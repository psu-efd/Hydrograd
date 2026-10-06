"""Time/memory benchmark of one explicit-solver configuration (one process).

Each process runs one (backend, device, dtype, mesh, closure) configuration of the explicit
SSP-RK time loop of the shallow water equations, so that peak-memory numbers are clean.
It measures:

- the forward loss evaluation L(theta) = mean_snapshots (h - h_obs)^2 (s), first call
  (compile + run) and the median of ``--repeats`` later calls;
- the gradient dL/dtheta, reverse mode with optional checkpointing (``--ckpt``) or forward
  mode (``--grad_mode fwd``), timed the same way;
- peak device memory and peak process memory (MB).

The run has ``n_save * k_inner`` time steps of size dt (s); dt is the CFL step unless
``--dt`` overrides it. The roughness parameters theta are either a Manning's n field
(``--mode field``, one value per cell, s/m^(1/3)) or the weights of an MLP n = f(h, |U|)
(``--mode nn``, layer sizes from ``--mlp``).

Usage:
    python bench_explicit.py --backend jax --device cpu --cells 8000 --mode nn --ckpt

Prints a single JSON line with timings (s) and peak memory (MB).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import psutil

from backends import cfl_dt, initial_state, true_n_field
from mesh import make_channel_mesh, read_srhgeom
from physics import FlowConfig

# Savannah River mesh, relative to the root of Hydrograd.jl.
_SAV_REL = os.path.join(
    "examples", "SWE_2D", "forward_simulation", "Savannah_River", "savana_SI.srhgeom"
)
_HERE = os.path.dirname(os.path.abspath(__file__))

# Hydrograd.jl location: $HYDROGRAD_JL, else the Windows layout (~/research), else a sibling
# of Hydrograd/ (Roar).
_JL_CANDIDATES = [
    os.environ.get("HYDROGRAD_JL", ""),
    os.path.join(_HERE, "..", "..", "..", "..", "..", "research", "Hydrograd.jl"),
    os.path.join(_HERE, "..", "..", "..", "Hydrograd.jl"),
]

# First candidate that holds the mesh file; fall back to the Windows layout.
SAVANNAH = next(
    (
        os.path.join(jl_root, _SAV_REL)
        for jl_root in _JL_CANDIDATES
        if jl_root and os.path.exists(os.path.join(jl_root, _SAV_REL))
    ),
    os.path.join(_JL_CANDIDATES[1], _SAV_REL),
)


def process_peak_mb() -> float:
    """Peak memory of the current process.

    Uses the peak working set on Windows and ``ru_maxrss`` (kB) on Linux.

    Returns:
        Peak resident memory (MB, 2^20 bytes).
    """
    mem_info = psutil.Process().memory_info()
    peak = getattr(mem_info, "peak_wset", None)

    # Linux: ru_maxrss is in kB.
    if peak is None:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return peak / 2**20


def timed(fn, sync, repeats: int):
    """Median wall time of ``fn`` over several calls.

    Args:
        fn: Zero-argument callable to time; its result is passed to ``sync``.
        sync: Callable that blocks until the result of ``fn`` is computed (asynchronous
            dispatch on JAX/GPU).
        repeats: Number of timed calls.

    Returns:
        Median wall time of one call (s).
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    """Parse the CLI, build the solver, time forward and gradient, print one JSON line."""
    # ---------------------------------------------------------------------------------
    # 1. Command-line options
    # ---------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["jax", "torch"], required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--cells", type=int, default=8000)
    p.add_argument("--mesh", choices=["channel", "savannah"], default="channel")
    p.add_argument("--mode", choices=["field", "nn"], default="nn")
    p.add_argument("--n_save", type=int, default=10)
    p.add_argument("--k_inner", type=int, default=50)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the step")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no_grad", action="store_true")
    p.add_argument("--dt", type=float, default=0.0, help="override CFL time step (s)")
    p.add_argument("--grad_mode", choices=["rev", "fwd"], default="rev")
    p.add_argument("--mlp", type=str, default="2,16,16,1", help="MLP layer sizes")
    a = p.parse_args()

    if a.backend == "jax" and a.device == "cpu":
        # jax.default_device does not pin jit calls made outside its context to the CPU;
        # hide the GPU so a "cpu" run cannot silently execute on an available GPU.
        os.environ["JAX_PLATFORMS"] = "cpu"

    # ---------------------------------------------------------------------------------
    # 2. Mesh, flow configuration and initial state
    # ---------------------------------------------------------------------------------
    if a.mesh == "savannah":
        mesh = read_srhgeom(SAVANNAH, inlet_ns=1, outlet_ns=2)
        cfg = FlowConfig(q_in=187.4 / 120.0, wse_out=29.3)
        U0 = initial_state(mesh, 29.3)
    else:
        mesh = make_channel_mesh(a.cells, length=1000.0, width=100.0)
        cfg = FlowConfig(q_in=1.0, wse_out=1.5)
        U0 = initial_state(mesh, 1.5)

    # Time step (s): the override, or the CFL step with 0.5 m of head-room on h_max.
    if a.dt > 0:
        dt = a.dt
    else:
        dt = cfl_dt(mesh, h_max=float(U0[:, 0].max()) + 0.5)

    mlp_sizes = tuple(int(v) for v in a.mlp.split(","))
    n_field = true_n_field(mesh)

    # Result record (keys are read by summarize.py).
    res = {
        "backend": a.backend,
        "device": a.device,
        "dtype": a.dtype,
        "mesh": a.mesh,
        "cells": mesh.n_cells,
        "faces": mesh.n_faces,
        "mode": a.mode,
        "ckpt": a.ckpt,
        "compile": a.compile,
        "steps": a.n_save * a.k_inner,
        "platform": platform.system(),
        "dt": dt,
        "mlp": a.mlp,
    }

    # ---------------------------------------------------------------------------------
    # 3. Backend: build the simulator, the truth observations and the parameters theta
    # ---------------------------------------------------------------------------------
    if a.backend == "jax":
        import jax
        import sim_jax as S

        B = S.build(mesh, cfg, dt, a.n_save, a.k_inner, a.ckpt, a.dtype, a.device)
        ops = B["ops"]
        with jax.default_device(B["device"]):
            U0d = ops.asarray(U0)
            h_obs = B["simulate"]({"n": ops.asarray(n_field)}, U0d)
            theta = S.to_theta(
                ops, a.mode, np.full(mesh.n_cells, 0.035), mlp_sizes, device=B["device"]
            )
        res["device_kind"] = B["device"].device_kind
        sync = S.block_until_ready

        def peak_dev():
            """Peak device memory of the JAX device (bytes)."""
            return S.peak_memory_bytes(B["device"])

        def reset():
            """No-op: JAX has no peak-memory reset."""

    else:
        import sim_torch as S

        B = S.build(mesh, cfg, dt, a.n_save, a.k_inner, a.ckpt, a.dtype, a.device, a.compile)
        ops = B["ops"]
        U0d = ops.asarray(U0)
        h_obs = B["simulate"]({"n": ops.asarray(n_field)}, U0d)
        theta = S.to_theta(ops, a.mode, np.full(mesh.n_cells, 0.035), mlp_sizes)
        if ops.device.type == "cuda":
            res["device_kind"] = ops.t.cuda.get_device_name(0)
        else:
            res["device_kind"] = platform.processor() or "cpu"

        def sync(x):
            """Wait for the torch device; the argument is ignored."""
            return S.synchronize(ops)

        def peak_dev():
            """Peak torch device memory since the last reset (bytes)."""
            return S.peak_memory_bytes(ops)

        def reset():
            """Reset the torch peak-memory counter."""
            return S.reset_peak(ops)

    # ---------------------------------------------------------------------------------
    # 4. Forward loss: first call (includes compilation), then the median of repeats
    # ---------------------------------------------------------------------------------
    t0 = time.perf_counter()
    sync(B["loss"](theta, U0d, h_obs))
    res["fwd_first_s"] = time.perf_counter() - t0
    res["fwd_s"] = timed(lambda: B["loss"](theta, U0d, h_obs), sync, a.repeats)
    res["fwd_peak_dev_mb"] = peak_dev() / 2**20

    # ---------------------------------------------------------------------------------
    # 5. Gradient (reverse mode by default, forward mode on request)
    # ---------------------------------------------------------------------------------
    if a.grad_mode == "fwd":
        B["grad"] = B["fwd_grad"]
        res["grad_mode"] = "fwd"
    if not a.no_grad:
        reset()
        t0 = time.perf_counter()
        sync(B["grad"](theta, U0d, h_obs))
        res["grad_first_s"] = time.perf_counter() - t0
        res["grad_s"] = timed(lambda: B["grad"](theta, U0d, h_obs), sync, a.repeats)
        res["grad_peak_dev_mb"] = peak_dev() / 2**20
        res["grad_over_fwd"] = res["grad_s"] / res["fwd_s"]

    # ---------------------------------------------------------------------------------
    # 6. Derived cost per cell and step (microseconds), process memory, output
    # ---------------------------------------------------------------------------------
    res["us_per_cell_step_fwd"] = 1e6 * res["fwd_s"] / (mesh.n_cells * res["steps"])
    res["proc_peak_mb"] = process_peak_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
