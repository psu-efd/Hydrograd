"""Time/memory benchmark of one explicit-solver configuration (one process).

Example:
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

_SAV_REL = os.path.join("examples", "SWE_2D", "forward_simulation", "Savannah_River", "savana_SI.srhgeom")
_HERE = os.path.dirname(os.path.abspath(__file__))
# Hydrograd.jl location: $HYDROGRAD_JL, else the Windows layout (~/research), else a sibling of Hydrograd/ (Roar).
_JL_CANDIDATES = [os.environ.get("HYDROGRAD_JL", ""),
                  os.path.join(_HERE, "..", "..", "..", "..", "..", "research", "Hydrograd.jl"),
                  os.path.join(_HERE, "..", "..", "..", "Hydrograd.jl")]
SAVANNAH = next((os.path.join(d, _SAV_REL) for d in _JL_CANDIDATES
                 if d and os.path.exists(os.path.join(d, _SAV_REL))),
                os.path.join(_JL_CANDIDATES[1], _SAV_REL))


def process_peak_mb() -> float:
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is None:  # Linux: ru_maxrss in kB
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return peak / 2 ** 20


def timed(fn, sync, repeats: int):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sync(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
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

    if a.mesh == "savannah":
        mesh = read_srhgeom(SAVANNAH, inlet_ns=1, outlet_ns=2)
        cfg = FlowConfig(q_in=187.4 / 120.0, wse_out=29.3)
        U0 = initial_state(mesh, 29.3)
    else:
        mesh = make_channel_mesh(a.cells, length=1000.0, width=100.0)
        cfg = FlowConfig(q_in=1.0, wse_out=1.5)
        U0 = initial_state(mesh, 1.5)
    dt = a.dt if a.dt > 0 else cfl_dt(mesh, h_max=float(U0[:, 0].max()) + 0.5)
    sizes = tuple(int(v) for v in a.mlp.split(","))
    nf = true_n_field(mesh)
    res = {"backend": a.backend, "device": a.device, "dtype": a.dtype, "mesh": a.mesh,
           "cells": mesh.n_cells, "faces": mesh.n_faces, "mode": a.mode, "ckpt": a.ckpt,
           "compile": a.compile, "steps": a.n_save * a.k_inner, "platform": platform.system(),
           "dt": dt, "mlp": a.mlp}

    if a.backend == "jax":
        import jax
        import sim_jax as S
        B = S.build(mesh, cfg, dt, a.n_save, a.k_inner, a.ckpt, a.dtype, a.device)
        ops = B["ops"]
        with jax.default_device(B["device"]):
            U0d = ops.asarray(U0)
            h_obs = B["simulate"]({"n": ops.asarray(nf)}, U0d)
            theta = S.to_theta(ops, a.mode, np.full(mesh.n_cells, 0.035), sizes,
                               device=B["device"])
        res["device_kind"] = B["device"].device_kind
        sync = S.block_until_ready
        peak_dev = lambda: S.peak_memory_bytes(B["device"])
        reset = lambda: None
    else:
        import sim_torch as S
        B = S.build(mesh, cfg, dt, a.n_save, a.k_inner, a.ckpt, a.dtype, a.device, a.compile)
        ops = B["ops"]
        U0d = ops.asarray(U0)
        h_obs = B["simulate"]({"n": ops.asarray(nf)}, U0d)
        theta = S.to_theta(ops, a.mode, np.full(mesh.n_cells, 0.035), sizes)
        res["device_kind"] = (ops.t.cuda.get_device_name(0) if ops.device.type == "cuda"
                              else platform.processor() or "cpu")
        sync = lambda x: S.synchronize(ops)
        peak_dev = lambda: S.peak_memory_bytes(ops)
        reset = lambda: S.reset_peak(ops)

    t0 = time.perf_counter()
    sync(B["loss"](theta, U0d, h_obs))
    res["fwd_first_s"] = time.perf_counter() - t0
    res["fwd_s"] = timed(lambda: B["loss"](theta, U0d, h_obs), sync, a.repeats)
    res["fwd_peak_dev_mb"] = peak_dev() / 2 ** 20
    if a.grad_mode == "fwd":
        B["grad"] = B["fwd_grad"]
        res["grad_mode"] = "fwd"
    if not a.no_grad:
        reset()
        t0 = time.perf_counter()
        sync(B["grad"](theta, U0d, h_obs))
        res["grad_first_s"] = time.perf_counter() - t0
        res["grad_s"] = timed(lambda: B["grad"](theta, U0d, h_obs), sync, a.repeats)
        res["grad_peak_dev_mb"] = peak_dev() / 2 ** 20
        res["grad_over_fwd"] = res["grad_s"] / res["fwd_s"]
    res["us_per_cell_step_fwd"] = 1e6 * res["fwd_s"] / (mesh.n_cells * res["steps"])
    res["proc_peak_mb"] = process_peak_mb()
    print(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
