"""Scaling profile of Hydrograd v2.0 on synthetic channels of growing size (paper Table/Fig. 10a).

Usage:
    python paper/perf_profile_scaling.py OUT.jsonl NX NY [NX NY ...] [--steady-max-cells N]

This is the paper copy of ``benchmarks/package/profile_scaling.py`` (v3 data in
``benchmarks/package/results``), extended with the steady IFT gradient, peak memory and an
environment record. The model, case and solver settings are identical, so the v3 numbers
and these can be compared directly.

Case: the default synthetic channel of ``hydrograd.io.synthetic.write_channel_case``
(1000 m x 100 m, bed slope 1e-4, Gaussian bump of 0.3 m, inlet Q = 100 m^3/s, exit WSE 2 m,
Manning n = 0.03 s m^(-1/3), walls at the sides), meshed with NX x NY quadrilaterals and
run with the package defaults (hydrostatic reconstruction + HLLC, second-order MUSCL with
the Venkatakrishnan limiter, float64). For each mesh the script times:

- setup (parse + mesh + model);
- one RHS evaluation dU/dt = R(U);
- a transient forward run from rest, 0 -> 20 s (Tsit5, rtol 1e-6, atol 1e-9);
- the reverse-mode gradient of L = mean(xi(T)^2) (m^2) with respect to the zone Manning n
  through that run (diffrax recursive checkpointing, 64 checkpoints);
- the Jacobian sparsity pattern (distance-2 graph colouring), the steady pseudo-transient
  Newton solve R(U*) = 0, and the gradient of L* = mean(xi*^2) through the steady state by
  the implicit function theorem (Newton solve + one transposed sparse solve), only for
  meshes with at most N cells.

Every jitted function is timed twice; ``*_compile_s`` is the first call (compilation + run),
the plain key is the second (compiled) call. Times in s, memory in GB. One JSON line per
mesh is appended to OUT.jsonl; the first line of a new file is the environment record.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.jacobian import jacobian_pattern
from hydrograd.solvers.ode import ODESettings, solve_swe
from hydrograd.solvers.steady import NewtonSettings, newton_solve, steady_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_common import environment_record, peak_memory_gb, timed

# Transient run: 20 s simulated from rest (s); Tsit5 tolerances as in the v3 profile.
T_END_S = 20.0
RTOL = 1.0e-6
ATOL = 1.0e-9

# Initial pseudo-time step of the steady Newton solver (s), as in the v3 profile.
DTAU0_S = 10.0


def git_commit() -> str:
    """Return the current git commit hash of the repository ("" if unavailable)."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip()


def profile_mesh(nx: int, ny: int, steady_max: int) -> dict:
    """Time setup, RHS, forward run, gradient and (optionally) steady solve on one mesh.

    Args:
        nx: quads along the channel.
        ny: quads across the channel.
        steady_max: largest cell count for which the steady Newton + IFT part is run.

    Returns:
        Record of times (s), step and iteration counts, and peak memory (GB).
    """
    # 1. Write the case and time the setup.
    case_dir = Path(tempfile.mkdtemp()) / "c"
    write_channel_case(case_dir, nx, ny)
    record = {"nx": nx, "ny": ny, "cells": nx * ny, "device": jax.devices()[0].device_kind}

    t0 = time.perf_counter()
    case = setup_case(read_control(case_dir / "run_control.json"))
    record["setup_s"] = time.perf_counter() - t0
    model = case.model
    U0 = case.U0
    record["spatial_order"] = int(getattr(model, "order", -1))
    record["scheme"] = str(getattr(model, "scheme", ""))

    # 2. One RHS evaluation.
    def rhs_eval(U):
        """RHS dU/dt (shape [3, n_cells]) at the state U (shape [3, n_cells])."""
        return rhs(model, {}, U)

    _, record["rhs_compile_s"], record["rhs_s"] = timed(jax.jit(rhs_eval), U0)

    # 3. Transient forward run, 0 -> 20 s.
    settings = ODESettings(0.0, T_END_S, 0.1, n_save=1, rtol=RTOL, atol=ATOL)

    def forward(U):
        """Transient solution from the initial state U (shape [3, n_cells])."""
        return solve_swe(model, {}, U, settings)

    sol, record["fwd_compile_s"], record["fwd_s"] = timed(jax.jit(forward), U0)
    record["fwd_steps"] = int(sol.stats["num_steps"])

    # 4. Reverse-mode gradient of mean(xi(T)^2) with respect to the zone Manning n.
    def loss(n_zone):
        """mean(xi(T)^2) (m^2) for zone Manning n (s m^(-1/3), shape [n_zones])."""
        return jnp.mean(solve_swe(model, {"n_zone": n_zone}, U0, settings).ys[-1][0] ** 2)

    n_zone0 = jnp.asarray(model.n_zone)
    grad_fn = jax.jit(jax.grad(loss))
    _, record["grad_compile_s"], record["grad_s"] = timed(grad_fn, n_zone0)
    record["peak_mem_transient_gb"] = peak_memory_gb()

    # 5. Steady Newton solve and IFT gradient (meshes up to steady_max cells).
    if nx * ny > steady_max:
        return record

    # Build the pattern through the package cache, so that the Newton timing below
    # excludes the colouring.
    t0 = time.perf_counter()
    pattern = jacobian_pattern(model)
    record["pattern_s"] = time.perf_counter() - t0
    record["n_colors"] = pattern.n_colors
    newton_settings = NewtonSettings(dtau0=DTAU0_S)

    def newton(U):
        """Steady Newton solve from U; returns (U_steady, iterations, residual)."""
        return newton_solve(model, {}, U, newton_settings)

    (_, iters, res), record["newton_compile_s"], record["newton_s"] = timed(jax.jit(newton), U0)
    record["newton_iters"] = int(iters)
    record["newton_res"] = float(res)

    # IFT gradient: Newton solve + one transposed sparse solve + one VJP.
    def steady_loss(n_zone):
        """mean(xi*^2) (m^2) at the steady state for zone Manning n (shape [n_zones])."""
        U_star = steady_state(model, {"n_zone": n_zone}, U0, newton_settings)
        return jnp.mean(U_star[0] ** 2)

    # The IFT gradient can exceed the device memory on the largest meshes (the Jacobian
    # linearization at U* is held together with the forward solve); record that and go on.
    ift_fn = jax.jit(jax.grad(steady_loss))
    try:
        _, record["ift_grad_compile_s"], record["ift_grad_s"] = timed(ift_fn, n_zone0)
    except jax.errors.JaxRuntimeError as error:
        record["ift_grad_error"] = str(error).split("\n")[0][:200]
    record["peak_mem_total_gb"] = peak_memory_gb()
    return record


def main() -> None:
    """Parse the command line, write the environment record, profile every mesh size."""
    # 1. Command line: optional --steady-max-cells, output file, (NX, NY) pairs.
    args = sys.argv[1:]
    steady_max = 10**9
    if "--steady-max-cells" in args:
        i = args.index("--steady-max-cells")
        steady_max = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    out = Path(args[0])
    dims = [int(a) for a in args[1:]]

    # 2. Environment record (one line, key "environment") at the top of the output file.
    env = environment_record()
    env["git_commit"] = git_commit()
    with open(out, "a") as f:
        f.write(json.dumps({"environment": env}) + "\n")
    print(json.dumps(env), flush=True)

    # 3. One record per mesh size.
    for nx, ny in zip(dims[::2], dims[1::2]):
        record = profile_mesh(nx, ny, steady_max)
        print(json.dumps(record), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
