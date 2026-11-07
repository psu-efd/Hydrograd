"""Reverse-mode gradient cost and memory against the number of checkpoints (paper Fig. 10b).

Usage:
    python paper/perf_checkpoints.py OUT.jsonl NX NY N_CHECKPOINTS

Run one process per checkpoint count (the peak-memory counter cannot be reset inside a
process); ``perf_gpu.sbatch`` loops over the counts.

Case and run: as in ``perf_profile_scaling.py``, the default synthetic channel with NX x NY
cells, a transient run of 20 s from rest (Tsit5, rtol 1e-6, atol 1e-9), and the gradient of
L = mean(xi(T)^2) (m^2) with respect to the zone Manning n.

Background. diffrax's ``RecursiveCheckpointAdjoint(checkpoints=c)`` stores c states of the
forward run and recomputes the steps in between during the backward pass (binomial
checkpointing, Griewank and Walther, 2000). With N time steps and c >= N nothing is
recomputed; with c < N the recomputation grows roughly like N log(N)/log(c) steps, while the
memory falls in proportion to c. The script records

- the compiled forward time t_fwd (s) and the compiled gradient time t_grad (s), and their
  ratio t_grad / t_fwd (-);
- the peak memory of the gradient (GB): device peak on a GPU, process peak RSS on the CPU.

The gradient is computed first, so the recorded peak is that of the gradient.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.ode import ODESettings, solve_swe

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_common import environment_record, peak_memory_gb, timed

# Transient run as in perf_profile_scaling.py: 20 s from rest (s), Tsit5 tolerances.
T_END_S = 20.0
RTOL = 1.0e-6
ATOL = 1.0e-9


def main() -> None:
    """Time the gradient and the forward run for one mesh and one checkpoint count."""
    # 1. Command line.
    out = Path(sys.argv[1])
    nx = int(sys.argv[2])
    ny = int(sys.argv[3])
    n_checkpoints = int(sys.argv[4])

    # 2. Case and model.
    case_dir = Path(tempfile.mkdtemp()) / "c"
    write_channel_case(case_dir, nx, ny)
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    U0 = case.U0
    settings = ODESettings(0.0, T_END_S, 0.1, n_save=1, rtol=RTOL, atol=ATOL,
                           checkpoints=n_checkpoints)

    # 3. Gradient first, so that the peak memory belongs to the backward pass.
    def loss(n_zone):
        """mean(xi(T)^2) (m^2) for zone Manning n (s m^(-1/3), shape [n_zones])."""
        return jnp.mean(solve_swe(model, {"n_zone": n_zone}, U0, settings).ys[-1][0] ** 2)

    n_zone0 = jnp.asarray(model.n_zone)
    grad, grad_compile_s, grad_s = timed(jax.jit(jax.grad(loss)), n_zone0)
    peak_gb = peak_memory_gb()

    # 4. Forward run with the same settings.
    def forward(U):
        """Transient solution from the initial state U (shape [3, n_cells])."""
        return solve_swe(model, {}, U, settings)

    sol, _, fwd_s = timed(jax.jit(forward), U0)

    # 5. Record (environment included, since each count runs in its own process).
    record = {
        "cells": nx * ny,
        "checkpoints": n_checkpoints,
        "fwd_steps": int(sol.stats["num_steps"]),
        "fwd_s": fwd_s,
        "grad_s": grad_s,
        "grad_compile_s": grad_compile_s,
        "ratio": grad_s / fwd_s,
        "peak_mem_grad_gb": peak_gb,
        "grad": [float(g) for g in jnp.ravel(grad)],
        "environment": environment_record(),
    }
    print(json.dumps(record), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
