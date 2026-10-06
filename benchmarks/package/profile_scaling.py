"""Profile the hydrograd package on synthetic channels of growing size.

Usage: python profile_scaling.py OUT.jsonl NX NY [NX NY ...] [--steady-max-cells N]

Case: the default synthetic channel of ``hydrograd.io.synthetic.write_channel_case`` meshed
with NX x NY cells. For each size the script times:

- setup (parse + mesh + model);
- one RHS evaluation dU/dt = R(U);
- a transient forward run (Tsit5, 0 -> 20 s, rtol 1e-6, atol 1e-9);
- the reverse-mode gradient of mean(xi(T)^2) (m^2) w.r.t. the zone Manning n through the
  transient run (checkpointed);
- the Jacobian sparsity pattern (graph coloring) and a steady Newton solve, only for meshes
  with at most N cells (``--steady-max-cells``).

Each jitted function is timed twice: the first call includes compilation (``*_compile_s``),
the second is the compiled run time. Times in s, one JSON line per size, appended to
OUT.jsonl and printed.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np  # noqa: F401

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.ode import ODESettings, solve_swe


def timed(f, *a):
    """Call ``f(*a)`` twice and time both calls.

    Args:
        f: function (typically jitted) returning JAX arrays.
        *a: its arguments.

    Returns:
        (output of the second call, wall time of the first call incl. compilation (s),
        wall time of the second call (s)).
    """
    # First call: compilation + run.
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(*a))
    t1 = time.perf_counter() - t0

    # Second call: compiled run only.
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(*a))
    return out, t1, time.perf_counter() - t0


def main():
    """Parse the command line, then profile each mesh size and append its JSON record."""
    # 1. Command line: optional --steady-max-cells, output file, (NX, NY) pairs.
    args = sys.argv[1:]
    steady_max = 10**9
    if "--steady-max-cells" in args:
        i = args.index("--steady-max-cells")
        steady_max = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    out = args[0]
    dims = list(map(int, args[1:]))
    dev = jax.devices()[0]

    for nx, ny in zip(dims[::2], dims[1::2]):
        # 2. Write the case and time the setup.
        case_dir = Path(tempfile.mkdtemp()) / "c"
        write_channel_case(case_dir, nx, ny)
        r = {"nx": nx, "ny": ny, "cells": nx * ny, "device": dev.device_kind}
        t0 = time.perf_counter()
        case = setup_case(read_control(case_dir / "run_control.json"))
        r["setup_s"] = time.perf_counter() - t0
        m = case.model
        U0 = case.U0

        # 3. One RHS evaluation.
        def rhs_eval(U):
            """RHS dU/dt (shape [3, n_cells]) at state U (shape [3, n_cells])."""
            return rhs(m, {}, U)

        _, r["rhs_compile_s"], r["rhs_s"] = timed(jax.jit(rhs_eval), U0)

        # 4. Transient forward run, 0 -> 20 s.
        s = ODESettings(0.0, 20.0, 0.1, n_save=1, rtol=1e-6, atol=1e-9)

        def forward(U):
            """Transient solution from the initial state U (shape [3, n_cells])."""
            return solve_swe(m, {}, U, s)

        sol, r["fwd_compile_s"], r["fwd_s"] = timed(jax.jit(forward), U0)
        r["fwd_steps"] = int(sol.stats["num_steps"])

        # 5. Reverse gradient of a scalar loss w.r.t. the zone Manning n.
        def loss(nz):
            """mean(xi(T)^2) (m^2) for zone Manning n ``nz`` (s/m^(1/3), shape [n_zones])."""
            return jnp.mean(solve_swe(m, {"n_zone": nz}, U0, s).ys[-1][0] ** 2)

        _, r["grad_compile_s"], r["grad_s"] = timed(jax.jit(jax.grad(loss)),
                                                     jnp.asarray(m.n_zone))

        # 6. Jacobian pattern and steady Newton (small enough meshes only).
        if nx * ny <= steady_max:
            from hydrograd.solvers.jacobian import JacobianPattern
            from hydrograd.solvers.steady import NewtonSettings, newton_solve

            t0 = time.perf_counter()
            P = JacobianPattern(m)
            r["pattern_s"] = time.perf_counter() - t0
            r["n_colors"] = P.n_colors

            # Seed the pattern cache so the Newton timing excludes the coloring.
            from hydrograd.solvers import jacobian as J
            J._CACHE[id(m)] = P

            def newton(U):
                """Steady Newton solve from U; returns (U_steady, iterations, residual)."""
                return newton_solve(m, {}, U, NewtonSettings(dtau0=10.0))

            (U, it, rn), r["newton_compile_s"], r["newton_s"] = timed(jax.jit(newton), U0)
            r.update(newton_iters=int(it), newton_res=float(rn))

        # 7. Report and append the record.
        print(json.dumps(r), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
