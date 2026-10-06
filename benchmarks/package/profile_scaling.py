"""Profile the hydrograd package on synthetic channels of growing size.

Usage: python profile_scaling.py OUT.jsonl NX NY [NX NY ...] [--steady-max-cells N]
Each size: setup (parse+mesh+model), RHS, transient forward (Tsit5, 20 s), reverse
gradient w.r.t. n_zone through the transient (checkpointed), Jacobian pattern and
steady Newton (up to --steady-max-cells). Times in s, one JSON line per size.
"""
import json, sys, tempfile, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.ode import ODESettings, solve_swe


def timed(f, *a):
    t0 = time.perf_counter(); out = jax.block_until_ready(f(*a)); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); out = jax.block_until_ready(f(*a)); return out, t1, time.perf_counter() - t0


def main():
    args = sys.argv[1:]
    steady_max = 10**9
    if "--steady-max-cells" in args:
        i = args.index("--steady-max-cells"); steady_max = int(args[i + 1]); args = args[:i] + args[i + 2:]
    out, dims = args[0], list(map(int, args[1:]))
    dev = jax.devices()[0]
    for nx, ny in zip(dims[::2], dims[1::2]):
        d = Path(tempfile.mkdtemp()) / "c"
        write_channel_case(d, nx, ny)
        r = {"nx": nx, "ny": ny, "cells": nx * ny, "device": dev.device_kind}
        t0 = time.perf_counter(); case = setup_case(read_control(d / "run_control.json")); r["setup_s"] = time.perf_counter() - t0
        m, U0 = case.model, case.U0
        _, r["rhs_compile_s"], r["rhs_s"] = timed(jax.jit(lambda U: rhs(m, {}, U)), U0)
        s = ODESettings(0.0, 20.0, 0.1, n_save=1, rtol=1e-6, atol=1e-9)
        sol, r["fwd_compile_s"], r["fwd_s"] = timed(jax.jit(lambda U: solve_swe(m, {}, U, s)), U0)
        r["fwd_steps"] = int(sol.stats["num_steps"])
        loss = lambda nz: jnp.mean(solve_swe(m, {"n_zone": nz}, U0, s).ys[-1][0] ** 2)
        _, r["grad_compile_s"], r["grad_s"] = timed(jax.jit(jax.grad(loss)), jnp.asarray(m.n_zone))
        if nx * ny <= steady_max:
            from hydrograd.solvers.jacobian import JacobianPattern
            from hydrograd.solvers.steady import NewtonSettings, newton_solve
            t0 = time.perf_counter(); P = JacobianPattern(m); r["pattern_s"] = time.perf_counter() - t0
            r["n_colors"] = P.n_colors
            from hydrograd.solvers import jacobian as J; J._CACHE[id(m)] = P
            (U, it, rn), r["newton_compile_s"], r["newton_s"] = timed(
                jax.jit(lambda U: newton_solve(m, {}, U, NewtonSettings(dtau0=10.0))), U0)
            r.update(newton_iters=int(it), newton_res=float(rn))
        print(json.dumps(r), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
