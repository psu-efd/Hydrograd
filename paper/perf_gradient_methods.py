"""Cost and accuracy of the gradient methods of Hydrograd v1.0 (paper Table "gradient methods").

Usage:
    python paper/perf_gradient_methods.py OUT.jsonl steady NX NY T1 [T2 ...]
    python paper/perf_gradient_methods.py OUT.jsonl unsteady METHOD

Part 1, steady state: IFT against backpropagation through pseudo-time
---------------------------------------------------------------------
Case: the default synthetic channel (``write_channel_case``, 1000 m x 100 m, Q = 100 m^3/s,
exit WSE 2 m, bump 0.3 m) with NX x NY cells and four material zones of equal length along
the channel (Manning n = 0.03 s m^(-1/3) each). Loss L = mean(xi*^2) (m^2) at the steady
state U*, R(U*; n) = 0, gradient with respect to the four zone n values.

- IFT: pseudo-transient Newton solve, then J^T lambda = dL/dU*, dL/dn = -lambda^T dR/dn.
  Checked against central finite differences of the Newton loss (step 1e-6 s m^(-1/3)).
- Backpropagation through pseudo-time: the transient run from rest to time T (Tsit5, rtol
  1e-6, atol 1e-9, 64 checkpoints, the package default) is used as an approximation of the
  steady state, and reverse-mode AD differentiates through all its steps. Its error
  |g_T - g_IFT| / |g_IFT| falls as U(T) approaches U*; the residual max|R(U(T))| is recorded.

Part 2, unsteady flood: explicit against implicit time stepping
---------------------------------------------------------------
Case: the 500-cell compound valley of ``benchmarks/validation/hydrograph_inversion.py``
(1 km x 200 m, 50 x 10 cells, n = 0.035, second-order scheme), a 1 h flood with the inflow
Q(t) (m^3/s) given at 900 s knots. Observations: WSE at three centreline gauges every 300 s
from an explicit run with the true flood. Loss: mean squared stage misfit (m^2) at the
perturbed hydrograph Q = 0.8 Q_true; gradient with respect to the five knot values.

METHOD is ``explicit64`` (SSPRK3 with CFL steps, 64 checkpoints), ``explicit_auto``
(SSPRK3, checkpoints = "auto" with a 2 GB budget) or ``be<DT>`` (backward Euler with the
fixed step DT s and per-step IFT adjoints, e.g. ``be30``). Accuracy of each gradient: the
directional derivative g.d along a fixed random unit vector d is compared with a central
finite difference of the same discrete model (step 0.05 m^3/s). The explicit model's CFL
step selection is not differentiated, so its loss is slightly rough and its finite
difference less clean.

Each part (and each unsteady METHOD) runs in its own process, so that the peak memory (GB)
belongs to that method. Times in s; one JSON line per result appended to OUT.jsonl.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import ODESettings, solve_swe
from hydrograd.solvers.steady import NewtonSettings, steady_state

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmarks" / "validation"))
import hydrograph_inversion as valley
from perf_common import environment_record, peak_memory_gb, timed

# Steady part: channel length (m), number of zones, Newton dtau0 (s), FD step (s m^(-1/3)).
CHANNEL_LENGTH_M = 1000.0
N_ZONES = 4
DTAU0_S = 10.0
FD_STEP_N = 1.0e-6

# Unsteady part: event duration (s), knot and observation intervals (s), FD step (m^3/s).
T_FLOOD_S = 3600.0
KNOT_INTERVAL_S = 900.0
OBS_INTERVAL_S = 300.0
FD_STEP_Q = 0.05


def write(out: Path, record: dict) -> None:
    """Print a record and append it as one JSON line to ``out``.

    Args:
        out: output JSONL file.
        record: JSON-serializable dictionary.
    """
    print(json.dumps(record), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")


def compiled_once(f_jit: Any, *args: Any) -> tuple[Any, float]:
    """Compile a jitted function ahead of time, then time one compiled call.

    Used for the long transient runs, where a second timed call would double the cost.

    Args:
        f_jit: a ``jax.jit``-wrapped function.
        *args: its arguments.

    Returns:
        (output, wall time of the compiled call (s)).
    """
    compiled = f_jit.lower(*args).compile()
    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(*args))
    return out, time.perf_counter() - t0


def zone_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Material ID (1..N_ZONES) of four equal reaches along the channel.

    Args:
        x: cell centroid x (m), any shape.
        y: cell centroid y (m), same shape (unused).

    Returns:
        Integer material IDs, same shape as x.
    """
    reach_length = CHANNEL_LENGTH_M / N_ZONES
    return np.minimum((x // reach_length).astype(int) + 1, N_ZONES)


def run_steady(out: Path, nx: int, ny: int, t_ends: list[float]) -> None:
    """Part 1: IFT gradient against backpropagation through pseudo-time.

    Args:
        out: output JSONL file.
        nx: quads along the channel.
        ny: quads across the channel.
        t_ends: pseudo-time lengths T (s) of the transient runs, ascending.
    """
    # 1. Case with four zones.
    case_dir = Path(tempfile.mkdtemp()) / "c"
    write_channel_case(case_dir, nx, ny, zone_fn=zone_of)
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    U0 = case.U0
    n0 = jnp.asarray(model.n_zone)
    newton_settings = NewtonSettings(dtau0=DTAU0_S)

    # 2. IFT gradient (Newton + one transposed solve) and the forward-only Newton loss.
    def steady_loss(n_zone):
        """mean(xi*^2) (m^2) at the steady state for zone n (shape [n_zones])."""
        return jnp.mean(steady_state(model, {"n_zone": n_zone}, U0, newton_settings)[0] ** 2)

    loss_fn = jax.jit(steady_loss)
    _, _, newton_s = timed(loss_fn, n0)
    g_ift, ift_compile_s, ift_s = timed(jax.jit(jax.grad(steady_loss)), n0)
    g_ift = np.asarray(g_ift)

    # 3. Central finite differences of the Newton loss, one zone at a time.
    g_fd = np.zeros_like(g_ift)
    for k in range(g_ift.size):
        step = jnp.zeros_like(n0).at[k].set(FD_STEP_N)
        g_fd[k] = (float(loss_fn(n0 + step)) - float(loss_fn(n0 - step))) / (2.0 * FD_STEP_N)
    fd_err = float(np.linalg.norm(g_ift - g_fd) / np.linalg.norm(g_fd))

    write(out, {
        "part": "steady", "method": "newton_ift", "cells": nx * ny,
        "newton_s": newton_s, "grad_s": ift_s, "grad_compile_s": ift_compile_s,
        "grad": g_ift.tolist(), "fd_rel_err": fd_err, "peak_mem_gb": peak_memory_gb(),
        "environment": environment_record(),
    })

    # 4. Backpropagation through transient runs of growing length T.
    for t_end in t_ends:
        settings = ODESettings(0.0, t_end, 0.1, n_save=1, rtol=1.0e-6, atol=1.0e-9)

        def transient_loss(n_zone, settings=settings):
            """mean(xi(T)^2) (m^2) after the transient run, for zone n (shape [n_zones])."""
            return jnp.mean(solve_swe(model, {"n_zone": n_zone}, U0, settings).ys[-1][0] ** 2)

        def forward(U, settings=settings):
            """Transient solution from U (shape [3, n_cells])."""
            return solve_swe(model, {}, U, settings)

        # Long runs: compile ahead of time, then time one compiled call of each function.
        sol, fwd_s = compiled_once(jax.jit(forward), U0)
        U_T = sol.ys[-1]
        residual = float(jnp.max(jnp.abs(rhs(model, {}, U_T))))
        g_T, grad_s = compiled_once(jax.jit(jax.grad(transient_loss)), n0)
        g_T = np.asarray(g_T)

        write(out, {
            "part": "steady", "method": "bptt", "cells": nx * ny, "t_end_s": t_end,
            "fwd_steps": int(sol.stats["num_steps"]), "fwd_s": fwd_s, "grad_s": grad_s,
            "residual_max": residual, "grad": g_T.tolist(),
            "rel_err_vs_ift": float(np.linalg.norm(g_T - g_ift) / np.linalg.norm(g_ift)),
            "peak_mem_gb": peak_memory_gb(),
        })


def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial WSE (m): base-flow depth 1.03 m above the sloping valley bed.

    Args:
        x: cell centroid x (m), shape [n_cells].
        y: cell centroid y (m), shape [n_cells] (unused).

    Returns:
        WSE (m), shape [n_cells].
    """
    return valley.S0 * (valley.L - x) + 1.03


def run_unsteady(out: Path, method: str) -> None:
    """Part 2: explicit against implicit gradients for the flood in the compound valley.

    Args:
        out: output JSONL file.
        method: ``explicit64``, ``explicit_auto`` or ``be<DT>`` (DT in s).
    """
    # 1. Case: true hydrograph at the knots.
    t_knots = np.arange(0.0, T_FLOOD_S + 1.0, KNOT_INTERVAL_S)
    q_knots = valley.q_true(t_knots)
    case_dir = Path(tempfile.mkdtemp()) / "c"
    write_channel_case(
        case_dir, valley.NX, valley.NY, length=valley.L, width=valley.W,
        n_manning=valley.N_MAN, zb_fn=valley.bed, wstill=0.0, dt=0.1, t_end=T_FLOOD_S,
        solver="SSPRK3()", inlet="INLET-Q", outlet="EXIT-H", wse_out=1.03,
        Q_series=(t_knots, q_knots), ic_wse=initial_wse,
    )
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    U0 = case.U0

    # 2. Gauge cells (nearest centroid to each centreline point) and observation times (s).
    cx = np.asarray(case.mesh.centroid[:, 0])
    cy = np.asarray(case.mesh.centroid[:, 1])
    gauge_list = [int(np.argmin((cx - gx) ** 2 + (cy - valley.W / 2) ** 2))
                  for gx in valley.GAUGES_X]
    gauges = jnp.asarray(gauge_list)
    t_obs = jnp.arange(0.0, T_FLOOD_S + 1.0, OBS_INTERVAL_S)
    n_obs = int(t_obs.size)

    # 3. Gauge stages of the explicit and the implicit models (m), shape [n_obs, n_gauges].
    def stages_explicit(q, settings):
        """Explicit SSPRK3 stages for knot discharges q (m^3/s), shape [n_knots]."""
        sol = solve_swe(model, {"Q": q[None, :]}, U0, settings, ts=t_obs)
        return sol.ys[:, 0, gauges] + model.wstill[gauges]

    def stages_implicit(q, settings):
        """Backward-Euler stages for knot discharges q (m^3/s), shape [n_knots]."""
        _, ys = solve_implicit(model, {"Q": q[None, :]}, U0, settings)
        return ys[:, 0, gauges] + model.wstill[gauges]

    # 4. Observations: explicit run with the true hydrograph.
    q_true = jnp.asarray(q_knots)
    observed = jax.jit(lambda q: stages_explicit(q, case.ode))(q_true)

    # 5. The method's stage function and step count.
    if method.startswith("explicit"):
        checkpoints = 64 if method == "explicit64" else "auto"
        settings = ODESettings(0.0, T_FLOOD_S, case.ode.dt, solver="SSPRK3()",
                               checkpoints=checkpoints)

        def stages(q):
            """Explicit stages (m), shape [n_obs, n_gauges]."""
            return stages_explicit(q, settings)

    else:
        dt_s = float(method[2:])
        settings = ImplicitSettings(0.0, T_FLOOD_S, dt_s, theta=1.0, n_save=n_obs - 1,
                                    tol=1.0e-6)

        def stages(q):
            """Backward-Euler stages (m), shape [n_obs, n_gauges]."""
            return stages_implicit(q, settings)

    # 6. Loss, gradient and forward timing at the perturbed hydrograph.
    def loss(q):
        """Mean squared stage misfit (m^2) for knot discharges q (m^3/s)."""
        return jnp.mean((stages(q) - observed) ** 2)

    q_eval = 0.8 * q_true
    grad, grad_compile_s, grad_s = timed(jax.jit(jax.grad(loss)), q_eval)
    peak_gb = peak_memory_gb()
    loss_fn = jax.jit(loss)
    _, _, fwd_s = timed(loss_fn, q_eval)

    # 7. Directional finite-difference check along a fixed random unit vector.
    direction = np.random.default_rng(0).normal(size=q_eval.shape)
    direction = jnp.asarray(direction / np.linalg.norm(direction))
    fd = (float(loss_fn(q_eval + FD_STEP_Q * direction))
          - float(loss_fn(q_eval - FD_STEP_Q * direction))) / (2.0 * FD_STEP_Q)
    ad = float(jnp.dot(grad, direction))

    # 8. Step count of the method.
    if method.startswith("explicit"):
        sol = jax.jit(lambda q: solve_swe(model, {"Q": q[None, :]}, U0, settings))(q_eval)
        n_steps = int(sol.stats["num_steps"])
    else:
        n_steps = settings.steps()[0]

    write(out, {
        "part": "unsteady", "method": method, "cells": valley.NX * valley.NY,
        "steps": n_steps, "fwd_s": fwd_s, "grad_s": grad_s, "grad_compile_s": grad_compile_s,
        "ratio": grad_s / fwd_s, "loss": float(loss_fn(q_eval)),
        "grad": [float(g) for g in grad], "dir_ad": ad, "dir_fd": fd,
        "fd_rel_err": abs(ad - fd) / abs(fd), "peak_mem_gb": peak_gb,
        "environment": environment_record(),
    })


def main() -> None:
    """Dispatch to the steady or the unsteady part."""
    out = Path(sys.argv[1])
    part = sys.argv[2]

    if part == "steady":
        nx = int(sys.argv[3])
        ny = int(sys.argv[4])
        t_ends = [float(a) for a in sys.argv[5:]]
        run_steady(out, nx, ny, t_ends)
    else:
        run_unsteady(out, sys.argv[3])


if __name__ == "__main__":
    main()
