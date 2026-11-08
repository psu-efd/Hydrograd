"""Loss of the hydrograph inversion (Sect. 4.4) between the inverted and the true knots.

Backward Euler with a fixed step can fail in one Newton solve when the flood first climbs
the bank (Sect. 5.3). This script evaluates the inversion loss, the mean squared stage
misfit (m^2) at the three gauges every 5 min against the explicit truth observations,
along the straight line

    Q(alpha) = Q_inv + alpha (Q_true - Q_inv),   alpha = 0, 0.1, ..., 1.2,

with Delta t = 60 s (the inversion setting) and Delta t = 30 s, and records which points
fail. It also computes the gradient of the loss at the inverted knots with Delta t = 60 s.

Inputs: ``paper/data/hydrograph_inversion/`` (observations, gauge cells, inverted knots)
written by ``prep_hydrograph_inversion.py``. Output: ``paper/results/hydrograph_line_scan.json``.
Runs on the CPU in about 15 min (240 implicit steps per 30 s evaluation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent
REPO = PAPER_DIR.parent
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(REPO / "benchmarks" / "validation"))

import hydrograph_inversion as valley
from gradient_checks import hydrograph_case, inverted_knots

from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit

DATA_DIR = PAPER_DIR / "data" / "hydrograph_inversion"
RESULTS_FILE = PAPER_DIR / "results" / "hydrograph_line_scan.json"
ALPHAS = np.round(np.arange(0.0, 1.21, 0.1), 2)
TIME_STEPS = (60.0, 30.0)   # s


def make_loss(model: Any, U0: jax.Array, dt: float, gauge: jax.Array, obs: jax.Array):
    """Jitted loss(q) -> (mean squared stage misfit in m^2, saved states).

    Args:
        model: the SWE model of the inversion case.
        U0: initial state, shape [3, n_cells].
        dt: backward Euler step (s).
        gauge: gauge cell indices, shape [3].
        obs: observed WSE at the gauges (m), shape [n_obs + 1, 3] (every 5 min from t = 0).

    Returns:
        A jitted function of the hydrograph knots q, shape [1, 9] (m^3/s).
    """
    n_frames = obs.shape[0] - 1
    settings = ImplicitSettings(0.0, valley.T_END, dt, theta=1.0, n_save=n_frames, tol=1e-6)

    def loss(q: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Mean squared misfit (m^2) and the saved states, shape [n_frames + 1, 3, n_cells]."""
        _, ys = solve_implicit(model, {"Q": q}, U0, settings)
        wse = ys[:, 0, gauge] + model.wstill[gauge]
        return jnp.mean((wse - obs) ** 2), ys

    return jax.jit(loss)


def main() -> None:
    """Scan both step sizes along the line, compute the gradient and write the results."""
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "temp" / "hydrograph_line_scan"
    work.mkdir(parents=True, exist_ok=True)

    # Case, observations and the two end points of the line.
    case, q_true, _t_knots = hydrograph_case(work)
    data = np.load(DATA_DIR / "hydrograph_inversion.npz")
    gauge = jnp.asarray(data["gauge_cells"])
    obs = jnp.asarray(data["obs_noise0mm_m"])
    q_inv = inverted_knots()

    results: dict[str, Any] = {
        "alphas": ALPHAS.tolist(),
        "q_inverted_m3_per_s": np.asarray(q_inv)[0].tolist(),
        "q_true_knots_m3_per_s": np.asarray(q_true)[0].tolist(),
        "scans": {},
    }

    # Loss along the line for each step size; a non-finite state marks a failed solve.
    for dt in TIME_STEPS:
        loss = make_loss(case.model, case.U0, dt, gauge, obs)
        rows = []
        for alpha in ALPHAS:
            value, ys = loss(q_inv + float(alpha) * (q_true - q_inv))
            finite_frames = jnp.all(jnp.isfinite(ys.reshape(ys.shape[0], -1)), axis=1)
            finite = bool(jnp.all(finite_frames))
            first_bad = -1 if finite else int(jnp.argmax(~finite_frames))
            rows.append({"alpha": float(alpha), "loss_m2": float(value) if finite else None,
                         "finite": finite, "first_non_finite_frame": first_bad})
            print(f"dt {dt:g} s  alpha {alpha:4.1f}  loss {float(value):.3e}  finite {finite}",
                  flush=True)
        results["scans"][f"dt{dt:g}"] = {"dt_s": dt, "points": rows,
                                         "n_failed": sum(not r["finite"] for r in rows)}

    # Gradient at the inverted knots with the inversion step: zero at a minimum.
    loss60 = make_loss(case.model, case.U0, 60.0, gauge, obs)
    gradient = jax.grad(lambda q: loss60(q)[0])(q_inv)
    results["gradient_at_inverted_knots_dt60"] = np.asarray(gradient)[0].tolist()
    results["max_abs_gradient_at_inverted_knots_dt60"] = float(jnp.max(jnp.abs(gradient)))

    RESULTS_FILE.write_text(json.dumps(results, indent=1))
    print("wrote", RESULTS_FILE)


if __name__ == "__main__":
    main()
