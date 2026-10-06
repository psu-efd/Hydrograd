"""Parameter inversion (port of ``swe_2D_inversion.jl``): Manning's n, zb or Q.

Problem
-------
Find the active parameters p (Manning's n per zone, s/m^(1/3); zb per cell, m; or Q per
inlet, m^3/s) that minimize

    L(p) = [b_WSE] L_WSE + [b_uv] L_uv + [b_bound] L_bound + [b_slope] L_slope,

where the switches [b] come from ``inversion_bInversion_*_loss`` and the terms are those
of :mod:`hydrograd.apps.losses`, evaluated on the final state U(t_end; p). A failed
solve gives L = inf.

Gradients
---------
Forward mode (``AutoForwardDiff``/``ForwardDiffSensitivity`` in the control file, and
few parameters) or reverse mode with recursive checkpointing (``AutoZygote``, adjoint
sensealgs, or many parameters, e.g. ``zb``). Steady and implicit runs always use
reverse-mode IFT gradients. The optimizer stages are run by
:func:`hydrograd.apps.optimize.run_stages`.

Files written
-------------
Julia names: ``inversion_callback_save_iter_<n>.json`` every
``inversion_save_frequency`` iterations, ``inversion_results_losses_and_ODE_solution.json``
(losses and final-state fields at the saved iterations), ``inversion_loss_history.csv``,
and ``forward_simulation_results_%04d.vtk`` per saved iteration.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from hydrograd.apps.forward import derived_fields, write_frame_vtk
from hydrograd.apps.losses import (
    LossWeights,
    bound_loss,
    data_losses,
    load_truth,
    slope_loss,
)
from hydrograd.apps.optimize import run_stages, stages_from_options
from hydrograd.apps.params import initial_params, to_theta
from hydrograd.apps.sensitivity import DERIV_RTOL
from hydrograd.apps.setup import Case
from hydrograd.io.output import write_csv, write_json
from hydrograd.solvers.final import final_state, reverse_only

logger = logging.getLogger(__name__)

__all__ = ["run_inversion", "use_forward_mode"]

_FORWARD_ADTYPES = ("AutoForwardDiff()", "ForwardSensitivity()")
FORWARD_MAX_PARAMS = 16  # forward mode costs one tangent per parameter


def use_forward_mode(adtype: str, sensealg: str, n_params: int) -> bool:
    """Gradient mode from the Optimization.jl AD type (``*_sensealg`` key).

    Forward mode (diffrax ``ForwardMode``) is used for ``AutoForwardDiff()`` with
    at most ``FORWARD_MAX_PARAMS`` parameters; otherwise reverse mode with
    recursive checkpointing. The ODE ``sensealg`` string is ignored (it selects
    a SciMLSensitivity algorithm that has no one-to-one diffrax counterpart).

    Args:
        adtype: Optimization.jl AD type, e.g. ``"AutoForwardDiff()"``.
        sensealg: SciMLSensitivity algorithm name (ignored).
        n_params: number of scalar parameters.

    Returns:
        True for forward-mode gradients, False for reverse mode.
    """
    del sensealg
    return adtype in _FORWARD_ADTYPES and n_params <= FORWARD_MAX_PARAMS


def _weights(o: Any) -> LossWeights:
    """Loss switches and parameter bounds from the run control.

    Args:
        o: option getter ``o(key, default)`` of the run control.

    Returns:
        The active loss terms, the parameter bounds (units of the parameter) and the
        bed-slope bound (-).
    """
    return LossWeights(
        wse=bool(o("inversion_bInversion_WSE_loss")),
        uv=bool(o("inversion_bInversion_uv_loss")),
        bound=bool(o("inversion_bInversion_bound_loss")),
        slope=bool(o("inversion_bInversion_slope_loss", False)),
        lb=float(o("inversion_parameter_value_lower_bound")),
        ub=float(o("inversion_parameter_value_upper_bound")),
        slope_ub=float(o("inversion_parameter_slope_upper_bound", np.inf)),
    )


def run_inversion(case: Case) -> dict[str, Any]:
    """Run the inversion and write Julia-compatible outputs.

    Args:
        case: the case.

    Returns:
        Dict with the final parameters ``"p"``, shape [n_params], and the loss
        ``"history"``: one row [iter, total, WSE, uv, bound, slope] per evaluation.
    """
    o = case.ctrl.opt
    m = case.model
    d = case.ctrl.case_dir

    # ------------------------------------------------------------------
    # 1. Initial parameters, truth, loss switches and gradient mode
    # ------------------------------------------------------------------
    name, p0 = initial_params(case, "inversion")
    truth = load_truth(json.loads((d / o("inversion_truth_file_name")).read_text()))
    w = _weights(o)

    # Steady / implicit IFT gradients are reverse mode only.
    ode_o = o("inversion_ode_solver_options")
    fwd = (
        use_forward_mode(o("inversion_sensealg"), ode_o.get("ode_solver_sensealg", ""), len(p0))
        and not reverse_only(case.ode)
    )
    logger.info("inversion of %s with %s-mode gradients", name, "forward" if fwd else "reverse")

    # ------------------------------------------------------------------
    # 2. Loss function
    # ------------------------------------------------------------------
    def make_loss(adjoint: str) -> Callable[[jnp.ndarray], Any]:
        """Loss p -> (L, aux) with the given adjoint and derivative tolerances.

        Args:
            adjoint: diffrax adjoint name (``"forward"`` or ``"checkpoint"``).

        Returns:
            The loss function of the parameters (see ``loss_fn``).
        """
        s = dataclasses.replace(case.ode, adjoint=adjoint, rtol=min(case.ode.rtol, DERIV_RTOL),
                                atol=min(case.ode.atol, DERIV_RTOL * 1e-3))

        def loss_of_p(p: jnp.ndarray) -> Any:
            """Loss at parameters p [n_params] with the ODE settings above."""
            return loss_fn(p, s)

        return loss_of_p

    def loss_fn(p: jnp.ndarray, s: Any) -> Any:
        """Total inversion loss of parameters p (see the module docstring).

        Args:
            p: parameters, shape [n_params].
            s: ODE settings (adjoint, tolerances).

        Returns:
            ``(total, aux)``: the scalar loss (dimensionless; inf if the solve failed) and
            a dict with the terms ``"wse"``, ``"uv"``, ``"bound"``, ``"slope"`` and the
            final state ``"U"`` [3, n_cells].
        """
        U, ok = final_state(m, to_theta(name, p), case.U0, s)

        # Data misfits; in zb inversion the parameters are the bed itself.
        zb = p if name == "zb" else None
        dl = data_losses(m, U, truth, zb)

        # Regularization terms.
        lb = bound_loss(p, w.lb, w.ub, m.n_cells) if w.bound else 0.0
        ls = slope_loss(m, p, w.slope_ub) if (w.slope and name == "zb") else 0.0

        total = w.wse * dl["wse"] + w.uv * dl["uv"] + lb + ls
        total = jnp.where(ok, total, jnp.inf)

        return total, {"wse": dl["wse"], "uv": dl["uv"], "bound": lb, "slope": ls, "U": U}

    # ------------------------------------------------------------------
    # 3. Recording callback (loss history every iteration, fields every save_every)
    # ------------------------------------------------------------------
    save_every = int(o("inversion_save_frequency", 1))
    record_keys = ("iter", "loss_totals", "loss_pred_WSEs", "loss_pred_uvs", "loss_bounds",
                   "loss_slopes", "inverted_WSEs", "inverted_hs", "inverted_us", "inverted_vs",
                   "theta")
    rec: dict[str, list[Any]] = {key: [] for key in record_keys}
    history: list[list[float]] = []

    def callback(it: int, p: Any, loss: float, aux: dict[str, Any]) -> None:
        """Record iteration ``it`` (loss at parameters p, before the update).

        Args:
            it: iteration number (1-based).
            p: parameters, shape [n_params].
            loss: total loss at p.
            aux: loss terms and final state from ``loss_fn``.
        """
        history.append([it, loss, float(aux["wse"]), float(aux["uv"]), float(aux["bound"]),
                        float(aux["slope"])])

        # Save the fields only every save_every iterations (and at iteration 1).
        if it % save_every and it != 1:
            return

        U = np.asarray(aux["U"])
        f = derived_fields(m, to_theta(name, p), U, case.wstill)

        rec["iter"].append(it)
        rec["loss_totals"].append(loss)
        rec["loss_pred_WSEs"].append(float(aux["wse"]))
        rec["loss_pred_uvs"].append(float(aux["uv"]))
        rec["loss_bounds"].append(float(aux["bound"]))
        rec["loss_slopes"].append(float(aux["slope"]))
        rec["inverted_WSEs"].append(f["wse"])
        rec["inverted_hs"].append(f["h"])
        rec["inverted_us"].append(f["u"])
        rec["inverted_vs"].append(f["v"])
        rec["theta"].append(np.asarray(p))

        write_json(d / f"inversion_callback_save_iter_{it}.json",
                   {"iter_number": it, "loss_total": loss, "theta": np.asarray(p)})
        write_frame_vtk(case, f"forward_simulation_results_{len(rec['iter']):04d}.vtk", f)

    # ------------------------------------------------------------------
    # 4. Optimize
    # ------------------------------------------------------------------
    p_final, _ = run_stages(make_loss, p0, stages_from_options(o, "inversion"), callback,
                            forward_mode=fwd)

    # ------------------------------------------------------------------
    # 5. Outputs: loss history and the recorded iterations
    # ------------------------------------------------------------------
    write_csv(d / "inversion_loss_history.csv",
              ["iter", "loss_total", "loss_WSE", "loss_uv", "loss_bound", "loss_slope"], history)

    loss_preds = [a + b for a, b in zip(rec["loss_pred_WSEs"], rec["loss_pred_uvs"])]
    out: dict[str, Any] = {"nIter": len(rec["iter"]), "loss_preds": loss_preds}

    # Field and parameter records become nested lists; scalar records are kept as is.
    for key, values in rec.items():
        if key.startswith("inverted") or key == "theta":
            out[key] = [np.asarray(x).tolist() for x in values]
        else:
            out[key] = values

    write_json(d / "inversion_results_losses_and_ODE_solution.json", out)

    logger.info("inversion finished: final %s = %s", name,
                np.array2string(np.asarray(p_final)[:8], precision=5))
    return {"p": np.asarray(p_final), "history": history}
