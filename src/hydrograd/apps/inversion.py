"""Parameter inversion (port of ``swe_2D_inversion.jl``): Manning's n, zb or Q.

Gradients: forward mode (``AutoForwardDiff``/``ForwardDiffSensitivity`` in the
control file, or few parameters) or reverse mode with recursive checkpointing
(``AutoZygote``, adjoint sensealgs, or many parameters, e.g. ``zb``).

Outputs (Julia names): ``inversion_callback_save_iter_<n>.json`` every
``inversion_save_frequency`` iterations, ``inversion_results_losses_and_ODE_solution.json``
(losses and final-state fields at the saved iterations), ``inversion_loss_history.csv``,
and ``forward_simulation_results_%04d.vtk`` per saved iteration.
"""

from __future__ import annotations

import dataclasses
import json
import logging
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
from hydrograd.solvers.final import final_state, is_steady

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
    """
    del sensealg
    return adtype in _FORWARD_ADTYPES and n_params <= FORWARD_MAX_PARAMS


def _weights(o: Any) -> LossWeights:
    return LossWeights(wse=bool(o("inversion_bInversion_WSE_loss")), uv=bool(o("inversion_bInversion_uv_loss")),
                       bound=bool(o("inversion_bInversion_bound_loss")),
                       slope=bool(o("inversion_bInversion_slope_loss", False)),
                       lb=float(o("inversion_parameter_value_lower_bound")),
                       ub=float(o("inversion_parameter_value_upper_bound")),
                       slope_ub=float(o("inversion_parameter_slope_upper_bound", np.inf)))


def run_inversion(case: Case) -> dict[str, Any]:
    """Run the inversion and write Julia-compatible outputs."""
    o, m, d = case.ctrl.opt, case.model, case.ctrl.case_dir
    name, p0 = initial_params(case, "inversion")
    truth = load_truth(json.loads((d / o("inversion_truth_file_name")).read_text()))
    w = _weights(o)
    ode_o = o("inversion_ode_solver_options")
    fwd = use_forward_mode(o("inversion_sensealg"), ode_o.get("ode_solver_sensealg", ""), len(p0)) \
        and not is_steady(case.ode)   # steady IFT gradients are reverse mode
    logger.info("inversion of %s with %s-mode gradients", name, "forward" if fwd else "reverse")

    def make_loss(adjoint: str):
        s = dataclasses.replace(case.ode, adjoint=adjoint, rtol=min(case.ode.rtol, DERIV_RTOL),
                                atol=min(case.ode.atol, DERIV_RTOL * 1e-3))
        return lambda p: loss_fn(p, s)

    def loss_fn(p: jnp.ndarray, s: Any):
        U, ok = final_state(m, to_theta(name, p), case.U0, s)
        zb = p if name == "zb" else None
        dl = data_losses(m, U, truth, zb)
        lb = bound_loss(p, w.lb, w.ub, m.n_cells) if w.bound else 0.0
        ls = slope_loss(m, p, w.slope_ub) if (w.slope and name == "zb") else 0.0
        total = w.wse * dl["wse"] + w.uv * dl["uv"] + lb + ls
        total = jnp.where(ok, total, jnp.inf)
        return total, {"wse": dl["wse"], "uv": dl["uv"], "bound": lb, "slope": ls, "U": U}

    save_every = int(o("inversion_save_frequency", 1))
    rec: dict[str, list[Any]] = {k: [] for k in ("iter", "loss_totals", "loss_pred_WSEs", "loss_pred_uvs",
                                                 "loss_bounds", "loss_slopes", "inverted_WSEs",
                                                 "inverted_hs", "inverted_us", "inverted_vs", "theta")}
    history: list[list[float]] = []

    def callback(it: int, p: Any, loss: float, aux: dict[str, Any]) -> None:
        history.append([it, loss, float(aux["wse"]), float(aux["uv"]), float(aux["bound"]), float(aux["slope"])])
        if it % save_every and it != 1:
            return
        U = np.asarray(aux["U"])
        f = derived_fields(m, to_theta(name, p), U, case.wstill)
        for k, v in (("iter", it), ("loss_totals", loss), ("loss_pred_WSEs", float(aux["wse"])),
                     ("loss_pred_uvs", float(aux["uv"])), ("loss_bounds", float(aux["bound"])),
                     ("loss_slopes", float(aux["slope"])), ("inverted_WSEs", f["wse"]),
                     ("inverted_hs", f["h"]), ("inverted_us", f["u"]), ("inverted_vs", f["v"]),
                     ("theta", np.asarray(p))):
            rec[k].append(v)
        write_json(d / f"inversion_callback_save_iter_{it}.json",
                   {"iter_number": it, "loss_total": loss, "theta": np.asarray(p)})
        write_frame_vtk(case, f"forward_simulation_results_{len(rec['iter']):04d}.vtk", f)

    p_final, _ = run_stages(make_loss, p0, stages_from_options(o, "inversion"), callback,
                            forward_mode=fwd)
    write_csv(d / "inversion_loss_history.csv", ["iter", "loss_total", "loss_WSE", "loss_uv", "loss_bound",
                                                 "loss_slope"], history)
    out = {"nIter": len(rec["iter"]), "loss_preds": [a + b for a, b in zip(rec["loss_pred_WSEs"], rec["loss_pred_uvs"])]}
    out.update({k: ([np.asarray(x).tolist() for x in v] if k.startswith("inverted") or k == "theta" else v) for k, v in rec.items()})
    write_json(d / "inversion_results_losses_and_ODE_solution.json", out)
    logger.info("inversion finished: final %s = %s", name, np.array2string(np.asarray(p_final)[:8], precision=5))
    return {"p": np.asarray(p_final), "history": history}
