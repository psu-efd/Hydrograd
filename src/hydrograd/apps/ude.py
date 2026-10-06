"""UDE training/inference (port of ``swe_2D_UDE.jl``): NN closures in the SWEs.

``UDE_choice``:

* ``ManningN_h``          : n = NN(h)
* ``ManningN_h_Umag_ks``  : n = NN(h, |U|, ks)
* ``FlowResistance``      : friction magnitude = NN(h, |q|)

Loss: ``L_WSE`` and/or ``L_uv`` of the final frame (no bound/slope terms).
Gradients are forward mode for ``AutoForwardDiff()`` (Julia default for these
small networks) and reverse mode with checkpointing otherwise.

Outputs: ``UDE_callback_save_iter_<n>.json``, ``UDE_iteration_results_%04d.vtk``,
``UDE_results_losses_and_ODE_solution.json``, ``UDE_loss_history.csv`` and the
trained weights as ``.npz`` (Julia writes ``.jld2``). ``UDE_mode = "inference"``
loads the weights and runs one forward simulation (Julia's inference is a stub).

Architecture default: a plain tanh MLP *without* LayerNorm (``UDE_NN_config.layernorm``,
default false). The closure must be a local law n_i = f(h_i); a per-cell LayerNorm over the
few hidden units of a scalar input collapses n to a constant during training, and Julia's
whole-array LayerNorm is non-local. Julia weights loaded ``from_pretrained`` keep their
LayerNorm parameters.

Note: initial weights come from ``jax.random.PRNGKey(1111)``; they differ from
Julia's ``StableRNG(1111)``, so training trajectories are not bit-comparable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.apps.forward import derived_fields, write_frame_vtk
from hydrograd.apps.inversion import use_forward_mode
from hydrograd.apps.losses import data_losses, load_truth
from hydrograd.apps.optimize import run_stages, stages_from_options
from hydrograd.apps.sensitivity import DERIV_RTOL
from hydrograd.apps.setup import Case
from hydrograd.closures.mlp import init_mlp
from hydrograd.io.output import write_csv, write_json
from hydrograd.solvers.final import final_state, is_steady

logger = logging.getLogger(__name__)

__all__ = ["load_weights", "run_ude", "save_weights"]

SEED = 1111


def save_weights(path: Path, params: list[dict[str, Any]]) -> None:
    """Save MLP weights to ``.npz`` (keys ``<layer>_<name>``)."""
    np.savez(path, **{f"{i}_{k}": np.asarray(v) for i, layer in enumerate(params) for k, v in layer.items()})


def load_weights(path: Path) -> list[dict[str, Any]]:
    """Inverse of :func:`save_weights`."""
    z = np.load(path)
    n = 1 + max(int(k.split("_", 1)[0]) for k in z.files)
    out: list[dict[str, Any]] = [{} for _ in range(n)]
    for k in z.files:
        i, name = k.split("_", 1)
        out[int(i)][name] = jnp.asarray(z[k])
    return out


def _npz_name(name: str) -> str:
    return str(Path(name).with_suffix(".npz"))


def _init_params(case: Case) -> list[dict[str, Any]]:
    o, d = case.ctrl.opt, case.ctrl.case_dir
    nn = o("UDE_NN_config")
    if nn.get("how_to_initialize_NN", "random") == "from_pretrained":
        return load_weights(d / _npz_name(nn["NN_weights_state_file_name"]))
    seed = int(nn.get("seed", SEED))  # extension: optional "seed" in UDE_NN_config
    return init_mlp(jax.random.PRNGKey(seed), int(nn["input_dim"]), [int(w) for w in nn["hidden_layers"]],
                    int(nn["output_dim"]), layernorm=bool(nn.get("layernorm", False)))


def run_ude(case: Case) -> dict[str, Any]:
    """Train (or apply) the NN closure and write outputs."""
    o, m, d = case.ctrl.opt, case.model, case.ctrl.case_dir
    params0 = _init_params(case)
    n_par = sum(int(np.size(v)) for layer in params0 for v in layer.values())
    logger.info("UDE %s: MLP with %d parameters", o("UDE_choice"), n_par)

    if o("UDE_mode") == "inference":
        U, _ = jax.jit(lambda th: final_state(m, {"nn": th}, case.U0, case.ode))(params0)
        f = derived_fields(m, {"nn": params0}, np.asarray(U), case.wstill)
        write_frame_vtk(case, "UDE_inference_results.vtk", f)
        write_json(d / Path(o("UDE_inference_save_file_name")).with_suffix(".json"),
                   {k: v for k, v in f.items()})
        return {"fields": f}

    truth = load_truth(json.loads((d / o("UDE_truth_file_name")).read_text()))
    ode_o = o("UDE_ode_solver_options")
    fwd = use_forward_mode(o("UDE_sensealg"), ode_o.get("ode_solver_sensealg", ""), n_par) \
        and not is_steady(case.ode)   # steady IFT gradients are reverse mode
    b_wse, b_uv = bool(o("UDE_bWSE_loss")), bool(o("UDE_b_uv_loss"))
    logger.info("UDE training with %s-mode gradients", "forward" if fwd else "reverse")

    def make_loss(adjoint: str):
        s = dataclasses.replace(case.ode, adjoint=adjoint, rtol=min(case.ode.rtol, DERIV_RTOL),
                                atol=min(case.ode.atol, DERIV_RTOL * 1e-3))
        return lambda th: loss_fn(th, s)

    def loss_fn(th: Any, s: Any):
        U, ok = final_state(m, {"nn": th}, case.U0, s)
        dl = data_losses(m, U, truth)
        total = b_wse * dl["wse"] + b_uv * dl["uv"]
        total = jnp.where(ok, total, jnp.inf)
        return total, {"wse": dl["wse"], "uv": dl["uv"], "U": U}

    save_every = int(o("UDE_training_save_frequency", 1))
    rec: dict[str, list[Any]] = {k: [] for k in ("iter", "loss_totals", "loss_pred_WSEs", "loss_pred_uvs",
                                                 "inverted_WSEs", "inverted_hs", "inverted_us",
                                                 "inverted_vs", "ManningN_cells")}
    history: list[list[float]] = []

    def callback(it: int, th: Any, loss: float, aux: dict[str, Any]) -> None:
        history.append([it, loss, float(aux["wse"]), float(aux["uv"])])
        if it % save_every and it != 1:
            return
        f = derived_fields(m, {"nn": th}, np.asarray(aux["U"]), case.wstill)
        for k, v in (("iter", it), ("loss_totals", loss), ("loss_pred_WSEs", float(aux["wse"])),
                     ("loss_pred_uvs", float(aux["uv"])), ("inverted_WSEs", f["wse"]), ("inverted_hs", f["h"]),
                     ("inverted_us", f["u"]), ("inverted_vs", f["v"]), ("ManningN_cells", f["ManningN"])):
            rec[k].append(v)
        write_json(d / f"UDE_callback_save_iter_{it}.json", {"iter_number": it, "loss_total": loss})
        save_weights(d / f"UDE_callback_save_iter_{it}.npz", th)
        write_frame_vtk(case, f"UDE_iteration_results_{len(rec['iter']):04d}.vtk", f)

    th, _ = run_stages(make_loss, params0, stages_from_options(o, "UDE"), callback, forward_mode=fwd)
    save_weights(d / _npz_name(o("UDE_training_save_NN_weights_state_file_name")), th)
    write_csv(d / "UDE_loss_history.csv", ["iter", "loss_total", "loss_WSE", "loss_uv"], history)
    out: dict[str, Any] = {"nIter": len(rec["iter"]),
                           "loss_preds": [a + b for a, b in zip(rec["loss_pred_WSEs"], rec["loss_pred_uvs"])]}
    out.update({k: ([np.asarray(x).tolist() for x in v] if k.startswith(("inverted", "Manning")) else v) for k, v in rec.items()})
    write_json(d / "UDE_results_losses_and_ODE_solution.json", out)
    return {"params": th, "history": history}
