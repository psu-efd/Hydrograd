"""UDE training/inference (port of ``swe_2D_UDE.jl``): NN closures in the SWEs.

Closures
--------
``UDE_choice`` selects what the multilayer perceptron (MLP) predicts:

* ``ManningN_h``          : n = NN(h)
* ``ManningN_h_Umag_ks``  : n = NN(h, |U|, ks)
* ``FlowResistance``      : friction magnitude = NN(h, |q|)

with h (m), |U| (m/s), ks (m), |q| (m^2/s) and n (s/m^(1/3)); inputs and outputs are
normalized with the bounds of ``UDE_NN_config`` (see :mod:`hydrograd.apps.setup`).

Training
--------
Loss: ``[b_WSE] L_WSE + [b_uv] L_uv`` of the final frame (no bound/slope terms; see
:mod:`hydrograd.apps.losses`), plus the optional force (``"force_data"``) and scattered
velocity (``"velocity_data"``) misfits; a failed solve gives L = inf. Gradients are forward mode
for ``AutoForwardDiff()`` (Julia default for these small networks) and reverse mode with
checkpointing otherwise.

Files written
-------------
``UDE_callback_save_iter_<n>.json``, ``UDE_iteration_results_%04d.vtk``,
``UDE_results_losses_and_ODE_solution.json``, ``UDE_loss_history.csv`` and the
trained weights as ``.npz`` (Julia writes ``.jld2``). ``UDE_mode = "inference"``
loads the weights and runs one forward simulation (Julia's inference is a stub).

Architecture
------------
Default: a plain tanh MLP *without* LayerNorm (``UDE_NN_config.layernorm``,
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.apps.force_data import force_loss, load_force_data
from hydrograd.apps.forward import derived_fields, write_frame_vtk
from hydrograd.apps.inversion import use_forward_mode
from hydrograd.apps.losses import data_losses, load_truth
from hydrograd.apps.optimize import run_stages, stages_from_options
from hydrograd.apps.sensitivity import DERIV_RTOL
from hydrograd.apps.setup import Case
from hydrograd.apps.velocity_data import load_velocity_data, velocity_loss
from hydrograd.closures.mlp import init_mlp
from hydrograd.io.output import write_csv, write_json
from hydrograd.solvers.final import final_state, reverse_only

logger = logging.getLogger(__name__)

__all__ = ["load_weights", "run_ude", "save_weights"]

SEED = 1111


def save_weights(path: Path, params: list[dict[str, Any]]) -> None:
    """Save MLP weights to ``.npz`` (keys ``<layer>_<name>``).

    Args:
        path: output file.
        params: MLP parameters, one dict of arrays per layer.
    """
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(params):
        for k, v in layer.items():
            arrays[f"{i}_{k}"] = np.asarray(v)

    np.savez(path, **arrays)  # type: ignore[arg-type]


def load_weights(path: Path) -> list[dict[str, Any]]:
    """Inverse of :func:`save_weights`.

    Args:
        path: ``.npz`` file written by :func:`save_weights`.

    Returns:
        MLP parameters, one dict of JAX arrays per layer.
    """
    z = np.load(path)

    # Number of layers = largest layer index + 1.
    n = 1 + max(int(k.split("_", 1)[0]) for k in z.files)
    out: list[dict[str, Any]] = [{} for _ in range(n)]

    for k in z.files:
        i, name = k.split("_", 1)
        out[int(i)][name] = jnp.asarray(z[k])

    return out


def _npz_name(name: str) -> str:
    """File name with its suffix replaced by ``.npz`` (Julia names ``.jld2`` files).

    Args:
        name: file name from the run control.

    Returns:
        The same name with the ``.npz`` suffix.
    """
    return str(Path(name).with_suffix(".npz"))


def _init_params(case: Case) -> list[dict[str, Any]]:
    """Initial MLP weights: pretrained from file, or random with a fixed seed.

    Args:
        case: the case (``UDE_NN_config`` in its run control).

    Returns:
        MLP parameters, one dict of arrays per layer.
    """
    o = case.ctrl.opt
    d = case.ctrl.case_dir
    nn = o("UDE_NN_config")

    if nn.get("how_to_initialize_NN", "random") == "from_pretrained":
        return load_weights(d / _npz_name(nn["NN_weights_state_file_name"]))

    # Extension: optional "seed" in UDE_NN_config.
    seed = int(nn.get("seed", SEED))
    hidden_widths = [int(w) for w in nn["hidden_layers"]]
    return init_mlp(jax.random.PRNGKey(seed), int(nn["input_dim"]), hidden_widths,
                    int(nn["output_dim"]), layernorm=bool(nn.get("layernorm", False)))


def run_ude(case: Case) -> dict[str, Any]:
    """Train (or apply) the NN closure and write outputs.

    Args:
        case: the case.

    Returns:
        Inference: ``{"fields": ...}``, the derived fields of the final state.
        Training: ``{"params": ..., "history": ...}``, the trained weights and one loss
        row [iter, total, WSE, uv] per evaluation.
    """
    o = case.ctrl.opt
    m = case.model
    d = case.ctrl.case_dir

    params0 = _init_params(case)
    n_par = sum(int(np.size(v)) for layer in params0 for v in layer.values())
    logger.info("UDE %s: MLP with %d parameters", o("UDE_choice"), n_par)

    # ------------------------------------------------------------------
    # 1. Inference: one forward simulation with the given weights
    # ------------------------------------------------------------------
    if o("UDE_mode") == "inference":

        def solve_with(th: Any) -> Any:
            """Final state and success flag with MLP weights ``th``."""
            return final_state(m, {"nn": th}, case.U0, case.ode)

        U, _ = jax.jit(solve_with)(params0)
        f = derived_fields(m, {"nn": params0}, np.asarray(U), case.wstill)

        write_frame_vtk(case, "UDE_inference_results.vtk", f)
        write_json(d / Path(o("UDE_inference_save_file_name")).with_suffix(".json"),
                   {k: v for k, v in f.items()})
        return {"fields": f}

    # ------------------------------------------------------------------
    # 2. Training setup: truth, gradient mode, loss switches
    # ------------------------------------------------------------------
    truth = load_truth(json.loads((d / o("UDE_truth_file_name")).read_text()))

    # Steady / implicit IFT gradients are reverse mode only.
    ode_o = o("UDE_ode_solver_options")
    fwd = (
        use_forward_mode(o("UDE_sensealg"), ode_o.get("ode_solver_sensealg", ""), n_par)
        and not reverse_only(case.ode)
    )

    b_wse = bool(o("UDE_bWSE_loss"))
    b_uv = bool(o("UDE_b_uv_loss"))

    # Extension: observed vegetation forces ("force_data" block), or None.
    force_data = load_force_data(case)

    # Extension: scattered velocities, e.g. LSPIV ("velocity_data" block), or None.
    velocity_data = load_velocity_data(case)
    logger.info("UDE training with %s-mode gradients", "forward" if fwd else "reverse")

    # ------------------------------------------------------------------
    # 3. Loss function
    # ------------------------------------------------------------------
    def make_loss(adjoint: str) -> Callable[[Any], Any]:
        """Loss th -> (L, aux) with the given adjoint and derivative tolerances.

        Args:
            adjoint: diffrax adjoint name (``"forward"`` or ``"checkpoint"``).

        Returns:
            The loss function of the MLP weights (see ``loss_fn``).
        """
        s = dataclasses.replace(case.ode, adjoint=adjoint, rtol=min(case.ode.rtol, DERIV_RTOL),
                                atol=min(case.ode.atol, DERIV_RTOL * 1e-3))

        def loss_of_weights(th: Any) -> Any:
            """Loss at MLP weights ``th`` with the ODE settings above."""
            return loss_fn(th, s)

        return loss_of_weights

    def loss_fn(th: Any, s: Any) -> Any:
        """UDE training loss of MLP weights ``th`` (see the module docstring).

        Args:
            th: MLP parameters (pytree).
            s: ODE settings (adjoint, tolerances).

        Returns:
            ``(total, aux)``: the scalar loss (dimensionless; inf if the solve failed) and
            a dict with the terms ``"wse"``, ``"uv"``, ``"force"``, ``"velocity"`` and the
            final state ``"U"`` [3, n_cells].
        """
        U, ok = final_state(m, {"nn": th}, case.U0, s)
        dl = data_losses(m, U, truth)

        # Optional force misfit (extension; e.g. sub-patch forces for a C_D closure).
        lf = force_loss(m, {"nn": th}, U, force_data) if force_data else 0.0

        # Optional misfit of scattered (LSPIV) velocities (extension).
        lv = velocity_loss(m, {"nn": th}, U, velocity_data) if velocity_data else 0.0

        total = b_wse * dl["wse"] + b_uv * dl["uv"] + lf + lv
        total = jnp.where(ok, total, jnp.inf)

        return total, {"wse": dl["wse"], "uv": dl["uv"], "force": lf, "velocity": lv, "U": U}

    # ------------------------------------------------------------------
    # 4. Recording callback (loss history every iteration, fields every save_every)
    # ------------------------------------------------------------------
    save_every = int(o("UDE_training_save_frequency", 1))
    record_keys = ("iter", "loss_totals", "loss_pred_WSEs", "loss_pred_uvs", "inverted_WSEs",
                   "inverted_hs", "inverted_us", "inverted_vs", "ManningN_cells")
    rec: dict[str, list[Any]] = {key: [] for key in record_keys}
    history: list[list[float]] = []

    def callback(it: int, th: Any, loss: float, aux: dict[str, Any]) -> None:
        """Record iteration ``it`` (loss at weights ``th``, before the update).

        Args:
            it: iteration number (1-based).
            th: MLP parameters (pytree).
            loss: total loss at ``th``.
            aux: loss terms and final state from ``loss_fn``.
        """
        history.append([it, loss, float(aux["wse"]), float(aux["uv"])])

        # Save the fields only every save_every iterations (and at iteration 1).
        if it % save_every and it != 1:
            return

        f = derived_fields(m, {"nn": th}, np.asarray(aux["U"]), case.wstill)

        rec["iter"].append(it)
        rec["loss_totals"].append(loss)
        rec["loss_pred_WSEs"].append(float(aux["wse"]))
        rec["loss_pred_uvs"].append(float(aux["uv"]))
        rec["inverted_WSEs"].append(f["wse"])
        rec["inverted_hs"].append(f["h"])
        rec["inverted_us"].append(f["u"])
        rec["inverted_vs"].append(f["v"])
        rec["ManningN_cells"].append(f["ManningN"])

        write_json(d / f"UDE_callback_save_iter_{it}.json", {"iter_number": it, "loss_total": loss})
        save_weights(d / f"UDE_callback_save_iter_{it}.npz", th)
        write_frame_vtk(case, f"UDE_iteration_results_{len(rec['iter']):04d}.vtk", f)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    th, _ = run_stages(make_loss, params0, stages_from_options(o, "UDE"), callback,
                       forward_mode=fwd)

    # ------------------------------------------------------------------
    # 6. Outputs: trained weights, loss history and the recorded iterations
    # ------------------------------------------------------------------
    save_weights(d / _npz_name(o("UDE_training_save_NN_weights_state_file_name")), th)
    write_csv(d / "UDE_loss_history.csv", ["iter", "loss_total", "loss_WSE", "loss_uv"], history)

    loss_preds = [a + b for a, b in zip(rec["loss_pred_WSEs"], rec["loss_pred_uvs"])]
    out: dict[str, Any] = {"nIter": len(rec["iter"]), "loss_preds": loss_preds}

    # Field records become nested lists; scalar records are kept as is.
    for key, values in rec.items():
        if key.startswith(("inverted", "Manning")):
            out[key] = [np.asarray(x).tolist() for x in values]
        else:
            out[key] = values

    write_json(d / "UDE_results_losses_and_ODE_solution.json", out)

    return {"params": th, "history": history}
