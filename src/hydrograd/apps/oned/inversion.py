"""Parameter inversion of a 1-D case from surveyed deposit volumes.

Problem
-------
Find the active parameters that minimize the negative log-posterior

    L(u) = sum_s sum_i (V[s, i](u) - V_obs[s, i])^2 / (2 sigma^2)  -  log p(u),

where V are the surveyed deposit volumes per cross-section of the model
(:mod:`hydrograd.apps.oned.observe`), V_obs those of the truth file, sigma the standard
deviation of the survey error (m^3) and log p the Gaussian priors of the active
parameters. The optimizer works on the unconstrained variables u of
:mod:`hydrograd.params.transforms` (logarithms of positive parameters, and so on), so no
bounds are needed.

Gradients
---------
Reverse mode through the whole run, with the checkpointed time loop of
:mod:`hydrograd.oned.stepper`. The optimizer stages (``Descent``, ``Adam``, ``LBFGS``)
are run by :func:`hydrograd.apps.optimize.run_stages`, as for 2-D inversions.

Options (``inversion_options`` in ``run_control.json``)
----------------------------------------------------
::

    "active_param_names": ["a_eh", "kappa_w[0]"],          (required)
    "inversion_parameter_initial_values": {"a_eh": 0.03, "kappa_w[0]": 0.7},
                                        start of the search; default: the case values
    "inversion_truth_file_name": "oned_survey_truth.json",
    "inversion_survey_sigma": 500.0,                       sigma (m^3), or
    "inversion_survey_relative_sigma": 0.05,               sigma as a fraction of the
                                                           root-mean-square observed
                                                           volume (default 0.05)
    "inversion_parameter_prior_sd": {"a_eh": 0.7},         prior standard deviation of
                                        the unconstrained variable, by parameter name;
                                        centred on the case values; default: no prior
    "inversion_optimizers": ["LBFGS"],                     (required, with the next two)
    "inversion_learning_rates": [1.0],
    "inversion_max_iterations": [30],
    "inversion_abs_tols": [0.0],
    "inversion_save_frequency": 1

Files written
-------------
``inversion_loss_history.csv`` (iteration, total loss, misfit, negative log-prior and the
physical value of every active scalar), ``inversion_callback_save_iter_<n>.json`` every
``inversion_save_frequency`` iterations, and ``inversion_results.json`` (labels, initial
and final physical values, final losses, the modelled and the observed volumes).

Units: SI.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.apps.oned.observe import load_truth, survey_volumes
from hydrograd.apps.optimize import run_stages, stages_from_options
from hydrograd.io.control import Control
from hydrograd.io.oned_case import OneDCase
from hydrograd.io.output import write_csv
from hydrograd.oned.config import default_parameter_specs
from hydrograd.oned.stepper import run
from hydrograd.params.transforms import ParameterMap, with_values

logger = logging.getLogger(__name__)

__all__ = ["run_inversion", "survey_sigma"]


def survey_sigma(o: Callable[..., Any], observed: np.ndarray) -> float:
    """Standard deviation of the survey error from the run-control options.

    Args:
        o: option getter ``o(key, default)`` of the run control.
        observed: observed volumes (m^3), shape [S, N].

    Returns:
        sigma (m^3): ``inversion_survey_sigma`` if given, otherwise
        ``inversion_survey_relative_sigma`` (default 0.05) times the root-mean-square
        observed volume.
    """
    sigma = o("inversion_survey_sigma", None)
    if sigma is not None:
        return float(sigma)

    relative = float(o("inversion_survey_relative_sigma", 0.05))
    return relative * float(np.sqrt(np.mean(observed**2)))


def _flat_values(physical: dict[str, Array]) -> list[float]:
    """Physical values of all active labels as one flat list.

    Args:
        physical: label -> physical value (scalar or array).

    Returns:
        The values in the order of the labels, vector parameters entry by entry.
    """
    values: list[float] = []
    for value in physical.values():
        values.extend(np.atleast_1d(np.asarray(value)).ravel().tolist())
    return values


def _flat_names(physical: dict[str, Array]) -> list[str]:
    """Column names matching :func:`_flat_values`.

    Args:
        physical: label -> physical value (scalar or array).

    Returns:
        One name per scalar: the label, or ``label[i]`` for the entries of a vector.
    """
    names: list[str] = []
    for label, value in physical.items():
        size = int(np.size(value))
        if np.ndim(value) == 0:
            names.append(label)
        else:
            names.extend(f"{label}[{entry}]" for entry in range(size))
    return names


def run_inversion(ctrl: Control, case: OneDCase) -> dict[str, Any]:
    """Run the inversion and write its output files.

    Args:
        ctrl: run control.
        case: the 1-D case.

    Returns:
        Dict with the final physical values ``"values"`` (label -> array), the final
        parameter dictionary ``"theta"`` and the loss ``"history"``: one row
        [iteration, total, misfit, negative log-prior, values...] per evaluation.
    """
    o = ctrl.opt
    case_dir = ctrl.case_dir
    sediment = case.config.sediment

    # ------------------------------------------------------------------
    # 1. Observations and their error
    # ------------------------------------------------------------------
    truth = load_truth(case_dir / o("inversion_truth_file_name", "oned_survey_truth.json"))
    observed = jnp.asarray(truth.volumes)
    sigma = survey_sigma(o, truth.volumes)

    n_sections = case.network.n_sections
    if truth.volumes.shape[1] != n_sections:
        raise ValueError(
            f"truth file has {truth.volumes.shape[1]} sections, the case has {n_sections}"
        )

    # ------------------------------------------------------------------
    # 2. Active parameters: transforms, priors and starting values
    # ------------------------------------------------------------------
    labels = list(o("active_param_names"))
    specs = default_parameter_specs()
    for name, prior_sd in o("inversion_parameter_prior_sd", {}).items():
        if name not in specs:
            raise ValueError(f"inversion_parameter_prior_sd: unknown parameter {name!r}")
        specs[name] = dataclasses.replace(specs[name], prior_sd=float(prior_sd))

    # The case values are the fixed values of the inactive parameters and the centre of
    # the priors; the search starts from the initial values of the run control.
    parameter_map = ParameterMap(specs, case.theta, labels)
    theta_start = with_values(case.theta, o("inversion_parameter_initial_values", {}))
    u_start = parameter_map.unconstrained(theta_start)
    initial = parameter_map.physical(u_start)

    logger.info(
        "1-D inversion of %s from %s; survey sigma %.4g m^3",
        labels, _flat_values(initial), sigma,
    )

    # ------------------------------------------------------------------
    # 3. Loss function
    # ------------------------------------------------------------------
    def loss_fn(u: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        """Negative log-posterior of the optimizer variables u.

        Args:
            u: label -> unconstrained value.

        Returns:
            ``(total, aux)``: the scalar loss and a dict with the ``"misfit"``, the
            negative log-prior ``"prior"`` and the modelled volumes ``"volumes"``
            (m^3), shape [S, N].
        """
        theta = parameter_map.theta(u)
        result = run(case.config, case.network, theta, case.forcing, case.bed0)
        volumes = survey_volumes(result, theta, sediment, truth.times, truth.consolidation)

        misfit = jnp.sum((volumes - observed) ** 2) / (2.0 * sigma**2)
        prior = -parameter_map.log_prior(u)
        total = misfit + prior

        # A failed run (non-finite volumes) gives an infinite loss.
        total = jnp.where(jnp.isfinite(total), total, jnp.inf)
        return total, {"misfit": misfit, "prior": prior, "volumes": volumes}

    def make_loss(adjoint: str) -> Callable[[dict[str, Array]], Any]:
        """The loss function; the 1-D mode has one (reverse-mode) gradient path.

        Args:
            adjoint: adjoint name requested by the optimizer loop (not used).

        Returns:
            ``loss_fn``.
        """
        del adjoint
        return loss_fn

    # ------------------------------------------------------------------
    # 4. Recording callback
    # ------------------------------------------------------------------
    save_every = int(o("inversion_save_frequency", 1))
    history: list[list[float]] = []
    last: dict[str, Any] = {}

    def callback(iteration: int, u: dict[str, Array], loss: float, aux: dict[str, Any]) -> None:
        """Record one iteration (loss at u, before the update).

        Args:
            iteration: iteration number (1-based).
            u: optimizer variables.
            loss: total loss at u.
            aux: loss terms and modelled volumes from ``loss_fn``.
        """
        physical = parameter_map.physical(u)
        values = _flat_values(physical)
        history.append([iteration, loss, float(aux["misfit"]), float(aux["prior"]), *values])

        last["volumes"] = np.asarray(aux["volumes"])
        last["misfit"] = float(aux["misfit"])
        last["prior"] = float(aux["prior"])
        last["loss"] = loss

        # Save the parameters only every save_every iterations (and at iteration 1).
        if iteration % save_every and iteration != 1:
            return

        record = {
            "iter_number": iteration,
            "loss_total": loss,
            "parameter_labels": _flat_names(physical),
            "parameter_values": values,
        }
        name = f"inversion_callback_save_iter_{iteration}.json"
        (case_dir / name).write_text(json.dumps(record))

    # ------------------------------------------------------------------
    # 5. Optimize
    # ------------------------------------------------------------------
    u_final, _ = run_stages(make_loss, u_start, stages_from_options(o, "inversion"), callback)
    final = parameter_map.physical(u_final)
    theta_final = parameter_map.theta(u_final)

    # ------------------------------------------------------------------
    # 6. Outputs
    # ------------------------------------------------------------------
    names = _flat_names(final)
    header = ["iter", "loss_total", "loss_misfit", "loss_prior", *names]
    write_csv(case_dir / "inversion_loss_history.csv", header, history)

    results = {
        "parameter_labels": names,
        "parameter_initial_values": _flat_values(initial),
        "parameter_final_values": _flat_values(final),
        "loss_total": last.get("loss"),
        "loss_misfit": last.get("misfit"),
        "loss_prior": last.get("prior"),
        "survey_sigma": sigma,
        "survey_times": truth.times.tolist(),
        "deposit_volume_model": last["volumes"].tolist(),
        "deposit_volume_observed": truth.volumes.tolist(),
        "n_iterations": len(history),
    }
    (case_dir / "inversion_results.json").write_text(json.dumps(results))

    logger.info("1-D inversion finished: %s = %s", names, _flat_values(final))
    return {"values": final, "theta": theta_final, "history": history}
