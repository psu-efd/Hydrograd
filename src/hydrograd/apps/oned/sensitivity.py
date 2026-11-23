"""Sensitivity analysis of a 1-D case.

What it computes
----------------
The Jacobian of the surveyed deposit volumes (see :mod:`hydrograd.apps.oned.observe`)
with respect to the active parameters:

    J[s, i, j] = d V[s, i] / d p_j          (m^3 per unit of p_j),

with s the survey, i the cross-section and p_j the physical value of the j-th active
scalar. A label of a whole vector parameter (``"kappa_w"``) contributes one column per
entry (``"kappa_w[0]"``, ``"kappa_w[1]"``, ...). The scaled Jacobian

    J_log[s, i, j] = p_j J[s, i, j]          (m^3 per unit of ln p_j)

is also written; its columns are comparable between parameters of different units.

How
---
Forward-mode automatic differentiation through the whole run (one tangent per active
scalar), which suits the few parameters of a sensitivity analysis.

Options (``sensitivity_analysis_options`` in ``run_control.json``)
---------------------------------------------------------------
::

    "active_param_names": ["a_eh", "kappa_w[0]", ...],     (required)
    "sensitivity_parameter_values": {"a_eh": 0.04, ...},   evaluation point; default:
                                                           the values of the case file
    "survey_times": [t1, ...],                             default: the end of the run
    "survey_consolidation": true

File written
------------
``sensitivity_results.json``: ``parameter_labels`` [P], ``parameter_values`` [P],
``survey_times`` [S], ``deposit_volume`` [S][N], ``jacobian`` [S][N][P] and
``jacobian_log`` [S][N][P] (nested lists, row-major).

Units: SI.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.apps.oned.observe import survey_times, survey_volumes
from hydrograd.io.control import Control
from hydrograd.io.oned_case import OneDCase
from hydrograd.oned.stepper import run
from hydrograd.params.transforms import parse_label, with_values

logger = logging.getLogger(__name__)

__all__ = ["run_sensitivity", "scalar_labels", "volume_jacobian"]


def scalar_labels(theta: dict[str, Any], labels: list[str]) -> list[str]:
    """Expand the labels of whole vector parameters into one label per entry.

    Args:
        theta: parameter dictionary.
        labels: active labels, e.g. ``["a_eh", "kappa_w"]``.

    Returns:
        Labels of scalars, e.g. ``["a_eh", "kappa_w[0]", "kappa_w[1]", "kappa_w[2]"]``.
    """
    expanded = []
    for label in labels:
        name, index = parse_label(label)
        if name not in theta:
            raise ValueError(f"Parameter {name!r} (label {label!r}) is not in theta")

        size = int(np.size(theta[name]))
        if index is not None or np.ndim(theta[name]) == 0:
            expanded.append(label)
        else:
            for entry in range(size):
                expanded.append(f"{name}[{entry}]")

    return expanded


def _values_of(theta: dict[str, Any], labels: list[str]) -> Array:
    """Physical values of scalar labels.

    Args:
        theta: parameter dictionary.
        labels: labels of scalars.

    Returns:
        Values, shape [P].
    """
    values = []
    for label in labels:
        name, index = parse_label(label)
        flat = jnp.reshape(jnp.asarray(theta[name], dtype=float), (-1,))
        values.append(flat[0 if index is None else index])

    return jnp.stack(values)


def volume_jacobian(
    case: OneDCase,
    theta: dict[str, Any],
    labels: list[str],
    times: np.ndarray,
    consolidation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surveyed volumes and their Jacobian with respect to scalar parameters.

    Args:
        case: the 1-D case.
        theta: evaluation point (all parameters, physical values).
        labels: labels of the active scalars, length P.
        times: survey times (s), shape [S].
        consolidation: include consolidation in the volumes.

    Returns:
        Tuple ``(values, volumes, jacobian)``: the parameter values, shape [P]; the
        volumes (m^3), shape [S, N]; and d V / d p, shape [S, N, P].
    """
    values = _values_of(theta, labels)

    def volumes_of(p: Array) -> Array:
        """Surveyed volumes for the values p of the active scalars."""
        parameters = with_values(theta, dict(zip(labels, p, strict=True)))
        result = run(case.config, case.network, parameters, case.forcing, case.bed0)
        return survey_volumes(result, parameters, case.config.sediment, times, consolidation)

    def volumes_and_jacobian(p: Array) -> tuple[Array, Array]:
        """Forward-mode Jacobian with the volumes as auxiliary output."""

        def with_aux(q: Array) -> tuple[Array, Array]:
            """Volumes twice: once to differentiate, once to return."""
            volumes = volumes_of(q)
            return volumes, volumes

        jacobian, volumes = jax.jacfwd(with_aux, has_aux=True)(p)
        return volumes, jacobian

    volumes, jacobian = jax.jit(volumes_and_jacobian)(values)
    return np.asarray(values), np.asarray(volumes), np.asarray(jacobian)


def run_sensitivity(ctrl: Control, case: OneDCase) -> dict[str, Any]:
    """Run the sensitivity analysis and write ``sensitivity_results.json``.

    Args:
        ctrl: run control.
        case: the 1-D case.

    Returns:
        Dict with ``"labels"``, ``"values"`` [P], ``"volumes"`` [S, N], ``"jacobian"``
        and ``"jacobian_log"`` [S, N, P].
    """
    o = ctrl.opt

    # Evaluation point: the case values, overridden by the run control.
    theta = with_values(case.theta, o("sensitivity_parameter_values", {}))
    labels = scalar_labels(theta, list(o("active_param_names")))
    time_end = case.forcing.time_end()
    times = survey_times(o("survey_times", None), time_end)
    consolidation = bool(o("survey_consolidation", True))

    start = time.perf_counter()
    values, volumes, jacobian = volume_jacobian(case, theta, labels, times, consolidation)
    logger.info(
        "1-D sensitivity Jacobian %s computed in %.2f s", jacobian.shape,
        time.perf_counter() - start,
    )

    # Scaled Jacobian, per unit of ln p.
    jacobian_log = jacobian * values[None, None, :]

    data = {
        "parameter_labels": labels,
        "parameter_values": values.tolist(),
        "survey_times": times.tolist(),
        "deposit_volume": volumes.tolist(),
        "jacobian": jacobian.tolist(),
        "jacobian_log": jacobian_log.tolist(),
    }
    (ctrl.case_dir / "sensitivity_results.json").write_text(json.dumps(data))
    logger.info("wrote sensitivity_results.json")

    return {
        "labels": labels,
        "values": values,
        "volumes": volumes,
        "jacobian": jacobian,
        "jacobian_log": jacobian_log,
    }
