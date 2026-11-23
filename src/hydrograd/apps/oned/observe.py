"""Observations of a 1-D run: surveyed deposit volumes per cross-section.

A repeat bathymetric survey of a reservoir gives, for each cross-section, the bulk volume
of sediment deposited since the previous survey. This is the observable of the 1-D
sensitivity analysis and inversion:

    V[s, i] = bulk deposit volume at section i at survey time t_s           (m^3),

from the solids volumes the model deposits per step and size class, either with the
consolidation of the deposits up to the survey (Lane and Koelzer) or at the initial
density (see :mod:`hydrograd.sediment.observation`). Negative values are net erosion.

Truth file
----------
The forward mode writes the surveys of a run as a JSON file, which the inversion reads
as the observations::

    {
      "survey_times": [S],              (s)
      "consolidation": true,
      "deposit_volume": [S][N],         (m^3, bulk)
      "noise_relative": 0.0             standard deviation of the added noise, relative
                                        to the root-mean-square volume of each survey
    }

Units: SI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.oned.stepper import RunResult
from hydrograd.sediment.config import SedimentConfig
from hydrograd.sediment.observation import consolidated_volume

__all__ = ["SurveyTruth", "load_truth", "survey_times", "survey_volumes", "write_truth"]

# Differentiable parameters: name -> array of physical values.
Theta = dict[str, Any]


@dataclass(frozen=True)
class SurveyTruth:
    """Observed surveys.

    Attributes:
        times: survey times (s), shape [S].
        volumes: bulk deposit volume per survey and section (m^3), shape [S, N].
        consolidation: True if the volumes include the consolidation of the deposits.
    """

    times: np.ndarray
    volumes: np.ndarray
    consolidation: bool


def survey_times(option: Any, time_end: Array) -> np.ndarray:
    """Survey times from the run-control option.

    Args:
        option: list of times (s), or None for one survey at the end of the run.
        time_end: time at the end of each step (s), shape [T].

    Returns:
        Survey times (s), shape [S].
    """
    if option is None:
        return np.asarray([float(time_end[-1])])
    return np.atleast_1d(np.asarray(option, dtype=float))


def survey_volumes(
    result: RunResult,
    theta: Theta,
    config: SedimentConfig,
    times: np.ndarray,
    consolidation: bool,
) -> Array:
    """Bulk deposit volume per section at each survey time.

    Args:
        result: result of a 1-D run (deposited solids volumes per step).
        theta: parameters (``gamma1`` and, with consolidation, ``K_cons``).
        config: sediment settings.
        times: survey times (s), shape [S].
        consolidation: include the consolidation of the deposits up to each survey.

    Returns:
        Volumes (m^3), shape [S, N].
    """
    volumes = []
    for time in times:
        if consolidation:
            volume = consolidated_volume(
                result.deposition, result.time_end, float(time), theta, config,
            )
        else:
            # Deposits of the steps that ended before the survey, at the initial density.
            counted = (result.time_end <= float(time)).astype(result.deposition.dtype)
            solid_fraction = theta["gamma1"] / config.constants.rho_s
            bulk = result.deposition / solid_fraction[None, :, None]
            volume = jnp.sum(bulk * counted[:, None, None], axis=(0, 1))
        volumes.append(volume)

    return jnp.stack(volumes)


def write_truth(
    path: str | Path, truth: SurveyTruth, noise_relative: float = 0.0,
) -> None:
    """Write the surveys as a truth file.

    Args:
        path: output ``.json`` file.
        truth: the surveys.
        noise_relative: relative standard deviation of the noise that was added (-),
            stored for reference.
    """
    data = {
        "survey_times": np.asarray(truth.times).tolist(),
        "consolidation": bool(truth.consolidation),
        "deposit_volume": np.asarray(truth.volumes).tolist(),
        "noise_relative": float(noise_relative),
    }
    Path(path).write_text(json.dumps(data))


def load_truth(path: str | Path) -> SurveyTruth:
    """Read a truth file.

    Args:
        path: the ``.json`` truth file.

    Returns:
        The surveys.
    """
    data = json.loads(Path(path).read_text())
    return SurveyTruth(
        times=np.asarray(data["survey_times"], dtype=float),
        volumes=np.asarray(data["deposit_volume"], dtype=float),
        consolidation=bool(data["consolidation"]),
    )
