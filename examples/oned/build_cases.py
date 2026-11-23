"""Build the example cases of the 1-D mode.

Three cases are written, each a folder with a case file (``case.json``) and run-control
files, ready for ``hydrograd run``:

``reservoir_delta``
    Forward simulation. A 20 km trapezoidal channel (80 m bottom width, slope 0.001) ends
    at a dam with an 18 m pool. The upper 2.5 km is a river with a non-erodible bed; the
    rest is the reservoir. Seasonal hydrograph between 100 and 250 m^3/s, three size
    classes (silt 0.02 mm, fine sand 0.1 mm, sand 0.5 mm), inflow load from the rating
    curve Q_t = a_r Q^2. Two years of daily steps: a delta of sand progrades into the
    pool, and the silt settles further downstream.

``reservoir_twin``
    Sensitivity analysis and inversion on the same reservoir with 2-day steps. The truth
    file (``oned_survey_truth.json``) holds the consolidated deposit volumes per section
    after two years of a run with a_eh = 0.05, kappa_w[0] = 1.5 and a_r = 2e-6, with 5 %
    noise. The case file starts from a_eh = 0.03, kappa_w[0] = 0.7 and a_r = 4e-6, and
    the inversion recovers the three values from the survey.

``confluence``
    Forward simulation on a network. Two tributaries (5 and 4 km, different widths and
    slopes) join at the head of a 10 km main stem with a 12 m pool at the dam; one year
    of daily steps with different hydrographs on the two branches.

Usage::

    python examples/oned/build_cases.py            (writes the folders next to this file)

The synthetic reservoir is also the first case of the comparison with SRH-1D
(``benchmarks/oned/``).

Units: SI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.apps.oned.observe import SurveyTruth, survey_volumes, write_truth
from hydrograd.io.oned_case import OneDCase, write_case
from hydrograd.oned import (
    Forcing,
    OneDConfig,
    default_parameters,
    network_from_positions,
    run,
)
from hydrograd.oned.synthetic import trapezoid_channel, uniform_bed, y_network
from hydrograd.sediment import SedimentConfig

HERE = Path(__file__).resolve().parent

DAY = 86400.0                               # s
YEAR = 365.25 * DAY                         # s
DIAMETERS = (2.0e-5, 1.0e-4, 5.0e-4)        # silt, fine sand, sand (m)

__all__ = ["build_all", "confluence_case", "reservoir_case", "write_control"]


def reservoir_case(years: float = 2.0, dt_days: float = 1.0, **parameters: Any) -> OneDCase:
    """The synthetic reservoir (see the module docstring).

    Args:
        years: length of the run (years).
        dt_days: step length (days).
        **parameters: parameter values that replace the defaults of this case, e.g.
            ``a_eh=0.03``.

    Returns:
        The case.
    """
    n_steps = round(years * 365.25 / dt_days)
    dt = dt_days * DAY

    # Channel: 41 sections 500 m apart; non-erodible bed in the upper 2.5 km.
    x, z, s = trapezoid_channel(length=20000.0, n_sections=41, slope=1.0e-3, bottom_width=80.0)
    bed0 = uniform_bed(z, s, n_classes=3, fractions=(0.2, 0.3, 0.5), fixed=x < 2500.0)

    # Seasonal hydrograph Q = 100 + 150 sin^2(pi t / year) (m^3/s); pool level 118 m.
    time_start = jnp.arange(n_steps) * dt
    Q_in = 100.0 + 150.0 * jnp.sin(math.pi * time_start / YEAR) ** 2
    forcing = Forcing(
        Q_in=Q_in,
        eta_outlet=jnp.full(n_steps, 118.0),
        dt=jnp.full(n_steps, dt),
    )

    # Parameters: rating curve a_r Q^2 with 60 % silt, 30 % fine sand and 10 % sand; the
    # silt settles 1.5 times faster than its Stokes velocity (flocculation).
    theta = default_parameters(3)
    theta["tau_cd"] = jnp.asarray(1.0)
    theta["a_r"] = jnp.asarray(2.0e-6)
    theta["p_in"] = jnp.asarray([0.6, 0.3, 0.1])
    theta["kappa_w"] = jnp.asarray([1.5, 1.0, 1.0])
    for name, value in parameters.items():
        theta[name] = jnp.asarray(value, dtype=float)

    return OneDCase(
        config=OneDConfig(sediment=SedimentConfig(diameters=DIAMETERS)),
        network=network_from_positions(x),
        theta=theta,
        forcing=forcing,
        bed0=bed0,
    )


def confluence_case(years: float = 1.0) -> OneDCase:
    """Two tributaries joining a main stem with a pool (see the module docstring).

    Args:
        years: length of the run (years), in daily steps.

    Returns:
        The case.
    """
    n_steps = round(years * 365.25)

    # Branch A: 5 km, 40 m wide, slope 0.0012. Branch B: 4 km, 30 m wide, slope 0.0015.
    y = y_network(
        main_length=10000.0,
        n_main=21,
        branch_lengths=(5000.0, 4000.0),
        n_branch=(10, 8),
        branch_slopes=(1.2e-3, 1.5e-3),
        bottom_width=60.0,
        branch_widths=(40.0, 30.0),
    )

    # The upper part of each branch (more than 2 km above the junction) is non-erodible.
    bed0 = uniform_bed(
        y.z, y.s, n_classes=3, fractions=(0.2, 0.3, 0.5), fixed=y.distance_to_junction > 2000.0,
    )

    # Hydrographs of the two branches, half a season apart (m^3/s), shape [T, 2].
    time_start = jnp.arange(n_steps) * DAY
    Q_a = 60.0 + 90.0 * jnp.sin(math.pi * time_start / YEAR) ** 2
    Q_b = 30.0 + 60.0 * jnp.cos(math.pi * time_start / YEAR) ** 2
    forcing = Forcing(
        Q_in=jnp.stack([Q_a, Q_b], axis=1),
        eta_outlet=jnp.full(n_steps, 112.0),
        dt=jnp.full(n_steps, DAY),
    )

    theta = default_parameters(3)
    theta["tau_cd"] = jnp.asarray(1.0)
    theta["a_r"] = jnp.asarray(3.0e-6)
    theta["p_in"] = jnp.asarray([0.6, 0.3, 0.1])

    return OneDCase(
        config=OneDConfig(sediment=SedimentConfig(diameters=DIAMETERS)),
        network=y.network,
        theta=theta,
        forcing=forcing,
        bed0=bed0,
    )


def write_control(path: Path, mode_flag: str, options: dict[str, Any]) -> None:
    """Write a run-control file for the 1-D case ``case.json`` in the same folder.

    Args:
        path: the run-control file to write.
        mode_flag: the ``bPerform_*`` flag to set.
        options: option blocks by name, e.g. ``{"inversion_options": {...}}``.
    """
    control = {
        "control_variables": {"model": "1D", "case_file_name": "case.json", mode_flag: True},
    }
    control.update(options)
    path.write_text(json.dumps(control, indent=2) + "\n")


def _build_reservoir_delta(folder: Path) -> None:
    """Write the forward case of the reservoir.

    Args:
        folder: output folder.
    """
    folder.mkdir(parents=True, exist_ok=True)
    write_case(folder / "case.json", reservoir_case(years=2.0, dt_days=1.0))

    # Surveys after one and two years.
    options = {
        "forward_simulation_options": {
            "survey_times": [YEAR, 2.0 * YEAR],
            "survey_consolidation": True,
        }
    }
    write_control(folder / "run_control.json", "bPerform_Forward_Simulation", options)


def _build_reservoir_twin(folder: Path) -> None:
    """Write the twin case: noisy truth file, starting case, sensitivity and inversion.

    Args:
        folder: output folder.
    """
    folder.mkdir(parents=True, exist_ok=True)

    # Truth run and its survey with 5 % noise (of the root-mean-square volume).
    truth_case = reservoir_case(years=2.0, dt_days=2.0)

    def run_truth(theta):
        """The truth run as a function of the parameters."""
        return run(
            truth_case.config, truth_case.network, theta, truth_case.forcing, truth_case.bed0,
        )

    result = jax.jit(run_truth)(truth_case.theta)
    times = np.asarray([float(result.time_end[-1])])
    volumes = np.asarray(
        survey_volumes(result, truth_case.theta, truth_case.config.sediment, times, True)
    )
    rng = np.random.default_rng(0)
    rms = np.sqrt(np.mean(volumes**2, axis=1, keepdims=True))
    noisy = volumes + 0.05 * rms * rng.standard_normal(volumes.shape)
    write_truth(folder / "oned_survey_truth.json", SurveyTruth(times, noisy, True), 0.05)

    # The case file holds the starting values, not the truth.
    start = {"a_eh": 0.03, "kappa_w": [0.7, 1.0, 1.0], "a_r": 4.0e-6}
    write_case(folder / "case.json", reservoir_case(years=2.0, dt_days=2.0, **start))

    labels = ["a_eh", "kappa_w[0]", "a_r"]
    sensitivity = {"sensitivity_analysis_options": {"active_param_names": labels}}
    write_control(
        folder / "run_control_sensitivity.json", "bPerform_Sensitivity_Analysis", sensitivity,
    )

    # Two Adam stages on the unconstrained (logarithmic) variables.
    inversion = {
        "inversion_options": {
            "active_param_names": labels,
            "inversion_truth_file_name": "oned_survey_truth.json",
            "inversion_survey_relative_sigma": 0.05,
            "inversion_optimizers": ["Adam", "Adam"],
            "inversion_learning_rates": [0.1, 0.01],
            "inversion_max_iterations": [150, 100],
            "inversion_abs_tols": [0.0, 0.0],
            "inversion_save_frequency": 50,
        }
    }
    write_control(folder / "run_control.json", "bPerform_Inversion", inversion)


def _build_confluence(folder: Path) -> None:
    """Write the forward case of the confluence.

    Args:
        folder: output folder.
    """
    folder.mkdir(parents=True, exist_ok=True)
    write_case(folder / "case.json", confluence_case(years=1.0))
    write_control(folder / "run_control.json", "bPerform_Forward_Simulation", {})


def build_all(root: Path = HERE) -> None:
    """Write the three example cases.

    Args:
        root: folder in which the case folders are created.
    """
    _build_reservoir_delta(root / "reservoir_delta")
    _build_reservoir_twin(root / "reservoir_twin")
    _build_confluence(root / "confluence")


if __name__ == "__main__":
    build_all()
    print(f"wrote the example cases in {HERE}")
