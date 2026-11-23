"""Forward simulation of a 1-D case.

Runs the model of the case file with its parameter values (no differentiation) and writes
the results and the surveys.

Options (``forward_simulation_options`` in ``run_control.json``; all optional)
--------------------------------------------------------------------------
::

    "forward_simulation_save_file_name": "oned_forward_results.json",
    "forward_simulation_save_solution_truth_file_name": "oned_survey_truth.json",
    "survey_times": [t1, t2, ...],        survey times (s); default: the end of the run
    "survey_consolidation": true,         consolidate the deposits up to each survey
    "survey_noise_relative": 0.0,         Gaussian noise added to the truth file, with a
                                          standard deviation of this fraction of the
                                          root-mean-square volume of each survey
    "survey_noise_seed": 0

Files written (in the case directory)
-------------------------------------
* the results file (JSON, arrays as nested lists in row-major order): ``time_end`` [T]
  (s); ``thalweg_initial`` and ``thalweg_final`` [N] (m); ``eta_final`` [N] (m);
  ``deposit_volume_final`` [N] (m^3, bulk at the initial density); ``p_active_final``
  [K][N]; ``load_in_total`` and ``load_out_total`` [T_rec] (m^3/s of solids, summed over
  the size classes and headwaters); ``trap_efficiency`` [T_rec];
* ``oned_forward_results.npz``: all arrays of the run (bed tables, deposition per step,
  class and section, histories);
* the truth file with the surveys (see :mod:`hydrograd.apps.oned.observe`).

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

from hydrograd.apps.oned.observe import (
    SurveyTruth,
    survey_times,
    survey_volumes,
    write_truth,
)
from hydrograd.io.control import Control
from hydrograd.io.oned_case import OneDCase
from hydrograd.oned.stepper import RunResult, run
from hydrograd.sediment.observation import bulk_volume_uncompacted

logger = logging.getLogger(__name__)

__all__ = ["run_forward", "simulate"]


def simulate(case: OneDCase, theta: dict[str, Any] | None = None) -> RunResult:
    """Run the model of a case, jit-compiled.

    Args:
        case: the 1-D case.
        theta: parameter values; None uses those of the case.

    Returns:
        The run result.
    """

    def run_with(parameters: dict[str, Any]) -> RunResult:
        """The run as a function of the parameters."""
        return run(case.config, case.network, parameters, case.forcing, case.bed0)

    return jax.jit(run_with)(case.theta if theta is None else theta)


def run_forward(ctrl: Control, case: OneDCase) -> dict[str, Any]:
    """Run the forward simulation and write the results and the truth file.

    Args:
        ctrl: run control.
        case: the 1-D case.

    Returns:
        Dict with the run ``"result"`` and the surveys ``"truth"`` (without noise).
    """
    o = ctrl.opt
    case_dir = ctrl.case_dir
    sediment = case.config.sediment

    # ------------------------------------------------------------------
    # 1. Run
    # ------------------------------------------------------------------
    start = time.perf_counter()
    result = simulate(case)
    result.deposition.block_until_ready()
    logger.info(
        "1-D forward run of %d steps finished in %.2f s (with compilation)",
        case.forcing.n_steps, time.perf_counter() - start,
    )

    if not bool(jnp.all(jnp.isfinite(result.bed.z))):
        logger.error("the run produced non-finite bed elevations")

    # ------------------------------------------------------------------
    # 2. Surveys
    # ------------------------------------------------------------------
    times = survey_times(o("survey_times", None), result.time_end)
    consolidation = bool(o("survey_consolidation", True))
    volumes = np.asarray(survey_volumes(result, case.theta, sediment, times, consolidation))
    truth = SurveyTruth(times=times, volumes=volumes, consolidation=consolidation)

    # Optional noise on the surveys written to the truth file (twin experiments).
    noise_relative = float(o("survey_noise_relative", 0.0))
    written = truth
    if noise_relative > 0.0:
        rng = np.random.default_rng(int(o("survey_noise_seed", 0)))
        rms = np.sqrt(np.mean(volumes**2, axis=1, keepdims=True))
        noisy = volumes + noise_relative * rms * rng.standard_normal(volumes.shape)
        written = SurveyTruth(times=times, volumes=noisy, consolidation=consolidation)

    truth_name = o("forward_simulation_save_solution_truth_file_name", "oned_survey_truth.json")
    write_truth(case_dir / truth_name, written, noise_relative)
    logger.info("wrote %s (%d survey(s))", truth_name, len(times))

    # ------------------------------------------------------------------
    # 3. Results files
    # ------------------------------------------------------------------
    deposit_volume = bulk_volume_uncompacted(result.deposition, case.theta, sediment)
    summary = {
        "time_end": np.asarray(result.time_end).tolist(),
        "thalweg_initial": np.asarray(jnp.min(case.bed0.z, axis=1)).tolist(),
        "thalweg_final": np.asarray(jnp.min(result.bed.z, axis=1)).tolist(),
        "eta_final": np.asarray(result.eta[-1]).tolist(),
        "deposit_volume_final": np.asarray(deposit_volume).tolist(),
        "p_active_final": np.asarray(result.bed.p_active).tolist(),
        "load_in_total": np.asarray(jnp.sum(result.load_in, axis=(1, 2))).tolist(),
        "load_out_total": np.asarray(jnp.sum(result.load_out, axis=1)).tolist(),
        "trap_efficiency": np.asarray(result.trap_efficiency).tolist(),
    }
    results_name = o("forward_simulation_save_file_name", "oned_forward_results.json")
    (case_dir / results_name).write_text(json.dumps(summary))

    arrays = {
        "time_end": result.time_end,
        "z_initial": case.bed0.z,
        "z_final": result.bed.z,
        "s": result.bed.s,
        "p_active_final": result.bed.p_active,
        "p_substrate_final": result.bed.p_substrate,
        "substrate_thickness_final": result.bed.substrate_thickness,
        "deposition": result.deposition,
        "thalweg": result.thalweg,
        "eta": result.eta,
        "load_in": result.load_in,
        "load_out": result.load_out,
        "trap_efficiency": result.trap_efficiency,
    }
    if result.z_snapshots is not None:
        arrays["z_snapshots"] = result.z_snapshots
    np.savez(case_dir / "oned_forward_results.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    logger.info("wrote %s and oned_forward_results.npz", results_name)

    return {"result": result, "truth": truth}
