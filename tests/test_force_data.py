"""Tests of force observations as calibration data (``apps.force_data``).

On the Savannah River case (steady flow), material zone 2 is given rigid emergent
vegetation (a = 1 1/m) with the true drag coefficient C_D = 1.2:

1. ``load_force_data`` builds the region weights from material IDs, boxes and cell lists,
   and masks unobserved components.
2. An inversion of ``veg_CD`` that uses only the observed drag force on zone 2 (the
   water-level and velocity losses switched off) recovers C_D = 1.2 from C_D = 0.6. This
   exercises the run-control block, the observation file, the loss term in the
   inversion mode and its implicit-function-theorem gradient through the steady solve.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from conftest import requires_jl

from hydrograd.apps.force_data import load_force_data
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.physics.vegetation import vegetation_force
from hydrograd.solvers.final import final_state

pytestmark = requires_jl

CD_TRUE = 1.2           # true drag coefficient of zone 2 (-)
A_ZONE2 = 1.0           # frontal area per unit volume in zone 2 (1/m)


def _vegetated_savannah(case_copy):
    """Savannah inversion example with vegetation in zone 2 and a steady forward control.

    Args:
        case_copy: fixture that copies a Hydrograd.jl example into a temporary folder.

    Returns:
        (case folder, inversion control dict, forward control dict).
    """
    case_dir = case_copy("inversion/ManningN_inversion/Savannah_River")
    control_text = (case_dir / "run_control.json").read_text()
    control = json.loads(re.sub(r",(\s*[}\]])", r"\1", control_text))
    control["vegetation"] = {"a": {"2": A_ZONE2}, "CD": {"2": CD_TRUE}}

    # Steady forward control (same files and initial condition as the inversion).
    forward = json.loads(json.dumps(control))
    forward["control_variables"].update(bPerform_Inversion=False, bPerform_Forward_Simulation=True)
    forward["forward_simulation_options"] = {
        "forward_simulation_ManningN_option": "constant",
        "forward_simulation_solver": "SciML",
        "forward_simulation_ode_solver": "SteadyNewton()",
        "forward_simulation_adaptive": True,
        "forward_simulation_nSave": 1,
        "forward_simulation_initial_condition_options": "from_file",
        "forward_simulation_initial_condition_file_name": "forward_simulation_initial_condition.json",
        "forward_simulation_save_solution_truth_file_name": "steady_truth.json",
    }
    return case_dir, control, forward


def test_load_force_data_regions(case_copy) -> None:
    """Regions from materials, a box and a cell list; masks follow the observed keys."""
    # Setup: an observation file with the three region kinds.
    case_dir, control, _ = _vegetated_savannah(case_copy)
    observations = {"regions": [
        {"name": "zone2", "materials": [2], "Fx": 1.0},
        {"name": "box", "box": [-1e9, 1e9, -1e9, 1e9], "Fx": 2.0, "Fy": 0.5},
        {"name": "cells", "cells": [1, 2, 3], "Fy": 0.1},
    ]}
    (case_dir / "forces.json").write_text(json.dumps(observations))
    control["force_data"] = {"file": "forces.json", "sigma_N": 0.2, "weight": 2.0}
    (case_dir / "run_control.json").write_text(json.dumps(control))

    # Run.
    case = setup_case(read_control(case_dir / "run_control.json"))
    data = load_force_data(case)

    # Compare: region sizes, observed values and masks.
    weights = np.asarray(data.weights)
    material = np.asarray(case.model.mat_id)
    assert data.names == ("zone2", "box", "cells")
    assert weights[0].sum() == np.sum(material == 2)
    assert weights[1].sum() == case.model.n_cells
    assert weights[2].sum() == 3
    np.testing.assert_array_equal(np.asarray(data.mask), [[1, 0], [1, 1], [0, 1]])
    assert data.sigma == 0.2 and data.factor == 2.0


@pytest.mark.slow
def test_inversion_from_force_only(case_copy) -> None:
    """veg_CD of zone 2 from the zone-2 drag force alone: C_D = 1.2 recovered (rtol 1e-4)."""
    from hydrograd.apps.forward import run_forward
    from hydrograd.apps.inversion import run_inversion

    # Setup: steady forward run at the true C_D, and the observed force on zone 2.
    case_dir, control, forward = _vegetated_savannah(case_copy)
    (case_dir / "run_control_forward.json").write_text(json.dumps(forward))
    forward_case = setup_case(read_control(case_dir / "run_control_forward.json"))
    run_forward(forward_case)

    U, converged = final_state(forward_case.model, {}, forward_case.U0, forward_case.ode)
    assert bool(converged)
    zone2 = (np.asarray(forward_case.model.mat_id) == 2).astype(float)[None, :]
    observed_fx = float(vegetation_force(forward_case.model, {}, U, zone2)[0, 0])
    assert observed_fx > 0.0
    observations = {"regions": [{"name": "zone2", "materials": [2], "Fx": observed_fx}]}
    (case_dir / "forces.json").write_text(json.dumps(observations))

    # Inversion of veg_CD with the force term only.
    options = control["inversion_options"]
    options.update(
        active_param_names=["veg_CD"],
        inversion_veg_CD_initial_values=[1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
        inversion_bInversion_WSE_loss=False,
        inversion_bInversion_uv_loss=False,
        inversion_bInversion_bound_loss=False,
        inversion_truth_file_name="steady_truth.json",
        inversion_optimizers=["LBFGS"],
        inversion_learning_rates=[1.0],
        inversion_max_iterations=[30],
        inversion_abs_tols=[1e-14],
        inversion_save_frequency=100,
    )
    options["inversion_ode_solver_options"]["ode_solver"] = "SteadyNewton()"
    control["force_data"] = {"file": "forces.json", "sigma_N": 0.01 * observed_fx}
    (case_dir / "run_control.json").write_text(json.dumps(control))

    # Run.
    result = run_inversion(setup_case(read_control(case_dir / "run_control.json")))

    # Compare: the zone-2 drag coefficient.
    cd_inverted = np.asarray(result["p"])
    assert cd_inverted[2] == pytest.approx(CD_TRUE, rel=1e-4)
