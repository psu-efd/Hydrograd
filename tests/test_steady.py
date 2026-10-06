"""Steady Newton solve with IFT gradients (``ode_solver = "SteadyNewton()"``)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import requires_jl

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.final import final_state, newton_settings
from hydrograd.solvers.jacobian import (
    JacobianPattern,
    dense_jacobian,
    full_jacobian,
    linearize,
)
from hydrograd.solvers.steady import newton_solve, steady_jacobian, steady_state

pytestmark = requires_jl

CASES = ["forward_simulation/oneD_channel_with_bump", "forward_simulation/Savannah_River"]


def _steady_case(case_copy, rel):
    case = setup_case(read_control(case_copy(rel) / "run_control.json"))
    return dataclasses.replace(case, ode=dataclasses.replace(case.ode, solver="SteadyNewton()"))


@pytest.mark.parametrize("rel", CASES)
def test_sparse_jacobian_is_exact(case_copy, rel: str) -> None:
    case = _steady_case(case_copy, rel)
    m = case.model
    U = case.U0 + 0.01 * jnp.sin(jnp.arange(3 * m.n_cells).reshape(3, -1))
    comp, A, C = jax.jit(lambda U: linearize(m, {}, U))(U)
    J = full_jacobian(m, np.asarray(comp), np.asarray(A), np.asarray(C))
    np.testing.assert_allclose(J, dense_jacobian(m, {}, U), atol=1e-12)
    # Independent of the inlet width; face stencil at first order, two-cell stencil at second order.
    assert JacobianPattern(m).n_colors <= (12 if m.order == 1 else 40)


@pytest.mark.parametrize("rel", CASES)
def test_newton_converges(case_copy, rel: str) -> None:
    case = _steady_case(case_copy, rel)
    U, it, rn = jax.jit(lambda U0: newton_solve(case.model, {}, U0, newton_settings(case.ode)))(case.U0)
    assert float(rn) < 1e-10 and int(it) < 100


def test_ift_gradient_and_sensitivity_match_finite_differences(case_copy) -> None:
    case = _steady_case(case_copy, "forward_simulation/Savannah_River")
    m, s = case.model, newton_settings(case.ode)
    nz = jnp.asarray(m.n_zone)
    w = jnp.asarray(np.random.default_rng(0).normal(size=case.U0.shape))
    L = jax.jit(lambda p: jnp.sum(w * steady_state(m, {"n_zone": p}, case.U0, s)))
    g = np.asarray(jax.jit(jax.grad(L))(nz))
    U, ok = final_state(m, {"n_zone": nz}, case.U0, case.ode)
    J = steady_jacobian(m, {"n_zone": nz}, U)
    eps = 1e-6
    for i in (1, 3):
        e = jnp.zeros_like(nz).at[i].set(eps)
        fd = (float(L(nz + e)) - float(L(nz - e))) / (2 * eps)
        assert abs(g[i] - fd) <= 1e-6 * abs(fd)
        # Sensitivity Jacobian column against the IFT gradient: w . dU/dn_i
        np.testing.assert_allclose(float(np.asarray(w).reshape(-1) @ J[:, i]), g[i], rtol=1e-10)


@pytest.mark.slow
def test_steady_inversion_recovers_n_without_bias(case_copy) -> None:
    """Observations from a steady forward run; L-BFGS inversion recovers the zone n exactly
    (consistent velocity definition in truth file and loss: no loss floor, no bias)."""
    import csv
    import json
    import re

    from hydrograd.apps.forward import run_forward
    from hydrograd.apps.inversion import run_inversion

    d = case_copy("inversion/ManningN_inversion/Savannah_River")
    c = json.loads(re.sub(r",(\s*[}\]])", r"\1", (d / "run_control.json").read_text()))
    fwd = json.loads(json.dumps(c))
    fwd["control_variables"].update(bPerform_Inversion=False, bPerform_Forward_Simulation=True)
    fwd["forward_simulation_options"] = {
        "forward_simulation_ManningN_option": "constant", "forward_simulation_solver": "SciML",
        "forward_simulation_ode_solver": "SteadyNewton()", "forward_simulation_adaptive": True,
        "forward_simulation_nSave": 1, "forward_simulation_initial_condition_options": "from_file",
        "forward_simulation_initial_condition_file_name": "forward_simulation_initial_condition.json",
        "forward_simulation_save_solution_truth_file_name": "steady_truth.json"}
    (d / "run_control_forward.json").write_text(json.dumps(fwd))
    run_forward(setup_case(read_control(d / "run_control_forward.json")))
    o = c["inversion_options"]
    o.update(inversion_truth_file_name="steady_truth.json", inversion_optimizers=["LBFGS"],
             inversion_learning_rates=[1.0], inversion_max_iterations=[40], inversion_abs_tols=[1e-9],
             inversion_ManningN_initial_values=[0.03, 0.044, 0.055, 0.033, 0.0495, 0.055],
             inversion_save_frequency=100)
    o["inversion_ode_solver_options"]["ode_solver"] = "SteadyNewton()"
    (d / "run_control.json").write_text(json.dumps(c))
    p = run_inversion(setup_case(read_control(d / "run_control.json")))["p"]
    truth = json.loads((d / "steady_truth.json").read_text())["ManningN_zone_values_truth"]
    np.testing.assert_allclose(p[1:], truth[1:], atol=1e-7)   # zone 0 has no cells
    losses = [float(r[1]) for r in list(csv.reader(open(d / "inversion_loss_history.csv")))[1:]]
    assert losses[-1] < 1e-14


def test_steady_sensitivity_app_matches_finite_differences(case_copy) -> None:
    """run_sensitivity with SteadyNewton (regression: UnboundLocalError in the steady branch)."""
    from hydrograd.apps.sensitivity import final_state_jacobian

    rel = "sensitivity_analysis/ManningN/oneD_channel_with_bump"
    d = case_copy(rel)
    case = setup_case(read_control(d / "run_control.json"))
    case = dataclasses.replace(case, ode=dataclasses.replace(case.ode, solver="SteadyNewton()"))
    p = jnp.asarray(np.asarray(case.ctrl.opt("sensitivity_ManningN_values"), dtype=float))
    J = final_state_jacobian(case, "ManningN", p)["J"]
    s = newton_settings(case.ode)
    f = jax.jit(lambda pp: steady_state(case.model, {"n_zone": pp}, case.U0, s).reshape(-1))
    e = jnp.zeros_like(p).at[2].set(1e-6)
    fd = (np.asarray(f(p + e)) - np.asarray(f(p - e))) / 2e-6
    assert np.linalg.norm(J[:, 2] - fd) <= 1e-4 * np.linalg.norm(fd)
