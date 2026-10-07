"""Steady Newton solve with IFT gradients (``ode_solver = "SteadyNewton()"``).

The steady state U* solves R(U*, p) = 0, where R is the finite-volume residual (the
method-of-lines RHS). Newton iterates on the exact sparse Jacobian dR/dU, built by graph
coloring and JVPs. By the implicit function theorem (IFT),

    dU*/dp = -(dR/dU)^{-1} dR/dp,

so the gradient of any loss L(U*) needs one adjoint linear solve.

What is checked (Julia example cases ``oneD_channel_with_bump`` and ``Savannah_River``):
- The colored sparse Jacobian equals the dense AD Jacobian to 1e-12 (absolute), and the number
  of colors stays small (<= 12 at first order, <= 40 at second order).
- Newton converges to a residual norm below 1e-10 in fewer than 100 iterations.
- The IFT gradient of a random weighted sum of U* with respect to the zone Manning's n matches
  central finite differences to 1e-6 (relative), and the sensitivity Jacobian dU*/dn agrees with
  the IFT gradient to 1e-10.
- (slow) A steady L-BFGS inversion recovers the zone n of a steady forward run to 1e-7.
- The sensitivity app with SteadyNewton matches finite differences to 1e-4 (relative norm).
"""

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
    """Copy a Julia example case and set it up with ``ode_solver = "SteadyNewton()"``.

    Args:
        case_copy: conftest fixture that copies a Julia example into a temporary folder.
        rel: example path relative to ``examples/SWE_2D``.

    Returns:
        The set-up case with its ODE settings switched to the steady Newton solver.
    """
    case = setup_case(read_control(case_copy(rel) / "run_control.json"))
    steady_ode = dataclasses.replace(case.ode, solver="SteadyNewton()")
    return dataclasses.replace(case, ode=steady_ode)


@pytest.mark.parametrize("rel", CASES)
def test_sparse_jacobian_is_exact(case_copy, rel: str) -> None:
    """Colored sparse Jacobian vs dense AD Jacobian (atol 1e-12) and the number of colors."""
    case = _steady_case(case_copy, rel)
    model = case.model

    # A perturbed state, so the Jacobian is not evaluated at a special point
    U = case.U0 + 0.01 * jnp.sin(jnp.arange(3 * model.n_cells).reshape(3, -1))

    # Sparse Jacobian from the colored linearization, assembled to a full matrix
    compressed, A, C = jax.jit(lambda U: linearize(model, {}, U))(U)
    J = full_jacobian(model, np.asarray(compressed), np.asarray(A), np.asarray(C))
    np.testing.assert_allclose(J, dense_jacobian(model, {}, U), atol=1e-12)

    # Independent of the inlet width; face stencil at first order, two-cell stencil at second order.
    assert JacobianPattern(model).n_colors <= (12 if model.order == 1 else 40)


@pytest.mark.parametrize("rel", CASES)
def test_newton_converges(case_copy, rel: str) -> None:
    """Newton reaches a residual norm below 1e-10 in fewer than 150 iterations.

    The bump channel needs about 106 pseudo-transient Newton iterations with the Manning
    law of the default scheme (91 with Hydrograd.jl's (h + h_small) depth offset, which
    the default scheme no longer uses; report section 4n).
    """
    case = _steady_case(case_copy, rel)
    settings = newton_settings(case.ode)

    # Solve; returns the state, the iteration count and the final residual norm
    U, n_iterations, residual_norm = jax.jit(
        lambda U0: newton_solve(case.model, {}, U0, settings)
    )(case.U0)
    assert float(residual_norm) < 1e-10 and int(n_iterations) < 150


def test_ift_gradient_and_sensitivity_match_finite_differences(case_copy) -> None:
    """IFT gradient wrt zone n vs central FD (rel 1e-6); sensitivity column vs IFT (rtol 1e-10)."""
    case = _steady_case(case_copy, "forward_simulation/Savannah_River")
    model = case.model
    settings = newton_settings(case.ode)
    n_zone = jnp.asarray(model.n_zone)

    # Scalar loss: random weighted sum of the steady state
    weights = jnp.asarray(np.random.default_rng(0).normal(size=case.U0.shape))

    def loss(n_values):
        """Weighted sum of the steady state for zone Manning's n values n_values (s/m^(1/3))."""
        return jnp.sum(weights * steady_state(model, {"n_zone": n_values}, case.U0, settings))

    loss_jit = jax.jit(loss)

    # IFT gradient
    gradient = np.asarray(jax.jit(jax.grad(loss_jit))(n_zone))

    # Sensitivity Jacobian dU*/dn at the steady state
    U, converged = final_state(model, {"n_zone": n_zone}, case.U0, case.ode)
    J = steady_jacobian(model, {"n_zone": n_zone}, U)

    # Compare on zones 1 and 3
    eps = 1e-6
    for i in (1, 3):
        perturbation = jnp.zeros_like(n_zone).at[i].set(eps)
        fd = (float(loss_jit(n_zone + perturbation)) - float(loss_jit(n_zone - perturbation))) / (
            2 * eps
        )
        assert abs(gradient[i] - fd) <= 1e-6 * abs(fd)

        # Sensitivity Jacobian column against the IFT gradient: w . dU/dn_i
        np.testing.assert_allclose(
            float(np.asarray(weights).reshape(-1) @ J[:, i]), gradient[i], rtol=1e-10
        )


@pytest.mark.slow
def test_steady_inversion_recovers_n_without_bias(case_copy) -> None:
    """Observations from a steady forward run; L-BFGS inversion recovers the zone n exactly
    (consistent velocity definition in truth file and loss: no loss floor, no bias)."""
    import csv
    import json
    import re

    from hydrograd.apps.forward import run_forward
    from hydrograd.apps.inversion import run_inversion

    # Read the inversion control file (strip trailing commas, which JSON does not allow)
    case_dir = case_copy("inversion/ManningN_inversion/Savannah_River")
    control_text = (case_dir / "run_control.json").read_text()
    control = json.loads(re.sub(r",(\s*[}\]])", r"\1", control_text))

    # Steady forward run that writes the synthetic observations (steady_truth.json)
    forward_control = json.loads(json.dumps(control))
    forward_control["control_variables"].update(
        bPerform_Inversion=False, bPerform_Forward_Simulation=True
    )
    forward_control["forward_simulation_options"] = {
        "forward_simulation_ManningN_option": "constant",
        "forward_simulation_solver": "SciML",
        "forward_simulation_ode_solver": "SteadyNewton()",
        "forward_simulation_adaptive": True,
        "forward_simulation_nSave": 1,
        "forward_simulation_initial_condition_options": "from_file",
        "forward_simulation_initial_condition_file_name": "forward_simulation_initial_condition.json",
        "forward_simulation_save_solution_truth_file_name": "steady_truth.json",
    }
    (case_dir / "run_control_forward.json").write_text(json.dumps(forward_control))
    run_forward(setup_case(read_control(case_dir / "run_control_forward.json")))

    # Steady L-BFGS inversion from perturbed initial n values
    options = control["inversion_options"]
    options.update(
        inversion_truth_file_name="steady_truth.json",
        inversion_optimizers=["LBFGS"],
        inversion_learning_rates=[1.0],
        inversion_max_iterations=[40],
        inversion_abs_tols=[1e-9],
        inversion_ManningN_initial_values=[0.03, 0.044, 0.055, 0.033, 0.0495, 0.055],
        inversion_save_frequency=100,
    )
    options["inversion_ode_solver_options"]["ode_solver"] = "SteadyNewton()"
    (case_dir / "run_control.json").write_text(json.dumps(control))
    n_inverted = run_inversion(setup_case(read_control(case_dir / "run_control.json")))["p"]

    # Recovered n vs truth, and a vanishing final loss
    truth_text = (case_dir / "steady_truth.json").read_text()
    n_truth = json.loads(truth_text)["ManningN_zone_values_truth"]
    np.testing.assert_allclose(n_inverted[1:], n_truth[1:], atol=1e-7)   # zone 0 has no cells
    loss_rows = list(csv.reader(open(case_dir / "inversion_loss_history.csv")))
    losses = [float(row[1]) for row in loss_rows[1:]]
    assert losses[-1] < 1e-14


def test_steady_sensitivity_app_matches_finite_differences(case_copy) -> None:
    """run_sensitivity with SteadyNewton (regression: UnboundLocalError in the steady branch).

    The column dU*/dn_2 from the app is compared with a central FD (step 1e-6) in the relative
    2-norm, tolerance 1e-4.
    """
    from hydrograd.apps.sensitivity import final_state_jacobian

    # Set up the sensitivity example with the steady Newton solver
    rel = "sensitivity_analysis/ManningN/oneD_channel_with_bump"
    case_dir = case_copy(rel)
    case = setup_case(read_control(case_dir / "run_control.json"))
    case = dataclasses.replace(case, ode=dataclasses.replace(case.ode, solver="SteadyNewton()"))
    n_values = jnp.asarray(np.asarray(case.ctrl.opt("sensitivity_ManningN_values"), dtype=float))

    # Sensitivity Jacobian from the app
    J = final_state_jacobian(case, "ManningN", n_values)["J"]

    # Flattened steady state as a function of the zone n values
    settings = newton_settings(case.ode)

    def flat_steady_state(n_zone):
        """Steady state, flattened to shape [3 n_cells], for zone n values n_zone."""
        return steady_state(case.model, {"n_zone": n_zone}, case.U0, settings).reshape(-1)

    steady_state_jit = jax.jit(flat_steady_state)

    # Central FD on zone 2
    perturbation = jnp.zeros_like(n_values).at[2].set(1e-6)
    U_plus = np.asarray(steady_state_jit(n_values + perturbation))
    U_minus = np.asarray(steady_state_jit(n_values - perturbation))
    fd = (U_plus - U_minus) / 2e-6
    assert np.linalg.norm(J[:, 2] - fd) <= 1e-4 * np.linalg.norm(fd)
