"""Wet/dry scheme (hydrostatic reconstruction + HLLC, SSPRK3): exact solutions and invariants.

What is checked:
- 1D dam break on a flat frictionless bed (h_L = 1 m, x0 = 500 m, t = 60 s) against the exact
  Stoker (wet downstream, h_R = 0.1 m) and Ritter (dry downstream, h_R = 0) solutions:
  h >= 0 (to -1e-12 m), mass conserved to 1e-10 (relative), mean absolute depth error below
  4 mm. For Ritter, the wet front (h > 1 mm) is within 50 m of the exact position; for Stoker,
  no expansion shock at the sonic point (max error below 2 cm within 30 m of x0).
- Lake at rest over an emerged island: the residual vanishes to 1e-12 (well balanced across
  wet/dry fronts).
- (slow) Gradient of a weighted depth sum with respect to a hydrograph knot while floodplain
  cells wet up, vs a central finite difference (relative 1e-3). The CFL step choice is not
  differentiated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.ode import solve_swe
from hydrograd.validation.dambreak import dam_break

T = 60.0  # dam-break end time (s)


def _dam_break_case(tmp_path, h_r: float):
    """1000 m x 10 m frictionless channel with walls; 1 m depth left of x = 500 m, h_r right.

    Args:
        tmp_path: folder where the synthetic case is written.
        h_r: initial downstream depth (m); 0 gives a dry bed.

    Returns:
        The set-up case (SSPRK3, t_end = T).
    """

    def initial_wse(x, y):
        """Initial WSE (m): 1 m upstream of the dam at x = 500 m, h_r downstream."""
        return np.where(x < 500.0, 1.0, h_r)

    write_channel_case(
        tmp_path,
        1000,
        2,
        length=1000.0,
        width=10.0,
        slope=0.0,
        bump=0.0,
        n_manning=0.0,
        inlet="WALL",
        outlet="WALL",
        wse_out=h_r,
        wstill=0.5,
        t_end=T,
        dt=0.05,
        solver="SSPRK3()",
        ic_wse=initial_wse,
    )
    return setup_case(read_control(tmp_path / "run_control.json"))


@pytest.mark.parametrize("h_r", [0.1, 0.0])
def test_dam_break_matches_exact_solution(tmp_path, h_r: float) -> None:
    """Stoker (h_r = 0.1 m) and Ritter (h_r = 0) dam breaks vs the exact solution at t = T."""
    case = _dam_break_case(tmp_path, h_r)
    model = case.model

    # Run to t = T and take the final depth h = xi + h_still (m)
    solution = jax.jit(lambda U0: solve_swe(model, {}, U0, case.ode))(case.U0)
    U = np.asarray(solution.ys[-1])
    h = U[0] + np.asarray(model.hstill)
    assert h.min() > -1e-12                                     # positivity

    # Mass conservation (closed channel)
    area = np.asarray(model.area)
    mass0 = np.sum((np.asarray(case.U0[0]) + np.asarray(model.hstill)) * area)
    assert abs(np.sum(h * area) - mass0) <= 1e-10 * mass0      # exact mass conservation

    # Depth error against the exact solution
    x = case.mesh.centroid[:, 0]
    h_exact, _ = dam_break(x, T, 1.0, h_r, x0=500.0)
    assert np.mean(np.abs(h - h_exact)) < 4e-3                  # L1 depth error (m)

    if h_r == 0.0:
        # Ritter: wet-front position, where the exact h = 1 mm
        front_speed = 2 * np.sqrt(9.81) - 3 * np.sqrt(9.81 * 1e-3)
        exact_front = 500.0 + front_speed * T
        assert abs(x[h > 1e-3].max() - exact_front) < 50.0
    else:  # no expansion shock at the sonic point (x = x0)
        near = np.abs(x - 500.0) < 30.0
        assert np.abs(h - h_exact)[near].max() < 0.02


def test_lake_at_rest_with_emerged_island_is_exact(tmp_path) -> None:
    """Lake at rest (WSE 0.5 m) over a bump that emerges: the residual is zero to 1e-12."""

    def flat_wse(x, y):
        """Flat initial WSE of 0.5 m (shape of x)."""
        return 0.5 + 0 * x

    write_channel_case(
        tmp_path,
        200,
        10,
        length=1000.0,
        width=100.0,
        slope=0.0,
        bump=1.0,
        n_manning=0.03,
        inlet="WALL",
        outlet="WALL",
        wse_out=0.5,
        ic_wse=flat_wse,
    )
    case = setup_case(read_control(tmp_path / "run_control.json"))

    # The case must contain both wet and dry cells
    h = np.asarray(case.U0[0] + case.model.hstill)
    assert (h <= 0.0).sum() > 0 and (h > 0.1).sum() > 0         # wet and dry cells

    # Residual at the initial state
    residual = np.asarray(rhs(case.model, {}, case.U0))
    assert np.abs(residual).max() < 1e-12


@pytest.mark.slow
def test_gradient_through_wetting_front_matches_finite_differences(tmp_path) -> None:
    """Hydrograph gradient while floodplain cells wet up (CFL step choice not differentiated)."""
    import dataclasses

    def bed(X, Y):
        """Valley bed (m): slope 1e-3, 20 m channel, 5 m banks rising 2 m, gentle floodplains."""
        r = np.abs(Y - 50.0)
        floodplain = 2.0 + 0.01 * (r - 15.0)
        cross_section = np.where(r <= 10.0, 0.0, np.where(r <= 15.0, 0.4 * (r - 10.0), floodplain))
        return 1e-3 * (500.0 - X) + cross_section

    def initial_wse(x, y):
        """Initial WSE (m): 1 m above a plane with the bed slope."""
        return 1e-3 * (500.0 - x) + 1.0

    # Inflow hydrograph rising 10 -> 150 m^3/s over 600 s, then constant; wall outlet
    write_channel_case(
        tmp_path,
        50,
        10,
        length=500.0,
        width=100.0,
        n_manning=0.035,
        zb_fn=bed,
        wstill=0.0,
        inlet="INLET-Q",
        outlet="WALL",
        dt=0.1,
        solver="SSPRK3()",
        t_end=1200.0,
        Q_series=([0.0, 600.0, 1200.0], [10.0, 150.0, 150.0]),
        ic_wse=initial_wse,
    )
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    settings = dataclasses.replace(case.ode, n_save=1)

    # Forward run: the wet front must advance (more than 50 extra cells wetter than 1 cm)
    h0 = np.asarray(case.U0[0] + model.hstill)
    U = np.asarray(jax.jit(lambda U0: solve_swe(model, {}, U0, settings))(case.U0).ys[-1])
    n_wet_final = (U[0] + np.asarray(model.hstill) > 0.01).sum()
    assert n_wet_final > (h0 > 0.01).sum() + 50                                      # front advanced

    # Scalar loss: random weighted sum of the final depths
    weights = jnp.asarray(np.random.default_rng(0).normal(size=model.n_cells))

    def loss_fn(q):
        """Weighted sum of the final depths (m) for the hydrograph values q (m^3/s)."""
        U_final = solve_swe(model, {"Q": q}, case.U0, settings).ys[-1]
        return jnp.sum(weights * (U_final[0] + model.hstill))

    loss = jax.jit(loss_fn)

    # Reverse-mode gradient vs central FD on hydrograph entry [0, 1] (step 0.1 m^3/s)
    q0 = jnp.asarray(model.Q)
    gradient = np.asarray(jax.jit(jax.grad(loss_fn))(q0))
    perturbation = jnp.zeros_like(q0).at[0, 1].set(0.1)
    fd = (float(loss(q0 + perturbation)) - float(loss(q0 - perturbation))) / 0.2
    assert abs(gradient[0, 1] - fd) <= 1e-3 * abs(fd)
