"""Wet/dry scheme (hydrostatic reconstruction + HLLC, SSPRK3): exact solutions and invariants."""

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

T = 60.0


def _dam_break_case(tmp_path, h_r: float):
    write_channel_case(tmp_path, 1000, 2, length=1000.0, width=10.0, slope=0.0, bump=0.0, n_manning=0.0,
                       inlet="WALL", outlet="WALL", wse_out=h_r, wstill=0.5, t_end=T, dt=0.05,
                       solver="SSPRK3()", ic_wse=lambda x, y: np.where(x < 500.0, 1.0, h_r))
    return setup_case(read_control(tmp_path / "run_control.json"))


@pytest.mark.parametrize("h_r", [0.1, 0.0])
def test_dam_break_matches_exact_solution(tmp_path, h_r: float) -> None:
    case = _dam_break_case(tmp_path, h_r)
    m = case.model
    sol = jax.jit(lambda U0: solve_swe(m, {}, U0, case.ode))(case.U0)
    U = np.asarray(sol.ys[-1])
    h = U[0] + np.asarray(m.hstill)
    assert h.min() > -1e-12                                     # positivity
    area = np.asarray(m.area)
    mass0 = np.sum((np.asarray(case.U0[0]) + np.asarray(m.hstill)) * area)
    assert abs(np.sum(h * area) - mass0) <= 1e-10 * mass0      # exact mass conservation
    x = case.mesh.centroid[:, 0]
    he, _ = dam_break(x, T, 1.0, h_r, x0=500.0)
    assert np.mean(np.abs(h - he)) < 4e-3                       # L1 depth error (m)
    if h_r == 0.0:
        exact_front = 500.0 + (2 * np.sqrt(9.81) - 3 * np.sqrt(9.81 * 1e-3)) * T   # where exact h = 1 mm
        assert abs(x[h > 1e-3].max() - exact_front) < 50.0
    else:  # no expansion shock at the sonic point (x = x0)
        near = np.abs(x - 500.0) < 30.0
        assert np.abs(h - he)[near].max() < 0.02


def test_lake_at_rest_with_emerged_island_is_exact(tmp_path) -> None:
    write_channel_case(tmp_path, 200, 10, length=1000.0, width=100.0, slope=0.0, bump=1.0, n_manning=0.03,
                       inlet="WALL", outlet="WALL", wse_out=0.5, ic_wse=lambda x, y: 0.5 + 0 * x)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    h = np.asarray(case.U0[0] + case.model.hstill)
    assert (h <= 0.0).sum() > 0 and (h > 0.1).sum() > 0         # wet and dry cells
    R = np.asarray(rhs(case.model, {}, case.U0))
    assert np.abs(R).max() < 1e-12


@pytest.mark.slow
def test_gradient_through_wetting_front_matches_finite_differences(tmp_path) -> None:
    """Hydrograph gradient while floodplain cells wet up (CFL step choice not differentiated)."""
    import dataclasses

    def bed(X, Y):
        r = np.abs(Y - 50.0)
        return 1e-3 * (500.0 - X) + np.where(r <= 10.0, 0.0, np.where(r <= 15.0, 0.4 * (r - 10.0), 2.0 + 0.01 * (r - 15.0)))

    write_channel_case(tmp_path, 50, 10, length=500.0, width=100.0, n_manning=0.035, zb_fn=bed, wstill=0.0,
                       inlet="INLET-Q", outlet="WALL", dt=0.1, solver="SSPRK3()", t_end=1200.0,
                       Q_series=([0.0, 600.0, 1200.0], [10.0, 150.0, 150.0]),
                       ic_wse=lambda x, y: 1e-3 * (500.0 - x) + 1.0)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    m = case.model
    s = dataclasses.replace(case.ode, n_save=1)
    h0 = np.asarray(case.U0[0] + m.hstill)
    U = np.asarray(jax.jit(lambda U0: solve_swe(m, {}, U0, s))(case.U0).ys[-1])
    assert (U[0] + np.asarray(m.hstill) > 0.01).sum() > (h0 > 0.01).sum() + 50      # front advanced
    w = jnp.asarray(np.random.default_rng(0).normal(size=m.n_cells))
    loss = jax.jit(lambda q: jnp.sum(w * (solve_swe(m, {"Q": q}, case.U0, s).ys[-1][0] + m.hstill)))
    q0 = jnp.asarray(m.Q)
    g = np.asarray(jax.jit(jax.grad(lambda q: jnp.sum(w * (solve_swe(m, {"Q": q}, case.U0, s).ys[-1][0] + m.hstill))))(q0))
    e = jnp.zeros_like(q0).at[0, 1].set(0.1)
    fd = (float(loss(q0 + e)) - float(loss(q0 - e))) / 0.2
    assert abs(g[0, 1] - fd) <= 1e-3 * abs(fd)
