"""Second-order MUSCL (hydrostatic reconstruction + HLLC): balance, exact Jacobian,
observed order of accuracy, dry-bed dam break, IFT gradient."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.jacobian import dense_jacobian, full_jacobian, linearize
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.steady import NewtonSettings, newton_solve, steady_state
from hydrograd.validation.dambreak import dam_break

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
from convergence_bump import H_OUT, Q, bump, exact_depth


@pytest.mark.parametrize("limiter", ["venkatakrishnan", "barth_jespersen"])
def test_lake_at_rest_wet_dry_is_exact(tmp_path, limiter: str) -> None:
    write_channel_case(tmp_path, 100, 10, length=1000.0, width=100.0, inlet="WALL", outlet="WALL",
                       wse_out=0.5, ic_wse=lambda x, y: 0.5 + 0 * x, order=2,
                       zb_fn=lambda X, Y: np.exp(-((X - 500) / 50) ** 2) + 0.2 * np.sin(X / 37.0) * np.cos(Y / 13.0))
    case = setup_case(read_control(tmp_path / "run_control.json"))
    m = dataclasses.replace(case.model, limiter=limiter)
    h = np.asarray(case.U0[0] + m.hstill)
    assert (h <= 0).sum() > 0
    assert np.abs(np.asarray(rhs(m, {}, case.U0))).max() < 1e-13


def test_order2_jacobian_is_exact(tmp_path) -> None:
    write_channel_case(tmp_path, 20, 5, length=200.0, width=50.0, order=2,
                       zb_fn=lambda X, Y: 1e-3 * (200 - X) + 0.3 * np.exp(-((X - 100) / 20) ** 2) * np.cos(Y / 15.0))
    case = setup_case(read_control(tmp_path / "run_control.json"))
    m = case.model
    U = case.U0 + 0.05 * jnp.sin(jnp.arange(3 * m.n_cells).reshape(3, -1) * 0.7)
    comp, A, C = jax.jit(lambda U: linearize(m, {}, U))(U)
    J = full_jacobian(m, np.asarray(comp), np.asarray(A), np.asarray(C))
    np.testing.assert_allclose(J, dense_jacobian(m, {}, U), atol=1e-12)


def test_observed_order_on_subcritical_bump(tmp_path) -> None:
    errs = []
    for nx in (50, 100, 200):
        d = tmp_path / f"n{nx}"
        write_channel_case(d, nx, 1, length=25.0, width=1.0, n_manning=1e-4, Q=Q, wse_out=H_OUT, wstill=H_OUT,
                           order=2, zb_fn=lambda X, Y: bump(X), ic_wse=lambda x, y: H_OUT + 0 * x)
        case = setup_case(read_control(d / "run_control.json"))
        U, it, rn = jax.jit(lambda U0: newton_solve(case.model, {}, U0, NewtonSettings(dtau0=0.05)))(case.U0)
        assert float(rn) < 1e-10
        h = np.asarray(U[0] + case.model.hstill)
        errs.append(np.mean(np.abs(h - exact_depth(case.mesh.centroid[:, 0]))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert rates.min() > 1.75, rates


def test_order2_dry_dam_break(tmp_path) -> None:
    write_channel_case(tmp_path, 1000, 2, length=1000.0, width=10.0, slope=0.0, bump=0.0, n_manning=0.0,
                       inlet="WALL", outlet="WALL", wse_out=0.0, wstill=0.5, t_end=60.0, dt=0.05,
                       solver="SSPRK3()", order=2, ic_wse=lambda x, y: np.where(x < 500.0, 1.0, 0.0))
    case = setup_case(read_control(tmp_path / "run_control.json"))
    m, s = case.model, dataclasses.replace(case.ode, cfl=0.25)
    U = np.asarray(jax.jit(lambda U0: solve_swe(m, {}, U0, s))(case.U0).ys[-1])
    h = U[0] + np.asarray(m.hstill)
    he, _ = dam_break(case.mesh.centroid[:, 0], 60.0, 1.0, 0.0, x0=500.0)
    area = np.asarray(m.area)
    m0 = np.sum((np.asarray(case.U0[0]) + np.asarray(m.hstill)) * area)
    assert h.min() > -1e-12 and abs(np.sum(h * area) - m0) < 1e-10 * m0
    assert np.mean(np.abs(h - he)) < 6e-4                     # first order: 2.6e-3


def test_order2_ift_gradient(tmp_path) -> None:
    write_channel_case(tmp_path, 40, 4, length=400.0, width=40.0, Q=20.0, wse_out=1.5, order=2,
                       zb_fn=lambda X, Y: 1e-3 * (400 - X) + 0.2 * np.exp(-((X - 200) / 40) ** 2))
    case = setup_case(read_control(tmp_path / "run_control.json"))
    m, ns = case.model, NewtonSettings(dtau0=1.0)
    w = jnp.asarray(np.random.default_rng(0).normal(size=case.U0.shape))
    L = jax.jit(lambda nz: jnp.sum(w * steady_state(m, {"n_zone": nz}, case.U0, ns)))
    nz = jnp.asarray(m.n_zone)
    g = np.asarray(jax.jit(jax.grad(L))(nz))
    e = jnp.zeros_like(nz).at[1].set(1e-6)
    fd = (float(L(nz + e)) - float(L(nz - e))) / 2e-6
    assert abs(g[1] - fd) <= 1e-5 * abs(fd)
