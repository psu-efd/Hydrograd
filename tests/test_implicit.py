"""Implicit theta-scheme stepper: accuracy vs explicit, mass balance, IFT gradient, wetting."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import solve_swe


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
import hydrograph_inversion as H  # noqa: E402


def _case(tmp_path, outlet="EXIT-H", t_end=1200.0):
    """Hydrograph-inversion valley (gentle floodplains), Q rising 40 -> ~180 m^3/s over t_end."""
    tq = np.linspace(0.0, t_end, 5)
    write_channel_case(tmp_path, H.NX, H.NY, length=H.L, width=H.W, n_manning=H.N_MAN, zb_fn=H.bed,
                       wstill=0.0, inlet="INLET-Q", outlet=outlet, dt=0.1, solver="SSPRK3()", t_end=t_end,
                       wse_out=1.03, Q_series=(tq, H.q_true(tq)), ic_wse=lambda x, y: H.S0 * (H.L - x) + 1.03)
    return setup_case(read_control(tmp_path / "run_control.json"))


def test_implicit_matches_explicit_and_wets_floodplain(tmp_path) -> None:
    case = _case(tmp_path)
    m = case.model
    s = dataclasses.replace(case.ode, n_save=1)
    Ue = np.asarray(jax.jit(lambda U0: solve_swe(m, {}, U0, s).ys[-1])(case.U0))
    si = ImplicitSettings(0.0, 1200.0, 30.0, n_save=1, tol=1e-8)
    Ui = np.asarray(jax.jit(lambda U0: solve_implicit(m, {}, U0, si)[1][-1])(case.U0))
    h0 = np.asarray(case.U0[0] + m.hstill)
    he, hi = Ue[0] + np.asarray(m.hstill), Ui[0] + np.asarray(m.hstill)
    assert hi.min() >= 0.0
    assert (hi > 0.01).sum() > (h0 > 0.01).sum()                          # floodplain wets
    wet = he > 0.01
    assert np.sqrt(np.mean((hi - he)[wet] ** 2)) < 0.01                    # within 1 cm (dt = 30 s)


def test_implicit_volume_balance_closed_channel(tmp_path) -> None:
    case = _case(tmp_path, outlet="WALL", t_end=600.0)
    m = case.model
    si = ImplicitSettings(0.0, 600.0, 20.0, n_save=1, tol=1e-10)
    U = np.asarray(jax.jit(lambda U0: solve_implicit(m, {}, U0, si)[1][-1])(case.U0))
    area = np.asarray(m.area)
    V0 = np.sum(np.asarray(case.U0[0] + m.hstill) * area)
    V1 = np.sum((U[0] + np.asarray(m.hstill)) * area)
    # Interior fluxes cancel and the inlet face carries exactly Q(t^{n+1}) per step, so the
    # storage change equals the right-end-rule quadrature of the hydrograph.
    tq = np.arange(20.0, 600.0 + 1e-9, 20.0)
    knots = np.linspace(0.0, 600.0, 5)
    inflow = 20.0 * np.sum(np.interp(tq, knots, H.q_true(knots)))
    assert abs((V1 - V0) - inflow) < 1e-5 * inflow


def test_implicit_ift_gradient_matches_finite_differences(tmp_path) -> None:
    case = _case(tmp_path, t_end=600.0)
    m = case.model
    si = ImplicitSettings(0.0, 600.0, 60.0, n_save=1, tol=1e-10)
    w = jnp.asarray(np.random.default_rng(0).normal(size=m.n_cells))
    f = lambda q: jnp.sum(w * (solve_implicit(m, {"Q": q}, case.U0, si)[1][-1][0] + m.hstill))
    q0 = jnp.asarray(m.Q)
    g = np.asarray(jax.jit(jax.grad(f))(q0))
    L = jax.jit(f)
    e = jnp.zeros_like(q0).at[0, 1].set(0.05)
    fd = (float(L(q0 + e)) - float(L(q0 - e))) / 0.1
    assert abs(g[0, 1] - fd) <= 2e-3 * abs(fd)
