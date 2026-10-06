"""Time-varying boundary conditions: hydrograph input, volume balance, hydrograph gradients."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.ode import solve_swe

T_HYD = np.array([0.0, 60.0, 180.0, 300.0])      # s
Q_HYD = np.array([5.0, 40.0, 10.0, 5.0])         # m^3/s


def _case(tmp_path, t_end: float = 300.0):
    # Closed channel (wall outlet): the stored volume must grow by exactly the inflow volume.
    write_channel_case(tmp_path, 100, 5, length=500.0, width=20.0, slope=1e-3, bump=0.0, n_manning=0.03,
                       inlet="INLET-Q", outlet="WALL", wse_out=2.0, t_end=t_end, dt=0.05, n_save=3,
                       solver="SSPRK3()", Q_series=(T_HYD, Q_HYD))
    return setup_case(read_control(tmp_path / "run_control.json"))


def test_hydrograph_is_read_from_xys(tmp_path) -> None:
    m = _case(tmp_path).model
    np.testing.assert_allclose(np.asarray(m.Q_times), T_HYD)
    np.testing.assert_allclose(np.asarray(m.Q)[0], Q_HYD)


def test_volume_balance_equals_inflow_volume(tmp_path) -> None:
    case = _case(tmp_path)
    m = case.model
    sol = jax.jit(lambda U0: solve_swe(m, {}, U0, case.ode))(case.U0)
    area = np.asarray(m.area)
    V = [float(np.sum((np.asarray(U[0]) + np.asarray(m.hstill)) * area)) for U in sol.ys]
    ts = np.asarray(sol.ts)
    for k in range(1, len(ts)):
        tq = np.linspace(0.0, ts[k], 20001)
        inflow = np.trapezoid(np.interp(tq, T_HYD, Q_HYD), tq)
        assert abs((V[k] - V[0]) - inflow) <= 1e-6 * inflow


def test_hydrograph_gradient_matches_finite_differences(tmp_path) -> None:
    case = _case(tmp_path, t_end=200.0)
    m = case.model
    s = dataclasses.replace(case.ode, n_save=1)
    w = jnp.asarray(np.random.default_rng(0).normal(size=m.n_cells))

    def loss(q):
        U = solve_swe(m, {"Q": q}, case.U0, s).ys[-1]
        return jnp.sum(w * (U[0] + m.hstill))

    q0 = jnp.asarray(m.Q)
    g = np.asarray(jax.jit(jax.grad(loss))(q0))
    L = jax.jit(loss)
    for k in (1, 2):
        e = jnp.zeros_like(q0).at[0, k].set(1e-3)
        fd = (float(L(q0 + e)) - float(L(q0 - e))) / 2e-3
        assert abs(g[0, k] - fd) <= 2e-3 * abs(fd), (k, g[0, k], fd)


def test_exit_stage_series_is_interpolated(tmp_path) -> None:
    from hydrograd.physics.rhs import boundary_values_at

    write_channel_case(tmp_path, 20, 2, length=100.0, width=10.0, slope=0.0, bump=0.0, wse_out=2.0)
    (tmp_path / "stage.xys").write_text('XY 1 3 0 0 0 0 0 0 "tide"\n0 2.0\n0.5 2.6\n1.0 2.2\n')  # hours
    hydro = (tmp_path / "channel.srhhydro").read_text().replace("EWSParamsC 2 2.0 SI C", "EWSParamsC 2 stage.xys SI C")
    (tmp_path / "channel.srhhydro").write_text(hydro)
    m = setup_case(read_control(tmp_path / "run_control.json")).model
    np.testing.assert_allclose(np.asarray(m.WSE_times), [0.0, 1800.0, 3600.0])
    w = boundary_values_at(m, {}, 900.0)["WSE"]
    np.testing.assert_allclose(np.asarray(w), [2.3])                      # halfway up the first ramp
    np.testing.assert_allclose(np.asarray(boundary_values_at(m, {}, 7200.0)["WSE"]), [2.2])  # held after the end


def test_auto_checkpoints_give_identical_gradient(tmp_path) -> None:
    from hydrograd.solvers.ode import n_checkpoints

    case = _case(tmp_path, t_end=120.0)
    m = case.model
    grads = {}
    for ck in (64, "auto"):
        s = dataclasses.replace(case.ode, n_save=1, checkpoints=ck)
        grads[ck] = np.asarray(jax.jit(jax.grad(
            lambda q: jnp.sum(solve_swe(m, {"Q": q}, case.U0, s).ys[-1][0])))(jnp.asarray(m.Q)))
    np.testing.assert_allclose(grads["auto"], grads[64], rtol=1e-12, atol=1e-15)
    s = dataclasses.replace(case.ode, checkpoints="auto", checkpoint_memory_gb=0.001)
    assert n_checkpoints(s, case.U0) == max(16, int(0.001 * 2 ** 30 // (3 * case.U0.size * 8)))
