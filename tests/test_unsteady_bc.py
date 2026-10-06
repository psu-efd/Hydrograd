"""Time-varying boundary conditions: hydrograph input, volume balance, hydrograph gradients.

The test channel is 500 m x 20 m (slope 1e-3, n = 0.03) with an inlet-Q hydrograph read from a
``.xys`` file and a wall outlet, run with SSPRK3.

What is checked:
- The hydrograph knots (s, m^3/s) are read exactly from the ``.xys`` file.
- In the closed channel the stored volume grows by exactly the inflow volume, the time integral
  of the piecewise-linear Q(t), to 1e-6 (relative) at every saved time.
- Reverse-mode gradients of a weighted depth sum with respect to hydrograph knots match central
  finite differences to 2e-3 (relative).
- An exit stage series (``.xys``, times in hours) is converted to seconds, interpolated linearly
  in time and held constant after its last knot.
- ``checkpoints = "auto"`` gives the same gradient as 64 checkpoints (rtol 1e-12), and the
  automatic count follows the memory budget formula.
"""

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

T_HYD = np.array([0.0, 60.0, 180.0, 300.0])      # hydrograph times (s)
Q_HYD = np.array([5.0, 40.0, 10.0, 5.0])         # hydrograph discharges (m^3/s)


def _case(tmp_path, t_end: float = 300.0):
    """Write and set up the closed channel with the (T_HYD, Q_HYD) inlet hydrograph.

    Args:
        tmp_path: folder where the synthetic case is written.
        t_end: simulation end time (s); three frames are saved after t = 0.

    Returns:
        The set-up case.
    """
    # Closed channel (wall outlet): the stored volume must grow by exactly the inflow volume.
    write_channel_case(
        tmp_path,
        100,
        5,
        length=500.0,
        width=20.0,
        slope=1e-3,
        bump=0.0,
        n_manning=0.03,
        inlet="INLET-Q",
        outlet="WALL",
        wse_out=2.0,
        t_end=t_end,
        dt=0.05,
        n_save=3,
        solver="SSPRK3()",
        Q_series=(T_HYD, Q_HYD),
    )
    return setup_case(read_control(tmp_path / "run_control.json"))


def test_hydrograph_is_read_from_xys(tmp_path) -> None:
    """The model's hydrograph times and discharges equal T_HYD and Q_HYD (default rtol)."""
    model = _case(tmp_path).model
    np.testing.assert_allclose(np.asarray(model.Q_times), T_HYD)
    np.testing.assert_allclose(np.asarray(model.Q)[0], Q_HYD)


def test_volume_balance_equals_inflow_volume(tmp_path) -> None:
    """Storage change at each saved time equals the integral of Q(t) to 1e-6 (relative).

    The reference integral uses the trapezoidal rule on 20001 points, which is exact for the
    piecewise-linear hydrograph up to the knots falling between sample points.
    """
    case = _case(tmp_path)
    model = case.model

    # Forward run with the saved frames
    solution = jax.jit(lambda U0: solve_swe(model, {}, U0, case.ode))(case.U0)

    # Stored volume (m^3) at each saved time
    area = np.asarray(model.area)
    volumes = []
    for U in solution.ys:
        volumes.append(float(np.sum((np.asarray(U[0]) + np.asarray(model.hstill)) * area)))
    times = np.asarray(solution.ts)

    # Compare the storage change with the inflow volume up to each saved time
    for k in range(1, len(times)):
        quadrature_times = np.linspace(0.0, times[k], 20001)
        discharge = np.interp(quadrature_times, T_HYD, Q_HYD)
        inflow = np.trapezoid(discharge, quadrature_times)
        assert abs((volumes[k] - volumes[0]) - inflow) <= 1e-6 * inflow


def test_hydrograph_gradient_matches_finite_differences(tmp_path) -> None:
    """d(loss)/dQ at knots 1 and 2: reverse mode vs central FD (step 1e-3 m^3/s), rel 2e-3."""
    case = _case(tmp_path, t_end=200.0)
    model = case.model
    settings = dataclasses.replace(case.ode, n_save=1)

    # Random weights for the scalar loss
    weights = jnp.asarray(np.random.default_rng(0).normal(size=model.n_cells))

    def loss(q):
        """Weighted sum of the final depths (m) for the hydrograph discharges q (m^3/s)."""
        U = solve_swe(model, {"Q": q}, case.U0, settings).ys[-1]
        return jnp.sum(weights * (U[0] + model.hstill))

    # Reverse-mode gradient
    q0 = jnp.asarray(model.Q)
    gradient = np.asarray(jax.jit(jax.grad(loss))(q0))
    loss_jit = jax.jit(loss)

    # Central FD at each tested knot
    for k in (1, 2):
        perturbation = jnp.zeros_like(q0).at[0, k].set(1e-3)
        fd = (float(loss_jit(q0 + perturbation)) - float(loss_jit(q0 - perturbation))) / 2e-3
        assert abs(gradient[0, k] - fd) <= 2e-3 * abs(fd), (k, gradient[0, k], fd)


def test_exit_stage_series_is_interpolated(tmp_path) -> None:
    """Exit stage series in hours: times in seconds, linear interpolation, held after the end."""
    from hydrograd.physics.rhs import boundary_values_at

    # Channel with a constant exit stage, then switch the exit to a stage series file
    write_channel_case(tmp_path, 20, 2, length=100.0, width=10.0, slope=0.0, bump=0.0, wse_out=2.0)
    stage_series = 'XY 1 3 0 0 0 0 0 0 "tide"\n0 2.0\n0.5 2.6\n1.0 2.2\n'  # hours
    (tmp_path / "stage.xys").write_text(stage_series)
    hydro = (tmp_path / "channel.srhhydro").read_text()
    hydro = hydro.replace("EWSParamsC 2 2.0 SI C", "EWSParamsC 2 stage.xys SI C")
    (tmp_path / "channel.srhhydro").write_text(hydro)
    model = setup_case(read_control(tmp_path / "run_control.json")).model

    # Knot times converted from hours to seconds
    np.testing.assert_allclose(np.asarray(model.WSE_times), [0.0, 1800.0, 3600.0])

    # Interpolated stage at t = 900 s and the held value after the last knot
    wse_mid_ramp = boundary_values_at(model, {}, 900.0)["WSE"]
    np.testing.assert_allclose(np.asarray(wse_mid_ramp), [2.3])           # halfway up the first ramp
    wse_after_end = boundary_values_at(model, {}, 7200.0)["WSE"]
    np.testing.assert_allclose(np.asarray(wse_after_end), [2.2])          # held after the end


def test_auto_checkpoints_give_identical_gradient(tmp_path) -> None:
    """Gradient with 64 vs "auto" checkpoints (rtol 1e-12, atol 1e-15); auto count formula.

    The automatic count is max(16, budget / (bytes per state)) with 3 n_cells float64 values per
    state.
    """
    from hydrograd.solvers.ode import n_checkpoints

    case = _case(tmp_path, t_end=120.0)
    model = case.model

    # Gradient of the summed final xi for each checkpoint setting
    gradients = {}
    for checkpoints in (64, "auto"):
        settings = dataclasses.replace(case.ode, n_save=1, checkpoints=checkpoints)

        def summed_final_xi(q, settings=settings):
            """Sum of the final free-surface displacement xi (m) for the hydrograph q."""
            return jnp.sum(solve_swe(model, {"Q": q}, case.U0, settings).ys[-1][0])

        gradient = jax.jit(jax.grad(summed_final_xi))(jnp.asarray(model.Q))
        gradients[checkpoints] = np.asarray(gradient)
    np.testing.assert_allclose(gradients["auto"], gradients[64], rtol=1e-12, atol=1e-15)

    # Automatic checkpoint count for a 0.001 GB budget
    settings = dataclasses.replace(case.ode, checkpoints="auto", checkpoint_memory_gb=0.001)
    expected = max(16, int(0.001 * 2 ** 30 // (3 * case.U0.size * 8)))
    assert n_checkpoints(settings, case.U0) == expected
