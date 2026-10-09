"""Tests of the depth-averaged k-epsilon model (``physics.k_epsilon``).

The model transports h k and h epsilon as state rows 3 and 4, with nu_t = c_mu k^2 / eps
in the momentum diffusion, bed production P_kV = c_f |U|^3 / h and
P_eV = 3.6 c_2 sqrt(c_mu) c_f^(5/4) |U|^4 / h^2, and shear production nu_t |S|^2. The tests
check:

1. The equilibrium values give the Rastogi-Rodi eddy viscosity nu_t = u_* h / 12.96.
2. Steady uniform flow in a sloped flume: the depth equals the Manning normal depth and k
   and epsilon equal their equilibrium values (no shear production, no net transport).
3. The colored sparse Jacobian of the 5-row residual equals the dense jacfwd Jacobian
   (first and second order).
4. The IFT gradient of a steady quantity with respect to Manning's n matches central
   differences with the 5-row state.
5. Explicit SSP-RK3 steps (point-implicit dissipation) keep h k and h epsilon positive.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.k_epsilon import equilibrium, turbulence_variables
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.final import final_state
from hydrograd.solvers.jacobian import dense_jacobian, full_jacobian, linearize

# Sloped flume (SI units).
LENGTH = 100.0          # m
WIDTH = 0.5             # m
SLOPE = 5.0e-4          # -
Q_TOTAL = 0.075         # m^3/s
N_MANNING = 0.012       # s/m^(1/3)
G = 9.81                # m/s^2


def _normal_depth() -> float:
    """Manning normal depth (m) of the flume (smooth depth floor of the scheme).

    Returns:
        h_n (m).
    """
    q = Q_TOTAL / WIDTH

    def balance(h: float) -> float:
        """Gravity minus bed friction per unit plan area (m^2/s^2)."""
        h_eff = np.sqrt(h * h + 1.0e-6)
        return G * h * SLOPE - G * N_MANNING**2 * q * q / h_eff ** (7.0 / 3.0)

    return brentq(balance, 0.02, 2.0)


def _write_flume(case_dir, nx: int, ny: int, solver: str, order: int = 2) -> float:
    """Write the sloped flume with the k-epsilon model at its normal depth.

    Args:
        case_dir: output folder (pytest ``tmp_path``).
        nx, ny: cells along and across.
        solver: ODE solver name (``SteadyNewton()`` or ``SSPRK3()``).
        order: spatial order (1 or 2).

    Returns:
        The normal depth h_n (m).
    """
    h_normal = _normal_depth()

    def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial water surface: bed z_b = S (L - x) plus the normal depth (m)."""
        return SLOPE * (LENGTH - x) + h_normal

    write_channel_case(
        case_dir, nx, ny,
        length=LENGTH,
        width=WIDTH,
        slope=SLOPE,
        bump=0.0,
        Q=Q_TOTAL,
        wse_out=h_normal,
        n_manning=N_MANNING,
        solver=solver,
        ic_wse=initial_wse,
        order=order,
        t_end=2.0,
        dt=0.05,
    )
    control_path = case_dir / "run_control.json"
    control = json.loads(control_path.read_text())
    control["turbulence"] = {"model": "k_epsilon"}
    control_path.write_text(json.dumps(control, indent=1))
    return h_normal


def test_equilibrium_gives_rastogi_rodi_viscosity(tmp_path) -> None:
    """nu_t = c_mu k_eq^2 / eps_eq equals u_* h / 12.96 for any depth and speed."""
    # Setup: the model of the flume (for g, k_n and the constants).
    _write_flume(tmp_path, 4, 1, "SteadyNewton()")
    model = setup_case(read_control(tmp_path / "run_control.json")).model
    h = jnp.array([0.1, 0.3, 1.0, 3.0])
    speed = jnp.array([0.2, 0.5, 1.0, 2.0])
    n = jnp.full(4, 0.03)

    # Run.
    k, epsilon = equilibrium(model, h, speed, n)
    nu_t = model.turbulence.c_mu * k**2 / epsilon

    # Compare: u_* = sqrt(g) n |U| / h^(1/6).
    u_star = np.sqrt(G) * 0.03 * np.asarray(speed) / np.asarray(h) ** (1.0 / 6.0)
    np.testing.assert_allclose(np.asarray(nu_t), u_star * np.asarray(h) / 12.96, rtol=1e-12)


def test_uniform_flow_is_at_equilibrium(tmp_path) -> None:
    """Steady uniform flow: normal depth, and k, epsilon at their equilibrium values."""
    # Setup.
    h_normal = _write_flume(tmp_path, 80, 1, "SteadyNewton()")
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    assert case.U0.shape == (5, model.n_cells)

    # Run: steady Newton solve.
    U, converged = final_state(model, {}, case.U0, case.ode)
    assert bool(converged)

    # Compare in the middle of the flume.
    h = U[0] + model.hstill
    x = np.asarray(case.mesh.centroid[:, 0])
    middle = np.abs(x - 0.5 * LENGTH) < 0.1 * LENGTH
    depth = np.asarray(h)
    np.testing.assert_allclose(depth[middle], h_normal, rtol=1e-3)

    k, epsilon, _ = turbulence_variables(model, h, U)
    speed = jnp.abs(U[1]) / h
    k_eq, epsilon_eq = equilibrium(model, h, speed, jnp.full(model.n_cells, N_MANNING))
    np.testing.assert_allclose(np.asarray(k)[middle], np.asarray(k_eq)[middle], rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(epsilon)[middle], np.asarray(epsilon_eq)[middle], rtol=1e-4
    )


def test_sparse_jacobian_matches_dense(tmp_path) -> None:
    """Colored 5-row Jacobian = dense jacfwd Jacobian, first and second order."""
    for order in (1, 2):
        # Setup: small 2D flume, perturbed state.
        folder = tmp_path / f"order{order}"
        _write_flume(folder, 6, 4, "SteadyNewton()", order=order)
        case = setup_case(read_control(folder / "run_control.json"))
        model = case.model
        rows = jnp.arange(5 * model.n_cells).reshape(5, -1)
        U = case.U0 * (1.0 + 0.05 * jnp.sin(rows * 0.7))
        U = U.at[1].add(0.01 * jnp.cos(rows[1] * 0.3))
        U = U.at[2].add(0.01 * jnp.sin(rows[2] * 0.5))

        # Run.
        compressed, A, C = jax.jit(lambda state, m=model: linearize(m, {}, state))(U)
        J = full_jacobian(model, np.asarray(compressed), np.asarray(A), np.asarray(C))

        # Compare (entries of the k-epsilon rows are small: absolute and relative tolerance).
        reference = dense_jacobian(model, {}, U)
        np.testing.assert_allclose(J, reference, rtol=1e-10, atol=1e-12)


def test_steady_gradient_wrt_manning_n(tmp_path) -> None:
    """d(mean nu_t)/d(n) by the IFT adjoint matches central differences (5-row state)."""
    # Setup.
    _write_flume(tmp_path, 40, 1, "SteadyNewton()")
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    n0 = jnp.asarray(model.n_zone)

    def mean_viscosity(n_zone: jax.Array) -> jax.Array:
        """Mean steady eddy viscosity (m^2/s) for the zone Manning's n."""
        U, _ = final_state(model, {"n_zone": n_zone}, case.U0, case.ode)
        h = U[0] + model.hstill
        return jnp.mean(turbulence_variables(model, h, U)[2])

    # Run: adjoint gradient and central differences for the channel material.
    material = int(np.asarray(model.mat_id)[0])
    gradient = float(jax.grad(mean_viscosity)(n0)[material])
    step = 1.0e-5
    unit = jnp.zeros_like(n0).at[material].set(1.0)
    finite_difference = float(
        (mean_viscosity(n0 + step * unit) - mean_viscosity(n0 - step * unit)) / (2.0 * step)
    )

    # Compare.
    assert finite_difference > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4)


def test_explicit_steps_keep_turbulence_positive(tmp_path) -> None:
    """SSP-RK3 from a still state with a k-epsilon spike: h k, h eps stay positive."""
    # Setup: explicit solver; a strong local spike of k tests the dissipation sink.
    _write_flume(tmp_path, 20, 3, "SSPRK3()")
    case = setup_case(read_control(tmp_path / "run_control.json"))
    U0 = case.U0.at[3, 0].multiply(1.0e3)
    case = dataclasses.replace(case, U0=U0)

    # Run: 2 s of simulation.
    U, ok = final_state(case.model, {}, case.U0, case.ode)

    # Compare.
    assert bool(ok)
    assert bool(jnp.all(jnp.isfinite(U)))
    assert float(jnp.min(U[3])) > 0.0
    assert float(jnp.min(U[4])) > 0.0
    assert np.isfinite(float(jnp.max(jnp.abs(rhs(case.model, {}, U)))))
