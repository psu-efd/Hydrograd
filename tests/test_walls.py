"""Tests of the log-law side-wall friction (``physics.walls``).

Without a ``"wall_friction"`` block WALL faces are free-slip. With it each wall face
exerts tau_w / rho = u_*^2 over the wetted wall area h L_f, with u_* from the combined
smooth/rough log law u_t / u_* = ln(y / y_0) / kappa, y_0 = nu / (E u_*) + k_s / 30, and
the viscous-sublayer law u_t = u_*^2 y / nu below y+ = 11.53. The tests check:

1. The friction velocity solves the smooth log law, the fully rough log law and the
   viscous-sublayer law, against independent root finding (scipy ``brentq``).
2. Uniform flow in a narrow channel one cell wide (both walls at y = b / 2 from the cell
   centroid). The steady depth equals the normal depth h_n of the momentum balance per
   unit plan area

       g h S = g n^2 q^2 / h_eff^(7/3) + (2 h / b) u_*(q / h, b / 2)^2,

   with h_eff = sqrt(h^2 + h_small^2) (the default Manning law), and is deeper than with
   free-slip walls. The flume has 80 cells (dx = 1.25 m): the steady depth carries a
   first-order discretization error of the sloped-bed treatment, the same with and
   without wall friction (+0.12%, +0.06%, +0.03% at dx = 2.5, 1.25, 0.625 m).
3. The gradient of the steady mean depth with respect to the wall roughness
   ``theta["wall_ks"]`` (implicit-function-theorem adjoint) matches central differences.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.walls import wall_friction_velocity
from hydrograd.solvers.final import final_state

# Narrow flume of the uniform-flow tests (SI units).
LENGTH = 100.0          # m
WIDTH = 0.5             # m
SLOPE = 5.0e-4          # -
Q_TOTAL = 0.075         # m^3/s (q = 0.15 m^2/s)
N_MANNING = 0.012       # s/m^(1/3), bed
G = 9.81                # m/s^2
NU = 1.0e-6             # m^2/s
KAPPA = 0.41            # -
E_SMOOTH = 9.8          # -
YPLUS_LAM = 11.53       # -


def _friction_velocity_reference(speed: float, distance: float, ks: float) -> float:
    """u_* (m/s) of the wall law by root finding (independent of the package).

    Args:
        speed: tangential speed |u_t| (m/s).
        distance: distance y from the wall (m).
        ks: equivalent sand roughness (m).

    Returns:
        The friction velocity of the viscous law if its y+ < y+_lam, else of the log law.
    """
    # Viscous sublayer u_t = u_*^2 y / nu.
    viscous = np.sqrt(NU * speed / distance)
    if distance * viscous / NU < YPLUS_LAM:
        return viscous

    def log_law_residual(friction_velocity: float) -> float:
        """u_t / u_* - ln(y / y_0) / kappa (-)."""
        roughness_length = NU / (E_SMOOTH * friction_velocity) + ks / 30.0
        return speed / friction_velocity - np.log(distance / roughness_length) / KAPPA

    return brentq(log_law_residual, 1e-6 * speed, speed)


def test_friction_velocity_regimes() -> None:
    """Smooth log law, fully rough log law and viscous sublayer against brentq (1e-9)."""
    # Cases: (|u_t| (m/s), y (m), k_s (m)): smooth log region, fully rough, transitional,
    # and a slow flow close to the wall (viscous sublayer).
    cases = [
        (0.5, 0.025, 0.0),
        (0.5, 0.025, 5.0e-3),
        (0.5, 0.025, 1.0e-4),
        (1.0e-3, 1.0e-3, 0.0),
    ]
    speed = jnp.array([case[0] for case in cases])
    distance = jnp.array([case[1] for case in cases])
    ks = jnp.array([case[2] for case in cases])

    # Run: the package law (12 fixed-point iterations, the default).
    friction_velocity = np.asarray(
        wall_friction_velocity(speed, distance, ks, NU, KAPPA, E_SMOOTH, YPLUS_LAM, 12)
    )

    # Compare with the independent root finding.
    expected = [_friction_velocity_reference(*case) for case in cases]
    np.testing.assert_allclose(friction_velocity, expected, rtol=1e-9)

    # The last case is in the viscous sublayer, the others are not.
    yplus = np.asarray(distance) * friction_velocity / NU
    assert yplus[3] < YPLUS_LAM
    assert np.all(yplus[:3] > YPLUS_LAM)


def _write_flume(case_dir, wse_out: float, h_initial: float, ks: float | None) -> None:
    """Write the narrow sloped flume, one cell wide, with a steady solver.

    Args:
        case_dir: output folder (pytest ``tmp_path``).
        wse_out: water-surface elevation at the exit (m); the bed there is at z = 0.
        h_initial: initial depth everywhere (m), the initial guess of the Newton solve.
        ks: wall roughness (m), or None for free-slip walls (no ``wall_friction`` block).
    """

    def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial water surface: the bed z_b = S (L - x) plus a uniform depth (m)."""
        return SLOPE * (LENGTH - x) + h_initial

    write_channel_case(
        case_dir, 80, 1,
        length=LENGTH,
        width=WIDTH,
        slope=SLOPE,
        bump=0.0,
        Q=Q_TOTAL,
        wse_out=wse_out,
        n_manning=N_MANNING,
        solver="SteadyNewton()",
        ic_wse=initial_wse,
        order=2,
    )

    # Add the wall-friction block.
    if ks is not None:
        control_path = case_dir / "run_control.json"
        control = json.loads(control_path.read_text())
        control["wall_friction"] = {"model": "loglaw", "ks": ks}
        control_path.write_text(json.dumps(control, indent=1))


def _normal_depth(ks: float | None) -> float:
    """Normal depth h_n (m) of the flume with bed and (optionally) wall friction.

    Args:
        ks: wall roughness (m), or None for free-slip walls.

    Returns:
        h_n (m).
    """
    q = Q_TOTAL / WIDTH

    def balance(h: float) -> float:
        """Gravity minus resistance per unit plan area (m^2/s^2) at depth h (m)."""
        h_eff = np.sqrt(h * h + 1.0e-6)          # smooth floor of the default scheme
        bed = G * N_MANNING ** 2 * q * q / h_eff ** (7.0 / 3.0)
        walls = 0.0
        if ks is not None:
            friction_velocity = _friction_velocity_reference(q / h, 0.5 * WIDTH, ks)
            walls = 2.0 * h / WIDTH * friction_velocity ** 2
        return G * h * SLOPE - bed - walls

    return brentq(balance, 0.02, 2.0)


def test_uniform_flow_with_wall_friction(tmp_path) -> None:
    """Steady depth = analytic normal depth with smooth-wall friction (0.1% mid-channel)."""
    # Setup: exit stage at the normal depth (the bed at the exit is z = 0).
    h_normal = _normal_depth(ks=0.0)
    _write_flume(tmp_path, wse_out=h_normal, h_initial=h_normal, ks=0.0)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    assert case.model.wall_friction.mode == "loglaw"

    # Run: steady Newton solve.
    U, converged = final_state(case.model, {}, case.U0, case.ode)
    assert bool(converged)

    # Compare: depth in the middle of the flume against h_n.
    depth = np.asarray(U[0] + case.model.hstill)
    x = case.mesh.centroid[:, 0]
    middle = np.abs(x - 0.5 * LENGTH) < 0.1 * LENGTH
    error = np.max(np.abs(depth[middle] - h_normal)) / h_normal
    assert error < 1e-3, (h_normal, depth[middle].min(), depth[middle].max())

    # The walls matter here: free-slip walls would give a clearly smaller depth.
    assert h_normal > 1.05 * _normal_depth(ks=None)


def test_steady_gradient_wrt_wall_roughness(tmp_path) -> None:
    """d(mean depth)/d(k_s of the walls) by the IFT adjoint matches central differences."""
    # Setup: rough walls, k_s = 2 mm, flume at its normal depth.
    ks = 2.0e-3
    h_normal = _normal_depth(ks=ks)
    _write_flume(tmp_path, wse_out=h_normal, h_initial=h_normal, ks=ks)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model

    def mean_depth(wall_ks: jax.Array) -> jax.Array:
        """Mean steady depth (m) for the wall roughness k_s (m)."""
        U, _ = final_state(model, {"wall_ks": wall_ks}, case.U0, case.ode)
        return jnp.mean(U[0] + model.hstill)

    # Run: adjoint gradient and central differences.
    gradient = float(jax.grad(mean_depth)(jnp.asarray(ks)))
    step = 1.0e-5
    plus = mean_depth(jnp.asarray(ks + step))
    minus = mean_depth(jnp.asarray(ks - step))
    finite_difference = float((plus - minus) / (2.0 * step))

    # Compare: rougher walls -> deeper flow; adjoint = finite differences.
    assert finite_difference > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4)
