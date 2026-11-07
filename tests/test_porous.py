"""Tests of the Darcy-Forchheimer drag of porous zones (``physics.porous``).

The drag per unit plan area (divided by the density) is (alpha + beta |u|) u h = c_df q,
with c_df = alpha + beta |u| (1/s). It is added to the bed friction rate. The tests check:

1. Ergun's coefficients alpha = 150 nu (1 - eps)^2 / (eps^3 d^2) and
   beta = 1.75 (1 - eps) / (eps^3 d), and the ``"porous"`` block that uses them.
2. The friction rate equals the bed rate plus alpha + beta |q| / h exactly, and models
   without a ``"porous"`` block are bit-identical to before.
3. Uniform open-channel flow through a porous zone: the steady depth equals the normal
   depth of the analytic momentum balance

       g h S = c(h) q,   c(h) = g n^2 |q| / h_eff^(7/3) + alpha + beta |q| / h,

   with h_eff = sqrt(h^2 + h_small^2). On the sloping bed the scheme carries an O(dx)
   error (see ``tests/test_vegetation.py``); for this deep flow the mid-channel depth is
   0.107%, 0.062% and 0.033% above h_n at dx = 5, 2.5 and 1.25 m (first order, as
   expected), so the test at dx = 5 m allows 0.15% mid-channel and 0.35% next to the inlet
   and exit.
4. The gradients of the mean steady depth with respect to ``theta["df_alpha"]`` and
   ``theta["df_beta"]`` (implicit-function-theorem adjoint) match central differences.
5. The force on a region (``porous_force``, and ``drag_force`` of the force-data loss)
   equals the analytic drag rho (alpha + beta q / h_n) q per unit area in uniform flow.
6. ``porous_alpha`` and ``porous_beta`` are valid active parameters of an inversion.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from hydrograd.apps.force_data import drag_force
from hydrograd.apps.params import THETA_KEY, to_theta
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.model import Porous
from hydrograd.physics.porous import ergun_coefficients, porous_force
from hydrograd.physics.rhs import friction_coefficient
from hydrograd.solvers.final import final_state

# Channel and porous zone of the uniform-flow tests (SI units).
LENGTH = 200.0          # m
WIDTH = 10.0            # m
SLOPE = 1.0e-3          # -
Q_TOTAL = 2.0           # m^3/s
N_MANNING = 0.02        # s/m^(1/3)
ALPHA = 0.002           # Darcy coefficient (1/s)
BETA = 0.3              # Forchheimer coefficient (1/m)
G = 9.81                # m/s^2


def _write_porous_channel(case_dir, wse_out: float, h_initial: float, block: dict) -> None:
    """Write a straight sloped channel with a ``"porous"`` block and a steady solver.

    Args:
        case_dir: output folder (pytest ``tmp_path``).
        wse_out: water-surface elevation at the exit (m); the bed there is at z = 0.
        h_initial: initial depth everywhere (m), the initial guess of the Newton solve.
        block: the ``"porous"`` block of the run control.
    """

    def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial water surface: the bed z_b = S (L - x) plus a uniform depth (m)."""
        return SLOPE * (LENGTH - x) + h_initial

    write_channel_case(
        case_dir, 40, 2,
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

    # Add the porous block (the same zone over the whole channel).
    control_path = case_dir / "run_control.json"
    control = json.loads(control_path.read_text())
    control["porous"] = block
    control_path.write_text(json.dumps(control, indent=1))


def _normal_depth(with_porous: bool = True) -> float:
    """Normal depth h_n (m) of the analytic momentum balance g h S = c(h) q.

    Args:
        with_porous: include the porous drag (True) or the bed friction only (False).

    Returns:
        h_n (m).
    """
    q = Q_TOTAL / WIDTH
    porous_factor = 1.0 if with_porous else 0.0

    def balance(h: float) -> float:
        """Gravity minus resistance (m^2/s^2) per unit plan area at depth h (m)."""
        h_eff = np.sqrt(h * h + 1.0e-6)          # smooth floor of the default scheme
        c_bed = G * N_MANNING**2 * q / h_eff ** (7.0 / 3.0)
        c_porous = porous_factor * (ALPHA + BETA * q / h)
        return G * h * SLOPE - (c_bed + c_porous) * q

    return brentq(balance, 0.01, 10.0)


def test_ergun_coefficients(tmp_path) -> None:
    """Ergun's alpha and beta, and the porosity form of the run-control block."""
    # Direct evaluation against the formulas (porosity 0.9, d = 1 cm, water).
    alpha, beta = ergun_coefficients(0.9, 0.01, 1.0e-6)
    np.testing.assert_allclose(alpha, 150.0 * 1.0e-6 * 0.1**2 / (0.9**3 * 0.01**2), rtol=1e-14)
    np.testing.assert_allclose(beta, 1.75 * 0.1 / (0.9**3 * 0.01), rtol=1e-14)

    # Porosity 1 gives no drag; invalid input is rejected.
    assert ergun_coefficients(1.0, 0.01) == (0.0, 0.0)
    with pytest.raises(ValueError):
        ergun_coefficients(0.0, 0.01)

    # The block with porosity and d gives the same coefficients; an explicit beta
    # replaces the Ergun value.
    block = {"porosity": 0.9, "d": 0.01, "beta": {"0": 5.0}}
    _write_porous_channel(tmp_path, wse_out=1.0, h_initial=1.0, block=block)
    porous = setup_case(read_control(tmp_path / "run_control.json")).model.porous
    assert porous.mode == "zone"
    np.testing.assert_allclose(porous.alpha_zone, alpha, rtol=1e-14)
    assert porous.beta_zone[0] == 5.0
    np.testing.assert_allclose(porous.beta_zone[1:], beta, rtol=1e-14)


def test_drag_rate_is_bed_plus_porous_drag(tmp_path) -> None:
    """c = c_bed + alpha + beta |q|/h exactly; c = c_bed with no porous block."""
    # Setup: a porous channel and the same model without the porous zone.
    block = {"alpha": ALPHA, "beta": BETA}
    _write_porous_channel(tmp_path, wse_out=1.0, h_initial=1.0, block=block)
    model = setup_case(read_control(tmp_path / "run_control.json")).model
    model_bare = dataclasses.replace(model, porous=Porous())

    # A random state with depths 0.2-0.8 m and discharges of both signs.
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.uniform(0.2, 0.8, model.n_cells))
    qx = jnp.asarray(rng.uniform(-0.5, 0.5, model.n_cells))
    qy = jnp.asarray(rng.uniform(-0.2, 0.2, model.n_cells))
    n = jnp.full(model.n_cells, N_MANNING)

    # Run: friction rates with and without the porous zone.
    c_with = friction_coefficient(model, {}, h, qx, qy, n)
    c_bare = friction_coefficient(model_bare, {}, h, qx, qy, n)

    # Compare: the difference is alpha + beta |q| / h.
    q_mag = jnp.sqrt(qx * qx + qy * qy)
    expected = ALPHA + BETA * q_mag / h
    np.testing.assert_allclose(c_with - c_bare, expected, rtol=1e-12)


def test_uniform_flow_through_porous_zone(tmp_path) -> None:
    """The steady depth equals the analytic normal depth (0.15% mid-channel, 0.35% overall)."""
    # Setup: exit stage at the normal depth (bed at the exit is z = 0).
    h_normal = _normal_depth()
    block = {"alpha": ALPHA, "beta": BETA}
    _write_porous_channel(tmp_path, wse_out=h_normal, h_initial=h_normal, block=block)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model

    # Run: steady Newton solve.
    U, converged = final_state(model, {}, case.U0, case.ode)
    assert bool(converged)

    # Compare: depth in the middle of the channel, and in every cell, against h_n.
    depth = np.asarray(U[0] + model.hstill)
    x = case.mesh.centroid[:, 0]
    middle = np.abs(x - 0.5 * LENGTH) < 0.1 * LENGTH
    middle_error = np.max(np.abs(depth[middle] - h_normal)) / h_normal
    overall_error = np.max(np.abs(depth - h_normal)) / h_normal
    assert middle_error < 1.5e-3, (h_normal, depth[middle].min(), depth[middle].max())
    assert overall_error < 3.5e-3, (h_normal, depth.min(), depth.max())

    # The porous zone raises the normal depth substantially.
    assert h_normal > 1.5 * _normal_depth(with_porous=False)


def test_steady_gradients_wrt_porous_coefficients(tmp_path) -> None:
    """d(mean depth)/d(alpha) and d/d(beta) by the IFT adjoint match central differences."""
    # Setup: the normal-depth channel; parameters are alpha and beta per material ID.
    h_normal = _normal_depth()
    block = {"alpha": ALPHA, "beta": BETA}
    _write_porous_channel(tmp_path, wse_out=h_normal, h_initial=h_normal, block=block)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    n_zones = len(model.n_zone)
    values = {"df_alpha": jnp.full(n_zones, ALPHA), "df_beta": jnp.full(n_zones, BETA)}

    for key, step in (("df_alpha", 1e-6), ("df_beta", 1e-4)):

        def mean_depth(coefficients: jax.Array, key: str = key) -> jax.Array:
            """Mean steady depth (m) for the given coefficients per zone."""
            theta = {key: coefficients}
            U, _ = final_state(model, theta, case.U0, case.ode)
            return jnp.mean(U[0] + model.hstill)

        # Run: adjoint gradient and central differences for the channel zone (ID 1).
        base = values[key]
        gradient = jax.grad(mean_depth)(base)
        plus = mean_depth(base.at[1].add(step))
        minus = mean_depth(base.at[1].add(-step))
        finite_difference = (plus - minus) / (2.0 * step)

        # Compare: more drag gives deeper flow.
        assert float(finite_difference) > 0.0
        np.testing.assert_allclose(float(gradient[1]), float(finite_difference), rtol=1e-4)


def test_region_force_in_uniform_flow(tmp_path) -> None:
    """Porous force on the middle half of the channel equals the analytic drag (0.5%)."""
    # Setup: the normal-depth channel and a region away from the inlet and exit.
    h_normal = _normal_depth()
    block = {"alpha": ALPHA, "beta": BETA}
    _write_porous_channel(tmp_path, wse_out=h_normal, h_initial=h_normal, block=block)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    x = case.mesh.centroid[:, 0]
    in_region = (x > 0.25 * LENGTH) & (x < 0.75 * LENGTH)
    weights = jnp.asarray(in_region.astype(float))[None, :]
    region_area = float(np.sum(np.asarray(model.area)[in_region]))

    # Run: the steady state and the region force.
    U, _ = final_state(model, {}, case.U0, case.ode)
    force = np.asarray(porous_force(model, {}, U, weights))
    total = np.asarray(drag_force(model, {}, U, weights, 1000.0))

    # Compare: rho (alpha + beta q / h_n) q per unit area, times the region area; the
    # force-data loss sees the same force (no vegetation in this model).
    q = Q_TOTAL / WIDTH
    expected = 1000.0 * (ALPHA + BETA * q / h_normal) * q * region_area
    np.testing.assert_allclose(force[0, 0], expected, rtol=5e-3)
    np.testing.assert_allclose(total, force, rtol=1e-14)


def test_porous_active_parameters() -> None:
    """The inversion parameter names map to the porous coefficients in theta."""
    assert THETA_KEY["porous_alpha"] == "df_alpha"
    assert THETA_KEY["porous_beta"] == "df_beta"
    theta = to_theta("porous_beta", jnp.asarray([0.0, BETA]))
    assert list(theta) == ["df_beta"]
