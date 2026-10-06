"""Tests of the vegetation drag source (``physics.vegetation``).

The drag of rigid emergent stems per unit plan area (divided by the density) is
1/2 C_D a h |u| u = c_veg q, with c_veg = 1/2 C_D a |u| (1/s). It is added to the bed
friction rate. The tests check:

1. The friction rate equals the bed rate plus 1/2 C_D a |q| / h exactly, is unchanged
   where a = 0, and models without a ``"vegetation"`` block are bit-identical to before.
2. Uniform open-channel flow through vegetation: the steady solution has the normal depth
   h_n of the analytic momentum balance

       g h S = c(h) q,   c(h) = g n^2 |q| / (h + h_small)^(7/3) + 1/2 C_D a |q| / h,

   (the h_small offset is the one used in the code's Manning term). Second-order
   reconstruction (the default) is used: on a sloping bed, first-order hydrostatic
   reconstruction adds a numerical flux of about sqrt(g h) S dx / 2, which for this deep,
   slow flow (Fr ~ 0.01) lowers the cell discharge by 5% at dx = 5 m (error O(dx)).
3. The gradient of a steady-state loss with respect to the zone drag coefficient
   ``theta["veg_CD"]`` (implicit-function-theorem adjoint) matches finite differences.
4. The network closure C_D = f(h, |u|, a) is local: the drag rate of cell i depends on
   the depth of cell i only.
5. The force integrated over a region (``vegetation_force``) equals the analytic stem drag
   rho 1/2 C_D a q^2 / h_n per unit area in uniform flow, and its gradient with respect to
   C_D through the steady solve matches central differences.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from hydrograd.apps.setup import setup_case
from hydrograd.closures.mlp import init_mlp
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.model import Vegetation
from hydrograd.physics.rhs import friction_coefficient
from hydrograd.physics.vegetation import vegetation_drag_rate, vegetation_force
from hydrograd.solvers.final import final_state

# Channel and vegetation of the uniform-flow tests (SI units).
LENGTH = 200.0          # m
WIDTH = 10.0            # m
SLOPE = 1.0e-3          # -
Q_TOTAL = 2.0           # m^3/s
N_MANNING = 0.02        # s/m^(1/3)
A_FRONTAL = 2.0         # 1/m (e.g. d = 0.01 m stems at dS = 0.07 m)
CD = 1.0                # -
G = 9.81                # m/s^2
H_SMALL = 1.0e-3        # m, the model's dry threshold


def _write_vegetated_channel(case_dir, wse_out: float, h_initial: float) -> None:
    """Write a straight sloped channel with a vegetation block and a steady solver.

    Args:
        case_dir: output folder (pytest ``tmp_path``).
        wse_out: water-surface elevation at the exit (m); the bed there is at z = 0.
        h_initial: initial depth everywhere (m), the initial guess of the Newton solve.
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

    # Add the vegetation block: the same stems over the whole channel.
    control_path = case_dir / "run_control.json"
    control = json.loads(control_path.read_text())
    control["vegetation"] = {"a": A_FRONTAL, "CD": CD}
    control_path.write_text(json.dumps(control, indent=1))


def _normal_depth(with_stems: bool = True) -> float:
    """Normal depth h_n (m) of the analytic momentum balance g h S = c(h) q.

    Args:
        with_stems: include the stem drag (True) or the bed friction only (False).
    """
    q = Q_TOTAL / WIDTH
    stem_factor = 1.0 if with_stems else 0.0

    def balance(h: float) -> float:
        """Gravity minus resistance (m^2/s^2) per unit plan area at depth h (m)."""
        c_bed = G * N_MANNING ** 2 * q / (h + H_SMALL) ** (7.0 / 3.0)
        c_veg = stem_factor * 0.5 * CD * A_FRONTAL * q / h
        return G * h * SLOPE - (c_bed + c_veg) * q

    return brentq(balance, 0.01, 5.0)


def test_drag_rate_is_bed_plus_stem_drag(tmp_path) -> None:
    """c = c_bed + 1/2 C_D a |q|/h exactly; c = c_bed with no vegetation (bit-identical)."""
    # Setup: a vegetated channel and the same model without vegetation.
    _write_vegetated_channel(tmp_path, wse_out=0.5, h_initial=0.5)
    model = setup_case(read_control(tmp_path / "run_control.json")).model
    model_bare = dataclasses.replace(model, vegetation=Vegetation())
    assert model.vegetation.mode == "zone"

    # A random state with depths 0.2-0.8 m and discharges of both signs.
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.uniform(0.2, 0.8, model.n_cells))
    qx = jnp.asarray(rng.uniform(-0.5, 0.5, model.n_cells))
    qy = jnp.asarray(rng.uniform(-0.2, 0.2, model.n_cells))
    n = jnp.full(model.n_cells, N_MANNING)

    # Run: friction rates with and without vegetation.
    c_with = friction_coefficient(model, {}, h, qx, qy, n)
    c_bare = friction_coefficient(model_bare, {}, h, qx, qy, n)

    # Compare: the difference is the stem drag rate 1/2 C_D a |q| / h.
    q_mag = jnp.sqrt(qx * qx + qy * qy)
    expected = 0.5 * CD * A_FRONTAL * q_mag / h
    np.testing.assert_allclose(c_with - c_bare, expected, rtol=1e-12)

    # Cells without stems (a = 0 through theta["veg_a"]) get no drag.
    a_zero = jnp.zeros(len(model.n_zone))
    c_no_stems = friction_coefficient(model, {"veg_a": a_zero}, h, qx, qy, n)
    assert float(jnp.max(jnp.abs(c_no_stems - c_bare))) == 0.0


def test_uniform_flow_through_vegetation(tmp_path) -> None:
    """The steady depth equals the analytic normal depth (0.1% mid-channel, 0.25% overall).

    The overall tolerance covers the cells next to the inlet and exit, where the boundary
    treatment perturbs the uniform state slightly (largest deviation 0.15% at dx = 5 m).
    """
    # Setup: exit stage at the normal depth (bed at the exit is z = 0).
    h_normal = _normal_depth()
    _write_vegetated_channel(tmp_path, wse_out=h_normal, h_initial=h_normal)
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
    assert middle_error < 1e-3, (h_normal, depth[middle].min(), depth[middle].max())
    assert overall_error < 2.5e-3, (h_normal, depth.min(), depth.max())

    # The cell discharge equals the inflow per unit width.
    np.testing.assert_allclose(np.asarray(U[1])[middle], Q_TOTAL / WIDTH, rtol=1e-3)

    # Without the stems the same channel would carry the flow at a much smaller depth.
    assert h_normal > 1.5 * _normal_depth(with_stems=False)


def test_steady_gradient_wrt_drag_coefficient(tmp_path) -> None:
    """d(mean depth)/d(C_D per zone) by the IFT adjoint matches central differences (1e-4)."""
    # Setup: the normal-depth channel; parameters are C_D per material ID.
    h_normal = _normal_depth()
    _write_vegetated_channel(tmp_path, wse_out=h_normal, h_initial=h_normal)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    CD_zone = jnp.full(len(model.n_zone), CD)

    def mean_depth(CD_values: jax.Array) -> jax.Array:
        """Mean steady depth (m) for the given drag coefficients per zone."""
        U, _ = final_state(model, {"veg_CD": CD_values}, case.U0, case.ode)
        return jnp.mean(U[0] + model.hstill)

    # Run: adjoint gradient and central differences for the vegetated zone (ID 1).
    gradient = jax.grad(mean_depth)(CD_zone)
    step = 1e-4
    plus = mean_depth(CD_zone.at[1].add(step))
    minus = mean_depth(CD_zone.at[1].add(-step))
    finite_difference = (plus - minus) / (2.0 * step)

    # Compare.
    assert float(finite_difference) > 0.0           # more drag -> deeper flow
    np.testing.assert_allclose(float(gradient[1]), float(finite_difference), rtol=1e-4)


def test_network_drag_coefficient_is_local(tmp_path) -> None:
    """With C_D = f(h, |u|, a), d c_veg_i / d h_j = 0 for i != j (a local closure)."""
    # Setup: the channel with a small C_D network.
    _write_vegetated_channel(tmp_path, wse_out=0.5, h_initial=0.5)
    model = setup_case(read_control(tmp_path / "run_control.json")).model
    vegetation = dataclasses.replace(
        model.vegetation,
        mode="nn_CD",
        activations=("tanh", "tanh"),
        CD_bounds=(0.5, 2.0),
        h_bounds=(0.0, 1.0),
        umag_bounds=(0.0, 2.0),
        a_bounds=(0.0, 5.0),
    )
    model = dataclasses.replace(model, vegetation=vegetation)
    theta = {"nn": init_mlp(jax.random.PRNGKey(0), 3, [6, 6], 1)}

    # A random wet state.
    rng = np.random.default_rng(1)
    h = jnp.asarray(rng.uniform(0.2, 0.8, model.n_cells))
    qx = jnp.asarray(rng.uniform(0.1, 0.5, model.n_cells))
    qy = jnp.zeros(model.n_cells)

    def drag_rate_of_depth(depth: jax.Array) -> jax.Array:
        """c_veg (1/s) of every cell as a function of all depths (m)."""
        return vegetation_drag_rate(model, theta, depth, qx, qy)

    # Run: the full Jacobian d c_veg / d h.
    jacobian = np.asarray(jax.jacfwd(drag_rate_of_depth)(h))

    # Compare: only the diagonal is non-zero (and it is non-zero everywhere).
    off_diagonal = jacobian - np.diag(np.diag(jacobian))
    assert np.max(np.abs(off_diagonal)) == 0.0
    assert np.min(np.abs(np.diag(jacobian))) > 0.0


def test_region_force_in_uniform_flow(tmp_path) -> None:
    """Force on the downstream half of the channel: analytic value (0.5%) and FD gradient."""
    # Setup: the normal-depth channel; region = cells in the middle half of the channel
    # (away from the inlet and exit, where the flow is uniform).
    h_normal = _normal_depth()
    _write_vegetated_channel(tmp_path, wse_out=h_normal, h_initial=h_normal)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    x = case.mesh.centroid[:, 0]
    in_region = (x > 0.25 * LENGTH) & (x < 0.75 * LENGTH)
    weights = jnp.asarray(in_region.astype(float))[None, :]
    region_area = float(np.sum(np.asarray(model.area)[in_region]))

    def region_force_x(CD_values: jax.Array) -> jax.Array:
        """Streamwise vegetation force (N) on the region for C_D per material."""
        U, _ = final_state(model, {"veg_CD": CD_values}, case.U0, case.ode)
        return vegetation_force(model, {"veg_CD": CD_values}, U, weights)[0, 0]

    # Run: force at the true C_D, its gradient, and central differences.
    CD_zone = jnp.full(len(model.n_zone), CD)
    force = float(region_force_x(CD_zone))
    gradient = float(jax.grad(region_force_x)(CD_zone)[1])
    step = 1e-4
    plus = region_force_x(CD_zone.at[1].add(step))
    minus = region_force_x(CD_zone.at[1].add(-step))
    finite_difference = float((plus - minus) / (2.0 * step))

    # Compare: analytic force rho 1/2 C_D a q^2 / h_n times the region area.
    q = Q_TOTAL / WIDTH
    expected = 1000.0 * 0.5 * CD * A_FRONTAL * q ** 2 / h_normal * region_area
    np.testing.assert_allclose(force, expected, rtol=5e-3)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4)
