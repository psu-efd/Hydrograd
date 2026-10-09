"""Tests of the horizontal turbulent diffusion (``physics.turbulence``).

The momentum equations get div(h nu_t grad u). The tests check:

1. With a constant eddy viscosity and the lateral profile u(y) = U_0 cos(pi y / W) in a
   flat channel of uniform depth h, the term equals the analytic value
   h nu d^2u/dy^2 = -h nu (pi / W)^2 u (the cosine has zero slope at the walls, where the
   scheme has no diffusive flux). Second-order accuracy gives a relative error of about
   (pi dy / W)^2 / 12, i.e. 5e-4 for 40 cells across.
2. The diffusion conserves momentum: sum over cells of area x rate = 0 (round-off).
3. The derivative with respect to the parabolic-model coefficient ``theta["eddy_Ct"]``
   matches central differences.
4. A model without a ``"turbulence"`` block has no diffusion (mode ``none``).
5. Mixing-length model, nu_t = C_t u_* h + (C_m h)^2 |S| + nu_0: in a linear shear flow
   u = gamma y the strain-rate magnitude |S| equals gamma in the cells away from the
   walls, so nu_t = (C_m h)^2 gamma for C_t = nu_0 = 0.
6. With C_m = 0 the mixing-length model gives the parabolic residual bit for bit.
7. The mixing-length viscosity widens the residual stencil to two cells; the colored sparse
   Jacobian then still equals the dense jacfwd Jacobian with first-order numerics.
8. The derivative with respect to ``theta["eddy_Cm"]`` matches central differences.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.model import Turbulence
from hydrograd.physics.rhs import rhs
from hydrograd.physics.turbulence import eddy_viscosity, turbulent_diffusion
from hydrograd.solvers.jacobian import dense_jacobian, full_jacobian, linearize

WIDTH = 10.0            # m
DEPTH = 0.5             # m
U0 = 0.4                # m/s
NU_T = 0.02             # m^2/s
NY = 40                 # cells across


def _flat_channel_model(tmp_path, turbulence_block: dict | None):
    """Flat channel (4 x NY cells, depth 0.5 m) with an optional turbulence block.

    Args:
        tmp_path: pytest temporary folder.
        turbulence_block: the ``"turbulence"`` entry of the run control, or None.

    Returns:
        (case, centroid y (m) of every cell).
    """
    write_channel_case(tmp_path, 4, NY, length=4.0, width=WIDTH, slope=0.0, bump=0.0,
                       Q=1.0, wse_out=DEPTH)
    if turbulence_block is not None:
        control_path = tmp_path / "run_control.json"
        control = json.loads(control_path.read_text())
        control["turbulence"] = turbulence_block
        control_path.write_text(json.dumps(control, indent=1))

    case = setup_case(read_control(tmp_path / "run_control.json"))
    return case, case.mesh.centroid[:, 1]


def test_cosine_profile_matches_analytic_diffusion(tmp_path) -> None:
    """Constant nu_t: the rate equals -h nu (pi/W)^2 u (relative error < 2e-3)."""
    # Setup: constant eddy viscosity, cosine lateral velocity profile.
    case, y = _flat_channel_model(tmp_path, {"model": "constant", "nu_t": NU_T, "nu0": 0.0})
    model = case.model
    h = jnp.full(model.n_cells, DEPTH)
    u = jnp.asarray(U0 * np.cos(np.pi * y / WIDTH))
    v = jnp.zeros(model.n_cells)
    n = jnp.full(model.n_cells, 0.03)

    # Run.
    rate = turbulent_diffusion(model, {}, h, u, v, n)

    # Compare with the analytic second derivative.
    expected = -DEPTH * NU_T * (np.pi / WIDTH) ** 2 * np.asarray(u)
    error = np.abs(np.asarray(rate[1]) - expected)
    assert np.max(error) < 2e-3 * np.max(np.abs(expected)), np.max(error)
    assert float(jnp.max(jnp.abs(rate[2]))) == 0.0
    assert float(jnp.max(jnp.abs(rate[0]))) == 0.0


def test_diffusion_conserves_momentum(tmp_path) -> None:
    """sum(area * rate) = 0 for a random state with the parabolic model."""
    # Setup: parabolic model and a random wet state.
    case, _ = _flat_channel_model(tmp_path, {"model": "parabolic", "coefficient": 0.7})
    model = case.model
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.uniform(0.2, 1.0, model.n_cells))
    u = jnp.asarray(rng.uniform(-0.5, 0.5, model.n_cells))
    v = jnp.asarray(rng.uniform(-0.5, 0.5, model.n_cells))
    n = jnp.full(model.n_cells, 0.03)

    # Run.
    rate = turbulent_diffusion(model, {}, h, u, v, n)

    # Compare: total momentum change is zero to round-off.
    total_x = float(jnp.sum(rate[1] * model.area))
    total_y = float(jnp.sum(rate[2] * model.area))
    scale = float(jnp.sum(jnp.abs(rate[1]) * model.area))
    assert abs(total_x) < 1e-12 * scale
    assert abs(total_y) < 1e-12 * scale


def test_gradient_wrt_eddy_coefficient(tmp_path) -> None:
    """d(sum of squared rates)/d(C_t) by AD matches central differences (rtol 1e-6)."""
    # Setup: parabolic model, a sheared state.
    case, y = _flat_channel_model(tmp_path, {"model": "parabolic", "coefficient": 0.7})
    model = case.model
    h = jnp.full(model.n_cells, DEPTH)
    u = jnp.asarray(U0 * (1.0 + np.tanh((y - 0.5 * WIDTH) / 1.0)))
    v = jnp.zeros(model.n_cells)
    n = jnp.full(model.n_cells, 0.03)

    def objective(coefficient: jax.Array) -> jax.Array:
        """Sum of squared x-momentum rates (m^4/s^4) for C_t = coefficient."""
        rate = turbulent_diffusion(model, {"eddy_Ct": coefficient}, h, u, v, n)
        return jnp.sum(rate[1] ** 2)

    # Run: AD gradient and central differences.
    gradient = float(jax.grad(objective)(0.7))
    step = 1e-4
    finite_difference = float((objective(0.7 + step) - objective(0.7 - step)) / (2 * step))

    # Compare.
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6)


def test_no_turbulence_block_means_no_diffusion(tmp_path) -> None:
    """Without the block the model has mode ``none``; adding the block changes the mode."""
    case, _ = _flat_channel_model(tmp_path, None)
    assert case.model.turbulence == Turbulence()

    with_block = dataclasses.replace(case.model, turbulence=Turbulence(mode="constant", nu_constant=NU_T))
    assert with_block.turbulence.mode == "constant"


def test_mixing_length_viscosity_in_linear_shear(tmp_path) -> None:
    """u = gamma y: nu_t = (C_m h)^2 gamma away from the walls (C_t = nu_0 = 0)."""
    # Setup: mixing-length model without the parabolic and background parts.
    mixing_coefficient = 0.4
    gamma = 0.05                                       # shear rate du/dy (1/s)
    block = {
        "model": "mixing_length",
        "coefficient": 0.0,
        "nu0": 0.0,
        "mixing_coefficient": mixing_coefficient,
    }
    case, y = _flat_channel_model(tmp_path, block)
    model = case.model
    h = jnp.full(model.n_cells, DEPTH)
    u = jnp.asarray(gamma * y)
    v = jnp.zeros(model.n_cells)
    n = jnp.full(model.n_cells, 0.03)

    # Run.
    nu_t = np.asarray(eddy_viscosity(model, {}, h, u, v, n))

    # Compare in the cells not next to a wall (Green-Gauss is exact there for a linear field).
    dy = WIDTH / NY
    interior = (y > dy) & (y < WIDTH - dy)
    expected = (mixing_coefficient * DEPTH) ** 2 * np.sqrt(gamma**2 + 1e-12)
    np.testing.assert_allclose(nu_t[interior], expected, rtol=1e-12)


def test_mixing_length_with_zero_coefficient_is_parabolic(tmp_path) -> None:
    """C_m = 0: the mixing-length residual equals the parabolic residual bit for bit."""
    # Setup: the same case with both models.
    case, _ = _flat_channel_model(tmp_path, {"model": "parabolic", "coefficient": 0.7})
    parabolic = case.model
    mixing = dataclasses.replace(
        parabolic,
        turbulence=dataclasses.replace(
            parabolic.turbulence, mode="mixing_length", mixing_coefficient=0.0
        ),
    )
    U = case.U0 + 0.01 * jnp.sin(jnp.arange(3 * parabolic.n_cells).reshape(3, -1) * 0.3)

    # Run and compare.
    np.testing.assert_array_equal(np.asarray(rhs(parabolic, {}, U)), np.asarray(rhs(mixing, {}, U)))


def test_mixing_length_sparse_jacobian_first_order(tmp_path) -> None:
    """First-order numerics + mixing length: colored Jacobian = dense Jacobian."""
    # Setup: small first-order channel with the mixing-length model and a sheared state.
    write_channel_case(tmp_path, 6, 5, length=6.0, width=2.5, slope=0.0, bump=0.0,
                       Q=0.5, wse_out=DEPTH, order=1)
    control_path = tmp_path / "run_control.json"
    control = json.loads(control_path.read_text())
    control["turbulence"] = {"model": "mixing_length", "coefficient": 0.7,
                             "mixing_coefficient": 0.5}
    control_path.write_text(json.dumps(control, indent=1))
    case = setup_case(read_control(control_path))
    model = case.model
    assert model.order == 1
    U = case.U0 + 0.05 * jnp.sin(jnp.arange(3 * model.n_cells).reshape(3, -1) * 0.7)

    # Run: sparse Jacobian from the colored linearization, and the dense reference.
    compressed, A, C = jax.jit(lambda state: linearize(model, {}, state))(U)
    J = full_jacobian(model, np.asarray(compressed), np.asarray(A), np.asarray(C))

    # Compare.
    np.testing.assert_allclose(J, dense_jacobian(model, {}, U), atol=1e-12)


def test_gradient_wrt_mixing_coefficient(tmp_path) -> None:
    """d(sum of squared rates)/d(C_m) by AD matches central differences (rtol 1e-6)."""
    # Setup: mixing-length model, a shear layer across the channel.
    block = {"model": "mixing_length", "coefficient": 0.7, "mixing_coefficient": 0.3}
    case, y = _flat_channel_model(tmp_path, block)
    model = case.model
    h = jnp.full(model.n_cells, DEPTH)
    u = jnp.asarray(U0 * (1.0 + np.tanh((y - 0.5 * WIDTH) / 1.0)))
    v = jnp.zeros(model.n_cells)
    n = jnp.full(model.n_cells, 0.03)

    def objective(coefficient: jax.Array) -> jax.Array:
        """Sum of squared x-momentum rates (m^4/s^4) for C_m = coefficient."""
        rate = turbulent_diffusion(model, {"eddy_Cm": coefficient}, h, u, v, n)
        return jnp.sum(rate[1] ** 2)

    # Run: AD gradient and central differences.
    gradient = float(jax.grad(objective)(0.3))
    step = 1e-4
    finite_difference = float((objective(0.3 + step) - objective(0.3 - step)) / (2 * step))

    # Compare.
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6)
