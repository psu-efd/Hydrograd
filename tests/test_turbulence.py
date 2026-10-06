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
from hydrograd.physics.turbulence import turbulent_diffusion

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
