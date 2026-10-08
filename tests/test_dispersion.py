"""Tests of the dispersive momentum flux (momentum coefficient beta, ``physics.dispersion``).

The term adds - div[(beta - 1) h U U] to the momentum equations, with beta per material
zone. The tests use a straight channel one cell wide with a zone of beta = 1.3 in its
middle and check:

1. For a uniform state (h, u) the discrete source is exactly - (beta - 1) h u^2 / (2 dx)
   in the two cells on either side of the leading edge of the zone, the opposite sign at
   the trailing edge, and zero elsewhere; its area integral is zero (conservative).
2. beta = 1 everywhere leaves the residual unchanged (bit-identical).
3. The gradient of the steady mean depth in the zone with respect to beta
   (``theta["disp_beta"]``, implicit-function-theorem adjoint) matches central differences.
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
from hydrograd.physics.dispersion import dispersive_flux_source
from hydrograd.physics.model import Dispersion
from hydrograd.physics.rhs_hr import rhs_hr
from hydrograd.solvers.final import final_state

# Flat channel, one cell wide (SI units).
N_CELLS = 40
LENGTH = 4.0            # m (dx = 0.1 m)
WIDTH = 0.5             # m
DEPTH = 0.3             # m
SPEED = 0.5             # m/s
BETA = 1.3              # momentum coefficient in the zone (-)
ZONE = (1.0, 3.0)       # streamwise extent of the beta zone (m)


def _write_channel(case_dir, beta: float | None) -> None:
    """Write the flat channel with a steady solver and an optional dispersion block.

    Args:
        case_dir: output folder (pytest ``tmp_path``).
        beta: momentum coefficient of material 2 (the middle zone), or None for no
            ``dispersion`` block.
    """

    def zone(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Material 2 in the middle zone, 1 elsewhere."""
        return np.where((x > ZONE[0]) & (x < ZONE[1]), 2, 1)

    write_channel_case(
        case_dir, N_CELLS, 1,
        length=LENGTH,
        width=WIDTH,
        slope=0.0,
        bump=0.0,
        Q=SPEED * DEPTH * WIDTH,
        wse_out=DEPTH,
        n_manning=0.02,
        solver="SteadyNewton()",
        order=2,
        zone_fn=zone,
    )
    if beta is not None:
        control_path = case_dir / "run_control.json"
        control = json.loads(control_path.read_text())
        control["dispersion"] = {"model": "zone", "beta": {"2": beta}}
        control_path.write_text(json.dumps(control, indent=1))


def test_source_at_beta_jumps_is_exact_and_conservative(tmp_path) -> None:
    """Uniform flow: -(beta-1) h u^2 / (2 dx) at the leading edge, opposite at the trailing."""
    # Setup.
    _write_channel(tmp_path, BETA)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    assert model.dispersion.mode == "zone"
    h = jnp.full(model.n_cells, DEPTH)
    u = jnp.full(model.n_cells, SPEED)
    v = jnp.zeros(model.n_cells)

    # Run.
    source = np.asarray(dispersive_flux_source(model, {}, h, u, v))

    # Compare: cells sorted along x; the edges of the zone fall on cell faces.
    order = np.argsort(np.asarray(case.mesh.centroid[:, 0]))
    source_x = source[1, order]
    dx = LENGTH / N_CELLS
    edge_value = (BETA - 1.0) * DEPTH * SPEED**2 / (2.0 * dx)
    first = round(ZONE[0] / dx)      # first cell inside the zone
    last = round(ZONE[1] / dx) - 1   # last cell inside the zone
    expected = np.zeros(N_CELLS)
    expected[[first - 1, first]] = -edge_value
    expected[[last, last + 1]] = edge_value
    np.testing.assert_allclose(source_x, expected, atol=1e-12)
    np.testing.assert_allclose(source[2], 0.0, atol=1e-14)

    # Conservative: the area integral vanishes.
    assert abs(float(np.sum(source[1] * np.asarray(model.area)))) < 1e-14


def test_beta_one_leaves_the_residual_unchanged(tmp_path) -> None:
    """beta = 1 in every zone gives the same residual as no dispersion block."""
    # Setup: the same model with and without a beta = 1 dispersion term.
    _write_channel(tmp_path, None)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    plain = case.model
    n_materials = len(plain.n_zone)
    with_beta = dataclasses.replace(
        plain, dispersion=Dispersion(mode="zone", beta_zone=(1.0,) * n_materials)
    )

    # Run and compare.
    np.testing.assert_array_equal(
        np.asarray(rhs_hr(plain, {}, case.U0)), np.asarray(rhs_hr(with_beta, {}, case.U0))
    )


def test_steady_gradient_wrt_beta(tmp_path) -> None:
    """d(mean depth in the zone)/d(beta) by the IFT adjoint matches central differences."""
    # Setup.
    _write_channel(tmp_path, BETA)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    beta0 = jnp.asarray(model.dispersion.beta_zone)
    x = np.asarray(case.mesh.centroid[:, 0])
    in_zone = jnp.asarray((x > ZONE[0]) & (x < ZONE[1]))

    def zone_depth(beta_zone: jax.Array) -> jax.Array:
        """Mean steady depth (m) in the zone for the momentum coefficients beta_zone."""
        U, _ = final_state(model, {"disp_beta": beta_zone}, case.U0, case.ode)
        depth = U[0] + model.hstill
        return jnp.sum(jnp.where(in_zone, depth, 0.0)) / jnp.sum(in_zone)

    # Run: adjoint gradient with respect to beta of material 2, and central differences.
    gradient = float(jax.grad(zone_depth)(beta0)[2])
    step = 1.0e-4
    unit = jnp.zeros_like(beta0).at[2].set(1.0)
    plus = zone_depth(beta0 + step * unit)
    minus = zone_depth(beta0 - step * unit)
    finite_difference = float((plus - minus) / (2.0 * step))

    # Compare.
    assert finite_difference != 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4)
