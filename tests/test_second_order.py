"""Second-order MUSCL (hydrostatic reconstruction + HLLC): balance, exact Jacobian,
observed order of accuracy, dry-bed dam break, IFT gradient.

All cases are synthetic channels with ``order=2`` (MUSCL reconstruction with a slope limiter).

What is checked:
- Lake at rest (WSE 0.5 m) over a rough bed with emerged parts, for both limiters: the residual
  vanishes to 1e-13.
- The colored sparse Jacobian equals the dense AD Jacobian to 1e-12 (absolute).
- Subcritical flow over a smooth bump (``benchmarks/validation/convergence_bump.py``): the
  steady Newton solution converges to the exact Bernoulli depth with an observed order above
  1.75 in the mean absolute depth error on meshes of 50, 100 and 200 cells.
- Ritter dry-bed dam break (h_L = 1 m, t = 60 s, CFL 0.25): h >= 0, mass to 1e-10, mean depth
  error below 0.6 mm (the first-order scheme gives 2.6 mm).
- The steady IFT gradient with respect to a zone Manning's n matches a central finite
  difference to 1e-5 (relative).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.physics.rhs import rhs
from hydrograd.solvers.jacobian import dense_jacobian, full_jacobian, linearize
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.steady import NewtonSettings, newton_solve, steady_state
from hydrograd.validation.dambreak import dam_break

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
from convergence_bump import H_OUT, Q, bump, exact_depth


def _flat_wse(x, y):
    """Flat initial water-surface elevation of 0.5 m (shape of x)."""
    return 0.5 + 0 * x


@pytest.mark.parametrize("limiter", ["venkatakrishnan", "barth_jespersen"])
def test_lake_at_rest_wet_dry_is_exact(tmp_path, limiter: str) -> None:
    """Lake at rest over a rough, partly emerged bed: residual below 1e-13 for each limiter."""

    def rough_bed(X, Y):
        """Bed (m): a 1 m Gaussian hump at x = 500 m plus 0.2 m sinusoidal roughness."""
        return np.exp(-((X - 500) / 50) ** 2) + 0.2 * np.sin(X / 37.0) * np.cos(Y / 13.0)

    write_channel_case(
        tmp_path,
        100,
        10,
        length=1000.0,
        width=100.0,
        inlet="WALL",
        outlet="WALL",
        wse_out=0.5,
        ic_wse=_flat_wse,
        order=2,
        zb_fn=rough_bed,
    )
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = dataclasses.replace(case.model, limiter=limiter)

    # The case must contain dry cells
    h = np.asarray(case.U0[0] + model.hstill)
    assert (h <= 0).sum() > 0

    # Residual at the initial state
    assert np.abs(np.asarray(rhs(model, {}, case.U0))).max() < 1e-13


def test_order2_jacobian_is_exact(tmp_path) -> None:
    """Second-order colored sparse Jacobian vs dense AD Jacobian (atol 1e-12)."""

    def sloped_bed(X, Y):
        """Bed (m): slope 1e-3 plus a 0.3 m hump at x = 100 m, modulated across the channel."""
        return 1e-3 * (200 - X) + 0.3 * np.exp(-((X - 100) / 20) ** 2) * np.cos(Y / 15.0)

    write_channel_case(tmp_path, 20, 5, length=200.0, width=50.0, order=2, zb_fn=sloped_bed)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model

    # A perturbed state, so the limiter is active in a non-trivial way
    U = case.U0 + 0.05 * jnp.sin(jnp.arange(3 * model.n_cells).reshape(3, -1) * 0.7)

    # Sparse Jacobian from the colored linearization vs the dense Jacobian
    compressed, A, C = jax.jit(lambda U: linearize(model, {}, U))(U)
    J = full_jacobian(model, np.asarray(compressed), np.asarray(A), np.asarray(C))
    np.testing.assert_allclose(J, dense_jacobian(model, {}, U), atol=1e-12)


def _bump_bed(X, Y):
    """Bed elevation (m) of the convergence-test bump, a function of x only."""
    return bump(X)


def _outlet_wse(x, y):
    """Flat initial WSE equal to the outlet stage H_OUT (m), shape of x."""
    return H_OUT + 0 * x


def test_observed_order_on_subcritical_bump(tmp_path) -> None:
    """Mean absolute depth error vs the exact solution on 50/100/200 cells: order > 1.75."""
    errors = []
    for nx in (50, 100, 200):
        # Near-frictionless 25 m flume with the bump, nx cells along the channel
        case_dir = tmp_path / f"n{nx}"
        write_channel_case(
            case_dir,
            nx,
            1,
            length=25.0,
            width=1.0,
            n_manning=1e-4,
            Q=Q,
            wse_out=H_OUT,
            wstill=H_OUT,
            order=2,
            zb_fn=_bump_bed,
            ic_wse=_outlet_wse,
        )
        case = setup_case(read_control(case_dir / "run_control.json"))

        # Steady Newton solve to a residual norm below 1e-10
        U, n_iterations, residual_norm = jax.jit(
            lambda U0: newton_solve(case.model, {}, U0, NewtonSettings(dtau0=0.05))
        )(case.U0)
        assert float(residual_norm) < 1e-10

        # Mean absolute depth error (m)
        h = np.asarray(U[0] + case.model.hstill)
        errors.append(np.mean(np.abs(h - exact_depth(case.mesh.centroid[:, 0]))))

    # Observed orders between successive refinements
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert rates.min() > 1.75, rates


def test_order2_dry_dam_break(tmp_path) -> None:
    """Ritter dam break at second order: positivity, mass to 1e-10, mean depth error < 0.6 mm."""

    def dam_wse(x, y):
        """Initial WSE (m): 1 m upstream of the dam at x = 500 m, dry downstream."""
        return np.where(x < 500.0, 1.0, 0.0)

    write_channel_case(
        tmp_path,
        1000,
        2,
        length=1000.0,
        width=10.0,
        slope=0.0,
        bump=0.0,
        n_manning=0.0,
        inlet="WALL",
        outlet="WALL",
        wse_out=0.0,
        wstill=0.5,
        t_end=60.0,
        dt=0.05,
        solver="SSPRK3()",
        order=2,
        ic_wse=dam_wse,
    )
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    settings = dataclasses.replace(case.ode, cfl=0.25)

    # Run to t = 60 s and take the final depth (m)
    U = np.asarray(jax.jit(lambda U0: solve_swe(model, {}, U0, settings))(case.U0).ys[-1])
    h = U[0] + np.asarray(model.hstill)

    # Exact Ritter solution and the initial water volume (m^3)
    h_exact, _ = dam_break(case.mesh.centroid[:, 0], 60.0, 1.0, 0.0, x0=500.0)
    area = np.asarray(model.area)
    mass0 = np.sum((np.asarray(case.U0[0]) + np.asarray(model.hstill)) * area)

    # Positivity, mass conservation and accuracy
    assert h.min() > -1e-12 and abs(np.sum(h * area) - mass0) < 1e-10 * mass0
    assert np.mean(np.abs(h - h_exact)) < 6e-4                # first order: 2.6e-3


def test_order2_ift_gradient(tmp_path) -> None:
    """Steady IFT gradient wrt zone-1 Manning's n vs a central FD (step 1e-6), rel 1e-5."""

    def sloped_bed(X, Y):
        """Bed (m): slope 1e-3 plus a 0.2 m Gaussian hump at x = 200 m."""
        return 1e-3 * (400 - X) + 0.2 * np.exp(-((X - 200) / 40) ** 2)

    write_channel_case(
        tmp_path, 40, 4, length=400.0, width=40.0, Q=20.0, wse_out=1.5, order=2, zb_fn=sloped_bed
    )
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    newton = NewtonSettings(dtau0=1.0)

    # Scalar loss: random weighted sum of the steady state
    weights = jnp.asarray(np.random.default_rng(0).normal(size=case.U0.shape))

    def loss_fn(n_zone):
        """Weighted sum of the steady state for zone Manning's n values n_zone."""
        return jnp.sum(weights * steady_state(model, {"n_zone": n_zone}, case.U0, newton))

    loss = jax.jit(loss_fn)

    # IFT gradient vs central FD on zone 1
    n_zone = jnp.asarray(model.n_zone)
    gradient = np.asarray(jax.jit(jax.grad(loss))(n_zone))
    perturbation = jnp.zeros_like(n_zone).at[1].set(1e-6)
    fd = (float(loss(n_zone + perturbation)) - float(loss(n_zone - perturbation))) / 2e-6
    assert abs(gradient[1] - fd) <= 1e-5 * abs(fd)
