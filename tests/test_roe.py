"""Property tests of the xi-form Roe flux (antisymmetry, dry branches, finite gradients).

``hydrograd.physics.roe.roe_flux`` is the face flux of the Julia-ported scheme
(``scheme="julia_xi_roe"``). It takes the left and right states (xi, h_still, h, q_x, q_y,
z_b) and the unit normal (n_x, n_y) and returns the three flux components. The tests use
random states in which about 30 % of the cells are dry (h = h_small = 1e-3 m, q = 0), so
every wet/dry branch of the flux is exercised:

- antisymmetry: swapping L and R and reversing the normal negates the flux (1e-12);
- lake at rest: a flat free surface (xi = 0) over a varying bed gives zero flux (1e-12);
- reverse-mode gradients through all branches are finite (no NaN from sqrt or division
  at dry cells).
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hydrograd.physics.roe import roe_flux

# Gravity g (m/s^2) and the dry-cell depth h_small (m).
G = 9.81
HMIN = 1.0e-3


def _random_states(n: int, seed: int, dry_frac: float = 0.3):
    """Random left/right face states over a bed below a still-water level of 1 m.

    Args:
        n: number of faces.
        seed: seed of the NumPy random generator.
        dry_frac: fraction of cells set dry (h = HMIN, q = 0).

    Returns:
        ``(xi, hs, h, q, zb, nx, ny)``: xi, h_still, h and z_b (m) with shape [2, n]
        (index 0 = left, 1 = right); unit discharges q (m^2/s) with shape [2, 2, n]
        (side, component x/y, face); normal components nx, ny with shape [n].
    """
    rng = np.random.default_rng(seed)

    # Bed elevation and still-water depth for a still-water level at WSE = 1 m.
    zb = rng.uniform(0.0, 0.5, (2, n))
    hs = np.maximum(1.0 - zb, 0.0)                      # still water at WSE = 1 m

    # Random depths; a fraction of cells is dry.
    h = rng.uniform(0.0, 1.5, (2, n))
    h[rng.random((2, n)) < dry_frac] = HMIN             # dry cells sit exactly at h_small
    xi = h - hs

    # Random discharges in wet cells only, and random face orientations.
    q = rng.normal(0.0, 0.5, (2, 2, n)) * (h > HMIN)
    angle = rng.uniform(0, 2 * np.pi, n)
    return xi, hs, h, q, zb, np.cos(angle), np.sin(angle)


def test_antisymmetry_all_branches() -> None:
    """F(L, R, n) = -F(R, L, -n) for every component, to rtol = atol = 1e-12."""
    xi, hs, h, q, zb, nx, ny = _random_states(4000, 0)

    # Flux from left to right, and from right to left with the normal reversed.
    flux_lr = roe_flux(
        xi[0], hs[0], h[0], q[0, 0], q[0, 1], zb[0],
        xi[1], hs[1], h[1], q[1, 0], q[1, 1], zb[1],
        nx, ny, G, HMIN,
    )
    flux_rl = roe_flux(
        xi[1], hs[1], h[1], q[1, 0], q[1, 1], zb[1],
        xi[0], hs[0], h[0], q[0, 0], q[0, 1], zb[0],
        -nx, -ny, G, HMIN,
    )

    # The two must be opposite, component by component.
    for component_lr, component_rl in zip(flux_lr, flux_rl):
        np.testing.assert_allclose(
            np.asarray(component_lr), -np.asarray(component_rl), rtol=1e-12, atol=1e-12
        )


def test_lake_at_rest_flux_balance() -> None:
    """A lake at rest over a sloping bed gives |flux| < 1e-12 in all components."""
    # Flat WSE: xi = 0 on both sides -> pressure flux vanishes, no mass flux.
    n = 100
    zb = np.linspace(0.0, 0.9, 2 * n).reshape(2, n)
    hs = 1.0 - zb
    zeros = np.zeros(n)

    # Still water: h = h_still, q = 0; normal (1, 0).
    flux = roe_flux(
        zeros, hs[0], hs[0], zeros, zeros, zb[0],
        zeros, hs[1], hs[1], zeros, zeros, zb[1],
        np.ones(n), zeros, G, HMIN,
    )
    for component in flux:
        assert np.abs(np.asarray(component)).max() < 1e-12


def test_gradients_finite() -> None:
    """The gradient of sum(F^2) with respect to the left q_x is finite everywhere."""
    xi, hs, h, q, zb, nx, ny = _random_states(500, 1)

    def total(qx0):
        """Sum of the squared flux components (scalar) as a function of the left q_x."""
        flux = roe_flux(
            xi[0], hs[0], h[0], qx0, q[0, 1], zb[0],
            xi[1], hs[1], h[1], q[1, 0], q[1, 1], zb[1],
            nx, ny, G, HMIN,
        )
        return sum(jnp.sum(component ** 2) for component in flux)

    # Reverse-mode gradient through all wet/dry branches.
    gradient = jax.grad(total)(jnp.asarray(q[0, 0]))
    assert np.all(np.isfinite(np.asarray(gradient)))
