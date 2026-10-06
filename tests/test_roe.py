"""Property tests of the xi-form Roe flux (antisymmetry, dry branches, finite gradients)."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hydrograd.physics.roe import roe_flux

G, HMIN = 9.81, 1.0e-3


def _random_states(n: int, seed: int, dry_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    zb = rng.uniform(0.0, 0.5, (2, n))
    hs = np.maximum(1.0 - zb, 0.0)                      # still water at WSE = 1 m
    h = rng.uniform(0.0, 1.5, (2, n))
    h[rng.random((2, n)) < dry_frac] = HMIN             # dry cells sit exactly at h_small
    xi = h - hs
    q = rng.normal(0.0, 0.5, (2, 2, n)) * (h > HMIN)
    th = rng.uniform(0, 2 * np.pi, n)
    return xi, hs, h, q, zb, np.cos(th), np.sin(th)


def test_antisymmetry_all_branches() -> None:
    xi, hs, h, q, zb, nx, ny = _random_states(4000, 0)
    fLR = roe_flux(xi[0], hs[0], h[0], q[0, 0], q[0, 1], zb[0],
                   xi[1], hs[1], h[1], q[1, 0], q[1, 1], zb[1], nx, ny, G, HMIN)
    fRL = roe_flux(xi[1], hs[1], h[1], q[1, 0], q[1, 1], zb[1],
                   xi[0], hs[0], h[0], q[0, 0], q[0, 1], zb[0], -nx, -ny, G, HMIN)
    for a, b in zip(fLR, fRL):
        np.testing.assert_allclose(np.asarray(a), -np.asarray(b), rtol=1e-12, atol=1e-12)


def test_lake_at_rest_flux_balance() -> None:
    # Flat WSE: xi = 0 on both sides -> pressure flux vanishes, no mass flux.
    n = 100
    zb = np.linspace(0.0, 0.9, 2 * n).reshape(2, n)
    hs = 1.0 - zb
    z = np.zeros(n)
    f = roe_flux(z, hs[0], hs[0], z, z, zb[0], z, hs[1], hs[1], z, z, zb[1],
                 np.ones(n), z, G, HMIN)
    for comp in f:
        assert np.abs(np.asarray(comp)).max() < 1e-12


def test_gradients_finite() -> None:
    xi, hs, h, q, zb, nx, ny = _random_states(500, 1)

    def total(qx0):
        f = roe_flux(xi[0], hs[0], h[0], qx0, q[0, 1], zb[0],
                     xi[1], hs[1], h[1], q[1, 0], q[1, 1], zb[1], nx, ny, G, HMIN)
        return sum(jnp.sum(c ** 2) for c in f)

    g = jax.grad(total)(jnp.asarray(q[0, 0]))
    assert np.all(np.isfinite(np.asarray(g)))
