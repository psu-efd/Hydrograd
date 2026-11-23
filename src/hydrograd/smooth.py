"""Smooth replacements for non-differentiable operations.

A gradient-based calibration needs every operation between the parameters and the loss to
have a finite, informative derivative. Operations with a kink or a jump (max, min, |x|, a
step) are therefore replaced by smooth functions with a stated transition width. The
functions here are shared by the sediment closures (:mod:`hydrograd.sediment`) and by the
1-D cross-section geometry; they contain no physics.

    softplus_beta(x, beta)     smooth max(x, 0)      width about 1 / beta
    sigmoid_eps(x, eps)        smooth step 0 -> 1    width eps
    smooth_abs(x, eps)         smooth |x|            sqrt(x^2 + eps^2)
    soft_min_one(a, beta)      smooth min(a, 1)      1 - softplus_beta(1 - a)
    expm1_ratio(a)             (1 - exp(-a)) / a     finite (= 1) as a -> 0
    normalize_fractions(p)     fractions floored and rescaled to sum to 1

Units: each function states the units of its arguments; the widths have the units of the
argument they smooth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "expm1_ratio",
    "normalize_fractions",
    "sigmoid_eps",
    "smooth_abs",
    "soft_min_one",
    "softplus_beta",
]

# Smallest argument of expm1_ratio and smallest sum of fractions used as a divisor (-).
TINY = 1.0e-12


def softplus_beta(x: Array, beta: float) -> Array:
    """Smooth max(x, 0): softplus(beta x) / beta.

    The function tends to x for beta x >> 1 and to 0 for beta x << -1; at x = 0 it equals
    ln(2) / beta. It is written as max(beta x, 0) + log(1 + exp(-|beta x|)), which does
    not overflow for large |beta x|.

    Args:
        x: input values (any units), any shape.
        beta: sharpness (1 / units of x); the transition width is about 1 / beta.

    Returns:
        Smoothed max(x, 0) (units of x), same shape as ``x``.
    """
    scaled = beta * x

    # Stable form of log(1 + exp(scaled)).
    positive_part = jnp.maximum(scaled, 0.0)
    correction = jnp.log1p(jnp.exp(-jnp.abs(scaled)))

    return (positive_part + correction) / beta


def sigmoid_eps(x: Array, eps: float) -> Array:
    """Smooth step from 0 to 1: 1 / (1 + exp(-x / eps)).

    Args:
        x: input values (any units), any shape.
        eps: transition width (units of x).

    Returns:
        Values in (0, 1) (-), same shape as ``x``; 0.5 at x = 0.
    """
    return jax.nn.sigmoid(x / eps)


def smooth_abs(x: Array, eps: float) -> Array:
    """Smooth absolute value sqrt(x^2 + eps^2).

    Args:
        x: input values (any units), any shape.
        eps: smoothing width (units of x); the result equals eps at x = 0.

    Returns:
        Smoothed |x| (units of x), same shape as ``x``.
    """
    return jnp.sqrt(x * x + eps * eps)


def soft_min_one(a: Array, beta: float) -> Array:
    """Smooth min(a, 1) = 1 - softplus_beta(1 - a).

    Args:
        a: input values (-), any shape.
        beta: sharpness (-); the transition width is about 1 / beta.

    Returns:
        Smoothed min(a, 1) (-), same shape as ``a``.
    """
    return 1.0 - softplus_beta(1.0 - a, beta)


def expm1_ratio(a: Array) -> Array:
    """phi(a) = (1 - exp(-a)) / a, which tends to 1 as a -> 0.

    It appears in the exact solution of the non-equilibrium transport equation over a
    reach: the load relaxes by the factor 1 - exp(-a) with a = (reach length) / (adaptation
    length), and writing it as a phi(a) keeps the limit of an infinite adaptation length
    (a -> 0) finite.

    Args:
        a: non-negative input values (-), any shape.

    Returns:
        phi(a) (-), same shape as ``a``.
    """
    # Floor a at a tiny positive value, so that the division is defined at a = 0.
    a_safe = jnp.maximum(a, TINY)

    return -jnp.expm1(-a_safe) / a_safe


def normalize_fractions(p: Array, axis: int = 0, floor: float = 0.0) -> Array:
    """Floor the fractions at ``floor`` and rescale them to sum to one along ``axis``.

    Args:
        p: fractions (-), any shape; e.g. [n_classes, n_locations].
        axis: axis that runs over the size classes.
        floor: lower bound applied to each fraction before rescaling (-).

    Returns:
        Fractions (-) that sum to one along ``axis``, same shape as ``p``.
    """
    # Apply the floor, then divide by the sum (floored to avoid a division by zero).
    floored = jnp.maximum(p, floor)
    total = jnp.sum(floored, axis=axis, keepdims=True)

    return floored / jnp.maximum(total, TINY)
