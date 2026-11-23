"""Observation operators of the 1-D mode: the storage (area-capacity) curve of a reservoir.

The storage below a water level eta is the sum over the cross-sections of the flow area
at that level times the control-volume length of the section,

    V(eta) = sum_i A_i(eta) dx_cv,i                                        (m^3).

Reservoir surveys are often reported as such a curve; comparing the curves of the initial
and the final bed gives the storage lost to sedimentation at each level. No consolidation
adjustment is made here (see :mod:`hydrograd.sediment.observation` for deposit volumes).

Units: SI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.oned.config import GeometrySmoothing
from hydrograd.oned.geometry import hydraulic_properties

__all__ = ["capacity_curve"]


def capacity_curve(
    z: Array, s: Array, cv_lengths: Array, levels: Array, smoothing: GeometrySmoothing,
) -> Array:
    """Storage volume below each of the given water levels.

    Args:
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        cv_lengths: control-volume length of each section (m), shape [N].
        levels: water levels (m), shape [L].
        smoothing: widths of the smooth waterline.

    Returns:
        Storage volume V (m^3) at each level, shape [L].
    """
    n_sections = z.shape[0]
    ones = jnp.ones(n_sections, dtype=z.dtype)

    def volume_below(level: Array) -> Array:
        """Storage below one level (m^3)."""
        eta = jnp.broadcast_to(level, (n_sections,))
        area = hydraulic_properties(eta, z, s, ones, smoothing).area
        return jnp.sum(area * cv_lengths)

    return jax.vmap(volume_below)(levels)
