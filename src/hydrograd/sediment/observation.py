"""Observation operators for deposits: bulk volumes with and without consolidation.

A bathymetric survey measures the bulk volume of a deposit, which depends on its dry bulk
density. The model tracks the solids volume V_k (m^3) of each size class deposited at each
location in each time step; the bulk volume of a cohort is

    V_bulk = V_k rho_s / gamma_k,

with the grain density rho_s and the dry bulk density gamma_k (kg/m^3).

Consolidation (Lane and Koelzer 1943): a cohort of age T (years) has the density

    gamma_k(T) = gamma1_k + K_k log10(T_eff),     T_eff = 1 + softplus(T - 1),

i.e. the initial density gamma1_k in the first year, and a logarithmic increase with the
coefficient K_k afterwards (the softplus makes the start smooth). Consolidation is applied
in the observation only: the bed elevation of the model uses gamma1. Eroded material
(negative deposition) leaves at its initial density.

The functions work on arrays over time steps [T], size classes [K] and locations [N]
(cross-sections in 1-D, cells in 2-D).

Units: SI.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from hydrograd.sediment.config import SedimentConfig
from hydrograd.smooth import softplus_beta

__all__ = ["bulk_volume_uncompacted", "consolidated_volume"]

# Differentiable parameters, e.g. {"gamma1": ..., "K_cons": ...}.
Theta = dict[str, Any]


def consolidated_volume(
    deposition: Array, time_end: Array, time_survey: float, theta: Theta, config: SedimentConfig,
) -> Array:
    """Bulk deposit volume a survey at ``time_survey`` would measure at each location.

    Args:
        deposition: solids volume deposited per step, class and location (m^3), positive
            for deposition, shape [T, K, N].
        time_end: time at the end of each step (s), shape [T].
        time_survey: time of the survey (s); later steps are not counted.
        theta: parameters ``gamma1`` (kg/m^3) and ``K_cons`` (kg/m^3 per log10 year), each
            of shape [K].
        config: sediment settings (grain density, length of a year, smoothing).

    Returns:
        Bulk volume (m^3), shape [N].
    """
    constants = config.constants
    gamma1 = theta["gamma1"]
    K_cons = theta["K_cons"]

    # Steps that ended before the survey, and the age of their deposits in years.
    counted = (time_end <= time_survey).astype(deposition.dtype)
    age_years = jnp.maximum(time_survey - time_end, 0.0) / constants.seconds_per_year

    # Density of each cohort after consolidation, shape [T, K].
    age_effective = 1.0 + softplus_beta(age_years - 1.0, config.smoothing.beta_softplus)
    density_aged = gamma1[None, :] + K_cons[None, :] * jnp.log10(age_effective)[:, None]

    # Deposited cohorts consolidate; eroded material leaves at the initial density.
    density_aged = jnp.broadcast_to(density_aged[:, :, None], deposition.shape)
    density_initial = jnp.broadcast_to(gamma1[None, :, None], deposition.shape)
    density = jnp.where(deposition > 0, density_aged, density_initial)

    # Bulk volume of the counted cohorts, summed over steps and classes.
    volume = deposition * constants.rho_s / density * counted[:, None, None]
    return jnp.sum(volume, axis=(0, 1))


def bulk_volume_uncompacted(deposition: Array, theta: Theta, config: SedimentConfig) -> Array:
    """Bulk deposit volume at the initial density, at each location.

    Args:
        deposition: solids volume deposited per step, class and location (m^3),
            shape [T, K, N].
        theta: parameters with ``gamma1`` (kg/m^3), shape [K].
        config: sediment settings (grain density).

    Returns:
        Bulk volume (m^3), shape [N].
    """
    porosity = 1.0 - theta["gamma1"] / config.constants.rho_s
    volume = deposition / (1.0 - porosity)[None, :, None]
    return jnp.sum(volume, axis=(0, 1))
