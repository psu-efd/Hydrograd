"""1-D mode of Hydrograd: flow and sediment transport on a network of cross-sections.

The 1-D mode follows the USBR model SRH-1D (as the 2-D solver follows SRH-2D). Its
hydraulics are either a quasi-steady standard-step backwater computation (subcritical
flow with critical-flow controls) or the unsteady shallow-water equations on the
cross-sections (sub- and supercritical flow with hydraulic jumps). Both are different
schemes from the 2-D finite-volume solver, so they have their own modules here. The
sediment closures are shared with (future) 2-D sediment transport and live in
:mod:`hydrograd.sediment`.

Modules
-------
config       Settings (hydraulics, morphodynamics) and default parameters.
network      Tree-shaped channel networks: numbering, discharge accumulation, bookkeeping
             of deposits.
geometry     Cross-section properties from station-elevation tables; critical depth.
backwater    Backwater solver with critical-flow controls and implicit-function gradients.
unsteady     Unsteady shallow-water solver (staggered, explicit) for mixed flow regimes.
bed          Bed state of the cross-sections, supply limit, Exner factor, bed update.
routing      Non-equilibrium total-load routing with a locally implicit bed feedback.
forcing      Boundary-condition time series and the sediment rating curve.
stepper      Time loop with checkpointed reverse-mode derivatives.
observation  Storage (area-capacity) curve.
synthetic    Trapezoidal test channels, a Y-shaped network, uniform initial beds.
"""

from hydrograd.oned.backwater import HydroState, NewtonState, attach, converge, solve
from hydrograd.oned.bed import BedState
from hydrograd.oned.config import (
    BackwaterNumerics,
    GeometrySmoothing,
    HydraulicsConfig,
    MorphNumerics,
    OneDConfig,
    UnsteadyNumerics,
    default_parameters,
)
from hydrograd.oned.forcing import Forcing, constant_forcing
from hydrograd.oned.geometry import SectionProperties, hydraulic_properties
from hydrograd.oned.network import (
    Network,
    as_network,
    chain_network,
    network_from_downstream,
    network_from_positions,
    section_discharge,
    with_withdrawals,
)
from hydrograd.oned.stepper import RunResult, run
from hydrograd.oned.unsteady import UnsteadyState

__all__ = [
    "BackwaterNumerics",
    "BedState",
    "Forcing",
    "GeometrySmoothing",
    "HydraulicsConfig",
    "HydroState",
    "MorphNumerics",
    "Network",
    "NewtonState",
    "OneDConfig",
    "RunResult",
    "SectionProperties",
    "UnsteadyNumerics",
    "UnsteadyState",
    "as_network",
    "attach",
    "chain_network",
    "constant_forcing",
    "converge",
    "default_parameters",
    "hydraulic_properties",
    "network_from_downstream",
    "network_from_positions",
    "run",
    "section_discharge",
    "solve",
    "with_withdrawals",
]
