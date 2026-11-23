"""Sediment transport closures shared by the 1-D mode and (in the future) 2-D transport.

The subpackage has no dependence on cross-sections or meshes: its functions take arrays
over the size classes [K] and the locations [N] (cross-sections in 1-D, cells in 2-D).

Modules
-------
config     Physical constants, smoothing widths, size classes, default parameters.
closures   Settling velocity, suspended fraction, Engelund-Hansen capacity with hiding,
           Krone and Partheniades cohesive fluxes, adaptation lengths.
exchange   Source and rate of the non-equilibrium total-load equation, with the supply
           limit of the bed.
bed        Hirano active layer, substrate exchange, erodible thickness above a floor.
observation  Bulk deposit volumes with Lane-Koelzer consolidation.

The formulation follows the USBR SRH-1D v4.0 manual (chapter 3).
"""

from hydrograd.sediment.bed import BedComposition, InterfaceMix, update_composition
from hydrograd.sediment.closures import ClosureState, FlowState, compute_closures
from hydrograd.sediment.config import (
    PhysicalConstants,
    SedimentConfig,
    SedimentSmoothing,
    default_bed_parameters,
    default_sediment_parameters,
    interface_parameters,
    wilcock_crowe_parameters,
)
from hydrograd.sediment.exchange import select_locations, source_and_rate
from hydrograd.sediment.observation import bulk_volume_uncompacted, consolidated_volume

__all__ = [
    "BedComposition",
    "ClosureState",
    "FlowState",
    "InterfaceMix",
    "PhysicalConstants",
    "SedimentConfig",
    "SedimentSmoothing",
    "bulk_volume_uncompacted",
    "compute_closures",
    "consolidated_volume",
    "default_bed_parameters",
    "default_sediment_parameters",
    "interface_parameters",
    "select_locations",
    "source_and_rate",
    "update_composition",
    "wilcock_crowe_parameters",
]
