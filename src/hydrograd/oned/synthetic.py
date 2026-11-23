"""Synthetic 1-D test channels: prismatic trapezoids on uniform slopes ending at a dam.

Each cross-section is a trapezoid of four station points,

    s = [-(b/2 + m H), -b/2, b/2, b/2 + m H],     z = z_bed + [H, 0, 0, H],

with bottom width b (m), side slope m (horizontal per vertical) and bank height H (m).
Section 0 is the upstream end and the last section is at the dam.

- :func:`trapezoid_channel`: a single channel.
- :func:`y_network`: two branches that join at the head of a main stem.
- :func:`uniform_bed`: an initial bed state with the same composition at all sections,
  optionally with a non-erodible reach.
- :func:`normal_depth_trapezoid`: normal depth, for tests.

Units: SI.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.oned.bed import BedState
from hydrograd.oned.network import Network, network_from_downstream

__all__ = [
    "YNetwork",
    "normal_depth_trapezoid",
    "trapezoid_channel",
    "uniform_bed",
    "y_network",
]

# Depth of the non-erodible floor below the bed where the bed is erodible (m).
DEEP_FLOOR = 1.0e3

# Substrate thickness of a fully erodible bed (m), and its smallest value on a floor (m).
SUBSTRATE_THICKNESS = 10.0
SUBSTRATE_FLOOR = 0.01


class YNetwork(NamedTuple):
    """A Y-shaped network of trapezoidal channels.

    Attributes:
        network: topology and reach lengths.
        z: bed elevation of the station points (m), shape [N, 4].
        s: station of the points (m), shape [N, 4].
        distance_to_dam: distance to the dam along the flow path (m), shape [N].
        distance_to_junction: distance upstream of the junction (m), shape [N]; zero on
            the main stem.
    """

    network: Network
    z: Array
    s: Array
    distance_to_dam: Array
    distance_to_junction: Array


def _trapezoid_tables(
    bed_elevation: Array, bottom_width: Array, side_slope: float, bank_height: float,
) -> tuple[Array, Array]:
    """Station-elevation tables of trapezoidal sections.

    Args:
        bed_elevation: elevation of the flat bottom of each section (m), shape [N].
        bottom_width: bottom width b of each section (m), shape [N].
        side_slope: side slope m (horizontal per vertical) (-).
        bank_height: bank height H above the bottom (m).

    Returns:
        Tuple ``(z, s)``: elevations and stations (m), each of shape [N, 4].
    """
    half_width = bottom_width / 2.0
    bank_offset = side_slope * bank_height

    # Stations: left bank top, left toe, right toe, right bank top.
    s = jnp.stack(
        [-(half_width + bank_offset), -half_width, half_width, half_width + bank_offset],
        axis=1,
    )

    # Elevations above the bottom: H, 0, 0, H.
    z_row = jnp.asarray([bank_height, 0.0, 0.0, bank_height])
    z = bed_elevation[:, None] + z_row[None, :]

    return z, s


def trapezoid_channel(
    length: float = 10_000.0,
    n_sections: int = 21,
    slope: float = 1.0e-3,
    bottom_width: float = 60.0,
    side_slope: float = 2.0,
    bank_height: float = 30.0,
    z_dam_bed: float = 100.0,
) -> tuple[Array, Array, Array]:
    """A straight trapezoidal channel on a uniform slope.

    Args:
        length: channel length (m).
        n_sections: number of equally spaced cross-sections N.
        slope: bed slope (-); the bed rises upstream.
        bottom_width: bottom width b (m).
        side_slope: side slope m (horizontal per vertical) (-).
        bank_height: bank height H (m).
        z_dam_bed: bed elevation at the dam (m).

    Returns:
        Tuple ``(x, z, s)``: streamwise positions (m), shape [N], and the tables (m), each
        of shape [N, 4].
    """
    x = jnp.linspace(0.0, length, n_sections)
    bed_elevation = z_dam_bed + slope * (length - x)
    widths = jnp.full(n_sections, bottom_width)
    z, s = _trapezoid_tables(bed_elevation, widths, side_slope, bank_height)

    return x, z, s


def y_network(
    main_length: float = 10_000.0,
    n_main: int = 21,
    branch_lengths: tuple[float, float] = (5_000.0, 3_000.0),
    n_branch: tuple[int, int] = (10, 6),
    slope: float = 1.0e-3,
    branch_slopes: tuple[float, float] | None = None,
    bottom_width: float = 60.0,
    branch_widths: tuple[float, float] | None = None,
    side_slope: float = 2.0,
    bank_height: float = 30.0,
    z_dam_bed: float = 100.0,
) -> YNetwork:
    """Two trapezoidal branches that join at the head of a main stem ending at a dam.

    Section order: branch A (upstream to the junction), branch B, then the main stem,
    whose first section is the junction and whose last section is the dam. A branch of
    length L with n sections has them L / n apart, the last one a reach length upstream
    of the junction. The branch beds continue upward from the junction elevation on their
    own slopes.

    Args:
        main_length: length of the main stem (m).
        n_main: number of sections on the main stem, junction and dam included.
        branch_lengths: lengths of branches A and B (m).
        n_branch: number of sections on branches A and B.
        slope: bed slope of the main stem (-).
        branch_slopes: bed slopes of the branches (-); None uses ``slope``.
        bottom_width: bottom width of the main stem (m).
        branch_widths: bottom widths of the branches (m); None uses ``bottom_width``.
        side_slope: side slope (horizontal per vertical) (-).
        bank_height: bank height (m).
        z_dam_bed: bed elevation at the dam (m).

    Returns:
        The network and its tables (:class:`YNetwork`).
    """
    n_a = n_branch[0]
    n_b = n_branch[1]
    length_a = branch_lengths[0]
    length_b = branch_lengths[1]
    slopes = branch_slopes if branch_slopes is not None else (slope, slope)
    widths = branch_widths if branch_widths is not None else (bottom_width, bottom_width)

    # Main stem: positions from the junction (0) to the dam, and bed elevations.
    x_main = jnp.linspace(0.0, main_length, n_main)
    z_main = z_dam_bed + slope * (main_length - x_main)
    z_junction = z_main[0]

    # Branches: distance of each section upstream of the junction.
    distance_a = length_a - jnp.arange(n_a, dtype=float) * (length_a / n_a)
    distance_b = length_b - jnp.arange(n_b, dtype=float) * (length_b / n_b)

    # Bed elevation and bottom width of all sections (A, B, main).
    bed_elevation = jnp.concatenate(
        [z_junction + slopes[0] * distance_a, z_junction + slopes[1] * distance_b, z_main]
    )
    n_sections = bed_elevation.shape[0]
    bottom_widths = jnp.concatenate(
        [jnp.full(n_a, widths[0]), jnp.full(n_b, widths[1]), jnp.full(n_main, bottom_width)]
    )
    z, s = _trapezoid_tables(bed_elevation, bottom_widths, side_slope, bank_height)

    # Topology: each branch is a chain whose last section flows into the junction.
    junction = n_a + n_b
    downstream = np.concatenate(
        [
            np.arange(1, n_a),
            [junction],
            np.arange(n_a + 1, n_a + n_b),
            [junction],
            np.arange(junction + 1, n_sections),
        ]
    )
    reach_length = jnp.concatenate(
        [
            jnp.full(n_a, length_a / n_a),
            jnp.full(n_b, length_b / n_b),
            jnp.full(n_main - 1, main_length / (n_main - 1)),
        ]
    )
    network = network_from_downstream(reach_length, downstream)

    # Distances along the flow path.
    distance_to_dam = jnp.concatenate(
        [main_length + distance_a, main_length + distance_b, main_length - x_main]
    )
    distance_to_junction = jnp.concatenate([distance_a, distance_b, jnp.zeros(n_main)])

    return YNetwork(
        network=network,
        z=z,
        s=s,
        distance_to_dam=distance_to_dam,
        distance_to_junction=distance_to_junction,
    )


def uniform_bed(
    z: Array,
    s: Array,
    n_classes: int = 3,
    fractions: tuple[float, ...] | None = None,
    fixed: Array | None = None,
) -> BedState:
    """Initial bed with the same composition at all sections.

    Args:
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        n_classes: number of size classes K.
        fractions: fractions of the active layer and the substrate (-), length K; None
            gives equal fractions.
        fixed: boolean array, True where the bed is non-erodible (the floor is at the
            initial bed), shape [N]; None gives a fully erodible bed without a floor.

    Returns:
        The bed state. The substrate is 10 m thick where the bed is erodible and 1 cm on
        a floor.
    """
    n_sections = z.shape[0]
    if fractions is None:
        fractions = tuple(1.0 / n_classes for _ in range(n_classes))

    # The same fractions at every section, shape [K, N].
    p = jnp.broadcast_to(jnp.asarray(fractions)[:, None], (n_classes, n_sections))

    # Fully erodible bed.
    if fixed is None:
        return BedState(
            z=z,
            s=s,
            p_active=p,
            p_substrate=p,
            substrate_thickness=jnp.full(n_sections, SUBSTRATE_THICKNESS),
            z_floor=None,
        )

    # Floor at the initial bed where fixed, far below elsewhere; the substrate is the
    # erodible thickness above the floor.
    bed_elevation = jnp.min(z, axis=1)
    z_floor = jnp.where(jnp.asarray(fixed), bed_elevation, bed_elevation - DEEP_FLOOR)
    substrate_thickness = jnp.maximum(bed_elevation - z_floor, SUBSTRATE_FLOOR)

    return BedState(
        z=z,
        s=s,
        p_active=p,
        p_substrate=p,
        substrate_thickness=substrate_thickness,
        z_floor=z_floor,
    )


def normal_depth_trapezoid(
    Q: float,
    n_manning: float,
    slope: float,
    bottom_width: float,
    side_slope: float,
    conveyance: str = "segments",
) -> float:
    """Normal depth of a trapezoidal channel (Manning, SI), for tests.

    Solves Q = K(y) sqrt(S) by the fixed-point iteration y <- y (Q / Q(y))^0.6, with the
    conveyance K of :mod:`hydrograd.oned.geometry`. For the depth y, bottom width b and
    side slope m:

        "segments"   n K = b y^(5/3) + 2 (m y^2 / 2)^(5/3) / (y sqrt(1 + m^2))^(2/3)
        "section"    n K = A R^(2/3),  A = (b + m y) y,  R = A / (b + 2 y sqrt(1 + m^2))

    Args:
        Q: discharge (m^3/s).
        n_manning: Manning's n (s/m^(1/3)).
        slope: bed slope S (-).
        bottom_width: bottom width b (m).
        side_slope: side slope m (horizontal per vertical) (-).
        conveyance: ``"segments"`` (default) or ``"section"``.

    Returns:
        Normal depth (m).
    """
    bank_length_factor = (1.0 + side_slope**2) ** 0.5

    y = 1.0
    for _ in range(200):
        if conveyance == "section":
            area = (bottom_width + side_slope * y) * y
            perimeter = bottom_width + 2.0 * y * bank_length_factor
            radius = area / perimeter
            conveyance_times_n = area * radius ** (2.0 / 3.0)
        else:
            # Bottom slice (area b y, perimeter b) and the two bank slices (area
            # m y^2 / 2, perimeter y sqrt(1 + m^2) each).
            bottom = bottom_width * y ** (5.0 / 3.0)
            bank_area = 0.5 * side_slope * y * y
            bank = bank_area ** (5.0 / 3.0) / (y * bank_length_factor) ** (2.0 / 3.0)
            conveyance_times_n = bottom + 2.0 * bank

        Q_of_y = conveyance_times_n * slope**0.5 / n_manning
        y = y * (Q / Q_of_y) ** 0.6

    return y
