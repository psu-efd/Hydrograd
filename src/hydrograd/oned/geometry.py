"""Cross-section geometry from station-elevation tables.

Tables
------
Every cross-section is a polyline of J points: station s (m, across the channel) and bed
elevation z (m). All N sections have the same J (pad by repeating the last point), so the
tables are arrays ``z[N, J]`` and ``s[N, J]``. Every function is smooth in the water
surface eta and in z, so water levels and bed changes can be differentiated through.

Segment formulas
----------------
Let h = eta - z be the water depth at a station point (negative where the point stands
above the water), hp = softplus(h) its smooth positive part and ha = smooth |h|. For a
segment between points 1 and 2 with horizontal length ds and bed rise dz:

    wet fraction      f = (hp1 + hp2) / (ha1 + ha2)       1 if both wet, h1 / (h1 - h2) if
                                                          only point 1 is wet
    area              A = ds (hp1 + hp2)^2 / (2 (ha1 + ha2))    trapezoid or triangle
    top width         B = ds f
    wetted perimeter  P = sqrt(ds^2 + dz^2) f

These are exact for a piecewise-linear bed away from the waterline and smooth at it. The
section values A, B and P are sums over the segments, and the hydraulic radius is
R = A / P.

Conveyance
----------
With Manning's n (s/m^(1/3)), two formulas are available (``conveyance``):

    "segments" (default)   K = (1/n) sum_k A_k^(5/3) / P_k^(2/3)
    "section"              K = A R^(2/3) / n

``"segments"`` adds the conveyances of the vertical slices between neighbouring station
points (no friction between the slices). With the segment formulas above,

    A_k^(5/3) / P_k^(2/3) = ds^(5/3) (hp1 + hp2)^(8/3) / (2^(5/3) L^(2/3) (ha1 + ha2)),

with L = sqrt(ds^2 + dz^2). This term rises with the water surface for every segment, so
K rises with the water surface for every section. A vertical wall (ds = 0) has no area
and adds nothing: its friction is neglected, and a rectangle of width b and depth h has
K = b h^(5/3) / n.

``"section"`` uses the hydraulic radius of the whole section. When the water spills onto
a bench or a floodplain, P jumps, R falls and K can fall with rising water surface (by
more than 20 % at 26 of the 279 irregular sections of one real reservoir). The energy
equation of the backwater solver then has several roots and its Newton iterations may not
converge.
``"section"`` is the usual formula for a single channel and includes the friction of
vertical walls; use it for sections without benches where that matters (a narrow
rectangular flume).

The two formulas differ for every section shape: ``"segments"`` gives the larger K, by
4 % for a trapezoid of 80 m bottom width with 2:1 banks at 5 m depth, and by 25 % for a
rectangle 10 m wide at 2 m depth.

Resistance law
--------------
Manning's formula is one of two resistance laws (``resistance``). Both give the mean
velocity as U = C sqrt(g R S_f) with a dimensionless coefficient C = U / u_*, and the
conveyance of a segment (or of the section) as K = A sqrt(g R) C:

    "manning" (default)    C = R^(1/6) / (n sqrt(g))
    "variable_power"       C = a1 a2 x / sqrt(a1^2 + a2^2 x^(5/3)),   x = R / k

The second is the variable-power equation of Ferguson (2007) with a1 = 6.5 and a2 = 2.5
and a roughness length k (m) in the place of n. For x >> 1 it tends to a1 x^(1/6), which
is Manning's formula with n = k^(1/6) / (a1 sqrt(g)) (Manning-Strickler); for x of order 1
it tends to a2 x: the resistance of a flow that is not much deeper than its roughness
rises faster than Manning's formula says (C is 13 % below the deep-flow limit at x = 6
and 37 % below it at x = 2.5). A segment has A_k = ds (hp1 + hp2)^2 / (2 (ha1 + ha2)) and
R_k = A_k / P_k = ds (hp1 + hp2) / (2 L).

A third law is the logarithmic one of Keulegan (1938), with a roughness height k_s:

    "keulegan"             C = 2.5 ln(11 x),   x = R / k_s

It turns negative below x = 1 / 11. Below x_0 = e / 11 = 0.247, where C = 2.5, it is
therefore continued by its tangent through the origin, C = (2.5 / x_0) x = 10.1 x, which
keeps C positive, increasing and smooth (value and slope) down to zero depth.

Reduced shear stress
--------------------
:func:`stress_reduction` gives the factor (C / C_deep)^e, e = 1.5, by which the bed shear
stress is reduced for the sediment in shallow flow (``transport_stress = "reduced"``).
C_deep is the law of a deep flow over the same roughness: a1 x^(1/6) for
``"variable_power"``, which gives (1 + (a1 / a2)^2 x^(-5/3))^(-3/4), and the power law
8.1 x^(1/6) for ``"keulegan"``. The logarithmic law lies within 2 % of that power law for
x of 9 to 200 and below it on either side; the factor is min(C / C_deep, 1)^e for x below
e^6 / 11 = 36.7, where the logarithmic law is highest relative to the power law, and 1
above (no reduction in deep flow). It is 1 for x above 14 and 0.92, 0.82 and 0.64 at
x = 5, 2.5 and 1.

First moment and inverse of the area
------------------------------------
The unsteady solver (:mod:`hydrograd.oned.unsteady`) needs the first moment of the flow
area about the water surface,

    I_1 = integral of (eta - z) b dz = sum over the segments of the integral of h^2 / 2,

which is the hydrostatic pressure force divided by rho g; dI_1 / d eta = A for a fixed
section. With the smooth depths above, a segment contributes

    I_1 = ds (hp1^2 + hp1 hp2 + hp2^2) (hp1 + hp2) / (6 (ha1 + ha2)),

which is ds (h1^2 + h1 h2 + h2^2) / 6 if both points are wet and the moment of the wet
triangle if one is. :func:`water_surface_from_area` inverts A(eta) by Newton iterations
from a given start and attaches the derivatives with respect to the area and the bed by
the implicit function theorem, as for the critical depth below. Its slope dA / d eta is
obtained by automatic differentiation of the smooth area: within a few widths of the
smooth waterline it differs from the top width B (by 11 % for a flat bottom under 6 cm of
water with the default smoothing), and the top width in its place would bias the
derivatives by that factor.

Critical depth
--------------
The critical depth y_c above the lowest point of the section solves Q^2 B / (g A^3) = 1.
It is found by Newton iterations on G(y) = A^3 / B - Q^2 / g with the approximate slope
dG/dy = 3 A^2 (dA/dy = B, dB/dy neglected), which is enough for convergence.
:func:`critical_depth_attached` adds the derivatives with respect to Q and z by the
implicit function theorem: y = y* - G(y*, Q, z) / G_y with y* and G_y held constant, which
has the value y* and the derivative -G_theta / G_y. Here G_y must be the exact slope,

    G_y = 3 A^2 - (A^3 / B^2) dB/dy,

or the derivative is biased by the factor 1 / (1 - A B' / (3 B^2)), where B' = dB/dy (zero
for a rectangle, 2 m for a trapezoid of side slope m). The exact slope is obtained by
automatic differentiation.

Units: SI.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.oned.config import (
    KEULEGAN_OFFSET,
    KEULEGAN_SLOPE,
    REDUCED_STRESS_EXPONENT,
    RESISTANCE_LAWS,
    STRICKLER_COEFFICIENT,
    VPE_A1,
    VPE_A2,
    GeometrySmoothing,
)
from hydrograd.smooth import sigmoid_eps, smooth_abs, softplus_beta

__all__ = [
    "SectionProperties",
    "area_width_moment",
    "critical_depth",
    "critical_depth_attached",
    "deposition_weights",
    "deposition_width",
    "hydraulic_properties",
    "thalweg",
    "tributary_widths",
    "water_surface_from_area",
    "wet_weights",
]

# Floor of the area (m^2), top width (m) and wetted perimeter (m) of a section.
PROPERTY_FLOOR = 1.0e-6

# Floor of n K (m^(8/3)) of a section with the conveyance summed over the segments.
CONVEYANCE_FLOOR = 1.0e-6

# Formulas of the conveyance (see the module docstring).
CONVEYANCE_OPTIONS = ("segments", "section")

# Guard of the segment denominator ha1 + ha2 (m).
SEGMENT_GUARD = 1.0e-12

# Vertical shift (m) of the finite difference that gives the deposition width.
SHIFT = 1.0e-3

# Floor of the largest depth of a section (m) and of the depth ratio (-) in the weights of
# a depth-weighted deposit.
DEPTH_RATIO_FLOOR = 1.0e-6
# Power of the relative depth in the reduction of the bed change of the deepest part of a
# section (``channel_deficit``): the reduction acts where the depth exceeds about 0.7 h_max.
CHANNEL_POWER = 4.0

# Smallest critical depth (m).
CRITICAL_DEPTH_FLOOR = 1.0e-3

# Smallest depth above the thalweg (m) and largest change of the water surface in one
# Newton iteration (m) of the inverse of the area.
SURFACE_DEPTH_FLOOR = 1.0e-6
SURFACE_STEP_LIMIT = 1.0


class SectionProperties(NamedTuple):
    """Hydraulic properties of N cross-sections at a given water surface.

    Attributes:
        area: flow area A (m^2), shape [N].
        top_width: width B of the water surface (m), shape [N].
        wetted_perimeter: P (m), shape [N].
        hydraulic_radius: R = A / P (m), shape [N].
        conveyance: K (m^3/s), shape [N].
    """

    area: Array
    top_width: Array
    wetted_perimeter: Array
    hydraulic_radius: Array
    conveyance: Array


def _segment_depths(
    eta: Array, z: Array, s: Array, smoothing: GeometrySmoothing,
) -> tuple[Array, Array, Array, Array]:
    """Smooth depths of the station points and their sums over each segment.

    Args:
        eta: water-surface elevation (m), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        smoothing: widths of the smooth waterline.

    Returns:
        Tuple ``(h_positive, ds, numerator, denominator)``: the smooth positive part of
        the depth at the points (m), shape [N, J]; the horizontal length of the segments
        (m); hp1 + hp2 and ha1 + ha2 (plus a guard) of each segment (m), shape [N, J - 1].
    """
    # Depth at each station point (negative above the water), its smooth positive part
    # and its smooth absolute value, shape [N, J].
    h = eta[:, None] - z
    h_positive = softplus_beta(h, smoothing.beta_softplus)
    h_absolute = smooth_abs(h, smoothing.eps_abs)

    ds = s[:, 1:] - s[:, :-1]
    numerator = h_positive[:, :-1] + h_positive[:, 1:]
    denominator = h_absolute[:, :-1] + h_absolute[:, 1:] + SEGMENT_GUARD

    return h_positive, ds, numerator, denominator


def variable_power_coefficient(x: Array) -> Array:
    """C = U / u_* of the variable-power resistance equation of Ferguson (2007).

    Args:
        x: hydraulic radius over the roughness length (-), not negative.

    Returns:
        a1 a2 x / sqrt(a1^2 + a2^2 x^(5/3)) (-), same shape as ``x``.
    """
    return VPE_A1 * VPE_A2 * x / jnp.sqrt(VPE_A1**2 + VPE_A2**2 * x ** (5.0 / 3.0))


def keulegan_coefficient(x: Array) -> Array:
    """C = U / u_* of the logarithmic resistance law of Keulegan.

    Args:
        x: hydraulic radius over the roughness height (-), not negative.

    Returns:
        2.5 ln(11 x) for x >= e / 11, and its tangent through the origin, (27.5 / e) x,
        below (-); same shape as ``x``.
    """
    x_tangent = math.e / KEULEGAN_OFFSET
    logarithmic = KEULEGAN_SLOPE * jnp.log(KEULEGAN_OFFSET * jnp.maximum(x, x_tangent))
    return jnp.where(x >= x_tangent, logarithmic, (KEULEGAN_SLOPE / x_tangent) * x)


def resistance_coefficient(x: Array, resistance: str) -> Array:
    """C = U / u_* of a resistance law with a roughness length.

    Args:
        x: hydraulic radius over the roughness length (-), not negative.
        resistance: ``"variable_power"`` or ``"keulegan"``.

    Returns:
        C (-), same shape as ``x``.
    """
    if resistance == "variable_power":
        return variable_power_coefficient(x)
    if resistance == "keulegan":
        return keulegan_coefficient(x)
    raise ValueError(f"no roughness length in the resistance law '{resistance}'")


def stress_reduction(x: Array, resistance: str) -> Array:
    """Factor on the bed shear stress for the sediment in shallow flow (see the module
    docstring).

    Args:
        x: hydraulic radius (or mean depth) over the roughness length (-), positive.
        resistance: ``"variable_power"`` or ``"keulegan"``.

    Returns:
        (C / C_deep)^1.5 (-), at most 1; same shape as ``x``.
    """
    if resistance == "variable_power":
        return (1.0 + (VPE_A1 / VPE_A2) ** 2 * x ** (-5.0 / 3.0)) ** (-0.5 * REDUCED_STRESS_EXPONENT)
    if resistance == "keulegan":
        x_closest = math.exp(6.0) / KEULEGAN_OFFSET
        ratio = keulegan_coefficient(x) / (STRICKLER_COEFFICIENT * x ** (1.0 / 6.0))
        ratio = jnp.where(x < x_closest, jnp.minimum(ratio, 1.0), 1.0)
        return ratio**REDUCED_STRESS_EXPONENT
    raise ValueError(f'transport_stress = "reduced" is not available with the resistance law \'{resistance}\'')


def hydraulic_properties(
    eta: Array,
    z: Array,
    s: Array,
    n_manning: Array,
    smoothing: GeometrySmoothing,
    conveyance: str = "segments",
    resistance: str = "manning",
    gravity: float = 9.81,
) -> SectionProperties:
    """Area, top width, wetted perimeter, hydraulic radius and conveyance of each section.

    Args:
        eta: water-surface elevation (m), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), non-decreasing along each section, shape [N, J].
        n_manning: parameter of the resistance law, shape [N] or scalar: Manning's n
            (s/m^(1/3)), or the roughness length k (m) for ``resistance =
            "variable_power"`` and ``"keulegan"``.
        smoothing: widths of the smooth waterline.
        conveyance: ``"segments"`` (default; sum over the segments) or ``"section"``
            (hydraulic radius of the whole section); see the module docstring.
        resistance: ``"manning"`` (default), ``"variable_power"`` or ``"keulegan"``; see
            the module docstring.
        gravity: acceleration of gravity (m/s^2); not used by ``"manning"``.

    Returns:
        The section properties (:class:`SectionProperties`).
    """
    if conveyance not in CONVEYANCE_OPTIONS:
        raise ValueError(
            f"conveyance must be one of {CONVEYANCE_OPTIONS}, not '{conveyance}'"
        )
    if resistance not in RESISTANCE_LAWS:
        raise ValueError(
            f"resistance must be one of {RESISTANCE_LAWS}, not '{resistance}'"
        )

    # Horizontal length of each segment and the sums of the smooth depths of its two
    # points, shape [N, J - 1]; bed rise of each segment.
    _, ds, numerator, denominator = _segment_depths(eta, z, s, smoothing)
    dz = z[:, 1:] - z[:, :-1]

    # Length L of each segment along the bed. A segment of zero length (repeated points
    # that pad a table) gets L = 0 with a zero derivative; the square root itself has no
    # finite derivative there.
    length_squared = ds * ds + dz * dz
    has_length = length_squared > 0.0
    length_safe = jnp.sqrt(jnp.where(has_length, length_squared, 1.0))
    length = jnp.where(has_length, length_safe, 0.0)

    # Wet fraction of each segment.
    wet_fraction = numerator / denominator

    # Section totals, floored so that a dry section does not divide by zero.
    area = jnp.sum(ds * numerator * numerator / (2.0 * denominator), axis=1)
    top_width = jnp.sum(ds * wet_fraction, axis=1)
    wetted_perimeter = jnp.sum(length * wet_fraction, axis=1)
    area = jnp.maximum(area, PROPERTY_FLOOR)
    top_width = jnp.maximum(top_width, PROPERTY_FLOOR)
    wetted_perimeter = jnp.maximum(wetted_perimeter, PROPERTY_FLOOR)

    # Hydraulic radius of the whole section.
    hydraulic_radius = area / wetted_perimeter

    if resistance != "manning":
        # K = A sqrt(g R) C(R / k) of the section, or the sum of it over the segments.
        if conveyance == "section":
            coefficient = resistance_coefficient(hydraulic_radius / n_manning, resistance)
            conveyance_value = area * jnp.sqrt(gravity * hydraulic_radius) * coefficient
        else:
            roughness = jnp.reshape(jnp.asarray(n_manning), (-1, 1))
            # R_k = ds (hp1 + hp2) / (2 L); A_k sqrt(R_k) written with one power of the
            # wet depth, as for Manning's formula below. Segments without width (vertical
            # walls, padding) are left out; they get a factor of 1 inside the square root,
            # whose derivative at zero is not finite.
            has_width = has_length & (ds > 0.0)
            radius_factor = jnp.where(has_width, ds / (2.0 * length_safe), 1.0)
            coefficient = resistance_coefficient(radius_factor * numerator / roughness, resistance)
            segment_conveyance = (
                ds * jnp.sqrt(gravity * radius_factor) * numerator**2.5 / (2.0 * denominator)
                * coefficient
            )
            segment_conveyance = jnp.where(has_width, segment_conveyance, 0.0)
            conveyance_value = jnp.maximum(jnp.sum(segment_conveyance, axis=1), CONVEYANCE_FLOOR)
    elif conveyance == "section":
        # Manning conveyance.
        conveyance_value = area * hydraulic_radius ** (2.0 / 3.0) / n_manning
    else:
        # Geometric factor ds^(5/3) / (2^(5/3) L^(2/3)) of each segment; segments of zero
        # length are left out.
        factor = ds ** (5.0 / 3.0) / (2.0 ** (5.0 / 3.0) * length_safe ** (2.0 / 3.0))
        factor = jnp.where(has_length, factor, 0.0)

        # A_k^(5/3) / P_k^(2/3) of each segment, written as one power of the wet depth so
        # that its derivative stays finite where the segment is dry.
        segment_conveyance = factor * numerator ** (8.0 / 3.0) / denominator
        conveyance_sum = jnp.maximum(jnp.sum(segment_conveyance, axis=1), CONVEYANCE_FLOOR)
        conveyance_value = conveyance_sum / n_manning

    return SectionProperties(
        area=area,
        top_width=top_width,
        wetted_perimeter=wetted_perimeter,
        hydraulic_radius=hydraulic_radius,
        conveyance=conveyance_value,
    )


def area_width_moment(
    eta: Array, z: Array, s: Array, smoothing: GeometrySmoothing,
) -> tuple[Array, Array, Array]:
    """Flow area, top width and first moment of the area of each section.

    Args:
        eta: water-surface elevation (m), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        smoothing: widths of the smooth waterline.

    Returns:
        Tuple ``(area, top_width, moment)``: A (m^2) and B (m), floored as in
        :func:`hydraulic_properties`, and the first moment I_1 of the area about the
        water surface (m^3), each of shape [N].
    """
    h_positive, ds, numerator, denominator = _segment_depths(eta, z, s, smoothing)

    area = jnp.sum(ds * numerator * numerator / (2.0 * denominator), axis=1)
    top_width = jnp.sum(ds * numerator / denominator, axis=1)

    # Integral of h^2 / 2 along each segment (see the module docstring).
    first = h_positive[:, :-1]
    second = h_positive[:, 1:]
    squares = first * first + first * second + second * second
    moment = jnp.sum(ds * squares * numerator / (6.0 * denominator), axis=1)

    area = jnp.maximum(area, PROPERTY_FLOOR)
    top_width = jnp.maximum(top_width, PROPERTY_FLOOR)

    return area, top_width, moment


def water_surface_from_area(
    area: Array,
    z: Array,
    s: Array,
    eta_start: Array,
    smoothing: GeometrySmoothing,
    n_iter: int = 3,
) -> Array:
    """Water surface of each section for a given flow area (inverse of A(eta)).

    Newton iterations eta <- eta - (A(eta) - A) / A_eta run on constants from
    ``eta_start``, with the exact slope A_eta = dA / d eta of the smooth area. One more
    step with the dependence on ``area`` and ``z`` kept, and the slope held constant,
    gives the derivatives d eta / dA = 1 / A_eta and d eta / dz = -A_z / A_eta (implicit
    function theorem).

    Args:
        area: flow area A (m^2), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        eta_start: start of the iterations (m), shape [N]; the water surface of the
            previous time step is close enough for a few iterations.
        smoothing: widths of the smooth waterline.
        n_iter: number of Newton iterations on constants.

    Returns:
        Water-surface elevation (m), shape [N].
    """
    target = jax.lax.stop_gradient(area)
    z_constant = jax.lax.stop_gradient(z)
    s_constant = jax.lax.stop_gradient(s)
    lowest = thalweg(z_constant) + SURFACE_DEPTH_FLOOR

    def area_of(eta: Array) -> Array:
        """Flow area as a function of the water surface, on the constant bed."""
        return area_width_moment(eta, z_constant, s_constant, smoothing)[0]

    def area_and_slope(eta: Array) -> tuple[Array, Array]:
        """A and dA / d eta. A_i depends on eta_i only, so one directional derivative
        along a vector of ones gives the slope of all sections."""
        area_now, slope = jax.jvp(area_of, (eta,), (jnp.ones_like(eta),))
        return area_now, jnp.maximum(slope, PROPERTY_FLOOR)

    def newton_step(_, eta: Array) -> Array:
        """One Newton update of the water surface, limited and kept above the bed."""
        area_now, slope = area_and_slope(eta)
        change = jnp.clip(
            (area_now - target) / slope, -SURFACE_STEP_LIMIT, SURFACE_STEP_LIMIT,
        )
        return jnp.maximum(eta - change, lowest)

    eta_star = jax.lax.fori_loop(
        0, n_iter, newton_step, jnp.maximum(jax.lax.stop_gradient(eta_start), lowest),
    )

    # Attached step: residual with its dependence on the area and the bed, constant slope.
    _, slope_star = area_and_slope(eta_star)
    area_star, _, _ = area_width_moment(eta_star, z, s, smoothing)

    return eta_star - (area_star - area) / slope_star


def tributary_widths(s: Array) -> Array:
    """Width attributed to each station point: half of each adjacent segment.

    Args:
        s: station of the points (m), shape [N, J].

    Returns:
        Widths (m), shape [N, J]; they sum to the total width of the table.
    """
    ds = s[:, 1:] - s[:, :-1]
    zero_column = jnp.zeros_like(ds[:, :1])

    # Segment to the left and to the right of each point (zero at the two ends).
    left = jnp.concatenate([zero_column, ds], axis=1)
    right = jnp.concatenate([ds, zero_column], axis=1)

    return 0.5 * (left + right)


def wet_weights(eta: Array, z: Array, smoothing: GeometrySmoothing) -> Array:
    """Smooth indicator that a station point is below the water surface.

    Args:
        eta: water-surface elevation (m), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        smoothing: widths of the smooth waterline (``eps_wet``).

    Returns:
        Weights in (0, 1) (-), shape [N, J].
    """
    return sigmoid_eps(eta[:, None] - z, smoothing.eps_wet)


def deposition_weights(
    eta: Array, z: Array, smoothing: GeometrySmoothing, depth_exponent: float = 0.0,
    channel_deficit: float = 0.0,
) -> Array:
    """Share of a bed change each station point receives, relative to the deepest point.

    w_j = wet_j (h_j / h_max)^m, with wet_j the wet weight, h_j the local depth, h_max the
    largest depth of the section and m the exponent. m = 0 raises all wetted points by the
    same amount (a veneer, as in SRH-1D). m = 1 places the deposit in proportion to the
    local depth, m = 5/3 in proportion to the local unit discharge of a Manning flow; both
    fill the deep parts of a compound section first and leave shallow flooded ground
    nearly unchanged.

    With ``channel_deficit`` a > 0 the weights are multiplied by
    1 - a (h_j / h_max)^CHANNEL_POWER: the deepest part of the section receives only the
    part 1 - a of the change of the ground beside it, so that a deposit builds up beside a
    channel that stays open (a = 1: the deepest point does not change).

    Args:
        eta: water-surface elevation (m), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        smoothing: widths of the smooth waterline (``eps_wet``).
        depth_exponent: exponent m (-), not negative.
        channel_deficit: part a (-) by which the change of the deepest point is reduced,
            between 0 and 1.

    Returns:
        Weights in (0, 1] (-), shape [N, J].
    """
    weights = wet_weights(eta, z, smoothing)
    if depth_exponent == 0.0 and channel_deficit == 0.0:
        return weights

    # Depth relative to the deepest point, floored so that the power stays differentiable.
    depth = jnp.maximum(eta[:, None] - z, 0.0)
    deepest = jnp.maximum(jnp.max(depth, axis=1, keepdims=True), DEPTH_RATIO_FLOOR)
    ratio = jnp.maximum(depth / deepest, DEPTH_RATIO_FLOOR)
    if depth_exponent != 0.0:
        weights = weights * ratio**depth_exponent
    if channel_deficit != 0.0:
        weights = weights * (1.0 - channel_deficit * ratio**CHANNEL_POWER)
    return weights


def deposition_width(
    eta: Array,
    z: Array,
    s: Array,
    smoothing: GeometrySmoothing,
    mode: str = "polygon",
    depth_exponent: float = 0.0,
    channel_deficit: float = 0.0,
) -> tuple[Array, Array]:
    """Point weights of a bed shift and the width over which a deposit is spread.

    A deposit of thickness dz raises the wetted points: z -> z + dz w, with w the weights
    of :func:`deposition_weights` (the wet weights by default; dz is then the bed change
    of every wetted point, otherwise that of the deepest point). The deposition width B_d is the section area lost per unit dz, so the volume
    placed in the section per unit length is exactly B_d dz.

    With ``mode="polygon"`` the area is that of the whole section polygon below a datum
    above every point. Raising the wetted points also tilts the bank segments up to the
    fixed dry points, and that material above the water surface is part of the deposit
    (SRH-1D books deposits in the same way: for a trapezoid with dry bank tops, B_d is the
    bank-to-bank mean width, not the wetted top width). ``mode="wetted"`` uses the change
    of the wetted area only (kept for ablation; it over-thickens the deposit).

    Args:
        eta: water-surface elevation (m), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        smoothing: widths of the smooth waterline.
        mode: ``"polygon"`` or ``"wetted"``.
        depth_exponent: exponent of the local depth in the weights
            (:func:`deposition_weights`); 0 for a uniform shift of the wetted points.
        channel_deficit: reduction of the change of the deepest part
            (:func:`deposition_weights`).

    Returns:
        Tuple ``(weights, width)``: w (-), shape [N, J], and B_d (m), shape [N].
    """
    weights = deposition_weights(eta, z, smoothing, depth_exponent, channel_deficit)
    ones = jnp.ones(z.shape[0], dtype=z.dtype)

    # Level below which the area is measured: the water surface, or 1 m above the highest
    # point (held constant in the derivative).
    if mode == "wetted":
        datum = eta
    else:
        datum = jax.lax.stop_gradient(jnp.max(z, axis=1) + 1.0)

    # Area below the datum before and after raising the weighted points by SHIFT.
    area_before = hydraulic_properties(datum, z, s, ones, smoothing).area
    area_after = hydraulic_properties(datum, z + SHIFT * weights, s, ones, smoothing).area
    width = (area_before - area_after) / SHIFT

    return weights, jnp.maximum(width, PROPERTY_FLOOR)


def thalweg(z: Array) -> Array:
    """Lowest bed elevation of each section.

    Where several points share the lowest elevation (a flat bottom), the derivative goes
    to the first of them.

    Args:
        z: bed elevation of the station points (m), shape [N, J].

    Returns:
        Thalweg elevation (m), shape [N].
    """
    lowest = jnp.argmin(z, axis=1)
    return jnp.take_along_axis(z, lowest[:, None], axis=1)[:, 0]


def _critical_residual(
    y: Array, Q: Array, z: Array, s: Array, smoothing: GeometrySmoothing, g: float,
) -> tuple[Array, Array]:
    """Residual G = A^3 / B - Q^2 / g of the critical-flow condition and its slope 3 A^2.

    Args:
        y: depth above the thalweg (m), shape [N].
        Q: discharge (m^3/s), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        smoothing: widths of the smooth waterline.
        g: gravitational acceleration (m/s^2).

    Returns:
        Tuple ``(G, dG_dy)`` (m^5 and m^4), each of shape [N].
    """
    ones = jnp.ones(z.shape[0], dtype=z.dtype)
    properties = hydraulic_properties(thalweg(z) + y, z, s, ones, smoothing)
    area = properties.area

    residual = area**3 / properties.top_width - Q * Q / g
    slope = 3.0 * area * area

    return residual, slope


def critical_depth(
    Q: Array,
    z: Array,
    s: Array,
    smoothing: GeometrySmoothing,
    g: float = 9.81,
    n_iter: int = 40,
) -> Array:
    """Critical depth of each section, without derivatives.

    Args:
        Q: discharge (m^3/s), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        smoothing: widths of the smooth waterline.
        g: gravitational acceleration (m/s^2).
        n_iter: number of Newton iterations.

    Returns:
        Critical depth y_c above the thalweg (m), shape [N]; a constant in derivatives.
    """
    Q = jax.lax.stop_gradient(Q)
    z = jax.lax.stop_gradient(z)
    s = jax.lax.stop_gradient(s)

    def newton_step(_, y: Array) -> Array:
        """One Newton update of the depth, floored at CRITICAL_DEPTH_FLOOR."""
        residual, slope = _critical_residual(y, Q, z, s, smoothing, g)
        return jnp.maximum(y - residual / slope, CRITICAL_DEPTH_FLOOR)

    # Start from a depth of 1 m everywhere.
    y_start = jnp.ones(z.shape[0], dtype=z.dtype)

    return jax.lax.fori_loop(0, n_iter, newton_step, y_start)


def critical_depth_attached(
    Q: Array,
    z: Array,
    s: Array,
    smoothing: GeometrySmoothing,
    g: float = 9.81,
    n_iter: int = 40,
) -> Array:
    """Critical depth with derivatives with respect to Q and z.

    One step y = y* - G(y*, Q, z) / G_y from the converged y*, with y* and G_y held
    constant: the value is y* and the derivative is -G_theta / G_y (implicit function
    theorem). G_y is the exact slope dG/dy (see the module docstring).

    Args:
        Q: discharge (m^3/s), shape [N].
        z: bed elevation of the station points (m), shape [N, J].
        s: station of the points (m), shape [N, J].
        smoothing: widths of the smooth waterline.
        g: gravitational acceleration (m/s^2).
        n_iter: number of Newton iterations.

    Returns:
        Critical depth y_c above the thalweg (m), shape [N].
    """
    y_star = critical_depth(Q, z, s, smoothing, g, n_iter)

    # Residual at y* with the dependence on Q and z kept.
    residual, _ = _critical_residual(y_star, Q, z, s, smoothing, g)

    # Exact slope dG/dy at y*. G_i depends on y_i only, so the directional derivative
    # along a vector of ones gives dG_i/dy_i for all sections at once.
    def residual_of_depth(y: Array) -> Array:
        """G as a function of the depth, at constant Q and z."""
        return _critical_residual(y, Q, z, s, smoothing, g)[0]

    _, slope = jax.jvp(residual_of_depth, (y_star,), (jnp.ones_like(y_star),))

    # The slope is a constant of the step.
    slope = jax.lax.stop_gradient(slope)

    return y_star - residual / slope
