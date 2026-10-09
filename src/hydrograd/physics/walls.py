"""Side-wall shear stress from the log law of the wall (no-slip walls in the 2D SWEs).

Physics
-------
A WALL boundary face without this term is free-slip: the HLLC flux through it carries
no mass and no tangential momentum, so the wall exerts only the hydrostatic pressure.
Real side walls are no-slip. On a depth-averaged mesh the wall boundary layer is not
resolved (the first cell centre lies centimetres to metres from the wall), so the no-slip
condition enters as a wall function: the shear stress tau_w that a log-law boundary layer
exerts between the wall and the tangential velocity u_t of the wall cell.

Friction velocity
-----------------
With y the distance from the cell centroid to the wall (m), nu the kinematic viscosity
(m^2/s), k_s the equivalent sand roughness of the wall (m), kappa = 0.41 and E = 9.8:

    log law:              u_t / u_* = (1 / kappa) ln(y / y_0),
                          y_0 = nu / (E u_*) + k_s / 30                      (m)
    viscous sublayer:     u_t / u_* = y u_* / nu,   i.e.  u_* = sqrt(nu |u_t| / y)

The roughness length y_0 blends the smooth law (k_s = 0: u_t / u_* = ln(E y+) / kappa)
and the fully rough law (k_s >> nu / u_*: u_t / u_* = ln(30 y / k_s) / kappa). The log
law is solved for u_* by the fixed-point iteration u_* <- kappa |u_t| / ln(y / y_0(u_*)),
started from u_* = |u_t| / 20. Its contraction factor is about 1 / ln(y / y_0) (0.15 at
y+ = 100), so a few iterations reach round-off. The iterations are unrolled, so the
derivatives follow by automatic differentiation.

As in OpenFOAM's wall functions, the viscous law is used where its y+ = y u_* / nu is
below y+_lam = 11.53 and the log law elsewhere; for a smooth wall the two laws meet at
y+_lam (ln(9.8 x 11.53) / 0.41 = 11.53).

Momentum source
---------------
The stress acts against u_t over the wetted wall area h L_f of the face. Per unit plan
area of the wall cell (divided by the density):

    S = -(u_*^2 h L_f / A) t_hat sign(u_t)                              (m^2/s^2)

with t_hat the unit tangent of the face, L_f its length (m) and A the cell area (m^2).
It is a local source (one cell per face), so the Jacobian sparsity pattern of the scheme
is unchanged and the steady, implicit and adjoint solvers include it with no change.

Equivalence with Manning's n: for fully rough walls the integrated rough log law is
close to the Manning-Strickler power law with n ~ k_s^(1/6) / 26 (SI) over
y / k_s ~ 10-1000, but for smooth walls (glass, acrylic) the friction depends on the
Reynolds number through the viscous sublayer, which a constant n cannot represent.

Units: SI by default; the formulas hold in any consistent unit system (set ``nu``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hydrograd.physics.model import KIND_WALL, SWEModel

__all__ = ["wall_face_state", "wall_friction_velocity", "wall_shear_source"]

# Differentiable parameters; ``wall_ks`` (m) overrides the wall roughness.
Theta = dict[str, Any]

# Smallest tangential speed used in the formulas (m/s); keeps u_t / |u_t| finite.
SPEED_FLOOR = 1.0e-8


def wall_friction_velocity(
    speed: Array, distance: Array, ks: Array | float, nu: float, kappa: float, E: float,
    yplus_lam: float, iterations: int,
) -> Array:
    """Friction velocity u_* of a log-law wall boundary layer.

    Args:
        speed: tangential speed |u_t| at the wall-cell centroid (m/s), shape [n_faces].
        distance: distance y from the centroid to the wall (m), shape [n_faces].
        ks: equivalent sand roughness k_s of the wall (m), scalar or [n_faces].
        nu: kinematic viscosity (m^2/s).
        kappa: von Karman constant (-).
        E: smooth-wall log-law constant (-).
        yplus_lam: y+ below which the viscous-sublayer law applies (-).
        iterations: number of fixed-point iterations of the log law.

    Returns:
        u_* (m/s), shape [n_faces].
    """
    # Log law u_t / u_* = ln(y / y_0) / kappa with y_0 = nu / (E u_*) + k_s / 30, solved
    # by fixed-point iteration from u_* = |u_t| / 20.
    friction_velocity = speed / 20.0
    for _ in range(iterations):
        roughness_length = nu / (E * friction_velocity) + ks / 30.0

        # The logarithm is floored at 1: below that the log law is invalid (the viscous
        # law is used there) and the floor keeps u_* bounded and the iteration contracting.
        log_ratio = jnp.maximum(jnp.log(distance / roughness_length), 1.0)
        friction_velocity = kappa * speed / log_ratio

    # Viscous sublayer u_t = u_*^2 y / nu, and its y+.
    friction_velocity_viscous = jnp.sqrt(nu * speed / distance)
    yplus_viscous = distance * friction_velocity_viscous / nu

    # Viscous law inside the sublayer, log law outside.
    in_sublayer = yplus_viscous < yplus_lam
    return jnp.where(in_sublayer, friction_velocity_viscous, friction_velocity)


def wall_face_state(
    m: SWEModel, theta: Theta, u: Array, v: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Log-law state of every boundary face (meaningful on WALL faces only).

    Args:
        m: the model (``wall_friction`` settings, boundary faces).
        theta: differentiable parameters; ``wall_ks`` (m) overrides the wall roughness.
        u, v: depth-averaged velocity components (m/s), shape [n_cells].

    Returns:
        (is_wall, tangent_x, tangent_y, distance, tangential_velocity, speed,
        friction_velocity), each of shape [n_fb]: 1.0 on WALL faces (else 0.0); the unit
        tangent t_hat = (-n_y, n_x); the distance y from the cell centroid to the face (m);
        the tangential velocity u_t (m/s) and its smoothed magnitude (m/s); and the
        log-law friction velocity u_* (m/s).
    """
    settings = m.wall_friction

    # Wall faces: their cells, unit normals and tangents t_hat = (-n_y, n_x).
    is_wall = (m.fb_kind == KIND_WALL).astype(u.dtype)
    cell = m.fb_c
    normal_x = m.fb_nx
    normal_y = m.fb_ny
    tangent_x = -normal_y
    tangent_y = normal_x

    # Distance from the cell centroid to the wall face (m).
    distance = jnp.abs(m.fb_ro[:, 0] * normal_x + m.fb_ro[:, 1] * normal_y)

    # Tangential velocity of the wall cell (m/s), smoothed |u_t| for the formulas.
    tangential_velocity = u[cell] * tangent_x + v[cell] * tangent_y
    speed = jnp.sqrt(tangential_velocity * tangential_velocity + SPEED_FLOOR * SPEED_FLOOR)

    # Friction velocity from the log law (m/s).
    ks = theta.get("wall_ks", settings.ks)
    friction_velocity = wall_friction_velocity(
        speed,
        distance,
        ks,
        settings.nu,
        settings.kappa,
        settings.E,
        settings.yplus_lam,
        settings.iterations,
    )
    return (
        is_wall, tangent_x, tangent_y, distance, tangential_velocity, speed, friction_velocity
    )


def wall_shear_source(m: SWEModel, theta: Theta, h: Array, u: Array, v: Array) -> Array:
    """Momentum source of the wall shear stress, as a contribution to dU/dt.

    Args:
        m: the model (``wall_friction`` settings, boundary faces, cell areas).
        theta: differentiable parameters; ``wall_ks`` (m) overrides the wall roughness.
        h: water depth (m), shape [n_cells].
        u, v: depth-averaged velocity components (m/s), shape [n_cells].

    Returns:
        dU/dt contribution [0, S_x, S_y] (m^2/s^2 in the momentum rows), shape
        [3, n_cells]; zero except in cells with WALL faces.
    """
    cell = m.fb_c
    (is_wall, tangent_x, tangent_y, _, tangential_velocity, speed,
     friction_velocity) = wall_face_state(m, theta, u, v)

    # The stress tau_w / rho = u_*^2 (m^2/s^2).
    stress = friction_velocity * friction_velocity

    # Force per unit density on the wetted wall area h L_f, against u_t (m^4/s^2).
    wetted_area = h[cell] * m.fb_L
    force = -is_wall * stress * wetted_area * tangential_velocity / speed
    force_x = force * tangent_x
    force_y = force * tangent_y

    # Sum per cell and divide by the cell area: source per unit plan area (m^2/s^2).
    source_x = jax.ops.segment_sum(force_x, cell, m.n_cells) / m.area
    source_y = jax.ops.segment_sum(force_y, cell, m.n_cells) / m.area

    return jnp.stack([jnp.zeros_like(source_x), source_x, source_y])
