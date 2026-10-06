"""Backend-agnostic 2D shallow water physics (first-order Godunov FV).

The residual is written once against a tiny ``ops`` shim (see backends.py),
so the PyTorch and JAX backends evaluate exactly the same discretization:

    dU/dt = R(U; theta),  U = [h, hu, hv] per cell

* Roe approximate Riemann solver with Harten entropy fix.
* Audusse et al. (2004) hydrostatic reconstruction -> well-balanced,
  depth-positive first-order scheme on unstructured polygons.
* Manning friction, either a per-cell n field or an MLP closure
  n = NN(h, |u|) (the USWE setting of Liu & Song 2025).
* Boundary types: wall (reflective), inlet-Q (specified unit discharge),
  exit-H (specified water-surface elevation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

G = 9.81  # gravity (m/s^2)
H_EPS = 1.0e-6  # desingularization depth for velocities (m)


@dataclass(frozen=True)
class FlowConfig:
    """Physical/boundary parameters of the benchmark case (SI units)."""

    q_in: float = 1.0  # inlet unit discharge (m^2/s)
    wse_out: float = 1.5  # outlet water-surface elevation (m)
    n_lo: float = 0.01  # NN closure lower bound on Manning's n
    n_hi: float = 0.08  # NN closure upper bound on Manning's n
    h_ref: float = 1.0  # depth scale for NN input normalization (m)
    u_ref: float = 1.0  # velocity scale for NN input normalization (m/s)


def _velocity(ops, h, hu):
    """Desingularized velocity u = hu/h (Kurganov & Petrova 2007).

    u = sqrt(2) h (hu) / sqrt(h^4 + max(h^4, eps^4)), with eps = H_EPS, which
    equals hu/h for h >> eps and tends to zero as h -> 0.

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        h: water depth (m), shape [n_cells].
        hu: unit discharge component (m^2/s), shape [n_cells].

    Returns:
        Velocity component (m/s), shape [n_cells].
    """
    h4 = h ** 4
    return ops.sqrt2 * h * hu / ops.sqrt(h4 + ops.maximum(h4, H_EPS ** 4))


def roe_flux(ops, hL, uL, vL, hR, uR, vR, nx, ny):
    """Roe flux F(U_L, U_R) . n per face (Harten entropy fix).

    F = 0.5 (F(U_L) + F(U_R)) . n - 0.5 sum_k |lambda_k| alpha_k r_k, with
    Roe-averaged velocities, the arithmetic-mean depth for the celerity, and
    Harten's fix |lambda| -> (lambda^2 + delta^2)/(2 delta) for |lambda| < delta.

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        hL: left depth (m), shape [n_faces].
        uL: left x-velocity (m/s), shape [n_faces].
        vL: left y-velocity (m/s), shape [n_faces].
        hR: right depth (m), shape [n_faces].
        uR: right x-velocity (m/s), shape [n_faces].
        vR: right y-velocity (m/s), shape [n_faces].
        nx: x-component of the unit normal (L -> R), shape [n_faces].
        ny: y-component of the unit normal (L -> R), shape [n_faces].

    Returns:
        Tuple (f0, f1, f2) of normal fluxes per unit face length: mass
        (m^2/s) and x-, y-momentum (m^3/s^2), each of shape [n_faces].
    """
    # Normal velocities on each side (m/s)
    unL = uL * nx + vL * ny
    unR = uR * nx + vR * ny

    # Roe-averaged velocities, weighted by sqrt(h) (m/s)
    sL = ops.sqrt(hL + H_EPS)
    sR = ops.sqrt(hR + H_EPS)
    uh = (sL * uL + sR * uR) / (sL + sR)
    vh = (sL * vL + sR * vR) / (sL + sR)
    unh = uh * nx + vh * ny

    # Mean depth (m) and gravity-wave celerity (m/s)
    hbar = 0.5 * (hL + hR)
    c = ops.sqrt(G * hbar + H_EPS)

    # Jumps in depth, normal and tangential velocity across the face
    dh = hR - hL
    dun = unR - unL
    dut = (-uR * ny + vR * nx) - (-uL * ny + vL * nx)

    # Wave strengths of the three characteristic fields
    a1 = 0.5 * (dh - hbar * dun / c)
    a2 = hbar * dut
    a3 = 0.5 * (dh + hbar * dun / c)

    # Harten entropy-fix threshold (m/s)
    delta = 0.1 * c

    def fix(lam):
        """Harten-smoothed |lambda| (m/s); lam has shape [n_faces]."""
        al = ops.abs(lam)
        return ops.where(al < delta, (lam * lam + delta * delta) / (2.0 * delta), al)

    # Entropy-fixed wave speeds |un - c|, |un|, |un + c|
    l1 = fix(unh - c)
    l2 = fix(unh)
    l3 = fix(unh + c)

    # Hydrostatic pressure terms g h^2 / 2 on each side (m^3/s^2)
    pL = 0.5 * G * hL * hL
    pR = 0.5 * G * hR * hR

    # Mass flux: central average minus upwind dissipation
    f0 = 0.5 * (hL * unL + hR * unR) - 0.5 * (l1 * a1 + l3 * a3)

    # x-momentum flux
    central_x = hL * uL * unL + pL * nx + hR * uR * unR + pR * nx
    upwind_x = l1 * a1 * (uh - c * nx) + l2 * a2 * (-ny) + l3 * a3 * (uh + c * nx)
    f1 = 0.5 * central_x - 0.5 * upwind_x

    # y-momentum flux
    central_y = hL * vL * unL + pL * ny + hR * vR * unR + pR * ny
    upwind_y = l1 * a1 * (vh - c * ny) + l2 * a2 * nx + l3 * a3 * (vh + c * ny)
    f2 = 0.5 * central_y - 0.5 * upwind_y
    return f0, f1, f2


def mlp(ops, params: List[Tuple[Any, Any]], x):
    """Plain tanh MLP; params = [(W, b), ...], x: [N, d_in].

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        params: list of (W [d_a, d_b], b [d_b]) per layer.
        x: input features (dimensionless), shape [N, d_in].

    Returns:
        Linear output of the last layer, shape [N, d_out].
    """
    # Hidden layers with tanh activation
    for W, b in params[:-1]:
        x = ops.tanh(x @ W + b)

    # Linear output layer
    W, b = params[-1]
    return x @ W + b


def manning_n(ops, cfg: FlowConfig, theta: Dict[str, Any], h, u, v):
    """Cell Manning's n from a field (theta['n']) or an MLP closure.

    The closure is n = n_lo + (n_hi - n_lo) sigmoid(NN(h/h_ref, |u|/u_ref)).

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        cfg: flow configuration (closure bounds and scales).
        theta: parameters, either ``{"n": [n_cells]}`` or ``{"mlp": [(W, b), ...]}``.
        h: depth (m), shape [n_cells].
        u: x-velocity (m/s), shape [n_cells].
        v: y-velocity (m/s), shape [n_cells].

    Returns:
        Manning's n per cell (s/m^(1/3)), shape [n_cells].
    """
    # Direct per-cell field
    if "n" in theta:
        return theta["n"]

    # Normalized NN inputs: depth and speed
    speed = ops.sqrt(u * u + v * v + 1.0e-12)
    x = ops.stack([h / cfg.h_ref, speed / cfg.u_ref], axis=1)

    # Bounded output via a sigmoid
    z = mlp(ops, theta["mlp"], x)[:, 0]
    return cfg.n_lo + (cfg.n_hi - cfg.n_lo) * ops.sigmoid(z)


def residual(ops, m: Dict[str, Any], cfg: FlowConfig, theta: Dict[str, Any], U):
    """Semi-discrete residual R(U; theta) = dU/dt, shape [n_cells, 3].

    ``m`` holds backend arrays produced by backends.mesh_arrays():
    interior faces (fi_*), boundary faces (fb_*), cell area and bed zb.

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        m: mesh arrays from ``backends.mesh_arrays``.
        cfg: flow configuration (boundary values, closure settings).
        theta: friction parameters (``{"n": ...}`` or ``{"mlp": ...}``).
        U: state [h (m), hu (m^2/s), hv (m^2/s)], shape [n_cells, 3].

    Returns:
        dU/dt [m/s, m^2/s^2, m^2/s^2], shape [n_cells, 3].
    """
    # ---- 1. cell primitives -------------------------------------------------
    h = U[:, 0]
    hu = U[:, 1]
    hv = U[:, 2]
    u = _velocity(ops, h, hu)
    v = _velocity(ops, h, hv)
    zb = m["zb"]
    n_cells = m["n_cells"]

    # ---- 2. interior faces --------------------------------------------------
    o = m["fi_o"]
    r = m["fi_r"]
    nx = m["fi_nx"]
    ny = m["fi_ny"]
    L = m["fi_L"]
    hL = h[o]
    hR = h[r]

    # Hydrostatic reconstruction: depths relative to the face bed max(zL, zR)
    zf = ops.maximum2(zb[o], zb[r])
    hLs = ops.maximum(hL + zb[o] - zf, 0.0)
    hRs = ops.maximum(hR + zb[r] - zf, 0.0)
    f0, f1, f2 = roe_flux(ops, hLs, u[o], v[o], hRs, u[r], v[r], nx, ny)

    # Hydrostatic-reconstruction pressure corrections (one per side)
    cL = 0.5 * G * (hL * hL - hLs * hLs)
    cR = 0.5 * G * (hR * hR - hRs * hRs)

    # Face contributions to the owner (+) and neighbour (-) cells
    own = ops.stack([f0 * L, (f1 + cL * nx) * L, (f2 + cL * ny) * L], axis=1)
    nbr = ops.stack([-f0 * L, (-f1 - cR * nx) * L, (-f2 - cR * ny) * L], axis=1)

    # ---- 3. boundary faces (ghost states) -----------------------------------
    b = m["fb_o"]
    bnx = m["fb_nx"]
    bny = m["fb_ny"]
    bL = m["fb_L"]
    is_wall = m["fb_wall"]
    is_in = m["fb_in"]
    is_out = m["fb_out"]
    hb = h[b]
    ub = u[b]
    vb = v[b]

    # Owner velocity in face-normal / tangential coordinates
    unb = ub * bnx + vb * bny
    utb = -ub * bny + vb * bnx

    # Ghost depth: prescribed WSE at the outlet, otherwise the owner depth
    hg = ops.where(is_out, ops.maximum(cfg.wse_out - zb[b], 0.0), hb)

    # Ghost normal velocity: reflected at walls, q_in/h inflow at the inlet
    un_in = -cfg.q_in / ops.maximum(hb, 1.0e-3)  # inflow: normal points outward
    ung = ops.where(is_wall, -unb, ops.where(is_in, un_in, unb))

    # Ghost tangential velocity: zero at the inlet
    utg = ops.where(is_in, 0.0 * utb, utb)

    # Ghost velocity back in x, y and the boundary Roe flux
    ug = ung * bnx - utg * bny
    vg = ung * bny + utg * bnx
    g0, g1, g2 = roe_flux(ops, hb, ub, vb, hg, ug, vg, bnx, bny)
    bnd = ops.stack([g0 * bL, g1 * bL, g2 * bL], axis=1)

    # ---- 4. assemble: dU/dt = -(1/A) sum_faces F.n L ------------------------
    acc = ops.scatter_add(own, o, n_cells)
    acc = acc + ops.scatter_add(nbr, r, n_cells)
    acc = acc + ops.scatter_add(bnd, b, n_cells)
    dU = -acc / m["area"][:, None]

    # ---- 5. Manning friction: -g n^2 |u| (hu, hv) / h^(4/3) -----------------
    n = manning_n(ops, cfg, theta, h, u, v)
    hf = ops.maximum(h, 1.0e-3)
    speed = ops.sqrt(u * u + v * v + 1.0e-12)
    cf = G * n * n * speed / hf ** (4.0 / 3.0)
    fric = ops.stack([0.0 * h, -cf * hu, -cf * hv], axis=1)
    return dU + fric


def ssprk2_step(ops, m, cfg, theta, U, dt: float):
    """One SSP-RK2 (Heun) step with depth clipping for positivity.

    U1 = clip(U + dt R(U)); U2 = clip(U/2 + (U1 + dt R(U1))/2).

    Args:
        ops: backend shim (``TorchOps`` or ``JaxOps``).
        m: mesh arrays from ``backends.mesh_arrays``.
        cfg: flow configuration.
        theta: friction parameters.
        U: state [h, hu, hv], shape [n_cells, 3].
        dt: time step (s).

    Returns:
        State after one step, shape [n_cells, 3].
    """
    # Stage 1: forward Euler predictor
    U1 = U + dt * residual(ops, m, cfg, theta, U)
    U1 = ops.clip_depth(U1)

    # Stage 2: average of U and a second Euler step from U1
    U2 = 0.5 * U + 0.5 * (U1 + dt * residual(ops, m, cfg, theta, U1))
    return ops.clip_depth(U2)
