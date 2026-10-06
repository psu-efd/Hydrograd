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
    """Desingularized velocity u = hu/h (Kurganov & Petrova 2007)."""
    h4 = h ** 4
    return ops.sqrt2 * h * hu / ops.sqrt(h4 + ops.maximum(h4, H_EPS ** 4))


def roe_flux(ops, hL, uL, vL, hR, uR, vR, nx, ny):
    """Roe flux F(U_L, U_R) . n per face (Harten entropy fix)."""
    unL, unR = uL * nx + vL * ny, uR * nx + vR * ny
    sL, sR = ops.sqrt(hL + H_EPS), ops.sqrt(hR + H_EPS)
    uh = (sL * uL + sR * uR) / (sL + sR)
    vh = (sL * vL + sR * vR) / (sL + sR)
    unh = uh * nx + vh * ny
    hbar = 0.5 * (hL + hR)
    c = ops.sqrt(G * hbar + H_EPS)
    dh, dun = hR - hL, unR - unL
    dut = (-uR * ny + vR * nx) - (-uL * ny + vL * nx)
    a1 = 0.5 * (dh - hbar * dun / c)
    a2 = hbar * dut
    a3 = 0.5 * (dh + hbar * dun / c)
    delta = 0.1 * c

    def fix(lam):
        al = ops.abs(lam)
        return ops.where(al < delta, (lam * lam + delta * delta) / (2.0 * delta), al)

    l1, l2, l3 = fix(unh - c), fix(unh), fix(unh + c)
    pL, pR = 0.5 * G * hL * hL, 0.5 * G * hR * hR
    f0 = 0.5 * (hL * unL + hR * unR) - 0.5 * (l1 * a1 + l3 * a3)
    f1 = 0.5 * (hL * uL * unL + pL * nx + hR * uR * unR + pR * nx) - 0.5 * (
        l1 * a1 * (uh - c * nx) + l2 * a2 * (-ny) + l3 * a3 * (uh + c * nx))
    f2 = 0.5 * (hL * vL * unL + pL * ny + hR * vR * unR + pR * ny) - 0.5 * (
        l1 * a1 * (vh - c * ny) + l2 * a2 * nx + l3 * a3 * (vh + c * ny))
    return f0, f1, f2


def mlp(ops, params: List[Tuple[Any, Any]], x):
    """Plain tanh MLP; params = [(W, b), ...], x: [N, d_in]."""
    for W, b in params[:-1]:
        x = ops.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def manning_n(ops, cfg: FlowConfig, theta: Dict[str, Any], h, u, v):
    """Cell Manning's n from a field (theta['n']) or an MLP closure."""
    if "n" in theta:
        return theta["n"]
    speed = ops.sqrt(u * u + v * v + 1.0e-12)
    x = ops.stack([h / cfg.h_ref, speed / cfg.u_ref], axis=1)
    z = mlp(ops, theta["mlp"], x)[:, 0]
    return cfg.n_lo + (cfg.n_hi - cfg.n_lo) * ops.sigmoid(z)


def residual(ops, m: Dict[str, Any], cfg: FlowConfig, theta: Dict[str, Any], U):
    """Semi-discrete residual R(U; theta) = dU/dt, shape [n_cells, 3].

    ``m`` holds backend arrays produced by backends.mesh_arrays():
    interior faces (fi_*), boundary faces (fb_*), cell area and bed zb.
    """
    h, hu, hv = U[:, 0], U[:, 1], U[:, 2]
    u, v = _velocity(ops, h, hu), _velocity(ops, h, hv)
    zb = m["zb"]
    n_cells = m["n_cells"]

    # ---- interior faces -------------------------------------------------
    o, r = m["fi_o"], m["fi_r"]
    nx, ny, L = m["fi_nx"], m["fi_ny"], m["fi_L"]
    hL, hR = h[o], h[r]
    zf = ops.maximum2(zb[o], zb[r])
    hLs = ops.maximum(hL + zb[o] - zf, 0.0)
    hRs = ops.maximum(hR + zb[r] - zf, 0.0)
    f0, f1, f2 = roe_flux(ops, hLs, u[o], v[o], hRs, u[r], v[r], nx, ny)
    cL = 0.5 * G * (hL * hL - hLs * hLs)  # hydrostatic-reconstruction correction
    cR = 0.5 * G * (hR * hR - hRs * hRs)
    own = ops.stack([f0 * L, (f1 + cL * nx) * L, (f2 + cL * ny) * L], axis=1)
    nbr = ops.stack([-f0 * L, (-f1 - cR * nx) * L, (-f2 - cR * ny) * L], axis=1)

    # ---- boundary faces (ghost states) ---------------------------------
    b = m["fb_o"]
    bnx, bny, bL = m["fb_nx"], m["fb_ny"], m["fb_L"]
    is_wall, is_in, is_out = m["fb_wall"], m["fb_in"], m["fb_out"]
    hb, ub, vb = h[b], u[b], v[b]
    unb, utb = ub * bnx + vb * bny, -ub * bny + vb * bnx
    hg = ops.where(is_out, ops.maximum(cfg.wse_out - zb[b], 0.0), hb)
    un_in = -cfg.q_in / ops.maximum(hb, 1.0e-3)  # inflow: normal points outward
    ung = ops.where(is_wall, -unb, ops.where(is_in, un_in, unb))
    utg = ops.where(is_in, 0.0 * utb, utb)
    ug, vg = ung * bnx - utg * bny, ung * bny + utg * bnx
    g0, g1, g2 = roe_flux(ops, hb, ub, vb, hg, ug, vg, bnx, bny)
    bnd = ops.stack([g0 * bL, g1 * bL, g2 * bL], axis=1)

    # ---- assemble: dU/dt = -(1/A) sum_faces F.n L + friction ------------
    acc = ops.scatter_add(own, o, n_cells)
    acc = acc + ops.scatter_add(nbr, r, n_cells)
    acc = acc + ops.scatter_add(bnd, b, n_cells)
    dU = -acc / m["area"][:, None]

    n = manning_n(ops, cfg, theta, h, u, v)
    hf = ops.maximum(h, 1.0e-3)
    speed = ops.sqrt(u * u + v * v + 1.0e-12)
    cf = G * n * n * speed / hf ** (4.0 / 3.0)
    fric = ops.stack([0.0 * h, -cf * hu, -cf * hv], axis=1)
    return dU + fric


def ssprk2_step(ops, m, cfg, theta, U, dt: float):
    """One SSP-RK2 (Heun) step with depth clipping for positivity."""
    U1 = U + dt * residual(ops, m, cfg, theta, U)
    U1 = ops.clip_depth(U1)
    U2 = 0.5 * U + 0.5 * (U1 + dt * residual(ops, m, cfg, theta, U1))
    return ops.clip_depth(U2)
