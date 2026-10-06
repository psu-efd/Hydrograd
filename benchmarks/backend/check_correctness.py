"""Correctness checks: well-balancing, backend agreement, gradient vs FD."""

from __future__ import annotations

import numpy as np

import sim_jax
import sim_torch
from backends import JaxOps, TorchOps, cfl_dt, initial_state, mesh_arrays, true_n_field
from mesh import make_channel_mesh
from physics import FlowConfig, residual


def main() -> None:
    mesh = make_channel_mesh(2000, length=200.0, width=20.0)
    print(f"mesh: {mesh.n_cells} cells, {mesh.n_faces} faces, min_dx={mesh.min_dx:.3f} m")
    U0 = initial_state(mesh, 1.5)
    nf = true_n_field(mesh)

    # 1) lake at rest: zero inflow, outlet WSE = still WSE -> R(U0) == 0
    rest = FlowConfig(q_in=0.0, wse_out=1.5)
    jops = JaxOps("float64")
    R = np.asarray(residual(jops, mesh_arrays(jops, mesh), rest, {"n": jops.asarray(nf)},
                            jops.asarray(U0)))
    print(f"[well-balanced] max|R(lake at rest)| = {np.abs(R).max():.3e}")

    # 2) torch vs jax forward agreement on an unsteady inflow run
    cfg = FlowConfig(q_in=1.0, wse_out=1.5)
    dt = cfl_dt(mesh)
    n_save, k = 10, 50
    J = sim_jax.build(mesh, cfg, dt, n_save, k, checkpoint=False)
    T = sim_torch.build(mesh, cfg, dt, n_save, k, checkpoint=False)
    thJ = sim_jax.to_theta(J["ops"], "field", nf, device=J["device"])
    thT = sim_torch.to_theta(T["ops"], "field", nf)
    hJ = np.asarray(J["simulate"](thJ, jops.asarray(U0)))
    hT = T["simulate"](thT, T["ops"].asarray(U0)).cpu().numpy()
    print(f"[torch vs jax] dt={dt:.4f}s  max|h_jax-h_torch| = {np.abs(hJ - hT).max():.3e}"
          f"  (h range {hJ.min():.3f}..{hJ.max():.3f} m)")

    # 3) gradient check (field + NN modes) against central finite differences
    h_obs = hJ + 0.0  # observations from the "true" n field
    for mode in ("field", "nn"):
        n_guess = np.full(mesh.n_cells, 0.035)
        th = sim_jax.to_theta(J["ops"], mode, n_guess, device=J["device"])
        L, g = J["grad"](th, jops.asarray(U0), jops.asarray(h_obs))
        thT2 = sim_torch.to_theta(T["ops"], mode, n_guess)
        LT, gT = T["grad"](thT2, T["ops"].asarray(U0), T["ops"].asarray(h_obs))
        import jax

        from jax.flatten_util import ravel_pytree
        flat, unravel = ravel_pytree(th)
        gflat, _ = ravel_pytree(g)
        gT_flat = np.concatenate([x.detach().cpu().numpy().ravel() for x in gT])
        rng = np.random.default_rng(1)
        d = rng.normal(size=flat.shape)
        d /= np.linalg.norm(d)
        eps = 1e-6 if mode == "field" else 1e-5
        Lp = J["loss"](unravel(flat + eps * d), jops.asarray(U0), jops.asarray(h_obs))
        Lm = J["loss"](unravel(flat - eps * d), jops.asarray(U0), jops.asarray(h_obs))
        fd = float(Lp - Lm) / (2 * eps)
        ad = float(np.dot(np.asarray(gflat), d))
        print(f"[grad {mode:5s}] L_jax={float(L):.6e} L_torch={float(LT):.6e} "
              f"AD={ad:.6e} FD={fd:.6e} relerr={abs(ad - fd) / abs(fd):.2e} "
              f"|g_jax-g_torch|/|g|={np.linalg.norm(np.asarray(gflat) - gT_flat) / np.linalg.norm(gflat):.2e}")

    # 4) march toward steady state; check mass balance q_out ~ q_in
    Js = sim_jax.build(mesh, cfg, dt, 20, 1000, checkpoint=False)
    import jax

    def run_U(theta, U0_):
        from physics import ssprk2_step
        ops = Js["ops"]
        m = mesh_arrays(ops, mesh)
        def body(U, _):
            return ssprk2_step(ops, m, cfg, theta, U, dt), None
        U, _ = jax.lax.scan(body, U0_, None, length=20000)
        return U

    U = np.asarray(jax.jit(run_U)(thJ, jops.asarray(U0)))
    xc = mesh.centroid[:, 0]
    for xs in (20.0, 100.0, 180.0):
        sel = np.abs(xc - xs) < 2.0
        print(f"  t={20000 * dt:.0f}s  x={xs:5.0f} m: mean hu={U[sel, 1].mean():.4f} m^2/s "
              f"(q_in={cfg.q_in}), mean h={U[sel, 0].mean():.3f} m")


if __name__ == "__main__":
    main()
