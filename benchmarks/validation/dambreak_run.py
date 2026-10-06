"""Dam break (Stoker wet bed, Ritter dry bed) with the default wet/dry scheme; writes dambreak_hr*.npz."""
import sys, tempfile, time, dataclasses
from pathlib import Path
import numpy as np, jax
import hydrograd
from hydrograd.io.synthetic import write_channel_case
from hydrograd.io.control import read_control
from hydrograd.apps.setup import setup_case
from hydrograd.solvers.ode import solve_swe
from hydrograd.validation.dambreak import dam_break
T = 60.0
for h_r in (0.1, 0.0):
    d = Path(tempfile.mkdtemp()) / "db"
    write_channel_case(d, 1000, 2, length=1000.0, width=10.0, slope=0.0, bump=0.0, n_manning=0.0,
                       inlet="WALL", outlet="WALL", wse_out=h_r, wstill=0.5, t_end=T, dt=0.05, n_save=1, solver="SSPRK3()",
                       ic_wse=lambda x, y: np.where(x < 500.0, 1.0, h_r))
    case = setup_case(read_control(d / "run_control.json"))
    t0 = time.time(); sol = jax.jit(lambda U0: solve_swe(case.model, {}, U0, case.ode))(case.U0)
    U = np.asarray(sol.ys[-1]); h = U[0] + np.asarray(case.model.hstill); h = np.where(h < 1e-3, 0.0, h)
    x = case.mesh.centroid[:, 0]; he, ue = dam_break(x, T, 1.0, h_r, x0=500.0)
    u = U[1] / np.maximum(h, 1e-3) * (h > 1e-3)
    e = h - he
    mass0 = np.sum((case.wse0 - 0.0) * case.mesh.area); mass1 = np.sum(h * case.mesh.area)
    front = x[h > 1e-3].max() if h_r == 0 else None
    print(f"h_r={h_r}: result {sol.result}, steps {int(sol.stats['num_steps'])}, {time.time()-t0:.1f}s; "
          f"h L1 err {np.mean(np.abs(e)):.4f} m, max {np.abs(e).max():.3f} m; u L1 err {np.mean(np.abs(u-ue)):.4f} m/s; "
          f"mass rel. change {(mass1-mass0)/mass0:.1e}" + (f"; wet front {front:.1f} m (exact {500 + 2*np.sqrt(9.81)*T:.1f})" if front else ""))
    np.savez(f"dambreak_hr{h_r}.npz", x=x, h=h, u=u, he=he, ue=ue)
