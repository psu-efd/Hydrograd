"""Dam break (Stoker wet bed, Ritter dry bed) with the default wet/dry scheme.

Usage: python dambreak_run.py   (writes dambreak_hr0.1.npz and dambreak_hr0.0.npz in the cwd)

Case: frictionless horizontal channel 1000 m x 10 m (1000 x 2 cells), walls at both ends,
dam at x0 = 500 m. Initial depth h_l = 1 m upstream and h_r downstream:
h_r = 0.1 m (Stoker, wet bed) or h_r = 0 (Ritter, dry bed). Run to T = 60 s with SSPRK3.

Comparison with the exact Riemann solution (``hydrograd.validation.dambreak.dam_break``):
L1 and max depth errors (m), L1 velocity error (m/s), relative mass change, and for the dry
bed the wet-front position against the Ritter front x0 + 2 sqrt(g h_l) T (m).
Depths below 1e-3 m count as dry.
"""
import dataclasses  # noqa: F401
import sys  # noqa: F401
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.ode import solve_swe
from hydrograd.validation.dambreak import dam_break

# Simulated time (s).
T = 60.0

for h_r in (0.1, 0.0):

    def initial_wse(x, y):
        """Initial WSE (m): 1 m upstream of the dam (x < 500 m), h_r downstream."""
        return np.where(x < 500.0, 1.0, h_r)

    # 1. Write and set up the case.
    case_dir = Path(tempfile.mkdtemp()) / "db"
    write_channel_case(case_dir, 1000, 2, length=1000.0, width=10.0, slope=0.0, bump=0.0,
                       n_manning=0.0, inlet="WALL", outlet="WALL", wse_out=h_r, wstill=0.5,
                       t_end=T, dt=0.05, n_save=1, solver="SSPRK3()", ic_wse=initial_wse)
    case = setup_case(read_control(case_dir / "run_control.json"))

    # 2. Run (jitted, timing includes compilation).
    def simulate(U0):
        """SSPRK3 solution from the initial state U0 (shape [3, n_cells])."""
        return solve_swe(case.model, {}, U0, case.ode)

    t0 = time.time()
    sol = jax.jit(simulate)(case.U0)

    # 3. Final depth (m), dry cells set to 0.
    U = np.asarray(sol.ys[-1])
    h = U[0] + np.asarray(case.model.hstill)
    h = np.where(h < 1e-3, 0.0, h)

    # 4. Exact solution and velocity (m/s), zero on dry cells.
    x = case.mesh.centroid[:, 0]
    he, ue = dam_break(x, T, 1.0, h_r, x0=500.0)
    u = U[1] / np.maximum(h, 1e-3) * (h > 1e-3)
    e = h - he

    # 5. Mass (m^3) before and after; wet-front position for the dry bed.
    mass0 = np.sum((case.wse0 - 0.0) * case.mesh.area)
    mass1 = np.sum(h * case.mesh.area)
    front = x[h > 1e-3].max() if h_r == 0 else None

    # 6. Report and save the profiles.
    front_text = (f"; wet front {front:.1f} m (exact {500 + 2*np.sqrt(9.81)*T:.1f})"
                  if front else "")
    print(f"h_r={h_r}: result {sol.result}, steps {int(sol.stats['num_steps'])}, "
          f"{time.time()-t0:.1f}s; "
          f"h L1 err {np.mean(np.abs(e)):.4f} m, max {np.abs(e).max():.3f} m; "
          f"u L1 err {np.mean(np.abs(u-ue)):.4f} m/s; "
          f"mass rel. change {(mass1-mass0)/mass0:.1e}" + front_text)
    np.savez(f"dambreak_hr{h_r}.npz", x=x, h=h, u=u, he=he, ue=ue)
