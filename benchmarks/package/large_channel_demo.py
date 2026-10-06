"""512k-cell synthetic channel: steady (Newton) and transient runs, VTK + PNG output.

Usage: python large_channel_demo.py OUT_DIR [NX NY]

Case: the default synthetic channel of ``hydrograd.io.synthetic.write_channel_case`` (a
straight channel with a bed bump near x = 500 m, inflow Q = 100 m^3/s, Manning n = 0.03),
meshed with NX x NY quadrilaterals (default 3200 x 160 = 512,000 cells). Two runs:

- ``steady``: steady state by pseudo-transient Newton (``SteadyNewton()``);
- ``transient``: 120 s from rest with Tsit5, 4 saved frames (inflow wave).

Writes OUT_DIR/steady and OUT_DIR/transient case folders (``hydrograd run`` outputs:
forward_simulation_results_*.vtk for ParaView) and PNG figures in OUT_DIR. Prints the wall
times (s), the maximum speed (m/s) and the WSE range (m) as one JSON line.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.forward import run_forward
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.plot import plot_cell_fields
from hydrograd.io.synthetic import write_channel_case

# 1. Command line: output folder and mesh size.
out = Path(sys.argv[1])
if len(sys.argv) > 3:
    nx, ny = (int(sys.argv[2]), int(sys.argv[3]))
else:
    nx, ny = (3200, 160)
out.mkdir(parents=True, exist_ok=True)

# 2. Write and run both cases (steady Newton, transient Tsit5).
runs = {
    "steady": dict(solver="SteadyNewton()", t_end=1.0, n_save=1),
    "transient": dict(solver="Tsit5()", t_end=120.0, n_save=4),
}
res = {}
for name, kw in runs.items():
    case_dir = out / name
    write_channel_case(case_dir, nx, ny, **kw)
    t0 = time.perf_counter()
    case = setup_case(read_control(case_dir / "run_control.json"))
    res[name] = (case, run_forward(case))
    print(
        f"{name}: {case.mesh.n_cells} cells, run + output {time.perf_counter() - t0:.1f} s",
        flush=True,
    )

# 3. Steady-state figures: full channel and a zoom on the bump with velocity arrows.
case, r = res["steady"]
mesh = case.mesh
f = r["fields"]
speed = np.hypot(f["u"], f["v"])
fields = {
    "Bed elevation z_b": (f["zb"], "m"),
    "Water depth h": (f["h"], "m"),
    "Water-surface elevation": (f["wse"], "m"),
    "Speed |U|": (speed, "m/s"),
}
cmaps = {"Bed elevation z_b": "terrain", "Water depth h": "Blues", "Speed |U|": "magma"}
title = f"Steady state (Newton), {mesh.n_cells:,} cells, Q = 100 m$^3$/s, n = 0.03"
plot_cell_fields(mesh, fields, out / "steady_full.png", title=title, cmaps=cmaps,
                 panel_height=1.6, width=13)
plot_cell_fields(mesh, fields, out / "steady_zoom_bump.png", xlim=(400, 600),
                 title=title + " — zoom on the bump", cmaps=cmaps, quiver=(f["u"], f["v"]),
                 panel_height=2.6, width=11)

# 4. Transient figure: free-surface rise above the initial level at each saved time.
case_t, rt = res["transient"]
from hydrograd.apps.forward import derived_fields

frames = {}
for k, tk in enumerate(rt["ts"][1:], start=1):
    fk = derived_fields(case_t.model, {}, rt["ys"][k], case_t.wstill)
    frames[f"t = {tk:.0f} s: free-surface rise above initial level"] = (fk["wse"] - case_t.wse0, "m")
plot_cell_fields(case_t.mesh, frames, out / "transient_wave.png",
                 cmaps={k: "RdBu_r" for k in frames},
                 title=f"Transient from rest: inflow wave, {case_t.mesh.n_cells:,} cells",
                 panel_height=1.5, width=13)

# 5. Summary line (``mid`` marks the centreline cells; computed but not used).
mid = np.abs(mesh.centroid[:, 1] - 50.0) < 50.0 / ny
wse_range = [float(f['wse'].min()), float(f['wse'].max())]
print(json.dumps({"steady_newton_wall_s": r["wall_s"], "transient_wall_s": rt["wall_s"],
                  "max_speed": float(speed.max()), "wse_range": wse_range}))
