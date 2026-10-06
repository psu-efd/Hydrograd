"""512k-cell synthetic channel: steady (Newton) and transient runs, VTK + PNG output.

Usage: python large_channel_demo.py OUT_DIR [NX NY]
Writes OUT_DIR/steady and OUT_DIR/transient case folders (``hydrograd run`` outputs:
forward_simulation_results_*.vtk for ParaView) and PNG figures in OUT_DIR.
"""
import json, sys, time
from pathlib import Path

import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.forward import run_forward
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.plot import plot_cell_fields
from hydrograd.io.synthetic import write_channel_case

out = Path(sys.argv[1]); nx, ny = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (3200, 160)
out.mkdir(parents=True, exist_ok=True)
runs = {"steady": dict(solver="SteadyNewton()", t_end=1.0, n_save=1),
        "transient": dict(solver="Tsit5()", t_end=120.0, n_save=4)}
res = {}
for name, kw in runs.items():
    d = out / name
    write_channel_case(d, nx, ny, **kw)
    t0 = time.perf_counter()
    case = setup_case(read_control(d / "run_control.json"))
    res[name] = (case, run_forward(case))
    print(f"{name}: {case.mesh.n_cells} cells, run + output {time.perf_counter() - t0:.1f} s", flush=True)

case, r = res["steady"]
m, f = case.mesh, r["fields"]
speed = np.hypot(f["u"], f["v"])
fields = {"Bed elevation z_b": (f["zb"], "m"), "Water depth h": (f["h"], "m"),
          "Water-surface elevation": (f["wse"], "m"), "Speed |U|": (speed, "m/s")}
cm = {"Bed elevation z_b": "terrain", "Water depth h": "Blues", "Speed |U|": "magma"}
t = f"Steady state (Newton), {m.n_cells:,} cells, Q = 100 m$^3$/s, n = 0.03"
plot_cell_fields(m, fields, out / "steady_full.png", title=t, cmaps=cm, panel_height=1.6, width=13)
plot_cell_fields(m, fields, out / "steady_zoom_bump.png", xlim=(400, 600), title=t + " — zoom on the bump",
                 cmaps=cm, quiver=(f["u"], f["v"]), panel_height=2.6, width=11)
case_t, rt = res["transient"]
from hydrograd.apps.forward import derived_fields
frames = {}
for k, tk in enumerate(rt["ts"][1:], start=1):
    fk = derived_fields(case_t.model, {}, rt["ys"][k], case_t.wstill)
    frames[f"t = {tk:.0f} s: free-surface rise above initial level"] = (fk["wse"] - case_t.wse0, "m")
plot_cell_fields(case_t.mesh, frames, out / "transient_wave.png", cmaps={k: "RdBu_r" for k in frames},
                 title=f"Transient from rest: inflow wave, {case_t.mesh.n_cells:,} cells", panel_height=1.5, width=13)
mid = np.abs(m.centroid[:, 1] - 50.0) < 50.0 / ny
print(json.dumps({"steady_newton_wall_s": r["wall_s"], "transient_wall_s": rt["wall_s"],
                  "max_speed": float(speed.max()), "wse_range": [float(f['wse'].min()), float(f['wse'].max())]}))
