"""Flood wave through a compound channel: hydrograph, floodplain wetting and drying.

Usage: python flood_wave_demo.py OUT_DIR [NX NY [T_END_H]]
Valley 2 km x 200 m (slope 1e-3), main channel 40 m wide and 2 m deep, floodplains rising
3 % to the valley sides, n = 0.035. Inflow hydrograph 40 -> 400 -> 40 m^3/s; exit stage =
normal depth of the compound section for the current discharge. The initial condition is the
base flow after a 2 h spin-up at constant discharge. Outputs: VTK frames (ParaView), figures, mass-balance summary.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import hydrograd  # noqa: F401
from hydrograd.apps.forward import derived_fields, run_forward
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.plot import plot_cell_fields
from hydrograd.io.synthetic import write_channel_case

L, W, S0, N_MAN = 2000.0, 200.0, 1.0e-3, 0.035
T_HYD = np.array([0.0, 0.5, 1.5, 4.0, 6.0]) * 3600.0
Q_HYD = np.array([40.0, 40.0, 400.0, 40.0, 40.0])


def bed(X, Y):
    """z_b (m): valley slope + trapezoidal main channel + inclined floodplains."""
    r = np.abs(Y - 0.5 * W)
    lateral = np.where(r <= 20.0, 0.0, np.where(r <= 25.0, 2.0 * (r - 20.0) / 5.0, 2.0 + 0.03 * (r - 25.0)))
    return S0 * (L - X) + lateral


def normal_stage(Q: float, ny: int) -> float:
    """WSE (m) at the outlet (x = L) carrying Q in uniform flow (strip Manning, dy = W/ny)."""
    y = (np.arange(ny) + 0.5) * W / ny
    zb = bed(np.full(ny, L), y)
    flow = lambda w: np.sum(np.maximum(w - zb, 0) ** (5 / 3)) * (W / ny) * np.sqrt(S0) / N_MAN - Q
    return brentq(flow, zb.min() + 1e-6, zb.min() + 20.0)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    nx, ny = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (400, 40)
    t_end = float(sys.argv[4]) * 3600.0 if len(sys.argv) > 4 else 6.0 * 3600.0
    stage = np.array([normal_stage(q, ny) for q in Q_HYD])
    h_base = normal_stage(Q_HYD[0], ny) - bed(np.array([L]), np.array([W / 2]))[0]

    # 1. Base flow: transient spin-up at constant Q from a normal-depth guess. (Steady Newton
    #    cannot wet cells that start dry: d max(h, 0)/dh = 0, so the wet extent must settle in time.)
    common = dict(length=L, width=W, n_manning=N_MAN, zb_fn=bed, inlet="INLET-Q", outlet="EXIT-H",
                  wstill=0.0, dt=0.1)
    d0 = out / "base_flow"
    write_channel_case(d0, nx, ny, Q=Q_HYD[0], wse_out=stage[0], solver="SSPRK3()", t_end=2 * 3600.0,
                       ic_wse=lambda x, y: S0 * (L - x) + h_base, **common)
    t0 = time.perf_counter(); base = run_forward(setup_case(read_control(d0 / "run_control.json")))
    print(f"base-flow spin-up (2 h): {time.perf_counter() - t0:.1f} s", flush=True)
    U = base["ys"][-1]
    case0 = setup_case(read_control(d0 / "run_control.json"))
    wse0 = (U[0] + case0.wstill).tolist(); n = len(wse0)

    # 2. Flood wave from the base flow.
    d1 = out / "flood_wave"; d1.mkdir(parents=True, exist_ok=True)
    (d1 / "ic_base_flow.json").write_text(json.dumps({"wse": wse0, "wstill": case0.wstill.tolist(),
                                                     "q_x": U[1].tolist(), "q_y": U[2].tolist()}))
    write_channel_case(d1, nx, ny, Q_series=(T_HYD, Q_HYD), WSE_series=(T_HYD, stage), solver="SSPRK3()",
                       t_end=t_end, n_save=int(round(t_end / 900.0)), ic_file="ic_base_flow.json", **common)
    case = setup_case(read_control(d1 / "run_control.json"))
    t0 = time.perf_counter(); r = run_forward(case); wall = time.perf_counter() - t0
    print(f"flood wave: {case.mesh.n_cells} cells, {t_end / 3600:.1f} h simulated in {wall:.1f} s", flush=True)

    # 3. Diagnostics: storage, inundated area, outflow from the mass balance.
    m, area = case.model, np.asarray(case.model.area)
    ts, ys = r["ts"], r["ys"]
    hs = [np.asarray(Uk[0]) + np.asarray(m.hstill) for Uk in ys]
    V = np.array([np.sum(h * area) for h in hs])
    wet_area = np.array([np.sum(area[h > 0.01]) for h in hs])
    Qin = np.interp(ts, T_HYD, Q_HYD)
    tq = np.linspace(0, ts[-1], 200001)
    cum_in = np.interp(ts, tq, np.concatenate([[0], np.cumsum(0.5 * (np.interp(tq[1:], T_HYD, Q_HYD) + np.interp(tq[:-1], T_HYD, Q_HYD)) * np.diff(tq))]))
    Qout = Qin - np.gradient(V, ts)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].plot(ts / 3600, Qin, "k-", label="inflow Q(t) (prescribed)")
    ax[0].plot(ts / 3600, Qout, "C0o-", ms=3, label="outflow = Q_in - dV/dt")
    ax[0].axhline(120, color="gray", ls=":", label="approx. bankfull")
    ax[0].set_xlabel("t (h)"); ax[0].set_ylabel("discharge (m$^3$/s)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(ts / 3600, wet_area / (L * W) * 100, "C2o-", ms=3)
    ax[1].set_xlabel("t (h)"); ax[1].set_ylabel("inundated area (% of valley, h > 1 cm)"); ax[1].grid(alpha=.3)
    fig.suptitle(f"Flood wave through a compound channel ({m.n_cells:,} cells)")
    fig.tight_layout(); fig.savefig(out / "flood_wave_hydrographs.png", dpi=140); plt.close(fig)

    pick = [int(np.argmin(np.abs(ts - 3600 * t))) for t in (0.0, 1.0, 1.5, 2.5, 4.0, t_end / 3600)]
    frames = {f"t = {ts[k] / 3600:.1f} h, Q_in = {Qin[k]:.0f} m$^3$/s: depth": (np.where(hs[k] > 0.01, hs[k], np.nan), "m")
              for k in dict.fromkeys(pick)}
    plot_cell_fields(case.mesh, frames, out / "flood_wave_inundation.png", panel_height=1.35, width=12,
                     cmaps={k: "Blues" for k in frames}, title="Water depth (white = dry, h < 1 cm)")
    summary = {"cells": m.n_cells, "wall_s": wall, "min_depth_m": float(min(h.min() for h in hs)),
               "max_inundated_pct": float(wet_area.max() / (L * W) * 100),
               "storage_change_m3": float(V[-1] - V[0]), "inflow_volume_m3": float(cum_in[-1]),
               "outflow_volume_m3": float(cum_in[-1] - (V[-1] - V[0]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


if __name__ == "__main__":
    main()
