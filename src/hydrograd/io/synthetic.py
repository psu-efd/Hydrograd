"""Synthetic SRH-2D cases of any size (benchmarks and tests).

A rectangular channel of length ``length`` (m, x) and width ``width`` (m, y),
meshed with nx x ny quads, bed ``zb = s0 (length - x) + a exp(-((x - x_b)/w)^2)``
(downstream slope plus a Gaussian bump), inlet-Q on the left (x = 0), exit-H on
the right (x = length), walls along y = 0 and y = width. One material zone.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

__all__ = ["write_channel_case"]


def _fmt_ids(ids: np.ndarray, per_line: int = 10) -> str:
    rows = [" ".join(str(int(i)) for i in ids[k:k + per_line]) for k in range(0, len(ids), per_line)]
    return "\n".join(rows)


def write_channel_case(case_dir: str | Path, nx: int, ny: int, length: float = 1000.0, width: float = 100.0,
                       slope: float = 1.0e-4, bump: float = 0.3, Q: float = 100.0, wse_out: float = 2.0,
                       n_manning: float = 0.03, t_end: float = 100.0, dt: float = 0.1,
                       solver: str = "Tsit5()", n_save: int = 1, inlet: str = "INLET-Q",
                       outlet: str = "EXIT-H", ic_wse: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                       wstill: float | None = None,
                       Q_series: tuple[Sequence[float], Sequence[float]] | None = None,
                       WSE_series: tuple[Sequence[float], Sequence[float]] | None = None,
                       zb_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                       ic_file: str | None = None, order: int = 2) -> dict[str, int]:
    """Write ``channel.srhgeom/.srhhydro/.srhmat`` and a forward ``run_control.json``.

    Args:
        case_dir: output directory (created).
        nx, ny: number of quads along and across the channel.
        length, width: channel size (m); slope: bed slope (-); bump: bump height (m).
        Q: inlet discharge (m^3/s); wse_out: exit WSE (m); n_manning: Manning's n.
        t_end, dt: simulated time and initial step (s); solver: ``ode_solver`` name.
        inlet, outlet: BC types of the x = 0 and x = length boundaries
            (``INLET-Q``/``EXIT-H``/``WALL``/``SYMM``).
        ic_wse: optional WSE(x, y) (m) at cell centroids for the IC (written to
            ``initial_condition.json``; clipped to the bed, q = 0); default: flat ``wse_out``.
        wstill: still-water elevation (m) for the xi-form state (default: ``wse_out``).
        Q_series: optional inlet hydrograph (times (s), Q (m^3/s)); written to
            ``inflow.xys`` (SRH-2D convention, times in hours) and used instead of ``Q``.
        WSE_series: optional exit stage series (times (s), WSE (m)) -> ``stage.xys``.
        zb_fn: optional bed elevation z_b(x, y) (m) at nodes, replacing slope + bump.
        ic_file: optional existing IC JSON (keys wse, wstill, q_x, q_y) in ``case_dir``.
        order: spatial order (``numerics.spatial_order`` in run_control.json).

    Returns:
        {"n_cells": ..., "n_nodes": ...}
    """
    d = Path(case_dir)
    d.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, length, nx + 1)
    y = np.linspace(0.0, width, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")             # node (i, j) -> id i*(ny+1)+j+1
    if zb_fn is not None:
        Z = np.asarray(zb_fn(X, Y), dtype=float)
    else:
        Z = slope * (length - X) + bump * np.exp(-(((X - 0.5 * length) / (0.05 * length)) ** 2))
    nid = lambda i, j: i * (ny + 1) + j + 1
    lines = ["SRHGEOM 30", 'Name "synthetic channel"', 'GridUnit "Meters"']
    k = 0
    for i in range(nx):
        for j in range(ny):
            k += 1   # counter-clockwise: (i,j) (i+1,j) (i+1,j+1) (i,j+1)
            lines.append(f"Elem {k} {nid(i, j)} {nid(i + 1, j)} {nid(i + 1, j + 1)} {nid(i, j + 1)}")
    for i in range(nx + 1):
        for j in range(ny + 1):
            lines.append(f"Node {nid(i, j)} {X[i, j]:.17g} {Y[i, j]:.17g} {Z[i, j]:.17g}")
    ns = {1: [nid(0, j) for j in range(ny, -1, -1)],          # inlet  (x = 0)
          2: [nid(nx, j) for j in range(ny + 1)],             # outlet (x = length)
          3: [nid(i, 0) for i in range(nx + 1)],              # wall y = 0
          4: [nid(i, ny) for i in range(nx, -1, -1)]}         # wall y = width
    for b, ids in ns.items():
        lines.append(f"NodeString {b} " + _fmt_ids(np.asarray(ids)))
    (d / "channel.srhgeom").write_text("\n".join(lines) + "\n")

    hydro = ["SRHHYDRO 30", 'Case "channel"', 'Description "synthetic channel"', "RunType FLOW",
             "ModelTemp OFF", "UnsteadyOutput UNSTEADY", f"SimTime 0.0 {dt} {t_end / 3600.0}",
             "TurbulenceModel PARABOLIC", "ParabolicTurbulence 0.7", "InitCondOption DRY",
             'Grid "channel.srhgeom"', 'HydroMat "channel.srhmat"', "OutputFormat XMDFC SI",
             "OutputOption 1", "OutputInterval 1", f"ManningsN 0 {n_manning}", f"ManningsN 1 {n_manning}",
             f"BC 1 {inlet}", f"BC 2 {outlet}", "BC 3 WALL", "BC 4 WALL"]
    if inlet == "INLET-Q" and Q_series is not None:
        ts, qs = Q_series
        rows = [f"{t / 3600.0:.17g} {q:.17g}" for t, q in zip(ts, qs)]
        (d / "inflow.xys").write_text(f'XY 1 {len(rows)} 0 0 0 0 0 0 "inflow"\n' + "\n".join(rows) + "\n")
        hydro.append("IQParams 1 inflow.xys SI CONVEYANCE")
    elif inlet == "INLET-Q":
        hydro.append(f"IQParams 1 {Q} SI CONVEYANCE")
    if outlet == "EXIT-H" and WSE_series is not None:
        ts, ws = WSE_series
        rows = [f"{t / 3600.0:.17g} {w:.17g}" for t, w in zip(ts, ws)]
        (d / "stage.xys").write_text(f'XY 1 {len(rows)} 0 0 0 0 0 0 "stage"\n' + "\n".join(rows) + "\n")
        hydro.append("EWSParamsC 2 stage.xys SI C")
    elif outlet == "EXIT-H":
        hydro.append(f"EWSParamsC 2 {wse_out} SI C")
    (d / "channel.srhhydro").write_text("\n".join(hydro) + "\n")
    cells = np.arange(1, nx * ny + 1)
    (d / "channel.srhmat").write_text("SRHMAT 30\nNMaterials 2\nMatName 1 \"channel\"\nMaterial 1 "
                                      + _fmt_ids(cells) + "\n")
    ws = wse_out if wstill is None else wstill
    ic_opts = {"forward_simulation_initial_condition_options": "constant",
               "forward_simulation_initial_condition_constant_values": [wse_out, ws, 0.0, 0.0]}
    if ic_wse is not None:
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        XC, YC = np.meshgrid(xc, yc, indexing="ij")             # cell k = i*ny + j
        zc = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[1:, 1:] + Z[:-1, 1:])
        w = np.maximum(np.asarray(ic_wse(XC, YC), dtype=float), zc).ravel()
        n = nx * ny
        (d / "initial_condition.json").write_text(json.dumps(
            {"wse": w.tolist(), "wstill": [ws] * n, "q_x": [0.0] * n, "q_y": [0.0] * n}))
        ic_opts = {"forward_simulation_initial_condition_options": "from_file",
                   "forward_simulation_initial_condition_file_name": "initial_condition.json"}
    if ic_file is not None:
        ic_opts = {"forward_simulation_initial_condition_options": "from_file",
                   "forward_simulation_initial_condition_file_name": ic_file}
    ctrl = {
        "bVerbose": False,
        "numerics": {"scheme": "hr_hllc", "spatial_order": order},
        "control_variables": {"srhhydro_file_name": "channel.srhhydro", "bPerform_Forward_Simulation": True},
        "time_settings": {"bUse_srhhydro_time_settings": False, "tspan": [0.0, t_end], "dt": dt},
        "forward_simulation_options": {
            "forward_simulation_ManningN_option": "constant", "forward_simulation_solver": "SciML",
            "forward_simulation_ode_solver": solver, "forward_simulation_adaptive": True,
            "forward_simulation_nSave": n_save, **ic_opts,
            "forward_simulation_save_solution_truth_file_name": "truth.json"},
    }
    (d / "run_control.json").write_text(json.dumps(ctrl, indent=1))
    return {"n_cells": nx * ny, "n_nodes": (nx + 1) * (ny + 1)}
