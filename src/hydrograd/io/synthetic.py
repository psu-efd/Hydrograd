"""Synthetic SRH-2D cases of any size (benchmarks and tests).

Geometry
--------
A rectangular channel of length ``length`` (m, x) and width ``width`` (m, y), meshed
with nx x ny quads. Default bed (m):

    zb = s0 (length - x) + a exp(-((x - x_b) / w)^2)

(downstream slope s0 plus a Gaussian bump of height a, centred at x_b = length/2 with
width w = 0.05 length).

Boundaries
----------
- nodestring 1: inlet-Q on the left (x = 0);
- nodestring 2: exit-H on the right (x = length);
- nodestrings 3 and 4: walls along y = 0 and y = width.

One material zone covers all cells.

Numbering
---------
Node (i, j), i = 0..nx, j = 0..ny, has the 1-based ID i (ny + 1) + j + 1. Cell (i, j),
i = 0..nx-1, j = 0..ny-1, has the 1-based ID i ny + j + 1 and the counter-clockwise
nodes (i, j), (i+1, j), (i+1, j+1), (i, j+1).

Files written
-------------
``channel.srhgeom``, ``channel.srhhydro``, ``channel.srhmat``, ``run_control.json``,
and optionally ``inflow.xys``, ``stage.xys`` and ``initial_condition.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

__all__ = ["write_channel_case"]


def _fmt_ids(ids: np.ndarray, per_line: int = 10) -> str:
    """Format integer IDs as text, ``per_line`` IDs per line.

    Args:
        ids: integer IDs, shape [n].
        per_line: number of IDs on each line.

    Returns:
        The IDs separated by spaces, with lines separated by newlines (no trailing newline).
    """
    rows = []
    for k in range(0, len(ids), per_line):
        chunk = ids[k:k + per_line]
        rows.append(" ".join(str(int(i)) for i in chunk))

    return "\n".join(rows)


def write_channel_case(
    case_dir: str | Path,
    nx: int,
    ny: int,
    length: float = 1000.0,
    width: float = 100.0,
    slope: float = 1.0e-4,
    bump: float = 0.3,
    Q: float = 100.0,
    wse_out: float = 2.0,
    n_manning: float = 0.03,
    t_end: float = 100.0,
    dt: float = 0.1,
    solver: str = "Tsit5()",
    n_save: int = 1,
    inlet: str = "INLET-Q",
    outlet: str = "EXIT-H",
    ic_wse: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    wstill: float | None = None,
    Q_series: tuple[Sequence[float], Sequence[float]] | None = None,
    WSE_series: tuple[Sequence[float], Sequence[float]] | None = None,
    zb_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ic_file: str | None = None,
    order: int = 2,
    zone_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, int]:
    """Write ``channel.srhgeom/.srhhydro/.srhmat`` and a forward ``run_control.json``.

    Args:
        case_dir: output directory (created).
        nx: number of quads along the channel (x).
        ny: number of quads across the channel (y).
        length: channel length (m).
        width: channel width (m).
        slope: bed slope (-).
        bump: bump height (m).
        Q: inlet discharge (m^3/s).
        wse_out: exit WSE (m).
        n_manning: Manning's n (s/m^(1/3)).
        t_end: simulated time (s).
        dt: initial time step (s).
        solver: ``ode_solver`` name.
        n_save: number of saved solutions (``forward_simulation_nSave``).
        inlet: BC type of the x = 0 boundary (``INLET-Q``/``EXIT-H``/``WALL``/``SYMM``).
        outlet: BC type of the x = length boundary (same choices).
        ic_wse: optional WSE(x, y) (m) at cell centroids for the IC (written to
            ``initial_condition.json``; clipped to the bed, q = 0); default: flat ``wse_out``.
        wstill: still-water elevation (m) for the xi-form state (default: ``wse_out``).
        Q_series: optional inlet hydrograph (times (s), Q (m^3/s)); written to
            ``inflow.xys`` (SRH-2D convention, times in hours) and used instead of ``Q``.
        WSE_series: optional exit stage series (times (s), WSE (m)) -> ``stage.xys``.
        zb_fn: optional bed elevation z_b(x, y) (m) at nodes, replacing slope + bump.
        ic_file: optional existing IC JSON (keys wse, wstill, q_x, q_y) in ``case_dir``.
        order: spatial order (``numerics.spatial_order`` in run_control.json).
        zone_fn: optional material ID (an integer >= 1) of each cell from its centroid
            (x, y) (m); every material gets Manning's n ``n_manning``. Default: all cells
            in material 1.

    Returns:
        {"n_cells": ..., "n_nodes": ...}
    """
    d = Path(case_dir)
    d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Node coordinates and bed elevation (m), arrays of shape [nx + 1, ny + 1].
    # ------------------------------------------------------------------
    x = np.linspace(0.0, length, nx + 1)
    y = np.linspace(0.0, width, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if zb_fn is not None:
        Z = np.asarray(zb_fn(X, Y), dtype=float)
    else:
        bump_shape = np.exp(-(((X - 0.5 * length) / (0.05 * length)) ** 2))
        Z = slope * (length - X) + bump * bump_shape

    def nid(i: int, j: int) -> int:
        """1-based ID of node (i, j).

        Args:
            i: node index along x, 0..nx.
            j: node index along y, 0..ny.

        Returns:
            The node ID i (ny + 1) + j + 1.
        """
        return i * (ny + 1) + j + 1

    # ------------------------------------------------------------------
    # .srhgeom: cells, nodes, and the four boundary nodestrings.
    # ------------------------------------------------------------------
    lines = ["SRHGEOM 30", 'Name "synthetic channel"', 'GridUnit "Meters"']

    # Cells, counter-clockwise: (i,j) (i+1,j) (i+1,j+1) (i,j+1).
    k = 0
    for i in range(nx):
        for j in range(ny):
            k += 1
            lines.append(
                f"Elem {k} {nid(i, j)} {nid(i + 1, j)} {nid(i + 1, j + 1)} {nid(i, j + 1)}"
            )

    for i in range(nx + 1):
        for j in range(ny + 1):
            lines.append(f"Node {nid(i, j)} {X[i, j]:.17g} {Y[i, j]:.17g} {Z[i, j]:.17g}")

    ns = {
        1: [nid(0, j) for j in range(ny, -1, -1)],    # inlet  (x = 0)
        2: [nid(nx, j) for j in range(ny + 1)],       # outlet (x = length)
        3: [nid(i, 0) for i in range(nx + 1)],        # wall y = 0
        4: [nid(i, ny) for i in range(nx, -1, -1)],   # wall y = width
    }
    for b, ids in ns.items():
        lines.append(f"NodeString {b} " + _fmt_ids(np.asarray(ids)))

    (d / "channel.srhgeom").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # .srhhydro: run settings, Manning's n, boundary conditions.
    # ------------------------------------------------------------------
    hydro = [
        "SRHHYDRO 30",
        'Case "channel"',
        'Description "synthetic channel"',
        "RunType FLOW",
        "ModelTemp OFF",
        "UnsteadyOutput UNSTEADY",
        f"SimTime 0.0 {dt} {t_end / 3600.0}",
        "TurbulenceModel PARABOLIC",
        "ParabolicTurbulence 0.7",
        "InitCondOption DRY",
        'Grid "channel.srhgeom"',
        'HydroMat "channel.srhmat"',
        "OutputFormat XMDFC SI",
        "OutputOption 1",
        "OutputInterval 1",
        f"ManningsN 0 {n_manning}",
        f"ManningsN 1 {n_manning}",
        f"BC 1 {inlet}",
        f"BC 2 {outlet}",
        "BC 3 WALL",
        "BC 4 WALL",
    ]

    # Inlet discharge: a constant, or a hydrograph written to inflow.xys (times in hours).
    if inlet == "INLET-Q" and Q_series is not None:
        ts, qs = Q_series
        rows = [f"{t / 3600.0:.17g} {q:.17g}" for t, q in zip(ts, qs)]
        xys_text = f'XY 1 {len(rows)} 0 0 0 0 0 0 "inflow"\n' + "\n".join(rows) + "\n"
        (d / "inflow.xys").write_text(xys_text)
        hydro.append("IQParams 1 inflow.xys SI CONVEYANCE")
    elif inlet == "INLET-Q":
        hydro.append(f"IQParams 1 {Q} SI CONVEYANCE")

    # Exit WSE: a constant, or a stage series written to stage.xys (times in hours).
    if outlet == "EXIT-H" and WSE_series is not None:
        stage_t, stage_v = WSE_series
        rows = [f"{t / 3600.0:.17g} {w:.17g}" for t, w in zip(stage_t, stage_v)]
        xys_text = f'XY 1 {len(rows)} 0 0 0 0 0 0 "stage"\n' + "\n".join(rows) + "\n"
        (d / "stage.xys").write_text(xys_text)
        hydro.append("EWSParamsC 2 stage.xys SI C")
    elif outlet == "EXIT-H":
        hydro.append(f"EWSParamsC 2 {wse_out} SI C")

    (d / "channel.srhhydro").write_text("\n".join(hydro) + "\n")

    # ------------------------------------------------------------------
    # .srhmat: one material zone with every cell, or the zones of zone_fn.
    # ------------------------------------------------------------------
    cells = np.arange(1, nx * ny + 1)
    if zone_fn is None:
        mat_header = "SRHMAT 30\nNMaterials 2\nMatName 1 \"channel\"\nMaterial 1 "
        mat_text = mat_header + _fmt_ids(cells) + "\n"
    else:
        # Cell centroids in the cell order of the .srhgeom file (i outer, j inner).
        x_centre = 0.5 * (x[:-1] + x[1:])
        y_centre = 0.5 * (y[:-1] + y[1:])
        XC, YC = np.meshgrid(x_centre, y_centre, indexing="ij")
        material = np.asarray(zone_fn(XC, YC), dtype=int).ravel()
        if material.min() < 1:
            raise ValueError("zone_fn must return material IDs >= 1")

        n_materials = int(material.max()) + 1
        mat_lines = ["SRHMAT 30", f"NMaterials {n_materials}"]
        for zone in range(1, n_materials):
            mat_lines.append(f'MatName {zone} "zone{zone}"')
            mat_lines.append(f"Material {zone} " + _fmt_ids(cells[material == zone]))
        mat_text = "\n".join(mat_lines) + "\n"

        # Manning's n for the extra materials (0 and 1 are already listed).
        extra = [f"ManningsN {zone} {n_manning}" for zone in range(2, n_materials)]
        hydro_text = (d / "channel.srhhydro").read_text()
        hydro_text = hydro_text.replace(f"ManningsN 1 {n_manning}\n",
                                        f"ManningsN 1 {n_manning}\n" + "".join(e + "\n" for e in extra))
        (d / "channel.srhhydro").write_text(hydro_text)
    (d / "channel.srhmat").write_text(mat_text)

    # ------------------------------------------------------------------
    # Initial condition: constant WSE, an IC file written here, or a given IC file.
    # ------------------------------------------------------------------
    ws: float = wse_out if wstill is None else wstill
    ic_opts = {
        "forward_simulation_initial_condition_options": "constant",
        "forward_simulation_initial_condition_constant_values": [wse_out, ws, 0.0, 0.0],
    }

    if ic_wse is not None:
        # Cell centroids, shape [nx, ny]; cell k = i*ny + j after ravel().
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        XC, YC = np.meshgrid(xc, yc, indexing="ij")

        # Cell bed = mean of the four corner nodes; the WSE is clipped to it (dry cells).
        zc = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[1:, 1:] + Z[:-1, 1:])
        w = np.maximum(np.asarray(ic_wse(XC, YC), dtype=float), zc).ravel()

        n = nx * ny
        ic_data = {"wse": w.tolist(), "wstill": [ws] * n, "q_x": [0.0] * n, "q_y": [0.0] * n}
        (d / "initial_condition.json").write_text(json.dumps(ic_data))

        ic_opts = {
            "forward_simulation_initial_condition_options": "from_file",
            "forward_simulation_initial_condition_file_name": "initial_condition.json",
        }

    if ic_file is not None:
        ic_opts = {
            "forward_simulation_initial_condition_options": "from_file",
            "forward_simulation_initial_condition_file_name": ic_file,
        }

    # ------------------------------------------------------------------
    # run_control.json for a forward simulation.
    # ------------------------------------------------------------------
    ctrl = {
        "bVerbose": False,
        "numerics": {"scheme": "hr_hllc", "spatial_order": order},
        "control_variables": {
            "srhhydro_file_name": "channel.srhhydro",
            "bPerform_Forward_Simulation": True,
        },
        "time_settings": {"bUse_srhhydro_time_settings": False, "tspan": [0.0, t_end], "dt": dt},
        "forward_simulation_options": {
            "forward_simulation_ManningN_option": "constant",
            "forward_simulation_solver": "SciML",
            "forward_simulation_ode_solver": solver,
            "forward_simulation_adaptive": True,
            "forward_simulation_nSave": n_save,
            **ic_opts,
            "forward_simulation_save_solution_truth_file_name": "truth.json",
        },
    }
    (d / "run_control.json").write_text(json.dumps(ctrl, indent=1))

    return {"n_cells": nx * ny, "n_nodes": (nx + 1) * (ny + 1)}
