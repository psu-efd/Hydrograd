"""Readers for SRH-2D case files: ``.srhhydro``, ``.srhgeom``, ``.srhmat``.

Semantics follow Hydrograd.jl (``src/utilities/SRH_2D/``) with two deliberate
fixes, both inert for the Julia examples:

* quotes are stripped from string values (Julia keeps them, which makes its
  ``GridUnit`` check fail);
* ``SimTime`` is read in SRH order ``t_start (h), dt (s), t_end (h)``.

All IDs in the files are 1-based and are kept 1-based here (dictionary keys);
conversion to 0-based array indices happens in :mod:`hydrograd.mesh`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GEOM_KEYS = ("Elem", "Node", "NodeString", "SRHGEOM", "Name", "GridUnit")


def _strip(s: str) -> str:
    return s.strip().strip('"').strip()


@dataclass(frozen=True)
class TimeSeries:
    """Piecewise-linear boundary time series (e.g. an inlet hydrograph).

    Attributes:
        t: times (s), increasing; values beyond the ends are held constant.
        v: values (m^3/s for discharge, m for WSE).
        source: file it was read from.
    """

    t: np.ndarray
    v: np.ndarray
    source: str = ""


def read_xys(path: str | Path) -> TimeSeries:
    """Read an SMS/SRH-2D ``.xys`` series: lines ``t value`` with t in hours.

    Header lines (e.g. ``XY 1 3 0 0 0 0 0 0 "name"``) and anything that is not two
    numbers are skipped. Times are converted to seconds.
    """
    t, v = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        t.append(a)
        v.append(b)
    if len(t) < 1 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: need increasing times in the .xys series")
    return TimeSeries(3600.0 * np.asarray(t), np.asarray(v), str(path))


def _value_or_series(token: str, base: Path) -> float | TimeSeries:
    """A number, or a time-series file name (relative to the .srhhydro)."""
    try:
        return float(token)
    except ValueError:
        return read_xys(base / _strip(token))


@dataclass(frozen=True)
class SRHHydro:
    """Parsed ``.srhhydro`` control file.

    Attributes:
        path: file path.
        raw: every keyword -> list of value tokens (last occurrence wins for scalars).
        manning_n: material ID (0-based zone IDs as in the file) -> Manning's n (s/m^(1/3)).
        bc: boundary (nodestring) ID -> BC type in upper case (``INLET-Q``, ``EXIT-H``,
            ``WALL``, ``SYMM``); ``MONITORING`` lines are excluded.
        monitoring: nodestring IDs of monitoring lines.
        iq_params: inlet-Q ID -> (Q, unit, distribution type); Q is a number
            (m^3/s or ft^3/s) or a :class:`TimeSeries` read from an ``.xys`` file.
        ews_params: exit-H ID -> (WSE, unit, type); WSE (m or ft) is a number or a TimeSeries.
        grid_file, mat_file: names of the ``.srhgeom`` / ``.srhmat`` files.
        sim_time: (t_start (h), dt (s), t_end (h)).
    """

    path: Path
    raw: dict[str, list[str]]
    manning_n: dict[int, float]
    bc: dict[int, str]
    monitoring: list[int]
    iq_params: dict[int, tuple[float | TimeSeries, str, str]]
    ews_params: dict[int, tuple[float | TimeSeries, str, str]]
    grid_file: str
    mat_file: str
    sim_time: tuple[float, float, float]


def read_srhhydro(path: str | Path) -> SRHHydro:
    """Parse an SRH-2D ``.srhhydro`` file."""
    path = Path(path)
    raw: dict[str, list[str]] = {}
    manning: dict[int, float] = {}
    bc: dict[int, str] = {}
    monitoring: list[int] = []
    iq: dict[int, tuple[float | TimeSeries, str, str]] = {}
    ews: dict[int, tuple[float | TimeSeries, str, str]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) <= 1:
            continue
        key, vals = parts[0], parts[1:]
        if key == "ManningsN":
            manning[int(vals[0])] = float(vals[1])
        elif key == "BC":
            if vals[1].upper() == "MONITORING":
                monitoring.append(int(vals[0]))
            else:
                bc[int(vals[0])] = vals[1].upper()
        elif key == "IQParams":
            iq[int(vals[0])] = (_value_or_series(vals[1], path.parent), vals[2], vals[3] if len(vals) > 3 else "")
        elif key == "EWSParamsC":
            ews[int(vals[0])] = (_value_or_series(vals[1], path.parent), vals[2], vals[3] if len(vals) > 3 else "")
        else:
            raw[key] = vals
    st = [float(v) for v in raw.get("SimTime", ["0", "1", "0"])[:3]]
    return SRHHydro(path, raw, manning, bc, monitoring, iq, ews,
                    _strip(" ".join(raw["Grid"])), _strip(" ".join(raw["HydroMat"])),
                    (st[0], st[1], st[2]))


@dataclass(frozen=True)
class SRHGeom:
    """Parsed ``.srhgeom`` mesh.

    Attributes:
        nodes: node coordinates (x, y, z_bed) (m or ft), shape [n_nodes, 3]; row i is node i+1.
        cells: node IDs (0-based) of each cell, in file order (counter-clockwise expected).
        nodestrings: nodestring ID (1-based) -> node IDs (0-based) in file order.
        grid_unit: e.g. ``Meters``/``Feet`` (empty if absent).
    """

    nodes: np.ndarray
    cells: list[np.ndarray]
    nodestrings: dict[int, np.ndarray]
    grid_unit: str


def read_srhgeom(path: str | Path) -> SRHGeom:
    """Parse an SRH-2D ``.srhgeom`` file (IDs must be contiguous 1..N, as in Julia)."""
    node_rows: dict[int, tuple[float, float, float]] = {}
    elem_rows: dict[int, list[int]] = {}
    ns: dict[int, list[int]] = {}
    grid_unit = ""
    current_ns = None
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "Node":
            node_rows[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
            current_ns = None
        elif key == "Elem":
            elem_rows[int(parts[1])] = [int(p) for p in parts[2:]]
            current_ns = None
        elif key == "NodeString":
            current_ns = int(parts[1])
            ns[current_ns] = [int(p) for p in parts[2:]]
        elif key == "GridUnit":
            grid_unit = _strip(" ".join(parts[1:]))
        elif key in _GEOM_KEYS or "srhgeom" in line.lower():
            current_ns = None
        elif current_ns is not None:  # continuation of a nodestring
            ns[current_ns].extend(int(p) for p in parts)
    n_nodes, n_cells = len(node_rows), len(elem_rows)
    if sorted(node_rows) != list(range(1, n_nodes + 1)) or sorted(elem_rows) != list(range(1, n_cells + 1)):
        raise ValueError(f"{path}: Node/Elem IDs must be contiguous 1..N")
    nodes = np.array([node_rows[i] for i in range(1, n_nodes + 1)])
    cells = [np.asarray(elem_rows[i], dtype=np.int64) - 1 for i in range(1, n_cells + 1)]
    nodestrings = {k: np.asarray(v, dtype=np.int64) - 1 for k, v in ns.items()}
    return SRHGeom(nodes, cells, nodestrings, grid_unit)


@dataclass(frozen=True)
class SRHMat:
    """Parsed ``.srhmat`` material file.

    Attributes:
        n_materials: number of materials including the default zone 0.
        names: material ID -> name.
        zone_cells: material ID -> cell IDs (0-based).
    """

    n_materials: int
    names: dict[int, str]
    zone_cells: dict[int, np.ndarray]

    def cell_material_ids(self, n_cells: int) -> np.ndarray:
        """Material ID per cell (0 = default for cells not listed in any zone)."""
        mat = np.zeros(n_cells, dtype=np.int64)
        assigned = np.zeros(n_cells, dtype=bool)
        for mid in sorted(self.zone_cells):
            c = self.zone_cells[mid]
            c = c[~assigned[c]]  # first zone wins, as in Julia
            mat[c], assigned[c] = mid, True
        return mat


def read_srhmat(path: str | Path) -> SRHMat:
    """Parse an SRH-2D ``.srhmat`` file."""
    n_mat = 0
    names: dict[int, str] = {0: "Default"}
    zones: dict[int, list[int]] = {}
    current = None
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "NMaterials":
            n_mat, current = int(parts[1]), None
        elif key == "MatName":
            names[int(parts[1])], current = _strip(" ".join(parts[2:])), None
        elif key == "Material":
            current = int(parts[1])
            zones.setdefault(current, []).extend(int(p) for p in parts[2:])
        elif key == "SRHMAT":
            current = None
        elif current is not None:
            zones[current].extend(int(p) for p in parts)
    return SRHMat(n_mat, names, {k: np.asarray(v, dtype=np.int64) - 1 for k, v in zones.items()})


@dataclass(frozen=True)
class SRHCase:
    """The three SRH-2D files of one case."""

    hydro: SRHHydro
    geom: SRHGeom
    mat: SRHMat
    case_dir: Path = field(default=Path("."))


def read_srh_case(srhhydro_path: str | Path) -> SRHCase:
    """Read ``.srhhydro`` and the ``.srhgeom``/``.srhmat`` files it references."""
    srhhydro_path = Path(srhhydro_path)
    d = srhhydro_path.parent
    hydro = read_srhhydro(srhhydro_path)
    geom = read_srhgeom(d / hydro.grid_file)
    mat = read_srhmat(d / hydro.mat_file)
    logger.info("Read SRH-2D case %s: %d cells, %d nodes, %d materials",
                srhhydro_path.name, len(geom.cells), len(geom.nodes), mat.n_materials)
    return SRHCase(hydro, geom, mat, d)
