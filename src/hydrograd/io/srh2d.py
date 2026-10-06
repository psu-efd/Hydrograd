"""Readers for SRH-2D case files: ``.srhhydro``, ``.srhgeom``, ``.srhmat`` and ``.xys``.

An SRH-2D case consists of three text files plus optional time-series files:

- ``.srhhydro``: run control (simulation time, Manning's n per material zone, boundary
  conditions and their values, names of the mesh and material files);
- ``.srhgeom``: the mesh (node coordinates and bed elevation, cell-to-node lists,
  nodestrings that mark the boundaries);
- ``.srhmat``: material zones (the list of cells that belong to each zone);
- ``.xys``: piecewise-linear time series (an inlet hydrograph or an exit stage series).

Semantics
---------
Semantics follow Hydrograd.jl (``src/utilities/SRH_2D/``) with two deliberate fixes,
both inert for the Julia examples:

- quotes are stripped from string values (Julia keeps them, which makes its
  ``GridUnit`` check fail);
- ``SimTime`` is read in SRH order ``t_start (h), dt (s), t_end (h)``.

Units
-----
Lengths are in the unit of the case (``GridUnit``: m or ft); no conversion is done here.
Times in ``.xys`` files are in hours and are converted to seconds on reading.

Indexing
--------
All IDs in the files are 1-based. Boundary (nodestring) IDs and material IDs stay
1-based here (dictionary keys); node and cell IDs are converted to 0-based array
indices, which :mod:`hydrograd.mesh` uses directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Keywords that start a new record in a .srhgeom file. Any of them ends a nodestring
# whose node list continues over several lines.
_GEOM_KEYS = ("Elem", "Node", "NodeString", "SRHGEOM", "Name", "GridUnit")


def _strip(s: str) -> str:
    """Remove surrounding whitespace and double quotes from a string value.

    Args:
        s: raw token, e.g. ``'"channel.srhgeom"'``.

    Returns:
        The token without surrounding whitespace and quotes, e.g. ``channel.srhgeom``.
    """
    return s.strip().strip('"').strip()


@dataclass(frozen=True)
class TimeSeries:
    """Piecewise-linear boundary time series (e.g. an inlet hydrograph).

    Attributes:
        t: times (s), increasing, shape [n_points]; values beyond the ends are held
            constant.
        v: values (m^3/s for discharge, m for WSE), shape [n_points].
        source: file it was read from.
    """

    t: np.ndarray
    v: np.ndarray
    source: str = ""


def read_xys(path: str | Path) -> TimeSeries:
    """Read an SMS/SRH-2D ``.xys`` series: lines ``t value`` with t in hours.

    Header lines (e.g. ``XY 1 3 0 0 0 0 0 0 "name"``) and anything that is not two
    numbers are skipped. Times are converted to seconds.

    Args:
        path: the ``.xys`` file.

    Returns:
        The time series, with times in seconds.

    Raises:
        ValueError: if the file holds no data line or the times are not increasing.
    """
    times_h: list[float] = []
    values: list[float] = []

    # Keep only lines whose first two tokens are both numbers.
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            time_value = float(parts[0])
            data_value = float(parts[1])
        except ValueError:
            continue

        times_h.append(time_value)
        values.append(data_value)

    # Interpolation needs at least one point and strictly increasing times.
    if len(times_h) < 1 or np.any(np.diff(times_h) <= 0):
        raise ValueError(f"{path}: need increasing times in the .xys series")

    # Hours -> seconds.
    times_s = 3600.0 * np.asarray(times_h)

    return TimeSeries(times_s, np.asarray(values), str(path))


def _value_or_series(token: str, base: Path) -> float | TimeSeries:
    """A number, or a time-series file name (relative to the .srhhydro).

    Args:
        token: a BC value token from the ``.srhhydro`` file, either a number or the
            name of an ``.xys`` file (quotes allowed).
        base: directory of the ``.srhhydro`` file.

    Returns:
        The number (m^3/s or m, in the case units), or the time series read from the file.
    """
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
        grid_file: name of the ``.srhgeom`` file.
        mat_file: name of the ``.srhmat`` file.
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
    """Parse an SRH-2D ``.srhhydro`` file.

    Each non-empty line is ``Keyword value1 value2 ...``. The keywords ``ManningsN``,
    ``BC``, ``IQParams`` and ``EWSParamsC`` may appear several times (once per zone or
    boundary) and are collected into dictionaries; every other keyword is stored in
    ``raw`` (the last occurrence wins).

    Args:
        path: the ``.srhhydro`` file.

    Returns:
        The parsed control file.

    Raises:
        KeyError: if the ``Grid`` or ``HydroMat`` keyword is missing.
    """
    path = Path(path)
    raw: dict[str, list[str]] = {}
    manning: dict[int, float] = {}
    bc: dict[int, str] = {}
    monitoring: list[int] = []
    iq: dict[int, tuple[float | TimeSeries, str, str]] = {}
    ews: dict[int, tuple[float | TimeSeries, str, str]] = {}

    for line in path.read_text().splitlines():
        # Skip blank lines and keywords without a value.
        parts = line.split()
        if len(parts) <= 1:
            continue

        key = parts[0]
        vals = parts[1:]

        if key == "ManningsN":
            # ManningsN <material ID> <n>
            manning[int(vals[0])] = float(vals[1])

        elif key == "BC":
            # BC <nodestring ID> <type>; monitoring lines are not boundaries.
            if vals[1].upper() == "MONITORING":
                monitoring.append(int(vals[0]))
            else:
                bc[int(vals[0])] = vals[1].upper()

        elif key == "IQParams":
            # IQParams <ID> <Q or .xys file> <unit> [<distribution type>]
            q_value = _value_or_series(vals[1], path.parent)
            distribution = vals[3] if len(vals) > 3 else ""
            iq[int(vals[0])] = (q_value, vals[2], distribution)

        elif key == "EWSParamsC":
            # EWSParamsC <ID> <WSE or .xys file> <unit> [<type>]
            wse_value = _value_or_series(vals[1], path.parent)
            wse_type = vals[3] if len(vals) > 3 else ""
            ews[int(vals[0])] = (wse_value, vals[2], wse_type)

        else:
            raw[key] = vals

    # SimTime: t_start (h), dt (s), t_end (h).
    sim_time_tokens = raw.get("SimTime", ["0", "1", "0"])[:3]
    sim_time = [float(v) for v in sim_time_tokens]

    # Names of the mesh and material files (may contain spaces and quotes).
    grid_file = _strip(" ".join(raw["Grid"]))
    mat_file = _strip(" ".join(raw["HydroMat"]))

    return SRHHydro(
        path,
        raw,
        manning,
        bc,
        monitoring,
        iq,
        ews,
        grid_file,
        mat_file,
        (sim_time[0], sim_time[1], sim_time[2]),
    )


@dataclass(frozen=True)
class SRHGeom:
    """Parsed ``.srhgeom`` mesh.

    Attributes:
        nodes: node coordinates (x, y, z_bed) (m or ft), shape [n_nodes, 3]; row i is
            node i+1.
        cells: node IDs (0-based) of each cell, in file order (counter-clockwise expected).
        nodestrings: nodestring ID (1-based) -> node IDs (0-based) in file order.
        grid_unit: e.g. ``Meters``/``Feet`` (empty if absent).
    """

    nodes: np.ndarray
    cells: list[np.ndarray]
    nodestrings: dict[int, np.ndarray]
    grid_unit: str


def read_srhgeom(path: str | Path) -> SRHGeom:
    """Parse an SRH-2D ``.srhgeom`` file (IDs must be contiguous 1..N, as in Julia).

    Record types:

        Node <ID> <x> <y> <z_bed>
        Elem <ID> <node ID> <node ID> ...          (3 or more nodes)
        NodeString <ID> <node ID> ...              (may continue on the next lines)
        GridUnit "<Meters|Feet>"

    Args:
        path: the ``.srhgeom`` file.

    Returns:
        The parsed mesh, with node and cell IDs converted to 0-based indices.

    Raises:
        ValueError: if the Node or Elem IDs are not contiguous 1..N.
    """
    node_rows: dict[int, tuple[float, float, float]] = {}
    elem_rows: dict[int, list[int]] = {}
    ns: dict[int, list[int]] = {}
    grid_unit = ""

    # ID of the nodestring being read; its node list can continue on the next lines.
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

        elif current_ns is not None:
            # Continuation line of a nodestring: node IDs only.
            ns[current_ns].extend(int(p) for p in parts)

    # Node and cell IDs must be exactly 1..N so that they map to array rows.
    n_nodes = len(node_rows)
    n_cells = len(elem_rows)
    nodes_contiguous = sorted(node_rows) == list(range(1, n_nodes + 1))
    cells_contiguous = sorted(elem_rows) == list(range(1, n_cells + 1))
    if not nodes_contiguous or not cells_contiguous:
        raise ValueError(f"{path}: Node/Elem IDs must be contiguous 1..N")

    # Arrays in ID order, with node IDs converted to 0-based indices.
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
        """Material ID per cell (0 = default for cells not listed in any zone).

        A cell listed in several zones gets the lowest zone ID (first zone wins, as in
        Julia).

        Args:
            n_cells: number of cells in the mesh.

        Returns:
            Material ID of each cell, shape [n_cells], dtype int64.
        """
        mat = np.zeros(n_cells, dtype=np.int64)
        assigned = np.zeros(n_cells, dtype=bool)

        for mid in sorted(self.zone_cells):
            # Cells of this zone that no lower zone has claimed yet.
            zone = self.zone_cells[mid]
            zone = zone[~assigned[zone]]

            mat[zone] = mid
            assigned[zone] = True

        return mat


def read_srhmat(path: str | Path) -> SRHMat:
    """Parse an SRH-2D ``.srhmat`` file.

    Record types:

        NMaterials <n>
        MatName <ID> <name>
        Material <ID> <cell ID> ...                (may continue on the next lines)

    Args:
        path: the ``.srhmat`` file.

    Returns:
        The parsed material file, with cell IDs converted to 0-based indices.
    """
    n_mat = 0
    names: dict[int, str] = {0: "Default"}
    zones: dict[int, list[int]] = {}

    # ID of the material being read; its cell list can continue on the next lines.
    current = None

    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue

        key = parts[0]

        if key == "NMaterials":
            n_mat = int(parts[1])
            current = None

        elif key == "MatName":
            names[int(parts[1])] = _strip(" ".join(parts[2:]))
            current = None

        elif key == "Material":
            current = int(parts[1])
            zones.setdefault(current, []).extend(int(p) for p in parts[2:])

        elif key == "SRHMAT":
            current = None

        elif current is not None:
            # Continuation line of a material: cell IDs only.
            zones[current].extend(int(p) for p in parts)

    # Cell IDs converted to 0-based indices.
    zone_cells = {k: np.asarray(v, dtype=np.int64) - 1 for k, v in zones.items()}

    return SRHMat(n_mat, names, zone_cells)


@dataclass(frozen=True)
class SRHCase:
    """The three SRH-2D files of one case.

    Attributes:
        hydro: the parsed ``.srhhydro`` file.
        geom: the parsed ``.srhgeom`` file.
        mat: the parsed ``.srhmat`` file.
        case_dir: directory that holds the files.
    """

    hydro: SRHHydro
    geom: SRHGeom
    mat: SRHMat
    case_dir: Path = field(default=Path("."))


def read_srh_case(srhhydro_path: str | Path) -> SRHCase:
    """Read ``.srhhydro`` and the ``.srhgeom``/``.srhmat`` files it references.

    Args:
        srhhydro_path: the ``.srhhydro`` file; the other files are looked up in its
            directory.

    Returns:
        The three parsed files and the case directory.
    """
    srhhydro_path = Path(srhhydro_path)
    case_dir = srhhydro_path.parent

    hydro = read_srhhydro(srhhydro_path)
    geom = read_srhgeom(case_dir / hydro.grid_file)
    mat = read_srhmat(case_dir / hydro.mat_file)

    logger.info(
        "Read SRH-2D case %s: %d cells, %d nodes, %d materials",
        srhhydro_path.name,
        len(geom.cells),
        len(geom.nodes),
        mat.n_materials,
    )

    return SRHCase(hydro, geom, mat, case_dir)
