"""Shared helpers for the Savannah River figures of the Hydrograd v2.0 paper (Figs. 2, 4, 5).

The Savannah River case is the Hydrograd.jl example ``Savannah_River`` (SRH-2D mesh of
1306 cells, five Manning roughness zones, one inlet-Q and one exit-H boundary). This module

- locates the case input files in ``paper/cases/``;
- copies a case into a scratch folder (the case folder itself is never written to) and
  rewrites its ``run_control.json``;
- draws scalar fields on the true unstructured polygon cells with a matplotlib
  ``PolyCollection``;
- records the machine (CPU model, core count) on which timings were measured.

Units are SI throughout: lengths in m, velocities in m/s, Manning's n in s m^(-1/3).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.collections import PolyCollection

# Repository root (paper/ is one level below it) and the paper's data and result folders.
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "paper" / "data"
RESULTS_DIR = REPO / "paper" / "results"

# Input files of the Savannah River cases (paper/cases/, read only; copies of the
# Hydrograd.jl examples forward_simulation/Savannah_River and
# inversion/ManningN_inversion/Savannah_River).
CASES_DIR = REPO / "paper" / "cases"
FORWARD_CASE = CASES_DIR / "savannah_river_steady"
INVERSION_CASE = CASES_DIR / "savannah_river_inversion"


def read_json_with_commas(path: Path) -> dict[str, Any]:
    """Read a JSON file that may contain trailing commas (as some Julia examples do).

    Args:
        path: the JSON file.

    Returns:
        The parsed dictionary.
    """
    text = path.read_text()
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(cleaned)


def copy_case(source: Path, name: str) -> Path:
    """Copy a Hydrograd.jl example into a fresh scratch folder.

    The SRH-2D reference-run folder, animations and VTK files are skipped.

    Args:
        source: the example folder in Hydrograd.jl.
        name: name of the copy (a sub-folder of a new temporary directory).

    Returns:
        Path of the copied case folder.
    """
    scratch = Path(tempfile.mkdtemp(prefix="hydrograd_paper_"))
    target = scratch / name
    ignore = shutil.ignore_patterns("*.vtk", "*.gif", "*.jld2", "SRH_2D")
    shutil.copytree(source, target, ignore=ignore)
    return target


def write_control(case_dir: Path, control: dict[str, Any], name: str = "run_control.json") -> Path:
    """Write a run-control dictionary into a case folder.

    Args:
        case_dir: the case folder.
        control: the run-control dictionary.
        name: file name.

    Returns:
        Path of the written file.
    """
    path = case_dir / name
    path.write_text(json.dumps(control, indent=2))
    return path


def mesh_polygons(mesh: Any) -> list[np.ndarray]:
    """Vertex coordinates of every cell polygon.

    Args:
        mesh: a ``hydrograd.mesh.Mesh``.

    Returns:
        List of n_cells arrays, each of shape [n_vertices, 2] (m).
    """
    polygons = []
    for nodes in mesh.cell_nodes:
        polygons.append(mesh.nodes[nodes, :2])
    return polygons


def poly_field(ax: Any, polygons: list[np.ndarray], values: np.ndarray, edge_width: float = 0.0,
               **kwargs: Any) -> PolyCollection:
    """Draw a cell-centred field on the unstructured polygon cells.

    Args:
        ax: matplotlib axes.
        polygons: cell polygons from ``mesh_polygons``.
        values: one value per cell, shape [n_cells].
        edge_width: width of the cell edges (pt); 0 hides the mesh lines.
        **kwargs: passed to ``PolyCollection`` (cmap, norm, ...).

    Returns:
        The added collection (usable as a colour-bar mappable).
    """
    # Edges in the face colour when hidden, so no hairline gaps appear between cells.
    edge_colour = "0.25" if edge_width > 0 else "face"
    collection = PolyCollection(polygons, array=values, edgecolors=edge_colour,
                                linewidths=max(edge_width, 0.1), **kwargs)
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.autoscale_view()
    return collection


def machine_info() -> dict[str, Any]:
    """CPU model, core counts and Slurm job of the current process (for timing records).

    Returns:
        Dictionary with the CPU model name, the logical cores visible to the process, the
        Slurm job ID and CPUs per task (if any), the host name and the Python version.
    """
    # CPU model from /proc/cpuinfo (Linux).
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break

    # Cores this process may run on (respects Slurm's CPU binding).
    cores = len(os.sched_getaffinity(0))

    return {
        "cpu_model": cpu_model,
        "cores_available": cores,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "host": platform.node(),
        "python": platform.python_version(),
    }


def write_results(name: str, results: dict[str, Any]) -> Path:
    """Write a results dictionary to ``paper/results/<name>.json``.

    Args:
        name: file stem.
        results: JSON-serialisable dictionary.

    Returns:
        Path of the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(results, indent=2))
    return path
