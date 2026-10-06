"""Result writers: legacy VTK (ParaView), JSON (Hydrograd.jl-compatible), CSV."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hydrograd.mesh import Mesh

logger = logging.getLogger(__name__)

__all__ = ["write_csv", "write_json", "write_vtk"]

VTK_POLYGON = 7


BINARY_MIN_CELLS = 20_000  # above this, write binary VTK (smaller, much faster)


def write_vtk(path: str | Path, mesh: Mesh, scalars: Mapping[str, np.ndarray],
              vectors: Mapping[str, np.ndarray] | None = None, title: str = "hydrograd",
              binary: bool | None = None) -> None:
    """Write cell data on the mesh as a legacy VTK unstructured grid (ParaView).

    Args:
        path: output ``.vtk`` file.
        mesh: the mesh (node z is written as the bed elevation).
        scalars: name -> per-cell array [n_cells].
        vectors: name -> per-cell array [n_cells, 2] (written with z = 0).
        title: header line.
        binary: binary (big-endian) or ASCII; default binary above ``BINARY_MIN_CELLS``.
    """
    if binary is None:
        binary = mesh.n_cells >= BINARY_MIN_CELLS
    counts = np.fromiter((len(c) for c in mesh.cell_nodes), dtype=np.int64, count=mesh.n_cells)
    conn = np.concatenate([np.concatenate([[len(c)], c]) for c in mesh.cell_nodes]).astype(np.int64)
    blocks = []

    def header(text: str) -> None:
        blocks.append((text + "\n").encode())

    def data(arr: np.ndarray, fmt: str, kind: str) -> None:
        if binary:
            blocks.append(np.ascontiguousarray(arr, dtype=">f8" if kind == "f" else ">i4").tobytes())
            blocks.append(b"\n")
        else:
            rows = arr.reshape(len(arr), -1) if arr.ndim > 1 else arr[:, None]
            blocks.append(("\n".join(" ".join(fmt % x for x in r) for r in rows) + "\n").encode())

    header("# vtk DataFile Version 3.0")
    header(title[:250])
    header("BINARY" if binary else "ASCII")
    header("DATASET UNSTRUCTURED_GRID")
    header(f"POINTS {len(mesh.nodes)} double")
    data(np.asarray(mesh.nodes, dtype=float), "%.12g", "f")
    header(f"CELLS {mesh.n_cells} {int(counts.sum() + mesh.n_cells)}")
    data(conn, "%d", "i")
    header(f"CELL_TYPES {mesh.n_cells}")
    data(np.full(mesh.n_cells, VTK_POLYGON), "%d", "i")
    header(f"CELL_DATA {mesh.n_cells}")
    for name, v in scalars.items():
        header(f"SCALARS {name} double 1")
        header("LOOKUP_TABLE default")
        data(np.asarray(v, dtype=float), "%.15g", "f")
    for name, v in (vectors or {}).items():
        v = np.asarray(v, dtype=float)
        header(f"VECTORS {name} double")
        data(np.column_stack([v, np.zeros(len(v))]), "%.15g", "f")
    Path(path).write_bytes(b"".join(blocks))


def write_json(path: str | Path, data: Mapping[str, object]) -> None:
    """Write a dict of arrays/scalars as JSON (arrays flattened column-major like Julia)."""
    def conv(v: object) -> object:
        if isinstance(v, np.ndarray) or hasattr(v, "__array__"):
            a = np.asarray(v)
            return a.ravel(order="F").tolist() if a.ndim > 1 else a.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    Path(path).write_text(json.dumps({k: conv(v) for k, v in data.items()}))
    logger.info("wrote %s", path)


def write_csv(path: str | Path, header: Sequence[str], rows: Sequence[Sequence[float]]) -> None:
    """Write a simple CSV table."""
    out = [",".join(header)] + [",".join(f"{x:.15g}" for x in r) for r in rows]
    Path(path).write_text("\n".join(out) + "\n")


def cell_fields(scalars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop non-finite / wrong-length entries before writing (defensive)."""
    return {k: np.nan_to_num(np.asarray(v, dtype=float)) for k, v in scalars.items()}
