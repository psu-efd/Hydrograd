"""Result writers: legacy VTK (ParaView), JSON (Hydrograd.jl-compatible), CSV.

Legacy VTK layout
-----------------
:func:`write_vtk` writes an ``UNSTRUCTURED_GRID`` in the legacy VTK format (version 3.0):

    # vtk DataFile Version 3.0
    <title>
    ASCII | BINARY
    DATASET UNSTRUCTURED_GRID
    POINTS <n_nodes> double            x y z_bed per node (m)
    CELLS <n_cells> <size>             per cell: <k> <node 0> ... <node k-1>
    CELL_TYPES <n_cells>               7 (VTK_POLYGON) for every cell
    CELL_DATA <n_cells>
    SCALARS <name> double 1            one value per cell
    LOOKUP_TABLE default
    VECTORS <name> double              (x, y, 0) per cell

In binary mode the data are big-endian float64 (``>f8``) and int32 (``>i4``), as the
legacy format requires; each data block is followed by a newline.

JSON and CSV
------------
:func:`write_json` flattens multi-dimensional arrays in column-major (Fortran) order,
like Julia, so the files can be compared with Hydrograd.jl output. :func:`write_csv`
writes a header line and rows of numbers with 15 significant digits.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hydrograd.mesh import Mesh

logger = logging.getLogger(__name__)

__all__ = ["write_csv", "write_json", "write_vtk"]

# VTK cell type of a general polygon.
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

    # Connectivity: for each cell, its node count followed by its node indices.
    counts = np.fromiter((len(c) for c in mesh.cell_nodes), dtype=np.int64, count=mesh.n_cells)
    cell_records = []
    for c in mesh.cell_nodes:
        cell_records.append(np.concatenate([[len(c)], c]))
    conn = np.concatenate(cell_records).astype(np.int64)

    # The file is assembled from byte blocks and written once at the end.
    blocks = []

    def header(text: str) -> None:
        """Append one text line to the file.

        Args:
            text: the line, without the newline.
        """
        blocks.append((text + "\n").encode())

    def data(arr: np.ndarray, fmt: str, kind: str) -> None:
        """Append a data block, in binary or ASCII according to ``binary``.

        Args:
            arr: values, shape [n] or [n, k]; ASCII writes one row per line.
            fmt: printf-style format of one value in ASCII mode (e.g. ``"%.15g"``).
            kind: ``"f"`` for float64 data, anything else for int32 data (binary mode).
        """
        if binary:
            # Big-endian float64 or int32, followed by a newline.
            dtype = ">f8" if kind == "f" else ">i4"
            blocks.append(np.ascontiguousarray(arr, dtype=dtype).tobytes())
            blocks.append(b"\n")
        else:
            # One line per row, values separated by spaces.
            rows = arr.reshape(len(arr), -1) if arr.ndim > 1 else arr[:, None]
            lines = []
            for r in rows:
                lines.append(" ".join(fmt % x for x in r))
            blocks.append(("\n".join(lines) + "\n").encode())

    # File header.
    header("# vtk DataFile Version 3.0")
    header(title[:250])
    header("BINARY" if binary else "ASCII")
    header("DATASET UNSTRUCTURED_GRID")

    # Nodes (x, y, z_bed).
    header(f"POINTS {len(mesh.nodes)} double")
    data(np.asarray(mesh.nodes, dtype=float), "%.12g", "f")

    # Cells: connectivity and cell types (all polygons).
    header(f"CELLS {mesh.n_cells} {int(counts.sum() + mesh.n_cells)}")
    data(conn, "%d", "i")
    header(f"CELL_TYPES {mesh.n_cells}")
    data(np.full(mesh.n_cells, VTK_POLYGON), "%d", "i")

    # Cell data: scalars, then vectors padded with z = 0.
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
    """Write a dict of arrays/scalars as JSON (arrays flattened column-major like Julia).

    Args:
        path: output ``.json`` file.
        data: name -> array (NumPy, JAX, ...), NumPy scalar, or plain JSON value.
    """
    def conv(v: object) -> object:
        """Convert one value to a JSON-serialisable object.

        Args:
            v: an array-like, a NumPy scalar, or a plain Python value.

        Returns:
            A list (arrays; multi-dimensional ones flattened in column-major order),
            a Python scalar (NumPy scalars), or ``v`` unchanged.
        """
        if isinstance(v, np.ndarray) or hasattr(v, "__array__"):
            a = np.asarray(v)
            return a.ravel(order="F").tolist() if a.ndim > 1 else a.tolist()

        if isinstance(v, (np.floating, np.integer)):
            return v.item()

        return v

    converted = {k: conv(v) for k, v in data.items()}
    Path(path).write_text(json.dumps(converted))
    logger.info("wrote %s", path)


def write_csv(path: str | Path, header: Sequence[str], rows: Sequence[Sequence[float]]) -> None:
    """Write a simple CSV table.

    Args:
        path: output ``.csv`` file.
        header: column names.
        rows: table rows; each value is written with 15 significant digits.
    """
    out = [",".join(header)]
    for r in rows:
        out.append(",".join(f"{x:.15g}" for x in r))

    Path(path).write_text("\n".join(out) + "\n")


def cell_fields(scalars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop non-finite / wrong-length entries before writing (defensive).

    Args:
        scalars: name -> per-cell array.

    Returns:
        name -> float array with NaN replaced by 0 and +/-inf by the largest finite
        float (``np.nan_to_num``).
    """
    return {k: np.nan_to_num(np.asarray(v, dtype=float)) for k, v in scalars.items()}
