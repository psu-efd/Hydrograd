"""Shared pytest fixtures and helpers for the hydrograd test suite.

Most parity tests compare hydrograd against the reference Julia package Hydrograd.jl. They
need its example cases (SRH-2D input files, ``run_control.json`` and Julia truth files),
which are looked up in this order:

1. the directory named by the environment variable ``HYDROGRAD_JL``;
2. ``Hydrograd.jl`` next to this repository;
3. ``~/research/Hydrograd.jl``.

Tests marked with ``requires_jl`` are skipped when none of these holds the examples.

Example cases are always copied to a temporary directory before they are run, because
hydrograd writes its outputs (e.g. ``forward_simulation_solution_truth.json``) into the
case folder and must never overwrite Julia's files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import hydrograd  # noqa: F401  (enables float64)

# Candidate locations of the Hydrograd.jl checkout, in order of preference.
_CANDIDATES = [
    os.environ.get("HYDROGRAD_JL", ""),
    str(Path(__file__).resolve().parents[2] / "Hydrograd.jl"),
    str(Path.home() / "research" / "Hydrograd.jl"),
]

# First candidate that contains an ``examples`` folder, or None if Julia is not available.
JL = next((Path(p) for p in _CANDIDATES if p and (Path(p) / "examples").is_dir()), None)

# Root of the 2D shallow-water examples, relative to which tests name their cases.
EXAMPLES = JL / "examples" / "SWE_2D" if JL else None

# Skip marker for tests that need the Julia examples or reference data.
requires_jl = pytest.mark.skipif(
    JL is None,
    reason="Hydrograd.jl examples not found (set HYDROGRAD_JL)",
)


@pytest.fixture
def case_copy(tmp_path: Path):
    """Return a function that copies a Julia example case into ``tmp_path``.

    The copy skips animations (``*.gif``), HDF5 output (``*.h5``) and the ``SRH_2D``
    reference-run folder, which the tests do not need. Never write into Hydrograd.jl.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A function ``copy_case(rel) -> Path``. ``rel`` is the case path relative to
        ``examples/SWE_2D``; the return value is the copied case folder.
    """

    def copy_case(rel: str) -> Path:
        """Copy ``EXAMPLES / rel`` to ``tmp_path / <case name>`` and return the copy."""
        destination = tmp_path / Path(rel).name
        shutil.copytree(
            EXAMPLES / rel,
            destination,
            ignore=shutil.ignore_patterns("*.gif", "*.h5", "SRH_2D"),
        )
        return destination

    return copy_case


def julia_compat(case):
    """Return a copy of ``case`` configured to reproduce Hydrograd.jl's numerics.

    The Julia-compatible settings are:
    - Julia's xi-form Roe scheme (``scheme="julia_xi_roe"``) at first order;
    - Julia's inlet-Q split, with face shares proportional to L^(5/3) h/n
      (``inlet_split="julia"``);
    - whole-array, non-local NN LayerNorm (``ln_axes="all"``);
    - loss velocities q/(h + eps) with eps = ``EPS_S``.

    Use it for every comparison with Julia data.

    Args:
        case: case object from ``hydrograd.apps.setup.setup_case``.

    Returns:
        The same case with a modified ``model``; the input is not changed.
    """
    import dataclasses

    model = case.model
    from hydrograd.apps.losses import EPS_S

    # Swap in the Julia-compatible scheme, inlet split, loss velocity and LayerNorm.
    julia_resistance = dataclasses.replace(model.resistance, ln_axes="all")
    model = dataclasses.replace(
        model,
        inlet_split="julia",
        loss_velocity_eps=EPS_S,
        scheme="julia_xi_roe",
        order=1,
        resistance=julia_resistance,
    )
    return dataclasses.replace(case, model=model)
