"""Shared fixtures: locate Hydrograd.jl examples and copy cases to a temp dir."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import hydrograd  # noqa: F401  (enables float64)

_CANDIDATES = [os.environ.get("HYDROGRAD_JL", ""),
               str(Path(__file__).resolve().parents[2] / "Hydrograd.jl"),
               str(Path.home() / "research" / "Hydrograd.jl")]
JL = next((Path(p) for p in _CANDIDATES if p and (Path(p) / "examples").is_dir()), None)
EXAMPLES = JL / "examples" / "SWE_2D" if JL else None

requires_jl = pytest.mark.skipif(JL is None, reason="Hydrograd.jl examples not found (set HYDROGRAD_JL)")


@pytest.fixture
def case_copy(tmp_path: Path):
    """Copy an example (relative to examples/SWE_2D) into tmp_path; never write into Hydrograd.jl."""
    def _copy(rel: str) -> Path:
        dst = tmp_path / Path(rel).name
        shutil.copytree(EXAMPLES / rel, dst, ignore=shutil.ignore_patterns("*.gif", "*.h5", "SRH_2D"))
        return dst
    return _copy


def julia_compat(case):
    """Case with Hydrograd.jl's xi-form Roe scheme, inlet-Q split (L^(5/3) h/n),
    whole-array (non-local) NN LayerNorm and loss velocities q/(h + eps),
    for comparisons with Julia data."""
    import dataclasses
    m = case.model
    from hydrograd.apps.losses import EPS_S
    m = dataclasses.replace(m, inlet_split="julia", loss_velocity_eps=EPS_S, scheme="julia_xi_roe", order=1,
                            resistance=dataclasses.replace(m.resistance, ln_axes="all"))
    return dataclasses.replace(case, model=m)
