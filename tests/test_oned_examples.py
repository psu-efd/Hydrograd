"""Tests of the 1-D example cases (``examples/oned``).

1. The case and run-control files in the repository are those ``build_cases.py`` writes
   (so the examples cannot go stale when the builder or the case format changes).
2. The forward example of the reservoir runs through the command-line entry point and
   builds the expected delta: a bed rise of 6 to 8 m in the pool, no change on the
   non-erodible river reach beyond its active layer, and a closed volume balance.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

from hydrograd.cli import run as run_control_file

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "oned"

# Files of each example folder that the builder writes.
BUILT_FILES = {
    "reservoir_delta": ("case.json", "run_control.json"),
    "reservoir_twin": (
        "case.json",
        "oned_survey_truth.json",
        "run_control.json",
        "run_control_sensitivity.json",
    ),
    "confluence": ("case.json", "run_control.json"),
}


def _load_builder():
    """Import ``examples/oned/build_cases.py`` as a module.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("build_cases", EXAMPLES / "build_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numbers_close(first, second) -> bool:
    """Compare two decoded JSON values, numbers to a relative tolerance of 1e-9.

    Args:
        first, second: nested lists, dictionaries, numbers, strings, booleans or None.

    Returns:
        True if the two have the same structure and equal values.
    """
    if isinstance(first, dict):
        if not isinstance(second, dict) or set(first) != set(second):
            return False
        return all(_numbers_close(first[key], second[key]) for key in first)

    if isinstance(first, list):
        if not isinstance(second, list) or len(first) != len(second):
            return False
        if first and all(isinstance(value, (int, float)) for value in first):
            return bool(np.allclose(first, second, rtol=1e-9, atol=1e-12))
        return all(_numbers_close(a, b) for a, b in zip(first, second, strict=True))

    if isinstance(first, float) and isinstance(second, (int, float)):
        return bool(np.isclose(first, second, rtol=1e-9, atol=1e-12))

    return first == second


def test_example_files_are_up_to_date(tmp_path: Path):
    """The files in the repository equal a fresh build."""
    builder = _load_builder()
    builder.build_all(tmp_path)

    for folder, names in BUILT_FILES.items():
        for name in names:
            fresh = json.loads((tmp_path / folder / name).read_text())
            stored = json.loads((EXAMPLES / folder / name).read_text())
            assert _numbers_close(fresh, stored), f"{folder}/{name} differs from a fresh build"


def test_reservoir_delta_example_runs(tmp_path: Path):
    """The forward example builds a delta in the pool and conserves sediment volume."""
    case_dir = tmp_path / "reservoir_delta"
    shutil.copytree(EXAMPLES / "reservoir_delta", case_dir)
    run_control_file(case_dir / "run_control.json")

    summary = json.loads((case_dir / "oned_forward_results.json").read_text())
    rise = np.asarray(summary["thalweg_final"]) - np.asarray(summary["thalweg_initial"])

    # Delta in the pool; the river reach (first 5 sections, non-erodible) barely changes.
    assert 6.0 < rise.max() < 8.0
    assert 20 <= int(rise.argmax()) <= 35
    assert np.all(rise[:5] > -0.15)
    assert 0.99 < summary["trap_efficiency"][-1] < 1.0

    # Two surveys were written; the second holds more deposit than the first.
    truth = json.loads((case_dir / "oned_survey_truth.json").read_text())
    volumes = np.asarray(truth["deposit_volume"])
    assert volumes.shape == (2, 41)
    assert volumes[1].sum() > volumes[0].sum() > 0.0

    # Volume balance from the stored arrays: inflow - outflow = deposition.
    arrays = np.load(case_dir / "oned_forward_results.npz")
    step = np.diff(np.concatenate([[0.0], arrays["time_end"]]))
    inflow = np.sum(arrays["load_in"].sum(axis=(1, 2)) * step)
    outflow = np.sum(arrays["load_out"].sum(axis=1) * step)
    np.testing.assert_allclose(arrays["deposition"].sum(), inflow - outflow, rtol=1e-8)
