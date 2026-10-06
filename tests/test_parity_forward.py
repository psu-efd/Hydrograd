"""Parity with Hydrograd.jl: mesh/bed geometry and forward simulations.

Each case is a Hydrograd.jl forward-simulation example. hydrograd runs it with the
Julia-compatible numerics (``julia_compat``: xi-form Roe scheme, Julia inlet split) and the
result is compared with Julia's ``forward_simulation_solution_truth.json``:

- static fields (cell bed elevation zb (m) and bed slopes S0 (-)) must agree to 1e-12,
  i.e. the mesh and bed geometry are read identically;
- dynamic fields come from two independent Tsit5 integrations at rtol 1e-3, so they agree
  only to the integrator tolerance: water-surface elevation (m) within ``wse_tol``,
  x-velocity u (m/s) within 5e-3 * max(max|u|, 1 m/s), Manning's n within 2e-3.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import EXAMPLES, julia_compat, requires_jl

from hydrograd.apps.forward import run_forward
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control

pytestmark = requires_jl

# (case path relative to examples/SWE_2D, absolute WSE tolerance in m)
CASES = [
    ("forward_simulation/oneD_channel_with_bump", 2e-3),
    ("forward_simulation/Savannah_River", 2e-3),
    ("forward_simulation/oneD_channel_with_bump_ManningN_h", 2e-3),
]


@pytest.mark.parametrize("rel,wse_tol", CASES)
def test_forward_matches_julia_truth(case_copy, rel: str, wse_tol: float) -> None:
    """Bed geometry matches Julia exactly; WSE, u and n match Julia's forward run.

    Tolerances: zb and S0 absolute 1e-12; max|WSE diff| < ``wse_tol`` (m);
    max|u diff| < 5e-3 * max(max|u_julia|, 1) (m/s); Manning's n absolute 2e-3.
    """
    # Copy the case and load Julia's truth file. Keep only 1D fields (scalars or flat
    # lists); nested lists (time series of 2D arrays) are not compared.
    case_dir = case_copy(rel)
    truth_path = EXAMPLES / rel / "forward_simulation_solution_truth.json"
    truth_raw = json.loads(truth_path.read_text())
    julia = {
        key: np.asarray(value)
        for key, value in truth_raw.items()
        if not isinstance(value, list) or not value or not isinstance(value[0], list)
    }

    # Set up the case with Julia-compatible numerics.
    case = julia_compat(setup_case(read_control(case_dir / "run_control.json")))

    # Static fields: exact.
    np.testing.assert_allclose(np.asarray(case.model.zb), julia["zb_cell_truth"], atol=1e-12)
    bed_slopes = np.asarray(case.model.S0).ravel(order="F")
    np.testing.assert_allclose(bed_slopes, julia["S0_cells_truth"], atol=1e-12)

    # Run the forward simulation.
    fields = run_forward(case)["fields"]

    # Dynamic fields: same scheme, both with Tsit5 at rtol 1e-3.
    assert np.abs(fields["wse"] - julia["wse_truth"]).max() < wse_tol
    u_max = np.abs(julia["u_truth"]).max()
    assert np.abs(fields["u"] - julia["u_truth"]).max() < 5e-3 * max(u_max, 1.0)
    np.testing.assert_allclose(fields["ManningN"], julia["ManningN_cells_truth"], atol=2e-3)
