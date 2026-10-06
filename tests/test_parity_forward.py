"""Parity with Hydrograd.jl: mesh/bed geometry and forward simulations."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import EXAMPLES, julia_compat, requires_jl

from hydrograd.apps.forward import run_forward
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control

pytestmark = requires_jl

CASES = [("forward_simulation/oneD_channel_with_bump", 2e-3),
         ("forward_simulation/Savannah_River", 2e-3),
         ("forward_simulation/oneD_channel_with_bump_ManningN_h", 2e-3)]


@pytest.mark.parametrize("rel,wse_tol", CASES)
def test_forward_matches_julia_truth(case_copy, rel: str, wse_tol: float) -> None:
    d = case_copy(rel)
    julia = {k: np.asarray(v) for k, v in json.loads((EXAMPLES / rel / "forward_simulation_solution_truth.json").read_text()).items()
             if not isinstance(v, list) or not v or not isinstance(v[0], list)}
    case = julia_compat(setup_case(read_control(d / "run_control.json")))
    # Static fields: exact.
    np.testing.assert_allclose(np.asarray(case.model.zb), julia["zb_cell_truth"], atol=1e-12)
    np.testing.assert_allclose(np.asarray(case.model.S0).ravel(order="F"), julia["S0_cells_truth"], atol=1e-12)
    out = run_forward(case)["fields"]
    # Dynamic fields: same scheme, both with Tsit5 at rtol 1e-3.
    assert np.abs(out["wse"] - julia["wse_truth"]).max() < wse_tol
    vmax = np.abs(julia["u_truth"]).max()
    assert np.abs(out["u"] - julia["u_truth"]).max() < 5e-3 * max(vmax, 1.0)
    np.testing.assert_allclose(out["ManningN"], julia["ManningN_cells_truth"], atol=2e-3)
