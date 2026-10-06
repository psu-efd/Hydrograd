"""Round-off parity of the semi-discrete residual with Hydrograd.jl's ``swe_2d_rhs``.

Reference data ``tests/julia_ref/<case>.json`` are produced on Roar by
``hpc/roar/julia_dump.sbatch`` (script ``tests/julia_ref/dump_rhs.jl``): Julia
evaluates its RHS at the IC, a perturbed state and a state with dry cells.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import julia_compat, requires_jl

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.physics.rhs import rhs

pytestmark = requires_jl

REF = Path(__file__).parent / "julia_ref"
CASES = {
    "forward_simulation_oneD_channel_with_bump": "forward_simulation/oneD_channel_with_bump",
    "forward_simulation_oneD_channel_with_bump_ManningN_h": "forward_simulation/oneD_channel_with_bump_ManningN_h",
    "forward_simulation_Savannah_River": "forward_simulation/Savannah_River",
    "forward_simulation_Savannah_River_ManningN_ks_h_Umag": "forward_simulation/Savannah_River_ManningN_ks_h_Umag",
    "UDE_ManningN_oneD_channel_with_bump_training": "UDE/ManningN/oneD_channel_with_bump/training",
}


def julia_nn_params(flat: np.ndarray, hidden: list[int], in_dim: int, out_dim: int) -> list[dict]:
    """Lux Chain(Dense, LayerNorm, ..., Dense) flat ComponentArray -> our MLP pytree.

    Lux stores Dense weights as (out, in) column-major; LayerNorm scale/bias per feature.
    """
    params, k = [], 0
    sizes = [in_dim, *hidden, out_dim]
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        W = flat[k:k + a * b].reshape((b, a), order="F"); k += a * b
        bias = flat[k:k + b]; k += b
        layer = {"W": jnp.asarray(W), "b": jnp.asarray(bias)}
        if i < len(hidden):  # Lux LayerNorm parameters are (bias, scale)
            layer["ln_bias"] = jnp.asarray(flat[k:k + b]); k += b
            layer["ln_scale"] = jnp.asarray(flat[k:k + b]); k += b
        params.append(layer)
    assert k == len(flat), (k, len(flat))
    return params


@pytest.mark.parametrize("name", sorted(CASES))
def test_rhs_matches_julia(case_copy, name: str) -> None:
    ref_file = REF / f"{name}.json"
    if not ref_file.exists():
        pytest.skip(f"no Julia dump {ref_file.name}")
    ref = json.loads(ref_file.read_text())
    d = case_copy(CASES[name])
    ctrl_path = d / "run_control.json"
    ctrl_path.write_text(re.sub(r'"Savana_SI.srhhydro"', '"savana_SI.srhhydro"', ctrl_path.read_text()))
    case = julia_compat(setup_case(read_control(ctrl_path)))
    m = case.model
    np.testing.assert_allclose(np.asarray(m.hstill), ref["hstill"], atol=1e-12)
    theta = {}
    if "nn_params_flat" in ref:
        nn = case.ctrl.opt("UDE_NN_config")
        theta["nn"] = julia_nn_params(np.asarray(ref["nn_params_flat"]), [int(w) for w in nn["hidden_layers"]],
                                      int(nn["input_dim"]), int(nn["output_dim"]))
    for k, (q, r_jl) in enumerate(zip(ref["Q"], ref["rhs"])):
        U = jnp.asarray(np.asarray(q).reshape(3, -1))
        r_py = np.asarray(rhs(m, theta, U)).reshape(-1)
        r_jl = np.asarray(r_jl)
        scale = np.abs(r_jl).max()
        err = np.abs(r_py - r_jl).max()
        assert err <= 1e-9 * max(scale, 1.0), f"state {k}: max|diff| {err:.3e} (max|rhs| {scale:.3e})"


def test_inlet_split_conserves_Q_and_follows_conveyance(case_copy) -> None:
    """Corrected split: sum of face inflows = Q and shares proportional to L h^(5/3)/n."""
    from hydrograd.physics.model import KIND_INLET
    from hydrograd.physics.rhs import _ghosts, clipped_state, manning_n_cells

    d = case_copy(CASES["forward_simulation_Savannah_River"])
    case = setup_case(read_control(d / "run_control.json"))
    m = case.model
    assert m.inlet_split == "conveyance"
    xi, h, qx, qy = clipped_state(m, case.U0)
    n = manning_n_cells(m, {}, h, qx, qy)
    hg, qxg, qyg = _ghosts(m, {}, m.zb, h, qx, qy, n)
    sel = np.asarray(m.fb_kind) == KIND_INLET
    L, nx, ny = (np.asarray(a)[sel] for a in (m.fb_L, m.fb_nx, m.fb_ny))
    flux_in = -(np.asarray(qxg)[sel] * nx + np.asarray(qyg)[sel] * ny) * L        # m^3/s per face
    np.testing.assert_allclose(flux_in.sum(), float(m.Q[0]), rtol=1e-12)
    hc, nc = np.asarray(h)[np.asarray(m.fb_c)[sel]], np.asarray(n)[np.asarray(m.fb_c)[sel]]
    w = L * hc ** (5 / 3) / nc
    np.testing.assert_allclose(flux_in / flux_in.sum(), w / w.sum(), rtol=1e-12)
