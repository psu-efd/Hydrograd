"""Round-off parity of the semi-discrete residual with Hydrograd.jl's ``swe_2d_rhs``.

Reference data ``tests/julia_ref/<case>.json`` are produced on Roar by
``hpc/roar/julia_dump.sbatch`` (script ``tests/julia_ref/dump_rhs.jl``): Julia
evaluates its RHS at the IC, a perturbed state and a state with dry cells.

hydrograd evaluates its RHS dU/dt (U = [xi, q_x, q_y]; units m/s and m^2/s^2) at the same
states with the Julia-compatible numerics (``julia_compat``). Both codes run the same
algorithm in float64, so the maximum difference must be at round-off level:
max|diff| <= 1e-9 * max(max|rhs_julia|, 1).

A second test checks hydrograd's own (corrected) inlet-Q split, which is not in Julia: the
face inflows sum to Q to 1e-12 and are shared in proportion to the conveyance L h^(5/3)/n.
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

# Folder with the Julia RHS dumps (git-ignored; exists only on Roar).
REF = Path(__file__).parent / "julia_ref"

# Dump name (file stem in REF) -> case path relative to examples/SWE_2D.
CASES = {
    "forward_simulation_oneD_channel_with_bump": "forward_simulation/oneD_channel_with_bump",
    "forward_simulation_oneD_channel_with_bump_ManningN_h": (
        "forward_simulation/oneD_channel_with_bump_ManningN_h"
    ),
    "forward_simulation_Savannah_River": "forward_simulation/Savannah_River",
    "forward_simulation_Savannah_River_ManningN_ks_h_Umag": (
        "forward_simulation/Savannah_River_ManningN_ks_h_Umag"
    ),
    "UDE_ManningN_oneD_channel_with_bump_training": "UDE/ManningN/oneD_channel_with_bump/training",
}


def julia_nn_params(flat: np.ndarray, hidden: list[int], in_dim: int, out_dim: int) -> list[dict]:
    """Lux Chain(Dense, LayerNorm, ..., Dense) flat ComponentArray -> our MLP pytree.

    Lux stores Dense weights as (out, in) column-major; LayerNorm scale/bias per feature.
    The flat vector is laid out layer by layer as [W, b] for each Dense layer, followed by
    [ln_bias, ln_scale] after every hidden layer.

    Args:
        flat: all Lux parameters as one vector, shape [n_params].
        hidden: widths of the hidden layers.
        in_dim: number of network inputs.
        out_dim: number of network outputs.

    Returns:
        One dict per Dense layer with ``W`` [out, in] and ``b`` [out], plus ``ln_bias`` and
        ``ln_scale`` [out] for hidden layers.
    """
    params = []
    offset = 0
    sizes = [in_dim, *hidden, out_dim]

    for layer_index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        # Dense weights (column-major, shape [out, in]) and bias.
        weights = flat[offset:offset + n_in * n_out].reshape((n_out, n_in), order="F")
        offset += n_in * n_out
        bias = flat[offset:offset + n_out]
        offset += n_out
        layer = {"W": jnp.asarray(weights), "b": jnp.asarray(bias)}

        # Hidden layers are followed by a LayerNorm; Lux stores (bias, scale).
        if layer_index < len(hidden):
            layer["ln_bias"] = jnp.asarray(flat[offset:offset + n_out])
            offset += n_out
            layer["ln_scale"] = jnp.asarray(flat[offset:offset + n_out])
            offset += n_out
        params.append(layer)

    # Every entry of the flat vector must have been used.
    assert offset == len(flat), (offset, len(flat))
    return params


@pytest.mark.parametrize("name", sorted(CASES))
def test_rhs_matches_julia(case_copy, name: str) -> None:
    """hydrograd's RHS equals Julia's at every dumped state to round-off.

    Compares h_still (m) to 1e-12 absolute, then the RHS vector at each dumped state with
    max|diff| <= 1e-9 * max(max|rhs_julia|, 1). Skips if the Julia dump is missing.
    """
    # Load the Julia dump, or skip if it has not been generated.
    ref_file = REF / f"{name}.json"
    if not ref_file.exists():
        pytest.skip(f"no Julia dump {ref_file.name}")
    ref = json.loads(ref_file.read_text())

    # Copy the case; fix the Savannah hydro-file name (the file on disk is lower case).
    case_dir = case_copy(CASES[name])
    ctrl_path = case_dir / "run_control.json"
    ctrl_text = re.sub(r'"Savana_SI.srhhydro"', '"savana_SI.srhhydro"', ctrl_path.read_text())
    ctrl_path.write_text(ctrl_text)

    # Set up with Julia-compatible numerics and check the still-water depth.
    case = julia_compat(setup_case(read_control(ctrl_path)))
    model = case.model
    np.testing.assert_allclose(np.asarray(model.hstill), ref["hstill"], atol=1e-12)

    # UDE cases: load Julia's NN weights so both codes use the same closure.
    theta = {}
    if "nn_params_flat" in ref:
        nn_config = case.ctrl.opt("UDE_NN_config")
        hidden_widths = [int(width) for width in nn_config["hidden_layers"]]
        theta["nn"] = julia_nn_params(
            np.asarray(ref["nn_params_flat"]),
            hidden_widths,
            int(nn_config["input_dim"]),
            int(nn_config["output_dim"]),
        )

    # Compare the RHS at each dumped state (Julia stores Q = [xi; q_x; q_y] flat).
    for state_index, (q_flat, rhs_julia) in enumerate(zip(ref["Q"], ref["rhs"])):
        state = jnp.asarray(np.asarray(q_flat).reshape(3, -1))
        rhs_python = np.asarray(rhs(model, theta, state)).reshape(-1)
        rhs_julia = np.asarray(rhs_julia)
        scale = np.abs(rhs_julia).max()
        err = np.abs(rhs_python - rhs_julia).max()
        assert err <= 1e-9 * max(scale, 1.0), (
            f"state {state_index}: max|diff| {err:.3e} (max|rhs| {scale:.3e})"
        )


def test_inlet_split_conserves_Q_and_follows_conveyance(case_copy) -> None:
    """Corrected split: sum of face inflows = Q and shares proportional to L h^(5/3)/n.

    Both checks use relative tolerance 1e-12 (exact up to round-off).
    """
    from hydrograd.physics.model import KIND_INLET
    from hydrograd.physics.rhs import _ghosts, clipped_state, manning_n_cells

    # Savannah case with hydrograd's default (conveyance) inlet split.
    case_dir = case_copy(CASES["forward_simulation_Savannah_River"])
    case = setup_case(read_control(case_dir / "run_control.json"))
    model = case.model
    assert model.inlet_split == "conveyance"

    # Ghost-cell states at the initial condition; the inlet ghosts carry the split.
    xi, h, qx, qy = clipped_state(model, case.U0)
    manning_n = manning_n_cells(model, {}, h, qx, qy)
    h_ghost, qx_ghost, qy_ghost = _ghosts(model, {}, model.zb, h, qx, qy, manning_n)

    # Geometry of the inlet faces: length L (m) and outward unit normal (nx, ny).
    is_inlet = np.asarray(model.fb_kind) == KIND_INLET
    face_length = np.asarray(model.fb_L)[is_inlet]
    nx = np.asarray(model.fb_nx)[is_inlet]
    ny = np.asarray(model.fb_ny)[is_inlet]

    # Inflow per face (m^3/s): minus the outward normal discharge times face length.
    normal_q = np.asarray(qx_ghost)[is_inlet] * nx + np.asarray(qy_ghost)[is_inlet] * ny
    flux_in = -normal_q * face_length
    np.testing.assert_allclose(flux_in.sum(), float(model.Q[0]), rtol=1e-12)

    # Shares must follow the conveyance L h^(5/3)/n of the inlet cells.
    inlet_cells = np.asarray(model.fb_c)[is_inlet]
    h_cell = np.asarray(h)[inlet_cells]
    n_cell = np.asarray(manning_n)[inlet_cells]
    conveyance = face_length * h_cell ** (5 / 3) / n_cell
    np.testing.assert_allclose(flux_in / flux_in.sum(), conveyance / conveyance.sum(), rtol=1e-12)
