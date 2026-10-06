"""The NN resistance closure is a local law n_i = f(h_i) (default), unlike Julia's whole-array
LayerNorm.

hydrograd's default MLP closure for Manning's n has no LayerNorm across cells
(``ln_axes="features"``), so the n of a cell depends only on that cell's depth h (m). Julia's
Lux model applies LayerNorm over the whole cell array (``ln_axes="all"``), which couples all
cells. The tests check:

- locality: changing h in other cells leaves n unchanged (bit-identical), equal depths give
  equal n, and the Jacobian dn_i/dh_j is diagonal;
- non-locality of the Julia option: the same change moves n by more than 1e-6;
- (slow) the default UDE learns the sigmoid n(h) of the bump example: the loss drops by
  more than a factor 10 in 12 iterations.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from conftest import requires_jl  # noqa: F401

import hydrograd  # noqa: F401
from hydrograd.closures.mlp import apply_mlp, init_mlp


def _n(h: jnp.ndarray, ln_axes: str) -> jnp.ndarray:
    """Manning's n from a fixed random MLP (1 -> [3, 3] -> 1, tanh) at depths ``h``.

    The depth is scaled from [0.1, 0.5] m to [-1, 1]; the output is mapped to
    n in [0.03, 0.06].

    Args:
        h: water depths (m), shape [n_cells].
        ln_axes: LayerNorm axes, ``"features"`` (local) or ``"all"`` (Julia, non-local).

    Returns:
        Manning's n (s/m^(1/3)), shape [n_cells].
    """
    params = init_mlp(jax.random.PRNGKey(0), 1, [3, 3], 1)
    x = (2.0 * (h - 0.1) / 0.4 - 1.0)[:, None]
    return apply_mlp(params, x, ("tanh", "tanh"), 0.03, 0.06, ln_axes)[:, 0]


def test_default_closure_is_local() -> None:
    """With ``ln_axes="features"``, n_i depends on h_i only (exact checks)."""
    # Change depths in all cells but the first 10; their n must not change.
    h = jnp.linspace(0.1, 0.5, 50)
    h2 = h.at[10:].set(h[10:] + 0.2)          # change depths in all other cells
    n1 = _n(h, "features")
    n2 = _n(h2, "features")
    np.testing.assert_array_equal(np.asarray(n1[:10]), np.asarray(n2[:10]))

    # Same depth -> same n anywhere in the domain.
    n3 = _n(jnp.full(5, 0.3), "features")
    assert float(jnp.ptp(n3)) == 0.0

    # Jacobian dn_i/dh_j is diagonal.
    def n_local(depths: jnp.ndarray) -> jnp.ndarray:
        """Local-closure n as a function of the depths only."""
        return _n(depths, "features")

    J = jax.jacfwd(n_local)(h[:6])
    assert np.allclose(np.asarray(J) - np.diag(np.diag(np.asarray(J))), 0.0)


def test_julia_whole_array_layernorm_is_nonlocal() -> None:
    """With ``ln_axes="all"``, changing other cells moves n of the first 10 by > 1e-6."""
    h = jnp.linspace(0.1, 0.5, 50)
    h2 = h.at[10:].set(h[10:] + 0.2)
    n_change = jnp.abs(_n(h, "all")[:10] - _n(h2, "all")[:10]).max()
    assert float(n_change) > 1e-6


@pytest.mark.slow
def test_default_ude_trains_local_law(case_copy) -> None:
    """Default UDE closure (local, no LayerNorm) learns the sigmoid n(h) of the bump example.

    Passes if the final loss of 12 iterations is below 0.1 times the initial loss.
    """
    import json
    import re

    from hydrograd.apps.setup import setup_case
    from hydrograd.apps.ude import run_ude
    from hydrograd.io.control import read_control

    # Copy the case and set 12 training iterations (trailing commas removed first).
    case_dir = case_copy("UDE/ManningN/oneD_channel_with_bump/training")
    ctrl_path = case_dir / "run_control.json"
    ctrl = json.loads(re.sub(r",(\s*[}\]])", r"\1", ctrl_path.read_text()))
    ctrl["UDE_options"]["UDE_max_iterations"] = [12, 0]
    ctrl_path.write_text(json.dumps(ctrl))

    # The default closure must be local.
    case = setup_case(read_control(ctrl_path))
    assert case.model.resistance.ln_axes == "features"

    # Train; entry 1 of each history record is the total loss.
    losses = [record[1] for record in run_ude(case)["history"]]
    assert losses[-1] < 0.1 * losses[0]   # ~0.055 after 12 iterations, ~1e-3 after 30
