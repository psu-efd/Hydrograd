"""The NN resistance closure is a local law n_i = f(h_i) (default), unlike Julia's whole-array LayerNorm."""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from conftest import requires_jl  # noqa: F401

import hydrograd  # noqa: F401
from hydrograd.closures.mlp import apply_mlp, init_mlp


def _n(h: jnp.ndarray, ln_axes: str) -> jnp.ndarray:
    params = init_mlp(jax.random.PRNGKey(0), 1, [3, 3], 1)
    x = (2.0 * (h - 0.1) / 0.4 - 1.0)[:, None]
    return apply_mlp(params, x, ("tanh", "tanh"), 0.03, 0.06, ln_axes)[:, 0]


def test_default_closure_is_local() -> None:
    h = jnp.linspace(0.1, 0.5, 50)
    h2 = h.at[10:].set(h[10:] + 0.2)          # change depths in all other cells
    n1, n2 = _n(h, "features"), _n(h2, "features")
    np.testing.assert_array_equal(np.asarray(n1[:10]), np.asarray(n2[:10]))
    # Same depth -> same n anywhere in the domain.
    n3 = _n(jnp.full(5, 0.3), "features")
    assert float(jnp.ptp(n3)) == 0.0
    # Jacobian dn_i/dh_j is diagonal.
    J = jax.jacfwd(lambda hh: _n(hh, "features"))(h[:6])
    assert np.allclose(np.asarray(J) - np.diag(np.diag(np.asarray(J))), 0.0)


def test_julia_whole_array_layernorm_is_nonlocal() -> None:
    h = jnp.linspace(0.1, 0.5, 50)
    h2 = h.at[10:].set(h[10:] + 0.2)
    assert float(jnp.abs(_n(h, "all")[:10] - _n(h2, "all")[:10]).max()) > 1e-6


@pytest.mark.slow
def test_default_ude_trains_local_law(case_copy) -> None:
    """Default UDE closure (local, no LayerNorm) learns the sigmoid n(h) of the bump example."""
    import json
    import re

    from hydrograd.apps.setup import setup_case
    from hydrograd.apps.ude import run_ude
    from hydrograd.io.control import read_control

    d = case_copy("UDE/ManningN/oneD_channel_with_bump/training")
    c = json.loads(re.sub(r",(\s*[}\]])", r"\1", (d / "run_control.json").read_text()))
    c["UDE_options"]["UDE_max_iterations"] = [12, 0]
    (d / "run_control.json").write_text(json.dumps(c))
    case = setup_case(read_control(d / "run_control.json"))
    assert case.model.resistance.ln_axes == "features"
    hist = [h[1] for h in run_ude(case)["history"]]
    assert hist[-1] < 0.1 * hist[0]   # ~0.055 after 12 iterations, ~1e-3 after 30
