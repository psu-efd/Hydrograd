"""Small MLP closures for the USWE (port of ``src/UDE/process_UDE.jl``).

Architecture
------------
Lux, as in Hydrograd.jl: for each hidden width w_i with activation a_i,
``Dense(in, w_i, a_i) -> LayerNorm(w_i)`` (LayerNorm optional; the UDE app omits it
by default, see ``hydrograd.apps.ude``); then ``Dense(in, out)``; then the
output is mapped into ``[lo, hi]`` by ``lo + (hi - lo) * sigmoid(.)``.

Initialization: Glorot-uniform weights W ~ U(-sqrt(6/(fan_in + fan_out)), +...);
zero biases; LayerNorm affine (scale 1, bias 0), eps = 1e-5.

Inputs are normalized to [-1, 1] with ``2 (x - lo)/(hi - lo) - 1``.

LayerNorm axes
--------------
The closure is a *local* law, n_i = f(h_i): each cell is normalized over its own
hidden units (``ln_axes="features"``, default). Hydrograd.jl uses Lux's default
``dims=Colon()``, which normalizes over the whole array (all cells together), making n
in one cell depend on the depths in all cells; ``ln_axes="all"`` reproduces that for
parity tests only. LayerNorm: z <- (z - mean) / sqrt(var + eps) * scale + bias.

Parameters
----------
Parameters are a plain pytree (list of dicts), so any JAX optimizer applies.
Note: JAX random numbers differ from Julia's StableRNG(1111), so the initial
weights are not bit-identical to Hydrograd.jl.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["ACTIVATIONS", "apply_mlp", "init_mlp", "normalize"]


def _leaky_relu(x: Array) -> Array:
    """Leaky ReLU with negative slope 0.01 (Lux ``leakyrelu`` default).

    Args:
        x: pre-activations, any shape.

    Returns:
        Activations, same shape as ``x``.
    """
    return jax.nn.leaky_relu(x, 0.01)


# Activation name (``run_control.json``) -> function.
ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    "relu": jax.nn.relu,
    "leakyrelu": _leaky_relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "softplus": jax.nn.softplus,
}

# LayerNorm epsilon (added to the variance).
LN_EPS = 1.0e-5


def init_mlp(key: Array, in_dim: int, hidden: Sequence[int], out_dim: int,
             layernorm: bool = True) -> list[dict[str, Array]]:
    """Glorot-uniform weights, zero biases, unit LayerNorm scales (if ``layernorm``).

    Args:
        key: JAX PRNG key.
        in_dim: number of inputs.
        hidden: widths of the hidden layers.
        out_dim: number of outputs.
        layernorm: add LayerNorm scale/bias to each hidden layer.

    Returns:
        One dict per layer with ``W`` [fan_out, fan_in] and ``b`` [fan_out], plus
        ``ln_scale`` and ``ln_bias`` [fan_out] on hidden layers when ``layernorm``.
    """
    params: list[dict[str, Array]] = []
    sizes = [in_dim, *hidden, out_dim]

    for i, (fan_in, fan_out) in enumerate(pairwise(sizes)):
        # Glorot-uniform weights and zero biases.
        key, sub = jax.random.split(key)
        lim = math.sqrt(6.0 / (fan_in + fan_out))
        layer = {
            "W": jax.random.uniform(sub, (fan_out, fan_in), jnp.float64, -lim, lim),
            "b": jnp.zeros(fan_out, jnp.float64),
        }

        # LayerNorm affine parameters on hidden layers only.
        if layernorm and i < len(hidden):
            layer["ln_scale"] = jnp.ones(fan_out, jnp.float64)
            layer["ln_bias"] = jnp.zeros(fan_out, jnp.float64)

        params.append(layer)

    return params


def apply_mlp(params: list[dict[str, Array]], x: Array, activations: Sequence[str],
              out_lo: float, out_hi: float, ln_axes: str = "features") -> Array:
    """Evaluate the bounded MLP on a batch.

    Args:
        params: from :func:`init_mlp`.
        x: normalized inputs, shape [n_samples, in_dim].
        activations: names, one per hidden layer.
        out_lo, out_hi: output bounds (units of the closure output).
        ln_axes: ``features`` (per cell, local law; default) or ``all`` (Julia/Lux, whole array).

    Returns:
        outputs in (out_lo, out_hi), shape [n_samples, out_dim].
    """
    z = x

    # Hidden layers: dense + activation, then optional LayerNorm.
    for layer, act in zip(params[:-1], activations):
        z = ACTIVATIONS[act](z @ layer["W"].T + layer["b"])
        if "ln_scale" not in layer:  # MLP without LayerNorm
            continue

        axes = None if ln_axes == "all" else -1
        mu = z.mean(axis=axes, keepdims=True)
        var = ((z - mu) ** 2).mean(axis=axes, keepdims=True)
        z = (z - mu) / jnp.sqrt(var + LN_EPS) * layer["ln_scale"] + layer["ln_bias"]

    # Linear output layer, then a sigmoid map into (out_lo, out_hi).
    z = z @ params[-1]["W"].T + params[-1]["b"]
    return out_lo + (out_hi - out_lo) * jax.nn.sigmoid(z)


def normalize(x: Array, lo: float, hi: float) -> Array:
    """Map [lo, hi] to [-1, 1]: 2 (x - lo)/(hi - lo) - 1.

    Args:
        x: values (any units), any shape.
        lo, hi: range of x (same units).

    Returns:
        Normalized values (-), same shape as ``x``.
    """
    return 2.0 * (x - lo) / (hi - lo) - 1.0
