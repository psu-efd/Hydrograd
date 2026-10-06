"""Sequential optimizer loop shared by inversion and UDE training.

Mirrors the Julia logic: optimizers listed in ``*_optimizers`` run in order
(``Descent``, ``Adam``, ``LBFGS``) with parallel lists of learning rates and
iteration caps; an optimizer with ``max_iterations <= 0`` is skipped, each
starts from the previous result, and iteration numbers keep counting.

Mapping to optax: ``Descent(lr)`` -> ``sgd(lr)``; ``Adam(lr)`` -> ``adam(lr,
b1=0.9, b2=0.999, eps=1e-8)`` (Optimisers.jl defaults); ``LBFGS`` ->
``optax.lbfgs()`` with its zoom line search (Julia: BackTracking).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import optax

logger = logging.getLogger(__name__)

__all__ = ["OptStage", "run_stages", "stages_from_options"]

LossFn = Callable[[Any], tuple[jax.Array, dict[str, Any]]]


@dataclass(frozen=True)
class OptStage:
    """One optimizer stage: name, learning rate, iteration cap, tolerances."""

    name: str
    lr: float
    max_iter: int
    abs_tol: float = 0.0
    rel_tol: float = 0.0


def stages_from_options(o: Callable[[str], Any], prefix: str) -> list[OptStage]:
    """Read ``<prefix>_optimizers`` and the parallel lists from the run control."""
    names = o(f"{prefix}_optimizers")
    n = len(names)
    lrs = o(f"{prefix}_learning_rates")
    its = o(f"{prefix}_max_iterations")
    at = o(f"{prefix}_abs_tols", [0.0] * n)
    rt = o(f"{prefix}_rel_tols", [0.0] * n)
    return [OptStage(names[i], float(lrs[i]), int(its[i]), float(at[i]), float(rt[i])) for i in range(n)]


def _optax(stage: OptStage) -> optax.GradientTransformation:
    if stage.name == "Descent":
        return optax.sgd(stage.lr)
    if stage.name == "Adam":
        return optax.adam(stage.lr, b1=0.9, b2=0.999, eps=1.0e-8)
    if stage.name == "LBFGS":
        return optax.lbfgs()
    raise ValueError(f"unknown optimizer '{stage.name}'")


def _value_and_grad(loss_fn: LossFn, forward_mode: bool) -> Callable[[Any], Any]:
    """Jitted theta -> ((loss, aux), grad), by jacfwd or reverse mode."""
    if not forward_mode:
        return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    def f(t: Any) -> tuple[jax.Array, tuple[jax.Array, dict[str, Any]]]:
        val, aux = loss_fn(t)
        return val, (val, aux)

    def vg(th: Any) -> tuple[tuple[jax.Array, dict[str, Any]], Any]:
        g, (val, aux) = jax.jacfwd(f, has_aux=True)(th)
        return (val, aux), g
    return jax.jit(vg)


def run_stages(make_loss: Callable[[str], LossFn], theta0: Any, stages: Sequence[OptStage],
               callback: Callable[[int, Any, float, dict[str, Any]], None],
               forward_mode: bool = False) -> tuple[Any, list[float]]:
    """Minimize the loss with the configured stages.

    Args:
        make_loss: adjoint name (``forward``/``checkpoint``) -> loss function
            theta -> (loss, aux dict).
        theta0: initial parameters (pytree).
        stages: optimizer stages.
        callback: called as ``callback(iter, theta, loss, aux)`` with the loss *at*
            ``theta``, before the update (iterations start at 1, as in Optimization.jl).
        forward_mode: first-order stages use forward-mode gradients (diffrax
            ``ForwardMode``; cost grows with #params). LBFGS always uses reverse mode,
            because optax's line search differentiates ``value_fn`` in reverse mode.

    Returns:
        (theta_final, loss_history).
    """
    loss_rev = make_loss("checkpoint")
    vg = _value_and_grad(make_loss("forward") if forward_mode else loss_rev, forward_mode)
    theta, hist, it = theta0, [], 0
    for st in stages:
        if st.max_iter <= 0:
            logger.info("skipping %s (max_iterations = %d)", st.name, st.max_iter)
            continue
        opt = _optax(st)
        state = opt.init(theta)
        if st.name == "LBFGS":
            def lbfgs_step(th: Any, state: Any, opt: optax.GradientTransformation = opt) -> Any:
                (loss, aux), g = jax.value_and_grad(loss_rev, has_aux=True)(th)
                upd, state = opt.update(g, state, th, value=loss, grad=g,
                                        value_fn=lambda t: loss_rev(t)[0])
                return optax.apply_updates(th, upd), state, loss, aux, g
            step = jax.jit(lbfgs_step)
        else:
            def first_order_step(th: Any, state: Any, opt: optax.GradientTransformation = opt) -> Any:
                (loss, aux), g = vg(th)
                upd, state = opt.update(g, state, th)
                return optax.apply_updates(th, upd), state, loss, aux, g
            step = first_order_step
        t0 = time.perf_counter()
        for k in range(st.max_iter):
            new_theta, new_state, loss, aux, g = step(theta, state)
            it += 1
            loss_f = float(loss)
            hist.append(loss_f)
            callback(it, theta, loss_f, aux)
            if not np.isfinite(loss_f):
                logger.error("non-finite loss at iteration %d; stopping", it)
                return theta, hist
            theta, state = new_theta, new_state
            gnorm = float(optax.tree.norm(g))
            if k % 10 == 0:
                logger.info("%s iter %d: loss %.6e |grad| %.3e (%.2f s/iter)", st.name, it, loss_f, gnorm,
                            (time.perf_counter() - t0) / (k + 1))
            if st.abs_tol > 0 and gnorm < st.abs_tol:
                logger.info("%s converged: |grad| %.2e < %.2e", st.name, gnorm, st.abs_tol)
                break
    (loss, aux), _ = vg(theta)
    hist.append(float(loss))
    callback(it + 1, theta, float(loss), aux)
    return theta, hist
