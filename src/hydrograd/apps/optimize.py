"""Sequential optimizer loop shared by inversion and UDE training.

Stages
------
Mirrors the Julia logic: optimizers listed in ``*_optimizers`` run in order
(``Descent``, ``Adam``, ``LBFGS``) with parallel lists of learning rates and
iteration caps; an optimizer with ``max_iterations <= 0`` is skipped, each
starts from the previous result, and iteration numbers keep counting. A stage stops
early when the gradient norm falls below its ``abs_tol``; the whole run stops at a
non-finite loss.

Mapping to optax
----------------
``Descent(lr)`` -> ``sgd(lr)``; ``Adam(lr)`` -> ``adam(lr, b1=0.9, b2=0.999, eps=1e-8)``
(Optimisers.jl defaults); ``LBFGS`` -> ``optax.lbfgs()`` with its zoom line search
(Julia: BackTracking).

Iteration bookkeeping
---------------------
Each iteration evaluates the loss and gradient at theta_k, calls the callback with
(k, theta_k, L(theta_k)), and then updates theta_k -> theta_{k+1}. After the last stage
the loss is evaluated once more at the final parameters and reported as one extra
iteration.
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

# Loss function: parameters (pytree) -> (scalar loss, aux dict).
LossFn = Callable[[Any], tuple[jax.Array, dict[str, Any]]]


@dataclass(frozen=True)
class OptStage:
    """One optimizer stage.

    Attributes:
        name: ``"Descent"``, ``"Adam"`` or ``"LBFGS"``.
        lr: learning rate (unused by LBFGS).
        max_iter: iteration cap; <= 0 skips the stage.
        abs_tol: stop when |grad| < abs_tol (0 disables the test).
        rel_tol: relative tolerance (read from the run control, not used).
    """

    name: str
    lr: float
    max_iter: int
    abs_tol: float = 0.0
    rel_tol: float = 0.0


def stages_from_options(o: Callable[..., Any], prefix: str) -> list[OptStage]:
    """Read ``<prefix>_optimizers`` and the parallel lists from the run control.

    Args:
        o: option getter ``o(key, default)`` of the run control.
        prefix: ``"inversion"`` or ``"UDE"``.

    Returns:
        The optimizer stages in run order.
    """
    names = o(f"{prefix}_optimizers")
    n = len(names)

    # Parallel lists, one entry per optimizer; tolerances default to 0 (off).
    lrs = o(f"{prefix}_learning_rates")
    its = o(f"{prefix}_max_iterations")
    at = o(f"{prefix}_abs_tols", [0.0] * n)
    rt = o(f"{prefix}_rel_tols", [0.0] * n)

    stages = []
    for i in range(n):
        stages.append(OptStage(names[i], float(lrs[i]), int(its[i]), float(at[i]), float(rt[i])))

    return stages


def _optax(stage: OptStage) -> optax.GradientTransformation:
    """The optax optimizer of a stage (see the module docstring for the mapping).

    Args:
        stage: the optimizer stage.

    Returns:
        The optax gradient transformation.
    """
    if stage.name == "Descent":
        return optax.sgd(stage.lr)

    if stage.name == "Adam":
        return optax.adam(stage.lr, b1=0.9, b2=0.999, eps=1.0e-8)

    if stage.name == "LBFGS":
        return optax.lbfgs()

    raise ValueError(f"unknown optimizer '{stage.name}'")


def _value_and_grad(loss_fn: LossFn, forward_mode: bool) -> Callable[[Any], Any]:
    """Jitted theta -> ((loss, aux), grad), by jacfwd or reverse mode.

    Args:
        loss_fn: theta -> (scalar loss, aux dict).
        forward_mode: True for ``jax.jacfwd`` (cost grows with the number of parameters),
            False for ``jax.value_and_grad``.

    Returns:
        The jitted function theta -> ((loss, aux), grad); grad has the structure of theta.
    """
    if not forward_mode:
        return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    def f(t: Any) -> tuple[jax.Array, tuple[jax.Array, dict[str, Any]]]:
        """Loss with (loss, aux) as auxiliary output, so jacfwd also returns the value."""
        val, aux = loss_fn(t)
        return val, (val, aux)

    def vg(th: Any) -> tuple[tuple[jax.Array, dict[str, Any]], Any]:
        """Forward-mode ((loss, aux), grad) at parameters ``th``."""
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
    # Reverse-mode loss (always needed for LBFGS) and the gradient of the first-order stages.
    loss_rev = make_loss("checkpoint")
    vg = _value_and_grad(make_loss("forward") if forward_mode else loss_rev, forward_mode)

    theta = theta0
    hist: list[float] = []
    it = 0

    for st in stages:
        if st.max_iter <= 0:
            logger.info("skipping %s (max_iterations = %d)", st.name, st.max_iter)
            continue

        opt = _optax(st)
        state = opt.init(theta)

        # ------------------------------------------------------------------
        # 1. One optimizer step: (theta, state) -> (theta_new, state_new, loss, aux, grad)
        # ------------------------------------------------------------------
        if st.name == "LBFGS":
            def lbfgs_step(th: Any, state: Any, opt: optax.GradientTransformation = opt) -> Any:
                """LBFGS step with reverse-mode gradients and optax's line search.

                Args:
                    th: parameters (pytree).
                    state: optimizer state.
                    opt: the optimizer (bound at definition time).

                Returns:
                    (new parameters, new state, loss at ``th``, aux, gradient at ``th``).
                """
                (loss, aux), g = jax.value_and_grad(loss_rev, has_aux=True)(th)

                def value_fn(t: Any) -> jax.Array:
                    """Scalar loss at parameters ``t`` (for the line search)."""
                    return loss_rev(t)[0]

                upd, state = opt.update(g, state, th, value=loss, grad=g, value_fn=value_fn)
                return optax.apply_updates(th, upd), state, loss, aux, g

            step: Callable[..., Any] = jax.jit(lbfgs_step)
        else:
            def first_order_step(
                th: Any, state: Any, opt: optax.GradientTransformation = opt
            ) -> Any:
                """Descent/Adam step with the gradient of ``vg``.

                Args:
                    th: parameters (pytree).
                    state: optimizer state.
                    opt: the optimizer (bound at definition time).

                Returns:
                    (new parameters, new state, loss at ``th``, aux, gradient at ``th``).
                """
                (loss, aux), g = vg(th)
                upd, state = opt.update(g, state, th)
                return optax.apply_updates(th, upd), state, loss, aux, g

            step = first_order_step

        # ------------------------------------------------------------------
        # 2. Iterate the stage
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        for k in range(st.max_iter):
            new_theta, new_state, loss, aux, g = step(theta, state)
            it += 1

            # Record the loss at the current theta (before the update).
            loss_f = float(loss)
            hist.append(loss_f)
            callback(it, theta, loss_f, aux)

            if not np.isfinite(loss_f):
                logger.error("non-finite loss at iteration %d; stopping", it)
                return theta, hist

            # Accept the update.
            theta = new_theta
            state = new_state

            # Progress log every 10 iterations and the gradient-norm stopping test.
            gnorm = float(optax.tree.norm(g))
            if k % 10 == 0:
                logger.info("%s iter %d: loss %.6e |grad| %.3e (%.2f s/iter)", st.name, it, loss_f,
                            gnorm, (time.perf_counter() - t0) / (k + 1))
            if st.abs_tol > 0 and gnorm < st.abs_tol:
                logger.info("%s converged: |grad| %.2e < %.2e", st.name, gnorm, st.abs_tol)
                break

    # ------------------------------------------------------------------
    # 3. Final evaluation at the optimized parameters (one extra iteration)
    # ------------------------------------------------------------------
    (loss, aux), _ = vg(theta)
    hist.append(float(loss))
    callback(it + 1, theta, float(loss), aux)

    return theta, hist
