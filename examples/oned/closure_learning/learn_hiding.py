"""Learn the hiding factor of a sediment mixture from surveys of a synthetic reach.

The truth is the reach of ``case.py`` with a hiding law of two power laws. Its surveys of
the aggradation scenario (deposit per size class and cross-section at three times, surface
composition at the end), with noise, are the training data. Three closures are fitted:

- ``none``: no hiding (nothing to fit);
- ``power``: xi_k = (d_k / d_m)^(-m_h), one parameter;
- ``network``: ln xi_k = f(ln(d_k / d_m)) - f(0) with a small neural network f.

Each fit minimizes the misfit of the surveys with L-BFGS; the gradient comes from
reverse-mode differentiation through the whole run. The fitted closures are then run on
the degradation scenario (clear-water inflow), which was not used for training.

Usage::

    JAX_PLATFORMS=cpu python learn_hiding.py [--iterations 150] [--noise 0.02] [--seed 0]

Writes ``results/hiding_twin.json`` and ``results/hiding_twin.png``.

Units: SI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydrograd.sediment.closures import hiding_factor, init_hiding_network

import case as reach

RESULTS = Path(__file__).parent / "results"
# Range of d_k / d_m mapped to the inputs of the network.
SIZE_RATIO_RANGE = (0.05, 20.0)
# Standard deviation of the noise of a surface fraction (-).
SURFACE_NOISE = 0.01


def closures(base: reach.Case) -> dict:
    """The three closures: case, free parameters at the start, and how they enter theta."""
    network_case = reach.with_hiding(base, "network", hiding_size_ratio_range=SIZE_RATIO_RANGE)
    network_start = init_hiding_network(jax.random.PRNGKey(0), network_case.config.sediment)
    return {
        "none": (base, None, lambda free: {}),
        "power": (base, jnp.asarray(0.0), lambda free: {"m_h": free}),
        "network": (network_case, network_start, lambda free: {"hiding_network": free}),
    }


def hiding_curve(config, extra: dict, ratios: np.ndarray) -> np.ndarray:
    """Hiding factor of a closure for grains of the given d / d_m (mean size 1 mm)."""
    mean = jnp.asarray([1.0e-3])
    theta = {"m_h": jnp.asarray(0.0), **extra}
    values = hiding_factor(jnp.asarray(ratios) * 1.0e-3, mean, jnp.asarray([1.0]), theta, config)
    return np.asarray(values[:, 0])


def main() -> None:
    """Make the data, fit the closures, test them and draw the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--noise", type=float, default=0.02, help="relative noise of the deposits")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    train = reach.build("aggradation")
    test = reach.build("degradation")
    truth_extra = reach.as_arrays(reach.TRUTH)

    def observations(base, form_case, extra):
        return jax.jit(lambda e: reach.observe(form_case, {**base.theta, **e}))(extra)

    # Training data: the truth with noise. The noise of a deposit is relative to the
    # root-mean-square deposit of its class.
    truth_train = observations(train, reach.with_hiding(train, "two_power"), truth_extra)
    truth_test = observations(test, reach.with_hiding(test, "two_power"), truth_extra)
    rng = np.random.default_rng(args.seed)
    deposit_scale = np.sqrt(np.mean(np.asarray(truth_train[0]) ** 2, axis=(0, 2), keepdims=True))
    deposit_sigma = np.maximum(args.noise, 1.0e-3) * deposit_scale
    deposit_data = np.asarray(truth_train[0]) + args.noise * deposit_scale * rng.standard_normal(truth_train[0].shape)
    surface_data = np.asarray(truth_train[1]) + SURFACE_NOISE * rng.standard_normal(truth_train[1].shape)

    def misfit_of(form_case, wrap):
        def misfit(free):
            deposit, surface, _ = reach.observe(form_case, {**train.theta, **wrap(free)})
            return 0.5 * (
                jnp.mean(((deposit - deposit_data) / deposit_sigma) ** 2)
                + jnp.mean(((surface - surface_data) / SURFACE_NOISE) ** 2)
            )
        return misfit

    ratios = np.geomspace(0.05, 10.0, 60)
    out = {
        "arguments": vars(args),
        "truth": reach.TRUTH,
        "x_m": train.x.tolist(),
        "diameters_m": list(reach.DIAMETERS),
        "survey_steps": list(reach.SURVEY_STEPS),
        "size_ratio": ratios.tolist(),
        "hiding_truth": hiding_curve(reach.with_hiding(train, "two_power").config.sediment, truth_extra, ratios).tolist(),
        "train": {"truth": {"deposit": np.asarray(truth_train[0]).tolist(), "surface": np.asarray(truth_train[1]).tolist(),
                            "thalweg": np.asarray(truth_train[2]).tolist()}},
        "test": {"truth": {"deposit": np.asarray(truth_test[0]).tolist(), "surface": np.asarray(truth_test[1]).tolist(),
                           "thalweg": np.asarray(truth_test[2]).tolist()}},
        "closures": {},
    }
    # Sizes relative to the mean size that occur in the training run (truth, final bed).
    mean_train = np.sum(np.asarray(truth_train[1]) * np.asarray(reach.DIAMETERS)[:, None], axis=0)
    seen = np.asarray(reach.DIAMETERS)[:, None] / mean_train[None, :]
    out["size_ratio_seen"] = [float(seen.min()), float(seen.max())]

    for name, (form_case, start, wrap) in closures(train).items():
        begin = time.perf_counter()
        history = []
        if start is None:
            free = None
            extra = {}
            history.append(float(jax.jit(misfit_of(form_case, wrap))(None)))
        else:
            flat_start, unravel = ravel_pytree(start)
            fit_misfit = misfit_of(form_case, wrap)
            value_and_gradient = jax.jit(jax.value_and_grad(lambda flat, f=fit_misfit, u=unravel: f(u(flat))))

            def objective(flat, history=history, value_and_gradient=value_and_gradient):
                value, gradient = value_and_gradient(jnp.asarray(flat))
                history.append(float(value))
                return float(value), np.asarray(gradient, dtype=float)

            solution = minimize(objective, np.asarray(flat_start), jac=True, method="L-BFGS-B",
                                options={"maxiter": args.iterations})
            free = unravel(jnp.asarray(solution.x))
            extra = wrap(free)
        seconds = time.perf_counter() - begin

        fitted_train = observations(train, form_case, extra)
        test_case = form_case._replace(forcing=test.forcing)
        fitted_test = observations(test, test_case, extra)
        entry = {
            "n_parameters": 0 if start is None else int(ravel_pytree(start)[0].size),
            "misfit_history": history,
            "seconds": seconds,
            "hiding": hiding_curve(form_case.config.sediment, extra, ratios).tolist(),
        }
        if name == "power":
            entry["m_h"] = float(free)
        for label, fitted, truth in (("train", fitted_train, truth_train), ("test", fitted_test, truth_test)):
            entry[label] = {
                "deposit": np.asarray(fitted[0]).tolist(),
                "surface": np.asarray(fitted[1]).tolist(),
                "thalweg": np.asarray(fitted[2]).tolist(),
                "rms_thalweg_m": float(np.sqrt(np.mean((np.asarray(fitted[2]) - np.asarray(truth[2])) ** 2))),
                "rms_surface": float(np.sqrt(np.mean((np.asarray(fitted[1]) - np.asarray(truth[1])) ** 2))),
            }
        out["closures"][name] = entry
        print(f"{name:8s} {entry['n_parameters']:3d} parameters, {len(history):3d} evaluations, {seconds:6.0f} s: "
              f"misfit {history[0]:10.2f} -> {min(history):8.3f}; thalweg error train {entry['train']['rms_thalweg_m']:.3f} m, "
              f"test {entry['test']['rms_thalweg_m']:.3f} m; surface error train {entry['train']['rms_surface']:.4f}, "
              f"test {entry['test']['rms_surface']:.4f}" + (f"; m_h {entry['m_h']:.3f}" if name == "power" else ""), flush=True)
        (RESULTS / "hiding_twin.json").write_text(json.dumps(out))

    figure(out)


def figure(out: dict) -> None:
    """Learned hiding factors, and the bed of the training and test runs."""
    colors = {"none": "0.6", "power": "C0", "network": "C3"}
    labels = {"none": "no hiding", "power": "power law (1 parameter)", "network": "neural network"}
    x = np.asarray(out["x_m"]) / 1000.0
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.0))

    ax = axs[0, 0]
    ratios = np.asarray(out["size_ratio"])
    ax.loglog(ratios, out["hiding_truth"], "k-", lw=2.5, label="truth")
    for name, entry in out["closures"].items():
        ax.loglog(ratios, entry["hiding"], color=colors[name], lw=1.5, label=labels[name])
    ax.axvspan(*out["size_ratio_seen"], color="0.92", zorder=0, label="sizes in the training run")
    ax.set_xlabel("grain size / mean size of the bed surface")
    ax.set_ylabel("hiding factor")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for name, entry in out["closures"].items():
        if len(entry["misfit_history"]) > 1:
            ax.semilogy(entry["misfit_history"], color=colors[name], label=labels[name])
        else:
            ax.axhline(entry["misfit_history"][0], color=colors[name], label=labels[name])
    ax.set_xlabel("model evaluations")
    ax.set_ylabel("misfit of the training surveys")
    ax.legend(fontsize=8)

    mean_size = lambda surface: 1000.0 * np.sum(np.asarray(surface) * np.asarray(out["diameters_m"])[:, None], axis=0)
    for column, label, title in ((1, "train", "aggradation (training)"), (2, "test", "degradation (not used for training)")):
        truth = out[label]["truth"]
        ax = axs[0, column]
        ax.plot(x, np.asarray(truth["thalweg"][-1]) - bed_start(out), "k-", lw=2.5, label="truth")
        for name, entry in out["closures"].items():
            ax.plot(x, np.asarray(entry[label]["thalweg"][-1]) - bed_start(out), color=colors[name], lw=1.5, label=labels[name])
        ax.set_ylabel("bed change at the last survey (m)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax = axs[1, column]
        ax.plot(x, mean_size(truth["surface"]), "k-", lw=2.5)
        for name, entry in out["closures"].items():
            ax.plot(x, mean_size(entry[label]["surface"]), color=colors[name], lw=1.5)
        ax.set_xlabel("distance (km)")
        ax.set_ylabel("mean size of the bed surface (mm)")
    fig.tight_layout()
    fig.savefig(RESULTS / "hiding_twin.png", dpi=150)
    print(f"wrote {RESULTS / 'hiding_twin.png'}")


def bed_start(out: dict) -> np.ndarray:
    """Initial thalweg (m) at the sections."""
    x = np.asarray(out["x_m"])
    return reach.SLOPE * (reach.LENGTH - x)


if __name__ == "__main__":
    main()
