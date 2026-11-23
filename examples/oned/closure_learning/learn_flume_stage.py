"""Does the hiding factor of the flume mixture depend on the flow strength?

The three runs of Seal et al. (1997) differ in feed rate by a factor of four, and the
Shields number of the mean surface size varies along each deposit. This script asks
whether a hiding factor that depends on the Shields number, learned as a neural network,
predicts a run that was not used for fitting better than a hiding factor of the grain-size
ratio alone. Two runs are the training runs and the third is the test run
(``--test-run``); the three choices give a leave-one-run-out test.

Closures, each fitted together with Manning's n and the multiplier of the reference
Shields stress (Wilcock-Crowe) or the coefficient (Engelund-Hansen):

- ``power``: xi_k = (d_k / d_mean)^(-m_h);
- ``network``: ln xi_k = f(r_k) - f(0), r_k = ln(d_k / d_mean), a neural network f;
- ``network_shields``: ln xi_k = f(r_k, ln theta_m) - f(0, ln theta_m) with the Shields
  number theta_m of the mean size as a second input.

Both networks start from the fitted power law, so that a difference between them comes
from what the second input adds.

Training data of a run: the thickness of the deposit at the times of the bed profiles, the
grain sizes of the deposit (run 2: four percentiles at eight stations, Table 1 of the
paper; all runs: D90 and D50 of the deposit samples of Table 2), and the mean water depth
over the deposit (Table 3). The depth fixes Manning's n, and with it the bed shear stress
and the Shields number that the second input of the network is; ``--no-depth`` leaves it
out.

Usage::

    JAX_PLATFORMS=cpu python learn_flume_stage.py --test-run 1 [--iterations 60] [--no-depth]
                                                  [--serial]

The two networks are fitted side by side (four gradients at a time, about 10 GB of
memory); ``--serial`` fits them one after the other (about 5 GB).

Writes ``results/hiding_flume_stage_depth_test<run>.json``
(``results/hiding_flume_stage_test<run>.json`` with ``--no-depth``).

Units: SI; psi = log2(d in mm) for grain sizes.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
from hydrograd.sediment.closures import hiding_factor, init_hiding_network

import flume_case as flume
import flume_data as data
import learn_flume as learn

RESULTS = Path(__file__).parent / "results"
RUNS = (1, 2, 3)
# Range of the Shields number of the mean size that the second input covers (values on
# the deposits are 0.05 to 0.2; lower in the pool).
SHIELDS_RANGE = (0.005, 0.5)
# Start of the power-law fit: values near the fits of run 2 alone.
START = {"scale": 1.5, "n": 0.03, "m_h": -0.8}
# Shields numbers at which the learned hiding factor is stored.
SHIELDS_SHOWN = (0.03, 0.06, 0.1, 0.15, 0.2, 0.3)


def run_misfit(case, free: dict, depth: bool = True):
    """Misfit of one run: thickness profiles, grain sizes of the deposit, and the mean
    water depth if ``depth`` is set."""
    model = learn.predict(case, learn.parameters(case, free))
    target = learn.targets(case.run)
    thickness = jnp.concatenate(
        [model["thickness"][hours] - value for hours, (_, value) in target["thickness"].items()]
    )
    sizes = [(model["table2"][interval] - value).ravel() for interval, (_, value) in target["table2"].items()]
    if "table1" in model:
        sizes += [model["table1"][i] - data.TABLE1_PSI[p] for i, p in enumerate(learn.PERCENTILES)]
    sizes = jnp.concatenate(sizes)
    value = 0.5 * (jnp.mean((thickness / learn.SIGMA_THICKNESS) ** 2) + jnp.mean((sizes / learn.SIGMA_PSI) ** 2))
    return value + learn.depth_term(model, case.run) if depth else value


def bounds_of(start: dict):
    """Lower and upper limits of the free parameters, flattened like ``start``."""

    def limit(index: int):
        tree = {
            key: jnp.full_like(jnp.asarray(value, dtype=float), learn.LIMITS[key][index]) if key in learn.LIMITS
            else jax.tree.map(lambda leaf: jnp.full_like(leaf, (-jnp.inf, jnp.inf)[index]), value)
            for key, value in start.items()
        }
        return np.asarray(ravel_pytree(tree)[0])

    return list(zip(limit(0), limit(1), strict=True))


def fit(cases: list, start: dict, iterations: int, label: str, depth: bool = True):
    """Minimize the mean misfit of the training runs.

    The runs are evaluated in parallel threads. Returns the free parameters and the
    history of the misfit.
    """
    flat_start, unravel = ravel_pytree(start)
    functions = [jax.jit(jax.value_and_grad(lambda flat, c=case: run_misfit(c, unravel(flat), depth))) for case in cases]
    history = []

    def objective(flat):
        point = jnp.asarray(flat)
        with ThreadPoolExecutor(len(functions)) as pool:
            parts = list(pool.map(lambda function: function(point), functions))
        value = float(np.mean([float(part[0]) for part in parts]))
        gradient = np.mean([np.asarray(part[1], dtype=float) for part in parts], axis=0)
        value = value if np.isfinite(value) else 1.0e6
        history.append(value)
        print(f"  {label}: evaluation {len(history):3d}: misfit {value:.4f}", flush=True)
        return value, np.nan_to_num(gradient)

    solution = minimize(objective, np.asarray(flat_start), jac=True, method="L-BFGS-B", bounds=bounds_of(start),
                        options={"maxiter": iterations})
    return unravel(jnp.asarray(solution.x)), history


def hiding_surface(config, closure: dict, ratios, shields) -> np.ndarray:
    """Hiding factor for grains of the given d / d_mean at the given Shields numbers, shape [R, S]."""
    constants = config.constants
    mean = jnp.full(len(shields), 1.0e-3)
    stress = jnp.asarray(shields) * (constants.rho_s - constants.rho) * constants.g * 1.0e-3
    values = hiding_factor(jnp.asarray(ratios) * 1.0e-3, mean, stress, {"m_h": jnp.asarray(0.0), **closure}, config)
    return np.asarray(values)


def network_from_power(config, m_h: float):
    """Network parameters that reproduce the power law at every Shields number."""
    ratios = np.geomspace(0.02, 30.0, 60)
    shields = np.geomspace(0.01, 0.4, 12)
    wanted = jnp.asarray(-m_h * np.log(ratios))[:, None]
    start = init_hiding_network(jax.random.PRNGKey(0), config)
    flat, unravel = ravel_pytree(start)
    constants = config.constants
    mean = jnp.full(len(shields), 1.0e-3)
    stress = jnp.asarray(shields) * (constants.rho_s - constants.rho) * constants.g * 1.0e-3

    def loss(flat):
        values = hiding_factor(jnp.asarray(ratios) * 1.0e-3, mean, stress, {"hiding_network": unravel(flat)}, config)
        return jnp.mean((jnp.log(values) - wanted) ** 2)

    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    solution = minimize(lambda v: tuple(np.asarray(item, dtype=float) for item in value_and_gradient(jnp.asarray(v))),
                        np.asarray(flat), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    print(f"network started from the power law: mean square error of ln xi {solution.fun:.2e}", flush=True)
    return unravel(jnp.asarray(solution.x))


def main() -> None:
    """Fit the closures on two runs and test them on the third."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", type=int, choices=RUNS, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--formula", choices=tuple(learn.SCALE), default="wilcock_crowe")
    parser.add_argument("--flow", choices=("backwater", "unsteady"), default="unsteady")
    parser.add_argument("--no-depth", action="store_true", help="leave the mean water depth out of the training data")
    parser.add_argument("--serial", action="store_true", help="fit the two networks one after the other (less memory)")
    args = parser.parse_args()
    depth = not args.no_depth
    RESULTS.mkdir(exist_ok=True)
    scale_key, scale_name, _ = learn.SCALE[args.formula]
    train_runs = [run for run in RUNS if run != args.test_run]

    cases = {run: flume.build(run, args.formula, args.flow) for run in RUNS}
    forms = {
        "power": lambda case: case,
        "network": lambda case: flume.with_hiding(
            case, "network", hiding_size_ratio_range=learn.SIZE_RATIO_RANGE,
        ),
        "network_shields": lambda case: flume.with_hiding(
            case, "network", hiding_size_ratio_range=learn.SIZE_RATIO_RANGE,
            hiding_inputs=("size_ratio", "shields"), hiding_shields_range=SHIELDS_RANGE,
        ),
    }
    ratios = np.geomspace(0.01, 30.0, 80)
    out = {
        "arguments": vars(args), "formula": args.formula, "flow": args.flow, "depth_data": depth,
        "measured_depth_m": {str(run): data.RUNS[run]["depth_m"] for run in RUNS}, "test_run": args.test_run,
        "train_runs": train_runs, "x_m": cases[2].x.tolist(), "size_ratio": ratios.tolist(),
        "shields_shown": list(SHIELDS_SHOWN), "closures": {},
    }
    path = RESULTS / f"hiding_flume_stage{'_depth' if depth else ''}_test{args.test_run}.json"

    def record(name: str, fitted: dict, history: list, seconds: float, n_parameters: int) -> None:
        """Scores of all runs and the learned hiding factor of a fitted closure."""
        form = forms[name]
        entry = {
            "n_parameters": n_parameters, "misfit_history": history, "seconds": seconds,
            scale_name: float(np.exp(fitted[scale_key])), "n_manning": float(np.exp(fitted["log_n"])), "runs": {},
        }
        if name == "power":
            entry["m_h"] = float(fitted["m_h"])
        closure = {k: v for k, v in fitted.items() if k in ("m_h", "hiding_network")}
        entry["hiding"] = hiding_surface(form(cases[2]).config.sediment, closure, ratios, SHIELDS_SHOWN).tolist()
        for run, case in cases.items():
            run_case = form(case)
            model = jax.jit(lambda f, c=run_case: learn.predict(c, learn.parameters(c, f)))(fitted)
            scores = learn.scores(model, run)
            scores["misfit"] = float(jax.jit(lambda f, c=run_case: run_misfit(c, f, depth))(fitted))
            entry["runs"][str(run)] = {"model": learn.as_lists(model), "scores": scores}
        out["closures"][name] = entry
        line = f"{name:16s} {n_parameters:3d} parameters, {len(history):3d} evaluations, {seconds:6.0f} s: " \
               f"misfit {history[0]:.3f} -> {min(history):.3f}; {scale_name} {entry[scale_name]:.3f}, n {entry['n_manning']:.4f}" \
               + (f", m_h {entry['m_h']:.3f}" if name == "power" else "")
        print(line, flush=True)
        for run in cases:
            role = "test" if run == args.test_run else "train"
            print(f"   run {run} ({role}): {entry['runs'][str(run)]['scores']}", flush=True)
        path.write_text(json.dumps(out))

    # 1. Power law.
    begin = time.perf_counter()
    start = {scale_key: jnp.asarray(np.log(START["scale"])), "log_n": jnp.asarray(np.log(START["n"])),
             "m_h": jnp.asarray(START["m_h"])}
    power, history = fit([cases[run] for run in train_runs], start, args.iterations, "power", depth)
    record("power", power, history, time.perf_counter() - begin, 3)

    # 2. The two networks, both started from the fitted power law, fitted side by side.
    def fit_network(name: str):
        begin = time.perf_counter()
        form = forms[name]
        network = network_from_power(form(cases[2]).config.sediment, float(power["m_h"]))
        start = {scale_key: power[scale_key], "log_n": power["log_n"], "hiding_network": network}
        fitted, history = fit([form(cases[run]) for run in train_runs], start, args.iterations, name, depth)
        return name, fitted, history, time.perf_counter() - begin, int(ravel_pytree(start)[0].size)

    names = ("network", "network_shields")
    if args.serial:
        for name in names:
            record(*fit_network(name))
    else:
        with ThreadPoolExecutor(2) as pool:
            for result in pool.map(fit_network, names):
                record(*result)


if __name__ == "__main__":
    main()
