"""Learn the hiding factor of a gravel-sand mixture from a flume experiment.

The data are the downstream-fining runs of Seal et al. (1997) (``flume_data.py``,
``flume_case.py``). Run 2 is the training run: the thickness of the deposit at five times
and the grain sizes (four percentiles at eight stations) of the deposit laid down between
hours 14 and 22. The hiding closures are fitted one after the other, each together with
Manning's n and one parameter of the transport formula (Engelund-Hansen: the coefficient
``a_eh``; Wilcock-Crowe: the multiplier ``tau_r_wc`` of the reference Shields stress):

- ``none``: no hiding;
- ``published`` (Wilcock-Crowe only): the hiding function of Wilcock and Crowe (2003);
- ``power``: xi_k = (d_k / d_mean)^(-m_h), one parameter more;
- ``network``: ln xi_k = f(ln(d_k / d_mean)) - f(0) with a small neural network f, started
  from the fitted power law (Engelund-Hansen) or from the published function
  (Wilcock-Crowe).

With ``--depth`` the mean water depth over the deposit of run 2 (Table 3 of the paper) is
a further measurement. The deposit alone does not fix Manning's n: a smooth bed with a
shallow, fast flow and a rough bed with a deep, slow flow can build nearly the same wedge
if the reference stress is adjusted, and the depth tells them apart.

With ``--exchange bedload_mix`` the surface layer passes down to the deposit a mixture
of the load and the surface instead of its own composition (the transfer function of
Hoey and Ferguson, 1994, and Toro-Escobar et al., 1996, with the sand relation of Cui,
2007; see ``hydrograd.sediment.bed``), and the share of the load in that mixture is fitted
with the other parameters (published value 0.7).

Each fit minimizes the misfit with L-BFGS; the gradient comes from reverse-mode
differentiation through the whole run. Runs 1 (twice the feed rate) and 3 (half the feed
rate) were not used for fitting and test the fitted closures.

The flow is computed by the backwater solver (default) or by the unsteady shallow-water
solver (``--flow unsteady``), which follows the supercritical flow over the deposit of
run 1; a fit with the unsteady solver takes about five times longer.

Usage::

    JAX_PLATFORMS=cpu python learn_flume.py [--formula wilcock_crowe] [--flow unsteady]
                                            [--depth] [--exchange bedload_mix]
                                            [--iterations 120]

Writes ``results/hiding_flume_<formula>.json`` (``..._<formula>_unsteady.json`` with the
unsteady solver, ``..._depth.json`` with the water depth as a measurement, and
``..._mix.json`` with the transfer function).

Units: SI; psi = log2(d in mm) for grain sizes.
"""

from __future__ import annotations

import argparse
import json
import time
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

RESULTS = Path(__file__).parent / "results"
SIZE_RATIO_RANGE = (0.005, 50.0)
SIGMA_THICKNESS = 0.03       # m
SIGMA_PSI = 0.3              # psi units
SIGMA_DEPTH = 0.01           # m
PERCENTILES = (90, 84, 50, 16)
TRAIN_RUN = 2
TEST_RUNS = (1, 3)
# Where the "downstream" sample of Table 2 lies, as a part of the distance to the gravel
# front at the end of the interval (run 2: 17 m with the front at 27 m).
DOWNSTREAM_PART = 0.63
# Fitted parameter of each formula: name of the free parameter (its logarithm), name in
# theta, start value.
SCALE = {"engelund_hansen": ("log_a_eh", "a_eh", 0.3), "wilcock_crowe": ("log_tau_r", "tau_r_wc", 1.0)}
START_N = 0.02
LIMITS = {"log_a_eh": (np.log(0.01), np.log(10.0)), "log_tau_r": (np.log(0.3), np.log(3.0)),
          "log_tau_r_direct": (np.log(0.3), np.log(3.0)),
          "log_n": (np.log(0.012), np.log(0.035)), "log_k": (np.log(0.002), np.log(0.3)),
          "log_alpha": (np.log(0.3), np.log(30.0)),
          "m_h": (-1.5, 1.0), "c_interface": (0.0, 0.95)}
# Start of the share of the load in the transfer function (Toro-Escobar et al., 1996).
START_SHARE = 0.7


def parameters(case, free: dict) -> dict:
    """Model parameters from the free ones."""
    key, name, _ = SCALE[case.config.sediment.transport_formula]
    theta = {**case.theta, name: jnp.exp(free[key])}
    if "log_n" in free:
        theta["n_manning"] = jnp.exp(free["log_n"])
    if "log_k" in free:
        theta["roughness_length"] = jnp.exp(free["log_k"])
    if "log_alpha" in free:
        theta["roughness_factor"] = jnp.exp(free["log_alpha"])
    if "m_h" in free:
        theta["m_h"] = free["m_h"]
    for network in ("hiding_network", "transport_network"):
        if network in free:
            theta[network] = free[network]
    if "c_interface" in free:
        theta["c_interface"] = free["c_interface"]
    return theta


def front_station(x, thickness) -> float:
    """Station of the top of the gravel front of a measured profile (m)."""
    return float(x[np.where(thickness > 0.12)[0][-1]])


def targets(run: int) -> dict:
    """Measurements of a run in the layout of ``predict``."""
    profiles = data.thickness_profiles(run)
    out = {"thickness": {hours: (x.astype(int), value) for hours, (x, value) in profiles.items()}, "table2": {}}
    for interval, sizes in data.TABLE2[run].items():
        x, thickness = profiles[interval[1]]
        stations = (3, round(DOWNSTREAM_PART * front_station(x, thickness)))
        out["table2"][interval] = (stations, np.log2([sizes["D90"], sizes["D50"]]))        # [2 sizes, 2 stations]
    out["surface"] = np.log2([data.SURFACE[run]["D90"], data.SURFACE[run]["D50"]])
    return out


def predict(case, theta, result=None) -> dict:
    """Model values of all measurements of a run (thickness in m, sizes in psi).

    ``result`` is the result of the run if it was already made (``flume.run_case``).
    """
    result = flume.run_case(case, theta) if result is None else result
    thickness, cumulative, surface, depth = flume.observables(case, result)
    target = targets(case.run)
    out = {"thickness": {hours: thickness[flume.step_of(hours) - 1][index] for hours, (index, _) in target["thickness"].items()}}
    out["profile_end"] = thickness[-1]
    out["depth"] = jnp.mean(depth[flume.deposit_sections(case.run)])
    out["table2"] = {
        interval: flume.percentile_psi(flume.deposit_fractions(cumulative, interval), [90, 50])[:, jnp.asarray(stations)]
        for interval, (stations, _) in target["table2"].items()
    }
    front = round(DOWNSTREAM_PART * data.RUNS[case.run]["front_m"])
    out["surface"] = flume.percentile_psi(surface, [90, 50])[:, jnp.asarray([3, front])]
    if case.run == TRAIN_RUN:
        fractions = flume.deposit_fractions(cumulative, data.TABLE1_HOURS)
        out["table1"] = flume.percentile_psi(fractions, list(PERCENTILES))[:, jnp.asarray(np.round(data.TABLE1_X).astype(int))]
    return out


def depth_term(model: dict, run: int):
    """Misfit of the mean water depth over the deposit."""
    return 0.5 * ((model["depth"] - data.RUNS[run]["depth_m"]) / SIGMA_DEPTH) ** 2


def misfit(case, free: dict, depth: bool = False):
    """Misfit of the training run: thickness profiles, grain sizes of Table 1, and the
    mean water depth if ``depth`` is set."""
    model = predict(case, parameters(case, free))
    target = targets(case.run)
    thickness = jnp.concatenate([model["thickness"][hours] - value for hours, (_, value) in target["thickness"].items()])
    sizes = jnp.concatenate([model["table1"][i] - data.TABLE1_PSI[p] for i, p in enumerate(PERCENTILES)])
    value = 0.5 * (jnp.mean((thickness / SIGMA_THICKNESS) ** 2) + jnp.mean((sizes / SIGMA_PSI) ** 2))
    return value + depth_term(model, case.run) if depth else value


def scores(model: dict, run: int) -> dict:
    """Root-mean-square errors of a run."""
    target = targets(run)
    thickness = np.concatenate([np.asarray(model["thickness"][hours]) - value for hours, (_, value) in target["thickness"].items()])
    table2 = np.concatenate([(np.asarray(model["table2"][interval]) - value).ravel() for interval, (_, value) in target["table2"].items()])
    out = {
        "rms_thickness_m": float(np.sqrt(np.mean(thickness**2))),
        "rms_deposit_psi": float(np.sqrt(np.mean(table2**2))),
        "rms_surface_psi": float(np.sqrt(np.mean((np.asarray(model["surface"]) - target["surface"]) ** 2))),
        "depth_m": float(model["depth"]),
    }
    if "table1" in model:
        table1 = np.concatenate([np.asarray(model["table1"][i]) - data.TABLE1_PSI[p] for i, p in enumerate(PERCENTILES)])
        out["rms_table1_psi"] = float(np.sqrt(np.mean(table1**2)))
    return out


def as_lists(tree):
    """Nested dictionaries of arrays as JSON-ready lists (keys as text)."""
    if isinstance(tree, dict):
        return {str(key): as_lists(value) for key, value in tree.items()}
    if isinstance(tree, (tuple, list)):
        return [as_lists(value) for value in tree]
    return np.asarray(tree).tolist()


def hiding_curve(config, theta: dict, ratios: np.ndarray) -> np.ndarray:
    """Hiding factor for grains of the given d / d_m (mean size 1 mm)."""
    values = hiding_factor(jnp.asarray(ratios) * 1.0e-3, jnp.asarray([1.0e-3]), jnp.asarray([1.0]),
                           {"m_h": jnp.asarray(0.0), **theta}, config)
    return np.asarray(values[:, 0])


def network_like(config, reference_config, reference_theta: dict):
    """Network parameters that reproduce the hiding factor of another closure."""
    ratios = jnp.asarray(np.geomspace(0.02, 30.0, 80))
    wanted = jnp.log(jnp.asarray(hiding_curve(reference_config, reference_theta, np.asarray(ratios))))
    start = init_hiding_network(jax.random.PRNGKey(0), config)
    flat, unravel = ravel_pytree(start)

    def loss(flat):
        values = hiding_factor(ratios * 1.0e-3, jnp.asarray([1.0e-3]), jnp.asarray([1.0]), {"hiding_network": unravel(flat)}, config)
        return jnp.mean((jnp.log(values[:, 0]) - wanted) ** 2)

    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    solution = minimize(lambda v: tuple(np.asarray(item, dtype=float) for item in value_and_gradient(jnp.asarray(v))),
                        np.asarray(flat), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    print(f"network started from another closure: mean square error of ln xi {solution.fun:.2e}", flush=True)
    return unravel(jnp.asarray(solution.x))


def fit(case, start: dict, iterations: int, depth: bool = False, ftol: float | None = None):
    """Minimize the misfit of the training run; returns the free parameters and the history.

    ``ftol`` replaces the relative decrease of the misfit at which L-BFGS stops (about
    2e-9 by default). A network whose first step gains little needs a smaller value.
    """
    flat_start, unravel = ravel_pytree(start)
    low = ravel_pytree({key: jnp.full_like(jnp.asarray(value, dtype=float), LIMITS[key][0]) if key in LIMITS
                        else jax.tree.map(lambda leaf: jnp.full_like(leaf, -jnp.inf), value) for key, value in start.items()})[0]
    high = ravel_pytree({key: jnp.full_like(jnp.asarray(value, dtype=float), LIMITS[key][1]) if key in LIMITS
                         else jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.inf), value) for key, value in start.items()})[0]
    value_and_gradient = jax.jit(jax.value_and_grad(lambda flat: misfit(case, unravel(flat), depth)))
    history = []

    def objective(flat):
        value, gradient = value_and_gradient(jnp.asarray(flat))
        value = float(value) if np.isfinite(float(value)) else 1.0e6
        gradient = np.nan_to_num(np.asarray(gradient, dtype=float))
        history.append(value)
        print(f"  evaluation {len(history):3d}: misfit {value:.5f}, gradient norm {np.linalg.norm(gradient):.2e}", flush=True)
        return value, gradient

    options = {"maxiter": iterations} if ftol is None else {"maxiter": iterations, "ftol": ftol}
    solution = minimize(objective, np.asarray(flat_start), jac=True, method="L-BFGS-B",
                        bounds=list(zip(np.asarray(low), np.asarray(high), strict=True)), options=options)
    return unravel(jnp.asarray(solution.x)), history


def main() -> None:
    """Fit the three closures on run 2 and test them on runs 1 and 3."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--formula", choices=tuple(SCALE), default="engelund_hansen")
    parser.add_argument("--flow", choices=("backwater", "unsteady"), default="backwater")
    parser.add_argument("--depth", action="store_true", help="use the mean water depth of run 2 as a measurement")
    parser.add_argument("--exchange", choices=("surface", "bedload_mix"), default="surface",
                        help="what the surface layer passes down to the deposit in aggradation")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    scale_key, scale_name, scale_start = SCALE[args.formula]
    published = args.formula == "wilcock_crowe"

    cases = {run: flume.build(run, args.formula, args.flow, args.exchange) for run in (TRAIN_RUN, *TEST_RUNS)}
    mixed = args.exchange == "bedload_mix"
    tag = args.formula if args.flow == "backwater" else f"{args.formula}_{args.flow}"
    tag = f"{tag}_depth" if args.depth else tag
    tag = f"{tag}_mix" if mixed else tag
    ratios = np.geomspace(0.01, 30.0, 80)
    out = {
        "arguments": vars(args), "formula": args.formula, "flow": args.flow, "depth_data": args.depth,
        "exchange": args.exchange,
        "measured_depth_m": {str(run): data.RUNS[run]["depth_m"] for run in (TRAIN_RUN, *TEST_RUNS)}, "train_run": TRAIN_RUN, "test_runs": list(TEST_RUNS), "percentiles": list(PERCENTILES),
        "x_m": cases[TRAIN_RUN].x.tolist(), "size_ratio": ratios.tolist(), "closures": {},
        "targets": {str(run): as_lists({k: v for k, v in targets(run).items() if k != "thickness"}) for run in cases},
    }
    fitted_by: dict = {}
    forms = {
        "none": lambda case: case,
        "published": lambda case: flume.with_hiding(case, "wilcock_crowe"),
        "power": lambda case: case,
        "network": lambda case: flume.with_hiding(case, "network", hiding_size_ratio_range=SIZE_RATIO_RANGE),
    }
    for name in ("none", "published", "power", "network") if published else ("none", "power", "network"):
        begin = time.perf_counter()
        form = forms[name]
        if name == "power":
            start = {**fitted_by["none"], "m_h": jnp.asarray(-0.5)}
            if mixed:
                # The share fitted without hiding sits where the model sorts least.
                start["c_interface"] = jnp.asarray(START_SHARE)
        elif name == "network":
            # Start from the closure the network is to improve on.
            origin = "published" if published else "power"
            reference_theta = {k: v for k, v in fitted_by[origin].items() if k == "m_h"}
            network = network_like(form(cases[TRAIN_RUN]).config.sediment, forms[origin](cases[TRAIN_RUN]).config.sediment,
                                   reference_theta)
            start = {scale_key: fitted_by[origin][scale_key], "log_n": fitted_by[origin]["log_n"], "hiding_network": network}
            if mixed:
                start["c_interface"] = fitted_by[origin]["c_interface"]
        else:
            start = {scale_key: jnp.asarray(np.log(scale_start)), "log_n": jnp.asarray(np.log(START_N))}
            if mixed:
                start["c_interface"] = jnp.asarray(START_SHARE)
        print(f"{name}: fitting", flush=True)
        fitted, history = fit(form(cases[TRAIN_RUN]), start, args.iterations, args.depth)
        fitted_by[name] = fitted
        entry = {
            "n_parameters": int(ravel_pytree(start)[0].size), "misfit_history": history, "seconds": time.perf_counter() - begin,
            "free": as_lists(fitted),
            scale_name: float(np.exp(fitted[scale_key])), "n_manning": float(np.exp(fitted["log_n"])), "runs": {},
        }
        if name == "power":
            entry["m_h"] = float(fitted["m_h"])
        if mixed:
            entry["c_interface"] = float(fitted["c_interface"])
        config = form(cases[TRAIN_RUN]).config.sediment
        entry["hiding"] = hiding_curve(config, {k: v for k, v in fitted.items() if k in ("m_h", "hiding_network")}, ratios).tolist()
        for run, case in cases.items():
            run_case = form(case)
            model = jax.jit(lambda f, c=run_case: predict(c, parameters(c, f)))(fitted)
            entry["runs"][str(run)] = {"model": as_lists(model), "scores": scores(model, run)}
        out["closures"][name] = entry
        line = f"{name:8s} {entry['n_parameters']:3d} parameters, {len(history):3d} evaluations, {entry['seconds']:6.0f} s: " \
               f"misfit {history[0]:.2f} -> {min(history):.3f}; {scale_name} {entry[scale_name]:.3f}, n {entry['n_manning']:.4f}" \
               + (f", m_h {entry['m_h']:.3f}" if name == "power" else "") \
               + (f", share of the load {entry['c_interface']:.3f}" if mixed else "")
        print(line, flush=True)
        for run in cases:
            print(f"   run {run} ({'train' if run == TRAIN_RUN else 'test'}): {entry['runs'][str(run)]['scores']}", flush=True)
        (RESULTS / f"hiding_flume_{tag}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
