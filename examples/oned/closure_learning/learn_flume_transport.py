"""Learn the transport function of a gravel-sand mixture from a flume experiment.

In the formula of Wilcock and Crowe (2003) the transport rate of every size class follows
from one number, the ratio phi of the bed shear stress to the reference shear stress of
the class, through one curve W*(phi) (``hydrograd.sediment.closures``). That curve was
fitted to transport rates measured in a recirculating flume. Here it is learned instead
from the deposit that the transport leaves behind: the thickness of the wedge of run 2 of
Seal et al. (1997) at five times, the grain sizes of the deposit laid down between hours
14 and 22, and the mean water depth over the deposit (``flume_data.py``,
``flume_case.py``, ``learn_flume.py``). No transport rate is measured anywhere along the
flume; the curve is found through the solver.

The hiding function is the published one, and the surface layer passes down to the
deposit a mixture of the load and the surface (``interface_exchange = "bedload_mix"``).
Three transport functions are fitted one after the other, each together with Manning's n,
the multiplier ``tau_r_wc`` of the reference Shields stress and the share of the load in
the mixture passed down:

- ``wilcock_crowe``: the published curve;
- ``parker``: the curve of Parker (1990) in the form used by Cui (2007) for these runs;
- ``network``: W*(phi) = W*_WC(phi) exp(f(ln phi) - f(0)) with a small neural network f,
  started from the fitted ``wilcock_crowe`` stage with f close to 0. Every curve gives
  W* = 0.002 at phi = 1, so the reference stress keeps its meaning.

Each fit minimizes the misfit with L-BFGS; the gradient comes from reverse-mode
differentiation through the whole run. Runs 1 (twice the feed rate) and 3 (half the feed
rate) are not used for fitting and test the fitted curves. For every stage the curve and
the stress ratios that the classes meet on the deposit at the end of each run are
written, so that one can see which part of the curve the data constrain.

Usage::

    JAX_PLATFORMS=cpu python learn_flume_transport.py [--flow unsteady] [--iterations 60]
                                                      [--stages wilcock_crowe parker network]
                                                      [--start TAU_R N SHARE]

Writes ``results/transport_flume_<flow>.json``; the entry ``free`` of each stage holds the
fitted parameters (with the weights of the network), from which ``predict_run6.py`` runs
an experiment that was not used for fitting.

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

jax.config.update("jax_enable_x64", True)
from hydrograd.sediment.closures import init_transport_network, transport_function

import flume_case as flume
import flume_data as data
import learn_flume as lf

RESULTS = Path(__file__).parent / "results"
STAGES = ("wilcock_crowe", "parker", "network")
# Start of the multiplier of the reference Shields stress, of Manning's n and of the share
# of the load in the mixture passed down to the deposit.
START = (1.0, 0.029, lf.START_SHARE)
# The network starts from the published curve with a small random output layer: with an
# output layer of zero the hidden layers have no gradient, the first step of L-BFGS gains
# almost nothing and the fit stops at once.
START_OUTPUT_SCALE = 0.05
NETWORK_FTOL = 1.0e-12
# Stress ratios at which the fitted curves are written.
STRESS_RATIOS = np.geomspace(0.2, 50.0, 140)


def with_transport(case, function: str):
    """The case with the published hiding function and the given transport function."""
    return flume.with_hiding(case, "wilcock_crowe", transport_function=function)


def curve(config, free: dict) -> np.ndarray:
    """W*(phi) of a fitted stage at ``STRESS_RATIOS``."""
    theta = {key: value for key, value in free.items() if key == "transport_network"}
    return np.asarray(transport_function(jnp.asarray(STRESS_RATIOS), theta, config))


def evaluate(case, free: dict) -> dict:
    """Model values of the measurements of a run, the stress ratio of every class at the
    end of the run (shape [K, N]) and the surface fractions there (shape [K, N])."""
    theta = lf.parameters(case, free)
    result = flume.run_case(case, theta)
    model = lf.predict(case, theta, result)
    model["stress_ratio"] = flume.stress_ratio(case, theta, result)
    model["surface_fractions"] = result.bed.p_active
    return model


def main() -> None:
    """Fit the transport functions on run 2 and test them on runs 1 and 3."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--flow", choices=("backwater", "unsteady"), default="unsteady")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--start", nargs=3, type=float, default=list(START), metavar=("TAU_R", "N", "SHARE"),
                        help="start of the reference-stress multiplier, Manning's n and the share of the load")
    parser.add_argument("--seed", type=int, default=0, help="seed of the initial weights of the network")
    parser.add_argument("--tag", default="", help="added to the name of the result file")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    cases = {run: flume.build(run, "wilcock_crowe", args.flow, "bedload_mix") for run in (lf.TRAIN_RUN, *lf.TEST_RUNS)}
    name_out = f"transport_flume_{args.flow}{'_' + args.tag if args.tag else ''}.json"
    out = {
        "arguments": vars(args), "flow": args.flow, "train_run": lf.TRAIN_RUN, "test_runs": list(lf.TEST_RUNS),
        "percentiles": list(lf.PERCENTILES), "x_m": cases[lf.TRAIN_RUN].x.tolist(),
        "diameters_m": list(flume.DIAMETERS), "stress_ratio": STRESS_RATIOS.tolist(),
        "measured_depth_m": {str(run): data.RUNS[run]["depth_m"] for run in cases},
        "deposit_sections": {str(run): flume.deposit_sections(run).tolist() for run in cases},
        "targets": {str(run): lf.as_lists({k: v for k, v in lf.targets(run).items() if k != "thickness"}) for run in cases},
        "stages": {},
    }
    scalars = {"log_tau_r": jnp.asarray(np.log(args.start[0])), "log_n": jnp.asarray(np.log(args.start[1])),
               "c_interface": jnp.asarray(args.start[2])}
    fitted_by: dict = {}
    for name in args.stages:
        begin = time.perf_counter()
        train = with_transport(cases[lf.TRAIN_RUN], name)
        start = dict(scalars)
        if name == "network":
            # Start from the fitted published curve, with a network close to f = 0.
            origin = fitted_by.get("wilcock_crowe", scalars)
            start = {key: origin[key] for key in scalars}
            network = init_transport_network(jax.random.PRNGKey(args.seed), train.config.sediment)
            noise = jax.random.normal(jax.random.PRNGKey(args.seed + 1), network[-1]["W"].shape)
            network[-1]["W"] = START_OUTPUT_SCALE * noise
            start["transport_network"] = network
        print(f"{name}: fitting", flush=True)
        fitted, history = lf.fit(train, start, args.iterations, depth=True, ftol=NETWORK_FTOL if name == "network" else None)
        fitted_by[name] = fitted
        entry = {
            "n_parameters": int(ravel_pytree(start)[0].size), "misfit_history": history, "free": lf.as_lists(fitted),
            "tau_r_wc": float(np.exp(fitted["log_tau_r"])), "n_manning": float(np.exp(fitted["log_n"])),
            "c_interface": float(fitted["c_interface"]), "transport_function": curve(train.config.sediment, fitted).tolist(),
            "runs": {},
        }
        for run, case in cases.items():
            run_case = with_transport(case, name)
            model = jax.jit(lambda f, c=run_case: evaluate(c, f))(fitted)
            entry["runs"][str(run)] = {"model": lf.as_lists(model), "scores": lf.scores(model, run)}
        entry["seconds"] = time.perf_counter() - begin
        out["stages"][name] = entry
        print(f"{name:13s} {entry['n_parameters']:3d} parameters, {len(history):3d} evaluations, {entry['seconds']:6.0f} s: "
              f"misfit {history[0]:.2f} -> {min(history):.3f}; tau_r_wc {entry['tau_r_wc']:.3f}, n {entry['n_manning']:.4f}, "
              f"share of the load {entry['c_interface']:.3f}", flush=True)
        for run in cases:
            print(f"   run {run} ({'train' if run == lf.TRAIN_RUN else 'test'}): {entry['runs'][str(run)]['scores']}", flush=True)
        (RESULTS / name_out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
