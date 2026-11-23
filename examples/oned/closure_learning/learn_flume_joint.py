"""Fit sediment closures on one or two flume experiments and test them on the others.

Four experiments in the same flume (0.3 m wide) are used (``flume_data.py``): runs 1 to 3
of Seal et al. (1997), which differ only in the feed rate, and run 6 of Toro-Escobar et
al. (2000), with a sandier feed, a quarter of the discharge and a floor five times
steeper. The training runs are run 2, run 6, or both (``--train``); every fitted closure
is then run on all four experiments without any change.

Runs 1 and 3 test a closure only for another feed rate of the same mixture. Run 6 (or
run 2, if run 6 is the training run) tests it for another mixture and another flow, which
is where closures that fit one family of runs equally well differ.

The model: unsteady flow solver, bed shear stress with the mean depth, transfer function
for what the surface layer passes down to the deposit (``flume_case.py``). The flow
resistance is Manning's formula, the variable-power law or the logarithmic law of
Keulegan (``--resistance``); the roughness length of the last two is one length for all
runs, or a factor times the mean grain size or the D90 of the bed surface of each section
(``--roughness``); and the shear stress for the sediment is the total one or the one
reduced for the drag on the largest grains in shallow flow (``--stress``). ``--resistance
keulegan --roughness surface_mean --stress total`` with the closure ``parker`` is the
treatment of Cui (2007).

Closures (``--closures``), fitted one after the other; each is fitted together with the
multiplier of the reference Shields stress, the parameter of the resistance law and the
share of the load in the transfer function:

- ``published``: hiding function and transport function of Wilcock and Crowe (2003);
- ``parker``: the same with the transport function of Parker (1990) in the form used by
  Cui (2007);
- ``power``: hiding factor (d_k / d_g)^(-m_h), one exponent more;
- ``hiding_network``: ln xi_k = f(ln(d_k / d_g)) - f(0), started from the published
  hiding function;
- ``transport_network``: W* = W*_WC(phi) exp(f(ln phi) - f(0)), started close to the
  published transport function.

Measurements of a training run. Runs 1 to 3 (run 2): thickness of the deposit at five
times, four percentiles of the deposit of hours 14 to 22 at eight stations, mean water
depth. Run 6: thickness at five times, three percentiles of the gravel of the deposit of
hours 8 to 18 at the five stations on the wedge, mean water depth. The misfit of a run is
half the mean squared error of each kind, in units of its standard deviation, and the
misfits of the training runs are added. With ``--surface`` the grain sizes of the surface
layer at the end of the run are a further kind of measurement (run 2: D90 and D50 at the
upstream and downstream ends; run 6: three percentiles of the gravel at the five stations
on the wedge). The surface layer is what the hiding function acts on, so its composition
says most about the relative mobility of the sizes; without ``--surface`` it is a test.

With ``--direct FOLDER`` the transport rates of every size fraction measured directly in
another flume (Wilcock et al., 2001; ``wkc_test.py``) are fitted together with the
deposits. The closure (hiding function, transport function) is the same for both kinds of
data; the direct measurements get a multiplier of the reference Shields stress of their
own, because the one of the deposit runs also carries what is particular to that flume
and its flow model. Their misfit is half the mean squared error of log10 W*, in units of
0.5 (the scatter of the published formula about these data), times ``--direct-weight``.
The sediments named with ``--direct-test`` are left out of the fit and scored separately,
so that an outside test remains.

Usage::

    JAX_PLATFORMS=cpu python learn_flume_joint.py --train 2 6 [--closures published power]
        [--resistance variable_power] [--roughness length] [--stress reduced] [--iterations 60]
        [--surface] [--direct FOLDER] [--direct-test j14 j27]
        [--start-from results/joint_....json] [--parallel] [--tag TAG]

``--start-from`` takes the ``published`` stage from another results file instead of
fitting it again. Writes
``results/joint_train<runs>_<resistance>_<stress>[_surfacemean][_surface][_direct][_<tag>].json``.

Units: SI; psi = log2(d in mm) for grain sizes.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from hydrograd.sediment import FlowState, wilcock_crowe_parameters
from hydrograd.sediment.closures import (
    init_transport_network,
    transport_function,
    wilcock_crowe_capacity,
)

import flume_case as flume
import flume_data as data
import learn_flume as lf
import wkc_test as wkc

RESULTS = Path(__file__).parent / "results"
ALL_RUNS = (1, 2, 3, 6)
CLOSURES = ("published", "parker", "power", "hiding_network", "transport_network")
# Run 6: percentiles of the gravel that are compared, standard deviation of the mean
# depth (m; read from a figure), and the sampling stations on the wedge, where the
# deposit of hours 8 to 18 is thick enough to be sampled in the model as in the flume.
PERCENTILES6 = (90, 50, 16)
SIGMA_DEPTH6 = 0.005
# Standard deviation of the grain sizes of the surface layer (psi): larger than that of
# the deposit, because the depth of a surface sample (to the base of the largest grain)
# and the active layer of the model are not the same layer.
SIGMA_SURFACE = 0.5
N_WEDGE_STATIONS = 5
N_SAND = int(np.sum(flume.PSI_LIMITS[1:] <= 1.0))        # classes finer than 2 mm
# Start values: multiplier of the reference Shields stress, Manning's n, roughness length
# (m), share of the load.
START = {"log_tau_r": np.log(1.0), "log_n": np.log(0.029), "log_k": np.log(0.02),
         "log_alpha": np.log(flume.ROUGHNESS_FACTOR), "c_interface": lf.START_SHARE}
# Start of the roughness factor on the D90 of the surface: 2 for the logarithmic law
# (Parker, 1990); 0.5 for the variable-power law, whose roughness length is smaller.
START_D90_FACTOR = {"keulegan": 2.0, "variable_power": 0.5}
START_OUTPUT_SCALE = 0.05
NETWORK_FTOL = 1.0e-12
# Direct measurements: standard deviation of log10 W* (the scatter of the published
# formula about them) and the start of their multiplier of the reference Shields stress.
SIGMA_LOG = 0.5
START_DIRECT = np.log(1.1)
SIZE_RATIOS = np.geomspace(0.01, 30.0, 80)
STRESS_RATIOS = np.geomspace(0.2, 50.0, 140)


def nearest_sections(case, stations) -> np.ndarray:
    """Index of the section nearest to each station (m)."""
    return np.array([int(np.argmin(np.abs(case.x - x))) for x in stations])


def gravel_psi(fractions, percent):
    """Percentiles (psi) of the gravel of mixtures given by class, shape [P, N]."""
    gravel = jnp.concatenate([jnp.zeros_like(fractions[:N_SAND]), fractions[N_SAND:]], axis=0)
    return flume.percentile_psi(gravel + 1.0e-12, list(percent))


def predict6(case, theta, result=None) -> dict:
    """Model values of the measurements of run 6 (thickness and depth in m, sizes in psi)."""
    result = flume.run_case(case, theta) if result is None else result
    thickness, cumulative, surface, depth = flume.observables(case, result)
    stations = nearest_sections(case, data.RUN6_STATIONS)
    sizes_at = nearest_sections(case, data.RUN6_SIZE_X[data.RUN6_SIZE_X <= case.x[-1]])
    first, last = (flume.step_of(hours) - 1 for hours in data.RUN6_DEPOSIT_HOURS)
    laid = jnp.maximum(cumulative[last] - cumulative[first], 0.0) + 1.0e-12
    return {
        "thickness": {hours: thickness[flume.step_of(hours) - 1][stations[: len(values)]]
                      for hours, values in data.RUN6_PROFILES.items()},
        "profile": {hours: thickness[flume.step_of(hours) - 1] for hours in data.RUN6_PROFILES},
        "depth": jnp.mean(depth[stations[:10]]),
        "depth_profile": depth,
        "deposit": gravel_psi(laid, PERCENTILES6)[:, sizes_at],
        "surface": gravel_psi(surface, PERCENTILES6)[:, sizes_at],
        "layer": (thickness[last] - thickness[first])[sizes_at],
    }


def targets6() -> dict:
    """Measurements of run 6 in the layout of ``predict6``."""
    return {
        "thickness": {hours: value for hours, (_, value) in data.run6_thickness_profiles().items()},
        "deposit": np.array([data.RUN6_DEPOSIT_GRAVEL_PSI[p] for p in PERCENTILES6]),
        "surface": np.array([data.RUN6_SURFACE_GRAVEL_PSI[p] for p in PERCENTILES6]),
        "depth": data.RUN6["depth_m"],
    }


def misfit6(model: dict, surface: bool = False):
    """Misfit of run 6: thickness, gravel sizes of the deposit on the wedge, mean depth,
    and the gravel sizes of the surface on the wedge if ``surface`` is set."""
    target = targets6()
    thickness = jnp.concatenate([model["thickness"][hours] - value for hours, value in target["thickness"].items()])
    sizes = (model["deposit"][:, :N_WEDGE_STATIONS] - target["deposit"][:, :N_WEDGE_STATIONS]).ravel()
    value = 0.5 * (jnp.mean((thickness / lf.SIGMA_THICKNESS) ** 2) + jnp.mean((sizes / lf.SIGMA_PSI) ** 2))
    value = value + 0.5 * ((model["depth"] - target["depth"]) / SIGMA_DEPTH6) ** 2
    if surface:
        top = (model["surface"][:, :N_WEDGE_STATIONS] - target["surface"][:, :N_WEDGE_STATIONS]).ravel()
        value = value + 0.5 * jnp.mean((top / SIGMA_SURFACE) ** 2)
    return value


def misfit2(case, model: dict, surface: bool = False):
    """Misfit of run 2 (``learn_flume.misfit`` with the water depth), and the grain sizes
    of the surface layer if ``surface`` is set."""
    target = lf.targets(case.run)
    thickness = jnp.concatenate([model["thickness"][hours] - value for hours, (_, value) in target["thickness"].items()])
    sizes = jnp.concatenate([model["table1"][i] - data.TABLE1_PSI[p] for i, p in enumerate(lf.PERCENTILES)])
    value = 0.5 * (jnp.mean((thickness / lf.SIGMA_THICKNESS) ** 2) + jnp.mean((sizes / lf.SIGMA_PSI) ** 2))
    value = value + lf.depth_term(model, case.run)
    if surface:
        top = (model["surface"] - target["surface"]).ravel()
        value = value + 0.5 * jnp.mean((top / SIGMA_SURFACE) ** 2)
    return value


def scores6(model: dict) -> dict:
    """Root-mean-square errors of run 6."""
    target = targets6()
    thickness = np.concatenate([np.asarray(model["thickness"][hours]) - value for hours, value in target["thickness"].items()])
    n_sizes = np.asarray(model["deposit"]).shape[1]
    wedge = np.asarray(model["deposit"])[:, :N_WEDGE_STATIONS] - target["deposit"][:, :N_WEDGE_STATIONS]
    surface = np.asarray(model["surface"]) - target["surface"][:, :n_sizes]
    return {
        "rms_thickness_m": float(np.sqrt(np.mean(thickness**2))),
        "upstream_thickness_m": float(np.asarray(model["thickness"][data.RUN6["hours"]])[0]),
        "rms_deposit_psi": float(np.sqrt(np.mean(wedge**2))),
        "rms_surface_psi": float(np.sqrt(np.mean(surface**2))),
        "depth_m": float(model["depth"]),
    }


def run_misfit(case, free: dict, surface: bool = False):
    """Misfit of one training run (run 2 or run 6)."""
    theta = lf.parameters(case, free)
    if case.run == 6:
        return misfit6(predict6(case, theta), surface)
    return misfit2(case, lf.predict(case, theta), surface)


def evaluate(case, free: dict, surface: bool = False) -> dict:
    """Model values of the measurements of a run and its misfit as a training run."""
    theta = lf.parameters(case, free)
    if case.run == 6:
        model = predict6(case, theta)
        model["misfit"] = misfit6(model, surface)
        return model
    model = lf.predict(case, theta)
    if case.run == lf.TRAIN_RUN:
        model["misfit"] = misfit2(case, model, surface)
    return model


class Direct:
    """Directly measured transport rates of size fractions (``wkc_test.measurements``).

    Attributes:
        tau: bed shear stress of the runs (Pa), shape [R].
        fractions: surface fractions, coarsest class first (-), shape [K, R].
        observed: log10 of the observed W* of every fraction, shape [K, R]; 0 where there
            is none.
        valid: True where a fraction was measured, shape [K, R].
        train: True for the runs used for fitting, shape [R].
        sediment: name of the sediment of each run.
    """

    def __init__(self, folder: Path, test_sediments: list[str]):
        runs = wkc.measurements(folder)
        self.tau = jnp.asarray([run["tau"] for run in runs])
        self.fractions = jnp.asarray(np.array([run["F"] for run in runs]).T)
        observed = np.array([run["observed"] for run in runs]).T
        self.valid = np.isfinite(observed)
        self.observed = jnp.asarray(np.log10(np.where(self.valid, observed, 1.0)))
        self.sediment = [run["sediment"] for run in runs]
        self.train = np.array([name not in test_sediments for name in self.sediment])

    def log_ratio(self, config, free: dict):
        """log10 of predicted over observed W*, shape [K, R] (0 where not measured)."""
        settings = dataclasses.replace(config, constants=dataclasses.replace(config.constants, rho_s=wkc.RHO_S))
        theta = {**wilcock_crowe_parameters(), "tau_r_wc": jnp.exp(free["log_tau_r_direct"])}
        for key in ("m_h", "hiding_network", "transport_network"):
            if key in free:
                theta[key] = free[key]
        u_star = jnp.sqrt(self.tau / wkc.RHO)
        ones = jnp.ones_like(self.tau)
        flow = FlowState(discharge=ones, width=ones, depth=ones, shear_stress=self.tau, shear_velocity=u_star,
                         friction_coefficient=ones)
        capacity = wilcock_crowe_capacity(flow, jnp.asarray(wkc.SIZE), self.fractions, theta, settings)
        safe = jnp.where(self.valid, self.fractions, 1.0)
        predicted = (wkc.RHO_S / wkc.RHO - 1.0) * wkc.G * capacity / safe / u_star[None, :] ** 3
        return jnp.where(self.valid, jnp.log10(jnp.maximum(predicted, 1.0e-30)) - self.observed, 0.0)

    def misfit(self, config, free: dict):
        """Half the mean squared error of log10 W* of the training runs, in units of SIGMA_LOG."""
        mask = self.valid & self.train[None, :]
        ratio = self.log_ratio(config, free)
        return 0.5 * jnp.sum(jnp.where(mask, (ratio / SIGMA_LOG) ** 2, 0.0)) / mask.sum()

    def scores(self, config, free: dict) -> dict:
        """Scatter of the log ratio for the runs used for fitting and for those left out."""
        ratio = np.asarray(self.log_ratio(config, free))
        relative = wkc.SIZE[:, None] / np.exp(np.sum(np.asarray(self.fractions) * np.log(wkc.SIZE)[:, None], axis=0))[None, :]
        out = {}
        for name, runs in (("train", self.train), ("test", ~self.train)):
            mask = self.valid & runs[None, :]
            if mask.sum() == 0:
                continue
            values = ratio[mask]
            sizes = np.broadcast_to(relative, ratio.shape)[mask]
            out[name] = {
                "n": int(values.size), "rms_log10": float(np.sqrt(np.mean(values**2))),
                "within_2": float(np.mean(np.abs(values) < np.log10(2.0))),
                "median_ratio_fine": float(10.0 ** np.median(values[sizes < 0.5])),
                "median_ratio_middle": float(10.0 ** np.median(values[(sizes >= 0.5) & (sizes <= 2.0)])),
                "median_ratio_coarse": float(10.0 ** np.median(values[sizes > 2.0])),
            }
        return out


def fit(cases: list, start: dict, iterations: int, ftol: float | None = None, parallel: bool = False,
        surface: bool = False, direct: Direct | None = None, direct_weight: float = 1.0):
    # ``direct``: the direct measurements to fit with the runs, for the sediment settings of
    # the first case.
    """Minimize the sum of the misfits of the training runs (and of the direct measurements).

    The runs are differentiated one after the other, so that only one run is in memory at
    a time, or side by side in threads (``parallel``; faster with two cores, twice the
    memory). Returns the free parameters and the history of the misfit.
    """
    flat_start, unravel = ravel_pytree(start)
    # Limits of each entry of the flat vector (ravel_pytree orders a dictionary by its
    # sorted keys).
    bounds = []
    for key in sorted(start):
        size = int(ravel_pytree(start[key])[0].size)
        bounds += [lf.LIMITS.get(key, (None, None))] * size
    gradients = [jax.jit(jax.value_and_grad(lambda flat, c=case: run_misfit(c, unravel(flat), surface))) for case in cases]
    if direct is not None:
        sediment = cases[0].config.sediment
        gradients.append(jax.jit(jax.value_and_grad(lambda flat: direct_weight * direct.misfit(sediment, unravel(flat)))))
    history = []

    pool = ThreadPoolExecutor(max_workers=len(gradients)) if parallel and len(gradients) > 1 else None

    def objective(flat):
        value, gradient = 0.0, np.zeros(flat.size)
        point = jnp.asarray(flat)
        parts = pool.map(lambda function: function(point), gradients) if pool else (f(point) for f in gradients)
        for v, g in parts:
            value += float(v)
            gradient += np.asarray(g, dtype=float)
        value = value if np.isfinite(value) else 1.0e6
        gradient = np.nan_to_num(gradient)
        history.append(value)
        print(f"  evaluation {len(history):3d}: misfit {value:.5f}, gradient norm {np.linalg.norm(gradient):.2e}", flush=True)
        return value, gradient

    options = {"maxiter": iterations} if ftol is None else {"maxiter": iterations, "ftol": ftol}
    solution = minimize(objective, np.asarray(flat_start), jac=True, method="L-BFGS-B", bounds=bounds, options=options)
    return unravel(jnp.asarray(solution.x)), history


def with_closure(case, closure: str):
    """The case with the settings of a closure."""
    if closure == "power":
        return case
    if closure == "hiding_network":
        return flume.with_hiding(case, "network", hiding_size_ratio_range=lf.SIZE_RATIO_RANGE)
    if closure == "transport_network":
        return flume.with_hiding(case, "wilcock_crowe", transport_function="network")
    if closure == "parker":
        return flume.with_hiding(case, "wilcock_crowe", transport_function="parker")
    return flume.with_hiding(case, "wilcock_crowe")


def as_arrays(tree):
    """Nested lists and dictionaries of a results file as arrays."""
    if isinstance(tree, dict):
        return {key: as_arrays(value) for key, value in tree.items()}
    if isinstance(tree, list) and tree and isinstance(tree[0], dict):
        return [as_arrays(value) for value in tree]
    return jnp.asarray(tree)


def main() -> None:
    """Fit the closures on the training runs and score them on all runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=int, choices=(2, 6), default=[2, 6])
    parser.add_argument("--closures", nargs="+", choices=CLOSURES, default=["published"])
    parser.add_argument("--resistance", choices=("manning", "variable_power", "keulegan"), default="variable_power")
    parser.add_argument("--roughness", choices=("length", "surface_mean", "surface_d90"), default="length",
                        help="roughness length: one length, or a factor times the mean size or the D90 of the bed surface")
    parser.add_argument("--stress", choices=("total", "reduced"), default="reduced")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--start-from", type=Path, default=None, help="results file with a fitted 'published' stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel", action="store_true", help="differentiate the training runs side by side")
    parser.add_argument("--surface", action="store_true", help="use the grain sizes of the surface layer for fitting")
    parser.add_argument("--direct", type=Path, default=None, help="folder of the direct measurements (see wkc_test.py)")
    parser.add_argument("--direct-weight", type=float, default=1.0)
    parser.add_argument("--direct-test", nargs="*", default=["j14", "j27"], choices=wkc.SEDIMENTS,
                        help="sediments of the direct measurements that are left out of the fit")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    name_out = f"joint_train{''.join(str(r) for r in sorted(args.train))}_{args.resistance}_{args.stress}"
    name_out += {"length": "", "surface_mean": "_surfacemean", "surface_d90": "_surfaced90"}[args.roughness]
    name_out += "_surface" if args.surface else ""
    name_out += "_direct" if args.direct else ""
    name_out += f"_{args.tag}" if args.tag else ""

    cases = {run: flume.build(run, "wilcock_crowe", "unsteady", "bedload_mix", resistance=args.resistance,
                              transport_stress=args.stress, roughness_scale=args.roughness) for run in ALL_RUNS}
    roughness = "log_n" if args.resistance == "manning" else "log_k" if args.roughness == "length" else "log_alpha"
    roughness_name = {"log_n": "n", "log_k": "k", "log_alpha": "factor"}[roughness]
    scalars = {key: jnp.asarray(START[key]) for key in ("log_tau_r", roughness, "c_interface")}
    if args.roughness == "surface_d90":
        scalars["log_alpha"] = jnp.asarray(np.log(START_D90_FACTOR[args.resistance]))
    direct = Direct(args.direct, args.direct_test) if args.direct else None
    if direct is not None:
        scalars["log_tau_r_direct"] = jnp.asarray(START_DIRECT)
    out = {
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "train_runs": sorted(args.train), "resistance": args.resistance, "stress": args.stress, "surface_data": args.surface,
        "roughness_scale": args.roughness,
        "direct_data": args.direct is not None, "direct_test": args.direct_test if args.direct else [],
        "x_m": {str(run): case.x.tolist() for run, case in cases.items()}, "percentiles6": list(PERCENTILES6),
        "size_ratio": SIZE_RATIOS.tolist(), "stress_ratio": STRESS_RATIOS.tolist(), "stages": {},
    }
    fitted_by: dict = {}
    if args.start_from is not None:
        earlier = json.loads(args.start_from.read_text())["stages"]["published"]
        fitted_by["published"] = as_arrays(earlier["free"])
        out["stages"]["published"] = earlier
        print(f"published: taken from {args.start_from}", flush=True)

    for closure in args.closures:
        if closure in fitted_by:
            continue
        begin = time.perf_counter()
        training = [with_closure(cases[run], closure) for run in sorted(args.train)]
        config = training[0].config.sediment
        origin = fitted_by.get("published", scalars)
        start = {key: origin.get(key, scalars[key]) for key in scalars}
        ftol = None
        if closure == "power":
            start["m_h"] = jnp.asarray(-0.5)
        elif closure == "hiding_network":
            start["hiding_network"] = lf.network_like(config, with_closure(cases[2], "published").config.sediment, {})
            ftol = NETWORK_FTOL
        elif closure == "transport_network":
            network = init_transport_network(jax.random.PRNGKey(args.seed), config)
            noise = jax.random.normal(jax.random.PRNGKey(args.seed + 1), network[-1]["W"].shape)
            network[-1]["W"] = START_OUTPUT_SCALE * noise
            start["transport_network"] = network
            ftol = NETWORK_FTOL
        print(f"{closure}: fitting on run(s) {sorted(args.train)}", flush=True)
        fitted, history = fit(training, start, args.iterations, ftol, args.parallel, args.surface, direct, args.direct_weight)
        fitted_by[closure] = fitted
        entry = {
            "n_parameters": int(ravel_pytree(start)[0].size), "misfit_history": history, "free": lf.as_lists(fitted),
            "tau_r_wc": float(np.exp(fitted["log_tau_r"])), "c_interface": float(fitted["c_interface"]),
            "resistance_parameter": float(np.exp(fitted[roughness])), "runs": {},
        }
        if closure == "power":
            entry["m_h"] = float(fitted["m_h"])
        hiding_theta = {k: v for k, v in fitted.items() if k in ("m_h", "hiding_network")}
        entry["hiding"] = lf.hiding_curve(config, hiding_theta, SIZE_RATIOS).tolist()
        transport_theta = {k: v for k, v in fitted.items() if k == "transport_network"}
        entry["transport_function"] = np.asarray(transport_function(jnp.asarray(STRESS_RATIOS), transport_theta, config)).tolist()
        if direct is not None:
            entry["tau_r_direct"] = float(np.exp(fitted["log_tau_r_direct"]))
            entry["direct"] = direct.scores(config, fitted)
        for run, case in cases.items():
            run_case = with_closure(case, closure)
            model = jax.jit(lambda f, c=run_case: evaluate(c, f, args.surface))(fitted)
            scores = scores6(model) if run == 6 else lf.scores(model, run)
            if "misfit" in model:
                scores["misfit"] = float(model["misfit"])
            entry["runs"][str(run)] = {"model": lf.as_lists(model), "scores": scores}
        entry["seconds"] = time.perf_counter() - begin
        out["stages"][closure] = entry
        extra = f", m_h {entry['m_h']:.3f}" if closure == "power" else ""
        print(f"{closure:17s} {entry['n_parameters']:3d} parameters, {len(history):3d} evaluations, {entry['seconds']:6.0f} s: "
              f"misfit {history[0]:.3f} -> {min(history):.3f}; tau_r_wc {entry['tau_r_wc']:.3f}, "
              f"{roughness_name} {entry['resistance_parameter']:.4f}, share {entry['c_interface']:.3f}{extra}", flush=True)
        for run in cases:
            kind = "train" if run in args.train else "test"
            print(f"   run {run} ({kind}): " + ", ".join(f"{k} {v:.4f}" for k, v in entry["runs"][str(run)]["scores"].items()), flush=True)
        if direct is not None:
            for kind, values in entry["direct"].items():
                print(f"   direct measurements ({kind}), multiplier {entry['tau_r_direct']:.3f}: "
                      + ", ".join(f"{k} {v:.3f}" for k, v in values.items()), flush=True)
        (RESULTS / f"{name_out}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
