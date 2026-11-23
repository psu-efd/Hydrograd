"""Test of fitted closures against transport rates measured directly in another flume.

Wilcock, Kenworthy and Crowe (2001) measured the transport rate of every size fraction in
48 runs of a recirculating flume with five sediments of 6 to 34 % sand, together with the
size distribution of the bed surface and the bed shear stress. For a fraction i the
measurements give the dimensionless transport rate

    W*_i = (s - 1) g q_b,i / (F_i u_*^3),

with q_b,i the volume transport rate per unit width, F_i the fraction on the surface and
u_* the shear velocity. A closure (reference stress, hiding function, transport function)
predicts W*_i from the surface composition and the shear stress alone, without any flow
or bed model. The closures fitted through the solver to the deposits of the Seal et al.
and Toro-Escobar et al. runs (``learn_flume_joint.py`` and the other fit scripts) never
saw these data; the formula of Wilcock and Crowe (2003) was derived from them and is the
reference.

Each closure is scored twice: with the multiplier of the reference Shields stress that
was fitted in the other flume, and with the multiplier that suits these data best (one
number), which leaves the shape of the hiding and transport functions to be judged. The
second score is the test: the multiplier fitted in the other flume also carries what is
particular to that flume and its model (no side-wall correction of the shear stress
there, a reduction of the shear stress for shallow flow), and a constant factor on the
shear stress is the same as a multiplier of the reference stress. The shear stress of
these runs is used as published, corrected for the side walls and without any reduction.

Data: the electronic supplement of the paper (files ``2001wr000683-<sediment>flow.txt``,
``...surf.txt``, ``...trans.txt``; pass the folder with ``--data``), which is not
distributed with this example, and the bed shear stress of each run from Table 2 of the
paper (``SHEAR_STRESS`` below).

Wilcock, P. R., Kenworthy, S. T., and Crowe, J. C. (2001). Experimental study of the
transport of mixed sand and gravel. Water Resources Research 37(12), 3349-3358.

Usage::

    JAX_PLATFORMS=cpu python wkc_test.py --data FOLDER [--results results/joint_....json ...]

Writes ``results/wkc_test.json`` and prints one line per closure.

Units: SI; sizes of the supplement in mm, transport rates in g/(m s).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from hydrograd.sediment import FlowState, SedimentConfig, wilcock_crowe_parameters
from hydrograd.sediment.closures import wilcock_crowe_capacity

import learn_flume as lf

RESULTS = Path(__file__).parent / "results"
DEFAULT_DATA = Path(__file__).resolve().parents[3] / "background" / "Wilcox_Crowe_data"
RHO, G, RHO_S = 1000.0, 9.81, 2610.0
# Sieve classes of the supplement (mm): "retained on" sizes and the next larger sieve.
LOWER_MM = np.array([45.3, 32.0, 22.6, 16.0, 11.3, 8.0, 5.66, 4.0, 2.83, 2.0, 1.41, 1.0, 0.5, 0.21])
UPPER_MM = np.array([64.0, 45.3, 32.0, 22.6, 16.0, 11.3, 8.0, 5.66, 4.0, 2.83, 2.0, 1.41, 1.0, 0.5])
SIZE = 1.0e-3 * np.sqrt(LOWER_MM * UPPER_MM)          # m, geometric mean of each class
SEDIMENTS = ("j06", "j14", "j21", "j27", "bmc")
# Bed shear stress (Pa, corrected for the side walls) of each run: Table 2 of the paper.
SHEAR_STRESS = {
    "J06.1": 4.10, "J06.2": 4.90, "J06.3": 8.70, "J06.4": 11.3, "J06.5": 6.18, "J06.6": 8.29, "J06.7": 16.0,
    "J06.8": 5.42, "J06.9": 17.5, "J06.10": 23.6,
    "J14.1": 16.5, "J14.2": 19.1, "J14.3": 6.43, "J14.4": 9.74, "J14.5": 16.1, "J14.6": 4.38, "J14.7": 8.63,
    "J14.8": 7.27, "J14.9": 20.1,
    "J21.1": 15.9, "J21.2": 4.07, "J21.3": 6.64, "J21.4": 10.8, "J21.5": 3.35, "J21.6": 7.21, "J21.7": 2.82,
    "J21.8": 16.1, "J21.9": 18.6,
    "J27.1": 2.78, "J27.2": 6.91, "J27.3": 1.10, "J27.4": 2.50, "J27.5": 6.57, "J27.6": 3.96, "J27.7": 7.91,
    "J27.8": 9.46, "J27.9": 11.5, "J27.10": 17.4,
    "BMC14c": 0.57, "BMC7a": 0.85, "BMC14b": 0.86, "BMC7b": 1.07, "BMC7c": 1.60, "BMC1": 1.83, "BMC2": 3.14,
    "BMC6": 5.94, "BMC4": 6.47, "BMC5": 13.1,
}


def number(cell: str) -> float:
    """A cell of the supplement as a number; nan where it is empty or not a number."""
    try:
        return float(cell)
    except ValueError:
        return float("nan")


def table(folder: Path, name: str) -> tuple[list[str], dict[str, list[float]]]:
    """Header and rows {run: numbers} of a file of the supplement."""
    text = (folder / f"2001wr000683-{name}.txt").read_text(errors="replace").replace("\r", "\n")
    lines = [line for line in text.split("\n") if line.strip()]
    rows = {line.split("\t")[0].strip(): [number(cell) for cell in line.split("\t")[1:]] for line in lines[1:]}
    return lines[0].split("\t"), rows


def measurements(folder: Path) -> list[dict]:
    """The runs with a complete set of measurements.

    Returns:
        One dictionary per run: name, sediment, bed shear stress (Pa), flow depth (m),
        surface fractions F (14 classes, coarsest first), and the observed W* of the
        fractions that are present in both the surface and the transport (nan elsewhere).
    """
    runs = []
    for sediment in SEDIMENTS:
        header, flow = table(folder, sediment + "flow")
        _, transport = table(folder, sediment + "trans")
        _, surface = table(folder, sediment + "surf")
        depth_column = header.index("Depth") - 1
        for name, values in flow.items():
            if name not in SHEAR_STRESS or name not in transport or name not in surface or len(values) < len(header) - 1:
                continue
            total = values[len(header) - 2]                               # g/(m s), last column
            if not np.isfinite(total):
                continue
            p, F = np.array(transport[name])[:14], np.array(surface[name])[:14]
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(F)):
                continue
            tau = SHEAR_STRESS[name]
            u_star = np.sqrt(tau / RHO)
            present = (p > 0.0) & (F > 0.0)
            rate = total / 1000.0 / RHO_S * p / p.sum()                   # m^2/s of solids per class
            observed = np.where(present, (RHO_S / RHO - 1.0) * G * rate / np.where(present, F / F.sum(), 1.0) / u_star**3, np.nan)
            runs.append({"name": name, "sediment": sediment, "tau": tau, "depth": values[depth_column],
                         "F": F / F.sum(), "observed": observed})
    return runs


def closures_of(path: Path) -> list[tuple[str, SedimentConfig, dict]]:
    """Closures of a results file: (label, sediment settings, parameters)."""
    out = json.loads(path.read_text())
    base = SedimentConfig(diameters=tuple(SIZE), cohesive_dmax=0.0, transport_formula="wilcock_crowe")
    base = dataclasses.replace(base, constants=dataclasses.replace(base.constants, rho_s=RHO_S))
    listed = []
    stages = out.get("stages", out.get("closures"))
    for name, entry in stages.items():
        if "free" not in entry:
            continue
        free = entry["free"]
        settings = {"hiding_form": "wilcock_crowe"}
        theta = {**wilcock_crowe_parameters(), "tau_r_wc": jnp.exp(jnp.asarray(free["log_tau_r"]))}
        if name in ("power", "none"):
            settings = {"hiding_form": "power"}
            theta["m_h"] = jnp.asarray(free.get("m_h", 0.0))
        elif name in ("hiding_network", "network") and "hiding_network" in free:
            settings = {"hiding_form": "network", "hiding_size_ratio_range": lf.SIZE_RATIO_RANGE}
            theta["hiding_network"] = [{k: jnp.asarray(v) for k, v in layer.items()} for layer in free["hiding_network"]]
        if "transport_network" in free:
            settings["transport_function"] = "network"
            theta["transport_network"] = [{k: jnp.asarray(v) for k, v in layer.items()} for layer in free["transport_network"]]
        elif name == "parker":
            settings["transport_function"] = "parker"
        listed.append((f"{path.stem}: {name}", dataclasses.replace(base, **settings), theta))
    return listed


def log_ratios(runs: list[dict], config: SedimentConfig, theta: dict, multiplier: float | None = None):
    """log10 of predicted over observed W* of every fraction, with the size relative to
    the mean size of the surface and the sediment of each."""
    ratios, relative, sediment = [], [], []
    theta = theta if multiplier is None else {**theta, "tau_r_wc": jnp.asarray(multiplier)}
    for run in runs:
        tau = run["tau"]
        u_star = np.sqrt(tau / RHO)
        flow = FlowState(discharge=jnp.ones(1), width=jnp.ones(1), depth=jnp.ones(1), shear_stress=jnp.asarray([tau]),
                         shear_velocity=jnp.asarray([u_star]), friction_coefficient=jnp.ones(1))
        F = run["F"]
        capacity = np.asarray(wilcock_crowe_capacity(flow, jnp.asarray(SIZE), jnp.asarray(F)[:, None], theta, config))[:, 0]
        predicted = (RHO_S / RHO - 1.0) * G * capacity / np.where(F > 0.0, F, np.nan) / u_star**3
        ok = np.isfinite(run["observed"]) & (predicted > 0.0)
        mean_size = np.exp(np.sum(F * np.log(SIZE)))
        ratios.append(np.log10(predicted[ok] / run["observed"][ok]))
        relative.append(SIZE[ok] / mean_size)
        sediment += [run["sediment"]] * int(ok.sum())
    return np.concatenate(ratios), np.concatenate(relative), np.array(sediment)


def scores(ratio: np.ndarray, relative: np.ndarray) -> dict:
    """Summary of the log ratios: scatter, and bias by relative grain size."""
    return {
        "n": int(ratio.size),
        "median_ratio": float(10.0 ** np.median(ratio)),
        "rms_log10": float(np.sqrt(np.mean(ratio**2))),
        "within_2": float(np.mean(np.abs(ratio) < np.log10(2.0))),
        "within_5": float(np.mean(np.abs(ratio) < np.log10(5.0))),
        "median_ratio_fine": float(10.0 ** np.median(ratio[relative < 0.5])),
        "median_ratio_middle": float(10.0 ** np.median(ratio[(relative >= 0.5) & (relative <= 2.0)])),
        "median_ratio_coarse": float(10.0 ** np.median(ratio[relative > 2.0])),
    }


def best_multiplier(runs: list[dict], config: SedimentConfig, theta: dict) -> float:
    """Multiplier of the reference Shields stress with the least root-mean-square log ratio."""
    candidates = np.geomspace(0.3, 4.0, 53)
    errors = [np.sqrt(np.mean(log_ratios(runs, config, theta, m)[0] ** 2)) for m in candidates]
    return float(candidates[int(np.argmin(errors))])


def main() -> None:
    """Score the published formula and the closures of the given results files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    runs = measurements(args.data)
    print(f"{len(runs)} runs with complete measurements")
    base = SedimentConfig(diameters=tuple(SIZE), cohesive_dmax=0.0, transport_formula="wilcock_crowe", hiding_form="wilcock_crowe")
    base = dataclasses.replace(base, constants=dataclasses.replace(base.constants, rho_s=RHO_S))
    listed = [("Wilcock and Crowe (2003)", base, wilcock_crowe_parameters())]
    for path in args.results:
        listed += closures_of(path)
    out = {}
    for label, config, theta in listed:
        fitted = scores(*log_ratios(runs, config, theta)[:2])
        multiplier = best_multiplier(runs, config, theta)
        ratio, relative, sediment = log_ratios(runs, config, theta, multiplier)
        shape = scores(ratio, relative)
        out[label] = {"multiplier_fitted": float(theta["tau_r_wc"]), "fitted": fitted, "multiplier_best": multiplier, "best": shape,
                      "log10_ratio": ratio.tolist(), "relative_size": relative.tolist(), "sediment": sediment.tolist()}
        print(f"{label}\n   multiplier as fitted ({float(theta['tau_r_wc']):.2f}): median predicted/observed {fitted['median_ratio']:.2f}, "
              f"within a factor of 2: {100 * fitted['within_2']:.0f} %, of 5: {100 * fitted['within_5']:.0f} %, rms log10 {fitted['rms_log10']:.2f}\n"
              f"   best multiplier ({multiplier:.2f}): median {shape['median_ratio']:.2f}, within 2: {100 * shape['within_2']:.0f} %, "
              f"within 5: {100 * shape['within_5']:.0f} %, rms log10 {shape['rms_log10']:.2f}; median by size (fine / middle / coarse): "
              f"{shape['median_ratio_fine']:.2f} / {shape['median_ratio_middle']:.2f} / {shape['median_ratio_coarse']:.2f}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "wkc_test.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
