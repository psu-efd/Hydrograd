"""Test of fitted closures on an experiment that was not used for fitting.

Run 6 of Toro-Escobar et al. (2000) used the flume of the runs of Seal et al. (1997) with
other conditions: a sandier feed (53 % sand instead of 33 %), a quarter of the water
discharge, a floor five times steeper and a low tailgate, so that no pond and no gravel
front formed (``flume_data.py``). The closures fitted on run 2 (``learn_flume.py``,
``learn_flume_transport.py``) are applied without any change; nothing is fitted here.

Compared with the measurements: the thickness of the deposit at the survey sections and
times, the grain sizes of the gravel (coarser than 2 mm) of the deposit laid down between
hours 8 and 18 and of the surface layer at the end, and the mean water depth over the
deposit at the end.

Usage::

    JAX_PLATFORMS=cpu python predict_run6.py --results results/transport_flume_unsteady.json
    JAX_PLATFORMS=cpu python predict_run6.py --published 1.0 0.029 0.7 [--flow unsteady]
                                             [--feed KG_PER_MIN] [--tag TAG]

With ``--results`` every stage of the file is run with its fitted parameters. With
``--published`` the published closures are run (Wilcock-Crowe transport and hiding
functions, transfer function) with the given multiplier of the reference Shields stress,
Manning's n and share of the load. ``--feed`` replaces the feed rate (the paper gives two
values, see ``flume_data.py``). Writes ``results/run6_<name>.json`` and ``.png``.

Units: SI; psi = log2(d in mm) for grain sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import flume_case as flume
import flume_data as data
import learn_flume as lf

RESULTS = Path(__file__).parent / "results"
PERCENTILES = (90, 50, 16)
N_SAND = int(np.sum(flume.PSI_LIMITS[1:] <= 1.0))        # classes finer than 2 mm
# Least thickness laid down between two times for its grain sizes to be compared (m).
MIN_LAYER = 0.005
COLORS = ("#1baf7a", "#2a78d6", "#eb6834", "#8a8985")
INK = "#0b0b0b"
TRANSPORT_LABELS = {"wilcock_crowe": "Wilcock-Crowe curve", "parker": "Parker curve", "network": "learned curve"}
HIDING_LABELS = {"none": "no hiding", "published": "published hiding", "power": "power-law hiding", "network": "learned hiding"}


def as_arrays(tree):
    """Nested lists and dictionaries of a results file as arrays."""
    if isinstance(tree, dict):
        return {key: as_arrays(value) for key, value in tree.items()}
    if isinstance(tree, list) and tree and isinstance(tree[0], dict):
        return [as_arrays(value) for value in tree]
    return jnp.asarray(tree)


def free_parameters(entry: dict):
    """Fitted parameters of an entry of a results file, or None if they are not in it.

    Files written before the entry ``free`` existed hold the fitted numbers only, which is
    enough for the closures without a network.
    """
    if "free" in entry:
        return as_arrays(entry["free"])
    if "tau_r_wc" not in entry or entry["n_parameters"] > 4:
        return None
    free = {"log_tau_r": jnp.log(entry["tau_r_wc"]), "log_n": jnp.log(entry["n_manning"])}
    for key in ("m_h", "c_interface"):
        if key in entry:
            free[key] = jnp.asarray(entry[key])
    return free


def entries(args) -> tuple[str, list]:
    """Name of the output and the closures to run: (label, case of run 6, free parameters)."""
    if args.results is None:
        tau_r, n_manning, share = args.published
        case = flume.with_hiding(flume.build(6, "wilcock_crowe", args.flow, "bedload_mix", args.feed), "wilcock_crowe")
        free = {"log_tau_r": jnp.log(tau_r), "log_n": jnp.log(n_manning), "c_interface": jnp.asarray(share)}
        return "published", [("published closures", case, free)]
    out = json.loads(args.results.read_text())
    listed = []
    if "stages" in out:
        base = flume.build(6, "wilcock_crowe", out["flow"], "bedload_mix", args.feed)
        for stage, entry in out["stages"].items():
            case = flume.with_hiding(base, "wilcock_crowe", transport_function=stage)
            listed.append((TRANSPORT_LABELS[stage], case, free_parameters(entry)))
    else:
        base = flume.build(6, out["formula"], out["flow"], out["exchange"], args.feed)
        forms = {
            "none": base, "power": base, "published": flume.with_hiding(base, "wilcock_crowe"),
            "network": flume.with_hiding(base, "network", hiding_size_ratio_range=lf.SIZE_RATIO_RANGE),
        }
        for closure, entry in out["closures"].items():
            listed.append((HIDING_LABELS[closure], forms[closure], free_parameters(entry)))
    return args.results.stem, [item for item in listed if item[2] is not None]


def gravel_psi(fractions: np.ndarray) -> np.ndarray:
    """Percentiles ``PERCENTILES`` (psi) of the gravel of mixtures, shape [P, N]."""
    gravel = np.array(fractions)
    gravel[:N_SAND] = 0.0
    return np.asarray(flume.percentile_psi(jnp.asarray(gravel + 1.0e-12), list(PERCENTILES)))


def simulate(case, free: dict) -> dict:
    """Run the case and return what is compared with the measurements."""
    theta = lf.parameters(case, free)
    result = jax.jit(lambda t: flume.run_case(case, t))(theta)
    thickness, cumulative, surface, depth = (np.asarray(v) for v in flume.observables(case, result))
    stations = np.array([int(np.argmin(np.abs(case.x - x))) for x in data.RUN6_STATIONS])
    sizes_at = np.array([int(np.argmin(np.abs(case.x - x))) for x in data.RUN6_SIZE_X if x <= case.x[-1]])
    first, last = (flume.step_of(hours) - 1 for hours in data.RUN6_DEPOSIT_HOURS)
    laid = np.maximum(cumulative[last] - cumulative[first], 0.0)
    layer = thickness[last] - thickness[first]
    deposit = gravel_psi(laid / np.maximum(laid.sum(axis=0, keepdims=True), 1.0e-30))[:, sizes_at]
    deposit[:, layer[sizes_at] < MIN_LAYER] = np.nan
    return {
        "x_m": case.x.tolist(),
        "thickness": {str(hours): thickness[flume.step_of(hours) - 1].tolist() for hours in data.RUN6_PROFILES},
        "thickness_at_stations": {str(hours): thickness[flume.step_of(hours) - 1][stations[: len(values)]].tolist()
                                  for hours, values in data.RUN6_PROFILES.items()},
        "depth": depth.tolist(),
        "depth_m": float(np.mean(depth[stations[:10]])),
        "deposit_gravel_psi": deposit.tolist(),
        "surface_gravel_psi": gravel_psi(surface)[:, sizes_at].tolist(),
        "sand_in_surface": surface[:N_SAND].sum(axis=0).tolist(),
        "size_x_m": case.x[sizes_at].tolist(),
    }


def scores(model: dict) -> dict:
    """Root-mean-square errors of run 6."""
    measured = data.run6_thickness_profiles()
    thickness = np.concatenate([np.asarray(model["thickness_at_stations"][str(hours)]) - value for hours, (_, value) in measured.items()])
    n_sizes = len(model["size_x_m"])
    deposit = np.asarray(model["deposit_gravel_psi"]) - np.array([data.RUN6_DEPOSIT_GRAVEL_PSI[p][:n_sizes] for p in PERCENTILES])
    surface = np.asarray(model["surface_gravel_psi"]) - np.array([data.RUN6_SURFACE_GRAVEL_PSI[p][:n_sizes] for p in PERCENTILES])
    return {
        "rms_thickness_m": float(np.sqrt(np.mean(thickness**2))),
        "upstream_thickness_m": float(model["thickness_at_stations"][str(data.RUN6["hours"])][0]),
        "rms_deposit_gravel_psi": float(np.sqrt(np.nanmean(deposit**2))),
        "n_deposit_sections": int(np.sum(np.isfinite(deposit[0]))),
        "rms_surface_gravel_psi": float(np.sqrt(np.mean(surface**2))),
        "depth_m": model["depth_m"],
    }


def figure(runs: dict, path: Path) -> None:
    """Measured and modelled deposit of run 6: thickness, grain sizes of the gravel, depth."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.6))
    measured = data.run6_thickness_profiles()
    ax = axs[0, 0]
    for k, (hours, (x, thickness)) in enumerate(measured.items()):
        ax.plot(x, thickness, "o", color=INK, ms=4, mfc="white", label="measured" if k == 0 else None, zorder=3)
        for color, (label, entry) in zip(COLORS, runs.items(), strict=False):
            ax.plot(entry["model"]["x_m"], entry["model"]["thickness"][str(hours)], color=color, lw=1.4, label=label if k == 0 else None)
    ax.set_xlabel("distance from the feed point (m)")
    ax.set_ylabel("thickness of the deposit (m)")
    ax.set_title(f"Run 6: deposit at {', '.join(f'{h:g}' for h in measured)} h", loc="left", fontsize=11)
    ax.set_xlim(11.0, 50.0)
    ax.set_ylim(0.0, 0.9)
    ax.legend(fontsize=8.5, frameon=False)

    markers = {90: "o", 50: "s", 16: "^"}
    panels = ((axs[0, 1], "deposit_gravel_psi", data.RUN6_DEPOSIT_GRAVEL_PSI, "Gravel of the deposit of hours 8 to 18"),
              (axs[1, 0], "surface_gravel_psi", data.RUN6_SURFACE_GRAVEL_PSI, "Gravel of the surface layer at the end"))
    for ax, key, table, title in panels:
        for i, percent in enumerate(PERCENTILES):
            ax.semilogy(data.RUN6_SIZE_X, 2.0 ** table[percent], markers[percent], color=INK, ms=6, mfc="white",
                        label=f"measured D{percent}")
            for color, entry in zip(COLORS, runs.values(), strict=False):
                ax.semilogy(entry["model"]["size_x_m"], 2.0 ** np.asarray(entry["model"][key][i]), color=color, lw=1.6)
        ax.set_xlabel("distance from the feed point (m)")
        ax.set_ylabel("grain size of the gravel (mm)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlim(11.0, 56.0)
        ax.set_ylim(2.0, 200.0)
        ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")

    ax = axs[1, 1]
    depth = data.RUN6_WATER_SURFACE - np.asarray(data.RUN6_PROFILES[data.RUN6["hours"]])
    ax.plot(data.RUN6_STATIONS, depth, "o", color=INK, ms=4, mfc="white", label="measured", zorder=3)
    for color, (label, entry) in zip(COLORS, runs.items(), strict=False):
        ax.plot(entry["model"]["x_m"], entry["model"]["depth"], color=color, lw=1.4, label=label)
    ax.set_xlabel("distance from the feed point (m)")
    ax.set_ylabel("water depth at the end (m)")
    ax.set_title("Water depth at 27 h", loc="left", fontsize=11)
    ax.set_xlim(11.0, 50.0)
    ax.set_ylim(0.0, 0.16)
    ax.legend(fontsize=8.5, frameon=False)
    for ax in axs.ravel():
        ax.grid(color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)


def main() -> None:
    """Run run 6 with the given closures and compare with the measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None, help="results file of learn_flume.py or learn_flume_transport.py")
    parser.add_argument("--published", nargs=3, type=float, default=(1.0, 0.029, 0.7), metavar=("TAU_R", "N", "SHARE"))
    parser.add_argument("--flow", choices=("backwater", "unsteady"), default="unsteady")
    parser.add_argument("--feed", type=float, default=None, help="feed rate (kg/min) in place of that of flume_data.py")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    name, listed = entries(args)
    name = f"run6_{name}{'_' + args.tag if args.tag else ''}"
    feed = data.RUN6["feed_kg_per_min"] if args.feed is None else args.feed
    out = {"arguments": {k: str(v) for k, v in vars(args).items()}, "feed_kg_per_min": feed,
           "measured_depth_m": data.RUN6["depth_m"], "percentiles": list(PERCENTILES), "runs": {}}
    print(f"run 6, feed {feed:.2f} kg/min; measured: thickness at the first section {data.run6_thickness_profiles()[27.0][1][0]:.3f} m, "
          f"depth {data.RUN6['depth_m']:.3f} m")
    for label, case, free in listed:
        model = simulate(case, free)
        out["runs"][label] = {"model": model, "scores": scores(model)}
        print(f"{label}: {out['runs'][label]['scores']}", flush=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out))
    figure(out["runs"], RESULTS / f"{name}.png")
    print(f"wrote {RESULTS / f'{name}.json'} and .png")


if __name__ == "__main__":
    main()
