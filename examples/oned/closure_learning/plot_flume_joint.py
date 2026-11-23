"""Figures of the closures fitted by ``learn_flume_joint.py``.

``--profiles FILE``: measured and modelled deposits of the four experiments for the
closures of one results file. Top row: thickness of the deposit of runs 2, 1, 3 and 6 at
the first, middle and last survey. Bottom row: grain sizes of the deposit of run 2
(hours 14 to 22), of the gravel of the deposit of run 6 (hours 8 to 18) and of the gravel
of its surface at the end, and the mean water depth of the four runs.

``--slides FILE``: the same in two figures for a talk: the thickness of the deposits in a
2 by 2 array (training runs on top, test runs below, ``<name>_thickness.png``) and the
three panels of grain sizes (``<name>_sizes.png``); no water depth.

``--history FILE``: misfit of the training data during the fits of a results file, with
the fitted parameters named on the figure (``<name>_history.png``).

``--scores FILE [FILE ...]``: errors of every closure of the given files on every
experiment, as one chart per kind of measurement (thickness, grain sizes of the deposit,
water depth); training runs are drawn as filled markers, test runs as open ones.

``--functions FILE [FILE ...]``: the hiding function (as the reference shear stress of a
size relative to that of the mean size of the surface) and the transport function of
every closure of the given files, with the published ones. ``--measurements FILE.npz``
adds directly measured transport rates (arrays ``stress_ratio`` and ``transport``) behind
the transport functions.

``--summary FILE``: one figure for a talk, for a results file with a learned hiding
function: the hiding function against the published one (and, with ``--single FILE``, the
one fitted on a single run) and the thickness errors of the four experiments. With
``--direct-panel`` a third panel shows the bias by grain size on the sediments of the
direct measurements that were left out of the fit; that comparison depends on which size
is taken for a sieve class and on a few runs, and is not drawn by default.

Usage::

    python plot_flume_joint.py --profiles results/joint_train26_variable_power_reduced.json
    python plot_flume_joint.py --scores results/joint_train*.json [--out results/joint_scores.png]
    python plot_flume_joint.py --functions results/joint_train26_*.json [--out results/joint_functions.png]

Units: SI; grain sizes in mm on logarithmic axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import flume_data as data

RESULTS = Path(__file__).parent / "results"
COLORS = {"published": "#1baf7a", "power": "#2a78d6", "hiding_network": "#eb6834", "transport_network": "#8a8985",
          "parker": "#8a8985"}
LABELS = {"published": "published closures", "power": "power-law hiding function", "hiding_network": "learned hiding function",
          "transport_network": "learned transport curve", "parker": "Parker transport curve"}
# Name of the published closures where they are compared with other hiding functions only.
HIDING_STAGES = ("published", "power", "hiding_network")
PUBLISHED_HIDING = "Wilcock-Crowe hiding function"
INK = "#0b0b0b"
# The parameters fitted with every closure, by their key in the results files, in the
# order in which a figure lists them.
PARAMETER_NAMES = {
    "log_tau_r": "reference shear stress: multiplier for the flume runs",
    "log_tau_r_direct": "reference shear stress: multiplier for the direct data",
    "log_n": "Manning's n",
    "log_k": "roughness length of the bed",
    "log_alpha": "roughness length over the grain size of the bed surface",
    "c_interface": "share of bedload in the material stored in the deposit",
    "m_h": "exponent of the power-law hiding function",
}
RUNS = (2, 1, 3, 6)
MEASURED_DEPTH = {1: data.RUNS[1]["depth_m"], 2: data.RUNS[2]["depth_m"], 3: data.RUNS[3]["depth_m"], 6: data.RUN6["depth_m"]}
MARKERS = {90: "o", 50: "s", 16: "^"}
# Percentiles of the deposit of run 2 in the results files (``learn_flume.PERCENTILES``).
TABLE1_PERCENTILES = (90, 84, 50, 16)


def profiles_of(run: int) -> dict:
    """Measured thickness of a run: {hours: (stations, thickness)}."""
    return data.run6_thickness_profiles() if run == 6 else data.thickness_profiles(run)


def tidy(ax) -> None:
    """Recessive grid and no top or right frame."""
    ax.grid(color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def stage_labels(stages: dict) -> dict:
    """Legend text of the closures of a results file."""
    labels = dict(LABELS)
    if all(name in HIDING_STAGES for name in stages):
        labels["published"] = PUBLISHED_HIDING
    return labels


def thickness_panel(ax, run: int, stages: dict, labels: dict, training: list, title_size: float = 11) -> None:
    """Measured and modelled thickness of the deposit of a run at its first, middle and
    last survey."""
    measured = profiles_of(run)
    times = list(measured)
    shown = [times[0], times[len(times) // 2], times[-1]]
    for k, hours in enumerate(shown):
        x, thickness = measured[hours]
        ax.plot(x, thickness, "o", color=INK, ms=4, mfc="white", label="measured" if k == 0 else None, zorder=3)
        for name, entry in stages.items():
            model = entry["runs"][str(run)]["model"]
            ax.plot(x, model["thickness"][str(hours)], color=COLORS[name], lw=1.5, label=labels[name] if k == 0 else None)
    kind = "training" if run in training else "test"
    ax.set_title(f"Run {run} ({kind}): deposit at {', '.join(f'{h:g}' for h in shown)} h", loc="left", fontsize=title_size)
    ax.set_xlabel("distance from the feed point (m)")
    ax.set_ylabel("thickness of the deposit (m)")
    ax.set_ylim(0.0, 1.0)


def size_panels(axs, out: dict, stages: dict, title_size: float = 11, legend_size: float = 8.5) -> None:
    """Grain sizes along the flume in three panels: the deposit of run 2, and the gravel
    of the deposit and of the surface of run 6."""
    ax = axs[0]
    for i, percent in enumerate(TABLE1_PERCENTILES):
        if percent not in MARKERS:
            continue
        ax.semilogy(data.TABLE1_X, 2.0 ** data.TABLE1_PSI[percent], MARKERS[percent], color=INK, ms=6, mfc="white",
                    label=f"measured D{percent}")
        for name, entry in stages.items():
            ax.semilogy(data.TABLE1_X, 2.0 ** np.asarray(entry["runs"]["2"]["model"]["table1"][i]), color=COLORS[name], lw=1.6)
    ax.set_title("Run 2: deposit of hours 14 to 22", loc="left", fontsize=title_size)
    ax.set_ylabel("grain size (mm)")
    ax.set_ylim(0.3, 400.0)

    n_sizes = len(next(iter(stages.values()))["runs"]["6"]["model"]["deposit"][0])
    x6 = data.RUN6_SIZE_X[:n_sizes]
    panels = ((axs[1], "deposit", data.RUN6_DEPOSIT_GRAVEL_PSI, "Run 6: gravel of the deposit of hours 8 to 18"),
              (axs[2], "surface", data.RUN6_SURFACE_GRAVEL_PSI, "Run 6: gravel of the surface at the end"))
    for ax, key, table, title in panels:
        for i, percent in enumerate(out["percentiles6"]):
            ax.semilogy(x6, 2.0 ** table[percent][:n_sizes], MARKERS[percent], color=INK, ms=6, mfc="white", label=f"measured D{percent}")
            for name, entry in stages.items():
                model = entry["runs"]["6"]["model"]
                values = np.asarray(model[key][i], dtype=float)
                if key == "deposit":
                    values = np.where(np.asarray(model["layer"]) > 0.005, values, np.nan)
                ax.semilogy(x6, 2.0**values, color=COLORS[name], lw=1.6)
        ax.set_title(title, loc="left", fontsize=title_size)
        ax.set_ylabel("grain size of the gravel (mm)")
        ax.set_ylim(2.0, 200.0)
    for ax in axs:
        ax.set_xlabel("distance from the feed point (m)")
        ax.legend(fontsize=legend_size, frameon=False, loc="upper right")


def draw_profiles(path: Path) -> Path:
    """Deposits of the four experiments for the closures of one results file."""
    out = json.loads(path.read_text())
    stages = out["stages"]
    training = out["train_runs"]
    labels = stage_labels(stages)
    fig, axs = plt.subplots(2, 4, figsize=(17.0, 7.8))

    for ax, run in zip(axs[0], RUNS, strict=True):
        thickness_panel(ax, run, stages, labels, training)
    axs[0, 0].legend(fontsize=8.5, frameon=False, loc="upper right")
    size_panels(axs[1, :3], out, stages)

    ax = axs[1, 3]
    positions = np.arange(len(RUNS))
    ax.plot(positions, [MEASURED_DEPTH[run] for run in RUNS], "o", color=INK, ms=7, mfc="white", label="measured", zorder=3)
    for k, (name, entry) in enumerate(stages.items()):
        depths = [entry["runs"][str(run)]["scores"]["depth_m"] for run in RUNS]
        ax.plot(positions + 0.08 * (k + 1), depths, "D", color=COLORS[name], ms=5, label=labels[name])
    ax.set_xticks(positions)
    ax.set_xticklabels([f"run {run}" for run in RUNS])
    ax.set_ylabel("mean water depth over the deposit (m)")
    ax.set_title("Water depth", loc="left", fontsize=11)
    ax.set_ylim(0.0, 0.2)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    for ax in axs.ravel():
        tidy(ax)
    fig.tight_layout()
    target = path.with_suffix(".png")
    fig.savefig(target, dpi=170)
    return target


def draw_slides(path: Path) -> tuple[Path, Path]:
    """Two figures for a talk: the thickness of the deposits (training runs on top, test
    runs below) and the grain sizes along the flume."""
    out = json.loads(path.read_text())
    stages = out["stages"]
    training = out["train_runs"]
    labels = stage_labels(stages)
    lines = [plt.Line2D([], [], color=COLORS[name], lw=2.0, label=labels[name]) for name in stages]
    marker = plt.Line2D([], [], marker="o", color=INK, ls="", mfc="white", ms=5, label="measured")
    targets = (path.with_name(path.stem + "_thickness.png"), path.with_name(path.stem + "_sizes.png"))
    with plt.rc_context({"font.size": 12.5}):
        test = [run for run in RUNS if run not in training]
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4))
        for ax, run in zip(axs.ravel(), [*sorted(training), *sorted(test)], strict=True):
            thickness_panel(ax, run, stages, labels, training, 13.5)
            tidy(ax)
        fig.legend(handles=[marker, *lines], loc="upper center", ncol=3, frameon=False, fontsize=12.5)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(targets[0], dpi=200)

        fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.3))
        size_panels(axs, out, stages, 13.5, 11.5)
        for ax in axs:
            tidy(ax)
        fig.legend(handles=lines, loc="upper center", ncol=len(lines), frameon=False, fontsize=12.5)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig.savefig(targets[1], dpi=200)
    return targets


def draw_history(path: Path, target: Path) -> Path:
    """Misfit of the training data during the fits of one results file.

    The fits follow one another (the parameters of the published closures first, then
    the closure that starts from them). The optimizer (L-BFGS-B) evaluates the misfit and
    its gradient once per point; some points are trial steps that it does not keep, so
    the line is the lowest misfit so far. The parameters of the first fit are named on
    the figure.
    """
    out = json.loads(path.read_text())
    stages = out["stages"]
    labels = stage_labels(stages)
    first = float(next(iter(stages.values()))["misfit_history"][0])
    lowest = min(min(entry["misfit_history"]) for entry in stages.values())
    low, high = lowest - 0.08 * (first - lowest), first + 0.12 * (first - lowest)
    fitted = next(iter(stages.values()))["free"]
    scalars = [key for key in PARAMETER_NAMES if key in fitted]
    with plt.rc_context({"font.size": 12.5}):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        start = 1
        for name, entry in stages.items():
            best = np.minimum.accumulate(np.asarray(entry["misfit_history"], dtype=float))
            count = start + np.arange(best.size)
            n_weights = entry["n_parameters"] - sum(key in PARAMETER_NAMES for key in entry["free"])
            if n_weights > 0:
                size = f"the same {entry['n_parameters'] - n_weights} + {n_weights} network weights"
            else:
                size = f"{entry['n_parameters']} parameters"
            ax.plot(count, best, color=COLORS[name], lw=2.2, label=f"{labels[name]}: {size}")
            ax.annotate(f"{best[0]:.2f}", (count[0], best[0]), textcoords="offset points", xytext=(7, 4), color=INK)
            # The last value of a fit: above the line at the end of the figure, below it
            # where the next fit starts (its first value stands above).
            offset = (4, 7) if name == list(stages)[-1] else (2, -17)
            ax.annotate(f"{best[-1]:.2f}", (count[-1], best[-1]), textcoords="offset points", xytext=offset, color=INK, ha="right")
            start += best.size
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.22, 1.0), fontsize=11.5)
        note = f"The {len(scalars)} parameters:\n" + "\n".join(f"({k + 1}) {PARAMETER_NAMES[key]}" for k, key in enumerate(scalars))
        ax.text(0.30, 0.72, note, transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color=INK, linespacing=1.5)
        ax.set_xlim(0.0, start + 1.0)
        ax.set_ylim(low, high)
        terms = [f"flume run {run}" for run in out["train_runs"]] + (["direct measurements"] if out.get("direct_data") else [])
        ax.set_xlabel("evaluation of the misfit and its gradient")
        ax.set_ylabel("misfit of the training data")
        ax.set_title("Training: " + " + ".join(terms), loc="left", fontsize=13)
        tidy(ax)
        fig.tight_layout()
        fig.savefig(target, dpi=200)
    return target


def draw_scores(paths: list[Path], target: Path) -> Path:
    """Errors of every closure of the given files on every experiment."""
    rows, seen = [], set()
    for path in paths:
        out = json.loads(path.read_text())
        training = "+".join(str(r) for r in out["train_runs"])
        training += " + surface" if out.get("surface_data") else ""
        training += " + direct" if out.get("direct_data") else ""
        physics = {"manning": "Manning", "variable_power": "variable power", "keulegan": "Keulegan"}[out["resistance"]]
        physics += {"surface_mean": ", roughness from the surface mean size", "surface_d90": ", roughness from the surface D90"}.get(
            out.get("roughness_scale"), "")
        physics += ", reduced stress" if out["stress"] == "reduced" else ""
        for name, entry in out["stages"].items():
            label = f"{physics} | train {training} | {LABELS[name]}"
            # The published closures are fitted again in every file of the same set-up.
            if label not in seen:
                seen.add(label)
                rows.append((label, out["train_runs"], name, entry))
    kinds = (("rms_thickness_m", 100.0, "thickness error (cm)"), ("rms_deposit_psi", 1.0, "deposit grain-size error (psi)"),
             ("depth_m", 100.0, "water depth error (cm)"))
    run_markers = {2: "o", 1: "v", 3: "^", 6: "s"}
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 2.0 + 0.36 * len(rows)), sharey=True)
    for ax, (key, factor, title) in zip(axs, kinds, strict=True):
        for i, (_, training, name, entry) in enumerate(rows):
            for run in RUNS:
                value = entry["runs"][str(run)]["scores"][key]
                value = abs(value - MEASURED_DEPTH[run]) if key == "depth_m" else value
                ax.plot(value * factor, i, run_markers[run], color=COLORS[name], ms=7,
                        mfc=COLORS[name] if run in training else "white", zorder=3)
        ax.set_xlabel(title)
        ax.set_xlim(left=0.0)
        tidy(ax)
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([row[0] for row in rows], fontsize=8.5)
    axs[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker=run_markers[run], color=INK, ls="", mfc="white", label=f"run {run}") for run in RUNS]
    handles += [plt.Line2D([], [], marker="o", color=INK, ls="", label="filled: training run")]
    fig.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper right", ncol=5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(target, dpi=170)
    return target


def setup_label(out: dict) -> str:
    """Short description of what a results file was trained on."""
    training = "runs " + " and ".join(str(r) for r in out["train_runs"]) if len(out["train_runs"]) > 1 else f"run {out['train_runs'][0]}"
    training += ", surface" if out.get("surface_data") else ""
    training += ", direct measurements" if out.get("direct_data") else ""
    return training


def draw_functions(paths: list[Path], target: Path, measurements: Path | None) -> Path:
    """Hiding and transport functions of the closures of the given files."""
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    styles = ("-", "--", ":", "-.")
    ratios = stress = None
    drawn_published = False
    for k, path in enumerate(paths):
        out = json.loads(path.read_text())
        ratios, stress = np.asarray(out["size_ratio"]), np.asarray(out["stress_ratio"])
        for name, entry in out["stages"].items():
            if name == "published":
                if drawn_published:
                    continue
                drawn_published = True
                label = "Wilcock and Crowe (2003)"
            else:
                label = f"{LABELS[name]}, trained on {setup_label(out)}"
            style = "-" if name == "published" else styles[k % len(styles)]
            width = 2.4 if name == "published" else 1.8
            if name not in ("transport_network", "parker"):
                axs[0].loglog(ratios, ratios / np.asarray(entry["hiding"]), color=COLORS[name], ls=style, lw=width, label=label)
            if name in ("published", "transport_network", "parker"):
                # Stress relative to the published reference stress: phi times the multiplier.
                axs[1].loglog(stress * entry["tau_r_wc"], entry["transport_function"], color=COLORS[name], ls=style, lw=width,
                              label=f"{label}, multiplier {entry['tau_r_wc']:.2f}")
    axs[0].loglog(ratios, np.ones_like(ratios), color="0.6", lw=0.9, zorder=0)
    axs[0].loglog(ratios, ratios, color="0.6", lw=0.9, zorder=0)
    axs[0].text(0.012, 1.15, "equal mobility", fontsize=8, color="0.35")
    axs[0].text(4.0, 9.0, "no hiding", fontsize=8, color="0.35", rotation=38)
    axs[0].set_xlim(0.01, 30.0)
    axs[0].set_ylim(0.05, 30.0)
    axs[0].set_xlabel("grain size / mean size of the bed surface")
    axs[0].set_ylabel("reference shear stress / that of the mean size")
    axs[0].set_title("Hiding function", loc="left", fontsize=11)
    axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    if measurements is not None:
        measured = np.load(measurements)
        axs[1].loglog(measured["stress_ratio"], measured["transport"], ".", color="0.6", ms=3.5, alpha=0.6, zorder=1,
                      label="measured directly (Wilcock et al., 2001)")
    axs[1].set_xlim(0.3, 30.0)
    axs[1].set_ylim(1.0e-6, 30.0)
    axs[1].set_xlabel("bed shear stress / published reference shear stress of the class")
    axs[1].set_ylabel("dimensionless transport rate W*")
    axs[1].set_title("Transport function", loc="left", fontsize=11)
    axs[1].legend(fontsize=8, frameon=False, loc="lower right")
    for ax in axs:
        tidy(ax)
    fig.tight_layout()
    fig.savefig(target, dpi=170)
    return target


def draw_summary(path: Path, single: Path | None, target: Path, direct_panel: bool = False) -> Path:
    """Learned hiding function and its errors in the flume (and its bias on outside data)."""
    out = json.loads(path.read_text())
    published, learned = out["stages"]["published"], out["stages"]["hiding_network"]
    ratios = np.asarray(out["size_ratio"])
    green, orange, muted = COLORS["published"], COLORS["hiding_network"], "#8a8985"
    plt.rcParams.update({"font.size": 12})
    if direct_panel:
        fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": (1.25, 1.0, 0.9)})
    else:
        fig, axs = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": (1.25, 1.0)})

    ax = axs[0]
    ax.loglog(ratios, np.ones_like(ratios), color="0.6", lw=1.0, zorder=0)
    ax.loglog(ratios, ratios, color="0.6", lw=1.0, zorder=0)
    ax.text(0.013, 1.13, "equal mobility", fontsize=10.5, color=INK)
    ax.text(3.6, 8.2, "no hiding", fontsize=10.5, color=INK, rotation=40)
    if single is not None:
        one = json.loads(single.read_text())
        entry = one["stages"]["hiding_network"]
        ax.loglog(np.asarray(one["size_ratio"]), np.asarray(one["size_ratio"]) / np.asarray(entry["hiding"]), color=muted, lw=2.0, ls="--",
                  label="learned from one mixture only")
    ax.loglog(ratios, ratios / np.asarray(published["hiding"]), color=green, lw=3.0, label="Wilcock and Crowe (2003)")
    ax.loglog(ratios, ratios / np.asarray(learned["hiding"]), color=orange, lw=3.0,
              label="learned: two mixtures + direct measurements")
    ax.set_xlim(0.01, 30.0)
    ax.set_ylim(0.08, 30.0)
    ax.set_xlabel("grain size / mean size of the bed surface")
    ax.set_ylabel("reference shear stress / that of the mean size")
    ax.set_title("Hiding function", loc="left", fontsize=13)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")

    ax = axs[1]
    names = {2: "run 2", 6: "run 6", 1: "run 1", 3: "run 3"}
    order = (2, 6, 1, 3)
    height = 0.36
    for k, (entry, color, label) in enumerate(((published, green, "Wilcock and Crowe"), (learned, orange, "learned"))):
        values = [100.0 * entry["runs"][str(run)]["scores"]["rms_thickness_m"] for run in order]
        positions = np.arange(len(order)) + (k - 0.5) * (height + 0.03)
        ax.barh(positions, values, height=height, color=color, label=label)
        for position, value in zip(positions, values, strict=True):
            ax.text(value + 0.08, position, f"{value:.1f}", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{names[run]}\n({'fitted' if run in out['train_runs'] else 'test'})" for run in order])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 7.5)
    ax.set_xlabel("error of the deposit thickness (cm)")
    ax.set_title("Flume deposits", loc="left", fontsize=13)
    if not direct_panel:
        ax.legend(fontsize=10.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.625))
        for axis in axs:
            tidy(axis)
        fig.tight_layout()
        fig.savefig(target, dpi=200)
        return target

    ax = axs[2]
    groups = (("median_ratio_fine", "finer than\nhalf the mean"), ("median_ratio_middle", "near the\nmean"),
              ("median_ratio_coarse", "coarser than\ntwice the mean"))
    ax.axhline(1.0, color="0.6", lw=1.0, zorder=0)
    for entry, color, label, shift in ((published, green, "Wilcock and Crowe", -0.09), (learned, orange, "learned", 0.09)):
        values = [entry["direct"]["test"][key] for key, _ in groups]
        ax.semilogy(np.arange(len(groups)) + shift, values, "o", color=color, ms=11, label=label)
        for position, value in zip(np.arange(len(groups)) + shift, values, strict=True):
            ax.text(position + (0.13 if shift > 0 else -0.13), value, f"{value:.2f}", va="center",
                    ha="left" if shift > 0 else "right", fontsize=10.5, color=INK)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([label for _, label in groups], fontsize=10.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0.2, 3.0)
    ax.set_yticks([0.25, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.25", "0.5", "1", "2"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("predicted / measured transport (median)")
    ax.set_title("Direct transport measurements\n(sediments not used in the fit)", loc="left", fontsize=13)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    for ax in axs:
        tidy(ax)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    return target


def main() -> None:
    """Draw the requested figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument("--slides", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--scores", type=Path, nargs="+", default=None)
    parser.add_argument("--functions", type=Path, nargs="+", default=None)
    parser.add_argument("--measurements", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--single", type=Path, default=None, help="results file with a hiding network fitted on one run")
    parser.add_argument("--direct-panel", action="store_true", help="add the panel of the direct measurements to the summary")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.profiles is not None:
        print(f"wrote {draw_profiles(args.profiles)}")
    if args.slides is not None:
        print("wrote " + " and ".join(str(target) for target in draw_slides(args.slides)))
    if args.history is not None:
        print(f"wrote {draw_history(args.history, args.out or args.history.with_name(args.history.stem + '_history.png'))}")
    if args.scores is not None:
        print(f"wrote {draw_scores(args.scores, args.out or RESULTS / 'joint_scores.png')}")
    if args.summary is not None:
        print(f"wrote {draw_summary(args.summary, args.single, args.out or RESULTS / 'joint_summary.png', args.direct_panel)}")
    if args.functions is not None:
        print(f"wrote {draw_functions(args.functions, args.out or RESULTS / 'joint_functions.png', args.measurements)}")


if __name__ == "__main__":
    main()
