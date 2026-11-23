"""Figure of the learned transport function of the flume case (``learn_flume_transport.py``).

Top row: the fitted transport functions W*(phi), with the range of stress ratios that the
size classes meet on the deposit of the training run (and, if given, transport rates
measured directly in another flume); the grain sizes of the deposit of the training run
along the flume; the misfit against the number of model evaluations. Bottom row: measured
and modelled thickness of the deposit at the survey times of the training run (run 2) and
of the two test runs (runs 1 and 3).

Usage::

    python plot_flume_transport.py [--flow unsteady] [--tag TAG] [--measurements FILE.npz]

Reads ``results/transport_flume_<flow>[_<tag>].json``; writes the same name with ``.png``.
``--measurements`` is a file with the arrays ``stress_ratio`` and ``transport`` (W*) of
directly measured fractional transport rates, drawn as points behind the curves.

Units: SI; grain sizes in mm on logarithmic axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flume_data as data

RESULTS = Path(__file__).parent / "results"
COLORS = {"wilcock_crowe": "#1baf7a", "parker": "#2a78d6", "network": "#eb6834"}
LABELS = {"wilcock_crowe": "Wilcock-Crowe (published)", "parker": "Parker (as in Cui, 2007)", "network": "neural network"}
SHORT = {"wilcock_crowe": "Wilcock-Crowe", "parker": "Parker", "network": "network"}
INK = "#0b0b0b"
MUTED = "#8a8985"
# Part of the surface (by fraction) whose stress ratios mark the range the data constrain.
RANGE_PERCENT = (5.0, 95.0)


def weighted_percentiles(values: np.ndarray, weights: np.ndarray, percent) -> np.ndarray:
    """Percentiles of ``values`` with ``weights`` (both flattened)."""
    order = np.argsort(values.ravel())
    cumulative = np.cumsum(weights.ravel()[order])
    return np.interp(np.asarray(percent) / 100.0, cumulative / cumulative[-1], values.ravel()[order])


def visited_range(out: dict, entry: dict, run: int) -> np.ndarray:
    """Range of the stress ratio on the deposit at the end of a run, weighted by the
    surface fraction of each class."""
    sections = np.asarray(out["deposit_sections"][str(run)])
    model = entry["runs"][str(run)]["model"]
    ratio = np.asarray(model["stress_ratio"])[:, sections]
    return weighted_percentiles(ratio, np.asarray(model["surface_fractions"])[:, sections], RANGE_PERCENT)


def main() -> None:
    """Draw the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=("backwater", "unsteady"), default="unsteady")
    parser.add_argument("--tag", default="")
    parser.add_argument("--measurements", type=Path, default=None)
    args = parser.parse_args()
    name = f"transport_flume_{args.flow}{'_' + args.tag if args.tag else ''}"
    out = json.loads((RESULTS / f"{name}.json").read_text())
    stages = out["stages"]
    train = out["train_run"]
    fig, axs = plt.subplots(2, 3, figsize=(14.0, 7.6))

    ax = axs[0, 0]
    ratios = np.asarray(out["stress_ratio"])
    if args.measurements is not None:
        measured = np.load(args.measurements)
        ax.loglog(measured["stress_ratio"], measured["transport"], ".", color=MUTED, ms=3.5, alpha=0.6, zorder=1,
                  label="measured directly (other flume)")
    for stage, entry in stages.items():
        ax.loglog(ratios, entry["transport_function"], color=COLORS[stage], lw=2, zorder=3,
                  label=f"{LABELS[stage]} ({entry['n_parameters']} parameters)")
    last = list(stages)[-1]
    low, high = visited_range(out, stages[last], train)
    ax.axvspan(low, high, color="0.92", zorder=0)
    ax.text(np.sqrt(low * high), 12.0, f"on the deposit of run {train}", ha="center", va="center", fontsize=8, color="0.3")
    ax.set_xlim(0.3, 30.0)
    ax.set_ylim(1.0e-6, 30.0)
    ax.set_xlabel("bed shear stress / reference shear stress of the class")
    ax.set_ylabel("dimensionless transport rate W*")
    ax.set_title("Transport function", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    ax = axs[0, 1]
    markers = {90: "o", 50: "s", 16: "^"}
    for i, percent in enumerate(out["percentiles"]):
        if percent not in markers:
            continue
        ax.semilogy(data.TABLE1_X, 2.0 ** data.TABLE1_PSI[percent], markers[percent], color=INK, ms=6, mfc="white",
                    label=f"measured D{percent}")
        for stage, entry in stages.items():
            ax.semilogy(data.TABLE1_X, 2.0 ** np.asarray(entry["runs"][str(train)]["model"]["table1"][i]), color=COLORS[stage], lw=1.6)
    ax.set_xlabel("distance from the feed point (m)")
    ax.set_ylabel("grain size of the deposit (mm)")
    ax.set_title(f"Run {train} (training): deposit of hours 14 to 22", loc="left", fontsize=11)
    ax.set_ylim(0.3, 400.0)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")

    ax = axs[0, 2]
    start = 0
    for stage, entry in stages.items():
        history = np.minimum.accumulate(entry["misfit_history"])
        ax.semilogy(start + np.arange(len(history)), history, color=COLORS[stage], lw=2, label=LABELS[stage])
        start += len(history)
    ax.set_xlabel("model evaluations (the fits follow one another)")
    ax.set_ylabel(f"misfit of run {train}")
    ax.set_title("Fitting", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)

    for ax, run in zip(axs[1], (train, *out["test_runs"]), strict=True):
        profiles = data.thickness_profiles(run)
        for k, (hours, (x, thickness)) in enumerate(profiles.items()):
            ax.plot(x, thickness, "o", color=INK, ms=4, mfc="white", label="measured" if k == 0 else None, zorder=3)
            for stage, entry in stages.items():
                model = entry["runs"][str(run)]["model"]["thickness"][str(hours)]
                ax.plot(x, model, color=COLORS[stage], lw=1.4, label=LABELS[stage] if k == 0 else None)
        scores = ", ".join(f"{SHORT[stage]} {entry['runs'][str(run)]['scores']['rms_thickness_m'] * 100:.0f} cm"
                           for stage, entry in stages.items())
        kind = "training" if run == train else "test"
        ax.set_title(f"Run {run} ({kind}), feed {data.feed_rate(run):.3f} kg/s, {len(profiles)} surveys", loc="left", fontsize=11)
        ax.text(0.99, 0.98, "error: " + scores, transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
        ax.set_xlabel("distance from the feed point (m)")
        ax.set_ylabel("thickness of the deposit (m)")
        ax.set_ylim(0.0, 1.1)
    axs[1, 0].legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    for ax in axs.ravel():
        ax.grid(color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=170)
    print(f"wrote {RESULTS / f'{name}.png'}")


if __name__ == "__main__":
    main()
