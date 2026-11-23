"""Figure of the flume closure-learning case (``learn_flume.py``).

Top row: the fitted hiding factors; the grain sizes of the deposit of the training run
along the flume; the misfit against the number of model evaluations. Bottom row: measured
and modelled thickness of the deposit at the survey times of the training run (run 2) and
of the two test runs (runs 1 and 3).

Usage::

    python plot_flume.py [--formula wilcock_crowe] [--flow unsteady] [--depth]
                         [--exchange bedload_mix]

Reads ``results/hiding_flume_<tag>.json`` and writes ``results/hiding_flume_<tag>.png``;
the tag is that of ``learn_flume.py`` for the same options (``<formula>``, then
``_unsteady``, ``_depth`` and ``_mix``).
For Wilcock-Crowe the first panel shows the reference shear stress of a size relative to
that of the mean size, (d / d_g) / xi, the form in which hiding functions are published.

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
COLORS = {"none": "#8a8985", "published": "#1baf7a", "power": "#2a78d6", "network": "#eb6834"}
LABELS = {"none": "no hiding", "published": "Wilcock-Crowe (published)", "power": "power law", "network": "neural network"}
SHORT = {"none": "no hiding", "published": "published", "power": "power law", "network": "network"}
INK = "#0b0b0b"


def main() -> None:
    """Draw the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula", default="engelund_hansen")
    parser.add_argument("--flow", choices=("backwater", "unsteady"), default="backwater")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--exchange", choices=("surface", "bedload_mix"), default="surface")
    args = parser.parse_args()
    tag = args.formula if args.flow == "backwater" else f"{args.formula}_{args.flow}"
    tag = f"{tag}_depth" if args.depth else tag
    tag = f"{tag}_mix" if args.exchange == "bedload_mix" else tag
    out = json.loads((RESULTS / f"hiding_flume_{tag}.json").read_text())
    reference_stress = args.formula == "wilcock_crowe"
    closures = out["closures"]
    fig, axs = plt.subplots(2, 3, figsize=(14.0, 7.6))

    ax = axs[0, 0]
    ratios = np.asarray(out["size_ratio"])
    for name, entry in closures.items():
        curve = ratios / np.asarray(entry["hiding"]) if reference_stress else np.asarray(entry["hiding"])
        ax.loglog(ratios, curve, color=COLORS[name], lw=2, label=f"{LABELS[name]} ({entry['n_parameters']} parameters)")
    ax.set_xlabel("grain size / mean size of the bed surface")
    ax.set_ylabel("reference stress / that of the mean size" if reference_stress else "hiding factor")
    ax.set_title("Hiding function" if reference_stress else "Learned hiding factor", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)

    ax = axs[0, 1]
    markers = {90: "o", 50: "s", 16: "^"}
    for i, percent in enumerate(out["percentiles"]):
        if percent not in markers:
            continue
        ax.semilogy(data.TABLE1_X, 2.0 ** data.TABLE1_PSI[percent], markers[percent], color=INK, ms=6, mfc="white",
                    label=f"measured D{percent}")
        for name, entry in closures.items():
            ax.semilogy(data.TABLE1_X, 2.0 ** np.asarray(entry["runs"]["2"]["model"]["table1"][i]), color=COLORS[name], lw=1.6)
    ax.set_xlabel("distance from the feed point (m)")
    ax.set_ylabel("grain size of the deposit (mm)")
    ax.set_title("Run 2 (training): deposit of hours 14 to 22", loc="left", fontsize=11)
    ax.set_ylim(0.3, 400.0)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")

    ax = axs[0, 2]
    start = 0
    for name, entry in closures.items():
        history = np.minimum.accumulate(entry["misfit_history"])
        ax.semilogy(start + np.arange(len(history)), history, color=COLORS[name], lw=2, label=LABELS[name])
        start += len(history)
    ax.set_xlabel("model evaluations (the fits follow one another)")
    ax.set_ylabel("misfit of run 2")
    ax.set_title("Fitting", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)

    for ax, run in zip(axs[1], (2, 1, 3), strict=True):
        profiles = data.thickness_profiles(run)
        for k, (hours, (x, thickness)) in enumerate(profiles.items()):
            ax.plot(x, thickness, "o", color=INK, ms=4, mfc="white", label="measured" if k == 0 else None, zorder=3)
            for name, entry in closures.items():
                model = entry["runs"][str(run)]["model"]["thickness"][str(hours)]
                ax.plot(x, model, color=COLORS[name], lw=1.4, label=LABELS[name] if k == 0 else None)
        scores = ", ".join(f"{SHORT[name]} {entry['runs'][str(run)]['scores']['rms_thickness_m'] * 100:.0f} cm"
                           for name, entry in closures.items())
        kind = "training" if run == out["train_run"] else "test"
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
    fig.savefig(RESULTS / f"hiding_flume_{tag}.png", dpi=170)
    print(f"wrote {RESULTS / f'hiding_flume_{tag}.png'}")


if __name__ == "__main__":
    main()
