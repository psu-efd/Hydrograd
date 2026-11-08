"""Figure 5: steady Manning's n inversion on the Savannah River (Newton + IFT gradients).

Five Manning zones (true n* = 0.040, 0.050, 0.030, 0.045, 0.050 s m^(-1/3)) are recovered
from synthetic WSE and velocity observations, starting from n = 0.03 everywhere, with Adam
(30 iterations, learning rate 1e-3) followed by L-BFGS. The computations are in
``fig05_compute.py`` (run them as a batch job on a quiet node for reproducible
timings).

Panels
------
(a) Loss L (dimensionless) against iteration, log scale: steady IFT inversion and the
    transient (time-marching, t = 200 s) inversion with the same optimiser stages (both with
    the default second-order scheme).
(b) Manning's n of each zone against iteration (steady inversion); the true values are the
    horizontal lines.
(c) Wall time (s) of the complete inversions on an 8-core CPU allocation (Xeon Gold 6342;
    set-up and JIT compilation included), first- and second-order scheme: steady IFT and
    transient with the same optimiser stages; and, first order, the Hydrograd.jl example's
    configuration (300 Adam iterations, transient, forward-mode AD).
(d) Gradient check: IFT gradient dL/dn_k against central finite differences, zones 1-5, at
    the start point and at an intermediate point.

Numbers go to ``paper/results/steady_inversion.json``.

Usage
-----
    python paper/fig05_steady_inversion.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import FULL_WIDTH, MODEL, MODEL_2, REFERENCE, apply_style, panel_label, save
from savannah_common import DATA_DIR, write_results

# Runs computed by fig05_compute.py.
INVERSION_RUNS = ("steady", "transient", "steady_o1", "transient_o1", "julia300")
RUNS = (*INVERSION_RUNS, "gradcheck")

# Zone colours, the same as the material zones in Figure 2 (Okabe-Ito palette), except
# zone 3, drawn in black instead of yellow for legibility on white.
ZONE_COLOURS = ["#56B4E9", "#009E73", "black", "#CC79A7", "#0072B2"]
ZONE_MARKERS = ["o", "s", "^", "D", "v"]


def load_runs() -> dict[str, dict[str, Any]]:
    """Load the run records, computing the missing ones with ``fig05_compute.py``.

    Returns:
        Run name -> record (see ``fig05_compute.py``).
    """
    missing = [run for run in RUNS if not (DATA_DIR / f"fig05_{run}.json").exists()]
    if missing:
        print(f"computing {missing} on this machine (timings: prefer the Slurm job)")
        script = Path(__file__).resolve().parent / "fig05_compute.py"
        subprocess.run([sys.executable, str(script), *missing], check=True)

    runs = {}
    for run in RUNS:
        runs[run] = json.loads((DATA_DIR / f"fig05_{run}.json").read_text())
    return runs


def _adam_end(record: dict[str, Any]) -> int:
    """Iteration number of the last Adam iteration of a run (0 if there is no Adam stage).

    Args:
        record: run record.

    Returns:
        The iteration number.
    """
    iterations = 0
    for name, count in zip(record["optimizers"], record["max_iterations"]):
        if name == "Adam":
            iterations += int(count)
    return iterations


def summarise(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Collect the figure's numbers and write ``paper/results/steady_inversion.json``.

    Args:
        runs: run records.

    Returns:
        The results dictionary.
    """
    results: dict[str, Any] = {"inversions": {}, "machine": runs["steady"]["machine"]}

    # Per inversion: iterations, wall time, final loss and n, error against the truth.
    for run in INVERSION_RUNS:
        record = runs[run]
        n_final = np.asarray(record["n_final"])[1:]
        n_truth = np.asarray(record["n_truth"])[1:]
        elapsed = record["elapsed_at_iteration_s"]
        n_iterations = len(record["loss_total"]) - 1        # the last row is the final check
        results["inversions"][run] = {
            "spatial_order": record["spatial_order"],
            "solver": record["solver"],
            "optimizers": record["optimizers"],
            "learning_rates": record["learning_rates"],
            "max_iterations": record["max_iterations"],
            "gradient_mode_first_order_stage": ("reverse (IFT)" if run == "steady" else
                                                record["sensealg"]),
            "truth_file": record["truth_file"],
            "tspan_s": record["tspan_s"],
            "n_iterations": n_iterations,
            "n_adam_iterations": min(_adam_end(record), n_iterations),
            "wall_time_total_s": record["wall_time_s"],
            "time_to_first_iteration_s": elapsed[0],
            "time_per_iteration_after_first_s": (elapsed[-1] - elapsed[0]) / max(n_iterations, 1),
            "loss_initial": record["loss_total"][0],
            "loss_final": record["loss_total"][-1],
            "n_initial": record["theta"][0][1:],
            "n_final": n_final.tolist(),
            "n_truth": n_truth.tolist(),
            "max_abs_n_error": float(np.max(np.abs(n_final - n_truth))),
        }

    # Gradient check and the cost of one gradient.
    check = runs["gradcheck"]
    worst = 0.0
    for point in check["checks"].values():
        worst = max(worst, max(point["relative_difference"]))
    results["gradient_check"] = {
        "fd_step": check["fd_step"],
        "points": check["checks"],
        "max_relative_difference": worst,
    }
    results["gradient_cost_s"] = check["gradient_cost_s"]

    # Speed-ups of the steady inversion over the transient ones (wall-time ratios).
    inversions = results["inversions"]
    results["speed_up_steady_vs_transient_same_stages_order2"] = (
        inversions["transient"]["wall_time_total_s"] / inversions["steady"]["wall_time_total_s"])
    results["speed_up_steady_vs_transient_same_stages_order1"] = (
        inversions["transient_o1"]["wall_time_total_s"]
        / inversions["steady_o1"]["wall_time_total_s"])
    results["speed_up_steady_order1_vs_julia300_order1"] = (
        inversions["julia300"]["wall_time_total_s"] / inversions["steady_o1"]["wall_time_total_s"])
    results["reported_earlier"] = {
        "steady_total_s": 37.0,
        "transient_300_adam_min": 9.0,
        "source": ("plan/phase1_parity_report.md section 6b (8-core Roar CPU node, first-order "
                   "scheme, before second order was the default)"),
    }
    write_results("steady_inversion", results)
    return results


def plot(runs: dict[str, dict[str, Any]], results: dict[str, Any]) -> Path:
    """Draw the four panels and save ``figures/fig05_steady_inversion``.

    Args:
        runs: run records.
        results: output of ``summarise``.

    Returns:
        Path of the PDF file.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.6),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.42})
    ax_loss, ax_n, ax_time, ax_grad = axes.ravel()
    steady = runs["steady"]
    transient = runs["transient"]

    # ------------------------------------------------------------------
    # (a) Loss histories
    # ------------------------------------------------------------------
    ax_loss.semilogy(steady["iteration"], steady["loss_total"], label="steady (Newton + IFT)",
                     **MODEL)
    ax_loss.semilogy(transient["iteration"], transient["loss_total"],
                     label=r"transient ($t = 200$ s)", **MODEL_2)
    ax_loss.axvline(_adam_end(steady) + 0.5, color="0.5", linestyle=":", linewidth=0.8)
    ax_loss.text(_adam_end(steady) * 0.5, 0.05, "Adam", transform=ax_loss.get_xaxis_transform(),
                 ha="center", fontsize=8, color="0.3")
    ax_loss.text(_adam_end(steady) + 2.0, 0.05, "L-BFGS", transform=ax_loss.get_xaxis_transform(),
                 ha="left", fontsize=8, color="0.3")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel(r"loss $L$ ($-$)")
    ax_loss.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=8)
    panel_label(ax_loss, "(a)", x=-0.20)

    # ------------------------------------------------------------------
    # (b) Zone n against iteration (steady), truth as horizontal lines
    # ------------------------------------------------------------------
    theta = np.asarray(steady["theta"])
    iterations = np.asarray(steady["iteration"])[:len(theta)]
    n_truth = np.asarray(steady["n_truth"])
    for zone in range(1, 6):
        colour = ZONE_COLOURS[zone - 1]
        ax_n.axhline(n_truth[zone], color=colour, linestyle="-", linewidth=0.8, alpha=0.8)
        ax_n.plot(iterations, theta[:, zone], color=colour, linestyle="--", linewidth=1.2,
                  marker=ZONE_MARKERS[zone - 1], markersize=3.0, markevery=4,
                  markeredgecolor="0.25", markeredgewidth=0.4, label=rf"zone {zone}")
    ax_n.axvline(_adam_end(steady) + 0.5, color="0.5", linestyle=":", linewidth=0.8)
    ax_n.set_xlabel("iteration")
    ax_n.set_ylabel(r"Manning's $n$ (s\,m$^{-1/3}$)")
    ax_n.plot([], [], color="0.4", linestyle="-", linewidth=0.8, label="true value")
    ax_n.set_ylim(0.0285, 0.0585)
    ax_n.legend(loc="upper left", ncol=3, fontsize=7, columnspacing=0.8, handlelength=1.8,
                labelspacing=0.3, borderaxespad=0.3)
    panel_label(ax_n, "(b)", x=-0.20)

    # ------------------------------------------------------------------
    # (c) Wall time of the complete inversions (set-up and compilation included)
    # ------------------------------------------------------------------
    inversions = results["inversions"]
    width = 0.36
    groups = [("first order", "steady_o1", "transient_o1"), ("second order", "steady", "transient")]
    for index, (_, steady_run, transient_run) in enumerate(groups):
        for offset, run, style in ((-0.5, steady_run, MODEL), (0.5, transient_run, MODEL_2)):
            record = inversions[run]
            position = index + offset * width
            label = None
            if index == 0:
                label = "steady (Newton + IFT)" if run.startswith("steady") else "transient"
            ax_time.bar(position, record["wall_time_total_s"], width=width * 0.95,
                        color=style["color"], label=label)
            ax_time.text(position, record["wall_time_total_s"] * 1.12,
                         rf"{record['wall_time_total_s']:.0f}", ha="center", va="bottom",
                         fontsize=7)

    # Hydrograd.jl example configuration (300 Adam iterations, transient, first order).
    julia = inversions["julia300"]
    ax_time.bar(2.0, julia["wall_time_total_s"], width=width * 0.95, color="0.6",
                label=r"transient, 300 Adam (\textit{Hydrograd} v1.0)")
    ax_time.text(2.0, julia["wall_time_total_s"] * 1.12, rf"{julia['wall_time_total_s']:.0f}",
                 ha="center", va="bottom", fontsize=7)

    ax_time.set_yscale("log")
    ax_time.set_ylim(top=julia["wall_time_total_s"] * 30.0)
    ax_time.set_xticks([0, 1, 2], ["first order", "second order", "first order"], fontsize=8)
    ax_time.tick_params(axis="x", length=0, top=False)
    ax_time.set_ylabel("wall time (s)")
    ax_time.legend(loc="upper left", fontsize=7, ncol=1)
    panel_label(ax_time, "(c)", x=-0.20)

    # ------------------------------------------------------------------
    # (d) IFT gradient against central finite differences
    # ------------------------------------------------------------------
    markers = {"start": "o", "intermediate": "s"}
    all_values = []
    for label, point in results["gradient_check"]["points"].items():
        fd = np.asarray(point["grad_fd"])
        ift = np.asarray(point["grad_ift"])
        all_values.extend(fd.tolist() + ift.tolist())
        ax_grad.plot(fd, ift, linestyle="none", marker=markers[label], markerfacecolor="none",
                     markeredgecolor="tab:blue", markersize=5, label=rf"$\mathbf{{n}}$ = {label}")
    limit = 1.1 * max(abs(value) for value in all_values)
    ax_grad.plot([-limit, limit], [-limit, limit], label="1:1", **REFERENCE)
    ax_grad.set_xlim(-limit, limit)
    ax_grad.set_ylim(-limit, limit)
    ax_grad.set_aspect("equal")
    ax_grad.ticklabel_format(style="sci", scilimits=(-2, 2))
    ax_grad.set_xlabel(r"finite difference $\partial L/\partial n_k$ (s$^{-1}$\,m$^{1/3}$)")
    ax_grad.set_ylabel(r"IFT $\partial L/\partial n_k$ (s$^{-1}$\,m$^{1/3}$)")
    worst = results["gradient_check"]["max_relative_difference"]
    mantissa, exponent = f"{worst:.1e}".split("e")
    worst_text = rf"difference ${mantissa} \times 10^{{{int(exponent)}}}$"
    ax_grad.text(0.97, 0.12, r"max.\ relative", transform=ax_grad.transAxes, ha="right",
                 va="bottom", fontsize=8)
    ax_grad.text(0.97, 0.04, worst_text, transform=ax_grad.transAxes, ha="right", va="bottom",
                 fontsize=8)
    ax_grad.legend(loc="upper left", fontsize=8)
    panel_label(ax_grad, "(d)", x=-0.20)

    return save(fig, "fig05_steady_inversion")


def main() -> None:
    """Load (or compute) the runs, write the numbers and draw Figure 5."""
    runs = load_runs()
    results = summarise(runs)
    print(f"wrote {plot(runs, results)}")
    print(json.dumps({key: {k: v for k, v in value.items() if k in
                            ("n_iterations", "wall_time_total_s", "loss_final", "n_final")}
                      for key, value in results["inversions"].items()}, indent=1))


if __name__ == "__main__":
    main()
