"""Figure 10 of the Hydrograd v2.0 paper: computational performance.

Usage:
    python paper/fig10_performance.py

The figure (single column) shows the wall time against the number of cells (log-log) on
the synthetic channel: transient forward run (20 s simulated), reverse-mode gradient through
that run, and steady pseudo-transient Newton solve, on one CPU node (16 cores) and on one
A100 GPU. A line of slope 1 marks linear scaling. Data in ``paper/data/performance/``,
written by the ``perf_*.py`` scripts on Roar.

The three other panels of the earlier four-panel version (2026-10-09) are kept below as
functions but are no longer drawn, after the performance section of the paper was
shortened (2026-11-07): (b) checkpoint trade-off, (c) ensemble throughput, (d) steady
gradients against backpropagation through pseudo-time. Their data files are unchanged.

(b) Reverse-mode gradient through the 20 s run at 128,000 cells on the A100 against the
    number of checkpoints: cost relative to the forward run (left axis) and peak device
    memory (right axis). Fewer checkpoints trade memory for recomputation.
(c) Ensemble throughput on the A100: simulations per second against the batch size for the
    1 h flood in the 500-cell compound valley (vmapped explicit solver). The dotted line is
    perfect scaling from a batch of one.
(d) Steady-state gradients, 8,000 cells: relative gradient error against gradient wall time
    for backpropagation through a transient run of length T (pseudo-time, labelled by T)
    and for the Newton + implicit-function-theorem (IFT) gradient (error against central
    finite differences).

Output: ``paper/figures/fig10_performance.pdf`` and ``.png``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figstyle import COLUMN_WIDTH, apply_style, save

DATA = HERE / "data" / "performance"

# Colours of the three timed operations (panel a) and marker style of the two machines.
COLOURS = {"fwd_s": "tab:blue", "grad_s": "tab:red", "newton_s": "k"}
LABELS = {"fwd_s": "forward (20 s)", "grad_s": "gradient", "newton_s": "steady Newton"}
MARKERS = {"fwd_s": "o", "grad_s": "s", "newton_s": "^"}


def read_records(name: str) -> list[dict[str, Any]]:
    """Data records of a JSON-lines file (environment-only lines skipped).

    Args:
        name: file name in ``DATA``.

    Returns:
        List of records ([] if the file does not exist).
    """
    path = DATA / name
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if set(record) != {"environment"}:
            records.append(record)
    return records


def panel_scaling(ax: plt.Axes) -> None:
    """Wall time against cell count on the CPU node and the A100 (the whole figure).

    Args:
        ax: the axes.
    """
    machines = (("scaling_cpu_v4.jsonl", "CPU", "--", "white"),
                ("scaling_a100_v4.jsonl", "A100", "-", None))

    # One line per operation and machine; filled markers on the GPU, open on the CPU.
    for file_name, machine, linestyle, face in machines:
        records = read_records(file_name)
        cells = np.array([r["cells"] for r in records], dtype=float)
        for key in ("fwd_s", "grad_s", "newton_s"):
            times = np.array([r.get(key, np.nan) for r in records], dtype=float)
            ax.plot(cells, times, linestyle=linestyle, marker=MARKERS[key],
                    color=COLOURS[key], markerfacecolor=face or COLOURS[key], lw=1.0, ms=4)

    # Linear-scaling guide below the A100 forward line.
    c_ref = np.array([2.0e4, 5.12e5])
    t_ref = 0.05 * c_ref / c_ref[0]
    ax.plot(c_ref, t_ref, color="0.5", lw=0.8, ls=":")
    ax.text(1.6e5, 0.05 * 8.0 / 2.6, "slope 1", color="0.4", fontsize=8, ha="left", va="top")

    # Legend: operations by colour/marker, machines by line style.
    for key in ("fwd_s", "grad_s", "newton_s"):
        ax.plot([], [], color=COLOURS[key], marker=MARKERS[key], ls="none", label=LABELS[key])
    ax.plot([], [], color="0.3", ls="--", marker="o", markerfacecolor="white",
            label="CPU, 16 cores")
    ax.plot([], [], color="0.3", ls="-", marker="o", label="A100 GPU")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cells (-)")
    ax.set_ylabel("Wall time (s)")
    ax.set_ylim(2.0e-2, 3.0e4)
    ax.legend(loc="upper left", fontsize=7, ncol=2, handlelength=2.0, columnspacing=0.8)


def panel_checkpoints(ax: plt.Axes) -> None:
    """Panel (b): gradient/forward cost ratio and peak memory against checkpoint count.

    Args:
        ax: the axes (memory goes on a twin y axis).
    """
    records = sorted(read_records("checkpoints_a100.jsonl"), key=lambda r: r["checkpoints"])
    checkpoints = np.array([r["checkpoints"] for r in records], dtype=float)
    ratio = np.array([r["ratio"] for r in records])
    memory = np.array([r["peak_mem_grad_gb"] for r in records])

    # Cost ratio on the left axis.
    ax.plot(checkpoints, ratio, color="tab:red", marker="s", ms=4, lw=1.0,
            label="gradient / forward time")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of checkpoints (-)")
    ax.set_ylabel("Gradient / forward time (-)", color="tab:red")
    ax.tick_params(axis="y", colors="tab:red")
    ax.set_ylim(bottom=0.0)

    # Peak device memory on the right axis.
    twin = ax.twinx()
    twin.plot(checkpoints, memory, color="tab:blue", marker="o", ms=4, lw=1.0, ls="-.",
              label="peak memory")
    twin.set_ylabel("Peak GPU memory (GB)", color="tab:blue")
    twin.tick_params(axis="y", colors="tab:blue", direction="in")
    twin.set_ylim(bottom=0.0)
    ax.tick_params(axis="y", right=False)

    # Number of time steps of the run (all states stored beyond this count).
    if records:
        steps = records[0]["fwd_steps"]
        ax.axvline(steps, color="0.5", lw=0.8, ls=":")
        ax.text(steps * 0.92, 0.55, f"{steps} time steps", rotation=90, ha="right",
                va="center", fontsize=8, color="0.4", transform=ax.get_xaxis_transform())

    # One legend for both axes.
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper center", fontsize=7.5)


def read_ensemble() -> tuple[np.ndarray, np.ndarray]:
    """Batch sizes and throughput from the ensemble log.

    Returns:
        (batch sizes (-), throughput (members/s)), each of shape [n_batches].
    """
    path = DATA / "ensemble_a100.log"
    pattern = re.compile(r"batch\s+(\d+):\s+([\d.]+) s,\s+([\d.]+) members/s")
    batches = []
    throughput = []
    if path.exists():
        for line in path.read_text().splitlines():
            match = pattern.search(line)
            if match:
                batches.append(int(match.group(1)))
                throughput.append(float(match.group(3)))
    return np.array(batches, dtype=float), np.array(throughput)


def panel_ensemble(ax: plt.Axes) -> None:
    """Panel (c): ensemble throughput against batch size on the A100.

    Args:
        ax: the axes.
    """
    batches, throughput = read_ensemble()
    ax.plot(batches, throughput, color="tab:blue", marker="o", ms=4, lw=1.0,
            label="measured")

    # Perfect scaling from a batch of one.
    if batches.size:
        ax.plot(batches, throughput[0] * batches / batches[0], color="0.5", lw=0.8, ls=":",
                label="perfect scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size (-)")
    ax.set_ylabel(r"Throughput (runs s$^{-1}$)")
    ax.legend(loc="upper left", fontsize=7.5)


def panel_steady(ax: plt.Axes) -> None:
    """Panel (d): steady-state gradient by backpropagation through pseudo-time vs Newton + IFT.

    Left axis: relative gradient error against the pseudo-time length T of the transient
    run; right axis: gradient wall time. Horizontal lines give the Newton + IFT values.

    Args:
        ax: the axes (wall time goes on a twin y axis).
    """
    records = [r for r in read_records("gradient_methods_a100.jsonl") if r["part"] == "steady"]
    bptt = [r for r in records if r["method"] == "bptt"]
    ift = [r for r in records if r["method"] == "newton_ift"]
    t_end = np.array([r["t_end_s"] for r in bptt])
    errors = np.array([r["rel_err_vs_ift"] for r in bptt])
    times = np.array([r["grad_s"] for r in bptt])

    # Gradient error of backpropagation (left axis) and the IFT error (vs finite differences).
    ax.plot(t_end, errors, color="tab:red", marker="s", ms=4, lw=1.0,
            label="error, backprop. through $T$")
    if ift:
        ax.axhline(ift[0]["fd_rel_err"], color="tab:red", lw=0.9, ls="--",
                   label="error, Newton + IFT")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-13, 1.0e7)
    ax.set_yticks([1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0e0])
    ax.set_xticks(t_end)
    ax.set_xticklabels([f"{t:.0f}" for t in t_end], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Pseudo-time $T$ of the transient run (s)")
    ax.set_ylabel("Relative gradient error (-)", color="tab:red")
    ax.tick_params(axis="y", colors="tab:red", right=False)

    # Gradient wall time (right axis): backpropagation against T, and the IFT time.
    twin = ax.twinx()
    twin.plot(t_end, times, color="tab:blue", marker="o", ms=4, lw=1.0, ls="-.",
              label="time, backprop. through $T$")
    if ift:
        twin.axhline(ift[0]["grad_s"], color="tab:blue", lw=0.9, ls=":",
                     label="time, Newton + IFT")
    twin.set_yscale("log")
    twin.set_ylim(0.3, 3.0e4)
    twin.set_ylabel("Gradient wall time (s)", color="tab:blue")
    twin.tick_params(axis="y", colors="tab:blue", direction="in")

    # One legend for both axes.
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper right", fontsize=7)


def main() -> None:
    """Draw the single-column scaling figure and save it."""
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.9 * COLUMN_WIDTH))
    panel_scaling(ax)
    fig.tight_layout()
    print(save(fig, "fig10_performance"))


if __name__ == "__main__":
    main()
