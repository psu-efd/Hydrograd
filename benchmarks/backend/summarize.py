"""Turn results/*.jsonl into Markdown tables and figures (results/*.png).

Reads the JSON records appended by run_suite.py and writes

- ``results/summary.md``: one Markdown table per suite (scaling, Savannah, steady,
  unsteady), also printed to stdout;
- ``results/scaling.png``: wall time per time step (ms) against the number of cells, for
  the forward run and the checkpointed reverse-mode gradient, one line per backend label.

Records with an ``error`` key (failed runs) are left out of the scaling table.

Usage:
    python summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent / "results"


def load(name: str) -> list:
    """Read one results file.

    Args:
        name: Suite name; the file is ``results/<name>.jsonl``.

    Returns:
        List of records (dicts), one per non-blank line; empty if the file is missing.
    """
    f = RES / f"{name}.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def label(r: dict) -> str:
    """Backend label of a record, e.g. ``torch-gpu-compile-fp32@roar``.

    Args:
        r: Result record from bench_explicit.py (``_src = "roar"`` marks Roar runs).

    Returns:
        Label ``<backend>-<device>`` plus ``-compile``/``-eager`` (torch), ``-fp32`` and
        ``@roar`` suffixes.
    """
    s = f"{r['backend']}-{r['device']}"
    if r.get("compile"):
        s += "-compile"
    elif r["backend"] == "torch":
        s += "-eager"
    if r.get("dtype") == "float32":
        s += "-fp32"
    if r.get("_src") == "roar":
        s += "@roar"
    return s


def fmt(v, nd=2):
    """Format a table cell.

    Args:
        v: Value; None (missing), float, or anything else.
        nd: Decimal places for floats.

    Returns:
        ``"–"`` for None, a fixed-point string for floats, ``str(v)`` otherwise.
    """
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def scaling() -> str:
    """Scaling table and figure (results/scaling.png) of the explicit-solver runs.

    Returns:
        Markdown table, one row per (label, cells), sorted by label then cells.
    """
    # Windows-PC and Roar records together; failed runs dropped.
    roar_rows = [dict(r, _src="roar") for r in load("roar_scaling")]
    rows = [r for r in load("scaling") + roar_rows if "error" not in r]

    # -------------------------------------------------------------------------------------
    # 1. Markdown table
    # -------------------------------------------------------------------------------------
    out = [
        (
            "| backend | cells | fwd (s) | µs/cell/step | grad (s) | grad/fwd "
            "| compile+1st grad (s) | peak GPU (MB) | peak proc (MB) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (label(r), r["cells"])):
        # Peak device memory is reported for GPU runs only.
        if r["device"] == "gpu":
            peak_gpu = fmt(r.get("grad_peak_dev_mb"), 0)
        else:
            peak_gpu = "–"
        out.append(
            f"| {label(r)} | {r['cells']} | {fmt(r['fwd_s'])} "
            f"| {fmt(r['us_per_cell_step_fwd'], 3)} | "
            f"{fmt(r.get('grad_s'))} | {fmt(r.get('grad_over_fwd'), 1)} "
            f"| {fmt(r.get('grad_first_s'), 1)} | "
            f"{peak_gpu} | {fmt(r['proc_peak_mb'], 0)} |"
        )

    # -------------------------------------------------------------------------------------
    # 2. Figure: ms per time step vs cells (forward left, gradient right)
    # -------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lab in sorted({label(r) for r in rows}):
        rr = sorted([r for r in rows if label(r) == lab], key=lambda r: r["cells"])
        cells = [r["cells"] for r in rr]
        fwd_ms = [r["fwd_s"] / r["steps"] * 1e3 for r in rr]
        ax[0].loglog(cells, fwd_ms, "o-", label=lab)

        # Only runs that computed a gradient.
        rg = [r for r in rr if r.get("grad_s")]
        grad_cells = [r["cells"] for r in rg]
        grad_ms = [r["grad_s"] / r["steps"] * 1e3 for r in rg]
        ax[1].loglog(grad_cells, grad_ms, "o-", label=lab)

    # Axis labels, titles and grids.
    for a, t in zip(ax, ("forward", "gradient (reverse, checkpointed)")):
        a.set_xlabel("cells")
        a.set_ylabel("ms per time step")
        a.set_title(t)
        a.grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "scaling.png", dpi=130)
    return "\n".join(out)


def generic(name: str, keys: list) -> str:
    """Markdown table of one suite with the given columns.

    Args:
        name: Suite name (``results/<name>.jsonl``).
        keys: Record keys, one per column; ``h_err*`` columns get 4 decimals, others 2.

    Returns:
        Markdown table, one row per record in file order.
    """
    rows = load(name)
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        cells = [fmt(r.get(k), 4 if k.startswith("h_err") else 2) for k in keys]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    """Build all tables, write results/summary.md and print it."""
    # Columns of each per-suite table.
    savannah_keys = [
        "backend", "device", "mode", "grad_mode", "compile", "fwd_s", "grad_s",
        "grad_first_s", "proc_peak_mb",
    ]
    roar_savannah_keys = [
        "backend", "device", "device_kind", "mode", "compile", "fwd_s", "grad_s",
        "grad_first_s", "grad_peak_dev_mb", "proc_peak_mb",
    ]
    steady_keys = [
        "method", "cells", "n_colors", "fwd_s", "newton_iters", "adjoint_s", "gmres_iters",
        "n_steps", "total_s", "final_res", "grad_norm", "proc_peak_mb",
    ]
    unsteady_keys = [
        "method", "theta", "mult", "dt", "steps", "newton_its", "fwd_s", "grad_s",
        "h_err_max_m", "h_err_rms_m", "proc_peak_mb",
    ]

    # Sections in order: heading, then table.
    parts = [
        (
            "## Scaling (explicit, NN closure, 500 steps; `@roar` = Roar A100 node, "
            "others = Windows PC)"
        ),
        scaling(),
        "## Savannah (1306 cells, 10,000 steps, dt=0.02 s)",
        generic("savannah", savannah_keys),
        "## Savannah on Roar (A100 node)",
        generic("roar_savannah", roar_savannah_keys),
        "## Steady state",
        generic("steady", steady_keys),
        "## Unsteady explicit vs implicit",
        generic("unsteady", unsteady_keys),
    ]
    text = "\n\n".join(parts)
    (RES / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
