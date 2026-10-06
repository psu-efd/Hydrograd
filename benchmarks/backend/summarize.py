"""Turn results/*.jsonl into Markdown tables and figures (results/*.png)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent / "results"


def load(name: str) -> list:
    f = RES / f"{name}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def label(r: dict) -> str:
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
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def scaling() -> str:
    rows = [r for r in load("scaling") + [dict(r, _src="roar") for r in load("roar_scaling")]
            if "error" not in r]
    out = ["| backend | cells | fwd (s) | µs/cell/step | grad (s) | grad/fwd | compile+1st grad (s) | peak GPU (MB) | peak proc (MB) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (label(r), r["cells"])):
        out.append(f"| {label(r)} | {r['cells']} | {fmt(r['fwd_s'])} | {fmt(r['us_per_cell_step_fwd'], 3)} | "
                   f"{fmt(r.get('grad_s'))} | {fmt(r.get('grad_over_fwd'), 1)} | {fmt(r.get('grad_first_s'), 1)} | "
                   f"{fmt(r.get('grad_peak_dev_mb'), 0) if r['device'] == 'gpu' else '–'} | {fmt(r['proc_peak_mb'], 0)} |")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lab in sorted({label(r) for r in rows}):
        rr = sorted([r for r in rows if label(r) == lab], key=lambda r: r["cells"])
        ax[0].loglog([r["cells"] for r in rr], [r["fwd_s"] / r["steps"] * 1e3 for r in rr], "o-", label=lab)
        rg = [r for r in rr if r.get("grad_s")]
        ax[1].loglog([r["cells"] for r in rg], [r["grad_s"] / r["steps"] * 1e3 for r in rg], "o-", label=lab)
    for a, t in zip(ax, ("forward", "gradient (reverse, checkpointed)")):
        a.set_xlabel("cells"); a.set_ylabel("ms per time step"); a.set_title(t); a.grid(True, which="both", alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "scaling.png", dpi=130)
    return "\n".join(out)


def generic(name: str, keys: list) -> str:
    rows = load(name)
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(k), 4 if k.startswith("h_err") else 2) for k in keys) + " |")
    return "\n".join(out)


def main() -> None:
    parts = ["## Scaling (explicit, NN closure, 500 steps; `@roar` = Roar A100 node, others = Windows PC)", scaling(),
             "## Savannah (1306 cells, 10,000 steps, dt=0.02 s)",
             generic("savannah", ["backend", "device", "mode", "grad_mode", "compile", "fwd_s", "grad_s",
                                  "grad_first_s", "proc_peak_mb"]),
             "## Savannah on Roar (A100 node)",
             generic("roar_savannah", ["backend", "device", "device_kind", "mode", "compile", "fwd_s", "grad_s",
                                       "grad_first_s", "grad_peak_dev_mb", "proc_peak_mb"]),
             "## Steady state", generic("steady", ["method", "cells", "n_colors", "fwd_s", "newton_iters",
                                                   "adjoint_s", "gmres_iters", "n_steps", "total_s",
                                                   "final_res", "grad_norm", "proc_peak_mb"]),
             "## Unsteady explicit vs implicit", generic("unsteady", ["method", "theta", "mult", "dt", "steps",
                                                                     "newton_its", "fwd_s", "grad_s",
                                                                     "h_err_max_m", "h_err_rms_m",
                                                                     "proc_peak_mb"])]
    text = "\n\n".join(parts)
    (RES / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
