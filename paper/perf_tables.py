"""Collect the performance data of the Hydrograd v2.0 paper into tables and one JSON file.

Usage:
    python paper/perf_tables.py

Inputs (``paper/data/performance/``, written by the ``perf_*.py`` scripts on Roar):

- ``scaling_cpu_v4.jsonl``, ``scaling_a100_v4.jsonl``: scaling profile (synthetic channel,
  8k-512k cells; forward 20 s, reverse gradient, steady Newton, IFT gradient);
- ``gradient_methods_cpu.jsonl``, ``gradient_methods_a100.jsonl``: steady IFT against
  backpropagation through pseudo-time, and explicit against implicit unsteady gradients;
- ``savannah_python_{cpu,a100}.jsonl``, ``savannah_julia_cpu.json``: Savannah River,
  Hydrograd v2.0 and Hydrograd.jl on the same node;
- ``checkpoints_a100.jsonl``, ``ensemble_a100.log``: used by the figure only.

Outputs (``paper/results/``):

- ``performance.json``: every number quoted in the paper, with hardware and versions;
- ``table_scaling.csv``, ``table_gradient_methods.csv``, ``table_savannah.csv``;
- ``tables.tex``: ready-to-paste LaTeX tabulars (GMD style: \\hline at the top, after the
  header and at the bottom only).

Times in s, memory in GB, errors are relative 2-norms (-).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "performance"
RESULTS = HERE / "results"


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a JSON-lines file; split environment lines from data records.

    Args:
        path: the file.

    Returns:
        (data records, the last environment record found; {} if none). Records that carry
        their own "environment" key keep it, and it is also returned as the environment.
    """
    records: list[dict[str, Any]] = []
    environment: dict[str, Any] = {}
    if not path.exists():
        return records, environment

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) == {"environment"}:
            environment = record["environment"]
            continue
        if "environment" in record:
            environment = record.pop("environment")
        records.append(record)
    return records, environment


def fmt(value: float | None, digits: int = 2) -> str:
    """Format a time or ratio for a table: 3 significant digits, "--" if missing.

    Args:
        value: the number, or None.
        digits: unused (kept for call-site clarity).

    Returns:
        The formatted string.
    """
    del digits
    if value is None:
        return "--"
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_err(value: float | None) -> str:
    """Format a relative error in LaTeX scientific notation ("--" if missing).

    Args:
        value: the error (-), or None.

    Returns:
        A LaTeX math string such as ``$3.1\\times10^{-9}$``.
    """
    if value is None or math.isnan(value):
        return "--"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_table(header: list[list[str]], rows: list[list[str]], align: str) -> str:
    """A GMD-style LaTeX tabular (\\hline at the top, after the header, at the bottom).

    Args:
        header: header rows, each a list of column headings.
        rows: table body, one list of strings per row.
        align: column alignment string, e.g. "rrrr".

    Returns:
        The tabular environment as a string.
    """
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline"]
    for header_row in header:
        lines.append(" & ".join(header_row) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def scaling_table(out: dict[str, Any]) -> str:
    """Table "scaling": cells, steps, forward, gradient, Newton and IFT times, CPU and GPU.

    Args:
        out: the performance dictionary (filled in place under "scaling").

    Returns:
        LaTeX tabular.
    """
    cpu, env_cpu = read_jsonl(DATA / "scaling_cpu_v4.jsonl")
    gpu, env_gpu = read_jsonl(DATA / "scaling_a100_v4.jsonl")
    by_cells_gpu = {r["cells"]: r for r in gpu}
    out["scaling"] = {"cpu": cpu, "a100": gpu, "environment_cpu": env_cpu,
                      "environment_a100": env_gpu}

    # CSV and LaTeX rows: one per mesh size.
    header = ["Cells", "Steps", "Fwd CPU", "Fwd GPU", "Grad CPU", "Grad GPU",
              "Newton CPU", "Newton GPU", "IFT CPU", "IFT GPU"]
    rows = []
    csv_rows = []
    for r in cpu:
        g = by_cells_gpu.get(r["cells"], {})
        values = [r.get("fwd_s"), g.get("fwd_s"), r.get("grad_s"), g.get("grad_s"),
                  r.get("newton_s"), g.get("newton_s"), r.get("ift_grad_s"), g.get("ift_grad_s")]
        rows.append([f"{r['cells']:,}".replace(",", "\\,"), str(r["fwd_steps"])]
                    + [fmt(v) for v in values])
        csv_rows.append([r["cells"], r["fwd_steps"]] + values)

    with open(RESULTS / "table_scaling.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([h.replace(" ", "_").lower() + ("" if i < 2 else "_s")
                         for i, h in enumerate(header)])
        writer.writerows(csv_rows)

    latex_header = ["Cells", "Steps", "\\multicolumn{2}{c}{Forward (s)}",
                    "\\multicolumn{2}{c}{Gradient (s)}", "\\multicolumn{2}{c}{Newton (s)}",
                    "\\multicolumn{2}{c}{Newton + IFT (s)}"]
    sub_header = ["", "", "CPU", "GPU", "CPU", "GPU", "CPU", "GPU", "CPU", "GPU"]
    table = latex_table([latex_header, sub_header], rows, "rrrrrrrrrr")
    return table


def gradient_methods_table(out: dict[str, Any]) -> str:
    """Table "gradient methods": steady IFT vs backpropagation; explicit vs implicit.

    Args:
        out: the performance dictionary (filled in place under "gradient_methods").

    Returns:
        LaTeX tabular.
    """
    gpu, env_gpu = read_jsonl(DATA / "gradient_methods_a100.jsonl")
    cpu, env_cpu = read_jsonl(DATA / "gradient_methods_cpu.jsonl")
    out["gradient_methods"] = {"a100": gpu, "cpu": cpu, "environment_a100": env_gpu,
                               "environment_cpu": env_cpu}

    header = ["Problem", "Method", "Steps", "Forward (s)", "Gradient (s)", "Memory (GB)",
              "Gradient error", "Difference"]
    rows = []
    csv_rows = []

    # Steady part (A100 node): IFT, then backpropagation through pseudo-time T.
    for r in [x for x in gpu if x["part"] == "steady"]:
        if r["method"] == "newton_ift":
            method = "Newton + IFT"
            steps = "--"
            forward = r["newton_s"]
            error = r["fd_rel_err"]
        else:
            method = f"BPTT, $T={r['t_end_s']:.0f}$\\,s"
            steps = str(r["fwd_steps"])
            forward = r["fwd_s"]
            error = r["rel_err_vs_ift"]
        rows.append(["Steady, 8\\,000 cells, A100", method, steps, fmt(forward), fmt(r["grad_s"]),
                     fmt(r["peak_mem_gb"]), fmt_err(error), "--"])
        csv_rows.append(["steady", r["method"], r.get("t_end_s"), r.get("fwd_steps"),
                         forward, r["grad_s"], r["peak_mem_gb"], error, None])

    # Unsteady part (CPU node): explicit SSPRK3 and backward Euler. The last column compares
    # each gradient with that of the explicit model (64 checkpoints): the implicit model is a
    # different discretization, so this is its discretization error in the gradient.
    names = {"explicit64": "SSPRK3, 64 checkpoints", "explicit_auto": "SSPRK3, all steps stored"}
    unsteady = [x for x in cpu if x["part"] == "unsteady"]
    reference = next((np.asarray(x["grad"]) for x in unsteady if x["method"] == "explicit64"),
                     None)
    for r in unsteady:
        method = names.get(r["method"], f"Backward Euler, $\\Delta t={r['method'][2:]}$\\,s")
        grad = np.asarray(r["grad"])
        diff = None
        if reference is not None:
            diff = float(np.linalg.norm(grad - reference) / np.linalg.norm(reference))
        r["rel_diff_vs_explicit"] = diff
        rows.append(["Flood, 500 cells, CPU", method, str(r["steps"]), fmt(r["fwd_s"]),
                     fmt(r["grad_s"]), fmt(r["peak_mem_gb"]), fmt_err(r["fd_rel_err"]),
                     fmt_err(diff) if r["method"] != "explicit64" else "--"])
        csv_rows.append(["unsteady", r["method"], None, r["steps"], r["fwd_s"], r["grad_s"],
                         r["peak_mem_gb"], r["fd_rel_err"], diff])

    with open(RESULTS / "table_gradient_methods.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "t_end_s", "steps", "forward_s", "gradient_s",
                         "peak_memory_gb", "gradient_rel_error", "rel_diff_vs_explicit"])
        writer.writerows(csv_rows)

    return latex_table([header], rows, "llrrrrrr")


def short_cpu(model: str) -> str:
    """Short CPU name, e.g. "Xeon Gold 6342" from "Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz".

    Args:
        model: CPU model string from lscpu or Julia.

    Returns:
        The short name.
    """
    name = model.split("@")[0]
    name = name.replace("Intel(R) Xeon(R) ", "Xeon ").replace(" CPU", "")
    return name.strip()


def savannah_table(out: dict[str, Any]) -> str:
    """Table "Hydrograd.jl vs Hydrograd v2.0" on the Savannah River case.

    Rows: Hydrograd.jl on one thread of the CPU node; Hydrograd v2.0 on one core, on 8 cores
    of the same CPU type, and on an A100, each with forward- and reverse-mode gradients.

    Args:
        out: the performance dictionary (filled in place under "savannah").

    Returns:
        LaTeX tabular.
    """
    py_cpu1, env_cpu1 = read_jsonl(DATA / "savannah_python_cpu1.jsonl")
    py_cpu, env_cpu = read_jsonl(DATA / "savannah_python_cpu.jsonl")
    py_gpu, env_gpu = read_jsonl(DATA / "savannah_python_a100.jsonl")
    julia_file = DATA / "savannah_julia_cpu.json"
    julia = json.loads(julia_file.read_text()) if julia_file.exists() else {}
    out["savannah"] = {"python_cpu_1core": py_cpu1, "python_cpu": py_cpu, "python_a100": py_gpu,
                       "julia_cpu": julia, "environment_cpu_1core": env_cpu1,
                       "environment_cpu": env_cpu, "environment_a100": env_gpu}

    header = ["Code", "Hardware", "Forward (s)", "Gradient mode", "Gradient (s)"]
    rows = []
    csv_rows = []

    # Hydrograd.jl on one thread of the CPU node.
    if julia:
        hw = f"{short_cpu(julia['cpu'])}, {julia['threads']} thread"
        rows.append(["Hydrograd.jl", hw, fmt(julia["fwd_s"]), "forward (ForwardDiff)",
                     fmt(julia["grad_s"])])
        csv_rows.append(["Hydrograd.jl", hw, julia["fwd_s"], "forward", julia["grad_s"]])

    # Hydrograd v2.0: forward- and reverse-mode gradients on 1 core, 8 cores and the A100.
    machines = ((py_cpu1, env_cpu1, "1 core"), (py_cpu, env_cpu, "cores"),
                (py_gpu, env_gpu, "gpu"))
    for records, env, label in machines:
        if not records:
            continue
        r = records[-1]
        if label == "gpu":
            hw = env.get("gpu_model", "A100").replace("NVIDIA ", "")
        elif label == "1 core":
            hw = f"{short_cpu(env['cpu_model'])}, 1 core"
        else:
            hw = f"{short_cpu(env['cpu_model'])}, {env['cores_available']} cores"
        forward = r.get("fwd_case_s", r["fwd_ctrl_s"])
        rows.append(["Hydrograd v2.0", hw, fmt(forward), "forward (JVP)", fmt(r["grad_fwd_s"])])
        rows.append(["Hydrograd v2.0", hw, fmt(forward), "reverse (checkpointed)",
                     fmt(r["grad_rev_s"])])
        csv_rows.append(["Hydrograd v2.0", hw, forward, "forward", r["grad_fwd_s"]])
        csv_rows.append(["Hydrograd v2.0", hw, forward, "reverse", r["grad_rev_s"]])

    with open(RESULTS / "table_savannah.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "hardware", "forward_s", "gradient_mode", "gradient_s"])
        writer.writerows(csv_rows)

    return latex_table([header], rows, "llrlr")


def main() -> None:
    """Build all tables and write performance.json and tables.tex."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}

    # Tables (each also fills its part of ``out``).
    tables = {
        "scaling": scaling_table(out),
        "gradient methods": gradient_methods_table(out),
        "Hydrograd.jl vs Hydrograd v2.0 (Savannah River)": savannah_table(out),
    }

    # Figure-only data: checkpoint sweep and ensemble throughput.
    out["checkpoints"], _ = read_jsonl(DATA / "checkpoints_a100.jsonl")
    for r in out["checkpoints"]:
        r.pop("grad", None)
    ensemble_log = DATA / "ensemble_a100.log"
    out["ensemble"] = ensemble_log.read_text() if ensemble_log.exists() else ""

    # Context that the tables need in their captions.
    out["notes"] = {
        "scaling_case": "write_channel_case default channel, 1000 m x 100 m, hr_hllc, "
                        "second-order MUSCL (Venkatakrishnan), float64; forward = 20 s from "
                        "rest, Tsit5 rtol 1e-6 atol 1e-9; gradient = reverse mode, 64 "
                        "checkpoints, loss mean(xi(T)^2) w.r.t. zone n; Newton dtau0 = 10 s.",
        "v3_superseded": "benchmarks/package/results/*_v3.jsonl (2026-10-05) were measured with "
                         "the first-order scheme (7 Jacobian colours, 55 steps at 8k cells); "
                         "the current default is second order (19 colours, 104 steps), so "
                         "v3 times are 4-9x lower and no longer represent v2.0.",
        "ift_512k_gpu": "Newton + IFT gradient at 512k cells on the A100 (40 GB) ran out of "
                        "device memory (one 22 GiB allocation in the jitted steady loss); the "
                        "table shows '--'. Newton alone (97 s) fits; on the CPU node it runs.",
        "memory": "GPU:XLA allocator peak_bytes_in_use; CPU: process peak RSS incl. the "
                  "Python/JAX baseline (~0.3-1 GB). In the scaling files the peak is cumulative "
                  "over the meshes of one process.",
        "savannah_caveats": "Hydrograd.jl: first-order Roe scheme, Tsit5 default tolerances, "
                            "1 Julia thread; forward time includes case setup and output; "
                            "gradient = (t(4 Adam iterations) - t(1 iteration)) / 3 (ForwardDiff "
                            "through the solve, plus the optimizer and callback). Hydrograd "
                            "v2.0: second-order HLLC scheme, compiled times without setup, "
                            "gradients at rtol 1e-7 (DERIV_RTOL). WRR paper (Liu and Song, "
                            "2025): 18667 s / 150 epochs = 124 s per gradient on a Xeon Gold "
                            "5222 workstation.",
    }

    # Write the JSON and the LaTeX tables.
    (RESULTS / "performance.json").write_text(json.dumps(out, indent=1, default=str))
    with open(RESULTS / "tables.tex", "w") as f:
        for name, table in tables.items():
            f.write(f"% Table: {name} (generated by paper/perf_tables.py)\n{table}\n\n")
    print((RESULTS / "tables.tex").read_text())


if __name__ == "__main__":
    main()
