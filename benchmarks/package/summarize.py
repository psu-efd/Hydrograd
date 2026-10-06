"""Markdown table from results/scaling_*.jsonl (profile_scaling.py output).

Usage: python summarize.py [NAME.jsonl ...]
(default: scaling_cpu_v3.jsonl scaling_a100_v3.jsonl in ``results/``)

Each input line is one JSON record per mesh size (times in s). Writes ``results/scaling.md``
with one table per input file and prints it.
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).parent / "results"

# (JSON key, column header, format) for each table column.
COLS = [("cells", "cells", "{:,}"), ("setup_s", "setup (s)", "{:.1f}"),
        ("fwd_s", "forward 20 s (s)", "{:.2f}"), ("fwd_steps", "steps", "{}"),
        ("grad_s", "reverse grad (s)", "{:.2f}"), ("n_colors", "colors", "{}"),
        ("newton_s", "steady Newton (s)", "{:.2f}"), ("newton_iters", "Newton its", "{}")]


def table(name: str) -> str:
    """Markdown table of one results file.

    Args:
        name: file name inside ``results/`` (JSON lines from profile_scaling.py).

    Returns:
        Markdown text: a bold title with the device name, then the table. Missing keys
        are shown as "–".
    """
    # Read the non-empty JSON lines.
    lines = (RES / name).read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    # Title and table header.
    device = rows[0]['device'] if rows else ''
    header = "| " + " | ".join(c[1] for c in COLS) + " |"
    out = [f"**{name}** ({device})", "", header, "|" + "---|" * len(COLS)]

    # One table row per mesh size.
    for r in rows:
        cells = [f.format(r[k]) if k in r else "–" for k, _, f in COLS]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    names = sys.argv[1:] or ["scaling_cpu_v3.jsonl", "scaling_a100_v3.jsonl"]
    text = "\n\n".join(table(n) for n in names if (RES / n).exists())
    (RES / "scaling.md").write_text(text + "\n")
    print(text)
