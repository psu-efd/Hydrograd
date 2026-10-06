"""Markdown table from results/scaling_*.jsonl (profile_scaling.py output)."""
import json, sys
from pathlib import Path

RES = Path(__file__).parent / "results"
COLS = [("cells", "cells", "{:,}"), ("setup_s", "setup (s)", "{:.1f}"),
        ("fwd_s", "forward 20 s (s)", "{:.2f}"), ("fwd_steps", "steps", "{}"),
        ("grad_s", "reverse grad (s)", "{:.2f}"), ("n_colors", "colors", "{}"),
        ("newton_s", "steady Newton (s)", "{:.2f}"), ("newton_iters", "Newton its", "{}")]


def table(name: str) -> str:
    rows = [json.loads(l) for l in (RES / name).read_text().splitlines() if l.strip()]
    out = [f"**{name}** ({rows[0]['device'] if rows else ''})", "",
           "| " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        out.append("| " + " | ".join(f.format(r[k]) if k in r else "–" for k, _, f in COLS) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    names = sys.argv[1:] or ["scaling_cpu_v3.jsonl", "scaling_a100_v3.jsonl"]
    text = "\n\n".join(table(n) for n in names if (RES / n).exists())
    (RES / "scaling.md").write_text(text + "\n")
    print(text)
