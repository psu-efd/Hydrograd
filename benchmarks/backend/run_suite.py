"""Run benchmark command lists sequentially; append JSON results to results/<suite>.jsonl.

Each suite is a list of benchmark commands (script name + arguments). Every command runs in
its own Python process (clean timings and peak-memory numbers); the last stdout line that
starts with ``{`` is appended to ``results/<suite>.jsonl``. A failed command appends
``{"cmd": ..., "error": <last 400 characters of stderr>}`` instead.

Usage:
    python run_suite.py savannah steady unsteady scaling

Suites: scaling, savannah, roar_scaling, roar_savannah, steady, unsteady. A full run takes
hours; summarize.py turns the .jsonl files into tables and figures.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).parent

# Explicit-solver scaling on the Windows PC: (backend, device, dtype, extra flags) per size.
_SCALING_CFGS = (
    ("jax", "cpu", "float64", ""),
    ("torch", "gpu", "float64", "--compile --repeats 2"),
    ("torch", "gpu", "float32", "--compile --repeats 2"),
    ("torch", "cpu", "float64", "--repeats 1"),
    ("torch", "gpu", "float64", "--repeats 1"),
)

# Roar A100 node: same layout as _SCALING_CFGS.
_ROAR_SCALING_CFGS = (
    ("jax", "gpu", "float64", "--repeats 2"),
    ("jax", "gpu", "float32", "--repeats 2"),
    ("torch", "gpu", "float64", "--compile --repeats 2"),
    ("torch", "gpu", "float32", "--compile --repeats 2"),
    ("jax", "cpu", "float64", "--repeats 1"),
)

# Common prefix of the Savannah River runs (1306 cells, 10,000 steps of 0.02 s).
_SAV_PREFIX = (
    "bench_explicit.py --mesh savannah --dt 0.02 --n_save 10 --k_inner 1000 "
    "--mlp 2,3,3,1 --repeats 1"
)

_SAVANNAH_VARIANTS = (
    "--backend jax --mode nn --ckpt",
    "--backend jax --mode nn --ckpt --grad_mode fwd",
    "--backend jax --mode field --ckpt",
    "--backend torch --device gpu --mode nn --ckpt --compile",
    "--backend torch --device gpu --mode field --ckpt --compile",
)

_ROAR_SAVANNAH_VARIANTS = (
    "--backend jax --device gpu --mode nn --ckpt",
    "--backend jax --device gpu --mode field --ckpt",
    "--backend jax --device cpu --mode nn --ckpt",
    "--backend torch --device gpu --mode nn --ckpt --compile",
    "--backend torch --device gpu --mode field --ckpt --compile",
)

# Template of one explicit-scaling command.
_SCALING_CMD = (
    "bench_explicit.py --backend {b} --device {d} --dtype {t} --cells {n} --mode nn --ckpt {x}"
)

SUITES = {
    # Uncompiled (--repeats 1) runs only up to 8000 cells.
    "scaling": [
        _SCALING_CMD.format(b=b, d=d, t=t, n=n, x=x)
        for n in (2000, 8000, 32000, 128000)
        for b, d, t, x in _SCALING_CFGS
        if not (n > 8000 and x == "--repeats 1")
    ],
    "savannah": [f"{_SAV_PREFIX} {v}" for v in _SAVANNAH_VARIANTS],
    # Roar A100 node (Linux, float64-capable GPU): adds the missing JAX-GPU runs and repeats
    # the torch-GPU and JAX-CPU runs on the same node, so all backends share one machine.
    "roar_scaling": [
        _SCALING_CMD.format(b=b, d=d, t=t, n=n, x=x)
        for n in (2000, 8000, 32000, 128000, 512000)
        for b, d, t, x in _ROAR_SCALING_CFGS
        if not (n > 128000 and d == "cpu")
    ],
    "roar_savannah": [f"{_SAV_PREFIX} {v}" for v in _ROAR_SAVANNAH_VARIANTS],
    # Newton (IFT adjoint; GMRES adjoint up to 8000 cells), then explicit pseudo-time march.
    "steady": (
        [
            f"steady_ift.py --method {mth} --cells {n}"
            for n in (2000, 8000, 32000)
            for mth in ("newton_ift", "newton_gmres")
            if not (mth == "newton_gmres" and n > 8000)
        ]
        + [
            f"steady_ift.py --method explicit_fwd --cells 2000 --n_steps {k}"
            for k in (40000, 160000)
        ]
        + [
            f"steady_ift.py --method explicit_bptt --cells 2000 --n_steps {k}"
            for k in (40000, 160000)
        ]
    ),
    # Explicit reference first (it writes the reference file the implicit runs compare to).
    "unsteady": (
        ["implicit_unsteady.py --method explicit --cells 4000"]
        + [
            f"implicit_unsteady.py --method implicit --cells 4000 --mult {mu} --theta {th}"
            for th in (1.0, 0.55)
            for mu in (5, 20, 50, 100)
        ]
    ),
}


def main() -> None:
    """Run every suite named on the command line and append one JSON record per command."""
    for suite in sys.argv[1:]:
        out = HERE / "results" / f"{suite}.jsonl"
        for cmd in SUITES[suite]:
            # Run the command in a fresh process from this folder.
            t0 = time.time()
            p = subprocess.run([PY] + cmd.split(), cwd=HERE, capture_output=True, text=True)

            # Keep the last JSON line, or record the error.
            json_lines = [line for line in p.stdout.splitlines() if line.startswith("{")]
            if json_lines:
                rec = json_lines[-1]
            else:
                rec = json.dumps({"cmd": cmd, "error": p.stderr[-400:]})

            with out.open("a") as f:
                f.write(rec + "\n")
            print(f"[{suite}] {time.time() - t0:7.1f}s  {cmd}", flush=True)


if __name__ == "__main__":
    main()
