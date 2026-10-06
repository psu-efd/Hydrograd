"""Run benchmark command lists sequentially; append JSON results to results/<suite>.jsonl."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).parent

SUITES = {
    "scaling": [f"bench_explicit.py --backend {b} --device {d} --dtype {t} --cells {n} --mode nn --ckpt {x}"
                for n in (2000, 8000, 32000, 128000)
                for b, d, t, x in (("jax", "cpu", "float64", ""),
                                   ("torch", "gpu", "float64", "--compile --repeats 2"),
                                   ("torch", "gpu", "float32", "--compile --repeats 2"),
                                   ("torch", "cpu", "float64", "--repeats 1"),
                                   ("torch", "gpu", "float64", "--repeats 1"))
                if not (n > 8000 and x == "--repeats 1")],
    "savannah": [f"bench_explicit.py --mesh savannah --dt 0.02 --n_save 10 --k_inner 1000 --mlp 2,3,3,1 --repeats 1 {v}"
                 for v in ("--backend jax --mode nn --ckpt",
                           "--backend jax --mode nn --ckpt --grad_mode fwd",
                           "--backend jax --mode field --ckpt",
                           "--backend torch --device gpu --mode nn --ckpt --compile",
                           "--backend torch --device gpu --mode field --ckpt --compile")],
    # Roar A100 node (Linux, float64-capable GPU): adds the missing JAX-GPU runs and repeats
    # the torch-GPU and JAX-CPU runs on the same node, so all backends share one machine.
    "roar_scaling": [f"bench_explicit.py --backend {b} --device {d} --dtype {t} --cells {n} --mode nn --ckpt {x}"
                     for n in (2000, 8000, 32000, 128000, 512000)
                     for b, d, t, x in (("jax", "gpu", "float64", "--repeats 2"),
                                        ("jax", "gpu", "float32", "--repeats 2"),
                                        ("torch", "gpu", "float64", "--compile --repeats 2"),
                                        ("torch", "gpu", "float32", "--compile --repeats 2"),
                                        ("jax", "cpu", "float64", "--repeats 1"))
                     if not (n > 128000 and d == "cpu")],
    "roar_savannah": [f"bench_explicit.py --mesh savannah --dt 0.02 --n_save 10 --k_inner 1000 --mlp 2,3,3,1 --repeats 1 {v}"
                      for v in ("--backend jax --device gpu --mode nn --ckpt",
                                "--backend jax --device gpu --mode field --ckpt",
                                "--backend jax --device cpu --mode nn --ckpt",
                                "--backend torch --device gpu --mode nn --ckpt --compile",
                                "--backend torch --device gpu --mode field --ckpt --compile")],
    "steady": ([f"steady_ift.py --method {mth} --cells {n}" for n in (2000, 8000, 32000)
                for mth in ("newton_ift", "newton_gmres") if not (mth == "newton_gmres" and n > 8000)]
               + [f"steady_ift.py --method explicit_fwd --cells 2000 --n_steps {k}" for k in (40000, 160000)]
               + [f"steady_ift.py --method explicit_bptt --cells 2000 --n_steps {k}" for k in (40000, 160000)]),
    "unsteady": (["implicit_unsteady.py --method explicit --cells 4000"]
                 + [f"implicit_unsteady.py --method implicit --cells 4000 --mult {mu} --theta {th}"
                    for th in (1.0, 0.55) for mu in (5, 20, 50, 100)]),
}


def main() -> None:
    for suite in sys.argv[1:]:
        out = HERE / "results" / f"{suite}.jsonl"
        for cmd in SUITES[suite]:
            t0 = time.time()
            p = subprocess.run([PY] + cmd.split(), cwd=HERE, capture_output=True, text=True)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = lines[-1] if lines else json.dumps({"cmd": cmd, "error": p.stderr[-400:]})
            with out.open("a") as f:
                f.write(rec + "\n")
            print(f"[{suite}] {time.time() - t0:7.1f}s  {cmd}", flush=True)


if __name__ == "__main__":
    main()
