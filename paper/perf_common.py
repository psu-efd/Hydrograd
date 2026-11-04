"""Shared helpers of the performance scripts of the Hydrograd v2.0 paper (Sect. "Performance").

The scripts ``perf_profile_scaling.py``, ``perf_checkpoints.py`` and
``perf_gradient_methods.py`` time jitted JAX functions and record the hardware and software
they ran on. This module holds the pieces they share:

- ``timed``: call a function twice and time both calls. The first call includes the XLA
  compilation; the second is the compiled run time that the paper reports.
- ``peak_memory_gb``: peak memory of the run, in GB (10^9 bytes). On a GPU this is the
  device allocator's ``peak_bytes_in_use``; on the CPU it is the peak resident set size of
  the process (``ru_maxrss``), which includes the Python/JAX baseline (about 0.3-0.5 GB).
- ``environment_record``: CPU model, core count, GPU model, Slurm job, software versions and
  the date, written once per output file so that every number can be traced to a machine.
"""

from __future__ import annotations

import datetime
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import jax


def timed(f: Callable[..., Any], *args: Any) -> tuple[Any, float, float]:
    """Call ``f(*args)`` twice and time both calls.

    Args:
        f: function (typically jitted) returning JAX arrays.
        *args: its arguments.

    Returns:
        (output of the second call, wall time of the first call including compilation (s),
        wall time of the second, compiled call (s)).
    """
    # First call: compilation plus run.
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    t_first = time.perf_counter() - t0

    # Second call: compiled run only.
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    t_second = time.perf_counter() - t0

    return out, t_first, t_second


def peak_memory_gb() -> float:
    """Peak memory of this process so far (GB).

    Returns:
        On a GPU: the device allocator's peak bytes in use / 1e9. On the CPU: the peak
        resident set size of the process / 1e9 (Linux reports ``ru_maxrss`` in KiB).
    """
    device = jax.devices()[0]

    # GPU: the XLA allocator keeps its own peak counter.
    if device.platform == "gpu":
        stats = device.memory_stats() or {}
        return float(stats.get("peak_bytes_in_use", float("nan"))) / 1.0e9

    # CPU: peak resident set size of the whole process.
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss_kib * 1024.0 / 1.0e9


def _command_output(command: list[str]) -> str:
    """Run a shell command and return its standard output ("" on failure).

    Args:
        command: program and arguments.

    Returns:
        Stripped standard output.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def environment_record() -> dict[str, Any]:
    """Hardware, software and date of this run.

    Returns:
        Dictionary with the date (ISO), host name, Slurm job ID, CPU model, number of cores
        available to the process, GPU model (or ""), JAX device kind, and the versions of
        Python, JAX, jaxlib, diffrax, NumPy and SciPy.
    """
    import diffrax
    import numpy
    import scipy

    # CPU model from lscpu (first "Model name" line).
    cpu_model = ""
    for line in _command_output(["lscpu"]).splitlines():
        if line.startswith("Model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break

    # GPU model from nvidia-smi (empty on CPU nodes).
    gpu_model = _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])

    # Software versions.
    versions = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__ if hasattr(jax.lib, "__version__") else "",
        "diffrax": diffrax.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }

    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cpu_model": cpu_model,
        "cores_available": len(os.sched_getaffinity(0)),
        "gpu_model": gpu_model,
        "jax_device": jax.devices()[0].device_kind,
        "versions": versions,
        "argv": sys.argv,
    }
