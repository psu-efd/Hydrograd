"""Figure of the Savannah unsteady roughness calibration (phase 2 report, section 4k).

Reads the per-iteration log lines of the calibration runs of
``savannah_unsteady_inversion.py``,

    iter  k: RMS stage misfit   x.xx mm, n = [n1 n2 n3 n4 n5], t s

and plots
(a) the RMS stage misfit (mm) against the L-BFGS iteration for every run, with the
    model-error floor (implicit Delta t = 30 s against the explicit truth, 0.17 mm);
(b) the final Manning's n of each zone against the true values.

Usage
-----
    python plot_savannah_calibration.py OUT.png LABEL1=LOG1 LABEL2=LOG2 ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

N_TRUE = np.array([0.040, 0.050, 0.030, 0.045, 0.050])     # zones 1-5 (s/m^(1/3))
MODEL_FLOOR_MM = 0.17                                       # implicit vs explicit (mm)
ITER_LINE = re.compile(r"iter\s+(\d+): RMS stage misfit\s+([\d.]+) mm, n = \[([^\]]+)\]")


def read_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Misfit history (mm) and n history (s/m^(1/3)) from a calibration log.

    Args:
        path: Slurm log of one calibration run.

    Returns:
        (misfit [n_iter], n [n_iter, 5]).
    """
    misfits = []
    n_values = []
    for match in ITER_LINE.finditer(path.read_text()):
        misfits.append(float(match.group(2)))
        n_values.append([float(value) for value in match.group(3).split()])
    return np.array(misfits), np.array(n_values)


def main() -> None:
    """Read the logs given on the command line and write the two-panel figure."""
    out = Path(sys.argv[1])
    runs = {}
    for argument in sys.argv[2:]:
        label, log = argument.split("=", 1)
        runs[label] = read_log(Path(log))

    figure, (ax_misfit, ax_n) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    # (a) Misfit histories with the model-error floor.
    for label, (misfit, _) in runs.items():
        ax_misfit.semilogy(np.arange(len(misfit)), misfit, "o-", ms=3, label=label)
    ax_misfit.axhline(MODEL_FLOOR_MM, color="k", ls="--", lw=1, label="model-error floor")
    ax_misfit.set(xlabel="L-BFGS iteration", ylabel="RMS stage misfit (mm)")
    ax_misfit.legend(fontsize=8)

    # (b) Final n per zone against the truth.
    zones = np.arange(1, 6)
    ax_n.plot(zones, N_TRUE, "ks", ms=9, mfc="none", label="true n")
    offsets = np.linspace(-0.2, 0.2, len(runs))
    for offset, (label, (_, n_history)) in zip(offsets, runs.items(), strict=True):
        ax_n.plot(zones + offset, n_history[-1], "o", label=label)
    ax_n.set(xlabel="Manning zone", ylabel="Manning's n (s/m$^{1/3}$)", xticks=zones)
    ax_n.legend(fontsize=8)

    figure.savefig(out, dpi=150)
    print(f"wrote {out}")
    for label, (misfit, n_history) in runs.items():
        relative_error = 100 * np.abs(n_history[-1] - N_TRUE) / N_TRUE
        print(f"{label}: {len(misfit)} iterations, final misfit {misfit[-1]:.3f} mm, "
              f"max |n error| {relative_error.max():.1f} %")


if __name__ == "__main__":
    main()
