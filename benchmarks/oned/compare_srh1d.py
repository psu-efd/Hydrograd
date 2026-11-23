"""Comparison of the 1-D mode with SRH-1D v4.0.1 (USBR).

Each case is a folder with

    case.json                 the hydrograd case (see ``hydrograd.io.oned_case``)
    benchmark.json            how the SRH-1D output maps onto the hydrograd case (below)
    srh1d/<name>.srh          the SRH-1D input file of the same case
    srh1d/<name>_OUT_Profile.DAT   the bed profiles written by SRH-1D

``benchmark.json``::

    "title"                   name of the case
    "srh1d_version"           version of SRH-1D that wrote the output
    "srh1d_input"             path of the SRH-1D input file, relative to the case folder
    "srh1d_profile"           path of the SRH-1D profile file, relative to the case folder
    "srh1d_length_unit_m"     length unit of the SRH-1D case in metres (1, or 0.3048)
    "srh1d_time_at_start_h"   SRH-1D time (h) at the start of the hydrograd run
    "srh1d_sections"          [N][2]: SRH-1D river and section number of each hydrograd
                              cross-section, in the order of the hydrograd network
    "junction_sections"       hydrograd cross-sections that are junctions

Cases in ``cases/`` are part of the repository:

``reservoir``
    A 20 km trapezoidal channel (80 m bottom width, 2:1 banks, slope 0.001) with an 18 m
    pool at the dam; seasonal discharge of 100 to 250 m^3/s; silt, fine sand and sand
    from a rating curve; two years of daily steps, 41 cross-sections.
``confluence``
    Two branches that join at the head of a main stem with a pool at the dam; five size
    classes (clay to fine gravel); five years of daily steps, 41 cross-sections.

Further cases with the same layout may be put into ``local/``, which is not part of the
repository (for cases whose data cannot be distributed). They are run as well, and their
results are written to ``local/results/``.

What is compared
----------------
hydrograd is run with the parameter values of the case file and both formulas of the
conveyance (``"segments"``, the default, and ``"section"``; see
``hydrograd.oned.geometry``). At every output time of SRH-1D after the start, the thalweg
of all cross-sections is compared with that of the hydrograd step which ends closest to
it:

    rms = sqrt(mean_i (zb_hydrograd,i - zb_SRH-1D,i)^2)        (m)

The same is given without the cross-sections at the ends of the rivers of SRH-1D, which
are the outlet and the junctions. There the two codes spread a deposit over different
lengths: an end section of SRH-1D owns about half a reach, and SRH-1D repeats a junction
cross-section once per river, with independent beds; the outlet section of hydrograd owns
its whole incoming reach, and hydrograd has one junction section, which owns all its
incoming reaches. For about the same deposited volume the bed of SRH-1D therefore rises
more at these sections (about twice as much at the dam of the reservoir case).

The largest rise of the thalweg and the total bulk volume of the deposits at the last
output time are compared as well.

Usage::

    python benchmarks/oned/compare_srh1d.py                (all cases)
    python benchmarks/oned/compare_srh1d.py reservoir      (the named cases)

Results: ``results/srh1d_comparison.json`` (cases of the repository).

Units: SI.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

import hydrograd  # noqa: F401  (enables float64)
from hydrograd.io.oned_case import read_case
from hydrograd.io.srh1d import read_profile, section_indices
from hydrograd.oned import run
from hydrograd.sediment import bulk_volume_uncompacted

HERE = Path(__file__).resolve().parent

# Folders of the cases: those of the repository, and local ones that are not distributed.
CASES = HERE / "cases"
LOCAL = HERE / "local"

# Formulas of the conveyance that are compared.
CONVEYANCES = ("segments", "section")

# Hours per year (365.25 days).
HOURS_PER_YEAR = 8766.0


def _rms(difference: np.ndarray) -> np.ndarray:
    """Root-mean-square over the cross-sections.

    Args:
        difference: values per output time and cross-section, shape [T, N].

    Returns:
        Root-mean-square per output time, shape [T].
    """
    return np.sqrt(np.mean(difference**2, axis=1))


def compare_case(folder: Path, conveyance: str = "segments") -> dict[str, Any]:
    """Run one case in hydrograd and compare its thalweg and deposits with SRH-1D.

    Args:
        folder: folder of the case (``case.json``, ``benchmark.json`` and the SRH-1D
            files).
        conveyance: formula of the conveyance, ``"segments"`` or ``"section"``.

    Returns:
        Dictionary with the differences (see the module docstring), the run times and the
        sizes of the case.
    """
    # ------------------------------------------------------------------
    # 1. The case, with the requested conveyance and every step recorded
    # ------------------------------------------------------------------
    case = read_case(folder / "case.json")
    meta = json.loads((folder / "benchmark.json").read_text())
    config = dataclasses.replace(
        case.config,
        hydraulics=dataclasses.replace(case.config.hydraulics, conveyance=conveyance),
        morph=dataclasses.replace(case.config.morph, record_every=1),
    )

    # ------------------------------------------------------------------
    # 2. hydrograd run (the first call includes the compilation)
    # ------------------------------------------------------------------
    def run_with(theta):
        """The run as a function of the parameters."""
        return run(config, case.network, theta, case.forcing, case.bed0)

    compiled = jax.jit(run_with)

    start = time.perf_counter()
    result = compiled(case.theta)
    result.deposition.block_until_ready()
    first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    result = compiled(case.theta)
    result.deposition.block_until_ready()
    run_seconds = time.perf_counter() - start

    # ------------------------------------------------------------------
    # 3. SRH-1D profiles in the section order of the hydrograd network
    # ------------------------------------------------------------------
    profile = read_profile(folder / meta["srh1d_profile"], meta["srh1d_length_unit_m"])
    index = section_indices(profile, [tuple(pair) for pair in meta["srh1d_sections"]])
    if index.shape[0] != case.network.n_sections:
        raise ValueError(f"{folder}: 'srh1d_sections' must name every cross-section once")

    # Output times of SRH-1D after the start and within the hydrograd run (h). The runs
    # of the two codes may end within one step of each other (years of 365.25 days in
    # SRH-1D, whole days in hydrograd), so an output up to one step after the last
    # hydrograd step still counts.
    hours = np.asarray(result.time_end) / 3600.0
    last_step_hours = float(case.forcing.dt[-1]) / 3600.0
    hours_srh = profile.time - meta["srh1d_time_at_start_h"]
    within = (hours_srh > 0.0) & (hours_srh <= hours[-1] + last_step_hours)
    hours_srh = hours_srh[within]
    thalweg_srh = profile.thalweg[within][:, index]

    # The hydrograd step that ends closest to each output time.
    steps = np.array([int(np.argmin(np.abs(hours - hour))) for hour in hours_srh])
    thalweg = np.asarray(result.thalweg)[steps]

    # ------------------------------------------------------------------
    # 4. Differences
    # ------------------------------------------------------------------
    rms = _rms(thalweg - thalweg_srh)
    thalweg_initial = np.asarray(case.bed0.z).min(axis=1)

    # Bulk volume of the deposits up to the last output time, all cross-sections. SRH-1D
    # holds the deposit of a junction in the copies of its rivers; the sum takes all.
    deposition = result.deposition[: steps[-1] + 1]
    bulk_volume = bulk_volume_uncompacted(deposition, case.theta, config.sediment)
    bulk_volume_srh = profile.deposit_volume[within][-1]

    out: dict[str, Any] = {
        "title": meta["title"],
        "conveyance": conveyance,
        "n_sections": int(case.network.n_sections),
        "n_classes": int(case.bed0.p_active.shape[0]),
        "n_steps": int(case.forcing.n_steps),
        "first_call_seconds": first_seconds,
        "run_seconds": run_seconds,
        "time_years": (hours_srh / HOURS_PER_YEAR).tolist(),
        "rms_thalweg_m": rms.tolist(),
        "rms_thalweg_final_m": float(rms[-1]),
        "rms_thalweg_max_m": float(rms.max()),
        "largest_rise_m": {
            "hydrograd": float((thalweg[-1] - thalweg_initial).max()),
            "srh1d": float((thalweg_srh[-1] - thalweg_initial).max()),
        },
        "bulk_deposit_m3": {
            "hydrograd": float(np.sum(np.asarray(bulk_volume))),
            "srh1d": float(np.sum(bulk_volume_srh)),
        },
    }

    # The same without the sections at the ends of the rivers of SRH-1D: the junctions
    # and the outlet (the last section of the network).
    n_sections = case.network.n_sections
    river_ends = [int(section) for section in meta.get("junction_sections", [])]
    river_ends.append(n_sections - 1)
    keep = np.array([i for i in range(n_sections) if i not in river_ends])
    rms_without = _rms(thalweg[:, keep] - thalweg_srh[:, keep])
    out["river_end_sections"] = sorted(river_ends)
    out["rms_thalweg_without_river_ends_m"] = rms_without.tolist()
    out["rms_thalweg_without_river_ends_final_m"] = float(rms_without[-1])
    out["rms_thalweg_without_river_ends_max_m"] = float(rms_without.max())

    return out


def case_folders(names: list[str]) -> list[Path]:
    """Folders of the cases to run.

    Args:
        names: names of the wanted cases; empty for all.

    Returns:
        The folders in ``cases/`` and ``local/`` that hold a ``benchmark.json``, limited
        to ``names`` if given.
    """
    folders = []
    for root in (CASES, LOCAL):
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if (folder / "benchmark.json").is_file():
                folders.append(folder)

    if names:
        unknown = sorted(set(names) - {folder.name for folder in folders})
        if unknown:
            raise SystemExit(f"unknown case(s) {unknown}")
        folders = [folder for folder in folders if folder.name in names]

    return folders


def main(arguments: list[str]) -> None:
    """Run the comparison and write the results.

    Args:
        arguments: names of the cases to run; empty for all.
    """
    results: dict[str, dict[str, Any]] = {}
    results_local: dict[str, dict[str, Any]] = {}

    for folder in case_folders(arguments):
        by_conveyance = {}
        for conveyance in CONVEYANCES:
            out = compare_case(folder, conveyance)
            by_conveyance[conveyance] = out

            line = (
                f"{folder.name} ({conveyance}): RMS thalweg difference to SRH-1D "
                f"{out['rms_thalweg_final_m']:.3f} m at the end, {out['rms_thalweg_max_m']:.3f} m "
                f"at most ({out['rms_thalweg_without_river_ends_final_m']:.3f} m and "
                f"{out['rms_thalweg_without_river_ends_max_m']:.3f} m without the river ends)"
            )
            rise = out["largest_rise_m"]
            volume = out["bulk_deposit_m3"]
            line += (
                f"; largest rise {rise['hydrograd']:.2f} m (SRH-1D {rise['srh1d']:.2f} m); "
                f"bulk deposit {volume['hydrograd']:.3e} m^3 (SRH-1D {volume['srh1d']:.3e} m^3); "
                f"run {out['run_seconds']:.2f} s"
            )
            print(line, flush=True)

        if folder.parent == LOCAL:
            results_local[folder.name] = by_conveyance
        else:
            results[folder.name] = by_conveyance

    # Results of the cases of the repository; a run of selected cases keeps the others.
    if results:
        path = HERE / "results" / "srh1d_comparison.json"
        path.parent.mkdir(exist_ok=True)
        stored = json.loads(path.read_text()) if path.is_file() else {}
        stored.update(results)
        path.write_text(json.dumps(stored, indent=1))
        print(f"wrote {path}")

    # Results of local cases stay in the local folder.
    if results_local:
        path = LOCAL / "results" / "srh1d_comparison.json"
        path.parent.mkdir(exist_ok=True)
        stored = json.loads(path.read_text()) if path.is_file() else {}
        stored.update(results_local)
        path.write_text(json.dumps(stored, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
