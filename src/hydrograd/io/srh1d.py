"""Reader of the bed-profile output of SRH-1D.

SRH-1D (USBR) writes ``<case>_OUT_Profile.DAT``: one line per cross-section and output
time, with whitespace-separated columns (the file's own header lists them)

    t  riv  xcid  xc  xt  q  qlatf  zb  z  zba  fslope  topw  hydrad  vel
    d16  d30  d50  d84  vol_dep  sed_load  bank_eros  rock_eros  vol_water

of which this reader keeps

    t         time (h)
    riv       river number (1-based)
    xc        cross-section number within its river (1-based, upstream to downstream)
    xt        location of the cross-section along its river (ft or m)
    zb        thalweg elevation (ft or m)
    z         water-surface elevation (ft or m)
    vol_dep   cumulative bulk volume of deposition of the cross-section (ft^3 or m^3)

Lines that start with ``#`` and the line of column names are skipped. SRH-1D writes its
last output time twice; the later record is kept.

The file is in the units of the SRH-1D case (US customary or SI). ``length_unit``
converts lengths and volumes to metres and cubic metres (0.3048 for a case in feet).

Only this output file is read. SRH-1D input files are not read: a 1-D case of hydrograd
is given by its own case file (:mod:`hydrograd.io.oned_case`).

Units: time in hours; lengths in m and volumes in m^3 after the conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["Srh1dProfile", "read_profile", "section_indices"]

# Columns of ``*_OUT_Profile.DAT`` (0-based, after splitting a line at whitespace).
COLUMN_TIME = 0
COLUMN_RIVER = 1
COLUMN_SECTION = 3
COLUMN_LOCATION = 4
COLUMN_THALWEG = 7
COLUMN_WATER_SURFACE = 8
COLUMN_DEPOSIT_VOLUME = 18


class Srh1dProfile(NamedTuple):
    """Bed profiles of an SRH-1D run at its T output times, for its M cross-sections.

    The cross-sections are ordered by river number, then by section number.

    Attributes:
        time: output times (h), ascending, shape [T].
        river: river number of each cross-section (1-based), shape [M].
        section: number of each cross-section within its river (1-based), shape [M].
        location: location of each cross-section along its river (m), shape [M].
        thalweg: thalweg elevation (m), shape [T, M].
        water_surface: water-surface elevation (m), shape [T, M].
        deposit_volume: cumulative bulk volume of deposition (m^3), shape [T, M].
    """

    time: np.ndarray
    river: np.ndarray
    section: np.ndarray
    location: np.ndarray
    thalweg: np.ndarray
    water_surface: np.ndarray
    deposit_volume: np.ndarray


def read_profile(path: str | Path, length_unit: float = 1.0) -> Srh1dProfile:
    """Read ``<case>_OUT_Profile.DAT`` of SRH-1D.

    Args:
        path: the profile file.
        length_unit: length of the file's length unit in metres (1 for SI, 0.3048 for a
            case in feet).

    Returns:
        The profiles (:class:`Srh1dProfile`), in metres and cubic metres.
    """
    # ------------------------------------------------------------------
    # 1. Records by (time, river, section); a repeated record replaces the earlier one
    # ------------------------------------------------------------------
    records: dict[tuple[float, int, int], tuple[float, float, float, float]] = {}
    with open(path) as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "t,")):
                continue
            parts = stripped.split()
            key = (
                float(parts[COLUMN_TIME]),
                int(parts[COLUMN_RIVER]),
                int(parts[COLUMN_SECTION]),
            )
            records[key] = (
                float(parts[COLUMN_LOCATION]),
                float(parts[COLUMN_THALWEG]),
                float(parts[COLUMN_WATER_SURFACE]),
                float(parts[COLUMN_DEPOSIT_VOLUME]),
            )
    if not records:
        raise ValueError(f"{path}: no profile records found")

    # ------------------------------------------------------------------
    # 2. Output times and cross-sections
    # ------------------------------------------------------------------
    times = sorted({time for time, _, _ in records})
    sections = sorted({(river, section) for _, river, section in records})

    missing = [
        (time, river, section)
        for time in times
        for river, section in sections
        if (time, river, section) not in records
    ]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} record(s) missing, the first at (t, riv, xc) = {missing[0]}"
        )

    # ------------------------------------------------------------------
    # 3. Arrays [T, M, 4]: location, thalweg, water surface, deposit volume
    # ------------------------------------------------------------------
    table = np.array(
        [[records[(time, river, section)] for river, section in sections] for time in times]
    )

    return Srh1dProfile(
        time=np.array(times),
        river=np.array([river for river, _ in sections]),
        section=np.array([section for _, section in sections]),
        location=table[0, :, 0] * length_unit,
        thalweg=table[:, :, 1] * length_unit,
        water_surface=table[:, :, 2] * length_unit,
        deposit_volume=table[:, :, 3] * length_unit**3,
    )


def section_indices(profile: Srh1dProfile, sections: list[tuple[int, int]]) -> np.ndarray:
    """Positions of given SRH-1D cross-sections in the arrays of a profile.

    Used to bring SRH-1D results into the section order of a hydrograd network:
    ``profile.thalweg[:, section_indices(profile, sections)]``.

    Args:
        profile: profiles read by :func:`read_profile`.
        sections: pairs (river number, section number), in the wanted order.

    Returns:
        Index of each pair along the section axis of the profile, shape [len(sections)].
    """
    position = {
        (int(river), int(section)): index
        for index, (river, section) in enumerate(zip(profile.river, profile.section, strict=True))
    }

    unknown = [tuple(pair) for pair in sections if tuple(pair) not in position]
    if unknown:
        raise KeyError(f"SRH-1D profile has no cross-section (riv, xc) = {unknown[0]}")

    return np.array([position[tuple(pair)] for pair in sections])
