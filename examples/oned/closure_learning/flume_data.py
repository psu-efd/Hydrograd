"""Data of the downstream-fining flume experiments of Seal et al. (1997), runs 1 to 3,
and of run 6 of Toro-Escobar et al. (2000) (second part of the file).

Source: Seal, R., Paola, C., Parker, G., Southard, J. B., and Wilcock, P. R. (1997),
"Experiments on downstream fining of gravel: I. Narrow-channel runs", Journal of
Hydraulic Engineering 123(10), 874-884. The tailgate depth is from Toro-Escobar et al.
(2000), Journal of Hydraulic Engineering 126(3), 198-208, Table 1, and Fig. 2 of Seal et al.

A poorly sorted gravel-sand mixture was fed at the upstream end of a flume 0.3 m wide with
a smooth floor; a wedge of sediment built up and its gravel fined downstream. The three
runs differ only in the feed rate.

How the numbers were obtained:

- set-up, durations, masses, front positions: text and Table 3 of the paper (exact);
- grain sizes of the deposit: Tables 1 and 2 of the paper (exact);
- feed size distribution: bar heights of Fig. 4 read from the page image (the bars read
  sum to 100.1 %; reading error about 0.1 %);
- bed profiles: lines of Fig. 5 read from the page image at the measuring stations
  (reading error about 0.01 m).

Units: SI unless a name says otherwise; psi = log2(d in mm) (the paper lists phi = -psi).
"""

from __future__ import annotations

import numpy as np

# Flume.
LENGTH = 45.0             # m, from the feed point to the end of the channel
WIDTH = 0.3               # m
WALL_HEIGHT = 1.2         # m
FLOOR_SLOPE = 0.002
DISCHARGE = 0.049         # m^3/s
# Water level held by the tailgate, in m above the floor at the end of the channel
# (x = 45 m), for runs 1 to 3. The paper gives no numbers. Run 2: its Fig. 2 shows the
# final water surface and the measured floor on the same axes; the pond stands 0.41 to
# 0.42 m above the floor. Runs 1 and 3: from the differences between the runs in other
# publications of the same data, which state the levels in another datum (about 0.03 m
# higher: 0.40 and 0.45 m for runs 1 and 2 in Wu, 2004, J. Hydraul. Eng. 130(10), whose
# Fig. 3 shows the measured water surfaces of both runs; 0.40, 0.45 and 0.50 m in Cui,
# 2007, Water Resour. Res. 43, W10436, above the floor at the end of a 50 m channel).
# Toro-Escobar et al. (2000, J. Hydraul. Eng. 126(3)) list a downstream depth of 0.45 m
# for run 3. The values are good to about 0.02 m; 0.01 m moves the gravel front of the
# model by about 1 m.
TAILWATER = {1: 0.38, 2: 0.42, 3: 0.46}
GRAIN_DENSITY = 2650.0    # kg/m^3 (assumed in the paper)

# Feed: sieve sizes at half-psi steps (mm, lower bound of each class; the last class is
# the pan, finer than 0.125 mm) and the percentage of each class in the target mixture and
# in the mixture fed in run 2.
SIEVE_MM = np.array([64, 45.3, 32, 22.6, 16, 11.3, 8, 5.66, 4, 2.83, 2, 1.41, 1, 0.707, 0.5, 0.354, 0.25, 0.177, 0.125, 0.088])
FEED_TARGET_PERCENT = np.array([3.6, 3.6, 5.4, 9.8, 9.4, 6.4, 5.7, 6.6, 8.3, 5.8, 2.3, 2.8, 3.5, 4.5, 5.5, 6.0, 4.7, 3.1, 1.3, 1.8])
FEED_RUN2_PERCENT = np.array([3.9, 3.1, 6.3, 9.7, 8.4, 6.7, 6.0, 6.8, 8.6, 6.0, 2.0, 2.5, 3.5, 4.2, 5.1, 6.0, 5.2, 3.4, 1.4, 1.1])

# Runs (Table 3): duration, total mass fed, final position of the gravel front, final
# thickness of the deposit at the upstream end, mean water depth over the deposit.
RUNS = {
    1: {"hours": 16.83, "mass_kg": 11413.0, "front_m": 36.8, "upstream_thickness_m": 0.7754, "depth_m": 0.120},
    2: {"hours": 32.4, "mass_kg": 10989.0, "front_m": 36.8, "upstream_thickness_m": 0.71, "depth_m": 0.139},
    3: {"hours": 65.0, "mass_kg": 11017.5, "front_m": 35.0, "upstream_thickness_m": 0.7131, "depth_m": 0.149},
}

# Bed profiles (Fig. 5): elevation as plotted (m) at the stations (m) and times (h). The
# floor of each plot is ``FLOOR_PLOTTED``; a profile ends at the foot of the gravel front.
STATIONS = np.array([2.0, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40])
FLOOR_PLOTTED = {1: (0.25, 0.17), 2: (0.075, 0.0), 3: (0.075, 0.0)}   # at x = 2 m and x = 45 m
PROFILES = {
    1: {
        2.0: [0.535, 0.48, 0.44, 0.26],
        5.0: [0.68, 0.60, 0.59, 0.50, 0.485, 0.43, 0.425, 0.22],
        6.5: [0.74, 0.65, 0.625, 0.56, 0.535, 0.50, 0.475, 0.31, 0.235],
        10.0: [0.855, 0.75, 0.73, 0.675, 0.625, 0.57, 0.56, 0.525, 0.475, 0.43, 0.24],
        16.83: [1.03, 0.93, 0.905, 0.80, 0.77, 0.73, 0.685, 0.64, 0.61, 0.56, 0.53, 0.475, 0.53, 0.425, 0.41, 0.235],
    },
    2: {
        3.0: [0.345, 0.28, 0.26, 0.06],
        8.0: [0.46, 0.405, 0.38, 0.325, 0.27, 0.215, 0.065],
        14.0: [0.55, 0.50, 0.465, 0.43, 0.38, 0.33, 0.32, 0.29, 0.115, 0.10, 0.065, 0.03],
        22.0: [0.65, 0.615, 0.565, 0.53, 0.51, 0.45, 0.41, 0.365, 0.325, 0.28, 0.265, 0.02],
        32.4: [0.785, 0.765, 0.68, 0.625, 0.60, 0.55, 0.51, 0.455, 0.415, 0.385, 0.35, 0.315, 0.285, 0.275, 0.23, 0.09],
    },
    3: {
        6.0: [0.39, 0.305, 0.145, 0.06],
        14.0: [0.50, 0.39, 0.385, 0.325, 0.295, 0.055],
        26.0: [0.57, 0.50, 0.475, 0.42, 0.39, 0.355, 0.31, 0.11, 0.035],
        42.0: [0.70, 0.595, 0.65, 0.50, 0.485, 0.43, 0.415, 0.36, 0.325, 0.30, 0.22, 0.08, 0.05],
        64.0: [0.79, 0.715, 0.73, 0.64, 0.60, 0.535, 0.495, 0.455, 0.445, 0.395, 0.375, 0.34, 0.305, 0.285, 0.045, 0.085],
    },
}

# Table 1: grain sizes of the deposit of run 2 laid down between hours 14 and 22, at the
# sampling stations (m); psi = log2(d in mm). The last station lies in the front deposit.
TABLE1_X = np.array([3.0, 6.5, 10.0, 13.5, 17.0, 20.5, 24.0, 27.5])
TABLE1_PSI = {
    90: np.array([5.44, 5.20, 5.09, 5.08, 5.03, 4.93, 4.75, 4.44]),
    84: np.array([5.10, 4.88, 4.80, 4.79, 4.80, 4.68, 4.48, 4.09]),
    50: np.array([2.75, 2.72, 2.72, 2.59, 2.90, 2.65, 2.67, 2.14]),
    16: np.array([-0.34, -0.40, -0.57, -0.48, -0.53, -0.59, -0.44, -1.08]),
}
TABLE1_HOURS = (14.0, 22.0)

# Table 2: D90 and D50 (mm) of the deposit laid down between two times (h), at the
# upstream end and at the downstream end of the main deposit.
TABLE2 = {
    1: {(6.5, 10.0): {"D90": (46.6, 33.07), "D50": (6.98, 7.37)}, (2.0, 5.0): {"D90": (45.68, 32.89), "D50": (6.5, 6.5)}},
    2: {(14.0, 22.0): {"D90": (43.35, 32.57), "D50": (6.73, 7.46)}, (8.0, 14.0): {"D90": (43.15, 29.24), "D50": (6.71, 5.78)},
        (3.0, 8.0): {"D90": (46.98, 35.5), "D50": (7.48, 6.28)}},
    3: {(26.0, 42.0): {"D90": (56.37, 35.26), "D50": (9.034, 6.73)}, (14.0, 26.0): {"D90": (57.79, 36.0), "D50": (9.752, 6.63)},
        (6.0, 14.0): {"D90": (49.14, 37.53), "D50": (8.33, 7.36)}},
}
# Surface layer at the end of the run: D90 and D50 (mm), upstream and downstream.
SURFACE = {
    1: {"D90": (39.67, 29.65), "D50": (8.81, 8.11)},
    2: {"D90": (57.98, 27.1), "D50": (13.72, 6.83)},
    3: {"D90": (56.78, 26.08), "D50": (15.89, 6.63)},
}


def floor_plotted(run: int, x):
    """Floor of the plots of Fig. 5 at the stations x (m)."""
    upstream, downstream = FLOOR_PLOTTED[run]
    return upstream + (downstream - upstream) * (np.asarray(x) - 2.0) / (LENGTH - 2.0)


def thickness_profiles(run: int) -> dict:
    """Thickness of the deposit of a run: {hours: (stations, thickness)} (m)."""
    out = {}
    for hours, values in PROFILES[run].items():
        x = STATIONS[: len(values)]
        out[hours] = (x, np.maximum(np.asarray(values) - floor_plotted(run, x), 0.0))
    return out


def feed_rate(run: int) -> float:
    """Mean feed rate of a run (kg/s): total mass over duration."""
    return RUNS[run]["mass_kg"] / (RUNS[run]["hours"] * 3600.0)


# ---------------------------------------------------------------------------------------
# Run 6 of Toro-Escobar, C. M., Paola, C., Parker, G., Wilcock, P. R., and Southard, J. B.
# (2000), "Experiments on downstream fining of gravel. II: Wide and sandy runs", Journal
# of Hydraulic Engineering 126(3), 198-208.
#
# The same flume width (0.3 m, smooth walls) with other conditions: a sandier feed (53 %
# sand by the bars of Fig. 2c; "55 %" in the paper), a quarter of the discharge, a floor
# five times steeper (0.01, roughened with gravel set in the concrete) and a tailgate low
# enough that no pond and no gravel front formed. The deposit is a short, steep, concave
# wedge; fine gravel and sand travel on beyond it.
#
# Coordinates as in the paper: x from the sediment feed point; elevations above the top of
# the step at the end of the sloping floor. The sloping floor is the line
# z = 0.01 (49.5 - x) (it fits the plotted initial bed at x = 13 to 18 m and the step
# "49 m downstream of the feed point"). The first survey section, at x = 13 m, has the
# thickest deposit; the flow between the feed point and the deposit was fast and shallow.
#
# How the numbers were obtained:
#
# - set-up: Table 1 and text of the paper. The feed rate is uncertain: Table 1 lists
#   8.5 kg/min, the text says that the water and sediment discharges of run 5 (110 L/s,
#   35 kg/min, 2.7 m wide) were scaled down to keep the values per unit width, which gives
#   12.2 L/s (as in Table 1) and 3.89 kg/min. The surveyed deposit at 27 h holds about
#   1.6 m^3 (3 400 kg at 2 140 kg/m^3) and, by the paper, all the gravel fed: 2 950 kg of
#   gravel with 3.89 kg/min, 6 450 kg with 8.5 kg/min. The model with the published
#   closures lays down 4.9 m^2 of deposit (long section, x = 13 to 46 m) with 3.89 kg/min
#   and 12.2 m^2 with 8.5 kg/min; the survey shows 5.3 m^2. The scaled value is used.
# - water level at the downstream end: tailgate level 0.45 m above the channel floor
#   (Table 1) less the 0.33 m step, 0.12 m; the plotted final water surface at x = 46 m
#   is at 0.12 m.
# - feed size distribution: bar heights of Fig. 2(c) read from the page image (the bars
#   sum to 100.3 %; their geometric mean size is 2.1 mm, the paper states 2.0 mm).
# - bed profiles and final water surface: markers and line of Fig. 5(c) read from the
#   page image (reading error about 0.005 m; where markers of several times overlap,
#   downstream of x = 29 m, about 0.015 m).
# - grain sizes: markers of Figs. 6(d) and 7(d) read from the page image (reading error
#   about 0.05 psi). They are percentiles of the gravel only (coarser than 2 mm).
RUN6 = {
    "hours": 27.0,
    "width_m": 0.3,
    "floor_slope": 0.01,
    "floor_end_m": 49.5,            # x where the sloping floor reaches the top of the step
    "discharge_m3s": 0.0122,
    "tailwater_m": 0.12,            # above the top of the step
    "feed_kg_per_min": 35.0 / 9.0,  # run 5 scaled to the width (see above)
    "feed_table_kg_per_min": 8.5,   # Table 1 of the paper
    "depth_m": 0.059,               # mean of final water surface minus final bed, x = 13 to 29 m
}
# Percentage of each class of ``SIEVE_MM`` in the feed of runs 5 and 6.
FEED_RUN6_PERCENT = np.array([1.52, 2.58, 2.71, 8.47, 6.01, 4.33, 5.12, 5.19, 5.19, 3.46, 2.34, 3.09, 3.19, 6.28, 9.25,
                              11.83, 9.67, 6.03, 2.62, 1.43])
# Bed profiles (Fig. 5c): elevation (m) at the survey sections (m) and times (h). Only the
# sections where the markers of a time can be told apart are listed.
RUN6_STATIONS = 13.0 + 1.75 * np.arange(20)
RUN6_PROFILES = {
    2.0: [0.484, 0.406, 0.368, 0.340, 0.302, 0.284, 0.275, 0.248, 0.208, 0.200],
    8.0: [0.728, 0.598, 0.483, 0.428, 0.358, 0.318, 0.283, 0.255, 0.208, 0.200],
    16.0: [0.948, 0.758, 0.654, 0.536, 0.466, 0.395, 0.330, 0.270, 0.226, 0.200],
    22.0: [0.986, 0.830, 0.758, 0.621, 0.545, 0.469, 0.410, 0.334, 0.289, 0.264, 0.203, 0.196],
    27.0: [1.028, 0.886, 0.770, 0.710, 0.601, 0.515, 0.444, 0.386, 0.338, 0.298, 0.236, 0.223, 0.198, 0.160, 0.134,
           0.106, 0.112, 0.090, 0.068, 0.062],
}
# Final water surface (m) at the survey sections (nan where the line is hidden by markers).
RUN6_WATER_SURFACE = np.array([1.078, 0.950, 0.834, np.nan, 0.650, 0.576, 0.500, 0.456, 0.392, 0.358, 0.293, 0.272, 0.246,
                               0.224, 0.195, 0.178, 0.172, 0.154, 0.138, 0.121])
# Grain sizes of the gravel (psi) at the sampling sections (m): the deposit laid down
# between hours 8 and 18 (Fig. 7d) and the surface layer at the end of the run (Fig. 6d).
RUN6_SIZE_X = np.array([14.0, 17.0, 20.0, 23.0, 26.0, 30.0, 36.0, 42.0, 48.0, 54.0])
RUN6_DEPOSIT_HOURS = (8.0, 18.0)
RUN6_DEPOSIT_GRAVEL_PSI = {
    90: np.array([5.81, 5.18, 4.96, 4.83, 4.80, 4.53, 3.97, 3.72, 3.38, 3.29]),
    84: np.array([5.63, 4.95, 4.84, 4.69, 4.66, 4.31, 3.77, 3.46, 3.17, 2.95]),
    50: np.array([4.30, 3.98, 3.81, 3.64, 3.58, 3.15, 2.72, 2.47, 2.20, 1.97]),
    30: np.array([3.15, 2.94, 2.87, 2.83, 2.78, 2.47, 2.18, 2.04, 1.79, 1.57]),
    16: np.array([2.35, 2.27, 2.23, 2.20, 2.17, 2.00, 1.74, 1.63, 1.46, 1.30]),
    5: np.array([1.61, 1.58, 1.56, 1.54, 1.52, 1.40, 1.27, 1.21, 1.15, 1.10]),
}
RUN6_SURFACE_GRAVEL_PSI = {
    90: np.array([6.01, 5.64, 5.32, 5.10, 4.99, 4.88, 4.50, 4.29, 4.12, 3.30]),
    84: np.array([5.89, 5.45, 5.16, 4.96, 4.91, 4.76, 4.38, 4.00, 3.89, 2.96]),
    50: np.array([4.87, 4.62, 4.41, 4.24, 4.18, 3.91, 3.53, 3.04, 2.87, 1.96]),
    30: np.array([4.05, 3.78, 3.36, 3.31, 3.33, 3.14, 2.88, 2.46, 2.36, 1.57]),
    16: np.array([2.90, 2.81, 2.53, 2.54, 2.56, 2.45, 2.28, 2.02, 1.94, 1.30]),
    5: np.array([1.93, 1.89, 1.73, 1.74, 1.75, 1.69, 1.63, 1.43, 1.39, 1.10]),
}


def run6_floor(x):
    """Elevation of the sloping floor of run 6 at x (m)."""
    return RUN6["floor_slope"] * (RUN6["floor_end_m"] - np.asarray(x))


def run6_thickness_profiles() -> dict:
    """Thickness of the deposit of run 6: {hours: (stations, thickness)} (m)."""
    out = {}
    for hours, values in RUN6_PROFILES.items():
        x = RUN6_STATIONS[: len(values)]
        out[hours] = (x, np.maximum(np.asarray(values) - run6_floor(x), 0.0))
    return out
