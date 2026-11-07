"""Figure 9: calibration of a vegetation drag closure against 3D RANS of two patches in tandem.

Application
-----------
Two emergent vegetation patches (0.5 m x 0.5 m, 15 x 15 rigid stems of diameter
d = 0.01 m at spacing dS = 0.0333 m) stand in tandem on the centreline of a 6 m x 1.5 m
flume, with a gap of 0.5 m. The flow depth is H = 0.3 m (rigid lid) and the approach
velocity U_0 = 0.5 m/s. The reference is a steady 3D RANS solution (k-omega SST,
OpenFOAM, 5.5 M cells), which gives the drag force F_x (N) on every stem and the
depth-averaged velocity and water level on its plan-view columns.

The 2D model (hydrograd, steady Newton solve, second order, 180 x 45 cells of 0.0333 m)
adds the vegetation drag per unit plan area

    f = 1/2 C_D a h |u| u   (m^2/s^2, per unit density),

with a = d / dS^2 = 9 1/m the frontal area per unit volume, h (m) the depth and u (m/s)
the depth-averaged velocity of the 2D model. One C_D is used per patch ("zone" closure).
The bed Manning's n comes from the CFD bed shear; the side walls have a log-law wall
function (smooth); there is no eddy viscosity.

Two calibrations (L-BFGS, gradients by the implicit function theorem through the steady
solve), from ``cfd/pilot_tandem/post/train_cd_multidensity.py``:

- force-calibrated: misfit of the two CFD patch forces F_1, F_2 (N);
- field-calibrated: misfit of the depth-averaged velocity (U, V) and water level eta
  outside the patches.

The 2D patch and row forces are rho sum(f A) over the cells of a patch or of one stem row
(rho = 1000 kg/m^3, A the cell area, m^2).

Panels: (a) CFD depth-averaged streamwise velocity U on the 2D cells; (b, c) U of the two
calibrated 2D models minus the CFD; (d) drag per stem row, CFD against both models;
(e) patch-force errors; (f) field misfits outside the patches.

Data (``paper/data/vegetation/``, a few hundred kB, made once by ``--extract`` from the
reduced CFD data in ``cfd/pilot_tandem``): CFD fields averaged onto the 2D cells, CFD
stem, row and patch forces, geometry, and the fitted C_D. The 2D steady solves are re-run
from these (about 1 min each on a CPU) and cached in ``swe_solutions.npz``.

Usage
-----
    python paper/fig09_vegetation.py [--extract] [--resolve]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from figstyle import (
    DIFFERENCE_CMAP,
    FULL_WIDTH,
    MAGNITUDE_CMAP,
    MODEL,
    MODEL_2,
    REFERENCE,
    apply_style,
    colorbar,
    panel_label,
    save,
)
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

PAPER = Path(__file__).resolve().parent
DATA = PAPER / "data" / "vegetation"
RESULTS = PAPER / "results" / "vegetation.json"
PILOT = PAPER.parent / "cfd" / "pilot_tandem"

# The two calibrations shown, with the fit files they come from.
FITS = {
    "force": "cd_multidensity_none_zone_dense-zone-patch.json",
    "field": "cd_multidensity_none_zone_dense-zone-fields.json",
    "field_interior": "cd_multidensity_none_zone_dense-zone-fieldsint.json",
}
LABELS = {"force": "fit to patch forces", "field": "fit to flow fields"}
SHORT_LABELS = {"force": "forces", "field": "fields"}
STYLES = {"force": MODEL, "field": MODEL_2}


# ----------------------------------------------------------------------------
# One-off extraction of the small data set from the CFD post-processing
# ----------------------------------------------------------------------------
def extract() -> None:
    """Write the CFD data on the 2D cells, the forces and the fitted C_D to ``DATA``.

    Reads ``cfd/pilot_tandem/rans_dense_coarse/post/{depth_averaged.npz, summary.json,
    stems.csv}`` and the fit JSONs in ``cfd/pilot_tandem/post``.
    """
    sys.path.insert(0, str(PILOT / "post"))
    sys.path.insert(0, str(PILOT / "mesh"))
    from compare_swe import bed_manning_n, cfd_on_swe_cells, swe_grid
    from make_mesh import Geometry

    DATA.mkdir(parents=True, exist_ok=True)
    case_dir = PILOT / "rans_dense_coarse"
    geo = Geometry(dS=0.0333333333333, n_s=15)
    nx, ny = swe_grid(geo)

    # CFD depth-averaged fields, area-averaged onto the 2D cells.
    cfd = cfd_on_swe_cells(case_dir, geo)
    np.savez_compressed(DATA / "cfd_on_swe_cells.npz", nx=nx, ny=ny, **cfd)

    # Stem forces and local C_D (one row per stem).
    columns = ("patch", "row", "col", "x", "y", "Fx", "u_local", "CD_local")
    with open(case_dir / "post" / "stems.csv") as handle:
        stems = list(csv.DictReader(handle))
    with open(DATA / "cfd_stems.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stem in stems:
            writer.writerow({key: stem[key] for key in columns})

    # Geometry, bed roughness (walls carry their own friction), patch and row forces.
    summary = json.loads((case_dir / "post" / "summary.json").read_text())
    case = {
        "source": "rigid-lid RANS (k-omega SST), rans_dense_coarse",
        "L_x_m": geo.L_x,
        "W_m": geo.W,
        "H_m": geo.H,
        "d_m": geo.d,
        "dS_m": geo.dS,
        "n_s": geo.n_s,
        "patch_x0_m": list(geo.patch_x0),
        "patch_y0_m": geo.patch_y0,
        "U0_m_s": 0.5,
        "n_bed": bed_manning_n(case_dir, geo, include_sidewalls=False),
        "patch_force_N": [summary["patch_force_N"]["1"], summary["patch_force_N"]["2"]],
        "row_force_N": [
            summary["force_by_streamwise_row_N"]["1"],
            summary["force_by_streamwise_row_N"]["2"],
        ],
        "CD_local_mean_per_patch": [
            summary["mean_CD_per_patch_local"]["1"],
            summary["mean_CD_per_patch_local"]["2"],
        ],
        "head_loss_mm": summary["eta_inlet_minus_outlet_mm"],
    }
    (DATA / "cfd_case.json").write_text(json.dumps(case, indent=1))

    # Fitted C_D per material [background, open channel, patch 1, patch 2] and options.
    fits = {}
    for name, file_name in FITS.items():
        fit = json.loads((PILOT / "post" / file_name).read_text())
        metrics = fit["cases"]["dense"]
        fits[name] = {
            "source": file_name,
            "loss_terms": fit["loss_terms"],
            "walls": fit["walls"],
            "eddy": fit["eddy"],
            "CD_zone": fit["CD_zone"],
            "patch_force_error_pct": metrics["patch_force_error_pct"],
            "rms_U_outside_m_s": metrics["rms_U_outside_m_s"],
            "rms_eta_outside_mm": metrics["rms_eta_outside_mm"],
        }
    (DATA / "fits.json").write_text(json.dumps(fits, indent=1))
    print(f"wrote the data set to {DATA}")


# ----------------------------------------------------------------------------
# 2D model
# ----------------------------------------------------------------------------
def patch_boxes(case: dict) -> list[tuple[float, float, float, float]]:
    """Patch rectangles (x0, y0, x1, y1) in m."""
    side = case["n_s"] * case["dS_m"]
    y0 = case["patch_y0_m"]
    return [(x0, y0, x0 + side, y0 + side) for x0 in case["patch_x0_m"]]


def build_model(case: dict, nx: int, ny: int, work_dir: Path):
    """Write and load the hydrograd flume case (walls on, no eddy viscosity).

    Args:
        case: geometry and roughness from ``cfd_case.json``.
        nx, ny: number of 2D cells along and across the flume.
        work_dir: folder for the SRH-2D style case files.

    Returns:
        The hydrograd case; materials 1 = open channel, 2 / 3 = patch 1 / 2.
    """
    from hydrograd.apps.setup import setup_case
    from hydrograd.io.control import read_control
    from hydrograd.io.synthetic import write_channel_case

    boxes = patch_boxes(case)

    def zone(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Material ID of each cell centroid (x, y in m)."""
        material = np.ones_like(x, dtype=int)
        for zone_id, (x0, y0, x1, y1) in zip((2, 3), boxes, strict=True):
            inside = (x > x0) & (x < x1) & (y > y0) & (y < y1)
            material[inside] = zone_id
        return material

    # Flat flume: inlet discharge Q = U0 W H (m^3/s), exit water level H (m).
    write_channel_case(
        work_dir,
        nx,
        ny,
        length=case["L_x_m"],
        width=case["W_m"],
        slope=0.0,
        bump=0.0,
        Q=case["U0_m_s"] * case["W_m"] * case["H_m"],
        wse_out=case["H_m"],
        n_manning=case["n_bed"],
        solver="SteadyNewton()",
        order=2,
        zone_fn=zone,
    )

    # Vegetation zones (a = d / dS^2) and smooth log-law side walls.
    frontal_area = case["d_m"] / case["dS_m"] ** 2
    control_path = work_dir / "run_control.json"
    control = json.loads(control_path.read_text())
    control["vegetation"] = {
        "a": {"2": frontal_area, "3": frontal_area},
        "CD": {"2": 1.0, "3": 1.0},
    }
    control["wall_friction"] = {"model": "loglaw", "ks": 0.0}
    control_path.write_text(json.dumps(control, indent=1))
    return setup_case(read_control(control_path))


def region_weights(centroid: np.ndarray, case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cell weights of the two patches and of the stem rows (1 inside, 0 outside).

    Args:
        centroid: cell centroids (m), shape [n_cells, 2].
        case: geometry.

    Returns:
        (patch weights [2, n_cells], row weights [2 n_s, n_cells]).
    """
    x = centroid[:, 0]
    y = centroid[:, 1]
    patches = []
    rows = []
    for x0, y0, x1, y1 in patch_boxes(case):
        in_y = (y > y0) & (y < y1)
        patches.append(in_y & (x > x0) & (x < x1))
        for j in range(case["n_s"]):
            x_start = x0 + j * case["dS_m"]
            rows.append(in_y & (x > x_start) & (x < x_start + case["dS_m"]))
    return np.array(patches, dtype=float), np.array(rows, dtype=float)


def solve_fits(case: dict, fits: dict, nx: int, ny: int) -> dict:
    """Steady 2D solutions with the fitted C_D of each calibration.

    Args:
        case: geometry and roughness.
        fits: fitted C_D per calibration.
        nx, ny: grid size.

    Returns:
        Dict per calibration: ``U``, ``V``, ``eta`` (m/s, m; shape [n_cells]), ``patch_force``
        [2] and ``row_force`` [2 n_s] (N); plus ``centroid`` [n_cells, 2] (m).
    """
    from hydrograd.physics.vegetation import vegetation_force
    from hydrograd.solvers.final import final_state

    solutions = {}
    with tempfile.TemporaryDirectory() as work_dir:
        swe = build_model(case, nx, ny, Path(work_dir))
        model = swe.model
        centroid = np.asarray(swe.mesh.centroid)
        patch_w, row_w = region_weights(centroid, case)
        solutions["centroid"] = centroid

        for name in LABELS:
            # Steady solve with this calibration's C_D per material.
            theta = {"veg_CD": jnp.asarray(fits[name]["CD_zone"])}
            state, converged = final_state(model, theta, swe.U0, swe.ode)
            print(f"{name}: steady solve converged = {bool(converged)}", flush=True)

            # Depth-averaged velocity and water level relative to the exit level.
            depth = state[0] + model.hstill
            solutions[name] = {
                "U": np.asarray(state[1] / depth),
                "V": np.asarray(state[2] / depth),
                "eta": np.asarray(state[0] + model.wstill - model.wstill.max()),
                "patch_force": np.asarray(
                    vegetation_force(model, theta, state, jnp.asarray(patch_w))[:, 0]
                ),
                "row_force": np.asarray(
                    vegetation_force(model, theta, state, jnp.asarray(row_w))[:, 0]
                ),
            }
    return solutions


def outside_mask(centroid: np.ndarray, case: dict, dx: float) -> np.ndarray:
    """Cells outside both patches and their one-cell margin (the field-misfit region)."""
    near = np.zeros(len(centroid), dtype=bool)
    for x0, y0, x1, y1 in patch_boxes(case):
        in_x = (centroid[:, 0] > x0 - dx) & (centroid[:, 0] < x1 + dx)
        in_y = (centroid[:, 1] > y0 - dx) & (centroid[:, 1] < y1 + dx)
        near |= in_x & in_y
    return ~near


def load_solutions(case: dict, fits: dict, nx: int, ny: int, resolve: bool) -> dict:
    """Cached 2D solutions, or new solves (``resolve`` or no cache)."""
    cache = DATA / "swe_solutions.npz"
    if cache.exists() and not resolve:
        flat = np.load(cache)
        solutions = {"centroid": flat["centroid"]}
        for name in LABELS:
            keys = ("U", "V", "eta", "patch_force", "row_force")
            solutions[name] = {key: flat[f"{name}_{key}"] for key in keys}
        return solutions

    solutions = solve_fits(case, fits, nx, ny)
    flat = {"centroid": solutions["centroid"]}
    for name in LABELS:
        for key, value in solutions[name].items():
            flat[f"{name}_{key}"] = value
    np.savez_compressed(cache, **flat)
    return solutions


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def draw_map(ax, field: np.ndarray, case: dict, nx: int, ny: int, **style):
    """Plan-view map of a cell field with the patch outlines.

    Args:
        ax: the axes.
        field: values on the 2D cells, shape [nx * ny] (x outer, y inner).
        case: geometry.
        nx, ny: grid size.
        style: keyword arguments of ``pcolormesh`` (cmap, vmin, vmax).

    Returns:
        The QuadMesh.
    """
    x_edges = np.linspace(0.0, case["L_x_m"], nx + 1)
    y_edges = np.linspace(0.0, case["W_m"], ny + 1)
    image = ax.pcolormesh(x_edges, y_edges, field.reshape(nx, ny).T, rasterized=True, **style)
    for x0, y0, x1, y1 in patch_boxes(case):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k", lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlim(0.0, case["L_x_m"])
    ax.set_ylim(0.0, case["W_m"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r"$y$ (m)")
    return image


def make_figure(case: dict, fits: dict, cfd: dict, solutions: dict, nx: int, ny: int) -> dict:
    """Draw Figure 9 and return its numbers.

    Args:
        case: geometry and CFD forces.
        fits: fitted C_D.
        cfd: CFD fields on the 2D cells (U, V, eta).
        solutions: 2D solutions per calibration.
        nx, ny: grid size.

    Returns:
        The numbers written to ``paper/results/vegetation.json``.
    """
    apply_style()
    figure = plt.figure(figsize=(FULL_WIDTH, 6.7))
    outer = GridSpec(2, 1, figure=figure, height_ratios=[3.0, 1.45], hspace=0.24)
    map_grid = outer[0].subgridspec(3, 1, hspace=0.3)
    bottom_grid = outer[1].subgridspec(1, 4, wspace=0.6)

    # (a) CFD streamwise velocity.
    ax_a = figure.add_subplot(map_grid[0])
    image = draw_map(ax_a, cfd["U"], case, nx, ny, cmap=MAGNITUDE_CMAP, vmin=0.0, vmax=0.75)
    colorbar(ax_a, image, r"(m\,s$^{-1}$)")
    ax_a.text(0.08, 1.4, r"3D RANS, $U$", fontsize=9, va="top", color="w")
    ax_a.tick_params(labelbottom=False)
    panel_label(ax_a, "(a)", x=-0.085, y=0.98)

    # (b, c) 2D model minus CFD, same symmetric colour scale.
    limit = 0.15
    numbers = {"calibrations": {}}
    centroid = solutions["centroid"]
    dx = case["L_x_m"] / nx
    outside = outside_mask(centroid, case, dx)
    for row, name in enumerate(LABELS, start=1):
        ax = figure.add_subplot(map_grid[row], sharex=ax_a)
        difference = solutions[name]["U"] - cfd["U"]
        image = draw_map(
            ax, difference, case, nx, ny, cmap=DIFFERENCE_CMAP, vmin=-limit, vmax=limit
        )
        colorbar(ax, image, r"(m\,s$^{-1}$)")
        cd_1, cd_2 = fits[name]["CD_zone"][2:]
        ax.text(
            0.08,
            1.4,
            rf"{LABELS[name]}, $U_\mathrm{{2D}} - U_\mathrm{{CFD}}$:"
            rf" $C_D = {cd_1:.2f}$ / ${cd_2:.2f}$",
            fontsize=9,
            va="top",
        )
        panel_label(ax, f"({'abc'[row]})", x=-0.085, y=0.98)
        if row == 2:
            ax.set_xlabel(r"$x$ (m)", labelpad=1)
        else:
            ax.tick_params(labelbottom=False)

        # Misfits and forces of this calibration.
        model_eta = solutions[name]["eta"]
        cfd_force = np.asarray(case["patch_force_N"])
        model_force = solutions[name]["patch_force"]
        cfd_rows = np.ravel(case["row_force_N"])
        numbers["calibrations"][name] = {
            "loss_terms": fits[name]["loss_terms"],
            "CD_patch1": cd_1,
            "CD_patch2": cd_2,
            "patch_force_N": model_force.tolist(),
            "patch_force_error_pct": (100.0 * (model_force / cfd_force - 1.0)).tolist(),
            "row_force_rms_error_N": float(
                np.sqrt(np.mean((solutions[name]["row_force"] - cfd_rows) ** 2))
            ),
            "rms_U_outside_m_s": float(np.sqrt(np.mean(difference[outside] ** 2))),
            "rms_eta_outside_mm": float(
                1000.0 * np.sqrt(np.mean((model_eta - cfd["eta"])[outside] ** 2))
            ),
        }

    # (d) Drag per stem row against the row-centre position.
    ax_d = figure.add_subplot(bottom_grid[0, :2])
    row_x = []
    for x0 in case["patch_x0_m"]:
        row_x.extend(x0 + (np.arange(case["n_s"]) + 0.5) * case["dS_m"])
    row_x = np.array(row_x)
    n_s = case["n_s"]
    for patch in range(2):
        rows = slice(patch * n_s, (patch + 1) * n_s)
        first = patch == 0
        ax_d.plot(
            row_x[rows], np.ravel(case["row_force_N"])[rows], marker="s", ms=2.5,
            label="3D RANS" if first else None, **REFERENCE,
        )
        for name, label in LABELS.items():
            ax_d.plot(
                row_x[rows],
                solutions[name]["row_force"][rows],
                label=label if first else None,
                **STYLES[name],
            )
    ax_d.set_xlabel(r"$x$ of stem row (m)")
    ax_d.set_ylabel("row drag (N)")
    ax_d.set_ylim(0.0, 5.0)
    ax_d.legend(loc="upper right", handlelength=2.2)
    panel_label(ax_d, "(d)", x=-0.2)

    # (e) Patch-force errors; (f) field misfits outside the patches.
    ax_e = figure.add_subplot(bottom_grid[0, 2])
    ax_f = figure.add_subplot(bottom_grid[0, 3])
    width = 0.38
    positions = np.arange(2)
    for k, name in enumerate(LABELS):
        values = numbers["calibrations"][name]
        color = STYLES[name]["color"]
        offset = (k - 0.5) * width
        errors = values["patch_force_error_pct"]
        ax_e.bar(positions + offset, errors, width, color=color, label=SHORT_LABELS[name])
        misfits = [100.0 * values["rms_U_outside_m_s"], values["rms_eta_outside_mm"]]
        ax_f.bar(positions + offset, misfits, width, color=color)

        # Value labels (the force-calibrated errors are zero, so the bars are invisible).
        for position, error in zip(positions + offset, errors, strict=True):
            above = error >= 0.0
            rounded = round(error)
            text = "$0$" if rounded == 0 else f"${rounded:+d}$"
            ax_e.text(
                position, error + (3.0 if above else -3.0), text,
                ha="center", va="bottom" if above else "top", fontsize=7,
            )
    ax_e.axhline(0.0, color="k", lw=0.6)
    ax_e.set_xticks(positions, [r"$F_1$", r"$F_2$"])
    ax_e.set_ylabel(r"patch-force error (\%)")
    ax_e.set_ylim(-90.0, 40.0)
    ax_e.legend(loc="lower left", fontsize=7, handlelength=1.0, borderaxespad=0.3)
    ax_f.set_xticks(positions, ["$U$\n(cm\\,s$^{-1}$)", "$\\eta$\n(mm)"])
    ax_f.set_ylabel("RMS misfit")
    ax_f.set_ylim(0.0, 4.6)
    panel_label(ax_e, "(e)", x=-0.62)
    panel_label(ax_f, "(f)", x=-0.42)

    save(figure, "fig09_vegetation")
    return numbers


def main() -> None:
    """Extract the data (optional), solve or load the 2D runs, draw, write the numbers."""
    parser = argparse.ArgumentParser(description="Figure 9: vegetation drag calibration")
    parser.add_argument("--extract", action="store_true", help="rebuild the data set from cfd/")
    parser.add_argument("--resolve", action="store_true", help="re-run the 2D steady solves")
    args = parser.parse_args()
    if args.extract:
        extract()

    # Small data set.
    case = json.loads((DATA / "cfd_case.json").read_text())
    fits = json.loads((DATA / "fits.json").read_text())
    stored = np.load(DATA / "cfd_on_swe_cells.npz")
    nx = int(stored["nx"])
    ny = int(stored["ny"])
    cfd = {key: stored[key] for key in ("U", "V", "eta")}

    # 2D solutions and the figure.
    solutions = load_solutions(case, fits, nx, ny, args.resolve)
    numbers = make_figure(case, fits, cfd, solutions, nx, ny)

    # Numbers for the text.
    numbers["case"] = {
        "description": "two 0.5 m patches in tandem, 15 x 15 stems, d = 0.01 m, dS = 0.0333 m",
        "a_per_m": case["d_m"] / case["dS_m"] ** 2,
        "cfd_patch_force_N": case["patch_force_N"],
        "cfd_sheltering_F2_over_F1": case["patch_force_N"][1] / case["patch_force_N"][0],
        "cfd_first_last_row_force_patch1_N": [case["row_force_N"][0][0], case["row_force_N"][0][-1]],
        "cfd_CD_local_mean_per_patch": case["CD_local_mean_per_patch"],
        "cfd_head_loss_mm": case["head_loss_mm"],
        "n_bed": case["n_bed"],
        "swe_grid": [nx, ny],
    }
    numbers["field_plus_interior_fit"] = fits["field_interior"]
    numbers["recorded_fit_metrics"] = {name: fits[name] for name in LABELS}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers["calibrations"], indent=1))


if __name__ == "__main__":
    main()
