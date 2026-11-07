"""Figure 3 of the Hydrograd v2.0 paper: verification of the forward solver.

Usage (from any folder, with the Hydrograd venv active)::

    JAX_PLATFORMS=cpu python paper/fig03_verification.py              # run all cases + plot
    JAX_PLATFORMS=cpu python paper/fig03_verification.py --plot-only  # plot from paper/data

The script runs five verification cases with the default scheme (hydrostatic reconstruction
+ HLLC, ``hr_hllc``), at first order (piecewise constant) and second order (MUSCL with the
Venkatakrishnan limiter). It writes the profiles to ``paper/data/fig03_verification.npz``, the
error numbers to ``paper/results/verification.json`` and the figure to
``paper/figures/fig03_verification.pdf``.

Cases (all SI units; g = 9.81 m/s^2)
------------------------------------
(a) Stoker dam break (wet bed): frictionless flat channel 1000 m x 10 m (1000 x 2 cells), walls
    at both ends, h = 1 m for x < 500 m and h = 0.1 m downstream; SSPRK3 with CFL-limited steps
    (Courant number 0.5 at order 1, 0.25 at order 2) to t = 60 s. Exact solution: the Riemann
    solution (rarefaction + shock) of ``hydrograd.validation.dambreak.dam_break``.
(b) Ritter dam break (dry bed): as (a) with h = 0 downstream; the exact front moves at
    2 sqrt(g h_l) = 6.26 m/s.
(c) Lake at rest over an emerged island: closed 1000 m x 100 m basin (200 x 20 cells) with a
    Gaussian island of height 1 m and a wavy bed, still water level 0.5 m (so the island top is
    dry). The residual dU/dt must vanish (well balancing across wet/dry fronts); the state is
    also integrated for 600 s with SSPRK3.
(d) Transcritical flow over a bump with a hydraulic jump (Hydrograd.jl example
    ``oneD_channel_with_bump``; SWASHES/FullSWOF case): 25 m channel, q = 0.18 m^2/s, outlet
    WSE 0.33 m, z_b = 0.2 - 0.05 (x - 10)^2 for 8 < x < 12 m, Manning's n = 0.02 s/m^(1/3)
    as in the example (the analytical solution is frictionless). Steady state by
    pseudo-transient Newton; WSE compared with FullSWOF's analytical solution.
(f) Grid convergence on smooth subcritical flow over the same bump (q = 4.42 m^2/s, outlet depth
    2 m, near frictionless), exact Bernoulli depth; ``benchmarks/validation/convergence_bump.py``.
    The L1 depth error e = mean |h - h_exact| (m) is fitted as e ~ dx^p, with dx the cell size.
(e) Close-up of the hydraulic jump of (d).

Error norms: L1 = mean over cells of |h - h_exact| (m); RMS and max of the WSE error (m);
relative mass change (V_end - V_0) / V_0 with V = sum h A (m^3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Paths: this folder (paper/), the repository root and the case input files.
PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
BUMP_EXAMPLE = PAPER_DIR / "cases" / "channel_bump_transcritical"
DATA_FILE = PAPER_DIR / "data" / "fig03_verification.npz"
RESULT_FILE = PAPER_DIR / "results" / "verification.json"

# Dam-break settings: end time (s), upstream depth (m), dam position (m).
T_DAM = 60.0
H_LEFT = 1.0
X_DAM = 500.0

# Lake at rest: still-water level (m) and simulated time (s).
WSE_LAKE = 0.5
T_LAKE = 600.0

# Mesh sizes (cells along the channel) of the convergence study.
CONVERGENCE_NX = [50, 100, 200, 400, 800]

# Depth below which a cell counts as dry (m).
H_DRY = 1.0e-3


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def hardware_info() -> dict:
    """Describe the machine and software used for the runs.

    Returns:
        Dict with CPU model, number of usable cores, JAX version and devices, host name and
        the Slurm job ID (if any).
    """
    import jax

    # CPU model from /proc/cpuinfo (Linux); platform.processor() is often empty.
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break

    return {
        "cpu": cpu,
        "usable_cores": len(os.sched_getaffinity(0)),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "precision": "float64",
    }


def setup_synthetic(case_dir: Path, **kwargs):
    """Write a synthetic channel case and set it up.

    Args:
        case_dir: folder for the SRH-2D files and ``run_control.json`` (created).
        **kwargs: arguments of ``hydrograd.io.synthetic.write_channel_case``.

    Returns:
        The set-up case (``hydrograd.apps.setup.Case``).
    """
    from hydrograd.apps.setup import setup_case
    from hydrograd.io.control import read_control
    from hydrograd.io.synthetic import write_channel_case

    write_channel_case(case_dir, **kwargs)
    return setup_case(read_control(case_dir / "run_control.json"))


# ----------------------------------------------------------------------
# (a), (b): dam breaks
# ----------------------------------------------------------------------
def run_dam_break(h_right: float, order: int, work_dir: Path) -> tuple[dict, dict]:
    """Dam break on a flat frictionless bed, compared with the exact Riemann solution.

    Args:
        h_right: initial downstream depth (m); 0.1 for Stoker, 0 for Ritter.
        order: spatial order (1 or 2).
        work_dir: scratch folder for the case files.

    Returns:
        (numbers, profile): error numbers (m, m/s, -) and the profile along the channel,
        {"x": [n] (m), "h": [n] (m)} for the cells of the first row (y < 5 m).
    """
    import jax

    from hydrograd.solvers.ode import solve_swe
    from hydrograd.validation.dambreak import dam_break

    def initial_wse(x, y):
        """Initial WSE (m): H_LEFT upstream of the dam, h_right downstream."""
        return np.where(x < X_DAM, H_LEFT, h_right)

    # 1. Case: 1000 x 2 cells, walls at both ends, SSPRK3 with CFL steps.
    case = setup_synthetic(
        work_dir / f"dam_{h_right}_{order}",
        nx=1000,
        ny=2,
        length=1000.0,
        width=10.0,
        slope=0.0,
        bump=0.0,
        n_manning=0.0,
        inlet="WALL",
        outlet="WALL",
        wse_out=h_right,
        wstill=0.5,
        t_end=T_DAM,
        dt=0.05,
        n_save=1,
        solver="SSPRK3()",
        order=order,
        ic_wse=initial_wse,
    )
    model = case.model

    # 2. Run to T_DAM (the first call includes compilation; time a second call).
    def simulate(U0):
        """SSPRK3 solution from U0 [3, n_cells]."""
        return solve_swe(model, {}, U0, case.ode)

    simulate_jit = jax.jit(simulate)
    solution = simulate_jit(case.U0)
    jax.block_until_ready(solution.ys)
    start = time.perf_counter()
    solution = simulate_jit(case.U0)
    jax.block_until_ready(solution.ys)
    run_seconds = time.perf_counter() - start

    # 3. Final depth (m) and velocity (m/s; zero in dry cells).
    U = np.asarray(solution.ys[-1])
    h = U[0] + np.asarray(model.hstill)
    wet = h > H_DRY
    u = np.where(wet, U[1] / np.maximum(h, H_DRY), 0.0)

    # 4. Exact solution at the cell centroids.
    x = case.mesh.centroid[:, 0]
    y = case.mesh.centroid[:, 1]
    h_exact, u_exact = dam_break(x, T_DAM, H_LEFT, h_right, x0=X_DAM)
    depth_error = h - h_exact

    # 5. Mass (m^3) at the start and the end.
    area = np.asarray(model.area)
    h_initial = np.asarray(case.U0[0]) + np.asarray(model.hstill)
    volume_initial = np.sum(h_initial * area)
    volume_final = np.sum(h * area)

    numbers = {
        "order": order,
        "n_cells": int(model.n_cells),
        "time_steps": int(solution.stats["num_steps"]),
        "run_time_s": run_seconds,
        "h_L1_error_m": float(np.mean(np.abs(depth_error))),
        "h_max_error_m": float(np.max(np.abs(depth_error))),
        "u_L1_error_m_per_s": float(np.mean(np.abs(u - u_exact))),
        "relative_mass_change": float((volume_final - volume_initial) / volume_initial),
        "min_depth_m": float(h.min()),
    }

    # 6. Dry bed: position of the 1 mm wet front against the exact one.
    if h_right == 0.0:
        exact_speed = 2.0 * np.sqrt(9.81 * H_LEFT) - 3.0 * np.sqrt(9.81 * H_DRY)
        numbers["front_1mm_m"] = float(x[wet].max())
        numbers["front_1mm_exact_m"] = float(X_DAM + exact_speed * T_DAM)

    # 7. Profile of the first row of cells, sorted by x.
    row = y < 5.0
    order_x = np.argsort(x[row])
    profile = {"x": x[row][order_x], "h": h[row][order_x]}
    return numbers, profile


# ----------------------------------------------------------------------
# (c): lake at rest over an emerged island
# ----------------------------------------------------------------------
def island_bed(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Bed of the lake-at-rest basin: Gaussian island plus a wavy bed.

    Args:
        X, Y: coordinates (m), any (equal) shape.

    Returns:
        Bed elevation z_b (m): 1 m Gaussian island (radius 100 m) at (500, 50) m plus a
        0.1 m sinusoidal undulation; the island top rises above the 0.5 m water level.
    """
    island = np.exp(-((X - 500.0) ** 2 + (Y - 50.0) ** 2) / 100.0 ** 2)
    undulation = 0.1 * np.sin(X / 37.0) * np.cos(Y / 13.0)
    return island + undulation


def run_lake_at_rest(order: int, work_dir: Path) -> tuple[dict, dict]:
    """Lake at rest over an emerged island: residual at t = 0 and drift after T_LAKE.

    Args:
        order: spatial order (1 or 2).
        work_dir: scratch folder for the case files.

    Returns:
        (numbers, profile): max |dU/dt| at t = 0 (m/s, m^2/s^2), max |WSE - 0.5| (m) and
        max |u| (m/s) over the wet cells after T_LAKE; profile along the row of cells
        nearest the island centre line, {"x", "zb", "wse"} (m), WSE = NaN in dry cells.
    """
    import jax

    from hydrograd.physics.rhs import rhs
    from hydrograd.solvers.ode import solve_swe

    def flat_wse(x, y):
        """Flat initial WSE equal to WSE_LAKE (m)."""
        return WSE_LAKE + 0.0 * x

    # 1. Closed basin, 200 x 20 cells, Manning's n = 0.03 s/m^(1/3).
    case = setup_synthetic(
        work_dir / f"lake_{order}",
        nx=200,
        ny=20,
        length=1000.0,
        width=100.0,
        n_manning=0.03,
        inlet="WALL",
        outlet="WALL",
        wse_out=WSE_LAKE,
        wstill=WSE_LAKE,
        t_end=T_LAKE,
        dt=0.05,
        n_save=1,
        solver="SSPRK3()",
        order=order,
        zb_fn=island_bed,
        ic_wse=flat_wse,
    )
    model = case.model
    h_initial = np.asarray(case.U0[0] + model.hstill)

    # 2. Residual at the initial state (exactly zero for a well-balanced scheme).
    residual = np.asarray(jax.jit(lambda U: rhs(model, {}, U))(case.U0))

    # 3. Integrate for T_LAKE seconds.
    solution = jax.jit(lambda U0: solve_swe(model, {}, U0, case.ode))(case.U0)
    U = np.asarray(solution.ys[-1])
    h = U[0] + np.asarray(model.hstill)
    zb = np.asarray(model.zb)
    wet = h > H_DRY
    speed = np.hypot(U[1], U[2]) / np.maximum(h, H_DRY)

    numbers = {
        "order": order,
        "n_cells": int(model.n_cells),
        "n_dry_cells": int((h_initial <= 0.0).sum()),
        "max_abs_dUdt_t0": float(np.abs(residual).max()),
        "time_s": T_LAKE,
        "time_steps": int(solution.stats["num_steps"]),
        "max_abs_wse_change_m": float(np.abs((h + zb) - WSE_LAKE)[wet].max()),
        "max_speed_m_per_s": float(speed[wet].max()),
        "max_depth_in_initially_dry_cells_m": float(h[h_initial <= 0.0].max()),
    }

    # 4. Profile along the row of cells closest to y = 50 m (through the island top).
    x = case.mesh.centroid[:, 0]
    y = case.mesh.centroid[:, 1]
    row_y = y[np.argmin(np.abs(y - 50.0))]
    row = np.isclose(y, row_y)
    order_x = np.argsort(x[row])
    wse = np.where(wet, h + zb, np.nan)
    profile = {"x": x[row][order_x], "zb": zb[row][order_x], "wse": wse[row][order_x]}
    return numbers, profile


# ----------------------------------------------------------------------
# (d), (e): transcritical bump (Hydrograd.jl example) vs FullSWOF analytical
# ----------------------------------------------------------------------
def run_bump(work_dir: Path, manning_n: float | None) -> tuple[dict, dict]:
    """Steady transcritical bump flow at orders 1 and 2 vs the analytical WSE.

    The case is copied into ``work_dir`` first (never run inside ``paper/cases``).
    The analytical (SWASHES/FullSWOF) solution is frictionless, whereas the example uses
    Manning's n = 0.02 s/m^(1/3); ``manning_n = 1e-4`` (n = 0 gives NaN in the inlet-Q
    conveyance split L h^(5/3) / n) removes the friction so that
    the comparison measures the discretization error only.

    Args:
        work_dir: scratch folder.
        manning_n: Manning's n (s/m^(1/3)) written to every material of the copied
            ``.srhhydro`` file, or None to keep the example's value (0.02).

    Returns:
        (numbers, profiles): per order the Newton iterations, residual, solve time and
        the RMS / max WSE errors (m); profiles {"x", "zb", "wse_o1", "wse_o2"} at the cell
        centroids (m) plus the analytical {"x_exact", "wse_exact", "zb_exact"} (m).
    """
    import jax

    from hydrograd.apps.setup import setup_case
    from hydrograd.io.control import read_control
    from hydrograd.solvers.final import newton_settings
    from hydrograd.solvers.steady import newton_solve

    # 1. Copy the example and set it up.
    case_dir = work_dir / f"bump_{manning_n}"
    shutil.copytree(BUMP_EXAMPLE, case_dir, ignore=shutil.ignore_patterns("*.gif", "*.h5"))

    # 2. Optionally overwrite Manning's n of every material ("ManningsN <id> <n>" lines).
    if manning_n is not None:
        hydro_file = case_dir / "oneD_channel_with_bump_refined.srhhydro"
        lines = hydro_file.read_text().splitlines()
        for k, line in enumerate(lines):
            fields = line.split()
            if fields and fields[0] == "ManningsN":
                lines[k] = f"ManningsN {fields[1]} {manning_n}"
        hydro_file.write_text("\n".join(lines) + "\n")

    case = setup_case(read_control(case_dir / "run_control.json"))
    settings = newton_settings(case.ode)

    # 3. Analytical solution (FullSWOF file: x, h, zb, wse in m).
    analytical = np.loadtxt(case_dir / "analytical_solution_from_FullSWOF.dat", skiprows=1)
    x_exact = analytical[:, 0]
    wse_exact = analytical[:, 3]

    # 4. Cell centroids, bed and analytical WSE interpolated to the centroids.
    x = case.mesh.centroid[:, 0]
    zb = np.asarray(case.model.zb)
    wse_exact_cells = np.interp(x, x_exact, wse_exact)
    order_x = np.argsort(x)

    numbers = {
        "n_cells": int(case.model.n_cells),
        "manning_n": 0.02 if manning_n is None else manning_n,
        "newton_dtau0_s": settings.dtau0,
    }
    profiles = {
        "x": x[order_x],
        "zb": zb[order_x],
        "x_exact": x_exact,
        "wse_exact": wse_exact,
        "zb_exact": analytical[:, 2],
    }

    for order in (1, 2):
        # 5. Steady Newton solve at this order (second call timed, after compilation).
        model = dataclasses.replace(case.model, order=order)

        def steady(U0, model=model):
            """Newton solve from U0; returns (U, iterations, max residual)."""
            return newton_solve(model, {}, U0, settings)

        steady_jit = jax.jit(steady)
        start = time.perf_counter()
        U, iterations, residual = steady_jit(case.U0)
        jax.block_until_ready(U)
        first_seconds = time.perf_counter() - start
        start = time.perf_counter()
        U, iterations, residual = steady_jit(case.U0)
        jax.block_until_ready(U)
        solve_seconds = time.perf_counter() - start

        # 6. WSE errors at the cell centroids.
        h = np.asarray(U[0] + model.hstill)
        wse = h + zb
        error = wse - wse_exact_cells

        # 7. Errors away from the jump: cells more than 0.5 m from the analytical jump,
        #    located at the largest WSE increase between neighbouring analytical points.
        x_jump = x_exact[np.argmax(np.diff(wse_exact)) + 1]
        smooth = np.abs(x - x_jump) > 0.5
        numbers["x_jump_analytical_m"] = float(x_jump)
        numbers[f"order{order}"] = {
            "newton_iterations": int(iterations),
            "final_max_residual": float(residual),
            "solve_time_s": solve_seconds,
            "first_call_time_incl_compile_s": first_seconds,
            "wse_rms_error_m": float(np.sqrt(np.mean(error ** 2))),
            "wse_max_error_m": float(np.abs(error).max()),
            "x_of_max_error_m": float(x[np.argmax(np.abs(error))]),
            "wse_rms_error_excluding_jump_m": float(np.sqrt(np.mean(error[smooth] ** 2))),
            "wse_max_error_excluding_jump_m": float(np.abs(error[smooth]).max()),
        }
        profiles[f"wse_o{order}"] = wse[order_x]

    return numbers, profiles


# ----------------------------------------------------------------------
# (f): grid convergence on the smooth subcritical bump
# ----------------------------------------------------------------------
def run_convergence(work_dir: Path) -> dict:
    """L1 depth error against the exact Bernoulli depth on a sequence of meshes.

    Args:
        work_dir: scratch folder.

    Returns:
        {"nx": [k], "dx_m": [k], "L1_o1_m": [k], "L1_o2_m": [k], "newton_iterations_o1",
        "newton_iterations_o2", "rates_o1", "rates_o2" (observed orders between successive
        meshes), "fit_order_o1", "fit_order_o2" (least-squares slope of log e vs log dx)}.
    """
    import jax

    from hydrograd.solvers.steady import NewtonSettings, newton_solve

    # The case definition is shared with benchmarks/validation/convergence_bump.py.
    sys.path.insert(0, str(REPO_DIR / "benchmarks" / "validation"))
    from convergence_bump import H_OUT, Q, bump, exact_depth

    def bed(X, Y):
        """Bed elevation z_b(X) (m) of the smooth bump."""
        return bump(X)

    def outlet_wse(x, y):
        """Flat initial WSE at the outlet level H_OUT (m)."""
        return H_OUT + 0.0 * x

    result = {"nx": CONVERGENCE_NX, "dx_m": [25.0 / nx for nx in CONVERGENCE_NX]}
    for order in (1, 2):
        errors = []
        iterations = []
        for nx in CONVERGENCE_NX:
            # 1. 25 m x 1 m flume, nx x 1 cells.
            case = setup_synthetic(
                work_dir / f"conv_{order}_{nx}",
                nx=nx,
                ny=1,
                length=25.0,
                width=1.0,
                n_manning=1.0e-4,
                inlet="INLET-Q",
                outlet="EXIT-H",
                Q=Q,
                wse_out=H_OUT,
                wstill=H_OUT,
                order=order,
                zb_fn=bed,
                ic_wse=outlet_wse,
            )

            # 2. Steady Newton solve (settings of convergence_bump.py).
            def steady(U0, model=case.model):
                """Newton solve from U0; returns (U, iterations, max residual)."""
                return newton_solve(model, {}, U0, NewtonSettings(dtau0=0.05, max_iter=400))

            U, n_iterations, residual = jax.jit(steady)(case.U0)
            if float(residual) > 1.0e-9:
                print(f"warning: order {order}, nx {nx}: residual {float(residual):.1e}")

            # 3. L1 depth error (m).
            h = np.asarray(U[0] + case.model.hstill)
            h_exact = exact_depth(case.mesh.centroid[:, 0])
            errors.append(float(np.mean(np.abs(h - h_exact))))
            iterations.append(int(n_iterations))

        # 4. Observed orders between successive meshes and a least-squares fit.
        rates = [float(np.log2(errors[k] / errors[k + 1])) for k in range(len(errors) - 1)]
        slope = np.polyfit(np.log(result["dx_m"]), np.log(errors), 1)[0]
        result[f"L1_o{order}_m"] = errors
        result[f"newton_iterations_o{order}"] = iterations
        result[f"rates_o{order}"] = rates
        result[f"fit_order_o{order}"] = float(slope)
    return result


# ----------------------------------------------------------------------
# Driver: run everything and store data + numbers
# ----------------------------------------------------------------------
def compute() -> None:
    """Run all verification cases; write the npz data file and the JSON numbers."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import hydrograd  # noqa: F401  (enables float64)

    work_dir = Path(tempfile.mkdtemp(prefix="fig03_"))
    data = {}
    numbers = {"hardware": hardware_info(), "scheme": "hr_hllc (HR + HLLC)",
               "limiter_order2": "venkatakrishnan"}

    # Dam breaks at orders 1 and 2.
    for name, h_right in (("stoker", 0.1), ("ritter", 0.0)):
        numbers[name] = {}
        for order in (1, 2):
            case_numbers, profile = run_dam_break(h_right, order, work_dir)
            numbers[name][f"order{order}"] = case_numbers
            data[f"{name}_x"] = profile["x"]
            data[f"{name}_h_o{order}"] = profile["h"]
            print(name, order, case_numbers, flush=True)

    # Lake at rest at orders 1 and 2.
    numbers["lake_at_rest"] = {}
    for order in (1, 2):
        case_numbers, profile = run_lake_at_rest(order, work_dir)
        numbers["lake_at_rest"][f"order{order}"] = case_numbers
        data["lake_x"] = profile["x"]
        data["lake_zb"] = profile["zb"]
        data[f"lake_wse_o{order}"] = profile["wse"]
        print("lake", order, case_numbers, flush=True)

    # Transcritical bump: frictionless (plotted) and the example as distributed (n = 0.02).
    bump_numbers, bump_profiles = run_bump(work_dir, manning_n=1.0e-4)
    numbers["bump_transcritical_frictionless"] = bump_numbers
    for key, value in bump_profiles.items():
        data[f"bump_{key}"] = value
    print("bump (n = 0)", bump_numbers, flush=True)
    bump_numbers, _ = run_bump(work_dir, manning_n=None)
    numbers["bump_transcritical_example_n0.02"] = bump_numbers
    print("bump (n = 0.02)", bump_numbers, flush=True)

    # Grid convergence.
    convergence = run_convergence(work_dir)
    numbers["convergence_subcritical_bump"] = convergence
    print("convergence", convergence, flush=True)

    # Store data and numbers; remove the scratch cases.
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_FILE, **data, conv_dx=np.asarray(convergence["dx_m"]),
             conv_L1_o1=np.asarray(convergence["L1_o1_m"]),
             conv_L1_o2=np.asarray(convergence["L1_o2_m"]))
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(numbers, indent=2))
    shutil.rmtree(work_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def plot() -> None:
    """Draw the six-panel figure from the stored data and numbers."""
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PAPER_DIR))
    from figstyle import (
        FULL_WIDTH,
        MODEL,
        MODEL_2,
        REFERENCE,
        apply_style,
        panel_label,
        save,
    )

    from hydrograd.validation.dambreak import dam_break

    apply_style()
    data = np.load(DATA_FILE)
    numbers = json.loads(RESULT_FILE.read_text())
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.9))
    order1_style = dict(MODEL, label="1st order")
    order2_style = dict(MODEL_2, label="2nd order")

    # (a), (b): dam-break depth profiles against the exact solution on a fine grid.
    x_fine = np.linspace(0.0, 1000.0, 4001)
    dam_breaks = (
        (axes[0, 0], "stoker", 0.1, "Stoker (wet bed)"),
        (axes[0, 1], "ritter", 0.0, "Ritter (dry bed)"),
    )
    for ax, name, h_right, title in dam_breaks:
        h_exact, _ = dam_break(x_fine, T_DAM, H_LEFT, h_right, x0=X_DAM)
        ax.plot(x_fine, h_exact, **dict(REFERENCE, label="Exact"))
        ax.plot(data[f"{name}_x"], data[f"{name}_h_o1"], **order1_style)
        ax.plot(data[f"{name}_x"], data[f"{name}_h_o2"], **order2_style)
        ax.set_xlim(200.0, 1000.0)
        ax.set_ylim(-0.03, 1.1)
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$h$ (m)")
        ax.set_title(title, fontsize=10)
        ax.text(0.04, 0.05, r"$t = 60$ s", transform=ax.transAxes, fontsize=9)
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")

    # (c): lake at rest, bed and water surface along the island centre line.
    ax = axes[0, 2]
    ax.fill_between(data["lake_x"], -0.2, data["lake_zb"], color="0.8", lw=0, label="Bed")
    ax.plot(data["lake_x"], data["lake_wse_o2"], **dict(MODEL_2, label="WSE"))
    lake = numbers["lake_at_rest"]
    residual = max(lake["order1"]["max_abs_dUdt_t0"], lake["order2"]["max_abs_dUdt_t0"])
    speed = max(lake["order1"]["max_speed_m_per_s"], lake["order2"]["max_speed_m_per_s"])
    ax.text(0.04, 0.96,
            rf"$\max|\partial_t U|_{{t=0}} = {latex_exp(residual)}$" + "\n"
            + rf"$\max|\mathbf{{u}}|_{{600\,\mathrm{{s}}}} = {latex_exp(speed)}$ m/s",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5, linespacing=1.4)
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(-0.2, 1.7)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$z$ (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax.set_title("Lake at rest", fontsize=10)

    # (d): transcritical bump, WSE against the analytical solution.
    ax = axes[1, 0]
    draw_bump(ax, data, order1_style, order2_style, REFERENCE)
    ax.set_xlim(0.0, 25.0)
    ax.set_ylim(0.0, 0.5)
    ax.legend(loc="lower right", fontsize=8, handlelength=1.3, handletextpad=0.4)
    ax.set_title("Transcritical bump", fontsize=10)

    # (e): close-up of the hydraulic jump.
    ax = axes[1, 1]
    draw_bump(ax, data, order1_style, order2_style, REFERENCE)
    ax.set_xlim(10.5, 13.0)
    ax.set_ylim(0.0, 0.45)
    ax.set_title("Hydraulic jump (close-up)", fontsize=10)

    # (f): grid convergence with reference slopes 1 and 2.
    ax = axes[1, 2]
    dx = data["conv_dx"]
    fit1 = numbers["convergence_subcritical_bump"]["fit_order_o1"]
    fit2 = numbers["convergence_subcritical_bump"]["fit_order_o2"]
    ax.loglog(dx, data["conv_L1_o1"], marker="o", mfc="none",
              **dict(MODEL, label=rf"1st order, $p = {fit1:.2f}$"))
    ax.loglog(dx, data["conv_L1_o2"], marker="s", mfc="none",
              **dict(MODEL_2, label=rf"2nd order, $p = {fit2:.2f}$"))

    # Reference slopes below the data, labelled "1" and "2".
    slope1_line = 0.4 * data["conv_L1_o1"][0] * (dx / dx[0])
    slope2_line = 0.4 * data["conv_L1_o2"][0] * (dx / dx[0]) ** 2
    ax.loglog(dx, slope1_line, color="0.5", lw=0.8, ls=":")
    ax.loglog(dx, slope2_line, color="0.5", lw=0.8, ls=":")
    ax.text(dx[2], slope1_line[2] * 0.5, "1", fontsize=8, color="0.4")
    ax.text(dx[2], slope2_line[2] * 0.4, "2", fontsize=8, color="0.4")

    # Plain decimal ticks on the cell-size axis.
    ax.set_xticks([0.03, 0.1, 0.3])
    ax.set_xticklabels(["0.03", "0.1", "0.3"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Cell size (m)")
    ax.set_ylabel(r"$L_1$ error in $h$ (m)")
    ax.set_ylim(top=1.5e-1)
    ax.legend(loc="upper left", fontsize=8, handlelength=2.2, handletextpad=0.4)
    ax.set_title("Grid convergence", fontsize=10)

    # Panel labels and spacing.
    for label, ax in zip("abcdef", axes.flat):
        panel_label(ax, f"({label})", x=-0.24, y=1.03)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, "fig03_verification")


def draw_bump(ax, data, order1_style: dict, order2_style: dict, reference_style: dict) -> None:
    """Bed, analytical WSE and the orders-1 and 2 WSE of the transcritical bump.

    Args:
        ax: matplotlib axes.
        data: the loaded npz data (bump_* arrays, m).
        order1_style, order2_style, reference_style: line styles.
    """
    ax.fill_between(data["bump_x_exact"], 0.0, data["bump_zb_exact"], color="0.8", lw=0,
                    label="Bed")
    ax.plot(data["bump_x_exact"], data["bump_wse_exact"],
            **dict(reference_style, label="Analytical"))
    ax.plot(data["bump_x"], data["bump_wse_o1"], **order1_style)
    ax.plot(data["bump_x"], data["bump_wse_o2"], **order2_style)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$z$ (m)")


def latex_exp(value: float) -> str:
    """Format a small positive number as LaTeX, e.g. 2.2e-16 -> '2.2 \\times 10^{-16}'.

    Args:
        value: the number (> 0); 0 gives '0'.

    Returns:
        LaTeX math string (without dollar signs).
    """
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0 ** exponent
    return rf"{mantissa:.1f} \times 10^{{{exponent}}}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="plot from paper/data and paper/results without rerunning")
    arguments = parser.parse_args()
    if not arguments.plot_only:
        compute()
    plot()
