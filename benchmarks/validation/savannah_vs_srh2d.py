"""Savannah River steady state: hydrograd against SRH-2D.

The Hydrograd.jl forward example ``Savannah_River`` starts from the SRH-2D steady
solution (``forward_simulation_initial_condition.json``: water-surface elevation and unit
discharges per cell, Q = 187.4 m^3/s). This script solves the steady state with hydrograd
(``SteadyNewton()``, default scheme ``hr_hllc``) at first and second order, and reports
the differences against SRH-2D:

- RMS and maximum of the water-surface elevation difference (mm);
- RMS of the depth-averaged speed difference (m/s).

The Julia case is copied to a temporary folder (it is never modified).

Usage
-----
    python savannah_vs_srh2d.py <Hydrograd.jl>/examples/SWE_2D/forward_simulation/Savannah_River
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.solvers.final import final_state


def main() -> None:
    """Solve the steady state at orders 1 and 2 and print the differences to SRH-2D."""
    source = Path(sys.argv[1])

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Copy the case and switch it to a steady Newton solve.
        case_dir = Path(tmp) / "case"
        shutil.copytree(source, case_dir, ignore=shutil.ignore_patterns("*.vtk", "SRH_2D"))
        control_text = (case_dir / "run_control.json").read_text()
        control = json.loads(re.sub(r",(\s*[}\]])", r"\1", control_text))
        control["forward_simulation_options"]["forward_simulation_ode_solver"] = "SteadyNewton()"

        # 2. SRH-2D reference: the initial-condition file.
        reference = json.loads((case_dir / "forward_simulation_initial_condition.json").read_text())
        wse_srh = np.asarray(reference["wse"])

        for order in (1, 2):
            control["numerics"] = {"spatial_order": order}
            (case_dir / "run_control.json").write_text(json.dumps(control))
            case = setup_case(read_control(case_dir / "run_control.json"))
            model = case.model

            # 3. Steady solve, starting from the SRH-2D state.
            U, converged = final_state(model, {}, case.U0, case.ode)
            depth = np.asarray(U[0] + model.hstill)
            wse = np.asarray(U[0] + model.wstill)
            speed = np.hypot(np.asarray(U[1]), np.asarray(U[2])) / np.maximum(depth, 1e-3)

            # 4. Differences on wet cells.
            depth_srh = wse_srh - np.asarray(model.zb)
            q_srh = np.hypot(np.asarray(reference["q_x"]), np.asarray(reference["q_y"]))
            speed_srh = q_srh / np.maximum(depth_srh, 1e-3)
            wet = (depth > 0.01) & (depth_srh > 0.01)
            wse_error = wse[wet] - wse_srh[wet]
            speed_error = speed[wet] - speed_srh[wet]
            print(
                f"order {order}: converged {bool(converged)}; WSE RMS "
                f"{1000 * np.sqrt(np.mean(wse_error ** 2)):.1f} mm "
                f"(max {1000 * np.max(np.abs(wse_error)):.0f} mm, mean "
                f"{1000 * np.mean(wse_error):+.1f} mm); speed RMS "
                f"{np.sqrt(np.mean(speed_error ** 2)):.3f} m/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
