"""Applications of the 1-D mode: forward simulation, sensitivity analysis and inversion.

``hydrograd run`` dispatches here when ``run_control.json`` has ``"model": "1D"`` (see
:mod:`hydrograd.io.control`). The case itself is one JSON file
(:mod:`hydrograd.io.oned_case`).

Modules
-------
forward      Run the case; write the results and the surveyed deposit volumes.
observe      The observable: bulk deposit volume per cross-section at the survey times.
sensitivity  Jacobian of the surveyed volumes with respect to the active parameters.
inversion    Calibration of the active parameters against surveyed volumes.

The UDE mode (neural-network closures) exists for 2-D cases only.
"""

from __future__ import annotations

from typing import Any

from hydrograd.io.control import Control
from hydrograd.io.oned_case import read_case

__all__ = ["run_oned"]


def run_oned(ctrl: Control) -> dict[str, Any]:
    """Read the 1-D case of a run control and run the selected mode.

    The application modules are imported only when needed.

    Args:
        ctrl: run control with ``model == "1D"``.

    Returns:
        The result dictionary of the mode.
    """
    case = read_case(ctrl.case_dir / ctrl.case_file)

    if ctrl.mode == "forward":
        from hydrograd.apps.oned.forward import run_forward
        return run_forward(ctrl, case)

    if ctrl.mode == "sensitivity":
        from hydrograd.apps.oned.sensitivity import run_sensitivity
        return run_sensitivity(ctrl, case)

    if ctrl.mode == "inversion":
        from hydrograd.apps.oned.inversion import run_inversion
        return run_inversion(ctrl, case)

    raise NotImplementedError("the UDE mode is not available for 1-D cases")
