"""Active parameters (``active_param_names``) <-> parameter dict ``theta``.

Hydrograd.jl supports exactly one active parameter group:

* ``ManningN`` -> ``theta["n_zone"]``, one value per material ID 0..n_materials-1;
* ``zb``       -> ``theta["zb"]``, one value per cell (m);
* ``Q``        -> ``theta["Q"]``, one value per inlet-Q boundary (m^3/s).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.apps.setup import Case

logger = logging.getLogger(__name__)

__all__ = ["THETA_KEY", "active_name", "initial_params", "to_theta"]

THETA_KEY = {"ManningN": "n_zone", "zb": "zb", "Q": "Q"}
_JSON_KEY = {"ManningN": "ManningN_list_param", "zb": "zb_cells_param", "Q": "inlet_discharges_param"}


def active_name(case: Case) -> str:
    """The single active parameter name."""
    names = case.ctrl.opt("active_param_names")
    if len(names) != 1 or names[0] not in THETA_KEY:
        raise ValueError(f"active_param_names must be one of {list(THETA_KEY)}, got {names}")
    return names[0]


def _expected_len(case: Case, name: str) -> int:
    m = case.model
    return {"ManningN": len(m.n_zone), "zb": m.n_cells, "Q": m.n_inlets}[name]


def initial_params(case: Case, prefix: str) -> tuple[str, Array]:
    """Initial/evaluation parameter vector from the run control.

    Args:
        case: the case.
        prefix: ``inversion`` or ``sensitivity`` (option-key prefix).

    Returns:
        (name, vector).
    """
    name = active_name(case)
    o = case.ctrl.opt
    vopt = f"{prefix}_parameter_initial_values_options" if prefix == "inversion" else f"{prefix}_parameter_values_options"
    how = o(vopt)
    if how == "from_file":
        fkey = (f"{prefix}_parameter_initial_values_file_name" if prefix == "inversion"
                else f"{prefix}_parameter_values_file_name")
        d = json.loads((case.ctrl.case_dir / o(fkey)).read_text())
        p = np.asarray(d[_JSON_KEY[name]], dtype=float)
    else:
        suffix = "initial_values" if prefix == "inversion" else "values"
        key = {"ManningN": f"{prefix}_ManningN_{suffix}", "zb": f"{prefix}_zb_{suffix}",
               "Q": f"{prefix}_inlet_discharge_{suffix}"}[name]
        p = np.asarray(o(key), dtype=float)
        if name == "zb":
            p = np.full(case.model.n_cells, p[0])  # Julia: first value fills all cells
    n = _expected_len(case, name)
    if len(p) != n:
        raise ValueError(f"{name}: expected {n} parameter values, got {len(p)}")
    logger.info("active parameter %s (%d values), initial %s", name, n,
                np.array2string(p[:8], precision=4))
    return name, jnp.asarray(p)


def to_theta(name: str, p: Array) -> dict[str, Any]:
    """Parameter vector -> theta dict for the residual."""
    return {THETA_KEY[name]: p}
