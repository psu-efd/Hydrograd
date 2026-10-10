"""Active parameters (``active_param_names``) <-> parameter dict ``theta``.

Inversion and sensitivity analysis differentiate with respect to one group of
parameters, named in ``run_control.json`` by ``active_param_names``. Hydrograd.jl
supports exactly one active parameter group:

* ``ManningN`` -> ``theta["n_zone"]``, one value per material ID 0..n_materials-1 (s/m^(1/3));
* ``zb``       -> ``theta["zb"]``, one value per cell (m);
* ``Q``        -> ``theta["Q"]``, one value per inlet-Q boundary (m^3/s);
* ``veg_CD``   -> ``theta["veg_CD"]``, the stem drag coefficient per material ID (-)
  (extension; needs a ``"vegetation"`` block in the run control);
* ``porous_alpha`` -> ``theta["df_alpha"]``, the Darcy coefficient per material ID (1/s),
  and ``porous_beta`` -> ``theta["df_beta"]``, the Forchheimer coefficient per material
  ID (1/m) (extension; need a ``"porous"`` block in the run control).

The initial (inversion) or evaluation (sensitivity) values come either from the run
control or from a JSON file with the keys of ``_JSON_KEY`` (the format Hydrograd.jl
writes). Nothing is written.
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

# Active parameter name (run control) -> key in the parameter dict theta.
THETA_KEY = {
    "ManningN": "n_zone",
    "zb": "zb",
    "Q": "Q",
    "veg_CD": "veg_CD",
    "porous_alpha": "df_alpha",
    "porous_beta": "df_beta",
}

# Active parameter name -> key in a parameter JSON file ("from_file" option).
_JSON_KEY = {
    "ManningN": "ManningN_list_param",
    "zb": "zb_cells_param",
    "Q": "inlet_discharges_param",
    "veg_CD": "veg_CD_list_param",
    "porous_alpha": "porous_alpha_list_param",
    "porous_beta": "porous_beta_list_param",
}


def active_name(case: Case) -> str:
    """The single active parameter name.

    Args:
        case: the case (its run control lists ``active_param_names``).

    Returns:
        ``"ManningN"``, ``"zb"``, ``"Q"``, ``"veg_CD"``, ``"porous_alpha"`` or
        ``"porous_beta"``.
    """
    names = case.ctrl.opt("active_param_names")
    if len(names) != 1 or names[0] not in THETA_KEY:
        raise ValueError(f"active_param_names must be one of {list(THETA_KEY)}, got {names}")

    return names[0]


def _expected_len(case: Case, name: str) -> int:
    """Number of values of the active parameter group.

    Args:
        case: the case.
        name: active parameter name (a key of ``THETA_KEY``).

    Returns:
        Number of material zones, cells, inlet-Q boundaries or material zones.
    """
    m = case.model
    if name == "veg_CD" and m.vegetation.mode != "zone":
        raise ValueError("active parameter veg_CD needs a 'vegetation' block in the run control")
    if name.startswith("porous_") and m.porous.mode == "none":
        raise ValueError(f"active parameter {name} needs a 'porous' block in the run control")

    lengths = {"ManningN": len(m.n_zone), "zb": m.n_cells, "Q": m.n_inlets,
               "veg_CD": len(m.n_zone), "porous_alpha": len(m.n_zone),
               "porous_beta": len(m.n_zone)}
    return lengths[name]


def initial_params(case: Case, prefix: str) -> tuple[str, Array]:
    """Initial (inversion) or evaluation (sensitivity) parameter vector from the run control.

    Args:
        case: the case.
        prefix: ``"inversion"`` or ``"sensitivity"`` (option-key prefix).

    Returns:
        ``(name, p)``: the active parameter name and its values, shape [n_params], in the
        units of the parameter (s/m^(1/3), m or m^3/s).
    """
    name = active_name(case)
    o = case.ctrl.opt

    # Inversion keys say "initial_values", sensitivity keys say "values".
    if prefix == "inversion":
        values_option_key = f"{prefix}_parameter_initial_values_options"
    else:
        values_option_key = f"{prefix}_parameter_values_options"
    how = o(values_option_key)

    if how == "from_file":
        # Values from a parameter JSON file.
        if prefix == "inversion":
            file_key = f"{prefix}_parameter_initial_values_file_name"
        else:
            file_key = f"{prefix}_parameter_values_file_name"
        param_data = json.loads((case.ctrl.case_dir / o(file_key)).read_text())
        p = np.asarray(param_data[_JSON_KEY[name]], dtype=float)
    else:
        # Values listed in the run control.
        suffix = "initial_values" if prefix == "inversion" else "values"
        value_keys = {
            "ManningN": f"{prefix}_ManningN_{suffix}",
            "zb": f"{prefix}_zb_{suffix}",
            "Q": f"{prefix}_inlet_discharge_{suffix}",
            "veg_CD": f"{prefix}_veg_CD_{suffix}",
            "porous_alpha": f"{prefix}_porous_alpha_{suffix}",
            "porous_beta": f"{prefix}_porous_beta_{suffix}",
        }
        p = np.asarray(o(value_keys[name]), dtype=float)

        # Julia: the first zb value fills all cells.
        if name == "zb":
            p = np.full(case.model.n_cells, p[0])

    # Check the length against the mesh / materials / boundaries.
    n_expected = _expected_len(case, name)
    if len(p) != n_expected:
        raise ValueError(f"{name}: expected {n_expected} parameter values, got {len(p)}")

    logger.info(
        "active parameter %s (%d values), initial %s",
        name, n_expected, np.array2string(p[:8], precision=4),
    )
    return name, jnp.asarray(p)


def to_theta(name: str, p: Array) -> dict[str, Any]:
    """Parameter vector -> theta dict for the residual.

    Args:
        name: active parameter name (a key of ``THETA_KEY``).
        p: parameter values, shape [n_params].

    Returns:
        ``{THETA_KEY[name]: p}``.
    """
    return {THETA_KEY[name]: p}
