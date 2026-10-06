"""Reader for Hydrograd.jl's ``run_control.json`` (same keys, same meaning).

Only the run mode and the time settings are typed here; each application reads
its own option block (``forward_simulation_options``, ``inversion_options``,
``sensitivity_analysis_options``, ``UDE_options``) through :meth:`Control.opt`,
which raises a clear error for missing keys.

File layout (the keys read here)
--------------------------------
::

    {
      "control_variables": {
        "srhhydro_file_name": "<case>.srhhydro",
        "bPerform_Forward_Simulation": true,      (exactly one bPerform_* is true)
        ...
      },
      "time_settings": {
        "bUse_srhhydro_time_settings": false,
        "tspan": [t0, t1],                        (s)
        "dt": dt                                  (s)
      },
      "<mode>_options": { ... }
    }

Trailing commas, which some example files contain, are tolerated.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["MODES", "Control", "read_control"]

# Run-mode flag in "control_variables" -> (mode name, name of its option block).
MODES = {
    "bPerform_Forward_Simulation": ("forward", "forward_simulation_options"),
    "bPerform_Inversion": ("inversion", "inversion_options"),
    "bPerform_Sensitivity_Analysis": ("sensitivity", "sensitivity_analysis_options"),
    "bPerform_UDE": ("ude", "UDE_options"),
}


@dataclass(frozen=True)
class Control:
    """Parsed run control.

    Attributes:
        path: the JSON file.
        case_dir: its directory (all files are relative to it).
        mode: ``forward``, ``inversion``, ``sensitivity`` or ``ude``.
        srhhydro_file: name of the ``.srhhydro`` file.
        use_srhhydro_time: take time span and dt from the ``.srhhydro`` SimTime.
        tspan: (t0, t1) (s).
        dt: (initial) time step (s).
        options: the option block of the active mode.
        raw: the whole JSON.
    """

    path: Path
    case_dir: Path
    mode: str
    srhhydro_file: str
    use_srhhydro_time: bool
    tspan: tuple[float, float]
    dt: float
    options: dict[str, Any]
    raw: dict[str, Any]

    def opt(self, key: str, default: Any = ...) -> Any:
        """Option of the active mode; raises KeyError naming the block if missing.

        Args:
            key: option name inside the active mode's option block.
            default: value returned when the key is missing; if omitted (``...``),
                a missing key is an error.

        Returns:
            The option value, or ``default``.

        Raises:
            KeyError: if the key is missing and no default is given.
        """
        if key in self.options:
            return self.options[key]

        if default is ...:
            raise KeyError(f"run_control.json: '{key}' missing in {MODES_BLOCK[self.mode]}")

        return default


# Mode name -> name of its option block, e.g. "forward" -> "forward_simulation_options".
MODES_BLOCK = {mode: block for mode, block in MODES.values()}


def _load_json_with_comments(text: str) -> dict[str, Any]:
    """json.loads tolerant of trailing commas (some example files have them).

    Args:
        text: JSON text, possibly with a comma before a closing ``}`` or ``]``.

    Returns:
        The decoded JSON object.
    """
    # Drop any comma that is followed (after optional whitespace) by } or ].
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)

    return json.loads(cleaned)


def read_control(path: str | Path) -> Control:
    """Parse ``run_control.json``.

    Args:
        path: the ``run_control.json`` file.

    Returns:
        The parsed control, with the option block of the active mode.

    Raises:
        ValueError: if not exactly one ``bPerform_*`` flag is true.
    """
    path = Path(path).resolve()
    raw = _load_json_with_comments(path.read_text())
    cv = raw["control_variables"]

    # Exactly one bPerform_* flag must be true; it selects the mode and its option block.
    active = [MODES[k] for k in MODES if cv.get(k, False)]
    if len(active) != 1:
        raise ValueError(f"{path}: exactly one bPerform_* flag must be true, got {len(active)}")
    mode, block = active[0]

    # Time settings: span (t0, t1) and step dt (s).
    ts = raw["time_settings"]

    ctrl = Control(
        path=path,
        case_dir=path.parent,
        mode=mode,
        srhhydro_file=cv["srhhydro_file_name"],
        use_srhhydro_time=bool(ts["bUse_srhhydro_time_settings"]),
        tspan=(float(ts["tspan"][0]), float(ts["tspan"][1])),
        dt=float(ts["dt"]),
        options=raw.get(block, {}),
        raw=raw,
    )

    logger.info("run_control: mode=%s, tspan=%s s, dt=%g s", mode, ctrl.tspan, ctrl.dt)

    return ctrl
