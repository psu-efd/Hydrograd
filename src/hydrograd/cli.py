"""Command line: ``hydrograd run [run_control.json]`` inside a case directory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hydrograd.io.control import read_control

logger = logging.getLogger("hydrograd")


def run(control_path: Path) -> None:
    """Dispatch on the bPerform_* flag of ``run_control.json``."""
    from hydrograd.apps.setup import setup_case

    ctrl = read_control(control_path)
    case = setup_case(ctrl)
    if ctrl.mode == "forward":
        from hydrograd.apps.forward import run_forward
        run_forward(case)
    elif ctrl.mode == "sensitivity":
        from hydrograd.apps.sensitivity import run_sensitivity
        run_sensitivity(case)
    elif ctrl.mode == "inversion":
        from hydrograd.apps.inversion import run_inversion
        run_inversion(case)
    else:
        from hydrograd.apps.ude import run_ude
        run_ude(case)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hydrograd", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the case described by run_control.json")
    r.add_argument("control", nargs="?", default="run_control.json", type=Path)
    r.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    run(a.control)
    return 0


if __name__ == "__main__":
    sys.exit(main())
