"""Grid convergence: steady subcritical flow over a bump (SWASHES), orders 1 and 2.

Channel 25 m, bump z = 0.2 - 0.05 (x - 10)^2 for 8 < x < 12, q = 4.42 m^2/s, h_out = 2 m,
frictionless (n = 1e-4). Exact depth from Bernoulli + continuity (subcritical root).
"""
import tempfile, sys
from pathlib import Path

import jax
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.steady import NewtonSettings, newton_solve

G, Q, H_OUT = 9.81, 4.42, 2.0
bump = lambda x: np.where((x > 8) & (x < 12), 0.2 - 0.05 * (x - 10) ** 2, 0.0)


def exact_depth(x: np.ndarray) -> np.ndarray:
    E = H_OUT + Q ** 2 / (2 * G * H_OUT ** 2)               # specific energy (z = 0 at the outlet)
    out = []
    for z in bump(x):
        roots = np.roots([1.0, z - E, 0.0, Q ** 2 / (2 * G)])
        roots = roots[np.isreal(roots)].real
        hc = (Q ** 2 / G) ** (1 / 3)
        out.append(roots[roots > hc].min())                 # subcritical branch
    return np.asarray(out)


def run(nx: int, order: int) -> float:
    d = Path(tempfile.mkdtemp()) / "c"
    write_channel_case(d, nx, 1, length=25.0, width=1.0, n_manning=1e-4, inlet="INLET-Q", outlet="EXIT-H",
                       Q=Q, wse_out=H_OUT, wstill=H_OUT, order=order, zb_fn=lambda X, Y: bump(X),
                       ic_wse=lambda x, y: H_OUT + 0 * x)
    case = setup_case(read_control(d / "run_control.json"))
    U, it, rn = jax.jit(lambda U0: newton_solve(case.model, {}, U0, NewtonSettings(dtau0=0.05, max_iter=400)))(case.U0)
    h = np.asarray(U[0] + case.model.hstill)
    x = case.mesh.centroid[:, 0]
    err = np.mean(np.abs(h - exact_depth(x)))
    print(f"order {order} nx {nx:4d}: Newton {int(it):3d} its, max|R| {float(rn):.1e}, L1(h) {err:.3e} m", flush=True)
    return err


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [50, 100, 200, 400]
    for order in (1, 2):
        e = [run(n, order) for n in ns]
        rates = [np.log2(e[k] / e[k + 1]) for k in range(len(e) - 1)]
        print(f"order {order}: observed convergence rates {np.round(rates, 2)}", flush=True)
