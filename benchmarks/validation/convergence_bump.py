"""Grid convergence: steady subcritical flow over a bump (SWASHES), orders 1 and 2.

Usage: python convergence_bump.py [NX ...]   (default 50 100 200 400)

Case: channel 25 m long and 1 m wide (one cell across), bump
z_b(x) = 0.2 - 0.05 (x - 10)^2 (m) for 8 < x < 12 m, zero elsewhere; unit discharge
q = Q = 4.42 m^2/s, outlet depth h_out = 2 m, frictionless (n = 1e-4).

Exact depth: with z = 0 at the outlet, the specific energy E = h_out + q^2 / (2 g h_out^2)
is constant (Bernoulli), so the depth h solves the cubic

    h^3 + (z_b - E) h^2 + q^2 / (2 g) = 0,

of which the subcritical root (h > h_c = (q^2 / g)^(1/3)) is taken.

The steady state is computed by pseudo-transient Newton; the script prints the L1 depth
error (m) on each mesh and the observed convergence rates log2(e_k / e_{k+1}).
"""
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.steady import NewtonSettings, newton_solve

# Gravity (m/s^2), unit discharge (m^2/s), outlet depth (m).
G, Q, H_OUT = 9.81, 4.42, 2.0


def bump(x):
    """Bed elevation of the bump (m).

    Args:
        x: streamwise coordinates (m), any shape.

    Returns:
        z_b (m), same shape as ``x``.
    """
    return np.where((x > 8) & (x < 12), 0.2 - 0.05 * (x - 10) ** 2, 0.0)


def exact_depth(x: np.ndarray) -> np.ndarray:
    """Exact subcritical depth over the bump.

    Args:
        x: streamwise coordinates (m), shape [n].

    Returns:
        Depth h (m), shape [n].
    """
    # Specific energy (z = 0 at the outlet).
    E = H_OUT + Q ** 2 / (2 * G * H_OUT ** 2)

    out = []
    for z in bump(x):
        # Real roots of h^3 + (z - E) h^2 + q^2 / (2 g) = 0.
        roots = np.roots([1.0, z - E, 0.0, Q ** 2 / (2 * G)])
        roots = roots[np.isreal(roots)].real

        # Subcritical branch: the smallest root above the critical depth.
        hc = (Q ** 2 / G) ** (1 / 3)
        out.append(roots[roots > hc].min())
    return np.asarray(out)


def run(nx: int, order: int) -> float:
    """Steady solve on an nx-cell mesh and its L1 depth error.

    Args:
        nx: number of cells along the channel.
        order: spatial order of the scheme (1 or 2).

    Returns:
        Mean absolute depth error against ``exact_depth`` (m).
    """

    def bed(X, Y):
        """Bed elevation z_b(X) (m) at coordinates X, Y (m)."""
        return bump(X)

    def initial_wse(x, y):
        """Flat initial water surface at the outlet level H_OUT (m)."""
        return H_OUT + 0 * x

    # 1. Write the case.
    case_dir = Path(tempfile.mkdtemp()) / "c"
    write_channel_case(case_dir, nx, 1, length=25.0, width=1.0, n_manning=1e-4,
                       inlet="INLET-Q", outlet="EXIT-H", Q=Q, wse_out=H_OUT, wstill=H_OUT,
                       order=order, zb_fn=bed, ic_wse=initial_wse)
    case = setup_case(read_control(case_dir / "run_control.json"))

    # 2. Steady state by pseudo-transient Newton.
    def steady(U0):
        """Newton solve from U0; returns (U_steady, iterations, max residual)."""
        return newton_solve(case.model, {}, U0, NewtonSettings(dtau0=0.05, max_iter=400))

    U, it, rn = jax.jit(steady)(case.U0)

    # 3. L1 depth error at the cell centroids.
    h = np.asarray(U[0] + case.model.hstill)
    x = case.mesh.centroid[:, 0]
    err = np.mean(np.abs(h - exact_depth(x)))
    print(f"order {order} nx {nx:4d}: Newton {int(it):3d} its, max|R| {float(rn):.1e}, "
          f"L1(h) {err:.3e} m", flush=True)
    return err


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [50, 100, 200, 400]
    for order in (1, 2):
        # Errors on the mesh sequence, then the observed rates between successive meshes.
        e = [run(n, order) for n in ns]
        rates = [np.log2(e[k] / e[k + 1]) for k in range(len(e) - 1)]
        print(f"order {order}: observed convergence rates {np.round(rates, 2)}", flush=True)
