"""Ensemble (batched) simulation throughput: vmap over hydrograph scenarios.

Usage: python ensemble_throughput.py [BATCH ...]   (default 1 8 32 128)

Case: the 500-cell compound valley of ``hydrograph_inversion.py`` (NX x NY cells, length L,
width W, Manning n = N_MAN), a 1 h flood (T = 3600 s), explicit SSPRK3 (order 2). The inflow
hydrograph Q(t) (m^3/s) is the "true" flood of ``hydrograph_inversion.q_true`` sampled every
900 s. Member b of a batch of size B uses the scaled hydrograph s_b Q(t), with s_b spaced
evenly on [0.5, 1.5]. The whole batch runs as one ``jax.vmap``-ed, jitted solve.

Output: for each batch size, the wall time (s) of the second (compiled) call, the throughput
(members/s) and the speed-up in throughput relative to the first batch size.
"""
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import hydrograph_inversion as H  # noqa: E402

import hydrograd  # noqa: F401,E402
from hydrograd.apps.setup import setup_case  # noqa: E402
from hydrograd.io.control import read_control  # noqa: E402
from hydrograd.io.synthetic import write_channel_case  # noqa: E402
from hydrograd.solvers.ode import solve_swe  # noqa: E402

# Simulated duration (s).
T = 3600.0


def initial_wse(x, y):
    """Initial water-surface elevation: base-flow depth 1.03 m above the sloping bed.

    Args:
        x: cell centroid x coordinates (m), shape [n_cells].
        y: cell centroid y coordinates (m), shape [n_cells] (unused).

    Returns:
        WSE (m), shape [n_cells].
    """
    return H.S0 * (H.L - x) + 1.03


# 1. Write the synthetic compound-valley case into a temporary folder.
case_dir = Path(tempfile.mkdtemp()) / "c"
t_knots = np.arange(0.0, T + 1.0, 900.0)
write_channel_case(
    case_dir, H.NX, H.NY, length=H.L, width=H.W, n_manning=H.N_MAN, zb_fn=H.bed, wstill=0.0,
    dt=0.1, t_end=T, solver="SSPRK3()", inlet="INLET-Q", outlet="EXIT-H", wse_out=1.03,
    Q_series=(t_knots, H.q_true(t_knots)), ic_wse=initial_wse,
)
case = setup_case(read_control(case_dir / "run_control.json"))
model = case.model
ode_settings = case.ode


def final_depth(q):
    """Water depth at the end of the run for one hydrograph.

    Args:
        q: hydrograph values at the inlet knots (m^3/s), shape of ``model.Q``.

    Returns:
        Final depth h = xi + h_still (m), shape [n_cells].
    """
    return solve_swe(model, {"Q": q}, case.U0, ode_settings).ys[-1][0] + model.hstill


# 2. Batched solver: vmap over the leading (member) axis, then jit.
run = jax.jit(jax.vmap(final_depth))
print(f"device: {jax.devices()[0].device_kind}", flush=True)

# 3. Time each batch size (first call compiles; the second call is timed).
base = None
batch_sizes = [int(a) for a in sys.argv[1:]] or [1, 8, 32, 128]
for B in batch_sizes:
    # Member hydrographs: scale factors 0.5..1.5 times the base hydrograph.
    scale = jnp.linspace(0.5, 1.5, B)[:, None, None]
    qs = scale * jnp.asarray(model.Q)[None]

    # Compile, then time a compiled call.
    jax.block_until_ready(run(qs))
    t0 = time.perf_counter()
    jax.block_until_ready(run(qs))
    dt = time.perf_counter() - t0

    # Throughput relative to the first batch size.
    base = base or dt
    print(
        f"batch {B:4d}: {dt:7.1f} s, {B / dt:6.2f} members/s "
        f"({B * base / dt:5.1f}x the throughput of batch 1)",
        flush=True,
    )
