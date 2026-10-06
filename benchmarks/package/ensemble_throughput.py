"""Ensemble (batched) simulation throughput: vmap over hydrograph scenarios.

Usage: python ensemble_throughput.py [BATCH ...]   (default 1 8 32 128)
500-cell compound valley, 1 h flood, explicit SSPRK3 (order 2). Each ensemble member has
its own scaled hydrograph; reports wall time and members per second.
"""
import sys, tempfile, time
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

T = 3600.0
d = Path(tempfile.mkdtemp()) / "c"
tk = np.arange(0.0, T + 1.0, 900.0)
write_channel_case(d, H.NX, H.NY, length=H.L, width=H.W, n_manning=H.N_MAN, zb_fn=H.bed, wstill=0.0, dt=0.1,
                   t_end=T, solver="SSPRK3()", inlet="INLET-Q", outlet="EXIT-H", wse_out=1.03,
                   Q_series=(tk, H.q_true(tk)), ic_wse=lambda x, y: H.S0 * (H.L - x) + 1.03)
case = setup_case(read_control(d / "run_control.json"))
m, s = case.model, case.ode
run = jax.jit(jax.vmap(lambda q: solve_swe(m, {"Q": q}, case.U0, s).ys[-1][0] + m.hstill))
print(f"device: {jax.devices()[0].device_kind}", flush=True)
base = None
for B in [int(a) for a in sys.argv[1:]] or [1, 8, 32, 128]:
    scale = jnp.linspace(0.5, 1.5, B)[:, None, None]
    qs = scale * jnp.asarray(m.Q)[None]
    jax.block_until_ready(run(qs))
    t0 = time.perf_counter(); jax.block_until_ready(run(qs)); dt = time.perf_counter() - t0
    base = base or dt
    print(f"batch {B:4d}: {dt:7.1f} s, {B / dt:6.2f} members/s ({B * base / dt:5.1f}x the throughput of batch 1)", flush=True)
