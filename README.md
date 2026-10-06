# hydrograd (Python/JAX)

Differentiable finite-volume solver for the 2D shallow water equations (SWEs) and Universal SWEs
(SWEs with neural-network flow-resistance closures). It is a Python/JAX port and extension of
[Hydrograd.jl](https://github.com/psu-efd/Hydrograd.jl) (Liu & Song 2025, *Water Resources Research*,
doi:10.1029/2025WR040265).

- **Input:** SRH-2D case files (`.srhhydro`, `.srhgeom`, `.srhmat`) and Hydrograd.jl's `run_control.json`.
- **Modes:** forward simulation, sensitivity analysis, inversion (Manning's n, bed elevation, inflow) and UDE training.
- **Hardware:** CPU and GPU (JAX, float64).

## Install

```bash
uv pip install -e ".[dev]"            # CPU
uv pip install -e ".[dev,cuda]"       # + JAX CUDA 12
uv pip install -e ".[pardiso]"        # optional: multithreaded sparse LU (faster steady solves)
```

On Roar the environment already exists: `source hpc/roar/env.sh`. See `CLAUDE.md` for the Roar workflow.

## Run a case

Inside a case folder containing the SRH-2D files and a `run_control.json`:

```bash
hydrograd run            # dispatches on bPerform_Forward_Simulation / _Inversion / _Sensitivity_Analysis / _UDE
```

**Copy Hydrograd.jl examples before running them.** A forward run writes `forward_simulation_solution_truth.json` and would overwrite Julia's truth file.

**Outputs** use Hydrograd.jl's names:
- `forward_simulation_results_%04d.vtk` (binary above 20k cells; open in ParaView);
- truth and result JSON files;
- loss-history CSVs.

### Extensions to `run_control.json`

**Numerics**, a top-level block (all keys optional):

```json
"numerics": {
  "scheme": "hr_hllc",            // or "julia_xi_roe" (Hydrograd.jl's scheme, wet domains only)
  "spatial_order": 2,             // 1 is ~6x faster on wet/dry transients but more diffusive
  "limiter": "venkatakrishnan",   // or "barth_jespersen" (sharper, but stalls steady convergence)
  "checkpoints": 64,              // or "auto": as many as fit in checkpoint_memory_gb
  "checkpoint_memory_gb": 2.0
}
```

**`ode_solver`** values:
- `Tsit5()`, `Euler()` and `RK4()`, as in Julia.
- `SSPRK3()`: wet/dry runs. CFL-limited steps, positivity-preserving, point-implicit friction.
- `SteadyNewton()`: steady state by Newton with implicit-function-theorem gradients. Requires the wet extent not to change during the solve.
- `BackwardEuler()` / `CrankNicolson()`: implicit θ-scheme with a fixed step `time_settings.dt` (tens of seconds) and per-step IFT gradients (reverse mode). Best for long, slowly varying, mostly wet events; large steps at fast-advancing wet fronts can fail.

**Time series:** `IQParams` and `EWSParamsC` in `.srhhydro` accept an SMS/SRH-2D `.xys` file (times in hours) in place of a number. The series are differentiable, so hydrographs can be inverted.

**UDE network:** `UDE_NN_config.layernorm` defaults to false. The closure is a local law n_i = f(h_i). `layernorm_axes: "all"` reproduces Julia's non-local LayerNorm.

## Python API (sketch)

```python
import hydrograd
from hydrograd.io.control import read_control
from hydrograd.apps.setup import setup_case
from hydrograd.solvers.ode import solve_swe
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit

case = setup_case(read_control("run_control.json"))
sol = solve_swe(case.model, {}, case.U0, case.ode)                  # explicit
ts, ys = solve_implicit(case.model, {"Q": q}, case.U0,              # implicit backward Euler
                        ImplicitSettings(0.0, 7200.0, 60.0, n_save=24))
# jax.grad of any loss of ys w.r.t. theta ({"n_zone", "zb", "Q", "WSE", "nn"}) works with every solver.
```

## Validation (details in `plan/`)

- **Parity with Hydrograd.jl:**
  - semi-discrete RHS to round-off (1e-15 to 6e-14) on 5 configurations;
  - forward runs to the ODE tolerance;
  - sensitivity, inversion and UDE trajectories to 1e-4 to 4e-3.
- **Exact solutions:**
  - Stoker and Ritter dam breaks: L1 depth error 0.4–0.5 mm at second order;
  - lake at rest with an emerged island: exact;
  - subcritical bump: observed order 1.9.
- **SRH-2D:** Savannah steady WSE within 6.3 mm RMS.
- **Gradients:** checked against finite differences for every solver (explicit, steady IFT, implicit IFT), including across wetting fronts.

## Tests

```bash
pytest tests             # ~10 min on 8 cores (slow tests skipped)
pytest -m "" tests       # all 45 tests (~25 min)
```

The tests need a Hydrograd.jl checkout next to this folder or at `$HYDROGRAD_JL`.
