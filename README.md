# hydrograd (Python/JAX)

Differentiable finite-volume solver for the 2D shallow water equations (SWEs) and Universal SWEs
(SWEs with neural-network flow-resistance closures). This is *Hydrograd* v2.0: it adds a Python/JAX
implementation (CPU and GPU) to, and extends the capabilities of, the first version (v1.0),
[Hydrograd.jl](https://github.com/psu-efd/Hydrograd.jl) (Liu & Song 2025, *Water Resources Research*,
doi:10.1029/2025WR040265).

- **Input:** SRH-2D case files (`.srhhydro`, `.srhgeom`, `.srhmat`) and Hydrograd.jl's `run_control.json`.
- **Modes:** forward simulation, sensitivity analysis, inversion (Manning's n, bed elevation, inflow) and UDE training.
- **Hardware:** CPU and GPU (JAX, float64).

**Documentation:** the user manual is [`docs/manual.md`](docs/manual.md). The examples of the model description paper (Liu, *Hydrograd v2.0*, Geoscientific Model Development, in preparation), with the scripts that reproduce every figure, are in [`paper/`](paper/). Licence: MIT ([`LICENSE`](LICENSE)); citation: [`CITATION.cff`](CITATION.cff).

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

**Vegetation drag:** an optional top-level `vegetation` block adds the drag of rigid emergent stems, ½ C_D a h |u| u per unit plan area (a = d/ΔS², the frontal area per unit volume). It acts as an extra friction rate ½ C_D a |u|, so every solver and adjoint includes it.

```json
"vegetation": {
  "a":  {"2": 1.0},         // 1/m, per material ID (or "d" and "spacing" in m, per material)
  "CD": {"2": 1.0}          // drag coefficient per material ID
}
```

- **Inversion and sensitivity:** `active_param_names: ["veg_CD"]` makes the drag coefficient per material zone the active parameter.
- **UDE:** `UDE_choice: "VegetationCD"` trains a local network C_D = f(h, |u|, a), with `input_dim` 3 and `a_bounds` in `UDE_NN_config`. Manning's n stays per zone for the bed.
- **Force data:** a `force_data` block (`file`, `sigma_N`, `weight`) adds observed drag forces on regions (material zones, boxes or cell lists) to the inversion and UDE losses. Flow fields alone do not identify how resistance is split between neighbouring patches; patch or sub-patch forces do.
- **Order of accuracy:** use second order (the default) for deep, slow vegetated flows. On a sloping bed, first-order hydrostatic reconstruction adds a numerical flux of about √(gh) S dx/2.

**Porous drag (Darcy–Forchheimer):** an optional `porous` block adds (α + β|u|) u h per unit plan area, the porous-media option of SRH-2D and HEC-RAS. Give `alpha` (1/s) and `beta` (1/m) per material ID, or `porosity` and `d` (m), from which Ergun's equation gives them. `active_param_names: ["porous_alpha"]` or `["porous_beta"]` inverts them, and force data include this drag.

**Scattered velocity data (LSPIV):** an optional `velocity_data` block reads scattered velocities from CSV or PIVlab ASCII files. It maps them to model coordinates, converts surface velocities to depth averages (`surface_factor`), averages them over the cells that contain them, and adds the misfit on those cells to the inversion and UDE losses. See `docs/manual.md`, Sect. 7.1.

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

## Backend choice: JAX over PyTorch

PyTorch and JAX were both prototyped (`benchmarks/backend/`) and benchmarked on the same residual, meshes and hardware before the package was written. Full results are in `plan/backend_benchmark_report.md`.

- **Speed:** on an NVIDIA A100 in float64, JAX was faster than `torch.compile` at every mesh size from 2k to 512k cells. Forward runs were 1.2–5× faster and reverse-mode gradients 2–11× faster.
- **Savannah River:** a 10,000-step gradient took 4.5 s with JAX against 57.5 s with PyTorch.
- **Why:** XLA compiles the whole time loop (`lax.scan`) into one program, while PyTorch pays a per-step Python dispatch cost, most of all in the checkpointed backward pass.
- **Solver tools:** JAX's `custom_vjp`, `jvp`/`vmap` and checkpointing support the implicit-function-theorem adjoints and the colored sparse Jacobians. Those are what the steady and implicit solvers are built on.

The package therefore uses JAX only. Peak GPU memory was similar for both backends.

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
