# Hydrograd v2.0 user manual

*Hydrograd* is a differentiable two-dimensional (2D) shallow-water model on unstructured meshes. It runs forward simulations and computes exact gradients of model outputs with respect to model parameters, for calibration (inversion), sensitivity analysis and the training of data-driven closures (universal shallow-water equations, USWEs). This manual explains how to install the model, set up and run a case, and choose the numerical options. The model, its numerics and its verification are described in the accompanying paper (Liu, *Geoscientific Model Development*, in preparation); the examples of the paper are in `paper/`.

Contents:
1. Installation
2. Quick start
3. A case folder
4. The run-control file
5. Numerical options
6. Physical options (closures)
7. Inversion, sensitivity analysis and closure learning
8. Outputs
9. Python interface
10. Choosing a solver and a gradient method
11. Tests
12. Troubleshooting

---

## 1. Installation

Requirements: Python 3.11 or newer on Linux (tested), or any platform supported by JAX; double precision is used throughout. An NVIDIA GPU is optional.

```bash
git clone https://github.com/psu-efd/Hydrograd.git
cd Hydrograd
pip install -e ".[dev]"            # CPU (JAX, diffrax, optax, NumPy, SciPy; pytest and matplotlib)
pip install -e ".[dev,cuda]"       # with JAX for CUDA 12 GPUs
pip install -e ".[pardiso]"        # optional: multithreaded sparse direct solver (Intel MKL PARDISO)
```

`uv pip install ...` works the same way and is faster. The figure scripts of the paper also need a LaTeX installation (matplotlib renders the labels with LaTeX).

Main dependencies: JAX (automatic differentiation and compilation for CPU and GPU), diffrax (adaptive time integration), Optax (optimisers), NumPy and SciPy (sparse linear algebra). The optional `pypardiso` package speeds up the sparse factorisations of the steady and implicit solvers on large meshes (above about 30,000 unknowns).

## 2. Quick start

A case is a folder with the SRH-2D input files and a `run_control.json`. Inside the folder:

```bash
hydrograd run
```

The run mode (forward simulation, inversion, sensitivity analysis or closure learning) is selected in `run_control.json`. Examples:
- `paper/` holds the scripts of all paper examples;
- the cases of the first version, Hydrograd.jl (`examples/SWE_2D/...` in https://github.com/psu-efd/Hydrograd.jl), run unchanged.

**Copy example folders before running them.** A forward run writes `forward_simulation_solution_truth.json`, which would overwrite a truth file of the example.

## 3. A case folder

| File | Content |
|---|---|
| `<case>.srhhydro` | SRH-2D hydraulic control file: boundary conditions (inlet discharge `IQParams`, exit water-surface elevation `EWSParamsC`), Manning's n per material, time settings |
| `<case>.srhgeom` | SRH-2D mesh: nodes, cells (triangles, quadrilaterals or general polygons), boundary node strings |
| `<case>.srhmat` | SRH-2D materials: the cells of each material (roughness zone) |
| `run_control.json` | run mode and options (Sect. 4) |
| optional | initial condition JSON, truth JSON (observations), `.xys` time series, force data |

The SRH-2D files are produced by SMS (Surface-water Modeling System) or by pyHMT2D. Units follow the SRH-2D `GridUnit` (SI or US customary).

## 4. The run-control file

The file has the same layout and keys as in Hydrograd.jl:

```json
{
  "control_variables": {
    "srhhydro_file_name": "savana_SI.srhhydro",
    "bPerform_Forward_Simulation": false,
    "bPerform_Inversion": true,
    "bPerform_Sensitivity_Analysis": false,
    "bPerform_UDE": false
  },
  "time_settings": {
    "bUse_srhhydro_time_settings": false,
    "tspan": [0.0, 200.0],
    "dt": 0.02
  },
  "inversion_options": { "...": "options of the active mode" }
}
```

- **Run mode:** exactly one `bPerform_*` flag is true.
- **Time settings:** `tspan` (s) and the initial (explicit) or fixed (implicit) time step `dt` (s).
- **Option block of the active mode:** `forward_simulation_options`, `inversion_options`, `sensitivity_analysis_options` or `UDE_options`. Its keys are those of Hydrograd.jl:
  - the ODE solver and its options;
  - the parameters to invert;
  - the loss terms;
  - the optimisers and learning rates;
  - the NN configuration;
  - file names.

  Copy a Hydrograd.jl example and edit it.

Hydrograd v2.0 adds optional **top-level blocks**: `numerics`, `turbulence`, `wall_friction`, `vegetation`, `porous`, `dispersion`, `force_data` and `velocity_data` (Sects. 5–7). It also adds **solver names**: `SSPRK3()`, `SteadyNewton()`, `BackwardEuler()`, `CrankNicolson()`. A file without these entries runs with the defaults.

## 5. Numerical options

### 5.1 Spatial scheme (`numerics` block)

```json
"numerics": {
  "scheme": "hr_hllc",
  "spatial_order": 2,
  "limiter": "venkatakrishnan",
  "checkpoints": 64,
  "checkpoint_memory_gb": 2.0
}
```

| Key | Values | Default | Notes |
|---|---|---|---|
| `scheme` | `hr_hllc`, `julia_xi_roe` | `hr_hllc` | `hr_hllc`: hydrostatic reconstruction with the HLLC flux; wet/dry capable, well balanced, positivity preserving. `julia_xi_roe`: the Roe scheme of Hydrograd.jl, wet domains only, kept for comparison. All closures below need `hr_hllc`. |
| `spatial_order` | 1, 2 | 2 | Second order (MUSCL with a limiter) is more accurate; first order is about 6 times cheaper and more diffusive. |
| `limiter` | `venkatakrishnan`, `barth_jespersen` | `venkatakrishnan` | Venkatakrishnan is smooth and converges better in steady solves. |
| `checkpoints` | integer or `"auto"` | 64 | Number of stored states for reverse-mode gradients through explicit time stepping; `"auto"` uses as many as fit in `checkpoint_memory_gb`. |

### 5.2 Solvers (`ode_solver` in the option block of the mode)

| Solver | Use | Gradient |
|---|---|---|
| `SteadyNewton()` | steady flows; Newton with pseudo-transient continuation, exact sparse Jacobian | implicit function theorem: one linear solve, independent of the number of parameters |
| `SSPRK3()` | unsteady flows with wetting and drying; CFL-limited explicit steps, point-implicit friction | reverse-mode AD with checkpointing (about 4–8 forward runs) |
| `BackwardEuler()`, `CrankNicolson()` | long, slowly varying unsteady events; fixed step `dt` | one adjoint linear solve per step (about 1.5 forward runs); reverse mode only |
| `Tsit5()`, `Euler()`, `RK4()` | general adaptive or fixed-step integration (as in Hydrograd.jl) | reverse or forward mode; use a tight tolerance for gradients (the sensitivity, inversion and UDE modes use `rtol = 1e-7`) |

Section 10 gives guidance on the choice.

### 5.3 Time series

`IQParams` (inlet discharge) and `EWSParamsC` (exit water-surface elevation) in the `.srhhydro` file accept an SMS/SRH-2D `.xys` file in place of a number. Times are in hours, interpolated linearly, and held constant beyond the ends. The series are differentiable, so hydrographs can be inverted.

## 6. Physical options (closures)

All closures are differentiable and included in every solver and gradient.

### 6.1 Bed friction

Manning's n per material is the default (from the `.srhhydro` file). Hydrograd.jl's variable-n options are available in the forward options (`forward_simulation_ManningN_option`, `..._function_type`, `..._function_parameters`):
- n(h) laws;
- n(h, |U|, k_s).

Neural-network closures are selected with `UDE_choice` (Sect. 7.3).

### 6.2 Turbulent diffusion (`turbulence` block)

```json
"turbulence": {"model": "parabolic", "coefficient": 0.7}
"turbulence": {"model": "constant", "nu_t": 0.01}
"turbulence": {"model": "mixing_length", "coefficient": 0.0, "mixing_coefficient": 0.1}
"turbulence": {"model": "k_epsilon"}
```

- `parabolic`: ν_t = C_t u_* h + ν_0 (as in SRH-2D); `coefficient` is C_t; differentiable as `eddy_Ct`.
- `mixing_length`: adds the shear-generated part (C_m h)² |S|, with |S| the strain rate of the depth-averaged velocity. `mixing_coefficient` is C_m (differentiable as `eddy_Cm`).
- `k_epsilon`: the depth-averaged k–ε model (Rastogi and Rodi 1978) with the standard constants; the keys `c_mu`, `c_1`, `c_2`, `sigma_k`, `sigma_e` change them. The state gains the rows h k and h ε, initialised at the local equilibrium (`initial_speed` sets the speed used when the initial flow is at rest).

The explicit time step accounts for the diffusion limit with the mixing-length and k–ε models.

### 6.3 Side walls (`wall_friction` block)

```json
"wall_friction": {"model": "loglaw", "ks": 0.0, "nu": 1e-6}
```

Without this block, wall boundaries are free-slip. With it, each wall face exerts the shear stress of a log-law boundary layer: `ks` is the wall roughness in m, 0 for a smooth wall, differentiable as `wall_ks`. With k–ε, wall functions add the wall production of k and relax ε to its log-layer value in the wall cells.

### 6.4 Vegetation drag (`vegetation` block)

```json
"vegetation": {"a": {"2": 1.0}, "CD": {"2": 1.0}}
```

Rigid emergent stems exert the drag ½ C_D a h |u| u per unit plan area.
- `a` is the frontal area per unit volume (1/m) per material ID. Alternatively give `d` (stem diameter, m) and `spacing` (m) per material, from which a = d/spacing².
- C_D is applied to the 2D cell velocity, so it must be defined with the velocity averaged over the vegetated area, not the approach velocity of an isolated stem.

### 6.5 Porous drag, Darcy–Forchheimer (`porous` block)

```json
"porous": {"alpha": {"3": 0.02}, "beta": {"3": 4.0}}
"porous": {"porosity": {"3": 0.9}, "d": {"3": 0.01}, "viscosity": 1e-6}
```

A porous zone (dense vegetation treated as a porous medium, a log jam, a rock or brush structure) exerts the drag (α + β|u|) u h per unit plan area, where u is the depth-averaged (superficial) velocity.
- `alpha` (1/s) is the linear (Darcy) coefficient and `beta` (1/m) the quadratic (Forchheimer) coefficient, per material ID.
- Alternatively, give the porosity ε and the grain or stem diameter `d` (m); Ergun's equation then gives α = 150 ν (1−ε)²/(ε³ d²) and β = 1.75 (1−ε)/(ε³ d). Explicit `alpha` or `beta` entries override the Ergun values of the materials they list.
- Materials that are not listed have no porous drag. The block can be combined with `vegetation`; the two drags add.
- Both coefficients are differentiable (`df_alpha`, `df_beta`) and can be inverted (Sect. 7.1).

### 6.6 Dispersive momentum flux (`dispersion` block)

```json
"dispersion": {"model": "zone", "beta": {"2": 1.3}}
```

`beta` is the momentum coefficient per material. It is the flux added for velocity variations within a cell, e.g. between stems; differentiable as `disp_beta`.

## 7. Inversion, sensitivity analysis and closure learning

### 7.1 Inversion

In `inversion_options`, `active_param_names` selects the parameters:
- `ManningN`: n per zone;
- `zb`: bed elevation per cell;
- `Q`: inlet discharge;
- `veg_CD`: drag coefficient per zone;
- `porous_alpha`, `porous_beta`: Darcy–Forchheimer coefficients per zone.

The loss combines normalised misfits of the water-surface elevation (WSE) and the velocities against the truth file, plus a bound penalty. Optional force data are added with the `force_data` block:

```json
"force_data": {"file": "forces.json", "sigma_N": 0.05, "weight": 1.0}
```

It gives observed drag forces on regions: material zones, boxes or cell lists. The predicted force is the drag of the vegetation and the porous zones in the region.

Scattered velocities, such as LSPIV or PIV surface velocities, ADV points or CFD probes, are added with the `velocity_data` block:

```json
"velocity_data": {
    "file": "lspiv.csv",
    "transform": {"origin": [0.0, 0.0], "rotation_deg": 0.0, "scale": 1.0,
                  "velocity_scale": 1.0, "flip_y": false},
    "surface_factor": 0.85, "sigma_point": 0.02, "sigma_floor": 0.005,
    "min_points": 1, "components": ["u", "v"], "weight": 1.0,
    "binned_file": "lspiv_cells.csv"
}
```

- **File:** a text file with columns `x`, `y`, `u`, `v`, separated by commas, semicolons, tabs or spaces. Units in brackets are ignored, so the ASCII export of PIVlab is read directly. Other column names can be mapped with `"columns": {"u": "vx", "v": "vy"}`, and rows with NaN velocities (masked vectors) are dropped.
- **Coordinates:** `transform` maps the measurement frame to the model frame. `flip_y` handles image rows that point down, then the axes are rotated by `rotation_deg`, scaled by `scale` (e.g. m per pixel) and shifted by `origin`. Velocities are rotated in the same way and multiplied by `velocity_scale`.
- **Depth averaging:** surface velocities are converted to depth averages with the velocity index `surface_factor`. About 0.85 suits a logarithmic profile; use 1 for data that are already depth averages. It is best measured with a few velocity profiles in the same flow.
- **Cells:** points are averaged over the mesh cells that contain them, and points outside the mesh are dropped. A cell with N points has the uncertainty √(σ_point²/N + σ_floor²), and cells with fewer than `min_points` points are not used. `binned_file` writes the cell observations as CSV, so the mapping can be checked.
- **Loss:** the term is `weight` × the mean of ((U_pred − U_obs)/σ)² over the observed cells and components. It works in inversion and in UDE training.

Optimisers are run in sequence: for example Adam, then L-BFGS (`inversion_optimizers`, `inversion_learning_rates`, `inversion_max_iterations`). Positive parameters are optimised in log form where appropriate.

### 7.2 Sensitivity analysis

Computes the derivatives of the solution with respect to the active parameters:
- forward-mode AD for unsteady runs;
- one sparse factorisation for steady runs.

### 7.3 Closure learning (UDE)

`UDE_choice` selects the closure:
- `ManningN_h`: n = NN(h);
- `ManningN_h_Umag_ks`;
- `FlowResistance`: the NN predicts the bed shear stress from h and |q|;
- `VegetationCD`: C_D = NN(h, |u|, a).

`UDE_NN_config` gives the layer sizes, activations, input and output bounds. The closure is a local law: the default network has no normalisation layer (`layernorm: false`). With `ode_solver: "SteadyNewton()"`, training uses steady states and implicit-function-theorem gradients, which is much faster than training through time marching (paper, Sect. 4).

## 8. Outputs

Hydrograd.jl's file names are kept:
- `forward_simulation_results_%04d.vtk`: cell fields for ParaView. These are WSE, depth, velocities, unit discharges, bed and Manning's n; with k–ε also k, ε and ν_t. Files are binary above 20,000 cells.
- `forward_simulation_solution_truth.json`: the final state, usable as observations.
- `inversion_results_losses_and_ODE_solution.json`, `UDE_results_*`, and the loss and parameter histories as CSV.
- Trained network weights as `.npz`.

## 9. Python interface

All modes can be scripted. A minimal steady calibration:

```python
import jax, jax.numpy as jnp
from hydrograd.io.control import read_control
from hydrograd.apps.setup import setup_case
from hydrograd.solvers.final import final_state

case = setup_case(read_control("my_case/run_control.json"))   # ode_solver = SteadyNewton()
model = case.model

def loss(log_n):
    """Mean squared WSE misfit (m^2) at the steady state for zone Manning's n = exp(log_n)."""
    U, _ = final_state(model, {"n_zone": jnp.exp(log_n)}, case.U0, case.ode)
    wse = U[0] + model.wstill
    return jnp.mean((wse - observed_wse) ** 2)

value, gradient = jax.value_and_grad(loss)(jnp.log(jnp.asarray(model.n_zone)))
```

- **Parameter names in `theta`:**
  - `n_zone`, `zb`, `Q`, `WSE`;
  - `nn` (network weights), `veg_CD`, `veg_struct`;
  - `eddy_Ct`, `eddy_Cm`, `wall_ks`, `disp_beta`.
- **Batching:** `jax.vmap` runs ensembles of simulations, e.g. for uncertainty analysis, efficiently on a GPU.

## 10. Choosing a solver and a gradient method

| Problem | Solver | Why |
|---|---|---|
| Steady flow (calibration to steady observations, closure learning) | `SteadyNewton()` | the exact steady state in 5–20 Newton iterations; a gradient costs one linear solve; independent of the path to steady state |
| Unsteady, short events, wetting and drying, many parameters | `SSPRK3()` with checkpoints | robust and positivity preserving; memory independent of the length of the run; GPU efficient |
| Unsteady, long and slowly varying events, very small cells | `BackwardEuler()` with a fixed `dt` | steps far above the CFL limit; gradient about 1.5 forward runs; smooth loss. Large steps at fast-advancing wet fronts can fail. |
| Few parameters, many outputs (sensitivity) | forward mode | one pass per parameter |

General advice:
- Run transient forward simulations and gradients on a GPU for meshes above about 50,000 cells.
- Steady and implicit solves are limited by the sparse factorisation on the CPU; install `pypardiso` for large meshes.
- Use log-parameters for n and C_D.
- Calibration losses can be non-convex: use several starting points and place observations in every roughness zone.

## 11. Tests

```bash
pytest tests              # about 12 minutes on 8 CPU cores
pytest -m "" tests        # including the slow tests
```

The tests check:
- conservation and well-balancedness;
- analytical solutions (dam breaks, lake at rest, bump);
- steady, implicit and explicit gradients against finite differences;
- the sparse Jacobians against dense ones;
- every closure;
- agreement with Hydrograd.jl. These tests need Hydrograd.jl reference data and are skipped without them.

## 12. Troubleshooting

- **Steady solve does not converge:**
  - start from a better initial condition (e.g. a short explicit run);
  - reduce the initial pseudo-time step;
  - steady states with a moving wet front or an oscillating hydraulic jump may not exist; use an unsteady solver.
- **Gradients differ from finite differences:**
  - tighten the integration tolerance;
  - check that the steady residual is near machine precision;
  - bed-elevation gradients on a perfectly flat bed are one-sided (subgradients).
- **Out of memory in reverse mode:** reduce `checkpoints`, or set `"checkpoints": "auto"` with a memory budget.
- **Implicit steps fail at wet fronts:** reduce `dt`, or use `SSPRK3()` for the wetting phase.
