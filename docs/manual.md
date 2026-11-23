# Hydrograd v2.0 user manual

*Hydrograd* is a differentiable two-dimensional (2D) shallow-water model on unstructured meshes; since v2.0.1 it also has a one-dimensional (1-D) mode for flow and sediment transport on networks of cross-sections (Sect. 13). Sections 2 to 12 describe the 2D model. It runs forward simulations and computes exact gradients of model outputs with respect to model parameters, for calibration (inversion), sensitivity analysis and the training of data-driven closures (universal shallow-water equations, USWEs). This manual explains how to install the model, set up and run a case, and choose the numerical options. The model, its numerics and its verification are described in the accompanying paper (Liu, 2026, [arXiv:2611.11540](https://arxiv.org/abs/2611.11540)); the examples of the paper are in `paper/`.

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
13. The 1-D mode: flow and sediment transport on cross-sections

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

Dry cells in steady gradients: a dry cell (h ≤ 0 with `hr_hllc`) has no effect on the residual, so `SteadyNewton()` gradients hold the dry set fixed (zero sensitivity in dry cells). The gradient is exact as long as a small parameter change wets or dries no cell; it does not see a wet/dry front move.

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
pytest tests              # about 25 minutes on 8 CPU cores
pytest -m "" tests        # including the slow tests
```

The tests check:
- conservation and well-balancedness;
- analytical solutions (dam breaks, lake at rest, bump);
- steady, implicit and explicit gradients against finite differences;
- the sparse Jacobians against dense ones;
- every closure;
- agreement with Hydrograd.jl. These tests need Hydrograd.jl reference data and are skipped without them;
- the 1-D mode (Sect. 13.10), including the comparison with SRH-1D on two cases.

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

## 13. The 1-D mode: flow and sediment transport on cross-sections

The 1-D mode simulates flow, sediment transport and bed change on a network of cross-sections, for example the sedimentation of a reservoir and of the rivers that enter it. It follows the USBR model SRH-1D (as the 2-D solver follows SRH-2D) and is differentiable in the same way: the gradient of any output with respect to the sediment and flow parameters is available for sensitivity analysis and calibration. All quantities are in SI units.

Examples are in `examples/oned/` (a reservoir delta, a twin inversion and a confluence), with a README.

### 13.1 What the model solves

One morphological step (typically a day) has two parts:

1. **Hydraulics.** One of two flow solvers (below): by default a quasi-steady backwater profile, the standard-step energy equation between each cross-section and its downstream neighbour, with Manning friction, for the discharge and the outlet water level of the step. Where the energy equation has no subcritical solution (a crest above the pool), the section is a control at critical depth. The system is solved by Newton iteration. The friction slope is S_f = Q²/K², with the conveyance K of the cross-section (below).
2. **Sediment and bed.** For each size class, the total load relaxes along each reach towards the transport capacity (non-cohesive classes; Engelund–Hansen for sand, or Wilcock–Crowe for sand–gravel mixtures, with an active-layer fraction and a hiding factor) or exchanges with the bed by settling and erosion (cohesive classes; Krone and Partheniades). The deposit of each reach raises the bed of its downstream cross-section (Exner equation), and the composition of the bed is tracked in an active layer over a substrate (Hirano).

The sediment closures and the bed-composition model are independent of the 1-D geometry (`hydrograd.sediment`); the same code is meant to serve 2-D sediment transport. The equations are given in the docstrings of the modules:

| Module | Content |
|---|---|
| `hydrograd.oned.network` | tree-shaped networks: numbering, discharge accumulation, bookkeeping of deposits |
| `hydrograd.oned.geometry` | cross-section properties from station–elevation tables, critical depth |
| `hydrograd.oned.backwater` | backwater solver and its implicit-function-theorem gradient |
| `hydrograd.oned.unsteady` | unsteady shallow-water solver for sub- and supercritical flow |
| `hydrograd.oned.routing` | non-equilibrium routing with the locally implicit bed feedback |
| `hydrograd.oned.bed`, `hydrograd.sediment.bed` | Exner equation, active layer, substrate, non-erodible floor |
| `hydrograd.sediment.closures`, `.exchange` | settling, capacity, cohesive fluxes, adaptation lengths, supply limit |
| `hydrograd.sediment.observation` | deposit volumes with consolidation (Lane and Koelzer) |
| `hydrograd.oned.stepper` | the time loop |
| `hydrograd.io.oned_case`, `hydrograd.io.srh1d` | the case file; reader of the bed profiles written by SRH-1D |

**Conveyance.** Two formulas are available (`"hydraulics": {"conveyance": ...}` in the case file):

- **`"segments"`** (default): K = (1/n) Σ A_k^(5/3) / P_k^(2/3), the sum over the slices between neighbouring station points (area A_k, wetted bed length P_k). It rises with the water surface for every section shape. Vertical walls add no friction: a rectangle of width b and depth h has K = b h^(5/3) / n.
- **`"section"`**: K = A R^(2/3) / n with the hydraulic radius R = A/P of the whole section. Where the water floods a bench or a floodplain, the wetted perimeter jumps and K can fall with rising water surface; the energy equation then has several roots, and the Newton iteration may stop at its limit without having converged. Use it only for sections without benches, for example a narrow rectangular flume, where the wall friction matters.

The two differ for every section shape: `"segments"` gives the larger K, by 4 % for a trapezoid of 80 m bottom width with 2:1 banks at 5 m depth. The bed shear stress is τ_b = ρ g R S_f. By default R is the hydraulic radius of the whole section, for both conveyance formulas (`"hydraulics": {"shear_radius": "section"}`). With `"shear_radius": "depth"` it is the mean depth A/B, which puts the whole friction force on the bed. In a wide section the two are nearly equal. In a narrow channel with smooth vertical walls, such as a flume, the hydraulic radius gives a bed shear stress that is too small (half, for a depth of half the width), while the `"segments"` conveyance has no wall friction; use `"depth"` there.

Manning's formula is one of two resistance laws (`"hydraulics": {"resistance": ...}`). Both give the mean velocity as U = C √(g R S_f) with a dimensionless coefficient C = U/u_*, and the conveyance of a segment or of the section as K = A √(g R) C:

- `"manning"` (default): C = R^(1/6) / (n √g), with Manning's n (`n_manning`).
- `"variable_power"`: C = a₁ a₂ x / √(a₁² + a₂² x^(5/3)) with x = R/k, a₁ = 6.5 and a₂ = 2.5 (Ferguson, 2007), and a roughness length k in metres (`roughness_length`) in the place of n. For deep flow it is Manning's formula with n = k^(1/6) / (a₁ √g); as the depth falls towards k the resistance rises faster (C is 13 % below the deep-flow value at x = 6 and 37 % below it at x = 2.5), as observed in shallow flow over gravel.

- `"keulegan"`: the logarithmic law C = 2.5 ln(11 x) with x = R/k_s and the roughness height k_s (the same parameter, `roughness_length`). The law turns negative below x = 1/11; below x₀ = e/11 = 0.247, where C = 2.5, it is continued by its tangent through the origin, C = 10.1 x, so that C stays positive and smooth down to zero depth.

The roughness length of the last two laws is a given length (`"roughness_scale": "length"`, default; parameter `roughness_length`) or follows the bed surface (`"roughness_scale": "surface_mean"`): k = `roughness_factor` × D_sg, with D_sg the geometric mean grain size of the active layer of the section, taken at the start of every step. Cui (2007) uses the logarithmic law with the factor 2.5. With `"roughness_scale": "surface_d90"` the D90 of the active layer takes the place of D_sg (Parker, 1990, uses the factor 2): the mean size falls when sand covers the bed, while the D90 follows the coarse grains that make the roughness. The D90 is found by linear interpolation in ln d between the class limits, which are the geometric means of neighbouring class diameters.

With these two laws the shear stress handed to the sediment closures can be reduced for the drag on the largest grains, which does not move sediment (`"transport_stress": "reduced"`; default `"total"`): τ_b (C / C_deep)^1.5, where C_deep is the law of a deep flow over the same roughness. For `"variable_power"` it is a₁ x^(1/6), which gives τ_b [1 + (a₁/a₂)² x^(−5/3)]^(−3/4): 0.81 at x = 6 and 0.50 at x = 2.5. For `"keulegan"` it is the power law 8.1 x^(1/6) (Manning-Strickler), which the logarithmic law follows within 2 % for x of 9 to 200: the factor is 1 for x above 14 and 0.92, 0.82 and 0.64 at x = 5, 2.5 and 1. This is the reduction of Rickenmann and Recking (2011) for steep, shallow streams with the exponent of Meyer-Peter and Müller (1948). The friction slope and the water surface always use the total resistance.

**Flow solver.** Two solvers are available (`"hydraulics": {"flow_solver": ...}` in the case file). Both use the same conveyance and the same Manning's n.

- **`"backwater"`** (default): the quasi-steady energy equation described above. The flow is subcritical; a section where it would be supercritical is held at critical depth. The step can be long (a day), which suits reservoirs and lowland rivers over decades. On a steep reach the depth is then too large and the bed shear stress too small, and the stress no longer responds to the bed slope.
- **`"unsteady"`**: the unsteady shallow-water (Saint-Venant) equations for the flow area A and the discharge Q,

  ∂A/∂t + ∂Q/∂x = 0,  ∂Q/∂t + ∂(βQ²/A)/∂x + g A ∂η/∂x + g A S_f = 0,

  with the water surface η and the momentum coefficient β. The pressure term is evaluated from the first moment of the flow area of each cross-section, so that a level water surface is at rest over any sequence of sections and a hydraulic jump conserves the specific force. The scheme is staggered (areas at the sections, velocities on the reaches) with upwind momentum advection (Stelling and Duinmeijer, 2003) and explicit in time. It handles subcritical and supercritical flow and the jumps between them, and at steady state the discharge is the same on every reach. Each step of the time loop is divided into `numerics.unsteady.n_substeps` flow steps, which must keep the Courant number (|u| + c) Δt/Δx below 1 (c = √(gA/B); `hydrograd.oned.unsteady.courant_number` returns it). Use it for steep channels and flumes; a reservoir over decades needs tens of millions of flow steps. It is limited to single channels without lateral withdrawals.

  Boundary conditions of the unsteady solver: the inflow at the headwater and the water level at the outlet, as for the backwater solver. A level below the critical depth of the outlet is replaced by the critical depth (a free overfall), and a supercritical flow that arrives with more specific force than the tailwater leaves freely. Where the flow leaves the headwater supercritically, `numerics.unsteady.headwater` sets the second condition: `"uniform"` (default; no acceleration over the first reach) or `"critical"` (the headwater is a critical-flow control). The run starts from the backwater profile of the first step.

  In subcritical flow the two solvers give the same profile up to the discretization (8 mm at a section spacing of 500 m in a channel 3.5 to 8 m deep, halved at 250 m).

### 13.2 Networks

A network is a tree: tributaries join, and one outlet (the dam) closes the system. With N cross-sections there are N − 1 reaches. Sections are numbered so that reach i runs from section i to a downstream section `downstream[i]` with a larger number, and the outlet is the last section. A single channel has `downstream = [1, 2, ..., N-1]`.

- **Headwaters** are the sections no reach flows into. Discharge and inflow load are prescribed there.
- **Junctions:** a junction is one cross-section with several incoming reaches. Its discharge is the sum of theirs, the incoming loads add, and its capacity and bed supply are shared among the incoming reaches in proportion to their discharge. (SRH-1D repeats the junction cross-section once per river, with independent bed elevations; compare deposited volumes, not bed elevations, at a junction.)
- **Headwater sections** receive a virtual inflow reach (half of their own outgoing reach by default, `head_inflow_fraction`), in which the prescribed load relaxes towards the capacity of the section; otherwise their bed could never change.

### 13.3 The case file

A 1-D case is one JSON file. `run_control.json` selects the 1-D mode and names it:

```json
{
  "control_variables": {
    "model": "1D",
    "case_file_name": "case.json",
    "bPerform_Forward_Simulation": true
  },
  "forward_simulation_options": {}
}
```

Without `"model"`, a run control describes a 2-D case. `"time_settings"` is not used: the steps are part of the case file.

Blocks of the case file (N sections, J station points per section, K size classes, T steps, H headwaters; arrays are nested lists, row by row):

| Block | Entries |
|---|---|
| `"format"`, `"version"` | `"hydrograd-1d-case"`, `1` |
| `"sediment"` | `diameters` [K] (m, finest first); `cohesive_dmax` (m; finer classes are cohesive, default 6.25e-5); `transport_formula` (`"engelund_hansen"`, or `"wilcock_crowe"` for sand–gravel mixtures); `hiding_form` (`"power"`; `"wilcock_crowe"` is the hiding function of that formula; `"network"` is a neural network); `krone_form` (`"ramp"`) |
| `"hydraulics"` | `conveyance`: `"segments"` (default) or `"section"`; `shear_radius`: `"section"` (default) or `"depth"`; `resistance`: `"manning"` (default), `"variable_power"` or `"keulegan"`; `roughness_scale`: `"length"` (default), `"surface_mean"` or `"surface_d90"`; `transport_stress`: `"total"` (default) or `"reduced"`; `flow_solver`: `"backwater"` (default) or `"unsteady"` (Sect. 13.1); the block may be absent |
| `"network"` | `reach_length` [N−1] (m); `downstream` [N−1] (0-based section numbers) |
| `"sections"` | `s` [N][J] stations (m), not decreasing along a section; `z` [N][J] initial bed elevations (m). All sections have the same J: repeat the last point of shorter tables. |
| `"bed"` | `p_active` [K][N] and `p_substrate` [K][N] (fractions); `substrate_thickness` [N] (m); `z_floor` [N] (m; non-erodible floor) or `null` |
| `"forcing"` | `Q_in` [T] or [T][H] (m³/s); `eta_outlet` [T] (m); `dt` [T] (s); `load_in` [K][T] or [K][T][H] (m³/s of solids), or `null` for the rating curve |
| `"parameters"` | parameter values by name (below); missing ones take their defaults |
| `"numerics"` | `morph`, `backwater`, `unsteady`, `geometry_smoothing`, `sediment_smoothing`; missing settings take their defaults |

Each step has its own length `dt`, so a long record can also be given as a few flow classes of different duration. A misspelled parameter or setting name, and an array of the wrong shape, are reported as errors. The case files of the examples are written by `examples/oned/build_cases.py`, which shows how to build a case from Python (`hydrograd.io.oned_case.write_case`).

### 13.4 Parameters

| Name | Default | Units | Meaning |
|---|---|---|---|
| `n_manning` | 0.03 | s/m^(1/3) | Manning's n: one value, or one per section |
| `beta_momentum` | (absent: 1) | – | momentum coefficient β of the unsteady flow solver: one value, or one per section; optional |
| `roughness_length` | (absent) | m | roughness length k of the resistance laws `"variable_power"` and `"keulegan"`: one value, or one per section; required for them with `roughness_scale` `"length"` |
| `roughness_factor` | (absent) | – | roughness length over the geometric mean size (or the D90) of the active layer; required with `roughness_scale` `"surface_mean"` (or `"surface_d90"`) |
| `a_eh`, `b_eh` | 0.05, 2.5 | – | coefficient and Shields exponent of Engelund–Hansen |
| `a_wc`, `tau_r_wc` | 1, 1 | – | multipliers of the transport function and of the reference Shields stress of Wilcock–Crowe (only with that formula) |
| `m_h` | 0 | – | hiding exponent, ξ_k = (d_k/d_m)^(−m_h); 0 is no hiding (SRH-1D), −1 is equal mobility of all sizes |
| `kappa_w` [K] | 1 | – | multiplier of the settling velocity (Ferguson and Church) of each class |
| `alpha_d`, `alpha_e` | 0.25, 1 | – | recovery coefficients of the suspended load in deposition and erosion |
| `c_b` | 7 | – | bed-load adaptation length L_b = c_b h |
| `M_e` | 1e-8 | m/s | cohesive erosion rate (volume of solids) |
| `tau_ce`, `tau_cd` | 0.5, 0.1 | Pa | critical shear stresses for cohesive erosion and for full deposition |
| `delta_a` | 0.05 | m | active-layer thickness |
| `gamma1` [K] | 800, 1500, … | kg/m³ | initial dry bulk density of the deposits (first class 800) |
| `K_cons` [K] | 250, 1, … | kg/m³ | consolidation coefficient per log10(year) |
| `a_r`, `b_r` | 1e-6, 2 | – | rating curve Q_t = a_r Q^b_r (m³/s of solids), used when `load_in` is `null` |
| `p_in` [K] | 1/K | – | size fractions of the rating-curve load |

### 13.5 Numerical settings

The defaults of `"numerics"` are the validated scheme; most cases need none of them.

- **`morph.hydro_every`** (1): solve the hydraulics every this many steps.
- **`morph.n_substeps`** (1): sediment and bed sub-steps per step, with the hydraulics frozen.
- **`morph.implicit_coupling`** (`"diagonal"`): the locally implicit bed feedback per size class; `"full"` couples the classes through the active-layer composition.
- **`morph.morph_factor`** (1): morphological acceleration.
- **`morph.checkpoint`** (`"auto"`): reverse-mode gradients recompute segments of about √T steps instead of storing all steps; an integer sets the segment length, `"none"` stores everything.
- **`morph.record_every`**, **`morph.snapshot_steps`**: thinning of the recorded histories; steps after which the whole bed table is stored.
- **`backwater.newton_tol`** (1e-9 m), **`backwater.newton_max_iter`** (40).
- **`unsteady.n_substeps`** (1): flow steps per step with the unsteady flow solver; **`unsteady.headwater`** (`"uniform"`): condition at a supercritical headwater (Sect. 13.1); **`unsteady.checkpoint_substeps`** (true): reverse-mode gradients recompute the flow steps of a step instead of storing them. The unsteady solver needs `morph.hydro_every` = 1.
- The options `deposition_scheme`, `implicit_exner`, `composition_update`, `attached_sweeps` and `lag_switch` exist to study the stability of the adjoint (derivatives that grow with the length of a run) and should otherwise keep their defaults.

**Time step.** The scheme is stable at daily steps, but its accuracy at a given step depends on the case: where a reach incises or aggrades quickly, the bed can differ from the time-converged solution by decimetres to metres. Repeat a run with half the step (or `n_substeps: 2`) and compare before relying on it.

### 13.6 Forward simulation

Options (`forward_simulation_options`, all optional): `survey_times` (s; default: the end of the run), `survey_consolidation` (true), `survey_noise_relative` (0) and `survey_noise_seed`, and the two file names below.

| File | Content |
|---|---|
| `oned_forward_results.json` | final thalweg and water surface, bulk deposit volume per section, final active-layer fractions, inflow and outflow loads, trap efficiency |
| `oned_forward_results.npz` | all arrays: bed tables, deposition per step, class and section, histories of thalweg, water surface and loads, bed snapshots |
| `oned_survey_truth.json` | the surveys: bulk deposit volume per section at each survey time |

A survey is the bulk volume of sediment deposited in the control volume of each cross-section since the start of the run, with the consolidation of the deposits up to the survey if requested. It is the observation of the sensitivity analysis and of the inversion.

### 13.7 Sensitivity analysis

```json
"sensitivity_analysis_options": {
  "active_param_names": ["a_eh", "kappa_w[0]", "a_r"],
  "sensitivity_parameter_values": {"a_eh": 0.04}
}
```

A label names a whole parameter (`"a_eh"`; a vector parameter then contributes one column per entry) or one entry of a vector parameter (`"kappa_w[0]"`). `sensitivity_results.json` holds the Jacobian of the surveyed volumes, `jacobian` [S][N][P] (m³ per unit of the parameter), and `jacobian_log`, the same per unit of the logarithm of the parameter, whose columns can be compared between parameters. The Jacobian is computed in forward mode, one pass per active scalar.

### 13.8 Inversion

The inversion minimizes the misfit of the surveyed volumes, Σ (V − V_obs)² / (2σ²), plus optional Gaussian priors, with reverse-mode gradients through the whole run.

```json
"inversion_options": {
  "active_param_names": ["a_eh", "kappa_w[0]", "a_r"],
  "inversion_parameter_initial_values": {"a_eh": 0.03},
  "inversion_truth_file_name": "oned_survey_truth.json",
  "inversion_survey_relative_sigma": 0.05,
  "inversion_parameter_prior_sd": {"a_eh": 0.7},
  "inversion_optimizers": ["Adam", "Adam"],
  "inversion_learning_rates": [0.1, 0.01],
  "inversion_max_iterations": [150, 100]
}
```

- **Variables.** The optimizer works on unconstrained variables: the logarithm of positive parameters, a logit for the bounded exponents (`b_eh` in (2, 3), `m_h` in (−1, 1), `b_r` in (1, 3)) and a softmax for `p_in`. Learning rates therefore refer to changes of ln(parameter); 0.1 is a good first value for Adam.
- **Starting values** default to the values of the case file, which are also the centre of the priors.
- **Survey error:** `inversion_survey_sigma` (m³), or `inversion_survey_relative_sigma` as a fraction of the root-mean-square observed volume (default 0.05).
- **Priors:** `inversion_parameter_prior_sd` gives the standard deviation of the unconstrained variable by parameter name; 0.7 on a logarithm means "within a factor of about 2".
- **Optimizers:** the stages of Sect. 7.1 (`Descent`, `Adam`, `LBFGS`). `LBFGS` needs fewer iterations, but compiling its line search around the whole run takes minutes; `Adam` compiles in seconds.
- **Outputs:** `inversion_results.json` (initial and final values, losses, modelled and observed volumes), `inversion_loss_history.csv`, and `inversion_callback_save_iter_<n>.json`.

Not every parameter can be identified from a deposit survey; check the sensitivities first, and start from several initial values.

### 13.9 Python interface

```python
import jax, jax.numpy as jnp
from hydrograd.io.oned_case import read_case
from hydrograd.oned import run
from hydrograd.sediment import bulk_volume_uncompacted

case = read_case("case.json")

def deposit_volume(log_a_eh):
    """Bulk deposit volume per section (m^3) for the transport coefficient exp(log_a_eh)."""
    theta = dict(case.theta)
    theta["a_eh"] = jnp.exp(log_a_eh)
    result = run(case.config, case.network, theta, case.forcing, case.bed0)
    return bulk_volume_uncompacted(result.deposition, theta, case.config.sediment)

volume = jax.jit(deposit_volume)(jnp.log(0.05))
jacobian = jax.jit(jax.jacfwd(deposit_volume))(jnp.log(0.05))     # d volume / d ln(a_eh)
```

`run` is a pure function of `theta` (and of the arrays of the forcing and the initial bed), so `jax.grad`, `jax.jacfwd` and `jax.vmap` apply. Keep `case.config` and `case.network` as constants of the compiled function (closed over, as above), not as arguments.

### 13.10 Verification

**Comparison with SRH-1D.** `benchmarks/oned/compare_srh1d.py` runs two synthetic cases for which SRH-1D v4.0.1 was run on the same set-up, and compares the thalweg with the bed profiles written by SRH-1D (its input and output files are in `benchmarks/oned/cases/`; results in `benchmarks/oned/results/srh1d_comparison.json`; details in `benchmarks/oned/README.md`). The numbers are for the end of the run and the default conveyance, with those for `"section"`, the formula of SRH-1D, in parentheses. Run times are for 8 CPU cores, without compilation (about 5 s).

| Case | Sections, classes, steps | RMS thalweg difference, all sections | Without the river ends | Largest bed rise, Hydrograd / SRH-1D | Bulk deposit, Hydrograd / SRH-1D | Run time |
|---|---|---|---|---|---|---|
| Reservoir, 2 years | 41, 3, 730 | 0.43 m (0.39 m) | 0.16 m (0.09 m) | 6.97 / 6.77 m | 10.52 / 10.37 ×10⁶ m³ | 0.5 s |
| Confluence, 5 years | 41, 5, 1826 | 0.13 m (0.13 m) | 0.06 m (0.06 m) | 2.05 / 2.67 m | 1.862 / 1.863 ×10⁶ m³ | 1.2 s |

- **River ends.** The largest differences are at the outlet and at the junction, where the two codes spread about the same deposited volume over different lengths. An end section of SRH-1D owns about half a reach, and SRH-1D repeats a junction cross-section once per river, with independent beds. In Hydrograd the outlet section owns its whole incoming reach, and the single junction section owns all its incoming reaches. At the dam of the reservoir the bed therefore rises by 4.8 m in SRH-1D and 2.2 m in Hydrograd (deposits of 167 000 and 156 000 m³); at the junction the three copies of SRH-1D rise by 2.7 to 3.1 m and the junction section of Hydrograd by 1.95 m.
- **Conveyance.** With `"section"` Hydrograd uses the conveyance of SRH-1D, and the reservoir differs by 0.09 m RMS away from the dam, for a bed rise of 7 m. The default conveyance is 4 % larger for this trapezoid and moves the front of the delta: 0.16 m RMS.

**Irregular cross-sections.** The default conveyance was chosen on a real reservoir with 279 irregular cross-sections and 34 years of daily steps (its data are not distributed). With `"section"`, the Newton iteration of the backwater profile stopped at its limit without having converged on 104 of the 12,417 steps, with residuals of the energy equation of 0.4 m (median) and up to 21 m: at 164 of the 279 sections that conveyance falls with rising water surface, by more than 20 % at 26 of them. The water surface of such a step depends on round-off; raising one bed point by 1e-10 m changed the thalweg within 100 steps by 5 cm. With `"segments"` all steps converge (35 iterations at most) and the same perturbation stays 1e-10 m.

**Tests** (`tests/test_oned_*.py`, `tests/test_sediment_closures.py`):

- **Geometry and hydraulics:** exact section properties and both conveyance formulas; uniform flow and backwater profiles; a crest as a critical-flow control; convergence of the backwater solve on a channel with a bench; networks.
- **Unsteady flow solver:** a lake at rest over irregular sections; conservation of the water volume; the dam-break solution of Stoker; a hydraulic jump behind a bump and at the foot of a steep channel, at the analytical positions; free supercritical outflow; agreement with the backwater solver in subcritical flow; a run of the time loop with sediment.
- **Sediment and bed:** exponential settling, relaxation to the transport capacity, volume conservation on a channel and on a network, non-erodible beds, the active layer.
- **Derivatives:** against central differences for the hydraulics (also through a control section) and for whole runs; forward mode against reverse mode; checkpointed against stored time loops. The relative sensitivity of the deposit volume of the reservoir to the coefficient of the load, d ln V / d ln a_r, stays near one when the run is four times as long.
- **Applications:** the case file, forward simulation, sensitivity analysis and inversion; the example cases; the comparison with SRH-1D above, within bounds.

The twin example (`examples/oned/reservoir_twin`) recovers a_eh = 0.054, kappa_w of the silt = 1.44 and a_r = 2.00e-6 from one survey with 5 % noise (truth 0.05, 1.5 and 2e-6).
