# Examples of the 1-D mode

Three synthetic cases of the 1-D mode (flow and sediment transport on a network of cross-sections; manual, Sect. 13). Each folder holds a case file `case.json` and a `run_control.json`. All quantities are in SI units.

**Copy a folder before running it:** a run writes its results into the case folder, and the forward mode writes `oned_survey_truth.json`, which would overwrite the truth file of the twin example.

```bash
cp -r examples/oned/reservoir_delta /tmp/reservoir_delta
cd /tmp/reservoir_delta
hydrograd run
```

The cases are written by `build_cases.py`; edit it to change the geometry, the forcing or the parameters and run `python examples/oned/build_cases.py` again.

## `reservoir_delta`: forward simulation

A 20 km trapezoidal channel (80 m bottom width, banks 2:1, slope 0.001) ends at a dam with an 18 m pool. The upper 2.5 km is a river with a non-erodible bed; the rest is the reservoir, with 41 cross-sections 500 m apart.

- **Flow:** seasonal hydrograph, 100 to 250 m³/s.
- **Sediment:** three size classes (silt 0.02 mm, fine sand 0.1 mm, sand 0.5 mm). The inflow load follows the rating curve Q_t = a_r Q² with 60 % silt, 30 % fine sand and 10 % sand.
- **Run:** two years of daily steps (730 steps), about 4 s including compilation on a CPU.

Results: a sand delta progrades into the pool and raises the bed by up to 7.0 m (14.5 km from the upstream end); the silt settles further downstream. The reservoir traps 99.5 % of the load at the end of the run. The surveys after one and two years are written to `oned_survey_truth.json`.

## `reservoir_twin`: sensitivity analysis and inversion

The same reservoir with 2-day steps. `oned_survey_truth.json` holds the consolidated deposit volume per cross-section after two years of a run with known parameters, with 5 % noise. The case file starts from different values, and the inversion recovers them from the survey:

| Parameter | Truth | Start | Recovered |
|---|---|---|---|
| `a_eh` (transport coefficient) | 0.05 | 0.03 | 0.054 |
| `kappa_w[0]` (settling multiplier of the silt) | 1.5 | 0.7 | 1.44 |
| `a_r` (rating coefficient) | 2.0e-6 | 4.0e-6 | 2.00e-6 |

```bash
hydrograd run run_control_sensitivity.json     # Jacobian of the surveyed volumes
hydrograd run                                  # inversion (run_control.json)
```

- **Sensitivity:** forward mode, about 10 s. `sensitivity_results.json` holds the Jacobian of the deposit volume of every section with respect to the three parameters. The survey is most sensitive to the rating coefficient, then to the settling multiplier, then to the transport coefficient.
- **Inversion:** two Adam stages (150 and 100 iterations) on the logarithms of the parameters; each iteration is one forward and one reverse pass through the 365 steps. The recovered values differ from the truth by the effect of the survey noise.

## `confluence`: forward simulation on a network

Two tributaries join at the head of a 10 km main stem with a 12 m pool at the dam (39 cross-sections).

- **Branch A:** 5 km, 40 m wide, slope 0.0012, 60 to 150 m³/s.
- **Branch B:** 4 km, 30 m wide, slope 0.0015, 30 to 90 m³/s, with its high flows half a season after those of branch A.
- **Bed:** non-erodible more than 2 km above the junction.
- **Run:** one year of daily steps, about 4 s including compilation.

Results: the bed erodes by up to 0.7 m in the erodible lower part of branch B, and sediment deposits in the main stem (up to 2.9 m). The network numbering and the treatment of the junction are described in the manual (Sect. 13.2).

## Output files

| File | Content |
|---|---|
| `oned_forward_results.json` | final thalweg, water surface, deposit volume per section, loads and trap efficiency |
| `oned_forward_results.npz` | all arrays of the run (bed tables, deposition per step, class and section, histories) |
| `oned_survey_truth.json` | surveyed deposit volumes (the observations of an inversion) |
| `sensitivity_results.json` | Jacobian of the surveyed volumes |
| `inversion_results.json`, `inversion_loss_history.csv` | recovered parameters and the loss history |
