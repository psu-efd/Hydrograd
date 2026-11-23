# Closure learning in the 1-D mode: sediment closures of a gravel-sand mixture

The transport of a size class in a mixture is described by closures: a reference shear stress, a hiding
function (fine grains are sheltered by coarse ones, coarse grains are exposed), a transport function,
and a rule for what the bed surface passes down to the deposit. Because the 1-D mode is differentiable,
any of them can be fitted through the solver to surveys of a deposit, whether it is one number or a
small neural network. This example does that for a synthetic case and for flume experiments, and tests
every fitted closure on data that were not used to fit it.

## Synthetic twin (`case.py`, `learn_hiding.py`)

A straight reach with five sand classes; the truth has a hiding law of two power laws. The surveys of
an aggradation run (with noise) are the training data, a degradation run is the test.

```bash
JAX_PLATFORMS=cpu python learn_hiding.py      # writes results/hiding_twin.json and .png
```

## Flume experiments

### Data (`flume_data.py`)

| Data | Source | Conditions | Measured |
|---|---|---|---|
| Runs 1 to 3 | Seal et al. (1997) | flume 0.3 m wide, floor slope 0.002, 49 L/s, feed with 33 % sand at 0.19, 0.09 and 0.05 kg/s, ponded downstream end | thickness of the deposit at five times, grain sizes of the deposit and of the surface, mean water depth |
| Run 6 | Toro-Escobar et al. (2000) | same width, floor slope 0.01, 12.2 L/s, feed with 53 % sand, no pond | thickness at five times, grain sizes of the gravel of deposit and surface, final water surface |
| Direct measurements | Wilcock et al. (2001) | recirculating flume, five sediments with 6 to 34 % sand, 47 runs | transport rate of every size fraction, surface composition, bed shear stress |

`flume_data.py` states how each number was obtained (tables of the papers, or figures read from the
page images, with the reading error). For run 6 the paper gives two feed rates; the one used
(3.89 kg/min) is the one the surveyed deposit volume supports (see the notes in the file). The direct
measurements are the electronic supplement of the paper, which is not distributed here; `wkc_test.py`
holds only the bed shear stresses of its Table 2.

### Model (`flume_case.py`)

The closures can only be judged if the rest of the model is right. Four errors of the model structure
had to be removed first; each showed up as a fitted parameter that ran to a limit or to a value that
has no physical meaning:

| Part of the model | First choice | What went wrong | Now |
|---|---|---|---|
| Flow | backwater curve, at most critical | run 1 is supercritical; sediment piled up at the inlet | unsteady shallow-water solver (`flow_solver = "unsteady"`) |
| Bed shear stress | hydraulic radius of the section | half the bed stress in a flume with smooth walls | mean depth (`shear_radius = "depth"`) |
| Exchange with the deposit | surface composition | no armour, far too little sorting of the deposit | transfer function: a share of the load and the rest surface (`interface_exchange = "bedload_mix"`) |
| Flow resistance and stress | Manning's n, total stress | the measured depths and slopes give n = 0.025, 0.031, 0.035 and 0.047 for runs 1, 2, 3 and 6; fits on run 2 and on run 6 disagree | variable-power resistance law with a roughness length, and the shear stress reduced for the drag on the largest grains (`resistance = "variable_power"`, `transport_stress = "reduced"`) |

The last two are published relations (see the hydraulics section of `docs/manual.md`; the logarithmic
law of Keulegan and a roughness that follows the bed surface are further options, compared below): the velocity is
U = C sqrt(g R S_f) with C = a1 a2 x / sqrt(a1^2 + a2^2 x^(5/3)), x = R / k, a1 = 6.5, a2 = 2.5
(Ferguson, 2007), and the stress for the sediment is tau_b [1 + (a1/a2)^2 x^(-5/3)]^(-3/4) (Rickenmann
and Recking, 2011, with the exponent of Meyer-Peter and Mueller, 1948). The roughness length k is the
only new parameter and takes the place of Manning's n. Run 6 is 0.03 to 0.07 m deep and needs a sharp
waterline in the section geometry (set in `flume_case.build`).

Three numbers are fitted with every closure: the multiplier of the reference Shields stress of
Wilcock and Crowe (2003), the roughness length, and the share of the load in the transfer function.

### Scripts

```bash
# Fit closures on run 2, run 6 or both, and score them on all four runs (about 1 h per closure on one
# run, 3 to 5 h for a network on two runs; learn_flume_joint.sbatch runs one fit as a batch job).
JAX_PLATFORMS=cpu python learn_flume_joint.py --train 2 6 --closures published power
JAX_PLATFORMS=cpu python learn_flume_joint.py --train 2 6 --closures published hiding_network --tag hnet \
    --direct FOLDER_OF_THE_SUPPLEMENT          # deposits and direct measurements together

python plot_flume_joint.py --profiles results/joint_train26_variable_power_reduced.json
python plot_flume_joint.py --scores results/joint_train*.json
python plot_flume_joint.py --functions results/joint_train26_variable_power_reduced_direct_hnet.json
# History of the misfit during the fits of one results file.
python plot_flume_joint.py --history results/joint_train26_variable_power_reduced_direct_hnet.json
# The same profiles in two figures for a talk (thickness; grain sizes).
python plot_flume_joint.py --slides results/joint_train26_variable_power_reduced_direct_hnet.json
# Animation of a run with the fitted closure: bed and water surface while the deposit grows (.gif, and
# .mp4 where ffmpeg is on the path).
JAX_PLATFORMS=cpu python animate_flume.py results/joint_train26_variable_power_reduced_direct_hnet.json --run 2

# Score closures against the direct measurements (no flow or bed model involved).
JAX_PLATFORMS=cpu python wkc_test.py --data FOLDER_OF_THE_SUPPLEMENT --results results/joint_train*.json
```

Closures (`--closures`): `published` (hiding and transport functions of Wilcock and Crowe), `power`
(hiding factor (d/d_g)^(-m_h)), `hiding_network` (a network of 97 weights for the hiding function, input
ln(d/d_g), started from the published function), `transport_network` (the same for the transport
function, input the ratio of the shear stress to the reference stress of the class).

Tests: runs 1 and 3 are never used for fitting (another feed rate of the mixture of run 2). Run 6 is the
test for another mixture and flow when run 2 is the training run, and the other way round. With
`--direct`, two of the five sediments of the direct measurements (`--direct-test`) are left out of the
fit and scored separately.

### Misfit (loss function)

A fit minimizes one number, the misfit J (`fit` in `learn_flume_joint.py`). It is the sum of one term
for every training run and, with `--direct`, one term for the direct measurements:

```
J = J_run2 + J_run6 + w J_direct                      w = 1 (--direct-weight)
```

**Term of a flume run** (`misfit2` and `misfit6`). Each kind of measurement enters as half its mean
squared error in units of a standard deviation:

```
J_run = 1/2 mean[ ((h - h_obs) / sigma_h)^2 ]             thickness of the deposit
      + 1/2 mean[ ((psi - psi_obs) / sigma_psi)^2 ]       grain sizes of the deposit
      + 1/2 ((H - H_obs) / sigma_H)^2                     mean water depth
      + 1/2 mean[ ((psi_s - psi_s,obs) / sigma_s)^2 ]     grain sizes of the surface (only with --surface)
```

with psi = log2(d / 1 mm) of a percentile of the grain-size distribution, and each mean taken over the
values of that kind:

| Kind | Run 2 | Run 6 | Standard deviation |
|---|---|---|---|
| Thickness h | surveys at 3, 8, 14, 22 and 32.4 h, 51 values | surveys at 2, 8, 16, 22 and 27 h, 62 values | 0.03 m |
| Grain sizes psi | D90, D84, D50 and D16 of the deposit of hours 14 to 22 at eight stations, 32 values | D90, D50 and D16 of the gravel in the deposit of hours 8 to 18 at the five stations on the wedge, 15 values | 0.3 psi |
| Water depth H | mean over the deposit (measured 0.139 m) | mean over the first ten survey stations (measured 0.059 m) | 0.01 m in run 2, 0.005 m in run 6 |
| Surface sizes psi_s (`--surface`) | D90 and D50 at the upstream and downstream ends, 4 values | D90, D50 and D16 of the gravel at the five stations on the wedge, 15 values | 0.5 psi |

**Term of the direct measurements** (`Direct.misfit`):

```
J_direct = 1/2 mean[ ((log10 W* - log10 W*_obs) / sigma_W)^2 ]           sigma_W = 0.5
W*_k     = (s - 1) g q_k / (F_k u*^3)
```

W*_k is the dimensionless transport rate of size fraction k: q_k its volume transport rate per unit
width, F_k its share of the bed surface, u* = sqrt(tau_b / rho) with the measured bed shear stress, and
s the specific gravity of the sediment. The mean is over all fractions with a measured transport in
the runs of the sediments used for fitting (245 values in three of the five sediments; the 200 values
of the two sediments named with `--direct-test` are left out). The model value is the transport formula
alone, at the measured shear stress and the measured surface composition of each run; no flow or bed
model is involved. The closure is the same as in the flume runs, but the reference Shields stress has a
multiplier of its own, which is a fourth fitted number.

**What follows from this form**

- Every term is a mean, so a kind of measurement does not weigh more because it has more values. The
  standard deviations are chosen values, not measured errors; they set the weight of the kinds against
  each other.
- A term of 0.5 means that the root-mean-square error equals the standard deviation; in general the
  root-mean-square error is sigma sqrt(2 x term).
- There is no penalty on the network weights. The fitted numbers have limits (`LIMITS` in
  `learn_flume.py`); the weights have none.
- J is minimized with L-BFGS-B on all data at once. The gradient of each term comes from reverse-mode
  differentiation of its model (the whole flume run, or the transport formula), and the gradients are
  added like the terms.
- Not in J, and so tests: runs 1 and 3, the sediments left out of the direct measurements, and the
  grain sizes of the surface unless `--surface` is given.

**Values** for the fit on runs 2 and 6 with the direct measurements
(`results/joint_train26_variable_power_reduced_direct_hnet.json`):

| Term | Published hiding function (4 fitted numbers) | Learned hiding function (the same 4 and 97 weights) |
|---|---|---|
| Run 2 | 2.33 | 1.69 |
| - thickness | 1.23 | 0.92 |
| - grain sizes | 1.01 | 0.69 |
| - water depth | 0.09 | 0.08 |
| Run 6 | 0.86 | 0.87 |
| - thickness | 0.41 | 0.45 |
| - grain sizes | 0.28 | 0.34 |
| - water depth | 0.17 | 0.08 |
| Direct measurements | 0.51 | 0.54 |
| J | 3.70 | 3.09 |
| Tests, not in J: | | |
| Run 1, thickness | 1.38 | 1.05 |
| Run 1, water depth | 1.57 | 1.57 |
| Run 3, thickness | 1.91 | 1.37 |
| Run 3, water depth | 0.01 | 0.02 |
| Direct measurements left out | 0.49 | 0.55 |

The gain of the learned hiding function is in run 2 (thickness and grain sizes); run 6 stays the same
and the direct measurements are slightly worse. The results file holds the term of each training run
(`stages.<closure>.runs.<run>.model.misfit`) and the value of J at every evaluation (`misfit_history`,
drawn by `plot_flume_joint.py --history`); the term of the direct measurements is
0.5 (`direct.train.rms_log10` / 0.5)^2. The column "Misfit, runs 2 + 6" of the tables below is
J_run2 + J_run6 without the term of the direct measurements.

### Results

Errors are root-mean-square errors; the misfit is defined in the section above.

**Flow resistance and stress, with the published closures** (three fitted numbers):

| Resistance, stress | Trained on | Multiplier | n or k | Share | Thickness, runs 2 / 1 / 3 / 6 (cm) | Depth, runs 2 / 1 / 3 / 6 (m) |
|---|---|---|---|---|---|---|
| measured | | | | | | 0.139 / 0.120 / 0.149 / 0.059 |
| Manning, total | run 2 | 1.36 | 0.030 | 0.84 | 4.6 / 4.8 / 5.5 / 5.1 | 0.138 / 0.133 / 0.142 / 0.048 |
| Manning, total | run 6 | 2.05 | 0.035 (limit) | 0.73 | 6.0 / 5.8 / 6.1 / 3.0 | 0.148 / 0.143 / 0.151 / 0.051 |
| variable power, total | run 6 | 2.90 | 0.029 m | 0.77 | 10.9 / 9.3 / 10.4 / 2.0 | 0.145 / 0.135 / 0.148 / 0.059 |
| variable power, reduced | run 2 | 1.13 | 0.022 m | 0.77 | 4.6 / 4.9 / 5.6 / 2.6 | 0.138 / 0.133 / 0.143 / 0.054 |
| variable power, reduced | run 6 | 1.06 | 0.029 m | 0.63 | 5.3 / 5.2 / 7.0 / 2.4 | 0.150 / 0.144 / 0.156 / 0.059 |
| variable power, reduced | runs 2 and 6 | 1.11 | 0.026 m | 0.75 | 4.7 / 5.0 / 5.9 / 2.7 | 0.143 / 0.138 / 0.148 / 0.056 |

With the variable-power law and the reduced stress the closures fitted on one mixture predict the other
(2.6 cm on run 6 from run 2; 5.3 cm on run 2 from run 6), and the three numbers are the same within
their spread whichever run is used: a reference stress 1.06 to 1.13 times the published one (a smooth-
wall correction of the shear stress, which the model does not make, would explain 1.13) and a share of
the load of 0.63 to 0.77 (published: 0.7). The multiplier that suits the direct measurements best is
1.11 as well.

**Resistance law and roughness scale, with the published closures** (trained on runs 2 and 6; the
roughness is one fitted length for all runs, or a fitted factor times the D90 or the geometric mean
size D_sg of the bed surface of each section; Cui, 2007, uses 2.5 D_sg, Parker, 1990, 2 D90):

| Resistance law | Roughness | Stress | Multiplier | Length or factor | Share | Misfit, runs 2 + 6 | Thickness, runs 2 / 1 / 3 / 6 (cm) | Depth, runs 2 / 1 / 3 / 6 (m) |
|---|---|---|---|---|---|---|---|---|
| variable power | one length | reduced | 1.11 | 0.026 m | 0.75 | 3.19 | 4.7 / 5.0 / 5.9 / 2.7 | 0.143 / 0.138 / 0.148 / 0.056 |
| Keulegan | one length | reduced | 1.07 | 0.094 m | 0.76 | 3.38 | 4.7 / 5.0 / 5.7 / 2.4 | 0.146 / 0.140 / 0.150 / 0.055 |
| variable power | factor x D90 | reduced | 1.07 | 0.48 | 0.76 | 3.85 | 4.7 / 5.0 / 5.8 / 2.7 | 0.146 / 0.141 / 0.151 / 0.054 |
| Keulegan | factor x D90 | reduced | 1.06 | 1.8 | 0.76 | 3.96 | 4.7 / 5.0 / 5.5 / 2.5 | 0.148 / 0.143 / 0.153 / 0.053 |
| variable power | factor x D_sg | reduced | 1.08 | 1.9 | 0.78 | 4.65 | 4.9 / 5.2 / 5.8 / 2.8 | 0.149 / 0.140 / 0.157 / 0.052 |
| Keulegan | factor x D_sg | reduced | 1.07 | 7.0 | 0.78 | 4.77 | 4.8 / 5.2 / 5.6 / 2.7 | 0.150 / 0.141 / 0.158 / 0.051 |
| Keulegan | factor x D90 | total | 1.58 | 1.5 | 0.79 | 4.78 | 4.3 / 4.3 / 4.8 / 5.5 | 0.143 / 0.139 / 0.146 / 0.056 |
| Keulegan (Cui) | factor x D_sg | total | 1.65 | 5.7 | 0.79 | 4.95 | 4.2 / 4.5 / 4.8 / 5.3 | 0.147 / 0.138 / 0.154 / 0.055 |
| Keulegan (Cui), Parker curve | factor x D_sg | total | 1.89 | 4.1 | 0.79 | 7.09 | 5.1 / 5.7 / 5.5 / 6.0 | 0.144 / 0.139 / 0.146 / 0.055 |

The logarithmic law does nearly as well as the variable-power law when its roughness height is one
length and the stress is reduced. A roughness tied to D_sg does not carry between the two mixtures:
fitted on one run at a time, the factor is 3.3 to 4.5 on run 2 and 8.7 to 12.9 on run 6, and the depth
of the other run is then missed (0.047 m for run 6 from run 2; 0.171 m for run 2 from run 6). The surface
of run 6 is sandier, which lowers D_sg, while its flow is rougher. The D90 does better (factor 1.2 on
run 2 and 2.7 on run 6 with the logarithmic law, around the 2 of Parker; 0.37 and 0.66 with the
variable-power law), but one length for both mixtures still fits the depths best. The roughness scale
changes the depths only: the thickness errors, the multiplier of the reference stress and the share of
the load are the same within a few percent in all rows with the reduced stress. With the total stress (the treatment
of Cui, 2007) run 2 alone is fitted well (misfit 1.99 with a factor of 3.3, the share of the load at its
limit of 0.95), but run 6 is then predicted to 5.8 cm, and a fit on run 6 needs a reference stress 2.5
times the published one and misses run 2 by 8 cm. The transport curve of Parker fits worse than that of
Wilcock and Crowe in every case.

**Closures** (variable power, reduced stress; misfit of runs 2 and 6 added; "direct" column for the two
sediments left out of every fit: scatter of log10 of predicted over observed transport):

| Hiding or transport closure | Trained on | Misfit, runs 2 + 6 | Thickness, runs 2 / 1 / 3 / 6 (cm) | Direct: scatter |
|---|---|---|---|---|
| published | runs 2 and 6 | 3.19 | 4.7 / 5.0 / 5.9 / 2.7 | 0.49 |
| power law, m_h = -0.89 | run 2 | 4.62 | 3.8 / 3.6 / 5.2 / 7.0 | |
| hiding network | run 2 | 5.81 | 3.8 / 3.6 / 5.5 / 8.2 | |
| hiding network | run 2 + direct | 3.65 | 3.8 / 3.7 / 5.0 / 5.1 | 0.53 |
| hiding network | run 6 | 3.64 | 4.7 / 5.1 / 5.6 / 1.7 | |
| power law, m_h = -0.81 | runs 2 and 6 | 3.14 | 4.4 / 4.4 / 5.1 / 3.5 | 0.53 |
| hiding network | runs 2 and 6 | 2.47 | 4.2 / 4.4 / 5.0 / 2.5 | |
| hiding network | runs 2 and 6 + direct | 2.55 | 4.1 / 4.4 / 5.0 / 2.9 | 0.52 |
| transport network | runs 2 and 6 + direct | 3.19 | 4.6 / 4.9 / 5.9 / 3.0 | 0.48 |

A comparison by grain size on the direct measurements is not given. For the sizes coarser than twice
the mean, the ratio of predicted to measured transport of the published formula is anywhere between 0.26
and 0.99 in total (0.44 to 1.9 as the median over the runs), depending on which size is taken for a
sieve class (its middle or its lower sieve, which the papers do not state), on the surface composition
used (that of each run or the mean of the sediment) and on the multiplier of the reference stress; two
runs carry 92 % of the coarse transport of the two sediments left out. The direct measurements were
used here with the middle of each class and the surface of each run, and only the fractions with a
measured transport; the fitted functions inherit those choices.

Scored on all 47 runs of the direct measurements with `wkc_test.py` (best multiplier for each closure),
the published formula has 53 % of the fractions within a factor of 2; the hiding networks fitted on
deposits alone have 34 to 40 %.

Figures: `results/joint_train26_variable_power_reduced_direct_hnet.png` (deposits of the four runs,
published closures and learned hiding function), `results/joint_functions.png` (hiding functions),
`results/joint_scores.png` (all errors).

### What the case shows, and its limits

- A flexible closure fitted to one family of runs absorbs errors of the rest of the model. Fitted on
  run 2, a hiding exponent or a hiding network halves the misfit of run 2 and passes the test on runs 1
  and 3, which share the mixture of run 2. It then fails on run 6 (7 to 8 cm against 2.6 cm for the
  published function) and against the direct measurements. Tests on the same mixture do not expose this.
- Once the flow resistance and the shear stress are right, the published closures need only three
  numbers, which come out near their published values, to reproduce four experiments.
- Fitted on both mixtures, one hiding exponent brings nothing (misfit 3.14 against 3.19). A hiding
  network does: it lowers the misfit by 20 % and the thickness error of the two test runs by 12 and
  15 %. Fitted on the deposits alone it does worse than the published function on the direct
  measurements; fitted on deposits and direct measurements together it keeps the gain in the flume and
  has nearly the scatter of the published function on the two sediments it never saw (0.52 against
  0.49). Relative to the reference stress of the mean size, the learned function stays within 17 % of
  the published one for sizes below the mean and gives sizes 3 to 10 times the mean a reference stress
  25 to 40 % lower. The multiplier of the reference stress of the mean size, fitted together with the
  network, is 24 % higher than with the published function (1.38 against 1.11), so that in absolute
  terms the sizes from 0.3 to 2 times the mean have a reference stress 12 to 24 % higher in the flume
  and the sizes 3 to 10 times the mean one 7 to 25 % lower. The network fitted on the two flume
  runs alone lowers the reference stress of the coarse sizes as well, so the deposits support the
  direction; how large the change is depends on how the direct measurements are used (see the note
  under the table) and is not settled.
- A network for the transport function finds nothing to improve when the direct measurements are part
  of the fit.
- The surface layer of the model is about 1.3 psi coarser than the surface samples of runs 1 to 3. The
  samples are bulk samples to the depth of the largest grain, the formula of Wilcock and Crowe is based
  on counts of the surface, and the two are not the same quantity; how much of the difference this
  explains is not known. Fits that are made to match the samples (`--surface`) push the share of the
  load to 0.13 to 0.2, which switches the transfer function off.
- Not reproduced: the depth of run 1 (0.133 to 0.144 m against 0.120 m measured). The flume walls carry
  no friction in the model. The inlet of run 6 is set at the head of the deposit, not at the feed point.
- The results do not change with half the time step or half the flow step; with half the section
  spacing the thickness error of run 6 changes from 2.7 to 2.9 cm.

### Earlier scripts

`learn_flume.py`, `learn_flume_stage.py`, `learn_flume_transport.py`, `predict_run6.py`, `plot_flume.py`
and `plot_flume_transport.py` are the first steps of this study (one training run, Manning's n). The
joint script uses the measurements and the misfit of `learn_flume.py`; their results are superseded by
the tables above.

## References

- Cui, Y. (2007). The Unified Gravel-Sand (TUGS) model: simulating sediment transport and gravel/sand
  grain size distributions in gravel-bedded rivers. Water Resources Research 43, W10436.
- Ferguson, R. (2007). Flow resistance equations for gravel- and boulder-bed streams. Water Resources
  Research 43, W05427.
- Meyer-Peter, E., and Mueller, R. (1948). Formulas for bed-load transport. Proceedings of the 2nd
  Meeting of the International Association for Hydraulic Structures Research, Stockholm, 39-64.
- Rickenmann, D., and Recking, A. (2011). Evaluation of flow resistance in gravel-bed rivers through a
  large field data set. Water Resources Research 47, W07538.
- Seal, R., Paola, C., Parker, G., Southard, J. B., and Wilcock, P. R. (1997). Experiments on downstream
  fining of gravel: I. Narrow-channel runs. Journal of Hydraulic Engineering 123(10), 874-884.
- Toro-Escobar, C. M., Paola, C., Parker, G., Wilcock, P. R., and Southard, J. B. (2000). Experiments on
  downstream fining of gravel. II: Wide and sandy runs. Journal of Hydraulic Engineering 126(3), 198-208.
- Wilcock, P. R., and Crowe, J. C. (2003). Surface-based transport model for mixed-size sediment.
  Journal of Hydraulic Engineering 129(2), 120-128.
- Wilcock, P. R., Kenworthy, S. T., and Crowe, J. C. (2001). Experimental study of the transport of
  mixed sand and gravel. Water Resources Research 37(12), 3349-3358.
