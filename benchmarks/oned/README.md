# Comparison of the 1-D mode with SRH-1D

`compare_srh1d.py` runs Hydrograd on cases for which the USBR model SRH-1D (v4.0.1) was run on the same set-up, and compares the bed profiles. SRH-1D runs on Windows only, so its input and output files are stored here and it is not run again.

```bash
python benchmarks/oned/compare_srh1d.py              # all cases
python benchmarks/oned/compare_srh1d.py reservoir    # one case
```

Results are written to `results/srh1d_comparison.json`. The two cases together take about 30 s on a CPU, most of it compilation.

## Cases

Each folder in `cases/` holds:

| File | Content |
|---|---|
| `case.json` | the Hydrograd case (manual, Sect. 13.3) |
| `benchmark.json` | how the SRH-1D output maps onto the Hydrograd case: the SRH-1D river and section number of each Hydrograd cross-section, the length unit, the junction sections |
| `srh1d/<name>.srh` | the SRH-1D input file |
| `srh1d/<name>_OUT_Profile.DAT` | the bed profiles written by SRH-1D (read by `hydrograd.io.srh1d`) |

- **`reservoir`:** a 20 km trapezoidal channel (80 m bottom width, 2:1 banks, slope 0.001) with an 18 m pool at the dam. The upper 2.5 km has a non-erodible bed. Seasonal discharge of 100 to 250 m³/s; silt, fine sand and sand from the rating curve Q_t = 2e-6 Q²; two years of daily steps, 41 cross-sections. This is also the reservoir of `examples/oned/reservoir_delta`.
- **`confluence`:** two branches join at the head of a main stem with a pool at the dam. Five size classes from clay to fine gravel, a prescribed inflow load on each branch; five years of daily steps, 41 cross-sections.

Both set-ups are synthetic. The Hydrograd case files and the SRH-1D input files describe the same geometry, boundary conditions, size classes and parameter values. They were written once from a common definition and are stored as data; no script in this repository rebuilds one from the other. The title lines (`YTT`) of the SRH-1D input files were changed after the runs; nothing else was.

The hiding exponent is zero (`m_h = 0`) in both cases, because the Engelund–Hansen formula of SRH-1D has no hiding factor.

## What is compared

At every output time of SRH-1D after the start, the thalweg of all cross-sections is compared with that of the Hydrograd step that ends closest to it, as a root-mean-square (RMS) difference over the cross-sections. The largest rise of the thalweg and the total bulk volume of the deposits at the last output time are compared as well. Each case is run with both formulas of the conveyance (manual, Sect. 13.1): `"segments"`, the default, and `"section"`, the formula of SRH-1D.

**River ends.** The RMS difference is also given without the cross-sections at the ends of the rivers of SRH-1D, which are the outlet and the junctions. There the two codes spread a deposit over different lengths:

- **SRH-1D:** an end section owns about half a reach. A junction cross-section is repeated once per river (as the last section of each branch and the first of the main stem), and the copies have independent beds.
- **Hydrograd:** the outlet section owns its whole incoming reach. There is one junction section, and it owns all its incoming reaches.

For about the same deposited volume, the bed of SRH-1D therefore rises more at these sections.

## Results

Times are for 8 CPU cores, without compilation. RMS differences are those at the end of the run.

| Case | Conveyance | RMS thalweg difference, all sections | Without the river ends | Largest bed rise, Hydrograd / SRH-1D | Bulk deposit, Hydrograd / SRH-1D | Run time |
|---|---|---|---|---|---|---|
| reservoir | `"segments"` | 0.43 m | 0.16 m | 6.97 / 6.77 m | 10.52 / 10.37 ×10⁶ m³ | 0.5 s |
| reservoir | `"section"` | 0.39 m | 0.09 m | 6.85 / 6.77 m | 10.47 / 10.37 ×10⁶ m³ | 0.3 s |
| confluence | `"segments"` | 0.13 m | 0.06 m | 2.05 / 2.67 m | 1.862 / 1.863 ×10⁶ m³ | 1.2 s |
| confluence | `"section"` | 0.13 m | 0.06 m | 2.03 / 2.67 m | 1.862 / 1.863 ×10⁶ m³ | 0.8 s |

- **Reservoir, dam section:** the two codes deposit 156 000 m³ (Hydrograd) and 167 000 m³ (SRH-1D) there, and the bed rises by 2.2 m and 4.8 m. This one section accounts for most of the RMS difference over all sections.
- **Reservoir, other sections:** with the default conveyance the bed of Hydrograd is up to 0.4 m higher at the front of the delta (bed rise 6 to 7 m) and up to 0.2 m lower in the pool below it.
- **Confluence, junction:** the three copies of SRH-1D hold 174 000, 184 000 and 160 000 m³ and rise by 2.9, 3.1 and 2.7 m, which is 0.7 to 1.2 m more than their neighbouring sections. The single junction section of Hydrograd holds 467 000 m³ and rises by 1.95 m, in line with its neighbours. The largest bed rise of SRH-1D in the table is that of its main-stem copy of the junction.
- **Confluence, other sections:** on the two branches the codes differ by at most 0.08 m; in the main stem below the junction by up to 0.19 m in the first sections, less further down.

## Cases that cannot be distributed

A case whose data cannot be published can be put into `local/<name>/` with the same four files. `local/` is not part of the repository. `compare_srh1d.py` runs such cases as well and writes their results to `local/results/srh1d_comparison.json`.
