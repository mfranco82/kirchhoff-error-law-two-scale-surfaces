# Code for: Empirical error scaling of the Kirchhoff approximation on periodic surfaces with small-scale roughness

[![DOI](https://zenodo.org/badge/1403194780.svg)](https://zenodo.org/badge/latestdoi/1403194780)

Python code that generates the figure of the Results section (Fig. 2) and the numbers quoted in the text of

> M. Franco, *Empirical error scaling of the Kirchhoff approximation on periodic surfaces with small-scale roughness*.

together with the scripts that validate the numerical reference and the formulation (`validation/`).

The Kirchhoff approximation (KA) is an expansion in the **curvature** of a surface (valid when `C = max|F''|/(2k cos³θ_i) ≪ 1`),
and the method of variation of boundaries is an expansion in its **height** (`kb ≪ 1`). The paper measures, against a rigorous
reference, how the error of KA grows on a 1D lossless dielectric interface made of a smooth periodic profile carrying a small-scale
roughness of low height but of wavelength comparable to the radiation, compares it with the first-order small-slope approximation
(SSA-1, `ssa.py`), and uses a boundary-variation series to see what KA misses and how far the small scale can be added to the
smooth-profile solution. The variations are taken **about the periodic surface** instead of a flat one, so the Taylor
coefficients `d±_{n,r}` of order `n` are coupled across all Floquet orders `r`. The zeroth order of the series is either the Kirchhoff
amplitudes of the propagating orders of the smooth profile ("KA zeroth order") or its exact (Rayleigh) solution ("exact zeroth order",
which contains no KA). TE and TM polarizations are supported. The reference solution is an independent C-method solver; the baseline is
KA applied to the whole profile.

Figure 1 of the paper is a schematic of the geometry and is not produced by this code.

## Requirements

* Python 3 with `numpy`, `scipy` and `matplotlib` (see `requirements.txt`; tested with Python 3.12.4–3.12.7, numpy 1.26.4,
  scipy 1.13.1, matplotlib 3.8.4 and 3.9.2).
* A LaTeX installation with the `mathpazo` and `amsmath` packages (e.g. TeX Live): the figure is drawn with `text.usetex`, so that its
  typography matches the paper.
* The cache-producing scripts use `multiprocessing.Pool(8)` (the 2026-10-15 regeneration used 6 workers for all but `robustness.py` and `phase_map.py`, which used 8). Every case is seeded independently, so the results do not depend on the
  number of cores (only the running time does).

## Contents

| file | role |
|---|---|
| `scattering.py` | KA amplitudes of the smooth profile (reflected and transmitted; `ka_amplitudes_propagating` sets the evanescent ones to zero), MVB recursion about the smooth surface (`mvb_coefficients`; `SmoothOperator` factorizes the system matrix once and reuses it for any roughness), Rayleigh solver (the exact zeroth order and a diagnostic), efficiencies |
| `ssa.py` | first-order small-slope approximation (Voronovich) for a 1D periodic dielectric profile, TE and TM, reflected and transmitted amplitudes; its first-order kernels are derived from the same boundary conditions and checked against the first order of the recursion about a flat surface (`validation/ssa_checks.py`) |
| `cmethod.py` | independent C-method (Chandezon et al.) solver, which does not use the Rayleigh hypothesis; it is the reference of the paper |
| `make_results_figures.py` | Fig. 2 and the numbers quoted in the text |
| `robustness.py`, `phase_map.py`, `random_stats.py` | studies that fill the caches in `data/`: cosine profiles one parameter at a time, the grid in `Λ` and `kb`, and the random-phase profiles |
| `data/` | cached results of the studies (and of the validation scripts), so that the figure and the numbers can be regenerated without recomputing; see below |
| `validation/` | tests of the solvers and of the recursion, and the scripts behind the numbers of Secs. 4 and 5; see `validation/README.md` |

Conventions: incident wave `exp(i(αx − βz))`, `α_r = α + rK`; TE has `ν² = 1` and TM `ν² = 1/ε`; the error of a model is the relative
`L¹` distance between its efficiencies and the reference ones, `Σ_r|e_r − e_r^ref| / Σ_r e_r^ref`, over the propagating reflected
orders (`ε⁺`) or the transmitted ones (`ε⁻`). The validity parameter `C` is computed for the full profile (for the cosine profiles, the
sum of the contributions of both scales; for the random-phase profiles, from the maximum of `|z''|`); `s = kb λ/λ_s` is the slope of the
small scale.

## Reproducing the figure and the numbers

Run the commands from this folder. The figure is written to `figures/`, which is created if it does not exist.

| result | command | what it reads | time |
|---|---|---|---|
| Fig. 2, error versus `C` | `python make_results_figures.py caxis` | `data/robustness.npz`, `data/phase_map.npz`, `data/random_stats.npz`, `data/ssa_sweeps.npz` | seconds |
| numbers of the abstract and of Secs. 4–5 (percentiles per bin of `C`, fit of the error law, gain of the series, 46 profiles with condition number above 1e15, stability by condition number, dependence on `kb` and on the slope, local versus global `C`, floor of the KA zeroth order against the incidence angle, base-case errors) | `python make_results_figures.py numbers` | the three caches above | ~10 s |
| Table 1, limits of `C` and `s` for a tolerance (KA and the series) | `python validation/tolerance_Cmax.py` | the caches | seconds |
| SSA-1: Sec. 4.4, its rows of Table 1, the medians of Fig. 2 | `python validation/ssa_sweeps.py summary` | `data/ssa_sweeps.npz` and the caches above | ~10 s |

`make_results_figures.py` also has the commands `spectra` and `phasemap` (exploratory figures of the spectra of one case and of the
plane `(Λ, kb)`), which are not in the paper.

The figure is in colour (free online in the journal): each model has a fixed colour from a colour-blind-safe categorical palette and
keeps its own marker and line style, so it also reads in grey; the slope `s` of the small scale is a single-hue (blue) scale. The
`caxis_v1` command draws the one-panel layout of Fig. 2 of the earlier submission (in the current colours).

The studies behind the caches can be rerun (they overwrite the files in `data/`):

| cache | command | content | time (Intel Core i7-9700) |
|---|---|---|---|
| `robustness.npz` | `python robustness.py compute` | 14 configurations of cosine profiles, one parameter at a time: `L/λ`, incidence angle, `ε`, `aK`; 7 values of `Λ`, 4 of `kb`, TE and TM | ~4 min (8 cores) |
| `phase_map.npz` | `python phase_map.py compute` | 9 values of `Λ` × 7 of `kb` at `θ_i` = 20°, TE and TM | ~1 min (8 cores) |
| `random_stats.npz` | `python random_stats.py compute` | 40 random-phase profiles for each of 12 configurations (3 bands × 4 amplitudes), TE and TM | ~6 min (6 cores) |
| `ssa_sweeps.npz` | `python validation/ssa_sweeps.py compute 6` | SSA-1 (and KA, recomputed as a check) against a recomputed C-method reference for all the profiles of the three caches above and of the decorrelated design (1958 cases) | ~5 min (6 cores) |

`python robustness.py summary`, `python phase_map.py table` and `python random_stats.py summary` print the contents of the caches. Two
configurations of `robustness.py` (grazing incidence, `sin θ_i` = 0.75 and 0.85, and `aK` = 0.38) are used only for the floor of the KA
zeroth order and for the stability study, and are left out of Fig. 2.

## Validation and other results of the paper

The scripts in `validation/` check the numerical machinery (Rayleigh solver against the C-method, the formulation against independent
calculations, the benchmark solvers) and produce the rest of the numbers of Secs. 4 and 5: bootstrap over the parameter sweeps, the
separation of the dependence on `C` and on the slope (design of cells of equal `C`), the conditioning of the matrix of the smooth
profile, convergence of the series with the order, other error metrics, the energy balance as a quality control of KA, and the timings.
The `test_*.py` files end with assertions and exit with a non-zero status if a check fails:

```
cd validation
python test_formulation.py     # 41 checks, ~20 s
python test_benchmarks.py      # 23 checks, ~10 s
python test_scattering.py      # ~2 s
python test_reference.py       # ~2 s
python test_reference_sweeps.py  # ~2 s
```

`validation/README.md` lists every script, what it prints, where the number is quoted in the paper, and its running time. The caches
that those scripts store are also in `data/`: `error_metrics.npz`, `series_convergence.npz`, `ka_energy_qc.npz`,
`decorrelated_grid.npz`, `reference_truncation.npz`, `conditioning_diagnostics.txt` and `cost_scaling_timings.txt`.

Timings are hardware dependent: `validation/cost_scaling.py` and `validation/benchmark_cost.py` must be run on an otherwise idle machine, on a
single thread; the ratios between methods are what matters (the paper quotes an Intel Core i7-9700).

## Versions

`v0.1.0` accompanied an earlier version of the paper (KA applied to the smooth profile as the baseline; an analytic average over random
roughness). `v0.2.0` corresponds to the approach of the paper at that stage, with KA of the full profile as the baseline, the validity
parameter `C` as the organizing axis, and both zeroth orders. The next version accompanies the revised manuscript: the KA zeroth order
with the evanescent Kirchhoff amplitudes set to zero, the exact zeroth order as a boundary perturbation about the exact smooth solution,
bootstrap over the sweeps, the design that separates `C` from the slope, the conditioning diagnostics, the other error metrics and the new
validation scripts. `v0.4.0` accompanies the version submitted to JQSRT, reframed around the error of KA: it adds the first-order
small-slope approximation (`ssa.py`, `validation/ssa_checks.py`, `validation/ssa_sweeps.py`, `data/ssa_sweeps.npz`) and the
four-panel colour Fig. 2. The DOI badge above always points to the latest version.

## Use of AI tools

Parts of the code were written with the assistance of Claude (Anthropic) and validated as described in `validation/`.

## License

The code is released under the MIT license (see `LICENSE`).
