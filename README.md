# Code for "Patient-Level Reassessment of Breast Raman Spectroscopy: Data Leakage and Limits of Subtype Classification"

Seyedeh Rahemeh Amiri (University at Albany, State University of New York) and Ali Khoshnami (Shiraz University).
Contact: samiri@albany.edu

This archive contains the Python and GNU Octave code, run outputs and figures for a secondary, donor-level
reanalysis of the public ex vivo breast Raman dataset of Ezzat et al. (Sci. Rep. 16, 26557, 2026;
https://doi.org/10.1038/s41598-026-54071-5). The data themselves are **not** redistributed; `python/get_data.py`
downloads them from the original Zenodo record (https://doi.org/10.5281/zenodo.21901443, version 2, CC-BY 4.0)
and verifies their MD5 checksums.

## Contents

| Folder | Contents |
|---|---|
| `python/` | All Python scripts (preprocessing, splitting-unit comparison, nested donor-level validation, conformal prediction, simulations, band analyses, figures) |
| `octave/` | GNU Octave re-execution of the deposited cross-validation statements; `octave/results/` holds the CSV outputs used in the manuscript |
| `results/` | JSON/NumPy outputs of every Python analysis, as used in the manuscript |
| `figures/` | Figures 1–8 and Supplementary Figure S1 as they appear in the manuscript |
| `data/` | Empty; filled by `python/get_data.py` |

## Requirements

- Python ≥ 3.11 with the packages in `requirements.txt`. The original analyses were run with scikit-learn 1.8
  (Figures 1–6, Tables 2–4) and scikit-learn 1.9.1 / Python 3.14 (band analyses, Table 5, Figures 7–8).
- GNU Octave 11.1 with the `signal` and `statistics` (2.0.1) packages, only for the re-execution in `octave/`.

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh                 # full Python pipeline (many hours on a single core)
./octave/run_octave.sh       # optional; requires GNU Octave, run after python/prep.py
```

All Python scripts are run with `results/` as the working directory (as in `run_all.sh`); they read and write
files there and write figures to `figures/`. Single steps can be run in the same way, e.g.
`cd results && python3 ../python/bands.py`. Set `OMP_NUM_THREADS=1` to avoid oversubscription.

## Scripts and where their results appear

| Script | Purpose | Output | Manuscript |
|---|---|---|---|
| `get_data.py` | Download data and deposited MATLAB code, verify MD5 | `data/` | Data |
| `prep.py` | Preprocessing (800–2000 cm⁻¹, Savitzky–Golay 11/3, iterative 5th-order baseline, z-score); exports Octave inputs | `prep.npz`, `raw.npz`, `octave/raw.csv`, `label.csv`, `sid.txt` | Methods, Fig. 1 |
| `hierarchy.py` | Spectral correlations within point, sample and class | `hierarchy.json` | Results: donor structure |
| `split_levels.py` | Fixed PCA–SVM pipeline with spectrum-, point-, sample- and donor-level folds (10 seeds) | `split_levels.json` | Fig. 3 |
| `nested.py <model> <task> <seeds>` | Donor-disjoint nested CV; models `svm lr lda pls rf`, tasks `binary four` | `nested_*.json` | Tables 2–3, Fig. 4 |
| `summarize.py` | Accuracy, balanced accuracy, sensitivities, donor bootstrap CI, sample-level voting | `summary.json` | Table 3, sample-level aggregation |
| `conformal.py` | Split-conformal prediction calibrated on held-out donors (marginal, class-conditional, sample-level) | `conformal.json` | Table 4, Fig. 5 |
| `simulate.py` | Variance-calibrated simulation in principal-component space | `simulate.json` | Fig. 6 |
| `feats.py` | High-wavenumber preprocessing and band intensities | `bc.npz`, `ratios.npy` | Methods: band features |
| `bands.py` | Band-resolved donor/class variance shares; donor-level band ratios (Mann–Whitney, Holm) | `bands.json` | Fig. 7 |
| `regions.py <lr/svm> <seeds>`, `sumreg.py` | Nested CV for fingerprint, high-wavenumber, combined and ratio features | `regions_*.json`, `regions_summary.json` | Fig. 8a |
| `prepsens.py` | Preprocessing sensitivity of the spectrum- vs donor-level gap | `prepsens.json` | Table 5 |
| `lcurve.py` | Learning curve over the number of training donors | `lcurve.json` | Fig. 8b |
| `sim_phys.py <replicates>` | Physics-based simulation (real fluorescence backgrounds, gain, Poisson shot noise, full preprocessing) | `sim_phys.json` | Fig. 8c |
| `figs.py`, `newfigs.py` | Figures 1–6, S1 and 7–8 | `figures/` | Figures |
| `common.py`, `pcaopt.py` | Shared data loading, model grids, PCA settings | – | – |

### Octave re-execution (`octave/`)

- `prep_orig.m`: preprocessing statements from the deposited `main_code.mlx` (plots removed).
- `run_dep.m`: the cross-validation core of the deposited `pcasvm.m` and `pcalda_k.m`, transcribed with plotting
  removed. `mode='deposited'` keeps the original statement `trainIdx(removeIdx) = [];`, `mode='fixed'` replaces it
  with `trainIdx = trainIdx & ~ismember(shuffledSimpleID, testSimpleIDs);`. Octave does not implement MATLAB's
  randomised `KernelScale 'auto'`; the kernel scale is set to the median pairwise Euclidean distance of up to 300
  training rows. This is therefore an approximate re-execution, not a bit-for-bit MATLAB replication.
- `job_all.m(model, task, mode)`: component sweep (seed 1) and 10 seeds at the published component numbers;
  writes `seeds_*.csv` (seed, accuracy, mean test spectra selected for training per fold, mean training size),
  `sweep_*.csv` and `leak_*.csv`.
- `toy.m`: ten-row demonstration of the indexing statement (Fig. 2a).
- The original MATLAB files are downloaded to `data/code/` by `get_data.py` for comparison.

## Reproducibility notes

- The Python preprocessing matches the Octave execution of the deposited statements to within 6 × 10⁻⁹ per element.
- In a clean re-run of this archive, `prep.py`, `hierarchy.py`, `feats.py` and `bands.py` reproduced the stored
  results exactly, and the Octave input files were identical to those used for the manuscript.
- PCA: with scikit-learn's default `svd_solver='auto'`, used for the reported results, a randomized SVD without a
  fixed random state is selected for matrices of this size, so repeated runs differ slightly. In a test, two runs
  of the same nested seed gave 99.5% identical predictions; differences in accuracy were below the reported
  seed-to-seed standard deviations. Set `RAMAN_PCA_SOLVER=full` for exact, deterministic runs (results may then
  differ from the stored values by a similar small amount).
- Random seeds for folds, bootstrap and simulations are fixed in the scripts.

## Licence and citation

Code: MIT licence (see `LICENSE`). Outputs and figures in `results/`, `octave/results/` and `figures/`: CC-BY 4.0.
They are derived from the CC-BY 4.0 dataset of Zhu, Ezzat, Roddan et al. (https://doi.org/10.5281/zenodo.21901443),
which should be cited together with this archive. See `CITATION.cff`.
