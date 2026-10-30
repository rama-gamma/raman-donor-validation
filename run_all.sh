#!/usr/bin/env bash
# Full Python pipeline. Outputs are written to results/ and figures/.
# The nested SVM runs are the slowest steps; the full run takes many hours on a single core.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="$(cd "$(dirname "$0")" && pwd)"; PY="$ROOT/python"
cd "$ROOT/results"
python3 "$PY/get_data.py"
python3 "$PY/prep.py"                      # prep.npz, raw.npz, octave/raw.csv, label.csv, sid.txt
python3 "$PY/hierarchy.py"                 # spectral correlations within point / sample / class
python3 "$PY/split_levels.py"              # Fig. 3: spectrum / point / sample / donor folds
for t in binary four; do
  python3 "$PY/nested.py" svm $t 5
  for m in lr lda pls; do python3 "$PY/nested.py" $m $t 3; done
done
python3 "$PY/nested.py" rf four 1
python3 "$PY/summarize.py"                 # Table 3, sample-level aggregation
python3 "$PY/conformal.py"                 # Table 4, Fig. 5
python3 "$PY/simulate.py"                  # Fig. 6
python3 "$PY/feats.py"                     # high-wavenumber block, band intensities -> bc.npz
python3 "$PY/bands.py"                     # band-resolved variance, band ratios, Mann-Whitney/Holm
python3 "$PY/regions.py" lr 3; python3 "$PY/regions.py" svm 3; python3 "$PY/sumreg.py"
python3 "$PY/prepsens.py"                  # Table 5
python3 "$PY/lcurve.py"                    # learning curve
python3 "$PY/sim_phys.py" 8                # physics-based simulation
python3 "$PY/figs.py"                      # Figs 1-6, S1 (S1 and Fig. 2 also need octave/results/)
python3 "$PY/newfigs.py"                   # Figs 7-8
