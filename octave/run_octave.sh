#!/usr/bin/env bash
# Re-execution of the deposited cross-validation statements in GNU Octave (11.1; packages signal, statistics 2.0.1).
# Requires octave/raw.csv, label.csv and sid.txt, written by python/prep.py.
set -euo pipefail
cd "$(dirname "$0")"
octave --no-gui --eval "pkg load signal; pkg load statistics; prep_orig"      # -> prep_oct.bin, Z_oct.csv
octave --no-gui --eval "toy"                                                  # ten-row demonstration (Fig. 2a)
for job in "svm binary deposited" "svm binary fixed" "svm four deposited" "svm four fixed" \
           "lda four deposited" "lda four fixed"; do
  set -- $job
  octave --no-gui --eval "job_all('$1','$2','$3')" > "log_$1_$2_$3.txt" 2>&1
done
mkdir -p results; mv -f seeds_*.csv sweep_*.csv leak_*.csv results/
