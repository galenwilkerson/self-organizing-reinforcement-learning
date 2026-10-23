#!/usr/bin/env bash
# Reproduce all of the paper's figures from the bundled data/ (CPU only; no retraining needed).
# Each script resolves the repo-relative data/ and code/ automatically.
# After this, rebuild the PDF with:
#   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
set -e
cd "$(dirname "$0")/figures"
for f in gen_ifd_fig_b64 gen_convergence_fig gen_b1_selforg gen_bridge_figs \
         gen_discrim_exp3 gen_marl_figs gen_stronger_f1 gen_learning_curves gen_robustness_sweep; do
  echo "=== $f ==="
  python3 "$f.py"
done
echo "Done: all figures regenerated in figures/."
