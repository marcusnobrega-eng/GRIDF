#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

python3 plot_bias_correction_figure_all_products.py \
  --estimator mean \
  --percentile p98

python3 plot_bias_correction_figure_all_products.py \
  --estimator median \
  --percentile p98

echo
echo "Figures created in:"
echo "figures/bias_correction/p98/mean"
echo "figures/bias_correction/p98/median"
