#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

PRODUCTS=(
  br_dwgd
  imerg_v06
  imerg_v07
  chirps
  persiann_cdr
)

PERCENTILE="p98"

echo "============================================================"
echo "FINAL PAPER BIAS-CORRECTION WORKFLOW"
echo "============================================================"
echo "This workflow uses:"
echo "1) existing paper-consistent pair files"
echo "2) legacy-equivalent zeta calculation"
echo "3) legacy-equivalent IDW interpolation"
echo "4) apply-bias and diagnostics from the pipeline"
echo "5) final Helvetica + biome figure"
echo "============================================================"

echo
echo "Step 1: Build legacy-equivalent station zeta tables"
python3 build_legacy_equivalent_zeta_from_pairs.py

echo
echo "Step 2: Interpolate zeta maps using legacy-equivalent IDW"
python3 interpolate_zeta_legacy_equivalent_from_pipeline.py

echo
echo "Step 3: Apply bias correction and diagnostics"
for ESTIMATOR in mean median; do
  for PRODUCT in "${PRODUCTS[@]}"; do
    echo "============================================================"
    echo "$PRODUCT / $ESTIMATOR"
    echo "============================================================"

    rm -rf "data/products/$PRODUCT/sensitivity/$PERCENTILE/annual_max_corrected/$ESTIMATOR"
    rm -rf "data/products/$PRODUCT/sensitivity/$PERCENTILE/diagnostics/$ESTIMATOR"

    python3 run_pipeline.py apply-bias \
      --product "$PRODUCT" \
      --percentile "$PERCENTILE" \
      --estimator "$ESTIMATOR"

    python3 run_pipeline.py diagnostics \
      --product "$PRODUCT" \
      --percentile "$PERCENTILE" \
      --estimator "$ESTIMATOR"

    python3 check_outputs.py \
      --product "$PRODUCT" \
      --percentile "$PERCENTILE" \
      --estimator "$ESTIMATOR"
  done
done

echo
echo "Step 4: Generate final figures"
python3 plot_bias_correction_legacy_style_5x3_helvetica_biomes.py --estimator mean
python3 plot_bias_correction_legacy_style_5x3_helvetica_biomes.py --estimator median

echo
echo "DONE."
echo "Final figures:"
echo "figures/bias_correction/legacy_style_5x3_helvetica_biomes/p98/mean"
echo "figures/bias_correction/legacy_style_5x3_helvetica_biomes/p98/median"
