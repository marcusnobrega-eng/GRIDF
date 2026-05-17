#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

PERCENTILE="p98"
MAX_FEATURES=1000

run_one () {
  PRODUCT="$1"
  START_YEAR="$2"
  END_YEAR="$3"

  DRIVE_FOLDER="GRIDF_BiasCorrection_pairs_${PRODUCT}_${PERCENTILE}_${START_YEAR}_${END_YEAR}_fullGauge"

  echo
  echo "============================================================"
  echo "Product:      ${PRODUCT}"
  echo "Years:        ${START_YEAR}-${END_YEAR}"
  echo "Drive folder: ${DRIVE_FOLDER}"
  echo "============================================================"

  python3 run_pipeline.py select-events \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR"

  python3 run_pipeline.py export-pairs \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --max-features-per-export "$MAX_FEATURES" \
    --drive-folder "$DRIVE_FOLDER"
}

run_one br_dwgd       1995 2006
run_one chirps        1995 2006
run_one persiann_cdr  1995 2006

# IMERG starts later. We try 2000–2006 to match the old logic.
# If GEE has no valid images for part of 2000, those sampled rows should be filtered out or will simply be sparse.
run_one imerg_v06     2000 2006
run_one imerg_v07     2000 2006

echo
echo "============================================================"
echo "All GEE pair-export tasks submitted."
echo "Monitor with:"
echo "earthengine task list"
echo "============================================================"
