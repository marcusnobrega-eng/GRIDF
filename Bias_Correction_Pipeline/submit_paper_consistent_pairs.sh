#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

RAIN_CSV="/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction/rainfall_timeseries_with_metadata_all.csv"

python3 export_pairs_paper_consistent.py \
  --product br_dwgd \
  --csv-path "$RAIN_CSV" \
  --start-year 1995 \
  --end-year 2006 \
  --drive-folder GRIDF_paper_consistent_br_dwgd_p98_1995_2006

python3 export_pairs_paper_consistent.py \
  --product chirps \
  --csv-path "$RAIN_CSV" \
  --start-year 1995 \
  --end-year 2006 \
  --drive-folder GRIDF_paper_consistent_chirps_p98_1995_2006

python3 export_pairs_paper_consistent.py \
  --product persiann_cdr \
  --csv-path "$RAIN_CSV" \
  --start-year 1995 \
  --end-year 2006 \
  --drive-folder GRIDF_paper_consistent_persiann_cdr_p98_1995_2006

python3 export_pairs_paper_consistent.py \
  --product imerg_v07 \
  --csv-path "$RAIN_CSV" \
  --start-year 2001 \
  --end-year 2006 \
  --drive-folder GRIDF_paper_consistent_imerg_v07_p98_2001_2006

echo
echo "Submitted paper-consistent GEE pair exports."
echo "Monitor with:"
echo "earthengine task list"
