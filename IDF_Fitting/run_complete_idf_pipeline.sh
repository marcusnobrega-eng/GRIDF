#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting

echo "============================================================"
echo "Running complete GRIDF IDF pipeline"
echo "============================================================"
echo "Products:       br_dwgd, chirps, persiann_cdr, imerg_v06, imerg_v07"
echo "States:         raw, bias_corrected_mean"
echo "Distributions:  GUMBEL, GEV"
echo "Modes:          RASTER, CETESB, STATION"
echo "Plots:          enabled"
echo "Output:         /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs"
echo "============================================================"

mkdir -p logs

PYTHONUNBUFFERED=1 python3 -u complete_idf_pipeline.py \
  --products br_dwgd,chirps,persiann_cdr,imerg_v06,imerg_v07 \
  --states raw,bias_corrected_mean \
  --modes RASTER,CETESB,STATION \
  --distributions GUMBEL,GEV \
  --raster-disag-dir /Users/mngomes/Documents/GitHub/GRIDF/Disag_Coefficients/relative_to_daily \
  --station-csv /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Subhourly_Disag_Log.csv \
  --overwrite \
  2>&1 | tee logs/complete_idf_pipeline_$(date +%Y%m%d_%H%M%S).log

echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Outputs:"
echo "/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs"
