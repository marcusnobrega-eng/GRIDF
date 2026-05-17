#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

PRODUCTS=(
  imerg_v07
  imerg_v06
  chirps
  persiann_cdr
  br_dwgd
)

PERCENTILE="p98"
ESTIMATOR="mean"
START_YEAR=2001
END_YEAR=2006

echo "============================================================"
echo "GRIDF bias-correction rerun from existing pair files"
echo "No GEE pair extraction will be run."
echo "Estimator: ${ESTIMATOR}"
echo "Calibration years: ${START_YEAR}-${END_YEAR}"
echo "============================================================"

echo
echo "Current method configuration:"
cat config/method.yml

echo
echo "============================================================"
echo "Checking pair files"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  PAIR_DIR="data/products/${PRODUCT}/sensitivity/${PERCENTILE}/pairs"

  echo
  echo "${PRODUCT}"
  find "$PAIR_DIR" -maxdepth 1 -type f -name "pairs_${PRODUCT}_${PERCENTILE}_*.csv" | sort
  COUNT=$(find "$PAIR_DIR" -maxdepth 1 -type f -name "pairs_${PRODUCT}_${PERCENTILE}_*.csv" | wc -l | tr -d ' ')
  echo "Pair file count: ${COUNT}"

  if [ "$COUNT" = "0" ]; then
    echo "ERROR: no pair files found for ${PRODUCT}"
    exit 1
  fi
done

echo
echo "============================================================"
echo "Cleaning previous outputs for estimator=${ESTIMATOR}"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  echo "Cleaning ${PRODUCT}"

  rm -rf "data/products/${PRODUCT}/sensitivity/${PERCENTILE}/zeta_station/${ESTIMATOR}"
  rm -rf "data/products/${PRODUCT}/sensitivity/${PERCENTILE}/zeta_grid/${ESTIMATOR}"
  rm -rf "data/products/${PRODUCT}/sensitivity/${PERCENTILE}/annual_max_corrected/${ESTIMATOR}"
  rm -rf "data/products/${PRODUCT}/sensitivity/${PERCENTILE}/diagnostics/${ESTIMATOR}"
  rm -rf "figures/diagnostics/${PRODUCT}/${PERCENTILE}/${ESTIMATOR}"
done

echo
echo "============================================================"
echo "Step 1: compute station zeta"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Computing zeta: ${PRODUCT}"
  echo "------------------------------------------------------------"

  python3 run_pipeline.py compute-zeta \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --estimator "$ESTIMATOR" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR"
done

echo
echo "============================================================"
echo "Step 2: interpolate zeta"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Interpolating zeta: ${PRODUCT}"
  echo "------------------------------------------------------------"

  python3 run_pipeline.py interpolate-zeta \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --estimator "$ESTIMATOR"
done

echo
echo "============================================================"
echo "Step 3: apply bias correction to annual-maximum rasters"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Applying bias correction: ${PRODUCT}"
  echo "------------------------------------------------------------"

  python3 run_pipeline.py apply-bias \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --estimator "$ESTIMATOR"
done

echo
echo "============================================================"
echo "Step 4: run diagnostics"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Diagnostics: ${PRODUCT}"
  echo "------------------------------------------------------------"

  python3 run_pipeline.py diagnostics \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --estimator "$ESTIMATOR"
done

echo
echo "============================================================"
echo "Step 5: check outputs"
echo "============================================================"

for PRODUCT in "${PRODUCTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Output check: ${PRODUCT}"
  echo "------------------------------------------------------------"

  python3 check_outputs.py \
    --product "$PRODUCT" \
    --percentile "$PERCENTILE" \
    --estimator "$ESTIMATOR"
done

echo
echo "============================================================"
echo "Step 6: make summary table"
echo "============================================================"

python3 - <<'PY'
import pandas as pd
from pathlib import Path

products = ["imerg_v07", "imerg_v06", "chirps", "persiann_cdr", "br_dwgd"]
percentile = "p98"
estimator = "mean"

rows = []

for product in products:
    summary_path = Path(
        f"data/products/{product}/sensitivity/{percentile}/annual_max_corrected/{estimator}/"
        f"annual_max_correction_summary_{product}_{percentile}_{estimator}.csv"
    )
    zeta_path = Path(
        f"data/products/{product}/sensitivity/{percentile}/zeta_station/{estimator}/"
        f"zeta_per_station_{product}_{percentile}_{estimator}.csv"
    )

    if not summary_path.exists():
        print("Missing correction summary:", summary_path)
        continue

    s = pd.read_csv(summary_path)
    z = pd.read_csv(zeta_path) if zeta_path.exists() else pd.DataFrame()

    rows.append({
        "product": product,
        "estimator": estimator,
        "n_corrected_years": len(s),
        "first_year": int(s["year"].min()),
        "last_year": int(s["year"].max()),
        "n_zeta_stations": z["station_id"].nunique() if len(z) else 0,
        "median_station_zeta": z["zeta_selected"].median() if len(z) else None,
        "mean_station_zeta": z["zeta_selected"].mean() if len(z) else None,
        "mean_raw_annual_max": s["raw_mean"].mean(),
        "mean_corrected_annual_max": s["corrected_mean"].mean(),
        "mean_corrected_over_raw": (s["corrected_mean"] / s["raw_mean"]).mean(),
        "max_raw_annual_max": s["raw_max"].max(),
        "max_corrected_annual_max": s["corrected_max"].max(),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))

out_path = Path(f"data/products/{percentile}_bias_correction_summary_all_products_{estimator}_old_method.csv")
out.to_csv(out_path, index=False)
print("\nSaved:", out_path)
PY

echo
echo "============================================================"
echo "Step 7: inspect corrected maximum pixels"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path

src = Path("inspect_corrected_max_pixels.py")
dst = Path("inspect_corrected_max_pixels_mean.py")

txt = src.read_text()
txt = txt.replace('ESTIMATOR = "median"', 'ESTIMATOR = "mean"')
dst.write_text(txt)

print("Created:", dst)
PY

python3 inspect_corrected_max_pixels_mean.py

echo
echo "============================================================"
echo "Rerun complete."
echo "============================================================"
