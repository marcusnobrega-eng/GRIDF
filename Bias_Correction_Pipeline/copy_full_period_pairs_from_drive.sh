#!/usr/bin/env bash
set -euo pipefail

cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

DRIVE_ROOT="/Users/mngomes/Library/CloudStorage/GoogleDrive-marcusep2025@gmail.com/My Drive"
PERCENTILE="p98"

copy_one () {
  PRODUCT="$1"
  START_YEAR="$2"
  END_YEAR="$3"

  DRIVE_DIR="${DRIVE_ROOT}/GRIDF_BiasCorrection_pairs_${PRODUCT}_${PERCENTILE}_${START_YEAR}_${END_YEAR}_fullGauge"
  PAIR_DIR="data/products/${PRODUCT}/sensitivity/${PERCENTILE}/pairs"
  BACKUP_DIR="data/products/${PRODUCT}/sensitivity/${PERCENTILE}/pairs_backup_before_${START_YEAR}_${END_YEAR}_fullGauge"

  echo
  echo "============================================================"
  echo "Product: ${PRODUCT}"
  echo "Drive:   ${DRIVE_DIR}"
  echo "Local:   ${PAIR_DIR}"
  echo "Backup:  ${BACKUP_DIR}"
  echo "============================================================"

  if [ ! -d "$DRIVE_DIR" ]; then
    echo "ERROR: Drive folder not found:"
    echo "$DRIVE_DIR"
    exit 1
  fi

  mkdir -p "$PAIR_DIR"

  if [ -d "$PAIR_DIR" ]; then
    rm -rf "$BACKUP_DIR"
    cp -R "$PAIR_DIR" "$BACKUP_DIR"
  fi

  rm -f "$PAIR_DIR"/pairs_"$PRODUCT"_"$PERCENTILE"_*.csv
  cp "$DRIVE_DIR"/pairs_"$PRODUCT"_"$PERCENTILE"_*.csv "$PAIR_DIR"/

  echo "Copied files:"
  find "$PAIR_DIR" -maxdepth 1 -type f -name "pairs_${PRODUCT}_${PERCENTILE}_*.csv" | sort
  echo "File count:"
  find "$PAIR_DIR" -maxdepth 1 -type f -name "pairs_${PRODUCT}_${PERCENTILE}_*.csv" | wc -l
}

copy_one br_dwgd       1995 2006
copy_one chirps        1995 2006
copy_one persiann_cdr  1995 2006
copy_one imerg_v06     2000 2006
copy_one imerg_v07     2000 2006

echo
echo "All full-period pair files copied."
