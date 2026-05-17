#!/usr/bin/env bash
set -euo pipefail

DRIVE_ROOT="/Users/mngomes/Library/CloudStorage/GoogleDrive-marcusep2025@gmail.com/My Drive"
PIPELINE_ROOT="/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline"

PRODUCTS=(
  imerg_v07
  imerg_v06
  chirps
  persiann_cdr
  br_dwgd
)

echo "Searching GEE export folders in:"
echo "$DRIVE_ROOT"
echo

for PRODUCT in "${PRODUCTS[@]}"; do
  echo "============================================================"
  echo "Product: $PRODUCT"
  echo "============================================================"

  PAIR_DIR="$PIPELINE_ROOT/data/products/$PRODUCT/sensitivity/p98/pairs"
  mkdir -p "$PAIR_DIR"

  FOUND_FOLDER=0
  COPIED_TOTAL=0

  while IFS= read -r -d '' FOLDER; do
    FOUND_FOLDER=1

    echo "Found Drive folder:"
    echo "  $FOLDER"

    FOUND_CSV=0

    while IFS= read -r -d '' CSV; do
      FOUND_CSV=1
      BASENAME="$(basename "$CSV")"

      echo "  Copying: $BASENAME"
      cp "$CSV" "$PAIR_DIR/$BASENAME"

      COPIED_TOTAL=$((COPIED_TOTAL + 1))
    done < <(find "$FOLDER" -maxdepth 1 -type f -name "pairs_${PRODUCT}_p98_*.csv" -print0 | sort -z)

    if [ "$FOUND_CSV" -eq 0 ]; then
      echo "  No matching CSVs in this folder."
    fi

    echo
  done < <(find "$DRIVE_ROOT" -maxdepth 1 -type d -name "GRIDF_BiasCorrection_pairs_${PRODUCT}_p98*" -print0 | sort -z)

  if [ "$FOUND_FOLDER" -eq 0 ]; then
    echo "No Drive folder found for $PRODUCT p98."
  fi

  echo "Copied total for $PRODUCT: $COPIED_TOTAL file(s)"
  echo "Local folder:"
  echo "  $PAIR_DIR"
  echo

  find "$PAIR_DIR" -maxdepth 1 -type f -name "pairs_${PRODUCT}_p98_*.csv" -print | sort || true
  echo
done

echo "============================================================"
echo "Copy complete."
echo "============================================================"
