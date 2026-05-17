#!/usr/bin/env bash
set -euo pipefail

DRIVE_ROOT="/Users/mngomes/Library/CloudStorage/GoogleDrive-marcusep2025@gmail.com/My Drive"
PIPELINE_ROOT="/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline"

PRODUCTS=(
  imerg_v06
  persiann_cdr
)

for PRODUCT in "${PRODUCTS[@]}"; do
  echo "============================================================"
  echo "Copying pair files for: ${PRODUCT}"
  echo "============================================================"

  PAIR_DIR="${PIPELINE_ROOT}/data/products/${PRODUCT}/sensitivity/p98/pairs"
  mkdir -p "$PAIR_DIR"

  COPIED=0
  FOUND_FOLDER=0

  while IFS= read -r -d '' FOLDER; do
    FOUND_FOLDER=1

    echo "Found Drive folder:"
    echo "  $FOLDER"

    while IFS= read -r -d '' CSV; do
      BASENAME="$(basename "$CSV")"
      echo "  Copying: $BASENAME"

      cp "$CSV" "$PAIR_DIR/$BASENAME"
      COPIED=$((COPIED + 1))
    done < <(find "$FOLDER" -maxdepth 1 -type f -name "pairs_${PRODUCT}_p98_*.csv" -print0 | sort -z)

    echo
  done < <(find "$DRIVE_ROOT" -maxdepth 1 -type d -name "GRIDF_BiasCorrection_pairs_${PRODUCT}_p98*" -print0 | sort -z)

  if [ "$FOUND_FOLDER" -eq 0 ]; then
    echo "WARNING: no Drive folder found for ${PRODUCT}."
  fi

  echo "Copied ${COPIED} file(s) for ${PRODUCT}."
  echo "Local folder:"
  echo "  $PAIR_DIR"
  echo

  find "$PAIR_DIR" -maxdepth 1 -type f -name "pairs_${PRODUCT}_p98_*.csv" -print | sort
  echo
done

echo "============================================================"
echo "Finished copying IMERG V06 and PERSIANN-CDR pair files."
echo "============================================================"
