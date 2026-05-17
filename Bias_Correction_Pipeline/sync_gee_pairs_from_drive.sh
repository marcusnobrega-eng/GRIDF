#!/usr/bin/env bash
set -euo pipefail

GRIDF_ROOT="/Users/mngomes/Documents/GitHub/GRIDF"
LOCAL_ROOT="${GRIDF_ROOT}/Bias_Correction_Pipeline/data/products"

# Find likely Google Drive "My Drive" folders.
echo "Searching for Google Drive folders..."
DRIVE_CANDIDATES=$(find "$HOME/Library/CloudStorage" -type d -name "My Drive" 2>/dev/null || true)

if [ -z "$DRIVE_CANDIDATES" ]; then
  echo "ERROR: Could not find Google Drive 'My Drive' under ~/Library/CloudStorage"
  exit 1
fi

echo "Google Drive candidates:"
echo "$DRIVE_CANDIDATES"
echo

PRODUCTS=("br_dwgd" "chirps" "persiann_cdr" "imerg_v06" "imerg_v07")
PCTS=("p90" "p95" "p99" "p995")

for PRODUCT in "${PRODUCTS[@]}"; do
  for PCT in "${PCTS[@]}"; do

    FOLDER="GRIDF_BiasCorrection_pairs_${PRODUCT}_${PCT}"
    DST="${LOCAL_ROOT}/${PRODUCT}/sensitivity/${PCT}/pairs"

    mkdir -p "$DST"

    echo "============================================================"
    echo "PRODUCT=${PRODUCT} | PERCENTILE=${PCT}"
    echo "Looking for Drive folder: ${FOLDER}"
    echo "Destination: ${DST}"
    echo "============================================================"

    FOUND_ANY=0

    while IFS= read -r MYDRIVE; do
      # Exact folder or duplicate folder like "..._p90 (1)"
      while IFS= read -r SRC; do
        [ -z "$SRC" ] && continue

        FOUND_ANY=1

        echo "Found source:"
        echo "  $SRC"

        N_SRC=$(find "$SRC" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "  files in source: $N_SRC"

        if [ "$N_SRC" -gt 0 ]; then
          rsync -av --ignore-existing "$SRC"/ "$DST"/
        else
          echo "  WARNING: source folder is empty or not downloaded locally."
        fi

      done < <(find "$MYDRIVE" -type d \( -name "$FOLDER" -o -name "${FOLDER} (*)" \) 2>/dev/null)

    done <<< "$DRIVE_CANDIDATES"

    if [ "$FOUND_ANY" -eq 0 ]; then
      echo "WARNING: no Drive folder found for ${FOLDER}"
    fi

    N_DST=$(find "$DST" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "Files now in local destination: $N_DST"
    echo

  done
done

echo "Done syncing GEE pair folders."
