#!/bin/bash
set -euo pipefail

# ============================================================
# GRIDF backup + staged cleanup preparation
# ============================================================
# This script:
# 1. Creates a complete backup of the current GRIDF folder.
# 2. Creates an external folder for large/generated outputs.
# 3. Moves selected large output folders out of the GitHub repo.
# 4. Does NOT delete the backup.
#
# Original project:
#   /Users/mngomes/Documents/GitHub/GRIDF
#
# Backup:
#   /Users/mngomes/Documents/GRIDF_backup_YYYYMMDD_HHMMSS
#
# External outputs:
#   /Users/mngomes/Documents/GRIDF_large_outputs_YYYYMMDD_HHMMSS
# ============================================================

ROOT="/Users/mngomes/Documents/GitHub/GRIDF"
DOCS="/Users/mngomes/Documents"
STAMP=$(date +"%Y%m%d_%H%M%S")

BACKUP_DIR="${DOCS}/GRIDF_backup_${STAMP}"
EXTERNAL_DIR="${DOCS}/GRIDF_large_outputs_${STAMP}"

echo "============================================================"
echo "GRIDF BACKUP + STAGED CLEANUP"
echo "============================================================"
echo "Root folder     : ${ROOT}"
echo "Backup folder   : ${BACKUP_DIR}"
echo "External outputs: ${EXTERNAL_DIR}"
echo "============================================================"

if [ ! -d "$ROOT" ]; then
    echo "ERROR: ROOT folder does not exist:"
    echo "$ROOT"
    exit 1
fi

echo ""
echo "Step 1/4: Creating full backup..."
echo "This may take a few minutes."
mkdir -p "$BACKUP_DIR"

rsync -a --progress \
    "$ROOT/" \
    "$BACKUP_DIR/"

echo ""
echo "Backup completed:"
du -sh "$BACKUP_DIR"

echo ""
echo "Step 2/4: Creating external output folder..."
mkdir -p "$EXTERNAL_DIR"

echo ""
echo "Step 3/4: Moving large/generated outputs outside GitHub repo..."
echo "The internal folder structure will be preserved."

move_if_exists () {
    local relpath="$1"
    local src="${ROOT}/${relpath}"
    local dst="${EXTERNAL_DIR}/${relpath}"

    if [ -e "$src" ]; then
        echo ""
        echo "Moving:"
        echo "  FROM: $src"
        echo "  TO  : $dst"

        mkdir -p "$(dirname "$dst")"
        mv "$src" "$dst"
    else
        echo ""
        echo "Skipping missing path:"
        echo "  $src"
    fi
}

# ============================================================
# Large generated IDF fitting outputs
# ============================================================
move_if_exists "IDF_Fitting/Outputs"
move_if_exists "IDF_Fitting/Percentile_Sensitivity"
move_if_exists "IDF_Fitting/Distribution_Diagnostics"
move_if_exists "IDF_Fitting/logs"

# ============================================================
# Bias-correction generated data and figures
# Keep Bias_Correction_Pipeline/src and config in GitHub.
# Move generated data/figures/logs out.
# ============================================================
move_if_exists "Bias_Correction_Pipeline/data/products"
move_if_exists "Bias_Correction_Pipeline/figures"
move_if_exists "Bias_Correction_Pipeline/logs"
move_if_exists "Bias_Correction_Pipeline/metadata/gee_tasks"

# ============================================================
# Older/large bias-correction working folder
# This appears to contain large ANA rainfall intermediate files.
# ============================================================
move_if_exists "Bias_Correction/ana_rainfall_1995_2025"

# ============================================================
# Large generated figures and legacy comparison outputs
# ============================================================
move_if_exists "Figures"
move_if_exists "Existing_IDFs/Figures_Spatial_Bias"
move_if_exists "Existing_IDFs/Figures_IDF_Curves_Comparison"
move_if_exists "Existing_IDFs/Figures_IDF_Curves_Comparison_XLSX"
move_if_exists "Existing_IDFs/Figures_IDF_Curves_Excel_BR_DWGD_Rasters"

# ============================================================
# Large final products that are better distributed externally
# Keep them outside the source-code repo and document them.
# ============================================================
move_if_exists "Zonal_Disaggregation_Coefficients_FINAL"
move_if_exists "IDF_Parameters_Bias_Corrected"
move_if_exists "IDF_Parameters_Raw_Data"
move_if_exists "Annual_Maximum_Precipitation"

# ============================================================
# Other generated/legacy outputs
# ============================================================
move_if_exists "GEV_Gumbel_RE_outputs"
move_if_exists "GEV_Gumbel_SA"
move_if_exists "_project_audit"
move_if_exists "Bias_Correction_Pipeline_cleanup_archive_20260510_212257"

# IDE/cache folder
move_if_exists ".idea"

echo ""
echo "Step 4/4: Size check after moving outputs"
echo "============================================================"
echo "Original cleaned repo:"
du -sh "$ROOT"

echo ""
echo "Backup:"
du -sh "$BACKUP_DIR"

echo ""
echo "External outputs:"
du -sh "$EXTERNAL_DIR"

echo ""
echo "Top-level cleaned repo structure:"
find "$ROOT" -maxdepth 2 -type d | sort

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "A complete backup is stored at:"
echo "$BACKUP_DIR"
echo ""
echo "Large/generated outputs were moved to:"
echo "$EXTERNAL_DIR"
echo ""
echo "Next step:"
echo "  1. Inspect the cleaned GRIDF folder."
echo "  2. If everything looks correct, we will create a .gitignore."
echo "  3. Then we will reorganize code/docs for GitHub."
echo "============================================================"