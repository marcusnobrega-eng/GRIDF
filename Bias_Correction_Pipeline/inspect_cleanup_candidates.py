#!/usr/bin/env python3
from pathlib import Path
import os
import csv

ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")

KEEP_EXACT = {
    "run_pipeline.py",
    "check_outputs.py",
    "build_legacy_equivalent_zeta_from_pairs.py",
    "interpolate_zeta_legacy_equivalent_from_pipeline.py",
    "plot_bias_correction_legacy_style_5x3_helvetica_biomes.py",
    "run_final_paper_bias_workflow.sh",
}

KEEP_PREFIXES = [
    "src/",
    "config/",
    "data/products/",
    "figures/bias_correction/legacy_style_5x3_helvetica_biomes/",
]

ARCHIVE_PATTERNS = [
    "patch_",
    "force_zeta_selected_to_slope0",
    "standardize_zeta_station_coordinates",
    "repair_zeta_coordinates_from_pairs",
    "compare_legacy_zeta_to_current_estimators",
    "legacy_export_pairs_old_style",
    "plot_bias_correction_legacy_style_from_pipeline",
    "plot_bias_correction_legacy_style_5x3_helvetica.py",
    "plot_bias_correction_figure_paper5",
    "submit_all_p98_pairs_again",
    "copy_all_p98_pairs_again",
    "collect_all_pairs_from_drive",
    "collect_pairs_from_duplicate_drive_folders",
    "rerun_p98_all_products_mean_median",
    "run_p98_",
]

DELETE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".DS_Store",
    ".backup",
    "_backup",
    "backup_before",
]

AUDIT_KEEP = {
    "diagnose_legacy_vs_current_bias.py",
}

def relpath(p):
    return str(p.relative_to(ROOT))

def size_mb(p):
    if p.is_dir():
        total = 0
        for q in p.rglob("*"):
            if q.is_file():
                total += q.stat().st_size
        return total / 1024 / 1024
    return p.stat().st_size / 1024 / 1024

def classify(p):
    r = relpath(p)

    if r in KEEP_EXACT:
        return "KEEP_FINAL"
    if r in AUDIT_KEEP:
        return "KEEP_AUDIT_OR_MOVE_TO_DEBUG"
    if any(r.startswith(pref) for pref in KEEP_PREFIXES):
        # Still flag backups inside kept data folders.
        if any(x in r for x in DELETE_PATTERNS):
            return "ARCHIVE_OR_DELETE_BACKUP"
        return "KEEP_DATA_OR_OUTPUT"
    if any(x in r for x in DELETE_PATTERNS):
        return "ARCHIVE_OR_DELETE_BACKUP"
    if p.name.startswith(tuple(ARCHIVE_PATTERNS)) or any(p.name.startswith(x) for x in ARCHIVE_PATTERNS):
        return "ARCHIVE_SUPERSEDED_SCRIPT"
    if p.suffix in [".py", ".sh"]:
        return "REVIEW_SCRIPT"
    return "REVIEW_OTHER"

rows = []

for p in sorted(ROOT.rglob("*")):
    # Skip hidden git internals
    if ".git" in p.parts:
        continue
    if p.is_file() or p.is_dir() and p.name == "__pycache__":
        rows.append({
            "path": relpath(p),
            "type": "dir" if p.is_dir() else "file",
            "size_mb": round(size_mb(p), 3),
            "classification": classify(p),
        })

out = ROOT / "cleanup_inventory.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["classification", "type", "size_mb", "path"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("Wrote:", out)
print()
print("Summary:")
summary = {}
for row in rows:
    summary[row["classification"]] = summary.get(row["classification"], 0) + 1
for k, v in sorted(summary.items()):
    print(f"{k:30s} {v}")

print()
print("Review likely garbage with:")
print("python3 - <<'PY'")
print("import pandas as pd")
print("df=pd.read_csv('cleanup_inventory.csv')")
print("print(df[df.classification.str.contains('ARCHIVE|DELETE|SUPERSEDED', regex=True)].to_string(index=False))")
print("PY")
