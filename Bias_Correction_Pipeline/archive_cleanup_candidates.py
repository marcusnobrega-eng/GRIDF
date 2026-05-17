#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE = ROOT.parent / f"Bias_Correction_Pipeline_cleanup_archive_{STAMP}"

df = pd.read_csv(ROOT / "cleanup_inventory.csv")

classes_to_archive = {
    "ARCHIVE_SUPERSEDED_SCRIPT",
    "ARCHIVE_OR_DELETE_BACKUP",
}

to_archive = df[df["classification"].isin(classes_to_archive)].copy()

print("Archive folder:")
print(ARCHIVE)
print()
print("Files/folders to archive:", len(to_archive))

ARCHIVE.mkdir(parents=True, exist_ok=True)

for _, row in to_archive.iterrows():
    src = ROOT / row["path"]
    dst = ARCHIVE / row["path"]

    if not src.exists():
        continue

    dst.parent.mkdir(parents=True, exist_ok=True)

    print("MOVE:", row["path"])

    if src.is_dir():
        shutil.move(str(src), str(dst))
    else:
        shutil.move(str(src), str(dst))

print()
print("Done. Nothing was permanently deleted.")
print("Archive is here:")
print(ARCHIVE)
