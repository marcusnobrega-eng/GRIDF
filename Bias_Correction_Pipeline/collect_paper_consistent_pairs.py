#!/usr/bin/env python3
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from datetime import datetime

DRIVE_ROOT = Path("/Users/mngomes/Library/CloudStorage/GoogleDrive-marcusep2025@gmail.com/My Drive")
PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")

PERCENTILE = "p98"

PRODUCTS = {
    "br_dwgd": {
        "drive_folder": "GRIDF_paper_consistent_br_dwgd_p98_1995_2006",
        "pattern": "pairs_br_dwgd_p98_*.csv",
        "expected_years": list(range(1995, 2007)),
    },
    "chirps": {
        "drive_folder": "GRIDF_paper_consistent_chirps_p98_1995_2006",
        "pattern": "pairs_chirps_p98_*.csv",
        "expected_years": list(range(1995, 2007)),
    },
    "persiann_cdr": {
        "drive_folder": "GRIDF_paper_consistent_persiann_cdr_p98_1995_2006",
        "pattern": "pairs_persiann_cdr_p98_*.csv",
        "expected_years": list(range(1995, 2007)),
    },
    "imerg_v07": {
        "drive_folder": "GRIDF_paper_consistent_imerg_v07_p98_2001_2006",
        "pattern": "pairs_imerg_v07_p98_*.csv",
        "expected_years": list(range(2001, 2007)),
    },
}

# Existing legacy folder for IMERG V06.
IMERG_V06_LEGACY_FOLDER = DRIVE_ROOT / "imerg_bias_pairs"


def year_from_name(name: str):
    import re
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None


def backup_local(product: str):
    local = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / PERCENTILE / "pairs"
    local.mkdir(parents=True, exist_ok=True)

    existing = sorted(local.glob(f"pairs_{product}_{PERCENTILE}_*.csv"))
    if existing:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = local.parent / f"pairs_backup_before_paper_consistent_{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        for f in existing:
            shutil.copy2(f, backup / f.name)
        print(f"Backed up {product} pairs to: {backup}")

    for f in existing:
        f.unlink()

    return local


def copy_product(product: str, info: dict):
    print("\n" + "=" * 90)
    print(product)
    print("=" * 90)

    local = backup_local(product)

    drive_dir = DRIVE_ROOT / info["drive_folder"]
    if not drive_dir.exists():
        raise FileNotFoundError(f"Drive folder not found: {drive_dir}")

    files = sorted(drive_dir.glob(info["pattern"]))
    print("Drive folder:", drive_dir)
    print("Files found:", len(files))

    for f in files:
        print(" ", f.name)
        shutil.copy2(f, local / f.name)

    verify_product(product, info["expected_years"])


def standardize_legacy_imerg_v06():
    product = "imerg_v06"

    print("\n" + "=" * 90)
    print("imerg_v06 from legacy imerg_bias_pairs")
    print("=" * 90)

    local = backup_local(product)

    if not IMERG_V06_LEGACY_FOLDER.exists():
        raise FileNotFoundError(f"Legacy IMERG folder not found: {IMERG_V06_LEGACY_FOLDER}")

    files = sorted(IMERG_V06_LEGACY_FOLDER.glob("imerg_bias_pairs_*.csv"))
    files = [f for f in files if (year_from_name(f.name) is not None and 2001 <= year_from_name(f.name) <= 2006)]

    print("Legacy folder:", IMERG_V06_LEGACY_FOLDER)
    print("Files found:", len(files))

    for f in files:
        year = year_from_name(f.name)
        df = pd.read_csv(f, low_memory=False)

        if "imerg_mm" not in df.columns:
            raise ValueError(f"Missing imerg_mm in {f}")

        if "pr_g" not in df.columns:
            raise ValueError(f"Missing pr_g in {f}")

        df["year"] = year
        df["product_mm"] = pd.to_numeric(df["imerg_mm"], errors="coerce")
        df["gauge_mm"] = pd.to_numeric(df["pr_g"], errors="coerce")

        if "ratio" not in df.columns:
            df["ratio"] = df["gauge_mm"] / df["product_mm"]

        df["ratio_gauge_over_product"] = pd.to_numeric(df["ratio"], errors="coerce")

        out = local / f"pairs_imerg_v06_p98_{year}_chunk001.csv"
        df.to_csv(out, index=False)
        print(" ", f.name, "->", out.name, "rows=", len(df))

    verify_product(product, list(range(2001, 2007)))


def verify_product(product: str, expected_years: list[int]):
    pair_dir = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / PERCENTILE / "pairs"
    files = sorted(pair_dir.glob(f"pairs_{product}_{PERCENTILE}_*.csv"))

    years = []
    rows = 0
    stations = set()

    n_product_le1 = 0
    n_gauge_le1 = 0

    for f in files:
        df = pd.read_csv(f, low_memory=False)
        rows += len(df)

        if "date" in df.columns:
            yy = pd.to_datetime(df["date"], errors="coerce").dt.year.dropna().astype(int).unique()
            years.extend(yy.tolist())
        elif "year" in df.columns:
            yy = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique()
            years.extend(yy.tolist())

        if "station_id" in df.columns:
            stations.update(df["station_id"].astype(str).tolist())

        if "product_mm" in df.columns:
            p = pd.to_numeric(df["product_mm"], errors="coerce")
            n_product_le1 += int((p <= 1).sum())

        if "gauge_mm" in df.columns:
            g = pd.to_numeric(df["gauge_mm"], errors="coerce")
        elif "pr_g" in df.columns:
            g = pd.to_numeric(df["pr_g"], errors="coerce")
        else:
            g = pd.Series(dtype=float)

        if len(g):
            n_gauge_le1 += int((g <= 1).sum())

    years = sorted(set(years))
    missing = sorted(set(expected_years) - set(years))

    print("Local folder:", pair_dir)
    print("files:", len(files))
    print("rows:", rows)
    print("years:", years)
    print("missing expected years:", missing if missing else "none")
    print("stations:", len(stations))
    print("product <= 1 rows:", n_product_le1)
    print("gauge <= 1 rows:", n_gauge_le1)


def main():
    for product, info in PRODUCTS.items():
        copy_product(product, info)

    standardize_legacy_imerg_v06()

    print("\nAll paper-consistent pair files copied into pipeline folders.")


if __name__ == "__main__":
    main()
