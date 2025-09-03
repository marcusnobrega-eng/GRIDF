# -*- coding: utf-8 -*-
"""
Download raw DAILY rainfall (ANA) and export a WIDE CSV:
[metadata..., 1/1/1995, 1/2/1995, 1/3/1995, ...].

- Reads your inventory CSV (prefiltered).
- (Optional) keeps only ANA-owned pluviometric stations.
- Normalizes station codes to 8 digits.
- Bulk-downloads daily rainfall with threading.
- Crops to [PERIOD_START, PERIOD_END], coerces to numeric, clips negatives to 0.
- Writes ONE header row: metadata columns + M/D/YYYY daily columns.
- Streams rows per station to avoid huge memory spikes.
"""

from hydrobr.get_data import ANA
import pandas as pd
import numpy as np
import csv, os, re
from typing import List

# ========== USER SETTINGS ==========
CSV_PATH = "stations_inventory_filtered_all.csv"   # your inventory file
OUTFILE  = "rainfall_timeseries_with_metadata_all.csv"

# Time window (inclusive)
PERIOD_START = "1994-01-01"
PERIOD_END   = "2020-01-01"

# ANA.prec options
ONLY_CONSISTED = True
BATCH_SIZE = 500
THREADS    = 36

# Keep only these owners and station types (recommended)
RESPONSIBLE_ALLOWED = {"ANA"}   # comment out to allow all
TYPE_ALLOWED        = {2}       # 2 = pluviometric (rainfall)
# ===================================

# -------- helpers --------
def normalize_code8(x) -> str:
    """Return an 8-digit ANA code string (e.g., '47002.0' -> '00047002')."""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(8)

def fmt_meta(val, colname: str):
    """Pretty-print metadata for CSV."""
    # US-style dates for Start/End (to match your example)
    if colname in {"StartDate", "EndDate"}:
        try:
            d = pd.to_datetime(val)
            return f"{d.month}/{d.day}/{d.year}"
        except Exception:
            return "" if pd.isna(val) else str(val)

    if isinstance(val, (pd.Timestamp, np.datetime64)):
        try:
            d = pd.to_datetime(val)
            return f"{d.month}/{d.day}/{d.year}"
        except Exception:
            return str(val)

    if colname.lower() in {"latitude", "longitude"} and isinstance(val, (int, float, np.floating)):
        return f"{float(val):.6f}"

    if isinstance(val, (int, np.integer)):
        return str(int(val))

    if isinstance(val, (float, np.floating)):
        s = f"{float(val):.2f}"
        s = s.rstrip("0").rstrip(".")
        return s

    return "" if pd.isna(val) else str(val)

def fmt_val(v):
    """Format daily rainfall cell: '' for NaN, else up to 2 decimals (trim zeros)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    s = f"{float(v):.2f}"
    return s.rstrip("0").rstrip(".")

def us_date_label(d: pd.Timestamp) -> str:
    """M/D/YYYY (no leading zeros) header label for a date."""
    return f"{d.month}/{d.day}/{d.year}"

# -------- read inventory (robust to encoding) --------
try:
    inv = pd.read_csv(CSV_PATH, dtype={"Code": str})
except UnicodeDecodeError:
    inv = pd.read_csv(CSV_PATH, dtype={"Code": str}, encoding="latin-1")

# Optional filters to keep only ANA pluviometric
if "Responsible" in inv.columns:
    inv = inv[inv["Responsible"].astype(str).str.upper().isin(RESPONSIBLE_ALLOWED) | inv["Responsible"].isna()]
if "Type" in inv.columns:
    inv = inv[pd.to_numeric(inv["Type"], errors="coerce").fillna(-1).astype(int).isin(TYPE_ALLOWED) | inv["Type"].isna()]

# Normalize codes, clean coords
inv["Code8"] = inv["Code"].map(normalize_code8)
if "Latitude" in inv.columns:
    inv["Latitude"]  = pd.to_numeric(inv["Latitude"], errors="coerce")
if "Longitude" in inv.columns:
    inv["Longitude"] = pd.to_numeric(inv["Longitude"], errors="coerce")

inv = inv.dropna(subset=["Code8"]).drop_duplicates(subset=["Code8"]).reset_index(drop=True)

# Metadata columns (use your preferred order if present)
preferred_meta = [
    "Code", "Name", "City", "State",
    "Latitude", "Longitude",
    "StartDate", "EndDate",
    "NYD", "MD", "N_YWOMD", "YWMD",
]
present_pref = [c for c in preferred_meta if c in inv.columns]
extra_meta = [c for c in inv.columns if c not in present_pref + ["Code8"]]
meta_cols = present_pref + [c for c in extra_meta if c not in present_pref]

# Metadata table indexed by Code8
meta_table = inv.set_index("Code8")[meta_cols].copy()
codes_all: List[str] = meta_table.index.tolist()

if not codes_all:
    raise SystemExit("No stations to request. Check filters and CSV columns.")

# -------- date grid & header labels --------
period_start = pd.to_datetime(PERIOD_START)
period_end   = pd.to_datetime(PERIOD_END)
all_days = pd.date_range(start=period_start, end=period_end, freq="D")
date_labels = [us_date_label(d) for d in all_days]   # e.g., 1/1/1995

# -------- prepare output CSV (UTF-8 BOM for Excel accents) --------
header = meta_cols + date_labels
wrote_header = False
codes_written = set()

missing_any_data = []  # codes we asked for but got no rows (in full window)

with open(OUTFILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # Write one header row
    writer.writerow(header)
    wrote_header = True

    # -------- download & write in batches --------
    total_batches = int(np.ceil(len(codes_all) / BATCH_SIZE))
    for i in range(0, len(codes_all), BATCH_SIZE):
        batch_codes = codes_all[i:i+BATCH_SIZE]
        print(f"Batch {i//BATCH_SIZE + 1}/{total_batches} — requesting {len(batch_codes)} stations...")

        try:
            dfi = ANA.prec(batch_codes, only_consisted=ONLY_CONSISTED, threads=THREADS)
        except Exception as e:
            print(f"  -> batch failed: {e}")
            dfi = None

        if dfi is None or dfi.empty:
            print("  -> no data returned for this batch.")
            missing_any_data.extend(batch_codes)
            continue

        # Clean & crop
        dfi = dfi.apply(pd.to_numeric, errors="coerce").clip(lower=0)
        # Crop to window & ensure full index for reindex
        dfi = dfi[(dfi.index >= period_start) & (dfi.index <= period_end)]

        # Make sure we have exactly the full date grid (fill gaps with NaN)
        dfi = dfi.reindex(all_days)

        # Ensure column names are Code8 strings
        dfi.columns = [normalize_code8(c) for c in dfi.columns]

        # Write one row per station in this batch
        for code8 in batch_codes:
            if code8 in codes_written:
                continue  # safety

            # Collect metadata
            if code8 not in meta_table.index:
                continue
            meta_vals = [fmt_meta(meta_table.at[code8, c], c) if c in meta_table.columns else "" for c in meta_cols]

            # Collect daily values for this station
            if code8 not in dfi.columns:
                # No data at all in window — leave blank cells for days
                row_vals = [""] * len(all_days)
                missing_any_data.append(code8)
            else:
                s = dfi[code8].to_numpy()  # aligned to all_days by reindex
                row_vals = [fmt_val(v) for v in s]

            writer.writerow(meta_vals + row_vals)
            codes_written.add(code8)

# Diagnostics file for stations with no data in window
if missing_any_data:
    pd.Series(sorted(set(missing_any_data)), name="Code8").to_csv("missing_no_data_codes.csv", index=False, encoding="utf-8-sig")

print(f"\nWrote {len(codes_written)} stations to: {os.path.abspath(OUTFILE)}")
if missing_any_data:
    print(f"Stations with no daily data in the window: {len(set(missing_any_data))} (see missing_no_data_codes.csv)")
