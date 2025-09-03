# -*- coding: utf-8 -*-
"""
Download daily rainfall in batches, compute annual maxima (per calendar year),
and export a wide CSV that includes ALL station metadata + per-year maxima.

What this script does
---------------------
1) Reads your prefiltered inventory CSV (from the inventory step).
2) (Optional) Filters to ANA-owned pluviometric stations (Responsible == "ANA", Type == 2),
   because ANA.prec() only fetches from ANA/Hidroweb.
3) Normalizes station codes to 8 digits (left-padded with zeros) for compatibility.
4) Downloads daily data in batches with threading, crops to [PERIOD_START, PERIOD_END],
   converts to numeric, and keeps nonnegative values.
5) Drops stations that don't meet a simple coverage threshold inside the window
   (MIN_YEARS_IN_PERIOD * 365 valid daily values).
6) Computes annual maximum daily rainfall with Dec-year-end ("A-DEC").
7) Writes a CSV with TWO header rows:
   - Row 1: empty cells for metadata columns, then "Year", then the list of years.
   - Row 2: the metadata column names, then "P (mm)" repeated for each year.
   - Each data row: all metadata values for the station, followed by the annual maxima per year.
"""

from hydrobr.get_data import ANA
import pandas as pd
import numpy as np
import csv, os, re
from typing import List

# ===== USER INPUT =====
CSV_PATH = "stations_inventory_filtered.csv"   # your inventory/filter output
ONLY_CONSISTED = True
PERIOD_START = "1995-01-01"
PERIOD_END   = "2020-01-01"
MIN_YEARS_IN_PERIOD = 0                        # >= this many years of daily data in window

OUTFILE = "annual_maximum_selected_stations_with_metadata.csv"

# Bulk-download tuning
BATCH_SIZE = 500
THREADS    = 36

# Keep only these owners and station types from the CSV (recommended)
RESPONSIBLE_ALLOWED = {"ANA"}   # comment this out to allow all owners
TYPE_ALLOWED        = {2}       # 2 = pluviometric (rainfall)
# ======================

# Time window / years
period_start = pd.to_datetime(PERIOD_START)
period_end   = pd.to_datetime(PERIOD_END)
years: List[int] = list(range(period_start.year, period_end.year + 1))
min_days_required = int(MIN_YEARS_IN_PERIOD * 365)

def normalize_code8(x) -> str:
    """Return an 8-digit ANA code string (e.g., '47002.0' -> '00047002')."""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(8)

def fmt_meta(val, colname: str):
    """Nice formatting for common metadata types."""
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        try:
            return pd.to_datetime(val).strftime("%Y-%m-%d")
        except Exception:
            return str(val)
    if colname.lower() in {"latitude", "longitude"} and isinstance(val, (int, float, np.floating)):
        return f"{float(val):.6f}"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        # Keep reasonable precision but avoid scientific notation
        return f"{float(val):.6f}".rstrip("0").rstrip(".")
    return "" if pd.isna(val) else str(val)

# 1) Read station list (inventory)
inv = pd.read_csv(CSV_PATH, dtype={"Code": str})

# Optional diagnostics: show owners in the file
if "Responsible" in inv.columns:
    print("\nCounts by Responsible in input CSV:")
    print(inv["Responsible"].fillna("Unknown").value_counts().to_string())

# Optional filters to keep only ANA pluviometric stations
if "Responsible" in inv.columns:
    inv = inv[inv["Responsible"].astype(str).str.upper().isin(RESPONSIBLE_ALLOWED) | inv["Responsible"].isna()]
if "Type" in inv.columns:
    # Type sometimes is float in CSV; coerce to int for comparison
    inv = inv[pd.to_numeric(inv["Type"], errors="coerce").fillna(-1).astype(int).isin(TYPE_ALLOWED) | inv["Type"].isna()]

# Normalize codes and clean coords
inv["Code8"] = inv["Code"].map(normalize_code8)
inv["Latitude"]  = pd.to_numeric(inv["Latitude"], errors="coerce")
inv["Longitude"] = pd.to_numeric(inv["Longitude"], errors="coerce")
inv = inv.dropna(subset=["Latitude", "Longitude", "Code8"]).drop_duplicates(subset=["Code8"]).reset_index(drop=True)

# Build a metadata table keyed by Code8 (we'll export these columns)
# Preferred order if present; any extra columns will be appended afterward.
preferred_meta = [
    "Code", "Name",
    "City", "State",
    "Latitude", "Longitude",
    "StartDate", "EndDate",
    "NYD", "MD", "N_YWOMD", "YWMD"
]
present_pref = [c for c in preferred_meta if c in inv.columns]
extra_meta = [c for c in inv.columns if c not in present_pref + ["Code8"]]  # keep any other columns the CSV might carry
meta_cols = present_pref + [c for c in extra_meta if c not in present_pref]

# Index by Code8 for quick lookup
meta_table = inv.set_index("Code8")[meta_cols].copy()

# Maps for quick lat/lon access (still used below)
lat_map = meta_table["Latitude"].to_dict() if "Latitude" in meta_table.columns else {}
lon_map = meta_table["Longitude"].to_dict() if "Longitude" in meta_table.columns else {}

codes_all = meta_table.index.tolist()
print(f"\nTotal stations to request (after filters & cleaning): {len(codes_all)} "
      f"(batch size={BATCH_SIZE}, threads={THREADS})")

if not codes_all:
    raise SystemExit("No stations to request. Check filters and CSV columns.")

# 2) Bulk download in batches
dfs = []
for i in range(0, len(codes_all), BATCH_SIZE):
    batch = codes_all[i:i+BATCH_SIZE]
    print(f"Downloading batch {i//BATCH_SIZE + 1} of {int(np.ceil(len(codes_all)/BATCH_SIZE))} "
          f"({len(batch)} stations)...")
    try:
        dfi = ANA.prec(batch, only_consisted=ONLY_CONSISTED, threads=THREADS)
        if dfi is not None and not dfi.empty:
            dfs.append(dfi)
        else:
            print("  -> batch returned empty frame.")
    except Exception as e:
        print(f"  -> batch failed: {e}")

if not dfs:
    raise SystemExit("No data returned for any batch.")

# Merge columns from all batches, crop, clean
df_all = pd.concat(dfs, axis=1)
df_all = df_all.loc[:, ~df_all.columns.duplicated()].sort_index()
df_all = df_all[(df_all.index >= period_start) & (df_all.index <= period_end)]
df_all = df_all.apply(pd.to_numeric, errors="coerce").clip(lower=0)

# Diagnostics: which requested codes didn’t come back as columns?
present_cols = df_all.columns.astype(str).tolist()
missing_no_data = sorted(set(codes_all) - set(present_cols))
print(f"\nDownloaded columns from ANA: {len(present_cols)}")
print(f"Codes with NO data returned in window: {len(missing_no_data)} (saved to missing_no_data_codes.csv)")
pd.Series(missing_no_data, name="Code8").to_csv("missing_no_data_codes.csv", index=False)

# 3) Coverage filter within window
non_na_days = df_all.notna().sum(axis=0)  # per station
keep_cols = non_na_days[non_na_days >= min_days_required].index.tolist()
print(f"Stations meeting coverage (≥{min_days_required} days): {len(keep_cols)}")

if not keep_cols:
    raise SystemExit("No station meets the coverage requirement within the window.")

# 4) Annual maxima (vectorized) for kept stations
amax_df = df_all[keep_cols].resample("YE-DEC").max()   # Dec-31 index, columns = station codes

# Reindex to exactly the years we want (Dec-31 of each year)
ts_years = pd.to_datetime([f"{y}-12-31" for y in years])
amax_grid = amax_df.reindex(ts_years)

# 5) Build output rows: [all metadata..., AMAX_1995, AMAX_1996, ...]
rows, kept = [], 0
for code8 in keep_cols:
    if code8 not in meta_table.index:
        continue

    meta_vals = [fmt_meta(meta_table.at[code8, c], c) if c in meta_table.columns else "" for c in meta_cols]

    if code8 in amax_grid.columns:
        vals_arr = amax_grid[code8].to_numpy()
    else:
        vals_arr = np.full(len(ts_years), np.nan, dtype=float)

    year_vals = [
        "" if not (isinstance(v, (int, float, np.floating)) and np.isfinite(v))
        else f"{float(v):.2f}"
        for v in vals_arr
    ]

    rows.append([*meta_vals, *year_vals])
    kept += 1

if not rows:
    raise SystemExit("After coverage filtering, no rows remained for export.")

# 6) Write CSV with two header rows:
# Row 1: blanks for metadata columns, then "Year" + years
# Row 2: metadata column names, then "P (mm)" repeated
with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([""] * len(meta_cols) + ["Year", *years])
    w.writerow(meta_cols + (["P (mm)"] * len(years)))
    w.writerows(rows)

print(f"\nSaved {kept} stations to {os.path.abspath(OUTFILE)}")
