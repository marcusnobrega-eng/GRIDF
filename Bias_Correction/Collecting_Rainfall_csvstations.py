
# -*- coding: utf-8 -*-
"""
Generate a CSV table of annual maximum daily rainfall for multiple ANA precipitation stations.

This script:
1. Reads a pre-filtered inventory of ANA precipitation stations (CSV) that already meet
   basic quality criteria (produced by a separate "inventory filter" step).
2. Iterates through each station and:
   - Downloads daily rainfall data from ANA via hydrobr.get_data.ANA.
   - Normalizes the station code (removing any trailing '.0' from floats).
   - Restricts the data to a specified period (PERIOD_START to PERIOD_END).
   - Checks if the station has at least MIN_YEARS_IN_PERIOD years of data coverage within that period.
   - Computes the annual maximum daily rainfall for each calendar year in the window.
3. Stores results in a "layout-like-example.csv" table, with:
   - First two columns: Latitude and Longitude of the station.
   - One column per year in the target period, containing the annual max value (mm) or blank if missing.
   - Two header rows: 
       Row 1 = empty cell, "Year", list of years.
       Row 2 = "Latitude", "Longitude", and "P (mm)" repeated for each year.

Notes:
- ONLY_CONSISTED controls whether to use ANA's "consisted" (QC-passed) data only.
- MIN_YEARS_IN_PERIOD and min_days_required set the coverage threshold for station inclusion.
- Missing annual maxima for a year are stored as empty strings.
- Output CSV has exactly the format required for further processing or mapping.

Dependencies:
- hydrobr (to fetch ANA data)
- pandas, numpy, csv, re, os

Example use:
- Prepare your filtered inventory: stations_inventory_filtered_test.csv
- Adjust PERIOD_START, PERIOD_END, and thresholds
- Run the script to get annual_max_layout_like_example.csv
"""


# -*- coding: utf-8 -*-
from hydrobr.get_data import ANA
import pandas as pd
import numpy as np
import csv, os, re

# ===== USER INPUT =====
CSV_PATH = "stations_inventory_filtered.csv"   # from your inventory filter
ONLY_CONSISTED = True                               # use ANA "consisted" data
PERIOD_START = "1995-01-01"
PERIOD_END   = "2020-01-01"
MIN_YEARS_IN_PERIOD = 0                             # min years of daily data in the window
OUTFILE = "annual_maximum_selected_stations.csv"

# Bulk-download tuning
BATCH_SIZE = 400                                    # number of stations per request
THREADS    = 12                                     # passed to ANA.prec(..., threads=THREADS)
# ======================

# Time window / years
period_start = pd.to_datetime(PERIOD_START)
period_end   = pd.to_datetime(PERIOD_END)
years = list(range(period_start.year, period_end.year + 1))
min_days_required = int(MIN_YEARS_IN_PERIOD * 365)

def normalize_code8(x) -> str:
    """Return an 8-digit ANA code string (e.g., '47002.0' -> '00047002')."""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)      # drop trailing .0 from floats
    return s.zfill(8)               # ANA often zero-pads to 8

# 1) Read station list and build code→lat/lon map
inv = pd.read_csv(CSV_PATH, dtype={"Code": str})
need_cols = {"Code", "Name", "Latitude", "Longitude"}
missing = need_cols - set(inv.columns)
if missing:
    raise SystemExit(f"Input CSV missing columns: {missing}")

inv["Code8"] = inv["Code"].map(normalize_code8)
inv["Latitude"]  = pd.to_numeric(inv["Latitude"], errors="coerce")
inv["Longitude"] = pd.to_numeric(inv["Longitude"], errors="coerce")
inv = inv.dropna(subset=["Latitude", "Longitude", "Code8"]).drop_duplicates(subset=["Code8"]).reset_index(drop=True)

codes_all = inv["Code8"].tolist()
lat_map = dict(zip(inv["Code8"], inv["Latitude"]))
lon_map = dict(zip(inv["Code8"], inv["Longitude"]))

print(f"Total stations to request: {len(codes_all)} (batch size={BATCH_SIZE}, threads={THREADS})")

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

# Merge columns from all batches
df_all = pd.concat(dfs, axis=1)
# Keep only unique columns (some batches might repeat)
df_all = df_all.loc[:, ~df_all.columns.duplicated()].sort_index()

# 3) Crop to analysis window and clean
df_all = df_all[(df_all.index >= period_start) & (df_all.index <= period_end)]
# Numeric + nonnegative
df_all = df_all.apply(pd.to_numeric, errors="coerce").clip(lower=0)

# 4) Coverage filter per station (within window)
non_na_days = df_all.notna().sum(axis=0)  # per column
keep_cols = non_na_days[non_na_days >= min_days_required].index.tolist()

if not keep_cols:
    raise SystemExit("No station meets the coverage requirement within the window.")

print(f"Stations meeting coverage (≥{min_days_required} days): {len(keep_cols)}")

# 5) Annual maxima (vectorized) for kept stations
amax_df = df_all[keep_cols].resample("YE-DEC").max()   # rows = Dec-31 of each year, cols = station codes

# 6) Build output rows [lat, lon, y1, y2, ...] for each kept station
rows, kept = [], 0
for code in keep_cols:
    lat = lat_map.get(code, np.nan)
    lon = lon_map.get(code, np.nan)
    if not np.isfinite(lat) or not np.isfinite(lon):
        continue

    vals = []
    for y in years:
        ts = pd.Timestamp(y, 12, 31)
        if (ts in amax_df.index) and (code in amax_df.columns):
            val = amax_df.loc[ts, code]
        else:
            val = np.nan
        vals.append("" if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val))
                    else f"{float(val):.2f}")
    rows.append([f"{lat:.6f}", f"{lon:.6f}", *vals])
    kept += 1

if not rows:
    raise SystemExit("After merging and coverage filtering, no rows remained for export.")

# 7) Write CSV with two header rows
with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["", "Year", *years])                      # header row 1
    w.writerow(["Latitude", "Longitude", *(["P (mm)"] * len(years))])  # header row 2
    w.writerows(rows)

print(f"\nSaved {kept} stations to {os.path.abspath(OUTFILE)}")
