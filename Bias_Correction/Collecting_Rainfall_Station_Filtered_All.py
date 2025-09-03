from hydrobr.get_data import ANA
import pandas as pd
import numpy as np
import csv
import os
import re

# ===== USER INPUT =====
STATE = "Paraná"
CITY  = "Londrina"
ONLY_CONSISTED = True            # use ANA “consisted” data

PERIOD_START = "1995-01-01"
PERIOD_END   = "2025-01-01"
MIN_YEARS_IN_PERIOD = 15         # min years of daily data in the window
OUTFILE = "annual_max_layout_like_example.csv"
# ======================

period_start = pd.to_datetime(PERIOD_START)
period_end   = pd.to_datetime(PERIOD_END)
years = list(range(period_start.year, period_end.year + 1))
min_days_required = int(MIN_YEARS_IN_PERIOD * 365)  # simple threshold

def _slug(s: str) -> str:
    s = s or ""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip()) or "all"

# 1) List stations
stations = ANA.list_prec(state=STATE, city=CITY, source="ANA")
if stations is None or stations.empty:
    raise SystemExit(f"No precipitation stations found for {CITY}, {STATE}.")

# keep coords
need_cols = {"Latitude", "Longitude", "Code", "Name"}
missing = need_cols - set(stations.columns)
if missing:
    raise SystemExit(f"Stations table missing columns: {missing}")

stations = stations.dropna(subset=["Latitude", "Longitude"]).copy()
stations["Code"] = stations["Code"].astype(str)

rows = []   # each row: [lat, lon, v2000, v2001, ...]
kept = 0
skipped = 0

for _, r in stations.iterrows():
    code = r["Code"]
    name = r.get("Name", code)
    lat  = float(r["Latitude"])
    lon  = float(r["Longitude"])

    # Fetch daily data safely
    try:
        df = ANA.prec([code], only_consisted=ONLY_CONSISTED)
    except Exception as e:
        print(f"Skipping {code} ({name}) due to error: {e}")
        skipped += 1
        continue

    if df is None or df.empty or code not in df.columns:
        print(f"Skipping {code} ({name}) – empty frame or missing column.")
        skipped += 1
        continue

    s = pd.to_numeric(df[code], errors="coerce").clip(lower=0)
    s = s[(s.index >= period_start) & (s.index <= period_end)]

    non_na_days = int(s.notna().sum())
    if non_na_days < min_days_required:
        print(f"Skipping {code} ({name}) – coverage {non_na_days} < {min_days_required} days.")
        skipped += 1
        continue

    # Annual max (calendar years end in Dec)
    amax = s.resample("A-DEC").max()  # index at Dec 31 of each year

    # Build row with formatted values ("" for missing)
    vals = []
    for y in years:
        v = float(amax.get(pd.Timestamp(y, 12, 31), np.nan))
        vals.append("" if not np.isfinite(v) else f"{v:.2f}")

    rows.append([f"{lat:.6f}", f"{lon:.6f}", *vals])
    kept += 1

if not rows:
    raise SystemExit(
        f"No stations met the coverage requirement (≥{MIN_YEARS_IN_PERIOD} years of data) "
        f"in [{PERIOD_START} .. {PERIOD_END}] for {CITY}, {STATE}."
    )

# 2) Write CSV exactly like your example (two header rows)
with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    # Header row 1
    w.writerow(["", "Year", *years])
    # Header row 2
    w.writerow(["Latitude", "Longitude", *(["P (mm)"] * len(years))])
    # Data rows
    w.writerows(rows)

print(f"\nSaved {kept} stations (skipped {skipped}).")
print(f"Output: {os.path.abspath(OUTFILE)}")
