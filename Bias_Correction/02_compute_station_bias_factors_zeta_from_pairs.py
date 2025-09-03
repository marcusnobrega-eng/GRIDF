# 02_compute_station_bias_factors_zeta_from_pairs.py
# ---------------------------------------------------------
# Purpose:
#   Read yearly "chirps_bias_pairs_YYYY.csv" files produced by Phase A,
#   compute a robust station-level bias factor ζ from event-wise ratios
#   (default: median of ratio = pr_g / chirps_mm), and save:
#     - zeta_per_station.csv  (station_id, zeta, n_pairs [, lat, lon])
#
# Optional:
#   - Merge station coordinates from the wide ANA CSV (so Phase B can run directly).
#   - Alternate estimators for ζ (mean, or slope through origin).
#
# Requirements:
#   pip install pandas numpy
# ---------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd

# ----------------- USER INPUTS -----------------
# Folder containing the yearly CSVs exported by GEE (downloaded locally)
IN_DIR  = 'G:\My Drive\chirps_bias_pairs'   # e.g., ...\chirps_bias_pairs\
IN_GLOB = "chirps_bias_pairs_*.csv"        # pattern for yearly files

# Optional: merge lat/lon from your wide ANA CSV (same file used in Phase A)
WIDE_ANA_CSV = 'C:/Users/marcu/PycharmProjects/hydrobr_project/Testing/rainfall_timeseries_with_metadata_all.csv'  # e.g., r"C:\path\to\rainfall_timeseries_with_metadata_all.csv"
STATION_ID_COL = "Code"
LAT_COL  = "Latitude"
LON_COL  = "Longitude"

# Output
OUT_CSV = 'G:\My Drive\chirps_bias_pairs\zeta_per_station.csv'

# QC thresholds
MIN_SAMPLES   = 10                 # require at least this many event pairs per station
RATIO_CLIP    = (0.25, 5.0)        # drop ratios outside this range
MAX_RAIN_MM   = 350.0              # drop pr_g / chirps_mm above this cap if desired
MIN_RAIN_MM   = 1.0                # drop tiny values (already applied in Phase A, but double-guard)

# Estimator for ζ at the station:
#   'median' -> robust median of ratios
#   'mean'   -> arithmetic mean of ratios
#   'slope0' -> regression slope through origin of pr_g ~ zeta * chirps_mm
ZETA_METHOD = 'mean'
# ---------------------------------------------------------


def zeta_from_pairs(df_station: pd.DataFrame, method: str = 'median') -> float:
    """
    Compute station-level ζ from event pairs.
    df_station columns: ['ratio', 'pr_g', 'chirps_mm'] (and 'date', 'station_id')
    """
    r = df_station['ratio'].to_numpy(dtype=float)

    if method == 'median':
        return float(np.median(r))
    elif method == 'mean':
        return float(np.mean(r))
    elif method == 'slope0':
        # slope through origin: minimize sum (pr_g - zeta * chirps)^2
        x = df_station['chirps_mm'].to_numpy(dtype=float)
        y = df_station['pr_g'].to_numpy(dtype=float)
        num = np.sum(x * y)
        den = np.sum(x * x)
        if den <= 0:
            return np.nan
        return float(num / den)
    else:
        raise ValueError(f"Unknown ZETA_METHOD: {method}")


def main():
    # 1) Load all yearly files
    files = sorted(glob.glob(os.path.join(IN_DIR, IN_GLOB)))
    if not files:
        raise FileNotFoundError(f"No input files found in {IN_DIR} matching {IN_GLOB}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['date'])
        # Basic sanity filters (some of these already applied in Phase A)
        if 'ratio' not in df.columns:
            raise ValueError(f"'ratio' not found in {f}")
        # Drop non-finite or crazy values
        if 'pr_g' in df.columns:
            df = df[(df['pr_g'].between(MIN_RAIN_MM, MAX_RAIN_MM))]
        if 'chirps_mm' in df.columns:
            df = df[(df['chirps_mm'].between(MIN_RAIN_MM, MAX_RAIN_MM))]
        df = df[df['ratio'].between(RATIO_CLIP[0], RATIO_CLIP[1])]
        dfs.append(df)

    pairs = pd.concat(dfs, ignore_index=True)
    if pairs.empty:
        raise ValueError("After QC, no rows remain. Loosen filters or check inputs.")

    # Ensure station_id exists and is string
    if 'station_id' not in pairs.columns:
        raise ValueError("'station_id' column not found in input files.")
    pairs['station_id'] = pairs['station_id'].astype(str)

    # 2) Compute ζ per station
    rows = []
    for sid, g in pairs.groupby('station_id'):
        n = len(g)
        if n < MIN_SAMPLES:
            continue
        zeta = zeta_from_pairs(g, method=ZETA_METHOD)
        rows.append({
            'station_id': sid,
            'zeta': zeta,
            'n_pairs': n,
            'ratio_median': float(np.median(g['ratio'])),
            'ratio_mean': float(np.mean(g['ratio'])),
            'ratio_p10': float(np.percentile(g['ratio'], 10)),
            'ratio_p90': float(np.percentile(g['ratio'], 90))
        })

    zeta_df = pd.DataFrame(rows)
    if zeta_df.empty:
        raise ValueError("No stations passed MIN_SAMPLES threshold; lower MIN_SAMPLES or check inputs.")

    # 3) (Optional) merge lat/lon from the wide ANA CSV
    if WIDE_ANA_CSV and os.path.exists(WIDE_ANA_CSV):
        meta = pd.read_csv(WIDE_ANA_CSV, usecols=[STATION_ID_COL, LAT_COL, LON_COL])
        meta = meta.dropna(subset=[STATION_ID_COL, LAT_COL, LON_COL]).copy()
        meta[STATION_ID_COL] = meta[STATION_ID_COL].astype(str)
        # Deduplicate station ids
        meta = meta.drop_duplicates(subset=[STATION_ID_COL])
        zeta_df = zeta_df.merge(
            meta.rename(columns={
                STATION_ID_COL: 'station_id',
                LAT_COL: 'lat',
                LON_COL: 'lon'
            }),
            on='station_id', how='left'
        )

    # 4) Save
    zeta_df = zeta_df.sort_values('station_id').reset_index(drop=True)
    zeta_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  (stations: {len(zeta_df)})")
    print("\nPreview:")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
        print(zeta_df.head())


if __name__ == "__main__":
    main()
