# 02_compute_station_bias_factors_zeta_from_pairs_IMERG.py
# ---------------------------------------------------------
# Purpose:
#   Read yearly "imerg_bias_pairs_YYYY.csv" files produced by Phase A,
#   compute a robust station-level bias factor ζ from event-wise ratios
#   (default: slope through origin of pr_g ~ ζ * imerg_mm), and save:
#     - zeta_per_station.csv  (station_id, zeta, n_pairs [, lat, lon])
#
# Optional:
#   - Merge station coordinates from the wide ANA CSV (so Phase B can run directly).
#   - Alternate estimators for ζ (median, mean, or slope through origin).
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
IN_DIR  = r'G:\My Drive\imerg_bias_pairs'     # <- IMERG folder
IN_GLOB = "imerg_bias_pairs_*.csv"            # <- IMERG file prefix

# Optional: merge lat/lon from your wide ANA CSV (same file used in Phase A)
WIDE_ANA_CSV   = r'C:/Users/marcu/PycharmProjects/hydrobr_project/Testing/rainfall_timeseries_with_metadata_all.csv'
STATION_ID_COL = "Code"
LAT_COL  = "Latitude"
LON_COL  = "Longitude"

# Output
OUT_CSV = r'G:\My Drive\imerg_bias_pairs\zeta_per_station.csv'  # <- IMERG output path

# QC thresholds
MIN_SAMPLES   = 10
RATIO_CLIP    = (0.25, 5.0)
MAX_RAIN_MM   = 350.0
MIN_RAIN_MM   = 1.0


# Estimator for ζ at the station:
#   'median' -> robust median of ratios (pr_g / precip)
#   'mean'   -> arithmetic mean of ratios
#   'slope0' -> regression slope through origin of pr_g ~ ζ * precip_mm
ZETA_METHOD = 'mean'
# ---------------------------------------------------------


def detect_precip_col(df: pd.DataFrame) -> str:
    """
    Return the name of the precip column from Phase A pairs.
    Priority: IMERG ('imerg_mm'), then PERSIANN ('persiann_mm'), then CHIRPS ('chirps_mm').
    """
    if 'imerg_mm' in df.columns:
        return 'imerg_mm'
    if 'persiann_mm' in df.columns:
        return 'persiann_mm'
    if 'chirps_mm' in df.columns:
        return 'chirps_mm'
    raise ValueError("No precip column found (expected one of 'imerg_mm', 'persiann_mm', 'chirps_mm').")


def zeta_from_pairs(df_station: pd.DataFrame, method: str = 'median', precip_col: str = 'imerg_mm') -> float:
    """
    Compute station-level ζ from event pairs.
    Expected columns: ['ratio', 'pr_g', precip_col] (and 'date', 'station_id').

    ζ interprets as the multiplicative bias to apply to the satellite product:
      pr_g ≈ ζ * precip_mm      -> slope through origin if method='slope0'
      ζ ≈ median(pr_g / precip) -> if method='median'
    """
    if method == 'median':
        r = df_station['ratio'].to_numpy(dtype=float)
        return float(np.median(r))
    elif method == 'mean':
        r = df_station['ratio'].to_numpy(dtype=float)
        return float(np.mean(r))
    elif method == 'slope0':
        # slope through origin: minimize sum (pr_g - ζ * precip)^2
        x = df_station[precip_col].to_numpy(dtype=float)
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

        # Determine which precip column we have (IMERG preferred)
        precip_col = detect_precip_col(df)

        # Basic sanity filters (some already applied in Phase A)
        for req_col in ['ratio', 'pr_g', precip_col]:
            if req_col not in df.columns:
                raise ValueError(f"Required column '{req_col}' not found in {f}")

        # Drop non-finite or out-of-bounds values
        df = df[df['pr_g'].between(MIN_RAIN_MM, MAX_RAIN_MM)]
        df = df[df[precip_col].between(MIN_RAIN_MM, MAX_RAIN_MM)]
        df = df[df['ratio'].between(RATIO_CLIP[0], RATIO_CLIP[1])]

        # Keep only needed columns to lighten memory
        keep_cols = ['station_id', 'date', 'pr_g', 'ratio', precip_col, 'lat', 'lon']
        keep_cols = [c for c in keep_cols if c in df.columns]
        dfs.append(df[keep_cols])

    pairs = pd.concat(dfs, ignore_index=True)
    if pairs.empty:
        raise ValueError("After QC, no rows remain. Loosen filters or check inputs.")

    # Ensure station_id exists and is string
    if 'station_id' not in pairs.columns:
        raise ValueError("'station_id' column not found in input files.")
    pairs['station_id'] = pairs['station_id'].astype(str)

    # Detect precip column on the concatenated frame (consistent with per-file)
    precip_col = detect_precip_col(pairs)

    # 2) Compute ζ per station
    rows = []
    for sid, g in pairs.groupby('station_id'):
        n = len(g)
        if n < MIN_SAMPLES:
            continue
        zeta = zeta_from_pairs(g, method=ZETA_METHOD, precip_col=precip_col)
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
        meta = meta.drop_duplicates(subset=[STATION_ID_COL])
        zeta_df = zeta_df.merge(
            meta.rename(columns={STATION_ID_COL: 'station_id', LAT_COL: 'lat', LON_COL: 'lon'}),
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
