#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIDF percentile-threshold sensitivity using the SAME pairwise bias-correction
methodology used for the paper bias-correction figure.

This script evaluates the sensitivity of the bias-correction percentile
threshold using the paired station-product rainfall values, not fitted IDF
return levels.

For each product and percentile threshold, it computes:

1) Pairwise scatter metrics, exactly in the spirit of the paper figure:
   - station rainfall depth vs raw product rainfall depth
   - station rainfall depth vs bias-corrected product rainfall depth
   - slope through origin
   - ordinary least-squares slope and intercept
   - R²
   - RMSE
   - MAE
   - mean bias
   - percent bias

2) Bias-factor statistics:
   - mean/median/std of zeta
   - station-level mean zeta when station IDs are available

3) Optional spatial bias maps from the annual_max_corrected/mean folders:
   - SpatialBiasFactor_AMDmean.tif
   - SpatialBiasPct_AMDmean.tif
   - SpatialBiasPct_delta_vs_<baseline>.tif

Expected pair folders:
  /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline/data/products/<product>/sensitivity/<percentile>/pairs

The script is intentionally robust to different pair-file column names.
It tries to detect station, product, corrected, and zeta columns automatically.

Outputs:
  /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Percentile_Sensitivity/PAIRWISE_BIAS/

Run:
  cd /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting

  python3 percentile_sensitivity_pairwise_bias_metrics.py \
    --products br_dwgd,chirps,persiann_cdr,imerg_v06,imerg_v07 \
    --percentiles p90,p95,p98,p99,p995 \
    --baseline p98

If corrected values are not stored in the pair files, the script reconstructs
the corrected value as:
    corrected = product_raw * station_mean_zeta
where:
    zeta = station / product_raw
and station_mean_zeta is computed from the same pair table.
This matches the mean-zeta correction logic used in the bias-correction workflow.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF")
RAW_ROOT = ROOT / "Annual_Maximum_Precipitation"
BIAS_PIPELINE_ROOT = ROOT / "Bias_Correction_Pipeline" / "data" / "products"
OUT_ROOT = ROOT / "IDF_Fitting" / "Percentile_Sensitivity" / "PAIRWISE_BIAS"

BASELINE_PERCENTILE = "p98"

COMPRESS = "deflate"
ZLEVEL = 6
PLOT_DPI = 300


@dataclass
class ProductConfig:
    label: str
    raw_dirs: List[Path]
    raw_pattern: str
    year_start: int
    year_end: int


PRODUCTS: Dict[str, ProductConfig] = {
    "br_dwgd": ProductConfig(
        label="BR-DWGD",
        raw_dirs=[
            RAW_ROOT / "BR-DWGD",
            RAW_ROOT / "BR_DWGD_Max",
            RAW_ROOT / "BRDWGD_Max",
            RAW_ROOT / "BR_DWGD",
        ],
        raw_pattern="*.tif",
        year_start=1995,
        year_end=2025,
    ),
    "chirps": ProductConfig(
        label="CHIRPS",
        raw_dirs=[
            RAW_ROOT / "CHIRPS_Max",
            RAW_ROOT / "CHRIPS_Max",
            RAW_ROOT / "CHIRPS",
        ],
        raw_pattern="*.tif",
        year_start=1995,
        year_end=2025,
    ),
    "persiann_cdr": ProductConfig(
        label="PERSIANN-CDR",
        raw_dirs=[
            RAW_ROOT / "PERSIANN_CDR_Max",
            RAW_ROOT / "PERSIANN_Max",
            RAW_ROOT / "PERSIANN_CDR",
        ],
        raw_pattern="*.tif",
        year_start=1995,
        year_end=2025,
    ),
    "imerg_v06": ProductConfig(
        label="IMERG V06",
        raw_dirs=[
            RAW_ROOT / "IMERG_V06_Max",
            RAW_ROOT / "IMERG_Max",
            RAW_ROOT / "IMERG_V06",
        ],
        raw_pattern="*.tif",
        year_start=2001,
        year_end=2020,
    ),
    "imerg_v07": ProductConfig(
        label="IMERG V07",
        raw_dirs=[
            RAW_ROOT / "IMERG_V07_Max",
            RAW_ROOT / "IMERG_V07",
        ],
        raw_pattern="*.tif",
        year_start=2001,
        year_end=2025,
    ),
}


# =============================================================================
# COLUMN CANDIDATES
# =============================================================================

STATION_ID_CANDIDATES = [
    "station_id", "station", "station_code", "code", "Code", "codigo", "Código",
    "id", "ID", "gauge_id", "gauge", "ana_code", "ANA_CODE",
]

LAT_CANDIDATES = [
    "latitude", "Latitude", "lat", "LAT", "y", "Y", "station_lat", "gauge_lat",
]

LON_CANDIDATES = [
    "longitude", "Longitude", "lon", "LON", "long", "Long", "x", "X",
    "station_lon", "gauge_lon",
]

STATION_VALUE_CANDIDATES = [
    "station_mm", "station_value", "station_rain_mm", "station_rainfall_mm",
    "gauge_mm", "gauge_value", "gauge_rain_mm", "obs_mm", "observed_mm",
    "observed", "obs", "ana_mm", "rain_station_mm", "rainfall_station_mm",
    "P_station", "P_obs", "depth_station_mm", "station_depth_mm",
]

PRODUCT_VALUE_CANDIDATES = [
    "product_mm", "product_value", "product_rain_mm", "product_rainfall_mm",
    "raw_product_mm", "raw_product_value", "raw_mm", "raw_value",
    "raster_mm", "raster_value", "grid_mm", "grid_value",
    "satellite_mm", "satellite_value", "gridded_mm", "gridded_value",
    "P_product", "P_grid", "P_raw", "depth_product_mm",
]

CORRECTED_VALUE_CANDIDATES = [
    "corrected_mm", "corrected_value", "bias_corrected_mm", "bias_corrected_value",
    "product_corrected_mm", "product_corrected_value",
    "corrected_product_mm", "corrected_product_value",
    "P_corrected", "P_bc", "bc_mm", "bc_value",
]

ZETA_CANDIDATES = [
    "zeta", "zeta_mean", "mean_zeta", "zeta_station_mean", "station_mean_zeta",
    "correction_factor", "bias_factor", "bias_correction_factor", "factor",
]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False, label: str = "column") -> Optional[str]:
    cols = list(df.columns)
    exact = {str(c): c for c in cols}
    norm = {normalize_colname(c): c for c in cols}

    # Exact and case-insensitive candidate matching.
    for c in candidates:
        if c in exact:
            return exact[c]
        nc = normalize_colname(c)
        if nc in norm:
            return norm[nc]

    # Fuzzy contains match, but only for sufficiently specific names.
    candidate_norms = [normalize_colname(c) for c in candidates]
    for col in cols:
        ncol = normalize_colname(col)
        for nc in candidate_norms:
            if len(nc) >= 5 and (nc in ncol or ncol in nc):
                return col

    if required:
        raise ValueError(
            f"Could not find {label}. Tried candidates:\n{candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def normalize_station_id(values) -> pd.Series:
    s = pd.Series(values).astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.upper()
    return s


def percentile_value(label: str) -> float:
    s = label.lower().strip()
    for token in ["percentile_", "percentile", "pct_", "pct"]:
        s = s.replace(token, "")
    s = s.strip("_-")
    if s.startswith("p"):
        s = s[1:]
    s = s.replace("p", ".").replace("_", ".").replace("-", ".")
    try:
        val = float(s)
    except Exception:
        return np.nan
    if val > 100 and val < 1000:
        val = val / 10.0
    return val


def sort_percentiles(labels: Sequence[str]) -> List[str]:
    return sorted(labels, key=lambda x: (np.nan_to_num(percentile_value(x), nan=9999), x))


def extract_year(path: Path) -> Optional[int]:
    matches = re.findall(r"(19\d{2}|20\d{2})", path.name)
    return int(matches[-1]) if matches else None


def first_existing_directory(candidates: Sequence[Path]) -> Optional[Path]:
    for folder in candidates:
        if folder.exists():
            return folder
    return None


def get_sensitivity_root(product: str, percentile: str) -> Path:
    return BIAS_PIPELINE_ROOT / product / "sensitivity" / percentile


def get_pairs_folder(product: str, percentile: str) -> Path:
    return get_sensitivity_root(product, percentile) / "pairs"


def get_corrected_annual_folder(product: str, percentile: str) -> Path:
    return get_sensitivity_root(product, percentile) / "annual_max_corrected" / "mean"


def discover_percentiles_for_product(product: str) -> List[str]:
    root = BIAS_PIPELINE_ROOT / product / "sensitivity"
    if not root.exists():
        return []

    found = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        pairs_dir = d / "pairs"
        corrected_dir = d / "annual_max_corrected" / "mean"
        if (pairs_dir.exists() and list(pairs_dir.rglob("*"))) or (corrected_dir.exists() and list(corrected_dir.glob("*.tif"))):
            found.append(d.name)

    return sort_percentiles(found)


# =============================================================================
# PAIR TABLE LOADING
# =============================================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def find_pair_files(product: str, percentile: str) -> List[Path]:
    """
    Find pair files for product/percentile.

    Preferred location:
      sensitivity/<percentile>/pairs

    Fallback:
      recursively search sensitivity/<percentile> for CSV/Parquet files whose
      names contain 'pair'.
    """
    sens_root = get_sensitivity_root(product, percentile)
    pairs_dir = get_pairs_folder(product, percentile)

    files: List[Path] = []

    if pairs_dir.exists():
        files.extend(sorted(pairs_dir.rglob("*.csv")))
        files.extend(sorted(pairs_dir.rglob("*.parquet")))
        files.extend(sorted(pairs_dir.rglob("*.pq")))

    if not files and sens_root.exists():
        for ext in ["*.csv", "*.parquet", "*.pq"]:
            for f in sorted(sens_root.rglob(ext)):
                name = f.name.lower()
                if "pair" in name or "zeta" in name or "sample" in name:
                    files.append(f)

    # Exclude obvious summary files that are not pair-level.
    clean = []
    for f in files:
        name = f.name.lower()
        if any(token in name for token in ["summary", "metrics", "inventory", "manifest"]):
            continue
        clean.append(f)

    return clean


def load_pair_table(product: str, percentile: str) -> pd.DataFrame:
    files = find_pair_files(product, percentile)

    if not files:
        raise FileNotFoundError(
            f"No pair files found for {product} / {percentile}.\n"
            f"Expected folder: {get_pairs_folder(product, percentile)}"
        )

    frames = []
    for f in files:
        try:
            df = read_table(f)
            if len(df) == 0:
                continue
            df["_source_file"] = str(f)
            frames.append(df)
        except Exception as exc:
            print(f"    WARNING: failed reading {f}: {exc}")

    if not frames:
        raise ValueError(f"Pair files were found but none could be read for {product}/{percentile}")

    df = pd.concat(frames, ignore_index=True)
    return df


def standardize_pair_table(df_raw: pd.DataFrame, product: str, percentile: str) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Standardize a pair table to:
      station_id, latitude, longitude, station_mm, product_raw_mm,
      product_corrected_mm, zeta_pair, zeta_mean_used

    IMPORTANT:
    To reproduce the paper-style bias-correction scatterplots, the correction
    factor is recomputed directly from the paired values:

        zeta_pair = station_mm / product_raw_mm

    The corrected product value is then reconstructed as:

        product_corrected_mm = product_raw_mm * mean(zeta_pair) per station

    This avoids accidentally using unrelated columns named zeta/factor in the
    pair files. A stored corrected-value column is used only if it is explicitly
    detected and finite; otherwise, the station-mean zeta reconstruction is used.
    """
    df = df_raw.copy()

    station_col = find_column(df, STATION_VALUE_CANDIDATES, required=True, label="station rainfall depth")
    product_col = find_column(df, PRODUCT_VALUE_CANDIDATES, required=True, label="product/raw rainfall depth")
    corrected_col = find_column(df, CORRECTED_VALUE_CANDIDATES, required=False, label="bias-corrected rainfall depth")
    station_id_col = find_column(df, STATION_ID_CANDIDATES, required=False, label="station ID")
    lat_col = find_column(df, LAT_CANDIDATES, required=False, label="latitude")
    lon_col = find_column(df, LON_CANDIDATES, required=False, label="longitude")

    # We intentionally do NOT use a stored zeta column for the correction,
    # because some files contain unrelated or placeholder factor columns.
    zeta_col = find_column(df, ZETA_CANDIDATES, required=False, label="zeta/correction factor")

    meta = {
        "station_col": station_col,
        "product_col": product_col,
        "corrected_col": corrected_col,
        "zeta_col_detected_but_not_used": zeta_col,
        "station_id_col": station_id_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }

    print("    detected columns:")
    print(f"      station depth : {station_col}")
    print(f"      product raw   : {product_col}")
    print(f"      corrected     : {corrected_col}")
    print(f"      station ID    : {station_id_col}")
    print(f"      detected zeta : {zeta_col}  [not used for reconstruction]")

    out = pd.DataFrame()
    out["product"] = product
    out["percentile"] = percentile
    out["percentile_value"] = percentile_value(percentile)

    if station_id_col is not None:
        out["station_id"] = normalize_station_id(df[station_id_col])
    else:
        out["station_id"] = "NO_STATION_ID"

    if lat_col is not None:
        out["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        out["latitude"] = np.nan

    if lon_col is not None:
        out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        out["longitude"] = np.nan

    out["station_mm"] = pd.to_numeric(df[station_col], errors="coerce")
    out["product_raw_mm"] = pd.to_numeric(df[product_col], errors="coerce")

    if corrected_col is not None:
        out["product_corrected_stored_mm"] = pd.to_numeric(df[corrected_col], errors="coerce")
    else:
        out["product_corrected_stored_mm"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)

    # Match the QC logic used for the paper-style paired comparison:
    # keep positive paired rainfall depths and remove impossible daily extremes.
    # The 350 mm cap is the same defensibility cap discussed for the bias
    # correction pair screening.
    out = out.dropna(subset=["station_mm", "product_raw_mm"]).copy()
    out = out[
        (out["station_mm"] > 0)
        & (out["product_raw_mm"] > 0)
        & (out["station_mm"] <= 350.0)
        & (out["product_raw_mm"] <= 350.0)
    ].copy()

    # Pairwise zeta from paired values. This is the only zeta used.
    out["zeta_pair"] = out["station_mm"] / out["product_raw_mm"]
    out = out.replace([np.inf, -np.inf], np.nan)

    # Remove extreme correction ratios from numerical/pathological product
    # near-zero behavior. This is not supposed to remove valid extremes; it
    # avoids a few bad denominators dominating station-mean zeta.
    out = out.dropna(subset=["zeta_pair"]).copy()
    out = out[(out["zeta_pair"] > 0) & (out["zeta_pair"] <= 20.0)].copy()

    # Reconstruct corrected product using station-mean zeta.
    # This is what makes the sensitivity comparable to the paper figure:
    # station-product pairs are corrected by the station-level mean factor.
    if station_id_col is not None and out["station_id"].nunique() > 1:
        zeta_station = out.groupby("station_id")["zeta_pair"].mean()
        out["zeta_mean_used"] = out["station_id"].map(zeta_station)
    else:
        out["zeta_mean_used"] = float(out["zeta_pair"].mean())

    out["product_corrected_reconstructed_mm"] = out["product_raw_mm"] * out["zeta_mean_used"]

    # Prefer a stored corrected column only if it is clearly meaningful.
    # If stored corrected is missing, equal to raw everywhere, or produces
    # absurd slopes, use reconstructed mean-zeta corrected values.
    use_stored = False
    if out["product_corrected_stored_mm"].notna().sum() >= max(10, int(0.5 * len(out))):
        stored = out["product_corrected_stored_mm"].to_numpy(float)
        raw = out["product_raw_mm"].to_numpy(float)
        good = np.isfinite(stored) & np.isfinite(raw) & (stored > 0) & (stored <= 350.0)

        if np.count_nonzero(good) >= max(10, int(0.5 * len(out))):
            ratio = stored[good] / raw[good]
            # If the stored corrected value is just raw or a placeholder, do
            # not use it. Otherwise, it may be the exact corrected column from
            # the original workflow.
            if np.nanstd(ratio) > 1e-6 and 0.05 <= np.nanmedian(ratio) <= 20.0:
                use_stored = True

    if use_stored:
        out["product_corrected_mm"] = out["product_corrected_stored_mm"]
        out["correction_source"] = "stored_corrected_column"
    else:
        out["product_corrected_mm"] = out["product_corrected_reconstructed_mm"]
        out["correction_source"] = "station_mean_zeta_reconstructed"

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["station_mm", "product_raw_mm", "product_corrected_mm", "zeta_mean_used"]).copy()

    # Final cap for scatterplot comparability.
    out = out[
        (out["product_corrected_mm"] > 0)
        & (out["product_corrected_mm"] <= 350.0)
    ].copy()

    return out, meta


# =============================================================================
# METRICS
# =============================================================================

def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Compute pairwise metrics for y = product, x = station.

    The paper scatterplot label y = ax uses a zero-intercept slope:
        a = sum(x*y) / sum(x^2)

    The R² shown in the paper-style figure is the uncentered through-origin
    coefficient of determination:
        R²_0 = 1 - sum((y - a*x)^2) / sum(y^2)

    This is different from the squared Pearson correlation. We keep the
    Pearson/correlation R² as corr_r2 for diagnostics, but the main r2 column
    is the paper-style through-origin R².
    """
    x = np.asarray(obs, dtype=float)
    y = np.asarray(pred, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n = int(np.count_nonzero(good))

    if n < 3:
        return {
            "n_pairs": n,
            "slope_through_origin": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "corr_r2": np.nan,
            "rmse_mm": np.nan,
            "mae_mm": np.nan,
            "mean_bias_mm": np.nan,
            "percent_bias": np.nan,
            "mean_station_mm": np.nan,
            "mean_product_mm": np.nan,
        }

    x = x[good]
    y = y[good]

    denom_x = np.sum(x ** 2)
    slope0 = float(np.sum(x * y) / denom_x) if denom_x > 0 else np.nan

    yhat0 = slope0 * x
    denom_y0 = np.sum(y ** 2)
    r2_zero = float(1.0 - np.sum((y - yhat0) ** 2) / denom_y0) if denom_y0 > 0 else np.nan

    slope, intercept = np.polyfit(x, y, 1)

    if np.std(x) > 0 and np.std(y) > 0:
        r = np.corrcoef(x, y)[0, 1]
        corr_r2 = float(r ** 2)
    else:
        corr_r2 = np.nan

    residual = y - x

    return {
        "n_pairs": n,
        "slope_through_origin": slope0,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2_zero,
        "corr_r2": corr_r2,
        "rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
        "mae_mm": float(np.mean(np.abs(residual))),
        "mean_bias_mm": float(np.mean(residual)),
        "percent_bias": float(100.0 * np.sum(y - x) / np.sum(x)) if np.sum(x) != 0 else np.nan,
        "mean_station_mm": float(np.mean(x)),
        "mean_product_mm": float(np.mean(y)),
    }


def summarize_zeta(df: pd.DataFrame) -> Dict[str, float]:
    z = df["zeta_mean_used"].to_numpy(float)
    z = z[np.isfinite(z)]

    out = {
        "zeta_n": int(z.size),
        "zeta_mean": np.nan,
        "zeta_median": np.nan,
        "zeta_std": np.nan,
        "zeta_p05": np.nan,
        "zeta_p95": np.nan,
        "zeta_min": np.nan,
        "zeta_max": np.nan,
        "n_stations": int(df["station_id"].nunique()) if "station_id" in df.columns else np.nan,
    }

    if z.size > 0:
        out.update(
            {
                "zeta_mean": float(np.mean(z)),
                "zeta_median": float(np.median(z)),
                "zeta_std": float(np.std(z, ddof=1)) if z.size > 1 else 0.0,
                "zeta_p05": float(np.percentile(z, 5)),
                "zeta_p95": float(np.percentile(z, 95)),
                "zeta_min": float(np.min(z)),
                "zeta_max": float(np.max(z)),
            }
        )

    return out


def summarize_pair_metrics(df: pd.DataFrame, product: str, percentile: str, product_label: str, meta: Dict[str, Optional[str]]) -> Dict[str, float]:
    raw = regression_metrics(df["station_mm"].to_numpy(float), df["product_raw_mm"].to_numpy(float))
    cor = regression_metrics(df["station_mm"].to_numpy(float), df["product_corrected_mm"].to_numpy(float))
    zeta = summarize_zeta(df)

    row = {
        "product": product,
        "product_label": product_label,
        "percentile": percentile,
        "percentile_value": percentile_value(percentile),
        "station_column": meta.get("station_col"),
        "product_column": meta.get("product_col"),
        "corrected_column": meta.get("corrected_col"),
        "zeta_column": meta.get("zeta_col"),
        "station_id_column": meta.get("station_id_col"),
    }

    for k, v in raw.items():
        row[f"raw_{k}"] = v

    for k, v in cor.items():
        row[f"corrected_{k}"] = v

    row.update(zeta)

    return row


# =============================================================================
# RASTER SPATIAL BIAS SENSITIVITY
# =============================================================================

def read_single_band_float(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        profile = ds.profile.copy()
        nodata = ds.nodata
        if nodata is not None and np.isfinite(nodata):
            arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where((arr < -1e20) | (arr > 1e20), np.nan, arr)
    return arr, profile


def same_grid(path: Path, base_profile: dict) -> bool:
    with rasterio.open(path) as ds:
        return (
            ds.height == base_profile["height"]
            and ds.width == base_profile["width"]
            and ds.transform == base_profile["transform"]
            and ds.crs == base_profile["crs"]
        )


def reproject_raster_to_base(src_path: Path, base_profile: dict, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    dst = np.full((base_profile["height"], base_profile["width"]), np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            src_arr = np.where(src_arr == nodata, np.nan, src_arr)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=base_profile["transform"],
            dst_crs=base_profile["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def list_annual_rasters(folder: Path, pattern: str, year_start: int, year_end: int) -> List[Tuple[int, Path]]:
    files = sorted(folder.glob(pattern))
    out = []
    for f in files:
        if not f.is_file():
            continue
        y = extract_year(f)
        if y is not None and year_start <= y <= year_end:
            out.append((y, f))

    by_year: Dict[int, Path] = {}
    for y, f in out:
        by_year.setdefault(y, f)

    return sorted(by_year.items(), key=lambda x: x[0])


def load_stack_as_year_dict(folder: Path, pattern: str, year_start: int, year_end: int, base_profile: Optional[dict] = None) -> Tuple[Dict[int, np.ndarray], dict]:
    year_files = list_annual_rasters(folder, pattern, year_start, year_end)
    if not year_files:
        raise FileNotFoundError(f"No annual maximum rasters found in {folder} for {year_start}-{year_end}")

    data: Dict[int, np.ndarray] = {}

    if base_profile is None:
        first_arr, first_profile = read_single_band_float(year_files[0][1])
        base_profile = first_profile.copy()
        base_profile.update(
            height=first_arr.shape[0],
            width=first_arr.shape[1],
            transform=first_profile["transform"],
            crs=first_profile["crs"],
            count=1,
            dtype="float32",
        )
        data[year_files[0][0]] = first_arr.astype("float32")
        iterator = year_files[1:]
    else:
        iterator = year_files

    for year, path in iterator:
        if same_grid(path, base_profile):
            arr, _ = read_single_band_float(path)
        else:
            print(f"    Regridding to base grid: {path.name}")
            arr = reproject_raster_to_base(path, base_profile, resampling=Resampling.bilinear)
        data[year] = arr.astype("float32")

    return data, base_profile


def stack_from_year_dict(year_dict: Dict[int, np.ndarray], years: Sequence[int]) -> np.ndarray:
    arrays = [year_dict[y] for y in years if y in year_dict]
    if not arrays:
        raise ValueError("No matching years available for stacking.")
    return np.stack(arrays, axis=0).astype("float32")


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype="float32")
    good = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[good] = (a[good] / b[good]).astype("float32")
    return out


def summarize_array(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {
            f"{prefix}_valid_pixels": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_valid_pixels": int(vals.size),
        f"{prefix}_mean": float(np.nanmean(vals)),
        f"{prefix}_median": float(np.nanmedian(vals)),
        f"{prefix}_p05": float(np.nanpercentile(vals, 5)),
        f"{prefix}_p95": float(np.nanpercentile(vals, 95)),
        f"{prefix}_min": float(np.nanmin(vals)),
        f"{prefix}_max": float(np.nanmax(vals)),
    }


def write_geotiff(path: Path, arr: np.ndarray, profile: dict, dtype: str = "float32", nodata: Optional[float] = np.nan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress=COMPRESS, zlevel=ZLEVEL)
    if dtype.startswith("float"):
        out_profile.update(predictor=2)
    else:
        out_profile.update(predictor=1)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(dtype), 1)


def compute_spatial_bias_for_product(product: str, cfg: ProductConfig, percentiles: Sequence[str], baseline: str) -> pd.DataFrame:
    """
    Compute spatial bias maps for annual maximum means:
      corrected_AMD_mean / raw_AMD_mean
    This is optional but keeps the spatial-bias sensitivity requested earlier.
    """
    raw_folder = first_existing_directory(cfg.raw_dirs)
    if raw_folder is None:
        print(f"  WARNING: raw annual-max folder not found for spatial bias: {product}")
        return pd.DataFrame()

    rows = []
    out_product_root = OUT_ROOT / product
    out_product_root.mkdir(parents=True, exist_ok=True)

    try:
        raw_year_dict, base_profile = load_stack_as_year_dict(
            raw_folder,
            cfg.raw_pattern,
            cfg.year_start,
            cfg.year_end,
            base_profile=None,
        )
        raw_years = sorted(raw_year_dict.keys())
    except Exception as exc:
        print(f"  WARNING: failed loading raw rasters for spatial bias ({product}): {exc}")
        return pd.DataFrame()

    bias_pct_by_pct: Dict[str, np.ndarray] = {}

    for pct in percentiles:
        corrected_folder = get_corrected_annual_folder(product, pct)
        if not corrected_folder.exists():
            print(f"  WARNING: corrected annual folder missing for {product}/{pct}: {corrected_folder}")
            continue

        try:
            bc_year_dict, _ = load_stack_as_year_dict(
                corrected_folder,
                "*.tif",
                cfg.year_start,
                cfg.year_end,
                base_profile=base_profile,
            )
        except Exception as exc:
            print(f"  WARNING: failed loading corrected rasters for {product}/{pct}: {exc}")
            continue

        common_years = sorted(set(raw_years).intersection(bc_year_dict.keys()))
        if len(common_years) == 0:
            continue

        raw_stack = stack_from_year_dict(raw_year_dict, common_years)
        bc_stack = stack_from_year_dict(bc_year_dict, common_years)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_mean = np.nanmean(raw_stack, axis=0).astype("float32")
            bc_mean = np.nanmean(bc_stack, axis=0).astype("float32")

        bias_factor = safe_ratio(bc_mean, raw_mean)
        bias_pct = (100.0 * (bias_factor - 1.0)).astype("float32")
        bias_pct_by_pct[pct] = bias_pct

        pct_out_dir = out_product_root / pct
        pct_out_dir.mkdir(parents=True, exist_ok=True)
        write_geotiff(pct_out_dir / "SpatialBiasFactor_AMDmean.tif", bias_factor, base_profile)
        write_geotiff(pct_out_dir / "SpatialBiasPct_AMDmean.tif", bias_pct, base_profile)

        row = {
            "product": product,
            "product_label": cfg.label,
            "percentile": pct,
            "percentile_value": percentile_value(pct),
            "baseline_percentile": baseline,
            "year_start": int(common_years[0]),
            "year_end": int(common_years[-1]),
            "n_common_years": int(len(common_years)),
        }
        row.update(summarize_array(bias_factor, "spatial_bias_factor_amdmean"))
        row.update(summarize_array(bias_pct, "spatial_bias_pct_amdmean"))
        rows.append(row)

    if baseline in bias_pct_by_pct:
        baseline_bias = bias_pct_by_pct[baseline]
        for row in rows:
            pct = row["percentile"]
            delta = (bias_pct_by_pct[pct] - baseline_bias).astype("float32")
            pct_out_dir = out_product_root / pct
            write_geotiff(pct_out_dir / f"SpatialBiasPct_delta_vs_{baseline}.tif", delta, base_profile)
            row.update(summarize_array(delta, f"spatial_bias_pct_delta_vs_{baseline}"))

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def make_metric_sensitivity_plot(summary: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB or summary.empty:
        return

    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=PLOT_DPI)

    for product, g in summary.groupby("product"):
        g = g.sort_values("percentile_value")
        label = PRODUCTS.get(product, ProductConfig(product, [], "", 0, 0)).label
        axes[0].plot(g["percentile_value"], g["corrected_r2"], marker="o", linewidth=1.8, label=label)
        axes[1].plot(g["percentile_value"], g["corrected_slope_through_origin"], marker="o", linewidth=1.8, label=label)

    axes[0].set_title("Bias-corrected pairwise R²")
    axes[0].set_xlabel("Percentile threshold")
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    axes[1].set_title("Bias-corrected pairwise slope")
    axes[1].set_xlabel("Percentile threshold")
    axes[1].set_ylabel("Slope through origin")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"pairwise_metric_sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_scatter_plot(df: pd.DataFrame, product: str, percentile: str, label: str) -> None:
    if not HAS_MATPLOTLIB or df.empty:
        return

    fig_dir = OUT_ROOT / product / percentile / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = regression_metrics(df["station_mm"].to_numpy(float), df["product_raw_mm"].to_numpy(float))
    cor_metrics = regression_metrics(df["station_mm"].to_numpy(float), df["product_corrected_mm"].to_numpy(float))

    allv = np.concatenate([
        df["station_mm"].to_numpy(float),
        df["product_raw_mm"].to_numpy(float),
        df["product_corrected_mm"].to_numpy(float),
    ])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return

    lo = 0.0
    hi = float(np.nanpercentile(allv, 99.5))
    hi = max(hi, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=PLOT_DPI)

    for ax, ycol, title, m, color in [
        (axes[0], "product_raw_mm", "Raw", raw_metrics, "#6E6E6E"),
        (axes[1], "product_corrected_mm", "Bias-corrected", cor_metrics, "#0072B2"),
    ]:
        x = df["station_mm"].to_numpy(float)
        y = df[ycol].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y) & (x > 0)

        if np.count_nonzero(good) > 0:
            ax.hexbin(x[good], y[good], gridsize=45, extent=(lo, hi, lo, hi), mincnt=1, cmap="viridis")
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.1, linestyle="--")
        if np.isfinite(m["slope_through_origin"]):
            ax.plot([lo, hi], [0, m["slope_through_origin"] * hi], color=color, linewidth=1.6)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Station rainfall depth (mm)")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        txt = f"y = {m['slope_through_origin']:.2f}x, R² = {m['r2']:.2f}"
        ax.text(0.03, 0.95, txt, transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, pad=2))

    axes[0].set_ylabel("Product rainfall depth (mm)")
    fig.suptitle(f"{label} | {percentile}", fontweight="bold")
    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"pairwise_scatter_{product}_{percentile}.{ext}", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def process_product(product: str, cfg: ProductConfig, percentiles_requested: Optional[Sequence[str]], baseline: str, make_scatter: bool, compute_spatial: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 110)
    print(f"PRODUCT: {product} ({cfg.label})")
    print("=" * 110)

    if percentiles_requested is None:
        percentiles = discover_percentiles_for_product(product)
    else:
        percentiles = list(percentiles_requested)

    percentiles = sort_percentiles(percentiles)

    if not percentiles:
        print("  WARNING: no percentiles found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print("  Percentiles:", percentiles)
    print("  Baseline:", baseline)

    summary_rows = []
    point_frames = []

    for pct in percentiles:
        print(f"\n  Pairwise metrics for {product} / {pct}")

        try:
            raw_pairs = load_pair_table(product, pct)
            pairs, meta = standardize_pair_table(raw_pairs, product, pct)
        except Exception as exc:
            print(f"    WARNING: failed pairwise metrics for {product}/{pct}: {exc}")
            continue

        print(
            f"    pairs: {len(pairs):,}; stations: {pairs['station_id'].nunique():,}; "
            f"raw slope={regression_metrics(pairs['station_mm'], pairs['product_raw_mm'])['slope_through_origin']:.3f}; "
            f"corrected slope={regression_metrics(pairs['station_mm'], pairs['product_corrected_mm'])['slope_through_origin']:.3f}"
        )

        row = summarize_pair_metrics(pairs, product, pct, cfg.label, meta)
        summary_rows.append(row)

        # Save standardized pair table for reproducibility.
        out_dir = OUT_ROOT / product / pct
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out_dir / f"standardized_pairs_{product}_{pct}.csv", index=False)

        point_frames.append(pairs)

        if make_scatter:
            make_scatter_plot(pairs, product, pct, cfg.label)

    summary = pd.DataFrame(summary_rows)
    points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()

    spatial = pd.DataFrame()
    if compute_spatial:
        print("\n  Computing spatial bias maps/statistics...")
        spatial = compute_spatial_bias_for_product(product, cfg, percentiles, baseline)

    return summary, points, spatial


def run_sensitivity(
    products: Optional[Sequence[str]] = None,
    percentiles: Optional[Sequence[str]] = None,
    baseline: str = BASELINE_PERCENTILE,
    make_scatter: bool = True,
    compute_spatial: bool = True,
) -> None:
    selected_products = list(products) if products is not None else list(PRODUCTS.keys())

    invalid = [p for p in selected_products if p not in PRODUCTS]
    if invalid:
        raise ValueError(f"Unknown products: {invalid}. Valid products: {list(PRODUCTS)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 110)
    print("GRIDF PERCENTILE SENSITIVITY USING PAPER PAIRWISE BIAS-CORRECTION METHODOLOGY")
    print("#" * 110)
    print("Products:", selected_products)
    print("Percentiles:", percentiles if percentiles is not None else "auto-discover")
    print("Baseline percentile:", baseline)
    print("Output root:", OUT_ROOT)
    print("#" * 110)

    all_summary = []
    all_points = []
    all_spatial = []

    for product in selected_products:
        summary, points, spatial = process_product(
            product=product,
            cfg=PRODUCTS[product],
            percentiles_requested=percentiles,
            baseline=baseline,
            make_scatter=make_scatter,
            compute_spatial=compute_spatial,
        )
        if not summary.empty:
            all_summary.append(summary)
        if not points.empty:
            all_points.append(points)
        if not spatial.empty:
            all_spatial.append(spatial)

    df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    df_points = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()
    df_spatial = pd.concat(all_spatial, ignore_index=True) if all_spatial else pd.DataFrame()

    summary_path = OUT_ROOT / "pairwise_bias_metrics_by_product_percentile.csv"
    points_path = OUT_ROOT / "pairwise_bias_standardized_points.csv"
    spatial_path = OUT_ROOT / "spatial_bias_summary_by_product_percentile.csv"

    df_summary.to_csv(summary_path, index=False)
    df_points.to_csv(points_path, index=False)
    df_spatial.to_csv(spatial_path, index=False)

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", points_path)
    print(" ", spatial_path)

    if not df_summary.empty:
        cols = [
            "product", "percentile",
            "raw_n_pairs", "raw_slope_through_origin", "raw_r2", "raw_corr_r2",
            "corrected_slope_through_origin", "corrected_r2", "corrected_corr_r2",
            "corrected_rmse_mm", "corrected_percent_bias",
            "zeta_mean", "zeta_std", "n_stations",
        ]
        print("\nPreview pairwise metrics:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(df_summary[[c for c in cols if c in df_summary.columns]].to_string(index=False))

    if not df_spatial.empty:
        cols = [
            "product", "percentile",
            "spatial_bias_pct_amdmean_mean",
            "spatial_bias_pct_amdmean_median",
            f"spatial_bias_pct_delta_vs_{baseline}_mean",
        ]
        print("\nPreview spatial bias metrics:")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(df_spatial[[c for c in cols if c in df_spatial.columns]].to_string(index=False))

    make_metric_sensitivity_plot(df_summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run percentile sensitivity using paper pairwise station-product bias-correction methodology."
    )
    parser.add_argument("--products", default=None, help="Comma-separated products. Default: all.")
    parser.add_argument("--percentiles", default=None, help="Comma-separated percentiles. Example: p90,p95,p98,p99,p995")
    parser.add_argument("--baseline", default=BASELINE_PERCENTILE, help="Baseline percentile for spatial-bias deltas. Default: p98.")
    parser.add_argument("--no-scatter", action="store_true", help="Skip scatterplot generation.")
    parser.add_argument("--no-spatial", action="store_true", help="Skip spatial bias maps/statistics.")
    args = parser.parse_args()

    run_sensitivity(
        products=parse_list_arg(args.products),
        percentiles=parse_list_arg(args.percentiles),
        baseline=args.baseline,
        make_scatter=not args.no_scatter,
        compute_spatial=not args.no_spatial,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
