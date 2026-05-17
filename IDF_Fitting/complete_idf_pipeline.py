#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete GRIDF IDF Pipeline

Generates IDF parameter rasters for:
  - raw annual maximum daily rainfall
  - bias-corrected annual maximum daily rainfall using mean zeta
  - Gumbel and GEV extreme-value models
  - RASTER, CETESB, and STATION disaggregation modes

Main function:
    run_complete_idf_pipeline()

Default output root:
    /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs

Recommended run:
    python3 complete_idf_pipeline.py --dry-run
    python3 complete_idf_pipeline.py

Fast test:
    python3 complete_idf_pipeline.py --products chirps --states raw --modes RASTER --distributions GUMBEL --overwrite

cd /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting

PYTHONUNBUFFERED=1 python3 -u complete_idf_pipeline.py \
  --products br_dwgd \
  --states bias_corrected_mean \
  --modes RASTER \
  --distributions GUMBEL \
  --bc-percentile p99 \
  --overwrite \
  --no-plots

"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import genextreme, kstwo

# Optional plotting support. The pipeline still runs if matplotlib is unavailable.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF")
RAW_ROOT = ROOT / "Annual_Maximum_Precipitation"
BIAS_PIPELINE_ROOT = ROOT / "Bias_Correction_Pipeline" / "data" / "products"
OUT_ROOT = ROOT / "IDF_Fitting" / "Outputs"

# RASTER mode uses one raster disaggregation folder by default.
# The IDF formulation needs ratios P(D)/P(24h), so relative_to_daily is the default.
DISAG_RASTER_DIR = ROOT / "Disag_Coefficients" / "relative_to_daily"

# STATION mode input.
STATION_DISAG_CSV = ROOT / "IDF_Fitting" / "Subhourly_Disag_Log.csv"

RETURN_PERIODS_YR = np.array([2, 5, 10, 25, 50, 75, 100], dtype=float)
DURATIONS_MIN = np.array([5, 10, 15, 20, 25, 30, 60, 360, 480, 600, 720, 1440], dtype=float)
MIN_YEARS_REQ = 5
ALPHA_KS = 0.05

# Sherman fitting settings matched to the MATLAB script.
# MATLAB requires at least 3 valid durations before fitting IDF.
# With 7 return periods, this gives at least 21 log-intensity points.
MIN_VALID_DURATIONS_FOR_IDF = 3

# MATLAB uses eps for lower bounds on K and c, and b >= 0.
# The upper bound for b is built dynamically as 5 * max(duration_used), matching:
#   ub = [Inf, 1, 5*max(DDg), 5]
SHERMAN_EPS = np.finfo(float).eps
SHERMAN_LOWER = np.array([SHERMAN_EPS, -1.0, 0.0, SHERMAN_EPS], dtype=float)

# CETESB depth ratios. We normalize by the 24 h coefficient and then force 24 h to 1.
CETESB_TABLE = {
    5: 0.120, 10: 0.191, 15: 0.248, 20: 0.287, 25: 0.322, 30: 0.354,
    60: 0.479, 360: 0.821, 480: 0.889, 600: 0.935, 720: 0.969, 1440: 1.140,
}

# Match MATLAB fallback: fill failed Sherman pixels from nearest successful fit inside domain.
FILL_FAILED_SHERMAN_PIXELS = True
COMPRESS = "deflate"
ZLEVEL = 6

# Quick-look plots for checking IDF results.
MAKE_QC_PLOTS = True
PLOT_DPI = 250
PLOT_PERCENTILE_LOW = 2.0
PLOT_PERCENTILE_HIGH = 98.0

# =============================================================================
# PRODUCT CONFIGURATION
# =============================================================================

@dataclass
class ProductConfig:
    label: str
    raw_dirs: List[Path]
    raw_pattern: str
    bc_dir: Path
    bc_pattern: str
    year_start: int
    year_end: int

PRODUCTS: Dict[str, ProductConfig] = {
    "br_dwgd": ProductConfig(
        label="BR-DWGD",
        raw_dirs=[RAW_ROOT / "BR-DWGD", RAW_ROOT / "BR_DWGD_Max", RAW_ROOT / "BRDWGD_Max", RAW_ROOT / "BR_DWGD"],
        raw_pattern="*.tif",
        bc_dir=BIAS_PIPELINE_ROOT / "br_dwgd" / "sensitivity" / "p98" / "annual_max_corrected" / "mean",
        bc_pattern="*.tif",
        year_start=1995,
        year_end=2025,
    ),
    "chirps": ProductConfig(
        label="CHIRPS",
        raw_dirs=[RAW_ROOT / "CHIRPS_Max", RAW_ROOT / "CHRIPS_Max", RAW_ROOT / "CHIRPS"],
        raw_pattern="*.tif",
        bc_dir=BIAS_PIPELINE_ROOT / "chirps" / "sensitivity" / "p98" / "annual_max_corrected" / "mean",
        bc_pattern="*.tif",
        year_start=1995,
        year_end=2025,
    ),
    "persiann_cdr": ProductConfig(
        label="PERSIANN-CDR",
        raw_dirs=[RAW_ROOT / "PERSIANN_CDR_Max", RAW_ROOT / "PERSIANN_Max", RAW_ROOT / "PERSIANN_CDR"],
        raw_pattern="*.tif",
        bc_dir=BIAS_PIPELINE_ROOT / "persiann_cdr" / "sensitivity" / "p98" / "annual_max_corrected" / "mean",
        bc_pattern="*.tif",
        year_start=1995,
        year_end=2025,
    ),
    "imerg_v06": ProductConfig(
        label="IMERG V06",
        raw_dirs=[RAW_ROOT / "IMERG_V06_Max", RAW_ROOT / "IMERG_Max", RAW_ROOT / "IMERG_V06"],
        raw_pattern="*.tif",
        bc_dir=BIAS_PIPELINE_ROOT / "imerg_v06" / "sensitivity" / "p98" / "annual_max_corrected" / "mean",
        bc_pattern="*.tif",
        year_start=2001,
        year_end=2020,
    ),
    "imerg_v07": ProductConfig(
        label="IMERG V07",
        raw_dirs=[RAW_ROOT / "IMERG_V07_Max", RAW_ROOT / "IMERG_V07"],
        raw_pattern="*.tif",
        bc_dir=BIAS_PIPELINE_ROOT / "imerg_v07" / "sensitivity" / "p98" / "annual_max_corrected" / "mean",
        bc_pattern="*.tif",
        year_start=2001,
        year_end=2025,
    ),
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnnualRasterStack:
    data: np.ndarray
    years: List[int]
    profile: dict
    height: int
    width: int
    transform: object
    crs: object

@dataclass
class DisaggregationResult:
    mode_name: str
    ratio: np.ndarray
    qc_outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

@dataclass
class DistributionResult:
    name: str
    q24: np.ndarray
    ks_d: np.ndarray
    ks_p: np.ndarray
    ks_reject: np.ndarray
    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)

# =============================================================================
# FILE AND RASTER UTILITIES
# =============================================================================

def extract_year(path: Path) -> Optional[int]:
    matches = re.findall(r"(19\d{2}|20\d{2})", path.name)
    return int(matches[-1]) if matches else None

def first_existing_directory(candidates: Sequence[Path]) -> Optional[Path]:
    for folder in candidates:
        if folder.exists():
            return folder
    return None

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
            ds.height == base_profile["height"] and ds.width == base_profile["width"]
            and ds.transform == base_profile["transform"] and ds.crs == base_profile["crs"]
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

def load_annual_stack(folder: Path, pattern: str, year_start: int, year_end: int) -> AnnualRasterStack:
    year_files = list_annual_rasters(folder, pattern, year_start, year_end)
    if not year_files:
        raise FileNotFoundError(f"No annual maximum rasters found in {folder} for {year_start}-{year_end}")
    years = [y for y, _ in year_files]
    paths = [p for _, p in year_files]
    first_arr, first_profile = read_single_band_float(paths[0])
    height, width = first_arr.shape
    base_profile = first_profile.copy()
    base_profile.update(height=height, width=width, transform=first_profile["transform"], crs=first_profile["crs"], count=1, dtype="float32")
    data = np.full((len(paths), height, width), np.nan, dtype="float32")
    data[0] = first_arr
    for i, p in enumerate(paths[1:], start=1):
        if same_grid(p, base_profile):
            arr, _ = read_single_band_float(p)
        else:
            print(f"    Regridding annual raster to base grid: {p.name}")
            arr = reproject_raster_to_base(p, base_profile, resampling=Resampling.bilinear)
        data[i] = arr
    return AnnualRasterStack(data=data, years=years, profile=base_profile, height=height, width=width, transform=base_profile["transform"], crs=base_profile["crs"])

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

def write_stack_geotiff(path: Path, arrays: Sequence[np.ndarray], names: Sequence[str], profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", count=len(arrays), dtype="float32", nodata=np.nan, compress=COMPRESS, predictor=2, zlevel=ZLEVEL)
    with rasterio.open(path, "w", **out_profile) as dst:
        for idx, (arr, name) in enumerate(zip(arrays, names), start=1):
            dst.write(arr.astype("float32"), idx)
            dst.set_band_description(idx, name)

# =============================================================================
# DISAGGREGATION LOADERS
# =============================================================================

def duration_labels(duration_min: int) -> List[str]:
    if duration_min < 60:
        return [f"{duration_min}m", f"{duration_min}min", f"{duration_min}_min", f"{duration_min}-min", f"p{duration_min}m"]
    hours = int(round(duration_min / 60))
    return [f"{hours}h", f"{hours}hr", f"{hours}_h", f"{hours}-h", f"p{hours}h", f"{duration_min}m", f"{duration_min}min"]

def label_matches_filename(label: str, filename: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(label.lower())}(?!\d)", filename.lower()) is not None

def find_duration_raster(disag_dir: Path, duration_min: int) -> Path:
    files = sorted(disag_dir.glob("*.tif")) + sorted(disag_dir.glob("*.tiff"))
    labels = duration_labels(duration_min)
    candidates = []
    for f in files:
        for label in labels:
            if label_matches_filename(label, f.name):
                candidates.append(f)
                break
    if not candidates:
        examples = "\n".join(f.name for f in files[:50])
        raise FileNotFoundError(f"Missing disaggregation raster for duration {duration_min} min in {disag_dir}\nAvailable examples:\n{examples}")
    def score(p: Path) -> Tuple[int, int, int, str]:
        name = p.name.lower()
        return (0 if "k10" in name else 1, 0 if ("p2.0" in name or "p2p0" in name) else 1, len(name), name)
    return sorted(candidates, key=score)[0]

def enforce_ratio_monotonicity_and_fill(ratio: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ratio = ratio.astype("float32", copy=True)
    ratio = np.maximum(ratio, 0)
    ratio[-1] = 1.0
    before = ratio.copy()
    ratio = np.maximum.accumulate(ratio, axis=0)
    ratio = np.minimum(ratio, 1.0)
    ratio[-1] = 1.0
    diff = np.where(np.isfinite(before) & np.isfinite(ratio), np.abs(ratio - before), 0)
    violation_count = np.sum(diff > 1e-6, axis=0).astype("float32")
    violation_magnitude = np.sum(diff, axis=0).astype("float32")
    clean = np.all(np.isfinite(ratio), axis=0)
    qc = {
        "QC_disagg_violation_count": violation_count,
        "QC_disagg_violation_magnitude": violation_magnitude,
        "QC_problem_mask": (~clean).astype("float32"),
    }
    if clean.all():
        rows, cols = clean.shape
        qc["QC_nearest_src_row"] = np.zeros((rows, cols), dtype="float32")
        qc["QC_nearest_src_col"] = np.zeros((rows, cols), dtype="float32")
        qc["QC_nearest_distance_px"] = np.zeros((rows, cols), dtype="float32")
        return ratio, qc
    if not clean.any():
        raise ValueError("No clean pixels in disaggregation ratio cube.")
    distance, indices = distance_transform_edt(~clean, return_indices=True)
    bad = ~clean
    src_r, src_c = indices[0], indices[1]
    filled = ratio.copy()
    for j in range(filled.shape[0]):
        band = filled[j]
        band[bad] = ratio[j, src_r[bad], src_c[bad]]
        filled[j] = band
    qc["QC_nearest_src_row"] = src_r.astype("float32")
    qc["QC_nearest_src_col"] = src_c.astype("float32")
    qc["QC_nearest_distance_px"] = distance.astype("float32")
    return filled, qc

def load_raster_disaggregation(base_profile: dict, disag_dir: Path) -> DisaggregationResult:
    if not disag_dir.exists():
        raise FileNotFoundError(disag_dir)
    n_dur = len(DURATIONS_MIN)
    ratio = np.full((n_dur, base_profile["height"], base_profile["width"]), np.nan, dtype="float32")
    print(f"  Loading RASTER disaggregation from: {disag_dir}")
    for j, duration in enumerate(DURATIONS_MIN.astype(int)):
        if duration == 1440:
            ratio[j] = 1.0
            continue
        path = find_duration_raster(disag_dir, duration)
        print(f"    {duration:4d} min -> {path.name}")
        ratio[j] = reproject_raster_to_base(path, base_profile, resampling=Resampling.bilinear).astype("float32")
    ratio, qc = enforce_ratio_monotonicity_and_fill(ratio)
    return DisaggregationResult(mode_name="RASTER", ratio=ratio, qc_outputs=qc)

def load_cetesb_disaggregation(base_profile: dict) -> DisaggregationResult:
    ratio_vec = np.array([CETESB_TABLE[int(d)] for d in DURATIONS_MIN], dtype="float32")
    ratio_vec = ratio_vec / ratio_vec[-1]
    ratio_vec[-1] = 1.0
    ratio_vec = np.maximum.accumulate(ratio_vec)
    ratio_vec = np.minimum(ratio_vec, 1.0)
    ratio_vec[-1] = 1.0
    ratio = np.broadcast_to(ratio_vec[:, None, None], (len(DURATIONS_MIN), base_profile["height"], base_profile["width"])).astype("float32")
    qc = {
        "QC_disagg_violation_count": np.zeros((base_profile["height"], base_profile["width"]), dtype="float32"),
        "QC_disagg_violation_magnitude": np.zeros((base_profile["height"], base_profile["width"]), dtype="float32"),
        "QC_problem_mask": np.zeros((base_profile["height"], base_profile["width"]), dtype="float32"),
    }
    return DisaggregationResult(mode_name="CETESB", ratio=ratio, qc_outputs=qc)

def raster_cell_center_coordinates(profile: dict) -> Tuple[np.ndarray, np.ndarray]:
    rows = np.arange(profile["height"])
    cols = np.arange(profile["width"])
    cc, rr = np.meshgrid(cols, rows)
    xs, ys = rasterio.transform.xy(profile["transform"], rr, cc, offset="center")
    return np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")

def load_station_disaggregation(base_profile: dict, station_csv: Path) -> DisaggregationResult:
    if not station_csv.exists():
        raise FileNotFoundError(station_csv)
    dur_cols = [f"all_c{int(d)}" for d in DURATIONS_MIN]
    df = pd.read_csv(station_csv)
    required = ["latitude", "longitude", "note"] + dur_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Station CSV is missing required columns: {missing}")
    has_ok = df["note"].astype(str).str.lower().str.contains("ok", na=False)
    vals = df[dur_cols].apply(pd.to_numeric, errors="coerce")
    has_all = vals.notna().all(axis=1)
    has_coords = pd.to_numeric(df["latitude"], errors="coerce").notna() & pd.to_numeric(df["longitude"], errors="coerce").notna()
    keep = has_ok & has_all & has_coords
    valid = df.loc[keep].copy()
    if valid.empty:
        raise ValueError("No valid station rows found for STATION mode.")
    ratio_station = np.array(valid[dur_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32"), copy=True)
    ratio_station[:, -1] = 1.0
    ratio_station = np.maximum.accumulate(ratio_station, axis=1)
    ratio_station = np.minimum(ratio_station, 1.0)
    ratio_station[:, -1] = 1.0
    station_lat = pd.to_numeric(valid["latitude"], errors="coerce").to_numpy(dtype="float64")
    station_lon = pd.to_numeric(valid["longitude"], errors="coerce").to_numpy(dtype="float64")
    lon_grid, lat_grid = raster_cell_center_coordinates(base_profile)
    tree = cKDTree(np.column_stack([station_lon, station_lat]))
    dist, idx = tree.query(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]), k=1)
    idx_grid = idx.reshape(base_profile["height"], base_profile["width"]).astype("int32")
    dist_grid = dist.reshape(base_profile["height"], base_profile["width"]).astype("float32")
    ratio = np.full((len(DURATIONS_MIN), base_profile["height"], base_profile["width"]), np.nan, dtype="float32")
    for j in range(len(DURATIONS_MIN)):
        ratio[j] = ratio_station[idx_grid, j]
    valid = valid.copy()
    valid["used_station_internal_index"] = np.arange(len(valid), dtype=int) + 1
    qc = {
        "QC_nearest_station_index": (idx_grid + 1).astype("float32"),
        "QC_nearest_station_distance_deg": dist_grid,
    }
    return DisaggregationResult(mode_name="STATION", ratio=ratio, qc_outputs=qc, tables={"station_mode_valid_stations": valid})

def get_disaggregation_result(mode: str, base_profile: dict, raster_disag_dir: Path, station_csv: Path) -> DisaggregationResult:
    mode = mode.upper()
    if mode == "RASTER":
        return load_raster_disaggregation(base_profile, raster_disag_dir)
    if mode == "CETESB":
        return load_cetesb_disaggregation(base_profile)
    if mode == "STATION":
        return load_station_disaggregation(base_profile, station_csv)
    raise ValueError(f"Unknown disaggregation mode: {mode}")

# =============================================================================
# EXTREME VALUE DISTRIBUTIONS
# =============================================================================

def gumbel_return_levels_moments(x: np.ndarray, return_periods: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    if len(x) < MIN_YEARS_REQ:
        return np.full(len(return_periods), np.nan, dtype="float32")
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return np.full(len(return_periods), np.nan, dtype="float32")
    yT = -np.log(np.log(return_periods / (return_periods - 1.0)))
    q = mean + std * (yT * (1.0 / 1.282) - 0.450047)
    return q.astype("float32")

def fit_gev_return_levels(x: np.ndarray, return_periods: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    """Fit GEV using scipy.stats.genextreme.

    SciPy uses the shape parameter c. In the usual hydrology convention,
    xi = -c. We use c for all scipy CDF/PPF calls and export xi separately.
    """
    x = x[np.isfinite(x)]
    if len(x) < MIN_YEARS_REQ:
        return np.full(len(return_periods), np.nan, dtype="float32"), np.nan, np.nan, np.nan, 0.0
    try:
        c, loc, scale = genextreme.fit(x)
        if not np.isfinite(c) or not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
            return np.full(len(return_periods), np.nan, dtype="float32"), float(c), float(loc), float(scale), 0.0
        q = genextreme.ppf(1.0 - 1.0 / return_periods, c, loc=loc, scale=scale)
        if not np.all(np.isfinite(q)):
            return np.full(len(return_periods), np.nan, dtype="float32"), float(c), float(loc), float(scale), 0.0
        return q.astype("float32"), float(c), float(loc), float(scale), 1.0
    except Exception:
        return np.full(len(return_periods), np.nan, dtype="float32"), np.nan, np.nan, np.nan, 0.0

def ks_statistic_from_cdf_values(cdf_values: np.ndarray) -> float:
    """Standard one-sample KS statistic from fitted CDF values.

    For sorted F(x_i):
      D+ = max(i/n - F(x_i))
      D- = max(F(x_i) - (i-1)/n)
      D  = max(D+, D-)
    """
    f = np.asarray(cdf_values, dtype=float)
    f = f[np.isfinite(f)]
    f = np.sort(f)
    n = len(f)
    if n == 0:
        return np.nan
    i = np.arange(1, n + 1, dtype=float)
    d_plus = np.max(i / n - f)
    d_minus = np.max(f - (i - 1) / n)
    return float(max(d_plus, d_minus))

def ks_pvalue_standard(d_stat: float, n: int) -> float:
    """Standard one-sample KS p-value.

    Because distribution parameters are fitted from the same sample, this p-value
    should be interpreted as a diagnostic screening metric.
    """
    if not np.isfinite(d_stat) or n < 1:
        return np.nan
    try:
        return float(kstwo.sf(float(d_stat), int(n)))
    except Exception:
        en = np.sqrt(float(n))
        lam = (en + 0.12 + 0.11 / en) * float(d_stat)
        p_val = 0.0
        for j in range(1, 101):
            p_val += ((-1.0) ** (j - 1)) * np.exp(-2.0 * (j ** 2) * (lam ** 2))
        return float(np.clip(2.0 * p_val, 0.0, 1.0))

def ks_gumbel_moments(x: np.ndarray) -> Tuple[float, float, float]:
    """KS diagnostic for Gumbel fitted by method of moments."""
    x = x[np.isfinite(x)].astype(float)
    x = x[x > 0]
    n = len(x)

    if n < MIN_YEARS_REQ:
        return np.nan, np.nan, np.nan

    mean = np.mean(x)
    std = np.std(x, ddof=1)

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return np.nan, np.nan, np.nan

    beta = std / (np.pi / np.sqrt(6.0))
    mu = mean - 0.5772156649015329 * beta

    if not np.isfinite(mu) or not np.isfinite(beta) or beta <= 0:
        return np.nan, np.nan, np.nan

    xs = np.sort(x)
    z = (xs - mu) / beta
    F = np.exp(-np.exp(-z))

    if not np.all(np.isfinite(F)):
        return np.nan, np.nan, np.nan

    d_stat = ks_statistic_from_cdf_values(F)
    p_val = ks_pvalue_standard(d_stat, n)
    reject = float(p_val < ALPHA_KS) if np.isfinite(p_val) else np.nan

    return float(d_stat), float(p_val), reject

def ks_gev(x: np.ndarray, c: float, loc: float, scale: float) -> Tuple[float, float, float]:
    """KS diagnostic for GEV fitted with scipy.stats.genextreme."""
    x = x[np.isfinite(x)].astype(float)
    x = x[x > 0]
    n = len(x)

    if (
        n < MIN_YEARS_REQ
        or not np.isfinite(c)
        or not np.isfinite(loc)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        return np.nan, np.nan, np.nan

    xs = np.sort(x)

    try:
        F = genextreme.cdf(xs, c, loc=loc, scale=scale)
    except Exception:
        return np.nan, np.nan, np.nan

    if not np.all(np.isfinite(F)):
        return np.nan, np.nan, np.nan

    d_stat = ks_statistic_from_cdf_values(F)
    p_val = ks_pvalue_standard(d_stat, n)
    reject = float(p_val < ALPHA_KS) if np.isfinite(p_val) else np.nan

    return float(d_stat), float(p_val), reject

def compute_distribution_outputs(annual_data: np.ndarray, distributions: Sequence[str]) -> Dict[str, DistributionResult]:
    _, rows, cols = annual_data.shape
    n_pix = rows * cols
    results = {}
    for dist in [d.upper() for d in distributions]:
        print(f"    Computing {dist} return levels and KS...")
        q24 = np.full((len(RETURN_PERIODS_YR), rows, cols), np.nan, dtype="float32")
        ks_d = np.full((rows, cols), np.nan, dtype="float32")
        ks_p = np.full((rows, cols), np.nan, dtype="float32")
        ks_reject = np.full((rows, cols), np.nan, dtype="float32")
        diagnostics = {}
        if dist == "GEV":
            diagnostics["GEV_shape_c"] = np.full((rows, cols), np.nan, dtype="float32")
            diagnostics["GEV_shape_xi_hydrology"] = np.full((rows, cols), np.nan, dtype="float32")
            diagnostics["GEV_location"] = np.full((rows, cols), np.nan, dtype="float32")
            diagnostics["GEV_scale"] = np.full((rows, cols), np.nan, dtype="float32")
            diagnostics["GEV_fit_success"] = np.full((rows, cols), np.nan, dtype="float32")
        for idx in range(n_pix):
            if idx % max(1, n_pix // 20) == 0:
                print(f"      {dist}: {100.0 * idx / n_pix:5.1f}%")
            r, cidx = divmod(idx, cols)
            x = annual_data[:, r, cidx]
            x = x[np.isfinite(x)]
            if len(x) < MIN_YEARS_REQ:
                continue
            if dist == "GUMBEL":
                q = gumbel_return_levels_moments(x, RETURN_PERIODS_YR)
                d_stat, p_val, reject = ks_gumbel_moments(x)
            elif dist == "GEV":
                q, gev_c, gev_loc, gev_scale, gev_ok = fit_gev_return_levels(x, RETURN_PERIODS_YR)
                diagnostics["GEV_shape_c"][r, cidx] = gev_c
                diagnostics["GEV_shape_xi_hydrology"][r, cidx] = -gev_c if np.isfinite(gev_c) else np.nan
                diagnostics["GEV_location"][r, cidx] = gev_loc
                diagnostics["GEV_scale"][r, cidx] = gev_scale
                diagnostics["GEV_fit_success"][r, cidx] = gev_ok
                d_stat, p_val, reject = ks_gev(x, gev_c, gev_loc, gev_scale)
            else:
                raise ValueError(f"Unknown distribution: {dist}")
            q24[:, r, cidx] = q
            ks_d[r, cidx] = d_stat
            ks_p[r, cidx] = p_val
            ks_reject[r, cidx] = reject
        results[dist] = DistributionResult(name=dist, q24=q24, ks_d=ks_d, ks_p=ks_p, ks_reject=ks_reject, diagnostics=diagnostics)
    return results

# =============================================================================
# SHERMAN FITTING
# =============================================================================

def sherman_log_model(xdata: np.ndarray, K: float, a: float, b: float, c: float) -> np.ndarray:
    T = xdata[:, 0]
    D = xdata[:, 1]
    return np.log10(K) + a * np.log10(T) - c * np.log10(b + D)

def fit_sherman_parameters(return_periods: np.ndarray, durations_min: np.ndarray, intensities: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Fit Sherman IDF parameters using the MATLAB workflow, except with SciPy's solver.

    MATLAB equivalent being matched:
      [TT, DD] = ndgrid(RETURN_PERIODS_YR, DURATIONS_MIN(validDur));
      y = log10(Igrid(:)); good = isfinite(y);
      if nnz(good) < 3, continue; end
      Xlin  = [ones(numel(y),1), log10(TT(:)), -log10(DD(:))];
      beta0 = Xlin(good,:) \ yg;
      K0 = 10.^beta0(1); a0 = beta0(2); c0 = max(0, -beta0(3));
      b0 = 0.1 * min(DURATIONS_MIN(validDur));
      lb = [eps, -1, 0, eps];
      ub = [Inf, 1, 5*max(DDg), 5];
    """
    TT, DD = np.meshgrid(return_periods, durations_min, indexing="ij")
    Igrid = intensities.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log10(Igrid.ravel())

    TTg_all = TT.ravel()
    DDg_all = DD.ravel()
    good = np.isfinite(y) & np.isfinite(TTg_all) & np.isfinite(DDg_all)

    # MATLAB uses: if nnz(good) < 3, continue; end
    if np.count_nonzero(good) < 3:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    xdata = np.column_stack([TTg_all[good], DDg_all[good]])
    ydata = y[good]

    # Linear Bernard warm start, matching MATLAB exactly in structure/sign.
    try:
        Xlin_all = np.column_stack([
            np.ones_like(y),
            np.log10(TTg_all),
            -np.log10(DDg_all),
        ])
        beta0, *_ = np.linalg.lstsq(Xlin_all[good, :], ydata, rcond=None)

        K0 = float(10.0 ** beta0[0])
        a0 = float(beta0[1])
        c0 = float(max(0.0, -beta0[2]))
        b0 = float(0.1 * np.min(durations_min))
    except Exception:
        # Fallback only if the MATLAB-equivalent linear warm start fails.
        K0, a0, b0, c0 = 1000.0, 0.1, 1.0, 0.7

    # MATLAB bounds:
    #   lb = [eps, -1, 0, eps]
    #   ub = [Inf, 1, 5*max(DDg), 5]
    lower = SHERMAN_LOWER
    upper = np.array([
        np.inf,
        1.0,
        5.0 * float(np.max(DDg_all)),
        5.0,
    ], dtype=float)

    # SciPy requires p0 to be feasible. MATLAB code provides the raw warm start;
    # this clipping is only to satisfy the SciPy solver while keeping the same bounds.
    p0 = np.array([K0, a0, b0, c0], dtype=float)
    p0 = np.where(np.isfinite(p0), p0, np.array([1000.0, 0.1, 1.0, 0.7], dtype=float))
    p0 = np.minimum(np.maximum(p0, lower), upper)

    try:
        popt, _ = curve_fit(
            sherman_log_model,
            xdata,
            ydata,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000,
        )
        K, a, b, cpar = [float(v) for v in popt]

        # Match MATLAB metric calculation.
        Ihat = (K * (TT ** a)) / ((b + DD) ** cpar)
        Iobs = Igrid.ravel()
        Ihat_flat = Ihat.ravel()
        valid = np.isfinite(Iobs) & np.isfinite(Ihat_flat)

        if not np.any(valid):
            return (K, a, b, cpar, np.nan, np.nan, np.nan)

        residual = Iobs[valid] - Ihat_flat[valid]
        mse = float(np.mean(residual ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum((Iobs[valid] - Ihat_flat[valid]) ** 2))
        mean_iobs = float(np.mean(Iobs[valid]))
        ss_tot = float(np.sum((Iobs[valid] - mean_iobs) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan

        return (K, a, b, cpar, r2, rmse, mse)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

def fill_failed_sherman_from_nearest(bands: np.ndarray, valid_fit: np.ndarray, domain: np.ndarray) -> np.ndarray:
    if not FILL_FAILED_SHERMAN_PIXELS or valid_fit.all() or not valid_fit.any():
        return bands
    _, indices = distance_transform_edt(~valid_fit, return_indices=True)
    bad = (~valid_fit) & domain
    src_r, src_c = indices[0][bad], indices[1][bad]
    filled = bands.copy()
    for j in range(filled.shape[0]):
        band = filled[j]
        band[bad] = bands[j, src_r, src_c]
        band[~domain] = np.nan
        filled[j] = band
    return filled

def fit_idf_for_distribution(
    dist_result: DistributionResult,
    intensity_scale: np.ndarray,
    fit_domain: np.ndarray,
    fill_domain: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Fit Sherman IDF parameters per pixel using MATLAB's fitting logic."""
    _, rows, cols = dist_result.q24.shape
    n_pix = rows * cols
    K_img = np.full((rows, cols), np.nan, dtype="float32")
    a_img = np.full((rows, cols), np.nan, dtype="float32")
    b_img = np.full((rows, cols), np.nan, dtype="float32")
    c_img = np.full((rows, cols), np.nan, dtype="float32")
    r2_img = np.full((rows, cols), np.nan, dtype="float32")
    rmse_img = np.full((rows, cols), np.nan, dtype="float32")
    mse_img = np.full((rows, cols), np.nan, dtype="float32")

    print(f"    Fitting Sherman IDF for {dist_result.name}...")
    for idx in range(n_pix):
        if idx % max(1, n_pix // 20) == 0:
            print(f"      Sherman {dist_result.name}: {100.0 * idx / n_pix:5.1f}%")

        r, cidx = divmod(idx, cols)

        # MATLAB fitting skip condition:
        #   if ~isfinite(AMD_mean(p)) || ~isfinite(AMD_std(p)) || AMD_std(p)==0, continue; end
        if not fit_domain[r, cidx]:
            continue

        q = dist_result.q24[:, r, cidx]
        if not np.all(np.isfinite(q)):
            continue

        scale_vec = intensity_scale[:, r, cidx]
        valid_dur = np.isfinite(scale_vec) & (scale_vec > 0)
        if np.count_nonzero(valid_dur) < MIN_VALID_DURATIONS_FOR_IDF:
            continue

        durations_use = DURATIONS_MIN[valid_dur]
        scale_use = scale_vec[valid_dur]

        intensities = q[:, None] * scale_use[None, :]

        K, a, b, cpar, r2, rmse, mse = fit_sherman_parameters(
            RETURN_PERIODS_YR,
            durations_use,
            intensities,
        )

        K_img[r, cidx] = K
        a_img[r, cidx] = a
        b_img[r, cidx] = b
        c_img[r, cidx] = cpar
        r2_img[r, cidx] = r2
        rmse_img[r, cidx] = rmse
        mse_img[r, cidx] = mse

    valid_fit = (
        np.isfinite(K_img)
        & np.isfinite(a_img)
        & np.isfinite(b_img)
        & np.isfinite(c_img)
        & np.isfinite(r2_img)
        & fill_domain
    )
    bands = np.stack([K_img, a_img, b_img, c_img, r2_img, rmse_img, mse_img], axis=0)
    bands = fill_failed_sherman_from_nearest(bands, valid_fit, fill_domain)
    K_img, a_img, b_img, c_img, r2_img, rmse_img, mse_img = bands

    return {"K": K_img, "a": a_img, "b": b_img, "c": c_img, "R2": r2_img, "RMSE": rmse_img, "MSE": mse_img}


# =============================================================================
# QUICK-LOOK PLOTS
# =============================================================================

def finite_values(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype="float64").ravel()
    return vals[np.isfinite(vals)]


def robust_limits(arr: np.ndarray, low: float = PLOT_PERCENTILE_LOW, high: float = PLOT_PERCENTILE_HIGH) -> Tuple[Optional[float], Optional[float]]:
    vals = finite_values(arr)
    if vals.size == 0:
        return None, None
    vmin = float(np.nanpercentile(vals, low))
    vmax = float(np.nanpercentile(vals, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-6)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def raster_extent_from_profile(profile: dict) -> List[float]:
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return [left, right, bottom, top]


def choose_plot_norm(arr: np.ndarray, key: str):
    key_lower = key.lower()
    vmin, vmax = robust_limits(arr)
    if vmin is None or vmax is None:
        return None

    if key_lower in {"r2", "ks_p", "gev_fit_success"}:
        return Normalize(vmin=0.0, vmax=1.0)

    if key_lower in {"a", "gev_shape_c"}:
        vals = finite_values(arr)
        if vals.size > 0 and np.nanmin(vals) < 0 < np.nanmax(vals):
            lim = float(np.nanpercentile(np.abs(vals), PLOT_PERCENTILE_HIGH))
            lim = max(lim, 1e-6)
            return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        return Normalize(vmin=vmin, vmax=vmax)

    if key_lower in {"k", "b", "rmse", "mse"}:
        vals = finite_values(arr)
        vals = vals[vals > 0]
        if vals.size > 10:
            lvmin = float(np.nanpercentile(vals, PLOT_PERCENTILE_LOW))
            lvmax = float(np.nanpercentile(vals, PLOT_PERCENTILE_HIGH))
            if lvmin > 0 and lvmax > lvmin:
                return LogNorm(vmin=lvmin, vmax=lvmax)

    return Normalize(vmin=vmin, vmax=vmax)


def choose_plot_cmap(key: str) -> str:
    key_lower = key.lower()
    if key_lower in {"r2", "ks_p", "gev_fit_success", "nyears"}:
        return "viridis"
    if key_lower in {"k", "q24_rp100", "amd_mean"}:
        return "magma"
    if key_lower in {"a", "gev_shape_c"}:
        return "coolwarm"
    if key_lower in {"b", "c"}:
        return "plasma"
    if key_lower in {"rmse", "mse", "amd_std"}:
        return "cividis"
    return "viridis"


def add_map_panel(fig, ax, arr: np.ndarray, profile: dict, title: str, key: str) -> None:
    extent = raster_extent_from_profile(profile)
    norm = choose_plot_norm(arr, key)
    cmap = choose_plot_cmap(key)

    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, length=2)


def plot_idf_parameter_maps(
    out_dir: Path,
    profile: dict,
    valid_count: np.ndarray,
    amd_mean: np.ndarray,
    amd_std: np.ndarray,
    dist: DistributionResult,
    sherman: Dict[str, np.ndarray],
) -> None:
    if not HAS_MATPLOTLIB or not MAKE_QC_PLOTS:
        return

    fig_dir = out_dir / "quicklook_plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    q100 = dist.q24[-1]

    panels: List[Tuple[str, str, np.ndarray]] = [
        ("K", "K", sherman["K"]),
        ("a", "a", sherman["a"]),
        ("b", "b", sherman["b"]),
        ("c", "c", sherman["c"]),
        ("R²", "R2", sherman["R2"]),
        ("RMSE", "RMSE", sherman["RMSE"]),
        ("KS p-value", "KS_p", dist.ks_p),
        ("Q24 RP100 (mm)", "Q24_RP100", q100),
        ("N years", "Nyears", valid_count),
        ("AMD mean (mm)", "AMD_mean", amd_mean),
        ("AMD std (mm)", "AMD_std", amd_std),
    ]

    if dist.name == "GEV" and "GEV_shape_c" in dist.diagnostics:
        panels.append(("GEV shape c", "GEV_shape_c", dist.diagnostics["GEV_shape_c"]))
    else:
        panels.append(("MSE", "MSE", sherman["MSE"]))

    fig, axs = plt.subplots(3, 4, figsize=(12.5, 8.8), dpi=PLOT_DPI)
    axs = axs.ravel()

    for ax, (title, key, arr) in zip(axs, panels):
        add_map_panel(fig, ax, arr, profile, title, key)

    for ax in axs[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"IDF quick-look maps — {dist.name}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"quicklook_maps_{dist.name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_idf_parameter_histograms(
    out_dir: Path,
    dist: DistributionResult,
    sherman: Dict[str, np.ndarray],
) -> None:
    if not HAS_MATPLOTLIB or not MAKE_QC_PLOTS:
        return

    fig_dir = out_dir / "quicklook_plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    q100 = dist.q24[-1]
    hist_items: List[Tuple[str, np.ndarray, str]] = [
        ("K", sherman["K"], "K"),
        ("a", sherman["a"], "a"),
        ("b", sherman["b"], "b"),
        ("c", sherman["c"], "c"),
        ("R²", sherman["R2"], "R2"),
        ("RMSE", sherman["RMSE"], "RMSE"),
        ("KS p-value", dist.ks_p, "KS_p"),
        ("Q24 RP100", q100, "Q24_RP100"),
    ]

    if dist.name == "GEV" and "GEV_shape_c" in dist.diagnostics:
        hist_items.append(("GEV shape c", dist.diagnostics["GEV_shape_c"], "GEV_shape_c"))

    n = len(hist_items)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(11.5, 3.0 * nrows), dpi=PLOT_DPI)
    axs = np.asarray(axs).ravel()

    for ax, (title, arr, key) in zip(axs, hist_items):
        vals = finite_values(arr)
        if vals.size == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        vmin, vmax = robust_limits(vals)
        if vmin is not None and vmax is not None:
            vals_plot = vals[(vals >= vmin) & (vals <= vmax)]
        else:
            vals_plot = vals

        ax.hist(vals_plot, bins=50, color="#3B7EA1", edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.axvline(np.nanmedian(vals), color="#D1495B", lw=1.8, ls="--", label="median")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, ls=":", lw=0.5, alpha=0.7)
        ax.legend(fontsize=7, frameon=False)

    for ax in axs[len(hist_items):]:
        ax.axis("off")

    fig.suptitle(f"IDF parameter histograms — {dist.name}", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"quicklook_histograms_{dist.name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_idf_quicklook_plots(
    out_dir: Path,
    profile: dict,
    valid_count: np.ndarray,
    amd_mean: np.ndarray,
    amd_std: np.ndarray,
    dist: DistributionResult,
    sherman: Dict[str, np.ndarray],
) -> None:
    """Create simple QC plots from fitted IDF parameters and distribution diagnostics."""
    if not HAS_MATPLOTLIB or not MAKE_QC_PLOTS:
        if not HAS_MATPLOTLIB:
            print("    Matplotlib not available; skipping quick-look plots.")
        return

    try:
        plot_idf_parameter_maps(out_dir, profile, valid_count, amd_mean, amd_std, dist, sherman)
        plot_idf_parameter_histograms(out_dir, dist, sherman)
        print("    Quick-look plots saved in:", out_dir / "quicklook_plots")
    except Exception as exc:
        print(f"    WARNING: quick-look plot generation failed for {dist.name}: {exc}")

# =============================================================================
# OUTPUTS
# =============================================================================


def summarize_distribution_result(dist: DistributionResult) -> Dict[str, float]:
    valid = np.isfinite(dist.ks_d)
    if not np.any(valid):
        return {
            "distribution": dist.name,
            "valid_pixels": 0,
            "ks_D_mean": np.nan,
            "ks_D_median": np.nan,
            "ks_p_mean": np.nan,
            "ks_p_median": np.nan,
            "reject_pct_p_lt_alpha": np.nan,
        }
    reject = np.isfinite(dist.ks_p) & (dist.ks_p < ALPHA_KS)
    return {
        "distribution": dist.name,
        "valid_pixels": int(np.sum(valid)),
        "ks_D_mean": float(np.nanmean(dist.ks_d[valid])),
        "ks_D_median": float(np.nanmedian(dist.ks_d[valid])),
        "ks_p_mean": float(np.nanmean(dist.ks_p[valid])),
        "ks_p_median": float(np.nanmedian(dist.ks_p[valid])),
        "reject_pct_p_lt_alpha": float(100.0 * np.sum(reject & valid) / np.sum(valid)),
    }

def write_distribution_fit_outputs(
    out_dir: Path,
    profile: dict,
    valid_count: np.ndarray,
    amd_mean: np.ndarray,
    amd_std: np.ndarray,
    distribution_results: Dict[str, DistributionResult],
) -> None:
    """Write daily annual-maximum distribution diagnostics once per product/state.

    These outputs are independent of temporal disaggregation and Sherman IDF
    fitting. Use this folder as the authoritative source for KS maps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    write_geotiff(out_dir / "DIST_AMD_mean.tif", amd_mean, profile)
    write_geotiff(out_dir / "DIST_AMD_std.tif", amd_std, profile)
    write_geotiff(out_dir / "DIST_Nyears.tif", valid_count, profile)

    rows = []
    for dname, dist in distribution_results.items():
        write_geotiff(out_dir / f"DIST_KS_D_{dname}.tif", dist.ks_d, profile)
        write_geotiff(out_dir / f"DIST_KS_p_{dname}.tif", dist.ks_p, profile)
        write_geotiff(out_dir / f"DIST_KS_reject_{dname}.tif", dist.ks_reject, profile)
        for i, T in enumerate(RETURN_PERIODS_YR.astype(int)):
            write_geotiff(out_dir / f"DIST_Q24_{dname}_RP{T:03d}.tif", dist.q24[i], profile)
        for name, arr in dist.diagnostics.items():
            write_geotiff(out_dir / f"DIST_{name}.tif", arr, profile)
        rows.append(summarize_distribution_result(dist))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "distribution_fit_summary.csv", index=False)

    readme = out_dir / "README_distribution_fit_outputs.txt"
    readme.write_text(
        "Authoritative daily annual-maximum distribution-fit diagnostics.\n"
        "\n"
        "These outputs are computed before temporal disaggregation and before\n"
        "Sherman IDF fitting. Therefore, they are independent of RASTER,\n"
        "CETESB, and STATION disaggregation modes.\n"
        "\n"
        "KS statistic: standard one-sample D = max(D+, D-).\n"
        "KS p-value: scipy.stats.kstwo.sf(D, n).\n"
        f"Reject threshold alpha: {ALPHA_KS}.\n"
        "\n"
        "Use DIST_KS_D_* and DIST_KS_p_* for paper figures.\n"
    )

    print("Distribution-fit diagnostics saved in:", out_dir)
    print(summary.to_string(index=False))

def write_mode_outputs(out_dir: Path, profile: dict, valid_count: np.ndarray, amd_mean: np.ndarray, amd_std: np.ndarray, disagg: DisaggregationResult, dist: DistributionResult, sherman: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dname = dist.name
    for key in ["K", "a", "b", "c", "R2", "RMSE", "MSE"]:
        write_geotiff(out_dir / f"IDF_{key}_{dname}.tif", sherman[key], profile)
    write_geotiff(out_dir / f"IDF_KS_D_{dname}.tif", dist.ks_d, profile)
    write_geotiff(out_dir / f"IDF_KS_p_{dname}.tif", dist.ks_p, profile)
    write_geotiff(out_dir / f"IDF_KS_reject_{dname}.tif", dist.ks_reject, profile)
    for i, T in enumerate(RETURN_PERIODS_YR.astype(int)):
        write_geotiff(out_dir / f"Q24_{dname}_RP{T:03d}.tif", dist.q24[i], profile)
    for name, arr in dist.diagnostics.items():
        write_geotiff(out_dir / f"{name}.tif", arr, profile)
    write_geotiff(out_dir / "IDF_AMD_mean.tif", amd_mean, profile)
    write_geotiff(out_dir / "IDF_AMD_std.tif", amd_std, profile)
    write_geotiff(out_dir / "IDF_Nyears.tif", valid_count, profile)
    for name, arr in disagg.qc_outputs.items():
        write_geotiff(out_dir / f"{name}.tif", arr.astype("float32"), profile)
    for table_name, df in disagg.tables.items():
        df.to_csv(out_dir / f"{table_name}.csv", index=False)
    arrays = [sherman["K"], sherman["a"], sherman["b"], sherman["c"], sherman["R2"], sherman["RMSE"], sherman["MSE"], dist.ks_d, dist.ks_p, dist.ks_reject, amd_mean, amd_std, valid_count]
    names = ["K", "a", "b", "c", "R2", "RMSE", "MSE", "KS_D", "KS_p", "KS_reject", "AMD_mean", "AMD_std", "Nyears"]
    for name, arr in dist.diagnostics.items():
        arrays.append(arr)
        names.append(name)
    write_stack_geotiff(out_dir / f"IDF_params_stack_{dname}.tif", arrays, names, profile)

    # Quick-look plots for visual quality control.
    make_idf_quicklook_plots(
        out_dir=out_dir,
        profile=profile,
        valid_count=valid_count,
        amd_mean=amd_mean,
        amd_std=amd_std,
        dist=dist,
        sherman=sherman,
    )

# =============================================================================
# PIPELINE
# =============================================================================

def normalize_bc_percentile(value: Optional[str]) -> Optional[str]:
    """
    Normalize bias-correction percentile labels.

    Accepted examples:
        p90, 90, 0.90
        p95, 95, 0.95
        p98, 98, 0.98
        p99, 99, 0.99
        p995, 99.5, 0.995

    Returns:
        None if value is None or empty, otherwise a label like "p98".
    """
    if value is None:
        return None

    s = str(value).strip().lower()
    if s == "":
        return None

    if s.startswith("p"):
        s2 = s[1:].replace(".", "")
        if s2 in {"90", "95", "98", "99", "995"}:
            return f"p{s2}"

    try:
        x = float(s)
        if 0 < x < 1:
            pct = x * 100.0
        else:
            pct = x

        if abs(pct - 90.0) < 1e-9:
            return "p90"
        if abs(pct - 95.0) < 1e-9:
            return "p95"
        if abs(pct - 98.0) < 1e-9:
            return "p98"
        if abs(pct - 99.0) < 1e-9:
            return "p99"
        if abs(pct - 99.5) < 1e-9:
            return "p995"
    except Exception:
        pass

    raise ValueError(
        f"Invalid --bc-percentile value: {value}. "
        "Use one of: p90, p95, p98, p99, p995."
    )


def output_root_for_product_state(
    state: str,
    product_key: str,
    bc_percentile: Optional[str] = None,
) -> Path:
    """
    Return the output folder for one product/state.

    Backward compatibility:
      - If --bc-percentile is not provided, outputs remain:
            Outputs/<state>/<product>/

    Sensitivity runs:
      - If --bc-percentile is provided and state is bias_corrected_mean:
            Outputs/sensitivity/<pXX>/bias_corrected_mean/<product>/
    """
    if state.lower() == "bias_corrected_mean" and bc_percentile is not None:
        return OUT_ROOT / "sensitivity" / bc_percentile / state / product_key

    return OUT_ROOT / state / product_key

def prepare_annual_statistics(stack: AnnualRasterStack) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_count = np.sum(np.isfinite(stack.data), axis=0).astype("float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        amd_mean = np.nanmean(stack.data, axis=0).astype("float32")
        amd_std = np.nanstd(stack.data, axis=0, ddof=1).astype("float32")
    amd_mean[valid_count < MIN_YEARS_REQ] = np.nan
    amd_std[valid_count < MIN_YEARS_REQ] = np.nan
    domain = np.isfinite(amd_mean) & np.isfinite(amd_std) & (amd_std > 0)
    return valid_count, amd_mean, amd_std, domain

def resolve_input_for_state(
    product_key: str,
    cfg: ProductConfig,
    state: str,
    bc_percentile: Optional[str] = None,
) -> Tuple[Path, str]:
    if state.lower() == "raw":
        folder = first_existing_directory(cfg.raw_dirs)
        if folder is None:
            raise FileNotFoundError("No raw annual maximum folder found among: " + ", ".join(str(p) for p in cfg.raw_dirs))
        return folder, cfg.raw_pattern

    if state.lower() == "bias_corrected_mean":
        # Default behavior remains p98 unless --bc-percentile is explicitly provided.
        pct = bc_percentile if bc_percentile is not None else "p98"
        folder = (
            BIAS_PIPELINE_ROOT
            / product_key
            / "sensitivity"
            / pct
            / "annual_max_corrected"
            / "mean"
        )
        return folder, cfg.bc_pattern

    raise ValueError(f"Unknown state: {state}")

def process_one_product_state(
    product_key: str,
    cfg: ProductConfig,
    state: str,
    modes: Sequence[str],
    distributions: Sequence[str],
    raster_disag_dir: Path,
    station_csv: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    bc_percentile: Optional[str] = None,
) -> None:
    folder, pattern = resolve_input_for_state(
        product_key=product_key,
        cfg=cfg,
        state=state,
        bc_percentile=bc_percentile,
    )
    product_out_root = output_root_for_product_state(
        state=state,
        product_key=product_key,
        bc_percentile=bc_percentile,
    )
    year_files = list_annual_rasters(folder, pattern, cfg.year_start, cfg.year_end)
    print("\n" + "=" * 110)
    print(f"PRODUCT: {product_key} ({cfg.label}) | STATE: {state}")
    if state.lower() == "bias_corrected_mean":
        print(f"Bias-correction percentile: {bc_percentile if bc_percentile is not None else 'p98'}")
    print("=" * 110)
    print("Input folder:", folder)
    print("Pattern:", pattern)
    print("Year window:", cfg.year_start, cfg.year_end)
    print("Files found:", len(year_files))
    if year_files:
        years = [y for y, _ in year_files]
        print("Years:", years[0], "to", years[-1], "|", years)
    else:
        print("WARNING: no files found.")
        return
    if dry_run:
        for mode in modes:
            print("Would write:", product_out_root / mode.upper())
        return
    print("Loading annual maximum raster stack...")
    stack = load_annual_stack(folder, pattern, cfg.year_start, cfg.year_end)
    print("Loaded stack:", stack.data.shape, "| native grid preserved")
    print("CRS:", stack.crs)
    valid_count, amd_mean, amd_std, domain = prepare_annual_statistics(stack)
    print("Valid domain pixels:", int(np.sum(domain)))
    print("Computing distribution return levels once for this product/state...")
    distribution_results = compute_distribution_outputs(stack.data, distributions)

    # Authoritative daily distribution diagnostics.
    # These are independent of RASTER/CETESB/STATION disaggregation modes.
    distribution_out_dir = product_out_root / "DISTRIBUTION_FIT"
    write_distribution_fit_outputs(
        out_dir=distribution_out_dir,
        profile=stack.profile,
        valid_count=valid_count,
        amd_mean=amd_mean,
        amd_std=amd_std,
        distribution_results=distribution_results,
    )

    for mode in [m.upper() for m in modes]:
        out_mode_dir = product_out_root / mode
        if out_mode_dir.exists() and not overwrite:
            print(f"\nSkipping existing output folder: {out_mode_dir}")
            print("Use --overwrite to rebuild.")
            continue
        print("\n" + "-" * 100)
        print(f"DISAGGREGATION MODE: {mode}")
        print("-" * 100)
        disagg = get_disaggregation_result(mode, stack.profile, raster_disag_dir, station_csv)
        intensity_scale = disagg.ratio / (DURATIONS_MIN / 60.0)[:, None, None]
        for _, dist_result in distribution_results.items():
            sherman = fit_idf_for_distribution(dist_result, intensity_scale, domain, amd_mean > 0)
            write_mode_outputs(out_mode_dir, stack.profile, valid_count, amd_mean, amd_std, disagg, dist_result, sherman)
        print("Saved mode output:", out_mode_dir)

def run_complete_idf_pipeline(
    products: Optional[Sequence[str]] = None,
    states: Optional[Sequence[str]] = None,
    modes: Optional[Sequence[str]] = None,
    distributions: Optional[Sequence[str]] = None,
    raster_disag_dir: Path = DISAG_RASTER_DIR,
    station_csv: Path = STATION_DISAG_CSV,
    overwrite: bool = False,
    dry_run: bool = False,
    bc_percentile: Optional[str] = None,
) -> None:
    selected_products = list(products) if products is not None else list(PRODUCTS.keys())
    selected_states = list(states) if states is not None else ["raw", "bias_corrected_mean"]
    selected_modes = [m.upper() for m in (modes if modes is not None else ["RASTER", "CETESB", "STATION"])]
    selected_distributions = [d.upper() for d in (distributions if distributions is not None else ["GUMBEL", "GEV"])]
    selected_bc_percentile = normalize_bc_percentile(bc_percentile)
    invalid_products = [p for p in selected_products if p not in PRODUCTS]
    if invalid_products:
        raise ValueError(f"Unknown products: {invalid_products}. Valid products: {list(PRODUCTS)}")
    invalid_modes = [m for m in selected_modes if m not in {"RASTER", "CETESB", "STATION"}]
    if invalid_modes:
        raise ValueError(f"Unknown modes: {invalid_modes}. Use RASTER,CETESB,STATION.")
    invalid_dists = [d for d in selected_distributions if d not in {"GUMBEL", "GEV"}]
    if invalid_dists:
        raise ValueError(f"Unknown distributions: {invalid_dists}. Use GUMBEL,GEV.")
    print("\n" + "#" * 110)
    print("COMPLETE GRIDF IDF PIPELINE")
    print("#" * 110)
    print("Output root:", OUT_ROOT)
    print("Products:", selected_products)
    print("States:", selected_states)
    print("Modes:", selected_modes)
    print("Distributions:", selected_distributions)
    print("Bias-correction percentile:", selected_bc_percentile if selected_bc_percentile is not None else "default p98")
    if selected_bc_percentile is not None and "bias_corrected_mean" in [s.lower() for s in selected_states]:
        print("Sensitivity output root:", OUT_ROOT / "sensitivity" / selected_bc_percentile)
    print("Raster disaggregation directory:", raster_disag_dir)
    print("Station disaggregation CSV:", station_csv)
    print("Overwrite:", overwrite)
    print("Dry run:", dry_run)
    print("#" * 110)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for product_key in selected_products:
        for state in selected_states:
            process_one_product_state(
                product_key,
                PRODUCTS[product_key],
                state,
                selected_modes,
                selected_distributions,
                raster_disag_dir,
                station_csv,
                overwrite,
                dry_run,
                bc_percentile=selected_bc_percentile,
            )
    print("\n" + "#" * 110)
    print("IDF PIPELINE FINISHED")
    print("#" * 110)
    print("Outputs are in:", OUT_ROOT)

# =============================================================================
# CLI
# =============================================================================

def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [v.strip() for v in value.split(",") if v.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete GRIDF IDF pipeline.")
    parser.add_argument("--products", default=None, help="Comma-separated products. Default: all. Example: chirps,imerg_v07")
    parser.add_argument("--states", default=None, help="Comma-separated states. Default: raw,bias_corrected_mean")
    parser.add_argument("--modes", default=None, help="Comma-separated disaggregation modes. Default: RASTER,CETESB,STATION")
    parser.add_argument("--distributions", default=None, help="Comma-separated distributions. Default: GUMBEL,GEV")
    parser.add_argument(
        "--bc-percentile",
        default=None,
        help=(
            "Bias-correction percentile folder to use for bias_corrected_mean. "
            "Examples: p90, p95, p98, p99, p995. "
            "If omitted, the pipeline keeps the default p98 behavior and old output paths."
        ),
    )
    parser.add_argument("--raster-disag-dir", default=str(DISAG_RASTER_DIR), help="Folder with raster disaggregation coefficients for RASTER mode.")
    parser.add_argument("--station-csv", default=str(STATION_DISAG_CSV), help="CSV with station coefficients for STATION mode.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing output folders.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs and planned outputs without processing rasters.")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate quick-look PNG/PDF/SVG plots.")
    args = parser.parse_args()
    global MAKE_QC_PLOTS
    if args.no_plots:
        MAKE_QC_PLOTS = False
    run_complete_idf_pipeline(
        products=parse_list_arg(args.products),
        states=parse_list_arg(args.states),
        modes=parse_list_arg(args.modes),
        distributions=parse_list_arg(args.distributions),
        raster_disag_dir=Path(args.raster_disag_dir),
        station_csv=Path(args.station_csv),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        bc_percentile=args.bc_percentile,
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
