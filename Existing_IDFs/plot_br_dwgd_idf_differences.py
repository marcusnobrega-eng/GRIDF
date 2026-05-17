#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIDF / BR-DWGD IDF comparison using ONLY IDF_Curves_Filtered.xlsx
for the station-based existing IDF equations.

This script makes ONE figure with four rows of subplots:

    Rows 1--2: Existing STANDARD/sub-daily IDFs versus GRIDF/BR-DWGD raster IDFs
    Rows 3--4: Existing DISAGGREGATION IDFs versus GRIDF/BR-DWGD raster IDFs

Important change relative to the previous script:
    - Existing IDF coefficients are read directly from IDF_Curves_Filtered.xlsx.
    - The script does NOT read IDF_metrics_by_station.csv.
    - The script does NOT read Disaggregation_Stations_WithMetrics.csv.
    - GRIDF/BR-DWGD coefficients are sampled directly from raster files at the
      station coordinates in the Excel workbook.

Therefore, you need four GRIDF/BR-DWGD raster coefficient files:

    K_r raster
    a_r raster
    b_r raster
    c_r raster

The script can try to auto-find these rasters, but the safest option is to set
explicit paths in Config.gridf_parameter_rasters.

IDF equation used:
    I(d,T) = K * T^a / (d + b)^c

where:
    I = rainfall intensity [mm h^-1]
    d = duration [min]
    T = return period [yr]

If quantity_mode = "depth":
    P(d,T) = I(d,T) * d / 60

Bias on the right y-axis:
    bias [%] = 100 * (GRIDF - Existing IDFs) / Existing IDFs

Positive bias means GRIDF/BR-DWGD is larger than the station-based IDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import warnings
import re

import numpy as np
import pandas as pd

import rasterio
from rasterio.warp import transform as rio_transform

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


# =============================================================================
# USER SETTINGS
# =============================================================================

@dataclass
class Config:
    # Main GRIDF folder on your Mac.
    root: Path = Path("/Users/mngomes/Documents/GitHub/GRIDF")

    # Existing IDF workbook. This is now the ONLY existing-IDF table used.
    idf_workbook_xlsx: Path = root / "Existing_IDFs" / "IDF_Curves_Filtered.xlsx"
    standard_sheet_name: str = "Standard"
    disaggregation_sheet_name: str = "Disaggregation"

    # ---------------------------------------------------------------------
    # GRIDF / BR-DWGD raster coefficient inputs
    # ---------------------------------------------------------------------
    # Safest option: explicitly set the four paths below.
    # Example:
    # gridf_parameter_rasters = {
    #     "K": Path("/Users/mngomes/Documents/GitHub/GRIDF/.../K.tif"),
    #     "a": Path("/Users/mngomes/Documents/GitHub/GRIDF/.../a.tif"),
    #     "b": Path("/Users/mngomes/Documents/GitHub/GRIDF/.../b.tif"),
    #     "c": Path("/Users/mngomes/Documents/GitHub/GRIDF/.../c.tif"),
    # }
    #
    # If left as None, the script will try to auto-find raster files under
    # gridf_raster_search_dirs.
    gridf_parameter_rasters: dict[str, Path] | None = None

    # Auto-search folders. Keep this reasonably narrow so the search is fast.
    gridf_raster_search_dirs: tuple[Path, ...] = (
        root / "IDF_Fitting",
        root / "Outputs",
        root / "Existing_IDFs",
        root,
    )
    auto_find_gridf_rasters: bool = True

    # Keywords preferred when auto-finding files.
    # The search still works if filenames do not contain all terms, but these
    # terms increase the score of the candidate.
    gridf_dataset_keywords: tuple[str, ...] = ("br", "dwgd")
    gridf_preferred_keywords: tuple[str, ...] = (
        "idf", "parameter", "parameters", "coefficient", "coefficients",
        "raster", "fit", "fitted"
    )

    # Output folder.
    out_dir: Path = root / "Existing_IDFs" / "Figures_IDF_Curves_Comparison_XLSX"

    # Blocks to plot. Options: "standard", "disaggregation".
    # With both blocks and 8 return periods, the figure has 4 rows.
    reference_blocks: tuple[str, ...] = ("standard", "disaggregation")

    # Return periods to plot in each block.
    # With 8 return periods and ncols=4, each block occupies 2 rows:
    #   Row 1 of a block: 2, 5, 10, 25 yr
    #   Row 2 of a block: 50, 75, 100, 500 yr
    return_periods_yr: tuple[int, ...] = (2, 5, 10, 25, 50, 75, 100, 500)

    # Durations in minutes.
    # 2880 min = 2 days; 10080 min = 7 days.
    durations_min: tuple[float, ...] = (
        5, 10, 15, 20, 25, 30,
        60, 120, 180, 240, 360, 480, 720, 1440,
        2880, 10080,
    )

    # Quantity to plot on the LEFT y-axis:
    #   "intensity" -> rainfall intensity [mm h^-1]
    #   "depth"     -> precipitation depth [mm]
    quantity_mode: str = "intensity"

    # LEFT y-axis scale.
    # Options:
    #   "log"    -> log-scale left y-axis
    #   "linear" -> linear left y-axis
    #   "auto"   -> log for intensity, linear for depth
    # The right bias axis is always linear because bias can be negative.
    left_y_axis_scale: str = "linear"

    # Use log-scale duration axis.
    use_log_duration_axis: bool = True

    # Optional quality filters for existing IDFs.
    # Leave as None to use all valid IDFs.
    min_r2: float | None = None
    min_years: float | None = None

    # ---------------------------------------------------------------------
    # Existing-IDF outlier investigation/filtering
    # ---------------------------------------------------------------------
    # The outlier filter is based only on the EXISTING IDF curves, not on GRIDF.
    # This avoids removing stations simply because they disagree with GRIDF.
    investigate_existing_idf_outliers: bool = True
    apply_existing_idf_outlier_filter: bool = True
    outlier_detection_quantity_mode: str = "depth"
    outlier_z_threshold: float = 4.5
    outlier_fraction_threshold: float = 0.20
    outlier_min_flagged_points: int = 4
    outlier_extreme_z_threshold: float = 9.0
    write_outlier_diagnostics: bool = True
    save_outlier_diagnostic_plot: bool = True

    # ---------------------------------------------------------------------
    # Figure style
    # ---------------------------------------------------------------------
    font_family: str = "Avenir Next"
    fallback_font: str = "DejaVu Sans"
    dpi: int = 300
    ncols: int = 4

    # Shaded band on the left axis.
    # Options:
    #   "std" -> median +/- 1 standard deviation
    #   "iqr" -> 25th to 75th percentile range
    left_band_mode: str = "iqr"

    # Shaded band on the right bias axis.
    # Options:
    #   "std" -> median bias +/- 1 standard deviation
    #   "iqr" -> 25th to 75th percentile range of paired bias
    bias_band_mode: str = "std"

    # ---------------------------------------------------------------------
    # Optional manual LEFT y-axis limits by data type
    # ---------------------------------------------------------------------
    # These are applied depending on quantity_mode. Leave as None to keep the
    # automatic row-wise limits.
    #
    # For intensity with a log y-axis, ymin MUST be > 0. Examples:
    #   manual_intensity_ylim = (1.0, 1000.0)
    #   manual_intensity_ylim_by_row = ((1.0, 250.0), (1.0, 450.0),
    #                                   (1.0, 350.0), (1.0, 600.0))
    #
    # If you give only two row-wise limits while plotting two blocks, they are
    # repeated for each block:
    #   row 1 standard        -> first tuple
    #   row 2 standard        -> second tuple
    #   row 1 disaggregation  -> first tuple
    #   row 2 disaggregation  -> second tuple
    #
    # If both global and row-wise limits are provided, row-wise limits have
    # priority.
    manual_intensity_ylim: tuple[float, float] | None = None
    manual_intensity_ylim_by_row: tuple[tuple[float, float], ...] | None = None

    # For depth with a linear y-axis. This default repeats for each block.
    manual_depth_ylim: tuple[float, float] | None = None
    manual_depth_ylim_by_row: tuple[tuple[float, float], ...] | None = ((0.0, 300.0), (0.0, 900.0))

    # Optional manual RIGHT y-axis limits for bias [%].
    manual_bias_ylim: tuple[float, float] | None = None
    manual_bias_ylim_by_row: tuple[tuple[float, float], ...] | None = None

    # Use one shared left y-axis limit per row.
    row_wise_left_y_limits: bool = True

    # Use one shared right y-axis limit per row for the bias axis.
    row_wise_bias_y_limits: bool = True

    # The lower edge of median +/- std can become negative. Since rainfall
    # cannot be negative, this clips the lower band at zero on the left axis.
    clip_std_lower_at_zero: bool = True

    # Secondary y-axis: relative bias = GRIDF - Existing IDFs.
    show_bias_axis: bool = True
    label_right_axis_only_on_last_column: bool = True

    # Colors.
    existing_color: str = "#b30000"       # red
    gridf_color: str = "#188b8b"          # teal
    bias_color: str = "#111111"           # black/dark gray
    zero_line_color: str = "#000000"      # black zero-bias line
    zero_line_linestyle: str = "--"
    grid_color: str = "#d0d0d0"

    # Line/marker settings.
    main_linewidth: float = 2.2
    bias_linewidth: float = 1.75
    marker_size: float = 3.8
    bias_marker_size: float = 3.3
    band_alpha: float = 0.18
    bias_band_alpha: float = 0.11

    # Save outputs.
    save_png: bool = True
    save_pdf: bool = True
    save_svg: bool = True

    def __post_init__(self):
        if self.gridf_parameter_rasters is None:
            self.gridf_parameter_rasters = {
                "K": None,
                "a": None,
                "b": None,
                "c": None,
            }


CFG = Config()


# =============================================================================
# STYLE
# =============================================================================

def setup_style(cfg: Config) -> None:
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if cfg.font_family in available_fonts:
        mpl.rcParams["font.family"] = cfg.font_family
    else:
        print(f"[WARNING] Font '{cfg.font_family}' not found. Using '{cfg.fallback_font}'.")
        mpl.rcParams["font.family"] = cfg.fallback_font

    mpl.rcParams.update({
        "font.size": 8.8,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.2,
        "figure.titlesize": 12.0,
        "axes.linewidth": 1.1,
        "xtick.major.width": 0.95,
        "ytick.major.width": 0.95,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# =============================================================================
# PATH HELPERS
# =============================================================================

def resolve_input_path(path: Path) -> Path:
    """Resolve configured input paths on Maria's Mac or in /mnt/data."""
    if path.exists():
        return path

    local_candidate = Path("/mnt/data") / path.name
    if local_candidate.exists():
        return local_candidate

    raise FileNotFoundError(
        f"Input file not found:\n  {path}\nAlso checked:\n  {local_candidate}"
    )


def resolve_output_dir(path: Path) -> Path:
    """Resolve output folder on Mac, or use /mnt/data when running here."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        local_out = Path("/mnt/data") / path.name
        local_out.mkdir(parents=True, exist_ok=True)
        return local_out


# =============================================================================
# EXCEL READING
# =============================================================================

LAT_CANDIDATES = [
    "Latitude (º)", "Latitude (°)", "Latitude", "latitude", "lat", "LAT", "LATITUDE", "y", "Y"
]
LON_CANDIDATES = [
    "Longitude (º)", "Longitude (°)", "Longitude", "longitude", "lon", "LON", "LONGITUDE", "x", "X"
]


def find_first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        c2 = cand.strip().lower()
        if c2 in lower_map:
            return lower_map[c2]
    for c in columns:
        c_lower = str(c).strip().lower()
        for cand in candidates:
            if cand.strip().lower() in c_lower:
                return c
    return None


def clean_numeric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_existing_idf_sheet(workbook: Path, sheet_name: str, reference_type: str) -> pd.DataFrame:
    """Load one sheet from IDF_Curves_Filtered.xlsx."""
    df = pd.read_excel(workbook, sheet_name=sheet_name)
    df = df.copy()

    lat_col = find_first_column(df.columns, LAT_CANDIDATES)
    lon_col = find_first_column(df.columns, LON_CANDIDATES)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find latitude/longitude columns in sheet '{sheet_name}'.")

    required = ["K", "a", "b", "c", lat_col, lon_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    df = clean_numeric_columns(df, ["K", "a", "b", "c", lat_col, lon_col, "R2", "Years"])
    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    df["reference_type"] = reference_type
    df["source_sheet"] = sheet_name
    df["station_index"] = np.arange(len(df), dtype=int)

    # Basic coefficient and coordinate validity.
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "K", "a", "b", "c"]).copy()
    df = df[
        df["longitude"].between(-76, -30)
        & df["latitude"].between(-36, 8)
        & (df["K"] > 0)
        & (df["b"] > -np.nanmin(np.asarray(CFG.durations_min, dtype=float)) + 1e-6)
        & (df["c"] > 0)
    ].copy()

    if CFG.min_r2 is not None and "R2" in df.columns:
        df = df[(df["R2"].isna()) | (df["R2"] >= CFG.min_r2)].copy()

    if CFG.min_years is not None and "Years" in df.columns:
        df = df[(df["Years"].isna()) | (df["Years"] >= CFG.min_years)].copy()

    print(f"[{reference_type}] loaded {len(df):,} valid rows from sheet '{sheet_name}' ({before:,} raw rows).")
    return df.reset_index(drop=True)


def load_existing_idfs_from_workbook(cfg: Config) -> dict[str, pd.DataFrame]:
    workbook = resolve_input_path(cfg.idf_workbook_xlsx)

    out: dict[str, pd.DataFrame] = {}
    if "standard" in cfg.reference_blocks:
        out["standard"] = load_existing_idf_sheet(workbook, cfg.standard_sheet_name, "Standard IDFs")
    if "disaggregation" in cfg.reference_blocks:
        out["disaggregation"] = load_existing_idf_sheet(workbook, cfg.disaggregation_sheet_name, "Disaggregation IDFs")

    return out


# =============================================================================
# GRIDF RASTER DISCOVERY AND SAMPLING
# =============================================================================

def param_regex(param: str) -> re.Pattern:
    # Match parameter as a standalone token: K, a, b, c.
    # This avoids matching random words containing the letters.
    return re.compile(rf"(^|[^a-z0-9]){re.escape(param.lower())}([^a-z0-9]|$)")


def score_raster_candidate(path: Path, param: str, cfg: Config) -> int:
    name = path.name.lower()
    stem = path.stem.lower()
    full = str(path).lower()

    score = 0

    # Strong preference for files that mention the parameter clearly.
    if param_regex(param).search(stem):
        score += 100
    if f"{param.lower()}_r" in stem or f"r_{param.lower()}" in stem:
        score += 40
    if f"param_{param.lower()}" in stem or f"parameter_{param.lower()}" in stem:
        score += 35
    if f"idf_{param.lower()}" in stem or f"{param.lower()}_idf" in stem:
        score += 30

    # Preferred dataset/method keywords.
    for kw in cfg.gridf_dataset_keywords:
        if kw.lower() in full:
            score += 15
    for kw in cfg.gridf_preferred_keywords:
        if kw.lower() in full:
            score += 8

    # Penalize likely diagnostic rasters.
    bad_terms = (
        "bias", "rmse", "nse", "r2", "ks", "pvalue", "p_value", "std",
        "mean", "median", "return", "period", "duration", "rain", "precip",
        "depth", "intensity", "mask", "count", "valid", "summary"
    )
    for bad in bad_terms:
        if bad in name:
            score -= 25

    # Prefer GeoTIFFs.
    if path.suffix.lower() in (".tif", ".tiff"):
        score += 5

    return score


def auto_find_gridf_raster(param: str, cfg: Config) -> Path:
    candidates: list[tuple[int, Path]] = []

    for search_dir in cfg.gridf_raster_search_dirs:
        if search_dir is None:
            continue
        search_dir = Path(search_dir)
        if not search_dir.exists():
            continue
        for p in search_dir.rglob("*.tif"):
            s = score_raster_candidate(p, param, cfg)
            if s > 0:
                candidates.append((s, p))
        for p in search_dir.rglob("*.tiff"):
            s = score_raster_candidate(p, param, cfg)
            if s > 0:
                candidates.append((s, p))

    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-find GRIDF/BR-DWGD raster for parameter '{param}'.\n"
            "Please set Config.gridf_parameter_rasters explicitly, for example:\n"
            "    gridf_parameter_rasters = {\n"
            "        'K': Path('/path/to/K.tif'),\n"
            "        'a': Path('/path/to/a.tif'),\n"
            "        'b': Path('/path/to/b.tif'),\n"
            "        'c': Path('/path/to/c.tif'),\n"
            "    }"
        )

    candidates = sorted(candidates, key=lambda x: (-x[0], str(x[1])))
    best_score, best_path = candidates[0]

    print(f"[AUTO-FIND] Parameter {param}: selected {best_path} (score={best_score})")
    if len(candidates) > 1:
        print("            Top candidates:")
        for score, p in candidates[:5]:
            print(f"              score={score:4d} | {p}")

    return best_path


def resolve_gridf_raster_paths(cfg: Config) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    for param in ["K", "a", "b", "c"]:
        configured = None
        if cfg.gridf_parameter_rasters is not None:
            configured = cfg.gridf_parameter_rasters.get(param)

        if configured is not None:
            configured = Path(configured)
            if configured.exists():
                paths[param] = configured
                print(f"[RASTER] Parameter {param}: {configured}")
                continue
            local_candidate = Path("/mnt/data") / configured.name
            if local_candidate.exists():
                paths[param] = local_candidate
                print(f"[RASTER] Parameter {param}: {local_candidate}")
                continue
            raise FileNotFoundError(f"Configured raster for parameter '{param}' not found: {configured}")

        if not cfg.auto_find_gridf_rasters:
            raise FileNotFoundError(
                f"No raster path configured for parameter '{param}', and auto_find_gridf_rasters=False."
            )

        paths[param] = auto_find_gridf_raster(param, cfg)

    return paths


def sample_one_raster_at_lonlat(raster_path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample raster values at lon/lat station coordinates."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        if str(src.crs).upper() in ("EPSG:4326", "OGC:CRS84") or src.crs.to_epsg() == 4326:
            xs, ys = lon, lat
        else:
            xs, ys = rio_transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)

        vals = np.full(len(lon), np.nan, dtype=float)
        for i, value in enumerate(src.sample(zip(xs, ys))):
            v = float(value[0])
            if src.nodata is not None and np.isfinite(src.nodata) and np.isclose(v, src.nodata):
                v = np.nan
            if (not np.isfinite(v)) or v < -1e20 or v > 1e20:
                v = np.nan
            vals[i] = v

    return vals


def attach_gridf_coefficients(df: pd.DataFrame, raster_paths: dict[str, Path], reference_key: str) -> pd.DataFrame:
    """Sample GRIDF rasters at station coordinates and append K_r/a_r/b_r/c_r."""
    df = df.copy()

    print(f"[{reference_key}] sampling GRIDF/BR-DWGD raster coefficients at {len(df):,} station locations...")
    for param, out_col in [("K", "K_r"), ("a", "a_r"), ("b", "b_r"), ("c", "c_r")]:
        vals = sample_one_raster_at_lonlat(
            raster_paths[param],
            df["longitude"].values,
            df["latitude"].values,
        )
        df[out_col] = vals
        print(
            f"    {out_col}: finite={np.isfinite(vals).sum():,}/{len(vals):,}, "
            f"min={np.nanmin(vals):.4g}, median={np.nanmedian(vals):.4g}, max={np.nanmax(vals):.4g}"
        )

    before = len(df)
    coeff_cols = ["K", "a", "b", "c", "K_r", "a_r", "b_r", "c_r"]
    df = df.dropna(subset=coeff_cols).copy()
    df = df[(df["K_r"] > 0) & (df["b_r"] > -np.nanmin(np.asarray(CFG.durations_min, dtype=float)) + 1e-6) & (df["c_r"] > 0)].copy()
    print(f"[{reference_key}] retained {len(df):,}/{before:,} rows after GRIDF raster sampling and coefficient checks.")

    if df.empty:
        raise ValueError(
            f"No valid rows remained for {reference_key} after sampling GRIDF rasters.\n"
            "Check that the raster CRS, raster extent, and station coordinates overlap."
        )

    return df.reset_index(drop=True)


# =============================================================================
# IDF COMPUTATION
# =============================================================================

def compute_idf_intensity(
    K: np.ndarray | float,
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float,
    return_period_yr: np.ndarray | float,
    duration_min: np.ndarray | float,
) -> np.ndarray:
    """Compute IDF rainfall intensity [mm h^-1]."""
    K = np.asarray(K, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    T = np.asarray(return_period_yr, dtype=float)
    d = np.asarray(duration_min, dtype=float)
    return K * np.power(T, a) / np.power(d + b, c)


def compute_quantity(
    K: np.ndarray | float,
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float,
    return_period_yr: float,
    duration_min: float,
    quantity_mode: str,
) -> np.ndarray:
    intensity = compute_idf_intensity(K, a, b, c, return_period_yr, duration_min)
    mode = quantity_mode.lower().strip()
    if mode == "intensity":
        return intensity
    if mode == "depth":
        return intensity * float(duration_min) / 60.0
    raise ValueError("quantity_mode must be 'intensity' or 'depth'.")


def robust_z_by_column(matrix: np.ndarray) -> np.ndarray:
    """Robust z-score by column using median and MAD."""
    med = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - med), axis=0)
    scale = 1.4826 * mad

    # Fallback to std if MAD is zero.
    std = np.nanstd(matrix, axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale <= 0), std, scale)
    scale = np.where((~np.isfinite(scale)) | (scale <= 0), np.nan, scale)

    return (matrix - med) / scale


def investigate_and_filter_existing_outliers(
    df: pd.DataFrame,
    cfg: Config,
    reference_key: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Flag/remove existing-IDF outliers using only existing IDF curves."""
    if not cfg.investigate_existing_idf_outliers:
        return df.reset_index(drop=True)

    curve_cols: list[str] = []
    curves = []

    for T in cfg.return_periods_yr:
        for d in cfg.durations_min:
            vals = compute_quantity(
                df["K"].values,
                df["a"].values,
                df["b"].values,
                df["c"].values,
                float(T),
                float(d),
                cfg.outlier_detection_quantity_mode,
            )
            curves.append(vals)
            curve_cols.append(f"T{T}_d{int(d)}")

    mat = np.vstack(curves).T
    mat = np.where((mat > 0) & np.isfinite(mat), mat, np.nan)
    log_mat = np.log(mat)
    z = robust_z_by_column(log_mat)

    flagged_points = np.abs(z) > cfg.outlier_z_threshold
    extreme_points = np.abs(z) > cfg.outlier_extreme_z_threshold

    n_points = np.sum(np.isfinite(z), axis=1)
    n_flagged = np.nansum(flagged_points, axis=1).astype(int)
    n_extreme = np.nansum(extreme_points, axis=1).astype(int)
    frac_flagged = np.divide(n_flagged, n_points, out=np.zeros_like(n_flagged, dtype=float), where=n_points > 0)
    max_abs_z = np.nanmax(np.abs(z), axis=1)

    is_outlier = (
        ((frac_flagged >= cfg.outlier_fraction_threshold) & (n_flagged >= cfg.outlier_min_flagged_points))
        | (n_extreme > 0)
    )

    diag = df.copy()
    diag["n_curve_points_checked"] = n_points
    diag["n_flagged_curve_points"] = n_flagged
    diag["fraction_flagged_curve_points"] = frac_flagged
    diag["max_abs_robust_z"] = max_abs_z
    diag["existing_idf_outlier"] = is_outlier

    if cfg.write_outlier_diagnostics:
        diag_csv = out_dir / f"BR_DWGD_existing_IDF_outlier_diagnostics_{reference_key}.csv"
        removed_csv = out_dir / f"BR_DWGD_existing_IDF_outliers_removed_{reference_key}.csv"
        diag.to_csv(diag_csv, index=False)
        diag.loc[is_outlier].to_csv(removed_csv, index=False)
        print(f"[{reference_key}] wrote outlier diagnostics: {diag_csv}")
        print(f"[{reference_key}] wrote removed-outlier table: {removed_csv}")

    print(
        f"[{reference_key}] existing-IDF outlier screen: "
        f"flagged {int(is_outlier.sum()):,}/{len(df):,} rows."
    )

    if cfg.save_outlier_diagnostic_plot:
        fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=cfg.dpi)
        ax.scatter(
            diag.loc[~is_outlier, "fraction_flagged_curve_points"],
            diag.loc[~is_outlier, "max_abs_robust_z"],
            s=16,
            facecolors="0.5",
            edgecolors="none",
            alpha=0.65,
            label="retained",
        )
        ax.scatter(
            diag.loc[is_outlier, "fraction_flagged_curve_points"],
            diag.loc[is_outlier, "max_abs_robust_z"],
            s=28,
            facecolors="#b30000",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
            label="flagged",
        )
        ax.axvline(cfg.outlier_fraction_threshold, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(cfg.outlier_extreme_z_threshold, color="black", linestyle=":", linewidth=0.9)
        ax.set_xlabel("Fraction of IDF curve points flagged")
        ax.set_ylabel("Maximum absolute robust z-score")
        ax.set_title(reference_key.replace("_", " ").title(), fontsize=10)
        ax.grid(True, color="0.85", linewidth=0.5, linestyle=":")
        ax.legend(frameon=False)
        fig.tight_layout()
        out_png = out_dir / f"BR_DWGD_existing_IDF_outlier_diagnostic_{reference_key}.png"
        fig.savefig(out_png, dpi=cfg.dpi)
        plt.close(fig)
        print(f"[{reference_key}] wrote outlier diagnostic plot: {out_png}")

    if cfg.apply_existing_idf_outlier_filter:
        df = df.loc[~is_outlier].copy()
        print(f"[{reference_key}] retained {len(df):,} rows after removing existing-IDF outliers.")

    return df.reset_index(drop=True)


def summarize_values(values: np.ndarray, band_mode: str, clip_lower_zero: bool) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "n": 0,
        }

    med = float(np.nanmedian(values))
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    q25 = float(np.nanpercentile(values, 25))
    q75 = float(np.nanpercentile(values, 75))

    if band_mode.lower().strip() == "iqr":
        lower, upper = q25, q75
    elif band_mode.lower().strip() == "std":
        lower, upper = med - std, med + std
        if clip_lower_zero:
            lower = max(0.0, lower)
    else:
        raise ValueError("band_mode must be 'std' or 'iqr'.")

    return {
        "median": med,
        "mean": mean,
        "std": std,
        "q25": q25,
        "q75": q75,
        "lower": float(lower),
        "upper": float(upper),
        "n": int(values.size),
    }


def build_station_values_and_summaries(
    df: pd.DataFrame,
    cfg: Config,
    reference_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute station-level existing/GRIDF curves and paired bias summaries."""
    station_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    bias_rows: list[dict] = []

    for T in cfg.return_periods_yr:
        for d in cfg.durations_min:
            existing = compute_quantity(
                df["K"].values,
                df["a"].values,
                df["b"].values,
                df["c"].values,
                float(T),
                float(d),
                cfg.quantity_mode,
            )
            gridf = compute_quantity(
                df["K_r"].values,
                df["a_r"].values,
                df["b_r"].values,
                df["c_r"].values,
                float(T),
                float(d),
                cfg.quantity_mode,
            )

            valid = (
                np.isfinite(existing)
                & np.isfinite(gridf)
                & (existing > 0)
                & (gridf > 0)
            )
            if not np.any(valid):
                continue

            existing_v = existing[valid]
            gridf_v = gridf[valid]
            station_ids = df.loc[valid, "station_index"].values

            for source_name, arr in [
                ("Existing IDFs", existing_v),
                ("GRIDF / BR-DWGD raster IDFs", gridf_v),
            ]:
                s = summarize_values(
                    arr,
                    cfg.left_band_mode,
                    cfg.clip_std_lower_at_zero,
                )
                s.update({
                    "reference_key": reference_key,
                    "source": source_name,
                    "return_period_yr": T,
                    "duration_min": d,
                })
                summary_rows.append(s)

            bias = 100.0 * (gridf_v - existing_v) / existing_v
            b = summarize_values(
                bias,
                cfg.bias_band_mode,
                clip_lower_zero=False,
            )
            b.update({
                "reference_key": reference_key,
                "return_period_yr": T,
                "duration_min": d,
                "median_bias_pct": b["median"],
                "mean_bias_pct": b["mean"],
                "std_bias_pct": b["std"],
                "q25_bias_pct": b["q25"],
                "q75_bias_pct": b["q75"],
                "lower_bias_pct": b["lower"],
                "upper_bias_pct": b["upper"],
                "n_idfs": b["n"],
            })
            bias_rows.append(b)

            station_rows.append(pd.DataFrame({
                "reference_key": reference_key,
                "reference_type": df.loc[valid, "reference_type"].values,
                "source_sheet": df.loc[valid, "source_sheet"].values,
                "station_index": station_ids,
                "return_period_yr": T,
                "duration_min": d,
                "existing_value": existing_v,
                "gridf_value": gridf_v,
                "bias_pct": bias,
            }))

    station_values = pd.concat(station_rows, ignore_index=True) if station_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    bias_summary = pd.DataFrame(bias_rows)

    return station_values, summary, bias_summary


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def duration_label(x: float) -> str:
    x = float(x)
    if np.isclose(x, 10080):
        return "7 d"
    if np.isclose(x, 7200):
        return "5 d"
    if np.isclose(x, 2880):
        return "2 d"
    if np.isclose(x, 1440):
        return "24 h"
    if x >= 60 and np.isclose(x % 60, 0):
        return f"{int(round(x / 60))} h"
    return f"{int(round(x))}"


def selected_xticks(durations: Sequence[float]) -> list[float]:
    preferred = [5, 15, 30, 60, 120, 360, 720, 1440, 2880, 10080]
    available = np.asarray(durations, dtype=float)
    ticks = []
    for p in preferred:
        if np.any(np.isclose(available, p)):
            ticks.append(float(p))
    return ticks


def quantity_ylabel(quantity_mode: str) -> str:
    mode = quantity_mode.lower().strip()
    if mode == "depth":
        return "Precipitation depth (mm)"
    if mode == "intensity":
        return "Rainfall intensity (mm h$^{-1}$)"
    return quantity_mode


def quantity_tag(quantity_mode: str) -> str:
    mode = quantity_mode.lower().strip()
    if mode == "depth":
        return "precipitation_depth_mm"
    if mode == "intensity":
        return "rainfall_intensity_mm_per_h"
    return mode


def left_axis_uses_log(cfg: Config) -> bool:
    choice = cfg.left_y_axis_scale.lower().strip()
    if choice == "log":
        return True
    if choice == "linear":
        return False
    if choice == "auto":
        return cfg.quantity_mode.lower().strip() == "intensity"
    raise ValueError("left_y_axis_scale must be 'log', 'linear', or 'auto'.")


def nice_upper_limit(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = 10 ** exp
    mant = x / base
    if mant <= 1.5:
        nice = 1.5 * base
    elif mant <= 2:
        nice = 2 * base
    elif mant <= 3:
        nice = 3 * base
    elif mant <= 5:
        nice = 5 * base
    else:
        nice = 10 * base
    return float(nice)


def nice_lower_positive_limit(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 0.1
    exp = np.floor(np.log10(x))
    base = 10 ** exp
    mant = x / base
    if mant <= 1:
        nice = 1 * base
    elif mant <= 2:
        nice = 2 * base
    elif mant <= 5:
        nice = 5 * base
    else:
        nice = 10 * base
    return float(max(nice, 1e-6))


def get_rowwise_manual_limit(
    row_index: int,
    block_rows: int,
    limits_by_row: tuple[tuple[float, float], ...] | None,
) -> tuple[float, float] | None:
    if limits_by_row is None:
        return None
    n = len(limits_by_row)
    if n == 0:
        return None
    # If exactly block_rows limits are provided, repeat them for each block.
    if n == block_rows:
        return limits_by_row[row_index % block_rows]
    # If explicit limits are provided for all rows, use row index directly.
    if row_index < n:
        return limits_by_row[row_index]
    # Fallback: cycle.
    return limits_by_row[row_index % n]


def get_manual_left_ylim(row_index: int, block_rows: int, cfg: Config) -> tuple[float, float] | None:
    mode = cfg.quantity_mode.lower().strip()

    if mode == "intensity":
        row_lim = get_rowwise_manual_limit(row_index, block_rows, cfg.manual_intensity_ylim_by_row)
        if row_lim is not None:
            return row_lim
        return cfg.manual_intensity_ylim

    if mode == "depth":
        row_lim = get_rowwise_manual_limit(row_index, block_rows, cfg.manual_depth_ylim_by_row)
        if row_lim is not None:
            return row_lim
        return cfg.manual_depth_ylim

    return None


def get_manual_bias_ylim(row_index: int, block_rows: int, cfg: Config) -> tuple[float, float] | None:
    row_lim = get_rowwise_manual_limit(row_index, block_rows, cfg.manual_bias_ylim_by_row)
    if row_lim is not None:
        return row_lim
    return cfg.manual_bias_ylim


def compute_row_limits(
    summary_all: pd.DataFrame,
    bias_all: pd.DataFrame,
    cfg: Config,
    nrows: int,
    block_rows: int,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    left_limits: dict[int, tuple[float, float]] = {}
    bias_limits: dict[int, tuple[float, float]] = {}

    use_log_y = left_axis_uses_log(cfg)
    durations = np.asarray(cfg.durations_min, dtype=float)

    for row in range(nrows):
        manual = get_manual_left_ylim(row, block_rows, cfg)
        if manual is not None:
            ymin, ymax = float(manual[0]), float(manual[1])
            if use_log_y and ymin <= 0:
                raise ValueError(
                    "Manual intensity/depth y-limit has ymin <= 0, but left y-axis is log-scale. "
                    f"Bad limit for row {row}: {manual}"
                )
            left_limits[row] = (ymin, ymax)
        else:
            row_records = []
            for i, T in enumerate(cfg.return_periods_yr):
                if i // cfg.ncols == (row % block_rows):
                    block_index = row // block_rows
                    if block_index < len(cfg.reference_blocks):
                        ref_key = cfg.reference_blocks[block_index]
                        sel = summary_all[
                            (summary_all["reference_key"] == ref_key)
                            & (summary_all["return_period_yr"] == T)
                        ]
                        row_records.append(sel)

            if row_records:
                row_df = pd.concat(row_records, ignore_index=True)
                lows = row_df["lower"].to_numpy(dtype=float)
                ups = row_df["upper"].to_numpy(dtype=float)
                lows = lows[np.isfinite(lows)]
                ups = ups[np.isfinite(ups)]

                if use_log_y:
                    positive_lows = lows[lows > 0]
                    if positive_lows.size and ups.size:
                        ymin = nice_lower_positive_limit(np.nanmin(positive_lows) * 0.75)
                        ymax = nice_upper_limit(np.nanmax(ups) * 1.10)
                        left_limits[row] = (ymin, ymax)
                else:
                    if ups.size:
                        ymax = nice_upper_limit(np.nanmax(ups) * 1.08)
                        left_limits[row] = (0.0, ymax)

        manual_bias = get_manual_bias_ylim(row, block_rows, cfg)
        if manual_bias is not None:
            bias_limits[row] = (float(manual_bias[0]), float(manual_bias[1]))
        else:
            row_bias = []
            for i, T in enumerate(cfg.return_periods_yr):
                if i // cfg.ncols == (row % block_rows):
                    block_index = row // block_rows
                    if block_index < len(cfg.reference_blocks):
                        ref_key = cfg.reference_blocks[block_index]
                        sel = bias_all[
                            (bias_all["reference_key"] == ref_key)
                            & (bias_all["return_period_yr"] == T)
                        ]
                        row_bias.append(sel)

            if row_bias:
                rb = pd.concat(row_bias, ignore_index=True)
                vals = np.concatenate([
                    rb["lower_bias_pct"].to_numpy(dtype=float),
                    rb["upper_bias_pct"].to_numpy(dtype=float),
                    rb["median_bias_pct"].to_numpy(dtype=float),
                    np.asarray([0.0]),
                ])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ymin = np.nanmin(vals)
                    ymax = np.nanmax(vals)
                    pad = 0.08 * max(1.0, ymax - ymin)
                    ymin = np.floor((ymin - pad) / 25.0) * 25.0
                    ymax = np.ceil((ymax + pad) / 25.0) * 25.0
                    bias_limits[row] = (float(ymin), float(ymax))

    return left_limits, bias_limits


def apply_axis_formatting(ax, ax_bias, row: int, col: int, block_rows: int, cfg: Config):
    durations = np.asarray(cfg.durations_min, dtype=float)
    xticks = selected_xticks(durations)

    if cfg.use_log_duration_axis:
        ax.set_xscale("log")
        if ax_bias is not None:
            ax_bias.set_xscale("log")
    else:
        ax.set_xscale("linear")
        if ax_bias is not None:
            ax_bias.set_xscale("linear")

    ax.set_xticks(xticks)
    ax.set_xticklabels([duration_label(x) for x in xticks], rotation=45, ha="right")

    if left_axis_uses_log(cfg):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yscale("linear")

    ax.grid(True, which="major", axis="both", color=cfg.grid_color, linestyle=":", linewidth=0.62, alpha=0.85)
    ax.grid(True, which="minor", axis="x", color=cfg.grid_color, linestyle=":", linewidth=0.38, alpha=0.55)

    # x-axis label only on the last row of each block.
    if (row % block_rows) == block_rows - 1:
        ax.set_xlabel("Duration", fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if col == 0:
        ax.set_ylabel(quantity_ylabel(cfg.quantity_mode), fontweight="bold")
    else:
        ax.set_ylabel("")

    if ax_bias is not None:
        ax_bias.set_yscale("linear")
        if cfg.label_right_axis_only_on_last_column and col != cfg.ncols - 1:
            ax_bias.set_yticklabels([])
            ax_bias.set_ylabel("")
        elif col == cfg.ncols - 1:
            ax_bias.set_ylabel("GRIDF $-$ Existing IDFs bias (%)", fontweight="bold")


# =============================================================================
# PLOT
# =============================================================================

def plot_figure(
    station_values_all: pd.DataFrame,
    summary_all: pd.DataFrame,
    bias_all: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
):
    source_order = ["Existing IDFs", "GRIDF / BR-DWGD raster IDFs"]
    colors = {
        "Existing IDFs": cfg.existing_color,
        "GRIDF / BR-DWGD raster IDFs": cfg.gridf_color,
    }
    labels = {
        "Existing IDFs": "Existing IDFs",
        "GRIDF / BR-DWGD raster IDFs": "GRIDF / BR-DWGD raster IDFs",
    }

    n_blocks = len(cfg.reference_blocks)
    block_rows = int(np.ceil(len(cfg.return_periods_yr) / cfg.ncols))
    nrows = n_blocks * block_rows
    ncols = cfg.ncols

    fig_height = 2.45 * nrows + 0.75
    fig_width = 12.6

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        dpi=cfg.dpi,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.070,
        right=0.940,
        top=0.985,
        bottom=0.090,
        wspace=0.21,
        hspace=0.48,
    )

    left_limits, bias_limits = compute_row_limits(summary_all, bias_all, cfg, nrows, block_rows)

    legend_handles = []
    legend_labels = []

    for block_i, ref_key in enumerate(cfg.reference_blocks):
        block_label = "Standard IDFs" if ref_key == "standard" else "Disaggregation IDFs"

        for i, T in enumerate(cfg.return_periods_yr):
            local_row = i // ncols
            col = i % ncols
            row = block_i * block_rows + local_row
            ax = axes[row, col]
            ax_bias = ax.twinx() if cfg.show_bias_axis else None

            # Left axis: existing and GRIDF curves.
            for source in source_order:
                d = summary_all[
                    (summary_all["reference_key"] == ref_key)
                    & (summary_all["source"] == source)
                    & (summary_all["return_period_yr"] == T)
                ].sort_values("duration_min")

                if d.empty:
                    continue

                x = d["duration_min"].to_numpy(dtype=float)
                y = d["median"].to_numpy(dtype=float)
                y1 = d["lower"].to_numpy(dtype=float)
                y2 = d["upper"].to_numpy(dtype=float)

                if left_axis_uses_log(cfg):
                    positive = np.concatenate([y[np.isfinite(y) & (y > 0)], y1[np.isfinite(y1) & (y1 > 0)]])
                    floor = np.nanmin(positive) * 0.5 if positive.size else 1e-3
                    y1 = np.where(y1 <= 0, floor, y1)

                line, = ax.plot(
                    x,
                    y,
                    color=colors[source],
                    linewidth=cfg.main_linewidth,
                    marker="o",
                    markersize=cfg.marker_size,
                    markerfacecolor=colors[source],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    label=labels[source],
                    zorder=4,
                )
                ax.fill_between(
                    x,
                    y1,
                    y2,
                    color=colors[source],
                    alpha=cfg.band_alpha,
                    linewidth=0,
                    zorder=2,
                )

                if labels[source] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(labels[source])

            # Right axis: paired bias.
            if ax_bias is not None:
                bd = bias_all[
                    (bias_all["reference_key"] == ref_key)
                    & (bias_all["return_period_yr"] == T)
                ].sort_values("duration_min")

                if not bd.empty:
                    x = bd["duration_min"].to_numpy(dtype=float)
                    y = bd["median_bias_pct"].to_numpy(dtype=float)
                    y1 = bd["lower_bias_pct"].to_numpy(dtype=float)
                    y2 = bd["upper_bias_pct"].to_numpy(dtype=float)

                    ax_bias.axhline(
                        0.0,
                        color=cfg.zero_line_color,
                        linestyle=cfg.zero_line_linestyle,
                        linewidth=1.05,
                        alpha=0.95,
                        zorder=0,
                    )
                    ax_bias.fill_between(
                        x,
                        y1,
                        y2,
                        color="0.45",
                        alpha=cfg.bias_band_alpha,
                        linewidth=0,
                        zorder=1,
                    )
                    bias_line, = ax_bias.plot(
                        x,
                        y,
                        color=cfg.bias_color,
                        linestyle="--",
                        linewidth=cfg.bias_linewidth,
                        marker="s",
                        markersize=cfg.bias_marker_size,
                        markerfacecolor="white",
                        markeredgecolor=cfg.bias_color,
                        markeredgewidth=0.8,
                        label="Median bias ± 1 std",
                        zorder=5,
                    )
                    if "Median bias ± 1 std" not in legend_labels:
                        legend_handles.append(bias_line)
                        legend_labels.append("Median bias ± 1 std")

            ax.set_title(f"T = {int(T)} yr", fontweight="bold", pad=3.0)

            # n label for this reference block/return period.
            n_panel = int(
                station_values_all[
                    (station_values_all["reference_key"] == ref_key)
                    & (station_values_all["return_period_yr"] == T)
                ]["station_index"].nunique()
            )
            ax.text(
                0.03,
                0.93,
                f"n={n_panel}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.75", lw=0.55, alpha=0.85),
                zorder=10,
            )

            apply_axis_formatting(ax, ax_bias, row, col, block_rows, cfg)

            if cfg.row_wise_left_y_limits and row in left_limits:
                ax.set_ylim(*left_limits[row])
            if ax_bias is not None and cfg.row_wise_bias_y_limits and row in bias_limits:
                ax_bias.set_ylim(*bias_limits[row])

        # Vertical block label along left side.
        top_ax = axes[block_i * block_rows, 0]
        bot_ax = axes[block_i * block_rows + block_rows - 1, 0]
        y_mid = 0.5 * (top_ax.get_position().y1 + bot_ax.get_position().y0)
        fig.text(
            0.022,
            y_mid,
            block_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10.0,
            fontweight="bold",
        )

    # Hide unused axes, if any.
    for j in range(len(cfg.return_periods_yr), block_rows * ncols):
        local_row = j // ncols
        col = j % ncols
        for block_i in range(n_blocks):
            row = block_i * block_rows + local_row
            axes[row, col].axis("off")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.020),
        handlelength=3.0,
        columnspacing=1.8,
    )

    out_tag = quantity_tag(cfg.quantity_mode)
    yscale_tag = "logy" if left_axis_uses_log(cfg) else "lineary"
    band_tag = f"{cfg.left_band_mode}band"
    filter_tag = "outlier_filtered" if cfg.apply_existing_idf_outlier_filter else "all_existing_idfs"
    base_name = (
        "BR_DWGD_existing_vs_gridf_IDF_curves_"
        f"from_IDF_Curves_Filtered_standard_disaggregation_{out_tag}_{yscale_tag}_{band_tag}_"
        f"with_right_axis_bias_{filter_tag}_one_figure"
    )

    if cfg.save_png:
        out = out_dir / f"{base_name}.png"
        fig.savefig(out, dpi=cfg.dpi)
        print(f"Saved: {out}")
    if cfg.save_pdf:
        out = out_dir / f"{base_name}.pdf"
        fig.savefig(out)
        print(f"Saved: {out}")
    if cfg.save_svg:
        out = out_dir / f"{base_name}.svg"
        fig.savefig(out)
        print(f"Saved: {out}")

    plt.close(fig)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_bias_diagnostics(bias_all: pd.DataFrame):
    check_T = [10, 25]
    check_d = [15, 60, 1440, 7200, 10080]

    d = bias_all[
        bias_all["return_period_yr"].isin(check_T)
        & bias_all["duration_min"].isin(check_d)
    ].copy()

    if d.empty:
        return

    print("\nBias diagnostic from current workbook + sampled GRIDF rasters")
    print("Formula: 100 * (GRIDF - Existing IDFs) / Existing IDFs")
    print("Positive means GRIDF/BR-DWGD > existing IDF.\n")

    cols = ["reference_key", "return_period_yr", "duration_min", "median_bias_pct", "mean_bias_pct", "std_bias_pct", "n_idfs"]
    print(d[cols].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = CFG
    setup_style(cfg)
    out_dir = resolve_output_dir(cfg.out_dir)

    print("\nLoading existing IDF equations from IDF_Curves_Filtered.xlsx...")
    existing_blocks = load_existing_idfs_from_workbook(cfg)

    print("\nResolving GRIDF/BR-DWGD raster coefficient files...")
    raster_paths = resolve_gridf_raster_paths(cfg)

    station_tables = []
    summary_tables = []
    bias_tables = []

    for ref_key in cfg.reference_blocks:
        if ref_key not in existing_blocks:
            continue

        df = existing_blocks[ref_key]
        df = attach_gridf_coefficients(df, raster_paths, ref_key)
        df = investigate_and_filter_existing_outliers(df, cfg, ref_key, out_dir)

        station_values, summary, bias_summary = build_station_values_and_summaries(df, cfg, ref_key)
        station_tables.append(station_values)
        summary_tables.append(summary)
        bias_tables.append(bias_summary)

    station_values_all = pd.concat(station_tables, ignore_index=True)
    summary_all = pd.concat(summary_tables, ignore_index=True)
    bias_all = pd.concat(bias_tables, ignore_index=True)

    print_bias_diagnostics(bias_all)

    prefix = (
        "BR_DWGD_existing_vs_gridf_from_IDF_Curves_Filtered_"
        f"{quantity_tag(cfg.quantity_mode)}"
    )
    station_csv = out_dir / f"{prefix}_station_values.csv"
    summary_csv = out_dir / f"{prefix}_summary.csv"
    bias_csv = out_dir / f"{prefix}_bias_summary.csv"

    station_values_all.to_csv(station_csv, index=False)
    summary_all.to_csv(summary_csv, index=False)
    bias_all.to_csv(bias_csv, index=False)

    print(f"\nSaved station-level values: {station_csv}")
    print(f"Saved summary curves:       {summary_csv}")
    print(f"Saved bias summary:         {bias_csv}")

    print("\nPlotting figure...")
    plot_figure(station_values_all, summary_all, bias_all, cfg, out_dir)

    print("\nDone.")
    print("This script used IDF_Curves_Filtered.xlsx for existing IDFs and sampled GRIDF/BR-DWGD raster coefficients directly.")


if __name__ == "__main__":
    main()
