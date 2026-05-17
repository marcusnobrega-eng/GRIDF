#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Median bias curves by biome, duration, and reference IDF type for BR-DWGD
GRIDF raster IDFs versus existing IDFs.

This script computes the bias over a continuous return-period range from
2 to 500 years and plots median bias curves by biome.

Figure layout
-------------
Rows    -> Existing IDF reference type:
           1) Standard
           2) Disaggregation

Columns -> Selected durations

Inside each panel:
           one median-bias curve per biome

Station counts
--------------
For each IDF type, the plot shows:
  - total number of valid stations/rows used for that IDF type
  - number of valid stations/rows used in each biome

Bias definition
---------------
bias [%] = 100 * (BR-DWGD raster IDF - Existing IDF) / Existing IDF

Main design choices
-------------------
- Standard and Disaggregation are separated in different rows.
- The number of valid stations used for each IDF type is shown in the row label.
- The number of valid stations used for each biome and IDF type is shown in
  a compact count box inside the first panel of each row.
- No figure-level title.
- No standard-deviation shading.
- Biome colors, fonts, and general style follow the final spatial-bias figure.
- Return periods are plotted on a logarithmic x-axis.

Author: ChatGPT
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator


# =============================================================================
# USER SETTINGS
# =============================================================================

@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Core inputs
    # -------------------------------------------------------------------------
    filtered_xlsx: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Existing_IDFs/IDF_Curves_Filtered.xlsx"
    )

    standard_sheet: str = "Standard"
    disaggregation_sheet: str = "Disaggregation"

    product_name: str = "BR-DWGD"
    product_raster_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER"
    )

    # Explicit BR-DWGD Gumbel raster paths.
    product_raster_paths: dict[str, Path] | None = field(default_factory=lambda: {
        "K": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_K_GUMBEL.tif"),
        "a": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_a_GUMBEL.tif"),
        "b": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_b_GUMBEL.tif"),
        "c": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_c_GUMBEL.tif"),
    })
    auto_find_product_rasters: bool = False

    biomes_shp: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes/Brazil_biomes.shp"
    )

    # Font setup.
    font_file: Path = Path(
        "/Users/mngomes/Library/Fonts/Avenir Next.ttc"
    )
    font_family: str = "Avenir Next"
    fallback_font: str = "DejaVu Sans"

    # Output.
    out_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Existing_IDFs/Figures_Spatial_Bias"
    )
    out_name: str = "Median_bias_return_period_curves_BR_DWGD_vs_existing_IDFs_standard_disaggregation_rows_with_biome_counts"

    # -------------------------------------------------------------------------
    # Curve settings
    # -------------------------------------------------------------------------
    # References are intentionally separated as rows in the final figure.
    references_to_plot: tuple[str, ...] = ("standard", "disaggregation")

    # Durations used for the curves, in minutes.
    # This default matches the durations in the final spatial-bias script.
    # Edit this tuple if your full IDF-fitting duration list is different.
    durations_min_for_curves: tuple[float, ...] = (15.0, 60.0, 720.0, 1440.0)

    # Return-period range.
    return_period_min_yr: float = 2.0
    return_period_max_yr: float = 500.0
    n_return_periods: int = 400

    # "intensity" -> mm/h ; "depth" -> mm.
    # For a fixed duration, the bias is the same for intensity and depth,
    # but this option is kept for consistency with the spatial-bias script.
    quantity_mode: str = "intensity"

    # Optional: restrict the displayed y-axis. Use None for automatic limits.
    manual_ymin_pct: float | None = None
    manual_ymax_pct: float | None = None

    # Automatic y-axis padding around all median curves.
    y_padding_pct: float = 8.0

    # -------------------------------------------------------------------------
    # Figure aesthetics, inspired by the final spatial-bias figure
    # -------------------------------------------------------------------------
    dpi: int = 600

    # Figure size is scaled by number of duration columns.
    figure_width_per_duration: float = 3.35
    figure_height: float = 6.20
    min_figure_width: float = 8.0

    line_width: float = 2.05
    marker_size: float = 2.9
    marker_every: int = 60

    grid_linewidth: float = 0.45
    zero_linewidth: float = 0.90
    axis_linewidth: float = 1.15

    legend_fontsize: float = 8.4
    panel_title_fontsize: float = 10.2
    label_fontsize: float = 9.6
    tick_fontsize: float = 8.4
    row_label_fontsize: float = 10.3
    panel_label_fontsize: float = 9.8
    count_box_fontsize: float = 7.4

    save_curve_summary_csv: bool = True
    save_station_counts_csv: bool = True


CFG = Config()


# =============================================================================
# STYLE HELPERS
# =============================================================================

def setup_style(cfg: Config) -> None:
    """Register Avenir Next from the provided file path and set Matplotlib style."""
    font_was_loaded = False

    if cfg.font_file is not None and cfg.font_file.exists():
        try:
            fm.fontManager.addfont(str(cfg.font_file))
            loaded_name = fm.FontProperties(fname=str(cfg.font_file)).get_name()
            mpl.rcParams["font.family"] = loaded_name
            font_was_loaded = True
            print(f"[INFO] Using custom font: {loaded_name}")
        except Exception as exc:
            print(f"[WARNING] Could not load font file {cfg.font_file}: {exc}")

    if not font_was_loaded:
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        if cfg.font_family in available_fonts:
            mpl.rcParams["font.family"] = cfg.font_family
            print(f"[INFO] Using installed font family: {cfg.font_family}")
        else:
            print(
                f"[WARNING] Font '{cfg.font_family}' not found and font file was not available. "
                f"Using '{cfg.fallback_font}'."
            )
            mpl.rcParams["font.family"] = cfg.fallback_font

    mpl.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.titlesize": 13.0,
        "axes.linewidth": cfg.axis_linewidth,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# =============================================================================
# COLUMN HELPERS
# =============================================================================

LAT_CANDIDATES = [
    "Latitude", "latitude", "lat", "LAT", "Latitude (º)", "Latitude (°)",
    "LATITUDE", "y", "Y",
]
LON_CANDIDATES = [
    "Longitude", "longitude", "lon", "LON", "Longitude (º)", "Longitude (°)",
    "LONGITUDE", "x", "X",
]
CODE_CANDIDATES = [
    "Code", "code", "station_id", "station_ID", "ID", "id", "station", "station_code",
]
NAME_CANDIDATES = [
    "Name", "name", "Station", "station", "station_name", "Nome", "NOME",
]
PARAM_COLS = ["K", "a", "b", "c"]


def find_first_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in columns}

    for cand in candidates:
        if cand in columns:
            return cand
        cand_lower = cand.strip().lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]

    for c in columns:
        c_lower = str(c).strip().lower()
        for cand in candidates:
            if cand.strip().lower() in c_lower:
                return c

    return None


# =============================================================================
# BIOME HELPERS
# =============================================================================

BIOME_COLORS = {
    "Amazônia": "#0072B2",
    "Caatinga": "#E69F00",
    "Cerrado": "#009E73",
    "Mata Atlântica": "#CC79A7",
    "Pampa": "#D55E00",
    "Pantanal": "#56B4E9",
}

BIOME_ABBR = {
    "Amazônia": "AMZ",
    "Caatinga": "CAT",
    "Cerrado": "CER",
    "Mata Atlântica": "MAT",
    "Pampa": "PAM",
    "Pantanal": "PAN",
}

BIOME_ORDER = ["Amazônia", "Caatinga", "Cerrado", "Mata Atlântica", "Pampa", "Pantanal"]


def canonical_biome_name(name) -> str:
    if pd.isna(name):
        return "Unknown"

    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    if "amaz" in s:
        return "Amazônia"
    if "caating" in s:
        return "Caatinga"
    if "cerrad" in s:
        return "Cerrado"
    if "mata" in s and ("atl" in s or "atlant" in s):
        return "Mata Atlântica"
    if "pampa" in s:
        return "Pampa"
    if "pantanal" in s:
        return "Pantanal"

    return str(name)


def load_biomes(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Biome shapefile not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)

    biome_col = find_first_column(
        gdf.columns,
        ["Bioma", "BIOMA", "bioma", "Biome", "BIOME", "biome", "Name", "name", "NOME", "nome"],
    )
    if biome_col is None:
        raise ValueError("Could not identify the biome-name column in the biome shapefile.")

    gdf["biome_name"] = gdf[biome_col].apply(canonical_biome_name)
    return gdf


def assign_biomes_to_stations(df: pd.DataFrame, biomes: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        gdf,
        biomes[["biome_name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined["biome_name"] = joined["biome_name"].fillna("Unknown")

    return pd.DataFrame(joined.drop(columns="geometry"))


# =============================================================================
# EXISTING IDF READER
# =============================================================================

def read_existing_idf_sheet(path: Path, sheet_name: str, reference_key: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find workbook: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    lat_col = find_first_column(df.columns, LAT_CANDIDATES)
    lon_col = find_first_column(df.columns, LON_CANDIDATES)
    code_col = find_first_column(df.columns, CODE_CANDIDATES)
    name_col = find_first_column(df.columns, NAME_CANDIDATES)

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify latitude/longitude columns in sheet {sheet_name}")

    out = df.copy()
    out["latitude"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(out[lon_col], errors="coerce")
    out["station_code"] = out[code_col].astype(str) if code_col is not None else ""
    out["station_name"] = out[name_col].astype(str) if name_col is not None else ""

    for col in PARAM_COLS:
        if col not in out.columns:
            raise ValueError(f"Sheet {sheet_name} is missing required parameter column '{col}'.")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["reference_key"] = reference_key
    out["source_sheet"] = sheet_name
    out["source_row"] = np.arange(len(out), dtype=int) + 2

    valid = (
        np.isfinite(out["latitude"])
        & np.isfinite(out["longitude"])
        & out["longitude"].between(-76, -30)
        & out["latitude"].between(-36, 8)
        & np.isfinite(out["K"])
        & np.isfinite(out["a"])
        & np.isfinite(out["b"])
        & np.isfinite(out["c"])
        & (out["K"] > 0)
    )

    before = len(out)
    out = out.loc[valid].copy().reset_index(drop=True)
    print(f"[{reference_key}] retained {len(out):,}/{before:,} rows after basic Excel checks.")

    return out


# =============================================================================
# RASTER HELPERS
# =============================================================================

def parameter_token_score(stem_lower: str, param: str) -> int:
    p = param.lower()
    tokens = re.split(r"[^a-z0-9]+", stem_lower)

    score = 0
    strong_patterns = [
        f"param_{p}", f"parameter_{p}", f"coef_{p}", f"coeff_{p}",
        f"coefficient_{p}", f"idf_{p}", f"{p}_gev", f"{p}_gumbel",
    ]

    for pat in strong_patterns:
        if pat in stem_lower:
            score += 80

    if p in tokens:
        score += 60

    if stem_lower == p or stem_lower.startswith(p + "_") or stem_lower.endswith("_" + p):
        score += 50

    return score


def resolve_product_rasters(cfg: Config) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    if cfg.product_raster_paths is not None:
        for p in PARAM_COLS:
            path = Path(cfg.product_raster_paths[p])
            if not path.exists():
                raise FileNotFoundError(f"Configured raster not found for {p}: {path}")
            paths[p] = path
        return paths

    if not cfg.auto_find_product_rasters:
        raise ValueError("No raster paths were provided and auto_find_product_rasters=False.")

    if not cfg.product_raster_dir.exists():
        raise FileNotFoundError(f"Raster directory not found: {cfg.product_raster_dir}")

    tif_files = sorted(list(cfg.product_raster_dir.glob("*.tif")) + list(cfg.product_raster_dir.glob("*.tiff")))
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No tif files found in {cfg.product_raster_dir}")

    for p in PARAM_COLS:
        ranked = []
        for tif in tif_files:
            score = parameter_token_score(tif.stem.lower(), p)
            ranked.append((score, tif))

        ranked.sort(key=lambda x: (x[0], x[1].name), reverse=True)
        best_score, best_path = ranked[0]

        if best_score <= 0:
            raise FileNotFoundError(
                f"Could not automatically identify raster for parameter '{p}' inside {cfg.product_raster_dir}."
            )

        paths[p] = best_path
        print(f"[AUTO] Parameter {p}: {best_path.name} (score={best_score})")

    return paths


def sample_one_raster_at_lonlat(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    with rasterio.open(path) as src:
        vals = np.full(lon.shape, np.nan, dtype="float64")
        sampled = list(src.sample(list(zip(lon, lat))))

        for i, item in enumerate(sampled):
            if len(item) > 0:
                vals[i] = float(item[0])

        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            vals[np.isclose(vals, nodata, rtol=0.0, atol=0.0)] = np.nan

        vals[(vals < -1e30) | (vals > 1e30)] = np.nan

    return vals


def attach_product_coefficients(df: pd.DataFrame, raster_paths: dict[str, Path]) -> pd.DataFrame:
    out = df.copy()

    out["K_r"] = sample_one_raster_at_lonlat(raster_paths["K"], out["longitude"], out["latitude"])
    out["a_r"] = sample_one_raster_at_lonlat(raster_paths["a"], out["longitude"], out["latitude"])
    out["b_r"] = sample_one_raster_at_lonlat(raster_paths["b"], out["longitude"], out["latitude"])
    out["c_r"] = sample_one_raster_at_lonlat(raster_paths["c"], out["longitude"], out["latitude"])

    return out


# =============================================================================
# IDF AND BIAS
# =============================================================================

def idf_intensity(K, a, b, c, T, dmin):
    return K * (T ** a) / ((dmin + b) ** c)


def convert_quantity(intensity_mm_h: np.ndarray, duration_min: float, quantity_mode: str) -> np.ndarray:
    mode = quantity_mode.lower().strip()

    if mode == "intensity":
        return intensity_mm_h
    if mode == "depth":
        return intensity_mm_h * duration_min / 60.0

    raise ValueError("quantity_mode must be 'intensity' or 'depth'.")


def duration_label(dmin: float) -> str:
    dmin = float(dmin)

    if dmin < 60:
        return f"{int(dmin)} min"

    if dmin < 1440:
        hours = dmin / 60.0
        if abs(hours - round(hours)) < 1e-9:
            return f"{int(round(hours))} h"
        return f"{hours:g} h"

    days = dmin / 1440.0
    if abs(days - round(days)) < 1e-9:
        return f"{int(round(days))} d"

    return f"{days:g} d"


def reference_label(reference_key: str) -> str:
    if reference_key == "standard":
        return "Standard"
    if reference_key == "disaggregation":
        return "Disaggregation"
    return reference_key


def read_and_prepare_station_table(cfg: Config) -> pd.DataFrame:
    print("Resolving BR-DWGD raster parameter files...")
    raster_paths = resolve_product_rasters(cfg)

    print("Reading existing IDF equations...")
    blocks = []

    if "standard" in cfg.references_to_plot:
        blocks.append(read_existing_idf_sheet(cfg.filtered_xlsx, cfg.standard_sheet, "standard"))

    if "disaggregation" in cfg.references_to_plot:
        blocks.append(read_existing_idf_sheet(cfg.filtered_xlsx, cfg.disaggregation_sheet, "disaggregation"))

    if not blocks:
        raise ValueError("references_to_plot must include 'standard', 'disaggregation', or both.")

    df = pd.concat(blocks, ignore_index=True)

    print("Sampling BR-DWGD raster coefficients at station locations...")
    df = attach_product_coefficients(df, raster_paths)

    valid_r = (
        np.isfinite(df["K_r"])
        & np.isfinite(df["a_r"])
        & np.isfinite(df["b_r"])
        & np.isfinite(df["c_r"])
        & (df["K_r"] > 0)
    )

    n_before = len(df)
    df = df.loc[valid_r].copy().reset_index(drop=True)
    print(f"Retained {len(df):,}/{n_before:,} rows after requiring sampled BR-DWGD raster coefficients.")

    print("Assigning biome to each station...")
    biomes = load_biomes(cfg.biomes_shp)
    df = assign_biomes_to_stations(df, biomes)

    df = df[df["biome_name"].isin(BIOME_ORDER)].copy().reset_index(drop=True)
    print(f"Retained {len(df):,} rows in the six main Brazilian biomes.")

    return df


def compute_station_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count valid stations/rows used for each IDF reference type and biome.

    The count is based on valid Excel rows after:
      - IDF-parameter checks
      - raster sampling
      - biome assignment

    If station_code is duplicated in your workbook and you want unique physical
    stations instead of rows, replace len(sub) with a station_code unique count.
    """
    rows = []

    for ref in sorted(df["reference_key"].unique()):
        ref_df = df[df["reference_key"] == ref].copy()

        rows.append({
            "reference_key": ref,
            "reference_label": reference_label(ref),
            "biome_name": "ALL",
            "biome_abbr": "ALL",
            "n_stations": int(len(ref_df)),
        })

        for biome in BIOME_ORDER:
            sub = ref_df[ref_df["biome_name"] == biome]
            rows.append({
                "reference_key": ref,
                "reference_label": reference_label(ref),
                "biome_name": biome,
                "biome_abbr": BIOME_ABBR.get(biome, biome),
                "n_stations": int(len(sub)),
            })

    return pd.DataFrame(rows)


def compute_median_bias_curves(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("Computing median bias curves by reference, biome, and duration...")

    T_values = np.geomspace(
        cfg.return_period_min_yr,
        cfg.return_period_max_yr,
        cfg.n_return_periods,
    )

    rows = []

    for ref in cfg.references_to_plot:
        ref_df = df[df["reference_key"] == ref].copy()
        if ref_df.empty:
            continue

        for duration_min in cfg.durations_min_for_curves:
            for biome in BIOME_ORDER:
                sub = ref_df[ref_df["biome_name"] == biome].copy()
                if sub.empty:
                    continue

                # Station arrays for vectorized calculation.
                K = sub["K"].to_numpy(dtype=float)
                a = sub["a"].to_numpy(dtype=float)
                b = sub["b"].to_numpy(dtype=float)
                c = sub["c"].to_numpy(dtype=float)

                K_r = sub["K_r"].to_numpy(dtype=float)
                a_r = sub["a_r"].to_numpy(dtype=float)
                b_r = sub["b_r"].to_numpy(dtype=float)
                c_r = sub["c_r"].to_numpy(dtype=float)

                for T in T_values:
                    existing_i = idf_intensity(K, a, b, c, T, duration_min)
                    gridf_i = idf_intensity(K_r, a_r, b_r, c_r, T, duration_min)

                    existing_v = convert_quantity(existing_i, duration_min, cfg.quantity_mode)
                    gridf_v = convert_quantity(gridf_i, duration_min, cfg.quantity_mode)

                    valid = (
                        np.isfinite(existing_v)
                        & np.isfinite(gridf_v)
                        & (existing_v > 0)
                    )

                    if not np.any(valid):
                        median_bias = np.nan
                        n_valid = 0
                    else:
                        bias = 100.0 * (gridf_v[valid] - existing_v[valid]) / existing_v[valid]
                        bias = bias[np.isfinite(bias)]
                        median_bias = float(np.nanmedian(bias)) if bias.size else np.nan
                        n_valid = int(bias.size)

                    rows.append({
                        "reference_key": ref,
                        "reference_label": reference_label(ref),
                        "biome_name": biome,
                        "biome_abbr": BIOME_ABBR.get(biome, biome),
                        "duration_min": float(duration_min),
                        "duration_label": duration_label(duration_min),
                        "return_period_yr": float(T),
                        "median_bias_pct": median_bias,
                        "n_stations": n_valid,
                    })

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def determine_y_limits(curves: pd.DataFrame, cfg: Config) -> tuple[float, float]:
    if cfg.manual_ymin_pct is not None and cfg.manual_ymax_pct is not None:
        return float(cfg.manual_ymin_pct), float(cfg.manual_ymax_pct)

    vals = curves["median_bias_pct"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -100.0, 100.0

    ymin = float(np.nanmin(vals))
    ymax = float(np.nanmax(vals))

    if ymin == ymax:
        ymin -= 5.0
        ymax += 5.0

    pad = cfg.y_padding_pct
    ymin -= pad
    ymax += pad

    if cfg.manual_ymin_pct is not None:
        ymin = float(cfg.manual_ymin_pct)
    if cfg.manual_ymax_pct is not None:
        ymax = float(cfg.manual_ymax_pct)

    return ymin, ymax


def style_curve_axis(ax, cfg: Config, show_xlabel: bool = True, show_ylabel: bool = True):
    ax.axhline(
        0.0,
        color="black",
        linewidth=cfg.zero_linewidth,
        linestyle="--",
        alpha=0.85,
        zorder=2,
    )

    ax.set_xscale("log")
    major_ticks = [2, 5, 10, 25, 50, 100, 250, 500]
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in major_ticks]))

    ax.grid(
        axis="both",
        which="major",
        linestyle=":",
        linewidth=cfg.grid_linewidth,
        alpha=0.48,
    )

    if show_xlabel:
        ax.set_xlabel("Return period, T (years)", fontsize=cfg.label_fontsize)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("Median bias (%)", fontsize=cfg.label_fontsize)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["left"].set_linewidth(cfg.axis_linewidth)
    ax.spines["bottom"].set_linewidth(cfg.axis_linewidth)


def make_biome_legend_handles(station_counts: pd.DataFrame | None = None) -> list[Line2D]:
    """
    Create legend handles for biome colors.

    The legend intentionally does not include counts, because counts differ
    between Standard and Disaggregation. Counts are shown in row-specific
    boxes instead.
    """
    handles = []

    for biome in BIOME_ORDER:
        color = BIOME_COLORS.get(biome, "0.45")
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=biome,
            )
        )

    return handles


def station_count_for_reference(station_counts: pd.DataFrame, reference_key: str) -> int:
    sub = station_counts[
        (station_counts["reference_key"] == reference_key)
        & (station_counts["biome_name"] == "ALL")
    ]

    if sub.empty:
        return 0

    return int(sub["n_stations"].iloc[0])


def biome_count_for_reference(station_counts: pd.DataFrame, reference_key: str, biome: str) -> int:
    sub = station_counts[
        (station_counts["reference_key"] == reference_key)
        & (station_counts["biome_name"] == biome)
    ]

    if sub.empty:
        return 0

    return int(sub["n_stations"].iloc[0])


def add_biome_count_box(
    ax,
    station_counts: pd.DataFrame,
    reference_key: str,
    cfg: Config,
):
    """
    Add a compact row-specific count box showing n by biome.

    Example:
        n by biome
        AMZ=100  CAT=30  CER=60
        MAT=120  PAM=10  PAN=5
    """
    count_lines = []
    pair_strings = []

    for biome in BIOME_ORDER:
        abbr = BIOME_ABBR.get(biome, biome)
        n = biome_count_for_reference(station_counts, reference_key, biome)
        pair_strings.append(f"{abbr}={n:,}")

    # Two compact lines of three biome counts each.
    count_lines.append("number of stations")
    count_lines.append("   ".join(pair_strings[:3]))
    count_lines.append("   ".join(pair_strings[3:]))

    ax.text(
        0.985,
        0.965,
        "\n".join(count_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=cfg.count_box_fontsize,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.22",
            fc="white",
            ec="0.35",
            lw=0.55,
            alpha=0.88,
        ),
        zorder=30,
    )


def plot_reference_duration_grid(
    curves: pd.DataFrame,
    station_counts: pd.DataFrame,
    cfg: Config,
) -> plt.Figure:
    references = list(cfg.references_to_plot)
    durations = list(cfg.durations_min_for_curves)

    nrows = len(references)
    ncols = len(durations)

    fig_width = max(cfg.min_figure_width, cfg.figure_width_per_duration * ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, cfg.figure_height),
        dpi=cfg.dpi,
        sharex=True,
        sharey=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    ymin, ymax = determine_y_limits(curves, cfg)

    if ymax - ymin > 125:
        y_major_step = 50
    else:
        y_major_step = 25

    panel_idx = 0

    for i, ref in enumerate(references):
        n_ref = station_count_for_reference(station_counts, ref)

        for j, duration_min in enumerate(durations):
            ax = axes[i, j]

            panel = curves[
                (curves["reference_key"] == ref)
                & (curves["duration_min"] == float(duration_min))
            ].copy()

            for biome in BIOME_ORDER:
                sub = panel[panel["biome_name"] == biome].copy()
                if sub.empty:
                    continue

                sub = sub.sort_values("return_period_yr")
                color = BIOME_COLORS.get(biome, "0.45")

                ax.plot(
                    sub["return_period_yr"],
                    sub["median_bias_pct"],
                    color=color,
                    linewidth=cfg.line_width,
                    marker="o",
                    markersize=cfg.marker_size,
                    markevery=cfg.marker_every,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    label=biome,
                    zorder=5,
                )

            ax.set_ylim(ymin, ymax)
            ax.yaxis.set_major_locator(MultipleLocator(y_major_step))

            show_xlabel = i == nrows - 1
            show_ylabel = j == 0
            style_curve_axis(ax, cfg, show_xlabel=show_xlabel, show_ylabel=show_ylabel)

            # Column titles only. No figure-level title.
            if i == 0:
                ax.set_title(
                    duration_label(duration_min),
                    fontsize=cfg.panel_title_fontsize,
                    fontweight="regular",
                    pad=6.0,
                )

            # Small panel label, e.g., a), b), c).
            ax.text(
                0.018,
                0.970,
                f"{chr(97 + panel_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=cfg.panel_label_fontsize,
                color="black",
                zorder=20,
            )

            # Add biome-specific station counts once per row, in the first panel.
            if j == 0:
                add_biome_count_box(ax, station_counts, ref, cfg)

            panel_idx += 1

        # Row label with total number of stations used for that IDF type.
        row_ax = axes[i, 0]
        row_ax.text(
            -0.32,
            0.50,
            f"{reference_label(ref)}\nTotal n={n_ref:,}",
            transform=row_ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=cfg.row_label_fontsize,
            color="black",
            zorder=30,
        )

    handles = make_biome_legend_handles(station_counts)
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        fontsize=cfg.legend_fontsize,
        bbox_to_anchor=(0.5, 0.016),
        borderpad=0.55,
        handlelength=2.15,
        handletextpad=0.48,
        columnspacing=1.05,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_linewidth(0.55)
    legend.get_frame().set_alpha(0.92)

    fig.subplots_adjust(
        left=0.105,
        right=0.990,
        top=0.945,
        bottom=0.140,
        wspace=0.105,
        hspace=0.165,
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    cfg = CFG
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    setup_style(cfg)

    df = read_and_prepare_station_table(cfg)
    station_counts = compute_station_counts(df)
    curves = compute_median_bias_curves(df, cfg)

    if cfg.save_station_counts_csv:
        counts_path = cfg.out_dir / f"{cfg.out_name}_station_counts.csv"
        station_counts.to_csv(counts_path, index=False)
        print(f"Saved: {counts_path}")

    if cfg.save_curve_summary_csv:
        csv_path = cfg.out_dir / f"{cfg.out_name}_curves.csv"
        curves.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    fig = plot_reference_duration_grid(curves, station_counts, cfg)

    for ext in ["png", "pdf", "svg"]:
        out = cfg.out_dir / f"{cfg.out_name}.{ext}"
        fig.savefig(out, dpi=cfg.dpi if ext == "png" else None)
        print(f"Saved: {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
