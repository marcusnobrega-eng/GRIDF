#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial bias maps for BR-DWGD GRIDF raster IDFs versus existing IDFs,
with small inset biome-wise boxplots inside each panel.

What this script does
---------------------
1) Reads existing IDF equations from IDF_Curves_Filtered.xlsx
   - sheet "Standard"
   - sheet "Disaggregation"
2) Samples BR-DWGD raster IDF parameters (K, a, b, c) at each station from:
   /Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER
3) Computes bias [%] at selected durations and return periods:
       bias = 100 * (BR-DWGD raster IDF - Existing IDF) / Existing IDF
4) Produces a multi-panel spatial figure over Brazil with:
   - the same DEM hillshade background used in the GRIDF study-area figure
   - biome polygons styled with the same colors, alpha, and boundaries as that figure
   - station dots colored by bias
   - a small inset vertical boxplot in the bottom-left corner of each map,
     showing the bias distribution by biome for that panel
   - a thin pointed colorbar with extensions below -100% and above +100%

Default figure layout
---------------------
Rows    -> selected durations
Columns -> [Standard T=10, Standard T=25, Disaggregation T=10, Disaggregation T=25]

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
from rasterio.mask import mask

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm


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

    # BR-DWGD raster folder ONLY.
    # This script does not search the whole GRIDF folder, so it will not
    # accidentally select IMERG/CHIRPS/PERSIANN rasters.
    product_name: str = "BR-DWGD"
    product_raster_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER"
    )

    # Optional exact raster paths. Here we explicitly force the BR-DWGD
    # Gumbel rasters so the script does not auto-select GEV or any other
    # distribution by filename.
    product_raster_paths: dict[str, Path] | None = field(default_factory=lambda: {
        "K": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_K_GUMBEL.tif"),
        "a": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_a_GUMBEL.tif"),
        "b": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_b_GUMBEL.tif"),
        "c": Path("/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs/bias_corrected_mean/br_dwgd/RASTER/IDF_c_GUMBEL.tif"),
    })
    auto_find_product_rasters: bool = False

    # Biomes shapefile.
    biomes_shp: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes/Brazil_biomes.shp"
    )

    # DEM used to create the same hillshade background as the GRIDF
    # study-area figure. The DEM is cropped to the Brazil polygon.
    dem_tif: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Misc/DEM.tif"
    )

    # Avenir Next font file path.
    # The script will use this if it exists; otherwise it falls back to any
    # installed Avenir Next family, then DejaVu Sans.
    font_file: Path = Path(
        "/Users/mngomes/Library/Fonts/Avenir Next.ttc"
    )
    font_family: str = "Avenir Next"
    fallback_font: str = "DejaVu Sans"

    # Output folder.
    out_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Existing_IDFs/Figures_Spatial_Bias"
    )
    out_name: str = "Spatial_bias_maps_BR_DWGD_raster_vs_existing_IDFs_with_biome_insets_pointed_colorbar"

    # -------------------------------------------------------------------------
    # What to plot
    # -------------------------------------------------------------------------
    reference_blocks: tuple[str, ...] = ("standard", "disaggregation")
    selected_return_periods_yr: tuple[float, ...] = (10, 25)
    selected_durations_min: tuple[float, ...] = (15, 60, 720, 1440)

    # "intensity" -> mm/h ; "depth" -> mm
    quantity_mode: str = "intensity"

    # -------------------------------------------------------------------------
    # Bias-color settings
    # -------------------------------------------------------------------------
    # Bias definition: 100 * (BR-DWGD raster IDF - Existing IDF) / Existing IDF.
    # Positive = BR-DWGD raster IDF larger than existing IDF.
    manual_bias_abs_limit_pct: float | None = 100.0
    bias_abs_percentile_for_limit: float = 95.0

    # Keep False for pointed colorbar behavior. Values below/above the colorbar
    # range are drawn using the end colors and represented by triangular extensions.
    clip_display_bias_to_limit: bool = False

    # Optional station-level bias filter before plotting.
    filter_extreme_bias_stations_for_plot: bool = False
    max_abs_bias_station_threshold_pct: float | None = 1000.0
    p95_abs_bias_station_threshold_pct: float | None = None

    # -------------------------------------------------------------------------
    # Marker and aesthetics
    # -------------------------------------------------------------------------
    marker_size: float = 24.0
    marker_edgecolor: str = "white"
    marker_linewidth: float = 0.35
    marker_alpha: float = 0.95

    # -------------------------------------------------------------------------
    # Figure aesthetics
    # -------------------------------------------------------------------------
    dpi: int = 600
    figure_width: float = 12.8
    figure_height: float = 12.9

    # Match the study-area figure styling.
    country_linewidth: float = 0.85
    biome_linewidth: float = 0.85
    biome_alpha: float = 0.27
    hillshade_alpha: float = 0.28
    map_padding_deg: float = 0.65

    bias_cmap_name: str = "RdBu_r"

    # Thin pointed colorbar.
    colorbar_left: float = 0.20
    colorbar_bottom: float = 0.030
    colorbar_width: float = 0.60
    colorbar_height: float = 0.0085
    colorbar_extend: str = "both"  # pointed ends on both sides

    # Inset boxplot position inside each panel: [x0, y0, width, height].
    # Matches the reference figure inset placement.
    inset_bbox: tuple[float, float, float, float] = (0.02, 0.28, 0.30, 0.25)

    # Inset font and line sizes matched to the reference figure.
    inset_xtick_fontsize: float = 8.0
    inset_ytick_fontsize: float = 6.0
    inset_ylabel_fontsize: float = 8.0
    inset_title_fontsize: float = 8.0
    inset_axis_linewidth: float = 0.45
    inset_tick_length: float = 1.2
    inset_tick_width: float = 0.4

    # Save diagnostics.
    save_station_bias_csv: bool = True
    save_full_panel_bias_csv: bool = True
    save_selected_raster_paths_csv: bool = True


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

    # Same global typography/line settings used in the reference GRIDF figure.
    mpl.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.titlesize": 13.0,
        "axes.linewidth": 1.15,
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

# Fixed biome colors from the previous GRIDF map style.
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


def dissolve_country(biomes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    country = biomes.dissolve()
    country = country.set_crs(biomes.crs)
    return country.to_crs("EPSG:4326")


def assign_biomes_to_stations(df: pd.DataFrame, biomes: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf, biomes[["biome_name", "geometry"]], how="left", predicate="intersects")
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined["biome_name"] = joined["biome_name"].fillna("Unknown")
    return pd.DataFrame(joined.drop(columns="geometry"))


def plot_biome_background(ax, biomes: gpd.GeoDataFrame, cfg: Config):
    """Plot biome polygons with the same colors and boundary styling as the reference figure."""
    for biome in BIOME_ORDER:
        sub = biomes[biomes["biome_name"] == biome]
        if len(sub) == 0:
            continue
        sub.plot(
            ax=ax,
            color=BIOME_COLORS.get(biome, "#BDBDBD"),
            alpha=cfg.biome_alpha,
            edgecolor="0.35",
            linewidth=cfg.biome_linewidth,
            zorder=1,
        )

    others = biomes[~biomes["biome_name"].isin(BIOME_ORDER)]
    if len(others) > 0:
        others.plot(
            ax=ax,
            color="#D9D9D9",
            alpha=cfg.biome_alpha,
            edgecolor="0.35",
            linewidth=cfg.biome_linewidth,
            zorder=1,
        )

    biomes.boundary.plot(ax=ax, color="0.25", linewidth=cfg.biome_linewidth, zorder=2)


def style_map_axis(ax, brazil: gpd.GeoDataFrame, cfg: Config):
    minx, miny, maxx, maxy = brazil.total_bounds
    ax.set_xlim(minx - cfg.map_padding_deg, maxx + cfg.map_padding_deg)
    ax.set_ylim(miny - cfg.map_padding_deg, maxy + cfg.map_padding_deg)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)



# =============================================================================
# DEM / HILLSHADE HELPERS
# =============================================================================

def crop_dem_to_brazil(dem_path: Path, brazil: gpd.GeoDataFrame):
    """Crop the DEM to Brazil and return DEM array, minimal profile, and plot extent."""
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        shapes = brazil.to_crs(src.crs).geometry
        out_image, out_transform = mask(src, shapes, crop=True, nodata=np.nan, filled=True)
        dem = out_image[0].astype("float32")

        if src.nodata is not None and np.isfinite(src.nodata):
            dem = np.where(dem == src.nodata, np.nan, dem)

        dem = np.where((dem < -1000) | (dem > 9000), np.nan, dem)

        profile = {
            "height": dem.shape[0],
            "width": dem.shape[1],
            "transform": out_transform,
            "crs": src.crs,
        }

        extent = [
            out_transform.c,
            out_transform.c + out_transform.a * dem.shape[1],
            out_transform.f + out_transform.e * dem.shape[0],
            out_transform.f,
        ]

    return dem, profile, extent


def compute_hillshade(dem: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """Compute the same simple hillshade used by the reference GRIDF figure."""
    arr = dem.astype(float)
    arr_fill = np.where(np.isfinite(arr), arr, np.nanmedian(arr))

    dy, dx = np.gradient(arr_fill)
    slope = np.pi / 2 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)

    shaded = (
        np.sin(alt) * np.sin(slope)
        + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    shaded = (shaded - np.nanmin(shaded)) / (np.nanmax(shaded) - np.nanmin(shaded))
    shaded[~np.isfinite(dem)] = np.nan
    return shaded

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
        np.isfinite(out["latitude"]) & np.isfinite(out["longitude"]) &
        out["longitude"].between(-76, -30) & out["latitude"].between(-36, 8) &
        np.isfinite(out["K"]) & np.isfinite(out["a"]) & np.isfinite(out["b"]) & np.isfinite(out["c"]) &
        (out["K"] > 0)
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
        f"coefficient_{p}", f"idf_{p.lower()}", f"{p}_gev", f"{p}_gumbel",
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


def compute_panel_bias_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for T in cfg.selected_return_periods_yr:
            for d in cfg.selected_durations_min:
                existing_i = idf_intensity(row["K"], row["a"], row["b"], row["c"], T, d)
                gridf_i = idf_intensity(row["K_r"], row["a_r"], row["b_r"], row["c_r"], T, d)
                existing_v = convert_quantity(existing_i, d, cfg.quantity_mode)
                gridf_v = convert_quantity(gridf_i, d, cfg.quantity_mode)

                if not np.isfinite(existing_v) or existing_v <= 0 or not np.isfinite(gridf_v):
                    bias = np.nan
                else:
                    bias = 100.0 * (gridf_v - existing_v) / existing_v

                records.append({
                    "reference_key": row["reference_key"],
                    "source_sheet": row["source_sheet"],
                    "source_row": int(row["source_row"]),
                    "station_code": row["station_code"],
                    "station_name": row["station_name"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "biome_name": row.get("biome_name", "Unknown"),
                    "return_period_yr": T,
                    "duration_min": d,
                    "existing_value": existing_v,
                    "gridf_value": gridf_v,
                    "bias_pct": bias,
                })
    return pd.DataFrame(records)


def compute_station_bias_diagnostics(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["reference_key", "source_sheet", "source_row", "station_code", "station_name", "latitude", "longitude"]
    for key, g in panel_df.groupby(group_cols):
        vals = pd.to_numeric(g["bias_pct"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append({
            "reference_key": key[0],
            "source_sheet": key[1],
            "source_row": int(key[2]),
            "station_code": key[3],
            "station_name": key[4],
            "latitude": key[5],
            "longitude": key[6],
            "n_bias_values": int(vals.size),
            "median_bias_pct": float(np.nanmedian(vals)) if vals.size else np.nan,
            "mean_bias_pct": float(np.nanmean(vals)) if vals.size else np.nan,
            "std_bias_pct": float(np.nanstd(vals)) if vals.size else np.nan,
            "max_abs_bias_pct": float(np.nanmax(np.abs(vals))) if vals.size else np.nan,
            "p95_abs_bias_pct": float(np.nanpercentile(np.abs(vals), 95)) if vals.size else np.nan,
        })
    return pd.DataFrame(rows)


def filter_extreme_bias_stations(panel_df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_diag = compute_station_bias_diagnostics(panel_df)
    removed = pd.DataFrame(columns=station_diag.columns)

    if not cfg.filter_extreme_bias_stations_for_plot:
        return panel_df.copy(), removed

    keep_mask = np.ones(len(station_diag), dtype=bool)
    if cfg.max_abs_bias_station_threshold_pct is not None:
        keep_mask &= station_diag["max_abs_bias_pct"].to_numpy(dtype=float) <= cfg.max_abs_bias_station_threshold_pct
    if cfg.p95_abs_bias_station_threshold_pct is not None:
        keep_mask &= station_diag["p95_abs_bias_pct"].to_numpy(dtype=float) <= cfg.p95_abs_bias_station_threshold_pct

    removed = station_diag.loc[~keep_mask].copy().reset_index(drop=True)
    keep_keys = station_diag.loc[keep_mask, [
        "reference_key", "source_sheet", "source_row", "station_code", "station_name", "latitude", "longitude"
    ]].copy()

    filtered = panel_df.merge(
        keep_keys,
        on=["reference_key", "source_sheet", "source_row", "station_code", "station_name", "latitude", "longitude"],
        how="inner",
    )
    print(f"[FILTER] Retained {filtered[['reference_key','source_row']].drop_duplicates().shape[0]:,} stations; removed {len(removed):,} extreme-bias stations.")
    return filtered, removed


def determine_bias_limit(panel_df: pd.DataFrame, cfg: Config) -> float:
    vals = pd.to_numeric(panel_df["bias_pct"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.abs(vals)
    if vals.size == 0:
        return 100.0
    if cfg.manual_bias_abs_limit_pct is not None:
        return float(cfg.manual_bias_abs_limit_pct)
    lim = float(np.nanpercentile(vals, cfg.bias_abs_percentile_for_limit))
    return max(5.0, lim)


# =============================================================================
# PANEL PLOTTING
# =============================================================================

def ref_label(ref_key: str) -> str:
    return "Standard" if ref_key == "standard" else "Disaggregation"


def duration_label(dmin: float) -> str:
    dmin = float(dmin)
    if dmin < 60:
        return f"t = {int(dmin)} min"
    if dmin < 1440:
        hours = dmin / 60.0
        if abs(hours - round(hours)) < 1e-9:
            return f"t = {int(round(hours))} h"
        return f"t = {hours:g} h"
    days = dmin / 1440.0
    if abs(days - round(days)) < 1e-9:
        return f"t = {int(round(days))} d"
    return f"t = {days:g} d"


def build_panel_matrix(cfg: Config) -> list[tuple[str, float]]:
    cols = []
    for ref in cfg.reference_blocks:
        for T in cfg.selected_return_periods_yr:
            cols.append((ref, float(T)))
    return cols


def add_biome_boxplot_inset(ax, sub: pd.DataFrame, bias_abs_lim: float, cfg: Config):
    """Add a vertical biome-wise boxplot inset styled like the reference figure."""
    inset = ax.inset_axes(cfg.inset_bbox)
    inset.set_facecolor((1, 1, 1, 0.82))

    data = []
    labels = []
    colors = []
    for biome in BIOME_ORDER:
        vals = sub.loc[sub["biome_name"] == biome, "bias_pct"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        # For inset readability, clip to the displayed colorbar range.
        vals = np.clip(vals, -bias_abs_lim, bias_abs_lim)
        data.append(vals)
        labels.append(biome)
        colors.append(BIOME_COLORS.get(biome, "#BDBDBD"))

    if len(data) == 0:
        inset.text(
            0.5,
            0.5,
            "No biome data",
            ha="center",
            va="center",
            fontsize=cfg.inset_ytick_fontsize,
        )
        inset.set_xticks([])
        inset.set_yticks([])
        for side in ["top", "right"]:
            inset.spines[side].set_visible(False)
        inset.spines["left"].set_linewidth(cfg.inset_axis_linewidth)
        inset.spines["bottom"].set_linewidth(cfg.inset_axis_linewidth)
        return

    bp = inset.boxplot(
        data,
        vert=True,
        patch_artist=True,
        labels=labels,
        widths=0.58,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.80),
        whiskerprops=dict(color="0.15", linewidth=0.45),
        capprops=dict(color="0.15", linewidth=0.45),
        boxprops=dict(linewidth=0.45, edgecolor="0.15"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.90)

    inset.axhline(0.0, color="black", linestyle="--", linewidth=0.65, alpha=0.90, zorder=0)
    inset.set_ylim(-bias_abs_lim, bias_abs_lim)
    inset.set_yticks([-bias_abs_lim, 0, bias_abs_lim])

    # Match the reference figure inset typography and tick styling.
    inset.set_xticks(range(1, len(labels) + 1))
    inset.set_xticklabels(
        labels,
        fontsize=cfg.inset_xtick_fontsize,
        rotation=90,
        ha="center",
        va="top",
    )
    inset.tick_params(
        axis="x",
        length=cfg.inset_tick_length,
        width=cfg.inset_tick_width,
        pad=0.6,
    )
    inset.tick_params(
        axis="y",
        labelsize=cfg.inset_ytick_fontsize,
        length=cfg.inset_tick_length,
        width=cfg.inset_tick_width,
        pad=0.7,
    )

    inset.set_ylabel("Bias (%)", fontsize=cfg.inset_ylabel_fontsize, labelpad=1.2)
    inset.grid(axis="y", linestyle=":", linewidth=0.35, alpha=0.50)

    for side in ["top", "right"]:
        inset.spines[side].set_visible(False)
    inset.spines["left"].set_linewidth(cfg.inset_axis_linewidth)
    inset.spines["bottom"].set_linewidth(cfg.inset_axis_linewidth)


def plot_spatial_bias_figure(
    panel_df: pd.DataFrame,
    biomes: gpd.GeoDataFrame,
    brazil: gpd.GeoDataFrame,
    cfg: Config,
    bias_abs_lim: float,
    hillshade: Optional[np.ndarray] = None,
    hillshade_extent: Optional[list[float]] = None,
) -> plt.Figure:
    col_specs = build_panel_matrix(cfg)
    nrows = len(cfg.selected_durations_min)
    ncols = len(col_specs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(cfg.figure_width, cfg.figure_height), dpi=cfg.dpi)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    fig.subplots_adjust(left=0.045, right=0.972, top=0.985, bottom=0.065, wspace=0.018, hspace=0.092)

    cmap = plt.get_cmap(cfg.bias_cmap_name).copy()
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    norm = TwoSlopeNorm(vmin=-bias_abs_lim, vcenter=0.0, vmax=bias_abs_lim)

    for i, d in enumerate(cfg.selected_durations_min):
        for j, (ref, T) in enumerate(col_specs):
            ax = axes[i, j]
            sub = panel_df[
                (panel_df["reference_key"] == ref) &
                (panel_df["return_period_yr"] == T) &
                (panel_df["duration_min"] == d)
            ].copy()

            if hillshade is not None and hillshade_extent is not None:
                hs_m = np.ma.masked_invalid(hillshade)
                ax.imshow(
                    hs_m,
                    extent=hillshade_extent,
                    origin="upper",
                    cmap="gray",
                    alpha=cfg.hillshade_alpha,
                    zorder=0,
                )

            plot_biome_background(ax, biomes, cfg)
            brazil.boundary.plot(ax=ax, color="black", linewidth=cfg.country_linewidth, zorder=3)

            if len(sub) > 0:
                x = sub["longitude"].to_numpy(dtype=float)
                y = sub["latitude"].to_numpy(dtype=float)
                c = sub["bias_pct"].to_numpy(dtype=float)
                if cfg.clip_display_bias_to_limit:
                    c = np.clip(c, -bias_abs_lim, bias_abs_lim)
                ax.scatter(
                    x, y,
                    c=c,
                    s=cfg.marker_size,
                    cmap=cmap,
                    norm=norm,
                    edgecolors=cfg.marker_edgecolor,
                    linewidths=cfg.marker_linewidth,
                    alpha=cfg.marker_alpha,
                    zorder=6,
                )

            style_map_axis(ax, brazil, cfg)
            ax.set_title(f"{ref_label(ref)} | T = {int(T)} yr | {duration_label(d)}", pad=3.5, fontweight="regular")

            add_biome_boxplot_inset(ax, sub, bias_abs_lim, cfg)

    # Shared thin pointed colorbar.
    cax = fig.add_axes([cfg.colorbar_left, cfg.colorbar_bottom, cfg.colorbar_width, cfg.colorbar_height])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        extend=cfg.colorbar_extend,
        extendfrac=0.045,
    )
    cb.set_label("Bias = 100 × (BR-DWGD raster IDF − Existing IDF) / Existing IDF (%)", fontsize=10.3)
    cb.ax.tick_params(labelsize=8.8, width=0.75, length=2.5)
    cb.outline.set_linewidth(0.75)

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

    print("Loading biome polygons...")
    biomes = load_biomes(cfg.biomes_shp)
    brazil = dissolve_country(biomes)

    print("Loading DEM and computing hillshade background...")
    dem, dem_profile, dem_extent = crop_dem_to_brazil(cfg.dem_tif, brazil)
    hillshade = compute_hillshade(dem)

    print("Resolving BR-DWGD raster parameter files...")
    raster_paths = resolve_product_rasters(cfg)
    if cfg.save_selected_raster_paths_csv:
        pd.DataFrame({"param": list(raster_paths.keys()), "path": [str(v) for v in raster_paths.values()]}).to_csv(
            cfg.out_dir / "00_selected_brdwgd_rasters_used_for_figure.csv",
            index=False,
        )

    print("Reading existing IDF sheets...")
    blocks = []
    if "standard" in cfg.reference_blocks:
        blocks.append(read_existing_idf_sheet(cfg.filtered_xlsx, cfg.standard_sheet, "standard"))
    if "disaggregation" in cfg.reference_blocks:
        blocks.append(read_existing_idf_sheet(cfg.filtered_xlsx, cfg.disaggregation_sheet, "disaggregation"))
    df = pd.concat(blocks, ignore_index=True)

    print("Sampling BR-DWGD raster coefficients at station locations...")
    df = attach_product_coefficients(df, raster_paths)

    valid_r = (
        np.isfinite(df["K_r"]) & np.isfinite(df["a_r"]) &
        np.isfinite(df["b_r"]) & np.isfinite(df["c_r"]) &
        (df["K_r"] > 0)
    )
    n_before = len(df)
    df = df.loc[valid_r].copy().reset_index(drop=True)
    print(f"Retained {len(df):,}/{n_before:,} stations after requiring sampled BR-DWGD raster coefficients.")

    print("Assigning biome to each station...")
    df = assign_biomes_to_stations(df, biomes)

    print("Computing panel bias values...")
    panel_df = compute_panel_bias_table(df, cfg)
    if cfg.save_full_panel_bias_csv:
        panel_df.to_csv(cfg.out_dir / "01_panel_bias_values_all_points.csv", index=False)

    station_diag_before = compute_station_bias_diagnostics(panel_df)
    if cfg.save_station_bias_csv:
        station_diag_before.to_csv(cfg.out_dir / "02_station_bias_diagnostics_before_filter.csv", index=False)

    panel_df_plot, removed = filter_extreme_bias_stations(panel_df, cfg)
    if len(removed) > 0:
        removed.to_csv(cfg.out_dir / "03_removed_extreme_bias_stations.csv", index=False)

    station_diag_after = compute_station_bias_diagnostics(panel_df_plot)
    if cfg.save_station_bias_csv:
        station_diag_after.to_csv(cfg.out_dir / "04_station_bias_diagnostics_after_filter.csv", index=False)

    bias_abs_lim = determine_bias_limit(panel_df_plot, cfg)
    print(
        f"Using pointed symmetric bias color limit: < -{bias_abs_lim:.2f}% "
        f"and > +{bias_abs_lim:.2f}% shown with colorbar extensions."
    )

    print("Making spatial bias figure...")
    fig = plot_spatial_bias_figure(
        panel_df_plot,
        biomes,
        brazil,
        cfg,
        bias_abs_lim,
        hillshade=hillshade,
        hillshade_extent=dem_extent,
    )

    for ext in ["png", "pdf", "svg"]:
        out = cfg.out_dir / f"{cfg.out_name}.{ext}"
        fig.savefig(out, dpi=cfg.dpi if ext == "png" else None)
        print(f"Saved: {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
