#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIDF-BR study-area and input-data figure.

This script creates a 12-panel figure for the GRIDF paper.

Panels a-f reproduce the logic of the original study-area figure:
  a) ANA sub-daily stations before quality control
  b) Existing IDF equations derived with sub-daily data before quality control
  c) Existing IDF equations derived with disaggregation methods before quality control
  d) ANA sub-daily stations after quality control
  e) Existing IDF equations derived with sub-daily data after quality control
  f) Existing IDF equations derived with disaggregation methods after quality control

Panels g-l add the datasets used in the updated GRIDF workflow:
  g) Topography and biome boundaries
  h) BR-DWGD mean annual-maximum daily precipitation
  i) CHIRPS mean annual-maximum daily precipitation
  j) IMERG V06 mean annual-maximum daily precipitation
  k) IMERG V07 mean annual-maximum daily precipitation
  l) PERSIANN-CDR mean annual-maximum daily precipitation

Key features:
  - all point datasets are clipped to the Brazil polygon derived from the biome shapefile
  - station markers are unfilled, smaller, and thinner
  - biome boundaries are used instead of state boundaries
  - the elevation colorbar is vertical beside the topography panel
  - mean and standard deviation labels are shown in the top-right corner of product maps
  - the product time period is shown in the bottom-right corner of product maps
  - product maps share one explicit ScalarMappable colorbar to avoid PDF/SVG rendering issues

Outputs:
  PNG, PDF, and SVG.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# USER SETTINGS
# =============================================================================

@dataclass
class Config:
    root: Path = Path("/Users/mngomes/Documents/GitHub/GRIDF")

    # -------------------------------------------------------------------------
    # Station inventories for panels a and d
    # -------------------------------------------------------------------------
    ana_subdaily_all_csv: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction/stations_inventory_filtered_all.csv"
    )
    ana_subdaily_qc_csv: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction/stations_inventory_filtered.csv"
    )

    # -------------------------------------------------------------------------
    # Existing IDF datasets for panels b, c, e, f
    # -------------------------------------------------------------------------
    idf_curves_all_xlsx: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Existing_IDFs/IDF_Curves_Brazil.xlsx"
    )
    idf_curves_filtered_xlsx: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Existing_IDFs/IDF_Curves_Filtered.xlsx"
    )

    # Sheet names in the Torres et al. workbook
    standard_sheet: str = "Standard"
    disaggregation_sheet: str = "Disaggregation"

    # -------------------------------------------------------------------------
    # Spatial inputs
    # -------------------------------------------------------------------------
    dem_tif: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Misc/DEM.tif"
    )
    biomes_shp: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes/Brazil_biomes.shp"
    )

    # -------------------------------------------------------------------------
    # Annual maximum precipitation folders for panels h-l
    # -------------------------------------------------------------------------
    annual_max_root: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Annual_Maximum_Precipitation"
    )
    annual_max_products: dict | None = None

    # Product-specific year windows for the mean annual-maximum maps.
    # Files outside these ranges are ignored when computing the map mean.
    # This keeps BR-DWGD limited to 1995--2025, as requested.
    product_year_windows: dict | None = None

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    out_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Figures"
    )
    out_name: str = "Figure_StudyArea_Stations_IDFs_Topography_AnnualMax_3x4"

    # -------------------------------------------------------------------------
    # Figure aesthetics
    # -------------------------------------------------------------------------
    font_family: str = "Avenir Next"
    fallback_font: str = "DejaVu Sans"
    dpi: int = 600

    # Use smaller and thinner unfilled markers.
    station_marker_size: float = 5.2
    station_linewidth: float = 0.38

    # Boundaries
    biome_linewidth: float = 0.85
    country_linewidth: float = 0.85

    # Product-map color scale.
    # Leave None to use percentiles from all products.
    precip_vmin: Optional[float] = 25.0
    precip_vmax: Optional[float] = 125.0
    precip_pmin: float = 2.0
    precip_pmax: float = 98.0

    # Manual bottom colorbar axis for product maps:
    # [left, bottom, width, height] in figure fraction.
    # This is intentionally manual to render correctly in PNG, PDF, and SVG.
    precip_cbar_axes: tuple[float, float, float, float] = (0.115, 0.042, 0.77, 0.012)

    def __post_init__(self):
        if self.annual_max_products is None:
            self.annual_max_products = {
                "BR-DWGD": self.annual_max_root / "BR-DWGD",
                "CHIRPS": self.annual_max_root / "CHIRPS_Max",
                "IMERG V06": self.annual_max_root / "IMERG_V06_Max",
                "IMERG V07": self.annual_max_root / "IMERG_V07_Max",
                "PERSIANN-CDR": self.annual_max_root / "PERSIANN_CDR_Max",
            }

        if self.product_year_windows is None:
            self.product_year_windows = {
                "BR-DWGD": (2001, 2020),
                "CHIRPS": (2001, 2020),
                "IMERG V06": (2001, 2020),
                "IMERG V07": (2001, 2020),
                "PERSIANN-CDR": (2001, 2020),
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
    "LATITUDE", "y", "Y"
]
LON_CANDIDATES = [
    "Longitude", "longitude", "lon", "LON", "Longitude (º)", "Longitude (°)",
    "LONGITUDE", "x", "X"
]
CODE_CANDIDATES = [
    "Code", "code", "station_id", "station_ID", "ID", "id", "station", "station_code"
]
NAME_CANDIDATES = [
    "Name", "name", "Station", "station", "station_name", "Nome", "NOME"
]


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
# GEOMETRY / POINT READERS
# =============================================================================

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
    return gdf


def dissolve_country(biomes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    country = biomes.dissolve()
    country = country.set_crs(biomes.crs)
    return country.to_crs("EPSG:4326")


def find_biome_name_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    return find_first_column(
        gdf.columns,
        ["Bioma", "BIOMA", "bioma", "Biome", "BIOME", "biome", "Name", "name", "NOME", "nome"],
    )


def dataframe_to_points(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    label: str,
    deduplicate: bool = False,
) -> gpd.GeoDataFrame:
    df = df.copy()

    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["source_label"] = label

    df = df.dropna(subset=["latitude", "longitude"])

    # Broad sanity filter. Exact country clipping occurs later.
    df = df[
        df["longitude"].between(-76, -30)
        & df["latitude"].between(-36, 8)
    ].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    if deduplicate:
        code_col = find_first_column(gdf.columns, CODE_CANDIDATES)
        if code_col is not None:
            gdf = gdf.drop_duplicates(subset=[code_col])
        else:
            gdf["_x"] = gdf.geometry.x.round(6)
            gdf["_y"] = gdf.geometry.y.round(6)
            gdf = gdf.drop_duplicates(subset=["_x", "_y"])
            gdf = gdf.drop(columns=["_x", "_y"])

    return gdf.reset_index(drop=True)


def clip_points_to_brazil(points: gpd.GeoDataFrame, brazil: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """
    Clip station points to Brazil polygon. This fixes offshore/outside-Brazil points.
    """
    if points.empty:
        return points

    points = points.to_crs("EPSG:4326")
    brazil_geom = brazil.to_crs("EPSG:4326").geometry.unary_union.buffer(1e-8)

    keep = points.geometry.intersects(brazil_geom)
    clipped = points.loc[keep].copy().reset_index(drop=True)

    removed = len(points) - len(clipped)
    if removed > 0:
        print(f"[INFO] {label}: removed {removed} points outside Brazil.")

    return clipped


def clean_biome_label(name) -> str:
    """Clean biome names for display."""
    s = str(name)
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def get_biome_color_map(biomes: gpd.GeoDataFrame) -> dict:
    """
    Assign distinguishable, high-contrast scientific colors to each biome.

    The colors follow a colorblind-friendly/high-contrast qualitative style
    suitable for scientific figures.
    """
    biome_col = find_biome_name_column(biomes)
    if biome_col is None:
        return {}

    names = [clean_biome_label(x) for x in biomes[biome_col].dropna().unique()]

    preferred_order = [
        "Amazônia", "Amazonia", "Amazon", "Amazônia Legal",
        "Caatinga",
        "Cerrado",
        "Mata Atlântica", "Mata Atlantica", "Atlantic Forest",
        "Pampa",
        "Pantanal",
    ]

    ordered = []
    for target in preferred_order:
        for n in names:
            if n not in ordered and target.lower() in n.lower():
                ordered.append(n)

    for n in sorted(names):
        if n not in ordered:
            ordered.append(n)

    palette = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#D55E00",  # vermillion
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#332288",  # deep indigo
        "#88CCEE",  # cyan
        "#AA4499",  # magenta
    ]

    return {name: palette[i % len(palette)] for i, name in enumerate(ordered)}


def plot_colored_biomes(ax, biomes: gpd.GeoDataFrame, cfg: Config, alpha: float = 0.30):
    """
    Plot biome polygons using a fixed distinguishable color per biome.
    """
    biome_col = find_biome_name_column(biomes)

    if biome_col is None:
        biomes.plot(
            ax=ax,
            color="#f3f3f3",
            alpha=alpha,
            edgecolor="0.35",
            linewidth=cfg.biome_linewidth,
            zorder=1,
        )
        return

    color_map = get_biome_color_map(biomes)

    for _, row in biomes.iterrows():
        label = clean_biome_label(row[biome_col])
        color = color_map.get(label, "#BDBDBD")
        gpd.GeoSeries([row.geometry], crs=biomes.crs).plot(
            ax=ax,
            color=color,
            alpha=alpha,
            edgecolor="0.35",
            linewidth=cfg.biome_linewidth,
            zorder=1,
        )

    biomes.boundary.plot(ax=ax, color="0.25", linewidth=cfg.biome_linewidth, zorder=2)


def station_density_by_biome(
    points: gpd.GeoDataFrame,
    biomes: gpd.GeoDataFrame,
    scale_area_km2: float = 10000.0,
) -> pd.DataFrame:
    """
    Compute station density by biome.

    Density is reported as stations per `scale_area_km2`.
    Default: stations per 10,000 km².
    """
    if points.empty:
        return pd.DataFrame(columns=["biome", "count", "area_km2", "density"])

    biome_col = find_biome_name_column(biomes)
    if biome_col is None:
        return pd.DataFrame(columns=["biome", "count", "area_km2", "density"])

    # ------------------------------------------------------------------
    # 1) Count stations per biome using lon/lat spatial join
    # ------------------------------------------------------------------
    points_ll = points[["geometry"]].copy().to_crs("EPSG:4326")
    biomes_ll = biomes[[biome_col, "geometry"]].copy().to_crs("EPSG:4326")

    try:
        try:
            joined = gpd.sjoin(
                points_ll,
                biomes_ll,
                how="left",
                predicate="intersects",
            )
        except Exception:
            joined = gpd.sjoin(points_ll, biomes_ll, how="left")
        labels = joined[biome_col].dropna().map(clean_biome_label)
        counts = labels.value_counts(dropna=True)
    except Exception:
        # Robust fallback if spatial-index backends are unavailable.
        values = []
        for geom in points_ll.geometry:
            hit = biomes_ll.loc[biomes_ll.geometry.intersects(geom), biome_col]
            values.append(clean_biome_label(hit.iloc[0]) if len(hit) > 0 else np.nan)
        counts = pd.Series(values).value_counts(dropna=True)

    # ------------------------------------------------------------------
    # 2) Compute biome areas in an equal-area projection
    # ------------------------------------------------------------------
    # Brazil/South-America Albers Equal Area projection.
    equal_area_crs = (
        "+proj=aea +lat_1=-5 +lat_2=-25 +lat_0=-15 "
        "+lon_0=-54 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    biomes_area = biomes[[biome_col, "geometry"]].copy().to_crs(equal_area_crs)
    biomes_area["biome"] = biomes_area[biome_col].map(clean_biome_label)

    # Dissolve because a biome may be split into multiple polygons.
    area_df = biomes_area.dissolve(by="biome").reset_index()
    area_df["area_km2"] = area_df.geometry.area / 1e6

    # ------------------------------------------------------------------
    # 3) Combine counts and areas
    # ------------------------------------------------------------------
    out = area_df[["biome", "area_km2"]].copy()
    out["count"] = out["biome"].map(counts).fillna(0).astype(float)
    out["density"] = out["count"] / out["area_km2"] * scale_area_km2

    out = out.sort_values("density", ascending=False).reset_index(drop=True)

    return out

def add_biome_hist_inset(ax, points: gpd.GeoDataFrame, biomes: gpd.GeoDataFrame):
    """
    Add a small station-density histogram by biome near the bottom-left.

    The plotted value is stations per 10,000 km².
    """
    dens = station_density_by_biome(
        points,
        biomes,
        scale_area_km2=10000.0,
    )

    if dens.empty:
        return

    # Keep the six main Brazilian biomes.
    dens = dens.iloc[:6].copy()

    color_map = get_biome_color_map(biomes)
    labels = dens["biome"].tolist()
    colors = [color_map.get(clean_biome_label(x), "0.5") for x in labels]

    # Bottom-left inset
    axins = ax.inset_axes([0.02, 0.28, 0.30, 0.25])

    axins.bar(
        range(len(dens)),
        dens["density"].values,
        color=colors,
        edgecolor="0.15",
        linewidth=0.35,
        alpha=0.90,
    )

    axins.set_facecolor((1, 1, 1, 0.82))
    axins.set_xticks(range(len(dens)))

    # Vertical biome names
    axins.set_xticklabels(
        labels,
        fontsize=8,
        rotation=90,
        ha="center",
        va="top",
    )

    axins.tick_params(axis="x", length=1.2, width=0.4, pad=0.6)
    axins.tick_params(axis="y", labelsize=6, length=1.2, width=0.4, pad=0.7)

    axins.set_ylabel(
        "stations / 10⁴ km²",
        fontsize=8,
        labelpad=1.2,
    )

    axins.grid(axis="y", linestyle=":", linewidth=0.35, alpha=0.5)

    for side in ["top", "right"]:
        axins.spines[side].set_visible(False)

    axins.spines["left"].set_linewidth(0.45)
    axins.spines["bottom"].set_linewidth(0.45)

def read_point_csv(path: Path, label: str, deduplicate: bool = False) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0, low_memory=False)
    cols = list(header.columns)

    lat_col = find_first_column(cols, LAT_CANDIDATES)
    lon_col = find_first_column(cols, LON_CANDIDATES)
    name_col = find_first_column(cols, NAME_CANDIDATES)
    code_col = find_first_column(cols, CODE_CANDIDATES)

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify latitude/longitude columns in {path}")

    usecols = [lat_col, lon_col]
    for col in [name_col, code_col]:
        if col is not None and col not in usecols:
            usecols.append(col)

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return dataframe_to_points(df, lat_col, lon_col, label=label, deduplicate=deduplicate)


def read_point_excel_sheet(
    path: Path,
    sheet_name: str,
    label: str,
    deduplicate: bool = False,
) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    header = pd.read_excel(path, sheet_name=sheet_name, nrows=0)
    cols = list(header.columns)

    lat_col = find_first_column(cols, LAT_CANDIDATES)
    lon_col = find_first_column(cols, LON_CANDIDATES)
    name_col = find_first_column(cols, NAME_CANDIDATES)
    code_col = find_first_column(cols, CODE_CANDIDATES)

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify latitude/longitude columns in {path}, sheet={sheet_name}")

    usecols = [lat_col, lon_col]
    for col in [name_col, code_col]:
        if col is not None and col not in usecols:
            usecols.append(col)

    df = pd.read_excel(path, sheet_name=sheet_name, usecols=usecols)
    return dataframe_to_points(df, lat_col, lon_col, label=label, deduplicate=deduplicate)


# =============================================================================
# RASTER HELPERS
# =============================================================================

def mask_array_to_brazil(arr: np.ndarray, profile: dict, brazil: gpd.GeoDataFrame) -> np.ndarray:
    brazil_raster = brazil.to_crs(profile["crs"])
    shapes = [geom for geom in brazil_raster.geometry if geom is not None and not geom.is_empty]

    outside = geometry_mask(
        shapes,
        out_shape=arr.shape,
        transform=profile["transform"],
        invert=False,
        all_touched=False,
    )

    out = arr.copy()
    out[outside] = np.nan
    return out


def crop_dem_to_brazil(dem_path: Path, brazil: gpd.GeoDataFrame):
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)

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


def make_topography_cmap() -> LinearSegmentedColormap:
    colors = ["#1f4e99", "#1c8dc5", "#23b37a", "#d8d56b", "#b28b56", "#f4efe6"]
    return LinearSegmentedColormap.from_list("gridf_topography", colors, N=256)


def list_tifs(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    return sorted([p for p in folder.glob("*.tif") if p.is_file()])


def get_year_from_name(path: Path) -> Optional[int]:
    """
    Extract the first 4-digit year from a raster filename.
    """
    matches = re.findall(r"(?:19|20)\d{2}", path.name)

    for m in matches:
        y = int(m)
        if 1900 <= y <= 2100:
            return y

    return None


def filter_tifs_by_year_window(
    tif_files: list[Path],
    year_window: Optional[tuple[int, int]],
) -> list[Path]:
    """
    Keep only rasters whose filename year falls within the selected analysis window.
    """
    if year_window is None:
        return tif_files

    y0, y1 = year_window
    kept: list[Path] = []

    for p in tif_files:
        y = get_year_from_name(p)

        if y is None:
            # Keep files with no detectable year rather than silently losing data.
            # This should normally not happen for annual-maximum rasters.
            kept.append(p)
        elif y0 <= y <= y1:
            kept.append(p)

    return sorted(kept)


def infer_years_from_tifs(tif_files: list[Path]) -> tuple[Optional[int], Optional[int], str]:
    """
    Infer year range from filenames such as:
      CHIRPS_MaxDaily_0p05deg_1995_Brazil.tif
      BR_DWGD_prmax_2025.tif
      something_2001_something.tif

    Returns:
      year_start, year_end, period_label
    """
    years: list[int] = []

    for p in tif_files:
        y = get_year_from_name(p)
        if y is not None:
            years.append(y)

    years = sorted(set(years))

    if len(years) == 0:
        return None, None, "unknown period"

    return years[0], years[-1], f"{years[0]}--{years[-1]}"


def read_raster_as_base_grid(path: Path, base_profile: Optional[dict] = None):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")

        if src.nodata is not None and np.isfinite(src.nodata):
            arr = np.where(arr == src.nodata, np.nan, arr)

        arr = np.where((arr < -1e20) | (arr > 1e20), np.nan, arr)

        if base_profile is None:
            profile = {
                "height": src.height,
                "width": src.width,
                "transform": src.transform,
                "crs": src.crs,
            }
            extent = [
                src.transform.c,
                src.transform.c + src.transform.a * src.width,
                src.transform.f + src.transform.e * src.height,
                src.transform.f,
            ]
            return arr, profile, extent

        same_grid = (
            src.height == base_profile["height"]
            and src.width == base_profile["width"]
            and src.transform == base_profile["transform"]
            and src.crs == base_profile["crs"]
        )

        if same_grid:
            return arr, base_profile, None

        dst = np.full((base_profile["height"], base_profile["width"]), np.nan, dtype="float32")
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=base_profile["transform"],
            dst_crs=base_profile["crs"],
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        return dst, base_profile, None


def compute_mean_annual_max_map(
    folder: Path,
    brazil: gpd.GeoDataFrame,
    year_window: Optional[tuple[int, int]] = None,
):
    tif_files_all = list_tifs(folder)
    tif_files = filter_tifs_by_year_window(tif_files_all, year_window)

    if len(tif_files) == 0:
        raise FileNotFoundError(
            f"No GeoTIFFs found in {folder} for year window {year_window}. "
            f"Total files before filtering: {len(tif_files_all)}"
        )

    year_start, year_end, period_label = infer_years_from_tifs(tif_files)

    first_arr, base_profile, extent = read_raster_as_base_grid(tif_files[0], base_profile=None)

    sum_arr = np.zeros_like(first_arr, dtype="float64")
    count_arr = np.zeros_like(first_arr, dtype="float64")

    for i, tif in enumerate(tif_files):
        arr, _, _ = read_raster_as_base_grid(tif, base_profile=base_profile if i > 0 else None)
        valid = np.isfinite(arr)
        sum_arr[valid] += arr[valid]
        count_arr[valid] += 1

    mean_arr = np.divide(
        sum_arr,
        count_arr,
        out=np.full_like(sum_arr, np.nan, dtype="float64"),
        where=count_arr > 0,
    ).astype("float32")

    mean_arr = mask_array_to_brazil(mean_arr, base_profile, brazil)

    spatial_mean = float(np.nanmean(mean_arr))
    spatial_std = float(np.nanstd(mean_arr))

    return (
        mean_arr,
        base_profile,
        extent,
        spatial_mean,
        spatial_std,
        len(tif_files),
        period_label,
        year_start,
        year_end,
    )


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def get_bounds_with_padding(gdf: gpd.GeoDataFrame, pad: float = 0.65):
    minx, miny, maxx, maxy = gdf.total_bounds
    return minx - pad, maxx + pad, miny - pad, maxy + pad


def style_map_axis(ax, brazil: gpd.GeoDataFrame):
    minx, maxx, miny, maxy = get_bounds_with_padding(brazil, pad=0.65)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def add_panel_label(ax, label: str):
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="regular",
        color="black",
        zorder=50,
    )


def plot_biome_background(ax, biomes: gpd.GeoDataFrame, cfg: Config):
    # Use fixed distinguishable colors so map polygons and inset histograms match.
    plot_colored_biomes(ax, biomes, cfg, alpha=0.27)

def plot_station_panel(
    ax,
    biomes: gpd.GeoDataFrame,
    brazil: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    title: str,
    label: str,
    edge_color: str,
    marker: str,
    cfg: Config,
    size: Optional[float] = None,
    linewidth: Optional[float] = None,
    hillshade: Optional[np.ndarray] = None,
    hillshade_extent: Optional[list[float]] = None,
):
    size = cfg.station_marker_size if size is None else size
    linewidth = cfg.station_linewidth if linewidth is None else linewidth

    if hillshade is not None and hillshade_extent is not None:
        hs_m = np.ma.masked_invalid(hillshade)
        ax.imshow(hs_m, extent=hillshade_extent, origin="upper", cmap="gray", alpha=0.28, zorder=0)

    plot_biome_background(ax, biomes, cfg)

    if not points.empty:
        ax.scatter(
            points.geometry.x,
            points.geometry.y,
            s=size,
            marker=marker,
            facecolors="none",
            edgecolors=edge_color,
            linewidths=linewidth,
            alpha=0.95,
            zorder=5,
        )

    brazil.boundary.plot(ax=ax, color="black", linewidth=cfg.country_linewidth, zorder=6)

    style_map_axis(ax, brazil)
    add_panel_label(ax, label)
    add_biome_hist_inset(ax, points, biomes)

    ax.set_title(f"{title}\n(n={len(points):,})", pad=3.5, fontweight="regular")


def plot_topography_panel(
    ax,
    dem: np.ndarray,
    extent: list[float],
    biomes: gpd.GeoDataFrame,
    brazil: gpd.GeoDataFrame,
    cfg: Config,
):
    hillshade = compute_hillshade(dem)
    dem_m = np.ma.masked_invalid(dem)
    hs_m = np.ma.masked_invalid(hillshade)

    vmax = np.nanpercentile(dem, 99.5)
    vmax = max(1200, min(2500, float(vmax)))

    ax.imshow(hs_m, extent=extent, origin="upper", cmap="gray", alpha=0.55, zorder=1)

    im = ax.imshow(
        dem_m,
        extent=extent,
        origin="upper",
        cmap=make_topography_cmap(),
        vmin=0,
        vmax=vmax,
        alpha=0.82,
        zorder=2,
    )

    biomes.boundary.plot(ax=ax, color="white", linewidth=cfg.biome_linewidth, alpha=0.70, zorder=5)
    brazil.boundary.plot(ax=ax, color="black", linewidth=cfg.country_linewidth, zorder=6)

    style_map_axis(ax, brazil)
    add_panel_label(ax, "g)")
    ax.set_title("Topography and biome boundaries", pad=3.5, fontweight="regular")

    return im


def plot_product_panel(
    ax,
    arr: np.ndarray,
    extent: list[float],
    biomes: gpd.GeoDataFrame,
    brazil: gpd.GeoDataFrame,
    title: str,
    label: str,
    vmin: float,
    vmax: float,
    mean_val: float,
    std_val: float,
    period_label: str,
    cfg: Config,
):
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad((1, 1, 1, 0))

    arr_m = np.ma.masked_invalid(arr)

    im = ax.imshow(
        arr_m,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    biomes.boundary.plot(ax=ax, color="white", linewidth=0.86, alpha=0.85, zorder=4)
    brazil.boundary.plot(ax=ax, color="black", linewidth=cfg.country_linewidth, zorder=5)

    style_map_axis(ax, brazil)
    add_panel_label(ax, label)
    ax.set_title(title, pad=3.5, fontweight="regular")

    # Top-right: spatial mean and spatial standard deviation.
    ax.text(
        0.35,
        0.10,
        f"μ={mean_val:.1f} mm\nσ={std_val:.1f} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.35", lw=0.55, alpha=0.86),
        zorder=20,
    )

    return im




def draw_horizontal_gradient_colorbar(
    fig,
    axes_for_position: list,
    cmap,
    vmin: float,
    vmax: float,
    label: str,
    labelsize: float = 12,
    major_step: float = 10.0,
    minor_step: float = 5.0,
):
    """
    Draw a horizontal colorbar manually using a gradient image with true
    triangular pointer ends. This renders reliably in PNG, PDF, and SVG.
    """
    positions = [ax.get_position() for ax in axes_for_position]
    x0 = min(p.x0 for p in positions)
    x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions)

    full_width = x1 - x0
    cbar_width = full_width * 0.80
    cbar_height = 0.006
    cbar_pad = 0.022
    cbar_x0 = x0 + 0.5 * (full_width - cbar_width)
    cax = fig.add_axes([cbar_x0, max(0.025, y0 - cbar_pad), cbar_width, cbar_height])

    # Triangular pointer length in data units.
    pointer = 0.045 * (vmax - vmin)
    left_inner = vmin + pointer
    right_inner = vmax - pointer

    gradient = np.linspace(vmin, vmax, 1024, dtype=float)[None, :]
    cax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[left_inner, right_inner, 0, 1],
        interpolation="nearest",
        zorder=1,
    )

    # True pointed ends, not rectangular caps.
    cax.add_patch(
        Polygon(
            [[vmin, 0.5], [left_inner, 0.0], [left_inner, 1.0]],
            closed=True,
            facecolor=cmap(0.0),
            edgecolor="none",
            zorder=2,
        )
    )
    cax.add_patch(
        Polygon(
            [[vmax, 0.5], [right_inner, 0.0], [right_inner, 1.0]],
            closed=True,
            facecolor=cmap(1.0),
            edgecolor="none",
            zorder=2,
        )
    )

    # Polygon outline around the full pointed bar.
    outline_x = [vmin, left_inner, right_inner, vmax, right_inner, left_inner, vmin]
    outline_y = [0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5]
    cax.plot(outline_x, outline_y, color="black", linewidth=0.9, zorder=3)

    cax.set_xlim(vmin, vmax)
    cax.set_ylim(-0.08, 1.08)
    cax.set_yticks([])

    tick_start = np.ceil(vmin / major_step) * major_step
    tick_end = np.floor(vmax / major_step) * major_step
    if tick_end >= tick_start:
        cax.set_xticks(np.arange(tick_start, tick_end + 1e-9, major_step))

    cax.xaxis.set_minor_locator(MultipleLocator(minor_step))
    cax.tick_params(axis="x", which="major", labelsize=11, width=0.9, length=3.2)
    cax.tick_params(axis="x", which="minor", width=0.6, length=1.7)
    cax.set_xlabel(label, fontsize=labelsize, fontweight="regular", labelpad=3.2)

    # Hide rectangular axes box so only the pointed colorbar outline appears.
    for spine in cax.spines.values():
        spine.set_visible(False)

    return cax

def draw_vertical_gradient_colorbar(
    ax_parent,
    mappable,
    label: str,
):
    """
    Draw a vertical colorbar manually using a gradient image.

    The axis is placed beside the parent map panel. This avoids blank colorbar
    gradients in SVG/PDF exports on some systems.
    """
    cax = inset_axes(
        ax_parent,
        width="3.6%",
        height="68%",
        loc="center right",
        bbox_to_anchor=(0.060, 0.0, 1.0, 1.0),
        bbox_transform=ax_parent.transAxes,
        borderpad=0,
    )

    vmin = float(mappable.norm.vmin)
    vmax = float(mappable.norm.vmax)
    cmap = mappable.cmap

    gradient = np.linspace(vmin, vmax, 512, dtype=float)[:, None]
    cax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[0, 1, vmin, vmax],
        interpolation="nearest",
    )

    cax.set_xticks([])
    cax.set_ylim(vmin, vmax)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(label, labelpad=4)

    tick_step = 200
    tick_start = np.ceil(vmin / tick_step) * tick_step
    tick_end = np.floor(vmax / tick_step) * tick_step
    if tick_end > tick_start:
        cax.set_yticks(np.arange(tick_start, tick_end + 1e-9, tick_step))

    cax.tick_params(axis="y", which="major", labelsize=7.4, width=0.85, length=3)

    for spine in cax.spines.values():
        spine.set_linewidth(0.85)

    return cax


# =============================================================================
# BUILD FIGURE
# =============================================================================

def build_figure(cfg: Config):
    setup_style(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading biome polygons...")
    biomes = load_biomes(cfg.biomes_shp)
    brazil = dissolve_country(biomes)

    print("Loading point datasets for panels a-f...")
    ana_subdaily_all = read_point_csv(
        cfg.ana_subdaily_all_csv,
        label="ANA sub-daily stations before QC",
        deduplicate=False,
    )
    ana_subdaily_qc = read_point_csv(
        cfg.ana_subdaily_qc_csv,
        label="ANA sub-daily stations after QC",
        deduplicate=False,
    )

    idf_standard_all = read_point_excel_sheet(
        cfg.idf_curves_all_xlsx,
        sheet_name=cfg.standard_sheet,
        label="Existing IDFs: sub-daily data before QC",
        deduplicate=False,
    )
    idf_disagg_all = read_point_excel_sheet(
        cfg.idf_curves_all_xlsx,
        sheet_name=cfg.disaggregation_sheet,
        label="Existing IDFs: disaggregation before QC",
        deduplicate=False,
    )

    idf_standard_qc = read_point_excel_sheet(
        cfg.idf_curves_filtered_xlsx,
        sheet_name=cfg.standard_sheet,
        label="Existing IDFs: sub-daily data after QC",
        deduplicate=False,
    )
    idf_disagg_qc = read_point_excel_sheet(
        cfg.idf_curves_filtered_xlsx,
        sheet_name=cfg.disaggregation_sheet,
        label="Existing IDFs: disaggregation after QC",
        deduplicate=False,
    )

    print("Clipping all point datasets to Brazil polygon...")
    ana_subdaily_all = clip_points_to_brazil(ana_subdaily_all, brazil, "ANA sub-daily stations before QC")
    ana_subdaily_qc = clip_points_to_brazil(ana_subdaily_qc, brazil, "ANA sub-daily stations after QC")
    idf_standard_all = clip_points_to_brazil(idf_standard_all, brazil, "Existing IDFs sub-daily before QC")
    idf_disagg_all = clip_points_to_brazil(idf_disagg_all, brazil, "Existing IDFs disaggregation before QC")
    idf_standard_qc = clip_points_to_brazil(idf_standard_qc, brazil, "Existing IDFs sub-daily after QC")
    idf_disagg_qc = clip_points_to_brazil(idf_disagg_qc, brazil, "Existing IDFs disaggregation after QC")

    print("Loading topography...")
    dem, dem_profile, dem_extent = crop_dem_to_brazil(cfg.dem_tif, brazil)
    hillshade = compute_hillshade(dem)

    print("Computing mean annual-maximum daily precipitation maps...")
    product_results = {}
    all_product_values = []

    for product_name, folder in cfg.annual_max_products.items():
        year_window = cfg.product_year_windows.get(product_name) if cfg.product_year_windows else None
        print(f"  {product_name}: {folder} | years={year_window}")
        (
            arr,
            profile,
            extent,
            mu,
            sigma,
            n_years,
            period_label,
            year_start,
            year_end,
        ) = compute_mean_annual_max_map(folder, brazil, year_window=year_window)

        product_results[product_name] = {
            "arr": arr,
            "profile": profile,
            "extent": extent,
            "mu": mu,
            "sigma": sigma,
            "n_years": n_years,
            "period": period_label,
            "year_start": year_start,
            "year_end": year_end,
        }
        all_product_values.append(arr[np.isfinite(arr)])

    all_product_values = np.concatenate(all_product_values)

    precip_vmin = cfg.precip_vmin
    precip_vmax = cfg.precip_vmax
    if precip_vmin is None:
        precip_vmin = float(np.nanpercentile(all_product_values, cfg.precip_pmin))
    if precip_vmax is None:
        precip_vmax = float(np.nanpercentile(all_product_values, cfg.precip_pmax))
    precip_vmin = max(0.0, precip_vmin)

    print(f"Shared product color scale: {precip_vmin:.2f} to {precip_vmax:.2f} mm")
    print("Product periods used in panels h-l:")
    for product_name, info in product_results.items():
        print(f"  {product_name}: {info['period']}")

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.6), dpi=cfg.dpi)
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.985,
        bottom=0.095,  # keeps the rainfall colorbar closer to the bottom row maps
        wspace=0.030,
        hspace=0.150,
    )

    # Row 1: untreated station/equation datasets
    plot_station_panel(
        axes[0, 0], biomes, brazil, ana_subdaily_all,
        "ANA sub-daily stations", "a)", "#145A8D", "o", cfg,
        size=cfg.station_marker_size, linewidth=cfg.station_linewidth,
        hillshade=hillshade, hillshade_extent=dem_extent,
    )
    plot_station_panel(
        axes[0, 1], biomes, brazil, idf_standard_all,
        "IDFs with sub-daily data", "b)", "#477B8E", "o", cfg,
        size=cfg.station_marker_size, linewidth=cfg.station_linewidth,
        hillshade=hillshade, hillshade_extent=dem_extent,
    )
    plot_station_panel(
        axes[0, 2], biomes, brazil, idf_disagg_all,
        "IDFs with disaggregation methods", "c)", "#6E7F80", "o", cfg,
        size=cfg.station_marker_size * 0.70, linewidth=cfg.station_linewidth * 0.80,
        hillshade=hillshade, hillshade_extent=dem_extent,
    )

    # Row 2: treated/QC datasets
    plot_station_panel(
        axes[1, 0], biomes, brazil, ana_subdaily_qc,
        "ANA sub-daily stations after QC", "d)", "#6C757D", "^", cfg,
        size=cfg.station_marker_size, linewidth=cfg.station_linewidth,
        hillshade=hillshade, hillshade_extent=dem_extent,
    )
    plot_station_panel(
        axes[1, 1], biomes, brazil, idf_standard_qc,
        "IDFs with sub-daily data after QC", "e)", "#8D6E63", "^", cfg,
        size=cfg.station_marker_size, linewidth=cfg.station_linewidth,
        hillshade=hillshade, hillshade_extent=dem_extent,
    )
    plot_station_panel(
        axes[1, 2], biomes, brazil, idf_disagg_qc,
        "IDFs with disaggregation methods after QC", "f)", "#7F2D26", "^", cfg,
        size=cfg.station_marker_size, linewidth=cfg.station_linewidth,
        hillshade=hillshade, hillshade_extent=dem_extent,
    )

    # Row 3
    topo_im = plot_topography_panel(axes[2, 0], dem, dem_extent, biomes, brazil, cfg)

    br = product_results["BR-DWGD"]
    plot_product_panel(
        axes[2, 1], br["arr"], br["extent"], biomes, brazil,
        "BR-DWGD mean annual-max", "h)", precip_vmin, precip_vmax,
        br["mu"], br["sigma"], br["period"], cfg,
    )

    ch = product_results["CHIRPS"]
    plot_product_panel(
        axes[2, 2], ch["arr"], ch["extent"], biomes, brazil,
        "CHIRPS mean annual-max", "i)", precip_vmin, precip_vmax,
        ch["mu"], ch["sigma"], ch["period"], cfg,
    )

    # Row 4
    iv6 = product_results["IMERG V06"]
    plot_product_panel(
        axes[3, 0], iv6["arr"], iv6["extent"], biomes, brazil,
        "IMERG V06 mean annual-max", "j)", precip_vmin, precip_vmax,
        iv6["mu"], iv6["sigma"], iv6["period"], cfg,
    )

    iv7 = product_results["IMERG V07"]
    plot_product_panel(
        axes[3, 1], iv7["arr"], iv7["extent"], biomes, brazil,
        "IMERG V07 mean annual-max", "k)", precip_vmin, precip_vmax,
        iv7["mu"], iv7["sigma"], iv7["period"], cfg,
    )

    pe = product_results["PERSIANN-CDR"]
    plot_product_panel(
        axes[3, 2], pe["arr"], pe["extent"], biomes, brazil,
        "PERSIANN-CDR mean annual-max", "l)", precip_vmin, precip_vmax,
        pe["mu"], pe["sigma"], pe["period"], cfg,
    )

    # -------------------------------------------------------------------------
    # Vertical elevation colorbar beside topography panel.
    # Manual gradient is used so the bar renders correctly in PDF/SVG.
    # -------------------------------------------------------------------------
    draw_vertical_gradient_colorbar(
        ax_parent=axes[2, 0],
        mappable=topo_im,
        label="Elevation (m)",
    )

    # -------------------------------------------------------------------------
    # Shared precipitation colorbar for panels h-l.
    # Manual gradient is used so the bar renders correctly in PDF/SVG.
    # -------------------------------------------------------------------------
    product_cmap = plt.get_cmap("inferno").copy()
    draw_horizontal_gradient_colorbar(
        fig=fig,
        axes_for_position=[axes[3, 0], axes[3, 1], axes[3, 2]],
        cmap=product_cmap,
        vmin=precip_vmin,
        vmax=precip_vmax,
        label="Mean annual-maximum daily precipitation (mm/day)",
        major_step=10.0,
        minor_step=5.0,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    for ext in ["png", "pdf", "svg"]:
        out = cfg.out_dir / f"{cfg.out_name}.{ext}"
        fig.savefig(out, dpi=cfg.dpi if ext == "png" else None)
        print(f"Saved: {out}")

    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    build_figure(CFG)


if __name__ == "__main__":
    main()
