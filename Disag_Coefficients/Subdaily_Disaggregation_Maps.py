#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRIDF - Publication-quality maps of subdaily disaggregation coefficients.

This script plots the subdaily rainfall disaggregation coefficient rasters
using a clean publication-style multi-panel layout.

Main characteristics:

    - Subdaily coefficient rasters only
    - Coefficients ordered from largest to shortest duration
    - White figure background
    - No grey terrain/hillshade background by default
    - No state boundaries
    - Biome polygons used as contextual background
    - Biome boundaries shown clearly
    - Brazil national boundary shown
    - User-selected coefficient colormap
    - Discrete color intervals from 0.00 to 1.20 every 0.05
    - Upper colorbar pointer for values > 1.20
    - Thin colorbar with thicker black borders
    - Mean and standard deviation label in each subplot using LaTeX:
        x-bar and sigma

Inputs:

    Subdaily coefficient rasters:
        /Users/mngomes/Documents/GitHub/GRIDF/Disag_Coefficients/relative_to_subdaily

    DEM:
        /Users/mngomes/Documents/GitHub/GRIDF/Misc/DEM.tif

    Brazil boundary:
        /Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/ADMLevels/bra_admbnda_adm0_ibge_2020.shp

    Biomes:
        /Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes

Outputs:

    /Users/mngomes/Documents/GitHub/GRIDF/Figures/Subdaily_Disaggregation_Maps/
        GRIDF_subdaily_coefficients_multipanel.png
        GRIDF_subdaily_coefficients_multipanel.pdf
        GRIDF_subdaily_coefficients_multipanel.svg
"""

from pathlib import Path
import re
import math
import unicodedata
import warnings
from matplotlib import font_manager

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable


# ============================================================
# 1. USER SETTINGS
# ============================================================

ROOT_DIR = Path("/Users/mngomes/Documents/GitHub/GRIDF")

SUBDAILY_DIR = ROOT_DIR / "Disag_Coefficients" / "relative_to_subdaily"

DEM_PATH = ROOT_DIR / "Misc" / "DEM.tif"

ADM_DIR = ROOT_DIR / "BrazilShapefiles" / "ADMLevels"
BIOME_DIR = ROOT_DIR / "BrazilShapefiles" / "Biomes"

BRAZIL_SHP = ADM_DIR / "bra_admbnda_adm0_ibge_2020.shp"

# If this exact file does not exist, the script automatically detects
# the first biome/bioma shapefile inside BIOME_DIR.
BIOMES_SHP = BIOME_DIR / "biomes.shp"

OUTPUT_DIR = ROOT_DIR / "Figures" / "Subdaily_Disaggregation_Maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Font settings
# ------------------------------------------------------------
USE_CUSTOM_FONT = True

# Local font file. Replace with your own font path if needed.
HELVETICA_FONT_PATH = "/Users/mngomes/Downloads/AvenirNextCyr-Regular.ttf"

# Used only when USE_CUSTOM_FONT = False, or as a fallback label.
FONT_FAMILY_NAME = "Helvetica"

# ------------------------------------------------------------
# Font-size settings
# ------------------------------------------------------------
# Change these values to control every text size in the figure.
GLOBAL_FONT_SIZE = 14
MAIN_TITLE_FONT_SIZE = 17
SUBPLOT_TITLE_FONT_SIZE = 17
STAT_LABEL_FONT_SIZE = 14
COLORBAR_LABEL_FONT_SIZE = 12
COLORBAR_TICK_FONT_SIZE = 12

# Optional related text controls.
MAIN_TITLE_FONT_WEIGHT = "bold"
SUBPLOT_TITLE_FONT_WEIGHT = "bold"
STAT_LABEL_FONT_WEIGHT = "bold"
COLORBAR_LABEL_FONT_WEIGHT = "bold"

# ------------------------------------------------------------
# Raster order
# ------------------------------------------------------------
ORDER_LARGEST_TO_SHORTEST = True

# ------------------------------------------------------------
# Coefficient raster style
# ------------------------------------------------------------
COEFF_CMAP_NAME = "gist_ncar"

COEFF_MIN = 0.0
COEFF_MAX = 1.2
COEFF_STEP = 0.05

COEFF_ALPHA = 0.82

# Keep False so values above 1.2 are shown using the colorbar extension.
CLIP_COEFFICIENTS_TO_COLORBAR = False

# ------------------------------------------------------------
# Terrain hillshade style
# ------------------------------------------------------------
# Keep True to show terrain shading behind the rasters.
SHOW_HILLSHADE = True

# Clip hillshade to the Brazil/domain polygon so the area outside stays white.
CLIP_HILLSHADE_TO_DOMAIN = True

# Used only when SHOW_HILLSHADE = True.
HILLSHADE_ALPHA = 0.75

HILLSHADE_Z_FACTOR = 1.5
HILLSHADE_AZIMUTH = 315
HILLSHADE_ALTITUDE = 45

# Higher VMIN makes the hillshade lighter.
HILLSHADE_VMIN = 0.18
HILLSHADE_VMAX = 1.00

# ------------------------------------------------------------
# Biome style
# ------------------------------------------------------------
SHOW_BIOME_FILLS = True
SHOW_BIOME_BOUNDARIES = True
SHOW_BRAZIL_BOUNDARY = True

BIOME_FILL_ALPHA = 0.34
BIOME_BOUNDARY_ALPHA = 0.92
BRAZIL_BOUNDARY_ALPHA = 0.98

BIOME_BOUNDARY_LINEWIDTH = 0.90
BRAZIL_BOUNDARY_LINEWIDTH = 1.10

BIOME_CMAP_NAME = "Set3"

# ------------------------------------------------------------
# Statistic label style
# ------------------------------------------------------------
SHOW_STAT_LABEL = True
STAT_LABEL_DECIMALS = 2
STAT_LABEL_BOX_ALPHA = 0.84

# ------------------------------------------------------------
# Raster plotting resolution
# ------------------------------------------------------------
MAX_PLOT_DIM_COEFF = 1500
MAX_PLOT_DIM_DEM = 1800

# ------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------
N_COLS = 4
FIG_WIDTH_IN = 14.0
FIG_HEIGHT_IN = 10.0
DPI = 350

FIGURE_FACE_COLOR = "white"
AXES_FACE_COLOR = "white"

TITLE = "Subdaily Rainfall Disaggregation Coefficients"


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def clean_text(text) -> str:
    """
    Convert text to ASCII-safe format.
    """

    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("ASCII")
    text = re.sub(r"[^A-Za-z0-9 _\-/\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def setup_font():
    """
    Configure matplotlib font family and global font sizes.
    """

    font_name = FONT_FAMILY_NAME

    if USE_CUSTOM_FONT:
        font_path = Path(HELVETICA_FONT_PATH)

        if not font_path.exists():
            raise FileNotFoundError(f"Custom font file not found:\n{font_path}")

        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["font.size"] = GLOBAL_FONT_SIZE

    # Math text uses the same font as the rest of the figure.
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = font_name
    plt.rcParams["mathtext.it"] = font_name + ":italic"
    plt.rcParams["mathtext.bf"] = font_name + ":bold"

    print(f"Using font: {font_name}")
    print("Font sizes:")
    print(f"  Global:         {GLOBAL_FONT_SIZE}")
    print(f"  Main title:     {MAIN_TITLE_FONT_SIZE}")
    print(f"  Subplot titles: {SUBPLOT_TITLE_FONT_SIZE}")
    print(f"  Statistics:     {STAT_LABEL_FONT_SIZE}")
    print(f"  Colorbar label: {COLORBAR_LABEL_FONT_SIZE}")
    print(f"  Colorbar ticks: {COLORBAR_TICK_FONT_SIZE}")

def check_path(path: Path, label: str):
    """
    Check whether a file/folder exists.
    """

    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def find_biome_shapefile() -> Path:
    """
    Automatically detect the biome shapefile.
    """

    if BIOMES_SHP.exists():
        return BIOMES_SHP

    shp_files = sorted(BIOME_DIR.glob("*.shp"))

    if not shp_files:
        raise FileNotFoundError(f"No biome shapefile found in:\n{BIOME_DIR}")

    preferred = [
        shp for shp in shp_files
        if "biome" in shp.name.lower() or "bioma" in shp.name.lower()
    ]

    if preferred:
        return preferred[0]

    return shp_files[0]


def unique_preserve_order(values):
    """
    Return unique values while preserving their original order.
    """

    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


# ============================================================
# 3. RASTER NAME PARSING AND SORTING
# ============================================================

def get_coefficient_name(raster_path: Path) -> str:
    """
    Convert raster filename to clean coefficient name.

    Examples:
        IDW_R_24h_1dia_res0.100_k10_p2.0.tif -> R_24h_1dia
        IDW_R_5m_30m_res0.100_k10_p2.0.tif  -> R_5m_30m
    """

    name = raster_path.stem
    name = re.sub(r"^IDW_", "", name)
    name = re.sub(r"_res.*$", "", name)
    name = clean_text(name).replace(" ", "_")

    return name


def duration_to_minutes(duration: str) -> float:
    """
    Convert duration strings to minutes.

    Examples:
        5m    -> 5
        30m   -> 30
        1h    -> 60
        24h   -> 1440
        1dia  -> 1440
    """

    duration = str(duration).lower().strip()
    duration = duration.replace("day", "dia")

    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)(m|h|dia)$", duration)

    if match is None:
        return -1.0

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "m":
        return value

    if unit == "h":
        return value * 60.0

    if unit == "dia":
        return value * 1440.0

    return -1.0


def coefficient_duration_minutes(raster_path: Path) -> float:
    """
    Extract the numerator duration from a subdaily coefficient raster.
    """

    coeff = get_coefficient_name(raster_path)
    parts = coeff.split("_")

    if len(parts) >= 2 and parts[0] == "R":
        return duration_to_minutes(parts[1])

    return -1.0


def list_subdaily_rasters() -> list[Path]:
    """
    List subdaily coefficient rasters and sort them by duration.
    """

    check_path(SUBDAILY_DIR, "Subdaily coefficient folder")

    rasters = sorted(SUBDAILY_DIR.glob("*.tif"))

    if not rasters:
        raise FileNotFoundError(f"No .tif files found in:\n{SUBDAILY_DIR}")

    rasters = sorted(
        rasters,
        key=coefficient_duration_minutes,
        reverse=ORDER_LARGEST_TO_SHORTEST,
    )

    return rasters


def format_duration_label(duration: str) -> str:
    """
    Format duration labels for plot titles.
    """

    duration = str(duration)

    if duration.endswith("m"):
        return duration.replace("m", "min")

    if duration.endswith("dia"):
        return duration.replace("dia", "day")

    return duration


def coefficient_plot_title(raster_path: Path) -> str:
    """
    Create clean subplot title.
    """

    coeff = get_coefficient_name(raster_path)
    parts = coeff.split("_")

    if len(parts) >= 3 and parts[0] == "R":
        numerator = format_duration_label(parts[1])
        denominator = format_duration_label(parts[2])
        return f"{numerator}/{denominator}"

    return coeff.replace("_", "/")


# ============================================================
# 4. RASTER READING AND HILLSHADE
# ============================================================

def raster_extent(bounds):
    """
    Convert rasterio bounds to matplotlib extent.
    """

    return (bounds.left, bounds.right, bounds.bottom, bounds.top)


def mask_array_to_geometries(array, extent, geometries):
    """
    Mask a raster array outside a vector geometry domain.

    This is used to keep the DEM hillshade visible only inside Brazil
    so the rectangular DEM extent does not create a grey background
    outside the map domain.
    """

    if array is None or extent is None:
        return array

    if np.ma.isMaskedArray(array):
        source = array.filled(np.nan)
        existing_mask = np.ma.getmaskarray(array) | ~np.isfinite(source)
    else:
        source = np.asarray(array, dtype=float)
        existing_mask = ~np.isfinite(source)

    height, width = source.shape
    left, right, bottom, top = extent

    transform = from_bounds(
        left,
        bottom,
        right,
        top,
        width,
        height,
    )

    valid_geometries = [
        geometry
        for geometry in geometries
        if geometry is not None and not geometry.is_empty
    ]

    if not valid_geometries:
        return np.ma.array(source, mask=np.ones_like(source, dtype=bool))

    inside_domain = geometry_mask(
        valid_geometries,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=True,
    )

    final_mask = existing_mask | ~inside_domain

    return np.ma.array(source, mask=final_mask)


def read_raster_for_plot(
    raster_path: Path,
    target_crs=None,
    max_dim: int = 1500,
    masked: bool = True,
    resampling=Resampling.bilinear,
):
    """
    Read and optionally reproject/downsample raster for plotting.
    """

    with rasterio.open(raster_path) as src:

        if target_crs is not None and src.crs is not None and src.crs != target_crs:
            dataset = WarpedVRT(
                src,
                crs=target_crs,
                resampling=resampling,
            )
        else:
            dataset = src

        height = dataset.height
        width = dataset.width

        scale = max(height, width) / float(max_dim)

        if scale < 1:
            out_height = height
            out_width = width
        else:
            out_height = int(round(height / scale))
            out_width = int(round(width / scale))

        data = dataset.read(
            1,
            out_shape=(out_height, out_width),
            masked=masked,
            resampling=resampling,
        )

        extent = raster_extent(dataset.bounds)
        crs = dataset.crs

        if isinstance(dataset, WarpedVRT):
            dataset.close()

    return data, extent, crs


def compute_hillshade(
    dem_array,
    azimuth: float = 315,
    altitude: float = 45,
    z_factor: float = 1.0,
):
    """
    Compute analytical hillshade from DEM.
    """

    if np.ma.isMaskedArray(dem_array):
        arr = dem_array.filled(np.nan).astype(float)
    else:
        arr = dem_array.astype(float)

    if np.isnan(arr).all():
        return np.zeros_like(arr, dtype=float)

    median_value = np.nanmedian(arr)
    arr = np.where(np.isfinite(arr), arr, median_value)

    dy, dx = np.gradient(arr * z_factor)

    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)

    shaded = (
        np.sin(altitude_rad) * np.sin(slope)
        + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    )

    shaded = (shaded - np.nanmin(shaded)) / (
        np.nanmax(shaded) - np.nanmin(shaded) + 1e-12
    )

    return shaded


def prepare_coefficient_for_display(data):
    """
    Mask invalid values and optionally clip coefficient values.
    """

    if np.ma.isMaskedArray(data):
        arr = data.astype(float)
    else:
        arr = np.ma.masked_invalid(data.astype(float))

    arr = np.ma.masked_where(~np.isfinite(arr), arr)

    if CLIP_COEFFICIENTS_TO_COLORBAR:
        arr = np.ma.clip(arr, COEFF_MIN, COEFF_MAX)

    return arr


def compute_raster_mean_std(data):
    """
    Compute spatial mean and standard deviation from valid pixels only.
    """

    if np.ma.isMaskedArray(data):
        valid_values = data.compressed().astype(float)
    else:
        arr = np.asarray(data, dtype=float)
        valid_values = arr[np.isfinite(arr)]

    if valid_values.size == 0:
        return np.nan, np.nan

    mean_value = float(np.nanmean(valid_values))
    std_value = float(np.nanstd(valid_values))

    return mean_value, std_value


# ============================================================
# 5. VECTOR LAYERS
# ============================================================

def read_vector(path: Path, fallback_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Read vector data and fix missing CRS/invalid geometries.
    """

    check_path(path, "Vector file")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"Vector file is empty:\n{path}")

    if gdf.crs is None:
        print(f"WARNING: Vector has no CRS. Assigning {fallback_crs}:")
        print(f"  {path}")
        gdf = gdf.set_crs(fallback_crs, allow_override=True)

    gdf = gdf[gdf.geometry.notnull()].copy()

    invalid_count = (~gdf.geometry.is_valid).sum()

    if invalid_count > 0:
        print(f"Fixing {invalid_count} invalid geometries in:")
        print(f"  {path}")
        gdf["geometry"] = gdf.geometry.buffer(0)

    return gdf


def detect_biome_name_column(gdf: gpd.GeoDataFrame) -> str:
    """
    Detect the most likely biome name field.
    """

    candidates = [
        "Bioma",
        "BIOMA",
        "bioma",
        "NOME",
        "Nome",
        "nome",
        "NM_BIOMA",
        "NM_BIOM",
        "CD_Bioma",
        "CD_BIOMA",
        "name",
        "NAME",
    ]

    for col in candidates:
        if col in gdf.columns:
            return col

    object_cols = [
        col for col in gdf.columns
        if col != "geometry" and gdf[col].dtype == "object"
    ]

    if object_cols:
        warnings.warn(
            f"No standard biome name field found. Using: {object_cols[0]}"
        )
        return object_cols[0]

    warnings.warn("No biome name field found. Using polygon index.")
    return "__index__"


def load_context_layers(target_crs):
    """
    Load Brazil boundary and biome polygons.
    """

    brazil = read_vector(BRAZIL_SHP)
    biomes = read_vector(find_biome_shapefile())

    if target_crs is not None:
        brazil = brazil.to_crs(target_crs)
        biomes = biomes.to_crs(target_crs)

    biome_name_col = detect_biome_name_column(biomes)

    if biome_name_col == "__index__":
        biomes["biome_name"] = [f"Biome_{i + 1}" for i in range(len(biomes))]
    else:
        biomes["biome_name"] = biomes[biome_name_col].astype(str).apply(clean_text)

    return brazil, biomes


# ============================================================
# 6. COLOR MAP AND COLORBAR
# ============================================================

def make_discrete_colormap():
    """
    Build discrete colormap from 0 to 1.2 every 0.05.
    """

    bounds = np.arange(
        COEFF_MIN,
        COEFF_MAX + COEFF_STEP * 0.5,
        COEFF_STEP,
    )

    cmap = plt.get_cmap(COEFF_CMAP_NAME, len(bounds) - 1).copy()

    # Keep nodata/masked raster pixels transparent so the background stays clean.
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    norm = BoundaryNorm(bounds, cmap.N, clip=False)

    return cmap, norm, bounds


def add_shared_colorbar(fig, axes, cmap, norm, bounds):
    """
    Add very thin shared colorbar with thicker borders.

    Bins are every 0.05. Tick labels are shown every 0.10 to avoid clutter.
    """

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    tick_values = np.arange(COEFF_MIN, COEFF_MAX + 0.001, 0.10)

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.010,
        pad=0.010,
        ticks=tick_values,
        shrink=0.80,
        aspect=60,
        extend="max",
    )

    cbar.set_label(
        "Ratio",
        fontsize=COLORBAR_LABEL_FONT_SIZE,
        fontweight=COLORBAR_LABEL_FONT_WEIGHT,
        labelpad=8,
    )

    cbar.ax.tick_params(
        labelsize=COLORBAR_TICK_FONT_SIZE,
        width=1.2,
        length=3.5,
    )

    cbar.outline.set_linewidth(1.4)
    cbar.outline.set_edgecolor("black")

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_edgecolor("black")

    return cbar


# ============================================================
# 7. MAP PLOTTING HELPERS
# ============================================================

def plot_biomes(ax, biomes):
    """
    Plot biome fills using categorical colors.
    """

    if not SHOW_BIOME_FILLS:
        return

    unique_biomes = unique_preserve_order(biomes["biome_name"].tolist())
    n_biomes = len(unique_biomes)

    biome_cmap = plt.get_cmap(BIOME_CMAP_NAME, max(n_biomes, 3))

    color_map = {
        biome: biome_cmap(i % biome_cmap.N)
        for i, biome in enumerate(unique_biomes)
    }

    for biome_name in unique_biomes:
        subset = biomes[biomes["biome_name"] == biome_name]

        subset.plot(
            ax=ax,
            facecolor=color_map[biome_name],
            edgecolor="none",
            alpha=BIOME_FILL_ALPHA,
            zorder=1,
        )


def plot_context(ax, brazil, biomes, hillshade=None, hillshade_extent=None):
    """
    Plot hillshade, biomes, and Brazil boundary.

    No state boundaries.
    """

    ax.set_facecolor(AXES_FACE_COLOR)

    if SHOW_HILLSHADE and hillshade is not None and hillshade_extent is not None:
        hillshade_cmap = plt.get_cmap("gray").copy()
        hillshade_cmap.set_bad((1, 1, 1, 0))

        ax.imshow(
            hillshade,
            extent=hillshade_extent,
            cmap=hillshade_cmap,
            vmin=HILLSHADE_VMIN,
            vmax=HILLSHADE_VMAX,
            alpha=HILLSHADE_ALPHA,
            interpolation="bilinear",
            zorder=0,
        )

    plot_biomes(ax, biomes)

    if SHOW_BIOME_BOUNDARIES:
        biomes.boundary.plot(
            ax=ax,
            linewidth=BIOME_BOUNDARY_LINEWIDTH,
            color="#252525",
            alpha=BIOME_BOUNDARY_ALPHA,
            zorder=4,
        )

    if SHOW_BRAZIL_BOUNDARY:
        brazil.boundary.plot(
            ax=ax,
            linewidth=BRAZIL_BOUNDARY_LINEWIDTH,
            color="black",
            alpha=BRAZIL_BOUNDARY_ALPHA,
            zorder=5,
        )


def set_clean_map_axis(ax, brazil):
    """
    Remove axes and zoom to Brazil.
    """

    minx, miny, maxx, maxy = brazil.total_bounds

    pad_x = (maxx - minx) * 0.035
    pad_y = (maxy - miny) * 0.035

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal", adjustable="box")


def add_stat_label(ax, mean_value, std_value):
    """
    Add mean and standard deviation label in the top-right corner
    using LaTeX notation.

    Mean:
        \bar{x}

    Standard deviation:
        \sigma
    """

    if not SHOW_STAT_LABEL:
        return

    if np.isnan(mean_value) or np.isnan(std_value):
        label = r"$\mu$ = NA" + "\n" + r"$\sigma$ = NA"
    else:
        label = (
                rf"$\mu$ = {mean_value:.{STAT_LABEL_DECIMALS}f}"
                + "\n"
                + rf"$\sigma$ = {std_value:.{STAT_LABEL_DECIMALS}f}"
        )

    ax.text(
        0.965,
        0.955,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=STAT_LABEL_FONT_SIZE,
        fontweight=STAT_LABEL_FONT_WEIGHT,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor="black",
            linewidth=0.65,
            alpha=STAT_LABEL_BOX_ALPHA,
        ),
        zorder=20,
    )


# ============================================================
# 8. MAIN FIGURE
# ============================================================

def make_multipanel_map():
    """
    Make and save multi-panel subdaily coefficient map.
    """

    rasters = list_subdaily_rasters()

    print("\nSubdaily rasters in plotting order:")
    for raster in rasters:
        print(f"  {coefficient_plot_title(raster):>12s}  |  {raster.name}")

    with rasterio.open(rasters[0]) as src:
        target_crs = src.crs

    if target_crs is None:
        print("WARNING: First coefficient raster has no CRS. Using EPSG:4326.")
        target_crs = "EPSG:4326"

    brazil, biomes = load_context_layers(target_crs)

    # --------------------------------------------------------
    # Read and prepare hillshade
    # --------------------------------------------------------
    hillshade = None
    hillshade_extent = None

    if SHOW_HILLSHADE:
        check_path(DEM_PATH, "DEM raster")

        dem_data, hillshade_extent, _ = read_raster_for_plot(
            DEM_PATH,
            target_crs=target_crs,
            max_dim=MAX_PLOT_DIM_DEM,
            masked=True,
            resampling=Resampling.bilinear,
        )

        hillshade = compute_hillshade(
            dem_data,
            azimuth=HILLSHADE_AZIMUTH,
            altitude=HILLSHADE_ALTITUDE,
            z_factor=HILLSHADE_Z_FACTOR,
        )

        if CLIP_HILLSHADE_TO_DOMAIN:
            hillshade = mask_array_to_geometries(
                hillshade,
                hillshade_extent,
                brazil.geometry,
            )

    cmap, norm, bounds = make_discrete_colormap()

    n_maps = len(rasters)
    n_cols = N_COLS
    n_rows = int(math.ceil(n_maps / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=False,
        facecolor=FIGURE_FACE_COLOR,
    )

    axes = np.array(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):

        ax.set_facecolor(AXES_FACE_COLOR)

        if ax_idx >= n_maps:
            ax.axis("off")
            continue

        raster_path = rasters[ax_idx]

        coeff_data_raw, coeff_extent, _ = read_raster_for_plot(
            raster_path,
            target_crs=target_crs,
            max_dim=MAX_PLOT_DIM_COEFF,
            masked=True,
            resampling=Resampling.bilinear,
        )

        coeff_data = prepare_coefficient_for_display(coeff_data_raw)

        mean_value, std_value = compute_raster_mean_std(coeff_data)

        plot_context(
            ax=ax,
            brazil=brazil,
            biomes=biomes,
            hillshade=hillshade,
            hillshade_extent=hillshade_extent,
        )

        ax.imshow(
            coeff_data,
            extent=coeff_extent,
            cmap=cmap,
            norm=norm,
            alpha=COEFF_ALPHA,
            interpolation="nearest",
            zorder=3,
        )

        # Replot biome and Brazil boundaries above coefficient raster.
        if SHOW_BIOME_BOUNDARIES:
            biomes.boundary.plot(
                ax=ax,
                linewidth=BIOME_BOUNDARY_LINEWIDTH,
                color="#202020",
                alpha=BIOME_BOUNDARY_ALPHA,
                zorder=6,
            )

        if SHOW_BRAZIL_BOUNDARY:
            brazil.boundary.plot(
                ax=ax,
                linewidth=BRAZIL_BOUNDARY_LINEWIDTH,
                color="black",
                alpha=BRAZIL_BOUNDARY_ALPHA,
                zorder=7,
            )

        set_clean_map_axis(ax, brazil)

        title = coefficient_plot_title(raster_path)

        ax.set_title(
            title,
            fontsize=SUBPLOT_TITLE_FONT_SIZE,
            fontweight=SUBPLOT_TITLE_FONT_WEIGHT,
            pad=5,
        )

        add_stat_label(ax, mean_value, std_value)

    fig.suptitle(
        TITLE,
        fontsize=MAIN_TITLE_FONT_SIZE,
        fontweight=MAIN_TITLE_FONT_WEIGHT,
        y=0.965,
    )

    add_shared_colorbar(fig, axes[:n_maps], cmap, norm, bounds)

    plt.subplots_adjust(
        left=0.025,
        right=0.915,
        bottom=0.035,
        top=0.925,
        wspace=0.040,
        hspace=0.105,
    )

    png_path = OUTPUT_DIR / "GRIDF_subdaily_coefficients_multipanel.png"
    pdf_path = OUTPUT_DIR / "GRIDF_subdaily_coefficients_multipanel.pdf"
    svg_path = OUTPUT_DIR / "GRIDF_subdaily_coefficients_multipanel.svg"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=FIGURE_FACE_COLOR,
    )

    fig.savefig(
        pdf_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=FIGURE_FACE_COLOR,
    )

    fig.savefig(
        svg_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=FIGURE_FACE_COLOR,
    )

    plt.close(fig)

    print("\nSaved multi-panel maps:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {svg_path}")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    setup_font()

    print("=" * 80)
    print("GRIDF SUBDAILY DISAGGREGATION COEFFICIENT MAPS")
    print("=" * 80)

    check_path(SUBDAILY_DIR, "Subdaily coefficient folder")

    # DEM is required only when terrain hillshade is enabled.
    if SHOW_HILLSHADE:
        check_path(DEM_PATH, "DEM raster")

    check_path(BRAZIL_SHP, "Brazil boundary shapefile")
    check_path(BIOME_DIR, "Biome folder")

    make_multipanel_map()

    print("\n" + "=" * 80)
    print("FINISHED SUCCESSFULLY")
    print("=" * 80)

    print("\nOutput folder:")
    print(f"  {OUTPUT_DIR}")

    print("\nNotes:")
    if SHOW_HILLSHADE:
        print("  - Terrain hillshade is included.")
    else:
        print("  - Terrain hillshade is disabled, so there is no grey background.")
    print("  - Background is white.")
    print("  - Only biome polygons and biome boundaries are shown as context.")
    print("  - No state boundaries are plotted.")
    print(f"  - Coefficient color scale is {COEFF_CMAP_NAME}.")
    print("  - Color bins are discretized from 0 to 1.2 every 0.05.")
    print("  - The colorbar uses an upper pointer for values above 1.2.")
    print("  - The colorbar is thinner and has thicker borders.")
    print("  - Each subplot includes LaTeX labels for mean and standard deviation:")
    print(r"      $\bar{x}$ and $\sigma$")
    print("  - Subdaily rasters are ordered from largest to shortest duration.")


if __name__ == "__main__":
    main()