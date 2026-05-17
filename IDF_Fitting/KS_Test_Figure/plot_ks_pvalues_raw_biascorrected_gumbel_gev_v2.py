#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot KS goodness-of-fit maps for GRIDF IDF pipeline outputs.

Updates in this version
-----------------------
- Values with KS p-value < 0.05 are shown in strong red.
- Values with KS p-value >= 0.05 are shown with a continuous grayscale ramp.
- Each map includes a small inset histogram showing:

      rejected-pixel area / biome area × 100

  for each Brazilian biome.
- The histogram y-axis label was removed.
- The histogram label is now the inset title: "Rejected area (%)".
- Biome histogram colors follow the same color logic used in the study-area code.
- The rejection percentage label prints "%" correctly, without "\%".
- The global figure title was removed.

Rows:
  1) RAW / Gumbel
  2) Bias-corrected / Gumbel
  3) RAW / GEV
  4) Bias-corrected / GEV

Columns:
  BR-DWGD, CHIRPS, IMERG V06, IMERG V07, PERSIANN-CDR
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter, MaxNLocator

from pyproj import Geod


# =============================================================================
# USER SETTINGS
# =============================================================================

@dataclass
class Config:
    # IDF pipeline output root
    idf_output_root: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs"
    )

    # Biome shapefile, used for Brazil boundary and biome boundaries
    biomes_shp: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes/Brazil_biomes.shp"
    )

    # Output folder
    out_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Figures"
    )
    out_name: str = "Figure_KS_pvalues_raw_biascorrected_Gumbel_GEV"

    # Product folders as used by the pipeline
    products: tuple[str, ...] = (
        "br_dwgd",
        "chirps",
        "imerg_v06",
        "imerg_v07",
        "persiann_cdr",
    )

    product_labels: dict[str, str] = None

    # States as used by the pipeline
    states: tuple[str, ...] = (
        "raw",
        "bias_corrected_mean",
    )

    state_labels: dict[str, str] = None

    # Distributions as used by the pipeline
    distributions: tuple[str, ...] = (
        "GUMBEL",
        "GEV",
    )

    distribution_labels: dict[str, str] = None

    # Figure style
    font_family: str = "Avenir Next"
    fallback_font: str = "DejaVu Sans"
    dpi: int = 600

    # KS p-value display
    p_threshold: float = 0.05

    # Strong red for rejected pixels, where p < 0.05.
    # This avoids confusion with the biome histogram colors.
    rejection_color: str = "#b2182b"

    # Map boundary styling
    biome_linewidth: float = 0.30
    country_linewidth: float = 0.75

    # Histogram inset position inside each map axis:
    # [left, bottom, width, height] in axis fraction.
    hist_inset_axes: tuple[float, float, float, float] = (0.005, 0.100, 0.330, 0.245)

    # Histogram style
    hist_face_alpha: float = 0.84
    hist_bar_alpha: float = 0.92

    # Optional: if the search picks the wrong raster, put explicit paths here.
    # Keys must be (state, product, distribution), e.g.
    # ("raw", "br_dwgd", "GUMBEL"): Path("/path/to/pvalue.tif")
    explicit_paths: dict[tuple[str, str, str], Path] = None

    def __post_init__(self):
        if self.product_labels is None:
            self.product_labels = {
                "br_dwgd": "BR-DWGD",
                "chirps": "CHIRPS",
                "imerg_v06": "IMERG V06",
                "imerg_v07": "IMERG V07",
                "persiann_cdr": "PERSIANN-CDR",
            }

        if self.state_labels is None:
            self.state_labels = {
                "raw": "Raw",
                "bias_corrected_mean": "Bias-corrected",
            }

        if self.distribution_labels is None:
            self.distribution_labels = {
                "GUMBEL": "Gumbel",
                "GEV": "GEV",
            }

        if self.explicit_paths is None:
            self.explicit_paths = {}


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
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.titlesize": 13.0,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# =============================================================================
# BIOME HELPERS
# =============================================================================

def find_first_column(columns, candidates) -> Optional[str]:
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


def find_biome_name_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    return find_first_column(
        gdf.columns,
        [
            "Bioma",
            "BIOMA",
            "bioma",
            "Biome",
            "BIOME",
            "biome",
            "Name",
            "name",
            "NOME",
            "nome",
        ],
    )


def clean_biome_label(name) -> str:
    """Clean biome names for display."""
    s = str(name)
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def get_biome_color_map(biomes: gpd.GeoDataFrame) -> dict:
    """
    Assign distinguishable, high-contrast scientific colors to each biome.

    These colors match the style used in the study-area figure code.
    """
    biome_col = find_biome_name_column(biomes)

    if biome_col is None:
        return {}

    names = [
        clean_biome_label(x)
        for x in biomes[biome_col].dropna().unique()
    ]

    preferred_order = [
        "Amazônia",
        "Amazonia",
        "Amazon",
        "Amazônia Legal",
        "Caatinga",
        "Cerrado",
        "Mata Atlântica",
        "Mata Atlantica",
        "Atlantic Forest",
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

    return {
        name: palette[i % len(palette)]
        for i, name in enumerate(ordered)
    }


def short_biome_label(label: str) -> str:
    """
    Short labels keep the tiny inset histograms readable.
    """
    s = clean_biome_label(label)

    replacements = {
        "Amazônia": "Amaz.",
        "Amazonia": "Amaz.",
        "Amazon": "Amaz.",
        "Caatinga": "Caat.",
        "Cerrado": "Cerr.",
        "Mata Atlântica": "M. Atl.",
        "Mata Atlantica": "M. Atl.",
        "Atlantic Forest": "Atl. For.",
        "Pampa": "Pampa",
        "Pantanal": "Pant.",
    }

    for key, val in replacements.items():
        if key.lower() in s.lower():
            return val

    return s


# =============================================================================
# SPATIAL HELPERS
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


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")

        if src.nodata is not None and np.isfinite(src.nodata):
            arr = np.where(arr == src.nodata, np.nan, arr)

        arr = np.where((arr < -1e20) | (arr > 1e20), np.nan, arr)

        extent = [
            src.transform.c,
            src.transform.c + src.transform.a * src.width,
            src.transform.f + src.transform.e * src.height,
            src.transform.f,
        ]

        profile = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
        }

    return arr, extent, profile


def mask_to_brazil(
    arr: np.ndarray,
    profile: dict,
    brazil: gpd.GeoDataFrame,
) -> np.ndarray:
    if profile["crs"] is None:
        return arr

    brazil_raster = brazil.to_crs(profile["crs"])

    shapes = [
        geom for geom in brazil_raster.geometry
        if geom is not None and not geom.is_empty
    ]

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


# =============================================================================
# AREA HELPERS FOR REJECTED-PIXEL HISTOGRAMS
# =============================================================================

def get_biome_area_m2_lookup(
    biomes: gpd.GeoDataFrame,
    biome_col: str,
) -> dict[str, float]:
    """
    Compute biome areas in m² using a Brazil/South-America Albers Equal Area CRS.
    """
    equal_area_crs = (
        "+proj=aea +lat_1=-5 +lat_2=-25 +lat_0=-15 "
        "+lon_0=-54 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    gdf = biomes[[biome_col, "geometry"]].copy()
    gdf["_biome_clean"] = gdf[biome_col].map(clean_biome_label)
    gdf = gdf.to_crs(equal_area_crs)

    dissolved = gdf.dissolve(by="_biome_clean").reset_index()
    dissolved["area_m2"] = dissolved.geometry.area

    return dict(zip(dissolved["_biome_clean"], dissolved["area_m2"]))


def pixel_area_grid_m2(profile: dict) -> np.ndarray:
    """
    Return a 2D grid of pixel areas in m².

    If the raster CRS is projected, this assumes map units are meters.
    If the raster CRS is geographic, geodesic pixel areas are computed row by row.
    """
    height = int(profile["height"])
    width = int(profile["width"])
    transform = profile["transform"]
    crs = profile["crs"]

    # If CRS is missing, fall back to unit-area pixels.
    # In that case, the histogram becomes an area-like raster fraction.
    if crs is None:
        return np.ones((height, width), dtype="float64")

    if not crs.is_geographic:
        pixel_area = abs(transform.a * transform.e)
        return np.full((height, width), pixel_area, dtype="float64")

    geod = Geod(ellps="WGS84")

    # Assumes a north-up grid with no rotation, which is the normal case
    # for the KS p-value rasters.
    x_left = transform.c
    x_right = transform.c + transform.a

    row_areas = np.zeros(height, dtype="float64")

    for row in range(height):
        y_top = transform.f + row * transform.e
        y_bottom = transform.f + (row + 1) * transform.e

        lons = [x_left, x_right, x_right, x_left]
        lats = [y_top, y_top, y_bottom, y_bottom]

        area, _ = geod.polygon_area_perimeter(lons, lats)
        row_areas[row] = abs(area)

    return np.repeat(row_areas[:, None], width, axis=1)


def biome_rejection_area_percent(
    arr: np.ndarray,
    profile: dict,
    biomes: gpd.GeoDataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    For each biome, compute:

        100 × area(p < threshold) / total biome area

    The rejected area is estimated from raster cells. The biome area denominator
    is computed in an equal-area projection.
    """
    biome_col = find_biome_name_column(biomes)

    if biome_col is None:
        return pd.DataFrame(columns=["biome", "rejected_area_percent"])

    arr_valid = np.isfinite(arr)
    rejected = arr_valid & (arr < cfg.p_threshold)

    if not np.any(arr_valid):
        return pd.DataFrame(columns=["biome", "rejected_area_percent"])

    pixel_area_m2 = pixel_area_grid_m2(profile)
    biome_area_lookup = get_biome_area_m2_lookup(biomes, biome_col)

    if profile["crs"] is not None:
        biomes_raster = biomes[[biome_col, "geometry"]].copy().to_crs(profile["crs"])
    else:
        biomes_raster = biomes[[biome_col, "geometry"]].copy()

    biomes_raster["_biome_clean"] = biomes_raster[biome_col].map(clean_biome_label)
    biomes_dissolved = biomes_raster.dissolve(by="_biome_clean").reset_index()

    rows = []

    for _, row in biomes_dissolved.iterrows():
        biome_name = row["_biome_clean"]
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        inside_biome = geometry_mask(
            [geom],
            out_shape=arr.shape,
            transform=profile["transform"],
            invert=True,
            all_touched=False,
        )

        rejected_area_m2 = float(np.nansum(pixel_area_m2[inside_biome & rejected]))

        biome_area_m2 = float(biome_area_lookup.get(biome_name, np.nan))

        # Fallback if a biome label cannot be matched for any reason.
        if not np.isfinite(biome_area_m2) or biome_area_m2 <= 0:
            biome_area_m2 = float(np.nansum(pixel_area_m2[inside_biome]))

        percent = 100.0 * rejected_area_m2 / biome_area_m2 if biome_area_m2 > 0 else np.nan

        rows.append({
            "biome": biome_name,
            "rejected_area_percent": percent,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    color_map = get_biome_color_map(biomes)
    order = list(color_map.keys())
    order_lookup = {name: i for i, name in enumerate(order)}

    out["_order"] = out["biome"].map(lambda x: order_lookup.get(x, 999))
    out = out.sort_values(["_order", "biome"]).drop(columns="_order").reset_index(drop=True)

    return out


def add_biome_rejection_hist_inset(
    ax,
    arr: np.ndarray,
    profile: dict,
    biomes: gpd.GeoDataFrame,
    cfg: Config,
):
    """
    Add a small inset histogram to a map panel.

    Bars show:
        rejected-pixel area / biome area × 100
    """
    biome_stats = biome_rejection_area_percent(
        arr=arr,
        profile=profile,
        biomes=biomes,
        cfg=cfg,
    )

    if biome_stats.empty:
        return None

    color_map = get_biome_color_map(biomes)

    labels_full = biome_stats["biome"].tolist()
    labels_short = [short_biome_label(x) for x in labels_full]
    values = biome_stats["rejected_area_percent"].values.astype(float)
    colors = [color_map.get(clean_biome_label(x), "0.5") for x in labels_full]

    axins = ax.inset_axes(cfg.hist_inset_axes)
    axins.set_zorder(30)

    x = np.arange(len(values))

    axins.bar(
        x,
        values,
        color=colors,
        edgecolor="0.15",
        linewidth=0.30,
        alpha=cfg.hist_bar_alpha,
        zorder=3,
    )

    axins.set_facecolor((1, 1, 1, cfg.hist_face_alpha))

    # Histogram label is now the title, not the y-axis label.
    axins.set_title(
        "Rejected area (%)",
        fontsize=5.8,
        fontweight="regular",
        pad=1.2,
    )

    axins.set_xticks(x)
    axins.set_xticklabels(
        labels_short,
        fontsize=5.4,
        rotation=90,
        ha="center",
        va="top",
    )

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        ymax = 1.0
    else:
        ymax = max(1.0, float(np.nanmax(finite_values)) * 1.18)

    axins.set_ylim(0, ymax)

    decimals = 1 if ymax <= 5 else 0
    axins.yaxis.set_major_formatter(
        PercentFormatter(xmax=100, decimals=decimals)
    )
    axins.yaxis.set_major_locator(MaxNLocator(nbins=3))

    axins.tick_params(
        axis="x",
        length=1.1,
        width=0.35,
        pad=0.4,
    )
    axins.tick_params(
        axis="y",
        labelsize=5.4,
        length=1.1,
        width=0.35,
        pad=0.5,
    )

    axins.grid(
        axis="y",
        linestyle=":",
        linewidth=0.32,
        alpha=0.55,
        zorder=1,
    )

    for side in ["top", "right"]:
        axins.spines[side].set_visible(False)

    axins.spines["left"].set_linewidth(0.40)
    axins.spines["bottom"].set_linewidth(0.40)

    return axins


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def path_tokens(path: Path) -> set[str]:
    """
    Return normalized path tokens. This helps detect names like:
      ks_p_GUMBEL.tif
      GUMBEL_ks_pvalue.tif
      ks_p_val.tif
    """
    s = normalize_text(str(path))
    return set(s.split("_"))


def has_pvalue_signal(path: Path) -> bool:
    """
    True if path looks like a p-value raster.

    Some GRIDF outputs use compact names like KS_p_GUMBEL.tif,
    so this function also accepts token-level "p" when "ks" is present.
    """
    s = normalize_text(str(path))
    toks = path_tokens(path)

    if "ks" not in toks and "ks" not in s:
        return False

    strong = [
        "pvalue",
        "p_value",
        "pval",
        "p_val",
        "pvalues",
        "p_values",
    ]

    if any(x in s for x in strong):
        return True

    if "p" in toks:
        return True

    if "ksp" in s or "kstestp" in s:
        return True

    return False


def candidate_score(
    path: Path,
    state: str,
    product: str,
    distribution: str,
) -> int:
    """
    Score candidate raster paths so the most likely KS p-value map is selected.
    """
    s = normalize_text(str(path))
    toks = path_tokens(path)
    dist = distribution.lower()

    score = 0

    # Product/state/distribution signals
    if product in s:
        score += 12
    if state in s:
        score += 12
    if dist in s:
        score += 18

    # KS / p-value signals
    if "ks" in toks or "ks" in s:
        score += 25
    if has_pvalue_signal(path):
        score += 30

    # Prefer files whose basename has the diagnostics,
    # not only the folder name.
    b = normalize_text(path.name)
    btoks = set(b.split("_"))

    if "ks" in btoks or "ks" in b:
        score += 8
    if has_pvalue_signal(Path(path.name)):
        score += 8
    if dist in b:
        score += 6

    # Prefer GeoTIFFs in diagnostic / distribution folders.
    for token in ["distribution", "extreme", "diagnostic", "gof", "fit", "ks"]:
        if token in s:
            score += 2

    # Penalize rasters that are likely not p-values.
    bad_tokens = [
        "return",
        "depth",
        "intensity",
        "sherman",
        "parameter",
        "mse",
        "rmse",
        "r2",
        "scale",
        "location",
        "shape",
        "mu",
        "beta",
        "sigma",
        "alpha",
        "rl",
    ]

    for token in bad_tokens:
        if token in toks or token in b:
            score -= 8

    return score


def find_ks_pvalue_raster(
    cfg: Config,
    state: str,
    product: str,
    distribution: str,
) -> Optional[Path]:
    key = (state, product, distribution)

    if key in cfg.explicit_paths:
        p = cfg.explicit_paths[key]
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit path does not exist for {key}: {p}")

    root = cfg.idf_output_root

    # Search first in likely product/state subtrees, then globally.
    likely_roots = [
        root / state / product,
        root / product / state,
        root / state,
        root,
    ]

    candidates = []
    nearby = []
    seen = set()

    for r in likely_roots:
        if not r.exists():
            continue

        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for p in r.rglob(pattern):
                if p in seen:
                    continue

                seen.add(p)

                s = normalize_text(str(p))

                # Keep a small diagnostic list of any file that looks relevant.
                if product in s and state in s and ("ks" in s or distribution.lower() in s):
                    nearby.append(p)

                # Main candidate logic:
                # 1) Must match product/state somewhere in the path
                # 2) Must match distribution somewhere in the path
                # 3) Must look like a KS p-value raster
                if product not in s:
                    continue
                if state not in s:
                    continue
                if distribution.lower() not in s:
                    continue
                if not has_pvalue_signal(p):
                    continue

                candidates.append(p)

    if not candidates:
        print(f"[WARNING] No KS p-value raster found for {state} / {product} / {distribution}")

        if nearby:
            print("          Nearby files that may help diagnose naming:")
            for cand in nearby[:8]:
                print(f"          {cand}")
        else:
            subtree = root / state / product
            if subtree.exists():
                sample = list(subtree.rglob("*.tif"))[:8]
                if sample:
                    print("          Sample GeoTIFFs in expected subtree:")
                    for cand in sample:
                        print(f"          {cand}")
            else:
                print(f"          Expected subtree does not exist: {subtree}")

        return None

    scored = [
        (candidate_score(p, state, product, distribution), p)
        for p in candidates
    ]

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    best_score, best_path = scored[0]
    print(f"[FOUND] {state} / {product} / {distribution}: {best_path} | score={best_score}")

    if len(scored) > 1:
        print("        next candidates:")
        for score, cand in scored[1:4]:
            print(f"        score={score:3d} | {cand}")

    return best_path


# =============================================================================
# COLOR HELPERS
# =============================================================================

def make_ks_colormaps_and_norms(cfg: Config):
    """
    Create two plotting layers:

    1. A single strong red color for rejected pixels:
         p < cfg.p_threshold

    2. A continuous grayscale ramp for non-rejected pixels:
         cfg.p_threshold <= p <= 1
    """
    rejection_cmap = ListedColormap([cfg.rejection_color])

    grayscale_cmap = LinearSegmentedColormap.from_list(
        "ks_gray_continuous",
        [
            "#d9d9d9",  # p near 0.05
            "#a6a6a6",
            "#737373",
            "#404040",
            "#111111",  # p near 1.0
        ],
        N=256,
    )

    grayscale_norm = Normalize(
        vmin=cfg.p_threshold,
        vmax=1.0,
        clip=True,
    )

    return rejection_cmap, grayscale_cmap, grayscale_norm


def draw_ks_colorbar(
    fig,
    axes_for_position,
    cfg: Config,
    grayscale_cmap,
    grayscale_norm,
):
    """
    Draw a horizontal colorbar with:
      - strong red segment for p < 0.05
      - continuous grayscale gradient for p >= 0.05
    """
    positions = [ax.get_position() for ax in axes_for_position]

    x0 = min(p.x0 for p in positions)
    x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions)

    full_width = x1 - x0
    cbar_width = full_width * 0.72
    cbar_height = 0.012
    cbar_x0 = x0 + 0.5 * (full_width - cbar_width)
    cbar_y0 = max(0.025, y0 - 0.045)

    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_height])

    vmin = 0.0
    vmax = 1.0
    threshold = cfg.p_threshold

    # Red segment: p < threshold
    cax.axvspan(
        vmin,
        threshold,
        ymin=0,
        ymax=1,
        color=cfg.rejection_color,
        linewidth=0,
        zorder=1,
    )

    # Continuous grayscale segment: threshold <= p <= 1
    gradient = np.linspace(threshold, vmax, 1024)[None, :]

    cax.imshow(
        gradient,
        aspect="auto",
        cmap=grayscale_cmap,
        norm=grayscale_norm,
        origin="lower",
        extent=[threshold, vmax, 0, 1],
        interpolation="nearest",
        zorder=2,
    )

    # Border
    cax.plot(
        [vmin, vmax, vmax, vmin, vmin],
        [0, 0, 1, 1, 0],
        color="black",
        linewidth=0.9,
        zorder=3,
    )

    # Vertical threshold separator
    cax.axvline(
        threshold,
        color="black",
        linewidth=0.9,
        zorder=4,
    )

    cax.set_xlim(vmin, vmax)
    cax.set_ylim(0, 1)
    cax.set_yticks([])

    ticks = [threshold, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["< 0.05", "0.2", "0.4", "0.6", "0.8", "1.0"]

    cax.set_xticks(ticks)
    cax.set_xticklabels(labels)

    cax.tick_params(
        axis="x",
        which="major",
        labelsize=8.0,
        width=0.9,
        length=3.2,
    )

    cax.set_xlabel(
        r"KS $p$-value",
        fontweight="bold",
        labelpad=3.0,
    )

    for spine in cax.spines.values():
        spine.set_visible(False)

    return cax


# =============================================================================
# PLOTTING
# =============================================================================

def rejection_and_mean_p(arr: np.ndarray, cfg: Config):
    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return np.nan, np.nan

    rej = 100.0 * np.sum(arr[valid] < cfg.p_threshold) / valid.sum()
    mean_p = float(np.nanmean(arr[valid]))

    return rej, mean_p


def plot_ks_panel(
    ax,
    path: Optional[Path],
    biomes: gpd.GeoDataFrame,
    brazil: gpd.GeoDataFrame,
    rejection_cmap,
    grayscale_cmap,
    grayscale_norm,
    cfg: Config,
):
    if path is None:
        ax.text(
            0.5,
            0.5,
            "missing\nKS raster",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )
        style_map_axis(ax, brazil)
        return None

    arr, extent, profile = read_raster(path)
    arr = mask_to_brazil(arr, profile, brazil)

    # Keep only valid p-values.
    arr = np.where((arr >= 0) & (arr <= 1), arr, np.nan)

    # Layer 1: non-rejected pixels, p >= threshold, continuous grayscale.
    arr_gray = np.where(arr >= cfg.p_threshold, arr, np.nan)

    im_gray = ax.imshow(
        np.ma.masked_invalid(arr_gray),
        extent=extent,
        origin="upper",
        cmap=grayscale_cmap,
        norm=grayscale_norm,
        interpolation="nearest",
        zorder=1,
    )

    # Layer 2: rejected pixels, p < threshold, strong red.
    arr_reject = np.where(arr < cfg.p_threshold, 1.0, np.nan)

    ax.imshow(
        np.ma.masked_invalid(arr_reject),
        extent=extent,
        origin="upper",
        cmap=rejection_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=2,
    )

    biomes.boundary.plot(
        ax=ax,
        color="black",
        linewidth=cfg.biome_linewidth,
        alpha=0.80,
        zorder=4,
    )

    brazil.boundary.plot(
        ax=ax,
        color="black",
        linewidth=cfg.country_linewidth,
        zorder=5,
    )

    style_map_axis(ax, brazil)

    rej, mean_p = rejection_and_mean_p(arr, cfg)

    ax.text(
        0.5,
        1.015,
        f"rej={rej:.2f}% | " + rf"$\bar{{p}}$={mean_p:.2f}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
    )

    # Small biome histogram:
    # rejected-pixel area / biome area × 100.
    add_biome_rejection_hist_inset(
        ax=ax,
        arr=arr,
        profile=profile,
        biomes=biomes,
        cfg=cfg,
    )

    return im_gray


def build_figure(cfg: Config):
    setup_style(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading biomes...")
    biomes = load_biomes(cfg.biomes_shp)
    brazil = dissolve_country(biomes)

    rejection_cmap, grayscale_cmap, grayscale_norm = make_ks_colormaps_and_norms(cfg)

    row_specs = [
        ("raw", "GUMBEL"),
        ("bias_corrected_mean", "GUMBEL"),
        ("raw", "GEV"),
        ("bias_corrected_mean", "GEV"),
    ]

    nrows = len(row_specs)
    ncols = len(cfg.products)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.5, 8.7),
        dpi=cfg.dpi,
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.925,
        bottom=0.125,
        wspace=0.030,
        hspace=0.220,
    )

    # Column headers only. No global figure title.
    for j, product in enumerate(cfg.products):
        axes[0, j].set_title(
            cfg.product_labels.get(product, product),
            fontsize=10.0,
            fontweight="bold",
            pad=16,
        )

    for i, (state, distribution) in enumerate(row_specs):
        row_label = (
            f"{cfg.state_labels.get(state, state)} / "
            f"{cfg.distribution_labels.get(distribution, distribution)}"
        )

        axes[i, 0].text(
            -0.13,
            0.5,
            row_label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
        )

        for j, product in enumerate(cfg.products):
            path = find_ks_pvalue_raster(
                cfg=cfg,
                state=state,
                product=product,
                distribution=distribution,
            )

            plot_ks_panel(
                ax=axes[i, j],
                path=path,
                biomes=biomes,
                brazil=brazil,
                rejection_cmap=rejection_cmap,
                grayscale_cmap=grayscale_cmap,
                grayscale_norm=grayscale_norm,
                cfg=cfg,
            )

    draw_ks_colorbar(
        fig=fig,
        axes_for_position=list(axes[-1, :]),
        cfg=cfg,
        grayscale_cmap=grayscale_cmap,
        grayscale_norm=grayscale_norm,
    )

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