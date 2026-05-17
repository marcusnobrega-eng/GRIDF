#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIDF distribution-fit diagnostics for daily annual maxima.

Purpose
-------
Diagnose why KS rejection percentages changed between the previous dataset
and the new GRIDF IDF pipeline.

This script recomputes the daily annual-maximum distribution diagnostics
directly from the annual-maximum GeoTIFF stacks, independently of the IDF
Sherman-curve fitting. Therefore, it helps determine whether high rejection
rates are coming from:

  1) the daily extreme-value distribution fit itself;
  2) GEV shape-parameter instability;
  3) a possible SciPy GEV sign-convention issue;
  4) bias-correction/capping effects in the bias-corrected annual maxima;
  5) differences between products and time periods.

Important
---------
The KS diagnostics are computed BEFORE temporal disaggregation and BEFORE
Sherman IDF fitting. Therefore, Sherman parameter bounds/clips cannot affect
these KS results.

Outputs
-------
For each product/state:
  - GeoTIFF maps:
      GUMBEL_KS_D.tif
      GUMBEL_KS_p.tif
      GEV_KS_D.tif
      GEV_KS_p.tif
      GEV_xi_hydrology_shape.tif
      GEV_scipy_shape_c.tif
      GEV_scale.tif
      deltaD_GEV_minus_GUMBEL.tif
      deltaD_GEV_flipped_shape_minus_GEV.tif

  - Summary CSV:
      distribution_fit_diagnostics_summary.csv

  - Quick-look figure:
      diagnostic_<state>_<product>.png/pdf/svg

How to run
----------
Run all configured products/states:

    python3 diagnose_distribution_fit_daily_amax.py

Run only one product/state first:

    python3 diagnose_distribution_fit_daily_amax.py --products br_dwgd --states raw

Run a faster diagnostic using a pixel limit:

    python3 diagnose_distribution_fit_daily_amax.py --products br_dwgd --states raw --max-pixels 20000

Dependencies
------------
numpy, pandas, scipy, rasterio, geopandas, matplotlib
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats


# =============================================================================
# USER SETTINGS
# =============================================================================

@dataclass
class Config:
    root: Path = Path("/Users/mngomes/Documents/GitHub/GRIDF")

    annual_max_root: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/Annual_Maximum_Precipitation"
    )

    idf_output_root: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Outputs"
    )

    biomes_shp: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes/Brazil_biomes.shp"
    )

    out_dir: Path = Path(
        "/Users/mngomes/Documents/GitHub/GRIDF/IDF_Fitting/Distribution_Diagnostics"
    )

    # Raw annual-maximum folders
    raw_dirs: dict[str, Path] = None

    # Optional manual bias-corrected annual-maximum folders.
    # Leave None to let the script discover them automatically.
    bias_corrected_dirs: dict[str, Optional[Path]] = None

    product_labels: dict[str, str] = None

    # Match the periods used in your IDF workflow.
    product_year_windows: dict[str, tuple[int, int]] = None

    products: tuple[str, ...] = (
        "br_dwgd",
        "chirps",
        "imerg_v06",
        "imerg_v07",
        "persiann_cdr",
    )

    states: tuple[str, ...] = (
        "raw",
        "bias_corrected_mean",
    )

    min_years: int = 15
    p_threshold: float = 0.05

    # Shape thresholds for diagnostic flags.
    xi_warn: float = 0.30
    xi_bad: float = 0.50

    # Bias-correction cap diagnostic.
    # Used only to report how many pixels/values are near the cap.
    possible_rainfall_cap_mm: float = 500.0
    cap_tolerance_mm: float = 1.0

    # Plotting
    font_family: str = "Avenir Next"
    fallback_font: str = "DejaVu Sans"
    dpi: int = 450

    def __post_init__(self):
        if self.raw_dirs is None:
            self.raw_dirs = {
                "br_dwgd": self.annual_max_root / "BR-DWGD",
                "chirps": self.annual_max_root / "CHIRPS_Max",
                "imerg_v06": self.annual_max_root / "IMERG_V06_Max",
                "imerg_v07": self.annual_max_root / "IMERG_V07_Max",
                "persiann_cdr": self.annual_max_root / "PERSIANN_CDR_Max",
            }

        if self.bias_corrected_dirs is None:
            self.bias_corrected_dirs = {
                "br_dwgd": None,
                "chirps": None,
                "imerg_v06": None,
                "imerg_v07": None,
                "persiann_cdr": None,
            }

        if self.product_labels is None:
            self.product_labels = {
                "br_dwgd": "BR-DWGD",
                "chirps": "CHIRPS",
                "imerg_v06": "IMERG V06",
                "imerg_v07": "IMERG V07",
                "persiann_cdr": "PERSIANN-CDR",
            }

        if self.product_year_windows is None:
            self.product_year_windows = {
                "br_dwgd": (1995, 2025),
                "chirps": (1995, 2025),
                "imerg_v06": (2001, 2020),
                "imerg_v07": (2001, 2025),
                "persiann_cdr": (1995, 2025),
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
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.titlesize": 12.0,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# =============================================================================
# BASIC HELPERS
# =============================================================================

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def extract_year(path: Path) -> Optional[int]:
    matches = re.findall(r"(?:19|20)\d{2}", path.name)
    for m in matches:
        y = int(m)
        if 1900 <= y <= 2100:
            return y
    return None


def list_year_tifs(folder: Path, year_window: Optional[tuple[int, int]]) -> list[tuple[int, Path]]:
    if folder is None or not folder.exists():
        return []

    out = []
    for p in sorted(folder.glob("*.tif")):
        y = extract_year(p)
        if y is None:
            continue
        if year_window is not None:
            y0, y1 = year_window
            if not (y0 <= y <= y1):
                continue
        out.append((y, p))

    out = sorted(out, key=lambda x: x[0])
    return out


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


def mask_to_brazil(arr: np.ndarray, profile: dict, brazil: gpd.GeoDataFrame) -> np.ndarray:
    if profile["crs"] is None:
        return arr

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


# =============================================================================
# BIAS-CORRECTED DIRECTORY DISCOVERY
# =============================================================================

def product_search_tokens(product: str) -> list[str]:
    if product == "br_dwgd":
        return ["br_dwgd", "br-dwgd", "brdwgd", "dwgd"]
    if product == "chirps":
        return ["chirps"]
    if product == "imerg_v06":
        return ["imerg_v06", "imerg-v06", "imergv06", "v06", "v6"]
    if product == "imerg_v07":
        return ["imerg_v07", "imerg-v07", "imergv07", "v07", "v7"]
    if product == "persiann_cdr":
        return ["persiann_cdr", "persiann-cdr", "persianncdr", "persiann"]
    return [product]


def discover_bias_corrected_dir(cfg: Config, product: str) -> Optional[Path]:
    """
    Try to find the folder containing bias-corrected annual maxima.

    The function is intentionally flexible because different pipeline runs may
    use slightly different folder names.
    """
    manual = cfg.bias_corrected_dirs.get(product)
    if manual is not None and manual.exists():
        return manual

    tokens = product_search_tokens(product)

    search_roots = [
        cfg.annual_max_root / "BiasCorrected",
        cfg.annual_max_root / "Bias_Corrected",
        cfg.annual_max_root / "bias_corrected",
        cfg.annual_max_root,
    ]

    candidates = []

    for root in search_roots:
        if not root.exists():
            continue

        for d in root.rglob("*"):
            if not d.is_dir():
                continue

            s = normalize_text(str(d))

            # Must mention bias/BC and the product.
            has_bias = any(tok in s for tok in ["bias", "corrected", "bc"])
            has_product = any(normalize_text(tok) in s for tok in tokens)

            if not (has_bias and has_product):
                continue

            tifs = list(d.glob("*.tif"))
            if len(tifs) == 0:
                continue

            candidates.append((len(tifs), d))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_product_state_folder(cfg: Config, product: str, state: str) -> Optional[Path]:
    if state == "raw":
        return cfg.raw_dirs.get(product)

    if state == "bias_corrected_mean":
        return discover_bias_corrected_dir(cfg, product)

    raise ValueError(f"Unknown state: {state}")


# =============================================================================
# RASTER STACK LOADING
# =============================================================================

def read_raster_to_base(path: Path, base_profile: Optional[dict] = None):
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
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "nodata": np.nan,
            }
            extent = [
                src.transform.c,
                src.transform.c + src.transform.a * src.width,
                src.transform.f + src.transform.e * src.height,
                src.transform.f,
            ]
            return arr, profile, extent

        same = (
            src.height == base_profile["height"]
            and src.width == base_profile["width"]
            and src.transform == base_profile["transform"]
            and src.crs == base_profile["crs"]
        )

        if same:
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


def load_stack(folder: Path, year_window: tuple[int, int], brazil: gpd.GeoDataFrame):
    year_files = list_year_tifs(folder, year_window)

    if len(year_files) == 0:
        raise FileNotFoundError(f"No annual-maximum GeoTIFFs found in {folder} for {year_window}")

    years = [y for y, _ in year_files]
    files = [p for _, p in year_files]

    first, profile, extent = read_raster_to_base(files[0], base_profile=None)
    first = mask_to_brazil(first, profile, brazil)

    stack = np.full((len(files), first.shape[0], first.shape[1]), np.nan, dtype="float32")
    stack[0] = first

    for i, p in enumerate(files[1:], start=1):
        arr, _, _ = read_raster_to_base(p, base_profile=profile)
        arr = mask_to_brazil(arr, profile, brazil)
        stack[i] = arr

    return stack, years, profile, extent


# =============================================================================
# DISTRIBUTION FIT HELPERS
# =============================================================================

EULER_GAMMA = 0.5772156649015329


def ks_statistic_from_cdf_values(cdf_values: np.ndarray) -> float:
    """
    One-sample KS statistic for sorted CDF values.

    D+ = max(i/n - F(x_i))
    D- = max(F(x_i) - (i-1)/n)
    D  = max(D+, D-)
    """
    f = np.sort(np.asarray(cdf_values, dtype=float))
    f = f[np.isfinite(f)]

    n = len(f)
    if n == 0:
        return np.nan

    i = np.arange(1, n + 1, dtype=float)
    d_plus = np.max(i / n - f)
    d_minus = np.max(f - (i - 1) / n)

    return float(max(d_plus, d_minus))


def ks_pvalue_asymptotic(D: float, n: int) -> float:
    """
    Approximate one-sample KS p-value.

    Note: because distribution parameters are estimated from the same sample,
    this is a diagnostic approximation rather than an exact hypothesis test.
    """
    if not np.isfinite(D) or n <= 0:
        return np.nan

    try:
        return float(stats.kstwo.sf(D, n))
    except Exception:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
        terms = [(-1) ** (j - 1) * np.exp(-2 * (j ** 2) * (lam ** 2)) for j in range(1, 101)]
        p = 2 * np.sum(terms)
        return float(np.clip(p, 0, 1))


def fit_gumbel_mom(x: np.ndarray):
    """
    Gumbel fit by method of moments, consistent with common hydrology practice.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 3:
        return np.nan, np.nan

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))

    if not np.isfinite(sd) or sd <= 0:
        return np.nan, np.nan

    beta = sd * np.sqrt(6.0) / np.pi
    mu = mean - EULER_GAMMA * beta

    return mu, beta


def fit_pixel(x: np.ndarray):
    """
    Fit Gumbel and GEV to a single pixel annual-maximum series.

    Returns a dict of diagnostic values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    n = len(x)

    out = {
        "n": n,
        "gumbel_mu": np.nan,
        "gumbel_beta": np.nan,
        "gumbel_D": np.nan,
        "gumbel_p": np.nan,
        "gev_c_scipy": np.nan,
        "gev_xi_hydro": np.nan,
        "gev_loc": np.nan,
        "gev_scale": np.nan,
        "gev_D": np.nan,
        "gev_p": np.nan,
        "gev_flipped_D": np.nan,
        "gev_flipped_p": np.nan,
    }

    if n < 3 or np.nanstd(x) <= 0:
        return out

    # ------------------------------------------------------------------
    # Gumbel MOM
    # ------------------------------------------------------------------
    mu, beta = fit_gumbel_mom(x)
    out["gumbel_mu"] = mu
    out["gumbel_beta"] = beta

    if np.isfinite(mu) and np.isfinite(beta) and beta > 0:
        f = stats.gumbel_r.cdf(x, loc=mu, scale=beta)
        D = ks_statistic_from_cdf_values(f)
        out["gumbel_D"] = D
        out["gumbel_p"] = ks_pvalue_asymptotic(D, n)

    # ------------------------------------------------------------------
    # GEV using SciPy convention:
    # scipy.stats.genextreme(c, loc, scale), where c = -xi_hydrology
    # ------------------------------------------------------------------
    try:
        c, loc, scale = stats.genextreme.fit(x)

        if np.isfinite(c) and np.isfinite(loc) and np.isfinite(scale) and scale > 0:
            out["gev_c_scipy"] = float(c)
            out["gev_xi_hydro"] = float(-c)
            out["gev_loc"] = float(loc)
            out["gev_scale"] = float(scale)

            f = stats.genextreme.cdf(x, c, loc=loc, scale=scale)
            D = ks_statistic_from_cdf_values(f)
            out["gev_D"] = D
            out["gev_p"] = ks_pvalue_asymptotic(D, n)

            # Diagnostic for sign-convention errors:
            # If someone accidentally uses xi_hydrology directly in SciPy,
            # this evaluates the same loc/scale with the flipped shape.
            f_flip = stats.genextreme.cdf(x, -c, loc=loc, scale=scale)
            D_flip = ks_statistic_from_cdf_values(f_flip)
            out["gev_flipped_D"] = D_flip
            out["gev_flipped_p"] = ks_pvalue_asymptotic(D_flip, n)

    except Exception:
        pass

    return out


# =============================================================================
# MAIN DIAGNOSTIC COMPUTATION
# =============================================================================

OUTPUT_NAMES = [
    "n",
    "gumbel_mu",
    "gumbel_beta",
    "gumbel_D",
    "gumbel_p",
    "gev_c_scipy",
    "gev_xi_hydro",
    "gev_loc",
    "gev_scale",
    "gev_D",
    "gev_p",
    "gev_flipped_D",
    "gev_flipped_p",
]


def save_geotiff(path: Path, arr: np.ndarray, profile: dict):
    profile_out = profile.copy()
    profile_out.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=2,
        zlevel=6,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype("float32"), 1)


def compute_diagnostics_for_stack(
    stack: np.ndarray,
    cfg: Config,
    max_pixels: int = 0,
    random_seed: int = 123,
):
    nt, ny, nx = stack.shape

    valid_count = np.sum(np.isfinite(stack) & (stack > 0), axis=0)
    valid_pixels = np.where(valid_count >= cfg.min_years)
    n_valid = len(valid_pixels[0])

    if n_valid == 0:
        raise ValueError("No valid pixels passed the min_years threshold.")

    if max_pixels and max_pixels > 0 and n_valid > max_pixels:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n_valid, size=max_pixels, replace=False)
        rows = valid_pixels[0][idx]
        cols = valid_pixels[1][idx]
        print(f"Using random subset of pixels: {max_pixels:,} of {n_valid:,}")
    else:
        rows, cols = valid_pixels
        print(f"Using all valid pixels: {n_valid:,}")

    result = {
        name: np.full((ny, nx), np.nan, dtype="float32")
        for name in OUTPUT_NAMES
    }

    total = len(rows)
    next_report = 0

    for k, (r, c) in enumerate(zip(rows, cols), start=1):
        vals = stack[:, r, c]
        d = fit_pixel(vals)

        for name in OUTPUT_NAMES:
            result[name][r, c] = d[name]

        pct = int(100 * k / total)
        if pct >= next_report:
            print(f"  progress: {pct:3d}% ({k:,}/{total:,})")
            next_report += 5

    # Derived maps
    result["deltaD_GEV_minus_GUMBEL"] = result["gev_D"] - result["gumbel_D"]
    result["deltaD_GEV_flipped_shape_minus_GEV"] = result["gev_flipped_D"] - result["gev_D"]

    return result, valid_count


def summarize_maps(
    maps: dict[str, np.ndarray],
    stack: np.ndarray,
    years: list[int],
    cfg: Config,
    product: str,
    state: str,
):
    valid = np.isfinite(maps["gumbel_D"])

    gD = maps["gumbel_D"]
    gp = maps["gumbel_p"]
    eD = maps["gev_D"]
    ep = maps["gev_p"]
    xi = maps["gev_xi_hydro"]
    scale = maps["gev_scale"]
    deltaD = maps["deltaD_GEV_minus_GUMBEL"]
    flip_delta = maps["deltaD_GEV_flipped_shape_minus_GEV"]

    data_values = stack[np.isfinite(stack)]
    near_cap = np.nan
    if state != "raw" and data_values.size > 0:
        near_cap = 100.0 * np.mean(
            data_values >= (cfg.possible_rainfall_cap_mm - cfg.cap_tolerance_mm)
        )

    def pct(mask):
        mask = mask & valid
        denom = np.sum(valid)
        if denom == 0:
            return np.nan
        return 100.0 * np.sum(mask) / denom

    summary = {
        "product": product,
        "state": state,
        "year_start": min(years),
        "year_end": max(years),
        "n_years": len(years),
        "valid_pixels": int(np.sum(valid)),

        "gumbel_D_mean": float(np.nanmean(gD)),
        "gumbel_D_median": float(np.nanmedian(gD)),
        "gumbel_p_mean": float(np.nanmean(gp)),
        "gumbel_reject_pct_p_lt_0p05": pct(gp < cfg.p_threshold),

        "gev_D_mean": float(np.nanmean(eD)),
        "gev_D_median": float(np.nanmedian(eD)),
        "gev_p_mean": float(np.nanmean(ep)),
        "gev_reject_pct_p_lt_0p05": pct(ep < cfg.p_threshold),

        "gev_worse_than_gumbel_pct_D_GEV_gt_D_GUMBEL": pct(deltaD > 0),
        "gev_much_worse_pct_deltaD_gt_0p05": pct(deltaD > 0.05),
        "gev_better_pct_deltaD_lt_minus_0p05": pct(deltaD < -0.05),

        "gev_xi_mean": float(np.nanmean(xi)),
        "gev_xi_median": float(np.nanmedian(xi)),
        "gev_abs_xi_gt_0p30_pct": pct(np.abs(xi) > cfg.xi_warn),
        "gev_abs_xi_gt_0p50_pct": pct(np.abs(xi) > cfg.xi_bad),

        "gev_scale_mean": float(np.nanmean(scale)),
        "gev_scale_median": float(np.nanmedian(scale)),
        "gev_scale_nonpositive_pct": pct(scale <= 0),

        # If flipped shape gives a much better D, that is a warning that
        # sign convention should be inspected in the pipeline.
        "gev_flipped_shape_better_pct": pct(flip_delta < 0),
        "gev_flipped_shape_much_better_pct_delta_lt_minus_0p05": pct(flip_delta < -0.05),

        "bias_corrected_values_near_500mm_cap_pct": near_cap,
    }

    return summary


# =============================================================================
# QUICK-LOOK FIGURES
# =============================================================================

def plot_diagnostic_figure(
    maps: dict[str, np.ndarray],
    extent,
    profile,
    biomes,
    brazil,
    cfg: Config,
    product: str,
    state: str,
    out_base: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.7), dpi=cfg.dpi)
    fig.subplots_adjust(wspace=0.05, hspace=0.18, left=0.04, right=0.96, top=0.91, bottom=0.08)

    panels = [
        ("Gumbel KS D", maps["gumbel_D"], "magma_r", 0, 0.35),
        ("GEV KS D", maps["gev_D"], "magma_r", 0, 0.35),
        ("ΔD = GEV - Gumbel", maps["deltaD_GEV_minus_GUMBEL"], "coolwarm", -0.15, 0.15),
        ("GEV hydrologic shape ξ", maps["gev_xi_hydro"], "coolwarm", -0.5, 0.5),
        ("GEV scale", maps["gev_scale"], "viridis", None, None),
        ("Flipped-shape ΔD", maps["deltaD_GEV_flipped_shape_minus_GEV"], "coolwarm", -0.15, 0.15),
    ]

    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        arrm = np.ma.masked_invalid(arr)

        if vmin is None:
            vmin = float(np.nanpercentile(arr, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(arr, 98))

        im = ax.imshow(
            arrm,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=1,
        )

        biomes.boundary.plot(ax=ax, color="black", linewidth=0.25, alpha=0.75, zorder=4)
        brazil.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        style_map_axis(ax, brazil)

        ax.set_title(title, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{cfg.product_labels.get(product, product)} | {state} | daily annual maxima distribution diagnostics",
        fontweight="bold",
        y=0.98,
    )

    for ext in ["png", "pdf", "svg"]:
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=cfg.dpi if ext == "png" else None)
        print(f"Saved figure: {out}")

    plt.close(fig)


# =============================================================================
# DRIVER
# =============================================================================

def run_one(cfg: Config, product: str, state: str, brazil: gpd.GeoDataFrame, biomes: gpd.GeoDataFrame, max_pixels: int):
    year_window = cfg.product_year_windows.get(product)
    folder = get_product_state_folder(cfg, product, state)

    if folder is None or not folder.exists():
        print(f"[SKIP] Could not find folder for {product} / {state}")
        print(f"       If this is a bias-corrected case, set Config.bias_corrected_dirs manually.")
        return None

    print("\n" + "=" * 110)
    print(f"PRODUCT: {product} ({cfg.product_labels.get(product, product)}) | STATE: {state}")
    print(f"Folder: {folder}")
    print(f"Year window: {year_window}")
    print("=" * 110)

    stack, years, profile, extent = load_stack(folder, year_window, brazil)
    print(f"Loaded stack: {stack.shape} | years: {years[0]}--{years[-1]} | n={len(years)}")

    maps, valid_count = compute_diagnostics_for_stack(
        stack=stack,
        cfg=cfg,
        max_pixels=max_pixels,
        random_seed=123,
    )

    summary = summarize_maps(
        maps=maps,
        stack=stack,
        years=years,
        cfg=cfg,
        product=product,
        state=state,
    )

    out_dir = cfg.out_dir / state / product
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all diagnostic maps.
    save_map_names = {
        "gumbel_D": "GUMBEL_KS_D.tif",
        "gumbel_p": "GUMBEL_KS_p.tif",
        "gev_D": "GEV_KS_D.tif",
        "gev_p": "GEV_KS_p.tif",
        "gev_xi_hydro": "GEV_xi_hydrology_shape.tif",
        "gev_c_scipy": "GEV_scipy_shape_c.tif",
        "gev_scale": "GEV_scale.tif",
        "deltaD_GEV_minus_GUMBEL": "deltaD_GEV_minus_GUMBEL.tif",
        "deltaD_GEV_flipped_shape_minus_GEV": "deltaD_GEV_flipped_shape_minus_GEV.tif",
    }

    for key, filename in save_map_names.items():
        save_geotiff(out_dir / filename, maps[key], profile)

    print(f"Saved GeoTIFF diagnostics to: {out_dir}")

    plot_diagnostic_figure(
        maps=maps,
        extent=extent,
        profile=profile,
        biomes=biomes,
        brazil=brazil,
        cfg=cfg,
        product=product,
        state=state,
        out_base=out_dir / f"diagnostic_{state}_{product}",
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose Gumbel/GEV distribution fits for GRIDF daily annual maxima."
    )

    parser.add_argument(
        "--products",
        type=str,
        default=",".join(CFG.products),
        help="Comma-separated product keys, e.g. br_dwgd,chirps,imerg_v06",
    )

    parser.add_argument(
        "--states",
        type=str,
        default=",".join(CFG.states),
        help="Comma-separated states, e.g. raw,bias_corrected_mean",
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="Maximum number of valid pixels to fit per product/state. 0 means all pixels.",
    )

    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = parse_args()
    setup_style(CFG)

    products = [x.strip() for x in args.products.split(",") if x.strip()]
    states = [x.strip() for x in args.states.split(",") if x.strip()]

    CFG.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading biomes...")
    biomes = load_biomes(CFG.biomes_shp)
    brazil = dissolve_country(biomes)

    summaries = []

    for product in products:
        for state in states:
            s = run_one(
                cfg=CFG,
                product=product,
                state=state,
                brazil=brazil,
                biomes=biomes,
                max_pixels=args.max_pixels,
            )
            if s is not None:
                summaries.append(s)

    if summaries:
        df = pd.DataFrame(summaries)
        out_csv = CFG.out_dir / "distribution_fit_diagnostics_summary.csv"
        df.to_csv(out_csv, index=False)
        print("\nSummary saved:")
        print(out_csv)

        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(df)


if __name__ == "__main__":
    main()
