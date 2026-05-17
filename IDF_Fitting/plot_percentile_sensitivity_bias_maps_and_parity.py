#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Percentile-threshold sensitivity for GRIDF bias correction.

This script uses the SAME pairwise bias-correction methodology and visual style
as the legacy/paper figure provided by the user.

It creates two main figures:

1) Bias-factor maps
   - rows    = products
   - columns = percentile thresholds
   - maps show spatial bias factor zeta
   - same colorbar style as the paper bias map
   - mean and std are shown in each panel

2) Bias-corrected parity plots
   - rows    = products
   - columns = percentile thresholds
   - x-axis  = rain gauge rainfall depth
   - y-axis  = bias-corrected product rainfall depth
   - density-colored points, 1:1 line, through-origin fit line
   - equation and paper-style R² shown in each panel

It also writes a CSV table with raw and corrected pairwise metrics for every
product-percentile combination.

Author: GRIDF workflow
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import rasterio
import geopandas as gpd
from rasterio.mask import mask


# ============================================================
# SETTINGS
# ============================================================

GRIDF_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF")
PIPELINE_ROOT = GRIDF_ROOT / "Bias_Correction_Pipeline"

# Percentile thresholds to compare.
PERCENTILES = ["p90", "p95", "p98", "p99", "p995"]

# Estimator used by the bias-correction pipeline.
DEFAULT_ESTIMATOR = "mean"

CONFIGS = [
    {"name": "BR-DWGD",       "product": "br_dwgd",      "precip_col": "xavier_pr_mm"},
    {"name": "IMERG V06",     "product": "imerg_v06",    "precip_col": "imerg_mm"},
    {"name": "IMERG V07",     "product": "imerg_v07",    "precip_col": "imerg_mm"},
    {"name": "CHIRPS",        "product": "chirps",       "precip_col": "chirps_mm"},
    {"name": "PERSIANN-CDR",  "product": "persiann_cdr", "precip_col": "persiann_mm"},
]

# Biomes instead of states.
BIOMES_SHP = GRIDF_ROOT / "BrazilShapefiles" / "Biomes" / "Brazil_biomes.shp"

SHOW_BIOME_BOUNDARIES_ON_BIAS_MAPS = False
BIOME_EDGE_COLOR_MAIN = "white"
BIOME_EDGE_COLOR_UNDER = "black"
BIOME_EDGE_WIDTH_MAIN = 0.40
BIOME_EDGE_WIDTH_UNDER = 0.85

# Font.
FONT_SIZE = 10
FONT_FAMILY = "Helvetica"

# Scatter/QC params matching the legacy figure.
AX_MIN = 0
AX_MAX = 200
AX_TICKS = [0, 50, 100, 150, 200]

MIN_MM = 1.0
RATIO_CLIP = (0.1, 10.0)
NBINS = 100
DOT_SIZE = 4.5
ALPHA = 0.72
TICK_WIDTH = 1.6
BORDER_WIDTH = 1.8
DPI = 600

# Bias map colorbar. Same style as the paper bias-map figure.
BIAS_VMIN = 1.0
BIAS_VMAX = 3.5
BIAS_LEVELS = np.linspace(BIAS_VMIN, BIAS_VMAX, 11)
BIAS_CMAP = "nipy_spectral"

CBAR_WIDTH_FIG = 0.014
CBAR_RIGHT_PAD = 0.025
CBAR_HEIGHT_REL = 0.74
CBAR_TICK_STEP = 0.5

# Density colorbar geometry.
DENSITY_CBAR_HEIGHT = 0.010
DENSITY_CBAR_Y = 0.035

# Contours on bias maps.
CONTOUR_LEVELS = np.arange(1.0, 3.5 + 1e-9, 0.5)
SHOW_BIAS_CONTOURS = True

# Bias-map statistic label style.
SHOW_BIAS_STAT_LABEL = True
BIAS_STAT_LABEL_DECIMALS = 2
BIAS_STAT_LABEL_FONT_SIZE = 7.2
BIAS_STAT_LABEL_FONT_WEIGHT = "bold"
BIAS_STAT_LABEL_BOX_ALPHA = 0.84

# Figure dimensions.
BIAS_MAP_FIG_W = 10.8
BIAS_MAP_FIG_H = 10.4

PARITY_FIG_W = 10.8
PARITY_FIG_H = 10.4


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_station_id(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def setup_font():
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    available = {f.name for f in fm.fontManager.ttflist}
    if FONT_FAMILY not in available:
        print(f"Warning: {FONT_FAMILY} not detected by Matplotlib. It will use the closest available fallback.")


def fit_origin(x, y):
    """
    Through-origin fit used in the paper figure.

    Returns:
        a  : slope in y = a x
        r2 : paper-style through-origin R²
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]

    if x.size == 0:
        return np.nan, np.nan

    denom = np.sum(x * x)
    denom_r2 = np.sum(y * y)

    if denom <= 0 or denom_r2 <= 0:
        return np.nan, np.nan

    a = np.sum(x * y) / denom
    r2 = 1.0 - np.sum((y - a * x) ** 2) / denom_r2

    return a, r2


def fit_origin_metrics(x, y):
    """
    Full pairwise metrics for product-vs-gauge comparisons.

    The main R² follows the same through-origin definition used in the paper
    figure. Pearson correlation squared is also returned as a diagnostic.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]

    if x.size < 3:
        return {
            "n_pairs": int(x.size),
            "slope_through_origin": np.nan,
            "r2_origin": np.nan,
            "corr_r2": np.nan,
            "ols_slope": np.nan,
            "ols_intercept": np.nan,
            "rmse_mm": np.nan,
            "mae_mm": np.nan,
            "mean_bias_mm": np.nan,
            "percent_bias": np.nan,
        }

    a, r2_origin = fit_origin(x, y)

    if np.std(x) > 0 and np.std(y) > 0:
        corr_r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    else:
        corr_r2 = np.nan

    ols_slope, ols_intercept = np.polyfit(x, y, 1)

    residual = y - x

    return {
        "n_pairs": int(x.size),
        "slope_through_origin": float(a),
        "r2_origin": float(r2_origin),
        "corr_r2": float(corr_r2) if np.isfinite(corr_r2) else np.nan,
        "ols_slope": float(ols_slope),
        "ols_intercept": float(ols_intercept),
        "rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
        "mae_mm": float(np.mean(np.abs(residual))),
        "mean_bias_mm": float(np.mean(residual)),
        "percent_bias": float(100.0 * np.sum(y - x) / np.sum(x)) if np.sum(x) > 0 else np.nan,
    }


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(BORDER_WIDTH)

    ax.tick_params(width=TICK_WIDTH, length=5.5)
    ax.grid(True, ls=":", lw=0.6)


def bin_counts_for_points(x, y, xedges, yedges):
    H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))

    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, len(xedges) - 2)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, len(yedges) - 2)

    vals = H[ix, iy]
    return vals, H.max()


def panel_scatter_density_corrected(ax, x, y, norm, title, fit_color):
    xedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)
    yedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)

    m = np.isfinite(x) & np.isfinite(y) & (x >= AX_MIN) & (x <= AX_MAX) & (y >= AX_MIN) & (y <= AX_MAX)
    x_plot = np.asarray(x)[m]
    y_plot = np.asarray(y)[m]

    vals, _ = bin_counts_for_points(x_plot, y_plot, xedges, yedges)

    sc = ax.scatter(
        x_plot,
        y_plot,
        c=vals,
        cmap="viridis",
        norm=norm,
        s=DOT_SIZE,
        alpha=ALPHA,
        edgecolors="none",
        rasterized=True,
    )

    ax.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], color="red", lw=1.8, linestyle="--")

    a, r2 = fit_origin(x_plot, y_plot)

    if np.isfinite(a):
        xx = np.linspace(AX_MIN, AX_MAX, 200)
        ax.plot(xx, a * xx, color=fit_color, lw=2.2)

        ax.text(
            0.045 * AX_MAX,
            0.89 * AX_MAX,
            rf"$y={a:.2f}x,\ R^2={r2:.2f}$",
            color=fit_color,
            fontsize=7.4,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.23",
                facecolor="white",
                edgecolor=fit_color,
                alpha=0.85,
            ),
        )

    ax.set_xlim(AX_MIN, AX_MAX)
    ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_xticks(AX_TICKS)
    ax.set_yticks(AX_TICKS)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=7)

    style_axes(ax)
    return sc


def parse_percentile_arg(value):
    if value is None or value.strip() == "":
        return PERCENTILES
    return [v.strip() for v in value.split(",") if v.strip()]


# ============================================================
# DATA LOADING: PAIRS AND ZETA
# ============================================================

def read_pairs(product, precip_col, percentile):
    pair_dir = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / percentile / "pairs"
    files = sorted(pair_dir.glob(f"pairs_{product}_{percentile}_*.csv"))

    if not files:
        # More flexible fallback.
        files = sorted(pair_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No pair files found in {pair_dir}")

    df = pd.concat(
        [pd.read_csv(f, parse_dates=["date"], low_memory=False) for f in files],
        ignore_index=True,
    )

    if precip_col not in df.columns:
        if "product_mm" in df.columns:
            df[precip_col] = df["product_mm"]
        else:
            raise ValueError(f"{product}/{percentile}: missing {precip_col} and product_mm")

    if "pr_g" not in df.columns:
        if "gauge_mm" in df.columns:
            df["pr_g"] = df["gauge_mm"]
        else:
            raise ValueError(f"{product}/{percentile}: missing pr_g and gauge_mm")

    if "station_id" not in df.columns:
        raise ValueError(f"{product}/{percentile}: missing station_id")

    df["station_id"] = normalize_station_id(df["station_id"])
    df["pr_g"] = pd.to_numeric(df["pr_g"], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")

    # Legacy plotting filter.
    df = df[(df["pr_g"] >= MIN_MM) & (df[precip_col] >= MIN_MM)].copy()

    if "ratio" in df.columns:
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
        df = df[df["ratio"].between(*RATIO_CLIP)].copy()

    return df


def read_zeta(product, percentile, estimator):
    zeta_csv = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / percentile
        / "zeta_station"
        / estimator
        / f"zeta_per_station_{product}_{percentile}_{estimator}.csv"
    )

    if not zeta_csv.exists():
        raise FileNotFoundError(zeta_csv)

    z = pd.read_csv(zeta_csv, low_memory=False)

    if "station_id" not in z.columns:
        raise ValueError(f"{zeta_csv} must contain station_id")

    z["station_id"] = normalize_station_id(z["station_id"])

    if "zeta" not in z.columns:
        if "zeta_selected" in z.columns:
            z["zeta"] = z["zeta_selected"]
        elif "zeta_mean" in z.columns:
            z["zeta"] = z["zeta_mean"]
        else:
            raise ValueError(f"{zeta_csv} must contain zeta, zeta_selected, or zeta_mean")

    z["zeta"] = pd.to_numeric(z["zeta"], errors="coerce")
    return z[["station_id", "zeta"]].dropna()


def load_product_percentile_arrays(product, precip_col, percentile, estimator):
    pairs = read_pairs(product, precip_col, percentile)
    zeta = read_zeta(product, percentile, estimator)

    df = pairs.merge(zeta, on="station_id", how="left").dropna(subset=["zeta"])

    if df.empty:
        raise ValueError(f"{product}/{percentile}: no pairs after merging zeta")

    df["product_corr"] = df[precip_col] * df["zeta"]

    x = df["pr_g"].to_numpy(float)
    y_raw = df[precip_col].to_numpy(float)
    y_cor = df["product_corr"].to_numpy(float)

    return {
        "df": df,
        "x": x,
        "y_raw": y_raw,
        "y_cor": y_cor,
        "n": len(df),
        "n_stations": df["station_id"].nunique(),
        "zeta_mean": float(df["zeta"].mean()),
        "zeta_std": float(df["zeta"].std()),
        "zeta_median": float(df["zeta"].median()),
    }


def find_zeta_tif(product, percentile, estimator):
    folder = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / percentile
        / "zeta_grid"
        / estimator
    )

    preferred = folder / f"zeta_map_{product}_{percentile}_{estimator}_idw_k8_p2p0.tif"

    if preferred.exists():
        return preferred

    files = sorted(folder.glob(f"zeta_map_{product}_{percentile}_{estimator}_idw_*.tif"))

    if files:
        return files[-1]

    raise FileNotFoundError(f"No zeta raster found in {folder}")


# ============================================================
# DATA LOADING: BIAS MAPS
# ============================================================

def load_biomes():
    if not BIOMES_SHP.exists():
        print(f"Warning: biomes shapefile not found: {BIOMES_SHP}")
        return None, None, None

    gdf = gpd.read_file(BIOMES_SHP)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    brazil_geom = gdf.dissolve().geometry.iloc[0]
    brazil_crs = gdf.crs

    return gdf, brazil_geom, brazil_crs


def load_bias_map(tif_path, brazil_geom=None, brazil_crs=None):
    with rasterio.open(tif_path) as ds:
        if brazil_geom is not None:
            gdf_tmp = gpd.GeoDataFrame(
                geometry=[brazil_geom],
                crs=brazil_crs if brazil_crs is not None else 4326,
            ).to_crs(ds.crs)

            data, _ = mask(
                ds,
                [gdf_tmp.geometry.iloc[0]],
                crop=False,
                filled=True,
                invert=False,
                nodata=np.nan,
            )

            arr = data[0].astype("float32")
        else:
            arr = ds.read(1, masked=True).astype("float32")
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)

            nod = ds.nodata
            if nod is not None and not np.isnan(nod):
                arr = np.where(arr == nod, np.nan, arr)

        bounds = ds.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    arr = np.where((arr < -1e20) | (arr > 1e20), np.nan, arr)

    return arr, extent


# ============================================================
# MAP PLOTTING
# ============================================================

def plot_bias_map(ax, arr, extent, norm, cmap, biomes_gdf):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)

    if SHOW_BIOME_BOUNDARIES_ON_BIAS_MAPS and biomes_gdf is not None and not biomes_gdf.empty:
        biomes_gdf.boundary.plot(
            ax=ax,
            edgecolor=BIOME_EDGE_COLOR_UNDER,
            linewidth=BIOME_EDGE_WIDTH_UNDER,
            alpha=0.65,
        )
        biomes_gdf.boundary.plot(
            ax=ax,
            edgecolor=BIOME_EDGE_COLOR_MAIN,
            linewidth=BIOME_EDGE_WIDTH_MAIN,
            alpha=0.95,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_frame_on(False)
    ax.set_facecolor("none")
    return im


def make_cbar_for_maps(fig, right_col_axes, mappable):
    rights, bottoms, tops = [], [], []

    for ax in right_col_axes:
        pos = ax.get_position()
        rights.append(pos.x1)
        bottoms.append(pos.y0)
        tops.append(pos.y1)

    x_right = max(rights)
    y0 = min(bottoms)
    y1 = max(tops)
    stack_h = y1 - y0

    cbar_h = stack_h * CBAR_HEIGHT_REL
    cbar_y = y0 + 0.5 * (stack_h - cbar_h)
    cbar_x = x_right + CBAR_RIGHT_PAD

    cbar_ax = fig.add_axes([cbar_x, cbar_y, CBAR_WIDTH_FIG, cbar_h])

    cb = fig.colorbar(
        mappable,
        cax=cbar_ax,
        extend="both",
        extendfrac=0.05,
        boundaries=BIAS_LEVELS,
        spacing="proportional",
    )

    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.tick_params(left=False, right=True, labelleft=False, labelright=True)

    cb.set_ticks(np.arange(BIAS_VMIN, BIAS_VMAX + 1e-9, CBAR_TICK_STEP))
    cb.ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    cb.ax.tick_params(which="major", length=6, width=1.2, labelsize=11)
    cb.ax.tick_params(which="minor", length=3, width=0.8)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    cb.set_label(r"Bias factor ($\zeta$)", rotation=90, labelpad=12, va="center", fontsize=11)
    cb.outline.set_linewidth(1.5)

    return cb


def density_ticks_from_vmax(vmax):
    vmax = int(np.ceil(float(vmax)))
    ticks = [1]

    for t in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if 1 < t < vmax:
            ticks.append(t)

    if vmax not in ticks:
        ticks.append(vmax)

    return ticks


def compute_raster_mean_std(data):
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


def add_bias_stat_label(ax, mean_value, std_value):
    if not SHOW_BIAS_STAT_LABEL:
        return

    if np.isnan(mean_value) or np.isnan(std_value):
        label = r"$\mu$ = NA" + "\n" + r"$\sigma$ = NA"
    else:
        label = (
            rf"$\mu$ = {mean_value:.{BIAS_STAT_LABEL_DECIMALS}f}"
            + "\n"
            + rf"$\sigma$ = {std_value:.{BIAS_STAT_LABEL_DECIMALS}f}"
        )

    ax.text(
        0.965,
        0.985,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=BIAS_STAT_LABEL_FONT_SIZE,
        fontweight=BIAS_STAT_LABEL_FONT_WEIGHT,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="black",
            linewidth=0.65,
            alpha=BIAS_STAT_LABEL_BOX_ALPHA,
        ),
        zorder=20,
    )


def make_cbar_for_density(fig, axes, norm_scatter):
    # Use bottom row first and last axes to define colorbar width.
    pos_left = axes[-1, 0].get_position()
    pos_right = axes[-1, -1].get_position()

    cbar_x = pos_left.x0
    cbar_w = pos_right.x1 - pos_left.x0
    cbar_y = DENSITY_CBAR_Y

    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, DENSITY_CBAR_HEIGHT])

    sm = plt.cm.ScalarMappable(norm=norm_scatter, cmap="viridis")
    sm.set_array([])

    vmax = int(np.ceil(float(norm_scatter.vmax)))
    ticks = density_ticks_from_vmax(vmax)

    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])

    cb.set_label("Point density per bin", labelpad=4)

    cb.ax.tick_params(
        which="major",
        length=4.5,
        width=1.4,
        labelsize=8,
        pad=2,
    )
    cb.outline.set_linewidth(1.4)

    return cb


# ============================================================
# MAIN ANALYSIS
# ============================================================

def compute_all_results(percentiles, estimator):
    results = {}
    metric_rows = []

    for cfg in CONFIGS:
        product = cfg["product"]
        results[product] = {}

        for pct in percentiles:
            print(f"Loading pairwise data: {cfg['name']} / {pct} / {estimator}")

            arrs = load_product_percentile_arrays(product, cfg["precip_col"], pct, estimator)
            arrs["name"] = cfg["name"]
            arrs["product"] = product
            arrs["percentile"] = pct
            results[product][pct] = arrs

            raw_metrics = fit_origin_metrics(arrs["x"], arrs["y_raw"])
            cor_metrics = fit_origin_metrics(arrs["x"], arrs["y_cor"])

            row = {
                "product": product,
                "product_name": cfg["name"],
                "percentile": pct,
                "estimator": estimator,
                "n_pairs": arrs["n"],
                "n_stations": arrs["n_stations"],
                "zeta_mean_pairweighted": arrs["zeta_mean"],
                "zeta_std_pairweighted": arrs["zeta_std"],
                "zeta_median_pairweighted": arrs["zeta_median"],
            }

            for k, v in raw_metrics.items():
                row[f"raw_{k}"] = v

            for k, v in cor_metrics.items():
                row[f"corrected_{k}"] = v

            metric_rows.append(row)

            print(
                f"  pairs={arrs['n']:,}; stations={arrs['n_stations']:,}; "
                f"corrected y={cor_metrics['slope_through_origin']:.2f}x; "
                f"R2={cor_metrics['r2_origin']:.2f}"
            )

    metrics = pd.DataFrame(metric_rows)
    return results, metrics


def load_all_bias_maps(percentiles, estimator, biomes_gdf, brazil_geom, brazil_crs):
    bias_maps = {}
    map_rows = []

    for cfg in CONFIGS:
        product = cfg["product"]
        bias_maps[product] = {}

        for pct in percentiles:
            print(f"Loading zeta map: {cfg['name']} / {pct} / {estimator}")

            tif = find_zeta_tif(product, pct, estimator)
            arr, extent = load_bias_map(tif, brazil_geom, brazil_crs)
            mean_value, std_value = compute_raster_mean_std(arr)

            bias_maps[product][pct] = {
                "arr": arr,
                "extent": extent,
                "tif": tif,
                "mean": mean_value,
                "std": std_value,
            }

            map_rows.append(
                {
                    "product": product,
                    "product_name": cfg["name"],
                    "percentile": pct,
                    "estimator": estimator,
                    "zeta_map_path": str(tif),
                    "zeta_map_mean": mean_value,
                    "zeta_map_std": std_value,
                    "zeta_map_median": float(np.nanmedian(arr)),
                    "zeta_map_p05": float(np.nanpercentile(arr[np.isfinite(arr)], 5)) if np.isfinite(arr).any() else np.nan,
                    "zeta_map_p95": float(np.nanpercentile(arr[np.isfinite(arr)], 95)) if np.isfinite(arr).any() else np.nan,
                }
            )

            print(f"  map mean={mean_value:.3f}; std={std_value:.3f}; tif={tif}")

    map_stats = pd.DataFrame(map_rows)
    return bias_maps, map_stats


def plot_bias_map_grid(bias_maps, percentiles, estimator, out_dir):
    biomes_gdf, brazil_geom, brazil_crs = load_biomes()

    # If already loaded outside, reload here to keep this function standalone.
    if biomes_gdf is None:
        print("Warning: plotting bias maps without biome mask/boundaries.")

    cmap_bias = plt.get_cmap(BIAS_CMAP).copy()
    cmap_bias.set_bad((0, 0, 0, 0))
    norm_bias = BoundaryNorm(BIAS_LEVELS, cmap_bias.N, clip=True)

    nrows = len(CONFIGS)
    ncols = len(percentiles)

    fig, axs = plt.subplots(nrows, ncols, figsize=(BIAS_MAP_FIG_W, BIAS_MAP_FIG_H), dpi=DPI)
    fig.subplots_adjust(wspace=0.06, hspace=0.16, right=0.90)

    first_map_mappable = None
    right_col_axes = [axs[i, -1] for i in range(nrows)]

    for i, cfg in enumerate(CONFIGS):
        product = cfg["product"]

        for j, pct in enumerate(percentiles):
            ax = axs[i, j]
            bm = bias_maps[product][pct]

            arr_show = np.array(bm["arr"], copy=True)
            arr_show = np.where(
                np.isfinite(arr_show),
                np.clip(arr_show, BIAS_VMIN, BIAS_VMAX),
                np.nan,
            )

            im = plot_bias_map(
                ax,
                arr_show,
                bm["extent"],
                norm_bias,
                cmap_bias,
                biomes_gdf,
            )

            if first_map_mappable is None:
                first_map_mappable = im

            if SHOW_BIAS_CONTOURS:
                try:
                    ny, nx = arr_show.shape
                    x = np.linspace(bm["extent"][0], bm["extent"][1], nx)
                    y = np.linspace(bm["extent"][3], bm["extent"][2], ny)
                    X, Y = np.meshgrid(x, y)
                    Z = np.ma.masked_invalid(arr_show)
                    cs = ax.contour(X, Y, Z, levels=CONTOUR_LEVELS, colors="k", linewidths=0.65)
                    ax.clabel(cs, fmt="%.1f", inline=True, fontsize=5.8)
                except Exception:
                    pass

            add_bias_stat_label(ax, bm["mean"], bm["std"])

            if i == 0:
                ax.set_title(pct, pad=8, fontsize=10, fontweight="bold")

            if j == 0:
                ax.text(
                    -0.06,
                    0.5,
                    cfg["name"],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="bold",
                )

    if first_map_mappable is not None:
        make_cbar_for_maps(fig, right_col_axes, first_map_mappable)

    out_png = out_dir / f"percentile_sensitivity_zeta_maps_{estimator}.png"
    out_pdf = out_dir / f"percentile_sensitivity_zeta_maps_{estimator}.pdf"
    out_svg = out_dir / f"percentile_sensitivity_zeta_maps_{estimator}.svg"

    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print("Saved:", out_svg)


def plot_corrected_parity_grid(results, percentiles, estimator, out_dir):
    # Global density scale across all corrected panels.
    xedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)
    yedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)

    vmax = 2
    for cfg in CONFIGS:
        product = cfg["product"]
        for pct in percentiles:
            r = results[product][pct]
            x = r["x"]
            y = r["y_cor"]
            m = np.isfinite(x) & np.isfinite(y) & (x >= AX_MIN) & (x <= AX_MAX) & (y >= AX_MIN) & (y <= AX_MAX)
            _, vmax_here = bin_counts_for_points(x[m], y[m], xedges, yedges)
            vmax = max(vmax, vmax_here)

    norm_scatter = LogNorm(vmin=1, vmax=vmax)

    nrows = len(CONFIGS)
    ncols = len(percentiles)

    fig, axs = plt.subplots(nrows, ncols, figsize=(PARITY_FIG_W, PARITY_FIG_H), dpi=DPI)
    fig.subplots_adjust(wspace=0.13, hspace=0.24, bottom=0.095)

    for i, cfg in enumerate(CONFIGS):
        product = cfg["product"]
        is_bottom = i == nrows - 1

        for j, pct in enumerate(percentiles):
            ax = axs[i, j]
            r = results[product][pct]

            title = pct if i == 0 else ""

            panel_scatter_density_corrected(
                ax,
                r["x"],
                r["y_cor"],
                norm_scatter,
                title=title,
                fit_color="tab:blue",
            )

            # Only left column keeps y labels.
            if j > 0:
                ax.tick_params(axis="y", which="both", labelleft=False)

            # Only bottom row keeps x labels.
            if is_bottom:
                ax.tick_params(axis="x", which="both", labelbottom=True)
            else:
                ax.tick_params(axis="x", which="both", labelbottom=False)

            if j == 0:
                ax.set_ylabel("Product (mm)")
                ax.text(
                    -0.35,
                    0.5,
                    cfg["name"],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.set_ylabel("")

            if is_bottom:
                ax.set_xlabel("Rain Gauge (mm)")
            else:
                ax.set_xlabel("")

    make_cbar_for_density(fig, axs, norm_scatter)

    out_png = out_dir / f"percentile_sensitivity_corrected_parity_{estimator}.png"
    out_pdf = out_dir / f"percentile_sensitivity_corrected_parity_{estimator}.pdf"
    out_svg = out_dir / f"percentile_sensitivity_corrected_parity_{estimator}.svg"

    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print("Saved:", out_svg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", default=DEFAULT_ESTIMATOR, choices=["mean", "median"])
    parser.add_argument(
        "--percentiles",
        default=",".join(PERCENTILES),
        help="Comma-separated percentile thresholds, e.g. p90,p95,p98,p99,p995",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip bias-factor map figure.",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip corrected parity plot figure.",
    )
    args = parser.parse_args()

    estimator = args.estimator
    percentiles = parse_percentile_arg(args.percentiles)

    setup_font()

    out_dir = (
        GRIDF_ROOT
        / "IDF_Fitting"
        / "Percentile_Sensitivity"
        / "PAIRWISE_BIAS_LEGACY_STYLE"
        / estimator
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pairwise results and compute metrics.
    results, metrics = compute_all_results(percentiles, estimator)

    metrics_csv = out_dir / f"pairwise_bias_metrics_percentile_sensitivity_{estimator}.csv"
    metrics.to_csv(metrics_csv, index=False)
    print("Saved:", metrics_csv)

    # Load maps and compute spatial map statistics.
    biomes_gdf, brazil_geom, brazil_crs = load_biomes()

    if not args.skip_maps:
        bias_maps, map_stats = load_all_bias_maps(percentiles, estimator, biomes_gdf, brazil_geom, brazil_crs)
        map_stats_csv = out_dir / f"zeta_map_statistics_percentile_sensitivity_{estimator}.csv"
        map_stats.to_csv(map_stats_csv, index=False)
        print("Saved:", map_stats_csv)

        plot_bias_map_grid(bias_maps, percentiles, estimator, out_dir)
    else:
        map_stats = pd.DataFrame()

    if not args.skip_parity:
        plot_corrected_parity_grid(results, percentiles, estimator, out_dir)

    # Compact printed summary.
    keep = [
        "product_name",
        "percentile",
        "n_pairs",
        "n_stations",
        "corrected_slope_through_origin",
        "corrected_r2_origin",
        "corrected_rmse_mm",
        "corrected_percent_bias",
        "zeta_mean_pairweighted",
        "zeta_std_pairweighted",
    ]

    print("\nPairwise bias-corrected sensitivity summary:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(metrics[[c for c in keep if c in metrics.columns]].to_string(index=False))

    if not map_stats.empty:
        keep_maps = [
            "product_name",
            "percentile",
            "zeta_map_mean",
            "zeta_map_std",
            "zeta_map_median",
            "zeta_map_p05",
            "zeta_map_p95",
        ]
        print("\nSpatial zeta-map sensitivity summary:")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(map_stats[[c for c in keep_maps if c in map_stats.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
