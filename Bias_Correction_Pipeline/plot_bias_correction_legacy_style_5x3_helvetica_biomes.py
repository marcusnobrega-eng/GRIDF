#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")

PERCENTILE = "p98"

CONFIGS = [
    {"name": "BR-DWGD",    "product": "br_dwgd",      "precip_col": "xavier_pr_mm"},
    {"name": "IMERG V06",  "product": "imerg_v06",    "precip_col": "imerg_mm"},
    {"name": "IMERG V07",  "product": "imerg_v07",    "precip_col": "imerg_mm"},
    {"name": "CHIRPS",     "product": "chirps",       "precip_col": "chirps_mm"},
    {"name": "PERSIANN",   "product": "persiann_cdr", "precip_col": "persiann_mm"},
]

# Biomes instead of states
BIOMES_SHP = Path("/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/Biomes/Brazil_biomes.shp")

SHOW_BIOME_BOUNDARIES_ON_BIAS_MAPS = False
BIOME_EDGE_COLOR_MAIN = "white"
BIOME_EDGE_COLOR_UNDER = "black"
BIOME_EDGE_WIDTH_MAIN = 0.40
BIOME_EDGE_WIDTH_UNDER = 0.85

# Font
FONT_SIZE = 10
FONT_FAMILY = "Helvetica"

# Scatter/QC params matching the legacy figure
AX_MIN = 0
AX_MAX = 200
AX_TICKS = [0, 50, 100, 150, 200]

MIN_MM = 1.0
RATIO_CLIP = (0.1, 10.0)
NBINS = 100
DOT_SIZE = 5
ALPHA = 0.75
TICK_WIDTH = 2.0
BORDER_WIDTH = 2.0
DPI = 600
PCT_LIM = 95

# Bias map colorbar
BIAS_VMIN = 1.0
BIAS_VMAX = 3.5
BIAS_LEVELS = np.linspace(BIAS_VMIN, BIAS_VMAX, 11)
BIAS_CMAP = "nipy_spectral"

CBAR_WIDTH_FIG = 0.015
CBAR_RIGHT_PAD = 0.05
CBAR_HEIGHT_REL = 0.75
CBAR_TICK_STEP = 0.5

# Density colorbar geometry
DENSITY_CBAR_HEIGHT = 0.010
DENSITY_CBAR_Y = 0.025

# Same width as legacy, taller for 5 rows + bottom density colorbar
FIG_W = 6.3
FIG_H = 10.25

CONTOUR_LEVELS = np.arange(1.0, 3.5 + 1e-9, 0.5)

# Bias-map statistic label style
SHOW_BIAS_STAT_LABEL = True
BIAS_STAT_LABEL_DECIMALS = 2
BIAS_STAT_LABEL_FONT_SIZE = 8
BIAS_STAT_LABEL_FONT_WEIGHT = "bold"
BIAS_STAT_LABEL_BOX_ALPHA = 0.84


# ============================================================
# HELPERS
# ============================================================

def normalize_station_id(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def setup_font():
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    available = {f.name for f in fm.fontManager.ttflist}
    if FONT_FAMILY not in available:
        print(f"Warning: {FONT_FAMILY} not detected by Matplotlib. It will use the closest available fallback.")


def fit_origin(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        return np.nan, np.nan

    denom = np.sum(x * x)
    if denom <= 0:
        return np.nan, np.nan

    a = np.sum(x * y) / denom
    r2 = 1.0 - np.sum((y - a * x) ** 2) / np.sum(y ** 2)

    return a, r2


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(BORDER_WIDTH)

    ax.tick_params(width=TICK_WIDTH, length=6)
    ax.grid(True, ls=":", lw=0.6)


def bin_counts_for_points(x, y, xedges, yedges):
    H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))

    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, len(xedges) - 2)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, len(yedges) - 2)

    vals = H[ix, iy]
    return vals, H.max()


def panel_scatter_density(ax, x, y, norm, title, fit_color):
    xedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)
    yedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)

    vals, _ = bin_counts_for_points(x, y, xedges, yedges)

    sc = ax.scatter(
        x,
        y,
        c=vals,
        cmap="viridis",
        norm=norm,
        s=DOT_SIZE,
        alpha=ALPHA,
        edgecolors="none",
        rasterized=True,
    )

    ax.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], color="red", lw=2.0, linestyle="--")

    a, r2 = fit_origin(x, y)

    if np.isfinite(a):
        xx = np.linspace(AX_MIN, AX_MAX, 200)
        ax.plot(xx, a * xx, color=fit_color, lw=2.4)

        ax.text(
            0.04 * AX_MAX,
            0.90 * AX_MAX,
            rf"$y={a:.2f}x,\ R^2={r2:.2f}$",
            color=fit_color,
            fontsize=8,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
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
    ax.set_title(title, pad=8)

    style_axes(ax)
    return sc


def read_pairs(product, precip_col):
    pair_dir = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / PERCENTILE / "pairs"
    files = sorted(pair_dir.glob(f"pairs_{product}_{PERCENTILE}_*.csv"))

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
            raise ValueError(f"{product}: missing {precip_col} and product_mm")

    if "pr_g" not in df.columns:
        if "gauge_mm" in df.columns:
            df["pr_g"] = df["gauge_mm"]
        else:
            raise ValueError(f"{product}: missing pr_g and gauge_mm")

    df["station_id"] = normalize_station_id(df["station_id"])
    df["pr_g"] = pd.to_numeric(df["pr_g"], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")

    # Legacy plotting filter
    df = df[(df["pr_g"] >= MIN_MM) & (df[precip_col] >= MIN_MM)].copy()

    if "ratio" in df.columns:
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
        df = df[df["ratio"].between(*RATIO_CLIP)].copy()

    return df


def read_zeta(product, estimator):
    zeta_csv = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / PERCENTILE
        / "zeta_station"
        / estimator
        / f"zeta_per_station_{product}_{PERCENTILE}_{estimator}.csv"
    )

    if not zeta_csv.exists():
        raise FileNotFoundError(zeta_csv)

    z = pd.read_csv(zeta_csv, low_memory=False)
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


def load_product_arrays(product, precip_col, estimator):
    pairs = read_pairs(product, precip_col)
    zeta = read_zeta(product, estimator)

    df = pairs.merge(zeta, on="station_id", how="left").dropna(subset=["zeta"])

    if df.empty:
        raise ValueError(f"{product}: no pairs after merging zeta")

    df["product_corr"] = df[precip_col] * df["zeta"]

    return {
        "x": df["pr_g"].to_numpy(float),
        "y_raw": df[precip_col].to_numpy(float),
        "y_cor": df["product_corr"].to_numpy(float),
        "n": len(df),
    }


def find_zeta_tif(product, estimator):
    folder = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / PERCENTILE
        / "zeta_grid"
        / estimator
    )

    preferred = folder / f"zeta_map_{product}_{PERCENTILE}_{estimator}_idw_k8_p2p0.tif"

    if preferred.exists():
        return preferred

    files = sorted(folder.glob(f"zeta_map_{product}_{PERCENTILE}_{estimator}_idw_*.tif"))

    if files:
        return files[-1]

    raise FileNotFoundError(f"No zeta raster found in {folder}")


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

    return arr, extent


def plot_bias_map(ax, arr, extent, norm, cmap, biomes_gdf):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)

    if SHOW_BIOME_BOUNDARIES_ON_BIAS_MAPS and biomes_gdf is not None and not biomes_gdf.empty:
        # Draw a dark under-stroke and a white over-stroke for visibility over the zeta map.
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
    """
    Build readable log-scale ticks and force the last tick to be the actual
    maximum bin count, so the colorbar clearly reports the maximum density.
    """
    vmax = int(np.ceil(float(vmax)))
    ticks = [1]

    for t in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        if 1 < t < vmax:
            ticks.append(t)

    if vmax not in ticks:
        ticks.append(vmax)

    return ticks


def compute_raster_mean_std(data):
    """
    Compute spatial mean and standard deviation from valid raster pixels only.
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


def add_bias_stat_label(ax, mean_value, std_value):
    """
    Add mean and standard deviation label in the top-right corner
    using the same style as the reference map figure.
    """

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
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor="black",
            linewidth=0.65,
            alpha=BIAS_STAT_LABEL_BOX_ALPHA,
        ),
        zorder=20,
    )


def make_cbar_for_density(fig, ax_left, ax_right, norm_scatter):
    pos_left = ax_left.get_position()
    pos_right = ax_right.get_position()

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

    # Thinner bar, but stronger ticks and outline for readability.
    cb.ax.tick_params(
        which="major",
        length=4.5,
        width=1.4,
        labelsize=8,
        pad=2
    )
    cb.outline.set_linewidth(1.4)

    return cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", default="mean", choices=["mean", "median"])
    args = parser.parse_args()

    estimator = args.estimator

    setup_font()

    out_dir = PIPELINE_ROOT / "figures" / "bias_correction" / "legacy_style_5x3_helvetica_biomes" / PERCENTILE / estimator
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / f"density_raw_vs_corrected_and_biasmaps_5x3_{PERCENTILE}_{estimator}_biomes.png"
    out_pdf = out_dir / f"density_raw_vs_corrected_and_biasmaps_5x3_{PERCENTILE}_{estimator}_biomes.pdf"
    out_svg = out_dir / f"density_raw_vs_corrected_and_biasmaps_5x3_{PERCENTILE}_{estimator}_biomes.svg"

    biomes_gdf, brazil_geom, brazil_crs = load_biomes()

    results = []
    bias_maps = []

    for cfg in CONFIGS:
        print(f"Loading {cfg['name']} / {estimator}")

        arrs = load_product_arrays(cfg["product"], cfg["precip_col"], estimator)
        arrs["name"] = cfg["name"]
        arrs["product"] = cfg["product"]
        results.append(arrs)

        tif = find_zeta_tif(cfg["product"], estimator)
        arr, extent = load_bias_map(tif, brazil_geom, brazil_crs)
        bias_maps.append({"arr": arr, "extent": extent, "tif": tif})

        print(f"  pairs used: {arrs['n']}")
        print(f"  zeta map: {tif}")

    all_vals = []
    for r in results:
        all_vals.extend([r["x"], r["y_raw"], r["y_cor"]])

    concat_vals = np.concatenate(all_vals)
    lim = max(10.0, float(np.nanpercentile(concat_vals, PCT_LIM)))

    xedges = np.linspace(0, lim, NBINS + 1)
    yedges = np.linspace(0, lim, NBINS + 1)

    vmax = 2
    for r in results:
        _, vmax1 = bin_counts_for_points(r["x"], r["y_raw"], xedges, yedges)
        _, vmax2 = bin_counts_for_points(r["x"], r["y_cor"], xedges, yedges)
        vmax = max(vmax, vmax1, vmax2)

    norm_scatter = LogNorm(vmin=1, vmax=vmax)

    cmap_bias = plt.get_cmap(BIAS_CMAP).copy()
    cmap_bias.set_bad((0, 0, 0, 0))
    norm_bias = BoundaryNorm(BIAS_LEVELS, cmap_bias.N, clip=True)

    nrows = len(CONFIGS)

    fig, axs = plt.subplots(nrows, 3, figsize=(FIG_W, FIG_H), dpi=DPI)

    # Extra bottom margin for shared density colorbar.
    fig.subplots_adjust(wspace=0.15, hspace=0.26, bottom=0.085)

    right_col_axes = [axs[i, 2] for i in range(nrows)]
    first_map_mappable = None

    for i, (cfg, r, bm) in enumerate(zip(CONFIGS, results, bias_maps)):
        ax_raw = axs[i, 0]
        ax_cor = axs[i, 1]
        ax_map = axs[i, 2]

        is_top = i == 0
        is_bottom = i == nrows - 1

        panel_scatter_density(
            ax_raw,
            r["x"],
            r["y_raw"],
            norm_scatter,
            title="Raw" if is_top else "",
            fit_color="tab:gray",
        )

        panel_scatter_density(
            ax_cor,
            r["x"],
            r["y_cor"],
            norm_scatter,
            title="Bias-corrected" if is_top else "",
            fit_color="tab:blue",
        )

        # Only show y-axis tick labels on the first column.
        ax_cor.tick_params(axis="y", which="both", labelleft=False)

        # Only one y-axis title: bottom-left panel.
        ax_raw.set_ylabel("Product (mm)" if is_bottom else "")

        # Only one x-axis title: bottom-left panel.
        if is_bottom:
            ax_raw.set_xlabel("Rain Gauge (mm)")
            ax_cor.set_xlabel("")
            ax_raw.tick_params(axis="x", which="both", labelbottom=True)
            ax_cor.tick_params(axis="x", which="both", labelbottom=True)
        else:
            ax_raw.set_xlabel("")
            ax_cor.set_xlabel("")
            ax_raw.tick_params(axis="x", which="both", labelbottom=False)
            ax_cor.tick_params(axis="x", which="both", labelbottom=False)

        arr_show = np.array(bm["arr"], copy=True)
        arr_show = np.where(
            np.isfinite(arr_show),
            np.clip(arr_show, BIAS_VMIN, BIAS_VMAX),
            np.nan,
        )

        im = plot_bias_map(
            ax_map,
            arr_show,
            bm["extent"],
            norm_bias,
            cmap_bias,
            biomes_gdf,
        )

        ny, nx = arr_show.shape
        x = np.linspace(bm["extent"][0], bm["extent"][1], nx)
        y = np.linspace(bm["extent"][3], bm["extent"][2], ny)

        X, Y = np.meshgrid(x, y)
        Z = np.ma.masked_invalid(arr_show)

        try:
            cs = ax_map.contour(X, Y, Z, levels=CONTOUR_LEVELS, colors="k", linewidths=0.9)
            ax_map.clabel(cs, fmt="%.1f", inline=True, fontsize=7)
        except Exception:
            pass

        bias_mean, bias_std = compute_raster_mean_std(bm["arr"])

        ax_map.set_title(cfg["name"], pad=8)

        add_bias_stat_label(ax_map, bias_mean, bias_std)

        if first_map_mappable is None:
            first_map_mappable = im

    if first_map_mappable is not None:
        make_cbar_for_maps(fig, right_col_axes, first_map_mappable)

    make_cbar_for_density(fig, axs[-1, 0], axs[-1, 1], norm_scatter)

    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print("Saved:", out_svg)


if __name__ == "__main__":
    main()
