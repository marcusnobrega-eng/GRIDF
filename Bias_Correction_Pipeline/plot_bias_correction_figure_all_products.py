#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-style bias-correction figure for GRIDF pipeline outputs.

Rows:
  BR-DWGD
  IMERG V07
  IMERG V06
  CHIRPS
  PERSIANN-CDR

Columns:
  1) Raw event pairs: gauge vs product
  2) Bias-corrected event pairs: gauge vs zeta*product
  3) Spatial zeta map

This script uses the current Bias_Correction_Pipeline folder structure.
It can generate figures for either estimator:
  --estimator mean
  --estimator median
"""

from __future__ import annotations

import argparse
import os
import glob
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
# DEFAULT PATHS
# ============================================================

PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")

BRAZIL_ADM0 = Path(
    "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/ADMLevels/"
    "bra_admbnda_adm0_ibge_2020.shp"
)

BRAZIL_ADM1 = Path(
    "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/ADMLevels/"
    "bra_admbnda_adm1_ibge_2020.shp"
)

# You can replace this with the exact Avenir Next font path you used before.
# On many macOS systems, this exists:
DEFAULT_FONT_PATH = "/System/Library/Fonts/Avenir Next.ttc"


# ============================================================
# PRODUCTS
# ============================================================

PRODUCTS = [
    {
        "key": "br_dwgd",
        "label": "BR-DWGD",
    },
    {
        "key": "imerg_v07",
        "label": "IMERG V07",
    },
    {
        "key": "imerg_v06",
        "label": "IMERG V06",
    },
    {
        "key": "chirps",
        "label": "CHIRPS",
    },
    {
        "key": "persiann_cdr",
        "label": "PERSIANN-CDR",
    },
]


# ============================================================
# FIGURE SETTINGS
# ============================================================

AX_MIN = 0.0
AX_MAX = 200.0
AX_TICKS = [0, 50, 100, 150, 200]

MIN_MM = 1.0
MAX_PAIR_MM = 350.0
RATIO_CLIP = (0.25, 5.0)

NBINS = 100
DOT_SIZE = 4.5
ALPHA = 0.75

TICK_WIDTH = 1.5
BORDER_WIDTH = 1.5
FONT_SIZE = 8.5
DPI = 600

SCATTER_CMAP = "viridis"

BIAS_VMIN = 1.0
BIAS_VMAX = 3.5
BIAS_LEVELS = np.linspace(BIAS_VMIN, BIAS_VMAX, 11)
BIAS_CMAP = "nipy_spectral"

CBAR_WIDTH_FIG = 0.014
CBAR_RIGHT_PAD = 0.035
CBAR_HEIGHT_REL = 0.76
CBAR_TICK_STEP = 0.5

CONTOUR_LEVELS = np.arange(1.0, 3.5 + 1e-9, 0.5)

COUNTRY_EDGE_COLOR = "black"
COUNTRY_EDGE_WIDTH = 0.6
STATE_EDGE_COLOR = "white"
STATE_EDGE_WIDTH = 0.35

FIG_W = 7.2
FIG_H = 10.2

FIT_WITHIN_AXIS_LIMITS = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def register_font(font_path: str | None) -> None:
    if font_path and Path(font_path).exists():
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass

    plt.rcParams["font.size"] = FONT_SIZE

    available = {f.name for f in fm.fontManager.ttflist}

    if "Avenir Next" in available:
        plt.rcParams["font.family"] = "Avenir Next"
    elif "Avenir" in available:
        plt.rcParams["font.family"] = "Avenir"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def read_csv_safely(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def find_existing_zeta_tif(product: str, percentile: str, estimator: str) -> Path:
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

    matches = sorted(folder.glob(f"zeta_map_{product}_{percentile}_{estimator}_idw_*.tif"))
    if matches:
        return matches[-1]

    raise FileNotFoundError(f"No zeta raster found in: {folder}")


def find_pair_qc_or_pairs(product: str, percentile: str) -> pd.DataFrame:
    base = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / percentile

    pair_qc = base / "tables" / f"pair_qc_{product}_{percentile}.csv"

    if pair_qc.exists():
        df = read_csv_safely(pair_qc)
        df["source_table"] = str(pair_qc)
        return df

    pairs_dir = base / "pairs"
    pair_files = sorted(pairs_dir.glob(f"pairs_{product}_{percentile}_*.csv"))

    if not pair_files:
        raise FileNotFoundError(f"No pair files found for {product} in {pairs_dir}")

    dfs = []
    for f in pair_files:
        d = read_csv_safely(f)
        d["source_table"] = str(f)
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True)


def detect_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            "Could not find required column. Tried: "
            + ", ".join(candidates)
            + "\nAvailable columns:\n"
            + ", ".join(df.columns)
        )

    return None


def load_zeta_table(product: str, percentile: str, estimator: str) -> pd.DataFrame:
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

    zeta = read_csv_safely(zeta_csv)
    zeta["station_id"] = zeta["station_id"].astype(str)

    zeta_col = detect_column(zeta, ["zeta_selected", "zeta", "ratio_mean", "ratio_median"])
    zeta = zeta[["station_id", zeta_col]].rename(columns={zeta_col: "zeta"})

    return zeta


def load_product_arrays(product: str, percentile: str, estimator: str) -> dict:
    pairs = find_pair_qc_or_pairs(product, percentile)

    station_col = detect_column(pairs, ["station_id", "Code", "code"])
    gauge_col = detect_column(pairs, ["gauge_mm", "pr_g", "rain_gauge_mm", "station_mm"])
    product_col = detect_column(
        pairs,
        [
            "product_mm",
            "imerg_mm",
            "chirps_mm",
            "persiann_mm",
            "xavier_pr_mm",
            "br_dwgd_mm",
            "precipitation",
        ],
    )
    ratio_col = detect_column(
        pairs,
        ["ratio_for_zeta", "ratio_gauge_over_product", "raw_ratio_gauge_over_product", "ratio"],
        required=False,
    )

    df = pairs.copy()
    df["station_id"] = df[station_col].astype(str)
    df["gauge_mm"] = pd.to_numeric(df[gauge_col], errors="coerce")
    df["product_mm"] = pd.to_numeric(df[product_col], errors="coerce")

    if ratio_col is not None:
        df["ratio"] = pd.to_numeric(df[ratio_col], errors="coerce")
    else:
        df["ratio"] = df["gauge_mm"] / df["product_mm"]

    if "qc_used_for_zeta" in df.columns:
        df = df[as_bool_series(df["qc_used_for_zeta"])].copy()
    else:
        df = df[
            df["gauge_mm"].between(MIN_MM, MAX_PAIR_MM)
            & df["product_mm"].between(MIN_MM, MAX_PAIR_MM)
            & df["ratio"].between(RATIO_CLIP[0], RATIO_CLIP[1])
        ].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["station_id", "gauge_mm", "product_mm", "ratio"])

    if df.empty:
        raise ValueError(f"No valid plotting pairs after QC for {product}")

    zeta = load_zeta_table(product, percentile, estimator)
    df = df.merge(zeta, on="station_id", how="inner")

    if df.empty:
        raise ValueError(f"No valid plotting pairs after merging zeta for {product}")

    df["product_corr"] = df["product_mm"] * df["zeta"]

    return {
        "x": df["gauge_mm"].to_numpy(dtype=float),
        "y_raw": df["product_mm"].to_numpy(dtype=float),
        "y_cor": df["product_corr"].to_numpy(dtype=float),
        "n": len(df),
    }


def fit_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if FIT_WITHIN_AXIS_LIMITS:
        m &= (x >= AX_MIN) & (x <= AX_MAX) & (y >= AX_MIN) & (y <= AX_MAX)

    x = x[m]
    y = y[m]

    if x.size == 0 or np.sum(x * x) <= 0:
        return np.nan, np.nan

    a = np.sum(x * y) / np.sum(x * x)
    r2 = 1.0 - np.sum((y - a * x) ** 2) / np.sum(y ** 2)

    return float(a), float(r2)


def style_scatter_axes(ax) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(BORDER_WIDTH)
    ax.tick_params(width=TICK_WIDTH, length=4.5)
    ax.grid(True, ls=":", lw=0.45, alpha=0.7)


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

    ax.scatter(
        x,
        y,
        c=vals,
        cmap=SCATTER_CMAP,
        norm=norm,
        s=DOT_SIZE,
        alpha=ALPHA,
        edgecolors="none",
        rasterized=True,
    )

    ax.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], color="red", lw=1.5)

    a, r2 = fit_origin(x, y)
    if np.isfinite(a):
        xx = np.linspace(AX_MIN, AX_MAX, 200)
        ax.plot(xx, a * xx, color=fit_color, lw=1.8)
        ax.text(
            0.04 * AX_MAX,
            0.89 * AX_MAX,
            rf"$y={a:.2f}x$" + "\n" + rf"$R^2={r2:.2f}$",
            color=fit_color,
            fontsize=7,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor=fit_color,
                alpha=0.88,
            ),
        )

    ax.set_xlim(AX_MIN, AX_MAX)
    ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_xticks(AX_TICKS)
    ax.set_yticks(AX_TICKS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=5, fontsize=8.5)
    style_scatter_axes(ax)


def load_boundaries():
    country_gdf = None
    states_gdf = None

    if BRAZIL_ADM0.exists():
        country_gdf = gpd.read_file(BRAZIL_ADM0)
        country_gdf = country_gdf.set_crs(4326) if country_gdf.crs is None else country_gdf.to_crs(4326)

    if BRAZIL_ADM1.exists():
        states_gdf = gpd.read_file(BRAZIL_ADM1)
        states_gdf = states_gdf.set_crs(4326) if states_gdf.crs is None else states_gdf.to_crs(4326)

    return country_gdf, states_gdf


def load_bias_map(tif_path: Path, country_gdf: gpd.GeoDataFrame | None):
    with rasterio.open(tif_path) as ds:
        if country_gdf is not None:
            tmp = country_gdf.to_crs(ds.crs)
            geom = tmp.dissolve().geometry.iloc[0]
            data, _ = mask(ds, [geom], crop=False, filled=True, invert=False, nodata=np.nan)
            arr = data[0].astype("float32")
        else:
            arr = ds.read(1, masked=True).astype("float32")
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)

        nod = ds.nodata
        if nod is not None and np.isfinite(nod):
            arr = np.where(arr == nod, np.nan, arr)

        bounds = ds.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    return arr, extent


def plot_bias_map(ax, arr, extent, norm, cmap, country_gdf, states_gdf):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm)

    if states_gdf is not None and not states_gdf.empty:
        states_gdf.boundary.plot(ax=ax, edgecolor=STATE_EDGE_COLOR, linewidth=STATE_EDGE_WIDTH)

    if country_gdf is not None and not country_gdf.empty:
        country_gdf.boundary.plot(ax=ax, edgecolor=COUNTRY_EDGE_COLOR, linewidth=COUNTRY_EDGE_WIDTH)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)

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
    cb.ax.tick_params(which="major", length=5, width=1.0)
    cb.ax.tick_params(which="minor", length=2.5, width=0.7)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.set_label(r"Bias factor, $\zeta$", rotation=90, labelpad=11)
    cb.outline.set_linewidth(1.2)

    return cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["mean", "median"], required=True)
    parser.add_argument("--percentile", default="p98")
    parser.add_argument("--font", default=os.environ.get("AVENIR_NEXT_FONT", DEFAULT_FONT_PATH))
    parser.add_argument("--outdir", default=str(PIPELINE_ROOT / "figures" / "bias_correction"))
    args = parser.parse_args()

    percentile = args.percentile
    estimator = args.estimator

    register_font(args.font)

    outdir = Path(args.outdir) / percentile / estimator
    outdir.mkdir(parents=True, exist_ok=True)

    out_png = outdir / f"bias_correction_raw_corrected_zeta_{percentile}_{estimator}.png"
    out_pdf = outdir / f"bias_correction_raw_corrected_zeta_{percentile}_{estimator}.pdf"

    country_gdf, states_gdf = load_boundaries()

    results = []
    maps = []

    for prod in PRODUCTS:
        key = prod["key"]
        label = prod["label"]

        print(f"Loading {label} / {percentile} / {estimator}")

        arrays = load_product_arrays(key, percentile, estimator)
        arrays["key"] = key
        arrays["label"] = label
        results.append(arrays)

        zeta_tif = find_existing_zeta_tif(key, percentile, estimator)
        arr, extent = load_bias_map(zeta_tif, country_gdf)
        maps.append({"arr": arr, "extent": extent, "path": zeta_tif})

        print(f"  pairs used: {arrays['n']}")
        print(f"  zeta map:   {zeta_tif}")

    xedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)
    yedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)

    vmax = 2
    for r in results:
        _, vmax1 = bin_counts_for_points(r["x"], r["y_raw"], xedges, yedges)
        _, vmax2 = bin_counts_for_points(r["x"], r["y_cor"], xedges, yedges)
        vmax = max(vmax, vmax1, vmax2)

    norm_scatter = LogNorm(vmin=1, vmax=vmax)

    cmap_bias = plt.get_cmap(BIAS_CMAP).copy()
    cmap_bias.set_bad((0, 0, 0, 0))
    norm_bias = BoundaryNorm(BIAS_LEVELS, cmap_bias.N, clip=True)

    nrows = len(PRODUCTS)
    fig, axs = plt.subplots(nrows, 3, figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.subplots_adjust(wspace=0.15, hspace=0.28)

    right_col_axes = [axs[i, 2] for i in range(nrows)]
    first_map_mappable = None

    for i, (prod, r, bm) in enumerate(zip(PRODUCTS, results, maps)):
        label = prod["label"]

        ax_raw = axs[i, 0]
        ax_cor = axs[i, 1]
        ax_map = axs[i, 2]

        panel_scatter_density(
            ax_raw,
            r["x"],
            r["y_raw"],
            norm_scatter,
            title="Raw",
            fit_color="0.25",
        )

        panel_scatter_density(
            ax_cor,
            r["x"],
            r["y_cor"],
            norm_scatter,
            title="Bias-corrected",
            fit_color="tab:blue",
        )

        ax_raw.set_ylabel("Product (mm)")

        if i == nrows - 1:
            ax_raw.set_xlabel("Rain gauge (mm)")
            ax_cor.set_xlabel("Rain gauge (mm)")
        else:
            ax_raw.tick_params(axis="x", labelbottom=False)
            ax_cor.tick_params(axis="x", labelbottom=False)

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
            country_gdf,
            states_gdf,
        )

        ny, nx = arr_show.shape
        x = np.linspace(bm["extent"][0], bm["extent"][1], nx)
        y = np.linspace(bm["extent"][3], bm["extent"][2], ny)
        X, Y = np.meshgrid(x, y)
        Z = np.ma.masked_invalid(arr_show)

        try:
            cs = ax_map.contour(X, Y, Z, levels=CONTOUR_LEVELS, colors="k", linewidths=0.55)
            ax_map.clabel(cs, fmt="%.1f", inline=True, fontsize=6)
        except Exception:
            pass

        ax_map.set_title(label, pad=5, fontsize=8.5)

        if first_map_mappable is None:
            first_map_mappable = im

        ax_raw.text(
            -0.18,
            1.06,
            f"{chr(97 + i * 3)})",
            transform=ax_raw.transAxes,
            fontsize=9,
            weight="bold",
            va="bottom",
            ha="left",
        )
        ax_cor.text(
            -0.18,
            1.06,
            f"{chr(97 + i * 3 + 1)})",
            transform=ax_cor.transAxes,
            fontsize=9,
            weight="bold",
            va="bottom",
            ha="left",
        )
        ax_map.text(
            -0.08,
            1.06,
            f"{chr(97 + i * 3 + 2)})",
            transform=ax_map.transAxes,
            fontsize=9,
            weight="bold",
            va="bottom",
            ha="left",
        )

    if first_map_mappable is not None:
        make_cbar_for_maps(fig, right_col_axes, first_map_mappable)

    fig.suptitle(
        f"Bias correction diagnostics ({percentile.upper()}, {estimator})",
        y=0.995,
        fontsize=10,
        weight="bold",
    )

    fig.savefig(out_png, bbox_inches="tight", dpi=DPI)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
