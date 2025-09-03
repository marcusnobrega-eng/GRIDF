# 04_apply_bias_correction_and_density_plots_ALL_4x3.py
# ---------------------------------------------------------
# 4×3 figure:
#   Rows: BR-DWGD (Xavier), IMERG, CHIRPS, PERSIANN
#   Col 1: Raw (x=gauge, y=product)
#   Col 2: Bias-corrected (x=gauge, y=zeta*product)
#   Col 3: Spatial bias (zeta map) with Brazil states overlay + single colorbar
#
# Reqs: pandas numpy matplotlib geopandas rasterio shapely
# ---------------------------------------------------------

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
import matplotlib.font_manager as fm
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# ---------------- USER INPUTS ----------------
CONFIGS = [
    {
        "name": "BR-DWGD",
        "pairs_dir": r'G:\My Drive\xavier_bias_pairs',
        "pairs_glob": "xavier_bias_pairs_*.csv",
        "zeta_csv": r'G:\My Drive\xavier_bias_pairs\zeta_per_station.csv',
        "zeta_tif": r'G:\My Drive\xavier_bias_pairs\zeta_map_010deg.tif',
    },
    {
        "name": "IMERG",
        "pairs_dir": r'G:\My Drive\imerg_bias_pairs',
        "pairs_glob": "imerg_bias_pairs_*.csv",
        "zeta_csv": r'G:\My Drive\imerg_bias_pairs\zeta_per_station.csv',
        "zeta_tif": r'G:\My Drive\imerg_bias_pairs\zeta_map_010deg.tif',
    },
    {
        "name": "CHIRPS",
        "pairs_dir": r'G:\My Drive\chirps_bias_pairs',
        "pairs_glob": "chirps_bias_pairs_*.csv",
        "zeta_csv": r'G:\My Drive\chirps_bias_pairs\zeta_per_station.csv',
        "zeta_tif": r'G:\My Drive\chirps_bias_pairs\zeta_map_005deg.tif',
    },
    {
        "name": "PERSIANN",
        "pairs_dir": r'G:\My Drive\persiann_bias_pairs',
        "pairs_glob": "persiann_bias_pairs_*.csv",
        "zeta_csv": r'G:\My Drive\persiann_bias_pairs\zeta_per_station.csv',
        "zeta_tif": r'G:\My Drive\persiann_bias_pairs\zeta_map_025deg.tif',
    },
]

OUT_PNG = r'G:\My Drive\multi_product_bias_plots\density_raw_vs_corrected_and_biasmaps_ALL_4x3.png'

# Fixed scatter-axis setup (x & y)
AX_MIN = 0
AX_MAX = 200
AX_TICKS = [0, 50, 100, 150, 200]


# Spatial overlay (Brazil states)
STATES_SHP = r'C:\Users\marcu\OneDrive - University of Arizona\Desktop\Desktop_Folder\ANA_Analysis\Rainfall_Distribution\Brasil_Shapefile\brazil_Brazil_States_Boundary.shp'
STATES_EDGE_COLOR = 'white'
STATES_EDGE_WIDTH = 0.6

# Font
FONT_PATH_MONTSERRAT = 'C:/Users/marcu/PycharmProjects/Sucept_Maps/Montserrat/Montserrat-Regular.ttf'
FONT_SIZE = 10

# Scatter/QC params (shared)
MIN_MM = 1.0
RATIO_CLIP = (0.1, 10.0)
NBINS = 100
DOT_SIZE = 5
ALPHA = 0.75
TICK_WIDTH = 2.0
BORDER_WIDTH = 2.0            # thicker borders for SCATTER plots only
DPI = 600
PCT_LIM = 95

# Bias map colormap (single bar for all maps)
# Personalized colorbar range: 1.0 → 4.0
BIAS_VMIN, BIAS_VMAX = 1, 3.5
BIAS_LEVELS = np.linspace(BIAS_VMIN, BIAS_VMAX, 11)  # 0.1 step
BIAS_CMAP = 'nipy_spectral'  # try 'magma' or 'viridis' if you prefer

# Personalized colorbar geometry & ticks
CBAR_WIDTH_FIG   = 0.015   # wider bar
CBAR_RIGHT_PAD   = 0.05    # push farther right
CBAR_HEIGHT_REL  = 0.75    # relative height of the stack of map axes
CBAR_TICK_STEP   = 0.5     # major tick step (used in helper)

# Figure size
FIG_W, FIG_H = 10.2, 10.6
FIG_W, FIG_H = 6.3, 8

# Map contour settings
CONTOUR_LEVELS = np.arange(1.0, 3.5 + 1e-9, 0.5)  # 1.0, 2.0, 3.0

# --------------------------------------------

# ---- helpers ----
def detect_precip_col(df: pd.DataFrame) -> str:
    for c in ('xavier_pr_mm','imerg_mm','chirps_mm','persiann_mm'):
        if c in df.columns:
            return c
    raise ValueError("No precip column found (expected one of: xavier_pr_mm, imerg_mm, chirps_mm, persiann_mm).")

def product_label_from_col(col: str) -> str:
    return {'xavier_pr_mm':'BR-DWGD', 'imerg_mm':'IMERG', 'chirps_mm':'CHIRPS', 'persiann_mm':'PERSIANN'}.get(col, col)

def fit_origin(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0: return np.nan, np.nan
    a  = np.sum(x*y) / np.sum(x*x)
    r2 = 1.0 - np.sum((y - a*x)**2) / np.sum(y**2)
    return a, r2

def try_register_montserrat(font_path: str):
    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams['font.size'] = FONT_SIZE
    if any(f.name.lower().startswith("montserrat") for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Montserrat'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'

def style_axes(ax):
    # apply thicker spines & ticks (for scatter plots)
    for sp in ax.spines.values():
        sp.set_linewidth(BORDER_WIDTH)
    ax.tick_params(width=TICK_WIDTH, length=6)
    ax.grid(True, ls=':', lw=0.6)

def bin_counts_for_points(x, y, xedges, yedges):
    H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
    ix = np.clip(np.searchsorted(xedges, x, side='right') - 1, 0, len(xedges)-2)
    iy = np.clip(np.searchsorted(yedges, y, side='right') - 1, 0, len(yedges)-2)
    vals = H[ix, iy]
    return vals, H.max()

def panel_scatter_density(ax, x, y, lim, norm, title, fit_color):
    # binning uses the fixed 0..200 range so density colors are comparable
    xedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)
    yedges = np.linspace(AX_MIN, AX_MAX, NBINS + 1)
    vals, _ = bin_counts_for_points(x, y, xedges, yedges)

    ax.scatter(x, y, c=vals, cmap='viridis', norm=norm,
               s=DOT_SIZE, alpha=ALPHA, edgecolors='none')

    # 45° (1:1) line locked to 0..200
    ax.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], color='red', lw=2.0, linestyle='-')

    # fit through origin and draw within 0..200
    a, r2 = fit_origin(x, y)
    if np.isfinite(a):
        xx = np.linspace(AX_MIN, AX_MAX, 200)
        ax.plot(xx, a * xx, color=fit_color, lw=2.4)
        ax.text(0.04 * AX_MAX, 0.90 * AX_MAX, rf"$y={a:.2f}x,\ R^2={r2:.2f}$",
                color=fit_color, fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=fit_color, alpha=0.85))

    # fixed limits & ticks
    ax.set_xlim(AX_MIN, AX_MAX)
    ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_xticks(AX_TICKS)
    ax.set_yticks(AX_TICKS)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, pad=8)
    style_axes(ax)



def load_product_arrays(pairs_dir, pairs_glob, zeta_csv):
    files = sorted(glob.glob(os.path.join(pairs_dir, pairs_glob)))
    if not files:
        return None
    pairs = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in files], ignore_index=True)

    precip_col = detect_precip_col(pairs)
    need = {'station_id','pr_g', precip_col}
    if not need.issubset(pairs.columns):
        raise ValueError(f"[{pairs_dir}] Pairs missing columns. Need {need}, got {pairs.columns.tolist()}")
    pairs['station_id'] = pairs['station_id'].astype(str)
    pairs = pairs[(pairs['pr_g'] >= MIN_MM) & (pairs[precip_col] >= MIN_MM)]
    if 'ratio' in pairs.columns:
        pairs = pairs[pairs['ratio'].between(*RATIO_CLIP)]
    if pairs.empty:
        return None

    zeta = pd.read_csv(zeta_csv)
    if 'station_id' not in zeta.columns or 'zeta' not in zeta.columns:
        raise ValueError(f"[{zeta_csv}] must contain: station_id, zeta")
    zeta['station_id'] = zeta['station_id'].astype(str)
    df = pairs.merge(zeta[['station_id','zeta']], on='station_id', how='left').dropna(subset=['zeta'])
    if df.empty:
        return None

    df['product_corr'] = df[precip_col] * df['zeta']
    return {"label": product_label_from_col(precip_col),
            "x": df['pr_g'].to_numpy(),
            "y_raw": df[precip_col].to_numpy(),
            "y_cor": df['product_corr'].to_numpy()}

def load_bias_map(tif_path, brazil_geom=None, brazil_crs=None):
    if not tif_path or not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as ds:
        if brazil_geom is not None:
            gdf_tmp = gpd.GeoDataFrame(geometry=[brazil_geom],
                                       crs=brazil_crs if brazil_crs is not None else 4326)
            gdf_tmp = gdf_tmp.to_crs(ds.crs)
            shapes = [gdf_tmp.geometry.iloc[0]]
            data, _ = mask(ds, shapes, crop=False, filled=True, invert=False, nodata=np.nan)
            arr = data[0].astype('float32')
        else:
            arr = ds.read(1, masked=True).astype('float32')
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            nod = ds.nodata
            if nod is not None and not np.isnan(nod):
                arr = np.where(arr == nod, np.nan, arr)
        bounds = ds.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return arr, extent

def plot_bias_map(ax, arr, extent, norm, cmap, states_gdf):
    im = ax.imshow(arr, extent=extent, origin='upper', cmap=cmap, norm=norm)
    if states_gdf is not None and not states_gdf.empty:
        states_gdf.boundary.plot(ax=ax, edgecolor=STATES_EDGE_COLOR, linewidth=STATES_EDGE_WIDTH)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)  # keep maps borderless
    ax.set_frame_on(False)
    ax.set_facecolor('none')  # transparent outside Brazil
    return im

def make_cbar_for_maps(fig, right_col_axes, mappable):
    # compute the stacked height of the map column
    rights, bottoms, tops = [], [], []
    for ax in right_col_axes:
        pos = ax.get_position()
        rights.append(pos.x1); bottoms.append(pos.y0); tops.append(pos.y1)
    x_right = max(rights)
    y0 = min(bottoms); y1 = max(tops)
    stack_h = y1 - y0

    # colorbar axes position
    cbar_h = stack_h * CBAR_HEIGHT_REL
    cbar_y = y0 + 0.5 * (stack_h - cbar_h)
    cbar_x = x_right + CBAR_RIGHT_PAD
    cbar_ax = fig.add_axes([cbar_x, cbar_y, CBAR_WIDTH_FIG, cbar_h])

    cb = fig.colorbar(
        mappable, cax=cbar_ax,
        extend='both',        # triangles at both ends
        extendfrac=0.05,      # triangle length
        boundaries=BIAS_LEVELS, spacing='proportional'
    )

    # ticks on the RIGHT only, majors every 0.5, minors every 0.25
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.tick_params(left=False, right=True, labelleft=False, labelright=True)
    cb.set_ticks(np.arange(BIAS_VMIN, BIAS_VMAX + 1e-9, CBAR_TICK_STEP))
    cb.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    cb.ax.tick_params(which='major', length=6, width=1.2)
    cb.ax.tick_params(which='minor', length=3, width=0.8)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    cb.set_label("Bias factor", rotation=90, labelpad=12, va='center')
    cb.outline.set_linewidth(1.5)
    return cb

def main():
    # Fonts
    try_register_montserrat(FONT_PATH_MONTSERRAT)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    # Brazil states (overlay + mask)
    states_gdf = None
    brazil_geom = None
    brazil_crs = None
    if STATES_SHP and os.path.exists(STATES_SHP):
        gdf = gpd.read_file(STATES_SHP)
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
        states_gdf = gdf
        brazil_geom = gdf.dissolve().geometry.iloc[0]
        brazil_crs = gdf.crs

    # Load products and bias maps
    results, bias_maps = [], []
    for cfg in CONFIGS:
        res = load_product_arrays(cfg["pairs_dir"], cfg["pairs_glob"], cfg["zeta_csv"])
        if res is None:
            results.append({"label": cfg["name"], "x": None, "y_raw": None, "y_cor": None, "missing": True})
        else:
            res["missing"] = False
            results.append(res)

        bm = load_bias_map(cfg.get("zeta_tif", ""), brazil_geom, brazil_crs)
        bias_maps.append({"arr": None, "extent": None} if bm is None else {"arr": bm[0], "extent": bm[1]})

    # Shared limits & density norm
    all_vals = []
    for r in results:
        if not r.get("missing"):
            all_vals.extend([r["x"], r["y_raw"], r["y_cor"]])
    if not all_vals:
        raise FileNotFoundError("No data found for any product. Check your paths and globs.")
    concat_vals = np.concatenate(all_vals)
    lim = max(10.0, float(np.nanpercentile(concat_vals, PCT_LIM)))

    xedges = np.linspace(0, lim, NBINS + 1)
    yedges = np.linspace(0, lim, NBINS + 1)
    vmax = 2
    for r in results:
        if r.get("missing"): continue
        _, vmax1 = bin_counts_for_points(r["x"], r["y_raw"], xedges, yedges)
        _, vmax2 = bin_counts_for_points(r["x"], r["y_cor"], xedges, yedges)
        vmax = max(vmax, vmax1, vmax2)
    norm_scatter = LogNorm(vmin=1, vmax=vmax)

    # Bias map color normalization (shared)
    cmap_bias = plt.get_cmap(BIAS_CMAP).copy()
    cmap_bias.set_bad((0, 0, 0, 0))  # transparent outside Brazil
    norm_bias = BoundaryNorm(BIAS_LEVELS, cmap_bias.N, clip=True)

    # Figure
    fig, axs = plt.subplots(4, 3, figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.subplots_adjust(wspace=0.15, hspace=0.26)
    right_col_axes = [axs[i, 2] for i in range(4)]
    first_map_mappable = None

    # Rows
    for i, r in enumerate(results):
        ax_raw, ax_cor = axs[i, 0], axs[i, 1]
        label = r["label"]

        if r.get("missing"):
            for ax in (ax_raw, ax_cor):
                ax.text(0.5, 0.5, f"No pairs for {label}", ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            axs[i, 2].text(0.5, 0.5, f"No ζ map for {label}", ha='center', va='center', transform=axs[i, 2].transAxes)
            axs[i, 2].set_axis_off()
            continue

        panel_scatter_density(ax_raw, r["x"], r["y_raw"], lim, norm_scatter,
                              title=f"Raw ", fit_color='tab:gray')
        panel_scatter_density(ax_cor, r["x"], r["y_cor"], lim, norm_scatter,
                              title=f"Bias-corrected", fit_color='tab:blue')

        # Labels
        ax_raw.set_ylabel("Product (mm)")

        # --- show Rain Gauge label + tick labels only on the bottom row ---
        is_bottom = (i == len(results) - 1)

        if is_bottom:
            ax_raw.set_xlabel("Rain Gauge (mm)")
            ax_cor.set_xlabel("Rain Gauge (mm)")
            # (keep tick labels visible on the bottom row)
            ax_raw.tick_params(axis='x', which='both', labelbottom=True)
            ax_cor.tick_params(axis='x', which='both', labelbottom=True)
        else:
            # remove x-axis titles and tick labels on the upper rows
            ax_raw.set_xlabel("")
            ax_cor.set_xlabel("")
            ax_raw.tick_params(axis='x', which='both', labelbottom=False)
            ax_cor.tick_params(axis='x', which='both', labelbottom=False)
        # --- end bottom-only x labeling ---

        ax_raw.set_ylabel("Product (mm)")
        if i == len(results) - 1:
            ax_raw.set_xlabel("Rain Gauge (mm)")
            ax_cor.set_xlabel("Rain Gauge (mm)")

        # Map
        ax_map = axs[i, 2]
        arr = bias_maps[i]["arr"]; extent = bias_maps[i]["extent"]
        if arr is None or extent is None:
            ax_map.text(0.5, 0.5, f"No ζ map for {label}", ha='center', va='center', transform=ax_map.transAxes)
            ax_map.set_axis_off()
        else:
            arr_show = np.array(arr, copy=True)
            arr_show = np.where(np.isfinite(arr_show),
                                np.clip(arr_show, BIAS_VMIN, BIAS_VMAX),
                                np.nan)
            im = plot_bias_map(ax_map, arr_show, extent, norm_bias, cmap_bias, states_gdf)

            # Contours at 1.0, 2.0, 3.0
            ny, nx = arr_show.shape
            x = np.linspace(extent[0], extent[1], nx)
            y = np.linspace(extent[3], extent[2], ny)  # origin='upper'
            X, Y = np.meshgrid(x, y)
            Z = np.ma.masked_invalid(arr_show)
            cs = ax_map.contour(X, Y, Z, levels=CONTOUR_LEVELS, colors='k', linewidths=0.9)
            ax_map.clabel(cs, fmt='%.1f', inline=True, fontsize=7)

            if first_map_mappable is None:
                first_map_mappable = im
            ax_map.set_title(f"{label}", pad=8)

    # Shared colorbar
    if first_map_mappable is not None:
        make_cbar_for_maps(fig, right_col_axes, first_map_mappable)

    fig.savefig(OUT_PNG, bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
