# 05c_zeta_perpair_vs_elevation_from_pairs_2x2.py
# ---------------------------------------------------------
# Use ALL per-pair bias factors (zeta) from the same folders/globs
# used in the 4x3 bias-correction script, relate them to elevation,
# and fit: zeta = a * exp(b * h/1000).
#
# Outputs: 2x2 PNG with one panel per product.
#
# Reqs: pandas numpy matplotlib geopandas rasterio shapely
# ---------------------------------------------------------

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.font_manager as fm
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import Point

# ---------------- USER INPUTS (same spirit as 04_... script) ----------------
CONFIGS = [
    {
        "name": "BR-DWGD",
        "pairs_dir": r'G:\My Drive\xavier_bias_pairs',
        "pairs_glob": "xavier_bias_pairs_*.csv",
        "precip_col_candidates": ["xavier_pr_mm", "xavier_mm", "xavier"],
    },
    {
        "name": "IMERG",
        "pairs_dir": r'G:\My Drive\imerg_bias_pairs',
        "pairs_glob": "imerg_bias_pairs_*.csv",
        "precip_col_candidates": ["imerg_mm", "imerg"],
    },
    {
        "name": "CHIRPS",
        "pairs_dir": r'G:\My Drive\chirps_bias_pairs',
        "pairs_glob": "chirps_bias_pairs_*.csv",
        "precip_col_candidates": ["chirps_mm", "chirps"],
    },
    {
        "name": "PERSIANN",
        "pairs_dir": r'G:\My Drive\persiann_bias_pairs',
        "pairs_glob": "persiann_bias_pairs_*.csv",
        "precip_col_candidates": ["persiann_mm", "persiann"],
    },
]

# DEM (0.1° MERIT in lat/lon; will reproject if different)
ELEV_TIF = r'G:\My Drive\multi_product_bias_plots\merit_dem_brazil_0p1deg.tif'

OUT_PNG = r'G:\My Drive\multi_product_bias_plots\zeta_perpair_vs_elevation_2x2.png'

# ---- Keep QC/visuals consistent with the 4×3 script ----
MIN_MM = 1.0
RATIO_CLIP = (0.1, 10.0)
NBINS = 140
DOT_SIZE = 6
ALPHA = 0.75
TICK_WIDTH = 2.0
BORDER_WIDTH = 2.0
DPI = 600
FONT_PATH_MONTSERRAT = r'C:/Users/marcu/PycharmProjects/Sucept_Maps/Montserrat/Montserrat-Regular.ttf'
FONT_SIZE = 10

# Plot ranges
ZETA_YMIN, ZETA_YMAX = 0.5, 3.5
X_TICK_STEP = 500  # meters

# If your 'ratio' column is product/gauge (instead of gauge/product), flip it:
RATIO_IS_PRODUCT_OVER_GAUGE = False
# ---------------------------------------------------------------------------


def try_register_montserrat(path):
    if path and os.path.exists(path):
        try:
            fm.fontManager.addfont(path)
        except Exception:
            pass
    plt.rcParams['font.size'] = FONT_SIZE
    if any(f.name.lower().startswith("montserrat") for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Montserrat'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(BORDER_WIDTH)
    ax.tick_params(width=TICK_WIDTH, length=6)
    ax.grid(True, ls=':', lw=0.7)


def guess_lat_lon_columns(df):
    lat_candidates = ['lat','latitude','Lat','Latitude','y','lat_g','y_g','LAT','LATITUDE','Latitude_dd']
    lon_candidates = ['lon','longitude','Lon','Longitude','x','lon_g','x_g','LON','LONGITUDE','Longitude_dd']
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col


def find_precip_col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def load_all_pairs(pairs_dir, pairs_glob, precip_candidates):
    """Load ALL per-pair rows; compute zeta; apply QC; return pairs with optional lat/lon."""
    files = sorted(glob.glob(os.path.join(pairs_dir, pairs_glob)))
    if not files:
        return pd.DataFrame()

    keep = []
    for f in files:
        # 1) Identify product precip column
        try:
            head = pd.read_csv(f, nrows=2000, low_memory=False)
        except Exception:
            continue
        pcol = find_precip_col(head, precip_candidates)
        if pcol is None:
            try:
                head = pd.read_csv(f, low_memory=False)
                pcol = find_precip_col(head, precip_candidates)
            except Exception:
                pcol = None
        if pcol is None:
            continue

        # 2) Read flexibly (let pandas ignore unknown columns)
        wanted = ['station_id','pr_g','ratio',pcol,'date',
                  # a wide net of possible coord columns (optional)
                  'lat','latitude','Lat','Latitude','y','lat_g','y_g','LAT','LATITUDE','Latitude_dd',
                  'lon','longitude','Lon','Longitude','x','lon_g','x_g','LON','LONGITUDE','Longitude_dd']
        try:
            df = pd.read_csv(
                f,
                usecols=lambda c: c in wanted,
                dtype={'station_id': str},
                parse_dates=['date'],
                infer_datetime_format=True,
                low_memory=False
            )
        except Exception:
            # fall back to full read if usecols fails
            df = pd.read_csv(f, dtype={'station_id': str}, low_memory=False)

        if 'station_id' not in df.columns or 'pr_g' not in df.columns or pcol not in df.columns:
            continue

        # 3) Normalize names & QC
        df = df.rename(columns={pcol: 'pr_p'})
        df['pr_g'] = pd.to_numeric(df['pr_g'], errors='coerce')
        df['pr_p'] = pd.to_numeric(df['pr_p'], errors='coerce')
        df = df[(df['pr_g'] >= MIN_MM) & (df['pr_p'] >= MIN_MM)]

        # ζ
        if 'ratio' in df.columns:
            df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
            df['zeta'] = (1.0 / df['ratio']) if RATIO_IS_PRODUCT_OVER_GAUGE else df['ratio']
        else:
            df['zeta'] = df['pr_g'] / np.maximum(df['pr_p'], 1e-9)

        df = df[(df['zeta'] > 0) & (df['zeta'] >= RATIO_CLIP[0]) & (df['zeta'] <= RATIO_CLIP[1])]

        # 4) Try to standardize coords if present in THIS file
        lat_col, lon_col = guess_lat_lon_columns(df)
        if lat_col and lon_col:
            df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
            df['lat'] = pd.to_numeric(df.get('lat'), errors='coerce')
            df['lon'] = pd.to_numeric(df.get('lon'), errors='coerce')

        # 5) Append only existing columns (avoid KeyError)
        base_cols = ['station_id', 'zeta']
        opt_cols = [c for c in ['lat', 'lon'] if c in df.columns]
        keep.append(df[base_cols + opt_cols].copy())

    if not keep:
        return pd.DataFrame()

    pairs = pd.concat(keep, ignore_index=True)
    # Don’t drop rows missing lat/lon here — we’ll fill from a coords map later.
    return pairs



def extract_unique_coords_from_pairs(pairs_dir, pairs_glob):
    """If some pair rows lack coords, scan files to build a station coord map."""
    files = sorted(glob.glob(os.path.join(pairs_dir, pairs_glob)))
    keep = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype={'station_id': str})
        except Exception:
            continue
        if 'station_id' not in df.columns:
            continue
        lat_col, lon_col = guess_lat_lon_columns(df)
        if not lat_col or not lon_col:
            continue
        sub = df[['station_id', lat_col, lon_col]].dropna().copy()
        sub['station_id'] = sub['station_id'].astype(str)
        sub = sub.rename(columns={lat_col: 'lat', lon_col: 'lon'})
        sub['lat'] = pd.to_numeric(sub['lat'], errors='coerce')
        sub['lon'] = pd.to_numeric(sub['lon'], errors='coerce')
        sub = sub.dropna(subset=['lat','lon'])
        keep.append(sub)
    if not keep:
        return pd.DataFrame(columns=['station_id','lat','lon'])
    coords = pd.concat(keep, ignore_index=True).drop_duplicates('station_id')
    return coords[['station_id','lat','lon']]


def sample_dem_elevation(tif_path, lons, lats):
    with rasterio.open(tif_path) as ds:
        crs_dst: CRS = ds.crs if ds.crs is not None else CRS.from_epsg(4326)
        gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(lons, lats)], crs=4326)
        if crs_dst.to_epsg() != 4326:
            gdf = gdf.to_crs(crs_dst)
        xy = [(geom.x, geom.y) for geom in gdf.geometry]
        vals = list(ds.sample(xy))
        elev = np.array([v[0] if (v is not None and np.isfinite(v[0])) else np.nan for v in vals], dtype=float)
        nod = ds.nodata
        if nod is not None and not np.isnan(nod):
            elev = np.where(elev == nod, np.nan, elev)
        return elev


def exp_fit_km(x_m, y):
    """Fit y = a * exp(b * (x_m/1000)). Return a, b, r2."""
    x_km = np.asarray(x_m, float) / 1000.0
    y = np.asarray(y, float)
    m = np.isfinite(x_km) & np.isfinite(y) & (y > 0)
    x_km = x_km[m]; y = y[m]
    if x_km.size < 3:
        return np.nan, np.nan, np.nan, m
    ln_y = np.log(y)
    A = np.vstack([np.ones_like(x_km), x_km]).T
    (ln_a, b), *_ = np.linalg.lstsq(A, ln_y, rcond=None)
    a = float(np.exp(ln_a))
    y_hat = a * np.exp(b * x_km)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, float(b), float(r2), m


def density_colors(x, y, nbins=140, vmin=1):
    xf = np.asarray(x, float); yf = np.asarray(y, float)
    mask = np.isfinite(xf) & np.isfinite(yf)
    xf = xf[mask]; yf = yf[mask]
    if xf.size == 0:
        return mask, np.array([]), 1
    x_min, x_max = np.nanpercentile(xf, [0.0, 99.5])
    y_min, y_max = np.nanpercentile(yf, [0.0, 99.5])
    x_pad = 0.02 * (x_max - x_min + 1e-9)
    y_pad = 0.02 * (y_max - y_min + 1e-9)
    xedges = np.linspace(max(0, x_min - x_pad), x_max + x_pad, nbins + 1)
    yedges = np.linspace(max(0, y_min - y_pad), y_max + y_pad, nbins + 1)
    H, _, _ = np.histogram2d(xf, yf, bins=(xedges, yedges))
    ix = np.clip(np.searchsorted(xedges, np.asarray(x, float), side='right') - 1, 0, len(xedges)-2)
    iy = np.clip(np.searchsorted(yedges, np.asarray(y, float), side='right') - 1, 0, len(yedges)-2)
    vals = H[ix, iy]
    vmax = max(int(H.max()), vmin + 1)
    return mask, vals, vmax


def panel(ax, elev_m, zeta, title):
    msk, vals, vmax = density_colors(elev_m, zeta, nbins=NBINS, vmin=1)
    sc = ax.scatter(np.asarray(elev_m)[msk], np.asarray(zeta)[msk],
                    c=vals[msk], cmap='viridis',
                    norm=LogNorm(vmin=1, vmax=vmax),
                    s=DOT_SIZE, alpha=ALPHA, edgecolors='none')

    a, b, r2, m_fit = exp_fit_km(elev_m, zeta)
    if np.isfinite(a) and np.isfinite(b):
        x0 = float(np.nanpercentile(np.asarray(elev_m)[m_fit], 0.5))
        x1 = float(np.nanpercentile(np.asarray(elev_m)[m_fit], 99.5))
        x_curve = np.linspace(max(0, x0), max(1.0, x1), 300)
        y_curve = a * np.exp(b * (x_curve / 1000.0))
        ax.plot(x_curve, y_curve, lw=2.4, color='tab:blue')
        ax.text(0.03, 0.95,
                rf"$\zeta = {a:.2f}\exp({b:.3f}\,h/1000)$"+"\n"+rf"$R^2={r2:.2f}$",
                transform=ax.transAxes, va='top', ha='left', fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='tab:blue', alpha=0.9))

    # Axes & labels
    if np.isfinite(elev_m).any():
        x_max = float(np.nanpercentile(elev_m, 99.5))
        x_max = int(np.ceil(x_max / X_TICK_STEP) * X_TICK_STEP)
        x_max = max(x_max, 1000)
    else:
        x_max = 3000
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.xaxis.set_minor_locator(MultipleLocator(X_TICK_STEP / 2))

    ax.set_ylim(ZETA_YMIN, ZETA_YMAX)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.set_title(title, pad=8)
    style_axes(ax)
    return sc


def main():
    try_register_montserrat(FONT_PATH_MONTSERRAT)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    if not os.path.exists(ELEV_TIF):
        raise FileNotFoundError(f"DEM not found: {ELEV_TIF}")

    fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.4), dpi=DPI)
    fig.subplots_adjust(wspace=0.18, hspace=0.28)
    mappables = []

    for i, cfg in enumerate(CONFIGS):
        row, col = divmod(i, 2)
        ax = axs[row, col]

        print(f"\n=== {cfg['name']} ===")
        pairs = load_all_pairs(cfg["pairs_dir"], cfg["pairs_glob"], cfg["precip_col_candidates"])
        if pairs.empty:
            ax.text(0.5, 0.5, f"No pairs for {cfg['name']}", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        print(f"Per-pair rows after QC: {len(pairs):,}")

        # If some rows lack coords, enrich using a station coord map from files
        if pairs[['lat','lon']].isna().any().any():
            coords_map = extract_unique_coords_from_pairs(cfg["pairs_dir"], cfg["pairs_glob"])
            if not coords_map.empty:
                pairs = pairs.merge(coords_map, on='station_id', how='left', suffixes=('','_map'))
                pairs['lat'] = pairs['lat'].fillna(pairs['lat_map'])
                pairs['lon'] = pairs['lon'].fillna(pairs['lon_map'])
                pairs = pairs.drop(columns=[c for c in ['lat_map','lon_map'] if c in pairs.columns])

        pairs = pairs.dropna(subset=['lat','lon','zeta'])
        # unique station DEM sample
        coords = pairs[['station_id','lat','lon']].drop_duplicates('station_id').copy()
        elev = sample_dem_elevation(ELEV_TIF, coords['lon'].to_numpy(), coords['lat'].to_numpy())
        coords['elev_m'] = elev
        coords = coords.dropna(subset=['elev_m'])

        # join back to ALL pairs
        pairs = pairs.merge(coords[['station_id','elev_m']], on='station_id', how='inner')
        pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=['elev_m','zeta'])
        print(f"Pairs with elevation: {len(pairs):,}")

        zeta_plot = np.clip(pairs['zeta'].to_numpy(), ZETA_YMIN, ZETA_YMAX)
        sc = panel(ax, pairs['elev_m'].to_numpy(), zeta_plot, cfg['name'])
        mappables.append(sc)

        # Labels: left column & bottom row only
        if col == 0:
            ax.set_ylabel(r"Bias correction factor $\zeta$ (–)")
        else:
            ax.set_ylabel("")
        if row == 1:
            ax.set_xlabel("Elevation (m)")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)

    if mappables:
        cbar = fig.colorbar(mappables[-1], ax=axs, orientation='vertical', fraction=0.04, pad=0.02)
        cbar.set_label("Point density (log scale)")

    fig.savefig(OUT_PNG, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
