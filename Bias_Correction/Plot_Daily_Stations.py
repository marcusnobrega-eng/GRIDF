# ---------------------------------------------------------
# Brazil rain-gauge map colored by NYD
# Reqs: pandas, geopandas, matplotlib, numpy, shapely
# Optional: Montserrat font (same style as your other script)
# ---------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.font_manager as fm
import geopandas as gpd
from shapely.geometry import Point

# ---------------- USER INPUTS ----------------
CSV_PATH = r"C:\Users\marcu\PycharmProjects\hydrobr_project\Testing\stations_inventory_filtered_all.csv"
STATES_SHP = r"C:/Users/marcu/PycharmProjects/Mapas_Rainfall_Distribution/Shapefiles/Brasil_estados.shp"
OUT_PNG   = r"C:\Users\marcu\PycharmProjects\hydrobr_project\Testing\map_rain_gauges_NYD.png"
OUT_PDF   = r"C:\Users\marcu\PycharmProjects\hydrobr_project\Testing\map_rain_gauges_NYD.pdf"

# Style to match your bias-map script
STATES_EDGE_COLOR = 'white'
STATES_EDGE_WIDTH = 0.6
FONT_PATH_MONTSERRAT = 'C:/Users/marcu/PycharmProjects/Sucept_Maps/Montserrat/Montserrat-Regular.ttf'
FONT_SIZE = 10

# Colorbar geometry (adapted from your script)
CBAR_WIDTH_FIG   = 0.015
CBAR_RIGHT_PAD   = 0.05
CBAR_HEIGHT_REL  = 0.75
CBAR_TICK_STEP   = 20  # NYD is integer, so 1-year steps make sense

# Dot aesthetics
DOT_SIZE = 4
ALPHA = 0.8

# Colormap for discrete integer NYD
NYD_CMAP = 'nipy_spectral'  # consistent with your bias maps; try 'viridis' if preferred

# ---------------- Helpers ----------------
def try_register_montserrat(font_path: str):
    """Register Montserrat font if available; otherwise fallback nicely."""
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

def find_lat_lon_columns(df: pd.DataFrame):
    """Heuristically find latitude/longitude columns."""
    lon_candidates = ['lon', 'longitude', 'long', 'x', 'coord_x', 'lng']
    lat_candidates = ['lat', 'latitude', 'y', 'coord_y']
    df_cols_lower = {c.lower(): c for c in df.columns}
    lon = next((df_cols_lower[c] for c in lon_candidates if c in df_cols_lower), None)
    lat = next((df_cols_lower[c] for c in lat_candidates if c in df_cols_lower), None)
    if lon is None or lat is None:
        raise ValueError(
            f"Could not detect lat/lon columns. Found columns: {df.columns.tolist()}\n"
            f"Expected one of lon/lonitude/long/x/lng and lat/latitude/y."
        )
    return lat, lon

def find_nyd_column(df: pd.DataFrame):
    """Find NYD (case-insensitive) column."""
    for c in df.columns:
        if c.strip().lower() == 'nyd':
            return c
    # fallback: a couple of common variants
    for c in df.columns:
        if c.strip().lower() in ('n_y_d', 'n_y', 'years_data', 'n_years_data'):
            return c
    raise ValueError("Could not find an 'NYD' column in the CSV.")

def add_right_colorbar(fig, ax, mappable, bounds, tick_step=1, label="NYD (years)"):
    """Place a vertical colorbar to the right of the given axis with your geometry."""
    pos = ax.get_position()
    cbar_h = pos.height * CBAR_HEIGHT_REL
    cbar_y = pos.y0 + 0.5 * (pos.height - cbar_h)
    cbar_x = pos.x1 + CBAR_RIGHT_PAD
    cbar_ax = fig.add_axes([cbar_x, cbar_y, CBAR_WIDTH_FIG, cbar_h])

    cb = fig.colorbar(
        mappable, cax=cbar_ax,
        boundaries=bounds, spacing='proportional'
    )
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.tick_params(left=False, right=True, labelleft=False, labelright=True)
    vmin = bounds[0] + 0.5
    vmax = bounds[-1] - 0.5
    ticks = np.arange(np.ceil(vmin), np.floor(vmax) + 1e-9, tick_step, dtype=int)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_minor_locator(MultipleLocator(max(tick_step/2, 1)))
    cb.ax.tick_params(which='major', length=6, width=1.2)
    cb.ax.tick_params(which='minor', length=3, width=0.8)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    cb.set_label(label, rotation=90, labelpad=12, va='center')
    cb.outline.set_linewidth(1.2)
    return cb

# ---------------- Main ----------------
def main():
    # Fonts
    try_register_montserrat(FONT_PATH_MONTSERRAT)

    # --- Load states shapefile (and get Brazil outline) ---
    if not os.path.exists(STATES_SHP):
        raise FileNotFoundError(f"States shapefile not found: {STATES_SHP}")
    states = gpd.read_file(STATES_SHP)
    states = states.set_crs(4326) if states.crs is None else states.to_crs(4326)
    brazil_outline = states.dissolve()  # single polygon for Brazil

    # --- Load stations CSV & build GeoDataFrame ---
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    lat_col, lon_col = find_lat_lon_columns(df)
    nyd_col = find_nyd_column(df)

    # clean rows
    df = df.dropna(subset=[lat_col, lon_col, nyd_col]).copy()
    df = df[np.isfinite(df[lat_col]) & np.isfinite(df[lon_col]) & np.isfinite(df[nyd_col])]

    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col].astype(float), df[lat_col].astype(float)),
        crs=4326
    )

    # Optionally clip to Brazil
    try:
        gdf_pts = gpd.clip(gdf_pts, brazil_outline)
    except Exception:
        # If clipping fails due to topology, fall back to within()
        geom = brazil_outline.geometry.iloc[0]
        gdf_pts = gdf_pts[gdf_pts.geometry.within(geom.buffer(0))]

    # --- NYD color scaling (discrete integers) ---
    nyd_vals = gdf_pts[nyd_col].astype(float)
    vmin = int(np.floor(np.nanmin(nyd_vals)))
    vmax = int(np.ceil(np.nanmax(nyd_vals)))
    if vmin == vmax:  # all the same value
        vmin = vmax - 1
    # Boundaries so each integer year gets its own color bin centered at k+0.5
    bounds = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    cmap = plt.get_cmap(NYD_CMAP)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.3, 5), dpi=300)
    # a faint fill to help white state boundaries pop
    brazil_outline.plot(ax=ax, facecolor='0.92', edgecolor='none')
    # state boundaries
    states.boundary.plot(ax=ax, edgecolor=STATES_EDGE_COLOR, linewidth=STATES_EDGE_WIDTH)

    # gauges
    sc = ax.scatter(
        gdf_pts.geometry.x.values,
        gdf_pts.geometry.y.values,
        c=nyd_vals.values,
        s=DOT_SIZE,
        cmap=cmap,
        norm=norm,
        alpha=ALPHA,
        edgecolors='none'
    )

    # tidy map
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Daily Rainfall Stations Used for Bias Correction", pad=10)

    # colorbar (to the right, styled like your script)
    add_right_colorbar(fig, ax, sc, bounds=bounds, tick_step=CBAR_TICK_STEP, label="NYD (years)")

    # save/show
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')  # <-- export to PDF
    plt.show()
    plt.close(fig)
    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)

if __name__ == "__main__":
    main()
