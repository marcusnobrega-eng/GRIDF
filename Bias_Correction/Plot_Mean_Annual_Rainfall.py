# -*- coding: utf-8 -*-
"""
Plot 2×2 panel of mean annual rainfall (mm/year) for:
  - BR-DWGD (Xavier)
  - IMERG
  - CHIRPS
  - PERSIANN

Personalized:
- Discrete colormap with a fixed range [VMIN, VMAX] = [50, 3000] mm
- Colorbar with 'pointers' (extend triangles) and major/minor ticks
- User-settable number of discrete intervals
- 250 mm contours on top of each raster
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ---------------- USER INPUTS ----------------
# Paths to your exported GeoTIFFs (update if different)
RASTER_PATHS = {
    "BR-DWGD": r"G:\My Drive\annual_means\BR_DWGD_pr_meanAnnual_2000_2023_0p1deg.tif",
    "IMERG v6":             r"G:\My Drive\annual_means\IMERG_MeanAnnual_2000_2023_01deg.tif",
    "CHIRPS":            r"G:\My Drive\annual_means\CHIRPS_MeanAnnual_2000_2024_01deg.tif",
    "PERSIANN-CDR":          r"G:\My Drive\annual_means\PERSIANN_MeanAnnual_2000_2024_025deg.tif",
}

OUT_PNG = r"G:\My Drive\annual_means\MeanAnnualRainfall_2x2.png"

# Brazil states shapefile
STATES_SHP = r"C:\Users\marcu\OneDrive - University of Arizona\Desktop\Desktop_Folder\ANA_Analysis\Rainfall_Distribution\Brasil_Shapefile\brazil_Brazil_States_Boundary.shp"

# Figure & colorbar geometry
FIGSIZE    = (10, 7.5)
DPI        = 300
CBAR_WIDTH = 0.01
CBAR_PAD   = 0.04

# Discrete colormap settings
VMIN, VMAX   = 0.0, 3000.0              # fixed colorbar limits (mm/year)
N_INTERVALS  = 10                        # number of discrete color intervals (edit to taste)
CMAP_NAME    = "magma"                 # any Matplotlib cmap name

# Colorbar tick settings
CBAR_MAJOR_STEP = 600                    # major tick every 500 mm
CBAR_MINOR_STEP = 300                    # minor tick every 250 mm

# Contours
DRAW_CONTOURS   = True
CONTOUR_STEP_MM = 500                    # contour every 500 mm
CONTOUR_COLOR   = "k"
CONTOUR_WIDTH   = 0.5
CONTOUR_ALPHA   = 0.6
CONTOUR_LABELS  = True                   # turn labels on/off

# ---------------- HELPERS ----------------
def read_raster(fp):
    with rasterio.open(fp) as ds:
        arr = ds.read(1).astype("float32")
        bounds = ds.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        nod = ds.nodata
        if nod is not None and not np.isnan(nod):
            arr = np.where(arr == nod, np.nan, arr)
    return arr, extent

# ---------------- MAIN ----------------
def main():
    arrays, extents, titles = [], [], []

    for name, fp in RASTER_PATHS.items():
        if os.path.exists(fp):
            arr, ext = read_raster(fp)
            arrays.append(arr); extents.append(ext); titles.append(name)
        else:
            print(f"[WARN] Missing file for {name}: {fp}")

    if not arrays:
        raise RuntimeError("No rasters found to plot.")

    # Build discrete colormap & norm
    # Levels are the *edges* of the discrete bins (N_INTERVALS bins => N_INTERVALS+1 edges)
    levels = np.linspace(VMIN, VMAX, N_INTERVALS + 1)
    cmap = plt.get_cmap(CMAP_NAME, N_INTERVALS)  # discrete ListedColormap
    try:
        # Make NaNs transparent
        cmap = cmap.copy()
        cmap.set_bad((0, 0, 0, 0))
    except Exception:
        pass
    norm = BoundaryNorm(levels, cmap.N, clip=True)  # clip so all values map into the range

    # Brazil states
    states = None
    if STATES_SHP and os.path.exists(STATES_SHP):
        states = gpd.read_file(STATES_SHP)
        states = states.set_crs(4326) if states.crs is None else states.to_crs(4326)

    # Figure
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI)
    axs = axs.ravel()
    for ax in axs: ax.axis("off")

    im0 = None
    for i, (arr, ext, tl) in enumerate(zip(arrays, extents, titles)):
        ax = axs[i]; ax.axis("on")

        # Clip to desired colorbar range (keeps mapping well-defined)
        arr_show = np.array(arr, copy=True)
        # We'll clip for imshow; colorbar will still show "pointers" via 'extend' for conceptual emphasis
        arr_show = np.where(np.isfinite(arr_show), np.clip(arr_show, VMIN, VMAX), np.nan)

        im = ax.imshow(arr_show, extent=ext, origin="upper", cmap=cmap, norm=norm)
        if im0 is None: im0 = im

        # Contours every 250 mm
        if DRAW_CONTOURS:
            ny, nx = arr_show.shape
            x = np.linspace(ext[0], ext[1], nx)
            y = np.linspace(ext[3], ext[2], ny)  # origin='upper' => y descends
            X, Y = np.meshgrid(x, y)
            Z = np.ma.masked_invalid(arr_show)

            clevels = np.arange(VMIN, VMAX + 1e-6, CONTOUR_STEP_MM)
            cs = ax.contour(X, Y, Z, levels=clevels, colors=CONTOUR_COLOR,
                            linewidths=CONTOUR_WIDTH, alpha=CONTOUR_ALPHA)
            if CONTOUR_LABELS:
                ax.clabel(cs, fmt='%d', inline=True, fontsize=7)

        # States overlay
        if states is not None and not states.empty:
            states.boundary.plot(ax=ax, edgecolor="white", linewidth=0.6)

        # === Stats box (mean & std) ===
        add_stats_box(ax, arr, fontsize=8)  # use `arr_show` instead if you prefer stats of the clipped display

        ax.set_title(tl, pad=6, fontsize=10, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Shared colorbar on the right (stack aligned to right column)
    # If fewer than 4 rasters, this still places the bar aligned to the existing right-most axes.
    right_axes = [axs[j] for j in range(len(axs)) if (j % 2 == 1)]
    if right_axes and im0 is not None:
        rights, bottoms, tops = [], [], []
        for ax in right_axes:
            pos = ax.get_position()
            rights.append(pos.x1); bottoms.append(pos.y0); tops.append(pos.y1)
        x_right = max(rights); y0 = min(bottoms); y1 = max(tops)
        cbar_ax = fig.add_axes([x_right + CBAR_PAD, y0, CBAR_WIDTH, y1 - y0])

        # Use 'boundaries' so the bar shows the discrete bins; 'extend' adds pointer triangles
        cb = fig.colorbar(
            im0, cax=cbar_ax,
            boundaries=levels,
            spacing='proportional',
            extend='both',
            extendfrac=0.05
        )
        cb.set_label("Mean annual rainfall (mm/year)", rotation=90, labelpad=10)
        # Major/minor ticking
        cb.ax.yaxis.set_major_locator(MultipleLocator(CBAR_MAJOR_STEP))
        cb.ax.yaxis.set_minor_locator(MultipleLocator(CBAR_MINOR_STEP))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        cb.ax.tick_params(which='major', length=6, width=1.2)
        cb.ax.tick_params(which='minor', length=3, width=0.8)
        cb.outline.set_linewidth(1.2)

    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_PNG)

def add_stats_box(ax, arr, fontsize=12):
    """
    Draw mean/std box at top-left of an axes.
    Uses finite values of `arr` (no clipping to VMIN/VMAX).
    """
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if not np.isfinite(m) or not np.isfinite(s):
        text = r"$\mu=\mathrm{NA}$" + "\n" + r"$\sigma=\mathrm{NA}$"
    else:
        text = rf"$\mu={m:.0f}$ mm/yr" + "\n" + rf"$\sigma={s:.0f}$ mm/yr"

    ax.text(
        0.02, 0.02, text,          # <-- bottom-left in axes coords
        transform=ax.transAxes,
        ha="left", va="bottom",    # <-- anchor at the bottom
        fontsize=fontsize,
        fontfamily="Montserrat",   # falls back if not installed
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", edgecolor="black",
                  linewidth=0.8, alpha=0.9)
    )


if __name__ == "__main__":
    main()
