# 04_apply_bias_correction_and_density_plots_XAVIER.py
# ---------------------------------------------------------
# Two side-by-side scatter-density panels with Turbo colormap.
# Left  : RAW   (x = Rain Gauge, y = XAVIER)
# Right : CORR  (x = Rain Gauge, y = zeta * XAVIER)
# Dots colored by local 2D-bin counts (log scale). No colorbar shown.
# ---------------------------------------------------------

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm

# ---------------- USER INPUTS ----------------
PAIRS_DIR  = r'G:\My Drive\xavier_bias_pairs'
PAIRS_GLOB = "xavier_bias_pairs_*.csv"
ZETA_CSV   = r'G:\My Drive\xavier_bias_pairs\zeta_per_station.csv'

OUT_PNG    = r'G:\My Drive\xavier_bias_pairs\density_raw_vs_corrected_turbo_xavier.png'

MIN_MM       = 1.0
RATIO_CLIP   = (0.1, 10.0)
NBINS        = 100
DOT_SIZE     = 3
ALPHA        = 0.75
TICK_WIDTH   = 2.0
BORDER_WIDTH = 2.0
FONT_SIZE    = 11
DPI          = 300
PCT_LIM      = 95
# --------------------------------------------

def detect_precip_col(df: pd.DataFrame) -> str:
    """Prefer XAVIER ('xavier_pr_mm'); fallback to IMERG/PERSIANN/CHIRPS for mixed runs."""
    if 'xavier_pr_mm' in df.columns: return 'xavier_pr_mm'
    if 'imerg_mm'     in df.columns: return 'imerg_mm'
    if 'persiann_mm'  in df.columns: return 'persiann_mm'
    if 'chirps_mm'    in df.columns: return 'chirps_mm'
    raise ValueError("Could not find precip column among: xavier_pr_mm, imerg_mm, persiann_mm, chirps_mm.")

def product_label_from_col(col: str) -> str:
    return {
        'xavier_pr_mm':'XAVIER',
        'imerg_mm':'IMERG',
        'persiann_mm':'PERSIANN',
        'chirps_mm':'CHIRPS'
    }.get(col, col)

def fit_origin(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0: return np.nan, np.nan
    a  = np.sum(x*y) / np.sum(x*x)
    r2 = 1.0 - np.sum((y - a*x)**2) / np.sum(y**2)
    return a, r2

def set_font():
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['font.family'] = 'Montserrat' if any(
        f.name == "Montserrat" for f in fm.fontManager.ttflist
    ) else 'DejaVu Sans'

def style_axes(ax):
    for sp in ax.spines.values(): sp.set_linewidth(BORDER_WIDTH)
    ax.tick_params(width=TICK_WIDTH, length=6)
    ax.grid(True, ls=':', lw=0.6)

def bin_counts_for_points(x, y, xedges, yedges):
    H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
    ix = np.clip(np.searchsorted(xedges, x, side='right') - 1, 0, len(xedges)-2)
    iy = np.clip(np.searchsorted(yedges, y, side='right') - 1, 0, len(yedges)-2)
    vals = H[ix, iy]
    return vals, H.max()

def panel_scatter_density(ax, x, y, lim, norm, title, fit_color):
    # Per-point density via 2D bins; Turbo colormap; no colorbar
    xedges = np.linspace(0, lim, NBINS + 1)
    yedges = np.linspace(0, lim, NBINS + 1)
    vals, _ = bin_counts_for_points(x, y, xedges, yedges)
    ax.scatter(x, y, c=vals, cmap='turbo', norm=norm,
               s=DOT_SIZE, alpha=ALPHA, edgecolors='none')

    # 45° solid line
    ax.plot([0, lim], [0, lim], color='red', lw=2.0, linestyle='-')

    # Best-fit through origin + boxed annotation
    a, r2 = fit_origin(x, y)
    xx = np.linspace(0, lim, 200)
    if np.isfinite(a):
        ax.plot(xx, a*xx, color=fit_color, lw=2.4)
        ax.text(0.04*lim, 0.90*lim, rf"$y={a:.2f}x,\ R^2={r2:.2f}$",
                color=fit_color, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=fit_color, alpha=0.85))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, pad=8)
    style_axes(ax)

def main():
    set_font()

    # Load pairs
    files = sorted(glob.glob(os.path.join(PAIRS_DIR, PAIRS_GLOB)))
    if not files:
        raise FileNotFoundError(f"No pair files in {PAIRS_DIR}\\{PAIRS_GLOB}")
    pairs = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in files], ignore_index=True)

    # Detect precip column (XAVIER preferred)
    precip_col = detect_precip_col(pairs)
    product_name = product_label_from_col(precip_col)

    # QC
    need = {'station_id','pr_g', precip_col}
    if not need.issubset(pairs.columns):
        raise ValueError(f"Pairs missing columns. Need {need}, got {pairs.columns.tolist()}")
    pairs['station_id'] = pairs['station_id'].astype(str)
    pairs = pairs[(pairs['pr_g'] >= MIN_MM) & (pairs[precip_col] >= MIN_MM)]
    if 'ratio' in pairs.columns:
        pairs = pairs[pairs['ratio'].between(*RATIO_CLIP)]

    # Merge zeta and correct
    zeta = pd.read_csv(ZETA_CSV)
    if 'station_id' not in zeta.columns or 'zeta' not in zeta.columns:
        raise ValueError("zeta_per_station.csv must contain: station_id, zeta")
    zeta['station_id'] = zeta['station_id'].astype(str)
    df = pairs.merge(zeta[['station_id','zeta']], on='station_id', how='left').dropna(subset=['zeta'])
    df['product_corr'] = df[precip_col] * df['zeta']

    # Arrays for plotting
    x_g   = df['pr_g'].to_numpy()
    y_raw = df[precip_col].to_numpy()
    y_cor = df['product_corr'].to_numpy()

    # Limits & shared normalization (log) for density colors
    lim = float(np.nanpercentile(np.concatenate([x_g, y_raw, y_cor]), PCT_LIM))
    lim = max(10.0, lim)
    xedges = np.linspace(0, lim, NBINS + 1)
    yedges = np.linspace(0, lim, NBINS + 1)
    _, vmax1 = bin_counts_for_points(x_g, y_raw, xedges, yedges)
    _, vmax2 = bin_counts_for_points(x_g, y_cor, xedges, yedges)
    norm = LogNorm(vmin=1, vmax=max(2, vmax1, vmax2))

    # Short figure (half-ish height), no extra right margin (no colorbar)
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3), dpi=DPI, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12)

    panel_scatter_density(
        axs[0], x_g, y_raw, lim, norm,
        title=f"Raw ({product_name})", fit_color='tab:gray'
    )
    panel_scatter_density(
        axs[1], x_g, y_cor, lim, norm,
        title=f"Bias-corrected ({product_name})", fit_color='tab:blue'
    )

    axs[0].set_xlabel("Rain Gauge (mm)"); axs[0].set_ylabel("Product (mm)")
    axs[1].set_xlabel("Rain Gauge (mm)")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
