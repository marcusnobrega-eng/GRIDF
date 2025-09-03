# 01_boxplot_bias_raw_vs_corrected_ALL.py
# ---------------------------------------------------------
# Creates a single 6.3x3.2 in boxplot figure with bias factors:
#   For each product (Xavier, IMERG, CHIRPS, PERSIANN), plot two boxes:
#     • Raw  bias  = product / gauge
#     • Corr bias  = (zeta * product) / gauge
#
# Reqs: pandas numpy matplotlib
# (font is optional; set FONT_PATH_MONTSERRAT if you have it)
# ---------------------------------------------------------

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------- USER INPUTS ----------------
CONFIGS = [
    {
        "name": "BR-DWGD",   # Xavier
        "pairs_dir": r'G:\My Drive\xavier_bias_pairs',
        "pairs_glob": "xavier_bias_pairs_*.csv",
        "zeta_csv":  r'G:\My Drive\xavier_bias_pairs\zeta_per_station.csv',
        "precip_col_hint": "xavier_pr_mm",  # optional hint; auto-detect also works
    },
    {
        "name": "IMERG",
        "pairs_dir": r'G:\My Drive\imerg_bias_pairs',
        "pairs_glob": "imerg_bias_pairs_*.csv",
        "zeta_csv":  r'G:\My Drive\imerg_bias_pairs\zeta_per_station.csv',
        "precip_col_hint": "imerg_mm",
    },
    {
        "name": "CHIRPS",
        "pairs_dir": r'G:\My Drive\chirps_bias_pairs',
        "pairs_glob": "chirps_bias_pairs_*.csv",
        "zeta_csv":  r'G:\My Drive\chirps_bias_pairs\zeta_per_station.csv',
        "precip_col_hint": "chirps_mm",
    },
    {
        "name": "PERSIANN",
        "pairs_dir": r'G:\My Drive\persiann_bias_pairs',
        "pairs_glob": "persiann_bias_pairs_*.csv",
        "zeta_csv":  r'G:\My Drive\persiann_bias_pairs\zeta_per_station.csv',
        "precip_col_hint": "persiann_mm",
    },
]

# Output
OUT_PNG = r'G:\My Drive\multi_product_bias_plots\boxplot_bias_raw_vs_corrected_ALL.png'

# Font
FONT_PATH_MONTSERRAT = r'C:/Users/marcu/PycharmProjects/Sucept_Maps/Montserrat/Montserrat-Regular.ttf'
FONT_SIZE = 9

# Data screening
MIN_MM = 1.0              # min rain for both gauge & product
RATIO_CLIP = (0.1, 10.0)  # drop extreme pairs as in your density plots

# Figure style
FIG_W, FIG_H = 6.3, 3.2
SPINE_WIDTH = 2.0
TICK_WIDTH = 2.0
TICK_LEN = 6
DPI = 600

# Y-axis range for bias factors (tweak if you like)
Y_MIN, Y_MAX = 0.0, 3.0

# Colors per dataset (customize here)
# raw uses the same color with lower alpha; corrected uses solid
COLORS = {
    "BR-DWGD": "#1f77b4",
    "IMERG":   "#2ca02c",
    "CHIRPS":  "#ff7f0e",
    "PERSIANN":"#d62728",
}

# --------------- helpers --------------------
def try_register_montserrat(font_path: str):
    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams['font.size'] = FONT_SIZE
    # prefer Montserrat if available
    if any(f.name.lower().startswith("montserrat") for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Montserrat'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'

def detect_precip_col(df: pd.DataFrame, hint: str | None = None) -> str:
    if hint and hint in df.columns:
        return hint
    for c in ('xavier_pr_mm','imerg_mm','chirps_mm','persiann_mm'):
        if c in df.columns:
            return c
    raise ValueError("No precip column found (expected one of: xavier_pr_mm, imerg_mm, chirps_mm, persiann_mm).")

def load_bias_series(pairs_dir, pairs_glob, zeta_csv, precip_col_hint=None):
    files = sorted(glob.glob(os.path.join(pairs_dir, pairs_glob)))
    if not files:
        return None, None

    pairs = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in files], ignore_index=True)

    pcol = detect_precip_col(pairs, precip_col_hint)
    need = {'station_id','pr_g', pcol}
    if not need.issubset(pairs.columns):
        raise ValueError(f"[{pairs_dir}] Pairs missing columns. Need {need}, got {pairs.columns.tolist()}")

    # Basic QC
    pairs = pairs[(pairs['pr_g'] >= MIN_MM) & (pairs[pcol] >= MIN_MM)].copy()
    if 'ratio' in pairs.columns:
        pairs = pairs[pairs['ratio'].between(*RATIO_CLIP)]
    if pairs.empty:
        return None, None

    # Merge ζ by station
    zeta = pd.read_csv(zeta_csv)
    if not {'station_id','zeta'}.issubset(zeta.columns):
        raise ValueError(f"[{zeta_csv}] must contain: station_id, zeta")
    pairs['station_id'] = pairs['station_id'].astype(str)
    zeta['station_id'] = zeta['station_id'].astype(str)
    df = pairs.merge(zeta[['station_id','zeta']], on='station_id', how='left').dropna(subset=['zeta'])
    if df.empty:
        return None, None

    # Bias factors
    raw_bias = (df[pcol].values / df['pr_g'].values)
    cor_bias = (df[pcol].values * df['zeta'].values / df['pr_g'].values)

    # Clip to sensible range for stability (same as screening)
    raw_bias = raw_bias[(raw_bias >= RATIO_CLIP[0]) & (raw_bias <= RATIO_CLIP[1])]
    cor_bias = cor_bias[(cor_bias >= RATIO_CLIP[0]) & (cor_bias <= RATIO_CLIP[1])]
    return raw_bias, cor_bias

# --------------- main -----------------------
def main():
    try_register_montserrat(FONT_PATH_MONTSERRAT)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    # Collect data in requested order
    order = ["BR-DWGD", "IMERG", "CHIRPS", "PERSIANN"]
    data_raw, data_cor, labels = [], [], []

    for cfg in CONFIGS:
        name = cfg["name"]
        raw, cor = load_bias_series(cfg["pairs_dir"], cfg["pairs_glob"], cfg["zeta_csv"], cfg.get("precip_col_hint"))
        labels.append(name)
        data_raw.append(np.asarray([]) if raw is None else raw)
        data_cor.append(np.asarray([]) if cor is None else cor)

    # Build interleaved list for boxplot: [X_raw, X_cor, I_raw, I_cor, ...]
    all_data = []
    all_colors = []
    xticks = []
    for name, r, c in zip(labels, data_raw, data_cor):
        base = COLORS.get(name, "#808080")
        all_data.extend([r, c])
        all_colors.extend([(*plt.matplotlib.colors.to_rgb(base), 0.45),  # raw (semi-transparent)
                           (*plt.matplotlib.colors.to_rgb(base), 1.00)]) # corrected (solid)
        xticks.extend([f"{name}\nRaw", "Bias-\ncorrected"])

    # Positions with small gap inside each pair and larger gap between products
    n_products = len(labels)
    positions = []
    step = 2.5         # distance between products
    inner = 0.35       # distance within a product (raw vs corrected)
    x0 = 1.0
    for i in range(n_products):
        positions += [x0 + i*step - inner/2, x0 + i*step + inner/2]

    # Figure
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    # Boxplot styling
    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        whis=(5, 95),   # 5–95th percentile whiskers (change if you prefer Tukey)
        showfliers=False,
        medianprops=dict(linewidth=SPINE_WIDTH),
        boxprops=dict(linewidth=SPINE_WIDTH),
        whiskerprops=dict(linewidth=SPINE_WIDTH),
        capprops=dict(linewidth=SPINE_WIDTH),
    )

    # Apply colors
    for patch, col in zip(bp['boxes'], all_colors):
        patch.set_facecolor(col)
        patch.set_edgecolor('black')

    # Axes style: ticks outside, thick spines, no grid
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_WIDTH)
    ax.tick_params(direction='out', width=TICK_WIDTH, length=TICK_LEN)
    ax.grid(False)

    # X ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(xticks, ha='center')
    # Y range & label
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("Bias factor (Product / Gauge)")

    # Light reference line at unbiased = 1
    ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
