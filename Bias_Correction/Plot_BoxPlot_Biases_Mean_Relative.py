# 01_boxplot_RELATIVE_bias_station_MEANS.py
# ---------------------------------------------------------
# Station-level RELATIVE bias (unbounded, centered at 0):
#   For each station i and product:
#     G_i  = mean(gauge at station i)
#     P_i  = mean(product at station i)
#     ζ_i  = station's multiplicative bias factor (from zeta_per_station.csv)
#
#   Raw   relative bias_i  = (P_i - G_i) / G_i
#   Corr  relative bias_i  = ((ζ_i * P_i) - G_i) / G_i
#
# Plots boxplots (Raw vs Bias-corrected) in order:
#   BR-DWGD → IMERG → CHIRPS → PERSIANN
#
# Styling per request:
#   - Figure size 6.3 x 3.2 inches, DPI 600
#   - Montserrat font (path below), font size 9
#   - Ticks outside, thick borders (2.5), no grid
#   - Custom colors per dataset (set in COLORS)
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
        "precip_col_hint": "xavier_pr_mm",
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

OUT_PNG = r'G:\My Drive\multi_product_bias_plots\boxplot_RELATIVEbias_station_means.png'

# Font
FONT_PATH_MONTSERRAT = r'C:/Users/marcu/PycharmProjects/Sucept_Maps/Montserrat/Montserrat-Regular.ttf'
FONT_SIZE = 9

# Screening for event pairs before station means
MIN_MM = 1.0              # min rain for both gauge & product
RATIO_CLIP = (0.1, 10.0)  # if an event-wise 'ratio' column exists

# Figure style
FIG_W, FIG_H = 6.3, 3.2
SPINE_WIDTH = 2.5
TICK_WIDTH = 2.5
TICK_LEN = 6
DPI = 600

# Y-axis display range for RELATIVE bias ((P-G)/G)
# Adjust if you expect wider/smaller spreads. Zero is unbiased.
Y_MIN, Y_MAX = -0.5, 0.5

# Colors per dataset (customize)
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

def load_station_mean_RELbiases(pairs_dir, pairs_glob, zeta_csv, precip_col_hint=None):
    files = sorted(glob.glob(os.path.join(pairs_dir, pairs_glob)))
    if not files:
        return None, None

    df = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in files], ignore_index=True)

    pcol = detect_precip_col(df, precip_col_hint)
    need = {'station_id','pr_g', pcol}
    if not need.issubset(df.columns):
        raise ValueError(f"[{pairs_dir}] Pairs missing columns. Need {need}, got {df.columns.tolist()}")

    # Event-level QC before computing means
    m = (df['pr_g'] >= MIN_MM) & (df[pcol] >= MIN_MM)
    if 'ratio' in df.columns:
        m &= df['ratio'].between(*RATIO_CLIP)
    df = df.loc[m].copy()
    if df.empty:
        return None, None

    # Mean per station for gauge and product
    grp = df.groupby('station_id', as_index=False).agg(
        Gmean=('pr_g','mean'),
        Pmean=(pcol,'mean')
    )

    # Merge ζ per station
    zeta = pd.read_csv(zeta_csv)
    if not {'station_id','zeta'}.issubset(zeta.columns):
        raise ValueError(f"[{zeta_csv}] must contain: station_id, zeta")
    zeta['station_id'] = zeta['station_id'].astype(str)
    grp['station_id'] = grp['station_id'].astype(str)
    grp = grp.merge(zeta[['station_id','zeta']], on='station_id', how='left').dropna(subset=['zeta'])

    # Keep stations with reasonable denominators
    grp = grp[(grp['Gmean'] >= MIN_MM)]
    if grp.empty:
        return None, None

    G = grp['Gmean'].to_numpy()
    P = grp['Pmean'].to_numpy()
    Z = grp['zeta'].to_numpy()

    # Station-level RELATIVE biases (unbounded, 0 = unbiased)
    raw_bias = (P - G) / G
    cor_bias = ((Z * P) - G) / G

    # Keep finite values only
    raw_bias = raw_bias[np.isfinite(raw_bias)]
    cor_bias = cor_bias[np.isfinite(cor_bias)]
    return raw_bias, cor_bias

# --------------- main -----------------------
def main():
    try_register_montserrat(FONT_PATH_MONTSERRAT)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    labels = [cfg["name"] for cfg in CONFIGS]
    data_raw, data_cor = [], []

    for cfg in CONFIGS:
        raw, cor = load_station_mean_RELbiases(
            cfg["pairs_dir"], cfg["pairs_glob"], cfg["zeta_csv"], cfg.get("precip_col_hint")
        )
        data_raw.append(np.asarray([]) if raw is None else raw)
        data_cor.append(np.asarray([]) if cor is None else cor)

    # Interleave Raw/Corrected series and aesthetics
    all_data, all_colors, xticks, positions = [], [], [], []
    step, inner, x0 = 2.5, 0.35, 1.0
    for i, name in enumerate(labels):
        base = COLORS.get(name, "#808080")
        rgb = plt.matplotlib.colors.to_rgb(base)
        all_data.extend([data_raw[i], data_cor[i]])
        all_colors.extend([(*rgb, 0.45), (*rgb, 1.00)])  # raw translucent, corrected solid
        xticks.extend([f"{name}\nRaw", "Bias-\ncorrected"])
        positions += [x0 + i*step - inner/2, x0 + i*step + inner/2]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        whis=(5, 95),      # 5–95th percentile whiskers (robust for skew)
        showfliers=False,
        medianprops=dict(linewidth=SPINE_WIDTH),
        boxprops=dict(linewidth=SPINE_WIDTH),
        whiskerprops=dict(linewidth=SPINE_WIDTH),
        capprops=dict(linewidth=SPINE_WIDTH),
    )
    for patch, col in zip(bp['boxes'], all_colors):
        patch.set_facecolor(col)
        patch.set_edgecolor('black')

    # Style: ticks outside, thick borders, no grid
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_WIDTH)
    ax.tick_params(direction='out', width=TICK_WIDTH, length=TICK_LEN)
    ax.grid(False)

    ax.set_xticks(positions)
    ax.set_xticklabels(xticks, ha='center')

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("Relative Bias ((Product − Gauge) / Gauge)")

    # Reference line at unbiased = 0
    ax.axhline(0.0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()
