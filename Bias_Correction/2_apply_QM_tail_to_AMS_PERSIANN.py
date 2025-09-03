# 2_apply_QM_tail_to_AMS_PERSIANN.py
# ---------------------------------------------------------
# Uses the tail-only QM models to correct *annual maxima* for PERSIANN.
# Steps:
#  1) Build AMS for product (PERSIANN) and gauge at each station.
#  2) For each AMS x: u = F_P_daily(x) (ECDF using full daily product)
#  3) If u >= q0: y = f_tail(x) via PCHIP(p_q -> g_q), with linear
#     extrapolation beyond top support; else y = x (unchanged).
#  4) Save per-station, per-year AMS (raw, corrected, gauge).
# ---------------------------------------------------------

import os, json
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# ---- paths (PERSIANN) ----
IN_DAILY_CSV   = r'G:\My Drive\qm_outputs\persiann\persiann_daily_pairs_clean.csv'
IN_MODELS_JSON = r'G:\My Drive\qm_outputs\persiann\qm_tail_models_persiann.json'
OUT_DIR        = r'G:\My Drive\qm_outputs\persiann'
os.makedirs(OUT_DIR, exist_ok=True)

OUT_AMS_CSV = os.path.join(OUT_DIR, "persiann_AMS_raw_QM_and_gauge.csv")

def ecdf_u(x_sorted: np.ndarray, v: float) -> float:
    """
    Empirical CDF value u in [0,1] for v given a sorted sample x_sorted.
    Linear interpolation between order stats, inclusive of tails.
    """
    xs = x_sorted
    n = xs.size
    if n == 0:
        return np.nan
    if v <= xs[0]:
        return 0.0
    if v >= xs[-1]:
        return 1.0
    # position in [0, n-1]
    i = np.searchsorted(xs, v, side='right') - 1
    x0, x1 = xs[i], xs[i+1]
    frac = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
    u = (i + frac) / (n - 1)
    return float(np.clip(u, 0.0, 1.0))

def apply_tail_mapping(x, p_q, g_q):
    """
    Value-space PCHIP from p_q -> g_q, with linear extrapolation outside.
    """
    p = np.asarray(p_q, float)
    g = np.asarray(g_q, float)
    f = PchipInterpolator(p, g, extrapolate=True)
    # For ultra tails, enforce linear behavior anchored by end segments
    if np.isscalar(x):
        x_arr = np.array([x], float)
    else:
        x_arr = np.asarray(x, float)
    y = f(x_arr)

    # linear tail extrapolation safeguards (use last two points)
    m_hi = (g[-1] - g[-2]) / max(1e-12, (p[-1] - p[-2]))
    m_lo = (g[1]  - g[0])  / max(1e-12, (p[1]  - p[0]))
    y = np.where(x_arr > p[-1], g[-1] + m_hi * (x_arr - p[-1]), y)
    y = np.where(x_arr < p[0],  g[0]  + m_lo * (x_arr - p[0]),  y)
    return float(y[0]) if np.isscalar(x) else y

def main():
    # Load daily data and models
    df = pd.read_csv(IN_DAILY_CSV, parse_dates=['date'])
    df['station_id'] = df['station_id'].astype(str)
    with open(IN_MODELS_JSON, "r") as f:
        models = json.load(f)

    # Build AMS per station/year from daily series
    df['year'] = df['date'].dt.year
    ams_rows = []
    for sid, g in df.groupby('station_id'):
        p = g['persiann_mm'].to_numpy(float)
        z = g['pr_g'].to_numpy(float)

        # sorted product distribution for ECDF u
        p_sorted = np.sort(p[np.isfinite(p)])

        # annual maxima (product & gauge)
        gp = g.groupby('year', as_index=False)
        p_ams = gp['persiann_mm'].max().rename(columns={'persiann_mm':'prod_AMS'})
        z_ams = gp['pr_g'].max().rename(columns={'pr_g':'gauge_AMS'})
        ams = pd.merge(p_ams, z_ams, on='year', how='outer').sort_values('year')
        ams['station_id'] = sid

        # Tail mapping (if model exists)
        m = models.get(sid)
        if m is not None:
            q0  = float(m['q0'])
            p_q = np.array(m['p_q'], float)
            g_q = np.array(m['g_q'], float)

            def correct_val(x):
                if not np.isfinite(x):
                    return np.nan
                u = ecdf_u(p_sorted, x)
                if u < q0:
                    return x  # unchanged below the tail start
                return apply_tail_mapping(x, p_q, g_q)

            ams['prod_AMS_QM'] = ams['prod_AMS'].apply(correct_val)
        else:
            ams['prod_AMS_QM'] = np.nan

        ams_rows.append(ams)

    out = pd.concat(ams_rows, ignore_index=True)
    out = out[['station_id','year','prod_AMS','prod_AMS_QM','gauge_AMS']]
    out.to_csv(OUT_AMS_CSV, index=False)

    print("Saved AMS (raw, QM-corrected, gauge):", OUT_AMS_CSV)
    print("Rows:", len(out))

if __name__ == "__main__":
    main()
