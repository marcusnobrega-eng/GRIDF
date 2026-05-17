#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")

PERCENTILE = "p98"
PERCENTILE_VALUE = 0.98

MIN_SAMPLES = 10
MIN_RAIN_MM = 1.0
MAX_RAIN_MM = 350.0
RATIO_LOW = 0.25
RATIO_HIGH = 5.0

PRODUCTS = {
    "br_dwgd": {
        "label": "BR-DWGD / Xavier",
        "precip_col": "xavier_pr_mm",
        "start_year": 1995,
        "end_year": 2006,
    },
    "imerg_v06": {
        "label": "IMERG V06",
        "precip_col": "imerg_mm",
        "start_year": 2001,
        "end_year": 2006,
    },
    "imerg_v07": {
        "label": "IMERG V07",
        "precip_col": "imerg_mm",
        "start_year": 2001,
        "end_year": 2006,
    },
    "chirps": {
        "label": "CHIRPS",
        "precip_col": "chirps_mm",
        "start_year": 1995,
        "end_year": 2006,
    },
    "persiann_cdr": {
        "label": "PERSIANN-CDR",
        "precip_col": "persiann_mm",
        "start_year": 1995,
        "end_year": 2006,
    },
}

ESTIMATORS = ["mean", "median"]


def normalize_station_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_pairs(product, spec):
    pair_dir = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / PERCENTILE / "pairs"
    files = sorted(pair_dir.glob(f"pairs_{product}_{PERCENTILE}_*.csv"))

    if not files:
        raise FileNotFoundError(f"No pair files found for {product}: {pair_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["source_file"] = f.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year_check"] = df["date"].dt.year
    elif "year" in df.columns:
        df["year_check"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year_check"] = np.nan

    df = df[(df["year_check"] >= spec["start_year"]) & (df["year_check"] <= spec["end_year"])].copy()

    if "station_id" not in df.columns:
        raise ValueError(f"{product}: missing station_id")

    df["station_id"] = normalize_station_id(df["station_id"])

    if "pr_g" not in df.columns:
        if "gauge_mm" in df.columns:
            df["pr_g"] = df["gauge_mm"]
        else:
            raise ValueError(f"{product}: missing pr_g/gauge_mm")

    precip_col = spec["precip_col"]
    if precip_col not in df.columns:
        if "product_mm" in df.columns:
            df[precip_col] = df["product_mm"]
        else:
            raise ValueError(f"{product}: missing {precip_col}/product_mm")

    df["pr_g"] = pd.to_numeric(df["pr_g"], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")

    df["raw_ratio_gauge_over_product"] = df["pr_g"] / df[precip_col]
    df["raw_ratio_gauge_over_product"] = df["raw_ratio_gauge_over_product"].replace([np.inf, -np.inf], np.nan)

    lat_col = find_col(df, ["lat", "latitude", "Latitude", "station_lat"])
    lon_col = find_col(df, ["lon", "longitude", "Longitude", "station_lon"])

    if lat_col is not None:
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        df["latitude"] = np.nan

    if lon_col is not None:
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        df["longitude"] = np.nan

    return df


def add_qc_columns(df, precip_col):
    out = df.copy()

    out["qc_basic_pass"] = (
        out["pr_g"].between(MIN_RAIN_MM, MAX_RAIN_MM)
        & out[precip_col].between(MIN_RAIN_MM, MAX_RAIN_MM)
        & np.isfinite(out["raw_ratio_gauge_over_product"])
    )

    out["ratio_clipped_low"] = out["qc_basic_pass"] & (out["raw_ratio_gauge_over_product"] < RATIO_LOW)
    out["ratio_clipped_high"] = out["qc_basic_pass"] & (out["raw_ratio_gauge_over_product"] > RATIO_HIGH)

    # Legacy-equivalent behavior:
    # FILTER outside ratios. Do not clip/winsorize them.
    out["ratio_pass"] = (
        out["qc_basic_pass"]
        & out["raw_ratio_gauge_over_product"].between(RATIO_LOW, RATIO_HIGH)
    )

    out["ratio_for_zeta"] = out["raw_ratio_gauge_over_product"].where(out["ratio_pass"], np.nan)
    out["qc_used_for_zeta"] = out["ratio_pass"]

    out["ratio_clip_low"] = RATIO_LOW
    out["ratio_clip_high"] = RATIO_HIGH

    return out


def slope0(g, precip_col):
    x = g[precip_col].to_numpy(float)
    y = g["pr_g"].to_numpy(float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    den = np.sum(x * x)
    if den <= 0:
        return np.nan

    return float(np.sum(x * y) / den)


def summarize_station(product, spec, df_used, estimator):
    precip_col = spec["precip_col"]
    rows = []

    for sid, g_all in df_used.groupby("station_id"):
        g = g_all[g_all["qc_used_for_zeta"]].copy()
        n_used = len(g)

        if n_used == 0:
            ratios = np.array([], dtype=float)
        else:
            ratios = g["ratio_for_zeta"].to_numpy(float)
            ratios = ratios[np.isfinite(ratios)]

        n_raw = len(g_all)
        n_basic = int(g_all["qc_basic_pass"].sum())
        n_low = int(g_all["ratio_clipped_low"].sum())
        n_high = int(g_all["ratio_clipped_high"].sum())

        lat = pd.to_numeric(g_all["latitude"], errors="coerce").dropna()
        lon = pd.to_numeric(g_all["longitude"], errors="coerce").dropna()

        if len(ratios) >= MIN_SAMPLES:
            zeta_mean = float(np.mean(ratios))
            zeta_median = float(np.median(ratios))
            zeta_slope0 = slope0(g, precip_col)
            zeta_std = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
            zeta_min = float(np.min(ratios))
            zeta_p10 = float(np.percentile(ratios, 10))
            zeta_p25 = float(np.percentile(ratios, 25))
            zeta_p75 = float(np.percentile(ratios, 75))
            zeta_p90 = float(np.percentile(ratios, 90))
            zeta_max = float(np.max(ratios))
            zeta_iqr = zeta_p75 - zeta_p25

            if estimator == "mean":
                zeta_selected = zeta_mean
            elif estimator == "median":
                zeta_selected = zeta_median
            elif estimator == "slope0":
                zeta_selected = zeta_slope0
            else:
                raise ValueError(estimator)

            passes = True
        else:
            zeta_mean = np.nan
            zeta_median = np.nan
            zeta_slope0 = np.nan
            zeta_std = np.nan
            zeta_min = np.nan
            zeta_p10 = np.nan
            zeta_p25 = np.nan
            zeta_p75 = np.nan
            zeta_p90 = np.nan
            zeta_max = np.nan
            zeta_iqr = np.nan
            zeta_selected = np.nan
            passes = False

        rows.append({
            "product": product,
            "product_label": spec["label"],
            "percentile_label": PERCENTILE,
            "percentile_value": PERCENTILE_VALUE,
            "zeta_method": estimator,
            "station_id": sid,
            "station_name": np.nan,
            "city": np.nan,
            "state": np.nan,
            "latitude": float(lat.median()) if len(lat) else np.nan,
            "longitude": float(lon.median()) if len(lon) else np.nan,
            "row_index": np.nan,
            "n_pairs_raw": n_raw,
            "n_pairs_after_basic_qc": n_basic,
            "n_pairs_used": int(n_used),
            "n_ratio_clipped_low": n_low,
            "n_ratio_clipped_high": n_high,
            "mean_gauge_mm": float(g["pr_g"].mean()) if n_used else np.nan,
            "mean_product_mm": float(g[precip_col].mean()) if n_used else np.nan,
            "median_gauge_mm": float(g["pr_g"].median()) if n_used else np.nan,
            "median_product_mm": float(g[precip_col].median()) if n_used else np.nan,
            "zeta_mean": zeta_mean,
            "zeta_median": zeta_median,
            "zeta_slope0": zeta_slope0,
            "zeta_std": zeta_std,
            "zeta_min": zeta_min,
            "zeta_p10": zeta_p10,
            "zeta_p25": zeta_p25,
            "zeta_p75": zeta_p75,
            "zeta_p90": zeta_p90,
            "zeta_max": zeta_max,
            "zeta_iqr": zeta_iqr,
            "zeta_selected": zeta_selected,
            "zeta": zeta_selected,
            "passes_min_pairs": passes,
        })

    return pd.DataFrame(rows)


def write_outputs(product, spec, estimator):
    print("\n" + "=" * 90)
    print(f"{product} / {estimator}")
    print("=" * 90)

    precip_col = spec["precip_col"]
    pairs = load_pairs(product, spec)
    pairs_qc = add_qc_columns(pairs, precip_col)

    out_base = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / PERCENTILE
    zeta_dir = out_base / "zeta_station" / estimator
    table_dir = out_base / "tables"

    zeta_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    all_station = summarize_station(product, spec, pairs_qc, estimator)
    retained = all_station[all_station["passes_min_pairs"]].copy()

    all_path = zeta_dir / f"zeta_station_all_{product}_{PERCENTILE}_{estimator}.csv"
    retained_path = zeta_dir / f"zeta_per_station_{product}_{PERCENTILE}_{estimator}.csv"
    qc_path = table_dir / f"pair_qc_{product}_{PERCENTILE}.csv"

    all_station.to_csv(all_path, index=False)
    retained.to_csv(retained_path, index=False)
    pairs_qc.to_csv(qc_path, index=False)

    print(f"pairs raw: {len(pairs_qc)}")
    print(f"pairs used for zeta: {int(pairs_qc['qc_used_for_zeta'].sum())}")
    print(f"stations all: {len(all_station)}")
    print(f"stations retained: {len(retained)}")
    if len(retained):
        print(f"median zeta selected: {retained['zeta_selected'].median():.4f}")
        print(f"mean zeta selected: {retained['zeta_selected'].mean():.4f}")
    print("wrote:", retained_path)


def main():
    for estimator in ESTIMATORS:
        for product, spec in PRODUCTS.items():
            write_outputs(product, spec, estimator)


if __name__ == "__main__":
    main()
