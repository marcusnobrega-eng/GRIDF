#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")
DRIVE_ROOT = Path("/Users/mngomes/Library/CloudStorage/GoogleDrive-marcusep2025@gmail.com/My Drive")

PERCENTILE = "p98"
ESTIMATOR = "mean"

CONFIGS = [
    {
        "label": "BR-DWGD",
        "product": "br_dwgd",
        "legacy_dir": DRIVE_ROOT / "xavier_bias_pairs",
        "legacy_pairs_glob": "xavier_bias_pairs_*.csv",
        "legacy_zeta": DRIVE_ROOT / "xavier_bias_pairs" / "zeta_per_station.csv",
        "precip_col": "xavier_pr_mm",
    },
    {
        "label": "IMERG V06",
        "product": "imerg_v06",
        "legacy_dir": DRIVE_ROOT / "imerg_bias_pairs",
        "legacy_pairs_glob": "imerg_bias_pairs_*.csv",
        "legacy_zeta": DRIVE_ROOT / "imerg_bias_pairs" / "zeta_per_station.csv",
        "precip_col": "imerg_mm",
    },
    {
        "label": "CHIRPS",
        "product": "chirps",
        "legacy_dir": DRIVE_ROOT / "chirps_bias_pairs",
        "legacy_pairs_glob": "chirps_bias_pairs_*.csv",
        "legacy_zeta": DRIVE_ROOT / "chirps_bias_pairs" / "zeta_per_station.csv",
        "precip_col": "chirps_mm",
    },
    {
        "label": "PERSIANN",
        "product": "persiann_cdr",
        "legacy_dir": DRIVE_ROOT / "persiann_bias_pairs",
        "legacy_pairs_glob": "persiann_bias_pairs_*.csv",
        "legacy_zeta": DRIVE_ROOT / "persiann_bias_pairs" / "zeta_per_station.csv",
        "precip_col": "persiann_mm",
    },
]


def norm_sid(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def fit_origin(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) == 0:
        return np.nan, np.nan
    a = np.sum(x * y) / np.sum(x * x)
    r2 = 1.0 - np.sum((y - a * x) ** 2) / np.sum(y ** 2)
    return a, r2


def load_pairs_from_files(files, precip_col):
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["source_file"] = f.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if "station_id" not in df.columns:
        raise ValueError("Missing station_id")

    if "pr_g" not in df.columns:
        if "gauge_mm" in df.columns:
            df["pr_g"] = df["gauge_mm"]
        else:
            raise ValueError("Missing pr_g/gauge_mm")

    if precip_col not in df.columns:
        if "product_mm" in df.columns:
            df[precip_col] = df["product_mm"]
        else:
            raise ValueError(f"Missing {precip_col}/product_mm")

    df["station_id"] = norm_sid(df["station_id"])
    df["pr_g"] = pd.to_numeric(df["pr_g"], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "ratio" not in df.columns:
        df["ratio"] = df["pr_g"] / df[precip_col]

    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")

    # Legacy plotting filter
    df = df[
        (df["pr_g"] >= 1.0)
        & (df[precip_col] >= 1.0)
        & (df["ratio"].between(0.1, 10.0))
    ].copy()

    return df


def load_current_pairs(product, precip_col):
    pair_dir = PIPELINE_ROOT / "data" / "products" / product / "sensitivity" / PERCENTILE / "pairs"
    files = sorted(pair_dir.glob(f"pairs_{product}_{PERCENTILE}_*.csv"))
    return load_pairs_from_files(files, precip_col)


def load_legacy_pairs(cfg):
    files = sorted(cfg["legacy_dir"].glob(cfg["legacy_pairs_glob"]))
    return load_pairs_from_files(files, cfg["precip_col"])


def load_current_zeta(product):
    p = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / PERCENTILE
        / "zeta_station"
        / ESTIMATOR
        / f"zeta_per_station_{product}_{PERCENTILE}_{ESTIMATOR}.csv"
    )

    z = pd.read_csv(p, low_memory=False)
    z["station_id"] = norm_sid(z["station_id"])

    if "zeta" not in z.columns:
        if "zeta_selected" in z.columns:
            z["zeta"] = z["zeta_selected"]
        elif "zeta_mean" in z.columns:
            z["zeta"] = z["zeta_mean"]
        else:
            raise ValueError(f"No zeta column in {p}")

    z["zeta"] = pd.to_numeric(z["zeta"], errors="coerce")

    return z[["station_id", "zeta"]].dropna()


def load_legacy_zeta(path):
    z = pd.read_csv(path, low_memory=False)
    z["station_id"] = norm_sid(z["station_id"])

    if "zeta" not in z.columns:
        # fallback options
        for c in ["zeta_selected", "zeta_mean", "mean_zeta"]:
            if c in z.columns:
                z["zeta"] = z[c]
                break

    if "zeta" not in z.columns:
        raise ValueError(f"No zeta column in legacy file: {path}")

    z["zeta"] = pd.to_numeric(z["zeta"], errors="coerce")

    return z[["station_id", "zeta"]].dropna()


def slope_with_zeta(pairs, zeta, precip_col):
    if pairs.empty or zeta.empty:
        return np.nan, np.nan, 0

    df = pairs.merge(zeta, on="station_id", how="left").dropna(subset=["zeta"])

    if df.empty:
        return np.nan, np.nan, 0

    y_corr = df[precip_col].to_numpy(float) * df["zeta"].to_numpy(float)
    x = df["pr_g"].to_numpy(float)

    a, r2 = fit_origin(x, y_corr)

    return a, r2, len(df)


def raw_slope(pairs, precip_col):
    if pairs.empty:
        return np.nan, np.nan, 0
    a, r2 = fit_origin(pairs["pr_g"], pairs[precip_col])
    return a, r2, len(pairs)


def compare_pair_overlap(a, b):
    if "date" not in a.columns or "date" not in b.columns:
        return None

    ka = set(zip(a["station_id"], a["date"]))
    kb = set(zip(b["station_id"], b["date"]))

    if not ka and not kb:
        return None

    inter = len(ka & kb)
    union = len(ka | kb)

    return inter, union, inter / union if union else np.nan


def main():
    for cfg in CONFIGS:
        label = cfg["label"]
        product = cfg["product"]
        precip_col = cfg["precip_col"]

        print("\n" + "=" * 100)
        print(label)
        print("=" * 100)

        legacy_pairs = load_legacy_pairs(cfg)
        current_pairs = load_current_pairs(product, precip_col)

        legacy_zeta = load_legacy_zeta(cfg["legacy_zeta"])
        current_zeta = load_current_zeta(product)

        print("\nPair counts:")
        print(f"  legacy pairs: {len(legacy_pairs)}")
        print(f"  current pairs: {len(current_pairs)}")
        print(f"  legacy stations: {legacy_pairs['station_id'].nunique() if not legacy_pairs.empty else 0}")
        print(f"  current stations: {current_pairs['station_id'].nunique() if not current_pairs.empty else 0}")

        overlap = compare_pair_overlap(legacy_pairs, current_pairs)
        if overlap is not None:
            inter, union, jacc = overlap
            print(f"  station-date overlap: {inter}/{union} = {jacc:.3f}")

        print("\nRaw slopes:")
        a, r2, n = raw_slope(legacy_pairs, precip_col)
        print(f"  legacy pairs raw:  y={a:.2f}x, R2={r2:.2f}, n={n}")

        a, r2, n = raw_slope(current_pairs, precip_col)
        print(f"  current pairs raw: y={a:.2f}x, R2={r2:.2f}, n={n}")

        print("\nCorrected slopes:")
        a, r2, n = slope_with_zeta(legacy_pairs, legacy_zeta, precip_col)
        print(f"  legacy pairs + legacy zeta:  y={a:.2f}x, R2={r2:.2f}, n={n}")

        a, r2, n = slope_with_zeta(current_pairs, current_zeta, precip_col)
        print(f"  current pairs + current zeta: y={a:.2f}x, R2={r2:.2f}, n={n}")

        a, r2, n = slope_with_zeta(current_pairs, legacy_zeta, precip_col)
        print(f"  current pairs + legacy zeta:  y={a:.2f}x, R2={r2:.2f}, n={n}")

        a, r2, n = slope_with_zeta(legacy_pairs, current_zeta, precip_col)
        print(f"  legacy pairs + current zeta:  y={a:.2f}x, R2={r2:.2f}, n={n}")

        print("\nZeta comparison:")
        z = legacy_zeta.rename(columns={"zeta": "zeta_legacy"}).merge(
            current_zeta.rename(columns={"zeta": "zeta_current"}),
            on="station_id",
            how="inner",
        )

        print(f"  matched zeta stations: {len(z)}")
        if len(z):
            z["diff"] = z["zeta_current"] - z["zeta_legacy"]
            z["ratio_current_over_legacy"] = z["zeta_current"] / z["zeta_legacy"]

            print(f"  legacy zeta median:  {z['zeta_legacy'].median():.3f}")
            print(f"  current zeta median: {z['zeta_current'].median():.3f}")
            print(f"  diff median:         {z['diff'].median():.3f}")
            print(f"  current/legacy med:  {z['ratio_current_over_legacy'].median():.3f}")
            print(f"  correlation:         {z[['zeta_legacy','zeta_current']].corr().iloc[0,1]:.3f}")

            print("\n  Largest absolute zeta differences:")
            print(
                z.reindex(z["diff"].abs().sort_values(ascending=False).index)
                .head(10)
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
