#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeta.py

Station-level bias-correction factor estimation for the GRIDF rainfall-product
bias-correction pipeline.

Part 05 scope
-------------
This module reads GEE-exported gauge/product bias-pair CSVs and computes
station-level correction factors:

    zeta = gauge_mm / product_mm

The main estimator is the median ratio, with mean retained as a sensitivity
estimator. The station table always saves multiple estimators so the paper can
compare them without recomputing the full event-pair stage.

Scientific workflow
-------------------
Input from Part 04:
    data/products/<product>/sensitivity/<pXX>/pairs/*.csv

Each event row should contain:
    station_id
    latitude
    longitude
    date
    year
    gauge_mm
    product_mm
    ratio_gauge_over_product

Quality control:
    - gauge_mm must be finite
    - product_mm must be finite
    - gauge_mm > min_gauge_rainfall_for_ratio_mm
    - product_mm > min_product_rainfall_for_ratio_mm
    - gauge_mm <= max_rainfall_for_ratio_mm
    - raw ratio must be finite and positive
    - ratio is clipped to ratio_clip before zeta aggregation by default

Why clip rather than silently drop?
-----------------------------------
The previous pipeline used clipping-style safeguards to prevent pathological
gauge/product mismatches from dominating the correction field. This module keeps
that philosophy while preserving audit columns:

    raw_ratio_gauge_over_product
    ratio_for_zeta
    ratio_clipped_low
    ratio_clipped_high

Station-level output:
    zeta_mean
    zeta_median
    zeta_slope0
    zeta_selected
    zeta_method
    n_pairs_used

Main correction:
    zeta_selected = zeta_median

Estimator sensitivity:
    zeta_selected = zeta_mean
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .event_selection import parse_percentile_arg
from .utils import ensure_dir, now_iso, print_header, print_section, timestamp, write_json


PAIR_REQUIRED_BASE_COLUMNS = [
    "station_id",
    "latitude",
    "longitude",
    "date",
    "year",
    "gauge_mm",
    "product_mm",
]


STATION_OUTPUT_COLUMNS = [
    "product",
    "product_label",
    "percentile_label",
    "percentile_value",
    "zeta_method",
    "station_id",
    "station_name",
    "city",
    "state",
    "latitude",
    "longitude",
    "row_index",
    "n_pairs_raw",
    "n_pairs_after_basic_qc",
    "n_pairs_used",
    "n_ratio_clipped_low",
    "n_ratio_clipped_high",
    "mean_gauge_mm",
    "mean_product_mm",
    "median_gauge_mm",
    "median_product_mm",
    "zeta_mean",
    "zeta_median",
    "zeta_slope0",
    "zeta_std",
    "zeta_min",
    "zeta_p10",
    "zeta_p25",
    "zeta_p75",
    "zeta_p90",
    "zeta_max",
    "zeta_iqr",
    "zeta_selected",
    "passes_min_pairs",
]


def _pairs_dir(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "pairs"
    )


def _zeta_station_dir(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "zeta_station"
        / estimator
    )


def _tables_dir(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "tables"
    )


def _manifest_dir(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "zeta_station"
    )


def list_pair_csvs(pairs_dir: Path) -> List[Path]:
    """
    List likely GEE pair CSVs in a local pairs folder.

    Excludes manifest and README files. Recurses one level if users copied
    Drive exports into subfolders.
    """
    pairs_dir = Path(pairs_dir)
    if not pairs_dir.exists():
        return []

    csvs: List[Path] = []
    for path in pairs_dir.rglob("*.csv"):
        name = path.name.lower()
        if "manifest" in name:
            continue
        if "readme" in name:
            continue
        if name.startswith("."):
            continue
        csvs.append(path)

    return sorted(csvs)


def _coerce_bool_like(series: pd.Series) -> pd.Series:
    """Convert bool/string bool columns to boolean with NA as False."""
    if series.dtype == bool:
        return series.fillna(False)

    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "t"])


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert dataframe column to numeric, handling comma decimals."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    if pd.api.types.is_numeric_dtype(df[col]):
        return pd.to_numeric(df[col], errors="coerce")

    return pd.to_numeric(
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan}),
        errors="coerce",
    )


def _normalize_station_id(series: pd.Series) -> pd.Series:
    """Normalize station IDs from CSVs."""
    def one(x: Any) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)) and float(x).is_integer():
            return str(int(x))
        text = str(x).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        return text

    return series.map(one)


def load_pair_csvs(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    pairs_folder: Optional[Path] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read all pair CSVs for one product/percentile.

    Parameters
    ----------
    pairs_folder:
        Optional override. If None, uses the configured local pairs directory.

    Returns
    -------
    Concatenated DataFrame with a source_file column.
    """
    if pairs_folder is None:
        pairs_folder = _pairs_dir(cfg, product_name, percentile_label)

    pairs_folder = Path(pairs_folder)
    csvs = list_pair_csvs(pairs_folder)

    if not csvs:
        raise FileNotFoundError(
            f"No pair CSV files found in:\n  {pairs_folder}\n\n"
            "Expected files exported by Part 04, e.g.:\n"
            f"  pairs_{product_name}_{percentile_label}_YYYY_chunkNNN.csv\n\n"
            "Remember: Earth Engine exports to Google Drive first. After the "
            "tasks finish, copy/sync the CSVs into this local pairs folder."
        )

    frames: List[pd.DataFrame] = []

    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
        except pd.errors.EmptyDataError:
            if verbose:
                print(f"[warning] empty CSV skipped: {csv}")
            continue

        if df.empty:
            if verbose:
                print(f"[warning] zero-row CSV skipped: {csv}")
            continue

        df["source_file"] = str(csv)
        frames.append(df)

    if not frames:
        raise ValueError(f"All CSV files were empty or unreadable in: {pairs_folder}")

    pairs = pd.concat(frames, ignore_index=True)

    # Normalize core fields.
    if "station_id" in pairs.columns:
        pairs["station_id"] = _normalize_station_id(pairs["station_id"])
    else:
        raise ValueError("Pair CSVs are missing required column: station_id")

    for col in ["latitude", "longitude", "gauge_mm", "product_mm"]:
        pairs[col] = _to_numeric(pairs, col)

    if "year" in pairs.columns:
        pairs["year"] = pd.to_numeric(pairs["year"], errors="coerce").astype("Int64")
    else:
        # Fall back from date.
        pairs["year"] = pd.to_datetime(pairs.get("date"), errors="coerce").dt.year.astype("Int64")

    if "date" in pairs.columns:
        pairs["date"] = pd.to_datetime(pairs["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        pairs["date"] = ""

    if "ratio_gauge_over_product" in pairs.columns:
        pairs["raw_ratio_gauge_over_product"] = _to_numeric(pairs, "ratio_gauge_over_product")
    else:
        pairs["raw_ratio_gauge_over_product"] = pairs["gauge_mm"] / pairs["product_mm"]

    if "product_valid_for_ratio" in pairs.columns:
        pairs["product_valid_for_ratio_original"] = _coerce_bool_like(pairs["product_valid_for_ratio"])
    else:
        pairs["product_valid_for_ratio_original"] = np.nan

    if start_year is not None:
        pairs = pairs.loc[pairs["year"] >= int(start_year)].copy()
    if end_year is not None:
        pairs = pairs.loc[pairs["year"] <= int(end_year)].copy()

    if verbose:
        print_section("Loaded pair CSVs")
        print(f"Pairs folder: {pairs_folder}")
        print(f"CSV files:    {len(csvs)}")
        print(f"Rows loaded:   {len(pairs)}")
        if len(pairs):
            years = sorted(pairs["year"].dropna().astype(int).unique().tolist())
            print(f"Years:         {years[0]}–{years[-1]} ({len(years)} years)")

    return pairs


def apply_pair_qc(
    pairs: pd.DataFrame,
    cfg: Any,
    product_name: str,
    percentile_label: str,
) -> pd.DataFrame:
    """
    Apply pair-level QC and create ratio_for_zeta.

    The function keeps all rows but adds QC flags.
    """
    out = pairs.copy()

    ratio_qc = cfg.method["ratio_qc"]
    ratio_clip = ratio_qc.get("ratio_clip", [0.25, 5.0])
    clip_low, clip_high = float(ratio_clip[0]), float(ratio_clip[1])

    min_gauge = float(ratio_qc["min_gauge_rainfall_for_ratio_mm"])
    min_product = float(ratio_qc["min_product_rainfall_for_ratio_mm"])
    max_rain = float(ratio_qc["max_rainfall_for_ratio_mm"])

    out["product"] = product_name
    out["percentile_label"] = percentile_label

    # Recompute raw ratio where possible to avoid CSV type issues.
    raw_ratio = out["raw_ratio_gauge_over_product"].copy()
    recompute_mask = ~np.isfinite(raw_ratio) & np.isfinite(out["gauge_mm"]) & np.isfinite(out["product_mm"]) & (out["product_mm"] != 0)
    raw_ratio.loc[recompute_mask] = out.loc[recompute_mask, "gauge_mm"] / out.loc[recompute_mask, "product_mm"]
    out["raw_ratio_gauge_over_product"] = raw_ratio

    out["qc_gauge_finite"] = np.isfinite(out["gauge_mm"])
    out["qc_product_finite"] = np.isfinite(out["product_mm"])
    out["qc_ratio_finite"] = np.isfinite(out["raw_ratio_gauge_over_product"])

    out["qc_gauge_above_min"] = out["gauge_mm"] > min_gauge
    out["qc_product_above_min"] = out["product_mm"] > min_product
    out["qc_gauge_below_max"] = out["gauge_mm"] <= max_rain
    out["qc_ratio_positive"] = out["raw_ratio_gauge_over_product"] > 0

    out["qc_basic_pass"] = (
        out["qc_gauge_finite"]
        & out["qc_product_finite"]
        & out["qc_ratio_finite"]
        & out["qc_gauge_above_min"]
        & out["qc_product_above_min"]
        & out["qc_gauge_below_max"]
        & out["qc_ratio_positive"]
    )

    out["ratio_clipped_low"] = out["qc_basic_pass"] & (out["raw_ratio_gauge_over_product"] < clip_low)
    out["ratio_clipped_high"] = out["qc_basic_pass"] & (out["raw_ratio_gauge_over_product"] > clip_high)

    out["ratio_for_zeta"] = np.nan
    valid_ratio = out["qc_basic_pass"].to_numpy()
    raw = out["raw_ratio_gauge_over_product"].to_numpy(dtype=float)
    clipped = np.clip(raw, clip_low, clip_high)
    out.loc[valid_ratio, "ratio_for_zeta"] = clipped[valid_ratio]

    out["qc_used_for_zeta"] = np.isfinite(out["ratio_for_zeta"])

    out["ratio_clip_low"] = clip_low
    out["ratio_clip_high"] = clip_high
    out["min_gauge_rainfall_for_ratio_mm"] = min_gauge
    out["min_product_rainfall_for_ratio_mm"] = min_product
    out["max_rainfall_for_ratio_mm"] = max_rain

    return out


def _safe_first(series: pd.Series, default: Any = "") -> Any:
    """Return first non-null value from a series."""
    valid = series.dropna()
    if valid.empty:
        return default
    return valid.iloc[0]


def _weighted_slope_origin(product: np.ndarray, gauge: np.ndarray) -> float:
    """
    Compute slope through origin for gauge = zeta * product.

    zeta_slope0 = sum(product * gauge) / sum(product^2)

    Returns NaN if denominator is invalid.
    """
    product = np.asarray(product, dtype=float)
    gauge = np.asarray(gauge, dtype=float)
    valid = np.isfinite(product) & np.isfinite(gauge)
    if not valid.any():
        return np.nan
    denom = np.sum(product[valid] ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(product[valid] * gauge[valid]) / denom)


def compute_station_statistics(
    qc_pairs: pd.DataFrame,
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute station-level zeta statistics.

    Returns
    -------
    station_all:
        All station groups with statistics.
    station_retained:
        Only stations passing min_pairs_per_station.
    """
    product_cfg = cfg.product(product_name)
    product_label = product_cfg.get("label", product_name)

    _, percentile_value = parse_percentile_arg(percentile_label)

    min_pairs = int(cfg.method["zeta"]["min_pairs_per_station"])
    estimator = str(estimator).lower()

    if estimator not in ["median", "mean", "slope0"]:
        raise ValueError("estimator must be one of: median, mean, slope0")

    rows: List[Dict[str, Any]] = []

    # n_pairs_raw should count all rows before QC for that station.
    for station_id, group in qc_pairs.groupby("station_id", dropna=False):
        used = group.loc[group["qc_used_for_zeta"]].copy()

        ratios = used["ratio_for_zeta"].to_numpy(dtype=float)
        gauges = used["gauge_mm"].to_numpy(dtype=float)
        products = used["product_mm"].to_numpy(dtype=float)

        n_raw = int(len(group))
        n_basic = int(group["qc_basic_pass"].sum())
        n_used = int(np.isfinite(ratios).sum())

        if n_used > 0:
            zeta_mean = float(np.nanmean(ratios))
            zeta_median = float(np.nanmedian(ratios))
            zeta_std = float(np.nanstd(ratios, ddof=1)) if n_used > 1 else 0.0
            zeta_min = float(np.nanmin(ratios))
            zeta_p10 = float(np.nanpercentile(ratios, 10))
            zeta_p25 = float(np.nanpercentile(ratios, 25))
            zeta_p75 = float(np.nanpercentile(ratios, 75))
            zeta_p90 = float(np.nanpercentile(ratios, 90))
            zeta_max = float(np.nanmax(ratios))
            zeta_iqr = float(zeta_p75 - zeta_p25)
            zeta_slope0 = _weighted_slope_origin(products, gauges)

            mean_gauge = float(np.nanmean(gauges))
            mean_product = float(np.nanmean(products))
            median_gauge = float(np.nanmedian(gauges))
            median_product = float(np.nanmedian(products))
        else:
            zeta_mean = np.nan
            zeta_median = np.nan
            zeta_std = np.nan
            zeta_min = np.nan
            zeta_p10 = np.nan
            zeta_p25 = np.nan
            zeta_p75 = np.nan
            zeta_p90 = np.nan
            zeta_max = np.nan
            zeta_iqr = np.nan
            zeta_slope0 = np.nan
            mean_gauge = np.nan
            mean_product = np.nan
            median_gauge = np.nan
            median_product = np.nan

        if estimator == "median":
            zeta_selected = zeta_median
        elif estimator == "mean":
            zeta_selected = zeta_mean
        else:
            zeta_selected = zeta_slope0

        passes_min_pairs = bool(n_used >= min_pairs and np.isfinite(zeta_selected))

        row = {
            "product": product_name,
            "product_label": product_label,
            "percentile_label": percentile_label,
            "percentile_value": percentile_value,
            "zeta_method": estimator,
            "station_id": str(station_id),
            "station_name": _safe_first(group.get("station_name", pd.Series([""]))),
            "city": _safe_first(group.get("city", pd.Series([""]))),
            "state": _safe_first(group.get("state", pd.Series([""]))),
            "latitude": float(_safe_first(group["latitude"], np.nan)),
            "longitude": float(_safe_first(group["longitude"], np.nan)),
            "row_index": _safe_first(group.get("row_index", pd.Series([np.nan])), np.nan),
            "n_pairs_raw": n_raw,
            "n_pairs_after_basic_qc": n_basic,
            "n_pairs_used": n_used,
            "n_ratio_clipped_low": int(group["ratio_clipped_low"].sum()),
            "n_ratio_clipped_high": int(group["ratio_clipped_high"].sum()),
            "mean_gauge_mm": mean_gauge,
            "mean_product_mm": mean_product,
            "median_gauge_mm": median_gauge,
            "median_product_mm": median_product,
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
            "passes_min_pairs": passes_min_pairs,
        }

        rows.append(row)

    station_all = pd.DataFrame(rows, columns=STATION_OUTPUT_COLUMNS)
    station_retained = station_all.loc[station_all["passes_min_pairs"]].copy()

    # Stable sort for reproducibility.
    station_all = station_all.sort_values(["station_id"]).reset_index(drop=True)
    station_retained = station_retained.sort_values(["station_id"]).reset_index(drop=True)

    return station_all, station_retained


def write_zeta_outputs(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
    qc_pairs: pd.DataFrame,
    station_all: pd.DataFrame,
    station_retained: pd.DataFrame,
    source_pair_files: Sequence[str],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    write_qc_pairs: bool = True,
) -> Dict[str, Path]:
    """
    Write QC pair table, station zeta tables, and manifest.
    """
    out_dir = _zeta_station_dir(cfg, product_name, percentile_label, estimator)
    tables_dir = _tables_dir(cfg, product_name, percentile_label)
    manifest_dir = _manifest_dir(cfg, product_name, percentile_label)

    ensure_dir(out_dir)
    ensure_dir(tables_dir)
    ensure_dir(manifest_dir)

    prefix = f"{product_name}_{percentile_label}_{estimator}"

    station_all_path = out_dir / f"zeta_station_all_{prefix}.csv"
    station_retained_path = out_dir / f"zeta_per_station_{prefix}.csv"

    station_all.to_csv(station_all_path, index=False)
    station_retained.to_csv(station_retained_path, index=False)

    qc_pairs_path = tables_dir / f"pair_qc_{product_name}_{percentile_label}.csv"
    if write_qc_pairs:
        qc_pairs.to_csv(qc_pairs_path, index=False)

    manifest = {
        "created_at": now_iso(),
        "product": product_name,
        "product_label": cfg.product(product_name).get("label", product_name),
        "percentile_label": percentile_label,
        "estimator": estimator,
        "start_year_filter": start_year,
        "end_year_filter": end_year,
        "zeta_definition": cfg.method["zeta"]["definition"],
        "zeta_method": estimator,
        "min_pairs_per_station": int(cfg.method["zeta"]["min_pairs_per_station"]),
        "ratio_qc": cfg.method["ratio_qc"],
        "ratio_handling": {
            "raw_ratio_column": "raw_ratio_gauge_over_product",
            "aggregation_ratio_column": "ratio_for_zeta",
            "clip_before_aggregation": True,
            "clip_bounds": cfg.method["ratio_qc"].get("ratio_clip", [0.25, 5.0]),
        },
        "n_pair_rows_raw": int(len(qc_pairs)),
        "n_pair_rows_used_for_zeta": int(qc_pairs["qc_used_for_zeta"].sum()),
        "n_stations_total": int(station_all.shape[0]),
        "n_stations_retained": int(station_retained.shape[0]),
        "source_pair_files": list(source_pair_files),
        "outputs": {
            "station_all": str(station_all_path),
            "station_retained": str(station_retained_path),
            "qc_pairs": str(qc_pairs_path) if write_qc_pairs else None,
        },
    }

    manifest_path = out_dir / f"zeta_manifest_{prefix}.json"
    write_json(manifest_path, manifest)

    latest_manifest_path = manifest_dir / f"zeta_manifest_{prefix}_latest.json"
    write_json(latest_manifest_path, manifest)

    return {
        "station_all": station_all_path,
        "station_retained": station_retained_path,
        "qc_pairs": qc_pairs_path,
        "manifest": manifest_path,
        "latest_manifest": latest_manifest_path,
    }


def compute_zeta_for_product_percentile(
    cfg: Any,
    product_name: str,
    percentile: str | float,
    estimator: str = "median",
    pairs_folder: Optional[Path] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    write_qc_pairs: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Compute station-level zeta for one product, percentile, and estimator.
    """
    percentile_label, percentile_value = parse_percentile_arg(percentile)
    estimator = str(estimator).lower()

    if verbose:
        print_header(f"Computing station zeta: {product_name} / {percentile_label} / {estimator}")

    pairs = load_pair_csvs(
        cfg=cfg,
        product_name=product_name,
        percentile_label=percentile_label,
        pairs_folder=pairs_folder,
        start_year=start_year,
        end_year=end_year,
        verbose=verbose,
    )

    source_files = sorted(pairs["source_file"].dropna().unique().tolist())

    qc_pairs = apply_pair_qc(
        pairs=pairs,
        cfg=cfg,
        product_name=product_name,
        percentile_label=percentile_label,
    )

    station_all, station_retained = compute_station_statistics(
        qc_pairs=qc_pairs,
        cfg=cfg,
        product_name=product_name,
        percentile_label=percentile_label,
        estimator=estimator,
    )

    outputs = write_zeta_outputs(
        cfg=cfg,
        product_name=product_name,
        percentile_label=percentile_label,
        estimator=estimator,
        qc_pairs=qc_pairs,
        station_all=station_all,
        station_retained=station_retained,
        source_pair_files=source_files,
        start_year=start_year,
        end_year=end_year,
        write_qc_pairs=write_qc_pairs,
    )

    if verbose:
        print_section("Zeta result")
        print(f"Raw pair rows:            {len(qc_pairs)}")
        print(f"Rows used for zeta:       {int(qc_pairs['qc_used_for_zeta'].sum())}")
        print(f"Stations with any pairs:  {station_all.shape[0]}")
        print(f"Stations retained:        {station_retained.shape[0]}")
        print(f"Estimator:                {estimator}")
        if station_retained.shape[0] > 0:
            print(f"Median selected zeta:     {station_retained['zeta_selected'].median():.4f}")
            print(f"Mean selected zeta:       {station_retained['zeta_selected'].mean():.4f}")
        print("\nOutputs:")
        for key, path in outputs.items():
            print(f"  {key:18s}: {path}")

    return outputs


def compute_zeta_batch(
    cfg: Any,
    products: Sequence[str],
    percentiles: Sequence[str | float],
    estimators: Sequence[str],
    pairs_folder: Optional[Path] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    write_qc_pairs: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Path]]:
    """
    Batch station-zeta computation.

    Note
    ----
    pairs_folder override should generally only be used for single product /
    single percentile runs. For batches, leave it as None.
    """
    outputs: List[Dict[str, Path]] = []

    if pairs_folder is not None and (len(products) > 1 or len(percentiles) > 1):
        raise ValueError(
            "--pairs-folder override is only allowed for a single product and "
            "single percentile run."
        )

    # Avoid rewriting the same potentially large QC pair table repeatedly if
    # both mean and median are requested. Write it for the first estimator only.
    for product_name in products:
        for percentile in percentiles:
            for i, estimator in enumerate(estimators):
                out = compute_zeta_for_product_percentile(
                    cfg=cfg,
                    product_name=product_name,
                    percentile=percentile,
                    estimator=estimator,
                    pairs_folder=pairs_folder,
                    start_year=start_year,
                    end_year=end_year,
                    write_qc_pairs=write_qc_pairs if i == 0 else False,
                    verbose=verbose,
                )
                outputs.append(out)

    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    compute_zeta_for_product_percentile(
        cfg=cfg,
        product_name="imerg_v07",
        percentile="p98",
        estimator="median",
    )


if __name__ == "__main__":
    main()
