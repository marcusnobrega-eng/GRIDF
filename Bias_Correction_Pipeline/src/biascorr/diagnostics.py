#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostics.py

Diagnostics and sensitivity analysis for the GRIDF rainfall-product
bias-correction pipeline.

Part 08 scope
-------------
This module analyzes outputs from Parts 04–07:

1. Pair-level QC diagnostics.
2. Station-level zeta diagnostics.
3. Interpolated zeta raster diagnostics.
4. Corrected annual maximum rainfall diagnostics.
5. Percentile sensitivity, using P98 as the default reference.
6. Mean-vs-median estimator sensitivity.

The module is designed to be conservative:
- It never silently assumes outputs exist.
- It writes clear CSV/JSON summaries.
- Figures are diagnostics first; paper-level figures can be refined from these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .event_selection import parse_percentile_arg
from .raster_utils import (
    profile_summary,
    raster_bounds_from_profile,
    read_or_resample_to_match,
    read_raster_masked,
)
from .utils import ensure_dir, now_iso, print_header, print_section, write_json
from .plot_utils import plot_histogram, plot_scatter, plot_timeseries, plot_raster_preview


def _base(cfg: Any, product: str, p_label: str) -> Path:
    return Path(cfg.data_root) / "products" / product / "sensitivity" / p_label


def _diag_dir(cfg: Any, product: str, p_label: str, estimator: str) -> Path:
    return _base(cfg, product, p_label) / "diagnostics" / estimator


def _fig_diag_dir(cfg: Any) -> Path:
    return Path(cfg.figures_root) / "diagnostics"


def _fig_sens_dir(cfg: Any) -> Path:
    return Path(cfg.figures_root) / "sensitivity"


def _pair_qc_path(cfg: Any, product: str, p_label: str) -> Path:
    return _base(cfg, product, p_label) / "tables" / f"pair_qc_{product}_{p_label}.csv"


def _zeta_station_path(cfg: Any, product: str, p_label: str, estimator: str) -> Path:
    return (
        _base(cfg, product, p_label)
        / "zeta_station"
        / estimator
        / f"zeta_per_station_{product}_{p_label}_{estimator}.csv"
    )


def _zeta_all_path(cfg: Any, product: str, p_label: str, estimator: str) -> Path:
    return (
        _base(cfg, product, p_label)
        / "zeta_station"
        / estimator
        / f"zeta_station_all_{product}_{p_label}_{estimator}.csv"
    )


def _correction_summary_path(cfg: Any, product: str, p_label: str, estimator: str) -> Path:
    return (
        _base(cfg, product, p_label)
        / "annual_max_corrected"
        / estimator
        / f"annual_max_correction_summary_{product}_{p_label}_{estimator}.csv"
    )


def _zeta_grid_dir(cfg: Any, product: str, p_label: str, estimator: str) -> Path:
    return _base(cfg, product, p_label) / "zeta_grid" / estimator


def find_zeta_raster(cfg: Any, product: str, p_label: str, estimator: str) -> Optional[Path]:
    folder = _zeta_grid_dir(cfg, product, p_label, estimator)
    if not folder.exists():
        return None
    candidates = sorted(folder.glob(f"zeta_map_{product}_{p_label}_{estimator}_*.tif"))
    if not candidates:
        candidates = sorted(folder.glob("zeta_map_*.tif"))
    return candidates[-1] if candidates else None


def finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "max": None,
            "std": None,
        }
    return {
        "n": int(arr.size),
        "min": float(np.nanmin(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p25": float(np.nanpercentile(arr, 25)),
        "median": float(np.nanmedian(arr)),
        "mean": float(np.nanmean(arr)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p90": float(np.nanpercentile(arr, 90)),
        "max": float(np.nanmax(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def summarize_pair_qc(pair_qc: pd.DataFrame) -> Dict[str, Any]:
    """Summarize event-level pair QC table."""
    summary: Dict[str, Any] = {
        "n_rows": int(len(pair_qc)),
        "n_stations": int(pair_qc["station_id"].nunique()) if "station_id" in pair_qc else None,
        "n_years": int(pair_qc["year"].nunique()) if "year" in pair_qc else None,
    }

    for col in [
        "gauge_mm",
        "product_mm",
        "raw_ratio_gauge_over_product",
        "ratio_for_zeta",
    ]:
        if col in pair_qc.columns:
            summary[col] = finite_stats(pair_qc[col].values)

    for col in [
        "qc_basic_pass",
        "qc_used_for_zeta",
        "ratio_clipped_low",
        "ratio_clipped_high",
    ]:
        if col in pair_qc.columns:
            summary[col] = int(pair_qc[col].astype(bool).sum())

    if "source_file" in pair_qc.columns:
        summary["n_source_files"] = int(pair_qc["source_file"].nunique())

    return summary


def summarize_zeta_table(zeta: pd.DataFrame) -> Dict[str, Any]:
    """Summarize station-level zeta table."""
    summary: Dict[str, Any] = {
        "n_stations": int(len(zeta)),
    }

    for col in [
        "n_pairs_used",
        "zeta_selected",
        "zeta_mean",
        "zeta_median",
        "zeta_slope0",
        "zeta_iqr",
    ]:
        if col in zeta.columns:
            summary[col] = finite_stats(zeta[col].values)

    return summary


def summarize_correction_summary(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize annual maximum correction summary table."""
    out: Dict[str, Any] = {
        "n_years": int(len(summary_df)),
        "years": summary_df["year"].astype(int).tolist() if "year" in summary_df else [],
    }

    for col in [
        "raw_mean",
        "corrected_mean",
        "raw_max",
        "corrected_max",
        "zeta_mean",
        "zeta_median",
    ]:
        if col in summary_df.columns:
            out[col] = finite_stats(summary_df[col].values)

    if "raw_mean" in summary_df.columns and "corrected_mean" in summary_df.columns:
        ratio = summary_df["corrected_mean"] / summary_df["raw_mean"]
        out["corrected_over_raw_mean_ratio"] = finite_stats(ratio.values)

    return out


def zeta_raster_summary(raster_path: Path) -> Dict[str, Any]:
    """Summarize gridded zeta raster."""
    profile, data, mask = read_raster_masked(raster_path)
    vals = data[mask & np.isfinite(data)]
    return {
        "raster": str(raster_path),
        "profile": profile_summary(profile),
        "stats": finite_stats(vals),
        "n_valid_pixels": int(vals.size),
    }


def create_basic_diagnostic_figures(
    cfg: Any,
    product: str,
    p_label: str,
    estimator: str,
    pair_qc: Optional[pd.DataFrame],
    zeta: Optional[pd.DataFrame],
    correction: Optional[pd.DataFrame],
    zeta_raster: Optional[Path],
    out_dir: Path,
) -> List[str]:
    """Create core diagnostic figures."""
    paths: List[str] = []

    fig_dir = _fig_diag_dir(cfg) / product / p_label / estimator
    ensure_dir(fig_dir)

    if pair_qc is not None and not pair_qc.empty:
        if "raw_ratio_gauge_over_product" in pair_qc.columns:
            p = fig_dir / f"hist_raw_ratio_{product}_{p_label}_{estimator}.png"
            plot_histogram(
                pair_qc["raw_ratio_gauge_over_product"].values,
                p,
                title=f"{product} {p_label}: raw gauge/product ratios",
                xlabel="Raw gauge/product ratio",
                bins=80,
            )
            paths.append(str(p))

        if "ratio_for_zeta" in pair_qc.columns:
            p = fig_dir / f"hist_ratio_for_zeta_{product}_{p_label}_{estimator}.png"
            plot_histogram(
                pair_qc["ratio_for_zeta"].values,
                p,
                title=f"{product} {p_label}: ratio used for zeta",
                xlabel="Clipped gauge/product ratio",
                bins=80,
            )
            paths.append(str(p))

        if "product_mm" in pair_qc.columns and "gauge_mm" in pair_qc.columns:
            p = fig_dir / f"scatter_gauge_vs_product_events_{product}_{p_label}_{estimator}.png"
            plot_scatter(
                pair_qc["product_mm"].values,
                pair_qc["gauge_mm"].values,
                p,
                title=f"{product} {p_label}: event rainfall pairs",
                xlabel="Product rainfall (mm/day)",
                ylabel="Gauge rainfall (mm/day)",
                identity=True,
                alpha=0.30,
                s=8,
            )
            paths.append(str(p))

    if zeta is not None and not zeta.empty:
        p = fig_dir / f"hist_station_zeta_{product}_{p_label}_{estimator}.png"
        plot_histogram(
            zeta["zeta_selected"].values,
            p,
            title=f"{product} {p_label}: station zeta ({estimator})",
            xlabel="Station zeta",
            bins=50,
        )
        paths.append(str(p))

        if "zeta_mean" in zeta.columns and "zeta_median" in zeta.columns:
            p = fig_dir / f"scatter_zeta_mean_vs_median_{product}_{p_label}_{estimator}.png"
            plot_scatter(
                zeta["zeta_mean"].values,
                zeta["zeta_median"].values,
                p,
                title=f"{product} {p_label}: mean vs median station zeta",
                xlabel="Mean zeta",
                ylabel="Median zeta",
                identity=True,
                alpha=0.50,
                s=15,
            )
            paths.append(str(p))

    if correction is not None and not correction.empty and "year" in correction.columns:
        if "raw_mean" in correction.columns and "corrected_mean" in correction.columns:
            p = fig_dir / f"timeseries_mean_annual_max_{product}_{p_label}_{estimator}.png"
            plot_timeseries(
                correction["year"].values,
                [correction["raw_mean"].values, correction["corrected_mean"].values],
                ["Raw mean", "Corrected mean"],
                p,
                title=f"{product} {p_label}: annual maximum mean rainfall",
                ylabel="Rainfall (mm/day)",
            )
            paths.append(str(p))

        if "raw_max" in correction.columns and "corrected_max" in correction.columns:
            p = fig_dir / f"timeseries_max_annual_max_{product}_{p_label}_{estimator}.png"
            plot_timeseries(
                correction["year"].values,
                [correction["raw_max"].values, correction["corrected_max"].values],
                ["Raw max", "Corrected max"],
                p,
                title=f"{product} {p_label}: annual maximum raster maximum",
                ylabel="Rainfall (mm/day)",
            )
            paths.append(str(p))

    if zeta_raster is not None and zeta_raster.exists():
        try:
            profile, data, mask = read_raster_masked(zeta_raster)
            bounds = raster_bounds_from_profile(profile)
            extent = (bounds[0], bounds[2], bounds[1], bounds[3])
            p = fig_dir / f"zeta_raster_preview_{product}_{p_label}_{estimator}.png"
            points_x = zeta["longitude"].values if zeta is not None and "longitude" in zeta.columns else None
            points_y = zeta["latitude"].values if zeta is not None and "latitude" in zeta.columns else None
            plot_raster_preview(
                data,
                p,
                title=f"{product} {p_label}: gridded zeta ({estimator})",
                extent=extent,
                points_x=points_x,
                points_y=points_y,
                colorbar_label="Zeta",
            )
            paths.append(str(p))
        except Exception:
            pass

    return paths


def run_diagnostics_for_product_percentile(
    cfg: Any,
    product: str,
    percentile: str | float,
    estimator: str = "median",
    make_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Run full diagnostics for one product/percentile/estimator."""
    p_label, p_value = parse_percentile_arg(percentile)
    estimator = str(estimator).lower()

    if verbose:
        print_header(f"Diagnostics: {product} / {p_label} / {estimator}")

    out_dir = _diag_dir(cfg, product, p_label, estimator)
    ensure_dir(out_dir)

    pair_path = _pair_qc_path(cfg, product, p_label)
    zeta_path = _zeta_station_path(cfg, product, p_label, estimator)
    zeta_all_path = _zeta_all_path(cfg, product, p_label, estimator)
    correction_path = _correction_summary_path(cfg, product, p_label, estimator)
    zraster = find_zeta_raster(cfg, product, p_label, estimator)

    pair_qc = pd.read_csv(pair_path, low_memory=False) if pair_path.exists() else None
    zeta = pd.read_csv(zeta_path, low_memory=False) if zeta_path.exists() else None
    zeta_all = pd.read_csv(zeta_all_path, low_memory=False) if zeta_all_path.exists() else None
    correction = pd.read_csv(correction_path, low_memory=False) if correction_path.exists() else None

    summary: Dict[str, Any] = {
        "created_at": now_iso(),
        "product": product,
        "percentile_label": p_label,
        "percentile_value": p_value,
        "estimator": estimator,
        "inputs": {
            "pair_qc": str(pair_path) if pair_path.exists() else None,
            "zeta_retained": str(zeta_path) if zeta_path.exists() else None,
            "zeta_all": str(zeta_all_path) if zeta_all_path.exists() else None,
            "correction_summary": str(correction_path) if correction_path.exists() else None,
            "zeta_raster": str(zraster) if zraster is not None else None,
        },
        "pair_qc_summary": summarize_pair_qc(pair_qc) if pair_qc is not None else None,
        "zeta_retained_summary": summarize_zeta_table(zeta) if zeta is not None else None,
        "zeta_all_summary": summarize_zeta_table(zeta_all) if zeta_all is not None else None,
        "correction_summary": summarize_correction_summary(correction) if correction is not None else None,
        "zeta_raster_summary": zeta_raster_summary(zraster) if zraster is not None and zraster.exists() else None,
        "figures": [],
    }

    if make_figures:
        figs = create_basic_diagnostic_figures(
            cfg=cfg,
            product=product,
            p_label=p_label,
            estimator=estimator,
            pair_qc=pair_qc,
            zeta=zeta,
            correction=correction,
            zeta_raster=zraster,
            out_dir=out_dir,
        )
        summary["figures"] = figs

    summary_json = out_dir / f"diagnostic_summary_{product}_{p_label}_{estimator}.json"
    write_json(summary_json, summary)

    # Compact CSV of key metrics.
    metrics = {
        "product": product,
        "percentile_label": p_label,
        "estimator": estimator,
        "n_pair_rows": None if summary["pair_qc_summary"] is None else summary["pair_qc_summary"]["n_rows"],
        "n_pairs_used_for_zeta": None if summary["pair_qc_summary"] is None else summary["pair_qc_summary"].get("qc_used_for_zeta"),
        "n_station_zeta_retained": None if summary["zeta_retained_summary"] is None else summary["zeta_retained_summary"]["n_stations"],
        "median_zeta_selected": None if summary["zeta_retained_summary"] is None else summary["zeta_retained_summary"]["zeta_selected"]["median"],
        "mean_zeta_selected": None if summary["zeta_retained_summary"] is None else summary["zeta_retained_summary"]["zeta_selected"]["mean"],
        "n_corrected_years": None if summary["correction_summary"] is None else summary["correction_summary"]["n_years"],
    }
    metrics_path = out_dir / f"diagnostic_metrics_{product}_{p_label}_{estimator}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    if verbose:
        print_section("Diagnostic outputs")
        print(f"JSON:    {summary_json}")
        print(f"Metrics: {metrics_path}")
        if summary["figures"]:
            print(f"Figures: {len(summary['figures'])}")

    return {
        "summary_json": summary_json,
        "metrics_csv": metrics_path,
    }


def compare_percentile_sensitivity(
    cfg: Any,
    product: str,
    estimator: str = "median",
    reference_percentile: str = "p98",
    percentiles: Optional[Sequence[str]] = None,
    compare_rasters: bool = True,
    make_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Compare station and raster zeta sensitivity across percentile thresholds.

    Reference is P98 by default.
    """
    if percentiles is None:
        percentiles = cfg.method["event_selection"]["percentile_labels"]

    ref_label, _ = parse_percentile_arg(reference_percentile)
    estimator = str(estimator).lower()

    out_dir = _fig_sens_dir(cfg) / "percentile" / product / estimator
    ensure_dir(out_dir)

    if verbose:
        print_header(f"Percentile sensitivity: {product} / {estimator} / reference={ref_label}")

    ref_path = _zeta_station_path(cfg, product, ref_label, estimator)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference zeta table not found: {ref_path}")

    ref = pd.read_csv(ref_path, low_memory=False)
    ref = ref[["station_id", "zeta_selected", "latitude", "longitude", "n_pairs_used"]].copy()
    ref = ref.rename(columns={
        "zeta_selected": f"zeta_{ref_label}",
        "n_pairs_used": f"n_pairs_{ref_label}",
    })
    ref["station_id"] = ref["station_id"].astype(str)

    rows: List[Dict[str, Any]] = []

    for pct in percentiles:
        p_label, _ = parse_percentile_arg(pct)
        path = _zeta_station_path(cfg, product, p_label, estimator)
        if not path.exists():
            rows.append({
                "product": product,
                "estimator": estimator,
                "reference_percentile": ref_label,
                "percentile": p_label,
                "status": "missing_zeta_table",
            })
            continue

        cur = pd.read_csv(path, low_memory=False)
        cur = cur[["station_id", "zeta_selected", "n_pairs_used"]].copy()
        cur["station_id"] = cur["station_id"].astype(str)
        cur = cur.rename(columns={
            "zeta_selected": f"zeta_{p_label}",
            "n_pairs_used": f"n_pairs_{p_label}",
        })

        merged = ref.merge(cur, on="station_id", how="inner")
        x = merged[f"zeta_{ref_label}"].astype(float).values
        y = merged[f"zeta_{p_label}"].astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() >= 2:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            mae = float(np.nanmean(np.abs(y[mask] - x[mask])))
            med_abs = float(np.nanmedian(np.abs(y[mask] - x[mask])))
            med_rel = float(np.nanmedian(np.abs(y[mask] - x[mask]) / np.maximum(np.abs(x[mask]), 1e-12)))
        else:
            corr = np.nan
            mae = np.nan
            med_abs = np.nan
            med_rel = np.nan

        row = {
            "product": product,
            "estimator": estimator,
            "reference_percentile": ref_label,
            "percentile": p_label,
            "status": "ok",
            "n_common_stations": int(mask.sum()),
            "n_reference_stations": int(len(ref)),
            "n_percentile_stations": int(len(cur)),
            "station_zeta_correlation": corr,
            "station_zeta_mae": mae,
            "station_zeta_median_abs_diff": med_abs,
            "station_zeta_median_relative_abs_diff": med_rel,
            "median_pairs_reference": float(np.nanmedian(merged[f"n_pairs_{ref_label}"])) if len(merged) else np.nan,
            "median_pairs_percentile": float(np.nanmedian(merged[f"n_pairs_{p_label}"])) if len(merged) else np.nan,
        }

        # Raster comparison if both rasters exist.
        if compare_rasters and p_label != ref_label:
            ref_raster = find_zeta_raster(cfg, product, ref_label, estimator)
            cur_raster = find_zeta_raster(cfg, product, p_label, estimator)
            if ref_raster is not None and cur_raster is not None:
                ref_prof, ref_data, ref_mask = read_raster_masked(ref_raster)
                cur_data, was_resampled, _ = read_or_resample_to_match(cur_raster, ref_prof)
                mask_r = ref_mask & np.isfinite(ref_data) & np.isfinite(cur_data)
                if mask_r.any():
                    diff = cur_data[mask_r] - ref_data[mask_r]
                    rel = np.abs(diff) / np.maximum(np.abs(ref_data[mask_r]), 1e-12)
                    row.update({
                        "raster_compared": True,
                        "raster_resampled_to_reference": bool(was_resampled),
                        "raster_mae": float(np.nanmean(np.abs(diff))),
                        "raster_median_abs_diff": float(np.nanmedian(np.abs(diff))),
                        "raster_median_relative_abs_diff": float(np.nanmedian(rel)),
                    })
                else:
                    row["raster_compared"] = False
            else:
                row["raster_compared"] = False

        rows.append(row)

        if make_figures and p_label != ref_label and mask.sum() > 0:
            fig_path = out_dir / f"scatter_station_zeta_{product}_{p_label}_vs_{ref_label}_{estimator}.png"
            plot_scatter(
                x,
                y,
                fig_path,
                title=f"{product}: station zeta {p_label} vs {ref_label}",
                xlabel=f"Zeta {ref_label}",
                ylabel=f"Zeta {p_label}",
                identity=True,
                alpha=0.55,
                s=18,
            )

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"percentile_sensitivity_{product}_{estimator}_ref_{ref_label}.csv"
    df.to_csv(out_csv, index=False)

    manifest = {
        "created_at": now_iso(),
        "product": product,
        "estimator": estimator,
        "reference_percentile": ref_label,
        "percentiles": list(percentiles),
        "compare_rasters": compare_rasters,
        "output_csv": str(out_csv),
    }
    out_json = out_dir / f"percentile_sensitivity_{product}_{estimator}_ref_{ref_label}.json"
    write_json(out_json, manifest)

    if verbose:
        print_section("Percentile sensitivity outputs")
        print(out_csv)
        print(out_json)

    return {
        "csv": out_csv,
        "manifest": out_json,
    }


def compare_mean_median_sensitivity(
    cfg: Any,
    product: str,
    percentile: str = "p98",
    compare_rasters: bool = True,
    make_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Compare mean and median station/raster zeta."""
    p_label, _ = parse_percentile_arg(percentile)

    out_dir = _fig_sens_dir(cfg) / "mean_vs_median" / product / p_label
    ensure_dir(out_dir)

    if verbose:
        print_header(f"Mean-vs-median sensitivity: {product} / {p_label}")

    med_path = _zeta_station_path(cfg, product, p_label, "median")
    mean_path = _zeta_station_path(cfg, product, p_label, "mean")

    if not med_path.exists():
        raise FileNotFoundError(f"Median zeta table not found: {med_path}")
    if not mean_path.exists():
        raise FileNotFoundError(f"Mean zeta table not found: {mean_path}")

    med = pd.read_csv(med_path, low_memory=False)
    mean = pd.read_csv(mean_path, low_memory=False)

    med = med[["station_id", "zeta_selected", "n_pairs_used"]].rename(columns={
        "zeta_selected": "zeta_median_selected",
        "n_pairs_used": "n_pairs_median",
    })
    mean = mean[["station_id", "zeta_selected", "n_pairs_used"]].rename(columns={
        "zeta_selected": "zeta_mean_selected",
        "n_pairs_used": "n_pairs_mean",
    })

    med["station_id"] = med["station_id"].astype(str)
    mean["station_id"] = mean["station_id"].astype(str)

    merged = med.merge(mean, on="station_id", how="inner")
    x = merged["zeta_median_selected"].astype(float).values
    y = merged["zeta_mean_selected"].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() >= 2:
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        mae = float(np.nanmean(np.abs(y[mask] - x[mask])))
        med_abs = float(np.nanmedian(np.abs(y[mask] - x[mask])))
        med_rel = float(np.nanmedian(np.abs(y[mask] - x[mask]) / np.maximum(np.abs(x[mask]), 1e-12)))
    else:
        corr = np.nan
        mae = np.nan
        med_abs = np.nan
        med_rel = np.nan

    row = {
        "product": product,
        "percentile": p_label,
        "n_common_stations": int(mask.sum()),
        "n_median_stations": int(len(med)),
        "n_mean_stations": int(len(mean)),
        "station_zeta_correlation": corr,
        "station_zeta_mae": mae,
        "station_zeta_median_abs_diff": med_abs,
        "station_zeta_median_relative_abs_diff": med_rel,
        "median_zeta_median_estimator": float(np.nanmedian(x[mask])) if mask.any() else np.nan,
        "median_zeta_mean_estimator": float(np.nanmedian(y[mask])) if mask.any() else np.nan,
    }

    if compare_rasters:
        med_raster = find_zeta_raster(cfg, product, p_label, "median")
        mean_raster = find_zeta_raster(cfg, product, p_label, "mean")
        if med_raster is not None and mean_raster is not None:
            med_prof, med_data, med_mask = read_raster_masked(med_raster)
            mean_data, was_resampled, _ = read_or_resample_to_match(mean_raster, med_prof)
            mask_r = med_mask & np.isfinite(med_data) & np.isfinite(mean_data)
            if mask_r.any():
                diff = mean_data[mask_r] - med_data[mask_r]
                rel = np.abs(diff) / np.maximum(np.abs(med_data[mask_r]), 1e-12)
                row.update({
                    "raster_compared": True,
                    "raster_resampled_to_median": bool(was_resampled),
                    "raster_mae": float(np.nanmean(np.abs(diff))),
                    "raster_median_abs_diff": float(np.nanmedian(np.abs(diff))),
                    "raster_median_relative_abs_diff": float(np.nanmedian(rel)),
                })
            else:
                row["raster_compared"] = False
        else:
            row["raster_compared"] = False

    out_csv = out_dir / f"mean_vs_median_sensitivity_{product}_{p_label}.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    if make_figures and mask.sum() > 0:
        fig_path = out_dir / f"scatter_station_zeta_mean_vs_median_{product}_{p_label}.png"
        plot_scatter(
            x,
            y,
            fig_path,
            title=f"{product} {p_label}: mean vs median station zeta",
            xlabel="Median-estimator zeta",
            ylabel="Mean-estimator zeta",
            identity=True,
            alpha=0.55,
            s=18,
        )

    out_json = out_dir / f"mean_vs_median_sensitivity_{product}_{p_label}.json"
    write_json(out_json, {
        "created_at": now_iso(),
        "product": product,
        "percentile": p_label,
        "output_csv": str(out_csv),
        "metrics": row,
    })

    if verbose:
        print_section("Mean-vs-median outputs")
        print(out_csv)
        print(out_json)

    return {
        "csv": out_csv,
        "manifest": out_json,
    }


def diagnostics_batch(
    cfg: Any,
    products: Sequence[str],
    percentiles: Sequence[str | float],
    estimators: Sequence[str],
    make_figures: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Path]]:
    """Run diagnostics for multiple products/percentiles/estimators."""
    outputs: List[Dict[str, Path]] = []
    for product in products:
        for pct in percentiles:
            for est in estimators:
                outputs.append(
                    run_diagnostics_for_product_percentile(
                        cfg=cfg,
                        product=product,
                        percentile=pct,
                        estimator=est,
                        make_figures=make_figures,
                        verbose=verbose,
                    )
                )
    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    run_diagnostics_for_product_percentile(cfg, "imerg_v07", "p98", "median")


if __name__ == "__main__":
    main()
