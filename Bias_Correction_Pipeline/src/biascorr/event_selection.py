#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_selection.py

Select gauge extreme rainfall events for the GRIDF bias-correction pipeline.

This module implements the gauge-side event selection before GEE sampling.

Scientific logic
----------------
For each rainfall product and percentile threshold:

1. Use the product's configured/available years.
2. For each station and calendar year, compute a station-year percentile
   threshold from all valid daily gauge rainfall values.
3. Select candidate days where gauge rainfall exceeds the threshold.
4. Apply ratio-oriented rainfall QC before sending events to GEE:
      gauge rainfall > min_gauge_rainfall_for_ratio_mm
      gauge rainfall <= max_rainfall_for_ratio_mm
5. Decluster candidate events using a minimum temporal separation.
   If multiple candidates fall within the exclusion window, the largest
   rainfall event is retained.
6. Save product/percentile-specific event tables.

Important
---------
This stage does NOT sample satellite/product data. It only prepares the
station/date/gauge rainfall events that will be used by the GEE pair-export
stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import product_available_years
from .gauges import GaugeTable, load_gauge_table, rainfall_values_to_numeric
from .utils import (
    ensure_dir,
    label_to_percentile,
    now_iso,
    percentile_to_label,
    print_header,
    print_section,
    write_json,
)


EVENT_COLUMNS = [
    "product",
    "product_label",
    "percentile_label",
    "percentile_value",
    "percentile_basis",
    "station_id",
    "station_name",
    "city",
    "state",
    "latitude",
    "longitude",
    "row_index",
    "year",
    "date",
    "gauge_mm",
    "threshold_mm",
    "n_valid_days_year",
    "n_candidates_above_threshold",
    "n_candidates_after_rain_qc",
    "n_events_after_declustering",
    "min_gap_days",
]


SUMMARY_COLUMNS = [
    "product",
    "percentile_label",
    "percentile_value",
    "station_id",
    "station_name",
    "row_index",
    "latitude",
    "longitude",
    "valid_coordinates",
    "year",
    "n_valid_days_year",
    "threshold_mm",
    "n_candidates_above_threshold",
    "n_candidates_after_rain_qc",
    "n_events_after_declustering",
    "status",
]


def parse_percentile_arg(value: str | float) -> Tuple[str, float]:
    """
    Parse percentile argument.

    Accepts:
        p98
        0.98
        98
        p995
    """
    if isinstance(value, (float, int)):
        pct = float(value)
        if pct > 1.0:
            pct = pct / 100.0
        return percentile_to_label(pct), pct

    text = str(value).strip().lower()
    if text.startswith("p"):
        pct = label_to_percentile(text)
        return percentile_to_label(pct), pct

    pct = float(text)
    if pct > 1.0:
        pct = pct / 100.0

    return percentile_to_label(pct), pct


def decluster_events(
    dates: Sequence[pd.Timestamp],
    rainfall_mm: Sequence[float],
    min_gap_days: int,
) -> List[int]:
    """
    Decluster candidate events.

    Strategy
    --------
    Candidates are sorted from largest to smallest rainfall. The largest event
    is accepted first. Any other candidate within min_gap_days of an accepted
    event is rejected. The final selected indices are returned in chronological
    order.

    If min_gap_days = 3, two retained events must be at least 3 days apart.
    """
    if len(dates) == 0:
        return []

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    rainfall = np.asarray(rainfall_mm, dtype=float)

    # Sort by rainfall descending; stable sort by date for ties.
    order = sorted(range(len(rainfall)), key=lambda i: (-rainfall[i], dates.iloc[i]))

    selected: List[int] = []
    selected_dates: List[pd.Timestamp] = []

    for i in order:
        d = dates.iloc[i]
        too_close = False
        for sd in selected_dates:
            if abs((d - sd).days) < int(min_gap_days):
                too_close = True
                break
        if not too_close:
            selected.append(i)
            selected_dates.append(d)

    selected_sorted = sorted(selected, key=lambda i: dates.iloc[i])
    return selected_sorted


def _year_date_columns(table: GaugeTable, years: Sequence[int]) -> Dict[int, List[str]]:
    """Map year -> date columns."""
    by_year: Dict[int, List[str]] = {int(y): [] for y in years}
    for col, date in table.parsed_dates.items():
        y = int(date.year)
        if y in by_year:
            by_year[y].append(col)
    return by_year


def _event_output_dir(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "events"
    )


def select_events_for_product_percentile(
    cfg: Any,
    product_name: str,
    percentile: str | float,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    station_limit: Optional[int] = None,
    overwrite: bool = True,
    write_yearly_files: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Select gauge events for one product and one percentile.

    Outputs
    -------
    data/products/<product>/sensitivity/<pXX>/events/
        events_<product>_<pXX>_all_years.csv
        events_<product>_<pXX>_<year>.csv
        event_selection_summary_<product>_<pXX>.csv
        event_selection_manifest_<product>_<pXX>.json
    """
    percentile_label, percentile_value = parse_percentile_arg(percentile)

    product_cfg = cfg.product(product_name)
    product_label = product_cfg.get("label", product_name)

    inventory = product_available_years(cfg, product_name)
    years = list(inventory["processed_years"])

    if start_year is not None:
        years = [y for y in years if y >= int(start_year)]
    if end_year is not None:
        years = [y for y in years if y <= int(end_year)]

    if not years:
        raise ValueError(
            f"No processing years available for product={product_name}, "
            f"start_year={start_year}, end_year={end_year}."
        )

    out_dir = _event_output_dir(cfg, product_name, percentile_label)
    ensure_dir(out_dir)

    all_events_path = out_dir / f"events_{product_name}_{percentile_label}_all_years.csv"
    summary_path = out_dir / f"event_selection_summary_{product_name}_{percentile_label}.csv"
    manifest_path = out_dir / f"event_selection_manifest_{product_name}_{percentile_label}.json"

    if all_events_path.exists() and not overwrite:
        if verbose:
            print(f"[skip] events already exist: {all_events_path}")
        return {
            "events_all_years": all_events_path,
            "summary": summary_path,
            "manifest": manifest_path,
        }

    if verbose:
        print_header(f"Selecting gauge events: {product_name} / {percentile_label}")
        print(f"Product label: {product_label}")
        print(f"Years:         {years[0]}–{years[-1]} ({len(years)} years)")
        print(f"Percentile:    {percentile_value}")
        print(f"Output folder: {out_dir}")

    table = load_gauge_table(Path(cfg.paths["inputs"]["gauge_timeseries_csv"]))

    if station_limit is not None:
        station_indices = list(table.df.index[: int(station_limit)])
    else:
        station_indices = list(table.df.index)

    year_cols = _year_date_columns(table, years)

    event_selection_cfg = cfg.method["event_selection"]
    gauge_qc = cfg.method["gauge_qc"]
    ratio_qc = cfg.method["ratio_qc"]

    min_valid = float(gauge_qc["min_valid_rain_mm"])
    max_valid = float(gauge_qc["max_valid_rain_mm"])

    min_ratio_rain = float(ratio_qc["min_gauge_rainfall_for_ratio_mm"])
    max_ratio_rain = float(ratio_qc["max_rainfall_for_ratio_mm"])

    min_gap_days = int(event_selection_cfg["min_gap_days"])
    percentile_basis = str(event_selection_cfg["percentile_basis"])

    events: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for count, idx in enumerate(station_indices, start=1):
        meta = table.normalized_metadata.loc[idx]

        station_id = meta["station_id"]
        station_name = meta["station_name"]
        lat = meta["latitude"]
        lon = meta["longitude"]
        valid_coordinates = bool(meta["valid_coordinates"])

        if verbose and (count == 1 or count % 250 == 0):
            print(f"  Processing station {count}/{len(station_indices)}: {station_id}")

        for year in years:
            cols = year_cols.get(year, [])

            if not cols:
                summary_rows.append({
                    "product": product_name,
                    "percentile_label": percentile_label,
                    "percentile_value": percentile_value,
                    "station_id": station_id,
                    "station_name": station_name,
                    "row_index": int(idx),
                    "latitude": lat,
                    "longitude": lon,
                    "valid_coordinates": valid_coordinates,
                    "year": int(year),
                    "n_valid_days_year": 0,
                    "threshold_mm": np.nan,
                    "n_candidates_above_threshold": 0,
                    "n_candidates_after_rain_qc": 0,
                    "n_events_after_declustering": 0,
                    "status": "no_date_columns_for_year",
                })
                continue

            vals = rainfall_values_to_numeric(table.df.loc[idx, cols].values)
            dates = table.parsed_dates.loc[cols].reset_index(drop=True)

            valid = (
                np.isfinite(vals)
                & (vals >= min_valid)
                & (vals <= max_valid)
            )

            n_valid = int(valid.sum())

            if n_valid == 0:
                summary_rows.append({
                    "product": product_name,
                    "percentile_label": percentile_label,
                    "percentile_value": percentile_value,
                    "station_id": station_id,
                    "station_name": station_name,
                    "row_index": int(idx),
                    "latitude": lat,
                    "longitude": lon,
                    "valid_coordinates": valid_coordinates,
                    "year": int(year),
                    "n_valid_days_year": 0,
                    "threshold_mm": np.nan,
                    "n_candidates_above_threshold": 0,
                    "n_candidates_after_rain_qc": 0,
                    "n_events_after_declustering": 0,
                    "status": "no_valid_rainfall",
                })
                continue

            threshold = float(np.nanpercentile(vals[valid], percentile_value * 100.0))

            candidate = valid & (vals > threshold)
            n_candidate = int(candidate.sum())

            # Ratio-oriented rain QC. We do it here to avoid sending unusable
            # tiny rainfall events to GEE in Part 04.
            after_rain_qc = (
                candidate
                & (vals > min_ratio_rain)
                & (vals <= max_ratio_rain)
            )
            n_after_rain_qc = int(after_rain_qc.sum())

            candidate_idx = np.where(after_rain_qc)[0]
            candidate_dates = dates.iloc[candidate_idx].reset_index(drop=True)
            candidate_vals = vals[candidate_idx]

            selected_local_idx = decluster_events(
                dates=candidate_dates,
                rainfall_mm=candidate_vals,
                min_gap_days=min_gap_days,
            )

            selected_global_idx = [int(candidate_idx[i]) for i in selected_local_idx]
            n_selected = int(len(selected_global_idx))

            if not valid_coordinates:
                status = "invalid_coordinates_events_kept_for_qc"
            elif n_selected == 0:
                status = "no_events_after_qc"
            else:
                status = "ok"

            summary_rows.append({
                "product": product_name,
                "percentile_label": percentile_label,
                "percentile_value": percentile_value,
                "station_id": station_id,
                "station_name": station_name,
                "row_index": int(idx),
                "latitude": lat,
                "longitude": lon,
                "valid_coordinates": valid_coordinates,
                "year": int(year),
                "n_valid_days_year": n_valid,
                "threshold_mm": threshold,
                "n_candidates_above_threshold": n_candidate,
                "n_candidates_after_rain_qc": n_after_rain_qc,
                "n_events_after_declustering": n_selected,
                "status": status,
            })

            for j in selected_global_idx:
                d = pd.Timestamp(dates.iloc[j])
                events.append({
                    "product": product_name,
                    "product_label": product_label,
                    "percentile_label": percentile_label,
                    "percentile_value": percentile_value,
                    "percentile_basis": percentile_basis,
                    "station_id": station_id,
                    "station_name": station_name,
                    "city": meta.get("city", ""),
                    "state": meta.get("state", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "row_index": int(idx),
                    "year": int(year),
                    "date": d.strftime("%Y-%m-%d"),
                    "gauge_mm": float(vals[j]),
                    "threshold_mm": threshold,
                    "n_valid_days_year": n_valid,
                    "n_candidates_above_threshold": n_candidate,
                    "n_candidates_after_rain_qc": n_after_rain_qc,
                    "n_events_after_declustering": n_selected,
                    "min_gap_days": min_gap_days,
                })

    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    events_df.to_csv(all_events_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    yearly_paths: List[str] = []
    if write_yearly_files:
        for year in years:
            yearly = events_df.loc[events_df["year"] == int(year)].copy()
            yearly_path = out_dir / f"events_{product_name}_{percentile_label}_{year}.csv"
            yearly.to_csv(yearly_path, index=False)
            yearly_paths.append(str(yearly_path))

    n_events = int(len(events_df))
    n_stations_with_events = int(events_df["station_id"].nunique()) if n_events else 0
    n_station_years_with_events = int(
        summary_df.loc[summary_df["n_events_after_declustering"] > 0, ["station_id", "year"]]
        .drop_duplicates()
        .shape[0]
    )

    manifest = {
        "created_at": now_iso(),
        "product": product_name,
        "product_label": product_label,
        "percentile_label": percentile_label,
        "percentile_value": percentile_value,
        "years": years,
        "n_years": len(years),
        "station_limit": station_limit,
        "n_stations_processed": len(station_indices),
        "n_events": n_events,
        "n_stations_with_events": n_stations_with_events,
        "n_station_years_with_events": n_station_years_with_events,
        "method": {
            "percentile_basis": percentile_basis,
            "min_gap_days": min_gap_days,
            "gauge_qc": gauge_qc,
            "ratio_qc": ratio_qc,
            "declustering": "largest_event_kept_within_min_gap_window",
        },
        "outputs": {
            "events_all_years": str(all_events_path),
            "summary": str(summary_path),
            "yearly_event_files": yearly_paths,
        },
    }

    write_json(manifest_path, manifest)

    if verbose:
        print_section("Event-selection result")
        print(f"Events selected:              {n_events}")
        print(f"Stations with events:         {n_stations_with_events}")
        print(f"Station-years with events:    {n_station_years_with_events}")
        print(f"All-years event file:         {all_events_path}")
        print(f"Summary file:                 {summary_path}")
        print(f"Manifest:                     {manifest_path}")

    return {
        "events_all_years": all_events_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def select_events_batch(
    cfg: Any,
    products: Sequence[str],
    percentiles: Sequence[str | float],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    station_limit: Optional[int] = None,
    overwrite: bool = True,
    write_yearly_files: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Path]]:
    """Run event selection for multiple products and percentiles."""
    outputs: List[Dict[str, Path]] = []

    for product_name in products:
        for pct in percentiles:
            out = select_events_for_product_percentile(
                cfg=cfg,
                product_name=product_name,
                percentile=pct,
                start_year=start_year,
                end_year=end_year,
                station_limit=station_limit,
                overwrite=overwrite,
                write_yearly_files=write_yearly_files,
                verbose=verbose,
            )
            outputs.append(out)

    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    select_events_for_product_percentile(
        cfg,
        product_name="imerg_v07",
        percentile="p98",
        start_year=2001,
        end_year=2001,
        station_limit=10,
    )


if __name__ == "__main__":
    main()



# ============================================================
# Paper-consistent chronological declustering
# ============================================================
def decluster_dates_chronological_first(sorted_dates, min_gap_days=3):
    """
    Keep exceedance dates separated by at least min_gap_days using the
    chronological-first rule used in the paper-consistent bias-pair extraction.

    This intentionally keeps the first exceedance in a cluster, rather than
    searching for the maximum rainfall day within the cluster.
    """
    kept = []
    last = None
    for d in sorted_dates:
        if last is None or (d - last).days >= min_gap_days:
            kept.append(d)
            last = d
    return kept

