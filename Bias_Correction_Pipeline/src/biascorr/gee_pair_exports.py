#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gee_pair_exports.py

Google Earth Engine export layer for gauge/product bias-pair CSVs.

Part 04 scope
-------------
This module takes gauge-event tables created by Part 02 and submits GEE
Export.table.toDrive tasks that sample the selected rainfall product at each
station/date.

Scientific workflow
-------------------
For each selected gauge event:

    event = (station_id, lon, lat, date, gauge_mm)

the module builds the daily product rainfall image in mm/day using
gee_products.get_daily_precip_image(), samples it at the station point, and
exports:

    product_mm
    ratio_gauge_over_product = gauge_mm / product_mm

The ratio is only computed when product_mm is finite and exceeds the minimum
product rainfall threshold in config/method.yml. Invalid or tiny product values
are retained in the CSV with flags so they can be audited and filtered later in
Part 05.

Important storage note
----------------------
Earth Engine cannot export directly to a local Mac/GitHub folder. It exports
CSV files to Google Drive. This module therefore:

1. Creates GEE table export tasks to a Drive folder.
2. Writes a local manifest under metadata/gee_tasks/.
3. Writes a copy/sync instruction file in the local pairs folder.

After the GEE tasks finish and Google Drive syncs, the exported CSV files should
be copied or moved into:

    data/products/<product>/sensitivity/<pXX>/pairs/

Part 05 will read the pair CSVs from that local pairs folder.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .event_selection import parse_percentile_arg
from .gee_products import (
    brazil_geometry,
    choose_precip_band,
    collection_basic_info,
    get_daily_precip_image,
    initialize_earth_engine,
)
from .utils import ensure_dir, now_iso, print_header, print_section, timestamp, write_json, write_text


REQUIRED_EVENT_COLUMNS = [
    "station_id",
    "latitude",
    "longitude",
    "date",
    "year",
    "gauge_mm",
]


DEFAULT_EXPORT_SELECTORS = [
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
    "gee_collection",
    "gee_band",
    "daily_aggregation",
    "sample_scale_m",
    "daily_image_count",
    "product_mm",
    "product_valid_for_ratio",
    "ratio_gauge_over_product",
]


# =============================================================================
# PRODUCT-SPECIFIC EXPORT-SIDE DECODING
# =============================================================================

# BR-DWGD / Xavier precipitation is stored in Earth Engine as encoded values.
# The physical precipitation depth is:
#
#     pr_mm = raw_pr * 0.00686666 + 225.0
#
# Important: valid encoded BR-DWGD precipitation values are often negative
# before applying this scale/offset. Therefore, do NOT mask negative raw values
# before scaling. The physical rainfall validity test is applied after scaling.
BR_DWGD_PR_SCALE = 0.00686666
BR_DWGD_PR_OFFSET = 225.0
BR_DWGD_EXPORT_DAILY_AGGREGATION = "daily_total_direct_mm_day_scaled_offset"


def _events_dir(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "events"
    )


def _pairs_dir(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "pairs"
    )


def _task_manifest_dir(cfg: Any) -> Path:
    return Path(cfg.metadata_root) / "gee_tasks"


def _default_events_path(cfg: Any, product_name: str, percentile_label: str) -> Path:
    return (
        _events_dir(cfg, product_name, percentile_label)
        / f"events_{product_name}_{percentile_label}_all_years.csv"
    )


def _default_drive_folder(product_name: str, percentile_label: str) -> str:
    """
    Return a flat Google Drive folder name.

    Flat names are intentionally used because Earth Engine's Drive folder
    handling can be ambiguous for nested paths.
    """
    return f"GRIDF_BiasCorrection_pairs_{product_name}_{percentile_label}"


def _safe_export_name(text: str) -> str:
    """Make a safe Earth Engine export description/file prefix."""
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ["_", "-"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _validate_event_table(events: pd.DataFrame, path: Path) -> None:
    """Validate event table columns."""
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(
            f"Event table is missing required columns {missing}: {path}\n"
            f"Available columns: {list(events.columns)}"
        )


def load_events_for_export(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    events_path: Optional[Path] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load selected gauge events and apply client-side sanity filters before
    creating Earth Engine features.
    """
    if events_path is None:
        events_path = _default_events_path(cfg, product_name, percentile_label)

    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(
            f"Event file not found: {events_path}\n\n"
            "Run event selection first, for example:\n"
            f"    python3 run_pipeline.py select-events --product {product_name} --percentile {percentile_label}"
        )

    events = pd.read_csv(events_path)
    _validate_event_table(events, events_path)

    # Normalize important fields.
    events["year"] = pd.to_numeric(events["year"], errors="coerce").astype("Int64")
    events["gauge_mm"] = pd.to_numeric(events["gauge_mm"], errors="coerce")
    events["latitude"] = pd.to_numeric(events["latitude"], errors="coerce")
    events["longitude"] = pd.to_numeric(events["longitude"], errors="coerce")
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if start_year is not None:
        events = events.loc[events["year"] >= int(start_year)]
    if end_year is not None:
        events = events.loc[events["year"] <= int(end_year)]

    # Keep only rows that can become valid GEE point features.
    valid = (
        events["year"].notna()
        & events["date"].notna()
        & np.isfinite(events["gauge_mm"])
        & np.isfinite(events["latitude"])
        & np.isfinite(events["longitude"])
        & events["latitude"].between(-90, 90)
        & events["longitude"].between(-180, 180)
    )

    events = events.loc[valid].copy()
    events["year"] = events["year"].astype(int)

    return events


def chunk_dataframe(df: pd.DataFrame, max_rows: int) -> List[pd.DataFrame]:
    """Split a DataFrame into row chunks."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    return [df.iloc[i:i + max_rows].copy() for i in range(0, len(df), max_rows)]


def _clean_property_value(value: Any) -> Any:
    """
    Convert pandas/numpy values into Earth Engine-safe properties.
    """
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")

    if pd.isna(value):
        return None

    return value


def dataframe_to_feature_collection(
    ee: Any,
    df: pd.DataFrame,
    property_columns: Optional[Sequence[str]] = None,
) -> Any:
    """
    Convert a pandas event chunk to an ee.FeatureCollection.
    """
    if property_columns is None:
        property_columns = list(df.columns)

    features = []

    for _, row in df.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        props = {
            col: _clean_property_value(row[col])
            for col in property_columns
            if col in row.index
        }

        feature = ee.Feature(ee.Geometry.Point([lon, lat]), props)
        features.append(feature)

    return ee.FeatureCollection(features)


def _resolve_precip_band_for_export(
    ee: Any,
    product_cfg: Mapping[str, Any],
    start_date: str,
    end_date: str,
    geom: Any,
) -> Tuple[str, Dict[str, Any]]:
    """
    Inspect collection and select precipitation band for export.
    """
    info = collection_basic_info(
        ee=ee,
        collection_id=product_cfg["gee_collection"],
        start_date=start_date,
        end_date=end_date,
        geometry=geom,
    )

    band = choose_precip_band(
        info["band_names_first_image"],
        configured_band=product_cfg.get("gee_band"),
    )

    return band, info


def sample_event_feature_collection(
    ee: Any,
    feature_collection: Any,
    product_name: str,
    product_cfg: Mapping[str, Any],
    precip_band: str,
    sample_scale_m: float,
    min_product_rainfall_for_ratio_mm: float,
) -> Any:
    """
    Map over an event FeatureCollection and add product_mm and ratio fields.

    This function is server-side. It constructs the product daily image using
    the event date from each feature.
    """
    gee_collection = str(product_cfg["gee_collection"])
    is_br_dwgd = str(product_name).lower() == "br_dwgd"

    daily_aggregation = str(product_cfg.get("daily_aggregation", ""))
    if is_br_dwgd:
        daily_aggregation = BR_DWGD_EXPORT_DAILY_AGGREGATION

    def _sample_one(feat):
        date = ee.String(feat.get("date"))
        gauge_mm = ee.Number(feat.get("gauge_mm"))

        daily = get_daily_precip_image(
            ee=ee,
            product_name=product_name,
            product_cfg=product_cfg,
            date=date,
            precip_band=precip_band,
            geometry=None,
        )

        region = daily.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feat.geometry(),
            scale=sample_scale_m,
            bestEffort=True,
            maxPixels=1e8,
        )

        raw_product = region.get("precipitation")
        is_null = ee.Algorithms.IsEqual(raw_product, None)

        # Use -9999 internally only to allow numeric comparisons when GEE
        # returns null at the sampling point. For BR-DWGD, the -9999 sentinel
        # must NOT be scaled; nulls are handled explicitly below.
        raw_numeric = ee.Number(
            ee.Algorithms.If(is_null, -9999.0, raw_product)
        )

        if is_br_dwgd:
            # BR-DWGD / Xavier precipitation is encoded in the SAT-IO asset.
            # Convert to physical daily precipitation depth in mm before
            # exporting product_mm and before computing the gauge/product ratio.
            # Do not mask negative raw values before scaling; they are valid
            # encoded rainfall values.
            scaled_product = raw_numeric.multiply(BR_DWGD_PR_SCALE).add(BR_DWGD_PR_OFFSET)

            # Clamp only tiny numerical negatives after scaling.
            scaled_product = ee.Number(
                ee.Algorithms.If(
                    scaled_product.lt(0).And(scaled_product.gt(-0.2)),
                    0.0,
                    scaled_product,
                )
            )

            product_numeric = ee.Number(
                ee.Algorithms.If(is_null, -9999.0, scaled_product)
            )
        else:
            product_numeric = raw_numeric

        product_valid = product_numeric.gt(float(min_product_rainfall_for_ratio_mm))

        ratio = ee.Algorithms.If(
            product_valid,
            gauge_mm.divide(product_numeric),
            None,
        )

        product_out = ee.Algorithms.If(is_null, None, product_numeric)

        return feat.set({
            "gee_collection": gee_collection,
            "gee_band": precip_band,
            "daily_aggregation": daily_aggregation,
            "sample_scale_m": float(sample_scale_m),
            "daily_image_count": daily.get("image_count"),
            "product_mm": product_out,
            "product_valid_for_ratio": product_valid,
            "ratio_gauge_over_product": ratio,
        })

    return feature_collection.map(_sample_one)


def submit_table_export_to_drive(
    ee: Any,
    collection: Any,
    description: str,
    drive_folder: str,
    file_name_prefix: str,
    selectors: Sequence[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Submit or describe an Earth Engine table export task.
    """
    description = _safe_export_name(description)
    file_name_prefix = _safe_export_name(file_name_prefix)

    record: Dict[str, Any] = {
        "description": description,
        "drive_folder": drive_folder,
        "file_name_prefix": file_name_prefix,
        "file_format": "CSV",
        "selectors": list(selectors),
        "dry_run": bool(dry_run),
        "task_id": None,
        "submitted": False,
    }

    if dry_run:
        return record

    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=drive_folder,
        fileNamePrefix=file_name_prefix,
        fileFormat="CSV",
        selectors=list(selectors),
    )

    task.start()

    record.update({
        "task_id": task.id,
        "submitted": True,
    })

    return record


def write_pair_sync_instructions(
    pairs_dir: Path,
    drive_folder: str,
    product_name: str,
    percentile_label: str,
) -> Path:
    """
    Write a local text file explaining where the GEE CSVs are expected.
    """
    text = f"""
GEE pair-export instructions
============================

Product:
  {product_name}

Percentile:
  {percentile_label}

Earth Engine exports the bias-pair CSVs to this Google Drive folder:

  {drive_folder}

After all GEE tasks finish and Google Drive syncs, copy or move the exported
CSV files into this local pipeline folder:

  {pairs_dir}

Part 05 will read pair CSVs from this local folder.

Expected file pattern:

  pairs_{product_name}_{percentile_label}_YYYY_chunkNNN.csv

Notes:
- Earth Engine cannot export directly to the local GitHub folder.
- The export manifest in metadata/gee_tasks records all task IDs and file names.
""".strip() + "\n"

    out = pairs_dir / f"README_GEE_EXPORTS_{product_name}_{percentile_label}.txt"
    write_text(out, text)
    return out


def export_pairs_for_product_percentile(
    cfg: Any,
    product_name: str,
    percentile: str | float,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    events_path: Optional[Path] = None,
    gee_project: Optional[str] = "ee-marcusep2025",
    drive_folder: Optional[str] = None,
    max_features_per_export: int = 3000,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Submit GEE exports for one product and one percentile.

    Exports are split by year and then by chunk if needed.
    """
    percentile_label, percentile_value = parse_percentile_arg(percentile)

    product_cfg = cfg.product(product_name)
    product_label = product_cfg.get("label", product_name)

    pairs_dir = _pairs_dir(cfg, product_name, percentile_label)
    ensure_dir(pairs_dir)

    if drive_folder is None:
        drive_folder = _default_drive_folder(product_name, percentile_label)

    events = load_events_for_export(
        cfg=cfg,
        product_name=product_name,
        percentile_label=percentile_label,
        events_path=events_path,
        start_year=start_year,
        end_year=end_year,
    )

    if events.empty:
        raise ValueError(
            f"No events available for export: product={product_name}, "
            f"percentile={percentile_label}, start_year={start_year}, end_year={end_year}"
        )

    years = sorted(events["year"].dropna().astype(int).unique().tolist())

    if verbose:
        print_header(f"GEE pair export: {product_name} / {percentile_label}")
        print(f"Product label:        {product_label}")
        print(f"Events loaded:        {len(events)}")
        print(f"Years:                {years[0]}–{years[-1]} ({len(years)} years)")
        print(f"Drive folder:         {drive_folder}")
        print(f"Local pairs folder:   {pairs_dir}")
        print(f"Max features/export:  {max_features_per_export}")
        print(f"Dry run:              {dry_run}")

    ee = initialize_earth_engine(project=gee_project)
    geom = brazil_geometry(ee)

    # Resolve band using the filtered configured period, not only sample date.
    first_year = int(min(years))
    last_year = int(max(years))
    band_start = f"{first_year}-01-01"
    band_end = f"{last_year + 1}-01-01"

    precip_band, collection_info = _resolve_precip_band_for_export(
        ee=ee,
        product_cfg=product_cfg,
        start_date=band_start,
        end_date=band_end,
        geom=geom,
    )

    native_res = product_cfg.get("native_resolution_deg", None)
    if native_res is not None:
        sample_scale_m = float(native_res) * 111_320.0
    else:
        # Conservative default. The inspection step should normally provide
        # more product-specific confidence.
        sample_scale_m = 10_000.0

    min_product = float(cfg.method["ratio_qc"]["min_product_rainfall_for_ratio_mm"])

    # Only pass useful event fields to GEE.
    property_columns = [
        col for col in DEFAULT_EXPORT_SELECTORS
        if col in events.columns
    ]

    # Ensure mandatory columns are included even if not listed above.
    for col in REQUIRED_EVENT_COLUMNS:
        if col not in property_columns:
            property_columns.append(col)

    task_records: List[Dict[str, Any]] = []
    export_count = 0

    for year in years:
        events_y = events.loc[events["year"] == int(year)].copy()
        chunks = chunk_dataframe(events_y, max_features_per_export)

        if verbose:
            print_section(f"Year {year}")
            print(f"Events this year: {len(events_y)}")
            print(f"Chunks:           {len(chunks)}")

        for chunk_id, chunk in enumerate(chunks, start=1):
            export_count += 1

            fc = dataframe_to_feature_collection(
                ee=ee,
                df=chunk,
                property_columns=property_columns,
            )

            sampled_fc = sample_event_feature_collection(
                ee=ee,
                feature_collection=fc,
                product_name=product_name,
                product_cfg=product_cfg,
                precip_band=precip_band,
                sample_scale_m=sample_scale_m,
                min_product_rainfall_for_ratio_mm=min_product,
            )

            chunk_label = f"chunk{chunk_id:03d}"
            description = f"pairs_{product_name}_{percentile_label}_{year}_{chunk_label}"
            fname = description

            record = submit_table_export_to_drive(
                ee=ee,
                collection=sampled_fc,
                description=description,
                drive_folder=drive_folder,
                file_name_prefix=fname,
                selectors=DEFAULT_EXPORT_SELECTORS,
                dry_run=dry_run,
            )

            record.update({
                "product": product_name,
                "product_label": product_label,
                "percentile_label": percentile_label,
                "percentile_value": percentile_value,
                "year": int(year),
                "chunk_id": int(chunk_id),
                "n_features": int(len(chunk)),
                "local_expected_folder": str(pairs_dir),
                "expected_csv_name": f"{fname}.csv",
                "gee_project": gee_project,
            })

            task_records.append(record)

            if verbose:
                if dry_run:
                    print(f"  [dry-run] {fname}.csv ({len(chunk)} features)")
                else:
                    print(f"  Submitted {fname}.csv | task_id={record['task_id']} | features={len(chunk)}")

    instructions_path = write_pair_sync_instructions(
        pairs_dir=pairs_dir,
        drive_folder=drive_folder,
        product_name=product_name,
        percentile_label=percentile_label,
    )

    manifest = {
        "created_at": now_iso(),
        "product": product_name,
        "product_label": product_label,
        "percentile_label": percentile_label,
        "percentile_value": percentile_value,
        "years": years,
        "n_events_total": int(len(events)),
        "n_exports": int(export_count),
        "dry_run": bool(dry_run),
        "gee_project": gee_project,
        "drive_folder": drive_folder,
        "local_expected_pairs_folder": str(pairs_dir),
        "sync_instructions": str(instructions_path),
        "max_features_per_export": int(max_features_per_export),
        "precip_band": precip_band,
        "sample_scale_m": float(sample_scale_m),
        "min_product_rainfall_for_ratio_mm": min_product,
        "collection_info": collection_info,
        "daily_aggregation": (
            BR_DWGD_EXPORT_DAILY_AGGREGATION
            if str(product_name).lower() == "br_dwgd"
            else product_cfg.get("daily_aggregation")
        ),
        "tasks": task_records,
    }

    manifest_dir = _task_manifest_dir(cfg)
    ensure_dir(manifest_dir)

    manifest_path = (
        manifest_dir
        / f"gee_pair_export_manifest_{product_name}_{percentile_label}_{timestamp()}.json"
    )
    write_json(manifest_path, manifest)

    # Also save latest manifest in local pairs folder for easy access.
    latest_manifest_path = pairs_dir / f"gee_pair_export_manifest_{product_name}_{percentile_label}_latest.json"
    write_json(latest_manifest_path, manifest)

    if verbose:
        print_section("GEE pair-export manifest")
        print(manifest_path)
        print(latest_manifest_path)
        print("\nAfter tasks complete, copy/sync CSVs from Drive folder:")
        print(f"  {drive_folder}")
        print("to local folder:")
        print(f"  {pairs_dir}")

    return {
        "manifest": manifest_path,
        "latest_manifest": latest_manifest_path,
        "pairs_dir": pairs_dir,
        "drive_folder": drive_folder,
        "tasks": task_records,
    }


def export_pairs_batch(
    cfg: Any,
    products: Sequence[str],
    percentiles: Sequence[str | float],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    gee_project: Optional[str] = "ee-marcusep2025",
    drive_folder: Optional[str] = None,
    drive_folder_prefix: Optional[str] = None,
    max_features_per_export: int = 3000,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Export pairs for multiple products and percentiles.

    If drive_folder is provided for a batch with more than one product or
    percentile, all exports go to the same folder. Otherwise a product-specific
    default or prefix-derived folder is used.
    """
    outputs: List[Dict[str, Any]] = []

    for product_name in products:
        for percentile in percentiles:
            p_label, _ = parse_percentile_arg(percentile)

            if drive_folder is not None:
                folder = drive_folder
            elif drive_folder_prefix is not None:
                folder = f"{drive_folder_prefix}_{product_name}_{p_label}"
            else:
                folder = None

            out = export_pairs_for_product_percentile(
                cfg=cfg,
                product_name=product_name,
                percentile=p_label,
                start_year=start_year,
                end_year=end_year,
                gee_project=gee_project,
                drive_folder=folder,
                max_features_per_export=max_features_per_export,
                dry_run=dry_run,
                verbose=verbose,
            )
            outputs.append(out)

    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    export_pairs_for_product_percentile(
        cfg=cfg,
        product_name="imerg_v07",
        percentile="p98",
        start_year=2001,
        end_year=2001,
        max_features_per_export=100,
        dry_run=True,
    )


if __name__ == "__main__":
    main()
