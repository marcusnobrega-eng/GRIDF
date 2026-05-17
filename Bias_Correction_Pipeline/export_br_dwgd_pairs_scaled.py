#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path

import ee
import numpy as np
import pandas as pd


GEE_PROJECT = "ee-marcusep2025"

PRODUCT = "br_dwgd"
PRODUCT_LABEL = "BR-DWGD / Xavier"
PERCENTILE_LABEL = "p98"
PERCENTILE_VALUE = 0.98

YEAR_START = 2001
YEAR_END = 2006

EVENT_FILE = Path(
    "data/products/br_dwgd/sensitivity/p98/events/events_br_dwgd_p98_all_years.csv"
)

DRIVE_FOLDER = "GRIDF_BiasCorrection_pairs_br_dwgd_p98_2001_2006_scaled"

MAX_FEATURES_PER_EXPORT = 1000

COLLECTION_ID = "projects/sat-io/open-datasets/BR-DWGD/PR"
BAND = "b1"

# Correct BR-DWGD precipitation conversion:
# precipitation [mm/day] = raw encoded value * scale + offset
BR_DWGD_SCALE = 0.006866665
BR_DWGD_OFFSET = 225.0

SAMPLE_SCALE_M = 11132.0
MIN_PRODUCT_RAINFALL_FOR_RATIO_MM = 1.0

EXPORT_SELECTORS = [
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


def clean_value(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return float(v)
    if pd.isna(v):
        return None
    return v


def dataframe_to_fc(df):
    features = []

    for _, row in df.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        props = {}
        for col in df.columns:
            props[col] = clean_value(row[col])

        features.append(
            ee.Feature(
                ee.Geometry.Point([lon, lat]),
                props
            )
        )

    return ee.FeatureCollection(features)


def make_br_dwgd_daily_image(date):
    date = ee.Date(date)
    next_date = date.advance(1, "day")

    ic = (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(date, next_date)
        .select(BAND)
    )

    count = ic.size()

    empty = (
        ee.Image.constant(0)
        .rename("precipitation")
        .updateMask(ee.Image.constant(0))
    )

    scaled = (
        ic.mean()
        .select(BAND)
        .multiply(BR_DWGD_SCALE)
        .add(BR_DWGD_OFFSET)
        .rename("precipitation")
    )

    daily = ee.Image(
        ee.Algorithms.If(
            count.gt(0),
            scaled,
            empty
        )
    ).set({
        "system:time_start": date.millis(),
        "date": date.format("YYYY-MM-dd"),
        "image_count": count,
        "system:index": date.format("YYYYMMdd"),
    })

    return daily


def sample_one_feature(feat):
    date = ee.String(feat.get("date"))
    gauge_mm = ee.Number(feat.get("gauge_mm"))

    daily = make_br_dwgd_daily_image(date)

    sampled = daily.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=feat.geometry(),
        scale=SAMPLE_SCALE_M,
        bestEffort=True,
        maxPixels=1e8,
    )

    raw_product = sampled.get("precipitation")
    is_null = ee.Algorithms.IsEqual(raw_product, None)

    product_numeric = ee.Number(
        ee.Algorithms.If(is_null, -9999.0, raw_product)
    )

    product_valid = product_numeric.gt(MIN_PRODUCT_RAINFALL_FOR_RATIO_MM)

    ratio = ee.Algorithms.If(
        product_valid,
        gauge_mm.divide(product_numeric),
        None
    )

    product_out = ee.Algorithms.If(is_null, None, product_numeric)

    return feat.set({
        "product": PRODUCT,
        "product_label": PRODUCT_LABEL,
        "percentile_label": PERCENTILE_LABEL,
        "percentile_value": PERCENTILE_VALUE,
        "gee_collection": COLLECTION_ID,
        "gee_band": BAND,
        "daily_aggregation": (
            "daily_total_direct_mm_day_scaled_offset:"
            f"precip_mm=raw*{BR_DWGD_SCALE}+{BR_DWGD_OFFSET}"
        ),
        "sample_scale_m": SAMPLE_SCALE_M,
        "daily_image_count": daily.get("image_count"),
        "product_mm": product_out,
        "product_valid_for_ratio": product_valid,
        "ratio_gauge_over_product": ratio,
    })


def submit_export(df_chunk, year, chunk_id):
    fc = dataframe_to_fc(df_chunk)
    sampled_fc = fc.map(sample_one_feature)

    desc = f"pairs_br_dwgd_p98_{year}_chunk{chunk_id:03d}"

    task = ee.batch.Export.table.toDrive(
        collection=sampled_fc,
        description=desc,
        folder=DRIVE_FOLDER,
        fileNamePrefix=desc,
        fileFormat="CSV",
        selectors=EXPORT_SELECTORS,
    )

    task.start()

    print(
        f"[{year}] submitted {desc}.csv | "
        f"features={len(df_chunk)} | task_id={task.id}"
    )


def main():
    ee.Initialize(project=GEE_PROJECT)

    if not EVENT_FILE.exists():
        raise FileNotFoundError(f"Event file not found: {EVENT_FILE}")

    events = pd.read_csv(EVENT_FILE)

    events["year"] = pd.to_numeric(events["year"], errors="coerce").astype("Int64")
    events["gauge_mm"] = pd.to_numeric(events["gauge_mm"], errors="coerce")
    events["latitude"] = pd.to_numeric(events["latitude"], errors="coerce")
    events["longitude"] = pd.to_numeric(events["longitude"], errors="coerce")
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    valid = (
        events["year"].notna()
        & events["date"].notna()
        & np.isfinite(events["gauge_mm"])
        & np.isfinite(events["latitude"])
        & np.isfinite(events["longitude"])
    )

    events = events.loc[valid].copy()
    events["year"] = events["year"].astype(int)

    events = events.loc[
        (events["year"] >= YEAR_START)
        & (events["year"] <= YEAR_END)
    ].copy()

    print("=" * 80)
    print("BR-DWGD scaled GEE pair export")
    print("=" * 80)
    print(f"Events:       {len(events)}")
    print(f"Years:        {YEAR_START}-{YEAR_END}")
    print(f"Drive folder: {DRIVE_FOLDER}")
    print(f"Formula:      precip_mm = raw * {BR_DWGD_SCALE} + {BR_DWGD_OFFSET}")
    print("=" * 80)

    for year in sorted(events["year"].unique()):
        df_y = events.loc[events["year"] == year].copy()

        for start in range(0, len(df_y), MAX_FEATURES_PER_EXPORT):
            chunk = df_y.iloc[start:start + MAX_FEATURES_PER_EXPORT].copy()
            chunk_id = start // MAX_FEATURES_PER_EXPORT + 1
            submit_export(chunk, int(year), int(chunk_id))

    print("\nAll BR-DWGD scaled exports submitted.")


if __name__ == "__main__":
    main()
