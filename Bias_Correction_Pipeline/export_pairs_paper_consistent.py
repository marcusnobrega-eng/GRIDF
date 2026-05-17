#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm


PRODUCTS = {
    "br_dwgd": {
        "collection": "projects/sat-io/open-datasets/BR-DWGD/PR",
        "band": None,
        "product_col": "xavier_pr_mm",
        "scale_m": 11132,
        "gee_scale": 0.006866665,
        "gee_offset": 225.0,
        "description": "BR_DWGD_BiasPairs",
    },
    "chirps": {
        "collection": "UCSB-CHG/CHIRPS/DAILY",
        "band": "precipitation",
        "product_col": "chirps_mm",
        "scale_m": 5560,
        "description": "CHIRPS_BiasPairs",
    },
    "persiann_cdr": {
        "collection": "NOAA/PERSIANN-CDR",
        "band": "precipitation",
        "product_col": "persiann_mm",
        "scale_m": 27830,
        "description": "PERSIANN_BiasPairs",
    },
    # This is your legacy IMERG product. We keep it available,
    # but you said we will use the existing legacy Drive folder for this.
    "imerg_v06": {
        "collection": "projects/climate-engine-pro/assets/ce-gpm-imerg-daily",
        "band": "precipitationCal",
        "product_col": "imerg_mm",
        "scale_m": 11132,
        "description": "IMERG_BiasPairs",
    },
    # New product added in the same logic.
    "imerg_v07": {
        "collection": "projects/climate-engine-pro/assets/ce-gpm-imerg-v07/early-daily",
        "band": "precipitation",
        "product_col": "imerg_mm",
        "scale_m": 11132,
        "description": "IMERG_V07_BiasPairs",
    },
}


def parse_date_header(col_name: str):
    try:
        return datetime.strptime(str(col_name), "%m/%d/%Y")
    except Exception:
        return None


def identify_columns(path: Path):
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    date_cols = []
    for c in cols:
        d = parse_date_header(c)
        if d is not None:
            date_cols.append(c)

    if not date_cols:
        raise ValueError("No daily date columns found. Expected headers like 1/1/1994.")

    dates = sorted(parse_date_header(c) for c in date_cols)
    return date_cols, dates[0].year, dates[-1].year


def year_to_cols(date_cols, year: int):
    return [c for c in date_cols if parse_date_header(c) and parse_date_header(c).year == year]


def decluster_dates(sorted_dates, min_gap_days=3):
    """
    Paper-consistent chronological declustering.
    Keeps the first exceedance in a cluster, not the largest value in the cluster.
    """
    kept = []
    last = None
    for d in sorted_dates:
        if last is None or (d - last).days >= min_gap_days:
            kept.append(d)
            last = d
    return kept


def station_to_feature(row, station_col, lat_col, lon_col):
    geom = ee.Geometry.Point([float(row[lon_col]), float(row[lat_col])])
    return ee.Feature(
        geom,
        {
            "station_id": str(row[station_col]),
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True, choices=sorted(PRODUCTS))
    parser.add_argument("--csv-path", default="/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction/rainfall_timeseries_with_metadata_all.csv")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--drive-folder", required=True)
    parser.add_argument("--gee-project", default="ee-marcusep2025")
    parser.add_argument("--percentile", type=float, default=0.98)
    parser.add_argument("--percentile-label", default="p98")
    parser.add_argument("--min-gap-days", type=int, default=3)
    parser.add_argument("--min-per-year", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--station-id-col", default="Code")
    parser.add_argument("--lat-col", default="Latitude")
    parser.add_argument("--lon-col", default="Longitude")
    args = parser.parse_args()

    product_key = args.product
    spec = PRODUCTS[product_key]
    csv_path = Path(args.csv_path)

    if args.gee_project:
        ee.Initialize(project=args.gee_project)
    else:
        ee.Initialize()

    ic = ee.ImageCollection(spec["collection"])

    date_cols, file_y0, file_y1 = identify_columns(csv_path)
    y0 = max(args.start_year, file_y0)
    y1 = min(args.end_year, file_y1)

    if y0 > y1:
        raise ValueError(f"No overlap between requested years and CSV years: {file_y0}-{file_y1}")

    years = list(range(y0, y1 + 1))

    print("=" * 90)
    print("Paper-consistent GEE pair export")
    print("=" * 90)
    print("Product:      ", product_key)
    print("Collection:   ", spec["collection"])
    print("Band:         ", spec.get("band"))
    print("Product col:  ", spec["product_col"])
    print("CSV:          ", csv_path)
    print("CSV years:    ", f"{file_y0}-{file_y1}")
    print("Export years: ", f"{years[0]}-{years[-1]}")
    print("Drive folder: ", args.drive_folder)
    print("=" * 90)

    stations_meta = pd.read_csv(
        csv_path,
        usecols=[args.station_id_col, args.lat_col, args.lon_col],
        low_memory=False,
    )

    stations_meta = stations_meta.dropna(
        subset=[args.station_id_col, args.lat_col, args.lon_col]
    ).copy()

    stations_meta[args.station_id_col] = stations_meta[args.station_id_col].astype(str)
    stations_meta = stations_meta.drop_duplicates(subset=[args.station_id_col])

    fc_all = ee.FeatureCollection(
        [
            station_to_feature(row, args.station_id_col, args.lat_col, args.lon_col)
            for _, row in stations_meta.iterrows()
        ]
    )

    for year in years:
        cols_y = year_to_cols(date_cols, year)

        if not cols_y:
            print(f"[{year}] No date columns found; skipping.")
            continue

        print(f"\n[{year}] Reading stations in chunks. Daily cols = {len(cols_y)}")

        per_date = {}
        dates_y = [parse_date_header(c) for c in cols_y]

        usecols = [args.station_id_col, args.lat_col, args.lon_col] + cols_y
        reader = pd.read_csv(csv_path, usecols=usecols, chunksize=args.chunk_size, low_memory=False)

        for chunk in tqdm(reader, desc=f"Parse extremes {year}"):
            chunk[args.station_id_col] = chunk[args.station_id_col].astype(str)

            for c in cols_y:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            for _, row in chunk.iterrows():
                sid = str(row[args.station_id_col])
                vals = row[cols_y].to_numpy(dtype=float)

                # Same gauge QC as the legacy scripts.
                vals = np.where((vals >= 0) & (vals <= 500), vals, np.nan)

                if np.count_nonzero(~np.isnan(vals)) < 30:
                    continue

                thr = np.nanpercentile(vals, args.percentile * 100.0)
                idx = np.where(vals > thr)[0]

                if idx.size == 0:
                    continue

                exc_dates = [dates_y[i] for i in idx]
                exc_vals = [vals[i] for i in idx]

                exc_df = pd.DataFrame({"date": exc_dates, "pr_g": exc_vals}).sort_values("date")

                kept_dates = decluster_dates(
                    list(exc_df["date"]),
                    min_gap_days=args.min_gap_days,
                )

                if len(kept_dates) < args.min_per_year:
                    continue

                pr_lookup = dict(zip(exc_df["date"], exc_df["pr_g"]))

                for d in kept_dates:
                    prg = float(pr_lookup.get(d, np.nan))
                    if not np.isfinite(prg) or prg <= 1.0:
                        continue

                    dstr = d.strftime("%Y-%m-%d")
                    entry = per_date.get(dstr)

                    if entry is None:
                        per_date[dstr] = {"sids": [sid], "prg": {sid: prg}}
                    else:
                        entry["sids"].append(sid)
                        entry["prg"][sid] = prg

        if not per_date:
            print(f"[{year}] No extreme-day records formed; skipping export.")
            continue

        print(f"[{year}] Unique event dates: {len(per_date)}")
        print(f"[{year}] Preparing GEE sampling.")

        fc_year = ee.FeatureCollection([])

        for dstr, entry in tqdm(per_date.items(), desc=f"EE sampling {year}"):
            sids = entry["sids"]
            prg_dict = entry["prg"]

            fc_subset = fc_all.filter(ee.Filter.inList("station_id", ee.List(sids)))

            d0 = ee.Date(dstr)
            img0 = ic.filterDate(d0, d0.advance(1, "day")).first()

            product_col = spec["product_col"]

            if product_key == "br_dwgd":
                img = ee.Image(
                    ee.Algorithms.If(
                        img0,
                        ee.Image(img0)
                        .select(0)
                        .multiply(spec["gee_scale"])
                        .add(spec["gee_offset"])
                        .toFloat()
                        .rename(product_col),
                        ee.Image.constant(0).rename(product_col).toFloat(),
                    )
                )
            else:
                band = spec["band"]
                img = ee.Image(
                    ee.Algorithms.If(
                        img0,
                        ee.Image(img0).select(band).rename(product_col).toFloat(),
                        ee.Image.constant(0).rename(product_col).toFloat(),
                    )
                )

            samples = img.sampleRegions(
                collection=fc_subset,
                properties=["station_id", "lat", "lon"],
                scale=spec["scale_m"],
            )

            prg = ee.Dictionary(prg_dict)

            def with_props(feat):
                sid = ee.String(feat.get("station_id"))
                prc = ee.Number(feat.get(product_col))
                prg_val = ee.Number(prg.get(sid))
                ratio = prg_val.divide(prc)

                return feat.set(
                    {
                        "date": dstr,
                        "year": year,
                        product_col: prc,
                        "product_mm": prc,
                        "pr_g": prg_val,
                        "gauge_mm": prg_val,
                        "ratio": ratio,
                        "ratio_gauge_over_product": ratio,
                    }
                )

            # Critical: same export filter as legacy scripts.
            samples2 = (
                ee.FeatureCollection(samples.map(with_props))
                .filter(ee.Filter.gt(product_col, 1))
                .filter(ee.Filter.gt("pr_g", 1))
                .filter(ee.Filter.notNull(["lat", "lon", "ratio"]))
            )

            fc_year = ee.FeatureCollection(
                ee.Algorithms.If(samples2.size().gt(0), fc_year.merge(samples2), fc_year)
            )

        prefix = f"pairs_{product_key}_{args.percentile_label}_{year}_chunk001"

        selectors = [
            "station_id",
            "lat",
            "lon",
            "date",
            "year",
            product_col,
            "product_mm",
            "pr_g",
            "gauge_mm",
            "ratio",
            "ratio_gauge_over_product",
        ]

        task = ee.batch.Export.table.toDrive(
            collection=fc_year,
            description=f"{spec['description']}_{year}",
            folder=args.drive_folder,
            fileNamePrefix=prefix,
            fileFormat="CSV",
            selectors=selectors,
        )

        task.start()
        print(f"[{year}] Export started. Task ID: {task.id}")
        print(f"[{year}] File prefix: {prefix}")

    print("\nAll yearly exports submitted.")


if __name__ == "__main__":
    main()
