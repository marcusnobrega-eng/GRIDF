# 01_extract_extreme_pairs_ANA_XAVIER_GEE.py
# ---------------------------------------------------------
# Purpose:
#   From a wide ANA CSV (one row per station, daily columns like 1/1/1994),
#   find per-year >P95 gauge extremes with declustering, sample BR-DWGD
#   (Xavier) precipitation at those dates/points in Google Earth Engine,
#   and export one CSV per year to Drive:
#     station_id, date, xavier_pr_mm, pr_g, ratio
#
# Requirements:
#   pip install earthengine-api pandas numpy tqdm
#   earthengine authenticate   (run once in your environment)
#
# Notes:
#   - Submits one GEE export task per processed year.
#   - BR-DWGD precipitation ImageCollection (0.1° for Brazil):
#       'projects/sat-io/open-datasets/BR-DWGD/PR'
#     Each image has a single raw band; apply:
#       pr_mm = raw * 0.006866665 + 225.0
#     (Scale/offset documented in the community catalog.)
# ---------------------------------------------------------

import os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import ee

# ----------------- USER INPUTS -----------------
CSV_PATH = 'C:/Users/marcu/PycharmProjects/hydrobr_project/Testing/rainfall_timeseries_with_metadata_all.csv'
STATION_ID_COL = "Code"
LAT_COL = "Latitude"
LON_COL = "Longitude"

# Extreme selection settings
PCT = 0.98           # percentile threshold for extremes (e.g., 0.99 = P99)
MIN_GAP_DAYS = 3     # decluster gap (days) between kept extreme dates
MIN_PER_YEAR = 1     # require at least this many declustered dates per station-year

# Year range you want to process (clip to what's in the CSV automatically)
YEAR_START = 1994
YEAR_END   = 2020  # BR-DWGD PR extends to 2022; adjust if desired

# GEE / BR-DWGD sampling
GEE_PROJECT = 'ee-marcusep2025'
OUT_GDRIVE_FOLDER = "xavier_bias_pairs"
SAMPLE_SCALE_M = 11132  # ~0.1° native BR-DWGD resolution (~11.1 km at equator)

# Reading chunks
CHUNK_SIZE = 500

# Optional gauge sanity filter (keeps non-negative <= 500 mm/day; set to None to skip)
GAUGE_MAX_MM = 500.0

# BR-DWGD (Xavier) PR scale/offset (raw -> physical mm/day)
# Source: BR-DWGD community catalog table (Offset=225, Scale=0.006866665)
XAVIER_PR_SCALE = 0.006866665
XAVIER_PR_OFFSET = 225.0
# ---------------------------------------------------------


# --------------------- Helpers ---------------------------
def parse_date_header(col_name: str):
    """Parse a column header like '1/1/1994' to a datetime (or None)."""
    try:
        return datetime.strptime(col_name, "%m/%d/%Y")
    except Exception:
        return None


def identify_columns(path):
    """Return (meta_cols, date_cols, first_year_in_file, last_year_in_file)."""
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()
    date_cols = []
    for c in cols:
        d = parse_date_header(c)
        if d is not None:
            date_cols.append(c)
    if not date_cols:
        raise ValueError("No daily date columns found (expected headers like '1/1/1994').")
    dates = sorted([parse_date_header(c) for c in date_cols])
    meta_cols = [c for c in cols if c not in date_cols]
    return meta_cols, date_cols, dates[0].year, dates[-1].year


def year_to_cols(date_cols, y):
    """All column headers that belong to calendar year y."""
    out = []
    for c in date_cols:
        d = parse_date_header(c)
        if d and d.year == y:
            out.append(c)
    return out


def decluster_dates(sorted_dates, min_gap_days=3):
    """Greedy declustering of sorted datetime list by gap (days)."""
    kept = []
    last = None
    for d in sorted_dates:
        if last is None or (d - last).days >= min_gap_days:
            kept.append(d)
            last = d
    return kept


def main():
    # ---------- Init GEE ----------
    if GEE_PROJECT:
        ee.Initialize(project=GEE_PROJECT)
    else:
        ee.Initialize()

    # ====== BR-DWGD precipitation (raw) ======
    # Docs/asset ref: projects/sat-io/open-datasets/BR-DWGD/PR
    PR = ee.ImageCollection('projects/sat-io/open-datasets/BR-DWGD/PR')

    # ---------- Discover schema / years ----------
    meta_cols, all_date_cols, file_y0, file_y1 = identify_columns(CSV_PATH)

    # Clip requested window to what's present in the CSV
    y0 = max(YEAR_START, file_y0)
    y1 = min(YEAR_END, file_y1)
    if y0 > y1:
        raise ValueError(f"No overlapping years between file [{file_y0}–{file_y1}] and requested "
                         f"[{YEAR_START}–{YEAR_END}].")
    years = list(range(y0, y1 + 1))
    print(f"Detected years in file: {file_y0}–{file_y1}. Processing: {years[0]}–{years[-1]}.")

    # ---------- Build station master FeatureCollection ----------
    stations_meta = pd.read_csv(CSV_PATH, usecols=[STATION_ID_COL, LAT_COL, LON_COL])
    stations_meta = stations_meta.dropna(subset=[STATION_ID_COL, LAT_COL, LON_COL]).copy()
    stations_meta[STATION_ID_COL] = stations_meta[STATION_ID_COL].astype(str)
    stations_meta = stations_meta.drop_duplicates(subset=[STATION_ID_COL])

    def to_feat(row):
        geom = ee.Geometry.Point([float(row[LON_COL]), float(row[LAT_COL])])
        return ee.Feature(geom, {
            'station_id': str(row[STATION_ID_COL]),
            'lat': float(row[LAT_COL]),
            'lon': float(row[LON_COL])
        })

    fc_all = ee.FeatureCollection([to_feat(r) for _, r in stations_meta.iterrows()])

    # ---------- Main yearly loop ----------
    for y in years:
        cols_y = year_to_cols(all_date_cols, y)
        if not cols_y:
            print(f"[{y}] No date columns found; skipping.")
            continue

        print(f"\n[{y}] Reading stations in chunks (daily cols={len(cols_y)}) …")

        # date_str -> {'sids': [...], 'prg': {sid: pr_g_mm}}
        per_date = {}

        usecols = [STATION_ID_COL, LAT_COL, LON_COL] + cols_y
        reader = pd.read_csv(CSV_PATH, usecols=usecols, chunksize=CHUNK_SIZE)

        for chunk in tqdm(reader, desc=f"Parse extremes {y}"):
            chunk[STATION_ID_COL] = chunk[STATION_ID_COL].astype(str)
            # To numeric (coerce errors to NaN)
            for c in cols_y:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

            for _, row in chunk.iterrows():
                sid = row[STATION_ID_COL]
                vals = row[cols_y].to_numpy(dtype=float)

                # Basic gauge QC: keep non-negative and optionally cap unrealistic daily maxima
                if GAUGE_MAX_MM is not None:
                    vals = np.where((vals >= 0) & (vals <= GAUGE_MAX_MM), vals, np.nan)
                else:
                    vals = np.where(vals >= 0, vals, np.nan)

                # Need enough valid points to compute a sensible percentile
                if np.count_nonzero(~np.isnan(vals)) < 30:
                    continue

                thr = np.nanpercentile(vals, PCT * 100.0)
                idx = np.where(vals > thr)[0]
                if idx.size == 0:
                    continue

                dates_y = [parse_date_header(c) for c in cols_y]
                exc_dates = [dates_y[i] for i in idx]
                exc_vals  = [vals[i] for i in idx]

                exc_df = pd.DataFrame({'date': exc_dates, 'pr_g': exc_vals}).sort_values('date')
                kept_dates = decluster_dates(list(exc_df['date']), MIN_GAP_DAYS)
                if len(kept_dates) < MIN_PER_YEAR:
                    continue

                pr_lookup = dict(zip(exc_df['date'], exc_df['pr_g']))
                for d in kept_dates:
                    prg = float(pr_lookup.get(d, np.nan))
                    if not np.isfinite(prg) or prg <= 1.0:
                        continue
                    dstr = d.strftime("%Y-%m-%d")
                    entry = per_date.get(dstr)
                    if entry is None:
                        per_date[dstr] = {'sids': [sid], 'prg': {sid: prg}}
                    else:
                        entry['sids'].append(sid)
                        entry['prg'][sid] = prg

        if not per_date:
            print(f"[{y}] No extreme-day records formed; skipping export.")
            continue

        print(f"[{y}] {len(per_date)} extreme dates; preparing Earth Engine sampling…")

        fc_year = ee.FeatureCollection([])

        for dstr, entry in tqdm(per_date.items(), desc=f"EE sampling {y}"):
            sids = entry['sids']
            prg_dict = entry['prg']

            fc_subset = fc_all.filter(ee.Filter.inList('station_id', ee.List(sids)))

            # BR-DWGD PR image for that date (raw -> scaled mm/day)
            d0 = ee.Date(dstr)
            img0 = PR.filterDate(d0, d0.advance(1, 'day')).first()

            # Select first band (collection images are single-band), apply scale & offset, rename
            img_scaled = ee.Image(
                ee.Algorithms.If(
                    img0,
                    ee.Image(img0).select(0)  # first/only band
                        .multiply(XAVIER_PR_SCALE).add(XAVIER_PR_OFFSET)
                        .toFloat().rename('xavier_pr_mm'),
                    ee.Image.constant(0).rename('xavier_pr_mm').toFloat()
                )
            )

            # Sample at stations (native ~0.1°)
            samples = ee.Image(img_scaled).sampleRegions(
                collection=fc_subset,
                properties=['station_id', 'lat', 'lon'],
                scale=SAMPLE_SCALE_M
            )

            prg = ee.Dictionary(prg_dict)

            def with_props(feat):
                sid = ee.String(feat.get('station_id'))
                prc = ee.Number(feat.get('xavier_pr_mm'))
                prg_val = ee.Number(prg.get(sid))
                ratio = prg_val.divide(prc)
                return feat.set({
                    'date': dstr,
                    'xavier_pr_mm': prc,
                    'pr_g': prg_val,
                    'ratio': ratio
                })

            samples2 = (
                ee.FeatureCollection(samples.map(with_props))
                .filter(ee.Filter.gt('xavier_pr_mm', 1))
                .filter(ee.Filter.gt('pr_g', 1))
                .filter(ee.Filter.notNull(['lat', 'lon', 'ratio']))
            )

            fc_year = ee.FeatureCollection(
                ee.Algorithms.If(samples2.size().gt(0), fc_year.merge(samples2), fc_year)
            )

        # Export yearly CSV to Drive
        task = ee.batch.Export.table.toDrive(
            collection=fc_year,
            description=f"XAVIER_BiasPairs_{y}",
            folder=OUT_GDRIVE_FOLDER,
            fileNamePrefix=f"xavier_bias_pairs_{y}",
            fileFormat='CSV',
            selectors=['station_id', 'lat', 'lon', 'date', 'xavier_pr_mm', 'pr_g', 'ratio']
        )
        task.start()
        print(f"[{y}] Export started. Task ID: {task.id}")

    print("\nAll yearly exports submitted. Monitor progress in the GEE Tasks panel.")


if __name__ == "__main__":
    main()
