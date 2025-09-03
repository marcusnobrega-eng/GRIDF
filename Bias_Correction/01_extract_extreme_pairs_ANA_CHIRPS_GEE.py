# 01_extract_extreme_pairs_ANA_CHIRPS_GEE.py
# ---------------------------------------------------------
# Purpose:
#   From a wide ANA CSV (one row per station, daily columns like 1/1/1994),
#   find per-year >P95 gauge extremes with declustering, sample CHIRPS at those
#   dates/points in Google Earth Engine, and export one CSV per year to Drive:
#     station_id, date, chirps_mm, pr_g, ratio
#
# Requirements:
#   pip install earthengine-api pandas numpy tqdm
#   earthengine authenticate   (run once in your environment)
#
# Notes:
#   - This script submits one GEE export task per year.
#   - It avoids loading the full CSV into memory by chunking rows (stations).
#   - Ratios guard against tiny values (both pr_g and CHIRPS must be > 1 mm).
# ---------------------------------------------------------

import os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import ee

# ----------------- USER INPUTS -----------------
CSV_PATH = 'C:/Users/marcu/PycharmProjects/hydrobr_project/Testing/rainfall_timeseries_with_metadata_all.csv'  # path to your wide ANA CSV
STATION_ID_COL = "Code"         # station identifier in your file
LAT_COL        = "Latitude"
LON_COL        = "Longitude"

PCT = 0.98          # extreme threshold (paper uses 0.99)
MIN_GAP_DAYS = 3    # decluster: keep peaks separated by >= 3 days
MIN_PER_YEAR = 1    # require at least this many declustered extremes in a year to include that year/station

# Year range (discover from file if None)
YEAR_START = 1994
YEAR_END   = 2020

# GEE / CHIRPS sampling
GEE_PROJECT = 'ee-marcusep2025'  # set to your project id string if applicable, else None
OUT_GDRIVE_FOLDER = "chirps_bias_pairs"  # Google Drive folder for exports
SAMPLE_SCALE_M = 5560  # ~0.05 degree (CHIRPS native ~5.56 km)

# Reading chunks (to keep RAM usage low)
CHUNK_SIZE = 500
# ---------------------------------------------------------


# ---------- Helpers ----------
def parse_date_header(col_name: str):
    """Return datetime if header is 'mm/dd/YYYY', else None."""
    try:
        return datetime.strptime(col_name, "%m/%d/%Y")
    except Exception:
        return None

def identify_columns(path):
    """Identify metadata vs date columns; return (meta_cols, date_cols, first_year, last_year)."""
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()
    date_cols = []
    for c in cols:
        d = parse_date_header(c)
        if d is not None:
            date_cols.append(c)
    meta_cols = [c for c in cols if c not in date_cols]
    if not date_cols:
        raise ValueError("No daily date columns found (expected headers like '1/1/1994').")
    dates = sorted([parse_date_header(c) for c in date_cols])
    return meta_cols, date_cols, dates[0].year, dates[-1].year

def year_to_cols(date_cols, y):
    """Filter the date column names belonging to year y."""
    out = []
    for c in date_cols:
        d = parse_date_header(c)
        if d and d.year == y:
            out.append(c)
    return out

def decluster_dates(sorted_dates, min_gap_days=3):
    """Keep dates separated by at least min_gap_days (sorted list of datetimes)."""
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

    CHIRPS = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

    # ---------- Discover schema / years ----------
    meta_cols, all_date_cols, file_y0, file_y1 = identify_columns(CSV_PATH)
    y0 = YEAR_START if YEAR_START is not None else file_y0
    y1 = YEAR_END   if YEAR_END   is not None else file_y1
    years = list(range(y0, y1 + 1))
    print(f"Detected years in file: {file_y0}–{file_y1}. Processing: {years[0]}–{years[-1]}.")

    # ---------- Build station master FeatureCollection ----------
    stations_meta = pd.read_csv(CSV_PATH, usecols=[STATION_ID_COL, LAT_COL, LON_COL])
    stations_meta = stations_meta.dropna(subset=[STATION_ID_COL, LAT_COL, LON_COL]).copy()
    stations_meta[STATION_ID_COL] = stations_meta[STATION_ID_COL].astype(str)
    stations_meta = stations_meta.drop_duplicates(subset=[STATION_ID_COL])  # one row per station id

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
            # coerce daily columns to numeric
            for c in cols_y:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

            # iterate rows (stations) in this chunk
            for _, row in chunk.iterrows():
                sid = row[STATION_ID_COL]

                # Array of that year daily rainfall
                vals = row[cols_y].to_numpy(dtype=float)
                # QC: drop negative and implausibly large values
                vals = np.where((vals >= 0) & (vals <= 500), vals, np.nan)

                # Need enough days to form a percentile
                if np.count_nonzero(~np.isnan(vals)) < 30:
                    continue

                thr = np.nanpercentile(vals, PCT * 100.0)
                idx = np.where(vals > thr)[0]
                if idx.size == 0:
                    continue

                dates_y = [parse_date_header(c) for c in cols_y]  # same order as vals
                exc_dates = [dates_y[i] for i in idx]
                exc_vals  = [vals[i] for i in idx]

                exc_df = pd.DataFrame({'date': exc_dates, 'pr_g': exc_vals}).sort_values('date')
                kept_dates = decluster_dates(list(exc_df['date']), MIN_GAP_DAYS)
                if len(kept_dates) < MIN_PER_YEAR:
                    continue

                pr_lookup = dict(zip(exc_df['date'], exc_df['pr_g']))
                for d in kept_dates:
                    prg = float(pr_lookup.get(d, np.nan))
                    # Guard tiny / NaN values to avoid wild ratios
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

        # Build the FeatureCollection for the full year by merging per-date samples
        fc_year = ee.FeatureCollection([])

        for dstr, entry in tqdm(per_date.items(), desc=f"EE sampling {y}"):
            sids = entry['sids']
            prg_dict = entry['prg']

            # Select only stations with extremes on this date
            fc_subset = fc_all.filter(ee.Filter.inList('station_id', ee.List(sids)))

            # CHIRPS image for that date (fallback to 0 image if missing)
            d0 = ee.Date(dstr)
            img0 = CHIRPS.filterDate(d0, d0.advance(1, 'day')).first()
            img = ee.Image(
                ee.Algorithms.If(
                    img0,
                    ee.Image(img0).select('precipitation'),
                    ee.Image.constant(0).rename('precipitation').toFloat()
                )
            )

            # Sample at stations
            samples = img.sampleRegions(
                collection=fc_subset,
                properties=['station_id', 'lat', 'lon'],  # NEW -> brings lat/lon from station FC
                scale=SAMPLE_SCALE_M
            )

            # Attach gauge rain and ratio WITHOUT returning nulls
            # Attach gauge rain & ratio WITHOUT returning nulls; preserve existing props (incl. lat/lon)
            prg = ee.Dictionary(prg_dict)

            def with_props(feat):
                sid = ee.String(feat.get('station_id'))
                prc = ee.Number(feat.get('precipitation'))
                prg_val = ee.Number(prg.get(sid))
                ratio = prg_val.divide(prc)  # may be inf if prc==0; filtered below

                # Keep original properties (station_id, lat, lon) and add new ones
                return feat.set({
                    'date': dstr,
                    'chirps_mm': prc,
                    'pr_g': prg_val,
                    'ratio': ratio
                })

            samples2 = (
                ee.FeatureCollection(samples.map(with_props))
                .filter(ee.Filter.gt('chirps_mm', 1))  # avoid tiny/zero
                .filter(ee.Filter.gt('pr_g', 1))
                .filter(ee.Filter.notNull(['lat', 'lon', 'ratio']))  # ensure coords & ratio exist
            )

            # Merge only if non-empty (defensive)
            fc_year = ee.FeatureCollection(
                ee.Algorithms.If(samples2.size().gt(0), fc_year.merge(samples2), fc_year)
            )

        # Export yearly CSV
        task = ee.batch.Export.table.toDrive(
            collection=fc_year,
            description=f"CHIRPS_BiasPairs_{y}",
            folder=OUT_GDRIVE_FOLDER,
            fileNamePrefix=f"chirps_bias_pairs_{y}",
            fileFormat='CSV',
            selectors=['station_id','lat','lon','date','chirps_mm','pr_g','ratio']
        )
        task.start()
        print(f"[{y}] Export started. Task ID: {task.id}")

    print("\nAll yearly exports submitted. Monitor progress in the GEE Tasks panel (Code Editor or CLI).")


if __name__ == "__main__":
    main()
