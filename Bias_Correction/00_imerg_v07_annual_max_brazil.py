# 03_imerg_v07_annual_max_brazil_native.py
# ---------------------------------------------------------
# Purpose:
#   For each year, compute the maximum daily precipitation
#   from IMERG V07 over Brazil, and export one GeoTIFF to Drive.
#
# Dataset:
#   NASA/GPM_L3/IMERG_V07
#
# Band:
#   precipitation, mm/hour
#
# Method:
#   IMERG V07 in GEE is half-hourly.
#   Daily total = mean daily precipitation rate * 24.
#
# Output:
#   One GeoTIFF per year
#   CRS: EPSG:4326
#   Native grid: 0.1° x 0.1°
# ---------------------------------------------------------

import ee

# ---------------- USER INPUTS ----------------
YEAR_START = 2000
YEAR_END = 2025

GEE_PROJECT = 'mngomes2026'  # or None

OUT_DRIVE_FOLDER = 'IMERG_V07_Max'

FNAME_PREFIX = 'IMERG_V07_MaxDaily_0p10deg_{year}_Brazil'

MAX_PIXELS = 1e13

CRS = 'EPSG:4326'

# Native IMERG grid: 0.1 degree
CRS_TRANSFORM = [0.1, 0, -180, 0, -0.1, 90]

IMERG_COLLECTION_ID = 'NASA/GPM_L3/IMERG_V07'
IMERG_BAND = 'precipitation'
# --------------------------------------------


def make_daily_imerg_total(imerg_collection, date, brazil_geom):
    """
    Converts one day of IMERG V07 half-hourly precipitation rate to daily total.

    The precipitation band is treated as mm/hour.

    Daily rainfall:
        daily total = mean daily rate * 24

    Empty days are handled by returning a fully masked image.
    """

    date = ee.Date(date)
    next_date = date.advance(1, 'day')

    day_ic = (
        imerg_collection
        .filterDate(date, next_date)
        .select(IMERG_BAND)
    )

    count = day_ic.size()

    empty_daily = (
        ee.Image.constant(0)
        .rename('precipitation')
        .updateMask(ee.Image.constant(0))
        .clip(brazil_geom)
    )

    daily_total = (
        day_ic
        .mean()
        .multiply(24.0)
        .rename('precipitation')
        .clip(brazil_geom)
    )

    daily = ee.Image(
        ee.Algorithms.If(
            count.gt(0),
            daily_total,
            empty_daily
        )
    ).set({
        'system:time_start': date.millis(),
        'date': date.format('YYYY-MM-dd'),
        'image_count': count,
        'system:index': date.format('YYYYMMdd')
    })

    return daily


def make_imerg_daily_collection_for_year(imerg_collection, year, brazil_geom):
    """
    Builds a daily IMERG total collection for one calendar year.
    """

    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')

    n_days = end.difference(start, 'day')

    dates = ee.List.sequence(0, n_days.subtract(1)).map(
        lambda d: start.advance(ee.Number(d), 'day')
    )

    daily_ic = ee.ImageCollection.fromImages(
        dates.map(lambda date: make_daily_imerg_total(imerg_collection, date, brazil_geom))
    )

    return daily_ic


def main():
    if GEE_PROJECT:
        ee.Initialize(project=GEE_PROJECT)
    else:
        ee.Initialize()

    brazil_fc = (
        ee.FeatureCollection('FAO/GAUL/2015/level0')
        .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
    )

    brazil_geom = brazil_fc.geometry()

    imerg = (
        ee.ImageCollection(IMERG_COLLECTION_ID)
        .filterBounds(brazil_geom)
        .filterDate(f'{YEAR_START}-01-01', f'{YEAR_END + 1}-01-01')
        .select(IMERG_BAND)
    )

    years = list(range(YEAR_START, YEAR_END + 1))
    print(f"Exporting IMERG V07 annual maxima for years: {years[0]}–{years[-1]}")

    for y in years:
        imerg_daily_y = make_imerg_daily_collection_for_year(
            imerg_collection=imerg,
            year=y,
            brazil_geom=brazil_geom
        )

        annual_max = (
            imerg_daily_y
            .max()
            .clip(brazil_geom)
            .rename('max_precip_mm_day')
            .set({
                'product': IMERG_COLLECTION_ID,
                'band': IMERG_BAND,
                'stat': 'annual_max_daily_precipitation',
                'year': y
            })
        )

        desc = f'IMERG_V07_AMaxDaily_{y}'
        fname = FNAME_PREFIX.format(year=y)

        task = ee.batch.Export.image.toDrive(
            image=annual_max,
            description=desc,
            folder=OUT_DRIVE_FOLDER,
            fileNamePrefix=fname,
            region=brazil_geom,
            crs=CRS,
            crsTransform=CRS_TRANSFORM,
            maxPixels=MAX_PIXELS
        )

        task.start()
        print(f"[{y}] Export started -> Drive/{OUT_DRIVE_FOLDER}/{fname}.tif | Task ID: {task.id}")

    print("\nAll yearly IMERG V07 exports submitted.")


if __name__ == "__main__":
    main()