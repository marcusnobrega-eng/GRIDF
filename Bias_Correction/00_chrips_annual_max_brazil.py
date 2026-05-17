# 01_chirps_annual_max_brazil_native.py
# ---------------------------------------------------------
# Purpose:
#   For each year, compute the maximum daily precipitation
#   in CHIRPS over Brazil, and export one GeoTIFF to Drive.
#
# Dataset:
#   UCSB-CHG/CHIRPS/DAILY
#
# Band:
#   precipitation, mm/day
#
# Output:
#   One GeoTIFF per year
#   CRS: EPSG:4326
#   Native grid: 0.05° x 0.05°
# ---------------------------------------------------------

import ee

# ---------------- USER INPUTS ----------------
YEAR_START = 1995
YEAR_END = 2025

GEE_PROJECT = 'ee-marcusep2025'  # or None

OUT_DRIVE_FOLDER = 'CHIRPS_Max'

FNAME_PREFIX = 'CHIRPS_MaxDaily_0p05deg_{year}_Brazil'

MAX_PIXELS = 1e13

CRS = 'EPSG:4326'

# Native CHIRPS grid: 0.05 degree
CRS_TRANSFORM = [0.05, 0, -180, 0, -0.05, 90]
# --------------------------------------------


def main():
    if GEE_PROJECT:
        ee.Initialize(project=GEE_PROJECT)
    else:
        ee.Initialize()

    chirps = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .select('precipitation')
    )

    brazil_fc = (
        ee.FeatureCollection('FAO/GAUL/2015/level0')
        .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
    )

    brazil_geom = brazil_fc.geometry()

    years = list(range(YEAR_START, YEAR_END + 1))
    print(f"Exporting CHIRPS annual maxima for years: {years[0]}–{years[-1]}")

    for y in years:
        start = ee.Date.fromYMD(y, 1, 1)
        end = start.advance(1, 'year')

        annual_max = (
            chirps
            .filterDate(start, end)
            .max()
            .clip(brazil_geom)
            .rename('max_precip_mm_day')
            .set({
                'product': 'UCSB-CHG/CHIRPS/DAILY',
                'band': 'precipitation',
                'stat': 'annual_max_daily_precipitation',
                'year': y
            })
        )

        desc = f'CHIRPS_AMaxDaily_{y}'
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

    print("\nAll yearly CHIRPS exports submitted.")


if __name__ == "__main__":
    main()