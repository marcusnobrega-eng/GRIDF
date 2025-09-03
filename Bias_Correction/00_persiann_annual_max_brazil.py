# 00_persiann_annual_max_brazil.py
# ---------------------------------------------------------
# Purpose:
#   For each year (1994–2024), compute the maximum daily precipitation
#   in PERSIANN-CDR over Brazil, and export one GeoTIFF to Drive.
#
# Requirements:
#   pip install earthengine-api
#   earthengine authenticate
#
# Notes:
#   - Output CRS: EPSG:4326, scale ~0.25° (≈27.83 km)
#   - Dataset: NOAA/PERSIANN-CDR, band 'precipitation' (mm/day)
# ---------------------------------------------------------

import ee

# ---------------- USER INPUTS ----------------
YEAR_START = 1994
YEAR_END   = 2024

# Optional: if you have an EE project, set it; else leave None
GEE_PROJECT = 'ee-marcusep2025'  # or None

# Output to Google Drive
OUT_DRIVE_FOLDER = 'PERSIANN_Max'  # change if you want another folder

# Export naming
FNAME_PREFIX = 'PERSIANN_MaxDaily_0p25deg_{year}_Brazil'

# Export scale (meters). 0.25° at equator ≈ 27.83 km
SCALE_M = 27830

# Max pixels for export
MAX_PIXELS = 1e13
# --------------------------------------------


def main():
    # Initialize Earth Engine
    if GEE_PROJECT:
        ee.Initialize(project=GEE_PROJECT)
    else:
        ee.Initialize()

    # PERSIANN-CDR daily collection
    persiann = ee.ImageCollection('NOAA/PERSIANN-CDR').select('precipitation')

    # Brazil boundary from GAUL Level-0
    # (Fields: ADM0_NAME contains country name)
    brazil_fc = (
        ee.FeatureCollection('FAO/GAUL/2015/level0')
        .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
    )
    brazil_geom = brazil_fc.geometry()

    years = list(range(YEAR_START, YEAR_END + 1))
    print(f"Exporting annual maxima for years: {years[0]}–{years[-1]}")

    for y in years:
        start = ee.Date.fromYMD(y, 1, 1)
        end   = start.advance(1, 'year')

        # Max over the year, clipped to Brazil
        annual_max = (
            persiann
            .filterDate(start, end)
            .max()
            .clip(brazil_geom)
            .rename('max_precip_mm_day')
            .set({
                'product': 'NOAA/PERSIANN-CDR',
                'band': 'precipitation',
                'stat': 'annual_max',
                'year': y
            })
        )

        # (Optional) force nominal scale (reprojection) so exports are consistent
        # annual_max = annual_max.reproject(crs='EPSG:4326', scale=SCALE_M)

        # Prepare export
        desc = f'PERSIANN_AMaxDaily_{y}'
        fname = FNAME_PREFIX.format(year=y)

        task = ee.batch.Export.image.toDrive(
            image=annual_max,
            description=desc,
            folder=OUT_DRIVE_FOLDER,
            fileNamePrefix=fname,
            region=brazil_geom,
            scale=SCALE_M,
            crs='EPSG:4326',
            maxPixels=MAX_PIXELS
        )
        task.start()
        print(f"[{y}] Export started -> Drive/{OUT_DRIVE_FOLDER}/{fname}.tif  (Task ID: {task.id})")

    print("\nAll yearly exports submitted. Monitor them in the GEE Tasks panel (Code Editor or CLI).")


if __name__ == "__main__":
    main()
