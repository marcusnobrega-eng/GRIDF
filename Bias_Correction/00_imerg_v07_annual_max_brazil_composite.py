# 02_imerg_v07_early_annual_max_brazil_daily_climate_engine.py
# ---------------------------------------------------------
# Purpose:
#   For each year, compute the maximum daily precipitation
#   from the Climate Engine GPM Early Daily V07 product over Brazil,
#   and export one GeoTIFF to Google Drive.
#
# Dataset page:
#   https://www.climateengine.org/datasets/climatehydrology/gpmearly_daily_11000/
#
# Earth Engine asset:
#   projects/climate-engine-pro/assets/ce-gpm-imerg-v07/early-daily
#
# Notes:
#   - Daily precipitation is already summed to daily totals.
#   - Daily boundary is 0 UTC.
#   - Units are mm/day.
#   - 2000 is partial because the product starts on 2000-06-01.
#   - 2026 is year-to-date unless the full year is already available.
#
# Output:
#   One GeoTIFF per year
#   CRS: EPSG:4326
#   Grid: 0.1° x 0.1°
# ---------------------------------------------------------

import ee

# ---------------- USER INPUTS ----------------
YEAR_START = 2000
YEAR_END = 2026

GEE_PROJECT = "ee-marcusep2025"

OUT_DRIVE_FOLDER = "GPM_Early_V07_Daily_Max_Composite"

FNAME_PREFIX = "GPM_Early_V07_MaxDaily_0p10deg_{year}_Brazil"

MAX_PIXELS = 1e13

CRS = "EPSG:4326"

# 0.1 degree grid, global origin
CRS_TRANSFORM = [0.1, 0, -180, 0, -0.1, 90]

GPM_EARLY_DAILY_COLLECTION_ID = (
    "projects/climate-engine-pro/assets/ce-gpm-imerg-v07/early-daily"
)

# Set to None to auto-detect from the first image.
# If auto-detect fails, run once and inspect the printed band list.
GPM_DAILY_BAND = None

# Likely band names to try first.
PRECIP_BAND_CANDIDATES = [
    "precipitation",
    "Precipitation",
    "precip",
    "precipitationCal",
    "precipitation_cal",
]
# --------------------------------------------


def get_precip_band(collection_id):
    """
    Detects the precipitation band name from the first image.
    This avoids hard-coding a band name that may differ between
    Climate Engine assets.
    """

    first = ee.Image(ee.ImageCollection(collection_id).first())
    band_names = first.bandNames().getInfo()

    print("Available bands in first image:")
    for b in band_names:
        print(f"  - {b}")

    for candidate in PRECIP_BAND_CANDIDATES:
        if candidate in band_names:
            print(f"Using precipitation band: {candidate}")
            return candidate

    precip_like = [
        b for b in band_names
        if "precip" in b.lower() or "prcp" in b.lower() or "rain" in b.lower()
    ]

    if len(precip_like) == 1:
        print(f"Using precipitation-like band: {precip_like[0]}")
        return precip_like[0]

    raise ValueError(
        "Could not auto-detect precipitation band. "
        f"Available bands are: {band_names}. "
        "Set GPM_DAILY_BAND manually."
    )


def make_annual_max_gpm_early_daily(year, brazil_geom, precip_band):
    """
    Computes annual maximum daily precipitation from the already-daily
    Climate Engine GPM Early Daily V07 collection.
    """

    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")

    daily_ic = (
        ee.ImageCollection(GPM_EARLY_DAILY_COLLECTION_ID)
        .filterDate(start, end)
        .filterBounds(brazil_geom)
        .select(precip_band)
    )

    annual_max = (
        daily_ic
        .max()
        .clip(brazil_geom)
        .rename("max_precip_mm_day")
        .set({
            "product": GPM_EARLY_DAILY_COLLECTION_ID,
            "climate_engine_id": "GPM_DAILY_EARLY",
            "source_product": "GPM IMERG V07 Early Daily",
            "band": precip_band,
            "units": "mm/day",
            "stat": "annual_max_daily_precipitation",
            "year": year,
            "daily_time_boundary": "0 UTC",
            "note": "GPM Early is provisional near-real-time data"
        })
    )

    return annual_max


def main():
    ee.Initialize(project=GEE_PROJECT)

    print("Earth Engine initialized.")
    print(f"Using project: {GEE_PROJECT}")
    print(f"Using collection: {GPM_EARLY_DAILY_COLLECTION_ID}")

    precip_band = GPM_DAILY_BAND
    if precip_band is None:
        precip_band = get_precip_band(GPM_EARLY_DAILY_COLLECTION_ID)

    brazil_fc = (
        ee.FeatureCollection("FAO/GAUL/2015/level0")
        .filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
    )

    brazil_geom = brazil_fc.geometry()

    years = list(range(YEAR_START, YEAR_END + 1))

    print(
        f"\nExporting GPM Early Daily V07 annual maxima "
        f"for years: {years[0]}–{years[-1]}"
    )

    if YEAR_START == 2000:
        print(
            "Warning: 2000 is a partial year because the Climate Engine "
            "GPM daily record starts on 2000-06-01."
        )

    if YEAR_END >= 2026:
        print(
            "Warning: 2026 is likely year-to-date, not a complete annual maximum."
        )

    for y in years:
        annual_max = make_annual_max_gpm_early_daily(
            year=y,
            brazil_geom=brazil_geom,
            precip_band=precip_band
        )

        desc = f"GPM_Early_V07_Daily_AMaxDaily_{y}"
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

        print(
            f"[{y}] Export started -> "
            f"Drive/{OUT_DRIVE_FOLDER}/{fname}.tif | Task ID: {task.id}"
        )

    print("\nAll yearly GPM Early Daily V07 exports submitted.")


if __name__ == "__main__":
    main()