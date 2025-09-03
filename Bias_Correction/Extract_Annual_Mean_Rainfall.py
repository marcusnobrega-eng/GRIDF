# -*- coding: utf-8 -*-
# Compute mean annual rainfall (mm/yr) ON GEE and export ONE GeoTIFF per product.

import ee
ee.Initialize(project=None)  # or ee.Initialize(project='ee-your-project')

YEAR_START_GLOBAL = 1994
YEAR_END_GLOBAL   = 2024

# Per-product overrides (useful for IMERG start year, etc.)
YEAR_OVERRIDES = {
    "IMERG": 2000,   # CE daily product begins 2000
    "BR-DWGD (Xavier)": 1994,  # earliest reliable range in BR-DWGD
}

# ---- Brazil boundary for region clip/export ----
BRA = (ee.FeatureCollection("FAO/GAUL/2015/level0")
         .filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
         .geometry())

# ---- Daily band mappers → 'pr' in mm/day ----
def daily_xavier(img):
    raw = img.select(0).toFloat()
    mm  = raw.multiply(0.006866665).subtract(225.0).max(0)
    return mm.rename("pr")

def daily_chirps(img):   return img.select("precipitation").toFloat().rename("pr")
def daily_persiann(img): return img.select("precipitation").toFloat().rename("pr")
def daily_imerg(img):    return img.select("precipitationCal").toFloat().rename("pr")

PRODUCTS = [
    ("BR-DWGD (Xavier)", "projects/sat-io/open-datasets/BR-DWGD/PR",         daily_xavier, 11132),
    ("IMERG",             "projects/climate-engine-pro/assets/ce-gpm-imerg-daily", daily_imerg, 11132),
    ("CHIRPS",            "UCSB-CHG/CHIRPS/DAILY",                           daily_chirps,  5560),
    ("PERSIANN",          "NOAA/PERSIANN-CDR",                               daily_persiann,27830),
]

# ---- Helpers ----
def set_time_from_index_xavier(img):
    idx = ee.String(img.get("system:index"))
    datestr = idx.slice(-8)  # last 8 chars are YYYYMMDD
    return img.set("system:time_start",
                   ee.Date.parse("yyyyMMdd", datestr).millis())

def with_time_fallback(ic, product_name):
    """Ensure images have system:time_start."""
    if "Xavier" in product_name:
        return ic.map(set_time_from_index_xavier)
    else:
        # generic fallback for YYYYMMDD or pr_YYYYMMDD formats
        def _parse_idx_to_millis(img):
            s = ee.String(img.get('system:index'))
            date_str = ee.Algorithms.If(
                s.match('^pr_\\d{8}$'), s.slice(3, 11),
                ee.Algorithms.If(s.match('^\\d{8}$'), s.slice(0, 8), None)
            )
            t = ee.Algorithms.If(date_str,
                                 ee.Date.parse('yyyyMMdd', ee.String(date_str)).millis(),
                                 img.date().millis())
            return img.set('system:time_start', t)
        return ic.map(_parse_idx_to_millis)

def mean_annual_mm_per_year(ic_daily, y0, y1):
    years = ee.List.sequence(y0, y1)

    def yearly_total(y):
        y = ee.Number(y)
        ann = (ic_daily
               .filter(ee.Filter.calendarRange(y, y, 'year'))
               .sum()
               .rename('pr')
               .set('year', y))
        return ann

    ann_ic = ee.ImageCollection(years.map(yearly_total))
    return ee.Image(ee.Algorithms.If(
        ann_ic.size().gt(0),
        ann_ic.mean().rename('mean_ann'),
        ee.Image.constant(0).rename('mean_ann').updateMask(ee.Image(0))
    ))

def export_one(name, asset_id, mapper, scale_m, y0, y1, folder="annual_means"):
    ic = ee.ImageCollection(asset_id).map(mapper).filterBounds(BRA)
    ic = with_time_fallback(ic, name)

    ic_f = ic.filterDate(f"{y0}-01-01", f"{y1}-12-31")
    mean_img = mean_annual_mm_per_year(ic_f, y0, y1).clip(BRA)

    cond = mean_img.bandNames().size().gt(0)
    mean_img = ee.Image(ee.Algorithms.If(cond, mean_img, None))

    # Safe description/fileName
    safe_name = (name.replace(" ", "_")
                      .replace("(", "")
                      .replace(")", "")
                      .replace("-", "_"))

    desc = f"MeanAnnual_{safe_name}_{y0}_{y1}"
    fname = f"{safe_name}_MeanAnnual_{y0}_{y1}"

    task = ee.batch.Export.image.toDrive(
        image=mean_img,
        description=desc[:95],
        folder=folder,
        fileNamePrefix=fname[:95],
        region=BRA,
        scale=scale_m,
        crs="EPSG:4326",
        maxPixels=1e13
    )
    task.start()
    print(f"→ Export started: {name} [{y0}-{y1}] to Drive/{folder}")

def main():
    for name, asset_id, mapper, scale_m in PRODUCTS:
        y0 = YEAR_OVERRIDES.get(name, YEAR_START_GLOBAL)
        y1 = YEAR_END_GLOBAL
        export_one(name, asset_id, mapper, scale_m, y0, y1)

if __name__ == "__main__":
    main()
