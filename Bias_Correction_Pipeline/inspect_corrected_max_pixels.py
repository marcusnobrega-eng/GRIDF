#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

PRODUCTS = ["imerg_v07", "imerg_v06", "chirps", "persiann_cdr", "br_dwgd"]
PERCENTILE = "p98"
ESTIMATOR = "median"

rows = []

for product in PRODUCTS:
    folder = Path(f"data/products/{product}/sensitivity/{PERCENTILE}/annual_max_corrected/{ESTIMATOR}")
    files = sorted(folder.glob(f"corrected_{product}_{PERCENTILE}_{ESTIMATOR}_*.tif"))

    for f in files:
        m = re.search(r"_(\d{4})\.tif$", f.name)
        year = int(m.group(1)) if m else None

        with rasterio.open(f) as src:
            arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

            valid = np.isfinite(arr)
            if not valid.any():
                continue

            max_val = float(np.nanmax(arr))
            r, c = np.unravel_index(np.nanargmax(arr), arr.shape)
            lon, lat = xy(src.transform, r, c)

            rows.append({
                "product": product,
                "year": year,
                "file": str(f),
                "max_corrected_mm_day": max_val,
                "lon": lon,
                "lat": lat,
                "mean_corrected_mm_day": float(np.nanmean(arr)),
                "p99_corrected_mm_day": float(np.nanpercentile(arr[valid], 99)),
                "p999_corrected_mm_day": float(np.nanpercentile(arr[valid], 99.9)),
                "n_valid_pixels": int(valid.sum()),
            })

out = pd.DataFrame(rows)
out = out.sort_values(["product", "max_corrected_mm_day"], ascending=[True, False])

print("=" * 100)
print("Top corrected annual-maximum pixels by product")
print("=" * 100)

for product in PRODUCTS:
    print("\n" + "=" * 100)
    print(product)
    print("=" * 100)
    sub = out[out["product"] == product].copy()
    cols = [
        "year",
        "max_corrected_mm_day",
        "p999_corrected_mm_day",
        "p99_corrected_mm_day",
        "mean_corrected_mm_day",
        "lon",
        "lat",
    ]
    print(sub[cols].head(10).to_string(index=False))

out_path = Path("data/products/p98_corrected_max_pixel_diagnostic.csv")
out.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)
