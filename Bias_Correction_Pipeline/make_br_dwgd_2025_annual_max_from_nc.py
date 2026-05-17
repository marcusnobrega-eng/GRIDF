#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import yaml


# ============================================================
# USER INPUT
# ============================================================

NC_FILE = Path("/Users/mngomes/Downloads/pr_20010101_20251231_BR-DWGD_UFES_UTEXAS_v_3.2.4.nc")
YEAR = 2025

PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")
PRODUCT_CONFIG = PIPELINE_ROOT / "config" / "products.yml"

OUT_NAME = f"BR_DWGD_MaxDaily_0p10deg_{YEAR}_Brazil.tif"

# If the NetCDF is already CF-decoded, xarray applies scale_factor/add_offset automatically.
# Keep this True unless we diagnose otherwise.
DECODE_CF = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_precip_variable(ds):
    """
    Robustly find the precipitation variable in the BR-DWGD NetCDF.
    """
    candidate_names = ["pr", "precip", "precipitation", "rain", "rainfall"]

    for name in candidate_names:
        if name in ds.data_vars:
            return name

    # Fallback: choose first 3D variable with time + lat + lon dimensions.
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        if any(d.lower() in ["time", "times"] for d in dims):
            if any(d.lower() in ["lat", "latitude", "y"] for d in dims):
                if any(d.lower() in ["lon", "longitude", "x"] for d in dims):
                    return name

    raise ValueError(
        "Could not identify precipitation variable. Available variables:\n"
        + "\n".join(ds.data_vars)
    )


def find_coord_name(ds, options):
    """
    Find coordinate name from common alternatives.
    """
    lower_map = {c.lower(): c for c in list(ds.coords) + list(ds.dims)}

    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]

    raise ValueError(
        f"Could not find coordinate among {options}. "
        f"Available coords: {list(ds.coords)}; dims: {list(ds.dims)}"
    )


def write_geotiff_from_latlon(output_tif, arr2d, lat, lon, nodata=-9999.0):
    """
    Write a lat/lon array to GeoTIFF.

    Assumes regular lon/lat grid in EPSG:4326.
    Handles latitude orientation automatically.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    data = np.asarray(arr2d, dtype=np.float32)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("This script expects 1D latitude and longitude coordinates.")

    # Sort longitude ascending if needed.
    if np.any(np.diff(lon) < 0):
        idx = np.argsort(lon)
        lon = lon[idx]
        data = data[:, idx]

    # GeoTIFF convention: first row is north. If latitude is ascending south->north,
    # flip data and latitude.
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    dx = float(np.median(np.abs(np.diff(lon))))
    dy = float(np.median(np.abs(np.diff(lat))))

    west = float(lon.min() - dx / 2.0)
    north = float(lat.max() + dy / 2.0)

    transform = from_origin(west, north, dx, dy)

    out = data.copy()
    out[~np.isfinite(out)] = nodata

    profile = {
        "driver": "GTiff",
        "height": out.shape[0],
        "width": out.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "BIGTIFF": "IF_SAFER",
    }

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(out.astype(np.float32), 1)

    return output_tif


# ============================================================
# MAIN
# ============================================================

def main():
    if not NC_FILE.exists():
        raise FileNotFoundError(f"NetCDF file not found:\n{NC_FILE}")

    if not PRODUCT_CONFIG.exists():
        raise FileNotFoundError(f"Product config not found:\n{PRODUCT_CONFIG}")

    with open(PRODUCT_CONFIG, "r") as f:
        products = yaml.safe_load(f)

    br_cfg = products["products"]["br_dwgd"]
    out_folder = Path(br_cfg["annual_max_folder"])
    out_tif = out_folder / OUT_NAME

    print("=" * 80)
    print("BR-DWGD 2025 annual maximum from local NetCDF")
    print("=" * 80)
    print("NetCDF:       ", NC_FILE)
    print("Output folder:", out_folder)
    print("Output file:  ", out_tif)
    print("=" * 80)

    print("\nOpening NetCDF...")
    ds = xr.open_dataset(NC_FILE, decode_cf=DECODE_CF)

    print("\nDataset summary:")
    print(ds)

    var_name = find_precip_variable(ds)
    time_name = find_coord_name(ds, ["time", "times"])
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])

    print("\nDetected names:")
    print("Variable:", var_name)
    print("Time:    ", time_name)
    print("Lat:     ", lat_name)
    print("Lon:     ", lon_name)

    da = ds[var_name]

    print("\nVariable attributes:")
    for k, v in da.attrs.items():
        print(f"  {k}: {v}")

    # Select only target year.
    start = f"{YEAR}-01-01"
    end = f"{YEAR}-12-31"

    da_year = da.sel({time_name: slice(start, end)})

    n_days = da_year.sizes.get(time_name, None)
    print(f"\nSelected year: {YEAR}")
    print("Number of daily slices:", n_days)

    if n_days is None or n_days == 0:
        raise RuntimeError(f"No data found for {YEAR} in the NetCDF.")

    # Convert to float and remove physically impossible negative values.
    # Small negative values can occur from decoded missing/fill artifacts.
    da_year = da_year.astype("float32")
    da_year = da_year.where(da_year >= 0)

    print("\n2025 daily precipitation quick stats before annual max:")
    print("min:   ", float(da_year.min(skipna=True).values))
    print("mean:  ", float(da_year.mean(skipna=True).values))
    print("max:   ", float(da_year.max(skipna=True).values))

    annual_max = da_year.max(dim=time_name, skipna=True)

    # Force dimensions to lat, lon order.
    annual_max = annual_max.transpose(lat_name, lon_name)

    arr = annual_max.values
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    print("\nAnnual maximum 2025 stats:")
    print("shape: ", arr.shape)
    print("min:   ", float(np.nanmin(arr)))
    print("mean:  ", float(np.nanmean(arr)))
    print("median:", float(np.nanmedian(arr)))
    print("max:   ", float(np.nanmax(arr)))

    # Basic sanity check.
    if np.nanmax(arr) <= 0:
        raise RuntimeError("Annual maximum is non-positive everywhere. Something is wrong.")
    if np.nanmedian(arr) > 1000:
        raise RuntimeError(
            "Annual maximum values are extremely high. "
            "The NetCDF may not be decoded correctly."
        )

    print("\nWriting GeoTIFF...")
    write_geotiff_from_latlon(out_tif, arr, lat, lon)

    print("\nDone.")
    print("Created:")
    print(out_tif)

    ds.close()


if __name__ == "__main__":
    main()
