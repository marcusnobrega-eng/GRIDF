#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree


PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")

BOUNDARY_PATH = Path(
    "/Users/mngomes/Documents/GitHub/GRIDF/BrazilShapefiles/ADMLevels/"
    "bra_admbnda_adm0_ibge_2020.shp"
)

PERCENTILE = "p98"

PRODUCTS = {
    "br_dwgd": {
        "grid_res_deg": 0.10,
        "label": "BR-DWGD",
    },
    "imerg_v06": {
        "grid_res_deg": 0.10,
        "label": "IMERG V06",
    },
    "imerg_v07": {
        "grid_res_deg": 0.10,
        "label": "IMERG V07",
    },
    "chirps": {
        "grid_res_deg": 0.05,
        "label": "CHIRPS",
    },
    "persiann_cdr": {
        "grid_res_deg": 0.25,
        "label": "PERSIANN-CDR",
    },
}

ESTIMATORS = ["mean", "median"]

PAD_DEG = 0.5

METHOD = "idw"
IDW_POWER = 2.0
IDW_K = 8

# Legacy filters
MIN_PAIRS = 10
ZETA_CLIP = (0.05, 10.0)
IDW_CAP_RANGE = (0.25, 10.0)


def load_boundary_shapes(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    import geopandas as gpd
    from shapely.geometry import mapping

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        warnings.warn("Boundary has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    geom = gdf.unary_union
    shapes = [mapping(geom)]
    bounds = tuple(gdf.total_bounds)

    return shapes, bounds


def build_grid(bounds, res_deg, pad_deg=0.0):
    minx, miny, maxx, maxy = bounds

    minx -= pad_deg
    miny -= pad_deg
    maxx += pad_deg
    maxy += pad_deg

    xs = np.arange(minx, maxx + res_deg, res_deg)
    ys = np.arange(miny, maxy + res_deg, res_deg)

    xx, yy = np.meshgrid(xs, ys)

    transform = from_origin(xs.min(), ys.max(), res_deg, res_deg)

    return xs, ys, xx, yy, transform


def idw_interpolate(lon, lat, z, xx, yy, k=8, power=2.0):
    pts = np.c_[lon, lat]
    tree = cKDTree(pts)

    grid_pts = np.c_[xx.ravel(), yy.ravel()]

    k = min(k, len(pts))

    try:
        dists, idxs = tree.query(grid_pts, k=k, workers=-1)
    except TypeError:
        dists, idxs = tree.query(grid_pts, k=k)

    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    dists = np.where(dists == 0, 1e-12, dists)

    w = 1.0 / (dists ** power)
    w /= w.sum(axis=1, keepdims=True)

    z_idw = np.sum(w * z[idxs], axis=1)

    return z_idw.reshape(xx.shape)


def save_geotiff(path, array2d, transform, nodata=np.nan, crs="EPSG:4326"):
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = array2d.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 6,
        "nodata": nodata,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array2d.astype("float32"), 1)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_points(product, estimator):
    zeta_csv = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / PERCENTILE
        / "zeta_station"
        / estimator
        / f"zeta_per_station_{product}_{PERCENTILE}_{estimator}.csv"
    )

    if not zeta_csv.exists():
        raise FileNotFoundError(zeta_csv)

    df = pd.read_csv(zeta_csv, low_memory=False)

    zeta_col = find_col(df, ["zeta", "zeta_selected", "zeta_mean", "zeta_median"])
    lat_col = find_col(df, ["lat", "latitude", "station_lat", "Latitude"])
    lon_col = find_col(df, ["lon", "longitude", "station_lon", "Longitude"])

    if zeta_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            f"{zeta_csv} must contain zeta and coordinate columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    df["zeta_use"] = pd.to_numeric(df[zeta_col], errors="coerce")
    df["lat_use"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon_use"] = pd.to_numeric(df[lon_col], errors="coerce")

    # Legacy MIN_PAIRS logic.
    if "n_pairs" in df.columns:
        n = pd.to_numeric(df["n_pairs"], errors="coerce")
        df = df[n.fillna(0) >= MIN_PAIRS].copy()
    elif "n_pairs_used" in df.columns:
        n = pd.to_numeric(df["n_pairs_used"], errors="coerce")
        df = df[n.fillna(0) >= MIN_PAIRS].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["zeta_use", "lat_use", "lon_use"])

    df = df[
        df["zeta_use"].between(ZETA_CLIP[0], ZETA_CLIP[1])
        & df["lat_use"].between(-90, 90)
        & df["lon_use"].between(-180, 180)
    ].copy()

    if df.empty:
        raise ValueError(f"No valid points after filtering: {zeta_csv}")

    return df[["lon_use", "lat_use", "zeta_use"]].rename(
        columns={"lon_use": "lon", "lat_use": "lat", "zeta_use": "zeta"}
    )


def interpolate_one(product, estimator, shapes, bounds):
    spec = PRODUCTS[product]
    res = spec["grid_res_deg"]

    df = load_points(product, estimator)

    lon = df["lon"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    z = df["zeta"].to_numpy(dtype=float)

    xs, ys, xx, yy, transform = build_grid(bounds, res, PAD_DEG)

    z_grid = idw_interpolate(
        lon=lon,
        lat=lat,
        z=z,
        xx=xx,
        yy=yy,
        k=IDW_K,
        power=IDW_POWER,
    )

    z_grid = np.clip(z_grid, IDW_CAP_RANGE[0], IDW_CAP_RANGE[1])

    mask_outside = geometry_mask(
        shapes,
        z_grid.shape,
        transform,
        invert=False,
        all_touched=False,
    )

    z_grid = np.where(~mask_outside, z_grid, np.nan)

    out_dir = (
        PIPELINE_ROOT
        / "data"
        / "products"
        / product
        / "sensitivity"
        / PERCENTILE
        / "zeta_grid"
        / estimator
    )

    out_tif = out_dir / f"zeta_map_{product}_{PERCENTILE}_{estimator}_idw_k8_p2p0.tif"

    save_geotiff(out_tif, z_grid, transform)

    valid = np.isfinite(z_grid)

    print("\n" + "=" * 90)
    print(f"{product} / {estimator}")
    print("=" * 90)
    print("stations:", len(df))
    print("resolution:", res)
    print("grid shape:", z_grid.shape)
    print("valid pixels:", int(valid.sum()))
    print("zeta min:", float(np.nanmin(z_grid)))
    print("zeta mean:", float(np.nanmean(z_grid)))
    print("zeta median:", float(np.nanmedian(z_grid)))
    print("zeta max:", float(np.nanmax(z_grid)))
    print("saved:", out_tif)


def main():
    shapes, bounds = load_boundary_shapes(BOUNDARY_PATH)

    for estimator in ESTIMATORS:
        for product in PRODUCTS:
            interpolate_one(product, estimator, shapes, bounds)


if __name__ == "__main__":
    main()
