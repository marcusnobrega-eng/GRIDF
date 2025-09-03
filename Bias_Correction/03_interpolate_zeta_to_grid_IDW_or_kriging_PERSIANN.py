# 03_interpolate_zeta_to_grid_IDW_or_kriging.py
# ---------------------------------------------------------
# Purpose:
#   Interpolate station-level bias factors (zeta) to a regular grid
#   using IDW (fast, default) or Ordinary Kriging (optional, needs pykrige).
#
# Inputs:
#   - zeta_per_station.csv  (must contain: station_id, zeta, lat, lon [, n_pairs])
#   - (optional) Brazil boundary shapefile/geojson to mask the raster
#
# Output:
#   - GeoTIFF bias map (EPSG:4326), e.g., zeta_map_025deg.tif
#
# Reqs:
#   pip install numpy pandas geopandas rasterio shapely scipy
#   (optional for kriging) pip install pykrige
# ---------------------------------------------------------

import os
import warnings
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
import rasterio
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree

# ------------- USER INPUTS -------------
# PERSIANN output from step 02
ZETA_CSV      = r'G:/My Drive/persiann_bias_pairs/zeta_per_station.csv'  # <- switched folder
BOUNDARY_PATH = r'C:/Users/marcu/OneDrive - University of Arizona/Desktop/Desktop_Folder/ANA_Analysis/Rainfall_Distribution/Brasil_Shapefile/brazil_Brazil_Country_Boundary.shp'  # optional; set "" to skip
OUT_TIF       = r'G:/My Drive/persiann_bias_pairs/zeta_map_025deg.tif'   # <- switched name

GRID_RES_DEG  = 0.25     # ~PERSIANN-CDR native resolution (0.25°)
PAD_DEG       = 0.5      # expand bbox a bit beyond boundary

METHOD        = "idw"    # "idw" (default) or "ok" for Ordinary Kriging

# IDW parameters
IDW_POWER     = 2.0
IDW_K         = 8
IDW_CAP_RANGE = (0.25, 10)  # clamp final raster (same as before)

# Kriging parameters (if METHOD="ok")
KRIG_VARIOGRAM = "spherical"   # 'linear','power','gaussian','spherical','exponential'
KRIG_NLAGS     = 12
KRIG_ENABLE_ANISOTROPY = False

# Filters
MIN_PAIRS     = 10
ZETA_CLIP     = (0.05, 10.0)
# ---------------------------------------

# --- PROJ/Rasterio fix (Windows/pip wheels) ---
def _ensure_proj_env():
    try:
        import rasterio
    except Exception:
        return
    cur = os.environ.get("PROJ_LIB", "")
    if cur and os.path.exists(os.path.join(cur, "proj.db")):
        return
    base = os.path.dirname(rasterio.__file__)
    candidates = [
        os.path.join(base, "proj_data"),
        os.path.join(base, "share", "proj"),
        os.path.join(base, "projections", "data"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "proj.db")):
            os.environ["PROJ_LIB"] = c
            return
# ----------------------------------------------

def load_points(zeta_csv: str, min_pairs: int, clip_range):
    df = pd.read_csv(zeta_csv)
    must_have = {"zeta", "lat", "lon"}
    if not must_have.issubset(df.columns):
        raise ValueError(f"{zeta_csv} must contain columns: {must_have}")

    if "n_pairs" in df.columns:
        df = df[df["n_pairs"].fillna(0) >= min_pairs]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["zeta", "lat", "lon"])
    df = df[(df["zeta"] >= clip_range[0]) & (df["zeta"] <= clip_range[1])]
    if df.empty:
        raise ValueError("No valid points after filtering.")
    return df[["lon", "lat", "zeta"]].reset_index(drop=True)

def load_boundary_shapes(path: str):
    """Return (shapes, bounds) in EPSG:4326; or (None, None) if no path."""
    if not path:
        return None, None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import geopandas as gpd
        from shapely.geometry import mapping
    except Exception as e:
        raise RuntimeError(
            "Boundary masking requested but 'geopandas' is not installed. "
            "Either install geopandas or set BOUNDARY_PATH = ''."
        ) from e

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        warnings.warn("Boundary has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    geom = gdf.unary_union
    shapes = [mapping(geom)]
    bounds = tuple(gdf.total_bounds)   # (minx, miny, maxx, maxy)
    return shapes, bounds

def build_grid(bounds, res_deg, pad_deg=0.0):
    minx, miny, maxx, maxy = bounds
    minx -= pad_deg; miny -= pad_deg; maxx += pad_deg; maxy += pad_deg
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
        dists, idxs = tree.query(grid_pts, k=k, workers=-1)  # SciPy >= 1.6
    except TypeError:
        dists, idxs = tree.query(grid_pts, k=k)              # older SciPy
    dists = np.where(dists == 0, 1e-12, dists)
    w = 1.0 / (dists ** power)
    w /= w.sum(axis=1, keepdims=True)
    z_idw = np.sum(w * z[idxs], axis=1)
    return z_idw.reshape(xx.shape)

def kriging_interpolate(lon, lat, z, xs, ys):
    try:
        from pykrige.ok import OrdinaryKriging
    except Exception as e:
        raise RuntimeError("pykrige is required for METHOD='ok'. Install via 'pip install pykrige'.") from e
    ok = OrdinaryKriging(
        lon, lat, z,
        variogram_model=KRIG_VARIOGRAM,
        nlags=KRIG_NLAGS,
        enable_anisotropy=KRIG_ENABLE_ANISOTROPY,
        verbose=False, enable_plotting=False
    )
    z_krig, _ = ok.execute("grid", xs, ys)
    return np.asarray(z_krig)

def save_geotiff(path, array2d, transform, nodata=np.nan, crs="EPSG:4326"):
    _ensure_proj_env()
    from rasterio.errors import CRSError

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
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array2d.astype("float32"), 1)
    except CRSError:
        profile_no_crs = dict(profile); profile_no_crs.pop("crs", None)
        with rasterio.open(path, "w", **profile_no_crs) as dst:
            dst.write(array2d.astype("float32"), 1)
        print("WARNING: PROJ database not found; wrote GeoTIFF without CRS. "
              "Assign EPSG:4326 in your GIS if needed.")

def main():
    # Load points
    df = load_points(ZETA_CSV, MIN_PAIRS, ZETA_CLIP)
    lon = df["lon"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    z   = df["zeta"].to_numpy(dtype=float)

    # Bounds from boundary or points
    shapes, bounds = load_boundary_shapes(BOUNDARY_PATH)
    if shapes is None:
        bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    # else: bounds already from boundary

    xs, ys, xx, yy, transform = build_grid(bounds, GRID_RES_DEG, PAD_DEG)

    # Interpolate
    if METHOD.lower() == "idw":
        z_grid = idw_interpolate(lon, lat, z, xx, yy, k=IDW_K, power=IDW_POWER)
        z_grid = np.clip(z_grid, IDW_CAP_RANGE[0], IDW_CAP_RANGE[1])
    elif METHOD.lower() == "ok":
        z_grid = kriging_interpolate(lon, lat, z, xs, ys)
    else:
        raise ValueError("METHOD must be 'idw' or 'ok'.")

    # Mask to boundary if provided
    if shapes is not None:
        # geometry_mask(..., invert=False) -> True outside shapes
        mask_outside = geometry_mask(shapes, z_grid.shape, transform, False, False)
        z_grid = np.where(~mask_outside, z_grid, np.nan)
        print("pixels inside:", (~mask_outside).sum(), " | outside:", mask_outside.sum())
    else:
        print("No boundary provided; skipping mask.")

    # Save
    os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)
    save_geotiff(OUT_TIF, z_grid, transform)
    print(f"Saved bias map: {OUT_TIF}")
    print(f"Method: {METHOD.upper()} | Grid: {z_grid.shape} | Res: {GRID_RES_DEG}°")

if __name__ == "__main__":
    main()
