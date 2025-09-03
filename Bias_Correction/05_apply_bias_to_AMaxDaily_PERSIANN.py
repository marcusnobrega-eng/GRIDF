# 05_apply_bias_to_AMaxDaily_PERSIANN.py
# ---------------------------------------------------------
# Multiply each annual-maximum PERSIANN GeoTIFF by the bias factor (zeta) map.
# Inputs  : G:\My Drive\GEE\PERSIANN_MaxDaily_0p25deg_YYYY_Brazil.tif (e.g., 1994–2024)
# Zeta    : single GeoTIFF with zeta(x,y) for Brazil (time-invariant, EPSG:4326)
# Output  : G:\My Drive\GEE\PERSIANN_BiasCorrected_AMaxDaily\PERSIANN_MaxDaily_0p25deg_YYYY_Brazil_BC.tif
#
# Requires: rasterio, numpy  (pip install rasterio numpy)
# ---------------------------------------------------------

import os, re, warnings, glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import Affine

# ---------- USER SETTINGS ----------
IN_DIR    = r"G:\My Drive\PERSIANN_Max"
IN_GLOB   = r"PERSIANN_MaxDaily_0p25deg_*_Brazil.tif"   # pattern to match your files
ZETA_TIF  = r"G:\My Drive\persiann_bias_pairs\zeta_map_025deg.tif"  # your PERSIANN ζ raster
OUT_DIR   = r"G:\My Drive\PERSIANN_Max\PERSIANN_BiasCorrected_AMaxDaily"

YEAR_MIN, YEAR_MAX = 1994, 2024
CAP_ZETA  = (0.25, 5)           # set to None to disable, or adjust range
RESAMPLE  = Resampling.bilinear  # resampling for zeta onto PERSIANN grid
COMPRESS  = "deflate"           # GeoTIFF compression
ZLEVEL    = 6
MAX_RAIN_MM = 500.0             # hard cap for corrected values (mm/day)
# -----------------------------------

# --- PROJ/Rasterio helper for Windows/pip wheels ---
def _ensure_proj_env():
    import os as _os
    try:
        import rasterio as _rio
    except Exception:
        return
    cur = _os.environ.get("PROJ_LIB", "")
    if cur and os.path.exists(os.path.join(cur, "proj.db")):
        return
    base = os.path.dirname(_rio.__file__)
    for c in [os.path.join(base, "proj_data"),
              os.path.join(base, "share", "proj"),
              os.path.join(base, "projections", "data")]:
        if os.path.exists(os.path.join(c, "proj.db")):
            _os.environ["PROJ_LIB"] = c
            return
# ---------------------------------------------------

def list_inputs():
    """
    Return sorted list of (path, year) matching IN_GLOB and YEAR_MIN/MAX.
    Extracts the first 4-digit year found in each filename.
    """
    files = glob.glob(os.path.join(IN_DIR, IN_GLOB))
    out = []
    for p in files:
        m = re.search(r"(\d{4})", os.path.basename(p))
        if m:
            y = int(m.group(1))
            if YEAR_MIN <= y <= YEAR_MAX:
                out.append((p, y))
    return sorted(out, key=lambda t: t[1])

def read_zeta():
    ds = rasterio.open(ZETA_TIF)
    zeta = ds.read(1, masked=True).astype("float32")
    if ds.nodata is not None:
        zeta = np.ma.masked_equal(zeta, ds.nodata)
    zeta = np.ma.filled(zeta, np.nan)
    if CAP_ZETA:
        zeta = np.clip(zeta, CAP_ZETA[0], CAP_ZETA[1])
    return ds, zeta

def reproject_to_match(src_arr, src_ds, dst_profile):
    """Reproject src_arr (from src_ds) into the destination grid described by dst_profile."""
    dst_arr = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=RESAMPLE,
        dst_nodata=np.nan,
    )
    return dst_arr

def _mul16_le(n, cap=512):
    """Largest multiple of 16 <= min(n, cap); at least 16."""
    m = min(int(n), int(cap))
    m = (m // 16) * 16
    return 16 if m == 0 else m

def save_geotiff(path, array2d, profile, nodata=np.nan):
    prof = profile.copy()
    prof.update({
        "count": 1,
        "dtype": "float32",
        "compress": COMPRESS,
        "predictor": 2 if COMPRESS in ("deflate", "lzw") else 1,
        "zlevel": ZLEVEL if COMPRESS == "deflate" else None,
        "nodata": nodata,
    })

    # --- Try tiled with safe tile sizes (multiples of 16) ---
    bx = _mul16_le(prof["width"], 512)
    by = _mul16_le(prof["height"], 512)
    try:
        prof_tiled = prof.copy()
        prof_tiled.update({"tiled": True, "blockxsize": bx, "blockysize": by})
        with rasterio.open(path, "w", **prof_tiled) as dst:
            dst.write(array2d.astype("float32"), 1)
        return
    except Exception:
        # --- Fallback: striped GeoTIFF ---
        prof_striped = prof.copy()
        prof_striped.update({"tiled": False})
        with rasterio.open(path, "w", **prof_striped) as dst:
            dst.write(array2d.astype("float32"), 1)
        return

def main():
    _ensure_proj_env()
    os.makedirs(OUT_DIR, exist_ok=True)

    inputs = list_inputs()
    if not inputs:
        raise SystemExit(f"No input files found in {IN_DIR} matching {IN_GLOB}")

    print(f"Found {len(inputs)} annual maps in {IN_DIR}")
    zeta_ds, zeta_src = read_zeta()
    print(f"Loaded zeta map: {ZETA_TIF}")

    for in_path, yr in inputs:
        base = os.path.basename(in_path)
        out_path = os.path.join(OUT_DIR, base.replace(".tif", "_BC.tif"))

        if os.path.exists(out_path):
            print(f"[{yr}] exists, skipping: {out_path}")
            continue

        with rasterio.open(in_path) as din:
            arr = din.read(1, masked=True).astype("float32")
            nod = din.nodata
            if nod is not None:
                arr = np.ma.masked_equal(arr, nod)
            arr = np.ma.filled(arr, np.nan)

            # Build destination profile for reprojection of zeta
            dst_profile = {
                "crs": din.crs,
                "transform": din.transform,
                "width": din.width,
                "height": din.height,
            }

            # Reproject zeta to match this PERSIANN map
            zeta_match = reproject_to_match(zeta_src, zeta_ds, dst_profile)

            # Multiply, respecting NaNs
            corrected = arr * zeta_match
            corrected = np.where(corrected > MAX_RAIN_MM, MAX_RAIN_MM, corrected)

            # keep NaN where original is NaN or zeta is NaN
            mask = ~np.isfinite(arr) | ~np.isfinite(zeta_match)
            corrected[mask] = np.nan

            # Save
            out_profile = din.profile
            out_profile.update(dtype="float32", nodata=np.nan,
                               compress=COMPRESS, predictor=2, zlevel=ZLEVEL)
            save_geotiff(out_path, corrected, out_profile)

        print(f"[{yr}] wrote: {out_path}")

    zeta_ds.close()
    print("All done.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
