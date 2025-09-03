# 05_apply_bias_to_AMaxDaily_XAVIER_local.py
# ---------------------------------------------------------
# Multiply each annual-maximum XAVIER (BR-DWGD) GeoTIFF by the bias factor (zeta) map.
# Inputs  : C:\Users\marcu\OneDrive\Documentos\GitHub\Rainfall_Temp_Distributor\Data\BR_DWGD_prmax_YYYY(.tif/.tiff)
# Zeta    : single GeoTIFF with zeta(x,y) for Brazil (time-invariant; Xavier-derived)
# Output  : ...\Data\BiasCorrected\BR_DWGD_prmax_YYYY_BC.tif
#
# Requires: rasterio, numpy  (pip install rasterio numpy)
# ---------------------------------------------------------

import os, re, warnings
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# ---------- USER SETTINGS ----------
IN_DIR    = r"C:\Users\marcu\OneDrive\Documentos\GitHub\Rainfall_Temp_Distributor\Data\Xavier"
IN_REGEX  = r"^BR_DWGD_prmax_(\d{4})(?:\.(?:tif|tiff))?$"   # matches BR_DWGD_prmax_1995.tif/.tiff
ZETA_TIF  = r"G:\My Drive\xavier_bias_pairs\zeta_map_010deg.tif"  # <- path to your Xavier ζ raster
OUT_DIR   = os.path.join(IN_DIR, "BiasCorrected")

YEAR_MIN, YEAR_MAX = 1990, 2024   # clip years discovered from filenames to this window
CAP_ZETA  = (0.25, 5.0)           # set to None to disable, or adjust range
RESAMPLE  = Resampling.bilinear   # resampling for zeta onto XAVIER grid
COMPRESS  = "deflate"             # GeoTIFF compression
ZLEVEL    = 6
MAX_RAIN_MM = 500.0               # hard cap for corrected values (mm/day)
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
    """Return sorted list of (path, year) matching IN_REGEX and year range."""
    out = []
    for f in os.listdir(IN_DIR):
        m = re.match(IN_REGEX, f, flags=re.IGNORECASE)
        if m:
            y = int(m.group(1))
            if YEAR_MIN <= y <= YEAR_MAX:
                out.append((os.path.join(IN_DIR, f), y))
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
        # --- Fallback: striped GeoTIFF (always valid) ---
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
        raise SystemExit(f"No input files found in {IN_DIR} matching regex {IN_REGEX}")

    print(f"Found {len(inputs)} annual maps in {IN_DIR}")
    zeta_ds, zeta_src = read_zeta()
    print(f"Loaded zeta map: {ZETA_TIF}")

    for in_path, yr in inputs:
        base = os.path.basename(in_path)
        name, ext = os.path.splitext(base)
        ext = ext if ext else ".tif"  # default if extensionless
        out_path = os.path.join(OUT_DIR, f"{name}_BC{ext}")

        if os.path.exists(out_path):
            print(f"[{yr}] exists, skipping: {out_path}")
            continue

        with rasterio.open(in_path) as din:
            arr = din.read(1, masked=True).astype("float32")
            nod = din.nodata
            if nod is not None:
                arr = np.ma.masked_equal(arr, nod)
            arr = np.ma.filled(arr, np.nan)

            # Destination profile for zeta reprojection
            dst_profile = {
                "crs": din.crs,
                "transform": din.transform,
                "width": din.width,
                "height": din.height,
            }

            # Reproject zeta to match this raster
            zeta_match = reproject_to_match(zeta_src, zeta_ds, dst_profile)

            # Multiply and cap
            corrected = arr * zeta_match
            corrected = np.where(corrected > MAX_RAIN_MM, MAX_RAIN_MM, corrected)

            # keep NaN where original or zeta are NaN
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
