#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_utils.py

Raster utilities for the GRIDF rainfall-product bias-correction pipeline.

Part 07 scope
-------------
This module supports both:
    - Part 06: zeta interpolation to product grids
    - Part 07: application of zeta rasters to annual maximum rainfall rasters

Design principles
-----------------
1. Zeta rasters and annual maximum rasters must be aligned before correction.
2. If alignment differs, zeta is reprojected/resampled to the raw raster grid.
3. Raw raster nodata/masks are preserved.
4. Outputs are written as single-band float32 GeoTIFFs with explicit nodata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def _import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rasterio is required for raster operations.\n\n"
            "Install it with:\n"
            "    python3 -m pip install rasterio\n"
        ) from exc
    return rasterio


def list_raster_files(folder: Path, extensions: Sequence[str] = (".tif", ".tiff")) -> List[Path]:
    """List raster files recursively."""
    folder = Path(folder)
    if not folder.exists():
        return []
    exts = {e.lower() for e in extensions}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def extract_year_from_name(path: Path) -> Optional[int]:
    """Extract first 4-digit year from filename."""
    matches = YEAR_RE.findall(path.name)
    if not matches:
        return None
    return int(matches[0])


def raster_year_mapping(folder: Path) -> Dict[int, Path]:
    """
    Map year -> raster path.

    If multiple rasters contain the same year, the first sorted path is used.
    """
    mapping: Dict[int, Path] = {}
    for path in list_raster_files(folder):
        year = extract_year_from_name(path)
        if year is None:
            continue
        mapping.setdefault(year, path)
    return dict(sorted(mapping.items()))


def choose_template_raster(
    annual_max_folder: Path,
    preferred_years: Optional[Sequence[int]] = None,
) -> Path:
    """
    Choose a template raster from the annual-maximum folder.

    The template defines the target grid for zeta interpolation.
    Prefer the first available year in preferred_years; otherwise use the first
    raster found.
    """
    annual_max_folder = Path(annual_max_folder)
    mapping = raster_year_mapping(annual_max_folder)

    if not mapping:
        rasters = list_raster_files(annual_max_folder)
        if not rasters:
            raise FileNotFoundError(f"No raster files found in: {annual_max_folder}")
        return rasters[0]

    if preferred_years:
        for year in preferred_years:
            if int(year) in mapping:
                return mapping[int(year)]

    return mapping[sorted(mapping.keys())[0]]


def read_template_mask(template_raster: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Read template raster metadata and valid mask.

    Returns
    -------
    profile:
        Rasterio profile dictionary.
    valid_mask:
        Boolean array where interpolation should be performed.
    template_data:
        First band as float array with nodata/masked values set to NaN.
    """
    rasterio = _import_rasterio()

    template_raster = Path(template_raster)
    with rasterio.open(template_raster) as src:
        arr_masked = src.read(1, masked=True)
        profile = src.profile.copy()

    data = np.asarray(arr_masked.filled(np.nan), dtype=float)
    valid_mask = ~np.ma.getmaskarray(arr_masked) & np.isfinite(data)

    if not valid_mask.any():
        valid_mask = np.ones(data.shape, dtype=bool)

    return profile, valid_mask, data


def read_raster_masked(path: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Read a single-band raster as float data and valid mask.

    Returns
    -------
    profile:
        Raster profile.
    data:
        Float array with masked/nodata cells as NaN.
    valid_mask:
        True for valid cells.
    """
    rasterio = _import_rasterio()

    path = Path(path)
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        profile = src.profile.copy()

    data = np.asarray(arr.filled(np.nan), dtype=float)
    mask = ~np.ma.getmaskarray(arr) & np.isfinite(data)

    return profile, data, mask


def write_float32_geotiff(
    output_path: Path,
    data: np.ndarray,
    template_profile: Mapping[str, Any],
    nodata: float = -9999.0,
    compress: str = "deflate",
) -> Path:
    """
    Write a single-band float32 GeoTIFF using a template profile.
    """
    rasterio = _import_rasterio()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = dict(template_profile)
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": float(nodata),
        "compress": compress,
        "predictor": 2,
        "BIGTIFF": "IF_SAFER",
    })

    out = np.asarray(data, dtype=np.float32).copy()
    out[~np.isfinite(out)] = nodata

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)

    return output_path


def grid_cell_centers_for_rows(
    transform: Any,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cell-center x/y coordinates for row/col arrays.
    """
    cc = cols.astype(float) + 0.5
    rr = rows.astype(float) + 0.5

    x = transform.a * cc + transform.b * rr + transform.c
    y = transform.d * cc + transform.e * rr + transform.f

    return x, y


def raster_bounds_from_profile(profile: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """Return raster bounds from rasterio profile."""
    rasterio = _import_rasterio()
    from rasterio.transform import array_bounds

    height = int(profile["height"])
    width = int(profile["width"])
    transform = profile["transform"]
    west, south, east, north = array_bounds(height, width, transform)
    return float(west), float(south), float(east), float(north)


def profile_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Small JSON-serializable profile summary."""
    bounds = raster_bounds_from_profile(profile)
    transform = profile["transform"]
    return {
        "crs": str(profile.get("crs")),
        "width": int(profile.get("width")),
        "height": int(profile.get("height")),
        "dtype": str(profile.get("dtype")),
        "nodata": None if profile.get("nodata") is None else float(profile.get("nodata")),
        "transform": [float(v) for v in transform[:6]],
        "bounds": {
            "west": bounds[0],
            "south": bounds[1],
            "east": bounds[2],
            "north": bounds[3],
        },
    }


def profiles_match_grid(a: Mapping[str, Any], b: Mapping[str, Any], tol: float = 1e-9) -> bool:
    """
    Check whether two raster profiles share the same grid.

    This checks CRS, transform, width, and height.
    """
    if str(a.get("crs")) != str(b.get("crs")):
        return False
    if int(a.get("width")) != int(b.get("width")):
        return False
    if int(a.get("height")) != int(b.get("height")):
        return False

    ta = a.get("transform")
    tb = b.get("transform")

    if ta is None or tb is None:
        return False

    return all(abs(float(x) - float(y)) <= tol for x, y in zip(ta[:6], tb[:6]))


def resample_to_match(
    source_path: Path,
    target_profile: Mapping[str, Any],
    resampling: str = "bilinear",
    dst_nodata: float = np.nan,
) -> np.ndarray:
    """
    Reproject/resample source raster to match the target grid.

    Parameters
    ----------
    source_path:
        Source raster path.
    target_profile:
        Target raster profile containing CRS, transform, width, and height.
    resampling:
        One of nearest, bilinear, cubic, average.
    dst_nodata:
        Destination nodata value. Use NaN for in-memory float arrays.

    Returns
    -------
    Float array matching target grid.
    """
    rasterio = _import_rasterio()
    from rasterio.warp import reproject, Resampling

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }

    if resampling not in resampling_map:
        raise ValueError(
            f"Unsupported resampling method '{resampling}'. "
            f"Choose from {list(resampling_map)}."
        )

    source_path = Path(source_path)
    with rasterio.open(source_path) as src:
        src_arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        src_profile = src.profile.copy()

        dst = np.full(
            (int(target_profile["height"]), int(target_profile["width"])),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling_map[resampling],
        )

    return np.asarray(dst, dtype=float)


def read_or_resample_to_match(
    source_path: Path,
    target_profile: Mapping[str, Any],
    resampling: str = "bilinear",
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Read source raster directly if it matches the target grid; otherwise resample.

    Returns
    -------
    data:
        Float array matching target grid.
    was_resampled:
        True if reprojection/resampling was performed.
    source_profile:
        Source raster profile.
    """
    source_profile, source_data, _source_mask = read_raster_masked(source_path)

    if profiles_match_grid(source_profile, target_profile):
        return source_data, False, source_profile

    data = resample_to_match(
        source_path=source_path,
        target_profile=target_profile,
        resampling=resampling,
    )

    return data, True, source_profile
