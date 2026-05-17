#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpolation.py

Spatial interpolation of station-level zeta correction factors to rainfall
product grids.

Part 06 scope
-------------
This module reads station-level zeta tables from Part 05 and creates gridded
zeta rasters aligned to the annual-maximum rainfall product grid.

Main correction field
---------------------
For the paper-ready default:

    zeta = median(gauge_mm / product_mm)
    IDW k = 10
    IDW power = 2

Distance metric
---------------
Station coordinates are longitude/latitude. To avoid degree-distance distortion,
the IDW nearest-neighbor search uses a spherical representation:

    lon/lat -> unit-sphere xyz

Nearest neighbors are found in 3-D chord distance using scipy.spatial.cKDTree,
then converted approximately to great-circle distance in kilometers. IDW
weights are then:

    w_i = 1 / d_i^power

If a grid cell coincides with a station, the nearest station zeta is used
directly.

Output
------
data/products/<product>/sensitivity/<pXX>/zeta_grid/<estimator>/
    zeta_map_<product>_<pXX>_<estimator>_idw_k10_p2.tif
    zeta_station_points_<product>_<pXX>_<estimator>.csv
    zeta_grid_manifest_<product>_<pXX>_<estimator>.json
    zeta_map_preview_<product>_<pXX>_<estimator>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import product_available_years
from .event_selection import parse_percentile_arg
from .raster_utils import (
    choose_template_raster,
    grid_cell_centers_for_rows,
    profile_summary,
    read_template_mask,
    write_float32_geotiff,
)
from .utils import ensure_dir, now_iso, print_header, print_section, write_json


EARTH_RADIUS_KM = 6371.0088


def _import_ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scipy is required for IDW interpolation in Part 06.\n\n"
            "Install it with:\n"
            "    python3 -m pip install scipy\n"
        ) from exc
    return cKDTree


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return None
    return plt


def _zeta_station_dir(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "zeta_station"
        / estimator
    )


def _zeta_grid_dir(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
) -> Path:
    return (
        Path(cfg.data_root)
        / "products"
        / product_name
        / "sensitivity"
        / percentile_label
        / "zeta_grid"
        / estimator
    )


def default_zeta_station_path(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
) -> Path:
    """Return the default retained station-zeta table path."""
    return (
        _zeta_station_dir(cfg, product_name, percentile_label, estimator)
        / f"zeta_per_station_{product_name}_{percentile_label}_{estimator}.csv"
    )


def load_station_zeta_table(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
    zeta_table: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load retained station zeta table from Part 05.
    """
    if zeta_table is None:
        zeta_table = default_zeta_station_path(cfg, product_name, percentile_label, estimator)

    zeta_table = Path(zeta_table)
    if not zeta_table.exists():
        raise FileNotFoundError(
            f"Station zeta table not found:\n  {zeta_table}\n\n"
            "Run Part 05 first, for example:\n"
            f"  python3 run_pipeline.py compute-zeta --product {product_name} "
            f"--percentile {percentile_label} --estimator {estimator}"
        )

    df = pd.read_csv(zeta_table, low_memory=False)

    required = ["station_id", "latitude", "longitude", "zeta_selected"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Zeta table is missing required columns {missing}: {zeta_table}"
        )

    for col in ["latitude", "longitude", "zeta_selected"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "n_pairs_used" in df.columns:
        df["n_pairs_used"] = pd.to_numeric(df["n_pairs_used"], errors="coerce")
    else:
        df["n_pairs_used"] = np.nan

    valid = (
        np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
        & np.isfinite(df["zeta_selected"])
        & df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    df = df.loc[valid].copy()

    if df.empty:
        raise ValueError(f"No valid station zeta rows found in: {zeta_table}")

    return df


def lonlat_to_unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Convert longitude/latitude in degrees to unit-sphere xyz coordinates.
    """
    lon_rad = np.deg2rad(np.asarray(lon, dtype=float))
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))

    cos_lat = np.cos(lat_rad)

    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack([x, y, z])


def chord_to_great_circle_km(chord: np.ndarray) -> np.ndarray:
    """
    Convert unit-sphere chord distance to great-circle distance in km.
    """
    chord = np.asarray(chord, dtype=float)
    clipped = np.clip(chord / 2.0, 0.0, 1.0)
    angle = 2.0 * np.arcsin(clipped)
    return EARTH_RADIUS_KM * angle


def idw_interpolate_points_to_targets(
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    station_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    k: int = 10,
    power: float = 2.0,
    chunk_size: int = 250000,
    exact_distance_km: float = 1e-9,
) -> np.ndarray:
    """
    Interpolate station values to target lon/lat points using spherical IDW.

    Parameters
    ----------
    station_lon, station_lat:
        Station coordinates in degrees.
    station_values:
        Zeta values at stations.
    target_lon, target_lat:
        Target grid cell centers in degrees.
    k:
        Number of nearest stations.
    power:
        IDW power.
    chunk_size:
        Number of target cells per processing chunk.

    Returns
    -------
    Interpolated values for each target point.
    """
    cKDTree = _import_ckdtree()

    station_lon = np.asarray(station_lon, dtype=float)
    station_lat = np.asarray(station_lat, dtype=float)
    station_values = np.asarray(station_values, dtype=float)

    target_lon = np.asarray(target_lon, dtype=float)
    target_lat = np.asarray(target_lat, dtype=float)

    valid_station = (
        np.isfinite(station_lon)
        & np.isfinite(station_lat)
        & np.isfinite(station_values)
    )

    station_lon = station_lon[valid_station]
    station_lat = station_lat[valid_station]
    station_values = station_values[valid_station]

    if station_values.size == 0:
        raise ValueError("No valid station values available for IDW interpolation.")

    k_eff = min(int(k), int(station_values.size))

    station_xyz = lonlat_to_unit_xyz(station_lon, station_lat)
    tree = cKDTree(station_xyz)

    out = np.full(target_lon.shape, np.nan, dtype=float)

    n = target_lon.size

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)

        target_xyz = lonlat_to_unit_xyz(target_lon[start:end], target_lat[start:end])

        dist_chord, idx = tree.query(target_xyz, k=k_eff)

        if k_eff == 1:
            dist_chord = dist_chord[:, None]
            idx = idx[:, None]

        dist_km = chord_to_great_circle_km(dist_chord)
        vals = station_values[idx]

        chunk_out = np.full(end - start, np.nan, dtype=float)

        exact = dist_km <= exact_distance_km
        has_exact = exact.any(axis=1)

        if has_exact.any():
            exact_rows = np.where(has_exact)[0]
            for r in exact_rows:
                first_exact = np.where(exact[r])[0][0]
                chunk_out[r] = vals[r, first_exact]

        normal_rows = ~has_exact
        if normal_rows.any():
            d = dist_km[normal_rows]
            v = vals[normal_rows]

            weights = 1.0 / np.power(d, float(power))
            weights[~np.isfinite(weights)] = 0.0

            wsum = np.sum(weights, axis=1)
            numerator = np.sum(weights * v, axis=1)

            valid = wsum > 0
            temp = np.full(normal_rows.sum(), np.nan, dtype=float)
            temp[valid] = numerator[valid] / wsum[valid]

            chunk_out[normal_rows] = temp

        out[start:end] = chunk_out

    return out


def create_target_points_from_template(
    profile: Mapping[str, Any],
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create target lon/lat arrays for valid template pixels.

    Returns
    -------
    rows, cols, lon, lat
    """
    rows, cols = np.where(valid_mask)
    lon, lat = grid_cell_centers_for_rows(profile["transform"], rows, cols)
    return rows, cols, lon, lat


def make_preview_png(
    output_png: Path,
    zeta_grid: np.ndarray,
    profile: Mapping[str, Any],
    stations: pd.DataFrame,
    title: str,
) -> Optional[Path]:
    """
    Create a simple preview map of interpolated zeta.

    The publication-quality plots will be refined later in Part 08. This is
    only a technical diagnostic preview.
    """
    plt = _import_matplotlib()
    if plt is None:
        return None

    try:
        from .raster_utils import raster_bounds_from_profile
        bounds = raster_bounds_from_profile(profile)
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            np.where(np.isfinite(zeta_grid), zeta_grid, np.nan),
            extent=extent,
            origin="upper",
        )
        ax.scatter(
            stations["longitude"],
            stations["latitude"],
            s=8,
            edgecolors="black",
            linewidths=0.2,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Zeta")
        fig.tight_layout()
        fig.savefig(output_png, dpi=200)
        plt.close(fig)
        return output_png
    except Exception:
        return None


def interpolate_zeta_for_product_percentile(
    cfg: Any,
    product_name: str,
    percentile: str | float,
    estimator: str = "median",
    zeta_table: Optional[Path] = None,
    template_raster: Optional[Path] = None,
    output_nodata: float = -9999.0,
    chunk_size: int = 250000,
    make_preview: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Interpolate station zeta values to the product annual-maximum grid.
    """
    percentile_label, percentile_value = parse_percentile_arg(percentile)
    estimator = str(estimator).lower()

    product_cfg = cfg.product(product_name)
    product_label = product_cfg.get("label", product_name)

    if verbose:
        print_header(f"Interpolating zeta: {product_name} / {percentile_label} / {estimator}")

    stations = load_station_zeta_table(
        cfg=cfg,
        product_name=product_name,
        percentile_label=percentile_label,
        estimator=estimator,
        zeta_table=zeta_table,
    )

    interp_cfg = cfg.method["interpolation"]
    k = int(interp_cfg["idw_neighbors"])
    power = float(interp_cfg["idw_power"])

    zeta_clip = interp_cfg.get("zeta_clip_before_interpolation", [0.05, 10.0])
    zeta_low, zeta_high = float(zeta_clip[0]), float(zeta_clip[1])

    stations["zeta_for_interpolation_raw"] = stations["zeta_selected"]
    stations["zeta_for_interpolation"] = stations["zeta_selected"].clip(zeta_low, zeta_high)
    stations["zeta_clipped_low_for_interpolation"] = stations["zeta_selected"] < zeta_low
    stations["zeta_clipped_high_for_interpolation"] = stations["zeta_selected"] > zeta_high

    if template_raster is None:
        inventory = product_available_years(cfg, product_name)
        preferred_years = inventory.get("processed_years", [])
        template_raster = choose_template_raster(
            product_cfg["annual_max_folder"],
            preferred_years=preferred_years,
        )

    template_raster = Path(template_raster)

    profile, valid_mask, _template_data = read_template_mask(template_raster)
    rows, cols, target_lon, target_lat = create_target_points_from_template(profile, valid_mask)

    if verbose:
        print_section("Inputs")
        print(f"Station zeta rows:       {len(stations)}")
        print(f"Template raster:         {template_raster}")
        print(f"Template shape:          {profile['height']} x {profile['width']}")
        print(f"Valid interpolation px:  {len(target_lon)}")
        print(f"IDW k:                   {k}")
        print(f"IDW power:               {power}")
        print(f"Zeta clip before IDW:    [{zeta_low}, {zeta_high}]")

    interp_values = idw_interpolate_points_to_targets(
        station_lon=stations["longitude"].to_numpy(dtype=float),
        station_lat=stations["latitude"].to_numpy(dtype=float),
        station_values=stations["zeta_for_interpolation"].to_numpy(dtype=float),
        target_lon=target_lon,
        target_lat=target_lat,
        k=k,
        power=power,
        chunk_size=chunk_size,
    )

    zeta_grid = np.full(valid_mask.shape, np.nan, dtype=float)
    zeta_grid[rows, cols] = interp_values

    out_dir = _zeta_grid_dir(cfg, product_name, percentile_label, estimator)
    ensure_dir(out_dir)

    power_label = str(power).replace(".", "p")
    out_tif = out_dir / f"zeta_map_{product_name}_{percentile_label}_{estimator}_idw_k{k}_p{power_label}.tif"
    out_csv = out_dir / f"zeta_station_points_{product_name}_{percentile_label}_{estimator}.csv"
    out_manifest = out_dir / f"zeta_grid_manifest_{product_name}_{percentile_label}_{estimator}.json"
    out_png = out_dir / f"zeta_map_preview_{product_name}_{percentile_label}_{estimator}.png"

    write_float32_geotiff(
        output_path=out_tif,
        data=zeta_grid,
        template_profile=profile,
        nodata=output_nodata,
    )

    stations.to_csv(out_csv, index=False)

    preview_path = None
    if make_preview:
        preview_path = make_preview_png(
            output_png=out_png,
            zeta_grid=zeta_grid,
            profile=profile,
            stations=stations,
            title=f"{product_label} {percentile_label} {estimator} zeta",
        )

    finite_grid = zeta_grid[np.isfinite(zeta_grid)]

    manifest = {
        "created_at": now_iso(),
        "product": product_name,
        "product_label": product_label,
        "percentile_label": percentile_label,
        "percentile_value": percentile_value,
        "estimator": estimator,
        "zeta_definition": cfg.method["zeta"]["definition"],
        "template_raster": str(template_raster),
        "output_tif": str(out_tif),
        "station_points_csv": str(out_csv),
        "preview_png": None if preview_path is None else str(preview_path),
        "interpolation": {
            "method": "idw",
            "distance_metric": "great_circle_km_from_unit_sphere_chord",
            "idw_neighbors": k,
            "idw_power": power,
            "chunk_size": int(chunk_size),
            "zeta_clip_before_interpolation": [zeta_low, zeta_high],
        },
        "stations": {
            "n_stations": int(len(stations)),
            "n_clipped_low": int(stations["zeta_clipped_low_for_interpolation"].sum()),
            "n_clipped_high": int(stations["zeta_clipped_high_for_interpolation"].sum()),
            "zeta_selected_min": float(stations["zeta_selected"].min()),
            "zeta_selected_median": float(stations["zeta_selected"].median()),
            "zeta_selected_mean": float(stations["zeta_selected"].mean()),
            "zeta_selected_max": float(stations["zeta_selected"].max()),
        },
        "grid": {
            "n_valid_pixels": int(valid_mask.sum()),
            "n_interpolated_pixels": int(np.isfinite(zeta_grid).sum()),
            "zeta_grid_min": None if finite_grid.size == 0 else float(np.nanmin(finite_grid)),
            "zeta_grid_median": None if finite_grid.size == 0 else float(np.nanmedian(finite_grid)),
            "zeta_grid_mean": None if finite_grid.size == 0 else float(np.nanmean(finite_grid)),
            "zeta_grid_max": None if finite_grid.size == 0 else float(np.nanmax(finite_grid)),
            "profile": profile_summary(profile),
        },
    }

    write_json(out_manifest, manifest)

    if verbose:
        print_section("Zeta interpolation outputs")
        print(f"Zeta raster:        {out_tif}")
        print(f"Station points CSV: {out_csv}")
        print(f"Manifest:           {out_manifest}")
        if preview_path is not None:
            print(f"Preview PNG:        {preview_path}")
        if finite_grid.size:
            print_section("Interpolated zeta stats")
            print(f"min:     {np.nanmin(finite_grid):.4f}")
            print(f"median:  {np.nanmedian(finite_grid):.4f}")
            print(f"mean:    {np.nanmean(finite_grid):.4f}")
            print(f"max:     {np.nanmax(finite_grid):.4f}")

    return {
        "zeta_raster": out_tif,
        "station_points": out_csv,
        "manifest": out_manifest,
        "preview_png": preview_path,
    }


def interpolate_zeta_batch(
    cfg: Any,
    products: Sequence[str],
    percentiles: Sequence[str | float],
    estimators: Sequence[str],
    zeta_table: Optional[Path] = None,
    template_raster: Optional[Path] = None,
    output_nodata: float = -9999.0,
    chunk_size: int = 250000,
    make_preview: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Path]]:
    """
    Batch zeta interpolation.

    zeta_table and template_raster overrides are allowed only for a single
    product/percentile/estimator run.
    """
    if zeta_table is not None and (len(products) > 1 or len(percentiles) > 1 or len(estimators) > 1):
        raise ValueError("--zeta-table override is only valid for a single interpolation run.")

    if template_raster is not None and len(products) > 1:
        raise ValueError("--template-raster override is only valid for a single-product run.")

    outputs: List[Dict[str, Path]] = []

    for product_name in products:
        for percentile in percentiles:
            for estimator in estimators:
                out = interpolate_zeta_for_product_percentile(
                    cfg=cfg,
                    product_name=product_name,
                    percentile=percentile,
                    estimator=estimator,
                    zeta_table=zeta_table,
                    template_raster=template_raster,
                    output_nodata=output_nodata,
                    chunk_size=chunk_size,
                    make_preview=make_preview,
                    verbose=verbose,
                )
                outputs.append(out)

    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    interpolate_zeta_for_product_percentile(
        cfg=cfg,
        product_name="imerg_v07",
        percentile="p98",
        estimator="median",
    )


if __name__ == "__main__":
    main()
