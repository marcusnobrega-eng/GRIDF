#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gee_products.py

Google Earth Engine rainfall-product definitions and inspection utilities for
the GRIDF bias-correction pipeline.

Part 03 scope
-------------
This module does NOT export bias pairs yet. It provides the product-specific
GEE logic that later stages will reuse.

Main responsibilities
---------------------
1. Initialize Earth Engine safely.
2. Build the Brazil geometry from FAO/GAUL.
3. Inspect GEE collections:
   - available bands
   - first/last image date
   - image count over configured period
   - nominal scale/projection when available
4. Auto-detect precipitation bands when config uses "AUTO_DETECT".
5. Construct one-day rainfall images using product-specific logic:
   - CHIRPS: daily product, mm/day.
   - PERSIANN-CDR: daily product, mm/day.
   - BR-DWGD / Xavier: daily product, band/scaling must be inspected.
   - IMERG V06: half-hourly precipitationCal treated as mm/hour;
                daily total = mean daily rate * 24.
   - IMERG V07: Climate Engine daily product, already daily total mm/day.
6. Sample one point/date for sanity checking.

Scientific consistency
----------------------
The daily rainfall image returned by this module is the image that Part 04 will
sample at station/date pairs to compute:

    ratio = gauge_mm / product_mm

Therefore, the daily aggregation definitions here must match the annual maximum
raster generation logic used for the final corrected rainfall products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils import ensure_dir, now_iso, print_header, print_section, timestamp, write_json


# Earth Engine is imported lazily so other pipeline commands can run even if
# earthengine-api is not installed yet.
def _import_ee():
    try:
        import ee  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The earthengine-api package is required for Part 03 and later.\n\n"
            "Install it with:\n"
            "    python3 -m pip install earthengine-api\n\n"
            "Then authenticate if needed:\n"
            "    earthengine authenticate\n"
        ) from exc
    return ee


def initialize_earth_engine(project: Optional[str] = "ee-marcusep2025") -> Any:
    """
    Initialize Earth Engine.

    Parameters
    ----------
    project:
        GEE project ID. Use None to initialize without specifying a project.

    Returns
    -------
    ee module
    """
    ee = _import_ee()

    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Earth Engine.\n\n"
            "Try running:\n"
            "    earthengine authenticate\n\n"
            "If you use a specific project, confirm it is available:\n"
            f"    project = {project}\n\n"
            f"Original error: {exc}"
        ) from exc

    return ee


def brazil_geometry(ee: Any) -> Any:
    """
    Return Brazil geometry using the same FAO/GAUL source used in the annual
    maximum export scripts.
    """
    brazil_fc = (
        ee.FeatureCollection("FAO/GAUL/2015/level0")
        .filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
    )
    return brazil_fc.geometry()


def _ee_date_string(ee: Any, millis: Any) -> Optional[str]:
    """Convert an Earth Engine time_start value to YYYY-MM-dd string."""
    if millis is None:
        return None
    try:
        return ee.Date(millis).format("YYYY-MM-dd").getInfo()
    except Exception:
        return None


def collection_basic_info(
    ee: Any,
    collection_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    geometry: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Inspect a GEE ImageCollection.

    This function makes small getInfo calls and is intended for pre-production
    checks, not large-scale processing.
    """
    ic = ee.ImageCollection(collection_id)

    if start_date and end_date:
        ic_filtered = ic.filterDate(start_date, end_date)
    else:
        ic_filtered = ic

    if geometry is not None:
        ic_filtered = ic_filtered.filterBounds(geometry)

    first_all = ee.Image(ic.sort("system:time_start").first())
    last_all = ee.Image(ic.sort("system:time_start", False).first())

    first_filtered = ee.Image(ic_filtered.sort("system:time_start").first())
    last_filtered = ee.Image(ic_filtered.sort("system:time_start", False).first())

    first_all_time = first_all.get("system:time_start").getInfo()
    last_all_time = last_all.get("system:time_start").getInfo()
    first_filtered_time = first_filtered.get("system:time_start").getInfo()
    last_filtered_time = last_filtered.get("system:time_start").getInfo()

    try:
        band_names = first_all.bandNames().getInfo()
    except Exception:
        band_names = []

    try:
        projection_info = first_all.projection().getInfo()
    except Exception:
        projection_info = None

    try:
        nominal_scale = first_all.projection().nominalScale().getInfo()
    except Exception:
        nominal_scale = None

    try:
        count_filtered = ic_filtered.size().getInfo()
    except Exception:
        count_filtered = None

    return {
        "collection_id": collection_id,
        "band_names_first_image": band_names,
        "first_date_collection": _ee_date_string(ee, first_all_time),
        "last_date_collection": _ee_date_string(ee, last_all_time),
        "first_date_filtered": _ee_date_string(ee, first_filtered_time),
        "last_date_filtered": _ee_date_string(ee, last_filtered_time),
        "filtered_start_date": start_date,
        "filtered_end_date": end_date,
        "filtered_count": count_filtered,
        "projection_info_first_image": projection_info,
        "nominal_scale_m_first_image": nominal_scale,
    }


def choose_precip_band(
    available_bands: Sequence[str],
    configured_band: Optional[str] = None,
) -> str:
    """
    Choose the precipitation band.

    If configured_band is not AUTO_DETECT, it must exist in the available bands.
    If AUTO_DETECT, try common precipitation names and then precipitation-like
    substrings.
    """
    bands = list(available_bands)

    if configured_band and str(configured_band).upper() != "AUTO_DETECT":
        if configured_band not in bands:
            raise ValueError(
                f"Configured band '{configured_band}' not found. "
                f"Available bands: {bands}"
            )
        return str(configured_band)

    priority = [
        "precipitation",
        "Precipitation",
        "precip",
        "precipitationCal",
        "precipitation_cal",
        "pr",
        "PR",
        "b1",
        "constant",
    ]

    for candidate in priority:
        if candidate in bands:
            return candidate

    precip_like = [
        b for b in bands
        if any(token in b.lower() for token in ["precip", "rain", "prcp", "ppt"])
    ]

    if len(precip_like) == 1:
        return precip_like[0]

    if len(precip_like) > 1:
        raise ValueError(
            "Multiple precipitation-like bands found. Please set gee_band in "
            f"config/products.yml. Candidates: {precip_like}. All bands: {bands}"
        )

    if len(bands) == 1:
        # For datasets such as BR-DWGD this may be acceptable, but we still
        # report it explicitly in the inspection file.
        return bands[0]

    raise ValueError(
        "Could not auto-detect precipitation band. "
        f"Available bands: {bands}. Set gee_band in config/products.yml."
    )


def _empty_daily_image(ee: Any, name: str = "precipitation") -> Any:
    """Return a fully masked empty daily image."""
    return (
        ee.Image.constant(0.0)
        .rename(name)
        .updateMask(ee.Image.constant(0))
    )


def _apply_optional_scale_offset(image: Any, product_cfg: Mapping[str, Any]) -> Any:
    """
    Apply optional scale_factor and offset if explicitly present in config.

    No scaling is applied unless the config says so. This is deliberate because
    BR-DWGD/Xavier scaling must be verified by inspection before production.
    """
    scale_factor = product_cfg.get("scale_factor", None)
    offset = product_cfg.get("offset", None)

    out = image

    if offset is not None:
        out = out.subtract(float(offset))

    if scale_factor is not None:
        out = out.multiply(float(scale_factor))

    return out


def get_collection_for_product(ee: Any, product_cfg: Mapping[str, Any]) -> Any:
    """Return ImageCollection for a product config."""
    return ee.ImageCollection(product_cfg["gee_collection"])


def get_daily_precip_image(
    ee: Any,
    product_name: str,
    product_cfg: Mapping[str, Any],
    date: str,
    precip_band: Optional[str] = None,
    geometry: Optional[Any] = None,
) -> Any:
    """
    Build one daily rainfall image in mm/day.

    Parameters
    ----------
    ee:
        Earth Engine module.
    product_name:
        Product key from products.yml.
    product_cfg:
        Product configuration dictionary.
    date:
        Date string YYYY-MM-DD.
    precip_band:
        Precipitation band. If None, uses configured band or auto-detects.
    geometry:
        Optional geometry for clipping.

    Returns
    -------
    ee.Image named "precipitation" with daily rainfall in mm/day.
    """
    date_ee = ee.Date(date)
    next_date = date_ee.advance(1, "day")

    ic = get_collection_for_product(ee, product_cfg)

    if precip_band is None:
        info = collection_basic_info(
            ee,
            product_cfg["gee_collection"],
            start_date=date,
            end_date=ee.Date(next_date).format("YYYY-MM-dd").getInfo(),
            geometry=geometry,
        )
        precip_band = choose_precip_band(info["band_names_first_image"], product_cfg.get("gee_band"))

    product_name = product_name.lower()
    aggregation = str(product_cfg.get("daily_aggregation", "")).lower()

    # ------------------------------------------------------------------
    # IMERG V06: native half-hourly rate in mm/hour.
    # Daily total = mean daily rate * 24.
    # ------------------------------------------------------------------
    if product_name == "imerg_v06" or aggregation == "mean_half_hourly_rate_mm_hour_times_24":
        day_ic = (
            ic
            .filterDate(date_ee, next_date)
            .select(precip_band)
        )

        count = day_ic.size()

        daily = (
            ee.Image(
                ee.Algorithms.If(
                    count.gt(0),
                    day_ic.mean().multiply(24.0).rename("precipitation"),
                    _empty_daily_image(ee, "precipitation"),
                )
            )
            .set({
                "system:time_start": date_ee.millis(),
                "date": date_ee.format("YYYY-MM-dd"),
                "product_key": product_name,
                "source_collection": product_cfg["gee_collection"],
                "source_band": precip_band,
                "units": "mm/day",
                "daily_aggregation": "mean half-hourly mm/hour rate * 24",
                "image_count": count,
            })
        )

    # ------------------------------------------------------------------
    # Daily products already in mm/day:
    # CHIRPS, PERSIANN-CDR, IMERG V07 Climate Engine daily, and BR-DWGD
    # once band/scaling is confirmed.
    # ------------------------------------------------------------------
    else:
        day_ic = (
            ic
            .filterDate(date_ee, next_date)
            .select(precip_band)
        )

        count = day_ic.size()

        first_image = ee.Image(day_ic.first())
        first_image = _apply_optional_scale_offset(first_image, product_cfg)

        daily = (
            ee.Image(
                ee.Algorithms.If(
                    count.gt(0),
                    first_image.rename("precipitation"),
                    _empty_daily_image(ee, "precipitation"),
                )
            )
            .set({
                "system:time_start": date_ee.millis(),
                "date": date_ee.format("YYYY-MM-dd"),
                "product_key": product_name,
                "source_collection": product_cfg["gee_collection"],
                "source_band": precip_band,
                "units": "mm/day",
                "daily_aggregation": product_cfg.get("daily_aggregation", "daily_total_direct_mm_day"),
                "image_count": count,
            })
        )

    if geometry is not None:
        daily = daily.clip(geometry)

    return daily


def sample_daily_image_at_point(
    ee: Any,
    image: Any,
    lon: float,
    lat: float,
    scale_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Sample the daily rainfall image at one point.

    Returns a dictionary with product_mm if available.
    """
    point = ee.Geometry.Point([float(lon), float(lat)])

    if scale_m is None:
        try:
            scale_m = image.projection().nominalScale().getInfo()
        except Exception:
            scale_m = 10000.0

    try:
        result = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=float(scale_m),
            bestEffort=True,
            maxPixels=1e8,
        ).getInfo()
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "lon": lon,
            "lat": lat,
            "scale_m": scale_m,
            "product_mm": None,
        }

    value = result.get("precipitation", None) if isinstance(result, dict) else None

    return {
        "success": True,
        "lon": lon,
        "lat": lat,
        "scale_m": scale_m,
        "raw_reduce_region": result,
        "product_mm": value,
    }


def inspect_product(
    cfg: Any,
    product_name: str,
    gee_project: Optional[str] = "ee-marcusep2025",
    sample_date: Optional[str] = None,
    sample_lon: float = -47.8825,
    sample_lat: float = -15.7942,
    write_output: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Inspect one configured GEE rainfall product.

    Default sample point is near Brasília, Brazil.
    """
    ee = initialize_earth_engine(project=gee_project)
    geom = brazil_geometry(ee)

    product_cfg = cfg.product(product_name)

    start_year = int(product_cfg["start_year"])
    end_year = int(product_cfg["end_year"])

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year + 1}-01-01"

    if sample_date is None:
        sample_date = f"{start_year}-01-15"

    if verbose:
        print_header(f"GEE product inspection: {product_name}")
        print(f"Collection: {product_cfg['gee_collection']}")
        print(f"Configured years: {start_year}–{end_year}")
        print(f"Sample date: {sample_date}")
        print(f"Sample point: lon={sample_lon}, lat={sample_lat}")

    basic = collection_basic_info(
        ee=ee,
        collection_id=product_cfg["gee_collection"],
        start_date=start_date,
        end_date=end_date,
        geometry=geom,
    )

    precip_band = choose_precip_band(
        basic["band_names_first_image"],
        configured_band=product_cfg.get("gee_band"),
    )

    daily = get_daily_precip_image(
        ee=ee,
        product_name=product_name,
        product_cfg=product_cfg,
        date=sample_date,
        precip_band=precip_band,
        geometry=geom,
    )

    try:
        daily_count = daily.get("image_count").getInfo()
    except Exception:
        daily_count = None

    try:
        daily_props = daily.toDictionary().getInfo()
    except Exception:
        daily_props = {}

    # Estimate sampling scale from configured native resolution if possible.
    native_res = product_cfg.get("native_resolution_deg", None)
    if native_res is not None:
        sample_scale = float(native_res) * 111_320.0
    else:
        sample_scale = None

    sample = sample_daily_image_at_point(
        ee=ee,
        image=daily,
        lon=sample_lon,
        lat=sample_lat,
        scale_m=sample_scale,
    )

    result = {
        "created_at": now_iso(),
        "gee_project": gee_project,
        "product_key": product_name,
        "product_config": dict(product_cfg),
        "configured_start_date": start_date,
        "configured_end_date_exclusive": end_date,
        "collection_info": basic,
        "selected_precip_band": precip_band,
        "daily_image_properties": daily_props,
        "daily_image_count_for_sample_date": daily_count,
        "sample": sample,
        "interpretation": {
            "daily_units": "mm/day",
            "ratio_later": "gauge_mm / product_mm",
            "notes": (
                "This inspection confirms the image construction that Part 04 "
                "will use for station/date bias-pair sampling."
            ),
        },
    }

    if write_output:
        out_dir = Path(cfg.metadata_root) / "data_inventory"
        ensure_dir(out_dir)
        out_path = out_dir / f"gee_product_inspection_{product_name}_{timestamp()}.json"
        write_json(out_path, result)
        result["output_json"] = str(out_path)

    if verbose:
        print_section("Collection")
        print(f"Available bands:       {basic['band_names_first_image']}")
        print(f"Selected precip band:  {precip_band}")
        print(f"First date collection: {basic['first_date_collection']}")
        print(f"Last date collection:  {basic['last_date_collection']}")
        print(f"First date filtered:   {basic['first_date_filtered']}")
        print(f"Last date filtered:    {basic['last_date_filtered']}")
        print(f"Filtered count:        {basic['filtered_count']}")
        print(f"Nominal scale:         {basic['nominal_scale_m_first_image']}")

        print_section("Daily sample")
        print(f"Daily image count:     {daily_count}")
        print(f"Sample success:        {sample.get('success')}")
        print(f"Product mm/day:        {sample.get('product_mm')}")
        print(f"Scale m:               {sample.get('scale_m')}")

        if write_output:
            print_section("Inspection output")
            print(result["output_json"])

    return result


def inspect_products(
    cfg: Any,
    products: Sequence[str],
    gee_project: Optional[str] = "ee-marcusep2025",
    sample_date: Optional[str] = None,
    sample_lon: float = -47.8825,
    sample_lat: float = -15.7942,
    write_output: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Inspect multiple products."""
    outputs: List[Dict[str, Any]] = []
    for product_name in products:
        outputs.append(
            inspect_product(
                cfg=cfg,
                product_name=product_name,
                gee_project=gee_project,
                sample_date=sample_date,
                sample_lon=sample_lon,
                sample_lat=sample_lat,
                write_output=write_output,
                verbose=verbose,
            )
        )
    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    inspect_product(cfg, product_name="imerg_v07")


if __name__ == "__main__":
    main()
