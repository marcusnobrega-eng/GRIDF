#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_bias.py

Apply gridded zeta correction factors to annual maximum rainfall rasters.

Part 07 scope
-------------
This module performs the final raster correction:

    corrected_annual_max(x, y, year) = raw_annual_max(x, y, year) * zeta(x, y)

where:

    zeta = gauge rainfall / product rainfall

Input
-----
1. Annual maximum rainfall rasters:
    /Users/mngomes/Documents/GitHub/GRIDF/Annual_Maximum_Precipitation/<product_folder>/

2. Gridded zeta raster from Part 06:
    data/products/<product>/sensitivity/<pXX>/zeta_grid/<estimator>/
        zeta_map_<product>_<pXX>_<estimator>_idw_k10_p2p0.tif

Output
------
data/products/<product>/sensitivity/<pXX>/annual_max_corrected/<estimator>/
    corrected_<product>_<pXX>_<estimator>_<year>.tif
    annual_max_correction_summary_<product>_<pXX>_<estimator>.csv
    apply_bias_manifest_<product>_<pXX>_<estimator>.json

Theory and safeguards
---------------------
- Zeta is clipped before application using method.yml:
      zeta_clip_before_application: [0.25, 5.0]
- The raw annual-maximum raster grid is preserved.
- If zeta and raw grid do not match exactly, zeta is resampled to the raw grid.
- Raw nodata/masked pixels remain nodata in the corrected raster.
- Output units remain mm/day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import product_available_years
from .event_selection import parse_percentile_arg
from .raster_utils import (
    profile_summary,
    raster_year_mapping,
    read_or_resample_to_match,
    read_raster_masked,
    write_float32_geotiff,
)
from .utils import ensure_dir, now_iso, print_header, print_section, write_json


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


def _corrected_dir(
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
        / "annual_max_corrected"
        / estimator
    )


def find_default_zeta_raster(
    cfg: Any,
    product_name: str,
    percentile_label: str,
    estimator: str,
) -> Path:
    """
    Find the zeta raster created by Part 06.

    Uses glob matching to be robust to power-label formatting.
    """
    folder = _zeta_grid_dir(cfg, product_name, percentile_label, estimator)

    candidates = sorted(folder.glob(f"zeta_map_{product_name}_{percentile_label}_{estimator}_*.tif"))
    if not candidates:
        candidates = sorted(folder.glob("zeta_map_*.tif"))

    if not candidates:
        raise FileNotFoundError(
            f"No zeta raster found in:\n  {folder}\n\n"
            "Run Part 06 first, for example:\n"
            f"  python3 run_pipeline.py interpolate-zeta --product {product_name} "
            f"--percentile {percentile_label} --estimator {estimator}"
        )

    if len(candidates) > 1:
        # Prefer IDW k10 p2 outputs if present.
        preferred = [
            p for p in candidates
            if "idw" in p.name.lower() and "k10" in p.name.lower()
        ]
        if preferred:
            return preferred[-1]

    return candidates[-1]


def _finite_stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute finite statistics for an array."""
    if mask is not None:
        vals = arr[mask]
    else:
        vals = arr.ravel()

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "n": 0,
            "min": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }

    return {
        "n": int(vals.size),
        "min": float(np.nanmin(vals)),
        "p10": float(np.nanpercentile(vals, 10)),
        "median": float(np.nanmedian(vals)),
        "mean": float(np.nanmean(vals)),
        "p90": float(np.nanpercentile(vals, 90)),
        "max": float(np.nanmax(vals)),
    }


def apply_zeta_to_one_raster(
    raw_raster: Path,
    zeta_raster: Path,
    output_raster: Path,
    zeta_clip: Sequence[float],
    resampling: str = "bilinear",
    output_nodata: float = -9999.0,
) -> Dict[str, Any]:
    """
    Apply zeta correction to one annual maximum raster.

    Returns a dictionary of per-raster statistics.
    """
    raw_profile, raw, raw_valid = read_raster_masked(raw_raster)

    zeta, zeta_was_resampled, zeta_profile = read_or_resample_to_match(
        source_path=zeta_raster,
        target_profile=raw_profile,
        resampling=resampling,
    )

    zeta_low, zeta_high = float(zeta_clip[0]), float(zeta_clip[1])

    zeta_finite = np.isfinite(zeta)
    zeta_clipped = np.full(zeta.shape, np.nan, dtype=float)
    zeta_clipped[zeta_finite] = np.clip(zeta[zeta_finite], zeta_low, zeta_high)

    correction_mask = raw_valid & np.isfinite(raw) & np.isfinite(zeta_clipped)

    corrected = np.full(raw.shape, np.nan, dtype=float)
    corrected[correction_mask] = raw[correction_mask] * zeta_clipped[correction_mask]

    write_float32_geotiff(
        output_path=output_raster,
        data=corrected,
        template_profile=raw_profile,
        nodata=output_nodata,
    )

    raw_stats = _finite_stats(raw, raw_valid)
    zeta_stats = _finite_stats(zeta_clipped, correction_mask)
    corrected_stats = _finite_stats(corrected, correction_mask)

    zeta_unc = zeta[np.isfinite(zeta)]
    n_zeta_clipped_low = int(np.sum(zeta_unc < zeta_low))
    n_zeta_clipped_high = int(np.sum(zeta_unc > zeta_high))

    stats = {
        "raw_raster": str(raw_raster),
        "zeta_raster": str(zeta_raster),
        "output_raster": str(output_raster),
        "zeta_was_resampled": bool(zeta_was_resampled),
        "resampling": resampling,
        "n_raw_valid_pixels": int(raw_valid.sum()),
        "n_pixels_corrected": int(correction_mask.sum()),
        "n_zeta_finite_pixels": int(zeta_finite.sum()),
        "n_zeta_clipped_low_pixels": n_zeta_clipped_low,
        "n_zeta_clipped_high_pixels": n_zeta_clipped_high,
        "zeta_clip_low": zeta_low,
        "zeta_clip_high": zeta_high,
        "raw_stats": raw_stats,
        "zeta_stats_on_corrected_pixels": zeta_stats,
        "corrected_stats": corrected_stats,
        "raw_profile": profile_summary(raw_profile),
        "zeta_source_profile": profile_summary(zeta_profile),
    }

    return stats


def corrected_filename(
    product_name: str,
    percentile_label: str,
    estimator: str,
    year: int,
) -> str:
    """Standard corrected annual maximum filename."""
    return f"corrected_{product_name}_{percentile_label}_{estimator}_{int(year)}.tif"


def apply_bias_for_product_percentile(
    cfg: Any,
    product_name: str,
    percentile: str | float,
    estimator: str = "median",
    zeta_raster: Optional[Path] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    resampling: str = "bilinear",
    output_nodata: float = -9999.0,
    overwrite: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Apply zeta correction for one product/percentile/estimator.
    """
    percentile_label, percentile_value = parse_percentile_arg(percentile)
    estimator = str(estimator).lower()

    product_cfg = cfg.product(product_name)
    product_label = product_cfg.get("label", product_name)
    annual_folder = Path(product_cfg["annual_max_folder"])

    if zeta_raster is None:
        zeta_raster = find_default_zeta_raster(
            cfg=cfg,
            product_name=product_name,
            percentile_label=percentile_label,
            estimator=estimator,
        )
    else:
        zeta_raster = Path(zeta_raster)

    inventory = product_available_years(cfg, product_name)
    years = list(inventory["processed_years"])

    if start_year is not None:
        years = [y for y in years if y >= int(start_year)]
    if end_year is not None:
        years = [y for y in years if y <= int(end_year)]

    if not years:
        raise ValueError(
            f"No years available for correction: product={product_name}, "
            f"start_year={start_year}, end_year={end_year}"
        )

    raw_mapping = raster_year_mapping(annual_folder)
    missing_years = [y for y in years if y not in raw_mapping]

    if missing_years:
        raise FileNotFoundError(
            f"Missing annual maximum rasters for years {missing_years} in:\n"
            f"  {annual_folder}"
        )

    out_dir = _corrected_dir(cfg, product_name, percentile_label, estimator)
    ensure_dir(out_dir)

    zeta_clip = cfg.method["application"]["zeta_clip_before_application"]

    if verbose:
        print_header(f"Applying zeta correction: {product_name} / {percentile_label} / {estimator}")
        print(f"Product label:      {product_label}")
        print(f"Annual max folder:  {annual_folder}")
        print(f"Zeta raster:        {zeta_raster}")
        print(f"Output folder:      {out_dir}")
        print(f"Years:              {years[0]}–{years[-1]} ({len(years)} years)")
        print(f"Correction:         corrected = raw_product * zeta")
        print(f"Zeta clip:          {zeta_clip}")
        print(f"Resampling:         {resampling}")

    rows = []

    for year in years:
        raw_raster = raw_mapping[int(year)]
        output_raster = out_dir / corrected_filename(product_name, percentile_label, estimator, int(year))

        if output_raster.exists() and not overwrite:
            if verbose:
                print(f"[skip] {year}: {output_raster}")
            continue

        stats = apply_zeta_to_one_raster(
            raw_raster=raw_raster,
            zeta_raster=zeta_raster,
            output_raster=output_raster,
            zeta_clip=zeta_clip,
            resampling=resampling,
            output_nodata=output_nodata,
        )

        row = {
            "product": product_name,
            "product_label": product_label,
            "percentile_label": percentile_label,
            "percentile_value": percentile_value,
            "estimator": estimator,
            "year": int(year),
            "raw_raster": str(raw_raster),
            "zeta_raster": str(zeta_raster),
            "corrected_raster": str(output_raster),
            "zeta_was_resampled": stats["zeta_was_resampled"],
            "n_raw_valid_pixels": stats["n_raw_valid_pixels"],
            "n_pixels_corrected": stats["n_pixels_corrected"],
            "n_zeta_finite_pixels": stats["n_zeta_finite_pixels"],
            "n_zeta_clipped_low_pixels": stats["n_zeta_clipped_low_pixels"],
            "n_zeta_clipped_high_pixels": stats["n_zeta_clipped_high_pixels"],
            "raw_min": stats["raw_stats"]["min"],
            "raw_p10": stats["raw_stats"]["p10"],
            "raw_median": stats["raw_stats"]["median"],
            "raw_mean": stats["raw_stats"]["mean"],
            "raw_p90": stats["raw_stats"]["p90"],
            "raw_max": stats["raw_stats"]["max"],
            "zeta_min": stats["zeta_stats_on_corrected_pixels"]["min"],
            "zeta_median": stats["zeta_stats_on_corrected_pixels"]["median"],
            "zeta_mean": stats["zeta_stats_on_corrected_pixels"]["mean"],
            "zeta_max": stats["zeta_stats_on_corrected_pixels"]["max"],
            "corrected_min": stats["corrected_stats"]["min"],
            "corrected_p10": stats["corrected_stats"]["p10"],
            "corrected_median": stats["corrected_stats"]["median"],
            "corrected_mean": stats["corrected_stats"]["mean"],
            "corrected_p90": stats["corrected_stats"]["p90"],
            "corrected_max": stats["corrected_stats"]["max"],
        }
        rows.append(row)

        if verbose:
            print(
                f"[{year}] corrected -> {output_raster.name} | "
                f"raw mean={row['raw_mean']:.3f}, "
                f"zeta mean={row['zeta_mean']:.3f}, "
                f"corrected mean={row['corrected_mean']:.3f}"
            )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / f"annual_max_correction_summary_{product_name}_{percentile_label}_{estimator}.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "created_at": now_iso(),
        "product": product_name,
        "product_label": product_label,
        "percentile_label": percentile_label,
        "percentile_value": percentile_value,
        "estimator": estimator,
        "zeta_definition": cfg.method["zeta"]["definition"],
        "correction_equation": cfg.method["application"]["correction_equation"],
        "annual_max_folder": str(annual_folder),
        "zeta_raster": str(zeta_raster),
        "output_folder": str(out_dir),
        "years_requested": years,
        "n_years_requested": len(years),
        "n_years_written": int(len(summary)),
        "resampling": resampling,
        "output_nodata": output_nodata,
        "zeta_clip_before_application": zeta_clip,
        "summary_csv": str(summary_path),
        "corrected_rasters": summary["corrected_raster"].tolist() if not summary.empty else [],
    }

    manifest_path = out_dir / f"apply_bias_manifest_{product_name}_{percentile_label}_{estimator}.json"
    write_json(manifest_path, manifest)

    if verbose:
        print_section("Bias-application outputs")
        print(f"Summary CSV: {summary_path}")
        print(f"Manifest:    {manifest_path}")
        print(f"Output dir:  {out_dir}")

    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "output_dir": out_dir,
    }


def apply_bias_batch(
    cfg: Any,
    products: Sequence[str],
    percentiles: Sequence[str | float],
    estimators: Sequence[str],
    zeta_raster: Optional[Path] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    resampling: str = "bilinear",
    output_nodata: float = -9999.0,
    overwrite: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Path]]:
    """
    Batch application of gridded zeta correction.

    zeta_raster override is allowed only for a single product/percentile/
    estimator run.
    """
    if zeta_raster is not None and (len(products) > 1 or len(percentiles) > 1 or len(estimators) > 1):
        raise ValueError("--zeta-raster override is only valid for a single apply-bias run.")

    outputs: List[Dict[str, Path]] = []

    for product_name in products:
        for percentile in percentiles:
            for estimator in estimators:
                out = apply_bias_for_product_percentile(
                    cfg=cfg,
                    product_name=product_name,
                    percentile=percentile,
                    estimator=estimator,
                    zeta_raster=zeta_raster,
                    start_year=start_year,
                    end_year=end_year,
                    resampling=resampling,
                    output_nodata=output_nodata,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                outputs.append(out)

    return outputs


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    apply_bias_for_product_percentile(
        cfg=cfg,
        product_name="imerg_v07",
        percentile="p98",
        estimator="median",
    )


if __name__ == "__main__":
    main()
