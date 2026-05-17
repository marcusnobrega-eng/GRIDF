#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_outputs.py

Lightweight output checker for the GRIDF bias-correction pipeline.

This script checks whether expected local outputs exist for a product,
percentile, and estimator.

It is intentionally simple; the detailed diagnostics come in Part 08.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from biascorr.config import DEFAULT_PIPELINE_ROOT, init_folders, load_config, product_available_years
from biascorr.event_selection import parse_percentile_arg
from biascorr.apply_bias import find_default_zeta_raster
from biascorr.raster_utils import raster_year_mapping
from biascorr.utils import print_header


def exists_text(path: Path) -> str:
    return "OK" if Path(path).exists() else "MISSING"


def check_one(cfg, product: str, percentile: str, estimator: str) -> int:
    p_label, _ = parse_percentile_arg(percentile)

    base = Path(cfg.data_root) / "products" / product / "sensitivity" / p_label

    events = base / "events" / f"events_{product}_{p_label}_all_years.csv"
    pairs = base / "pairs"
    zeta = base / "zeta_station" / estimator / f"zeta_per_station_{product}_{p_label}_{estimator}.csv"

    try:
        zeta_raster = find_default_zeta_raster(cfg, product, p_label, estimator)
    except Exception:
        zeta_raster = base / "zeta_grid" / estimator / "NO_ZETA_RASTER_FOUND.tif"

    corrected_dir = base / "annual_max_corrected" / estimator
    corrected = raster_year_mapping(corrected_dir)

    inv = product_available_years(cfg, product)
    expected_years = inv["processed_years"]

    print_header(f"Output check: {product} / {p_label} / {estimator}")
    print(f"Events:       {exists_text(events):8s} {events}")
    print(f"Pairs folder: {exists_text(pairs):8s} {pairs}")
    print(f"Zeta station: {exists_text(zeta):8s} {zeta}")
    print(f"Zeta raster:  {exists_text(zeta_raster):8s} {zeta_raster}")
    print(f"Corrected dir:{exists_text(corrected_dir):8s} {corrected_dir}")
    print()
    print(f"Expected corrected years: {expected_years}")
    print(f"Available corrected years: {sorted(corrected.keys())}")
    missing = [y for y in expected_years if y not in corrected]
    print(f"Missing corrected years: {missing if missing else 'none'}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GRIDF bias-correction outputs.")
    parser.add_argument("--pipeline-root", type=Path, default=PROJECT_ROOT if PROJECT_ROOT.name == "Bias_Correction_Pipeline" else DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--product", required=True)
    parser.add_argument("--percentile", default="p98")
    parser.add_argument("--estimator", default="median")
    args = parser.parse_args()

    cfg = load_config(args.pipeline_root)
    init_folders(cfg)

    return check_one(cfg, args.product, args.percentile, args.estimator)


if __name__ == "__main__":
    raise SystemExit(main())
