#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_gee_products.py

Standalone helper for inspecting Google Earth Engine rainfall products before
running large bias-pair exports.

Examples
--------
Inspect one product:

    python3 inspect_gee_products.py --product imerg_v07

Inspect all products:

    python3 inspect_gee_products.py --all-products

Use a specific sample date and point:

    python3 inspect_gee_products.py --product chirps --date 2020-01-15 --lon -47.8825 --lat -15.7942

Use a different GEE project:

    python3 inspect_gee_products.py --product imerg_v07 --gee-project ee-marcusep2025
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


from biascorr.config import DEFAULT_PIPELINE_ROOT, init_folders, load_config  # noqa: E402
from biascorr.gee_products import inspect_products  # noqa: E402
from biascorr.utils import print_header  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect GEE rainfall products for the GRIDF bias-correction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=PROJECT_ROOT if PROJECT_ROOT.name == "Bias_Correction_Pipeline" else DEFAULT_PIPELINE_ROOT,
        help="Pipeline root.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--product", type=str, help="Product key, e.g. imerg_v07.")
    group.add_argument("--all-products", action="store_true", help="Inspect all configured products.")

    parser.add_argument("--gee-project", type=str, default="ee-marcusep2025", help="Earth Engine project ID.")
    parser.add_argument("--date", type=str, default=None, help="Sample date YYYY-MM-DD. Default uses product start year Jan 15.")
    parser.add_argument("--lon", type=float, default=-47.8825, help="Sample longitude.")
    parser.add_argument("--lat", type=float, default=-15.7942, help="Sample latitude.")
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON inspection files.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.pipeline_root)
    init_folders(cfg)

    if args.all_products:
        products = cfg.product_names
    else:
        products = [args.product]

    print_header("Running GEE product inspection")
    print(f"Products:    {products}")
    print(f"GEE project: {args.gee_project}")
    print(f"Date:        {args.date}")
    print(f"Point:       lon={args.lon}, lat={args.lat}")

    inspect_products(
        cfg=cfg,
        products=products,
        gee_project=args.gee_project,
        sample_date=args.date,
        sample_lon=args.lon,
        sample_lat=args.lat,
        write_output=not args.no_write,
        verbose=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
