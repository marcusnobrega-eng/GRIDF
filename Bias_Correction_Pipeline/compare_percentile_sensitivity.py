#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone wrapper for percentile zeta sensitivity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from biascorr.config import DEFAULT_PIPELINE_ROOT, init_folders, load_config
from biascorr.diagnostics import compare_percentile_sensitivity


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare percentile sensitivity of zeta correction factors.")
    parser.add_argument("--pipeline-root", type=Path, default=PROJECT_ROOT if PROJECT_ROOT.name == "Bias_Correction_Pipeline" else DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--product", required=True)
    parser.add_argument("--estimator", default="median")
    parser.add_argument("--reference-percentile", default="p98")
    parser.add_argument("--no-raster-compare", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.pipeline_root)
    init_folders(cfg)

    compare_percentile_sensitivity(
        cfg=cfg,
        product=args.product,
        estimator=args.estimator,
        reference_percentile=args.reference_percentile,
        compare_rasters=not args.no_raster_compare,
        make_figures=not args.no_figures,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
