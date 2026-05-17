#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py

Convenience wrapper to generate the current set of diagnostics and sensitivity
figures for paper development.

This script does not replace manual figure refinement. It calls the Part 08
diagnostic functions and stores outputs in figures/diagnostics and
figures/sensitivity.
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

from biascorr.config import DEFAULT_PIPELINE_ROOT, init_folders, load_config
from biascorr.diagnostics import (
    compare_mean_median_sensitivity,
    compare_percentile_sensitivity,
    diagnostics_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-support diagnostics/figures.")
    parser.add_argument("--pipeline-root", type=Path, default=PROJECT_ROOT if PROJECT_ROOT.name == "Bias_Correction_Pipeline" else DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--product", required=True)
    parser.add_argument("--percentile", default="p98")
    parser.add_argument("--estimator", default="median")
    parser.add_argument("--skip-percentile-sensitivity", action="store_true")
    parser.add_argument("--skip-mean-median", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.pipeline_root)
    init_folders(cfg)

    diagnostics_batch(
        cfg=cfg,
        products=[args.product],
        percentiles=[args.percentile],
        estimators=[args.estimator],
        make_figures=True,
        verbose=True,
    )

    if not args.skip_percentile_sensitivity:
        compare_percentile_sensitivity(
            cfg=cfg,
            product=args.product,
            estimator=args.estimator,
            reference_percentile="p98",
            compare_rasters=True,
            make_figures=True,
            verbose=True,
        )

    if not args.skip_mean_median:
        compare_mean_median_sensitivity(
            cfg=cfg,
            product=args.product,
            percentile=args.percentile,
            compare_rasters=True,
            make_figures=True,
            verbose=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
