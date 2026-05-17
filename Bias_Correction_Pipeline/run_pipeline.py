#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

Main command-line runner for the GRIDF rainfall-product bias-correction pipeline.

Part 08 implements the full pipeline interface:
    - show-config
    - check-paths
    - init-folders
    - list-products
    - inventory-years
    - write-manifest
    - prepare-gauges
    - select-events
    - inspect-gee
    - export-pairs
    - compute-zeta
    - interpolate-zeta
    - apply-bias
    - diagnostics
    - percentile-sensitivity
    - mean-median-sensitivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from biascorr.config import (  # noqa: E402
    DEFAULT_PIPELINE_ROOT,
    all_product_year_inventory,
    init_folders,
    load_config,
    print_config_summary,
    validate_input_paths,
    write_inventory,
    write_run_manifest,
)
from biascorr.gauges import prepare_gauges  # noqa: E402
from biascorr.event_selection import parse_percentile_arg, select_events_batch  # noqa: E402
from biascorr.utils import print_header  # noqa: E402


def add_product_percentile_estimator_args(parser, require_product=True, require_percentile=True):
    product_group = parser.add_mutually_exclusive_group(required=require_product)
    product_group.add_argument("--product", type=str)
    product_group.add_argument("--all-products", action="store_true")

    pct_group = parser.add_mutually_exclusive_group(required=require_percentile)
    pct_group.add_argument("--percentile", type=str)
    pct_group.add_argument("--all-percentiles", action="store_true")

    est_group = parser.add_mutually_exclusive_group(required=False)
    est_group.add_argument("--estimator", type=str, default=None)
    est_group.add_argument("--all-estimators", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRIDF rainfall-product bias-correction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=PROJECT_ROOT if PROJECT_ROOT.name == "Bias_Correction_Pipeline" else DEFAULT_PIPELINE_ROOT,
    )

    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("show-config")
    sub.add_parser("check-paths")
    sub.add_parser("init-folders")
    sub.add_parser("list-products")
    sub.add_parser("inventory-years")
    sub.add_parser("write-manifest")

    p_gauge = sub.add_parser("prepare-gauges")
    p_gauge.add_argument("--no-overwrite", action="store_true")

    p_events = sub.add_parser("select-events")
    prod_group = p_events.add_mutually_exclusive_group(required=True)
    prod_group.add_argument("--product", type=str)
    prod_group.add_argument("--all-products", action="store_true")
    pct_group = p_events.add_mutually_exclusive_group(required=True)
    pct_group.add_argument("--percentile", type=str)
    pct_group.add_argument("--all-percentiles", action="store_true")
    p_events.add_argument("--start-year", type=int, default=None)
    p_events.add_argument("--end-year", type=int, default=None)
    p_events.add_argument("--station-limit", type=int, default=None)
    p_events.add_argument("--no-overwrite", action="store_true")
    p_events.add_argument("--no-yearly-files", action="store_true")

    p_gee = sub.add_parser("inspect-gee")
    gee_group = p_gee.add_mutually_exclusive_group(required=True)
    gee_group.add_argument("--product", type=str)
    gee_group.add_argument("--all-products", action="store_true")
    p_gee.add_argument("--gee-project", type=str, default="ee-marcusep2025")
    p_gee.add_argument("--date", type=str, default=None)
    p_gee.add_argument("--lon", type=float, default=-47.8825)
    p_gee.add_argument("--lat", type=float, default=-15.7942)
    p_gee.add_argument("--no-write", action="store_true")

    p_pairs = sub.add_parser("export-pairs")
    prod_group = p_pairs.add_mutually_exclusive_group(required=True)
    prod_group.add_argument("--product", type=str)
    prod_group.add_argument("--all-products", action="store_true")
    pct_group = p_pairs.add_mutually_exclusive_group(required=True)
    pct_group.add_argument("--percentile", type=str)
    pct_group.add_argument("--all-percentiles", action="store_true")
    p_pairs.add_argument("--start-year", type=int, default=None)
    p_pairs.add_argument("--end-year", type=int, default=None)
    p_pairs.add_argument("--gee-project", type=str, default="ee-marcusep2025")
    p_pairs.add_argument("--drive-folder", type=str, default=None)
    p_pairs.add_argument("--drive-folder-prefix", type=str, default=None)
    p_pairs.add_argument("--max-features-per-export", type=int, default=3000)
    p_pairs.add_argument("--dry-run", action="store_true")

    p_zeta = sub.add_parser("compute-zeta")
    add_product_percentile_estimator_args(p_zeta)
    p_zeta.add_argument("--pairs-folder", type=Path, default=None)
    p_zeta.add_argument("--start-year", type=int, default=None)
    p_zeta.add_argument("--end-year", type=int, default=None)
    p_zeta.add_argument("--no-qc-pairs", action="store_true")

    p_interp = sub.add_parser("interpolate-zeta")
    add_product_percentile_estimator_args(p_interp)
    p_interp.add_argument("--zeta-table", type=Path, default=None)
    p_interp.add_argument("--template-raster", type=Path, default=None)
    p_interp.add_argument("--chunk-size", type=int, default=250000)
    p_interp.add_argument("--output-nodata", type=float, default=-9999.0)
    p_interp.add_argument("--no-preview", action="store_true")

    p_apply = sub.add_parser("apply-bias")
    add_product_percentile_estimator_args(p_apply)
    p_apply.add_argument("--zeta-raster", type=Path, default=None)
    p_apply.add_argument("--start-year", type=int, default=None)
    p_apply.add_argument("--end-year", type=int, default=None)
    p_apply.add_argument("--resampling", type=str, default="bilinear", choices=["nearest", "bilinear", "cubic", "average"])
    p_apply.add_argument("--output-nodata", type=float, default=-9999.0)
    p_apply.add_argument("--no-overwrite", action="store_true")

    p_diag = sub.add_parser("diagnostics")
    add_product_percentile_estimator_args(p_diag)
    p_diag.add_argument("--no-figures", action="store_true")

    p_ps = sub.add_parser("percentile-sensitivity")
    p_ps.add_argument("--product", required=True)
    p_ps.add_argument("--estimator", default="median")
    p_ps.add_argument("--reference-percentile", default="p98")
    p_ps.add_argument("--no-raster-compare", action="store_true")
    p_ps.add_argument("--no-figures", action="store_true")

    p_mm = sub.add_parser("mean-median-sensitivity")
    p_mm.add_argument("--product", required=True)
    p_mm.add_argument("--percentile", default="p98")
    p_mm.add_argument("--no-raster-compare", action="store_true")
    p_mm.add_argument("--no-figures", action="store_true")

    return parser


def _resolve_products(args, cfg):
    return cfg.product_names if getattr(args, "all_products", False) else [args.product]


def _resolve_percentiles(args, cfg):
    if getattr(args, "all_percentiles", False):
        return cfg.method["event_selection"]["percentile_labels"]
    p_label, _ = parse_percentile_arg(args.percentile)
    return [p_label]


def _resolve_estimators(args, cfg):
    if getattr(args, "all_estimators", False):
        return cfg.method["zeta"]["save_estimators"]
    if getattr(args, "estimator", None) is not None:
        return [args.estimator]
    return [cfg.method["zeta"]["main_estimator"]]


def print_products(cfg) -> None:
    print_header("Configured rainfall products")
    for name in cfg.product_names:
        p = cfg.product(name)
        print(f"{name:14s} {p['start_year']}-{p['end_year']}  {p['label']}")
        print(f"{'':14s} GEE:  {p['gee_collection']}")
        print(f"{'':14s} band: {p['gee_band']}")
        print(f"{'':14s} max:  {p['annual_max_folder']}")
        print()


def print_inventory(cfg) -> None:
    print_header("Annual maximum raster year inventory")
    inventory = all_product_year_inventory(cfg)
    for product_name, inv in inventory.items():
        print(f"\n{product_name}")
        print(f"  configured: {inv['configured_start_year']}–{inv['configured_end_year']}")
        print(f"  annual max folder: {inv['annual_max_folder']}")
        print(f"  available years ({inv['n_available_years']}): {inv['available_years']}")
        print(f"  processed years ({inv['n_processed_years']}): {inv['processed_years']}")
        print(f"  missing configured years: {inv['missing_configured_years'] if inv['missing_configured_years'] else 'none'}")
    out = write_inventory(cfg)
    print(f"\nInventory written to:\n  {out}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "show-config"

    cfg = load_config(args.pipeline_root)

    if command == "init-folders":
        init_folders(cfg)
        print_header("Folder initialization complete")
        print(f"Pipeline root:\n  {cfg.pipeline_root}")
        return 0

    init_folders(cfg)

    if command == "show-config":
        print_config_summary(cfg)
        return 0
    if command == "check-paths":
        validate_input_paths(cfg, verbose=True)
        return 0
    if command == "list-products":
        print_products(cfg)
        return 0
    if command == "inventory-years":
        print_inventory(cfg)
        return 0
    if command == "write-manifest":
        out = write_run_manifest(cfg, "manual")
        print_header("Manifest written")
        print(out)
        return 0
    if command == "prepare-gauges":
        prepare_gauges(cfg, overwrite=not args.no_overwrite, verbose=True)
        return 0
    if command == "select-events":
        outputs = select_events_batch(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            percentiles=_resolve_percentiles(args, cfg),
            start_year=args.start_year,
            end_year=args.end_year,
            station_limit=args.station_limit,
            overwrite=not args.no_overwrite,
            write_yearly_files=not args.no_yearly_files,
            verbose=True,
        )
        print_header("Event-selection batch complete")
        for out in outputs:
            print(f"events:  {out['events_all_years']}")
            print(f"summary: {out['summary']}")
            print()
        return 0
    if command == "inspect-gee":
        from biascorr.gee_products import inspect_products
        inspect_products(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            gee_project=args.gee_project,
            sample_date=args.date,
            sample_lon=args.lon,
            sample_lat=args.lat,
            write_output=not args.no_write,
            verbose=True,
        )
        return 0
    if command == "export-pairs":
        from biascorr.gee_pair_exports import export_pairs_batch
        outputs = export_pairs_batch(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            percentiles=_resolve_percentiles(args, cfg),
            start_year=args.start_year,
            end_year=args.end_year,
            gee_project=args.gee_project,
            drive_folder=args.drive_folder,
            drive_folder_prefix=args.drive_folder_prefix,
            max_features_per_export=args.max_features_per_export,
            dry_run=args.dry_run,
            verbose=True,
        )
        print_header("GEE pair-export batch complete")
        for out in outputs:
            print(f"manifest:     {out['manifest']}")
            print(f"drive folder: {out['drive_folder']}")
            print(f"local folder: {out['pairs_dir']}")
            print(f"tasks:        {len(out['tasks'])}")
            print()
        return 0
    if command == "compute-zeta":
        from biascorr.zeta import compute_zeta_batch
        outputs = compute_zeta_batch(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            percentiles=_resolve_percentiles(args, cfg),
            estimators=_resolve_estimators(args, cfg),
            pairs_folder=args.pairs_folder,
            start_year=args.start_year,
            end_year=args.end_year,
            write_qc_pairs=not args.no_qc_pairs,
            verbose=True,
        )
        print_header("Station-zeta batch complete")
        for out in outputs:
            print(f"retained: {out['station_retained']}")
            print(f"all:      {out['station_all']}")
            print(f"manifest: {out['manifest']}")
            print()
        return 0
    if command == "interpolate-zeta":
        from biascorr.interpolation import interpolate_zeta_batch
        outputs = interpolate_zeta_batch(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            percentiles=_resolve_percentiles(args, cfg),
            estimators=_resolve_estimators(args, cfg),
            zeta_table=args.zeta_table,
            template_raster=args.template_raster,
            output_nodata=args.output_nodata,
            chunk_size=args.chunk_size,
            make_preview=not args.no_preview,
            verbose=True,
        )
        print_header("Zeta interpolation batch complete")
        for out in outputs:
            print(f"zeta raster: {out['zeta_raster']}")
            print(f"manifest:    {out['manifest']}")
            if out.get("preview_png"):
                print(f"preview:     {out['preview_png']}")
            print()
        return 0
    if command == "apply-bias":
        from biascorr.apply_bias import apply_bias_batch
        outputs = apply_bias_batch(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            percentiles=_resolve_percentiles(args, cfg),
            estimators=_resolve_estimators(args, cfg),
            zeta_raster=args.zeta_raster,
            start_year=args.start_year,
            end_year=args.end_year,
            resampling=args.resampling,
            output_nodata=args.output_nodata,
            overwrite=not args.no_overwrite,
            verbose=True,
        )
        print_header("Bias-application batch complete")
        for out in outputs:
            print(f"summary:  {out['summary']}")
            print(f"manifest: {out['manifest']}")
            print(f"output:   {out['output_dir']}")
            print()
        return 0
    if command == "diagnostics":
        from biascorr.diagnostics import diagnostics_batch
        outputs = diagnostics_batch(
            cfg=cfg,
            products=_resolve_products(args, cfg),
            percentiles=_resolve_percentiles(args, cfg),
            estimators=_resolve_estimators(args, cfg),
            make_figures=not args.no_figures,
            verbose=True,
        )
        print_header("Diagnostics batch complete")
        for out in outputs:
            print(f"summary: {out['summary_json']}")
            print(f"metrics: {out['metrics_csv']}")
            print()
        return 0
    if command == "percentile-sensitivity":
        from biascorr.diagnostics import compare_percentile_sensitivity
        out = compare_percentile_sensitivity(
            cfg=cfg,
            product=args.product,
            estimator=args.estimator,
            reference_percentile=args.reference_percentile,
            compare_rasters=not args.no_raster_compare,
            make_figures=not args.no_figures,
            verbose=True,
        )
        print_header("Percentile sensitivity complete")
        print(out)
        return 0
    if command == "mean-median-sensitivity":
        from biascorr.diagnostics import compare_mean_median_sensitivity
        out = compare_mean_median_sensitivity(
            cfg=cfg,
            product=args.product,
            percentile=args.percentile,
            compare_rasters=not args.no_raster_compare,
            make_figures=not args.no_figures,
            verbose=True,
        )
        print_header("Mean-vs-median sensitivity complete")
        print(out)
        return 0

    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
