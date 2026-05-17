#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Configuration loader and path manager for the GRIDF bias-correction pipeline.

Part 01 scope:
    - read paths.yml, products.yml, and method.yml
    - validate key input paths
    - create expected output folders
    - scan available annual-maximum raster years
    - write a small run manifest
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .utils import (
    ensure_dir,
    ensure_dirs,
    now_iso,
    path_status,
    print_header,
    print_section,
    restrict_years_to_available,
    scan_years_from_filenames,
    write_json,
)


try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for this pipeline. Install it with:\n\n"
        "    pip install pyyaml\n"
    ) from exc


DEFAULT_PIPELINE_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline")


@dataclass
class PipelineConfig:
    """Container for loaded pipeline configuration."""

    pipeline_root: Path
    paths: Dict[str, Any]
    products: Dict[str, Any]
    method: Dict[str, Any]

    @property
    def config_dir(self) -> Path:
        return self.pipeline_root / "config"

    @property
    def data_root(self) -> Path:
        return Path(self.paths["outputs"]["data_root"])

    @property
    def figures_root(self) -> Path:
        return Path(self.paths["outputs"]["figures_root"])

    @property
    def metadata_root(self) -> Path:
        return Path(self.paths["outputs"]["metadata_root"])

    @property
    def logs_root(self) -> Path:
        return Path(self.paths["outputs"]["logs_root"])

    @property
    def product_names(self) -> List[str]:
        return sorted(self.products["products"].keys())

    def product(self, product_name: str) -> Dict[str, Any]:
        """Return product config by key."""
        if product_name not in self.products["products"]:
            raise KeyError(
                f"Unknown product '{product_name}'. "
                f"Available: {', '.join(self.product_names)}"
            )
        return self.products["products"][product_name]


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary: {path}")
    return data


def load_config(pipeline_root: Path | str = DEFAULT_PIPELINE_ROOT) -> PipelineConfig:
    """Load the three central YAML files."""
    pipeline_root = Path(pipeline_root).expanduser().resolve()
    config_dir = pipeline_root / "config"

    paths = read_yaml(config_dir / "paths.yml")
    products = read_yaml(config_dir / "products.yml")
    method = read_yaml(config_dir / "method.yml")

    # Normalize pipeline_root from paths.yml if present, but keep command-line root authoritative.
    paths.setdefault("project", {})
    paths["project"]["pipeline_root"] = str(pipeline_root)

    return PipelineConfig(
        pipeline_root=pipeline_root,
        paths=paths,
        products=products,
        method=method,
    )


def expected_base_folders(cfg: PipelineConfig) -> List[Path]:
    """Return base folders that should exist."""
    folders = [
        cfg.pipeline_root,
        cfg.config_dir,
        cfg.data_root,
        cfg.figures_root,
        cfg.metadata_root,
        cfg.logs_root,
        cfg.data_root / "gauges",
        cfg.data_root / "gauges" / "raw",
        cfg.data_root / "gauges" / "processed",
        cfg.data_root / "gauges" / "qc",
        cfg.data_root / "gauges" / "events",
        cfg.data_root / "boundaries",
        cfg.data_root / "static",
        cfg.figures_root / "diagnostics",
        cfg.figures_root / "sensitivity",
        cfg.figures_root / "paper",
        cfg.metadata_root / "manifests",
        cfg.metadata_root / "gee_tasks",
        cfg.metadata_root / "run_summaries",
        cfg.metadata_root / "data_inventory",
    ]
    return folders


def expected_product_folders(cfg: PipelineConfig) -> List[Path]:
    """Return product-specific folders that should exist."""
    folders: List[Path] = []
    percentile_labels = cfg.method["event_selection"]["percentile_labels"]
    estimators = cfg.method["zeta"]["save_estimators"]

    for product_name in cfg.product_names:
        product_root = cfg.data_root / "products" / product_name
        folders.extend([
            product_root,
            product_root / "annual_max_raw",
            product_root / "annual_max_corrected",
            product_root / "gee_exports",
            product_root / "manifests",
            product_root / "logs",
        ])

        for p_label in percentile_labels:
            p_root = product_root / "sensitivity" / p_label
            folders.extend([
                p_root,
                p_root / "events",
                p_root / "pairs",
                p_root / "zeta_station",
                p_root / "zeta_grid",
                p_root / "annual_max_corrected",
                p_root / "diagnostics",
                p_root / "tables",
            ])
            for estimator in estimators:
                folders.extend([
                    p_root / "zeta_station" / estimator,
                    p_root / "zeta_grid" / estimator,
                    p_root / "annual_max_corrected" / estimator,
                    p_root / "diagnostics" / estimator,
                ])

    return folders


def init_folders(cfg: PipelineConfig) -> None:
    """Create expected output folders."""
    ensure_dirs(expected_base_folders(cfg))
    ensure_dirs(expected_product_folders(cfg))


def validate_input_paths(cfg: PipelineConfig, verbose: bool = True) -> Dict[str, bool]:
    """Validate key input paths from paths.yml and products.yml."""
    results: Dict[str, bool] = {}

    input_paths = {
        "gauge_timeseries_csv": Path(cfg.paths["inputs"]["gauge_timeseries_csv"]),
        "station_inventory_csv": Path(cfg.paths["inputs"]["station_inventory_csv"]),
        "annual_max_root": Path(cfg.paths["inputs"]["annual_max_root"]),
        "brazil_shapefile": Path(cfg.paths["inputs"]["brazil_shapefile"]),
        "dem": Path(cfg.paths["inputs"]["dem"]),
    }

    if verbose:
        print_section("Input path check")

    for key, path in input_paths.items():
        exists = path.exists()
        results[key] = exists
        if verbose:
            print(f"{key:24s}: {path_status(path)}")

    if verbose:
        print_section("Annual maximum product folder check")

    for product_name in cfg.product_names:
        p = cfg.product(product_name)
        folder = Path(p["annual_max_folder"])
        exists = folder.exists()
        results[f"{product_name}.annual_max_folder"] = exists
        if verbose:
            print(f"{product_name:24s}: {path_status(folder)}")

    return results


def product_available_years(cfg: PipelineConfig, product_name: str) -> Dict[str, Any]:
    """
    Return configured and available annual-maximum raster years for one product.
    """
    p = cfg.product(product_name)
    folder = Path(p["annual_max_folder"])

    configured_start = int(p["start_year"])
    configured_end = int(p["end_year"])

    available = scan_years_from_filenames(folder)
    processed = restrict_years_to_available(configured_start, configured_end, available)

    return {
        "product": product_name,
        "label": p.get("label", product_name),
        "annual_max_folder": str(folder),
        "configured_start_year": configured_start,
        "configured_end_year": configured_end,
        "available_years": available,
        "processed_years": processed,
        "n_available_years": len(available),
        "n_processed_years": len(processed),
        "missing_configured_years": [
            y for y in range(configured_start, configured_end + 1)
            if y not in set(available)
        ],
    }


def all_product_year_inventory(cfg: PipelineConfig) -> Dict[str, Any]:
    """Return annual-raster year inventory for all products."""
    return {
        product_name: product_available_years(cfg, product_name)
        for product_name in cfg.product_names
    }


def print_config_summary(cfg: PipelineConfig) -> None:
    """Print a human-readable configuration summary."""
    print_header("GRIDF Bias Correction Pipeline — Configuration Summary")

    print(f"Pipeline root: {cfg.pipeline_root}")
    print(f"Data root:     {cfg.data_root}")
    print(f"Figures root:  {cfg.figures_root}")
    print(f"Metadata root: {cfg.metadata_root}")
    print(f"Logs root:     {cfg.logs_root}")

    validate_input_paths(cfg, verbose=True)

    print_section("Products")
    inventory = all_product_year_inventory(cfg)

    for product_name in cfg.product_names:
        p = cfg.product(product_name)
        inv = inventory[product_name]

        print(f"\n{product_name}")
        print(f"  label:                  {p.get('label')}")
        print(f"  configured years:       {p.get('start_year')}–{p.get('end_year')}")
        print(f"  available raster years: {inv['n_available_years']}")
        if inv["available_years"]:
            print(f"  available span:         {min(inv['available_years'])}–{max(inv['available_years'])}")
        else:
            print("  available span:         none found")
        print(f"  processed years:        {inv['n_processed_years']}")
        if inv["processed_years"]:
            print(f"  processed span:         {min(inv['processed_years'])}–{max(inv['processed_years'])}")
        print(f"  GEE collection:         {p.get('gee_collection')}")
        print(f"  GEE band:               {p.get('gee_band')}")
        print(f"  daily aggregation:      {p.get('daily_aggregation')}")

    print_section("Method")
    es = cfg.method["event_selection"]
    zeta = cfg.method["zeta"]
    interp = cfg.method["interpolation"]

    print(f"Percentile basis:    {es['percentile_basis']}")
    print(f"Main percentile:     {es['main_percentile']}")
    print(f"Sensitivity labels:  {', '.join(es['percentile_labels'])}")
    print(f"Minimum gap days:    {es['min_gap_days']}")
    print(f"Zeta definition:     {zeta['definition']}")
    print(f"Main estimator:      {zeta['main_estimator']}")
    print(f"Save estimators:     {', '.join(zeta['save_estimators'])}")
    print(f"Minimum pairs/stn:   {zeta['min_pairs_per_station']}")
    print(f"Interpolation:       {interp['method']}, k={interp['idw_neighbors']}, power={interp['idw_power']}")


def write_run_manifest(cfg: PipelineConfig, command: str, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Write a small manifest for a command."""
    manifest = {
        "created_at": now_iso(),
        "command": command,
        "pipeline_root": str(cfg.pipeline_root),
        "paths": cfg.paths,
        "products": cfg.products,
        "method": cfg.method,
    }
    if extra:
        manifest["extra"] = dict(extra)

    out = cfg.metadata_root / "manifests" / f"manifest_{command}_{now_iso().replace(':', '').replace('-', '')}.json"
    return write_json(out, manifest)


def write_inventory(cfg: PipelineConfig) -> Path:
    """Write product annual-raster inventory to metadata/data_inventory."""
    inventory = all_product_year_inventory(cfg)
    out = cfg.metadata_root / "data_inventory" / "annual_max_year_inventory.json"
    return write_json(out, inventory)


def main(argv: Optional[List[str]] = None) -> int:
    """Small CLI for config module debugging."""
    cfg = load_config()
    init_folders(cfg)
    print_config_summary(cfg)
    write_inventory(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
