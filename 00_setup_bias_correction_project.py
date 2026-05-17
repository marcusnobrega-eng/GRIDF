#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_setup_bias_correction_project.py

Create the folder structure and starter configuration files for the GRIDF
rainfall-product bias-correction pipeline.

Design:
    Everything is kept inside:
        /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

This includes code, configs, intermediate outputs, GEE pair CSVs, station zeta
tables, gridded zeta rasters, corrected annual maximum rasters, diagnostics,
figures, logs, and metadata.

If the folder becomes too large later, data/products and figures can be moved
to Google Drive or archived on Zenodo.

Usage:
    python 00_setup_bias_correction_project.py

Dry run:
    python 00_setup_bias_correction_project.py --dry-run

Overwrite existing YAML configs:
    python 00_setup_bias_correction_project.py --overwrite-config
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from datetime import datetime


GRIDF_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF")
PIPELINE_ROOT_DEFAULT = GRIDF_ROOT / "Bias_Correction_Pipeline"

GAUGE_TIMESERIES = GRIDF_ROOT / "Bias_Correction" / "rainfall_timeseries_with_metadata_all.csv"
STATION_INVENTORY = GRIDF_ROOT / "Bias_Correction" / "stations_inventory_filtered_all.csv"
ANNUAL_MAX_ROOT = GRIDF_ROOT / "Annual_Maximum_Precipitation"
BRAZIL_SHP = GRIDF_ROOT / "BrazilShapefiles" / "ADMLevels" / "bra_admbnda_adm0_ibge_2020.shp"
DEM = GRIDF_ROOT / "Misc" / "DEM.tif"

PRODUCTS = {
    "chirps": {
        "label": "CHIRPS",
        "start_year": 1995,
        "end_year": 2025,
        "annual_max_folder": ANNUAL_MAX_ROOT / "CHIRPS_Max",
        "gee_collection": "UCSB-CHG/CHIRPS/DAILY",
        "gee_band": "precipitation",
        "daily_aggregation": "daily_total_direct_mm_day",
        "native_resolution_deg": 0.05,
    },
    "persiann_cdr": {
        "label": "PERSIANN-CDR",
        "start_year": 1995,
        "end_year": 2025,
        "annual_max_folder": ANNUAL_MAX_ROOT / "PERSIANN_CDR_Max",
        "gee_collection": "NOAA/PERSIANN-CDR",
        "gee_band": "precipitation",
        "daily_aggregation": "daily_total_direct_mm_day",
        "native_resolution_deg": 0.25,
    },
    "br_dwgd": {
        "label": "BR-DWGD / Xavier",
        "start_year": 1995,
        "end_year": 2025,
        "annual_max_folder": ANNUAL_MAX_ROOT / "BR-DWGD",
        "gee_collection": "projects/sat-io/open-datasets/BR-DWGD/PR",
        "gee_band": "AUTO_DETECT",
        "daily_aggregation": "daily_total_direct_mm_day_confirm_band_and_scale",
        "native_resolution_deg": 0.10,
    },
    "imerg_v06": {
        "label": "IMERG V06",
        "start_year": 2001,
        "end_year": 2020,
        "annual_max_folder": ANNUAL_MAX_ROOT / "IMERG_V06_Max",
        "gee_collection": "NASA/GPM_L3/IMERG_V06",
        "gee_band": "precipitationCal",
        "daily_aggregation": "mean_half_hourly_rate_mm_hour_times_24",
        "native_resolution_deg": 0.10,
    },
    "imerg_v07": {
        "label": "IMERG V07 Early Daily",
        "start_year": 2001,
        "end_year": 2025,
        "annual_max_folder": ANNUAL_MAX_ROOT / "IMERG_V07_Max",
        "gee_collection": "projects/climate-engine-pro/assets/ce-gpm-imerg-v07/early-daily",
        "gee_band": "AUTO_DETECT",
        "daily_aggregation": "daily_total_direct_mm_day",
        "native_resolution_deg": 0.10,
    },
}

PERCENTILES = {
    "p90": 0.90,
    "p95": 0.95,
    "p98": 0.98,
    "p99": 0.99,
    "p995": 0.995,
}

ESTIMATORS = ["median", "mean"]


def mkdir(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] mkdir -p {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, text: str, overwrite: bool = False, dry_run: bool = False) -> None:
    if path.exists() and not overwrite:
        print(f"[skip] {path}")
        return

    if dry_run:
        print(f"[dry-run] write {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[write] {path}")


def make_paths_yml(root: Path) -> str:
    return dedent(f"""
    # paths.yml
    project:
      gridf_root: "{GRIDF_ROOT}"
      pipeline_root: "{root}"

    inputs:
      gauge_timeseries_csv: "{GAUGE_TIMESERIES}"
      station_inventory_csv: "{STATION_INVENTORY}"
      annual_max_root: "{ANNUAL_MAX_ROOT}"
      brazil_shapefile: "{BRAZIL_SHP}"
      dem: "{DEM}"

    outputs:
      output_root: "{root}"
      data_root: "{root / 'data'}"
      figures_root: "{root / 'figures'}"
      metadata_root: "{root / 'metadata'}"
      logs_root: "{root / 'logs'}"

    notes:
      storage_policy: "Everything is stored inside the GitHub pipeline folder for now. If outputs become too heavy, move data/products and figures to Google Drive or Zenodo."
    """).strip() + "\n"


def make_products_yml() -> str:
    lines = [
        "# products.yml",
        "# Product-specific settings for GEE sampling and annual-maximum correction.",
        "",
        "products:",
    ]
    for name, cfg in PRODUCTS.items():
        lines += [
            f"  {name}:",
            f"    label: \"{cfg['label']}\"",
            f"    start_year: {cfg['start_year']}",
            f"    end_year: {cfg['end_year']}",
            f"    annual_max_folder: \"{cfg['annual_max_folder']}\"",
            f"    gee_collection: \"{cfg['gee_collection']}\"",
            f"    gee_band: \"{cfg['gee_band']}\"",
            f"    daily_aggregation: \"{cfg['daily_aggregation']}\"",
            f"    native_resolution_deg: {cfg['native_resolution_deg']}",
            "    crs: \"EPSG:4326\"",
            "    scan_available_annual_max_rasters: true",
            "",
        ]
    return "\n".join(lines)


def make_method_yml() -> str:
    return dedent("""
    # method.yml
    event_selection:
      percentile_basis: "all_valid_days"
      main_percentile: 0.98
      sensitivity_percentiles: [0.90, 0.95, 0.98, 0.99, 0.995]
      percentile_labels: ["p90", "p95", "p98", "p99", "p995"]
      min_gap_days: 3
      min_events_per_station_year: 1

    gauge_qc:
      min_valid_rain_mm: 0.0
      max_valid_rain_mm: 500.0

    ratio_qc:
      min_gauge_rainfall_for_ratio_mm: 1.0
      min_product_rainfall_for_ratio_mm: 0.1
      max_rainfall_for_ratio_mm: 350.0
      ratio_clip: [0.25, 5.0]

    zeta:
      definition: "gauge_over_product"
      main_estimator: "median"
      sensitivity_estimators: ["mean"]
      save_estimators: ["mean", "median"]
      min_pairs_per_station: 10

    interpolation:
      method: "idw"
      idw_neighbors: 10
      idw_power: 2.0
      zeta_clip_before_interpolation: [0.05, 10.0]

    application:
      correction_equation: "corrected = raw_product * zeta"
      zeta_clip_before_application: [0.25, 5.0]
      preserve_input_grid: true
      preserve_nodata: true

    sensitivity_design:
      percentile_sensitivity:
        estimator: "median"
        percentiles: [0.90, 0.95, 0.98, 0.99, 0.995]
      estimator_sensitivity:
        percentile: 0.98
        estimators: ["median", "mean"]
    """).strip() + "\n"


def make_readme(root: Path) -> str:
    return dedent(f"""
    # GRIDF Bias Correction Pipeline

    Project root:

    ```text
    {root}
    ```

    Core correction:

    ```text
    zeta = gauge_rainfall / product_rainfall
    corrected_product = raw_product * zeta
    ```

    Products:
    - CHIRPS: 1995-2025
    - PERSIANN-CDR: 1995-2025
    - BR-DWGD / Xavier: 1995-2025
    - IMERG V06: 2001-2020
    - IMERG V07: 2001-2025

    Main method:
    - Percentile: P98
    - Sensitivity: P90, P95, P98, P99, P99.5
    - Zeta estimator: median
    - Estimator sensitivity: mean, primarily for P98
    - Interpolation: IDW, k = 10, power = 2

    Pipeline phases:
    1. Configuration and folder setup
    2. Gauge reading and validation
    3. Event selection
    4. GEE product inspection
    5. GEE bias-pair exports
    6. Station zeta estimation
    7. IDW interpolation
    8. Bias application and diagnostics
    """).strip() + "\n"


def make_gitignore() -> str:
    return dedent("""
    # Python
    __pycache__/
    *.py[cod]
    .pytest_cache/
    .mypy_cache/
    .ruff_cache/
    .ipynb_checkpoints/

    # Environments
    .venv/
    venv/
    env/
    .env

    # OS/editor
    .DS_Store
    Thumbs.db
    .vscode/
    .idea/

    # Logs
    *.log

    # Large-output policy:
    # Outputs are intentionally kept in this project for now.
    # If the repository becomes too heavy later, uncomment:
    #
    # data/products/
    # data/gauges/events/
    # figures/diagnostics/
    # figures/sensitivity/
    # *.tif
    # *.tiff
    # *.csv
    # *.parquet
    """).strip() + "\n"


def make_stub(module_name: str, description: str) -> str:
    return dedent(f"""
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    \"\"\"
    {module_name}.py

    {description}

    Placeholder created by 00_setup_bias_correction_project.py.
    Implementation will be added in later phases.
    \"\"\"

    from __future__ import annotations


    def main() -> None:
        print("{module_name}.py placeholder. Implementation pending.")


    if __name__ == "__main__":
        main()
    """).strip() + "\n"


def make_runner() -> str:
    return dedent("""
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    \"\"\"
    run_pipeline.py

    Main runner for the GRIDF bias-correction pipeline.
    Full command implementation will be added in Phase 1.
    \"\"\"

    from __future__ import annotations

    import argparse


    def main() -> None:
        parser = argparse.ArgumentParser(description="GRIDF bias-correction pipeline.")
        parser.add_argument("command", nargs="?", default="show-config")
        args = parser.parse_args()

        print("GRIDF Bias Correction Pipeline")
        print(f"Requested command: {args.command}")
        print("Runner implementation pending.")


    if __name__ == "__main__":
        main()
    """).strip() + "\n"


def create_folders(root: Path, dry_run: bool = False) -> None:
    # Code/config/documentation
    for folder in [
        root,
        root / "config",
        root / "src",
        root / "src" / "biascorr",
        root / "scripts",
        root / "notebooks",
        root / "docs",
        root / "tests",
        root / "logs",
        root / "metadata",
        root / "metadata" / "manifests",
        root / "metadata" / "gee_tasks",
        root / "metadata" / "run_summaries",
        root / "metadata" / "data_inventory",
        root / "data",
        root / "data" / "gauges",
        root / "data" / "gauges" / "raw",
        root / "data" / "gauges" / "processed",
        root / "data" / "gauges" / "qc",
        root / "data" / "gauges" / "events",
        root / "data" / "boundaries",
        root / "data" / "static",
        root / "data" / "static" / "dem",
        root / "data" / "static" / "masks",
        root / "figures",
        root / "figures" / "diagnostics",
        root / "figures" / "diagnostics" / "event_selection",
        root / "figures" / "diagnostics" / "pairs",
        root / "figures" / "diagnostics" / "zeta_station",
        root / "figures" / "diagnostics" / "zeta_maps",
        root / "figures" / "diagnostics" / "corrected_rasters",
        root / "figures" / "sensitivity",
        root / "figures" / "sensitivity" / "percentile",
        root / "figures" / "sensitivity" / "mean_vs_median",
        root / "figures" / "paper",
        root / "figures" / "paper" / "svg",
        root / "figures" / "paper" / "pdf",
        root / "figures" / "paper" / "png",
    ]:
        mkdir(folder, dry_run=dry_run)

    # Product-specific structure
    for product in PRODUCTS:
        product_root = root / "data" / "products" / product

        for folder in [
            product_root,
            product_root / "annual_max_raw",
            product_root / "annual_max_corrected",
            product_root / "gee_exports",
            product_root / "manifests",
            product_root / "logs",
        ]:
            mkdir(folder, dry_run=dry_run)

        for p_label in PERCENTILES:
            p_root = product_root / "sensitivity" / p_label

            for folder in [
                p_root,
                p_root / "events",
                p_root / "pairs",
                p_root / "zeta_station",
                p_root / "zeta_grid",
                p_root / "annual_max_corrected",
                p_root / "diagnostics",
                p_root / "tables",
            ]:
                mkdir(folder, dry_run=dry_run)

            for estimator in ESTIMATORS:
                for sub in ["zeta_station", "zeta_grid", "annual_max_corrected", "diagnostics"]:
                    mkdir(p_root / sub / estimator, dry_run=dry_run)


def create_files(root: Path, overwrite_config: bool = False, dry_run: bool = False) -> None:
    write_file(root / "config" / "paths.yml", make_paths_yml(root), overwrite=overwrite_config, dry_run=dry_run)
    write_file(root / "config" / "products.yml", make_products_yml(), overwrite=overwrite_config, dry_run=dry_run)
    write_file(root / "config" / "method.yml", make_method_yml(), overwrite=overwrite_config, dry_run=dry_run)

    write_file(root / "README.md", make_readme(root), overwrite=False, dry_run=dry_run)
    write_file(root / ".gitignore", make_gitignore(), overwrite=False, dry_run=dry_run)

    src = root / "src" / "biascorr"
    write_file(src / "__init__.py", '"""GRIDF bias-correction utilities."""\n', overwrite=False, dry_run=dry_run)

    modules = {
        "config": "Read YAML configuration files and validate project paths.",
        "utils": "General utilities for paths, logging, manifests, and labels.",
        "gauges": "Read and validate ANA gauge rainfall time-series files.",
        "event_selection": "Select station-year extreme rainfall events.",
        "gee_products": "Define GEE daily rainfall products and aggregation logic.",
        "gee_pair_exports": "Create GEE export tasks for gauge/product bias pairs.",
        "zeta": "Compute station-level zeta correction factors.",
        "interpolation": "Interpolate station zeta values using IDW.",
        "raster_utils": "Read, write, resample, and align rasters.",
        "apply_bias": "Apply gridded zeta factors to annual maximum rasters.",
        "diagnostics": "Generate diagnostic tables and figures.",
        "plot_utils": "Shared plotting utilities.",
    }

    for module, desc in modules.items():
        write_file(src / f"{module}.py", make_stub(module, desc), overwrite=False, dry_run=dry_run)

    write_file(root / "run_pipeline.py", make_runner(), overwrite=False, dry_run=dry_run)

    helper_scripts = {
        "inspect_gee_products.py": "Inspect GEE product bands, dates, units, and sample values.",
        "check_outputs.py": "Check missing or incomplete pipeline outputs.",
        "compare_mean_median.py": "Compare median and mean zeta correction factors.",
        "compare_percentile_sensitivity.py": "Compare P90, P95, P98, P99, and P99.5 sensitivity outputs.",
        "make_paper_figures.py": "Generate final paper-ready figures.",
    }

    for filename, desc in helper_scripts.items():
        stem = filename.replace(".py", "")
        write_file(root / filename, make_stub(stem, desc), overwrite=False, dry_run=dry_run)

    manifest = dedent(f"""
    setup_time: "{datetime.now().isoformat(timespec='seconds')}"
    pipeline_root: "{root}"
    gridf_root: "{GRIDF_ROOT}"
    storage_policy: "all_outputs_inside_github_folder_for_now"
    main_zeta_estimator: "median"
    interpolation: "IDW k=10 power=2"
    products: {list(PRODUCTS.keys())}
    percentile_labels: {list(PERCENTILES.keys())}
    """).strip() + "\n"
    write_file(root / "metadata" / "manifests" / "setup_manifest.yml", manifest, overwrite=True, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up GRIDF bias-correction pipeline folders.")
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.pipeline_root.expanduser().resolve()

    print("\nCreating GRIDF bias-correction pipeline structure")
    print(f"Pipeline root: {root}")
    print(f"Dry run: {args.dry_run}\n")

    create_folders(root, dry_run=args.dry_run)
    create_files(root, overwrite_config=args.overwrite_config, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nSetup complete.")
        print(f"Pipeline root: {root}")
        print("\nNext commands:")
        print(f"  cd {root}")
        print("  python run_pipeline.py show-config")
        print("\nImportant:")
        print("  Outputs are intentionally kept inside the GitHub folder for now.")
        print("  If files become too large later, move data/products and figures to Zenodo or Google Drive.\n")


if __name__ == "__main__":
    main()
