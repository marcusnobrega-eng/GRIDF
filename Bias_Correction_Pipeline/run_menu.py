#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_menu.py

Interactive menu for the GRIDF Bias Correction Pipeline.

Purpose
-------
Instead of typing long terminal commands, run:

    python3 run_menu.py

Then choose what you want to run.

This script simply builds and executes the correct run_pipeline.py commands.
It does not replace run_pipeline.py. It is a user-friendly wrapper.

Recommended usage
-----------------
From the pipeline folder:

    cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
    python3 run_menu.py
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
RUN_PIPELINE = PIPELINE_ROOT / "run_pipeline.py"


PRODUCTS = [
    "imerg_v07",
    "imerg_v06",
    "chirps",
    "persiann_cdr",
    "br_dwgd",
]

PERCENTILES = [
    "p90",
    "p95",
    "p98",
    "p99",
    "p995",
]

ESTIMATORS = [
    "median",
    "mean",
]


def clear_screen() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def print_header() -> None:
    print("=" * 78)
    print("GRIDF Bias Correction Pipeline — Interactive Menu")
    print("=" * 78)
    print(f"Pipeline root: {PIPELINE_ROOT}")
    print()


def ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    print(prompt)
    for i, choice in enumerate(choices, start=1):
        marker = " [default]" if default == choice else ""
        print(f"  {i}) {choice}{marker}")

    while True:
        raw = input("> ").strip()
        if raw == "" and default is not None:
            return default

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]

        if raw in choices:
            return raw

        print("Invalid choice. Try again.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix} ").strip().lower()
    if raw == "":
        return default
    return raw in ["y", "yes", "s", "sim"]


def ask_optional_int(prompt: str) -> int | None:
    raw = input(f"{prompt} (press Enter to skip): ").strip()
    if raw == "":
        return None
    return int(raw)


def ask_optional_path(prompt: str) -> str | None:
    raw = input(f"{prompt} (press Enter to skip): ").strip()
    if raw == "":
        return None
    return raw


def run_command(args: list[str]) -> int:
    cmd = [sys.executable, str(RUN_PIPELINE)] + args

    print()
    print("=" * 78)
    print("Command to run")
    print("=" * 78)
    print(" ".join(shlex.quote(x) for x in cmd))
    print()

    if not ask_yes_no("Run this command?", default=True):
        print("Cancelled.")
        return 0

    print()
    print("=" * 78)
    print("Running")
    print("=" * 78)

    result = subprocess.run(cmd, cwd=str(PIPELINE_ROOT))
    print()
    print("=" * 78)
    print(f"Finished with return code: {result.returncode}")
    print("=" * 78)
    input("Press Enter to return to the menu...")
    return int(result.returncode)


def build_product_percentile_args(require_estimator: bool = False) -> list[str]:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")

    args = ["--product", product, "--percentile", percentile]

    if require_estimator:
        estimator = ask_choice("Choose estimator:", ESTIMATORS, default="median")
        args += ["--estimator", estimator]

    return args


def menu_show_config() -> None:
    run_command(["show-config"])


def menu_check_paths() -> None:
    run_command(["check-paths"])


def menu_inventory() -> None:
    run_command(["inventory-years"])


def menu_prepare_gauges() -> None:
    run_command(["prepare-gauges"])


def menu_select_events_debug() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")

    start_year = ask_optional_int("Start year")
    end_year = ask_optional_int("End year")
    station_limit = ask_optional_int("Station limit for debugging")

    args = ["select-events", "--product", product, "--percentile", percentile]

    if start_year is not None:
        args += ["--start-year", str(start_year)]
    if end_year is not None:
        args += ["--end-year", str(end_year)]
    if station_limit is not None:
        args += ["--station-limit", str(station_limit)]

    run_command(args)


def menu_select_events_full_one_product() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")
    run_command(["select-events", "--product", product, "--percentile", percentile])


def menu_select_events_all_percentiles_one_product() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    run_command(["select-events", "--product", product, "--all-percentiles"])


def menu_inspect_gee() -> None:
    product = ask_choice("Choose product:", PRODUCTS + ["all-products"], default="imerg_v07")
    if product == "all-products":
        args = ["inspect-gee", "--all-products"]
    else:
        args = ["inspect-gee", "--product", product]

    date = input("Sample date YYYY-MM-DD (press Enter for default): ").strip()
    if date:
        args += ["--date", date]

    run_command(args)


def menu_export_pairs_dry_run() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")

    start_year = ask_optional_int("Start year")
    end_year = ask_optional_int("End year")

    args = ["export-pairs", "--product", product, "--percentile", percentile, "--dry-run"]

    if start_year is not None:
        args += ["--start-year", str(start_year)]
    if end_year is not None:
        args += ["--end-year", str(end_year)]

    run_command(args)


def menu_export_pairs_submit() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")

    start_year = ask_optional_int("Start year")
    end_year = ask_optional_int("End year")
    max_features = ask_optional_int("Max features per export")

    args = ["export-pairs", "--product", product, "--percentile", percentile]

    if start_year is not None:
        args += ["--start-year", str(start_year)]
    if end_year is not None:
        args += ["--end-year", str(end_year)]
    if max_features is not None:
        args += ["--max-features-per-export", str(max_features)]

    print()
    print("WARNING: This will submit Earth Engine export tasks to Google Drive.")
    run_command(args)


def menu_compute_zeta() -> None:
    args = ["compute-zeta"] + build_product_percentile_args(require_estimator=True)

    pairs_folder = ask_optional_path("Optional local pairs folder override")
    if pairs_folder is not None:
        args += ["--pairs-folder", pairs_folder]

    run_command(args)


def menu_interpolate_zeta() -> None:
    args = ["interpolate-zeta"] + build_product_percentile_args(require_estimator=True)
    run_command(args)


def menu_apply_bias() -> None:
    args = ["apply-bias"] + build_product_percentile_args(require_estimator=True)

    start_year = ask_optional_int("Start year")
    end_year = ask_optional_int("End year")

    if start_year is not None:
        args += ["--start-year", str(start_year)]
    if end_year is not None:
        args += ["--end-year", str(end_year)]

    run_command(args)


def menu_diagnostics() -> None:
    args = ["diagnostics"] + build_product_percentile_args(require_estimator=True)
    run_command(args)


def menu_percentile_sensitivity() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    estimator = ask_choice("Choose estimator:", ESTIMATORS, default="median")
    run_command(["percentile-sensitivity", "--product", product, "--estimator", estimator])


def menu_mean_median_sensitivity() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")
    run_command(["mean-median-sensitivity", "--product", product, "--percentile", percentile])


def menu_check_outputs() -> None:
    product = ask_choice("Choose product:", PRODUCTS, default="imerg_v07")
    percentile = ask_choice("Choose percentile:", PERCENTILES, default="p98")
    estimator = ask_choice("Choose estimator:", ESTIMATORS, default="median")

    cmd = [
        sys.executable,
        str(PIPELINE_ROOT / "check_outputs.py"),
        "--product", product,
        "--percentile", percentile,
        "--estimator", estimator,
    ]

    print()
    print("=" * 78)
    print("Command to run")
    print("=" * 78)
    print(" ".join(shlex.quote(x) for x in cmd))
    print()

    if ask_yes_no("Run this command?", default=True):
        subprocess.run(cmd, cwd=str(PIPELINE_ROOT))

    input("Press Enter to return to the menu...")


def print_menu() -> None:
    print_header()
    print("Basic checks")
    print("  1) Show configuration")
    print("  2) Check input paths")
    print("  3) Inventory annual maximum rasters")
    print()
    print("Gauge and events")
    print("  4) Prepare gauges")
    print("  5) Select events — debug/test run")
    print("  6) Select events — full one product/percentile")
    print("  7) Select events — all percentiles for one product")
    print()
    print("GEE")
    print("  8) Inspect GEE product")
    print("  9) Export pairs — dry run")
    print(" 10) Export pairs — submit real GEE tasks")
    print()
    print("Local correction")
    print(" 11) Compute station zeta")
    print(" 12) Interpolate zeta")
    print(" 13) Apply bias to annual maximum rasters")
    print()
    print("Diagnostics")
    print(" 14) Run diagnostics")
    print(" 15) Percentile sensitivity")
    print(" 16) Mean-vs-median sensitivity")
    print(" 17) Check outputs")
    print()
    print("  0) Exit")
    print()


def main() -> int:
    actions = {
        "1": menu_show_config,
        "2": menu_check_paths,
        "3": menu_inventory,
        "4": menu_prepare_gauges,
        "5": menu_select_events_debug,
        "6": menu_select_events_full_one_product,
        "7": menu_select_events_all_percentiles_one_product,
        "8": menu_inspect_gee,
        "9": menu_export_pairs_dry_run,
        "10": menu_export_pairs_submit,
        "11": menu_compute_zeta,
        "12": menu_interpolate_zeta,
        "13": menu_apply_bias,
        "14": menu_diagnostics,
        "15": menu_percentile_sensitivity,
        "16": menu_mean_median_sensitivity,
        "17": menu_check_outputs,
    }

    while True:
        clear_screen()
        print_menu()
        choice = input("Choose an option: ").strip()

        if choice == "0":
            print("Exiting.")
            return 0

        action = actions.get(choice)
        if action is None:
            input("Invalid option. Press Enter to continue...")
            continue

        try:
            action()
        except KeyboardInterrupt:
            print("\nInterrupted.")
            input("Press Enter to return to the menu...")
        except Exception as exc:
            print()
            print("=" * 78)
            print("ERROR")
            print("=" * 78)
            print(f"{type(exc).__name__}: {exc}")
            input("Press Enter to return to the menu...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
