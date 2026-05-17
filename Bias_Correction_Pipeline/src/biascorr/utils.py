#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

General utilities for the GRIDF bias-correction pipeline.

Part 01 scope:
    - safe folder creation
    - label/percentile formatting
    - annual-raster year scanning
    - simple JSON/YAML manifest writing
    - console formatting
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def ensure_dir(path: Path) -> Path:
    """Create a folder if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create multiple folders."""
    for path in paths:
        ensure_dir(Path(path))


def percentile_to_label(percentile: float) -> str:
    """
    Convert a percentile fraction to a compact label.

    Examples
    --------
    0.90  -> p90
    0.98  -> p98
    0.995 -> p995
    """
    value = float(percentile) * 100.0
    if abs(value - round(value)) < 1e-10:
        return f"p{int(round(value))}"
    return "p" + str(value).replace(".", "").rstrip("0")


def label_to_percentile(label: str) -> float:
    """
    Convert labels such as p90, p98, p995 to percentile fractions.
    """
    clean = str(label).strip().lower().replace("p", "")
    if clean == "995":
        return 0.995
    return float(clean) / 100.0


def scan_years_from_filenames(folder: Path, extensions: Sequence[str] = (".tif", ".tiff")) -> List[int]:
    """
    Scan a folder for years embedded in raster filenames.

    Parameters
    ----------
    folder:
        Folder containing annual maximum rasters.
    extensions:
        File extensions to consider.

    Returns
    -------
    Sorted list of unique years found in filenames.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    years = set()
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {e.lower() for e in extensions}:
            continue
        for match in YEAR_RE.findall(path.name):
            years.add(int(match))

    return sorted(years)


def restrict_years_to_available(
    configured_start: int,
    configured_end: int,
    available_years: Sequence[int],
) -> List[int]:
    """Return available years that fall inside the configured range."""
    available = set(int(y) for y in available_years)
    return [y for y in range(int(configured_start), int(configured_end) + 1) if y in available]


def write_json(path: Path, data: Mapping[str, Any], indent: int = 2) -> Path:
    """Write a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=indent, sort_keys=False), encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    """Write a text file."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    """Return current local time as ISO string."""
    return datetime.now().isoformat(timespec="seconds")


def print_header(title: str) -> None:
    """Print a readable console header."""
    line = "=" * 78
    print("\n" + line)
    print(title)
    print(line)


def print_section(title: str) -> None:
    """Print a readable console section."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def status(ok: bool) -> str:
    """Return text status mark."""
    return "OK" if ok else "MISSING"


def path_status(path: Path) -> str:
    """Return status string for a path."""
    path = Path(path)
    return f"{status(path.exists()):8s} {path}"


def deep_get(mapping: Mapping[str, Any], keys: Sequence[str], default: Optional[Any] = None) -> Any:
    """Safely get a nested dictionary value."""
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_path(value: str | Path) -> Path:
    """Convert value to expanded Path."""
    return Path(value).expanduser()
