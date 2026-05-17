#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_utils.py

Shared plotting utilities for the GRIDF rainfall-product bias-correction
pipeline.

Part 08 scope
-------------
Provide consistent, publication-oriented plot styling for diagnostics and
sensitivity analysis.

The functions here intentionally avoid hard-coded product-specific decisions.
They are helpers used by diagnostics.py and the standalone comparison scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


def import_matplotlib():
    """Import matplotlib lazily with a clear error message."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for diagnostics figures.\n\n"
            "Install it with:\n"
            "    python3 -m pip install matplotlib\n"
        ) from exc
    return plt


def set_plot_style() -> None:
    """
    Apply a clean scientific plotting style.

    We avoid assuming system-specific fonts. If Helvetica is available,
    matplotlib will use it; otherwise it falls back gracefully.
    """
    plt = import_matplotlib()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })


def ensure_parent(path: Path) -> Path:
    """Create parent folder of a file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: Any, output_path: Path, close: bool = True) -> Path:
    """Save a matplotlib figure with tight layout."""
    plt = import_matplotlib()
    output_path = ensure_parent(Path(output_path))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return output_path


def finite_values(values: Iterable[float]) -> np.ndarray:
    """Return finite float values as numpy array."""
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def add_identity_line(ax: Any, values_x: Sequence[float], values_y: Sequence[float]) -> None:
    """Add 1:1 line spanning both x/y values."""
    x = finite_values(values_x)
    y = finite_values(values_y)
    if x.size == 0 or y.size == 0:
        return
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)


def plot_histogram(
    values: Sequence[float],
    output_path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
) -> Path:
    """Create a simple histogram."""
    set_plot_style()
    plt = import_matplotlib()
    vals = finite_values(values)

    fig, ax = plt.subplots(figsize=(7, 4))
    if vals.size:
        ax.hist(vals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_scatter(
    x: Sequence[float],
    y: Sequence[float],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    identity: bool = False,
    alpha: float = 0.35,
    s: float = 10.0,
) -> Path:
    """Create a scatter plot."""
    set_plot_style()
    plt = import_matplotlib()

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(x_arr[mask], y_arr[mask], s=s, alpha=alpha, edgecolors="none")
    if identity:
        add_identity_line(ax, x_arr[mask], y_arr[mask])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_timeseries(
    x: Sequence[Any],
    ys: Sequence[Sequence[float]],
    labels: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    """Create a simple multi-line time-series plot."""
    set_plot_style()
    plt = import_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y, label in zip(ys, labels):
        ax.plot(x, y, marker="o", linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, output_path)


def plot_raster_preview(
    raster: np.ndarray,
    output_path: Path,
    title: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    points_x: Optional[Sequence[float]] = None,
    points_y: Optional[Sequence[float]] = None,
    colorbar_label: str = "",
) -> Path:
    """Create a raster preview image."""
    set_plot_style()
    plt = import_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        np.where(np.isfinite(raster), raster, np.nan),
        origin="upper",
        extent=extent,
    )
    if points_x is not None and points_y is not None:
        ax.scatter(points_x, points_y, s=5, edgecolors="black", linewidths=0.15)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=ax, shrink=0.80)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    return save_figure(fig, output_path)
