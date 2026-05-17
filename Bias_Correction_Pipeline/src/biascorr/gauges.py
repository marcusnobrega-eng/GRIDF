#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gauges.py

Gauge reading, metadata normalization, date-column detection, and QC summaries
for the GRIDF rainfall bias-correction pipeline.

Bugfix version:
---------------
This version removes the deprecated/unsupported pandas argument:

    infer_datetime_format=False

because newer pandas versions no longer accept it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, now_iso, print_header, print_section, write_json


DATE_LIKE_RE = re.compile(
    r"(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(\d{8})"
)


@dataclass
class GaugeTable:
    """Container for the wide gauge rainfall table and inferred metadata."""

    df: pd.DataFrame
    date_columns: List[str]
    parsed_dates: pd.Series
    metadata_columns: List[str]
    station_id_col: str
    station_name_col: Optional[str]
    latitude_col: str
    longitude_col: str
    city_col: Optional[str]
    state_col: Optional[str]
    date_dayfirst: bool
    date_score: Dict[str, Any]
    normalized_metadata: pd.DataFrame


# ---------------------------------------------------------------------
# Column detection utilities
# ---------------------------------------------------------------------

def _normalize_name(name: Any) -> str:
    """Normalize a column name for matching."""
    text = str(name).strip().lower()
    text = (
        text.replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )
    text = (
        text.replace("á", "a")
        .replace("à", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )
    return text


def find_first_existing_column(
    columns: Sequence[str],
    aliases: Sequence[str],
    required: bool = False,
    label: str = "column",
) -> Optional[str]:
    """Find the first column matching one of several aliases."""
    normalized = {_normalize_name(c): c for c in columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not find required {label}. Tried aliases: {aliases}. "
            f"Available columns include: {list(columns)[:20]}..."
        )

    return None


def detect_metadata_columns(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """Detect standard station metadata columns from a wide gauge table."""
    station_id_aliases = [
        "Code", "codigo", "código", "station_id", "stationid",
        "station_code", "stationcode", "codestacao", "cod_estacao",
        "id", "ana_code", "gauge", "gauge_id",
    ]

    station_name_aliases = [
        "Name", "nome", "station_name", "stationname",
        "nomeestacao", "station", "label", "label_name", "labelname",
    ]

    latitude_aliases = [
        "Latitude", "latitude", "lat", "Lat", "LAT",
        "y", "Y", "lat_dec", "latitude_decimal", "latitude_dd",
    ]

    longitude_aliases = [
        "Longitude", "longitude", "lon", "long", "Lon", "Long",
        "LON", "LONG", "x", "X", "lon_dec", "longitude_decimal",
        "longitude_dd",
    ]

    city_aliases = [
        "City", "city", "municipio", "município", "munic",
        "cidade", "localidade",
    ]

    state_aliases = [
        "State", "state", "uf", "UF", "estado", "province",
    ]

    return {
        "station_id_col": find_first_existing_column(
            columns, station_id_aliases, required=True, label="station id column"
        ),
        "station_name_col": find_first_existing_column(
            columns, station_name_aliases, required=False, label="station name column"
        ),
        "latitude_col": find_first_existing_column(
            columns, latitude_aliases, required=True, label="latitude column"
        ),
        "longitude_col": find_first_existing_column(
            columns, longitude_aliases, required=True, label="longitude column"
        ),
        "city_col": find_first_existing_column(columns, city_aliases, required=False),
        "state_col": find_first_existing_column(columns, state_aliases, required=False),
    }


# ---------------------------------------------------------------------
# Date-column detection
# ---------------------------------------------------------------------

def _candidate_date_columns(columns: Sequence[str]) -> List[str]:
    """Return columns that look like dates."""
    candidates = []
    for col in columns:
        text = str(col).strip()
        if DATE_LIKE_RE.search(text):
            candidates.append(col)
    return candidates


def _parse_date_candidates(
    columns: Sequence[str],
    dayfirst: bool,
) -> pd.Series:
    """
    Parse column names into dates.

    Important:
    ----------
    Do not use infer_datetime_format here. Recent pandas versions removed or
    changed support for that keyword.
    """
    parsed = pd.to_datetime(
        list(columns),
        errors="coerce",
        dayfirst=dayfirst,
    )
    return pd.Series(parsed, index=list(columns))


def _date_sequence_score(parsed: pd.Series) -> Dict[str, Any]:
    """
    Score parsed dates by how much the original column order resembles a daily
    sequence.
    """
    valid = parsed.dropna()
    if valid.empty:
        return {
            "n_valid": 0,
            "n_total": int(parsed.size),
            "valid_fraction": 0.0,
            "one_day_fraction": 0.0,
            "duplicate_count": 0,
            "score": -np.inf,
        }

    diffs = valid.diff().dropna().dt.days
    if diffs.empty:
        one_day_fraction = 0.0
    else:
        one_day_fraction = float((diffs == 1).mean())

    duplicate_count = int(valid.duplicated().sum())
    valid_fraction = float(valid.size / parsed.size)

    score = one_day_fraction + 0.001 * valid_fraction - 0.001 * duplicate_count

    return {
        "n_valid": int(valid.size),
        "n_total": int(parsed.size),
        "valid_fraction": valid_fraction,
        "one_day_fraction": one_day_fraction,
        "duplicate_count": duplicate_count,
        "score": float(score),
        "first_date": str(valid.min().date()) if valid.size else None,
        "last_date": str(valid.max().date()) if valid.size else None,
    }


def infer_date_columns_and_format(columns: Sequence[str]) -> Tuple[List[str], pd.Series, bool, Dict[str, Any]]:
    """
    Detect daily date columns and infer whether they are day-first.
    """
    date_columns = _candidate_date_columns(columns)

    if not date_columns:
        raise ValueError(
            "No date-like columns were found. Expected daily columns such as "
            "'1995-01-01', '1/1/1995', or similar."
        )

    parsed_mdy = _parse_date_candidates(date_columns, dayfirst=False)
    parsed_dmy = _parse_date_candidates(date_columns, dayfirst=True)

    score_mdy = _date_sequence_score(parsed_mdy)
    score_dmy = _date_sequence_score(parsed_dmy)

    if score_dmy["score"] > score_mdy["score"]:
        chosen_dayfirst = True
        chosen_parsed = parsed_dmy
    else:
        chosen_dayfirst = False
        chosen_parsed = parsed_mdy

    valid_mask = chosen_parsed.notna()
    final_date_columns = list(chosen_parsed.index[valid_mask])
    final_parsed_dates = chosen_parsed.loc[final_date_columns]

    score_info = {
        "candidate_date_columns": len(date_columns),
        "selected_date_columns": len(final_date_columns),
        "selected_dayfirst": chosen_dayfirst,
        "dayfirst_false": score_mdy,
        "dayfirst_true": score_dmy,
    }

    return final_date_columns, final_parsed_dates, chosen_dayfirst, score_info


# ---------------------------------------------------------------------
# Data conversion and metadata normalization
# ---------------------------------------------------------------------

def clean_station_id(value: Any) -> str:
    """Convert station ID to a stable string."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.match(r"^\d+\.0$", text):
        return text[:-2]
    return text


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, accepting comma decimals."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan}),
        errors="coerce",
    )


def rainfall_values_to_numeric(values: Any) -> np.ndarray:
    """Convert rainfall row values to float array."""
    s = pd.Series(values)
    return to_numeric_series(s).to_numpy(dtype=float)


def normalize_station_metadata(
    df: pd.DataFrame,
    station_id_col: str,
    station_name_col: Optional[str],
    latitude_col: str,
    longitude_col: str,
    city_col: Optional[str] = None,
    state_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create normalized station metadata table."""
    meta = pd.DataFrame(index=df.index)

    meta["station_id"] = df[station_id_col].map(clean_station_id)

    if station_name_col is not None:
        meta["station_name"] = df[station_name_col].astype(str).str.strip()
    else:
        meta["station_name"] = meta["station_id"].map(lambda x: f"Station {x}")

    meta["latitude"] = to_numeric_series(df[latitude_col])
    meta["longitude"] = to_numeric_series(df[longitude_col])

    if city_col is not None:
        meta["city"] = df[city_col].astype(str).str.strip()
    else:
        meta["city"] = ""

    if state_col is not None:
        meta["state"] = df[state_col].astype(str).str.strip()
    else:
        meta["state"] = ""

    meta["row_index"] = df.index.astype(int)

    meta["valid_coordinates"] = (
        meta["latitude"].between(-90, 90)
        & meta["longitude"].between(-180, 180)
    )

    return meta


# ---------------------------------------------------------------------
# Main loading and QC
# ---------------------------------------------------------------------

def load_gauge_table(csv_path: Path) -> GaugeTable:
    """Load and inspect the wide gauge rainfall CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Gauge rainfall CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    date_columns, parsed_dates, dayfirst, date_score = infer_date_columns_and_format(df.columns)
    metadata_columns = [c for c in df.columns if c not in set(date_columns)]

    detected = detect_metadata_columns(df.columns)

    normalized_metadata = normalize_station_metadata(
        df=df,
        station_id_col=detected["station_id_col"],
        station_name_col=detected["station_name_col"],
        latitude_col=detected["latitude_col"],
        longitude_col=detected["longitude_col"],
        city_col=detected["city_col"],
        state_col=detected["state_col"],
    )

    return GaugeTable(
        df=df,
        date_columns=date_columns,
        parsed_dates=parsed_dates,
        metadata_columns=metadata_columns,
        station_id_col=detected["station_id_col"],
        station_name_col=detected["station_name_col"],
        latitude_col=detected["latitude_col"],
        longitude_col=detected["longitude_col"],
        city_col=detected["city_col"],
        state_col=detected["state_col"],
        date_dayfirst=dayfirst,
        date_score=date_score,
        normalized_metadata=normalized_metadata,
    )


def summarize_station_coverage(
    table: GaugeTable,
    min_valid_rain_mm: float = 0.0,
    max_valid_rain_mm: float = 500.0,
) -> pd.DataFrame:
    """Summarize gauge rainfall coverage by station."""
    rows: List[Dict[str, Any]] = []
    dates = table.parsed_dates

    min_date = dates.min()
    max_date = dates.max()
    n_total_dates = int(len(table.date_columns))

    for idx, meta in table.normalized_metadata.iterrows():
        vals = rainfall_values_to_numeric(table.df.loc[idx, table.date_columns].values)

        valid = (
            np.isfinite(vals)
            & (vals >= min_valid_rain_mm)
            & (vals <= max_valid_rain_mm)
        )

        wet = valid & (vals > 0.0)

        if valid.any():
            valid_dates = dates.iloc[np.where(valid)[0]]
            first_valid = valid_dates.min()
            last_valid = valid_dates.max()
        else:
            first_valid = pd.NaT
            last_valid = pd.NaT

        rows.append({
            "row_index": int(idx),
            "station_id": meta["station_id"],
            "station_name": meta["station_name"],
            "city": meta["city"],
            "state": meta["state"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "valid_coordinates": bool(meta["valid_coordinates"]),
            "n_total_days_in_table": n_total_dates,
            "n_valid_days": int(valid.sum()),
            "n_missing_or_invalid_days": int((~valid).sum()),
            "valid_fraction": float(valid.mean()) if valid.size else np.nan,
            "n_wet_days_gt0": int(wet.sum()),
            "wet_fraction_gt0": float(wet.mean()) if wet.size else np.nan,
            "mean_rain_mm_valid_days": float(np.nanmean(vals[valid])) if valid.any() else np.nan,
            "max_rain_mm_valid_days": float(np.nanmax(vals[valid])) if valid.any() else np.nan,
            "first_valid_date": None if pd.isna(first_valid) else str(first_valid.date()),
            "last_valid_date": None if pd.isna(last_valid) else str(last_valid.date()),
            "table_first_date": str(min_date.date()),
            "table_last_date": str(max_date.date()),
        })

    return pd.DataFrame(rows)


def prepare_gauges(
    cfg: Any,
    overwrite: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Prepare gauge metadata and QC outputs.
    """
    input_csv = Path(cfg.paths["inputs"]["gauge_timeseries_csv"])
    out_processed = Path(cfg.data_root) / "gauges" / "processed"
    out_qc = Path(cfg.data_root) / "gauges" / "qc"
    ensure_dir(out_processed)
    ensure_dir(out_qc)

    if verbose:
        print_header("Preparing gauge rainfall database")
        print(f"Input CSV: {input_csv}")

    table = load_gauge_table(input_csv)

    gauge_qc = cfg.method["gauge_qc"]
    summary = summarize_station_coverage(
        table,
        min_valid_rain_mm=float(gauge_qc["min_valid_rain_mm"]),
        max_valid_rain_mm=float(gauge_qc["max_valid_rain_mm"]),
    )

    metadata_out = out_processed / "gauge_metadata_normalized.csv"
    summary_out = out_qc / "gauge_station_coverage_summary.csv"
    date_cols_out = out_qc / "gauge_date_columns.csv"
    manifest_out = out_qc / "gauge_detection_manifest.json"

    table.normalized_metadata.to_csv(metadata_out, index=False)
    summary.to_csv(summary_out, index=False)

    date_df = pd.DataFrame({
        "column_name": table.date_columns,
        "parsed_date": table.parsed_dates.dt.strftime("%Y-%m-%d").values,
        "year": table.parsed_dates.dt.year.values,
        "month": table.parsed_dates.dt.month.values,
        "day": table.parsed_dates.dt.day.values,
    })
    date_df.to_csv(date_cols_out, index=False)

    manifest = {
        "created_at": now_iso(),
        "input_csv": str(input_csv),
        "n_rows_stations": int(table.df.shape[0]),
        "n_columns_total": int(table.df.shape[1]),
        "n_metadata_columns": int(len(table.metadata_columns)),
        "n_date_columns": int(len(table.date_columns)),
        "station_id_col": table.station_id_col,
        "station_name_col": table.station_name_col,
        "latitude_col": table.latitude_col,
        "longitude_col": table.longitude_col,
        "city_col": table.city_col,
        "state_col": table.state_col,
        "date_dayfirst_selected": bool(table.date_dayfirst),
        "date_detection_score": table.date_score,
        "gauge_qc": gauge_qc,
        "outputs": {
            "metadata": str(metadata_out),
            "coverage_summary": str(summary_out),
            "date_columns": str(date_cols_out),
        },
    }
    write_json(manifest_out, manifest)

    if verbose:
        print_section("Gauge detection")
        print(f"Stations:           {table.df.shape[0]}")
        print(f"Date columns:       {len(table.date_columns)}")
        print(f"Date first:         {table.parsed_dates.min().date()}")
        print(f"Date last:          {table.parsed_dates.max().date()}")
        print(f"Selected dayfirst:  {table.date_dayfirst}")
        print(f"Station ID column:  {table.station_id_col}")
        print(f"Station name col:   {table.station_name_col}")
        print(f"Latitude column:    {table.latitude_col}")
        print(f"Longitude column:   {table.longitude_col}")

        print_section("Gauge outputs")
        print(metadata_out)
        print(summary_out)
        print(date_cols_out)
        print(manifest_out)

    return {
        "metadata": metadata_out,
        "coverage_summary": summary_out,
        "date_columns": date_cols_out,
        "manifest": manifest_out,
    }


def main() -> None:
    """Debug entry point."""
    from .config import load_config, init_folders

    cfg = load_config()
    init_folders(cfg)
    prepare_gauges(cfg)


if __name__ == "__main__":
    main()
