# download_ana_daily_rainfall_1995_2025.py

from __future__ import annotations

import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# =========================
# CONFIGURATION
# =========================

INPUT_CSV = Path(
    "/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction/rainfall_timeseries_with_metadata_all.csv"
)

START_DATE = "1995-01-01"
END_DATE = "2025-12-31"

# "overlap" = use stations that have any overlap with 1995–2025
# "full"    = use only stations covering the whole period
COVERAGE_MODE = "overlap"

# ANA HidroWeb consistency level:
# 1 = raw data
# 2 = quality-controlled / consistent data
CONSISTENCY_LEVEL = "2"

MAX_WORKERS = 4
SLEEP_BETWEEN_REQUESTS = 0.2

OUTPUT_DIR = Path("ana_rainfall_1995_2025")
CACHE_DIR = OUTPUT_DIR / "station_cache"

OUTPUT_EXCEL = OUTPUT_DIR / "rainfall_timeseries_with_metadata_ANA_1995_2025.xlsx"
OUTPUT_LONG_CSV = OUTPUT_DIR / "rainfall_ANA_1995_2025_long.csv"
OUTPUT_LOG = OUTPUT_DIR / "download_log.csv"

ANA_URL = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"

METADATA_COLUMNS = [
    "Code", "Name", "City", "State", "Latitude", "Longitude",
    "StartDate", "EndDate", "NYD", "MD", "N_YWOMD", "YWMD",
    "SubBasin", "Responsible"
]


# =========================
# HELPER FUNCTIONS
# =========================

def create_session() -> requests.Session:
    """
    Create a requests session with retry logic.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "python-requests ANA daily rainfall downloader"
    })

    return session


def remove_xml_namespace(tag: str) -> str:
    """
    Remove XML namespace from tag names.
    """
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_ana_xml_to_dataframe(xml_text: str) -> pd.DataFrame:
    """
    Parse the XML returned by the ANA HidroSerieHistorica service.

    The function searches for monthly records containing the field 'DataHora'.
    """
    if not xml_text or "DataHora" not in xml_text:
        return pd.DataFrame()

    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError:
        root = ET.fromstring(xml_text)

    rows = []

    for element in root.iter():
        if remove_xml_namespace(element.tag).lower().startswith("table"):
            row = {
                remove_xml_namespace(child.tag): child.text
                for child in list(element)
            }

            if "DataHora" in row:
                rows.append(row)

    return pd.DataFrame(rows)


def parse_number(value):
    """
    Convert ANA numeric values to float.

    ANA data may use commas as decimal separators.
    """
    if value is None:
        return np.nan

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.nan

    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def convert_monthly_rainfall_to_daily(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ANA monthly rainfall records into daily rainfall records.

    ANA returns rainfall as:
    Chuva01, Chuva02, ..., Chuva31

    Output:
    Code, Date, Rain_mm
    """
    if monthly_df.empty:
        return pd.DataFrame(columns=["Code", "Date", "Rain_mm"])

    rows = []

    for _, record in monthly_df.iterrows():
        station_code = str(record.get("EstacaoCodigo", "")).strip()

        month_start = pd.to_datetime(
            record.get("DataHora"),
            errors="coerce",
            dayfirst=True
        )

        if pd.isna(month_start):
            continue

        month_start = pd.Timestamp(month_start.year, month_start.month, 1)

        for day in range(1, 32):
            rainfall_column = f"Chuva{day:02d}"

            if rainfall_column not in record:
                continue

            date = month_start + pd.Timedelta(days=day - 1)

            # Avoid invalid dates such as February 30 or April 31
            if date.month != month_start.month:
                continue

            rows.append({
                "Code": station_code,
                "Date": date,
                "Rain_mm": parse_number(record[rainfall_column])
            })

    daily_df = pd.DataFrame(rows)

    if daily_df.empty:
        return pd.DataFrame(columns=["Code", "Date", "Rain_mm"])

    return daily_df


def download_station_daily_rainfall(
    station_code: str,
    session: requests.Session
) -> tuple[str, pd.DataFrame, str]:
    """
    Download daily rainfall data for one ANA station.

    A cache file is saved for each station so the process can be resumed
    without downloading everything again.
    """
    station_code = str(station_code).strip()
    cache_file = CACHE_DIR / f"{station_code}.csv"

    if cache_file.exists():
        try:
            cached_df = pd.read_csv(
                cache_file,
                parse_dates=["Date"],
                dtype={"Code": str}
            )
            return station_code, cached_df, "cached"
        except Exception:
            cache_file.unlink(missing_ok=True)

    params = {
        "codEstacao": station_code,
        "dataInicio": pd.Timestamp(START_DATE).strftime("%d/%m/%Y"),
        "dataFim": pd.Timestamp(END_DATE).strftime("%d/%m/%Y"),
        "tipoDados": "2",  # 2 = rainfall
        "nivelConsistencia": CONSISTENCY_LEVEL,
    }

    try:
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        response = session.get(
            ANA_URL,
            params=params,
            timeout=90
        )

        response.raise_for_status()

        monthly_df = parse_ana_xml_to_dataframe(response.text)
        daily_df = convert_monthly_rainfall_to_daily(monthly_df)

        if not daily_df.empty:
            daily_df["Code"] = daily_df["Code"].astype(str).str.strip()

            daily_df = daily_df[
                (daily_df["Date"] >= START_DATE) &
                (daily_df["Date"] <= END_DATE)
            ].copy()

        daily_df.to_csv(cache_file, index=False)

        return station_code, daily_df, "ok"

    except Exception as error:
        empty_df = pd.DataFrame(columns=["Code", "Date", "Rain_mm"])
        return station_code, empty_df, f"error: {error}"


def select_stations_by_period(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select stations based on the requested time period.

    COVERAGE_MODE = "overlap":
        Select stations with any overlap with the target period.

    COVERAGE_MODE = "full":
        Select only stations that cover the entire target period.
    """
    metadata_df = metadata_df.copy()

    metadata_df["Code"] = metadata_df["Code"].astype(str).str.strip()

    metadata_df["StartDate_dt"] = pd.to_datetime(
        metadata_df.get("StartDate"),
        errors="coerce"
    )

    metadata_df["EndDate_dt"] = pd.to_datetime(
        metadata_df.get("EndDate"),
        errors="coerce"
    )

    start_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp(END_DATE)

    if COVERAGE_MODE == "full":
        selected = (
            (metadata_df["StartDate_dt"].isna() | (metadata_df["StartDate_dt"] <= start_date)) &
            (metadata_df["EndDate_dt"].isna() | (metadata_df["EndDate_dt"] >= end_date))
        )
    else:
        selected = (
            (metadata_df["StartDate_dt"].isna() | (metadata_df["StartDate_dt"] <= end_date)) &
            (metadata_df["EndDate_dt"].isna() | (metadata_df["EndDate_dt"] >= start_date))
        )

    return metadata_df.loc[selected].drop(
        columns=["StartDate_dt", "EndDate_dt"]
    )


def format_date_as_column_name(date: pd.Timestamp) -> str:
    """
    Format date columns to match the original wide-format file style.
    """
    return f"{date.month}/{date.day}/{date.year}"


def recompute_basic_statistics(
    wide_df: pd.DataFrame,
    date_columns: list[str]
) -> pd.DataFrame:
    """
    Recompute basic metadata statistics based on the downloaded daily data.

    MD       = number of missing daily values
    NYD      = number of years with at least one valid daily value
    N_YWOMD  = number of complete years without missing data
    YWMD     = number of years with data but with at least one missing day
    """
    rainfall_df = wide_df[date_columns]
    years = pd.to_datetime(date_columns).year

    wide_df["StartDate"] = rainfall_df.apply(
        lambda row: pd.to_datetime(row.first_valid_index()).strftime("%-m/%-d/%Y")
        if row.first_valid_index() is not None else np.nan,
        axis=1
    )

    wide_df["EndDate"] = rainfall_df.apply(
        lambda row: pd.to_datetime(row.last_valid_index()).strftime("%-m/%-d/%Y")
        if row.last_valid_index() is not None else np.nan,
        axis=1
    )

    wide_df["MD"] = rainfall_df.isna().sum(axis=1)

    number_of_years_with_data = []
    number_of_complete_years = []
    number_of_years_with_missing_data = []

    for _, row in rainfall_df.iterrows():
        yearly_df = pd.DataFrame({
            "year": years,
            "has_data": row.notna().to_numpy()
        })

        grouped_by_year = yearly_df.groupby("year")["has_data"]

        years_with_any_data = grouped_by_year.any()
        complete_years = grouped_by_year.all()

        number_of_years_with_data.append(int(years_with_any_data.sum()))
        number_of_complete_years.append(int(complete_years.sum()))
        number_of_years_with_missing_data.append(
            int((years_with_any_data & ~complete_years).sum())
        )

    wide_df["NYD"] = number_of_years_with_data
    wide_df["N_YWOMD"] = number_of_complete_years
    wide_df["YWMD"] = number_of_years_with_missing_data

    return wide_df


# =========================
# MAIN SCRIPT
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading input metadata file...")

    metadata_df = pd.read_csv(INPUT_CSV)

    if "Code" not in metadata_df.columns:
        raise ValueError("The input CSV must contain a 'Code' column.")

    for column in METADATA_COLUMNS:
        if column not in metadata_df.columns:
            metadata_df[column] = np.nan

    metadata_df = metadata_df[METADATA_COLUMNS].copy()

    metadata_df["Code"] = (
        metadata_df["Code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    selected_stations = select_stations_by_period(metadata_df)

    station_codes = (
        selected_stations["Code"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Selected stations: {len(station_codes)}")

    session = create_session()

    all_daily_data = []
    log_rows = []

    print("Downloading ANA daily rainfall data...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                download_station_daily_rainfall,
                station_code,
                session
            ): station_code
            for station_code in station_codes
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            station_code, daily_df, status = future.result()

            log_rows.append({
                "Code": station_code,
                "status": status,
                "n_rows": len(daily_df)
            })

            if not daily_df.empty:
                all_daily_data.append(daily_df)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUTPUT_LOG, index=False)

    if all_daily_data:
        daily_rainfall_df = pd.concat(all_daily_data, ignore_index=True)
    else:
        daily_rainfall_df = pd.DataFrame(columns=["Code", "Date", "Rain_mm"])

    daily_rainfall_df["Code"] = (
        daily_rainfall_df["Code"]
        .astype(str)
        .str.strip()
    )

    daily_rainfall_df["Date"] = pd.to_datetime(
        daily_rainfall_df["Date"],
        errors="coerce"
    )

    daily_rainfall_df = daily_rainfall_df.dropna(subset=["Date"])

    daily_rainfall_df = daily_rainfall_df[
        (daily_rainfall_df["Date"] >= START_DATE) &
        (daily_rainfall_df["Date"] <= END_DATE)
    ].copy()

    print("Saving long-format CSV...")

    daily_rainfall_df.to_csv(OUTPUT_LONG_CSV, index=False)

    print("Creating wide-format table...")

    full_date_range = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D"
    )

    date_columns = [
        format_date_as_column_name(date)
        for date in full_date_range
    ]

    if daily_rainfall_df.empty:
        rainfall_pivot = pd.DataFrame(
            index=selected_stations["Code"].astype(str).unique()
        )
    else:
        daily_rainfall_df["DateColumn"] = daily_rainfall_df["Date"].map(
            format_date_as_column_name
        )

        rainfall_pivot = daily_rainfall_df.pivot_table(
            index="Code",
            columns="DateColumn",
            values="Rain_mm",
            aggfunc="first"
        )

    rainfall_pivot = rainfall_pivot.reindex(
        index=selected_stations["Code"].astype(str).tolist()
    )

    rainfall_pivot = rainfall_pivot.reindex(columns=date_columns)

    wide_df = (
        selected_stations
        .set_index("Code")
        .join(rainfall_pivot, how="left")
        .reset_index()
    )

    wide_df = recompute_basic_statistics(
        wide_df,
        date_columns
    )

    ordered_columns = METADATA_COLUMNS + date_columns
    wide_df = wide_df[ordered_columns]

    print("Exporting Excel file...")

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="xlsxwriter",
        engine_kwargs={"options": {"constant_memory": True}}
    ) as writer:
        wide_df.to_excel(
            writer,
            sheet_name="rain_daily_wide",
            index=False
        )

        log_df.to_excel(
            writer,
            sheet_name="download_log",
            index=False
        )

        selected_stations.to_excel(
            writer,
            sheet_name="metadata_input",
            index=False
        )

        workbook = writer.book

        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAD3",
            "border": 1
        })

        number_format = workbook.add_format({
            "num_format": "0.0"
        })

        text_format = workbook.add_format({
            "text_wrap": False
        })

        worksheet = writer.sheets["rain_daily_wide"]

        worksheet.freeze_panes(1, len(METADATA_COLUMNS))
        worksheet.autofilter(
            0,
            0,
            len(wide_df),
            len(wide_df.columns) - 1
        )

        for column_index, column_name in enumerate(wide_df.columns):
            worksheet.write(
                0,
                column_index,
                column_name,
                header_format
            )

        worksheet.set_column(0, 0, 10, text_format)
        worksheet.set_column(1, 3, 18, text_format)
        worksheet.set_column(4, 5, 12, text_format)
        worksheet.set_column(6, 13, 12, text_format)

        worksheet.set_column(
            len(METADATA_COLUMNS),
            len(wide_df.columns) - 1,
            8,
            number_format
        )

    print("Done.")
    print(f"Excel file: {OUTPUT_EXCEL}")
    print(f"Long-format CSV: {OUTPUT_LONG_CSV}")
    print(f"Download log: {OUTPUT_LOG}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()