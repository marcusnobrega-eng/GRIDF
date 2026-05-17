from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path(
    "/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction/ana_rainfall_1995_2025"
)

LONG_CSV = OUTPUT_DIR / "rainfall_ANA_1995_2025_long.csv"
LOG_CSV = OUTPUT_DIR / "download_log.csv"

START_DATE = "1995-01-01"
END_DATE = "2025-12-31"

print("Reading files...")
print(f"Looking for rainfall file at: {LONG_CSV}")
print(f"Looking for log file at:      {LOG_CSV}")

if not LONG_CSV.exists():
    raise FileNotFoundError(f"Rainfall file not found: {LONG_CSV}")

if not LOG_CSV.exists():
    raise FileNotFoundError(f"Download log file not found: {LOG_CSV}")

rain = pd.read_csv(LONG_CSV, parse_dates=["Date"], dtype={"Code": str})
log = pd.read_csv(LOG_CSV, dtype={"Code": str})

rain["Rain_mm"] = pd.to_numeric(rain["Rain_mm"], errors="coerce")

print("\n==============================")
print("DOWNLOAD LOG SUMMARY")
print("==============================")

print(log["status"].value_counts(dropna=False))

print("\nTotal stations attempted:", log["Code"].nunique())
print("Total downloaded rows:", len(rain))
print("Stations present in rainfall file:", rain["Code"].nunique())

valid = rain.dropna(subset=["Rain_mm"]).copy()

print("\nStations with at least one valid rainfall value:", valid["Code"].nunique())
print("Stations with no valid rainfall values:", log["Code"].nunique() - valid["Code"].nunique())

print("\nOverall date range:")
print("Start:", rain["Date"].min())
print("End:  ", rain["Date"].max())

print("\nRainfall value summary:")
print(valid["Rain_mm"].describe())

print("\nLargest rainfall values:")
print(
    valid.sort_values("Rain_mm", ascending=False)
    .head(20)
    [["Code", "Date", "Rain_mm"]]
)

print("\n==============================")
print("COVERAGE SUMMARY")
print("==============================")

total_days = len(pd.date_range(START_DATE, END_DATE, freq="D"))

coverage = (
    valid
    .drop_duplicates(subset=["Code", "Date"])
    .groupby("Code")
    .agg(
        actual_start=("Date", "min"),
        actual_end=("Date", "max"),
        valid_days=("Date", "count"),
        mean_rain_mm=("Rain_mm", "mean"),
        max_rain_mm=("Rain_mm", "max"),
    )
    .reset_index()
)

all_codes = pd.DataFrame({
    "Code": log["Code"].drop_duplicates()
})

coverage = all_codes.merge(
    coverage,
    on="Code",
    how="left"
)

coverage["valid_days"] = coverage["valid_days"].fillna(0).astype(int)
coverage["total_days"] = total_days
coverage["missing_days"] = coverage["total_days"] - coverage["valid_days"]
coverage["coverage_percent"] = 100 * coverage["valid_days"] / coverage["total_days"]

coverage["has_any_data"] = coverage["valid_days"] > 0
coverage["covers_start"] = coverage["actual_start"] <= pd.Timestamp(START_DATE)
coverage["covers_end"] = coverage["actual_end"] >= pd.Timestamp(END_DATE)
coverage["covers_full_period_by_dates"] = (
    coverage["covers_start"] &
    coverage["covers_end"]
)

print("\nCoverage percent summary:")
print(coverage["coverage_percent"].describe())

print("\nStations by coverage threshold:")
for threshold in [0, 50, 70, 80, 90, 95, 99, 100]:
    n = (coverage["coverage_percent"] >= threshold).sum()
    print(f">= {threshold}%: {n}")

print("\nStations covering start date:", coverage["covers_start"].sum())
print("Stations covering end date:", coverage["covers_end"].sum())
print("Stations covering full period by dates:", coverage["covers_full_period_by_dates"].sum())

print("\nTop 20 stations by coverage:")
print(
    coverage.sort_values("coverage_percent", ascending=False)
    .head(20)
)

print("\nBottom 20 stations by coverage:")
print(
    coverage.sort_values("coverage_percent", ascending=True)
    .head(20)
)

out_file = OUTPUT_DIR / "coverage_summary_inspected.csv"
coverage.to_csv(out_file, index=False)

print(f"\nSaved coverage summary to: {out_file}")