from hydrobr.get_data import ANA
import pandas as pd
import plotly.express as px

# Rainfall within a region following some defined criteria

# ===== USER INPUT =====
STATE = "Minas Gerais"
CITY  = "Belo Horizonte"
ONLY_CONSISTED = False            # True => ANA “consisted” (QC) data only

# Target analysis window
PERIOD_START = "2000-01-01"
PERIOD_END   = "2020-12-31"

# Require at least this many years of daily data (non-NaN) within the window
MIN_YEARS_IN_PERIOD = 10
# ======================

period_start = pd.to_datetime(PERIOD_START)
period_end   = pd.to_datetime(PERIOD_END)
min_days_required = int(MIN_YEARS_IN_PERIOD * 365)  # simple threshold

# 1) List precipitation stations
stations = ANA.list_prec(state=STATE, city=CITY, source="ANA")
if stations is None or stations.empty:
    raise SystemExit(f"No precipitation stations found for {CITY}, {STATE}.")

# 2) Find the first station with sufficient data coverage in [PERIOD_START, PERIOD_END]
selected = None
series = None

for _, row in stations.iterrows():
    code = str(row["Code"])
    name = row.get("Name", code)

    try:
        df = ANA.prec([code], only_consisted=ONLY_CONSISTED)
    except Exception as e:
        print(f"Skipping {code} ({name}) due to error: {e}")
        continue

    if df is None or df.empty or code not in df.columns:
        print(f"Skipping {code} ({name}) – empty frame or missing column.")
        continue

    s = pd.to_numeric(df[code], errors="coerce")

    # Crop to analysis window
    s = s[(s.index >= period_start) & (s.index <= period_end)]

    # Count non-NaN days in the window
    non_na_days = int(s.notna().sum())
    print(f"Station {code} ({name}) coverage in window: {non_na_days} days")

    if non_na_days >= min_days_required:
        selected = (code, name)
        series = s
        break
    else:
        print(f"Skipping {code} ({name}) – insufficient coverage (< {min_days_required} days).")

if selected is None:
    raise SystemExit(
        f"No stations with ≥{MIN_YEARS_IN_PERIOD} years of daily data in "
        f"[{PERIOD_START} .. {PERIOD_END}] for {CITY}, {STATE}."
    )

code, name = selected
print(f"Selected station: {name} (Code {code}) with {int(series.notna().sum())} valid days in window.")

# 3) Plot daily rainfall (within period)
plot_df = series.reset_index()
plot_df.columns = ["Date", "Rain_mm_day"]

fig = px.line(
    plot_df, x="Date", y="Rain_mm_day",
    title=f"Daily Rainfall — {name} (Code {code}) • {CITY}/{STATE} • {PERIOD_START}–{PERIOD_END}"
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (mm/day)",
    margin=dict(l=40, r=20, t=60, b=40)
)
fig.show()

# 4) Save period-cropped daily CSV
output_file = f"rainfall_{code}_{PERIOD_START[:4]}_{PERIOD_END[:4]}.csv"
plot_df.to_csv(output_file, index=False, float_format="%.2f")
print(f"Daily data exported to {output_file}")

# === Annual maximum daily rainfall (per calendar year in the window) ===
series = pd.to_numeric(series, errors="coerce").clip(lower=0).dropna()

# Resample by calendar year (ending in December) within the cropped window
annual = series.resample("YE-DEC")
annual_max = annual.max()
annual_when = annual.apply(pd.Series.idxmax)

annual_df = pd.DataFrame({
    "Year": annual_max.index.year,
    "MaxDaily_mm": annual_max.values,
    "DateOfMax": annual_when.values
}).dropna(subset=["MaxDaily_mm"])

print(annual_df.head())

# Plot annual maxima
fig2 = px.line(
    annual_df, x="Year", y="MaxDaily_mm",
    markers=True,
    title=f"Annual Max Daily Rainfall — {name} (Code {code}) • {CITY}/{STATE} • {PERIOD_START[:4]}–{PERIOD_END[:4]}"
)
fig2.update_layout(xaxis_title="Year", yaxis_title="Max daily rainfall (mm/day)")
fig2.show()

# Export annual maxima
annual_out = f"annual_max_daily_{code}_{PERIOD_START[:4]}_{PERIOD_END[:4]}.csv"
annual_df.to_csv(annual_out, index=False, float_format="%.2f")
print(f"Exported annual maxima to {annual_out}")
