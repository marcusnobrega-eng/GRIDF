from hydrobr.get_data import ANA
import pandas as pd
import plotly.express as px

# ===== USER INPUT =====
STATE = "Minas Gerais"
CITY  = "Belo Horizonte"
ONLY_CONSISTED = False     # True => ANA “consisted” (QC) data only
START_DATE = None          # e.g. "2000-01-01" or None
END_DATE   = None          # e.g. "2020-12-31" or None
# ======================

# 1) List precipitation stations
stations = ANA.list_prec(state=STATE, city=CITY, source="ANA")
if stations is None or stations.empty:
    raise SystemExit(f"No precipitation stations found for {CITY}, {STATE}.")

# 2) Find the first station that successfully returns non-NaN data
selected = None
series = None

for _, row in stations.iterrows():
    code = str(row["Code"])
    name = row.get("Name", code)

    try:
        # Important: call per-station and guard against “No objects to concatenate”
        df = ANA.prec([code], only_consisted=ONLY_CONSISTED)
    except Exception as e:
        # e.g., ValueError: No objects to concatenate – skip this station
        print(f"Skipping {code} ({name}) due to error: {e}")
        continue

    if df is None or df.empty or code not in df.columns:
        print(f"Skipping {code} ({name}) – empty frame or missing column.")
        continue

    s = pd.to_numeric(df[code], errors="coerce")
    if START_DATE:
        s = s[s.index >= pd.to_datetime(START_DATE)]
    if END_DATE:
        s = s[s.index <= pd.to_datetime(END_DATE)]

    if s.dropna().empty:
        print(f"Skipping {code} ({name}) – all NaN after filtering.")
        continue

    # Found one with data
    selected = (code, name)
    series = s
    break

if selected is None:
    raise SystemExit("No stations with available precipitation data were found.")

code, name = selected
print(f"Selected station: {name} (Code {code})")

# 3) Plot daily rainfall
plot_df = series.reset_index()
plot_df.columns = ["Date", "Rain_mm_day"]

fig = px.line(
    plot_df, x="Date", y="Rain_mm_day",
    title=f"Daily Rainfall — {name} (Code {code}) • {CITY}/{STATE}"
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (mm/day)",
    margin=dict(l=40, r=20, t=60, b=40)
)
fig.show()

# 4) Save to CSV
output_file = f"rainfall_{code}.csv"

# Reset index so 'Date' is a column
out_df = series.reset_index()
out_df.columns = ["Date", "Rain_mm_day"]

# Export
out_df.to_csv(output_file, index=False, float_format="%.2f")
print(f"Data exported to {output_file}")

# === Annual maximum daily rainfall (per calendar year) ===
# Clean & prepare
series = pd.to_numeric(series, errors="coerce").clip(lower=0)
series = series.dropna()

# Resample by calendar year (ending in December)
annual = series.resample("YE-DEC")

# Max value each year
annual_max = annual.max()

# Date of the max each year (use apply to call idxmax on each group)
annual_when = annual.apply(pd.Series.idxmax)
# alternatively: annual_when = annual.apply(lambda s: s.idxmax())

# Build tidy table
annual_df = pd.DataFrame({
    "Year": annual_max.index.year,
    "MaxDaily_mm": annual_max.values,
    "DateOfMax": annual_when.values
})

print(annual_df.head())

# (Optional) quick plot
fig = px.line(
    annual_df, x="Year", y="MaxDaily_mm",
    markers=True,
    title=f"Annual Maximum Daily Rainfall — {name} (Code {code}) • {CITY}/{STATE}"
)
fig.update_layout(xaxis_title="Year", yaxis_title="Max daily rainfall (mm/day)")
fig.show()

# (Optional) export
annual_df.to_csv(f"annual_max_daily_{code}.csv", index=False, float_format="%.2f")
print(f"Exported annual maxima to annual_max_daily_{code}.csv")

