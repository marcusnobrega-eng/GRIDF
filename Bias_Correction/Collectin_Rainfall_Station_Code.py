from hydrobr.get_data import ANA
import pandas as pd
import plotly.express as px
import re, numpy as np

# Rainfall for a particular station (code) within a range

# ===== USER INPUT =====
STATION_CODE = 1649013       # can be int, "2549017", or "02549017"
ONLY_CONSISTED = True
START_DATE = "1995-01-01"
END_DATE   = "2020-01-01"
# ======================

def normalize_code(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)      # drop trailing .0 if present (from CSV floats)
    return s.zfill(8)               # <-- pad to 8 digits to match ANA format

code = normalize_code(STATION_CODE)
name = code  # display name (optional)

# 1) Download daily precipitation for the single station
df = ANA.prec([code], only_consisted=ONLY_CONSISTED)

# Find the right column (prefer exact, else match ignoring leading zeros)
col = code
if col not in df.columns:
    col = next((c for c in df.columns if c.lstrip("0") == code.lstrip("0")), None)
    if col is None:
        raise SystemExit(f"No data column matching station code {code} found. Got columns: {list(df.columns)}")

# 2) Prepare the series and apply optional date filters
series = pd.to_numeric(df[col], errors="coerce")
if START_DATE:
    series = series[series.index >= pd.to_datetime(START_DATE)]
if END_DATE:
    series = series[series.index <= pd.to_datetime(END_DATE)]
if series.dropna().empty:
    raise SystemExit("The time series is empty or all-NaN after filtering.")

# 3) Plot daily rainfall
plot_df = series.reset_index()
plot_df.columns = ["Date", "Rain_mm_day"]
fig = px.line(plot_df, x="Date", y="Rain_mm_day", title=f"Daily Rainfall — {name}")
fig.update_layout(xaxis_title="Date", yaxis_title="Rainfall (mm/day)", margin=dict(l=40, r=20, t=60, b=40))
fig.show()

# 4) Save daily CSV
plot_df.to_csv(f"rainfall_{code}.csv", index=False, float_format="%.2f")

# === Annual maximum daily rainfall (per calendar year) ===
series = series.clip(lower=0).dropna()
annual = series.resample("YE-DEC")
annual_max = annual.max()
annual_when = annual.apply(pd.Series.idxmax)

annual_df = pd.DataFrame({
    "Year": annual_max.index.year,
    "MaxDaily_mm": annual_max.values,
    "DateOfMax": annual_when.values
})
annual_df.to_csv(f"annual_max_daily_{code}.csv", index=False, float_format="%.2f")

import plotly.express as px
import pandas as pd

# --- Daily rainfall ---
fig_daily = px.line(
    series.rename("Rain_mm_day").reset_index(),
    x=series.index.name or "index", y="Rain_mm_day",
    title="Daily Rainfall"
)
fig_daily.update_layout(xaxis_title="Date", yaxis_title="Rainfall (mm/day)")
fig_daily.show()

# --- Annual maximum (per calendar year) ---
annual_max = series.resample("YE-DEC").max()
annual_max.index = annual_max.index.year  # use year on x-axis

fig_amax = px.line(
    x=annual_max.index, y=annual_max.values,
    title="Annual Maximum Daily Rainfall", markers=True
)
fig_amax.update_layout(xaxis_title="Year", yaxis_title="Max daily (mm/day)")
fig_amax.show()


