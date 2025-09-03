# -*- coding: utf-8 -*-
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ========= USER PATHS =========
CSV_PATH   = 'C:/Users/marcu/PycharmProjects/hydrobr_project/Testing/stations_inventory_filtered_all.csv'
FONT_PATH  = 'C:/Users/marcu/PycharmProjects/Sucept_Maps/Montserrat/Montserrat-Regular.ttf'
OUTPUT_PDF = 'C:/Users/marcu/PycharmProjects/hydrobr_project/Testing/rain_gauge_overview.pdf'

# ========= FONT: MONTSERRAT =========
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_name = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = font_name
else:
    print(f"Warning: Montserrat not found at {FONT_PATH}. Using default font.")

# ========= GLOBAL GRAY STYLE =========
plt.rcParams.update({
    "axes.facecolor": "0.96",
    "figure.facecolor": "0.96",
    "savefig.facecolor": "0.96",
    "axes.edgecolor": "0.3",
    "axes.labelcolor": "0.1",
    "text.color": "0.1",
    "xtick.color": "0.2",
    "ytick.color": "0.2",
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.linewidth": 2.0,  # thicker axis border globally
})

def style_axis(ax):
    ax.tick_params(axis="both", which="both", length=6, width=1.2, colors="0.2")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.0)   # thicker borders
        ax.spines[side].set_color("0.3")

# ========= LOAD =========
df = pd.read_csv(CSV_PATH)

# --------- Find state column ---------
def find_state_col(columns):
    candidates = ["UF", "uf", "state", "STATE", "State", "ST", "st", "region", "Region"]
    for c in candidates:
        if c in columns:
            return c
    rx = re.compile(r"(state|province|region|uf|admin).*", re.I)
    for c in columns:
        if rx.fullmatch(str(c)) or rx.search(str(c)):
            return c
    return None

state_col = find_state_col(df.columns)

# --------- Robust name → abbreviation mapping ---------
state_abbrev_norm = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM",
    "bahia": "BA", "ceara": "CE", "distrito federal": "DF", "espirito santo": "ES",
    "goias": "GO", "maranhao": "MA", "mato grosso": "MT", "mato grosso do sul": "MS",
    "minas gerais": "MG", "para": "PA", "paraiba": "PB", "parana": "PR",
    "pernambuco": "PE", "piaui": "PI", "rio de janeiro": "RJ",
    "rio grande do norte": "RN", "rio grande do sul": "RS",
    "rondonia": "RO", "roraima": "RR", "santa catarina": "SC",
    "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}

def deaccent(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def normalize_name(x) -> str:
    x = str(x)
    x = deaccent(x)
    x = x.strip().lower()
    x = ' '.join(x.split())
    return x

# Create a plotting column with abbreviations if possible
state_plot_col = None
if state_col is not None:
    s = df[state_col].astype(str)
    # If already 2-letter codes for most rows, use as-is
    if s.str.fullmatch(r"[A-Za-z]{2}").mean() >= 0.9:
        state_plot_col = state_col
    else:
        normalized = s.map(normalize_name)
        mapped = normalized.map(state_abbrev_norm)
        df["STATE_ABBR"] = mapped.fillna(s)  # keep original if not mapped
        state_plot_col = "STATE_ABBR"

# --------- Assure numeric columns we care about ---------
num_targets = ["NYD", "MD", "N_YWOMD", "YWMD"]
for c in num_targets:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        print(f"Note: column '{c}' not found.")

# --------- Detect start/end date/year columns ---------
def guess_year_series(frame, type_="start"):
    cols = list(frame.columns)
    patterns = {
        "start": [r"^start", r"^begin", r"first", r"^from", r"^year_start", r"^start_year"],
        "end":   [r"^end", r"last", r"^to", r"^year_end", r"^end_year"]
    }[type_]

    for c in cols:
        name = str(c).lower()
        if any(re.search(p, name) for p in patterns) and "year" in name:
            s = pd.to_numeric(frame[c], errors="coerce")
            if s.notna().sum() > 0:
                return s

    for c in cols:
        name = str(c).lower()
        if any(re.search(p, name) for p in patterns):
            try:
                dt = pd.to_datetime(frame[c], errors="coerce", infer_datetime_format=True, utc=True)
                years = dt.dt.year
                if years.notna().sum() > 0:
                    return years
            except Exception:
                pass

    fallback_map = {"start": ["FIRSTYEAR", "FirstYear", "firstyear"],
                    "end":   ["LASTYEAR", "LastYear", "lastyear"]}
    for c in fallback_map[type_]:
        if c in frame.columns:
            s = pd.to_numeric(frame[c], errors="coerce")
            if s.notna().sum() > 0:
                return s

    return pd.Series(index=frame.index, dtype="float64")

start_year = guess_year_series(df, "start")
end_year   = guess_year_series(df, "end")

# ========= PLOTTING (3×3 layout; 7 charts used) =========
fig = plt.figure(figsize=(15, 12), dpi=120)
axes = [plt.subplot(3, 3, i) for i in range(1, 10)]

# (1) Stations per state (UFs), **no x-label**
ax = axes[0]
if state_plot_col is not None:
    counts = df[state_plot_col].astype("category").value_counts().sort_values(ascending=False)
    counts.plot(kind="bar", ax=ax, color="0.7", edgecolor="0.2", linewidth=0.8)
    ax.set_title("Stations per State", pad=10)
    ax.set_xlabel(None)  # <-- removed x-label as requested
    ax.set_ylabel("Count")
    if counts.size > 15:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
else:
    ax.text(0.5, 0.5, "No state column found", ha="center", va="center")
style_axis(ax)

# (2-5) Histograms of NYD, MD, N_YWOMD, YWMD
for i, col in enumerate(num_targets, start=1):
    ax = axes[i]
    if col in df.columns:
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        if data.size:
            ax.hist(data, bins="auto", color="0.7", edgecolor="0.2", linewidth=0.8)
            ax.set_title(f"Histogram: {col}", pad=10)
            ax.set_xlabel(col); ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, f"No data in '{col}'", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, f"'{col}' not in file", ha="center", va="center")
    style_axis(ax)

# (6) Start year histogram
ax = axes[5]
if start_year.notna().sum() > 0:
    ax.hist(start_year.dropna(), bins="auto", color="0.7", edgecolor="0.2", linewidth=0.8)
    ax.set_title("Histogram: Start Year", pad=10)
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
else:
    ax.text(0.5, 0.5, "No start-year column found", ha="center", va="center")
style_axis(ax)

# (7) End year histogram
ax = axes[6]
if end_year.notna().sum() > 0:
    ax.hist(end_year.dropna(), bins="auto", color="0.7", edgecolor="0.2", linewidth=0.8)
    ax.set_title("Histogram: End Year", pad=10)
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
else:
    ax.text(0.5, 0.5, "No end-year column found", ha="center", va="center")
style_axis(ax)

# (8) Coverage length (if possible)
ax = axes[7]
if start_year.notna().sum() > 0 and end_year.notna().sum() > 0:
    span = (end_year - start_year).dropna()
    span = span[span >= 0]
    if span.size:
        ax.hist(span, bins="auto", color="0.7", edgecolor="0.2", linewidth=0.8)
        ax.set_title("Histogram: Record Length (years)", pad=10)
        ax.set_xlabel("Years"); ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "Invalid spans (end < start)", ha="center", va="center")
else:
    ax.text(0.5, 0.5, "Need both start & end year", ha="center", va="center")
style_axis(ax)

# (9) Summary stats
ax = axes[8]
preview_cols = [c for c in num_targets if c in df.columns]
if preview_cols:
    stats = df[preview_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    ax.axis("off")
    ax.table(cellText=np.round(stats.values, 3),
             rowLabels=stats.index,
             colLabels=stats.columns,
             loc="center", cellLoc="right")
    ax.set_title("Summary stats (NYD/MD/N_YWOMD/YWMD)", pad=10)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "No NYD/MD/N_YWOMD/YWMD columns to summarize",
            ha="center", va="center")

plt.tight_layout()

# --------- SAVE TO PDF ---------
plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
print(f"Saved PDF to: {OUTPUT_PDF}")

plt.show()
