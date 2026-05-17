import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# Input and output paths
# ============================================================
input_dir = (
    "/Users/mngomes/Documents/GitHub/GRIDF/"
    "Zonal_Disaggregation_Coefficients_FINAL/CSV/relative_to_subdaily"
)

output_dir = "/Users/mngomes/Documents/GitHub/GRIDF/Disag_Coefficients/Biases"
os.makedirs(output_dir, exist_ok=True)

biomes_file = os.path.join(input_dir, "Biomes_relative_to_subdaily_coefficients.csv")
brazil_file = os.path.join(input_dir, "Brazil_relative_to_subdaily_coefficients.csv")
cities_file = os.path.join(input_dir, "Cities_relative_to_subdaily_coefficients.csv")

# ============================================================
# Font
# ============================================================
font_path = "/Users/mngomes/Downloads/AvenirNextCyr-Regular.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    print(f"Warning: font file not found at {font_path}. Using Matplotlib default font.")

# Keep SVG text editable
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# Matplotlib style
# ============================================================
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 2.0

plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7

plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120

# ============================================================
# CETESB reference values shown in the table
# ============================================================
cetesb = {
    "R_5m_30m": 0.34,
    "R_10m_30m": 0.54,
    "R_15m_30m": 0.70,
    "R_20m_30m": 0.81,
    "R_25m_30m": 0.91,
    "R_30m_1h": 0.74,
    "R_1h_24h": 0.42,
    "R_6h_24h": 0.72,
    "R_8h_24h": 0.78,
    "R_10h_24h": 0.82,
    "R_12h_24h": 0.85,
    "R_24h_1dia": 1.14,
}

coef_order = [
    "R_5m_30m",
    "R_10m_30m",
    "R_15m_30m",
    "R_20m_30m",
    "R_25m_30m",
    "R_30m_1h",
    "R_1h_24h",
    "R_6h_24h",
    "R_8h_24h",
    "R_10h_24h",
    "R_12h_24h",
    "R_24h_1dia",
]

coef_labels = {
    "R_5m_30m": "5min/30min",
    "R_10m_30m": "10min/30min",
    "R_15m_30m": "15min/30min",
    "R_20m_30m": "20min/30min",
    "R_25m_30m": "25min/30min",
    "R_30m_1h": "30min/1h",
    "R_1h_24h": "1h/24h",
    "R_6h_24h": "6h/24h",
    "R_8h_24h": "8h/24h",
    "R_10h_24h": "10h/24h",
    "R_12h_24h": "12h/24h",
    "R_24h_1dia": "24h/1dia",
}

# Durations in hours, used to compute the integral of absolute bias
duration_hours = np.array([
    5 / 60,
    10 / 60,
    15 / 60,
    20 / 60,
    25 / 60,
    30 / 60,
    1,
    6,
    8,
    10,
    12,
    24,
])

# ============================================================
# Nature-inspired distinguishable colors
# ============================================================
region_colors = {
    "Caatinga": "#B8860B",          # golden earth
    "Cerrado": "#6B8E23",           # savanna olive
    "Pantanal": "#1F78B4",          # wetland blue
    "Pampa": "#8C6D31",             # grassland brown
    "Amazonia": "#1B7837",          # forest green
    "Mata Atlantica": "#5AB4AC",    # Atlantic teal
    "National average": "#2F2F2F",  # charcoal
}

city_colors = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple-pink
    "#E69F00",  # golden orange
]

# ============================================================
# Helper functions
# ============================================================
def compute_bias_table(df, region_col="region_name"):
    """
    Compute bias using the same logic as the LaTeX table:
    GRIDF mean is rounded to 2 decimals first, then bias is computed.

    Bias = (GRIDF_2dec - CETESB) / CETESB * 100
    """

    df = df.copy()

    df["gridf_2dec"] = df["mean"].round(2)
    df["cetesb"] = df["coefficient_name"].map(cetesb)

    missing_ref = df[df["cetesb"].isna()]["coefficient_name"].unique()
    if len(missing_ref) > 0:
        raise ValueError(f"Missing CETESB reference values for: {missing_ref}")

    df["bias_percent"] = (
        (df["gridf_2dec"] - df["cetesb"]) / df["cetesb"]
    ) * 100

    df["bias_percent"] = df["bias_percent"].round(0)

    bias_table = (
        df.pivot_table(
            index=region_col,
            columns="coefficient_name",
            values="bias_percent",
            aggfunc="mean"
        )
        .reindex(columns=coef_order)
    )

    bias_table.columns = [coef_labels[c] for c in coef_order]

    return bias_table


def integrated_absolute_bias(row, x_values):
    """
    Integral of absolute bias across the duration sequence.
    Absolute values are used so positive and negative biases do not cancel.
    """

    y = row.values.astype(float)
    valid = ~np.isnan(y)

    if valid.sum() < 2:
        return np.nan

    try:
        return np.trapezoid(np.abs(y[valid]), x=x_values[valid])
    except AttributeError:
        return np.trapz(np.abs(y[valid]), x=x_values[valid])


def style_axes(ax):
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=7,
        width=2.0,
        top=False,
        right=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    ax.grid(axis="y", linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)


def save_figure(fig, filename_base):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(output_dir, f"{filename_base}.{ext}"),
            bbox_inches="tight"
        )


# ============================================================
# Read datasets
# ============================================================
df_biomes = pd.read_csv(biomes_file)
df_brazil = pd.read_csv(brazil_file)
df_cities = pd.read_csv(cities_file)

# ============================================================
# Plot 1: biomes + national average
# ============================================================
df_brazil = df_brazil.copy()
df_brazil["region_name"] = df_brazil["region_name"].replace({
    "Brasil": "National average",
    "Brazil": "National average",
    "Brazil average": "National average",
    "Brazil national average": "National average",
})

df_regions = pd.concat([df_biomes, df_brazil], ignore_index=True)

bias_regions = compute_bias_table(df_regions, region_col="region_name")

region_order = [
    "Caatinga",
    "Cerrado",
    "Pantanal",
    "Pampa",
    "Amazonia",
    "Mata Atlantica",
    "National average",
]

bias_regions = bias_regions.reindex(region_order)

bias_regions.to_csv(
    os.path.join(output_dir, "biases_biomes_and_national_average.csv")
)

fig, ax = plt.subplots(figsize=(13.5, 6.2))

x = np.arange(len(bias_regions.columns))

for region in bias_regions.index:
    linewidth = 3.4 if region == "National average" else 2.5
    markersize = 8.5 if region == "National average" else 6.8
    zorder = 10 if region == "National average" else 5

    ax.plot(
        x,
        bias_regions.loc[region].values,
        marker="o",
        linewidth=linewidth,
        markersize=markersize,
        color=region_colors.get(region, "#333333"),
        label=region,
        zorder=zorder
    )

ax.axhline(
    0,
    color="black",
    linewidth=1.5,
    linestyle="--",
    alpha=0.8,
    zorder=2
)

ax.set_xticks(x)
ax.set_xticklabels(bias_regions.columns, rotation=45, ha="right")

ax.set_ylabel("Bias relative to CETESB (%)")
ax.set_xlabel("Time relation")
ax.set_title("Bias of GRIDF sub-daily rainfall coefficients by biome")

style_axes(ax)

# Legend inside the plotting area
ax.legend(
    frameon=True,
    framealpha=0.88,
    facecolor="white",
    edgecolor="none",
    fontsize=9.5,
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(0.015, 0.985),
    handlelength=2.4,
    borderpad=0.6,
    labelspacing=0.5
)

plt.tight_layout()

save_figure(fig, "bias_biomes_national_average")

plt.show()

# ============================================================
# Plot 2: top 5 cities with largest integrated absolute bias
# ============================================================
df_cities = df_cities.copy()

# Create unique city labels
if "state_name" in df_cities.columns:
    df_cities["city_label"] = (
        df_cities["region_name"].astype(str)
        + " ("
        + df_cities["state_name"].astype(str)
        + ")"
    )
else:
    df_cities["city_label"] = df_cities["region_name"].astype(str)

bias_cities = compute_bias_table(df_cities, region_col="city_label")

# Keep only cities with all coefficients available
bias_cities_complete = bias_cities.dropna(how="any")

city_integrals = bias_cities_complete.apply(
    integrated_absolute_bias,
    axis=1,
    x_values=duration_hours
)

city_integrals = city_integrals.sort_values(ascending=False)

top5_cities = city_integrals.head(5).index.tolist()
bias_top5_cities = bias_cities_complete.loc[top5_cities]

city_ranking = city_integrals.rename("integrated_absolute_bias").reset_index()
city_ranking.to_csv(
    os.path.join(output_dir, "city_bias_integral_ranking.csv"),
    index=False
)

bias_top5_cities.to_csv(
    os.path.join(output_dir, "top5_city_biases.csv")
)

fig, ax = plt.subplots(figsize=(13.5, 6.2))

x = np.arange(len(bias_top5_cities.columns))

for i, city in enumerate(bias_top5_cities.index):
    integral_value = city_integrals.loc[city]

    ax.plot(
        x,
        bias_top5_cities.loc[city].values,
        marker="o",
        linewidth=3.0,
        markersize=7.2,
        color=city_colors[i],
        label=f"{city} | IAB = {integral_value:.1f}",
        zorder=6
    )

ax.axhline(
    0,
    color="black",
    linewidth=1.5,
    linestyle="--",
    alpha=0.8,
    zorder=2
)

ax.set_xticks(x)
ax.set_xticklabels(bias_top5_cities.columns, rotation=45, ha="right")

ax.set_ylabel("Bias relative to CETESB (%)")
ax.set_xlabel("Time relation")
ax.set_title("Top 5 cities with largest integrated absolute bias")

style_axes(ax)

# Legend inside the plotting area
ax.legend(
    frameon=True,
    framealpha=0.88,
    facecolor="white",
    edgecolor="none",
    fontsize=9.2,
    loc="upper left",
    bbox_to_anchor=(0.015, 0.985),
    handlelength=2.4,
    borderpad=0.6,
    labelspacing=0.5
)

plt.tight_layout()

save_figure(fig, "bias_top5_cities_integrated_absolute_bias")

plt.show()

# ============================================================
# Console summary
# ============================================================
print("\nTop 5 cities ranked by integrated absolute bias:")
print(city_integrals.head(5))

print(f"\nOutputs saved to:\n{output_dir}")