#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRIDF - Final simple zonal extraction of disaggregation coefficients.

This script creates separate outputs for:

    1. relative_to_daily
    2. relative_to_subdaily

For each family, it creates one CSV and one shapefile for:

    1. Brazil
    2. States
    3. Cities
    4. Biomes

For each polygon and each coefficient raster, it computes:

    - Mean coefficient value
    - Spatial standard deviation

Important behavior:

    - Coefficients are ordered from largest duration to shortest duration.
    - Cities table is grouped by city; each city appears once per coefficient,
      and the coefficients are ordered largest to shortest.
    - If a small polygon contains no raster pixel center, the raster value at
      the polygon representative point is used as a fallback.

CSV columns:

    Brazil / States / Biomes:
        region_name
        coefficient_name
        mean
        std

    Cities:
        region_name
        state_name
        biome_name
        coefficient_name
        mean
        std

Shapefile columns:

    Brazil / States / Biomes:
        reg_name
        coefficient fields
        geometry

    Cities:
        reg_name
        state_name
        biome_name
        coefficient fields
        geometry

Example coefficient fields:

    P24h_M, P24h_SD
    P12h_M, P12h_SD
    P1h_M,  P1h_SD
    P5m_M,  P5m_SD

    R24h_M, R24h_SD
    R12h_M, R12h_SD
    R1h_M,  R1h_SD
    R5m_M,  R5m_SD
"""

from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm


# ============================================================
# 1. USER SETTINGS
# ============================================================

ROOT_DIR = Path("/Users/mngomes/Documents/GitHub/GRIDF")

ADM_DIR = ROOT_DIR / "BrazilShapefiles" / "ADMLevels"
BIOME_DIR = ROOT_DIR / "BrazilShapefiles" / "Biomes"

COEFF_DIR_DAILY = ROOT_DIR / "Disag_Coefficients" / "relative_to_daily"
COEFF_DIR_SUBDAILY = ROOT_DIR / "Disag_Coefficients" / "relative_to_subdaily"

OUTPUT_DIR = ROOT_DIR / "Zonal_Disaggregation_Coefficients_FINAL"
OUTPUT_CSV_DIR = OUTPUT_DIR / "CSV"
OUTPUT_SHP_DIR = OUTPUT_DIR / "Shapefiles"
OUTPUT_GPKG_DIR = OUTPUT_DIR / "GeoPackage"

OUTPUT_CSV_DAILY_DIR = OUTPUT_CSV_DIR / "relative_to_daily"
OUTPUT_CSV_SUBDAILY_DIR = OUTPUT_CSV_DIR / "relative_to_subdaily"

OUTPUT_SHP_DAILY_DIR = OUTPUT_SHP_DIR / "relative_to_daily"
OUTPUT_SHP_SUBDAILY_DIR = OUTPUT_SHP_DIR / "relative_to_subdaily"

for folder in [
    OUTPUT_CSV_DAILY_DIR,
    OUTPUT_CSV_SUBDAILY_DIR,
    OUTPUT_SHP_DAILY_DIR,
    OUTPUT_SHP_SUBDAILY_DIR,
    OUTPUT_GPKG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

SAVE_SHAPEFILES = True
SAVE_GEOPACKAGE = True

DEFAULT_VECTOR_CRS = "EPSG:4326"
DEFAULT_RASTER_CRS = "EPSG:4326"

# False = conservative zonal statistics.
# True = include all pixels touched by each polygon boundary.
ALL_TOUCHED = False


# ============================================================
# 2. INPUT FILES
# ============================================================

BRAZIL_SHP = ADM_DIR / "bra_admbnda_adm0_ibge_2020.shp"
STATES_SHP = ADM_DIR / "bra_admbnda_adm1_ibge_2020.shp"
CITIES_SHP = ADM_DIR / "bra_admbnda_adm2_ibge_2020.shp"

# If this exact file does not exist, the code automatically detects
# the first .shp inside the Biomes folder.
BIOMES_SHP = BIOME_DIR / "biomes.shp"


# ============================================================
# 3. LAYER CONFIGURATION
# ============================================================

LAYERS = {
    "Brazil": {
        "path": BRAZIL_SHP,
        "name_col": "ADM0_PT",
    },
    "States": {
        "path": STATES_SHP,
        "name_col": "ADM1_PT",
    },
    "Cities": {
        "path": CITIES_SHP,
        "name_col": "ADM2_PT",
        "state_col": "ADM1_PT",
    },
    "Biomes": {
        "path": BIOMES_SHP,
        "name_col": None,
    },
}


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def remove_accents(text) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ASCII", "ignore").decode("ASCII")

    return ascii_text


def clean_text(text) -> str:
    text = remove_accents(text)

    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("'", "")
    text = text.replace('"', "")

    text = re.sub(r"[^A-Za-z0-9 _\-.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_filename_text(text) -> str:
    text = clean_text(text)
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)

    return text


# ============================================================
# 5. BASIC HELPERS
# ============================================================

def check_path(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def find_biome_shapefile() -> Path:
    shp_files = sorted(BIOME_DIR.glob("*.shp"))

    if len(shp_files) == 0:
        raise FileNotFoundError(f"No biome shapefile found in:\n{BIOME_DIR}")

    preferred = [
        shp for shp in shp_files
        if "biome" in shp.name.lower() or "bioma" in shp.name.lower()
    ]

    if preferred:
        return preferred[0]

    return shp_files[0]


def list_rasters(folder: Path) -> list[Path]:
    check_path(folder, "Raster folder")

    rasters = sorted(folder.glob("*.tif"))

    if len(rasters) == 0:
        raise FileNotFoundError(f"No .tif rasters found in:\n{folder}")

    return rasters


def get_coefficient_name(raster_path: Path) -> str:
    """
    Example:
        IDW_P24h_Pday_res0.100_k10_p2.0.tif -> P24h_Pday
        IDW_R_24h_1dia_res0.100_k10_p2.0.tif -> R_24h_1dia
    """

    name = raster_path.stem
    name = re.sub(r"^IDW_", "", name)
    name = re.sub(r"_res.*$", "", name)
    name = clean_filename_text(name)

    return name


def duration_to_minutes(duration: str) -> float:
    """
    Convert duration strings to minutes.

    Examples:
        5m    -> 5
        30m   -> 30
        1h    -> 60
        24h   -> 1440
        1dia  -> 1440
        Pday  -> 1440
    """

    duration = str(duration).lower().strip()

    duration = duration.replace("pday", "1dia")
    duration = duration.replace("day", "1dia")
    duration = duration.replace("dia", "dia")

    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)(m|h|dia)$", duration)

    if match is None:
        return -1

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "m":
        return value
    if unit == "h":
        return value * 60.0
    if unit == "dia":
        return value * 1440.0

    return -1


def coefficient_sort_key(raster_path: Path, family: str) -> float:
    """
    Sort rasters from largest duration to shortest duration.

    Daily-relative:
        P24h_Pday -> 24h
        P1h_Pday  -> 1h
        P5m_Pday  -> 5m

    Subdaily-relative:
        R_24h_1dia -> 24h
        R_12h_24h  -> 12h
        R_30m_1h   -> 30m
        R_5m_30m   -> 5m
    """

    coeff = get_coefficient_name(raster_path)

    if family == "relative_to_daily":
        # P24h_Pday -> P24h -> 24h
        first = coeff.split("_")[0]
        duration = first.replace("P", "")

    elif family == "relative_to_subdaily":
        # R_24h_1dia -> 24h
        parts = coeff.split("_")
        if len(parts) >= 2 and parts[0] == "R":
            duration = parts[1]
        else:
            duration = coeff

    else:
        duration = coeff

    return duration_to_minutes(duration)


def sort_rasters_largest_to_shortest(rasters: list[Path], family: str) -> list[Path]:
    """
    Sort rasters by duration from largest to shortest.
    """

    return sorted(
        rasters,
        key=lambda p: coefficient_sort_key(p, family),
        reverse=True,
    )


def get_short_field_base(raster_path: Path, family: str) -> str:
    """
    Create simple shapefile field base names.

    Daily-relative:
        P24h_Pday -> P24h

    Subdaily-relative:
        R_24h_1dia -> R24h
        R_30m_1h   -> R30m
    """

    coeff = get_coefficient_name(raster_path)

    if family == "relative_to_daily":
        base = coeff.split("_")[0]

    elif family == "relative_to_subdaily":
        parts = coeff.split("_")
        if len(parts) >= 2 and parts[0] == "R":
            base = "R" + parts[1]
        else:
            base = coeff.replace("_", "")

    else:
        base = coeff.replace("_", "")

    base = clean_filename_text(base)

    # Shapefile field limit is 10 characters.
    # Need room for "_M" or "_SD".
    base = base[:7]

    return base


def detect_biome_name_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "Bioma",
        "BIOMA",
        "bioma",
        "NOME",
        "Nome",
        "nome",
        "NM_BIOMA",
        "NM_BIOM",
        "CD_Bioma",
        "CD_BIOMA",
        "name",
        "NAME",
    ]

    for col in candidates:
        if col in gdf.columns:
            return col

    object_cols = [
        col for col in gdf.columns
        if col != "geometry" and gdf[col].dtype == "object"
    ]

    if object_cols:
        warnings.warn(
            f"No standard biome name field found. Using: {object_cols[0]}"
        )
        return object_cols[0]

    warnings.warn("No biome name field found. Using polygon index.")
    return "__index__"


def make_unique_fields(existing_fields: set, base: str) -> tuple[str, str]:
    """
    Create unique mean and std field names.

    Example:
        base = P24h
        mean field = P24h_M
        std field  = P24h_SD
    """

    mean_field = f"{base}_M"
    std_field = f"{base}_SD"

    if mean_field not in existing_fields and std_field not in existing_fields:
        existing_fields.add(mean_field)
        existing_fields.add(std_field)
        return mean_field, std_field

    counter = 2

    while True:
        suffix = str(counter)
        base2 = base[: max(1, 7 - len(suffix))] + suffix

        mean_field = f"{base2}_M"
        std_field = f"{base2}_SD"

        if mean_field not in existing_fields and std_field not in existing_fields:
            existing_fields.add(mean_field)
            existing_fields.add(std_field)
            return mean_field, std_field

        counter += 1


# ============================================================
# 6. BIOME CONTEXT FOR CITIES
# ============================================================

def load_biome_reference() -> gpd.GeoDataFrame:
    path = BIOMES_SHP

    if not path.exists():
        path = find_biome_shapefile()

    check_path(path, "Biome shapefile")

    biome_gdf = gpd.read_file(path)

    if biome_gdf.empty:
        raise ValueError("Biome shapefile is empty.")

    if biome_gdf.crs is None:
        print("WARNING: Biome shapefile has no CRS.")
        print(f"Assigning {DEFAULT_VECTOR_CRS}.")
        biome_gdf = biome_gdf.set_crs(DEFAULT_VECTOR_CRS, allow_override=True)

    biome_gdf = biome_gdf[biome_gdf.geometry.notnull()].copy()

    invalid_count = (~biome_gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        print(f"Fixing {invalid_count} invalid geometries in Biomes reference...")
        biome_gdf["geometry"] = biome_gdf.geometry.buffer(0)

    biome_name_col = detect_biome_name_column(biome_gdf)

    if biome_name_col == "__index__":
        biome_names = [f"Biome_{i + 1}" for i in range(len(biome_gdf))]
    else:
        biome_names = biome_gdf[biome_name_col].astype(str).tolist()

    biome_gdf["biome_name"] = [clean_text(x) for x in biome_names]

    biome_gdf = biome_gdf[["biome_name", "geometry"]].copy()

    return biome_gdf


def add_biome_name_to_cities(city_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add biome_name to the Cities layer using representative points.
    """

    print("Assigning biome_name to cities using representative points...")

    biome_gdf = load_biome_reference()
    biome_gdf = biome_gdf.to_crs(city_gdf.crs)

    city_points = city_gdf[["reg_name", "state_name", "geometry"]].copy()
    city_points["geometry"] = city_points.geometry.representative_point()

    joined = gpd.sjoin(
        city_points,
        biome_gdf,
        how="left",
        predicate="within",
    )

    joined = joined[~joined.index.duplicated(keep="first")]

    biome_names = joined["biome_name"].reindex(city_gdf.index)

    city_gdf = city_gdf.copy()
    city_gdf["biome_name"] = biome_names.fillna("Unknown").apply(clean_text)

    n_unknown = (city_gdf["biome_name"] == "Unknown").sum()
    if n_unknown > 0:
        print(f"WARNING: {n_unknown} cities did not match a biome polygon.")

    return city_gdf


# ============================================================
# 7. LOAD VECTOR LAYER
# ============================================================

def load_layer(layer_name: str, cfg: dict) -> gpd.GeoDataFrame:
    path = cfg["path"]

    if layer_name == "Biomes" and not path.exists():
        path = find_biome_shapefile()

    check_path(path, f"{layer_name} shapefile")

    print("\n" + "-" * 80)
    print(f"Reading {layer_name}")
    print(f"File: {path}")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"{layer_name} shapefile is empty.")

    if gdf.crs is None:
        print(f"WARNING: {layer_name} shapefile has no CRS.")
        print(f"Assigning {DEFAULT_VECTOR_CRS}.")
        gdf = gdf.set_crs(DEFAULT_VECTOR_CRS, allow_override=True)

    gdf = gdf[gdf.geometry.notnull()].copy()

    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        print(f"Fixing {invalid_count} invalid geometries in {layer_name}...")
        gdf["geometry"] = gdf.geometry.buffer(0)

    name_col = cfg["name_col"]

    if layer_name == "Biomes" and name_col is None:
        name_col = detect_biome_name_column(gdf)

    if name_col == "__index__":
        region_names = [f"{layer_name}_{i + 1}" for i in range(len(gdf))]
    else:
        if name_col not in gdf.columns:
            raise KeyError(
                f"Name column '{name_col}' not found in {layer_name}.\n"
                f"Available columns:\n{list(gdf.columns)}"
            )
        region_names = gdf[name_col].astype(str).tolist()

    gdf["reg_name"] = [clean_text(x) for x in region_names]

    if layer_name == "Cities":
        state_col = cfg.get("state_col", "ADM1_PT")

        if state_col not in gdf.columns:
            raise KeyError(
                f"State column '{state_col}' not found in Cities shapefile.\n"
                f"Available columns:\n{list(gdf.columns)}"
            )

        gdf["state_name"] = gdf[state_col].astype(str).apply(clean_text)

        gdf = gdf[["reg_name", "state_name", "geometry"]].copy()
        gdf = add_biome_name_to_cities(gdf)
        gdf = gdf[["reg_name", "state_name", "biome_name", "geometry"]].copy()

    else:
        gdf = gdf[["reg_name", "geometry"]].copy()

    print(f"Number of polygons: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Name field used: {name_col}")

    if layer_name == "Cities":
        print(f"State field used: {cfg.get('state_col', 'ADM1_PT')}")
        print("Biome field added from biome shapefile.")

    print("Names simplified to ASCII-safe text.")

    return gdf


# ============================================================
# 8. RASTER POINT FALLBACK
# ============================================================

def sample_raster_at_points(
    raster_path: Path,
    points_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Sample raster values at point locations.

    Used when a polygon is too small to contain any raster pixel center.
    """

    sampled_values = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        if raster_crs is None:
            raster_crs = DEFAULT_RASTER_CRS

        points_raster_crs = points_gdf.to_crs(raster_crs)
        coords = [(geom.x, geom.y) for geom in points_raster_crs.geometry]

        for value in src.sample(coords):
            v = value[0]

            if nodata is not None and np.isclose(v, nodata):
                sampled_values.append(np.nan)
            else:
                sampled_values.append(float(v))

    return np.array(sampled_values, dtype=float)


# ============================================================
# 9. ZONAL STATISTICS FOR ONE FAMILY
# ============================================================

def compute_family_for_layer(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    raster_paths: list[Path],
    family: str,
):
    """
    Compute mean and standard deviation for one layer and one coefficient family.

    Output:
        - one GeoDataFrame for this family
        - one CSV DataFrame for this family

    The CSV is grouped by region first, then coefficients from largest to shortest.
    """

    gdf_out = gdf.copy()
    csv_records = []

    existing_fields = set(gdf_out.columns)

    point_gdf = gdf.copy()
    point_gdf["geometry"] = point_gdf.geometry.representative_point()

    print("\n" + "=" * 80)
    print(f"Layer: {layer_name}")
    print(f"Coefficient family: {family}")
    print("=" * 80)

    # Store all coefficient results first so CSV can be written grouped by city/region.
    coefficient_results = []

    for raster_path in tqdm(raster_paths, desc=f"{layer_name} | {family}"):

        coefficient_name = get_coefficient_name(raster_path)
        base_field = get_short_field_base(raster_path, family)
        mean_field, std_field = make_unique_fields(existing_fields, base_field)

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            nodata = src.nodata

        if raster_crs is None:
            print(f"WARNING: Raster has no CRS: {raster_path.name}")
            print(f"Assuming {DEFAULT_RASTER_CRS}.")
            raster_crs = DEFAULT_RASTER_CRS

        gdf_raster_crs = gdf.to_crs(raster_crs)

        stats = zonal_stats(
            vectors=gdf_raster_crs.geometry,
            raster=str(raster_path),
            stats=["mean", "std"],
            nodata=nodata,
            all_touched=ALL_TOUCHED,
            geojson_out=False,
        )

        mean_values = np.array(
            [s["mean"] if s["mean"] is not None else np.nan for s in stats],
            dtype=float,
        )

        std_values = np.array(
            [s["std"] if s["std"] is not None else np.nan for s in stats],
            dtype=float,
        )

        # --------------------------------------------------------
        # Fallback for polygons with no raster pixels
        # --------------------------------------------------------
        missing_mask = np.isnan(mean_values)

        if np.any(missing_mask):
            sampled_values = sample_raster_at_points(raster_path, point_gdf)

            n_fixed = 0

            for idx in np.where(missing_mask)[0]:
                sampled_value = sampled_values[idx]

                if not np.isnan(sampled_value):
                    mean_values[idx] = sampled_value
                    std_values[idx] = 0.0
                    n_fixed += 1

            if n_fixed > 0:
                print(
                    f"  Fallback applied for {n_fixed} polygons in {raster_path.name}: "
                    f"used representative-point pixel value."
                )

        gdf_out[mean_field] = mean_values
        gdf_out[std_field] = std_values

        coefficient_results.append(
            {
                "coefficient_name": coefficient_name,
                "mean_values": mean_values,
                "std_values": std_values,
            }
        )

    # --------------------------------------------------------
    # Build CSV grouped by region first, coefficients second
    # --------------------------------------------------------
    for row in range(len(gdf)):
        for result in coefficient_results:

            record = {
                "region_name": gdf.iloc[row]["reg_name"],
                "coefficient_name": result["coefficient_name"],
                "mean": result["mean_values"][row],
                "std": result["std_values"][row],
            }

            if layer_name == "Cities":
                record = {
                    "region_name": gdf.iloc[row]["reg_name"],
                    "state_name": gdf.iloc[row]["state_name"],
                    "biome_name": gdf.iloc[row]["biome_name"],
                    "coefficient_name": result["coefficient_name"],
                    "mean": result["mean_values"][row],
                    "std": result["std_values"][row],
                }

            csv_records.append(record)

    csv_df = pd.DataFrame(csv_records)

    return gdf_out, csv_df


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

def save_family_outputs(
    layer_name: str,
    family: str,
    gdf: gpd.GeoDataFrame,
    csv_df: pd.DataFrame,
):
    """
    Save one CSV, one shapefile, and one GeoPackage layer for a layer/family pair.
    """

    safe_layer = clean_filename_text(layer_name)
    safe_family = clean_filename_text(family)

    if family == "relative_to_daily":
        csv_dir = OUTPUT_CSV_DAILY_DIR
        shp_dir = OUTPUT_SHP_DAILY_DIR
    elif family == "relative_to_subdaily":
        csv_dir = OUTPUT_CSV_SUBDAILY_DIR
        shp_dir = OUTPUT_SHP_SUBDAILY_DIR
    else:
        csv_dir = OUTPUT_CSV_DIR
        shp_dir = OUTPUT_SHP_DIR

    csv_path = csv_dir / f"{safe_layer}_{safe_family}_coefficients.csv"
    shp_path = shp_dir / f"{safe_layer}_{safe_family}_coefficients.shp"
    gpkg_path = OUTPUT_GPKG_DIR / "GRIDF_disaggregation_coefficients.gpkg"
    gpkg_layer = f"{safe_layer}_{safe_family}"

    if layer_name == "Cities":
        csv_columns = [
            "region_name",
            "state_name",
            "biome_name",
            "coefficient_name",
            "mean",
            "std",
        ]
    else:
        csv_columns = [
            "region_name",
            "coefficient_name",
            "mean",
            "std",
        ]

    csv_df = csv_df[csv_columns].copy()

    for col in csv_df.select_dtypes(include=["object"]).columns:
        csv_df[col] = csv_df[col].apply(clean_text)

    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved CSV:")
    print(f"  {csv_path}")

    gdf_save = gdf.copy()

    for col in gdf_save.columns:
        if col != "geometry" and gdf_save[col].dtype == "object":
            gdf_save[col] = gdf_save[col].apply(clean_text).str.slice(0, 250)

    if SAVE_SHAPEFILES:
        print(f"Saving shapefile:")
        print(f"  {shp_path}")

        gdf_save.to_file(
            shp_path,
            driver="ESRI Shapefile",
            encoding="UTF-8",
        )

    if SAVE_GEOPACKAGE:
        print(f"Saving GeoPackage layer:")
        print(f"  {gpkg_path} | layer = {gpkg_layer}")

        gdf_save.to_file(
            gpkg_path,
            layer=gpkg_layer,
            driver="GPKG",
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GRIDF FINAL ZONAL DISAGGREGATION COEFFICIENT EXTRACTION")
    print("=" * 80)

    check_path(BRAZIL_SHP, "Brazil ADM0 shapefile")
    check_path(STATES_SHP, "States ADM1 shapefile")
    check_path(CITIES_SHP, "Cities ADM2 shapefile")
    check_path(BIOME_DIR, "Biome folder")

    daily_rasters = list_rasters(COEFF_DIR_DAILY)
    subdaily_rasters = list_rasters(COEFF_DIR_SUBDAILY)

    daily_rasters = sort_rasters_largest_to_shortest(
        daily_rasters,
        "relative_to_daily",
    )

    subdaily_rasters = sort_rasters_largest_to_shortest(
        subdaily_rasters,
        "relative_to_subdaily",
    )

    print("\nDaily-relative coefficient rasters ordered largest to shortest:")
    for r in daily_rasters:
        print(f"  {get_coefficient_name(r)}  |  {r.name}")

    print("\nSubdaily-relative coefficient rasters ordered largest to shortest:")
    for r in subdaily_rasters:
        print(f"  {get_coefficient_name(r)}  |  {r.name}")

    all_csv_tables = []

    for layer_name, cfg in LAYERS.items():

        gdf = load_layer(layer_name, cfg)

        # --------------------------------------------------------
        # Relative to daily
        # --------------------------------------------------------
        daily_gdf, daily_csv_df = compute_family_for_layer(
            gdf=gdf,
            layer_name=layer_name,
            raster_paths=daily_rasters,
            family="relative_to_daily",
        )

        save_family_outputs(
            layer_name=layer_name,
            family="relative_to_daily",
            gdf=daily_gdf,
            csv_df=daily_csv_df,
        )

        daily_all = daily_csv_df.copy()
        daily_all.insert(0, "coefficient_family", "relative_to_daily")
        daily_all.insert(0, "source_layer", layer_name)
        all_csv_tables.append(daily_all)

        # --------------------------------------------------------
        # Relative to subdaily
        # --------------------------------------------------------
        subdaily_gdf, subdaily_csv_df = compute_family_for_layer(
            gdf=gdf,
            layer_name=layer_name,
            raster_paths=subdaily_rasters,
            family="relative_to_subdaily",
        )

        save_family_outputs(
            layer_name=layer_name,
            family="relative_to_subdaily",
            gdf=subdaily_gdf,
            csv_df=subdaily_csv_df,
        )

        subdaily_all = subdaily_csv_df.copy()
        subdaily_all.insert(0, "coefficient_family", "relative_to_subdaily")
        subdaily_all.insert(0, "source_layer", layer_name)
        all_csv_tables.append(subdaily_all)

    # ------------------------------------------------------------
    # Optional combined CSV
    # ------------------------------------------------------------
    all_csv_df = pd.concat(all_csv_tables, ignore_index=True)

    for col in all_csv_df.select_dtypes(include=["object"]).columns:
        all_csv_df[col] = all_csv_df[col].apply(clean_text)

    all_csv_path = OUTPUT_CSV_DIR / "All_coefficients.csv"
    all_csv_df.to_csv(all_csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("FINISHED SUCCESSFULLY")
    print("=" * 80)

    print("\nMain output folder:")
    print(f"  {OUTPUT_DIR}")

    print("\nRelative-to-daily CSV folder:")
    print(f"  {OUTPUT_CSV_DAILY_DIR}")

    print("\nRelative-to-subdaily CSV folder:")
    print(f"  {OUTPUT_CSV_SUBDAILY_DIR}")

    print("\nRelative-to-daily shapefile folder:")
    print(f"  {OUTPUT_SHP_DAILY_DIR}")

    print("\nRelative-to-subdaily shapefile folder:")
    print(f"  {OUTPUT_SHP_SUBDAILY_DIR}")

    print("\nGeoPackage output:")
    print(f"  {OUTPUT_GPKG_DIR / 'GRIDF_disaggregation_coefficients.gpkg'}")

    print("\nCombined CSV:")
    print(f"  {all_csv_path}")

    print("\nOrdering:")
    print("  Coefficients were ordered from largest duration to shortest duration.")
    print("  Example daily order: P24h, P12h, P10h, ..., P5m")
    print("  Example subdaily order: R24h, R12h, R10h, ..., R5m")

    print("\nCities output:")
    print("  Cities include state_name and biome_name.")
    print("  In the CSV, each city is kept together before moving to the next city.")

    print("\nFallback:")
    print("  If a polygon had no raster pixel center, the representative-point pixel value was used.")
    print("  In those cases, std = 0.")


if __name__ == "__main__":
    main()