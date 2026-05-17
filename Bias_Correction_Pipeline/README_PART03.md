# Part 03 — GEE Product Inspection

This ZIP implements the Google Earth Engine product-inspection layer.

It does not export station/product bias pairs yet. That will come in Part 04.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/03_gee_product_inspection.zip
```

## Files included

```text
run_pipeline.py
inspect_gee_products.py
src/biascorr/gee_products.py
README_PART03.md
```

## Dependencies

Install Earth Engine API if needed:

```bash
python3 -m pip install earthengine-api
```

Authenticate if needed:

```bash
earthengine authenticate
```

Your pipeline uses the default project:

```text
ee-marcusep2025
```

You can override it with:

```bash
--gee-project YOUR_PROJECT
```

## Main commands

Inspect one product using the runner:

```bash
python3 run_pipeline.py inspect-gee --product imerg_v07
```

Inspect all products:

```bash
python3 run_pipeline.py inspect-gee --all-products
```

Use the standalone helper:

```bash
python3 inspect_gee_products.py --product imerg_v07
python3 inspect_gee_products.py --all-products
```

Use a specific sample date and point:

```bash
python3 run_pipeline.py inspect-gee --product chirps --date 2020-01-15 --lon -47.8825 --lat -15.7942
```

## Outputs

Inspection JSON files are written to:

```text
metadata/data_inventory/
```

Example:

```text
metadata/data_inventory/gee_product_inspection_imerg_v07_YYYYMMDD_HHMMSS.json
```

## Product daily rainfall logic

The module constructs daily rainfall images as follows:

### CHIRPS

Daily product. Treated as:

```text
mm/day
```

### PERSIANN-CDR

Daily product. Treated as:

```text
mm/day
```

### BR-DWGD / Xavier

Daily product, but the band and scaling must be confirmed during inspection.
The configuration currently uses:

```text
gee_band: AUTO_DETECT
```

No scaling is applied unless explicit `scale_factor` and/or `offset` are later
added to `config/products.yml`.

### IMERG V06

Native GEE collection:

```text
NASA/GPM_L3/IMERG_V06
band = precipitationCal
```

The band is treated as mm/hour. Daily total is computed as:

```text
daily total = mean half-hourly precipitation rate * 24
```

This matches the annual-maximum logic already discussed for V06.

### IMERG V07

Climate Engine daily asset:

```text
projects/climate-engine-pro/assets/ce-gpm-imerg-v07/early-daily
```

The daily product is treated as already accumulated daily rainfall in:

```text
mm/day
```

## Why this step matters

Part 04 will use the exact same daily-image construction to sample product
rainfall at gauge-event dates. Therefore, this inspection step is the safeguard
against wrong bands, wrong units, or wrong daily aggregation before exporting
many GEE bias-pair CSVs.
