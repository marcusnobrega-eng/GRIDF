# Part 04 — GEE Bias-Pair Exports

This ZIP implements the Google Earth Engine export layer for gauge/product
bias-pair CSVs.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/04_gee_pair_exports.zip
```

## Files included

```text
run_pipeline.py
src/biascorr/gee_pair_exports.py
README_PART04.md
```

It reuses:

```text
src/biascorr/gee_products.py
```

from Part 03.

## Dependencies

Install Earth Engine API if needed:

```bash
python3 -m pip install earthengine-api pandas numpy pyyaml
```

Authenticate if needed:

```bash
earthengine authenticate
```

## Required previous steps

Before exporting pairs, event selection must already exist.

For a small test:

```bash
python3 run_pipeline.py select-events --product imerg_v07 --percentile p98 --start-year 2001 --end-year 2001 --station-limit 20
```

## Dry-run test

This builds the export manifest but does not submit GEE tasks:

```bash
python3 run_pipeline.py export-pairs --product imerg_v07 --percentile p98 --start-year 2001 --end-year 2001 --dry-run
```

## Submit a small real GEE export

```bash
python3 run_pipeline.py export-pairs --product imerg_v07 --percentile p98 --start-year 2001 --end-year 2001 --max-features-per-export 500
```

## Submit full P98 for IMERG V07

```bash
python3 run_pipeline.py export-pairs --product imerg_v07 --percentile p98
```

## Submit all percentiles for one product

```bash
python3 run_pipeline.py export-pairs --product imerg_v07 --all-percentiles
```

## Submit all products and all percentiles

Only do this after small tests work:

```bash
python3 run_pipeline.py export-pairs --all-products --all-percentiles
```

## Output behavior

Earth Engine exports table CSVs to Google Drive, not directly to your local
GitHub folder.

Default Drive folder pattern:

```text
GRIDF_BiasCorrection_pairs_<product>_<pXX>
```

Example:

```text
GRIDF_BiasCorrection_pairs_imerg_v07_p98
```

Expected local destination after the Drive tasks finish and sync:

```text
data/products/<product>/sensitivity/<pXX>/pairs/
```

The module writes a local README file in each pairs folder explaining the
Drive folder and expected filenames.

## Manifest outputs

Task manifests are written to:

```text
metadata/gee_tasks/
```

and also copied to the local pairs folder as:

```text
gee_pair_export_manifest_<product>_<pXX>_latest.json
```

## Scientific logic

For each selected gauge event:

```text
station_id, lon, lat, date, gauge_mm
```

the code constructs the daily product rainfall image in mm/day using the same
logic as Part 03:

- CHIRPS: daily mm/day
- PERSIANN-CDR: daily mm/day
- BR-DWGD: daily product, band/scaling to be confirmed
- IMERG V06: half-hourly mm/hour, daily total = mean rate * 24
- IMERG V07: Climate Engine daily mm/day

Then it samples the product at the station point and exports:

```text
product_mm
ratio_gauge_over_product = gauge_mm / product_mm
```

The ratio is only computed when product rainfall is greater than the configured
minimum product rainfall threshold. Otherwise, product values and validity flags
are preserved for audit, and filtering is handled in Part 05.
