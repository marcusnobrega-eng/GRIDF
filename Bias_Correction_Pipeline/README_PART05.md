# Part 05 — Station Zeta Computation

This ZIP implements local station-level correction-factor estimation.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/05_zeta_station.zip
```

## Files included

```text
run_pipeline.py
src/biascorr/zeta.py
README_PART05.md
```

## Dependencies

```bash
python3 -m pip install pandas numpy pyyaml
```

## Required previous steps

You need local pair CSVs in:

```text
data/products/<product>/sensitivity/<pXX>/pairs/
```

These files are created by Earth Engine in Part 04, exported to Google Drive,
and then copied/synced into the local pairs folder.

Expected file pattern:

```text
pairs_<product>_<pXX>_YYYY_chunkNNN.csv
```

## Compute zeta for one product/percentile

Main paper estimator, median:

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --percentile p98 --estimator median
```

Mean sensitivity:

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --percentile p98 --estimator mean
```

Both configured estimators:

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --percentile p98 --all-estimators
```

All percentiles for one product using default estimator:

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --all-percentiles
```

All products and all percentiles using default estimator:

```bash
python3 run_pipeline.py compute-zeta --all-products --all-percentiles
```

## Optional year filter

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --percentile p98 --estimator median --start-year 2001 --end-year 2005
```

## Optional pairs folder override

Useful for debugging only:

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --percentile p98 --pairs-folder /path/to/local/csvs
```

## Outputs

Estimator-specific station outputs:

```text
data/products/<product>/sensitivity/<pXX>/zeta_station/<estimator>/
```

Main retained station table:

```text
zeta_per_station_<product>_<pXX>_<estimator>.csv
```

All station statistics, including stations failing min-pair threshold:

```text
zeta_station_all_<product>_<pXX>_<estimator>.csv
```

Manifest:

```text
zeta_manifest_<product>_<pXX>_<estimator>.json
```

Event-level pair QC table:

```text
data/products/<product>/sensitivity/<pXX>/tables/pair_qc_<product>_<pXX>.csv
```

## Theory and QC

For each event:

```text
raw_ratio = gauge_mm / product_mm
```

The event is considered usable for zeta if:

```text
gauge_mm is finite
product_mm is finite
gauge_mm > min_gauge_rainfall_for_ratio_mm
product_mm > min_product_rainfall_for_ratio_mm
gauge_mm <= max_rainfall_for_ratio_mm
raw_ratio > 0
```

Then:

```text
ratio_for_zeta = clipped(raw_ratio, ratio_clip_low, ratio_clip_high)
```

The clipping bounds come from:

```text
config/method.yml
ratio_qc:
  ratio_clip: [0.25, 5.0]
```

Station zeta statistics saved:

```text
zeta_mean
zeta_median
zeta_slope0
zeta_std
zeta_min
zeta_p10
zeta_p25
zeta_p75
zeta_p90
zeta_max
zeta_iqr
```

The main selected value for the paper is:

```text
zeta_selected = zeta_median
```

when `--estimator median` is used.

The station is retained for interpolation only if:

```text
n_pairs_used >= min_pairs_per_station
```

where the default is:

```text
10
```

from `config/method.yml`.
