# Part 08 — Diagnostics and Sensitivity Analysis

This is the final code block of the pipeline.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/08_diagnostics_sensitivity.zip
```

## Files included

```text
run_pipeline.py
compare_mean_median.py
compare_percentile_sensitivity.py
make_paper_figures.py
README_PART08.md
src/biascorr/diagnostics.py
src/biascorr/plot_utils.py
```

## Dependencies

```bash
python3 -m pip install pandas numpy rasterio matplotlib pyyaml
```

## Main diagnostics for one run

```bash
python3 run_pipeline.py diagnostics --product imerg_v07 --percentile p98 --estimator median
```

Outputs:

```text
data/products/<product>/sensitivity/<pXX>/diagnostics/<estimator>/
figures/diagnostics/<product>/<pXX>/<estimator>/
```

The diagnostic summary includes:

```text
pair QC summary
station zeta summary
zeta raster summary
corrected annual maximum summary
basic figures
```

## Percentile sensitivity

Compare P90, P95, P98, P99, and P99.5 against P98:

```bash
python3 run_pipeline.py percentile-sensitivity --product imerg_v07 --estimator median
```

Standalone equivalent:

```bash
python3 compare_percentile_sensitivity.py --product imerg_v07 --estimator median
```

Outputs:

```text
figures/sensitivity/percentile/<product>/<estimator>/
```

Metrics include:

```text
common station count
station zeta correlation against P98
mean absolute difference
median relative absolute difference
optional raster difference metrics
```

## Mean-vs-median sensitivity

```bash
python3 run_pipeline.py mean-median-sensitivity --product imerg_v07 --percentile p98
```

Standalone equivalent:

```bash
python3 compare_mean_median.py --product imerg_v07 --percentile p98
```

Outputs:

```text
figures/sensitivity/mean_vs_median/<product>/<pXX>/
```

Metrics include:

```text
station zeta correlation
mean absolute difference
median relative difference
optional raster difference metrics
```

## Paper-support wrapper

```bash
python3 make_paper_figures.py --product imerg_v07 --percentile p98 --estimator median
```

This runs:

```text
basic diagnostics
percentile sensitivity
mean-vs-median sensitivity
```

for the selected product.

## Important note

The figures generated here are diagnostics and paper-support figures. They are
scientifically organized and reproducible, but final manuscript figures can be
further refined after checking the numerical outputs.
