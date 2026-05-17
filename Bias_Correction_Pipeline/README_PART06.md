# Part 06 — IDW Zeta Interpolation

This ZIP implements the spatial interpolation stage.

It reads station-level zeta tables from Part 05 and creates a gridded zeta
GeoTIFF aligned to the annual-maximum raster grid of each rainfall product.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/06_idw_interpolation.zip
```

## Files included

```text
run_pipeline.py
src/biascorr/interpolation.py
src/biascorr/raster_utils.py
README_PART06.md
```

## Dependencies

```bash
python3 -m pip install pandas numpy scipy rasterio matplotlib pyyaml
```

`matplotlib` is only needed for quick preview PNGs. If unavailable, the raster
is still written.

## Required previous steps

You need a retained station zeta table from Part 05:

```bash
python3 run_pipeline.py compute-zeta --product imerg_v07 --percentile p98 --estimator median
```

Expected input:

```text
data/products/<product>/sensitivity/<pXX>/zeta_station/<estimator>/
    zeta_per_station_<product>_<pXX>_<estimator>.csv
```

## Run one interpolation

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --percentile p98 --estimator median
```

## Run all percentiles for one product

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --all-percentiles --estimator median
```

## Run both mean and median for P98

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --percentile p98 --all-estimators
```

## Optional debug overrides

Use a custom station zeta table:

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --percentile p98 --estimator median --zeta-table /path/to/zeta.csv
```

Use a custom template raster:

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --percentile p98 --estimator median --template-raster /path/to/template.tif
```

Disable preview PNG:

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --percentile p98 --estimator median --no-preview
```

## Outputs

```text
data/products/<product>/sensitivity/<pXX>/zeta_grid/<estimator>/
```

Main raster:

```text
zeta_map_<product>_<pXX>_<estimator>_idw_k10_p2p0.tif
```

Station CSV used for interpolation:

```text
zeta_station_points_<product>_<pXX>_<estimator>.csv
```

Manifest:

```text
zeta_grid_manifest_<product>_<pXX>_<estimator>.json
```

Preview PNG:

```text
zeta_map_preview_<product>_<pXX>_<estimator>.png
```

## Theory

The station zeta values are interpolated using inverse distance weighting:

```text
zeta(x) = sum_i w_i zeta_i / sum_i w_i
w_i = 1 / d_i^p
```

with:

```text
k = 10 nearest stations
p = 2
```

The distance calculation uses great-circle distance in kilometers based on a
unit-sphere coordinate transform. This avoids treating degrees of longitude and
latitude as equal linear distances across Brazil.

Before interpolation, `zeta_selected` is clipped using:

```text
config/method.yml:
  zeta_clip_before_interpolation: [0.05, 10.0]
```

The output raster uses the same CRS, transform, width, height, and mask as the
annual-maximum raster template for the corresponding product.
