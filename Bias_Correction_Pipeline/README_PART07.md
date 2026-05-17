# Part 07 — Apply Bias to Annual Maximum Rasters

This ZIP implements the final raster-correction stage.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/07_apply_bias_rasters.zip
```

## Files included

```text
run_pipeline.py
check_outputs.py
src/biascorr/apply_bias.py
src/biascorr/raster_utils.py
README_PART07.md
```

## Dependencies

```bash
python3 -m pip install pandas numpy rasterio pyyaml
```

## Required previous steps

You need a gridded zeta raster from Part 06:

```bash
python3 run_pipeline.py interpolate-zeta --product imerg_v07 --percentile p98 --estimator median
```

Expected input:

```text
data/products/<product>/sensitivity/<pXX>/zeta_grid/<estimator>/
    zeta_map_<product>_<pXX>_<estimator>_idw_k10_p2p0.tif
```

## Apply bias for one product/percentile/estimator

```bash
python3 run_pipeline.py apply-bias --product imerg_v07 --percentile p98 --estimator median
```

## Apply only a test year range

```bash
python3 run_pipeline.py apply-bias --product imerg_v07 --percentile p98 --estimator median --start-year 2001 --end-year 2002
```

## Apply all percentiles for one product

```bash
python3 run_pipeline.py apply-bias --product imerg_v07 --all-percentiles --estimator median
```

## Apply both mean and median for P98

```bash
python3 run_pipeline.py apply-bias --product imerg_v07 --percentile p98 --all-estimators
```

## Optional custom zeta raster

```bash
python3 run_pipeline.py apply-bias --product imerg_v07 --percentile p98 --estimator median --zeta-raster /path/to/zeta.tif
```

## Outputs

```text
data/products/<product>/sensitivity/<pXX>/annual_max_corrected/<estimator>/
```

Corrected GeoTIFFs:

```text
corrected_<product>_<pXX>_<estimator>_<year>.tif
```

Summary CSV:

```text
annual_max_correction_summary_<product>_<pXX>_<estimator>.csv
```

Manifest:

```text
apply_bias_manifest_<product>_<pXX>_<estimator>.json
```

## Theory

For each annual maximum raster:

```text
corrected = raw_product * zeta
```

where:

```text
zeta = gauge / product
```

The zeta field is clipped before application using:

```text
config/method.yml:
  zeta_clip_before_application: [0.25, 5.0]
```

If zeta and raw rainfall rasters are not on the same grid, zeta is resampled
to the raw raster grid using bilinear resampling by default.

The output raster preserves:

```text
CRS
transform
width/height
nodata behavior
```

from the original annual maximum rainfall raster.

## Check outputs

```bash
python3 check_outputs.py --product imerg_v07 --percentile p98 --estimator median
```
