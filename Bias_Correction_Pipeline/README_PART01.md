# Part 01 — Core Configuration and Runner

This ZIP updates the existing `Bias_Correction_Pipeline` folder with the real
core configuration system and a functional `run_pipeline.py` runner.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip ~/Downloads/01_core_config_runner.zip
```

## Files included

```text
config/paths.yml
config/products.yml
config/method.yml
src/biascorr/__init__.py
src/biascorr/config.py
src/biascorr/utils.py
run_pipeline.py
```

## Commands to test

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline

python run_pipeline.py show-config
python run_pipeline.py check-paths
python run_pipeline.py list-products
python run_pipeline.py inventory-years
python run_pipeline.py write-manifest
```

## What this part does

- Loads YAML configs.
- Checks input paths.
- Creates expected folders.
- Scans annual maximum raster folders for years.
- Writes a year inventory JSON.
- Writes run manifests.
- Defines placeholder CLI commands for later phases.

## What this part does not do yet

It does not read gauge time series, select events, connect to GEE, export pairs,
compute zeta, interpolate zeta, or apply correction. Those come in Parts 02–08.
