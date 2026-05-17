# Part 02 — Gauge Reading and Event Selection

This ZIP implements the gauge-side portion of the pipeline.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/02_gauge_event_selection.zip
```

## Files included

```text
run_pipeline.py
src/biascorr/gauges.py
src/biascorr/event_selection.py
README_PART02.md
```

## Dependencies

Install if needed:

```bash
python3 -m pip install pandas numpy pyyaml
```

## Main commands

Prepare and validate the gauge database:

```bash
python3 run_pipeline.py prepare-gauges
```

Test event selection for a small case:

```bash
python3 run_pipeline.py select-events --product imerg_v07 --percentile p98 --start-year 2001 --end-year 2001 --station-limit 20
```

Run full P98 event selection for IMERG V07:

```bash
python3 run_pipeline.py select-events --product imerg_v07 --percentile p98
```

Run all percentiles for one product:

```bash
python3 run_pipeline.py select-events --product imerg_v07 --all-percentiles
```

Run all products and all percentiles:

```bash
python3 run_pipeline.py select-events --all-products --all-percentiles
```

## Outputs

Gauge QC outputs:

```text
data/gauges/processed/gauge_metadata_normalized.csv
data/gauges/qc/gauge_station_coverage_summary.csv
data/gauges/qc/gauge_date_columns.csv
data/gauges/qc/gauge_detection_manifest.json
```

Event-selection outputs:

```text
data/products/<product>/sensitivity/<pXX>/events/
```

Main event table:

```text
events_<product>_<pXX>_all_years.csv
```

Yearly event files:

```text
events_<product>_<pXX>_<year>.csv
```

Summary table:

```text
event_selection_summary_<product>_<pXX>.csv
```

Manifest:

```text
event_selection_manifest_<product>_<pXX>.json
```

## Method implemented

For each station-year:

1. Valid rainfall is defined using `gauge_qc` from `config/method.yml`.
2. The percentile threshold is computed from all valid daily rainfall values.
3. Candidate events are days with rainfall strictly above the threshold.
4. Events with rainfall less than or equal to the minimum ratio rainfall
   threshold are removed before GEE sampling.
5. Events above the maximum ratio rainfall threshold are removed.
6. Declustering keeps the largest event within a `min_gap_days` window.
7. Output events are saved for later GEE product sampling.

This stage does not sample GEE yet. It only creates the station/date/gauge
event tables required by Part 04.
