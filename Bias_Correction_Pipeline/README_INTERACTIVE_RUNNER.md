# Interactive Runner + Pandas Date Bugfix

This ZIP adds:

```text
run_menu.py
src/biascorr/gauges.py
README_INTERACTIVE_RUNNER.md
```

## Why this was created

You asked for a way to choose what to run instead of typing long terminal commands.

Also, your current error was:

```text
TypeError: to_datetime() got an unexpected keyword argument 'infer_datetime_format'
```

This happens because your installed pandas version no longer accepts the
`infer_datetime_format` keyword. The included `gauges.py` removes that argument.

## Where to unzip

Unzip directly inside:

```bash
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Example:

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
unzip -o ~/Downloads/09_interactive_runner_bugfix.zip
```

## How to run the interactive menu

```bash
cd /Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
python3 run_menu.py
```

Then choose from the menu.

## Recommended first menu steps

Start with:

```text
1) Show configuration
2) Check input paths
3) Inventory annual maximum rasters
4) Prepare gauges
5) Select events — debug/test run
```

For the debug event selection, use:

```text
product: imerg_v07
percentile: p98
start year: 2001
end year: 2001
station limit: 20
```

If that works, continue to:

```text
6) Select events — full one product/percentile
```

## Important

The menu does not change the theory or pipeline. It simply runs the same
`run_pipeline.py` commands for you.
