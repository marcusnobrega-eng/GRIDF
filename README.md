# 🌧️ GRIDF-BR: Gridded Intensity–Duration–Frequency Curves for Brazil

**GRIDF-BR** is an open-source code repository for generating, comparing, and visualizing gridded rainfall Intensity–Duration–Frequency (IDF) curves across Brazil using long-term gridded and satellite rainfall products, locally derived sub-daily disaggregation coefficients, upper-tail bias correction, and pixelwise Sherman IDF equation fitting.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
![](https://img.shields.io/github/issues/marcusnobrega-eng/GRIDF)
![](https://img.shields.io/github/forks/marcusnobrega-eng/GRIDF)
![](https://img.shields.io/github/last-commit/marcusnobrega-eng/GRIDF)

---

## 📌 Project Overview

GRIDF-BR stands for **Gridded Intensity–Duration–Frequency Curves for Brazil**.

This repository contains the source code, configuration files, Google Earth Engine interface files, and supporting scripts used to develop the GRIDF-BR framework. The project supports the manuscript:

> **A Nationally Consistent Assessment of Intensity–Duration–Frequency Curves for Brazil Using Long-Term Gridded and Satellite Rainfall Data**

The objective of GRIDF-BR is to provide a reproducible, nationally consistent workflow for estimating gridded IDF curves over Brazil. The framework was developed to address the limitations of existing Brazilian IDF equations, many of which were derived from heterogeneous station records, different probability distributions, variable temporal periods, and fixed or regionally inconsistent disaggregation assumptions.

The workflow combines:

- Long-term gridded and satellite rainfall products.
- Annual maximum daily rainfall extraction.
- Bias correction of upper-tail rainfall extremes.
- Locally derived sub-daily disaggregation coefficients from ANA telemetric stations.
- Gumbel baseline and GEV sensitivity analysis.
- Pixelwise Sherman IDF equation fitting.
- Diagnostics for frequency analysis, bias correction, and IDF fit quality.
- An interactive Google Earth Engine application for map visualization and point-based IDF retrieval.

---

## 🔗 Related Resources

| Resource | Link |
|---|---|
| GitHub repository | <https://github.com/marcusnobrega-eng/GRIDF> |
| Zenodo data archive | <https://doi.org/10.5281/zenodo.20262939> |
| GRIDF-BR Google Earth Engine App | <https://gridf-470516.projects.earthengine.app/view/gridf-br> |

The GitHub repository is intentionally kept lightweight and contains primarily **code, configuration files, and documentation**. Large datasets, generated rasters, intermediate files, diagnostics, and manuscript results are stored separately in the Zenodo archive.

---

## 📦 Zenodo Dataset

The full data archive is available at:

**GRIDF-BR: Code, Datasets, and Results for Gridded Intensity–Duration–Frequency Curves for Brazil**  
DOI: [10.5281/zenodo.20262939](https://doi.org/10.5281/zenodo.20262939)

The Zenodo record contains the full working archive and separate results-only ZIP files, including:

- Annual maximum daily precipitation rasters.
- Raw and bias-corrected IDF parameter rasters.
- Bias-correction outputs and diagnostics.
- Sub-daily disaggregation coefficient products.
- GEV–Gumbel sensitivity outputs.
- Kolmogorov–Smirnov diagnostics.
- Manuscript and supplementary figures.
- Full reproducible working folder.

Users who want to reproduce the full analysis or access the generated raster products should download the appropriate ZIP files from Zenodo and place them according to the folder structure described in the Zenodo record.

---

## 🧭 Conceptual Workflow

```text
Daily gridded rainfall products
        │
        ├── Annual maximum daily rainfall extraction
        │
        ├── Gauge–product upper-tail event pairing
        │       └── Multiplicative bias correction
        │
        ├── Extreme-value frequency analysis
        │       ├── Gumbel baseline
        │       └── GEV sensitivity
        │
        ├── Daily design rainfall depths
        │
        ├── Sub-daily disaggregation coefficients
        │       ├── CETESB fixed ratios
        │       ├── Locally disaggregated raster ratios
        │       └── Station-derived nearest-station ratios
        │
        ├── Sub-daily design rainfall intensities
        │
        ├── Pixelwise Sherman IDF fitting
        │
        └── GRIDF-BR products
                ├── IDF parameter rasters
                ├── fit diagnostics
                ├── point IDF curves
                ├── product comparisons
                └── Google Earth Engine visualization
```

---

## 🌎 Rainfall Products

The GRIDF-BR workflow uses the following rainfall products:

| Product | Period used in GRIDF-BR | Native grid |
|---|---:|---:|
| BR-DWGD | 1995–2025 | 0.1° |
| CHIRPS | 1995–2025 | 0.05° |
| PERSIANN-CDR | 1995–2025 | 0.25° |
| IMERG V06 | 2001–2020 | 0.1° |
| IMERG V07 | 2001–2025 | 0.1° |

Each product is analyzed on its native grid whenever possible. The IDF fitting workflow preserves product-specific spatial support instead of forcing all products to a common grid.

---

## 📁 Repository Organization

This GitHub repository contains the code and lightweight supporting files needed to understand, run, and maintain the GRIDF-BR workflows.

```text
GRIDF/
├── Bias_Correction/
├── Bias_Correction_Pipeline/
├── Disag_Coefficients/
├── Existing_IDFs/
├── GEE_Interface/
├── IDF_Fitting/
├── Misc/
├── README.md
├── LICENSE
└── .gitignore
```

Large generated outputs are not stored directly in GitHub. They are available through the Zenodo archive.

---

## 📂 Folder Descriptions

### `Bias_Correction/`

Contains earlier and product-specific scripts used to develop and evaluate rainfall-product bias correction. These scripts support analyses of gauge–product relationships, rainfall-product comparison, and diagnostic plotting.

This folder may include scripts for:

- Reading rainfall station metadata.
- Comparing observed and gridded rainfall extremes.
- Producing bias-correction diagnostic figures.
- Supporting exploratory bias-correction workflows.

Large rainfall time series and generated outputs are not tracked in GitHub and are distributed through Zenodo.

---

### `Bias_Correction_Pipeline/`

Contains the modular bias-correction pipeline developed for the final workflow.

Typical structure:

```text
Bias_Correction_Pipeline/
├── config/
├── docs/
├── scripts/
├── src/
├── tests/
├── notebooks/
├── run_pipeline.py
└── run_menu.py
```

Main components:

- `config/`: YAML configuration files for paths, rainfall products, and method settings.
- `src/biascorr/`: core Python modules for reading inputs, selecting events, estimating correction factors, interpolating bias fields, applying corrections, and generating diagnostics.
- `scripts/`: helper scripts for running specific pipeline steps.
- `docs/`: pipeline-specific documentation.
- `tests/`: basic tests or validation utilities.
- `notebooks/`: exploratory or diagnostic notebooks when available.
- `run_pipeline.py`: command-line entry point for the modular pipeline.
- `run_menu.py`: interactive menu-style runner.

The large `data/`, `figures/`, `logs/`, and generated `metadata/` outputs are ignored by Git and archived externally.

---

### `Disag_Coefficients/`

Contains scripts used to derive, interpolate, summarize, and visualize daily-to-sub-daily rainfall disaggregation coefficients.

These coefficients are used to convert daily design rainfall depths into sub-daily design rainfall depths before fitting IDF curves.

Typical products generated by this workflow include:

- Ratios relative to daily rainfall.
- Ratios relative to sub-daily reference durations.
- Interpolated coefficient rasters.
- Biome, state, and city-level summaries.
- Comparisons with CETESB reference ratios.
- Zonal statistics tables.

Large coefficient rasters, GeoPackages, and zonal summary products are distributed through Zenodo.

---

### `Existing_IDFs/`

Contains scripts used to compare GRIDF-BR estimates with previously published Brazilian IDF equations, including the national IDF database compiled from existing station-based studies.

This folder supports analyses such as:

- Sampling GRIDF-BR rasters at existing IDF station locations.
- Comparing gridded IDF estimates with existing station-based equations.
- Computing relative differences by duration and return period.
- Summarizing differences by biome or reference-IDF type.
- Generating comparison maps and diagnostic plots.

Generated comparison outputs and figures are stored in the Zenodo archive.

---

### `GEE_Interface/`

Contains files associated with the Google Earth Engine implementation of the GRIDF-BR interface.

The GRIDF-BR app allows users to:

- Select rainfall product.
- Select raw or bias-corrected data.
- Select disaggregation method.
- Visualize IDF parameters and diagnostics.
- Query grid cells.
- Generate IDF curves for clicked locations.
- Export graphics and tables.

Interactive app:

<https://gridf-470516.projects.earthengine.app/view/gridf-br>

---

### `IDF_Fitting/`

Contains the main scripts for fitting gridded IDF curves.

This folder supports:

- Reading annual maximum rainfall rasters.
- Applying raw or bias-corrected annual maxima.
- Fitting Gumbel and GEV distributions.
- Converting daily design rainfall depths to sub-daily depths.
- Fitting the Sherman IDF equation pixel by pixel.
- Exporting IDF parameter rasters.
- Producing KS diagnostics and goodness-of-fit outputs.
- Running distributional and percentile-threshold sensitivity analyses.

Main scripts may include:

```text
complete_idf_pipeline.py
run_complete_idf_pipeline.sh
diagnose_distribution_fit_daily_amax.py
percentile_sensitivity_pairwise_bias_metrics.py
plot_percentile_sensitivity_bias_maps_and_parity.py
```

Large IDF rasters and diagnostic outputs are distributed through Zenodo.

---

### `Misc/`

Contains supporting scripts used for figure generation, study-area visualization, or auxiliary geospatial tasks.

Large support rasters such as DEMs are not tracked in GitHub and are distributed through the Zenodo archive when needed.

---

## 📊 Main Data Products

The full data products are distributed through Zenodo. The most important generated outputs are:

| Product family | Description |
|---|---|
| Annual maximum precipitation rasters | Yearly maximum daily precipitation for each rainfall product |
| Bias-correction factors | Product-specific multiplicative correction fields |
| Bias-corrected annual maxima | Corrected annual maximum rainfall rasters |
| Disaggregation coefficients | Daily-to-sub-daily duration ratios |
| Sherman IDF parameters | Pixelwise `K`, `a`, `b`, and `c` parameters |
| Fit diagnostics | RMSE, MSE, R², KS statistics, rejection masks |
| Sensitivity analyses | GEV–Gumbel and percentile-threshold sensitivity products |
| Figures and diagnostics | Manuscript figures, supplementary figures, parity plots, and maps |

---

## 📐 IDF Equation

GRIDF-BR uses a four-parameter Sherman equation:

```math
i(t,T) = \frac{K T^a}{(b+t)^c}
```

where:

- `i(t,T)` is rainfall intensity in mm h⁻¹,
- `t` is rainfall duration in minutes,
- `T` is return period in years,
- `K`, `a`, `b`, and `c` are fitted empirical parameters.

The parameters are fitted independently for each grid cell, rainfall product, correction state, disaggregation mode, and extreme-value distribution.

---

## 🧪 Extreme-Value Analysis

The workflow fits annual maximum daily rainfall series using:

- **Gumbel distribution** as the baseline model.
- **Generalized Extreme Value (GEV) distribution** as a sensitivity model.

Kolmogorov–Smirnov diagnostics are generated for each distribution, product, and correction state.

The Gumbel baseline is used in the main GRIDF-BR interface, while GEV products are retained as sensitivity outputs in the Zenodo archive.

---

## ⏱️ Disaggregation Modes

GRIDF-BR evaluates three temporal-distribution modes:

| Mode | Description |
|---|---|
| `CETESB` | Fixed national ratios traditionally used in Brazilian engineering practice |
| `Locally-Disaggregated` | Spatially interpolated ratios derived from ANA telemetric stations |
| `Station-derived` | Nearest quality-controlled station coefficient vector assigned to each grid cell |

These modes allow users to evaluate how design rainfall estimates change depending on the disaggregation assumption.

---

## 🧮 Bias Correction

Bias correction is applied using multiplicative correction factors estimated from upper-tail gauge–product rainfall pairs.

The general correction is:

```math
P_\mathrm{corrected}(x,y,t) = \zeta(x,y) P_\mathrm{raw}(x,y,t)
```

where:

- `P_raw` is the raw annual maximum precipitation,
- `ζ` is the spatially interpolated multiplicative correction factor,
- `P_corrected` is the corrected annual maximum precipitation.

The reference analysis uses the 98th percentile threshold for upper-tail event selection, with sensitivity analyses at additional thresholds.

---

## 🚀 Basic Installation

Create a Python environment:

```bash
conda create -n gridf python=3.10
conda activate gridf
```

Install core dependencies:

```bash
pip install numpy pandas geopandas rasterio rioxarray xarray scipy matplotlib scikit-learn pyyaml tqdm earthengine-api
```

Optional packages:

```bash
pip install contextily cartopy seaborn openpyxl
```

For Google Earth Engine workflows:

```bash
earthengine authenticate
```

---

## ⚙️ Running the Workflows

### 1. Bias-Correction Pipeline

```bash
cd Bias_Correction_Pipeline
python run_pipeline.py
```

or use the interactive runner:

```bash
python run_menu.py
```

Users may need to update the YAML configuration files in:

```text
Bias_Correction_Pipeline/config/
```

before running the workflow on a new machine.

---

### 2. IDF Fitting

```bash
cd IDF_Fitting
bash run_complete_idf_pipeline.sh
```

The full workflow requires annual maximum precipitation rasters and disaggregation coefficient rasters from the Zenodo archive.

---

### 3. Disaggregation Coefficients

```bash
cd Disag_Coefficients
python Subdaily_Disaggregation_Maps.py
```

or run the appropriate coefficient-generation or zonal-statistics script depending on the desired output.

---

### 4. Existing IDF Comparison

```bash
cd Existing_IDFs
python diagnose_gridf_bias_from_filtered_xlsx.py
```

Additional scripts in this folder generate comparison plots, biome summaries, and return-period bias diagnostics.

---

## 🧾 Data Availability

Large datasets and generated outputs are available from Zenodo:

[https://doi.org/10.5281/zenodo.20262939](https://doi.org/10.5281/zenodo.20262939)

The Zenodo archive includes both a full reproducible working archive and results-only ZIP files. Users who want to run the full workflow should download the full archive or place the required data folders from the results-only archives into the expected locations.

---

## 🌐 GRIDF-BR Google Earth Engine App

The GRIDF-BR web application is available at:

[https://gridf-470516.projects.earthengine.app/view/gridf-br](https://gridf-470516.projects.earthengine.app/view/gridf-br)

The app provides:

- Interactive maps of IDF parameters.
- Product selection.
- Raw versus bias-corrected comparison.
- Disaggregation method selection.
- Point-based IDF curve generation.
- Parameter and diagnostic extraction.
- Exportable plots and tables.

---

## ⚠️ Notes and Limitations

- Some scripts may preserve local path assumptions from the original development environment.
- Users may need to update path variables or YAML configuration files before rerunning workflows.
- The GitHub repository is intentionally lightweight and does not include large raster outputs.
- The full reproducible working archive is available from Zenodo.
- Sub-daily gauge coverage is uneven across Brazil.
- Very short rainfall durations, especially 5 and 10 minutes, have greater uncertainty where direct observations are unavailable.
- Bias-correction diagnostics should be interpreted as calibration diagnostics, not independent validation.
- High-return-period estimates remain sensitive to distributional assumptions and record length.

---

## 📄 Citation

If you use GRIDF-BR, please cite both the manuscript and the Zenodo archive.

### Dataset citation

```text
Gomes Jr., M. N. (2026). GRIDF-BR: Code, Datasets, and Results for Gridded Intensity–Duration–Frequency Curves for Brazil. Zenodo. https://doi.org/10.5281/zenodo.20262939
```

### Software citation

```text
Gomes Jr., M. N. (2026). GRIDF-BR: Source Code for Gridded Intensity–Duration–Frequency Curves for Brazil. GitHub. https://github.com/marcusnobrega-eng/GRIDF
```

### BibTeX

```bibtex
@dataset{gomes_gridf_br_zenodo_2026,
  author    = {Gomes Jr., Marcus N.},
  title     = {GRIDF-BR: Code, Datasets, and Results for Gridded Intensity--Duration--Frequency Curves for Brazil},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20262939},
  url       = {https://doi.org/10.5281/zenodo.20262939}
}
```

```bibtex
@misc{gomes_gridf_br_github_2026,
  author       = {Gomes Jr., Marcus N.},
  title        = {GRIDF-BR: Source Code for Gridded Intensity--Duration--Frequency Curves for Brazil},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/marcusnobrega-eng/GRIDF}
}
```

---

## 👤 Developer

**Marcus N. Gomes Jr., PhD**  
Postdoctoral Researcher, Stanford University  

GitHub: [@marcusnobrega-eng](https://github.com/marcusnobrega-eng)  
ORCID: [0000-0002-8250-8195](https://orcid.org/0000-0002-8250-8195)  
GRIDF-BR repository: [https://github.com/marcusnobrega-eng/GRIDF](https://github.com/marcusnobrega-eng/GRIDF)

---

## 📜 License

This repository is released under the MIT License. See [`LICENSE`](./LICENSE) for details.

---

## Acknowledgment

This project uses publicly available rainfall products and station data from national and international data providers, including ANA, BR-DWGD, CHIRPS, IMERG, and PERSIANN-CDR. The GRIDF-BR framework was developed to support open, reproducible, and spatially consistent rainfall design analysis across Brazil.
