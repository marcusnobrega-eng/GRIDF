## 🌧️ GRIDF-BR: Gridded Intensity–Duration–Frequency Curves for Brazil

A national framework for bias-corrected rainfall Intensity–Duration–Frequency (IDF) curves across Brazil, combining daily gridded datasets, sub-daily disaggregation from gauges, and open-access visualization through Google Earth Engine.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
![](https://img.shields.io/github/issues/marcusnobrega-eng/GRIDF)
![](https://img.shields.io/github/forks/marcusnobrega-eng/GRIDF)
![](https://img.shields.io/github/last-commit/marcusnobrega-eng/GRIDF)

## 📘 Project Summary

GRIDF-BR is an open-source modeling and visualization toolbox that delivers updated, bias-corrected IDF curves for the entire Brazilian territory.
It integrates:

## 🌍 Gridded rainfall datasets (BR-DWGD, IMERG, CHIRPS, PERSIANN)

⏱️ Locally derived sub-daily disaggregation ratios from 3,165 ANA telemetric stations

📊 Bias correction of extremes using exceedances above the 98th percentile

📈 Sherman-equation fits for multiple return periods and durations

🌐 A Google Earth Engine (GEE) application for direct retrieval and visualization

The toolbox addresses the lack of consistent and recent IDFs in Brazil by providing municipality- and basin-scale rainfall design data that are reproducible, spatially continuous, and openly accessible.

## 🔍 Model Capabilities
| Feature                  | Description                                                              |
| ------------------------ | ------------------------------------------------------------------------ |
| 🛰️ Gridded rainfall     | BR-DWGD (0.1°), IMERG (0.1°), CHIRPS (0.05°→0.1°), PERSIANN-CDR (0.25°)  |
| ⏱️ Sub-daily scaling     | Ratios for 5–1440 min durations from ANA telemetric stations             |
| 🧮 Bias correction       | Multiplicative factors computed from 98th-percentile station exceedances |
| 📈 Extreme value fitting | Gumbel distribution + 4-parameter Sherman equation                       |
| 🧪 Disaggregation modes  | CETESB (fixed), STATION (nearest), RASTER (interpolated with QC)         |
| 🌐 Visualization         | Google Earth Engine app for maps, curves, metrics, and CSV/SVG export    |
| 📊 Diagnostics           | RMSE, R², Kolmogorov–Smirnov test, parity plots of observed vs gridded   |


## 📚 Documentation

The full methodology is described in the paper:

Gridded Bias-Corrected Intensity–Duration–Frequency Curves for Brazil using BR-DWGD, IMERG, CHIRPS, and PERSIANN datasets with Locally-Derived Disaggregation Coefficients

Supplementary materials include:

📖 Derivation of local disaggregation coefficients

💻 MATLAB/Python code for data retrieval, QC, and fitting

🧩 Bias correction methods and diagnostics

📊 Comparison with Torres et al. (2025) national IDF database

🌐 Link to the GRIDF-BR GEE application: [GRIDF-BR App](https://gridf-470516.projects.earthengine.app/view/gridf-br)

## 🚀 Google Earth Engine Toolbox

The GRIDF-BR GEE app provides:

Interactive maps of IDF parameters (K, a, b, c)

Bias correction toggle (RAW vs BC)

Disaggregation method choice (CETESB / Station / Raster)

On-click IDF curves and parameter exports

Downloadable graphics (PNG/SVG) and tables (CSV)

## 👤 Developer
**Marcus N. Gomes Jr.**  
Postdoctoral Researcher II, University of Arizona  
📧 Email: [marcusnobrega.engcivil@gmail.com](mailto:marcusnobrega.engcivil@gmail.com)  
🌐 Website: [marcusnobrega-eng.github.io/profile](https://marcusnobrega-eng.github.io/profile)  
📄 CV: [Download PDF](https://marcusnobrega-eng.github.io/profile/files/CV___Marcus_N__Gomes_Jr_.pdf)  
🧪 ORCID: [0000-0002-8250-8195](https://orcid.org/0000-0002-8250-8195)  
🐙 GitHub: [@marcusnobrega-eng](https://github.com/marcusnobrega-eng)
