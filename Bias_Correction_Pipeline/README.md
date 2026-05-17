# GRIDF Bias Correction Pipeline

Project root:

```text
/Users/mngomes/Documents/GitHub/GRIDF/Bias_Correction_Pipeline
```

Core correction:

```text
zeta = gauge_rainfall / product_rainfall
corrected_product = raw_product * zeta
```

Products:
- CHIRPS: 1995-2025
- PERSIANN-CDR: 1995-2025
- BR-DWGD / Xavier: 1995-2025
- IMERG V06: 2001-2020
- IMERG V07: 2001-2025

Main method:
- Percentile: P98
- Sensitivity: P90, P95, P98, P99, P99.5
- Zeta estimator: median
- Estimator sensitivity: mean, primarily for P98
- Interpolation: IDW, k = 10, power = 2

Pipeline phases:
1. Configuration and folder setup
2. Gauge reading and validation
3. Event selection
4. GEE product inspection
5. GEE bias-pair exports
6. Station zeta estimation
7. IDW interpolation
8. Bias application and diagnostics
