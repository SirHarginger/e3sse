---
name: conformal-selection
description: "Implement and validate conformal selection and FDR control for E3-SSE fast-hop screening without calibration/test leakage."
---

# Conformal Selection

## Requirements

- define the outcome and fast-hop threshold explicitly;
- build the predictor without test outcomes;
- separate calibration and test chemical systems;
- respect dependence by using the pre-specified chemical-system procedure;
- calculate conformal p-values exactly as specified;
- apply the intended BH/conformal-selection procedure;
- report nominal FDR, realized FDP/FDR, power, calibration size and test size;
- include dependence/covariate-shift sensitivity analysis when requested.

Predictive accuracy and FDR validity are different claims. Do not conflate
them.
