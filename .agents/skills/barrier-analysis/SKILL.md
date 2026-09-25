---
name: barrier-analysis
description: "Evaluate raw and SRT-corrected Li migration barriers against DFT while keeping fixed-geometry and relaxed ML-NEB analyses distinct."
---

# Barrier Analysis

Keep these result types separate:

- DFT reference barrier;
- ML fixed-geometry barrier on DFT NEB images;
- relaxed ML-NEB barrier;
- SRT-corrected barrier;
- label-using recalibration comparator.

## Checks

- verify hop identifiers;
- verify image ordering;
- verify barrier definition against the dataset loader;
- verify energy units;
- report MAE, RMSE, signed error and rank metrics as appropriate;
- stratify only according to pre-specified analyses;
- preserve transition-metal/reference-functional caveats;
- attach confidence intervals using the specified resampling unit.

Do not describe fixed-geometry evaluation as ML-NEB.
