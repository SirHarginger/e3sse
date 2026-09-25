---
name: estimate-softening
description: "Estimate E3-SSE force softening scales with leakage control, uncertainty estimation, diagnostics, and global/family/material/element resolution."
---

# Estimate Softening

The primary estimator is the zero-intercept force-parity projection of ML
forces onto DFT forces.

## Workflow

1. identify the correction target;
2. construct the permitted DFT-labelled frame set;
3. exclude prohibited same-hop pathway frames for leakage-free estimates;
4. align ML and DFT force components exactly;
5. verify units;
6. estimate the softening scale;
7. obtain uncertainty by resampling whole frames/material units as required;
8. report sample size and force-magnitude coverage;
9. diagnose dependence on chemistry, element, and configurational energy;
10. retain oracle estimates separately and label them clearly.

Never use held-out barrier outcomes to choose the softening scale.
