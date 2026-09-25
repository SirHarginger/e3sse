---
name: transport-analysis
description: "Analyze E3-SSE ML-MD versus FPMD diffusion and activation energies with equilibration, uncertainty, structural-stability, and unit checks."
---

# Transport Analysis

## Workflow

1. verify material, temperature, cell, trajectory, timestep and model identity;
2. inspect equilibration;
3. inspect host-framework stability;
4. compute MSD with the intended tracer/collective definition;
5. estimate diffusion with an uncertainty method appropriate for correlated
   trajectories;
6. fit Arrhenius behavior only where justified;
7. compare raw and SRT quantities against FPMD;
8. retain all temperatures and uncertainty estimates;
9. report excluded trajectories and reasons.

Do not equate tracer diffusion and conductivity without stating the
correlation/Haven-ratio assumption.
