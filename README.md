# E3-SSE

**Softening-Rescaled Transport and Conformal Selection**

E3-SSE develops a training-free correction and statistically controlled screening framework for Li-ion solid electrolytes using foundation machine-learning interatomic potentials.

## Repository locations

- Local VS Code workspace: `e3sse`
- GitHub: `github.com/SirHarginger/e3sse`
- Server: `/srv/ben/e3sse`

## Scientific workflow

```text
data
  ↓
softening estimation
  ↓
barrier correction
  ↓
transport validation
  ↓
conformal / FDR selection
  ↓
prospective screening
```

## Study 1 (Softening Atlas)

Production commands, outcome-seal requirements and outputs are in
`docs/workflow.md` ("Study 1: Softening Atlas").  Entry points:
`scripts/run_study1_softening_atlas.py`, `scripts/analyze_study1_softening_atlas.py`,
`scripts/plot_study1_figures.py`.
