# E3-SSE Copilot Instructions

E3-SSE is a scientific Python project for training-free correction and
statistically controlled screening of Li-ion solid electrolytes with
foundation interatomic potentials.

Read `AGENTS.md` for the complete research and engineering contract.

Always:

- treat `data/` and especially `data/raw/` as immutable research inputs;
- never fabricate missing scientific data;
- inspect actual dataset schemas before implementing loaders;
- preserve train/calibration/test separation and chemical-system leakage rules;
- distinguish DFT, raw MLIP, SRT-corrected, FPMD, and fitted quantities;
- put reusable implementation under `src/e3sse/`;
- use configuration rather than hard-coded paths;
- add or update tests for meaningful behavior changes;
- run `.agents/scripts/validate_repo.sh` before considering a change complete;
- write generated scientific outputs under `outputs/`, never `data/`;
- preserve provenance for production runs.

Use relevant skills from `.agents/skills/`.
