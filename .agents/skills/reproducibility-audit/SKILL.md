---
name: reproducibility-audit
description: "Audit an E3-SSE analysis for exact reconstruction from Git commit, configuration, datasets, models, seeds, commands, outputs, and environment metadata."
---

# Reproducibility Audit

Confirm that a result can be reconstructed from:

- Git SHA and branch;
- clean/dirty state;
- configuration;
- dataset release and split;
- model/checkpoint;
- package environment;
- seed;
- command;
- output location;
- run metadata.

Flag hidden notebook state, manual spreadsheet edits, unstated exclusions,
untracked configuration, and copied values whose generating command cannot be
identified.
