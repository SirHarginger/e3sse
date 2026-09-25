---
name: mlip-inference
description: "Run reproducible uMLIP single-point or MD inference for E3-SSE while preserving model identity, device settings, units, and input provenance."
---

# MLIP Inference

Use for CHGNet, M3GNet, MACE, SevenNet, or other approved foundation
interatomic potentials.

## Before execution

Record:

- model name;
- exact checkpoint/version;
- package version;
- device and precision;
- atomic structure identifier;
- reference functional compatibility;
- configuration used.

## Rules

- Do not fine-tune unless the task explicitly calls for the fine-tuned
  comparator.
- Do not overwrite reference values.
- Store ML predictions in separate fields/tables.
- Check energy and force units.
- Test a small batch before a full inference campaign.
- Preserve failed predictions with explicit failure metadata rather than
  silently deleting them.
