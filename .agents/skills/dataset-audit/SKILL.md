---
name: dataset-audit
description: "Inspect and validate E3-SSE scientific datasets, schemas, splits, units, identifiers, provenance, and integrity before analysis."
---

# Dataset Audit

Use before implementing analysis against a dataset that has not already been
fully characterized.

## Workflow

1. Locate the dataset through configuration; do not hard-code a new path.
2. Inspect filenames and directory structure.
3. Inspect a small representative sample.
4. Record:
   - format;
   - number of structures/hops/frames;
   - identifiers;
   - train/validation/test labels;
   - atomic species;
   - energy fields and units;
   - force fields and units;
   - stress fields and units;
   - calculation metadata.
5. Identify links between MPLiTrj and nebDFT2k.
6. Check missing values, duplicate identifiers, malformed records, and
   unexpected dimensions.
7. Do not modify the source.
8. Write reusable inspection logic under `src/e3sse/data/`.
9. Add tiny synthetic/test fixtures under `tests/fixtures/`, never copies of
   the full scientific dataset.
10. Produce a concise audit report with unresolved ambiguities.
