---
name: data-loader
description: "Build or modify validated loaders for MPLiTrj, nebDFT2k, BVEL13k, nebBVSE122k, and FPMD data without mutating source files."
---

# Data Loader

Read the `dataset-audit` skill first if the schema has not been established.

## Requirements

- Preserve original identifiers.
- Preserve published split labels.
- Expose units explicitly.
- Validate required fields.
- Fail clearly on malformed records.
- Avoid silently dropping structures or atoms.
- Keep loading separate from scientific transformations.
- Never mutate source files.
- Use streaming/chunking when the full dataset would be unnecessarily loaded
  into memory.

## Testing

Use synthetic fixtures covering:

- valid record;
- missing field;
- unexpected shape;
- invalid unit/metadata;
- duplicate identifier where relevant.
