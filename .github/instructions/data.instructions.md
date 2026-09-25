---
description: E3-SSE research-data handling and leakage rules
applyTo: "src/e3sse/data/**"
---

Research data handling is conservative.

- Inspect source files before assuming field names.
- Preserve original identifiers and partition labels.
- Do not mutate source datasets.
- Load only required fields when practical.
- Validate units, atom counts, shapes, and metadata.
- Make dataset provenance visible in returned objects or manifests.
- Never allow a hop's own pathway labels to leak into a leakage-free
  correction of that hop.
