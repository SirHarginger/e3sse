---
description: E3-SSE Python implementation and testing rules
applyTo: "**/*.py"
---

Use clear scientific Python.

- Prefer explicit units in variable names or documentation.
- Avoid hidden global state.
- Validate array dimensions and expected columns.
- Handle NaN/inf deliberately.
- Never silently coerce incompatible units.
- Use pathlib for filesystem paths.
- Keep reusable logic out of notebooks.
- Add tests for numerical transformations and edge cases.
- Preserve deterministic seeds where randomness is used.
- Do not introduce a new dependency when the standard library or an existing
  project dependency is sufficient.
