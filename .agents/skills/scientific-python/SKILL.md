---
name: scientific-python
description: "Implement numerically reliable scientific Python for E3-SSE with explicit units, tests, deterministic behavior, and maintainable APIs."
---

# Scientific Python

When implementing numerical code:

1. make shapes and units explicit;
2. validate finite values where required;
3. avoid unnecessary copies of large arrays;
4. use stable numerical operations;
5. separate I/O from computation;
6. preserve deterministic seeds;
7. document formulas and their source in docstrings;
8. test known simple cases before large-scale execution;
9. compare vectorized and reference implementations where correctness is not
   obvious;
10. never optimize before establishing numerical correctness.
