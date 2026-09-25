---
name: code-review
description: "Review E3-SSE code for scientific correctness, data leakage, destructive behavior, numerical problems, reproducibility, tests, and maintainability."
---

# E3-SSE Code Review

Prioritize genuine defects over stylistic noise.

Review in this order:

1. raw-data safety;
2. scientific-definition correctness;
3. leakage and split integrity;
4. unit/shape correctness;
5. numerical correctness;
6. reproducibility;
7. error handling;
8. tests;
9. maintainability;
10. performance.

For every finding provide:

- severity;
- file/location;
- concrete failure mode;
- why it matters scientifically;
- minimal safe fix.

Do not invent a problem merely to produce review comments.
