"""Test-only reference for Weighted Conformalized Selection, transcribed from the paper.

Jin & Candes, "Model-free selective inference under covariate shift via weighted
conformal p-values", arXiv:2307.09291v2: eq. 4 (randomized p-value), eq. 5, eq. 6
(auxiliary p-values) and Algorithm 1 with deterministic pruning (eq. 9).  This is
written independently of ``e3sse.selection.conformal`` (plain Python loops, paper
notation) and is never imported by production code.
"""

from __future__ import annotations


def p_value_eq4(V, w, V_hat_j, w_j, U_j):
    """Eq. 4: [sum w_i 1{V_i < Vhat} + (w_j + sum w_i 1{V_i = Vhat}) U_j] / (sum w_i + w_j)."""

    below = sum(wi for Vi, wi in zip(V, w) if Vi < V_hat_j)
    tied = sum(wi for Vi, wi in zip(V, w) if Vi == V_hat_j)
    return (below + (w_j + tied) * U_j) / (sum(w) + w_j)


def p_value_eq5(V, w, V_hat_j, w_j):
    """Eq. 5: [sum w_i 1{V_i < Vhat_j} + w_j] / (sum w_i + w_j)."""

    return (sum(wi for Vi, wi in zip(V, w) if Vi < V_hat_j) + w_j) / (sum(w) + w_j)


def auxiliary_eq6(V, w, V_hat, w_test, j):
    """Eq. 6 for l != j (strict <; the added term carries w_j, the replaced unit)."""

    out = {}
    for l in range(len(V_hat)):
        if l == j:
            continue
        cal = sum(wi for Vi, wi in zip(V, w) if Vi < V_hat[l])
        out[l] = (cal + w_test[j] * (1 if V_hat[j] < V_hat[l] else 0)) / (sum(w) + w_test[j])
    return out


def bh_with_zero_at_j(aux, m, q):
    """|R_{j->0}|: BH on {p^(j)_l} with p_j replaced by 0 (Algorithm 1, steps 5-6)."""

    k_star = 1
    for k in range(1, m + 1):
        if 1 + sum(1 for value in aux.values() if value <= q * k / m) >= k:
            k_star = k
    return 1 + sum(1 for value in aux.values() if value <= q * k_star / m)


def wcs_deterministic(V, w, V_hat, w_test, q, *, count_ties):
    """Algorithm 1, deterministic pruning; ``count_ties``: main p-value is eq. 4 at U = 1."""

    m = len(V_hat)
    if count_ties:
        p = [p_value_eq4(V, w, V_hat[j], w_test[j], 1.0) for j in range(m)]
    else:
        p = [p_value_eq5(V, w, V_hat[j], w_test[j]) for j in range(m)]
    sizes = [bh_with_zero_at_j(auxiliary_eq6(V, w, V_hat, w_test, j), m, q)
             for j in range(m)]
    first = [p[j] <= q * sizes[j] / m for j in range(m)]
    r_star = 0
    for r in range(1, m + 1):
        if sum(1 for j in range(m) if first[j] and sizes[j] <= r) >= r:
            r_star = r
    return [bool(first[j] and sizes[j] <= r_star) for j in range(m)], p, sizes


def unit_keys(n_calibration, n_test, analysis="synthetic"):
    """Frozen per-unit tie keys for synthetic units (test helper)."""

    from e3sse.selection.conformal import unit_tie_key

    return {"calibration_tie_keys": [unit_tie_key("test-protocol", analysis, f"cal-{i}")
                                     for i in range(n_calibration)],
            "test_tie_keys": [unit_tie_key("test-protocol", analysis, f"test-{i}")
                              for i in range(n_test)]}
