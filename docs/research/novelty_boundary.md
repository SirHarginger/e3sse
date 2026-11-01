# E3-SSE novelty boundary

What prior work established, and what E3-SSE newly tests. No "first" claim is made
until the full-text verification below is complete. Every checkmark in the
manuscript's prior-work table must cite the paper's own text, not its abstract.

## Verification status of sources

| Source | Checked against | Status |
|---|---|---|
| Deng et al., npj Comput. Mater. 11, 9 (2025), doi 10.1038/s41524-024-01500-6 | full text (Ceder-group PDF) | verified for the claims below |
| Jin & Candès, weighted conformal selection, arXiv:2307.09291v2 (Biometrika 113(1) asaf066, 2026) | full text, arXiv v2 (Sections 2–3) | method verified; Biometrika version not yet compared |
| LiTraj, Dembitskiy et al., npj Comput. Mater. 11, 131 (2025) | repository code at commit c3ca5c2 (NEB notebooks, DFT driver) | code verified; paper text not re-read |
| Bheemaguli et al., arXiv:2512.03642 | abstract only | unverified beyond abstract |
| Schultz et al., npj Comput. Mater. (2025), doi 10.1038/s41524-025-01573-x | arXiv abstract and metadata | unverified beyond abstract |
| Wong & Yang, arXiv:2603.10159 (JCTC 2026) | abstract only | unverified beyond abstract |
| OMat24 (Barros-Luque et al. 2026), MatPES (Kaplan et al. 2025), Bai et al. JCIM 2025, Jin & Candès JMLR 2023, Li & Walsh (preprint) | not read in this pass | unverified |
| `E3-SSE_intro_literature-sent to the team.docx` | — | supporting source not locally available during this implementation pass; claims depending on it alone remain unconfirmed |

## Established by prior work (not claimed)

- Systematic softening of uMLIP potential-energy surfaces. The softening scale is the fitted slope of ML vs DFT forces in a material (Deng et al.).
- A scalar energy correction `E_corr = c · MLIP`, fitted from one high-energy structure, and tested on forces of the same material's configurations (Deng et al.). Dividing a barrier by ŝ is this correction applied to barriers.
- uMLIP Li-migration benchmarks and the nebDFT2k/MPLiTrj data (LiTraj). Fine-tuning comparisons. Endpoint-only fine-tuning reintroduces softening (LiTraj; bears directly on the risk of the material-level scale estimated from endpoint frames).
- Mechanical similarity (classical). The barrier decomposition (s−1)ΔE_DFT + Δδ is algebra.
- ρ_F = |sin θ| = sqrt(1 − uncentred R²) of the zero-intercept parity fit (standard statistic).
- Feature-space kNN/KDE applicability domains (Schultz et al.). Latent-space atypicality as a failure signal in uMLIPs (Wong & Yang; Li & Walsh, to verify).
- Risk–coverage curves and AURC (Geifman et al.).
- Conformal selection and weighted conformalized selection under covariate shift (Jin & Candès). Conformal selection in drug discovery (Bai et al., to verify).

## Newly tested or contributed by E3-SSE (subject to results)

| Claim | Evidence required |
|---|---|
| Measured transfer of a leakage-free scalar correction to unseen Li hops across six frozen models (ρ_SRT on paired hops). "Leakage-free" describes the design, not a contribution | Paired raw vs SRT on identical hops; oracle gap; per-hop leakage audit |
| Pre-outcome diagnostics predict where the scalar correction can be trusted (correct-or-abstain) | ΔAURC vs the random abstention baseline with a cluster-bootstrap CI (P4 / H6), compared descriptively with the \|raw barrier\| baseline |
| Downstream label budget (K → barrier error) with common random numbers | K-curves with Monte Carlo and cluster uncertainty; endpoint frames, so not comparable to Deng's single high-energy structure |
| Reliability-gated, FDR-controlled selection for uMLIP Li-hop screening | Realised FDR ≤ α + 2 MC s.e. (retrospective); approximate FDR control (prospective) |
| Frozen prospective screen with a deployment-faithful ML-NEB predictor | Shortlist manifest frozen before literature lookup |

## Explicitly not claimed

- No transport rescaling (Research Design contribution C1, transport part): Study 3 is deferred.
- No pre-registration: the active protocol is an internal, version-controlled freeze.
- No new conformal theory, applicability-domain method or MLIP architecture.
