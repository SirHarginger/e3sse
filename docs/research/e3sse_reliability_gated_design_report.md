# E3-SSE reliability-gated design report

Status: began as the pre-implementation report (task Section 49) and is kept current; it states plans and
implementation status, not scientific results. Repository HEAD `0fab51e` on `feature/study1-softening-atlas`,
now developed on `feature/reliability-gated-e3sse`. Date: 2026-11-01.

Public names for the method stages, used in figures, tables and manuscript
outputs:

| Internal (code, provenance) | Public stage |
|---|---|
| Study 1 | CALIBRATION + APPLICABILITY |
| Study 2a | UNSEEN-KINETICS TRANSFER |
| Study 4 | RISK-CONTROLLED SELECTION |
| Study 5 | PROSPECTIVE DISCOVERY |

Internal names stay in code, because renaming them would break resume
identities and pinned provenance.

---

## Protocol identity and governance

Protocols are named for what they do. Their identity comes from content hashes, never from a sequence number.

- **Registration-gated historical protocol.** `protocol_id 5f9aa711…`, `protocol.json` SHA-256 `6b3ed746…`, freeze commit `971f4c42`. It is immutable; its files, ID and OSF-registration semantics are never modified.
- **Reliability-gated internal-freeze protocol** (`configs/reliability_gated_internal_freeze_protocol.json`, `src/e3sse/protocol/internal_freeze.py`). Lineage fields: `parent_protocol_id`, `parent_protocol_sha256`, `supersedes_protocol_id`, `supersedes_protocol_sha256`, `governance_mode`. Its own `protocol_id` is content-derived.

The active internal-freeze protocol:
- does **not** claim pre-registration (`preregistration_claimed: false`);
- does **not** require OSF at runtime (`external_registration_required: false`);
- is specified before any real production outcome is inspected. This is declared by the owner. The builder checks it only against Study-1 outcome-access and run records; barrier, pilot and FPMD reads are excluded by the code-level seals;
- inherits every unchanged scientific decision from the parent through its immutable `protocol_id` and file SHA-256;
- intentionally changes governance (internal freeze) and scope (Study 3 deferred, Study 2b folded into ML-NEB calibration). Both are declared under `declared_scope_revisions`.

It is **not frozen yet**: blocking open items remain (Section K).

### Development, calibration and prospective populations

```text
DEVELOPMENT POPULATION
    → define/freeze predictor, applicability metric, support reference population,
      threshold, ML-NEB settings, prospective model
CONFORMAL CALIBRATION POPULATION
    → calibrate conformal evidence only
PROSPECTIVE POPULATION (BVEL13k)
```

Calibration outcomes may never select the model, tune the applicability gate, tune ML-NEB, or otherwise change the predictor. The support population and threshold come from the disjoint development population, not from the calibration set. The reviewers flagged the calibration-set variant because it would make the predictor depend asymmetrically on calibration covariates.

### Residual equivalence

ρ_F = |sin θ| for the stacked DFT and ML force vectors, and the existing Study-1 `residual_fraction` = 1 − S_xy²/(S_xx S_yy) = sin²θ = ρ_F². The canonical stored quantity is `residual_fraction`; ρ_F = sqrt(residual_fraction) is a derived display value. Because the square root is strictly increasing on [0, 1], both give identical rankings, risk–coverage orderings and AURC. No duplicate statistic is stored. This is tested algebraically (`test_rho_f_identities_and_scale_invariance`).

### Random abstention baseline

Under absolute-error risk, a uniformly random abstention order has E[risk(k)] = full-coverage MAE at every coverage, so AURC_random = MAE exactly. It is used analytically, labelled "random abstention baseline", and never presented as a method.

### Supporting source not available

`E3-SSE_intro_literature-sent to the team.docx` was not locally available during this implementation pass. No content is derived from it, and any novelty claim that depends on it alone remains unconfirmed.

---

## A. Repository map

| Concern | Files (exact) | State |
|---|---|---|
| Protocol content and freeze | `src/e3sse/protocol/freeze.py` (`_protocol_content` L783–1160, `materialize_protocol`), `decisions.py` (`DECISION_REGISTER`, `REQUIRED_RESOLUTION_ITEMS`), `configuration.py`, `partitions.py`, `chemistry.py`, `linkage.py`; config `configs/protocol_freeze.json`; CLI `scripts/build_protocol_freeze.py` | Frozen registration-gated historical protocol `5f9aa711…` materialized on the server (`protocol.json` SHA `6b3ed746…`, freeze commit `971f4c42`) |
| Registration (OSF) | `src/e3sse/protocol/registration.py`, `scripts/record_registration.py` | OSF addendum gates every outcome (seal 1) |
| Outcome access control | `src/e3sse/protocol/contracts.py`: `require_outcome_access`, `validate_protocol_freeze_for_outcome_access`, `build/validate_pilot_amendment`, `build/validate_transport_addendum`, `require_repeat_training_side` | Default-deny; two seals |
| Leakage | `contracts.py`: `audit_softening_frame_eligibility(_batch)`, `_audit_selection`, `require_primary_softening_selection`, `ELIGIBILITY_RULE="research_design_3.6_v1"`, `NEB_PATH_FILE_KINDS`, `ENDPOINT_RELAXATION_FILE_KINDS` | Implemented, tested |
| Support rule | `contracts.validate_softening_support`, constants `FULL_ESTIMATE_MIN_FRAMES=10`, `…_MIN_INFORMATIVE_COMPONENTS=100`, `INFORMATIVE_FORCE_EV_PER_A=0.1`, `POOLED_MIN_MATERIALS=3`, `LABEL_BUDGET_MIN_POPULATION=60`; `softening/frame_statistics.support_status` | Implemented |
| Study 1 (calibration) | `src/e3sse/softening/{frame_statistics,study1_inputs,study1_inference,study1_analysis,study1_tables,study1_figures,extxyz_frames}.py`; `scripts/{run,analyze}_study1_softening_atlas.py`, `scripts/plot_study1_figures.py`; `configs/study1_softening_atlas.json` | Complete pipeline. Production inference not yet run (it was waiting on OSF registration) |
| Model adapters | `src/e3sse/protocol/adapters.py` (`build_calculator`, `smoke_test`, `checkpoint_digest`); registry in `configs/protocol_freeze.json` → `protocol.model_groups` | 6 frozen models, smoke-tested, 5 isolated runtimes (server `~/.venvs`) |
| Embeddings | — | **None.** No extraction code; no local model runtimes (`~/.venvs` absent locally) |
| Barrier code | `src/e3sse/barriers/__init__.py` (empty) | **Not implemented.** Contract only (`study_2a_contract`) |
| Conformal / FDR | `src/e3sse/selection/__init__.py` (empty) | **Not implemented.** Contract and frozen draw seeds only (`study_4_contract`, `statistical_plan.randomness.study4_draw_seeds`) |
| Prospective screen | `src/e3sse/screening/__init__.py` (empty); `study5_exclusions` materialized by `freeze.py` | **Not implemented** beyond the exclusion set |
| Plotting | `src/e3sse/softening/study1_figures.py` (seaborn + OO Matplotlib, IEEE widths, 600 dpi, figure JSON) | Study-1 only; no shared style module, no figure registry |
| Execution | `src/e3sse/execution/{sharded_execution,resource_planning,thread_control,worker_memory_probe}.py`; `scripts/probe_model_worker_memory.py` | Deterministic, resumable, memory-planned. Reused as is |
| Manifests | JSON throughout: `e3sse.protocol.freeze.v1`, `e3sse.execution.manifest.v1`, `e3sse.study1.*`, `e3sse.mlip.inference.run.v1` | Reused |
| Data provenance | `src/e3sse/data/{dataset_audit,mplitrj_provenance,litraj_provenance,unit_evidence,fpmd_aiida_inventory}.py` | G0 audit + frame→hop index on server |
| Tests | `tests/test_protocol_*.py` (freeze, contracts, leakage, label budget, registration, safeguards, linkage, chemistry/models, configuration), `test_study1_*.py`, `test_parallel_execution_determinism.py`, `test_g0_parallel.py`, `test_dataset_audit.py`, fixtures `tests/{protocol,study1,parallel_execution}_fixtures.py`, `tests/fixtures/g0/` | Synthetic-only |
| Safety hooks | `.agents/hooks/safety_guard.py`, `.agents/scripts/validate_repo.sh`, `.agents/scripts/record_provenance.py` | Active |

Source documents read: `AGENTS.md`, `CLAUDE.md`, `docs/research/E3-SSE_Research_Design.docx` (RD), `E3-SSE_Paper_Narrative_and_Related_Work.docx`, `AI__Coding_Agent_Config_and_Prompt_Guide.docx`, `docs/workflow.md`, `configs/protocol_freeze.json`.
**Not found anywhere on this machine:** `E3-SSE_intro_literature-sent to the team.docx`.

---

## B. Scientific-contract map

"—" means it does not exist yet. "internal-freeze" means it is new in the internal-freeze protocol and must be frozen before any evaluation outcome is read.

| Rule | Source | Canonical config / decision | Implementation | Test | Output artifact |
|---|---|---|---|---|---|
| Zero-intercept Eq. 17 ŝ | RD 3.6 | `study_1_contract.estimator` | `frame_statistics.eq17`, `combine` | `test_study1_statistics` | `derived/study1_softening_estimates.csv` |
| Whole-frame bootstrap, 10 000 (≥2 000) | RD 3.6, 5.11 | `study_1_contract.bootstrap_resamples`; `DRAW_MAPPING` | `frame_statistics.iter_bootstrap_indices`, `study1_analysis._bootstrap` | `test_study1_statistics` | CI columns |
| Support: ≥10 frames, ≥100 comps with \|F_DFT\|≥0.1 eV/Å, pooled ≥3 materials, no fallback | RD 3.6 + `estimator_minimum_support` | protocol resolution | `contracts.validate_softening_support`, `support_status` | `test_protocol_contracts`, `test_study1_*` | `support_reasons` columns |
| Leakage: own-path prohibited (oracle only), endpoint relaxation admissible, other hops admissible, unknown fails closed | RD 3.6, 5.4 | `leakage_rules`, `research_design_3.6_v1` | `contracts.audit_softening_frame_eligibility*`, `require_primary_softening_selection` | `test_protocol_leakage` | `frozen/` eligibility units |
| Test-side material ŝ: non-path frames only | RD 3.6, 5.4 | `test_side_material_scope_frame_population` | `contracts._audit_selection` | `test_protocol_leakage` | — |
| 10 × 60/40 chemical-system partitions, base seed 2026; published split primary | RD 5.4 | `partition_definition` | `protocol/partitions.py` | `test_protocol_freeze`, `test_protocol_safeguards` | partition manifest |
| TM/Hubbard-U primary exclusion (+ f-block sensitivity) | RD 5.3, 3.9 | `primary_tm_hubbard_u` | `protocol/chemistry.py` | `test_protocol_chemistry_models` | population labels |
| Anion family `dominant-anion-count-v1` | RD 5.6 | `mixed_anion_family` | `protocol/chemistry.py` | same | family column |
| Configurational-energy deciles | RD 5.6 | `configurational_energy_deciles` | `study1_analysis.energy_deciles` | `test_study1_statistics` | `study1_energy_deciles.json` |
| Label budget K∈{1,3,5,10,30}, R=1000, N≥60, CRN across models, plateau ≤0.03 | RD 3.6, 4 H4 | `label_budget_stability` | `study1_analysis.label_budget_*` | `test_protocol_label_budget` | label-budget CSVs |
| s̄ = median hop-level material ŝ; s_host = non-Li components | RD 4, 3.5 | `s_bar_definition`, `s_host_definition` | `study1_tables.model_summary`, `COMPONENT_SETS` | `test_study1_*` | `study1_model_summary.csv` |
| Barrier = max − min reference image energy | RD 5.3; unit evidence | `inclusion_exclusion_rules.barrier_definition` | — | — | — |
| Study 2a comparators, metrics, c ∈ {0.3, 0.5} eV, ρ_SRT (Eq. 16), paired units | RD 5.7, 3.5 | `study_2a_contract` | — | — | — |
| Chemical-system cluster bootstrap (frozen 10 000) | RD 5.11 | `statistical_plan` | — (Study 1 uses whole-frame only) | — | — |
| Holm over {P1_m} ∪ P2; P3 is a validity check | RD 5.11 | `holm_family` | — | — | — |
| Jin–Candès clipped score, U=1, BH; BY sensitivity | RD 3.8 | `statistical_plan` | — | — | — |
| Study 4: 10 partitions × 200 draws, frozen seeds, SHA hop rank | RD 5.9 | `statistical_plan.randomness.study4_*` | — (seeds frozen) | — | — |
| MC SE = SD_p(FDR_p)/√10 | RD 6 G4 | `g4_monte_carlo_standard_error_definition` | — | — | — |
| Study 4 predictor independence (pooled ŝ from systems without nebDFT2k hops) | RD 3.8 | `study4_predictor_calibration_independence` | — | — | — |
| Study 5 exclusion = all partition-universe systems; fail closed | RD 5.4, 5.10 | `study5_evaluation_system_scope` | `freeze.py` `study5_exclusions` | `test_protocol_freeze` | in protocol |
| Validation split = sealed reserve | RD 5.2 | `published_validation_split_role` | `contracts` roles | `test_protocol_safeguards` | — |
| Outcome seals (freeze + OSF; pilot amendment) | RD 4, 5.1 | decision `two_stage_outcome_seal` | `contracts.require_outcome_access` | `test_protocol_registration` | `outcome_access/` |
| 6 frozen models, adapters, units | RD 5.5 | `model_groups`, `unit_contract` | `adapters.py`, `freeze._verify_production_evidence` | `test_protocol_chemistry_models`, `test_unit_evidence` | smoke reports |
| **Governance: internal freeze, no OSF** | task §10 | internal-freeze amendment | — | — | — |
| **Applicability diagnostics + gate** | task §2C, §3 | internal-freeze amendment | — | — | — |
| **Risk–coverage / AURC** | task §4 | internal-freeze amendment | — | — | — |
| **B4 gated SRT, B5/B6 definitions, downstream label budget** | task §5, §17 | internal-freeze amendment (B5/B6 already in RD) | — | — | — |
| **Representation support** | task §21 | internal-freeze amendment | — | — | — |
| **Figure/table registries** | task §25, §31 | registries (not science-bearing) | — | — | — |

Three places where the task and the frozen protocol differ. The internal-freeze protocol must record each one as a decision; none may change silently.

1. The registration-gated historical protocol's `study_4_contract.predictors` = [raw, SRT_global, SRT_family]. The task adds reliability-gated SRT. **The internal-freeze protocol adds it**, and the registration-gated historical protocol predictors are unchanged.
2. The registration-gated historical protocol primary endpoint P1 is the material-level ρ_SRT. The task's thesis centres on correct-or-abstain. **Recommendation:** keep P1 unchanged and add the applicability endpoint P4 (below) before any test outcome is read.
3. The registration-gated historical protocol's seal 1 requires OSF registration. The task removes OSF from runtime. **The internal-freeze protocol governance replaces it for the internal-freeze protocol only.** the registration-gated historical protocol code paths and the registration-gated historical protocol stay byte-for-byte.

---

## C. Novelty-boundary map (first pass; see the verification status at the end)

| Proposed contribution | What prior work already did | What E3-SSE does differently | Evidence required |
|---|---|---|---|
| A. Scalar softening correction of barriers | **Deng et al. 2025** (verified from full text): softening scale = "linear fitted slope of uMLIPs vs. DFT forces in a material". Proposes `E_corr = c·MLIP` (a scalar on the energy, implemented as a frozen model plus one scalar layer), fitted from one high-energy configuration. Tested on **forces of the same material's configurations** (test forces "originate from the same set of atomic arrangements") and on surfaces (SI). Barriers (477 Mg paths, ApproxNEB) are benchmarked **raw only**. Mechanical similarity is classical. | Dividing ΔE_ML by ŝ is *mathematically Deng's scalar correction applied to barriers*, so the correction itself must **not** be claimed as new. What is new is the barrier-level decomposition (s−1)ΔE_DFT + Δδ, the explained fraction ρ_SRT, and its empirical test on Li hops | Paired raw vs SRT on identical hops; ρ_SRT with a chemical-system CI; residual decomposition |
| B. Leakage-free transfer to unseen kinetics | Deng: no transfer to a held-out process. LiTraj (Dembitskiy 2025): fine-tuning on MPLiTrj and benchmarks; no scalar transfer | ŝ from frames that exclude the target hop's own path (audited, fail-closed), applied to that hop; own-path oracle as an upper bound | Paired test-split improvement; oracle gap; leakage audit artifact per hop |
| C. Applicability / correct-or-abstain | Schultz, Wang, Jacobs, Morgan, npj Comput. Mater. 2025 (DOI 10.1038/s41524-025-01573-x): KDE feature-space applicability domain for general ML models. Wong & Yang 2026 (arXiv 2603.10159; JCTC 22, 6820): PCA / Q-residual as an epistemic proxy in uMLIP fine-tuning. Li & Walsh (preprint, to verify): latent-geometry distortions signal MLIP failures. Chorna et al. (arXiv 2512.05717): fine-tuned uMLIPs keep a pretraining latent bias | Diagnostics of whether *one scalar* is an adequate local model of the error (force non-collinearity, scale uncertainty, Li/host disagreement, energy dependence, latent support), evaluated as selective prediction on unseen kinetics | ΔAURC versus random abstention, with a cluster-bootstrap CI that excludes 0; frozen gate; paired gated-vs-ungated results |
| D. Label budget → downstream barrier error | Deng: one structure suffices for the scalar layer (single case plus a 1000-compound force-MAE distribution) | Repeated subsampling with common random numbers across 6 models, K→barrier error on unseen hops | K-curve with Monte Carlo and cluster uncertainty |
| E. FDR-controlled selection | Jin & Candès 2023 (JMLR); Bai et al. 2025 (JCIM, drug discovery); Jin & Candès 2026 (Biometrika, weighted) | Application to uMLIP Li-hop screening, with a reliability-gated predictor and one-hop-per-system exchangeability | Realized FDR ≤ α + 2 MC SE; paired power gain |
| F. Prospective screen | Sendek 2017, Kahle 2020 funnels; LiTraj BVEL13k precedent; Bheemaguli 2025 (arXiv 2512.03642, 500 meV classification, no correction or gating) | Frozen, FDR-controlled shortlist plus an evidence ladder | Frozen shortlist manifest before any literature lookup |

Explicit non-claims (added after the 2026-11-01 research-architect review):
- The barrier decomposition (s−1)ΔE_DFT + Δδ is algebra. The contribution is the *measurement* of ρ_SRT across six frozen models on unseen hops, not the identity.
- ρ_F is sqrt(1 − uncentred R²) of the zero-intercept parity fit. It is a standard statistic and is not claimed as new.
- kNN/KDE applicability domains (Schultz et al. 2025) and AURC / risk–coverage (Geifman et al.) are borrowed tools. Latent support is presented as borrowed, against Wong & Yang 2026.
- "Leakage-free" describes the design, not a contribution. LiTraj's finding that endpoint-only fine-tuning reintroduces softening bears directly on the B3 risk and will be cited. The claim "LiTraj performs no scalar transfer" is still unverified.
- RD contribution C1 (transport rescaling) is **not** claimed while Study 3 is deferred.
- Row E rests on Bai 2025 and Jin–Candès, which have not yet been read in full.

Verification status: the Deng 2025 claims above were checked against the full text. The Bheemaguli 2025, Schultz 2025 and Wong & Yang 2026 entries were checked from the abstract or metadata only. LiTraj, OMat24, MatPES, Bai 2025 and both Jin–Candès papers still rest on the narrative document's 23 Sep 2026 check and **must be read in full before any checkmark in the prior-work table**. A dedicated `docs/research/novelty_boundary.md` will be kept from this table, and "first" claims are deferred until that pass is done.

---

## D. Benchmark matrix (information budget)

| ID | Predictor | DFT force labels used | Barrier labels used | Target-hop labels | Role |
|---|---|---|---|---|---|
| B0 | raw ΔE_ML on DFT-NEB images | none | none | none | baseline |
| B1 | ΔE_ML / ŝ_global | pooled training-side MPLiTrj (published train; repeat p: its training side) | none | none | deployable |
| B2 | ΔE_ML / ŝ_family | same, restricted to the target's anion family | none | none | deployable |
| B3 | ΔE_ML / ŝ_material | the target material's leakage-eligible frames (test side: non-path endpoint frames only); support-gated, no fallback | none | none | deployable; the registration-gated historical protocol P1 |
| B4 | gated B3 (and gated B1/B2 variants) | same frames as the underlying ŝ, plus the diagnostics computed from them | none | none | deployable; the internal-freeze protocol |
| B5 | a·ΔE_ML + b | none | training-partition nebDFT2k barriers | none | supervised comparator; visually separated |
| B6 | ΔE_ML / ŝ_own-path | the target hop's own NEB-path frames | none | **yes** | oracle upper bound only |
| B7 | published fine-tuned metrics | — | — | — | literature context table only; never paired |
| (sec.) | element-resolved (ŝ_Li-based) SRT | as B3 | none | none | secondary diagnostic |

Every comparison is computed on a frozen common-hop set and carries its SHA-256 (the `common_hop_population` pattern from Study 1). Excluded hops are counted.

---

## E. Figure registry (proposal; frozen as `configs/figure_registry.json` before production)

Main paper (IEEE two-column, 7.16 in unless noted):

| Fig. | Panel | Scientific question | Input (derived/) | Plot type | Statistics | Expected reading |
|---|---|---|---|---|---|---|
| 1 | — | What does E3-SSE do; where is the novelty? | none (TikZ) | method schematic plus PES inset (reference vs softened; ΔE_SRT = ΔE_ML/ŝ) | — | estimate → diagnose → correct/abstain → select → discover |
| 2 | A | How soft is each model? | Study-1 estimates | raincloud/violin of hop-level ŝ_material by model, median ± IQR points, line s=1 | whole-frame CI | s̄<1 for Group A |
| 2 | B | Does softening depend on chemistry? | Study-1 family estimates | heatmap: models × anion families, median ŝ, unsupported cells hatched | n per cell annotated | — |
| 2 | C | Is one scalar adequate across energies? | decile estimates | ŝ vs configurational-energy decile, per model, CIs | whole-frame CI | a slope indicates energy dependence |
| 2 | D | How many labels are needed? | label-budget + transfer K-curves | K vs \|ŝ_K − ŝ_full\| (left) and K vs barrier MAE after SRT (right, same K axis, small multiples, no dual axis) | MC + cluster CI | plateau K |
| 3 | A | Raw barriers vs DFT | transfer predictions | parity, hexbin or scatter, identical axes to B, parity line | n | underestimation |
| 3 | B | SRT barriers vs DFT | same hop identities | parity | n | shift toward parity |
| 3 | C | Does SRT help, paired? | paired errors | per-model paired ΔMAE (raw → SRT) with 95% cluster CI, B5/B6 drawn in a separate, visually distinct style | cluster bootstrap | effect size |
| 3 | D | Can we predict where to trust it? | risk–coverage | risk–coverage curves: ρ_F, scale-SD, latent, combined, random band, oracle (dashed) | AURC with CI | curves below random |
| 4 | A | Is the selection valid and useful? | Study-4 per-draw table | realized FDR vs mean true discoveries, nominal α line, MC-SE bars | MC SE | left of α, higher power |
| 4 | B | FDP and selection-size behaviour | per-draw | FDP distribution (box/strip) by method; zero-selection fraction | — | — |
| 4 | C | Prospective funnel | screen manifest | horizontal funnel with real counts per gate | — | — |
| 4 | D | What was shortlisted? | candidate evidence | evidence matrix (candidates × evidence levels) or PCA map with shortlist overlay | — | — |

Supplementary figures (all generated; the paper selection is decided later): S1 force parity per model (hexbin, zero-intercept line, ŝ, n); S2 force residuals vs \|F_DFT\|, \|F_ML\|, decile, element, family; S3 bootstrap distributions of ŝ; S4 ŝ_Li vs ŝ_host; S5 per-model raw/SRT barrier parity; S6 paired error distributions; S7 signed residual (Bland–Altman); S8 confusion matrices at c=0.3 and 0.5 eV; S9 risk–coverage per model × score, including the correct-or-raw variant; S10 latent PCA maps colored by ŝ, ρ_F, benefit, split membership (UMAP optional); S11 latent support vs raw error, SRT error and benefit; S12 FDR diagnostics (FDP, power, size, paired raw–SRT, partition spread); S13 prospective candidate-space map. Presentation only: 3-D PCA (Plotly HTML plus fixed static views).

## F. Table registry (proposal; `configs/table_registry.json`)

A1 datasets (role, n materials, n hops/frames from audited manifests, reference method, partition role, exclusions); A2 models (training set, checkpoint/version, family, runtime, regime); B core transfer (model × method: N paired, coverage, MAE, RMSE, signed error, Spearman, P/R/F1 at c, CI, relative improvement; deployable rows separated from B5/B6); C applicability (score: coverage, retained N, MAE at full/target coverage, AURC, E-AURC, normalized gain, abstention rate, Spearman with benefit); D label efficiency (K: softening error, bootstrap SD, downstream MAE, fraction supported); E FDR (method, α, c, realized FDR, MC SE, power, mean size, zero-selection fraction); F prospective candidates (filled only after the run); G prior-work comparison (source-verified checkmarks only); H literature context (published metrics, protocol mismatch stated). Full-precision tables go in `derived/`; rounded tables in `tables/`.

---

## G. Applicability plan

**Force non-collinearity ρ_F.** The task's candidate

  ρ_F = sqrt( Σ‖F_ML − ŝF_DFT‖² / Σ‖F_ML‖² )

has a closed form. With ŝ = S_xy/S_xx:

  ρ_F² = 1 − S_xy²/(S_xx·S_yy) = sin²θ,

where θ is the angle between the stacked DFT and ML force vectors of the calibration set. Consequences:

- ρ_F² is **exactly** the `residual_fraction` that Study 1 already computes from its per-frame sufficient statistics. No new inference or storage is needed, and it uses the same component weighting as ŝ.
- It is invariant to rescaling either force field, symmetric in DFT and ML, bounded in [0, 1], and 0 iff the forces are exactly proportional. That is the precise "uniform softening" null.
- Denominator choice: the DFT denominator gives sqrt(Σ‖·‖²/(ŝ²S_xx)) = tan θ. This is monotone in ρ_F, so **rankings, risk–coverage and AURC are identical**. Recommendation: freeze sin θ because it is bounded.
- Near zero: undefined when S_xx=0 or S_yy=0. The estimate is then unsupported, with a reason, and never imputed.
- Known bias: DFT force noise (σ_η) inflates ρ_F most on low-force frames. Test-side material calibration uses endpoint-relaxation frames, which are low-force, so ρ_F is reported together with the informative-component count and the mean \|F_DFT\|. The synthetic test checks that adding noise increases ρ_F.
- Frame weighting: the primary statistic is component-pooled (as ŝ is). A frame-median sin θ_f is a pre-specified sensitivity analysis. CI: whole-frame bootstrap, using the existing draws.

**Scale uncertainty.** Whole-frame bootstrap SD and the relative half-width (q₉₇.₅ − q₂.₅)/(2ŝ). This is stable because support guarantees that ŝ is far from 0.

**Element non-uniformity.** \|ln(ŝ_Li/ŝ_host)\| over the same frames, defined only when both component sets meet support (no fallback). This is the RD 3.5 "expected SRT failure" diagnostic.

**Energy dependence.** Within the calibration frames, a frozen median split on dE_f (eV/atom, using the existing decile energy reference): \|ln(ŝ_high/ŝ_low)\|, support-gated. A pooled view is the decile curve (Fig. 2C).

**Representation support.** See H.

**Combined score.** Equal-weight mean of within-population percentile ranks of the pre-declared diagnostics. **No learned weights**, because fitting a combiner would need barrier labels, and the registration-gated historical protocol rule `repeated_test_pilot_overlap_rule` forbids nebDFT2k barriers as tuning inputs.

**Risk–coverage (primary, threshold-free).** Sort hops by score, most trustworthy first. Coverage is k/n. risk(k) is the mean \|ΔE_SRT − ΔE_DFT\| over the top k. AURC = (1/n)·Σ_k risk(k) (Geifman et al.); ties are scored by their expected risk under random within-tie order, in closed form, so the result is order-invariant. E-AURC = AURC − AURC_oracle, and normalized gain = (AURC_rand − AURC)/(AURC_rand − AURC_oracle). The random baseline is analytic: E[risk(k)] = full-coverage MAE, so AURC_rand = MAE. A Monte Carlo band is shown for the figure only. CIs use a paired chemical-system cluster bootstrap across scores. Two framings are pre-specified (see question Q2):
  1. *selective*: abstained hops receive no prediction;
  2. *correct-or-raw*: abstained hops keep the raw prediction, and risk is computed over all hops. This framing is explicitly labelled, because it is not the "no-fallback" scale rule.

Proposed new endpoint **P4**: ΔAURC(ρ_F − random) < 0 under framing 1 for the SRT-material predictor, chemical-system cluster CI. It would be pre-specified (not pre-registered) in the internal-freeze protocol.

**Deployment gate (secondary, B4).** Pass iff the frozen support rule holds, ρ_F ≤ q_ρ, and the relative half-width ≤ q_CI. Each q is a **pre-declared quantile (proposal: 0.80)** of the hop-level distribution of that diagnostic on Study-1 published-train target hops. These are force-only and use no barrier label. The quantile level is frozen in the internal-freeze protocol. The gate is evaluated once on test hops.

## H. Latent-analysis plan

Status: **unverified locally.** The model runtimes exist only on the server. The plan below must be confirmed by a read-only probe in each pinned environment before the internal-freeze protocol is frozen. Pinned package versions are from the registry.

| Model (pinned) | Candidate access | Layer / module | Dim (to verify) | Note |
|---|---|---|---|---|
| MACE-MP-0 small (mace-torch 0.3.6) | `MACECalculator.get_descriptors(atoms, invariants_only=True)` | invariant node features of every interaction layer | expected 128 per layer | official API |
| CHGNet 0.3.0 | `CHGNet.predict_structure(..., return_atom_feas=True)` | final atom features before the readout | expected 64 | official API |
| M3GNet-MP / M3GNet-MatPES (matgl 1.2.7) | forward hook on the last graph layer node features | last `graph_layers[-1]` output | expected 64 | hook only; no architecture change |
| SevenNet-0 (0.10.1) / SevenNet-omat (0.11.0) | forward hook on the module before the atomwise readout | scalar (0e) node features | to verify | separate runtimes; hooks differ by version |

- Representation: atom-level invariant features of the **migrating Li**. It is identified from geometry only, as the atom with the largest minimum-image displacement between the first and last image. Features are taken on interior images and pooled by mean. A pre-specified variant uses the ML-highest-energy image only. **The image with the highest DFT energy must not be used**: choosing it needs the target outcome.
- Calibration population: Li-atom features of the frames that produced the hop's ŝ (local support). A global variant uses the pooled training-side frames.
- Normalization: per-model z-scoring with mean and SD fitted on published-train calibration features, frozen.
- Metric: mean Euclidean distance to k=5 nearest calibration neighbours, divided by the median leave-one-out 5-NN distance inside the calibration set. This makes the score scale-free per model. KDE log-density (Schultz et al.) is a sensitivity analysis.
- Cross-model: never compare distances across architectures. Analysis is per model, shown as small multiples. Canonical presentation model: MACE-MP-0 small, if its descriptor API is confirmed.
- Projection: PCA fitted on training-side calibration features is primary. UMAP is optional (fixed `random_state`, `n_neighbors`, `min_dist`). 3-D PCA in Plotly HTML for presentation. **Projected coordinates never enter any score.**
- Cost: captured in the same pass as the Study-2a single points (needed anyway for test-side calibration frames and nebDFT2k images). Only Li rows are stored, as float32.

## I. Implementation plan

### I.1 Files to create

| File | Purpose |
|---|---|
| `src/e3sse/protocol/internal_freeze.py` | internal-freeze protocol: embeds the registration-gated historical protocol by id + SHA, governance block (`internal_version_controlled_freeze`, `external_registration_required=false`, `preregistration_claimed=false`), new contracts; content-addressed `protocol_id`; validator |
| `configs/reliability_gated_internal_freeze_protocol.json` | internal-freeze decisions (applicability, gate quantiles, P4, B4–B6, latent, downstream label budget, prospective, registry hashes) |
| `scripts/build_internal_freeze_protocol.py` | materializes `outputs/protocol_freeze/<protocol_id>/` on the server; requires a clean tree and pinned SHA; `--resume` |
| `src/e3sse/softening/applicability.py` | ρ_F, scale uncertainty, Li/host and energy contrasts (pure functions on sufficient statistics) |
| `src/e3sse/barriers/fixed_geometry.py` | barrier definition, offset invariance, SRT, paired metrics, ρ_SRT, classification metrics, common-population hash |
| `src/e3sse/barriers/transfer_inputs.py`, `transfer_inference.py` | nebDFT2k image and test-side endpoint-frame loading, per-model single points plus embedding capture (reusing `execution/`) |
| `scripts/run_fixed_geometry_transfer.py`, `scripts/analyze_fixed_geometry_transfer.py` | plan/index/probe/infer/verify; analysis with `--resume` |
| `src/e3sse/selection/selective_prediction.py`, `scripts/analyze_selective_risk_coverage.py` | risk–coverage, AURC, E-AURC, ties, random/oracle, cluster CIs |
| `src/e3sse/selection/conformal.py`, `scripts/analyze_conformal_candidate_selection.py` | Eq. 20 p-values (U=1), BH, BY, weighted CS, per-draw FDP, MC SE |
| `src/e3sse/representation/{embeddings,support}.py`, `scripts/extract_foundation_model_embeddings.py` | adapters' hooks and API calls, kNN support, PCA/UMAP |
| `src/e3sse/screening/prospective.py`, `scripts/run_prospective_candidate_screen.py` | Gate 0–3 funnel, candidate manifest, frozen shortlist |
| `src/e3sse/figures/{publication_style,registry,manifest}.py` | central style (palette, markers, line styles, sizes 8–10 pt), registry loader, figure-manifest writer |
| `scripts/plot_{softening_calibration_atlas,transfer_and_applicability,selection_and_discovery}.py` | figures from `derived/` only |
| `configs/{figure,table}_registry.json` | registries |
| `docs/figures/e3sse_method.tex` | Fig. 1 (TikZ) |
| `docs/research/novelty_boundary.md` | maintained from section C |

### I.2 Files to modify (minimal)

- `src/e3sse/protocol/contracts.py`: `require_outcome_access` accepts a validated internal-freeze protocol (seal 1 = validated internal freeze; seal 2 = an internally hashed pilot amendment bound to its `protocol_id`). The registration-gated historical protocol path is unchanged.
- `src/e3sse/softening/study1_inputs.py` and the Study-1 scripts: accept an internal-freeze protocol whose parent equals the pinned registration-gated historical protocol. Study-1 inference identity stays bound to the registration-gated historical protocol `protocol_id`, so no rerun is needed.
- `src/e3sse/softening/study1_analysis.py`: emit ρ_F and the applicability diagnostics into a **new** derived table. Existing outputs stay byte-stable; analysis code hashes change, so analysis (not inference) recomputes.
- `docs/workflow.md`: an internal-freeze protocol section and production commands.

### I.3 Protected (no change)

`configs/protocol_freeze.json`; `freeze._protocol_content`, `decisions.DECISION_REGISTER` and `REQUIRED_RESOLUTION_ITEMS` (any edit changes what re-materializing the registration-gated historical protocol produces); `registration.py` semantics for the registration-gated historical protocol; `configs/study1_softening_atlas.json` pinned identity; `docs/research/*.docx`; the Study-1 inference identity code (`study1_inference.INFERENCE_CODE_FILES`); every raw data root; `.gitignore` beyond the user's pending edit.

### I.4 Tests to add

The full task §42 list:
- softening/applicability: perfect-scaling synthetic PES (ŝ → s, ΔE_SRT → ΔE_DFT, ρ_F → 0); non-uniform synthetic (ρ_F increases monotonically with the residual amplitude, SRT leaves the residual); noise inflation; DFT-denominator ranking equivalence;
- leakage: own-hop rejected, other-hop policy, unknown rejected, mixed list rejected (extend `test_protocol_leakage`);
- barrier: definition, additive-offset invariance, paired membership, unsupported means no fallback;
- applicability: determinism, risk–coverage, AURC with ties, random-baseline identity, no target label reachable (API takes no outcome argument for scores);
- latent: determinism, layer/pooling recorded, finite, original-space distances, projection random state;
- conformal: orientation, p-value edges, BH, empty/all selections, common draws, one hop per system, FDR, MC SE;
- prospective: exclusions, determinism, shortlist frozen before evidence lookup;
- governance: the registration-gated historical protocol bytes unchanged, the internal-freeze protocol id content-addressed, no OSF needed for the internal-freeze protocol, the registration-gated historical protocol still needs OSF.

### I.5 Unresolved scientific risks

1. **Calibration regime of test-side material ŝ.** It comes only from endpoint-relaxation frames (near-equilibrium, low force). Support may fail often, ρ_F carries a noise floor, and the near-equilibrium → transition-state extrapolation is exactly what is being tested. Coverage of B3 is unknown until the support pass, which is label-free and can run before outcomes.
2. **Prospective predictor mismatch** (Q4). Conformal calibration on nebDFT2k uses the *fixed-geometry* predictor. Prospective candidates have no DFT-NEB geometry and need ML-NEB, so f(X) would differ between calibration and test and exchangeability fails. Prospective candidates also have no DFT force labels, so only ŝ_global or ŝ_family, family-level diagnostics and latent support are available.
3. **Downstream label budget on test hops.** The frozen N ≥ 60 eligible-frame rule may leave few test-side hops (endpoint frames only).
4. **Transition-state image selection** must use ML energy or all images; the DFT maximum is target leakage.
5. **Embedding access** is unverified in the pinned runtimes (M3GNet and SevenNet need hooks).
6. **Outcome-access history on the server is unknown** to me (Q1).
7. **Endpoint changes** (P4 and gated predictors) alter the Holm/endpoint structure. This must be decided before freeze.
8. **Superseded OSF-pending work.** The earlier uncommitted Study-1 OSF-pending path is archived on branch `archive/osf-pending-study1-work` (commit `2c5bbb0`) as historical development evidence. It is not in the reliability-gated branch's ancestry, and the active protocol has no registration-dependent execution path.
9. **Literature verification pending** for LiTraj, OMat24, MatPES, Bai 2025, Jin–Candès 2023/2026 and Li & Walsh (full text).
10. **Schedule.** The RD plan has Study 2a in week 7 (now). Task priorities 1–8 are several weeks of work. Optional items (information-efficient calibration, transport) stay deferred.

---

## J. Owner decisions (2026-11-01), now encoded in `configs/reliability_gated_internal_freeze_protocol.json`

| Question | Decision | Where it is frozen |
|---|---|---|
| Outcome-access history under parent `5f9aa711…` | Nothing read yet: no Study-1 inference or analysis, no barrier, no FPMD reference | `outcome_access_declaration` |
| Meaning of "abstain" | Both framings. **Selective is primary** (no SRT prediction on abstained hops); correct-or-raw is a labelled secondary framing and is the gated predictor in conformal selection | `amendments.selective_prediction.framings`, `amendments.selection` |
| Deployable threshold | **Label-free quantile** (0.80) of force-derived diagnostics; no barrier labels; the registration-gated historical protocol tuning rule is untouched | `amendments.deployment_gate` |
| Study 3 / Study 2b | Study 3 and P2 are **deferred**; Holm runs over {P1_m} only; no transport claim. Study 2b is **folded** into the ML-NEB calibration runs as the secondary ΔE_ML-NEB − ΔE_fixed comparison. Recorded as a scope revision made before any outcome inspection | `amendments.scope`, `endpoints.primary_family` |
| Prospective gate | **Representation-support gate** in the original latent space, applied identically to development, calibration and pool units. In the selective analysis an unsupported unit abstains; in the conformal predictor it keeps the raw ML-NEB prediction. Support population and τ_d come from the disjoint development set (for exchangeability; see Section K). Fallback order: embeddings → pre-declared descriptor-space support → ungated SRT-family, the last only as a labelled secondary baseline | `amendments.prospective.support_gate` |
| Prospective predictor | **BVSE-initialised ML-NEB** for calibration and deployment alike. Development set (model selection) → separate ML-NEB conformal calibration set → BVEL13k pool with the identical pipeline. Fixed-geometry transfer is the physics validation of SRT, not the deployment predictor. Conformal-calibration outcomes are never used for model selection or ML-NEB tuning | `amendments.prospective` |

## L. Weighted conformal selection: conformance

- **Canonical source.** The published Biometrika main text (113(1), asaf066) is canonical. The equations were read from arXiv:2307.09291v2, including appendices A–C with the proofs of Theorems 3.1 (C.1) and 3.5 (C.4). The OUP supplement and article pages return HTTP 403 from this machine.
- **Equation mapping.**
  - The WCS p-value is eq. 5 (strict `<`, plus w_j). Eq. 4 is the randomized marginal p-value and is not the WCS p-value.
  - Auxiliary p-values follow eq. 6. Their added term carries w(X_{n+j}), consistent with the oracle eq. 33 in the C.1 proof.
  - Selection follows Algorithm 1 with deterministic pruning (eq. 9).
- **Tie rules.**
  - Primary: `tie_free_lexicographic`, i.i.d. protocol-RNG tie keys attached to units, giving an exact (score, key) order.
    - The proposed argument: the augmented problem is tie-free almost surely, monotone in y, and i.i.d., so Theorem 3.1 would cover it.
    - The statistical-validator approved this on 2026-11-01, after the keys were bound to unit identifiers: FDR ≤ q holds marginally over the key draw, not conditionally on the realised keys.
  - Sensitivity: `e3sse_conservative_le`, which equals the authors' official `weighted_CS` with its tie randomization fixed at U_j = 1.
- **Triangulation.**
  - Production vs the independent reference transcription: exact agreement on 480 randomized fixtures under both tie rules.
  - Production vs the official ConfSelect code (`ying531/conformal-selection` @ `feffee17`, `weighted_CS(rand="dtm")`, U ≡ 1): 200/200 tie fixtures agree with the sensitivity rule, and 195/200 continuous fixtures agree.
    - All five differences come from the official code using `test_weights[k]` (w_l) in eq. 6 instead of w(X_{n+j}).
    - With that one index changed, 200/200 agree.
    - Production follows the published Biometrika eq. 6 (w(X_{n+j})), consistent with the proof's eq. 33. This is recorded as a published-paper / reference-code discrepancy and pinned by a regression test.
- **Estimated-weight claim.**
  - Known weights: exact finite-sample control, Theorem 3.1.
  - Estimated weights fitted on data independent of all calibration and test units: only the Theorem 3.5 bound, q·E[γ²/(1 + q(γ² − 1)/m)]. It is not exact.
- **Simulation (q = 0.2), family "unmodelled shift-region predictor bias".**
  - Unweighted: FDR 0.786, power 0.798.
  - Known weights: FDR 0.000, power 0.000.
  - Estimated weights: FDR 0.000, power 0.000.
  - Weighting is necessary for validity here. With poor overlap it is valid but powerless, which the ESS and overlap diagnostics detect.

## M. Server status

`aiserver` (192.168.200.62) was unreachable from the development machine: SSH port 22 timed out, the machine was on Wi-Fi 192.168.0.200/24, and the wired link was down. The following are therefore pending:

- embedding capability probe;
- real-data rate-limiting-hop linkage audit;
- `ions` reproduction;
- ASE/FIRE pinning.

No server state was modified. The tooling is ready: `scripts/probe_embedding_capability.py` and `scripts/audit_rate_limiting_hop_linkage.py`.

## K. Implementation and freeze status

Implemented (synthetic tests only; no real-data outcome has been read):

| Component | Module |
|---|---|
| Internal-freeze protocol layer, internal pilot record, governance provenance | `protocol/internal_freeze.py`, `scripts/build_internal_freeze_protocol.py`, `contracts.require_outcome_access` |
| Applicability diagnostics, label-free gate | `softening/applicability.py` |
| Risk–coverage, AURC, analytic random abstention baseline | `selection/selective_prediction.py` |
| Conformal selection: Eq. 20, BH/BY, frozen draws, MC SE, weighted conformalized selection | `selection/conformal.py` |
| Fixed-geometry barrier core | `barriers/fixed_geometry.py` |
| Prospective support: development-population kNN, leave-one-chemical-system-out threshold, descriptor fallback, deterministic hierarchy | `screening/applicability_support.py` |
| Embedding-capability probe (server-side) | `representation/embedding_capability.py`, `scripts/probe_embedding_capability.py` |
| Publication style, figure/table registries | `figures/publication_style.py`, `configs/figure_registry.json`, `configs/table_registry.json` |
| Semantic naming check | `.agents/scripts/check_semantic_names.py` |

Not implemented yet: the fixed-geometry transfer inference runner, the Study-1 internal-freeze path, the ML-NEB runner, the prospective screen, plotting scripts and the TikZ Fig. 1.

**The protocol is not frozen.** Blocker ledger, generated from `configs/reliability_gated_internal_freeze_protocol.json`:

- **local_engineering** items must all be resolved, with reviewer evidence, before the engineering checkpoint commit. A test enforces this.
- **server_evidence** items must be resolved before the internal scientific freeze.

| Item | Class | Status | Resolution evidence / required resolution |
|---|---|---|---|
| `embedding_extraction_verification` | server_evidence | **blocking** | read-only probe in every pinned runtime: module, layer, dimension, determinism; record per model or mark latent support unavailable |
| `ml_neb_settings` | local_engineering | resolved | mlip-specialist: LiTraj recovery verified; every numeric setting now recorded; version slots are server evidence |
| `statistical_validator_review` | local_engineering | resolved | statistical-validator local specialist review 2026-11-01: 1,3,5,6,8 approve-with-fixes (applied), 2,4,9,10 approve, 7 rejected then fixed |
| `research_architect_review` | local_engineering | resolved | research-architect local specialist review 2026-11-01: rejection items implemented (access paths, splits, numeric settings, stale text) |
| `mlip_specialist_review` | local_engineering | resolved | mlip-specialist local specialist review 2026-11-01: probe fixes applied (translation, permutation, A-B-A, environment sensitivity, cloned capture, pre-hooks, call counts); criteria pre-specified |
| `literature_full_text_verification` | documentation | non-blocking | full-text check of LiTraj, OMat24, MatPES, Bai 2025, Jin-Candes 2023/2026 before novelty claims; does not change any analysis rule |
| `study3_and_study2b_scope` | documentation | resolved | project owner decision 2026-11-01, before any outcome inspection |
| `prospective_gate_source` | documentation | resolved | project owner decision 2026-11-01; development-set support population per statistical-validator and research-architect reviews |
| `prospective_pipeline_details` | local_engineering | resolved | research-architect: remaining parts split into server items (bvse_reproduction, descriptor_availability) |
| `combined_score_without_latent` | local_engineering | resolved | research-architect: resolvable from existing config text |
| `prospective_descriptor_fallback_metric` | local_engineering | resolved | research-architect and data-engineer fixes applied (local specialist review 2026-11-01); availability is server evidence |
| `specialist_rereview` | local_engineering | resolved | statistical-validator, research-architect, code-reviewer, data-engineer, mlip-specialist (local specialist review 2026-11-01); findings applied in this pass |
| `ions_bvse_version_pin` | server_evidence | **blocking** | pin the ions package version for BVSE initialization and verify its paths reproduce nebDFT2k _init.xyz for calibration hops |
| `failed_neb_policy` | local_engineering | resolved | statistical-validator APPROVE after fixes (local specialist review 2026-11-01); tests/test_prospective_contract.py |
| `density_ratio_features` | local_engineering | resolved | statistical-validator APPROVE estimator and split (local specialist review 2026-11-01); feature availability is server evidence (descriptor_availability) |
| `prospective_gate1_rule` | server_evidence | **blocking** | rate-limiting-hop rule on the BVSE percolating network, from the G0-audited BVEL13k/nebBVSE122k schemas (data-engineer) |
| `weighted_conformal_publication_check` | local_engineering | resolved | published eq. 5/6 mapping and official ConfSelect triangulation; statistical-validator APPROVE |
| `prospective_model_selection_rule` | local_engineering | resolved | statistical-validator rejection items (circular coverage, hop population, winner's curse) addressed (local specialist review 2026-11-01) |
| `applicability_fallback_hierarchy` | server_evidence | **blocking** | record which hierarchy branch applies per model from embedding_capability_manifest (rule itself is closed locally) |
| `representation_geometry_source` | documentation | resolved | owner decision 2026-11-01, before any outcome inspection |
| `hop_reduction_rule` | documentation | resolved | owner decision 2026-11-01, before any outcome inspection |
| `development_access_route` | local_engineering | resolved | research-architect requirement implemented |
| `study1_internal_freeze_path` | local_engineering | resolved | research-architect requirement implemented; tests/test_internal_freeze_protocol.py |
| `descriptor_family_handling` | local_engineering | resolved | statistical-validator APPROVE after one-hot fix; screening.prospective_contract.anion_family_record |
| `embedding_probe_server_run` | server_evidence | **blocking** | run scripts/probe_embedding_capability.py in every pinned runtime, pin hook captures where a documented API is absent, merge embedding_capability_manifest.json |
| `path_representation_mapping` | server_evidence | **blocking** | freeze the per-model path-level representation after embedding verification (mlip-specialist), before any calibration outcome |
| `rate_limiting_hop_algorithm` | server_evidence | **blocking** | data-engineer verification of amendments.prospective.rate_limiting_hop_algorithm on the actual data |
| `server_unreachable` | server_evidence | **blocking** | aiserver (192.168.200.62) was unreachable from the development machine (port 22 timeout; wired link down); every server audit is pending network access |
| `published_eq6_confirmation` | local_engineering | resolved | owner-supplied published equation; statistical-validator APPROVE (eq. 33 symmetry argument); regression test test_published_eq6_focal_weight_is_w_j_not_the_reference_code_w_l |
| `tie_rule_theory_review` | local_engineering | resolved | statistical-validator APPROVE (local specialist review 2026-11-01) after unit-bound keys; tests/test_weighted_conformal_conformance.py |
| `population_dimension_and_structure_rule` | local_engineering | resolved | statistical-validator and research-architect: d = 3 rationale sound and fixed before any count |
| `model_selection_constraints` | local_engineering | resolved | statistical-validator: floor and tie-breakers acceptable |
| `bvse_reproduction` | server_evidence | **blocking** | ions version pin and ML-NEB-to-nebDFT2k path correspondence from the server reproduction audit |
| `descriptor_availability` | server_evidence | **blocking** | source-field availability of every descriptor on the real schemas (data-engineer server audit) |
| `development_manifest` | server_evidence | **blocking** | build and pin development_manifest_sha256 from the server linkage audit before any development outcome |
