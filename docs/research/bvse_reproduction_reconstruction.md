# nebDFT2k BVSE initial paths: reconstruction of the generation method

Date: 2026-11-04. Candidate records: `bvse_reproduction_candidates.json`.
Re-run: `scripts/bvse_reproduction_diagnostic/`. `harness.py` re-implements every candidate on a fixed
pre-amendment base recipe (C0 is the former production task). `run_prod.py` runs the current
production `recipe()` + `bvse_path_task`. `summarize.py` writes one record per candidate.

## Outcome

The pinned recipe (LiTraj `notebooks/bvse_neb.ipynb`, ions 0.4.1) is **refuted**. The
server audit found a calibration reproduction rate of 0.0, with a max-position median of
0.2476 Å. The cause is the recipe, not the ions release. The reconstructed method
reproduces 1666 of 1667 checked public nebDFT2k paths to about 1e-8 Å. It runs on a lock
byte-identical to the server BVSE runtime.

The amendment is post-evidence and outcome-free:
- Six historically motivated candidates were compared against calibration-side
  `_init.xyz` geometry.
- No DFT label, relaxed path or ML-NEB result was read.
- The server rate under the amended recipe is therefore an in-sample fidelity check of
  the reconstruction, not a pre-declared test.
- The acceptance rule (0.01 Å per atom, 1e-4 Å cell, identical counts, rate > 0.5) is
  unchanged.

## Timeline

| Date | Event |
|---|---|
| 2024-04-23 | ions 0.3.3.1 on PyPI (= git `2d23c65`). No `distort`; BVSECalculator without stress. |
| 2024-08-20 | ions 0.3.5 on PyPI. No matching commit (snapshot of the 0.4 refactor). First release with `distort`, `SaddleFinder.run` and per-atom `stresses`. |
| 2024-08-29 | ions 0.4 on PyPI (= `5bc0674`). Its README documents the procedure: FIRE, subtract `info['perturbation']`, FIRE again. |
| 2024-10-30 | LiTraj initial commit. |
| 2024-10-31 | LiTraj manuscript received. |
| 2025-03-04 | ions 0.4.1 (= `b01c1e0`) released. LiTraj code upload `a195f62` the same day; the `bvse_neb.ipynb` demo targets 0.4.1. |

For this recipe's code path, ions 0.3.5, 0.4 and 0.4.1 are algorithmically identical.
The only differences are prints, the `ase.neb` → `ase.mep` import, featurizers and a
version string. On the diagnostic panel, C2 gives the same result on 0.4.1, on 0.4, and
on 0.3.5 with the 2024-08-21 stack (ASE 3.23.0); the agreement is about 1e-13 Å.

## Archive evidence (all 1681 public `_init.xyz`)

- **Code family.** Every frame carries per-atom `stresses` and a `stress`, which only the
  ≥ 0.3.5 `BVSECalculator` computes. The archive was therefore written by the
  0.3.5/0.4/0.4.1 code family.
- **No perturbation.** Framework copies of the same unit-cell atom are exact lattice
  translates (max 1.5e-8 Å), and the framework is identical across images.
- **Recipe mismatch explained.** The kept perturbation (`0.2·z/max|z|`, rng 42) gives a
  per-structure max |noise| of 0.2441, 0.2476 or 0.2608 Å. That is exactly the server
  distribution.
- **Endpoints not relaxed.** Endpoint Li sits on its lattice site (max 1.3e-8 Å). In
  94.5% of endpoints it still feels a BVSE force above 0.1 eV/Å, so `relax_endpoints`
  was off.
- **Oxidation states.** The archived `oxi_states` equal pymatgen BVAnalyzer valences.
  - Sampled misses (paths not reproduced): 103 of 112 match BVAnalyzer; ions `Decorator`
    matches 0 of 112.
  - Sampled controls (reproduced paths): 112 of 112 match BVAnalyzer.
  - LiTraj's notebook recommends BVAnalyzer for many structures.
- **Image count.** 1253 paths have 5 images (`Edge.length` 2.049–4.494 Å) and 428 have 7
  (4.502–5.989 Å).
  - All 1681 paths fit "7 if length ≥ 4.5 Å else 5".
  - This rule appears in no ions or LiTraj source. The ions default spacing rule matches
    1409; `n_images=5` matches 1253.
  - It is adopted under the frozen `image_count` provision as audit evidence. Keeping
    `n_images=5` would exclude every calibration hop ≥ 4.5 Å while pool hops of every
    length remain. The `hop_length_A` density-ratio feature would then lack calibration
    support (statistical-validator review).
- **Comparator.** The comparator was audited and is sound. The reader is bitwise equal to
  ASE, and there is no wrapping, triclinic or ordering artefact. A false pass would need
  same-element atoms within 0.02 Å; the minimum is 1.999 Å.

## Candidates (frozen rule; full set unless noted)

| Candidate | Reproduced / checked | Rate | Max-position median |
|---|---|---|---|
| C0 current pin | 0 / 1653 | 0.0 | 0.2476 Å |
| C1 `distort=False` (panel) | 3 / 24 | 0.125 | 0.022 Å |
| C2 `run` without endpoint relaxation, Decorator | 1148 / 1653 | 0.694 | 1.3e-8 Å |
| C3 subtract, no re-relaxation (panel) | 0 / 24 | 0.0 | 0.146 Å |
| C4 `run` with endpoint relaxation (panel) | 0 / 24 | 0.0 | 0.284 Å |
| C5 = C2 + BVAnalyzer, `n_images=5` | 1243 / 1667 | 0.746 | 1.3e-8 Å |
| **Amended production recipe** (C5 + 5/7 rule) | **1666 / 1667** | **0.9994** | **1.3e-8 Å** |

Production-recipe exclusions:
- **1 geometry miss:** `mp-775285_1_3_0_0_0`, 0.070 Å.
- **14 ions failures:**
  - 10 hops of `mp-752986`. BVAnalyzer assigns O⁻¹, which has no BVSE parameter; the
    archive has Ni³⁺/Ni⁴⁺ and O²⁻, and neither decorator reproduces it.
  - 4 pre-existing RuntimeErrors.
- **Not converged:** 8 paths are not BVSE-converged (recorded, not excluded).

## Caveats for the server run

- The public download must be compared with the server G0 manifest. The zip and shard
  SHA-256s are in the JSON.
- The server checks one bottleneck unit per system for development and P, not all 1681
  hops. Its rate will differ from this hop-level figure.
- `ions` must come from the pinned 0.4.1 lock, never git main (the 2026 refactor).
- BVSE paths change, so ML-NEB and later stages of the new run use new initial paths.
  Earlier ML-NEB, representation and resource evidence stays in its own run directory
  as historical evidence.
