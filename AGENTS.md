# E3-SSE Agent Instructions

## Project

E3-SSE develops Softening-Rescaled Transport (SRT) and statistically
controlled screening methods for Li-ion solid electrolytes using universal
machine-learning interatomic potentials.

The scientific pipeline is:

data
-> softening estimation
-> migration-barrier correction
-> finite-temperature transport validation
-> conformal/FDR selection
-> prospective screening

This repository contains research software. Scientific correctness,
reproducibility, data provenance, leakage prevention, and auditability take
priority over speed.

## Repository locations

Local development:
    ~/projects/e3sse

GitHub:
    github.com/SirHarginger/e3sse

Production server:
    /srv/ben/e3sse

The server contains the full scientific datasets. Git contains code,
configuration, documentation, tests, and small fixtures only.

## Non-negotiable data rules

1. Treat `/srv/ben/e3sse/data` as research source data.
2. Treat `data/raw/` as immutable.
3. Never delete, rename, overwrite, truncate, reformat, or rewrite raw data.
4. Never use raw-data directories as output directories.
5. Write generated results under `outputs/`.
6. Write temporary material under `scratch/`.
7. Never commit large research datasets to Git.
8. Never fabricate missing scientific values.
9. Never infer an unseen dataset schema. Inspect it first.
10. Never put credentials, API keys, tokens, passwords, or private keys in
    tracked files.

## Main datasets

Expected server datasets include:

- MPLiTrj
- nebDFT2k
- BVEL13k
- nebBVSE122k
- FPMD reference archives

Before implementing a loader, inspect the actual files, metadata, identifiers,
splits, units, ASE fields, and reference settings.

## Scientific integrity

Always distinguish:

- DFT reference values
- raw uMLIP predictions
- SRT-corrected predictions
- fitted/calibrated comparators
- FPMD reference transport
- prospective predictions

Do not silently convert one into another.

Report assumptions and exclusions explicitly.

## Leakage control

The correction applied to a Li hop must not use DFT information generated from
that hop's own NEB pathway unless the analysis is explicitly labelled an
oracle analysis.

All frames and hops belonging to the same chemical system must remain in the
same partition when grouped splitting is required.

Test/reference outcomes must not be inspected to choose thresholds,
hyperparameters, correction rules, or screening criteria.

## Softening estimation

The primary softening scale is based on the zero-intercept force-parity
projection between ML and DFT forces.

Estimate uncertainty by resampling whole frames/material units where
appropriate rather than treating correlated Cartesian force components as
independent samples.

Keep global, family-level, material-level, and element-level estimates
distinguishable.

## Barrier analysis

Keep these analyses distinct:

1. fixed-geometry evaluation on DFT NEB images;
2. relaxed ML-NEB evaluation;
3. raw uMLIP barriers;
4. SRT-corrected barriers;
5. label-using recalibration comparators.

Never label a fixed-geometry single-point calculation as an ML-NEB result.

## Transport analysis

For MD/FPMD comparisons:

- retain temperature and trajectory metadata;
- verify units;
- inspect equilibration and structural stability;
- distinguish tracer from collective diffusion;
- report statistical uncertainty;
- do not extrapolate an Arrhenius model without checking its adequacy.

## Conformal/FDR analysis

Keep calibration and test units separated.

Respect the chemical-system dependence structure.

Report:

- nominal FDR level;
- realized FDP/FDR;
- power or true discoveries;
- calibration/test sample sizes;
- assumptions required for the guarantee.

Do not describe an empirical uncertainty estimate as a formal FDR guarantee.

## Engineering rules

Production code belongs under `src/e3sse/`.

Use the existing scientific modules:

- `data`
- `softening`
- `barriers`
- `transport`
- `selection`
- `screening`

Use configuration files for paths and settings. Do not scatter hard-coded
server paths throughout Python modules.

Notebooks are for exploration and figures, not the only implementation of
core scientific logic.

Prefer small pure functions, typed interfaces where useful, and explicit unit
conversions.

Every important transformation should be testable.

## Reproducibility

A production analysis should record enough information to reconstruct it:

- UTC timestamp;
- Git commit SHA;
- Git dirty/clean state;
- branch;
- configuration file;
- model/checkpoint identity;
- dataset/split identity;
- Python version;
- relevant package versions;
- random seed;
- command used.

Use the `experiment-run` and `reproducibility-audit` skills.

## Development workflow

Normal workflow:

local VS Code
-> feature branch
-> tests/lint
-> commit
-> GitHub
-> review/merge
-> server git pull
-> full-data execution

Do not develop directly against the production server unless the task
specifically requires server-only inspection or computation.

## Before changing code

For any non-trivial task:

1. inspect relevant existing code;
2. inspect configuration and tests;
3. identify scientific assumptions;
4. identify dataset fields actually available;
5. state the implementation plan;
6. make the smallest coherent change.

Do not redesign unrelated parts of the project.

## Validation

Before declaring implementation complete, normally run:

    bash .agents/scripts/validate_repo.sh

For numerical work also run task-specific sanity checks.

A passing unit test is necessary but does not establish scientific validity.

## Completion standard

A task is complete only when:

- the implementation matches the requested scientific definition;
- no raw data were mutated;
- relevant tests pass;
- units and shapes are explicit;
- leakage rules are respected;
- provenance requirements are met;
- documentation/configuration is updated when behavior changes;
- unresolved uncertainty is reported rather than hidden.

## External resources

The project is intended to use open datasets, open-source software, and
available institutional compute.

Do not introduce paid data, paid APIs, cloud compute, or licensed services
without explicit user approval.

## Agent behavior

Use repository skills when their descriptions match the task.

Delegate focused work to specialist agents where useful.

For scientific claims or rapidly changing software APIs, verify against
primary documentation rather than guessing.

If existing code and the research protocol disagree, stop and surface the
difference before silently choosing one.
