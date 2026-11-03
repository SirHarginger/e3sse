"""Production stage runners: thin drivers over the tested E3-SSE libraries.

Each runner module exposes ``run(ctx)`` (and, for per-model stages,
``run_model(ctx)`` executed in the model's frozen runtime and ``reduce(ctx)`` executed
in the project environment).  Runners compute through existing library functions
(``e3sse.barriers``, ``e3sse.softening``, ``e3sse.selection``, ``e3sse.screening``,
``e3sse.representation``), read every outcome through
``RunnerContext.require_outcome`` (the frozen contract plus the stage scope), publish
create-once JSON artifacts, and resume through ``e3sse.execution`` shards.
"""

RUNNER_MODULES = {
    "gate1_population_audit": None,  # script-driven (audit_gate1_structural_populations.py)
    "descriptor_availability": "e3sse.production.runners.descriptors",
    "development_population": "e3sse.production.runners.population",
    "bvse_reproduction": "e3sse.production.runners.bvse_paths",
    "development_ml_neb": "e3sse.production.runners.ml_neb_units",
    "path_representation": "e3sse.production.runners.representation",
    "pilot": "e3sse.production.runners.pilot",
    "family_scale": "e3sse.production.runners.family_scale",
    "transfer": "e3sse.production.runners.transfer",
    "reliability": "e3sse.production.runners.reliability",
    "weight_slice_ml_neb": "e3sse.production.runners.ml_neb_units",
    "model_selection": "e3sse.production.runners.model_selection",
    "deployment_calibration": "e3sse.production.runners.ml_neb_units",
    "linear_recalibration_comparator": "e3sse.production.runners.recalibration",
    "controlled_selection": "e3sse.production.runners.study4",
    "density_ratio": "e3sse.production.runners.density",
    "discovery": "e3sse.production.runners.ml_neb_units",
    "discovery_selection": "e3sse.production.runners.discovery_selection",
    "publication": "e3sse.production.runners.publication",
}
