"""Synthetic universe for in-process integration tests of the outcome-reading runners.

A real frozen-protocol layout (``freeze._protocol_content``) over synthetic MPLiTrj
frames (train/test/val sources), nebDFT2k relaxed bands whose DFT barrier is set
exactly, a G0 input manifest pinning those files, and two repeated partitions.  DFT
labels come from E = k/2 sum |r|^2; each synthetic model predicts ``scale`` times it,
so every Eq. 17 slope and every raw barrier (scale x DFT) is known in closed form.
``Synth-A3`` fails on every structure (a recorded model failure, never an
infrastructure failure), which leaves its P1 p-value undefined.

``SyntheticContext`` is the production ``RunnerContext`` with its frozen inputs
injected; its ``require_outcome`` enforces the real stage boundary
(``stage_graph.PERMITTED_OUTCOME_ACCESS`` through the contract's own
``_require_stage_outcome_scope``) and logs every access in order.  Nothing here reads
real data: every file lives under the test's temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from protocol_fixtures import canonical_units

from e3sse.production.runners.context import RunnerContext

REPOSITORY = Path(__file__).resolve().parents[1]
K_SPRING = 1.0
MODELS = {"Synth-A1": 0.8, "Synth-A2": 0.9, "Synth-A3": None}  # None: fails on every structure
# material, chemical system, published split, anion family, nebDFT2k hops (0: frames only)
MATERIALS = (
    ("mp-1", "Li-O", "train", "oxide", 3),
    ("mp-2", "Li-S", "train", "sulfide", 3),
    ("mp-3", "Li-P-S", "train", "sulfide", 3),
    ("mp-4", "Cl-Li", "train", "halide", 3),
    ("mp-5", "Br-Li", "train", "halide", 0),
    ("mp-6", "Li-O-Si", "train", "oxide", 0),
    ("mp-7", "Li-O-P", "test", "oxide", 2),
    ("mp-8", "Li-Se", "test", "other", 2),
    ("mp-9", "Li-N", "val", "nitride", 1),
)
COMPOSITION = {"Li": 3, "X": 3}
ENDPOINT_FRAMES_PER_KIND = 6
PATH_FRAMES = 2
REPEATS = (
    {"Li-O": "test", "Li-S": "training_or_calibration", "Li-P-S": "training_or_calibration",
     "Cl-Li": "training_or_calibration", "Br-Li": "training_or_calibration",
     "Li-O-Si": "training_or_calibration", "Li-O-P": "training_or_calibration",
     "Li-Se": "test", "Li-N": "training_or_calibration"},
    {"Li-O": "training_or_calibration", "Li-S": "test", "Li-P-S": "training_or_calibration",
     "Cl-Li": "training_or_calibration", "Br-Li": "training_or_calibration",
     "Li-O-Si": "test", "Li-O-P": "test", "Li-Se": "training_or_calibration",
     "Li-N": "training_or_calibration"},
)


def hop_id(material: str, h: int) -> str:
    return f"{material}_0_{h}_0_0_0"


def dft_barrier_of(hop: str) -> float:
    """Exact synthetic DFT barrier (eV): training hops are all above c = 0.5 eV."""

    digest = int(hashlib.sha256(hop.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    material = hop.split("_")[0]
    split = next(s for m, _, s, _, _ in MATERIALS if m == material)
    return 0.6 + 0.6 * digest if split == "train" else 0.2 + 0.8 * digest


# ----------------------------------------------------------------------- models

class HarmonicCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def __init__(self, scale: float | None) -> None:
        super().__init__()
        self.scale = scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.scale is None:
            raise ValueError("synthetic model failure on this structure")
        r = self.atoms.positions
        self.results = {"energy": self.scale * 0.5 * K_SPRING * float((r * r).sum()),
                        "forces": -self.scale * K_SPRING * r}


def synthetic_inference_setup(model, checkpoint, source_path, device):
    """study1_inference.inference_worker_setup without a checkpoint (top level: spawn-safe)."""

    calc = None if model is None else HarmonicCalculator(MODELS[model["canonical_name"]])
    return {"calc": calc, "handle": open(source_path, "rb")}


def synthetic_single_point_setup(model, checkpoint):
    return {"calc": HarmonicCalculator(MODELS[model["canonical_name"]])}


def model_entries() -> dict[str, dict[str, Any]]:
    return {name: {"canonical_name": name, "frozen_group": "A", "adapter_name": "synthetic",
                   "weights_sha256": hashlib.sha256(name.encode()).hexdigest(),
                   "package_name": "synthetic", "package_version": "0", "adapter_version": "0"}
            for name in MODELS}


# ----------------------------------------------------------------------- files

def _extxyz(frames) -> str:
    lines = []
    for material, positions, symbols in frames:
        forces = -K_SPRING * positions
        energy = 0.5 * K_SPRING * float((positions * positions).sum()) - 3.0 * len(symbols)
        lines.append(str(len(symbols)))
        lines.append('Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" '
                     "Properties=species:S:1:pos:R:3:forces:R:3 "
                     f'energy={energy!r} material_id={material} pbc="T T T"')
        for symbol, r, f in zip(symbols, positions, forces, strict=True):
            lines.append(" ".join([symbol, *(repr(float(v)) for v in (*r, *f))]))
    return "\n".join(lines) + "\n"


def _symbols(system: str) -> list[str]:
    other = next(e for e in system.split("-") if e != "Li")
    return ["Li"] * COMPOSITION["Li"] + [other] * COMPOSITION["X"]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relaxed_band(hop: str, symbols: list[str], rng) -> str:
    """Five images whose harmonic DFT energies give exactly ``dft_barrier_of(hop)``."""

    r0 = rng.uniform(0.5, 2.5, size=(len(symbols), 3))
    e0 = 0.5 * K_SPRING * float((r0 * r0).sum())
    target = dft_barrier_of(hop)
    profile = (0.0, 0.5, 1.0, 0.5, 0.1)
    frames = [(hop.split("_")[0], r0 * math.sqrt(1.0 + target / e0 * p), symbols) for p in profile]
    return _extxyz(frames)


def build(tmp_path: Path) -> dict[str, Any]:
    from e3sse.protocol import freeze

    rng = np.random.default_rng(11)
    data_root = tmp_path / "data"
    sources = {split: data_root / "raw" / "MPLiTrj" / f"MPLiTrj_{split}.xyz"
               for split in ("train", "test", "val")}
    sources["train"].parent.mkdir(parents=True)
    frames_by_split: dict[str, list] = {split: [] for split in sources}
    rows_by_split: dict[str, list] = {split: [] for split in sources}
    hops = []
    for material, system, split, family, n_hops in MATERIALS:
        symbols = _symbols(system)
        frame_hops = [hop_id(material, h) for h in range(1, n_hops + 1)] or [hop_id(material, 9)]
        for hop in frame_hops:
            kinds = (["source"] * ENDPOINT_FRAMES_PER_KIND + ["target"] * ENDPOINT_FRAMES_PER_KIND
                     + ["band_optim_step"] * PATH_FRAMES)
            for kind in kinds:
                frames_by_split[split].append((material, rng.uniform(0.3, 3.5, size=(6, 3)), symbols))
                rows_by_split[split].append((material, system, hop, kind))
        hops.extend((material, system, split, hop) for hop in
                    [hop_id(material, h) for h in range(1, n_hops + 1)])
    membership, flattened = [], []
    for split, path in sources.items():
        path.write_text(_extxyz(frames_by_split[split]), encoding="utf-8")
        digest = _sha(path)
        flattened.append({"identifier": f"raw/MPLiTrj/{path.name}", "sha256": digest,
                          "frames": len(frames_by_split[split]), "published_split": split})
        for index, (material, system, hop, kind) in enumerate(rows_by_split[split]):
            membership.append({
                "record_kind": "mplitrj_frame", "frame_id": f"{path.name}:{index}",
                "provenance_index_id": "index-synthetic", "source_file_sha256": digest,
                "source_frame_index": index, "input_locator_sha256": "0" * 64,
                "material_id": material, "hop_id": hop, "chemical_system_id": system,
                "published_split": split, "mapping_status": "mapped_unique",
                "raw_file_kind": kind, "origin_evidence_status": "unequivocal",
                "origin_evidence": f"synthetic raw_archive_path={hop}/{kind}",
                "candidate_count": 1, "raw_archive_path": f"{hop}/{kind}"})
    neb_dir = data_root / "raw" / "nebDFT2k"
    neb_dir.mkdir(parents=True)
    g0_items = []
    for material, system, split, hop in hops:
        membership.append({"record_kind": "nebdft2k_hop", "frame_id": None, "hop_id": hop,
                           "material_id": material, "chemical_system_id": system,
                           "published_split": split, "eligibility_status": "included",
                           "exclusion_reason": None, "source_identifier": "index.csv",
                           "source_sha256": "1" * 64})
        path = neb_dir / f"{hop}_relaxed.xyz"
        path.write_text(_relaxed_band(hop, _symbols(system), rng), encoding="utf-8")
        stat = path.stat()
        g0_items.append({"path": str(path), "sha256": _sha(path), "bytes": stat.st_size,
                         "mtime_ns": stat.st_mtime_ns})
    g0_path = tmp_path / "g0_inputs.json"
    g0_path.write_text(json.dumps({"inputs": {"nebDFT2k": g0_items}}), encoding="utf-8")
    membership_sha = hashlib.sha256(
        json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    systems = sorted({m[1] for m in MATERIALS})
    repeated = []
    for index, assignment in enumerate(REPEATS):
        assert sorted(assignment) == systems
        repeated.append({"repeat_index": index, "assignment_by_chemical_system": assignment,
                         "training_or_calibration_chemical_systems": sorted(
                             s for s, role in assignment.items() if role == "training_or_calibration"),
                         "test_chemical_systems": sorted(
                             s for s, role in assignment.items() if role == "test"),
                         "overlap_count": 0})
    partition_manifest = {
        "schema_version": "e3sse.partition.manifest.v1",
        "partition_universe": {"membership_sha256": membership_sha},
        "repeated_partitions": {"partitions": repeated},
        "primary_population": {"status": "resolved",
                               "assignment_by_chemical_system": {s: "primary" for s in systems}},
        "anion_families": {"status": "resolved",
                           "assignment_by_material": {m: f for m, _, _, f, _ in MATERIALS}},
    }
    dataset_manifest = {"datasets": [{"dataset": "MPLiTrj", "provenance": {
        "index_id": "index-synthetic", "flattened_sources": flattened}}],
        "g0_status": "partial", "static_branch_status": "ready", "transport_branch_status": "partial"}
    config = json.loads((REPOSITORY / "configs" / "protocol_freeze.json").read_text())
    protocol = freeze._protocol_content(
        git_sha="a" * 40, source_documents={"research_design": "b" * 64},
        dataset_manifest=dataset_manifest, partition_manifest=partition_manifest,
        exclusions={}, units=canonical_units(), config=config)
    protocol["study_1_contract"].update(bootstrap_resamples=200, minimum_bootstrap_resamples=200)
    protocol["study_1_contract"]["label_budget_stability"]["repeat_count"] = 20
    protocol["study_4_contract"]["draws_per_partition"] = 4
    # A tiny universe has at most two materials per family; the frozen pooled minimum
    # (three) would leave every family scale unsupported.  Test-only relaxation.
    protocol["study_1_contract"]["minimum_support"]["resolution"]["parameters"][
        "pooled_min_materials"] = 1
    protocol.update(materialization_mode="production", production_eligible=True,
                    freeze_blockers=[], freeze_status="frozen")
    protocol["protocol_id"] = freeze.canonical_sha256(protocol)
    active = json.loads((REPOSITORY / "configs" /
                         "reliability_gated_internal_freeze_protocol.json").read_text())
    return {"tmp": tmp_path, "data_root": data_root, "protocol": protocol,
            "membership": membership, "partition_manifest": partition_manifest,
            "active": active, "models": model_entries(),
            "freeze_config": {"inputs": {"g0_input_manifest": {"path": str(g0_path),
                                                               "sha256": _sha(g0_path)}}},
            "runner_config": {"runners": {"transfer": {"test_source_file": "MPLiTrj_test.xyz",
                                                       "validation_source_file": "MPLiTrj_val.xyz",
                                                       "frames_per_shard": 16}}},
            "hops": hops}


# ----------------------------------------------------------------------- context

class SyntheticContext(RunnerContext):
    """The production RunnerContext with frozen inputs injected (no data root read)."""

    universe: ClassVar[dict[str, Any]] = {}

    @property
    def config(self):
        return self.universe["runner_config"]

    @property
    def active_protocol(self):
        return self.universe["active"]

    @property
    def freeze_config(self):
        return self.universe["freeze_config"]

    @property
    def study1_config(self):
        return {"inference": {"source_file": "MPLiTrj_train.xyz"}}

    @property
    def parent_protocol(self):
        return self.universe["protocol"]

    @property
    def membership(self):
        return self.universe["membership"]

    @property
    def partition_manifest(self):
        return self.universe["partition_manifest"]

    @property
    def models(self):
        return self.universe["models"]

    @property
    def data_root(self):
        return self.universe["data_root"]

    def checkpoint(self, model):
        return self.universe["tmp"] / "checkpoints" / model

    def internal_freeze(self):
        return {"protocol_id": "i" * 64, "git_sha": "a" * 40,
                "amendments": self.universe["active"]["amendments"]}

    def execute(self, *, configure_torch=False, **kwargs):
        # Synthetic models are not torch modules and the project venv has no torch;
        # the frozen runtimes pin torch threads (configure_torch=True) in production.
        return super().execute(configure_torch=False, **kwargs)

    def require_outcome(self, *, role, outcome, outcome_kind, unit_ids=None,
                        purpose="prespecified_analysis", development_manifest=None):
        """The real stage boundary (orchestrator allowlist + contract scope check) and the
        contract's own training-unit and tuning-purpose checks; the protocol seals
        (freeze, pilot amendment, seal 2) are covered by the contract's own tests."""

        from e3sse.production.stage_graph import EVALUATION_BARRIERS, PERMITTED_OUTCOME_ACCESS
        from e3sse.protocol.contracts import (
            _TUNING_PURPOSES,
            PILOT_TUNING_OUTCOME_KINDS,
            STAGE_OUTCOME_ACCESS_ENV,
            OutcomeAccessDenied,
            _require_stage_outcome_scope,
            _verify_training_units,
        )

        previous = os.environ.get(STAGE_OUTCOME_ACCESS_ENV)
        os.environ[STAGE_OUTCOME_ACCESS_ENV] = json.dumps(
            sorted(PERMITTED_OUTCOME_ACCESS[self.stage]))
        try:
            _require_stage_outcome_scope(outcome_kind)
        finally:
            if previous is None:
                os.environ.pop(STAGE_OUTCOME_ACCESS_ENV, None)
            else:
                os.environ[STAGE_OUTCOME_ACCESS_ENV] = previous
        if purpose in _TUNING_PURPOSES and (role != "train"
                                            or outcome_kind not in PILOT_TUNING_OUTCOME_KINDS):
            raise OutcomeAccessDenied(f"{purpose} is training-only over non-path force kinds")
        if role == "train":  # unit IDs must be published-train units of the frozen membership
            try:
                kinds = _verify_training_units(self.parent_protocol, self.membership, unit_ids)
            except ValueError as exc:
                raise OutcomeAccessDenied(str(exc)) from exc
            if purpose in _TUNING_PURPOSES and kinds != {"mplitrj_nonpath_frame"}:
                raise OutcomeAccessDenied("tuning units must be published-train non-path frames")
        entry = {"stage": self.stage, "model": self.model, "role": role,
                 "outcome_kind": outcome_kind, "purpose": purpose,
                 "units": None if unit_ids is None else len(unit_ids)}
        if self.stage == "transfer" and outcome_kind == EVALUATION_BARRIERS:
            entry["hop_sets_published_before"] = (self.stage_dir / "transfer_hop_sets.json").is_file()
        log = self.run_dir / "synthetic_outcome_access.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def context(universe, run_dir: Path, stage: str, model: str | None = None,
            workers: int = 1) -> SyntheticContext:
    SyntheticContext.universe = universe
    return SyntheticContext(repository=REPOSITORY, config_path=universe["tmp"] / "config.json",
                            run_dir=run_dir, stage=stage, model=model, workers=workers)


def access_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "synthetic_outcome_access.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ----------------------------------------------------------------------- upstream stages

def publish_upstream(run_dir: Path, stage: str, name: str, value: Any, *,
                     model: str | None = None, completion: bool = False) -> Path:
    """Write a JSON output and register it in the stage's manifest (verified upstream)."""

    directory = run_dir / stage / ("models/" + model if model else "")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name.replace(':', '_')}.json"
    path.write_text(json.dumps(value, sort_keys=True))
    register(run_dir, stage, name, path, model=model, completion=completion)
    return path


def register(run_dir: Path, stage: str, name: str, path: Path, *, model: str | None = None,
             completion: bool = False) -> None:
    directory = run_dir / stage / ("models/" + model if model else "")
    manifest = directory / ("stage_completion.json" if completion else
                            "model_manifest.json" if model else "stage_manifest.json")
    record = json.loads(manifest.read_text()) if manifest.exists() else {"outputs": {}}
    record["outputs"][name] = {"path": str(path), "sha256": _sha(path)}
    manifest.write_text(json.dumps(record, sort_keys=True))


def build_estimate_and_pilot(universe, run_dir: Path) -> None:
    """The Study-1 stage outputs the transfer and Study-4 runners consume (synthetic
    frozen inputs): train-source frame statistics per run (real inference shards over
    the synthetic models), pooled-scale and s-bar tables, and the pilot amendment."""

    import csv

    from e3sse.production.runners import transfer
    from e3sse.softening import study1_inference as inference
    from e3sse.softening.study1_inputs import mplitrj_source, published_train_frames

    ctx = context(universe, run_dir, "estimate")
    source = mplitrj_source(universe["protocol"], "MPLiTrj_train.xyz")
    path = (universe["data_root"] / source["identifier"]).resolve()
    frames = published_train_frames(universe["membership"], source)
    _, index = inference.build_frame_index(path, source, frames, ctx.stage_dir / "frame_index",
                                           resume=True)
    shards = inference.plan_frame_shards(frames, index, 16)
    for name in ("dft_reference", *MODELS):
        entry = None if name == "dft_reference" else universe["models"][name]
        results = ctx.execute(namespace=f"estimate.{name.lower().replace('-', '_')}",
                              units=[(s.identity, s.payload) for s in shards],
                              task=inference.infer_shard,
                              identity={"run": name, "code_sha256": ctx.code_sha256(
                                  ["src/e3sse/softening/study1_inference.py"])},
                              setup=synthetic_inference_setup,
                              setup_args=(entry, None, str(path), "cpu"))
        manifest = transfer._merge_statistics(ctx, f"inference/{name}", results,
                                              reference=name == "dft_reference")
        register(run_dir, "estimate", f"frame_statistics:{name}", manifest, completion=True)
    analysis = ctx.stage_dir / "analysis"
    (analysis / "derived").mkdir(parents=True, exist_ok=True)
    families = sorted({f for _, _, _, f, _ in MATERIALS})
    rows = []
    for model, scale in MODELS.items():
        for scope, target in [("global", ""), *[("family", f) for f in families]]:
            rows.append({"model": model, "kind": "pooled", "population": "primary",
                         "estimator_kind": "primary", "component_set": "all", "decile": "",
                         "scope": scope, "scope_target": target,
                         "status": "estimated" if scale else "unsupported",
                         "s_hat": scale or "", "residual_fraction": 0.0 if scale else "",
                         "ci_low": scale - 0.01 if scale else "",
                         "ci_high": scale + 0.01 if scale else ""})
    for name, table in (("study1_softening_estimates.csv", rows),
                        ("study1_model_summary.csv",
                         [{"model": m, "population": "primary", "s_bar": s or 0.85}
                          for m, s in MODELS.items()])):
        with (analysis / "derived" / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(table[0]))
            writer.writeheader()
            writer.writerows(table)
    manifest = analysis / "study1_analysis_manifest.json"
    manifest.write_text(json.dumps({"synthetic": True}))
    register(run_dir, "estimate", "study1_analysis_manifest", manifest, completion=True)
    publish_upstream(run_dir, "pilot", "pilot_amendment", {"frozen_thresholds": {
        "study2_p1_eligible_s_bar": {m: s or 0.85 for m, s in MODELS.items()}}})
    publish_upstream(run_dir, "pilot", "internal_pilot_record", {"synthetic": True})


def run_phase(universe, run_dir: Path, stage: str, phase: str, model: str | None = None,
              workers: int = 1) -> int:
    """scripts/run_production_stage.py in process, on the synthetic context."""

    import importlib.util

    from e3sse.production.runners import context as context_module

    spec = importlib.util.spec_from_file_location(
        "run_production_stage", REPOSITORY / "scripts" / "run_production_stage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SyntheticContext.universe = universe
    original = context_module.RunnerContext
    context_module.RunnerContext = SyntheticContext
    try:
        argv = ["--config", str(universe["tmp"] / "config.json"), "--run-dir", str(run_dir),
                "--stage", stage, "--phase", phase, "--workers", str(workers), "--threads", "1"]
        return module.main(argv + (["--model", model] if model else []))
    finally:
        context_module.RunnerContext = original


def sha_tree(directory: Path, names) -> dict[str, str]:
    return {name: _sha(directory / name) for name in names}


def canonical(value: Any) -> str:
    from e3sse.production.runners.context import canonical_sha256

    return canonical_sha256(value)


# ----------------------------------------------------------------------- ML-NEB

class NonFiniteCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": float("nan"), "forces": np.full((len(self.atoms), 3), np.nan)}


class AllocatorFailureCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        raise RuntimeError("DefaultCPUAllocator: can't allocate memory")


def synthetic_ml_neb_setup(kind: str) -> dict[str, Any]:
    """ml_neb_units.ml_neb_worker_setup with a synthetic calculator (spawn-safe)."""

    from ase.calculators.lj import LennardJones

    from e3sse.barriers.ml_neb import settings_from_protocol

    active = json.loads((REPOSITORY / "configs" /
                         "reliability_gated_internal_freeze_protocol.json").read_text())
    factories = {"lj": lambda: LennardJones(sigma=2.75, epsilon=0.05, rc=6.0, smooth=True),
                 "non_finite": NonFiniteCalculator, "allocator_failure": AllocatorFailureCalculator}
    return {"load": factories[kind], "pool": [], "settings": settings_from_protocol(active),
            "extractor": None,
            "provenance": {"model": "Synth-A1", "weights_sha256": "x", "environment_lock_sha256": "y",
                           "dtype": "float64", "device": "cpu", "torch_intra_op_threads": 1,
                           "torch_inter_op_threads": 1}}


def li_vacancy_band() -> list[dict[str, Any]]:
    """bcc-Li vacancy hop, linear 5-image band (endpoints + 3 interior images)."""

    from ase.build import bulk

    host = bulk("Li", "bcc", a=3.51, cubic=True).repeat(3)
    vectors = host.get_distances(0, range(len(host)), mic=True, vector=True)
    lengths = np.linalg.norm(vectors, axis=1)
    lengths[0] = np.inf
    j = int(np.argmin(lengths))
    initial, final = host.copy(), host.copy()
    final.positions[j] = host.positions[j] - vectors[j]
    del initial[0]
    del final[0]
    images = [initial.copy() for _ in range(5)]
    for k, image in enumerate(images):
        image.positions = initial.positions + (final.positions - initial.positions) * k / 4
    return [{"numbers": a.numbers.tolist(), "positions": a.positions.tolist(),
             "cell": a.cell.array.tolist(), "pbc": a.pbc.tolist()} for a in images]
