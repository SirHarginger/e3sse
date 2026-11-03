"""Synthetic Study-1 universe: a frozen protocol directory, MPLiTrj extxyz and models.

The protocol is assembled with the real ``freeze._protocol_content`` from the tracked
``configs/protocol_freeze.json`` (so Study-1 code reads the frozen field layout), then
registered with a synthetic OSF addendum.  DFT labels come from a harmonic potential
E = k/2 sum |r|^2 (F = -k r, k = 1 eV/A^2); each synthetic model predicts exactly
``scale`` times it, so every Eq. 17 slope is known in closed form.  Frames of the
contaminated hop's own NEB path carry DFT forces F = -(k/c) r instead, so their
slope is scale * c: the leakage stress test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from protocol_fixtures import canonical_units

from e3sse.production.checkpoints import SCHEMA as CHECKPOINT_INVENTORY_SCHEMA
from e3sse.protocol import freeze
from e3sse.protocol.registration import build_registration_addendum

K_SPRING = 1.0
MODEL_SCALES = {"CHGNet-0.3.0": 0.8, "M3GNet-MatPES-PBE-v2025.1-PES": 0.9, "SevenNet-omat": 0.95}
CONTAMINATED_HOP = "mp-1_0_1"
CONTAMINATION = 0.625  # own-path slope = 0.8 * 0.625 = 0.5 for CHGNet
# material -> (chemical system, composition per frame, hops, path frames per hop)
MATERIALS = {
    "mp-1": ("Li-O", {"Li": 4, "O": 2}, 3, 30),
    "mp-2": ("Li-S", {"Li": 4, "S": 2}, 2, 6),
    "mp-3": ("Li-P-S", {"Li": 3, "P": 1, "S": 4}, 2, 6),
    "mp-4": ("Cl-Li", {"Li": 3, "Cl": 3}, 2, 6),
    "mp-5": ("Fe-Li-O", {"Li": 2, "Fe": 2, "O": 4}, 2, 6),
}
VALIDATION_MATERIAL = ("mp-9", "Li-N")


class HarmonicCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.model = _FakeTorchModule()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        r = self.atoms.positions
        self.results = {"energy": self.scale * 0.5 * K_SPRING * float((r * r).sum()),
                        "forces": -self.scale * K_SPRING * r}


class _FakeParameter:
    class device:
        type = "cpu"

    dtype = "torch.float32"


class _FakeTorchModule:
    def parameters(self):
        return iter([_FakeParameter()])


def _stand_in_checkpoints(root: Path, models: list[dict]) -> dict[str, Path]:
    """Write placeholder checkpoint bytes and bind their hashes into the synthetic registry.

    The synthetic worker never loads weights, but the checkpoint inventory re-hashes
    every checkpoint against the registry, so the registry must name these bytes.
    MatGL models are directories whose weights digest is ``state.pt``.
    """

    paths = {}
    for model in models:
        name = model["canonical_name"]
        if model.get("artifact_files"):
            path = root / name
            path.mkdir(parents=True)
            for file in model["artifact_files"]:
                (path / file).write_bytes(f"{name}:{file}".encode())
            model["artifact_files"] = {file: _sha(path / file) for file in model["artifact_files"]}
            model["weights_sha256"] = _sha(path / "state.pt")
        else:
            path = root / Path(model["checkpoint_id"]).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            model["weights_sha256"] = _sha(path)
        paths[name] = path
    return paths


def synthetic_worker_setup(model, checkpoint, source_path, device):
    calc = None if model is None else HarmonicCalculator(MODEL_SCALES[model["canonical_name"]])
    return {"calc": calc, "handle": open(source_path, "rb")}


def _frames():
    """(material, hop, raw_file_kind, contaminated, positions, symbols) in source order."""

    rng = np.random.default_rng(7)
    frames = []
    for material, (_, composition, hops, path) in MATERIALS.items():
        symbols = [symbol for symbol, count in composition.items() for _ in range(count)]
        for h in range(1, hops + 1):
            hop = f"{material}_0_{h}"
            kinds = ["source", "target"] + ["band_optim_step"] * path
            for kind in kinds:
                positions = rng.uniform(0.3, 3.5, size=(len(symbols), 3))
                contaminated = hop == CONTAMINATED_HOP and kind == "band_optim_step"
                frames.append((material, hop, kind, contaminated, positions, symbols))
    return frames


def _extxyz(frames) -> str:
    lines = []
    for material, _, _, contaminated, positions, symbols in frames:
        forces = -K_SPRING * positions / (CONTAMINATION if contaminated else 1.0)
        energy = 0.5 * K_SPRING * float((positions * positions).sum()) - 3.0 * len(symbols)
        lines.append(str(len(symbols)))
        lines.append(
            'Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" '
            "Properties=species:S:1:pos:R:3:forces:R:3 "
            f'energy={energy!r} material_id={material} pbc="T T T"'
        )
        for symbol, r, f in zip(symbols, positions, forces, strict=True):
            lines.append(" ".join([symbol, *(repr(float(v)) for v in (*r, *f))]))
    return "\n".join(lines) + "\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_universe(tmp_path: Path, *, register: bool = True) -> dict:
    """Write data, frozen protocol directory, configs; return their paths."""

    data_root = tmp_path / "data"
    source = data_root / "raw" / "MPLiTrj" / "MPLiTrj_train.xyz"
    source.parent.mkdir(parents=True)
    frames = _frames()
    source.write_text(_extxyz(frames), encoding="utf-8")
    source_sha = _sha(source)
    membership = []
    for index, (material, hop, kind, _, _, _) in enumerate(frames):
        membership.append({
            "record_kind": "mplitrj_frame",
            "frame_id": f"MPLiTrj_train.xyz:{index}",
            "provenance_index_id": "index-synthetic",
            "source_file_sha256": source_sha,
            "source_frame_index": index,
            "input_locator_sha256": "0" * 64,
            "material_id": material,
            "hop_id": hop,
            "chemical_system_id": MATERIALS[material][0],
            "published_split": "train",
            "mapping_status": "mapped_unique",
            "raw_file_kind": kind,
            "origin_evidence_status": "unequivocal",
            "origin_evidence": f"synthetic raw_archive_path={hop}/{kind}",
            "candidate_count": 1,
            "raw_archive_path": f"{hop}/{kind}",
        })
    material, system = VALIDATION_MATERIAL
    membership.append({**membership[0], "frame_id": "MPLiTrj_val.xyz:0", "source_frame_index": 0,
                       "material_id": material, "hop_id": f"{material}_0_1",
                       "chemical_system_id": system, "published_split": "val"})
    for material, (system, _, hops, _) in MATERIALS.items():
        for h in range(1, hops + 1):
            membership.append({
                "record_kind": "nebdft2k_hop", "frame_id": None, "hop_id": f"{material}_0_{h}",
                "material_id": material, "chemical_system_id": system,
                "published_split": "train", "eligibility_status": "included",
                "exclusion_reason": None, "source_identifier": "index.csv",
                "source_sha256": "1" * 64,
            })
    membership_sha = hashlib.sha256(
        json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    groups = config["protocol"]["model_groups"]
    groups["A"] = [m for m in groups["A"] if m["canonical_name"] == "CHGNet-0.3.0"]
    checkpoint_paths = _stand_in_checkpoints(tmp_path / "checkpoints",
                                             [m for g in ("A", "B1", "B2") for m in groups[g]])
    families = {"mp-1": "oxide", "mp-2": "sulfide", "mp-3": "sulfide", "mp-4": "halide",
                "mp-5": "oxide", "mp-9": "nitride"}
    population = {system: "tm_hubbard_u_secondary" if "Fe" in system else "primary"
                  for system, *_ in [*MATERIALS.values(), (VALIDATION_MATERIAL[1],)]}
    partition_manifest = {
        "schema_version": "e3sse.partition.manifest.v1",
        "partition_universe": {"membership_sha256": membership_sha},
        "repeated_partitions": {"partitions": []},
        "primary_population": {"status": "resolved", "assignment_by_chemical_system": population},
        "anion_families": {"status": "resolved", "assignment_by_material": families},
    }
    dataset_manifest = {
        "datasets": [{"dataset": "MPLiTrj", "provenance": {
            "index_id": "index-synthetic",
            "flattened_sources": [{"identifier": "raw/MPLiTrj/MPLiTrj_train.xyz",
                                   "sha256": source_sha, "frames": len(frames),
                                   "published_split": "train"}],
        }}],
        "g0_status": "partial", "static_branch_status": "ready", "transport_branch_status": "partial",
    }
    protocol = freeze._protocol_content(
        git_sha="a" * 40, source_documents={"research_design": "b" * 64},
        dataset_manifest=dataset_manifest, partition_manifest=partition_manifest,
        exclusions={}, units=canonical_units(), config=config,
    )
    # Small resample counts keep tests fast; production reads the frozen values.
    protocol["study_1_contract"].update(bootstrap_resamples=200, minimum_bootstrap_resamples=200)
    protocol["study_1_contract"]["label_budget_stability"]["repeat_count"] = 40
    environments = config["inputs"]["production_evidence"]["runtime_environments"]
    protocol["production_evidence"] = {"runtime_environments": {
        name: {**{k: v for k, v in env.items() if k != "environment_lock"},
               "environment_lock": {"sha256": env["environment_lock"]["sha256"]}}
        for name, env in environments.items()
    }}
    protocol.update(materialization_mode="production", production_eligible=True,
                    freeze_blockers=[], freeze_status="frozen")
    protocol["protocol_id"] = freeze.canonical_sha256(protocol)

    protocol_dir = tmp_path / "outputs" / "protocol_freeze" / protocol["protocol_id"]
    protocol_dir.mkdir(parents=True)
    payload = json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (protocol_dir / "protocol.json").write_text(payload, encoding="utf-8")
    (protocol_dir / "partition_membership.json").write_text(
        json.dumps({"records": membership}), encoding="utf-8")
    (protocol_dir / "partition_manifest.json").write_text(
        json.dumps(partition_manifest), encoding="utf-8")
    if register:
        addendum = build_registration_addendum(
            protocol, status="registered", osf_registration_id="synth1",
            submitted_at_utc="2026-10-20T00:00:00Z", visibility="embargoed")
        (protocol_dir / "registration_registered.json").write_text(json.dumps(addendum))
    server = tmp_path / "server.json"
    server.write_text(json.dumps({"paths": {"data_root": str(data_root),
                                            "outputs_root": str(tmp_path / "outputs")}}))
    checkpoints = tmp_path / "checkpoint_inventory.json"
    checkpoints.write_text(json.dumps({
        "schema_version": CHECKPOINT_INVENTORY_SCHEMA,
        "models": {m["canonical_name"]: {
            "path": str(checkpoint_paths[m["canonical_name"]]),
            "checkpoint_id": m["checkpoint_id"], "weights_sha256": m["weights_sha256"],
            "artifact_files": m.get("artifact_files", {}),
        } for g in ("A", "B1", "B2") for m in groups[g]},
    }))
    study1 = json.loads(Path("configs/study1_softening_atlas.json").read_text(encoding="utf-8"))
    study1["frozen_protocol"].update(
        protocol_id=protocol["protocol_id"], freeze_git_sha="a" * 40,
        protocol_json_sha256=_sha(protocol_dir / "protocol.json"), directory=str(protocol_dir))
    study1.update(server_config=str(server), output_root=str(tmp_path / "outputs" / "study1"),
                  checkpoint_map=str(checkpoints))
    study1["inference"]["frames_per_shard"] = 16
    study1_path = tmp_path / "study1.json"
    study1_path.write_text(json.dumps(study1, indent=2))
    return {"config": study1_path, "protocol": protocol, "protocol_dir": protocol_dir,
            "data_root": data_root, "source": source, "frames": frames, "membership": membership,
            "output": tmp_path / "outputs" / "study1" / protocol["protocol_id"][:16]}
