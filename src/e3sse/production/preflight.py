"""Blocking preflight evidence for the production run; reads no scientific outcome.

Each check reads only identities, hashes, manifests and structural audit summaries.
Each returns ``passed``, ``failed`` (the evidence contradicts a requirement) or
``missing_evidence`` (the evidence does not exist yet).  Each says what it blocks:
``freeze`` (the internal scientific freeze may not be materialized) or
``production_compute`` (scientific computation may not start, e.g. without a
measured resource profile).  These are distinct from a scientific failure (a
recorded unit status) and from an infrastructure failure (a crashed command).

The protocol's own blocking open items are reported one by one.  Preflight never
resolves them: resolving an item is a reviewed protocol edit and commit, and the
freeze then binds it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from e3sse.production.run_identity import file_sha256

PREFLIGHT_SCHEMA = "e3sse.production.preflight-report"
STATUSES = ("passed", "failed", "missing_evidence")
BLOCKS = ("freeze", "production_compute", "nothing")
GATE1_DIMENSION = "3"


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    status: str
    blocks: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES or self.blocks not in BLOCKS:
            raise ValueError(f"invalid preflight check {self.check_id}")

    @property
    def blocking(self) -> bool:
        return self.status != "passed" and self.blocks != "nothing"


def _passed(check_id: str, blocks: str, detail: str, **evidence: Any) -> PreflightCheck:
    return PreflightCheck(check_id, "passed", blocks, detail, evidence)


def _failed(check_id: str, blocks: str, detail: str, **evidence: Any) -> PreflightCheck:
    return PreflightCheck(check_id, "failed", blocks, detail, evidence)


def _missing(check_id: str, blocks: str, detail: str, **evidence: Any) -> PreflightCheck:
    return PreflightCheck(check_id, "missing_evidence", blocks, detail, evidence)


def _pinned_file(check_id: str, path: Path, expected: str, blocks: str) -> PreflightCheck:
    if not path.is_file():
        return _missing(check_id, blocks, f"pinned file {path} is absent", path=str(path))
    actual = file_sha256(path)
    if actual != expected:
        return _failed(check_id, blocks, "pinned SHA-256 differs", path=str(path),
                       expected_sha256=expected, actual_sha256=actual)
    return _passed(check_id, blocks, "pinned SHA-256 verified", path=str(path), sha256=actual)


# ------------------------------------------------------------------- identity


def check_git_identity(git: Mapping[str, Any], *, require_clean: bool,
                       expected_commit: str | None) -> PreflightCheck:
    evidence = {"commit": git["commit"], "tree": git["tree"], "branch": git["branch"],
                "dirty": git["dirty"], "dirty_paths": git.get("dirty_paths", [])}
    if expected_commit and git["commit"] != expected_commit:
        return _failed("git_identity", "freeze", f"HEAD {git['commit']} is not the expected "
                       f"commit {expected_commit}", **evidence)
    if require_clean and git["dirty"]:
        return _failed("git_identity", "freeze", "worktree is dirty; production requires the "
                       "reviewed clean commit", **evidence)
    return _passed("git_identity", "freeze", "Git identity recorded", **evidence)


def check_output_root(output_root: Path, protected_roots: Sequence[Path]) -> PreflightCheck:
    from e3sse.execution.sharded_execution import ensure_safe_output_dir

    check_id = f"output_root_safety:{output_root}"
    from e3sse.softening.study1_inputs import Study1InputError, refuse_tmpfs

    try:
        resolved = ensure_safe_output_dir(output_root, protected_roots)
        refuse_tmpfs(resolved)
    except (PermissionError, ValueError, Study1InputError) as exc:
        return _failed(check_id, "production_compute", str(exc),
                       output_root=str(output_root))
    anchor = next((p for p in [resolved, *resolved.parents] if p.exists()), None)
    import os

    if anchor is None or not os.access(anchor, os.W_OK):
        return _failed(check_id, "production_compute",
                       f"no writable existing ancestor for {resolved}")
    return _passed(check_id, "production_compute",
                   "outside every protected data root and not on tmpfs/ramfs",
                   output_root=str(resolved), protected_roots=[str(p) for p in protected_roots])


# ------------------------------------------------------------------- pinned inputs


def _pins(value: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield prefix, dict(value)
            return
        for key, item in value.items():
            yield from _pins(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _pins(item, f"{prefix}[{index}]")


def check_freeze_input_pins(freeze_config: Mapping[str, Any], *, config_dir: Path) -> list[PreflightCheck]:
    """Every {path, sha256} pinned by the freeze-input configuration (G0, locks, smoke...)."""

    checks = []
    for name, pin in _pins(freeze_config.get("inputs", {})):
        path = Path(pin["path"])
        path = path if path.is_absolute() else (config_dir / path).resolve()
        checks.append(_pinned_file(f"pinned_input:{name}", path, pin["sha256"], "freeze"))
    if not checks:
        return [_failed("pinned_inputs", "freeze", "the freeze-input configuration pins no file")]
    # {root, manifest_sha256} pins (MPLiTrj provenance) are re-verified by the freeze builder.
    return checks


def _manifest_identities(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every file identity in a G0 input manifest: ``inputs.<dataset>`` lists and any
    top-level list of identities (e.g. ``mplitrj_raw_archive``)."""

    found = []
    for section in (manifest.get("inputs", {}), manifest):
        for value in section.values():
            if isinstance(value, list):
                found.extend(item for item in value
                             if isinstance(item, Mapping) and item.get("path"))
    unique = {str(item["path"]): item for item in found}
    return [unique[key] for key in sorted(unique)]


def check_dataset_hashes(g0_inputs_path: Path, *, rehash: bool) -> PreflightCheck:
    """Files named by the pinned G0 input manifest: present, same size, and (when
    ``rehash``) the same SHA-256 wherever G0 recorded one."""

    if not g0_inputs_path.is_file():
        return _missing("dataset_hashes", "freeze", "G0 input manifest absent",
                        path=str(g0_inputs_path))
    manifest = json.loads(g0_inputs_path.read_text(encoding="utf-8"))
    identities = _manifest_identities(manifest)
    if not identities:
        return _failed("dataset_hashes", "freeze", "the G0 input manifest names no file",
                       path=str(g0_inputs_path))
    absent, size_changed, hash_changed, mtime_changed = [], [], [], []
    rehashed, size_only = 0, 0
    for item in identities:
        path = Path(item["path"])
        if not path.is_file():
            absent.append(str(path))
            continue
        stat = path.stat()
        if item.get("bytes") is not None and stat.st_size != item["bytes"]:
            size_changed.append(str(path))
            continue
        if item.get("mtime_ns") is not None and stat.st_mtime_ns != item["mtime_ns"]:
            mtime_changed.append(str(path))  # recorded; content is judged by size/hash
        if rehash and item.get("sha256"):
            if file_sha256(path) != item["sha256"]:
                hash_changed.append(str(path))
            else:
                rehashed += 1
        else:
            size_only += 1
    evidence = {"files": len(identities), "sha256_reverified": rehashed,
                "size_verified_without_sha256": size_only, "absent": absent,
                "size_changed": size_changed, "sha256_changed": hash_changed,
                "mtime_changed_warning": mtime_changed}
    if absent or size_changed or hash_changed:
        return _failed("dataset_hashes", "freeze", "dataset files absent or changed", **evidence)
    return _passed("dataset_hashes", "freeze",
                   f"{len(identities)} G0 inputs present with recorded sizes; "
                   f"{rehashed} SHA-256 re-verified, {size_only} checked by size only "
                   "(G0 records no SHA-256 above its hash limit, or rehash is off)", **evidence)


# ------------------------------------------------------------------- Gate 1


def check_gate1_linkage(path: Path) -> PreflightCheck:
    """Structural invariants of the read-only Gate-1 linkage audit (no outcome)."""

    if not path.is_file():
        return _missing("gate1_linkage", "freeze", "Gate-1 linkage audit absent; run "
                        "scripts/audit_rate_limiting_hop_linkage.py", path=str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    calibration, quality = report.get("calibration_side", {}), report.get("edge_quality", {})
    problems = []
    if report.get("outcome_columns_read") is not False:
        problems.append("audit read an outcome column")
    edges, linked = (calibration.get("nebDFT2k_edges"),
                     calibration.get("nebDFT2k_edges_in_nebBVSE122k"))
    if not (isinstance(edges, int) and edges > 0 and edges == linked):
        problems.append("not every nebDFT2k edge is found in nebBVSE122k (or counts absent)")
    short = {name: item.get("short_rows_skipped") for name, item in report.get("inputs", {}).items()}
    if not short or any(value != 0 for value in short.values()):
        problems.append(f"input rows were skipped or input provenance is absent: {short}")
    if quality.get("ambiguous_materials") != 0:
        problems.append("ambiguous materials present")
    if quality.get("chemsys_inconsistent_materials") != 0:
        problems.append("chemical-system-inconsistent materials present")
    if GATE1_DIMENSION not in calibration.get("by_dimension", {}):
        problems.append("frozen d = 3 analysis absent")
    evidence = {
        "path": str(path), "sha256": file_sha256(path),
        "nebDFT2k_edges": calibration.get("nebDFT2k_edges"),
        "nebDFT2k_edges_in_nebBVSE122k": calibration.get("nebDFT2k_edges_in_nebBVSE122k"),
        "ambiguous_materials": quality.get("ambiguous_materials"),
        "duplicate_conflicts": quality.get("ambiguous_reasons", {}),
        "dimension_3_calibration": calibration.get("by_dimension", {}).get(GATE1_DIMENSION),
        "dimension_3_pool": report.get("pool", {}).get("by_dimension", {}).get(GATE1_DIMENSION),
        "unparseable_pool_chemical_systems": report.get("pool", {}).get(
            "unparseable_chemical_systems"),
        "input_sha256": {name: item.get("sha256") for name, item in report.get("inputs", {}).items()},
    }
    if problems:
        return _failed("gate1_linkage", "freeze", "; ".join(problems), **evidence)
    return _passed("gate1_linkage", "freeze", "complete outcome-free linkage at d = 3", **evidence)


def check_gate1_structural_populations(path: Path, linkage_path: Path) -> PreflightCheck:
    if not path.is_file():
        return _missing("gate1_structural_populations", "freeze",
                        "structural population audit absent; run "
                        "scripts/audit_gate1_structural_populations.py", path=str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("outcome_columns_read") is not False or report.get("dimension") != 3:
        return _failed("gate1_structural_populations", "freeze",
                       "audit is not an outcome-free d = 3 audit", path=str(path))
    if linkage_path.is_file():
        linkage = json.loads(linkage_path.read_text(encoding="utf-8"))
        theirs = {k: v.get("sha256") for k, v in linkage.get("inputs", {}).items()}
        ours = {k: v.get("sha256") for k, v in report.get("inputs", {}).items()}
        if theirs != ours:
            return _failed("gate1_structural_populations", "freeze",
                           "population audit read different CSVs from the linkage audit",
                           linkage_inputs=theirs, population_inputs=ours)
    return _passed("gate1_structural_populations", "freeze", "structural counts recorded",
                   path=str(path), sha256=file_sha256(path), counts=report.get("counts"))


# ------------------------------------------------------------------- models


def check_embedding_evidence(active_protocol: Mapping[str, Any],
                             models: Mapping[str, Mapping[str, Any]]) -> list[PreflightCheck]:
    """Re-hash each pinned report; its content must agree with the pin and the registry.

    Agreement: canonical name, checkpoint SHA-256 = frozen ``weights_sha256``, capture
    mode = pinned ``capture_method``, latent usability (as ``merge_manifest`` decides)
    = pinned status, and the applicability branch consistent with that status.
    """

    from e3sse.representation.embedding_capability import merge_manifest

    evidence = active_protocol["amendments"]["representation"].get(
        "embedding_capability_evidence", {}).get("reports", {})
    checks = []
    for name in sorted(models):
        check_id = f"embedding_evidence:{name}"
        pin = evidence.get(name)
        if pin is None:
            checks.append(_missing(check_id, "freeze", "no pinned embedding report"))
            continue
        pinned = _pinned_file(check_id, Path(pin["path"]), pin["sha256"], "freeze")
        if pinned.status != "passed":
            checks.append(pinned)
            continue
        report = json.loads(Path(pin["path"]).read_text(encoding="utf-8"))
        if report.get("canonical_name") != name:
            checks.append(_failed(check_id, "freeze", "report names a different model",
                                  report_model=report.get("canonical_name")))
            continue
        try:
            usable = name in merge_manifest([report])["usable_for_latent_support"]
        except ValueError as exc:
            checks.append(_failed(check_id, "freeze", f"invalid capability report: {exc}"))
            continue
        validated = pin["latent_support_status"] == "validated"
        problems = []
        weights = models[name].get("weights_sha256")
        if report.get("checkpoint_sha256") not in (None, weights):
            problems.append("report checkpoint SHA-256 differs from the frozen weights")
        if report.get("capture_mode") not in (None, pin.get("capture_method")):
            problems.append(f"capture mode {report.get('capture_mode')!r} differs from the pin")
        branch_latent = str(pin.get("applicability_branch", "")).startswith("latent_support")
        if branch_latent != validated:
            problems.append("applicability branch is inconsistent with the latent status")
        if problems:
            checks.append(_failed(check_id, "freeze", "; ".join(problems)))
            continue
        if usable != validated:
            checks.append(_failed(check_id, "freeze", "pinned latent-support status disagrees "
                                  "with the report's checks", pin_status=pin[
                                      "latent_support_status"], report_usable=usable))
            continue
        checks.append(_passed(check_id, "freeze", "pinned report verified",
                              sha256=pin["sha256"], latent_support=usable,
                              applicability_branch=pin["applicability_branch"]))
    return checks


def check_checkpoints(models: Mapping[str, Mapping[str, Any]], checkpoint_map: Path) -> list[PreflightCheck]:
    from e3sse.protocol.adapters import checkpoint_digest

    if not checkpoint_map.is_file():
        return [_missing("checkpoints", "production_compute",
                         f"checkpoint map {checkpoint_map} absent")]
    mapping = json.loads(checkpoint_map.read_text(encoding="utf-8"))
    checks = []
    for name, model in sorted(models.items()):
        check_id = f"checkpoint:{name}"
        if name not in mapping:
            checks.append(_missing(check_id, "production_compute", "not in checkpoint map"))
            continue
        path = Path(mapping[name]).expanduser()
        if not path.exists():
            checks.append(_missing(check_id, "production_compute", f"{path} absent"))
            continue
        digest = checkpoint_digest(path)
        bad = [file for file, sha in sorted(model.get("artifact_files", {}).items())
               if not (path / file).is_file() or file_sha256(path / file) != sha]
        if digest != model["weights_sha256"] or bad:
            checks.append(_failed(check_id, "production_compute", "checkpoint SHA-256 mismatch",
                                  path=str(path), actual_sha256=digest, artifact_mismatch=bad))
        else:
            checks.append(_passed(check_id, "production_compute", "weights SHA-256 verified",
                                  path=str(path), sha256=digest))
    return checks


def check_runtimes(runtimes: Mapping[str, Any]) -> list[PreflightCheck]:
    checks = []
    for name, runtime in sorted(runtimes.items()):
        record = runtime.as_json()
        if record["python_present"]:
            checks.append(_passed(f"runtime_interpreter:{name}", "production_compute",
                                  "frozen interpreter present", python=record["python"],
                                  environment=record["environment"]))
        else:
            checks.append(_missing(f"runtime_interpreter:{name}", "production_compute",
                                   "frozen interpreter absent; production never installs",
                                   python=record["python"], environment=record["environment"]))
    return checks


def check_ml_neb_runtime_audits(audit_dir: Path, environments: Iterable[str], *,
                                expected_code_sha256: Mapping[str, str],
                                expected_settings_sha256: str) -> list[PreflightCheck]:
    """Per-runtime ASE/FIRE/NEB audits, then the protocol's one-ASE-version rule."""

    checks, versions, barriers = [], {}, {}
    for environment in sorted(set(environments)):
        check_id = f"ml_neb_runtime:{environment}"
        candidates = sorted(audit_dir.glob(f"{environment}-*.json")) if audit_dir.is_dir() else []
        current = []
        for candidate in candidates:
            stored = json.loads(candidate.read_text(encoding="utf-8"))
            if (stored.get("identity", {}).get("code_sha256") == dict(expected_code_sha256)
                    and stored.get("settings_sha256") == expected_settings_sha256):
                current.append((candidate, stored))
        if not current:
            if candidates:
                checks.append(_failed(check_id, "freeze", "every audit is stale (code or frozen "
                                      "settings changed since it ran); rerun runtime_audit",
                                      stale=[str(c) for c in candidates]))
            else:
                checks.append(_missing(check_id, "freeze", "runtime audit absent",
                                       directory=str(audit_dir)))
            continue
        path, report = current[-1]
        identity = report.get("identity", {})
        versions[environment] = identity.get("ase_version")
        barriers[environment] = report.get("checks", {}).get("reference_neb", {}).get("barrier_eV")
        if not report.get("passed"):
            checks.append(_failed(check_id, "freeze", "runtime audit failed", path=str(path),
                                  runtime=report.get("runtime"),
                                  checks=report.get("checks", {}).get("passed")))
        else:
            checks.append(_passed(check_id, "freeze", "frozen NEB/FIRE settings accepted",
                                  path=str(path), sha256=file_sha256(path),
                                  ase_version=versions[environment],
                                  reference_barrier_eV=barriers[environment]))
    if versions:
        distinct = sorted({str(v) for v in versions.values()})
        finite = [b for b in barriers.values() if isinstance(b, float)]
        spread = max(finite) - min(finite) if len(finite) > 1 else 0.0
        evidence = {"ase_versions": dict(sorted(versions.items())),
                    "reference_barrier_spread_eV": spread}
        if len(distinct) != 1:
            checks.append(_failed(
                "ml_neb_single_ase_version", "freeze",
                "protocol amendments.prospective.ml_neb.runtime_requirements.ase requires one "
                f"pinned ASE version across all model runtimes; found {distinct}.  Resolve by an "
                "owner decision (rebuild the runtimes on one ASE with new locks and smoke "
                "reports, or amend the protocol); production never upgrades a package", **evidence))
        else:
            checks.append(_passed("ml_neb_single_ase_version", "freeze",
                                  f"one ASE version {distinct[0]}", **evidence))
    return checks


def check_resource_profiles(profiles: Mapping[tuple[str, str], Any],
                            required: Iterable[tuple[str, str]], *,
                            ambiguous: Sequence[str] = ()) -> list[PreflightCheck]:
    checks = [_failed(f"resource_profile:{key}", "production_compute",
                      "several profiles exist and none is bound to this run; run the "
                      "resource_profile stage") for key in sorted(set(ambiguous))]
    duplicate = set(ambiguous)
    for model, workload in sorted(set(required)):
        if workload == "statistics_numpy":
            continue
        check_id = f"resource_profile:{model}/{workload}"
        if f"{model}/{workload}" in duplicate:
            continue
        profile = profiles.get((model, workload))
        if profile is None:
            checks.append(_missing(check_id, "production_compute",
                                   "no measured worker memory/throughput profile"))
        else:
            checks.append(_passed(check_id, "production_compute", "measured profile present",
                                  path=profile.evidence_path, sha256=profile.evidence_sha256,
                                  peak_resident_bytes=profile.peak_resident_bytes,
                                  benchmark_points=len(profile.benchmark)))
    return checks


# ------------------------------------------------------------------- protocol


def check_protocol_open_items(active_protocol: Mapping[str, Any]) -> list[PreflightCheck]:
    checks = []
    for item in active_protocol["open_items"]:
        if item["blocking"]:
            checks.append(_missing(f"protocol_open_item:{item['id']}", "freeze",
                                   item["required_resolution"],
                                   blocker_class=item.get("blocker_class")))
    if not checks:
        checks.append(_passed("protocol_open_items", "freeze", "no blocking open item"))
    return checks


def check_outcome_runners_implemented(statuses: Mapping[str, str], *,
                                      required: bool) -> PreflightCheck:
    """Before the freeze, every stage that reads an outcome must already exist as code.

    Otherwise analysis code would be written after outcomes were seen.  The owner may
    switch this off (``freeze.require_outcome_runners_implemented: false``), which is
    then recorded with every run as a declared deviation.
    """

    missing = sorted(name for name, status in statuses.items() if status != "implemented")
    if not required:
        return _passed("outcome_runners_before_freeze", "nothing",
                       "declared deviation: outcome-reading runners may be written after "
                       "the freeze", not_yet_implemented=missing)
    if missing:
        return _missing("outcome_runners_before_freeze", "freeze",
                        "outcome-reading stages without a tested runner: freeze only after "
                        "the analysis code exists", not_yet_implemented=missing)
    return _passed("outcome_runners_before_freeze", "freeze",
                   "every outcome-reading stage has an implemented runner")


def check_scope(active_protocol: Mapping[str, Any]) -> PreflightCheck:
    scope = active_protocol["amendments"]["scope"]
    governance = active_protocol["governance"]
    ok = (str(scope.get("study_3", "")).startswith("deferred")
          and governance.get("external_registration_required") is False
          and governance.get("preregistration_claimed") is False)
    record = {"study_3": scope.get("study_3"), "governance_mode": governance.get("governance_mode")}
    if not ok:
        return _failed("scope_and_governance", "freeze", "transport must stay deferred under the "
                       "internal freeze without external registration", **record)
    return _passed("scope_and_governance", "freeze",
                   "internal version-controlled freeze; transport deferred", **record)


def check_outcome_access_declaration(study1_config: Path, *, freeze_bound: bool) -> PreflightCheck:
    """No Study-1 outcome record may exist before the freeze is materialized."""

    if freeze_bound:
        return _passed("outcome_access_declaration", "freeze",
                       "freeze already bound; the declaration was verified when it was built")
    config = json.loads(study1_config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    found = sorted(str(p) for pattern in ("*/inference/*/outcome_access/*.json",
                                          "*/inference/*/study1_runs/*.json")
                   for p in root.glob(pattern)) if root.is_dir() else []
    if found:
        return _failed("outcome_access_declaration", "freeze",
                       f"{len(found)} Study-1 outcome record(s) predate the freeze",
                       first=found[0])
    return _passed("outcome_access_declaration", "freeze",
                   "no Study-1 outcome record exists", study1_output_root=str(root))


def summarize(checks: Sequence[PreflightCheck]) -> dict[str, Any]:
    blocking = [c for c in checks if c.blocking]
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "classification": "outcome_free_preflight_evidence",
        "freeze_permitted": not any(c.blocks == "freeze" for c in blocking),
        "production_compute_permitted": not blocking,
        "counts": {status: sum(c.status == status for c in checks) for status in STATUSES},
        "blocking": [c.check_id for c in blocking],
        "checks": [asdict(c) | {"blocking": c.blocking} for c in checks],
    }
