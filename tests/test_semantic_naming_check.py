"""The naming validator flags new numeric version labels but not external versions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".agents" / "scripts" / "check_semantic_names.py"
spec = importlib.util.spec_from_file_location("check_semantic_names", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_new_numeric_version_labels_are_flagged():
    text = 'SCHEMA = "e3sse.selection.v2"\nbuild_v4 = 1\nv3_protocol = 2\nsee v4 for details'
    assert checker._tokens(text) == {"e3sse.selection.v2", "build_v4", "v3_protocol", "v4"}
    assert {"V4", "protocol-v4", "protocolV4"} <= checker._tokens("Protocol V4 protocol-v4 protocolV4")
    for name in ("protocol_v4.json", "analysis_v2.py", "study1_v1_manifest.json",
                 "e3sse_v4_design_report.md", "plan-version2.md", "protocol_v4", "protocolV4.py"):
        assert checker.PATH_PATTERN.search(name), name


def test_external_and_dotted_versions_are_not_flagged():
    text = ("sevenn==0.11.0 CHGNet (v0.3.0) M3GNet-MatPES-PBE-v2025.1-PES Python 3.11 "
            "doi.org/10.1038/s41524-024-01500-6 ~/.venvs/e3sse-sevennet-0.10.1 "
            "arXiv v2 arXiv:2307.09291v2 CUDA v12")
    assert checker._tokens(text) == set()
    for name in ("chgnet-0.3.0.txt", "reliability_gated_internal_freeze_protocol.json",
                 "e3sse_reliability_gated_design_report.md"):
        assert not checker.PATH_PATTERN.search(name), name
