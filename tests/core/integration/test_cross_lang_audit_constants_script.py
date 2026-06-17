from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / ".claude" / "skills" / "cross-lang-sync" / "audit_constants.py"
TIER1_CONSTANTS = {
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "ADL_GINI_THRESHOLD",
    "ADL_HARBERGER_TAX_RATE",
    "MIN_CONFIDENCE",
    "MAX_HARM_SCORE",
}


def test_cross_lang_audit_json_reports_full_tier1_set() -> None:
    report = _run_audit_json()

    result_names = {item["constant"] for item in report["results"]}

    assert report["status"] == "ALIGNED"
    assert TIER1_CONSTANTS <= result_names
    for item in report["results"]:
        if item["constant"] in TIER1_CONSTANTS:
            assert item["status"] == "ALIGNED", item
            assert item["python_value"] is not None, item
            assert item["rust_definitions"], item


def test_cross_lang_audit_json_reports_proofspace_reexports() -> None:
    report = _run_audit_json()

    proofspace = report["proofspace_reexports"]

    assert proofspace["status"] == "ALIGNED"
    assert proofspace["violations"] == []
    assert {
        "IHSAN_THRESHOLD",
        "ADL_GINI_MAX",
        "MAX_HARM_SCORE",
        "MIN_CONFIDENCE",
        "SNR_FLOOR",
    } <= {item["constant"] for item in proofspace["reexports"]}


def test_cross_lang_audit_json_checks_python_mirror_surfaces() -> None:
    report = _run_audit_json()

    mirrors = report["python_mirror_surfaces"]

    assert mirrors["status"] == "ALIGNED"
    assert mirrors["violations"] == []
    assert {
        "scripts/ci_proof_pyramid_gate.py",
        "runtime/core/constants.py",
        "bizra-node0/core/integration/constants.py",
    } <= {item["file"] for item in mirrors["surfaces"]}


def _run_audit_json() -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)
