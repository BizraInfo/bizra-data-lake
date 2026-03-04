"""
Tests for proof_forge chain verification hardening.

Focus:
- evidence hash recomputation catches receipt tampering
- signature verification is enforced by default
- optional legacy unsigned override works only when explicitly enabled
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.proof_forge import forge_evidence


def _create_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "app.py").write_text("print('hello world')\n", encoding="utf-8")
    return project_dir


def _load_receipt(receipt_path: Path) -> dict:
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def test_verify_chain_detects_tampered_receipt_body(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path)
    _, receipt_path = forge_evidence.forge_receipt(
        project_dir=project_dir,
        description="initial receipt",
        verification={
            "checks": [],
            "checks_run": 0,
            "checks_passed": 0,
            "overall_pass": None,
        },
    )

    receipt_file = Path(receipt_path)
    receipt = _load_receipt(receipt_file)
    receipt["description"] = "tampered description"
    receipt_file.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    result = forge_evidence.verify_chain(project_dir)
    assert result["valid"] is False
    assert any(
        "evidence_hash_mismatch" in r.get("error", "")
        for r in result.get("results", [])
    )


def test_verify_chain_requires_signature_by_default(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path)
    _, receipt_path = forge_evidence.forge_receipt(
        project_dir=project_dir,
        description="signed receipt",
        verification={
            "checks": [],
            "checks_run": 0,
            "checks_passed": 0,
            "overall_pass": None,
        },
    )

    receipt_file = Path(receipt_path)
    receipt = _load_receipt(receipt_file)
    receipt.pop("signature", None)
    receipt_file.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    strict = forge_evidence.verify_chain(project_dir)
    assert strict["valid"] is False
    assert any("signature_missing" in r.get("error", "") for r in strict["results"])

    legacy = forge_evidence.verify_chain(project_dir, allow_legacy_unsigned=True)
    assert legacy["valid"] is True


def test_forge_receipt_fails_closed_when_signing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _create_project(tmp_path)

    def _raise_signing_error(receipt_body: str, private_key_hex: str) -> str:
        raise RuntimeError("signing backend unavailable")

    monkeypatch.setattr(forge_evidence, "_sign_receipt", _raise_signing_error)

    with pytest.raises(RuntimeError, match="fail-closed"):
        forge_evidence.forge_receipt(
            project_dir=project_dir,
            description="must not emit unsigned receipt",
            verification={
                "checks": [],
                "checks_run": 0,
                "checks_passed": 0,
                "overall_pass": None,
            },
        )
