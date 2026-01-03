import json
import re
from hashlib import sha256
from pathlib import Path

import pytest


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def test_write_receipt_adds_integrity_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from core import main as kernel_main

    monkeypatch.setenv("BIZRA_KERNEL_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setenv("BIZRA_KERNEL_RECEIPTS", "1")

    payload = {
        "schema": "test_receipt_v1",
        "request_id": "unit_test_request_id",
        "status": "ok",
        "meta": {"a": 1, "b": 2},
    }

    out_path = kernel_main._write_receipt(payload)
    assert out_path is not None
    assert out_path.exists()

    receipt = json.loads(out_path.read_text(encoding="utf-8"))

    assert "integrity_hash" in receipt
    assert re.match(r"^sha256:[0-9a-f]{64}$", receipt["integrity_hash"])

    # Verify integrity hash matches canonical JSON of receipt excluding integrity_hash.
    expected_hash = _sha256_hex(_canonical_json_bytes({k: v for k, v in receipt.items() if k != "integrity_hash"}))
    assert receipt["integrity_hash"] == f"sha256:{expected_hash}"


def test_write_receipt_emits_evidence_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from core import main as kernel_main

    monkeypatch.setenv("BIZRA_KERNEL_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setenv("BIZRA_KERNEL_RECEIPTS", "1")

    evidence = [
        {"hash": "sha256:" + "0" * 64, "path": "docs/a.md"},
        {"hash": "sha256:" + "1" * 64, "path": "docs/b.md"},
    ]

    payload = {
        "schema": "test_receipt_v1",
        "request_id": "unit_test_request_with_evidence",
        "status": "ok",
        "evidence": evidence,
    }

    out_path = kernel_main._write_receipt(payload)
    assert out_path is not None

    folder = out_path.parent
    evidence_path = folder / "evidence.json"
    assert evidence_path.exists()

    receipt = json.loads(out_path.read_text(encoding="utf-8"))
    artifact = receipt.get("evidence_artifact")
    assert isinstance(artifact, dict)
    assert artifact.get("file") == "evidence.json"
    assert artifact.get("count") == 2
    assert re.match(r"^sha256:[0-9a-f]{64}$", str(artifact.get("sha256")))

    expected_evidence_sha = _sha256_hex(_canonical_json_bytes(evidence))
    assert artifact["sha256"] == f"sha256:{expected_evidence_sha}"

    # Evidence file should be canonical JSON + trailing newline.
    assert evidence_path.read_bytes() == _canonical_json_bytes(evidence) + b"\n"
