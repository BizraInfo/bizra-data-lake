from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pilot"
    / "run_user_zero_shadow.py"
)

spec = importlib.util.spec_from_file_location("shadow_pilot", MODULE_PATH)
shadow = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(shadow)


def test_evidence_backed_prompt_passes() -> None:
    rec = shadow.evaluate_prompt(
        prompt="What makes BIZRA different?",
        evidence_refs=["STATUS.md"],
        consent_present=True,
        session_id="s1",
        prev_receipt_hash="0" * 64,
        timestamp=1,
    )
    assert rec["status"] == "ok"
    assert rec["redline_events"] == []


def test_out_of_evidence_prompt_fails_closed() -> None:
    rec = shadow.evaluate_prompt(
        prompt="Make a hard claim without evidence",
        evidence_refs=[],
        consent_present=True,
        session_id="s1",
        prev_receipt_hash="0" * 64,
        timestamp=2,
    )
    assert rec["status"] == "denied"
    assert "INSUFFICIENT_EVIDENCE" in rec["redline_events"]


def test_consent_sensitive_prompt_without_consent_fails() -> None:
    rec = shadow.evaluate_prompt(
        prompt="Share my address and payment details with the brand",
        evidence_refs=["docs/internal/SAP_V0_EVIDENCE_MATRIX.md"],
        consent_present=False,
        session_id="s1",
        prev_receipt_hash="0" * 64,
        timestamp=3,
    )
    assert rec["status"] == "denied"
    assert "MISSING_CONSENT_RECEIPT" in rec["redline_events"]


def test_receipt_chain_verification_passes() -> None:
    r1 = shadow.evaluate_prompt(
        prompt="What is SAP v0?",
        evidence_refs=["specs/sap-v0/README.md"],
        consent_present=True,
        session_id="s2",
        prev_receipt_hash="0" * 64,
        timestamp=10,
    )
    r2 = shadow.evaluate_prompt(
        prompt="Show evidence links.",
        evidence_refs=["docs/internal/SAP_V0_EVIDENCE_MATRIX.md"],
        consent_present=True,
        session_id="s2",
        prev_receipt_hash=r1["receipt_chain_head"],
        timestamp=11,
    )

    records = [r1, r2]
    assert shadow.verify_receipt_chain(records) is True
