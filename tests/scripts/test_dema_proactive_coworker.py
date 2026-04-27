"""Dema Proactive Coworker Policy v0.1 contract tests.

Covers:

  Core layer
  ----------
  1. Signal validation: unknown kind / out-of-range confidence rejected.
  2. Intent prediction: each VALID_SIGNAL_KIND maps to a defined intent
     with consistent risk + reversibility.
  3. Policy decisions:
       a. low-risk reversible + confidence >= 0.85 → auto_low_risk
       b. low-risk reversible + 0.55 <= confidence < 0.85 → notify
       c. low-risk reversible + confidence < 0.55 → require_approval
       d. medium-risk reversible → require_approval
       e. medium-risk non-reversible → require_approval (but never auto)
       f. high-risk → forbid
       g. destructive intent (format_drive_candidate) → forbid
       h. credential_exposure_candidate → forbid

  CLI + receipt layer
  -------------------
  4. dema_proactive.py evaluate: end-to-end JSON shape + receipt emitted
     under sovereign_state sandbox.
  5. Receipt non-claims: network, desktop, MEMORY.md, docs/canon/,
     destructive_action, social_publish, long_term_memory_promotion all
     listed as not_touched_paths.
  6. forbid decision DOES NOT mark approval_status="granted" — it stays
     "pending" and the proposal text says Dema will not act.
  7. auto_low_risk decision marks approval_required=False.
  8. Boundary check: every proposal/receipt path lives under --root.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema.proactive import (  # noqa: E402
    VALID_SIGNAL_KINDS,
    AmbientSignal,
    decide,
    predict_intent,
)

# ── Core: signal validation ──────────────────────────────────────────


def test_signal_rejects_unknown_kind():
    with pytest.raises(ValueError):
        AmbientSignal(kind="bogus_kind", confidence=0.9)


def test_signal_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        AmbientSignal(kind="downloads_folder_large", confidence=1.5)
    with pytest.raises(ValueError):
        AmbientSignal(kind="downloads_folder_large", confidence=-0.1)


def test_signal_rejects_invalid_urgency():
    with pytest.raises(ValueError):
        AmbientSignal(
            kind="downloads_folder_large",
            confidence=0.9,
            urgency="extreme",
        )


# ── Core: intent prediction ──────────────────────────────────────────


def test_every_valid_signal_kind_has_an_intent_rule():
    for kind in VALID_SIGNAL_KINDS:
        signal = AmbientSignal(kind=kind, confidence=0.5)
        intent = predict_intent(signal)
        assert intent.intent
        assert intent.risk in ("low", "medium", "high")
        assert isinstance(intent.reversible, bool)
        assert intent.explanation


# ── Core: policy decision matrix ─────────────────────────────────────


def _decide_for(kind: str, confidence: float, *, user_preference: str = "default"):
    signal = AmbientSignal(kind=kind, confidence=confidence)
    intent = predict_intent(signal)
    return decide(intent, user_preference=user_preference)


def test_low_risk_reversible_high_confidence_auto_low_risk():
    d = _decide_for("downloads_folder_large", 0.92)
    assert d.verdict == "auto_low_risk"


def test_low_risk_reversible_mid_confidence_notify():
    d = _decide_for("downloads_folder_large", 0.70)
    assert d.verdict == "notify"


def test_low_risk_reversible_low_confidence_require_approval():
    d = _decide_for("downloads_folder_large", 0.30)
    assert d.verdict == "require_approval"


def test_medium_risk_reversible_require_approval():
    d = _decide_for("resource_pressure", 0.95)
    assert d.verdict == "require_approval"


def test_medium_risk_non_reversible_require_approval_never_auto():
    d = _decide_for("duplicate_delete_candidate", 0.99)
    assert d.verdict == "require_approval"
    assert "approval" in d.reason.lower() or "reversible" in d.reason.lower()


def test_high_risk_forbid():
    d = _decide_for("format_drive_candidate", 0.99)
    assert d.verdict == "forbid"


def test_destructive_intent_format_drive_forbid():
    d = _decide_for("format_drive_candidate", 0.51)
    assert d.verdict == "forbid"
    assert "destructive" in d.reason.lower() or "high" in d.reason.lower()


def test_destructive_intent_credential_audit_forbid():
    d = _decide_for("credential_exposure_candidate", 0.97)
    assert d.verdict == "forbid"


def test_social_post_candidate_requires_approval():
    d = _decide_for("social_post_candidate", 0.99)
    assert d.verdict == "require_approval"


# ── CLI + receipt layer ──────────────────────────────────────────────


def _run_cli(*args: str, root: Path) -> dict:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dema" / "dema_proactive.py"),
        *args,
        "--root",
        str(root),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
        check=True,
    )
    return json.loads(res.stdout)


REQUIRED_NON_CLAIMS = (
    "network",
    "desktop",
    "MEMORY.md",
    "docs/canon/",
    "destructive_action",
    "social_publish",
    "long_term_memory_promotion",
)


def test_cli_evaluate_emits_receipt_and_proposal(tmp_path):
    out = _run_cli(
        "evaluate",
        "--signal",
        "downloads_folder_large",
        "--confidence",
        "0.87",
        "--urgency",
        "low",
        root=tmp_path,
    )
    assert out["ok"] is True
    assert out["kind"] == "dema_proactive_evaluate"
    assert out["decision"]["verdict"] == "auto_low_risk"

    proposal_path = Path(out["proposal_path"])
    receipt_path = Path(out["receipt_path"])
    assert proposal_path.exists()
    assert receipt_path.exists()

    # Both artifacts live under the supplied --root.
    assert proposal_path.is_relative_to(tmp_path)
    assert receipt_path.is_relative_to(tmp_path)


def test_cli_evaluate_forbid_does_not_grant_approval(tmp_path):
    out = _run_cli(
        "evaluate",
        "--signal",
        "format_drive_candidate",
        "--confidence",
        "0.99",
        "--urgency",
        "low",
        root=tmp_path,
    )
    assert out["decision"]["verdict"] == "forbid"

    receipt = json.loads(Path(out["receipt_path"]).read_text(encoding="utf-8"))
    # forbid is not auto_low_risk, so approval_required is True.
    assert receipt["approval_required"] is True
    # Pending, never granted just because Dema noticed it.
    assert receipt["approval_status"] in ("pending", "n/a")
    assert receipt["approval_status"] != "granted"

    # Proposal text reflects the forbid verdict.
    proposal = json.loads(Path(out["proposal_path"]).read_text(encoding="utf-8"))
    assert "not act" in proposal["proposal"].lower()


def test_cli_evaluate_auto_low_risk_does_not_require_approval(tmp_path):
    out = _run_cli(
        "evaluate",
        "--signal",
        "long_idle_session",
        "--confidence",
        "0.91",
        root=tmp_path,
    )
    assert out["decision"]["verdict"] == "auto_low_risk"
    receipt = json.loads(Path(out["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["approval_required"] is False
    assert receipt["approval_status"] == "n/a"


def test_receipt_non_claims_present_on_every_decision(tmp_path):
    """forbid, require_approval, notify, auto_low_risk: all carry the same
    non_touched_paths boundary."""
    pairs = [
        ("downloads_folder_large", 0.95, "auto_low_risk"),  # auto
        ("downloads_folder_large", 0.70, "notify"),  # notify
        ("downloads_folder_large", 0.30, "require_approval"),  # low conf
        ("format_drive_candidate", 0.99, "forbid"),  # forbid
    ]
    for kind, conf, expected in pairs:
        out = _run_cli(
            "evaluate",
            "--signal",
            kind,
            "--confidence",
            str(conf),
            root=tmp_path,
        )
        assert out["decision"]["verdict"] == expected
        receipt = json.loads(Path(out["receipt_path"]).read_text(encoding="utf-8"))
        for claim in REQUIRED_NON_CLAIMS:
            assert (
                claim in receipt["not_touched_paths"]
            ), f"{claim} missing from {kind} receipt non_touched_paths"
        # Touched paths confined to sandbox under --root.
        for p in receipt["touched_paths"]:
            assert (
                str(tmp_path) in p
            ), f"touched path {p} escapes sandbox root {tmp_path}"


def test_no_destructive_or_token_language_in_proposal_text(tmp_path):
    """Proposal text must never imply destruction or financial reward."""
    forbidden_terms = (
        "deleted",
        "formatted",
        "wiped",
        "yield",
        "earnings",
        "investment",
        "guaranteed",
        "AGI achieved",
        "first in the world",
        "token profit",
    )
    for kind in VALID_SIGNAL_KINDS:
        out = _run_cli(
            "evaluate",
            "--signal",
            kind,
            "--confidence",
            "0.5",
            root=tmp_path,
        )
        proposal = json.loads(Path(out["proposal_path"]).read_text(encoding="utf-8"))
        haystack = (
            (proposal.get("proposal") or "")
            + " "
            + (proposal.get("why_matters") or "")
            + " "
            + (proposal.get("noticed") or "")
        ).lower()
        for term in forbidden_terms:
            assert (
                term.lower() not in haystack
            ), f"forbidden term {term!r} appeared in {kind} proposal"


def test_receipt_id_is_present_and_well_formed(tmp_path):
    out = _run_cli(
        "evaluate",
        "--signal",
        "downloads_folder_large",
        "--confidence",
        "0.9",
        root=tmp_path,
    )
    rid = out["receipt_id"]
    assert isinstance(rid, str)
    assert re.fullmatch(r"[0-9a-f]{64}", rid), "receipt_id must be BLAKE3 hex"
