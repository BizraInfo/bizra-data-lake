"""Canonical Semantics Layer drift tests.

CSL declares the authoritative vocabulary. These tests assert that:

  1. Each Enum's `.value` strings match the corresponding canonical tuple
     1:1 (no drift between the tuple and the Enum).
  2. Receipt-tier and display-tier truth-label sets relate correctly
     (RECEIPT ⊆ DISPLAY; DISPLAY adds exactly UNKNOWN).
  3. Existing consumer modules (DemaReceipt, FourStateModel, proactive
     intent + policy) carry value sets that match CSL — if anyone changes
     a label in a consumer without updating CSL (or vice versa), this
     test fails.
  4. The on-disk TypeScript mirror equals what `dema_csl.py emit-ts`
     would emit right now. If anyone edits ``frontend/src/lib/dema-csl.ts``
     by hand without regenerating, this test fails.
  5. Canonical TypedDicts cover the keys consumer dataclasses produce
     (so adding a new field to a consumer requires a CSL update).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema.csl import (  # noqa: E402
    APPROVAL_STATUSES,
    DECISION_VERDICTS,
    DISPLAY_TRUTH_LABELS,
    MISSION_TRUTH_LABELS,
    RECEIPT_TRUTH_LABELS,
    RISK_LEVELS,
    ApprovalStatus,
    CanonicalFourStateModel,
    CanonicalProactiveProposal,
    CanonicalReceiptEnvelope,
    DecisionVerdict,
    RiskLevel,
    TruthLabel,
)

# ── 1. Enum ↔ tuple parity ──────────────────────────────────────────


def test_truth_label_enum_matches_display_tuple():
    enum_values = {m.value for m in TruthLabel}
    assert enum_values == set(DISPLAY_TRUTH_LABELS)
    # Order independence is fine for sets, but the Enum order should also
    # match canonical order to keep diffs predictable.
    assert [m.value for m in TruthLabel] == list(DISPLAY_TRUTH_LABELS)


def test_risk_level_enum_matches_tuple():
    assert [m.value for m in RiskLevel] == list(RISK_LEVELS)


def test_approval_status_enum_matches_tuple():
    assert [m.value for m in ApprovalStatus] == list(APPROVAL_STATUSES)


def test_decision_verdict_enum_matches_tuple():
    assert [m.value for m in DecisionVerdict] == list(DECISION_VERDICTS)


# ── 2. Truth-label tier relationships ────────────────────────────────


def test_receipt_truth_subset_of_display():
    assert set(RECEIPT_TRUTH_LABELS).issubset(set(DISPLAY_TRUTH_LABELS))


def test_display_adds_exactly_unknown():
    assert set(DISPLAY_TRUTH_LABELS) - set(RECEIPT_TRUTH_LABELS) == {"UNKNOWN"}


def test_mission_truth_equals_display_truth():
    assert MISSION_TRUTH_LABELS == DISPLAY_TRUTH_LABELS


# ── 3. Consumer module drift ─────────────────────────────────────────


def test_dema_receipt_truth_labels_match_csl():
    from core.dema.receipts import VALID_TRUTH_LABELS as receipts_labels

    assert set(receipts_labels) == set(RECEIPT_TRUTH_LABELS), (
        "DemaReceipt's truth-label set drifted from CSL "
        "RECEIPT_TRUTH_LABELS. Update one or the other to stay aligned."
    )


def test_dema_receipt_approval_statuses_match_csl():
    from core.dema.receipts import VALID_APPROVAL as receipts_approval

    assert set(receipts_approval) == set(APPROVAL_STATUSES), (
        "DemaReceipt's approval-status set drifted from CSL " "APPROVAL_STATUSES."
    )


def test_mission_state_truth_labels_match_csl():
    from core.dema.mission_state import (
        VALID_TRUTH_LABELS as mission_labels,
    )

    assert set(mission_labels) == set(MISSION_TRUTH_LABELS), (
        "FourStateModel's truth-label set drifted from CSL " "MISSION_TRUTH_LABELS."
    )


def test_proactive_intent_risk_levels_match_csl():
    from core.dema.proactive.intent_model import _RULES

    consumer_risks = {risk for (_intent, risk, _rev, _expl) in _RULES.values()}
    # Consumer may legitimately use a SUBSET (not every signal triggers
    # every risk), but every risk it uses must be in CSL.
    assert consumer_risks.issubset(set(RISK_LEVELS)), (
        f"intent_model risk values {consumer_risks - set(RISK_LEVELS)} "
        "are not in CSL RISK_LEVELS"
    )


def test_proactive_decision_verdicts_match_csl():
    from typing import get_args

    from core.dema.proactive.interruption_policy import DecisionVerdict as PolDV

    # PolDV is a Literal type alias.
    pol_values = set(get_args(PolDV))
    assert pol_values == set(DECISION_VERDICTS), (
        "interruption_policy.DecisionVerdict drifted from CSL " "DECISION_VERDICTS."
    )


# ── 4. TypeScript mirror drift ──────────────────────────────────────


def test_typescript_mirror_matches_python():
    """The on-disk dema-csl.ts must equal what `dema_csl.py emit-ts` emits."""
    ts_path = REPO_ROOT / "frontend" / "src" / "lib" / "dema-csl.ts"
    assert ts_path.exists(), (
        f"{ts_path} is missing; regenerate with "
        "`python scripts/dema/dema_csl.py emit-ts --write`."
    )

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "dema" / "dema_csl.py"),
            "emit-ts",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
        check=True,
    )
    expected = res.stdout
    actual = ts_path.read_text(encoding="utf-8")
    assert actual == expected, (
        "frontend/src/lib/dema-csl.ts is out of sync with CSL. "
        "Regenerate with `python scripts/dema/dema_csl.py emit-ts --write`."
    )


# ── 5. Canonical envelope key coverage ──────────────────────────────


def test_canonical_receipt_envelope_covers_dema_receipt_keys():
    from core.dema.receipts import DemaReceipt

    receipt = DemaReceipt(
        action="dema.csl.test",
        truth_label="MEASURED",
        touched_paths=["x"],
    )
    keys = set(receipt.to_dict().keys())
    # The receipt as written by DemaReceipt does not yet include
    # receipt_id / payload_digest — those are added by ReceiptWriter
    # at seal time. CSL covers the sealed envelope, which is a superset.
    csl_keys = set(CanonicalReceiptEnvelope.__annotations__.keys())
    missing_in_csl = keys - csl_keys
    assert missing_in_csl == set(), (
        f"DemaReceipt has fields {missing_in_csl} not in CSL envelope; "
        "add them to CanonicalReceiptEnvelope."
    )


def test_canonical_four_state_covers_mission_state_keys():
    from core.dema.mission_state import FourStateModel

    state = FourStateModel()
    keys = set(state.to_dict().keys())
    csl_keys = set(CanonicalFourStateModel.__annotations__.keys())
    missing_in_csl = keys - csl_keys
    assert missing_in_csl == set(), (
        f"FourStateModel has fields {missing_in_csl} not in CSL shape; "
        "add them to CanonicalFourStateModel."
    )


def test_canonical_proactive_proposal_covers_proposal_keys():
    from core.dema.proactive.proposal import ProactiveProposal

    proposal = ProactiveProposal(
        noticed="x",
        why_matters="y",
        proposal="z",
        confidence=0.5,
        risk="low",
        reversibility=True,
        decision="auto_low_risk",
        decision_reason="r",
    )
    keys = set(proposal.to_dict().keys())
    csl_keys = set(CanonicalProactiveProposal.__annotations__.keys())
    missing_in_csl = keys - csl_keys
    assert missing_in_csl == set(), (
        f"ProactiveProposal has fields {missing_in_csl} not in CSL shape; "
        "add them to CanonicalProactiveProposal."
    )
