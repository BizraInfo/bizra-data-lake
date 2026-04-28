"""Canonical envelope shapes — TypedDicts for stable cross-language shapes.

These are runtime-checkable structural contracts. Every consumer (DemaReceipt,
FourStateModel, ProactiveProposal, CanonicalReceipt-on-the-wire) must produce
output that satisfies the matching TypedDict here.

v0.1 ships TypedDicts (no pydantic dep) for stdlib portability. Stronger
runtime validation is a v0.2 follow-up.
"""

from __future__ import annotations

from typing import Any, TypedDict


class CanonicalReceiptEnvelope(TypedDict):
    """Local Dema receipt envelope — written by DemaReceipt + ReceiptWriter.

    String values for ``truth_label`` MUST be in RECEIPT_TRUTH_LABELS.
    String values for ``approval_status`` MUST be in APPROVAL_STATUSES.
    """

    schema_version: str
    receipt_id: str
    payload_digest: str
    action: str
    truth_label: str
    touched_paths: list[str]
    not_touched_paths: list[str]
    approval_required: bool
    approval_status: str
    timestamp: str
    payload: dict[str, Any]


class CanonicalFourStateModel(TypedDict):
    """Mission state — current → ideal → gap → next admissible action.

    String values for ``truth_label`` MUST be in MISSION_TRUTH_LABELS.
    """

    current: str
    ideal: str
    gap: str
    next_admissible_action: str
    truth_label: str
    timestamp: str


class CanonicalProactiveProposal(TypedDict):
    """Proactive proposal artifact — emitted by the proactive layer.

    String values for ``risk`` MUST be in RISK_LEVELS.
    String values for ``decision`` MUST be in DECISION_VERDICTS.
    """

    schema_version: str
    receipt_id: str | None
    noticed: str
    why_matters: str
    proposal: str
    confidence: float
    risk: str
    reversibility: bool
    decision: str
    decision_reason: str
    timestamp: str
