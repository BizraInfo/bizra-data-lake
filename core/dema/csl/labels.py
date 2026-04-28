"""Canonical label sets and Enum types.

Two distinct truth-label tiers:

  RECEIPT_TRUTH_LABELS   — values that may appear inside a DemaReceipt's
                           ``truth_label`` field. Receipts always describe
                           something real; UNKNOWN is not a receipt label.
  DISPLAY_TRUTH_LABELS   — values that may appear in UI surfaces (Goal
                           cards, status strips). Adds UNKNOWN for honest
                           "no data" placeholders.

MISSION_TRUTH_LABELS == DISPLAY_TRUTH_LABELS because the mission state
machine starts UNKNOWN before any data lands.

RISK_LEVELS, APPROVAL_STATUSES, and DECISION_VERDICTS are flat
authoritative tuples consumed by the proactive layer and the receipt
envelope.

These tuples are the SOURCE OF TRUTH. The accompanying Enums are
convenience wrappers; their `.value` strings match the tuples 1:1 so JSON
serialisation never drifts from the canonical strings.
"""

from __future__ import annotations

from enum import Enum

# CSL itself is versioned; bumping this signals a breaking change to any
# consumer (Python module, TS mirror, downstream service).
SCHEMA_VERSION = "0.1.0"


# ── Truth labels ────────────────────────────────────────────────────────

RECEIPT_TRUTH_LABELS: tuple[str, ...] = (
    "MEASURED",
    "DERIVED",
    "PLANNED",
    "SANDBOX",
)

DISPLAY_TRUTH_LABELS: tuple[str, ...] = (
    "MEASURED",
    "DERIVED",
    "PLANNED",
    "SANDBOX",
    "UNKNOWN",
)

MISSION_TRUTH_LABELS: tuple[str, ...] = DISPLAY_TRUTH_LABELS


class TruthLabel(str, Enum):
    """Display-tier truth label (superset). Use the string value when
    serialising to JSON / when crossing the Python ↔ TS boundary."""

    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    PLANNED = "PLANNED"
    SANDBOX = "SANDBOX"
    UNKNOWN = "UNKNOWN"


# ── Risk levels ─────────────────────────────────────────────────────────

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Approval statuses ───────────────────────────────────────────────────

APPROVAL_STATUSES: tuple[str, ...] = ("granted", "pending", "n/a", "denied")


class ApprovalStatus(str, Enum):
    GRANTED = "granted"
    PENDING = "pending"
    NA = "n/a"
    DENIED = "denied"


# ── Decision verdicts ───────────────────────────────────────────────────

DECISION_VERDICTS: tuple[str, ...] = (
    "auto_low_risk",
    "notify",
    "require_approval",
    "forbid",
)


class DecisionVerdict(str, Enum):
    AUTO_LOW_RISK = "auto_low_risk"
    NOTIFY = "notify"
    REQUIRE_APPROVAL = "require_approval"
    FORBID = "forbid"
