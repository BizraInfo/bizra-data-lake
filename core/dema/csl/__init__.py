"""Canonical Semantics Layer (CSL) — single source of truth for Dema's
cross-cutting vocabulary.

CSL fixes a recurring drift hazard: TruthLabel, RiskLevel, ApprovalStatus,
DecisionVerdict, and the canonical envelope shapes (Receipt, Four-state
Mission, Proactive Proposal) appear in multiple modules and on both sides
of the Python ↔ TypeScript boundary. CSL declares each set ONCE, and
companion drift tests (Python + TS) refuse to ship when a consumer module
or the TS mirror falls out of sync.

v0.1 scope:
  - Define the canonical sets (labels.py).
  - Define the canonical TypedDict shapes (schemas.py).
  - Provide a CLI to emit the TypeScript mirror so Python stays
    authoritative (`scripts/dema/dema_csl.py emit-ts`).
  - Drift tests in tests/scripts/test_dema_csl.py + frontend/tests/.

v0.1 does NOT migrate existing module-local constants to import from CSL —
that is a v0.2 follow-up to keep this PR scoped. Tests assert today's
local sets match CSL; the moment they diverge, CI fails.
"""

from __future__ import annotations

from core.dema.csl.labels import (
    APPROVAL_STATUSES,
    DECISION_VERDICTS,
    DISPLAY_TRUTH_LABELS,
    MISSION_TRUTH_LABELS,
    RECEIPT_TRUTH_LABELS,
    RISK_LEVELS,
    SCHEMA_VERSION,
    ApprovalStatus,
    DecisionVerdict,
    RiskLevel,
    TruthLabel,
)
from core.dema.csl.schemas import (
    CanonicalFourStateModel,
    CanonicalProofSurface,
    CanonicalProactiveProposal,
    CanonicalReceiptEnvelope,
)

__all__ = [
    "SCHEMA_VERSION",
    # Labels
    "RECEIPT_TRUTH_LABELS",
    "DISPLAY_TRUTH_LABELS",
    "MISSION_TRUTH_LABELS",
    "RISK_LEVELS",
    "APPROVAL_STATUSES",
    "DECISION_VERDICTS",
    "TruthLabel",
    "RiskLevel",
    "ApprovalStatus",
    "DecisionVerdict",
    # Shapes
    "CanonicalReceiptEnvelope",
    "CanonicalFourStateModel",
    "CanonicalProactiveProposal",
    "CanonicalProofSurface",
]
