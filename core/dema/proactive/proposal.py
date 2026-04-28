"""Proactive proposal — composes the operator-visible artifact.

A ProactiveProposal carries a structured "what I noticed / why it matters /
what I propose / decision". It is paired with a DemaReceipt so every
proposal — accepted or not — leaves a hash-chained trail under
sovereign_state/dema/proposals/.

No proposal triggers an action by itself. The decision verdict tells the
caller (or the operator) whether anything may run; only `auto_low_risk`
permits silent queueing of a reversible action, and even then a receipt is
written first.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.dema.proactive.interruption_policy import Decision
from core.dema.proactive.signals import AmbientSignal
from core.dema.receipts import DemaReceipt, ReceiptWriter


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class ProactiveProposal:
    noticed: str
    why_matters: str
    proposal: str
    confidence: float
    risk: str
    reversibility: bool
    decision: str
    decision_reason: str
    schema_version: str = "0.1.0"
    timestamp: str = field(default_factory=_utc_now)
    receipt_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProposalWriter:
    """Writes a ProactiveProposal + a paired DemaReceipt under sovereign_state."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.proposals_root = self.root / "proposals"
        self.proposals_root.mkdir(parents=True, exist_ok=True)
        self.receipt_writer = ReceiptWriter(self.root)

    def _date_dir(self) -> Path:
        d = self.proposals_root / _today_utc()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(
        self,
        proposal: ProactiveProposal,
    ) -> tuple[str, Path, Path]:
        # 1. Write the receipt.
        receipt = DemaReceipt(
            action="dema.proactive.proposal",
            truth_label="MEASURED",
            touched_paths=[str(self.proposals_root)],
            not_touched_paths=[
                "network",
                "desktop",
                "MEMORY.md",
                "docs/canon/",
                "destructive_action",
                "social_publish",
                "long_term_memory_promotion",
            ],
            approval_required=proposal.decision != "auto_low_risk",
            approval_status=(
                "n/a" if proposal.decision == "auto_low_risk" else "pending"
            ),
            payload={
                "noticed": proposal.noticed,
                "why_matters": proposal.why_matters,
                "proposal": proposal.proposal,
                "confidence": proposal.confidence,
                "risk": proposal.risk,
                "reversibility": proposal.reversibility,
                "decision": proposal.decision,
                "decision_reason": proposal.decision_reason,
            },
        )
        rid, receipt_path = self.receipt_writer.write(receipt)
        proposal.receipt_id = rid

        # 2. Write the proposal artifact.
        proposal_path = self._date_dir() / f"{rid[:16]}.json"
        proposal_path.write_text(
            json.dumps(proposal.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return rid, proposal_path, receipt_path


def compose_proposal(signal: AmbientSignal, decision: Decision) -> ProactiveProposal:
    """Render the operator-visible proposal text from signal + decision."""
    intent = decision.intent
    why = intent.explanation
    propose = {
        "auto_low_risk": "Dema queued a reversible suggestion (silent).",
        "notify": "Dema notified — review when convenient; no action taken.",
        "require_approval": ("Dema awaits your explicit approval before any action."),
        "forbid": "Dema will NOT act proactively on this signal.",
    }[decision.verdict]

    return ProactiveProposal(
        noticed=f"{signal.kind} (confidence {signal.confidence:.2f})",
        why_matters=why,
        proposal=propose,
        confidence=intent.confidence,
        risk=intent.risk,
        reversibility=intent.reversible,
        decision=decision.verdict,
        decision_reason=decision.reason,
    )
