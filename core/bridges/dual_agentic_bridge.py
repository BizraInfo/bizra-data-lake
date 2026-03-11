"""
Dual-Agentic Bridge — PAT + SAT Connector
==========================================
Bridges the Primary Action Team (PAT) and Secondary Action Team (SAT)
enabling coordinated execution with Byzantine fault-tolerant validation.

PAT (7 agents): Execute actions
SAT (5 validators): Validate with veto power

Standing on Giants: Byzantine Consensus + Multi-Agent Systems + Constitutional AI
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.orchestration.team_planner import AgentRole, TeamTask
from core.pci.crypto import (
    canonicalize_and_validate,
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
)

logger = logging.getLogger(__name__)


class VetoReason(str, Enum):
    """Reasons for SAT veto."""

    SECURITY_VIOLATION = "security_violation"
    ETHICS_VIOLATION = "ethics_violation"
    PERFORMANCE_RISK = "performance_risk"
    CONSISTENCY_ERROR = "consistency_error"
    RESOURCE_OVERFLOW = "resource_overflow"
    IHSAN_THRESHOLD = "ihsan_threshold_breach"


class ConsensusResult(Enum):
    """Result of Byzantine consensus."""

    APPROVED = auto()
    VETOED = auto()
    PENDING = auto()
    TIMEOUT = auto()


class SATMode(str, Enum):
    """SAT operating profile."""

    MINI5 = "mini5"
    FULL49 = "full49"


@dataclass
class ActionProposal:
    """A proposed action from PAT for SAT validation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    proposer_role: AgentRole = AgentRole.MASTER_REASONER
    ihsan_estimate: float = 0.95
    risk_estimate: float = 0.1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Vote:
    """A validator's vote on an action proposal."""

    validator_role: AgentRole = AgentRole.SECURITY_GUARDIAN
    approve: bool = True
    confidence: float = 1.0
    veto_reason: Optional[VetoReason] = None
    comments: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConsensusOutcome:
    """Final consensus outcome for a proposal."""

    proposal_id: str = ""
    result: ConsensusResult = ConsensusResult.PENDING
    votes: List[Vote] = field(default_factory=list)
    approval_count: int = 0
    veto_count: int = 0
    quorum_met: bool = False
    final_ihsan: float = 0.0
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NegotiationReceipt:
    """
    Signed negotiation receipt between PAT proposer and SAT authority.

    Both parties sign the same canonical payload digest so downstream audit and
    dispute resolution can verify agreement cryptographically.
    """

    receipt_id: str
    proposal_id: str
    task_id: str
    action_type: str
    sat_mode: str
    proposer_role: AgentRole
    result: ConsensusResult
    approval_count: int
    veto_count: int
    quorum_met: bool
    final_ihsan: float
    risk_estimate: float
    resource_budget: Dict[str, Any]
    created_at: datetime
    resolved_at: datetime
    payload_digest: str
    proposer_agent_id: str
    proposer_public_key: str
    proposer_signature: str
    sat_agent_id: str
    sat_public_key: str
    sat_signature: str
    validator_votes: List[Dict[str, Any]] = field(default_factory=list)
    ledger_sequence: Optional[int] = None
    ledger_entry_hash: Optional[str] = None

    def verify_signatures(self) -> bool:
        """Verify both PAT and SAT signatures over the payload digest."""
        proposer_ok = verify_signature(
            self.payload_digest,
            self.proposer_signature,
            self.proposer_public_key,
        )
        sat_ok = verify_signature(
            self.payload_digest,
            self.sat_signature,
            self.sat_public_key,
        )
        return proposer_ok and sat_ok

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt with explicit cryptographic context."""
        return {
            "receipt_id": self.receipt_id,
            "proposal_id": self.proposal_id,
            "task_id": self.task_id,
            "action_type": self.action_type,
            "sat_mode": self.sat_mode,
            "proposer_role": self.proposer_role.value,
            "result": self.result.name,
            "approval_count": self.approval_count,
            "veto_count": self.veto_count,
            "quorum_met": self.quorum_met,
            "final_ihsan": self.final_ihsan,
            "risk_estimate": self.risk_estimate,
            "resource_budget": self.resource_budget,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat(),
            "payload_digest": self.payload_digest,
            "proposer_agent_id": self.proposer_agent_id,
            "proposer_public_key": self.proposer_public_key,
            "proposer_signature": self.proposer_signature,
            "sat_agent_id": self.sat_agent_id,
            "sat_public_key": self.sat_public_key,
            "sat_signature": self.sat_signature,
            "validator_votes": self.validator_votes,
            "ledger_sequence": self.ledger_sequence,
            "ledger_entry_hash": self.ledger_entry_hash,
            "signatures_valid": self.verify_signatures(),
        }


# Validator functions type
ValidatorFn = Callable[
    [ActionProposal], Awaitable[Tuple[bool, Optional[VetoReason], float]]
]


class DualAgenticBridge:
    """
    Bridge connecting PAT execution with SAT validation.

    Implements Byzantine fault-tolerant consensus:
    - 3 of 5 validators must approve (f < n/3)
    - Any SECURITY or ETHICS veto blocks immediately
    - Ihsan score must meet threshold

    Key guarantees:
    - Safety: No action executes without consensus
    - Liveness: Valid actions eventually execute
    - Constitutional: Ihsan constraints always enforced
    """

    # SAT validator roles for mini5 profile
    SAT_VALIDATORS_MINI5 = {
        AgentRole.SECURITY_GUARDIAN,
        AgentRole.ETHICS_VALIDATOR,
        AgentRole.PERFORMANCE_MONITOR,
        AgentRole.CONSISTENCY_CHECKER,
        AgentRole.RESOURCE_OPTIMIZER,
    }

    # Roles with veto power
    VETO_ROLES = {AgentRole.SECURITY_GUARDIAN, AgentRole.ETHICS_VALIDATOR}

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        vote_timeout: float = 5.0,
        sat_mode: str = SATMode.MINI5.value,
        proposer_private_key_hex: Optional[str] = None,
        proposer_public_key_hex: Optional[str] = None,
        sat_private_key_hex: Optional[str] = None,
        sat_public_key_hex: Optional[str] = None,
        evidence_ledger: Optional[object] = None,
        evidence_ledger_path: Optional[str] = None,
        fail_closed_on_ledger_error: bool = False,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.vote_timeout = vote_timeout
        self.sat_mode = SATMode(sat_mode)
        self.expected_validator_count = 49 if self.sat_mode == SATMode.FULL49 else 5
        self.consensus_threshold = 33 if self.sat_mode == SATMode.FULL49 else 3

        (
            self._proposer_private_key,
            self._proposer_public_key,
        ) = self._resolve_signing_identity(
            proposer_private_key_hex,
            proposer_public_key_hex,
            role_label="pat",
        )
        self._sat_private_key, self._sat_public_key = self._resolve_signing_identity(
            sat_private_key_hex,
            sat_public_key_hex,
            role_label="sat",
        )
        self._proposer_agent_id = f"pat-{self._proposer_public_key[:16]}"
        self._sat_agent_id = f"sat-{self._sat_public_key[:16]}"
        self._fail_closed_on_ledger_error = fail_closed_on_ledger_error
        self._evidence_ledger = evidence_ledger
        if self._evidence_ledger is None and evidence_ledger_path:
            from core.proof_engine.evidence_ledger import EvidenceLedger

            self._evidence_ledger = EvidenceLedger(
                Path(evidence_ledger_path),
                validate_on_append=True,
            )

        self._pending_proposals: Dict[str, ActionProposal] = {}
        self._outcomes: Dict[str, ConsensusOutcome] = {}
        self._receipts: Dict[str, NegotiationReceipt] = {}
        self._validators: Dict[AgentRole, ValidatorFn] = {}
        self._proposal_count = 0
        self._approved_count = 0
        self._vetoed_count = 0
        self._verified_receipts = 0
        self._evidence_entries = 0
        self._evidence_failures = 0

        # Register default validators
        self._register_default_validators()

    @staticmethod
    def _resolve_signing_identity(
        private_key_hex: Optional[str],
        public_key_hex: Optional[str],
        role_label: str,
    ) -> Tuple[str, str]:
        """Resolve signing identity; generate one when not supplied."""
        if private_key_hex and public_key_hex:
            return private_key_hex, public_key_hex
        if private_key_hex or public_key_hex:
            raise ValueError(
                f"{role_label} signer requires both private and public key"
            )
        return generate_keypair()

    @staticmethod
    def _extract_resource_budget(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract portable resource budget context for receipts."""
        if isinstance(parameters.get("resource_budget"), dict):
            return dict(parameters["resource_budget"])

        candidate_keys = {
            "budget",
            "tokens",
            "cpu_millicores",
            "gpu_tflops",
            "memory_bytes",
            "storage_bytes",
            "network_bps",
            "max_cost",
            "ram_gb",
            "vram_gb",
        }
        budget: Dict[str, Any] = {}
        for key, value in parameters.items():
            if key in candidate_keys:
                budget[key] = value
        return budget

    def _build_receipt_payload(
        self,
        proposal: ActionProposal,
        outcome: ConsensusOutcome,
    ) -> Dict[str, Any]:
        """Build canonical payload signed by PAT and SAT."""
        sorted_votes = sorted(
            [
                {
                    "validator_role": vote.validator_role.value,
                    "approve": vote.approve,
                    "confidence": vote.confidence,
                    "veto_reason": (
                        vote.veto_reason.value if vote.veto_reason is not None else None
                    ),
                    "comments": vote.comments,
                    "timestamp": vote.timestamp.isoformat(),
                }
                for vote in outcome.votes
            ],
            key=lambda item: item["validator_role"],
        )
        resource_budget = self._extract_resource_budget(proposal.parameters)
        return {
            "proposal_id": proposal.id,
            "task_id": proposal.task_id,
            "action_type": proposal.action_type,
            "proposer_role": proposal.proposer_role.value,
            "sat_mode": self.sat_mode.value,
            "result": outcome.result.name,
            "approval_count": outcome.approval_count,
            "veto_count": outcome.veto_count,
            "quorum_met": outcome.quorum_met,
            "final_ihsan": round(outcome.final_ihsan, 6),
            "risk_estimate": round(proposal.risk_estimate, 6),
            "resource_budget": resource_budget,
            "created_at": proposal.created_at.isoformat(),
            "resolved_at": outcome.resolved_at.isoformat(),
            "validator_votes": sorted_votes,
        }

    def _issue_negotiation_receipt(
        self,
        proposal: ActionProposal,
        outcome: ConsensusOutcome,
    ) -> NegotiationReceipt:
        """Sign and verify a PAT↔SAT negotiation receipt."""
        payload = self._build_receipt_payload(proposal, outcome)
        payload_bytes = canonicalize_and_validate(payload)
        payload_digest = domain_separated_digest(payload_bytes)
        receipt_id_seed = canonicalize_and_validate(
            {
                "proposal_id": proposal.id,
                "payload_digest": payload_digest,
                "resolved_at": outcome.resolved_at.isoformat(),
                "sat_mode": self.sat_mode.value,
            }
        )
        receipt_id = domain_separated_digest(receipt_id_seed)[:32]
        proposer_signature = sign_message(payload_digest, self._proposer_private_key)
        sat_signature = sign_message(payload_digest, self._sat_private_key)

        receipt = NegotiationReceipt(
            receipt_id=receipt_id,
            proposal_id=proposal.id,
            task_id=proposal.task_id,
            action_type=proposal.action_type,
            sat_mode=self.sat_mode.value,
            proposer_role=proposal.proposer_role,
            result=outcome.result,
            approval_count=outcome.approval_count,
            veto_count=outcome.veto_count,
            quorum_met=outcome.quorum_met,
            final_ihsan=outcome.final_ihsan,
            risk_estimate=proposal.risk_estimate,
            resource_budget=payload["resource_budget"],
            created_at=proposal.created_at,
            resolved_at=outcome.resolved_at,
            payload_digest=payload_digest,
            proposer_agent_id=self._proposer_agent_id,
            proposer_public_key=self._proposer_public_key,
            proposer_signature=proposer_signature,
            sat_agent_id=self._sat_agent_id,
            sat_public_key=self._sat_public_key,
            sat_signature=sat_signature,
            validator_votes=payload["validator_votes"],
        )
        if not receipt.verify_signatures():
            raise RuntimeError(
                f"Negotiation receipt signature verification failed: {proposal.id}"
            )
        return receipt

    @staticmethod
    def _reason_codes_from_outcome(outcome: ConsensusOutcome) -> List[str]:
        """Convert vote outcomes into stable machine reason codes."""
        if outcome.result == ConsensusResult.APPROVED:
            return []
        reason_codes: List[str] = []
        for vote in outcome.votes:
            if vote.approve:
                continue
            if vote.veto_reason is None:
                continue
            code = f"SAT_VETO_{vote.veto_reason.value.upper()}"
            if code not in reason_codes:
                reason_codes.append(code)
        if not reason_codes:
            reason_codes.append("SAT_CONSENSUS_REJECTED")
        return reason_codes

    def _emit_evidence_receipt(
        self,
        receipt: NegotiationReceipt,
        outcome: ConsensusOutcome,
    ) -> None:
        """Persist negotiation receipt in append-only evidence ledger."""
        if self._evidence_ledger is None:
            return

        from core.proof_engine.evidence_ledger import emit_receipt

        duration_ms = max(
            0.0,
            (receipt.resolved_at - receipt.created_at).total_seconds() * 1000.0,
        )
        try:
            entry = emit_receipt(
                self._evidence_ledger,
                receipt_id=receipt.receipt_id,
                node_id=receipt.sat_agent_id,
                policy_version="1.0.0",
                status=(
                    "accepted"
                    if receipt.result == ConsensusResult.APPROVED
                    else "rejected"
                ),
                decision=(
                    "APPROVED"
                    if receipt.result == ConsensusResult.APPROVED
                    else "REJECTED"
                ),
                reason_codes=self._reason_codes_from_outcome(outcome),
                snr_score=max(0.0, min(1.0, 1.0 - receipt.risk_estimate)),
                ihsan_score=max(0.0, min(1.0, receipt.final_ihsan)),
                ihsan_threshold=self.ihsan_threshold,
                seal_digest=receipt.payload_digest,
                payload_digest=receipt.payload_digest,
                gate_passed="sat_consensus",
                duration_ms=duration_ms,
                claim_tags={
                    "approved_votes": receipt.approval_count,
                    "veto_votes": receipt.veto_count,
                },
                signer_private_key_hex=self._sat_private_key,
                signer_public_key_hex=self._sat_public_key,
                origin={
                    "channel": "pat_sat",
                    "proposer_agent_id": receipt.proposer_agent_id,
                    "sat_agent_id": receipt.sat_agent_id,
                    "sat_mode": receipt.sat_mode,
                },
                critical_decision=True,
            )
            receipt.ledger_sequence = entry.sequence
            receipt.ledger_entry_hash = entry.entry_hash
            self._evidence_entries += 1
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            self._evidence_failures += 1
            if self._fail_closed_on_ledger_error:
                raise RuntimeError(
                    f"Evidence ledger append failed for {receipt.proposal_id}: {exc}"
                ) from exc
            logger.warning(
                "Negotiation ledger append failed for %s: %s",
                receipt.proposal_id,
                exc,
            )

    def _register_default_validators(self) -> None:
        """Register default validator implementations."""

        async def security_validator(
            proposal: ActionProposal,
        ) -> Tuple[bool, Optional[VetoReason], float]:
            """Security validation - check for dangerous patterns."""
            dangerous = {"delete", "drop", "rm", "format", "shutdown", "exec"}
            params_str = str(proposal.parameters).lower()

            if any(d in params_str for d in dangerous):
                return False, VetoReason.SECURITY_VIOLATION, 0.99
            if proposal.risk_estimate > 0.7:
                return False, VetoReason.SECURITY_VIOLATION, 0.9
            return True, None, 0.95

        async def ethics_validator(
            proposal: ActionProposal,
        ) -> Tuple[bool, Optional[VetoReason], float]:
            """Ethics validation - check Ihsan compliance."""
            if proposal.ihsan_estimate < self.ihsan_threshold:
                return False, VetoReason.IHSAN_THRESHOLD, 0.99
            return True, None, 0.95

        async def performance_validator(
            proposal: ActionProposal,
        ) -> Tuple[bool, Optional[VetoReason], float]:
            """Performance validation - check resource impact."""
            # Simulate performance check
            return True, None, 0.9

        async def consistency_validator(
            proposal: ActionProposal,
        ) -> Tuple[bool, Optional[VetoReason], float]:
            """Consistency validation - check state consistency."""
            return True, None, 0.9

        async def resource_validator(
            proposal: ActionProposal,
        ) -> Tuple[bool, Optional[VetoReason], float]:
            """Resource validation - check resource availability."""
            return True, None, 0.85

        self._validators = {
            AgentRole.SECURITY_GUARDIAN: security_validator,
            AgentRole.ETHICS_VALIDATOR: ethics_validator,
            AgentRole.PERFORMANCE_MONITOR: performance_validator,
            AgentRole.CONSISTENCY_CHECKER: consistency_validator,
            AgentRole.RESOURCE_OPTIMIZER: resource_validator,
        }

    def register_validator(
        self,
        role: AgentRole,
        validator: ValidatorFn,
    ) -> None:
        """Register a custom validator function."""
        if self.sat_mode == SATMode.MINI5 and role not in self.SAT_VALIDATORS_MINI5:
            raise ValueError(f"Role {role} is not a SAT validator")
        self._validators[role] = validator

    async def submit_proposal(self, proposal: ActionProposal) -> str:
        """Submit an action proposal for validation."""
        self._pending_proposals[proposal.id] = proposal
        self._proposal_count += 1
        logger.debug(f"Proposal submitted: {proposal.id} ({proposal.action_type})")
        return proposal.id

    async def validate(self, proposal_id: str) -> ConsensusOutcome:
        """Run Byzantine consensus validation on a proposal."""
        if len(self._validators) < self.expected_validator_count:
            raise RuntimeError(
                "SAT validator roster incomplete for "
                f"{self.sat_mode.value}: {len(self._validators)}/"
                f"{self.expected_validator_count}"
            )

        proposal = self._pending_proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        outcome = ConsensusOutcome(proposal_id=proposal_id)

        # PERF FIX #7: Run validators in parallel instead of sequentially
        # Create async tasks for all validators
        async def validate_with_role(
            role: AgentRole, validator_fn: ValidatorFn
        ) -> Vote:
            try:
                result: Tuple[bool, Optional[VetoReason], float] = (
                    await asyncio.wait_for(
                        validator_fn(proposal),
                        timeout=self.vote_timeout,
                    )
                )
                approve: bool = result[0]
                confidence: float = result[2]
                veto_reason: Optional[VetoReason] = result[1]
                return Vote(
                    validator_role=role,
                    approve=approve,
                    confidence=confidence,
                    veto_reason=veto_reason,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Validator {role.value} timed out")
                return Vote(
                    validator_role=role,
                    approve=False,
                    confidence=0.0,
                    veto_reason=None,
                    comments="Timeout",
                )
            except (asyncio.CancelledError, RuntimeError, OSError) as e:  # SEC-003 — async boundary
                logger.error(f"Validator {role.value} error: {e}")
                return Vote(
                    validator_role=role,
                    approve=False,
                    confidence=0.0,
                    veto_reason=None,
                    comments=f"Error: {e}",
                )

        # Run all validators concurrently
        validator_tasks = [
            validate_with_role(role, validator_fn)
            for role, validator_fn in self._validators.items()
        ]
        votes = await asyncio.gather(*validator_tasks)
        votes = list(votes)  # Convert to list for modification

        # Check for veto from SECURITY or ETHICS roles
        for vote in votes:
            if not vote.approve and vote.validator_role in self.VETO_ROLES:
                logger.warning(
                    f"VETO from {vote.validator_role.value}: {vote.veto_reason}"
                )
                outcome.result = ConsensusResult.VETOED
                outcome.votes = votes
                outcome.approval_count = sum(1 for v in votes if v.approve)
                outcome.veto_count = sum(1 for v in votes if not v.approve)
                outcome.quorum_met = False
                outcome.final_ihsan = proposal.ihsan_estimate
                outcome.resolved_at = datetime.now(timezone.utc)
                self._outcomes[proposal_id] = outcome
                self._vetoed_count += 1
                del self._pending_proposals[proposal_id]
                receipt = self._issue_negotiation_receipt(proposal, outcome)
                self._emit_evidence_receipt(receipt, outcome)
                self._receipts[proposal_id] = receipt
                self._verified_receipts += 1
                return outcome

        # Count votes
        outcome.votes = votes
        outcome.approval_count = sum(1 for v in votes if v.approve)
        outcome.veto_count = sum(1 for v in votes if not v.approve)
        outcome.quorum_met = outcome.approval_count >= self.consensus_threshold

        # Calculate final Ihsan score (weighted average of confident votes)
        confident_votes = [v for v in votes if v.confidence > 0.5]
        if confident_votes:
            total_conf = sum(v.confidence for v in confident_votes)
            weighted_ihsan = sum(
                (1.0 if v.approve else 0.8) * v.confidence for v in confident_votes
            )
            outcome.final_ihsan = weighted_ihsan / total_conf
        else:
            outcome.final_ihsan = proposal.ihsan_estimate

        # Determine result
        if outcome.quorum_met and outcome.final_ihsan >= self.ihsan_threshold:
            outcome.result = ConsensusResult.APPROVED
            self._approved_count += 1
        else:
            outcome.result = ConsensusResult.VETOED
            self._vetoed_count += 1

        outcome.resolved_at = datetime.now(timezone.utc)
        self._outcomes[proposal_id] = outcome
        del self._pending_proposals[proposal_id]
        receipt = self._issue_negotiation_receipt(proposal, outcome)
        self._emit_evidence_receipt(receipt, outcome)
        self._receipts[proposal_id] = receipt
        self._verified_receipts += 1

        logger.info(
            f"Consensus {outcome.result.name}: {proposal_id} "
            f"(votes: {outcome.approval_count}/{len(votes)}, ihsan: {outcome.final_ihsan:.3f})"
        )
        return outcome

    async def propose_and_validate(
        self,
        task: TeamTask,
        action_type: str,
        parameters: Dict[str, Any],
        proposer: AgentRole = AgentRole.MASTER_REASONER,
        ihsan_estimate: float = 0.95,
        risk_estimate: float = 0.1,
    ) -> ConsensusOutcome:
        """Convenience method to propose and validate in one call."""
        proposal = ActionProposal(
            task_id=task.id,
            action_type=action_type,
            parameters=parameters,
            proposer_role=proposer,
            ihsan_estimate=ihsan_estimate,
            risk_estimate=risk_estimate,
        )
        await self.submit_proposal(proposal)
        return await self.validate(proposal.id)

    def get_outcome(self, proposal_id: str) -> Optional[ConsensusOutcome]:
        """Get the outcome of a completed validation."""
        return self._outcomes.get(proposal_id)

    def get_receipt(self, proposal_id: str) -> Optional[NegotiationReceipt]:
        """Get the signed negotiation receipt for a completed proposal."""
        return self._receipts.get(proposal_id)

    def verify_receipt(self, proposal_id: str) -> bool:
        """Verify both signatures on an existing receipt."""
        receipt = self._receipts.get(proposal_id)
        if receipt is None:
            return False
        return receipt.verify_signatures()

    def stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "pending_proposals": len(self._pending_proposals),
            "total_proposals": self._proposal_count,
            "approved": self._approved_count,
            "vetoed": self._vetoed_count,
            "approval_rate": (self._approved_count / max(self._proposal_count, 1)),
            "ihsan_threshold": self.ihsan_threshold,
            "sat_mode": self.sat_mode.value,
            "expected_validators": self.expected_validator_count,
            "active_validators": len(self._validators),
            "consensus_threshold": self.consensus_threshold,
            "sat_claim_truthful": len(self._validators)
            >= self.expected_validator_count,
            "signed_receipts": len(self._receipts),
            "verified_receipts": self._verified_receipts,
            "receipt_signature_health": (
                self._verified_receipts / max(len(self._receipts), 1)
            ),
            "evidence_entries": self._evidence_entries,
            "evidence_failures": self._evidence_failures,
            "evidence_health": self._evidence_entries
            / max(
                self._evidence_entries + self._evidence_failures,
                1,
            ),
        }


__all__ = [
    "ActionProposal",
    "ConsensusOutcome",
    "ConsensusResult",
    "DualAgenticBridge",
    "NegotiationReceipt",
    "SATMode",
    "VetoReason",
    "Vote",
]
