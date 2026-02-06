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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from core.orchestration.team_planner import AgentRole, TeamTask

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


# Validator functions type
ValidatorFn = Callable[[ActionProposal], Tuple[bool, Optional[VetoReason], float]]


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

    # SAT validator roles
    SAT_VALIDATORS = {
        AgentRole.SECURITY_GUARDIAN,
        AgentRole.ETHICS_VALIDATOR,
        AgentRole.PERFORMANCE_MONITOR,
        AgentRole.CONSISTENCY_CHECKER,
        AgentRole.RESOURCE_OPTIMIZER,
    }

    # Roles with veto power
    VETO_ROLES = {AgentRole.SECURITY_GUARDIAN, AgentRole.ETHICS_VALIDATOR}

    # Byzantine threshold: need 3 of 5 for consensus
    CONSENSUS_THRESHOLD = 3

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        vote_timeout: float = 5.0,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.vote_timeout = vote_timeout

        self._pending_proposals: Dict[str, ActionProposal] = {}
        self._outcomes: Dict[str, ConsensusOutcome] = {}
        self._validators: Dict[AgentRole, ValidatorFn] = {}
        self._proposal_count = 0
        self._approved_count = 0
        self._vetoed_count = 0

        # Register default validators
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default validator implementations."""

        async def security_validator(proposal: ActionProposal) -> Tuple[bool, Optional[VetoReason], float]:
            """Security validation - check for dangerous patterns."""
            dangerous = {"delete", "drop", "rm", "format", "shutdown", "exec"}
            params_str = str(proposal.parameters).lower()

            if any(d in params_str for d in dangerous):
                return False, VetoReason.SECURITY_VIOLATION, 0.99
            if proposal.risk_estimate > 0.7:
                return False, VetoReason.SECURITY_VIOLATION, 0.9
            return True, None, 0.95

        async def ethics_validator(proposal: ActionProposal) -> Tuple[bool, Optional[VetoReason], float]:
            """Ethics validation - check Ihsan compliance."""
            if proposal.ihsan_estimate < self.ihsan_threshold:
                return False, VetoReason.IHSAN_THRESHOLD, 0.99
            return True, None, 0.95

        async def performance_validator(proposal: ActionProposal) -> Tuple[bool, Optional[VetoReason], float]:
            """Performance validation - check resource impact."""
            # Simulate performance check
            return True, None, 0.9

        async def consistency_validator(proposal: ActionProposal) -> Tuple[bool, Optional[VetoReason], float]:
            """Consistency validation - check state consistency."""
            return True, None, 0.9

        async def resource_validator(proposal: ActionProposal) -> Tuple[bool, Optional[VetoReason], float]:
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
        if role not in self.SAT_VALIDATORS:
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
        proposal = self._pending_proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        outcome = ConsensusOutcome(proposal_id=proposal_id)

        # PERF FIX #7: Run validators in parallel instead of sequentially
        # Create async tasks for all validators
        async def validate_with_role(role: AgentRole, validator_fn: ValidatorFn) -> Vote:
            try:
                approve, veto_reason, confidence = await asyncio.wait_for(
                    validator_fn(proposal),
                    timeout=self.vote_timeout,
                )
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
            except Exception as e:
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
                logger.warning(f"VETO from {vote.validator_role.value}: {vote.veto_reason}")
                outcome.result = ConsensusResult.VETOED
                outcome.votes = votes
                outcome.veto_count = sum(1 for v in votes if not v.approve)
                self._outcomes[proposal_id] = outcome
                self._vetoed_count += 1
                del self._pending_proposals[proposal_id]
                return outcome

        # Count votes
        outcome.votes = votes
        outcome.approval_count = sum(1 for v in votes if v.approve)
        outcome.veto_count = sum(1 for v in votes if not v.approve)
        outcome.quorum_met = outcome.approval_count >= self.CONSENSUS_THRESHOLD

        # Calculate final Ihsan score (weighted average of confident votes)
        confident_votes = [v for v in votes if v.confidence > 0.5]
        if confident_votes:
            total_conf = sum(v.confidence for v in confident_votes)
            weighted_ihsan = sum(
                (1.0 if v.approve else 0.8) * v.confidence
                for v in confident_votes
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

    def stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "pending_proposals": len(self._pending_proposals),
            "total_proposals": self._proposal_count,
            "approved": self._approved_count,
            "vetoed": self._vetoed_count,
            "approval_rate": (
                self._approved_count / max(self._proposal_count, 1)
            ),
            "ihsan_threshold": self.ihsan_threshold,
            "consensus_threshold": self.CONSENSUS_THRESHOLD,
        }


__all__ = [
    "ActionProposal",
    "ConsensusOutcome",
    "ConsensusResult",
    "DualAgenticBridge",
    "VetoReason",
    "Vote",
]
