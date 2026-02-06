"""
Guardian Council — Multi-Agent Consensus for Sovereign Decisions

Standing on the Shoulders of:
- DDAGI Constitution v1.1.0 (Ihsān Constraint Framework)
- Byzantine Fault Tolerance (Lamport et al., 1982)
- Weighted Voting Systems (Shapley-Shubik Index)
- Ensemble Methods (Breiman, 1996)

The Council ensures no single agent can compromise the system.
Every significant decision requires multi-guardian consensus.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
from datetime import datetime
import asyncio
import hashlib
import math


class GuardianRole(Enum):
    """The Eight Guardians of BIZRA — Each with specific domain expertise."""
    ARCHITECT = auto()      # System design, coherence, structure
    SECURITY = auto()       # Safety, adversarial detection, boundaries
    ETHICS = auto()         # Ihsān compliance, beneficence, fairness
    REASONING = auto()      # Logic validation, contradiction detection
    KNOWLEDGE = auto()      # Factual grounding, citation, provenance
    CREATIVE = auto()       # Novel solutions, lateral thinking
    INTEGRATION = auto()    # Cross-domain synthesis, compatibility
    NUCLEUS = auto()        # Final arbiter, tie-breaker, meta-oversight


class VoteType(Enum):
    """Types of votes a Guardian can cast."""
    APPROVE = auto()        # Fully endorses the proposal
    APPROVE_WITH_CONCERNS = auto()  # Endorses with noted reservations
    ABSTAIN = auto()        # Neither approves nor rejects
    REJECT_SOFT = auto()    # Rejects but allows override
    REJECT_HARD = auto()    # Absolute veto — blocks proposal


class ConsensusMode(Enum):
    """Consensus protocols for different decision types."""
    UNANIMOUS = auto()      # All must approve (critical decisions)
    SUPERMAJORITY = auto()  # 2/3 must approve (important decisions)
    MAJORITY = auto()       # >50% must approve (standard decisions)
    WEIGHTED = auto()       # Weighted by expertise (domain-specific)
    NUCLEUS_OVERRIDE = auto()  # Nucleus can decide alone (emergencies)


@dataclass
class IhsanVector:
    """
    Five-dimensional quality vector for Ihsān (excellence) measurement.
    Each dimension in [0, 1], combined score must exceed threshold.
    """
    correctness: float = 0.0    # Factual accuracy, logical validity
    safety: float = 0.0         # Harm prevention, boundary respect
    beneficence: float = 0.0    # Positive impact, helpfulness
    transparency: float = 0.0   # Explainability, auditability
    sustainability: float = 0.0 # Long-term viability, resource efficiency

    def score(self, weights: Optional[dict[str, float]] = None) -> float:
        """Compute weighted Ihsān score."""
        if weights is None:
            weights = {
                "correctness": 0.25,
                "safety": 0.25,
                "beneficence": 0.20,
                "transparency": 0.15,
                "sustainability": 0.15,
            }

        return (
            self.correctness * weights.get("correctness", 0.2) +
            self.safety * weights.get("safety", 0.2) +
            self.beneficence * weights.get("beneficence", 0.2) +
            self.transparency * weights.get("transparency", 0.2) +
            self.sustainability * weights.get("sustainability", 0.2)
        )

    def passes_gate(self, threshold: float = 0.95) -> bool:
        """Check if vector passes Ihsān gate."""
        return self.score() >= threshold and min(
            self.correctness, self.safety, self.beneficence,
            self.transparency, self.sustainability
        ) >= 0.7  # No dimension can be too weak


@dataclass
class GuardianVote:
    """A single Guardian's vote on a proposal."""
    guardian: GuardianRole
    vote_type: VoteType
    confidence: float  # [0, 1] — how certain the guardian is
    reasoning: str
    ihsan_assessment: IhsanVector
    timestamp: datetime = field(default_factory=datetime.now)
    signature: str = ""  # Cryptographic signature for audit

    def __post_init__(self):
        # Generate vote signature for tamper detection
        vote_data = f"{self.guardian.name}:{self.vote_type.name}:{self.confidence}:{self.timestamp.isoformat()}"
        self.signature = hashlib.sha256(vote_data.encode()).hexdigest()[:16]

    @property
    def numeric_value(self) -> float:
        """Convert vote to numeric value for aggregation."""
        base_values = {
            VoteType.APPROVE: 1.0,
            VoteType.APPROVE_WITH_CONCERNS: 0.7,
            VoteType.ABSTAIN: 0.0,
            VoteType.REJECT_SOFT: -0.5,
            VoteType.REJECT_HARD: -1.0,
        }
        return base_values[self.vote_type] * self.confidence


@dataclass
class Proposal:
    """A proposal submitted to the Guardian Council for review."""
    id: str
    title: str
    content: Any
    proposer: str
    context: dict[str, Any] = field(default_factory=dict)
    required_mode: ConsensusMode = ConsensusMode.MAJORITY
    urgency: float = 0.5  # [0, 1] — affects timeout
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CouncilVerdict:
    """The final verdict from the Guardian Council."""
    proposal_id: str
    approved: bool
    consensus_mode: ConsensusMode
    votes: list[GuardianVote]
    aggregate_score: float
    ihsan_passed: bool
    dissenting_opinions: list[str]
    recommendations: list[str]
    deliberation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def vote_summary(self) -> dict[VoteType, int]:
        """Count votes by type."""
        summary = {vt: 0 for vt in VoteType}
        for vote in self.votes:
            summary[vote.vote_type] += 1
        return summary

    @property
    def unanimous(self) -> bool:
        """Check if verdict was unanimous."""
        approvals = sum(1 for v in self.votes if v.vote_type in
                       [VoteType.APPROVE, VoteType.APPROVE_WITH_CONCERNS])
        return approvals == len(self.votes)


class Guardian:
    """
    A single Guardian agent with domain expertise.

    Each Guardian evaluates proposals through its specialized lens
    and casts a vote with reasoning.
    """

    # Domain-specific weights for Ihsān evaluation
    DOMAIN_WEIGHTS = {
        GuardianRole.ARCHITECT: {"correctness": 0.3, "transparency": 0.25, "sustainability": 0.25, "safety": 0.1, "beneficence": 0.1},
        GuardianRole.SECURITY: {"safety": 0.4, "correctness": 0.25, "transparency": 0.2, "beneficence": 0.1, "sustainability": 0.05},
        GuardianRole.ETHICS: {"beneficence": 0.35, "safety": 0.25, "transparency": 0.2, "correctness": 0.1, "sustainability": 0.1},
        GuardianRole.REASONING: {"correctness": 0.4, "transparency": 0.25, "beneficence": 0.15, "safety": 0.1, "sustainability": 0.1},
        GuardianRole.KNOWLEDGE: {"correctness": 0.35, "transparency": 0.3, "beneficence": 0.15, "safety": 0.1, "sustainability": 0.1},
        GuardianRole.CREATIVE: {"beneficence": 0.3, "sustainability": 0.25, "correctness": 0.2, "transparency": 0.15, "safety": 0.1},
        GuardianRole.INTEGRATION: {"sustainability": 0.3, "correctness": 0.25, "beneficence": 0.2, "transparency": 0.15, "safety": 0.1},
        GuardianRole.NUCLEUS: {"safety": 0.25, "correctness": 0.2, "beneficence": 0.2, "transparency": 0.2, "sustainability": 0.15},
    }

    def __init__(
        self,
        role: GuardianRole,
        evaluate_fn: Optional[Callable[[Proposal], IhsanVector]] = None,
    ):
        self.role = role
        self.evaluate_fn = evaluate_fn or self._default_evaluate
        self.vote_history: list[GuardianVote] = []

    def _default_evaluate(self, proposal: Proposal) -> IhsanVector:
        """Default evaluation — override with LLM-based evaluation in production."""
        # Placeholder scores — in real implementation, this calls an LLM
        return IhsanVector(
            correctness=0.85,
            safety=0.90,
            beneficence=0.88,
            transparency=0.82,
            sustainability=0.80,
        )

    async def evaluate(self, proposal: Proposal) -> GuardianVote:
        """Evaluate a proposal and cast a vote."""
        # Get domain-specific Ihsān assessment
        ihsan = self.evaluate_fn(proposal)

        # Calculate weighted score using domain expertise
        weights = self.DOMAIN_WEIGHTS[self.role]
        score = ihsan.score(weights)

        # Determine vote type based on score
        if score >= 0.95:
            vote_type = VoteType.APPROVE
            confidence = min(1.0, score)
        elif score >= 0.85:
            vote_type = VoteType.APPROVE_WITH_CONCERNS
            confidence = score
        elif score >= 0.70:
            vote_type = VoteType.ABSTAIN
            confidence = 0.5
        elif score >= 0.50:
            vote_type = VoteType.REJECT_SOFT
            confidence = 1.0 - score
        else:
            vote_type = VoteType.REJECT_HARD
            confidence = 1.0 - score

        # Generate reasoning
        reasoning = self._generate_reasoning(proposal, ihsan, score)

        vote = GuardianVote(
            guardian=self.role,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=reasoning,
            ihsan_assessment=ihsan,
        )

        self.vote_history.append(vote)
        return vote

    def _generate_reasoning(
        self,
        proposal: Proposal,
        ihsan: IhsanVector,
        score: float
    ) -> str:
        """Generate reasoning for the vote."""
        weakest = min(
            ("correctness", ihsan.correctness),
            ("safety", ihsan.safety),
            ("beneficence", ihsan.beneficence),
            ("transparency", ihsan.transparency),
            ("sustainability", ihsan.sustainability),
            key=lambda x: x[1]
        )

        return (
            f"[{self.role.name}] Score: {score:.3f} | "
            f"Weakest dimension: {weakest[0]} ({weakest[1]:.2f})"
        )


class GuardianCouncil:
    """
    The Guardian Council — Multi-agent consensus system.

    Ensures sovereign decisions pass through collective wisdom
    with Byzantine fault tolerance and Ihsān constraints.
    """

    # Voting power weights (Shapley-Shubik inspired)
    VOTING_POWER = {
        GuardianRole.NUCLEUS: 1.5,      # Tie-breaker authority
        GuardianRole.SECURITY: 1.3,     # Safety-critical weight
        GuardianRole.ETHICS: 1.3,       # Ethical oversight weight
        GuardianRole.REASONING: 1.1,    # Logic validation
        GuardianRole.KNOWLEDGE: 1.0,    # Standard weight
        GuardianRole.ARCHITECT: 1.0,
        GuardianRole.CREATIVE: 0.9,
        GuardianRole.INTEGRATION: 0.9,
    }

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        enable_veto: bool = True,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.enable_veto = enable_veto
        self.guardians: dict[GuardianRole, Guardian] = {}
        self.verdicts: list[CouncilVerdict] = []

        # Initialize all guardians
        for role in GuardianRole:
            self.guardians[role] = Guardian(role)

    def set_guardian_evaluator(
        self,
        role: GuardianRole,
        evaluate_fn: Callable[[Proposal], IhsanVector]
    ):
        """Set custom evaluation function for a guardian."""
        self.guardians[role].evaluate_fn = evaluate_fn

    async def deliberate(
        self,
        proposal: Proposal,
        timeout_seconds: float = 30.0,
    ) -> CouncilVerdict:
        """
        Convene the council to deliberate on a proposal.

        Returns a CouncilVerdict with the collective decision.
        """
        start_time = datetime.now()

        # Adjust timeout based on urgency
        adjusted_timeout = timeout_seconds * (2.0 - proposal.urgency)

        # Gather votes from all guardians concurrently
        vote_tasks = [
            asyncio.wait_for(
                guardian.evaluate(proposal),
                timeout=adjusted_timeout
            )
            for guardian in self.guardians.values()
        ]

        try:
            votes = await asyncio.gather(*vote_tasks, return_exceptions=True)
            # Filter out exceptions
            valid_votes = [v for v in votes if isinstance(v, GuardianVote)]
        except asyncio.TimeoutError:
            valid_votes = []

        # Check for hard vetoes
        hard_vetoes = [v for v in valid_votes if v.vote_type == VoteType.REJECT_HARD]
        if self.enable_veto and hard_vetoes:
            # Any hard veto blocks the proposal
            verdict = self._create_veto_verdict(proposal, valid_votes, hard_vetoes)
        else:
            # Calculate consensus based on mode
            verdict = self._calculate_consensus(proposal, valid_votes)

        # Record deliberation time
        end_time = datetime.now()
        verdict.deliberation_time_ms = (end_time - start_time).total_seconds() * 1000

        self.verdicts.append(verdict)
        return verdict

    def _calculate_consensus(
        self,
        proposal: Proposal,
        votes: list[GuardianVote]
    ) -> CouncilVerdict:
        """Calculate consensus based on the required mode."""
        mode = proposal.required_mode

        # Calculate weighted aggregate score
        total_weight = 0.0
        weighted_sum = 0.0

        for vote in votes:
            weight = self.VOTING_POWER.get(vote.guardian, 1.0)
            weighted_sum += vote.numeric_value * weight
            total_weight += weight

        aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine approval based on consensus mode
        if mode == ConsensusMode.UNANIMOUS:
            approved = all(
                v.vote_type in [VoteType.APPROVE, VoteType.APPROVE_WITH_CONCERNS]
                for v in votes
            )
        elif mode == ConsensusMode.SUPERMAJORITY:
            approval_count = sum(
                1 for v in votes
                if v.vote_type in [VoteType.APPROVE, VoteType.APPROVE_WITH_CONCERNS]
            )
            approved = approval_count >= len(votes) * (2/3)
        elif mode == ConsensusMode.MAJORITY:
            approved = aggregate_score > 0
        elif mode == ConsensusMode.WEIGHTED:
            approved = aggregate_score >= 0.5
        elif mode == ConsensusMode.NUCLEUS_OVERRIDE:
            nucleus_vote = next(
                (v for v in votes if v.guardian == GuardianRole.NUCLEUS),
                None
            )
            approved = nucleus_vote and nucleus_vote.vote_type in [
                VoteType.APPROVE, VoteType.APPROVE_WITH_CONCERNS
            ]
        else:
            approved = aggregate_score > 0

        # Check Ihsān compliance
        combined_ihsan = self._combine_ihsan_vectors(votes)
        ihsan_passed = combined_ihsan.passes_gate(self.ihsan_threshold)

        # If Ihsān fails, override approval
        if not ihsan_passed:
            approved = False

        # Collect dissenting opinions
        dissenting = [
            v.reasoning for v in votes
            if v.vote_type in [VoteType.REJECT_SOFT, VoteType.REJECT_HARD]
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(votes, aggregate_score)

        return CouncilVerdict(
            proposal_id=proposal.id,
            approved=approved,
            consensus_mode=mode,
            votes=votes,
            aggregate_score=aggregate_score,
            ihsan_passed=ihsan_passed,
            dissenting_opinions=dissenting,
            recommendations=recommendations,
        )

    def _create_veto_verdict(
        self,
        proposal: Proposal,
        votes: list[GuardianVote],
        vetoes: list[GuardianVote]
    ) -> CouncilVerdict:
        """Create a verdict when a hard veto has been cast."""
        veto_reasons = [v.reasoning for v in vetoes]

        return CouncilVerdict(
            proposal_id=proposal.id,
            approved=False,
            consensus_mode=proposal.required_mode,
            votes=votes,
            aggregate_score=-1.0,
            ihsan_passed=False,
            dissenting_opinions=veto_reasons,
            recommendations=[
                f"VETO by {v.guardian.name}: Address concerns before resubmission"
                for v in vetoes
            ],
        )

    def _combine_ihsan_vectors(self, votes: list[GuardianVote]) -> IhsanVector:
        """Combine Ihsān vectors from all votes using weighted average."""
        if not votes:
            return IhsanVector()

        total_weight = 0.0
        combined = IhsanVector()

        for vote in votes:
            weight = self.VOTING_POWER.get(vote.guardian, 1.0) * vote.confidence
            combined.correctness += vote.ihsan_assessment.correctness * weight
            combined.safety += vote.ihsan_assessment.safety * weight
            combined.beneficence += vote.ihsan_assessment.beneficence * weight
            combined.transparency += vote.ihsan_assessment.transparency * weight
            combined.sustainability += vote.ihsan_assessment.sustainability * weight
            total_weight += weight

        if total_weight > 0:
            combined.correctness /= total_weight
            combined.safety /= total_weight
            combined.beneficence /= total_weight
            combined.transparency /= total_weight
            combined.sustainability /= total_weight

        return combined

    def _generate_recommendations(
        self,
        votes: list[GuardianVote],
        score: float
    ) -> list[str]:
        """Generate actionable recommendations based on votes."""
        recommendations = []

        # Find weakest dimensions across all assessments
        dimension_scores = {
            "correctness": [],
            "safety": [],
            "beneficence": [],
            "transparency": [],
            "sustainability": [],
        }

        for vote in votes:
            dimension_scores["correctness"].append(vote.ihsan_assessment.correctness)
            dimension_scores["safety"].append(vote.ihsan_assessment.safety)
            dimension_scores["beneficence"].append(vote.ihsan_assessment.beneficence)
            dimension_scores["transparency"].append(vote.ihsan_assessment.transparency)
            dimension_scores["sustainability"].append(vote.ihsan_assessment.sustainability)

        # Identify dimensions needing improvement
        for dim, scores in dimension_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            if avg < 0.8:
                recommendations.append(f"Improve {dim}: current average {avg:.2f}")

        return recommendations


# Convenience function for quick council creation
def create_council(
    ihsan_threshold: float = 0.95,
    enable_veto: bool = True,
) -> GuardianCouncil:
    """Create a fully initialized Guardian Council."""
    return GuardianCouncil(
        ihsan_threshold=ihsan_threshold,
        enable_veto=enable_veto,
    )


# =============================================================================
# RUNTIME API EXTENSION
# =============================================================================

# Extend GuardianCouncil with validate() method for SovereignRuntime API
async def _guardian_council_validate(
    self: GuardianCouncil,
    content: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate content against Guardian Council consensus.

    This is the SovereignRuntime API wrapper around deliberate().
    It creates a Proposal from the content/context and returns
    a simplified validation result dict.

    Args:
        content: The content to validate
        context: Context dict with metadata

    Returns:
        Dict containing:
        - approved: bool - whether content passed validation
        - votes: dict - vote counts by type
        - consensus_score: float - aggregate consensus score
        - verdict: str - APPROVED/REJECTED/VETOED
        - ihsan_score: float - Ihsan compliance score
        - recommendations: List[str] - improvement suggestions
    """
    # Create a Proposal from the content
    proposal = Proposal(
        id=hashlib.sha256(content.encode()).hexdigest()[:12],
        title=f"Validation: {content[:50]}...",
        content=content,
        proposer="sovereign_runtime",
        context=context,
        required_mode=ConsensusMode.MAJORITY,
        urgency=0.7,  # Default urgency for validation
    )

    # Deliberate on the proposal
    verdict = await self.deliberate(proposal, timeout_seconds=10.0)

    # Convert vote summary to simple dict
    vote_counts = {
        "approve": sum(1 for v in verdict.votes if v.vote_type in [VoteType.APPROVE, VoteType.APPROVE_WITH_CONCERNS]),
        "reject": sum(1 for v in verdict.votes if v.vote_type in [VoteType.REJECT_SOFT, VoteType.REJECT_HARD]),
        "abstain": sum(1 for v in verdict.votes if v.vote_type == VoteType.ABSTAIN),
    }

    # Determine verdict string
    if verdict.aggregate_score <= -1.0:
        verdict_str = "VETOED"
    elif verdict.approved:
        verdict_str = "APPROVED"
    else:
        verdict_str = "REJECTED"

    return {
        "approved": verdict.approved,
        "votes": vote_counts,
        "consensus_score": max(verdict.aggregate_score, 0.0),  # Normalize to 0-1
        "verdict": verdict_str,
        "ihsan_score": 1.0 if verdict.ihsan_passed else max(verdict.aggregate_score * 0.8, 0.0),
        "recommendations": verdict.recommendations,
        "dissenting_opinions": verdict.dissenting_opinions,
        "deliberation_time_ms": verdict.deliberation_time_ms,
    }

# Monkey-patch the validate method onto GuardianCouncil
GuardianCouncil.validate = _guardian_council_validate
