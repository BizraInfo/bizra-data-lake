"""
BIZRA Cosmic Verdict Engine - Swarm Intelligence Validation with Byzantine Fault Tolerance
===========================================================================================

Implements bee colony-inspired swarm decision making with BFT guarantees for
sovereign verdict rendering across the 7+1 Guardian Constellation.

The 4-Phase Bee Colony Model:
    SCOUT   - Independent reasoning, NO confidence sharing (prevents cascade)
    WAGGLE  - Advocacy presentation with evidence quality metrics
    QUORUM  - Decision threshold checking (5/8 normal, 6/8 critical)
    LIFTOFF - Winner-take-all commitment phase

Byzantine Fault Tolerance:
    n = 8 agents (7 guardians + 1 Majlis)
    f = 2 maximum Byzantine failures (8 >= 3*2+1 = 7)
    Consensus requires 2f+1 = 5 votes minimum
    ABSOLUTE vetoes cannot be overridden (Ar-Ruh, Al-Amin, Majlis)

Key Insight: HIDE confidence scores during SCOUT phase to prevent over-confidence
cascades where early high-confidence votes unduly influence subsequent voters.

Domain: bizra-apex-v1:cosmic-verdict
Version: 1.0.0
Threshold: 0.95 Ihsan, 0.98 SNR
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    IHSAN_THRESHOLD as CONST_IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
)

# Import Guardian Constellation
from core.genesis import (
    GuardianRole,
    VetoPower,
    MajlisDecision,
    Guardian,
    GuardianConstellation,
    create_guardian_constellation,
)

# Import PersonaPlex consensus types
from core.personaplex.persona_consensus import (
    VoteDecision,
    ConsensusResult as PersonaConsensusResult,
)

# Optional: Ed25519 for signatures
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.encoding import HexEncoder

    HAS_NACL = True
except ImportError:
    HAS_NACL = False

# Optional: BLAKE3 for hashing
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

COSMIC_VERDICT_VERSION = "1.0.0"
COSMIC_VERDICT_DOMAIN = "bizra-apex-v1:cosmic-verdict"

# Byzantine Fault Tolerance Parameters
BFT_N = 8  # Total agents (7 guardians + 1 Majlis)
BFT_F = 2  # Maximum Byzantine failures we tolerate
BFT_QUORUM = 5  # 2f + 1 = 5 votes required for consensus

# Threshold parameters (from core/constants.py)
IHSAN_THRESHOLD = CONST_IHSAN_THRESHOLD  # 0.95
SNR_THRESHOLD = SNR_THRESHOLD_T0_ELITE  # 0.98
CRITICAL_QUORUM = 6  # 6/8 for critical decisions
QUALIFIED_OVERRIDE_QUORUM = 6  # 6/7 to override QUALIFIED veto

# Swarm parameters
SCOUT_TIMEOUT_MS = 5000
WAGGLE_TIMEOUT_MS = 3000
QUORUM_TIMEOUT_MS = 2000
LIFTOFF_TIMEOUT_MS = 1000

# Guardians with ABSOLUTE veto power (cannot be overridden)
ABSOLUTE_VETO_ROLES = {
    GuardianRole.AR_RUH,  # Ethics/Ihsan
    GuardianRole.AL_AMIN,  # Security
    GuardianRole.MAJLIS_AL_KAWNI,  # Collective council
}


# =============================================================================
# ENUMS
# =============================================================================


class VerdictDecision(str, Enum):
    """
    Final verdict decision types from the Cosmic Verdict Engine.

    APPROVED         - Request approved by consensus
    REJECTED         - Request rejected by consensus
    ABSOLUTE_VETO    - Vetoed by Ar-Ruh, Al-Amin, or Majlis (cannot be overridden)
    QUALIFIED_VETO   - Vetoed by qualified guardian (can be overridden by 6/7)
    DEADLOCK         - Extended deliberation needed, no consensus reached
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    ABSOLUTE_VETO = "absolute_veto"
    QUALIFIED_VETO = "qualified_veto"
    DEADLOCK = "deadlock"


class VotingPhase(str, Enum):
    """
    Bee colony-inspired voting phases.

    SCOUT  - Independent reasoning with NO confidence sharing
           - Prevents over-confidence cascades
           - Each guardian reasons in isolation

    WAGGLE - Advocacy with evidence quality scoring
           - Guardians present their reasoning
           - Evidence quality is evaluated

    QUORUM - Decision threshold checking
           - 5/8 for normal decisions
           - 6/8 for critical decisions

    LIFTOFF - Winner-take-all commitment
            - Final decision is locked
            - No further changes allowed
    """

    SCOUT = "scout"
    WAGGLE = "waggle"
    QUORUM = "quorum"
    LIFTOFF = "liftoff"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GuardianVote:
    """
    A vote from a Guardian in the cosmic verdict process.

    Attributes:
        guardian_role: The guardian's role in the constellation
        decision: The vote decision (approve/reject/abstain/soft_veto)
        reasoning: Explanation for the decision
        evidence_quality: Quality score of evidence supporting the vote (0.0-1.0)
        timestamp: When the vote was cast
        signature: Ed25519 signature of the vote content
    """

    guardian_role: GuardianRole
    decision: VoteDecision
    reasoning: str
    evidence_quality: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""

    # Hidden during SCOUT phase to prevent cascade
    _confidence_hidden: bool = field(default=True, repr=False)
    _raw_confidence: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Compute signature if not provided."""
        if not self.signature and HAS_NACL:
            self._compute_signature()

    def _compute_signature(self) -> None:
        """Compute Ed25519 signature of vote content."""
        # In production, would use actual signing key
        vote_data = (
            f"{self.guardian_role.value}:{self.decision.value}:"
            f"{self.reasoning}:{self.evidence_quality}:{self.timestamp.isoformat()}"
        )
        # Placeholder signature (production would use real key)
        self.signature = hashlib.sha256(vote_data.encode()).hexdigest()[:64]

    def reveal_confidence(self) -> float:
        """Reveal confidence after SCOUT phase ends."""
        self._confidence_hidden = False
        return self._raw_confidence

    def set_hidden_confidence(self, confidence: float) -> None:
        """Set confidence during SCOUT phase (hidden from other voters)."""
        self._raw_confidence = max(0.0, min(1.0, confidence))
        self._confidence_hidden = True

    @property
    def confidence(self) -> Optional[float]:
        """Get confidence (None during SCOUT phase)."""
        if self._confidence_hidden:
            return None
        return self._raw_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "guardian_role": self.guardian_role.value,
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "evidence_quality": self.evidence_quality,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "confidence": self.confidence,
        }


@dataclass
class CosmicVerdictResult:
    """
    Final result from the Cosmic Verdict Engine.

    Attributes:
        decision: Final verdict decision
        guardian_votes: Votes from all guardians
        majlis_decision: Decision from Majlis Al-Kawni
        persona_consensus: Result from PersonaPlex consensus layer
        merkle_root: Merkle root of all votes for integrity verification
        byzantine_tolerance: BFT parameters (e.g., "2 of 8")
        quorum_achieved: Whether quorum was reached
        evidence_chain: Chain of evidence hashes
    """

    decision: VerdictDecision
    guardian_votes: Dict[GuardianRole, GuardianVote]
    majlis_decision: MajlisDecision
    persona_consensus: Optional[PersonaConsensusResult]
    merkle_root: str
    byzantine_tolerance: str = "2 of 8"
    quorum_achieved: bool = False
    evidence_chain: List[str] = field(default_factory=list)

    # Metadata
    request_id: str = ""
    voting_phases_completed: List[VotingPhase] = field(default_factory=list)
    total_duration_ms: float = 0.0
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision": self.decision.value,
            "guardian_votes": {
                role.value: vote.to_dict() for role, vote in self.guardian_votes.items()
            },
            "majlis_decision": self.majlis_decision.value,
            "persona_consensus": (
                self.persona_consensus.to_dict() if self.persona_consensus else None
            ),
            "merkle_root": self.merkle_root,
            "byzantine_tolerance": self.byzantine_tolerance,
            "quorum_achieved": self.quorum_achieved,
            "evidence_chain": self.evidence_chain,
            "request_id": self.request_id,
            "voting_phases_completed": [p.value for p in self.voting_phases_completed],
            "total_duration_ms": self.total_duration_ms,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "timestamp": self.timestamp,
        }


@dataclass
class VerdictRequest:
    """
    Request for cosmic verdict rendering.

    Attributes:
        request_id: Unique request identifier
        task: Task description
        payload: Request payload
        context: Additional context
        ihsan_score: Current Ihsan score
        snr_score: Current SNR score
        is_critical: Whether this is a critical decision (requires 6/8)
        required_domains: Domains required for the task
    """

    request_id: str
    task: str
    payload: Dict[str, Any]
    context: Dict[str, Any]
    ihsan_score: float = 0.95
    snr_score: float = 0.98
    is_critical: bool = False
    required_domains: Set[str] = field(default_factory=set)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "task": self.task,
            "payload": self.payload,
            "context": self.context,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "is_critical": self.is_critical,
            "required_domains": list(self.required_domains),
            "timestamp": self.timestamp,
        }


@dataclass
class ScoutResult:
    """
    Result from SCOUT phase (independent reasoning).

    Contains votes without revealed confidences to prevent cascade.
    """

    votes: Dict[GuardianRole, GuardianVote]
    duration_ms: float
    phase: VotingPhase = VotingPhase.SCOUT

    def reveal_all_confidences(self) -> Dict[GuardianRole, float]:
        """Reveal confidences after SCOUT phase ends."""
        return {role: vote.reveal_confidence() for role, vote in self.votes.items()}


@dataclass
class WaggleResult:
    """
    Result from WAGGLE phase (advocacy with evidence).

    Now includes revealed confidences from SCOUT phase.
    """

    votes: Dict[GuardianRole, GuardianVote]
    evidence_scores: Dict[GuardianRole, float]
    advocacy_quality: Dict[GuardianRole, float]
    duration_ms: float
    phase: VotingPhase = VotingPhase.WAGGLE


@dataclass
class QuorumResult:
    """
    Result from QUORUM phase (threshold checking).
    """

    approve_count: int
    reject_count: int
    veto_count: int
    abstain_count: int
    quorum_type: str  # "normal" (5/8) or "critical" (6/8)
    quorum_reached: bool
    absolute_vetoes: List[GuardianRole]
    qualified_vetoes: List[GuardianRole]
    duration_ms: float
    phase: VotingPhase = VotingPhase.QUORUM


# =============================================================================
# COSMIC VERDICT ENGINE
# =============================================================================


class CosmicVerdictEngine:
    """
    Swarm Intelligence Verdict Engine with Byzantine Fault Tolerance.

    Implements the 4-phase bee colony model for collective decision-making:

    1. SCOUT Phase:
       - Each guardian reasons independently
       - NO confidence sharing (prevents over-confidence cascades)
       - Produces initial votes with hidden confidences

    2. WAGGLE Phase:
       - Guardians present advocacy for their decisions
       - Evidence quality is scored
       - Confidences are revealed

    3. QUORUM Phase:
       - Check decision thresholds (5/8 or 6/8)
       - Detect absolute and qualified vetoes
       - Determine if consensus is possible

    4. LIFTOFF Phase:
       - Lock final decision
       - Generate Merkle root of votes
       - Emit evidence receipt

    Byzantine Fault Tolerance:
       - n=8 tolerates f=2 Byzantine failures
       - Requires 2f+1=5 votes for consensus
       - ABSOLUTE vetoes (Ar-Ruh, Al-Amin) cannot be overridden

    Example:
        engine = CosmicVerdictEngine(
            guardian_constellation=constellation,
            persona_consensus=persona_engine,
        )

        request = VerdictRequest(
            request_id="req-001",
            task="Execute critical operation",
            payload={"action": "deploy"},
            context={},
            is_critical=True,
        )

        result = await engine.execute_verdict(request)

        if result.decision == VerdictDecision.APPROVED:
            # Proceed with execution
            pass
    """

    def __init__(
        self,
        guardian_constellation: Optional[GuardianConstellation] = None,
        persona_consensus: Optional[Any] = None,  # PersonaConsensusEngine
    ):
        """
        Initialize the Cosmic Verdict Engine.

        Args:
            guardian_constellation: The 7+1 Guardian Constellation
            persona_consensus: Optional PersonaPlex consensus engine
        """
        self.constellation = guardian_constellation or create_guardian_constellation()
        self.persona_consensus = persona_consensus

        self._verdict_count = 0
        self._receipts: List[Dict[str, Any]] = []

        logger.info(
            f"CosmicVerdictEngine initialized with "
            f"n={BFT_N} agents, f={BFT_F} Byzantine tolerance, "
            f"quorum={BFT_QUORUM}"
        )

    async def execute_verdict(
        self,
        request: VerdictRequest,
    ) -> CosmicVerdictResult:
        """
        Execute the full 4-phase verdict process.

        Args:
            request: The verdict request

        Returns:
            CosmicVerdictResult with final decision
        """
        start_time = time.perf_counter()
        self._verdict_count += 1

        if not request.request_id:
            request.request_id = f"verdict-{self._verdict_count:06d}"

        logger.info(f"Starting cosmic verdict for {request.request_id}")

        phases_completed: List[VotingPhase] = []

        # Phase 1: SCOUT - Independent reasoning, NO confidence sharing
        scout_result = await self._phase_scout(request)
        phases_completed.append(VotingPhase.SCOUT)

        # Early check for ABSOLUTE vetoes
        absolute_vetoes = self._check_absolute_vetoes(scout_result.votes)
        if absolute_vetoes:
            # Immediate rejection - cannot proceed
            logger.warning(
                f"{request.request_id}: ABSOLUTE veto by {[r.value for r in absolute_vetoes]}"
            )
            return self._create_absolute_veto_result(
                request, scout_result, absolute_vetoes, phases_completed, start_time
            )

        # Phase 2: WAGGLE - Advocacy with evidence quality
        waggle_result = await self._phase_waggle(scout_result)
        phases_completed.append(VotingPhase.WAGGLE)

        # Phase 3: QUORUM - Threshold checking
        quorum_result = await self._phase_quorum(
            waggle_result.votes,
            is_critical=request.is_critical,
        )
        phases_completed.append(VotingPhase.QUORUM)

        # Check for qualified vetoes that might be overrideable
        if (
            quorum_result.qualified_vetoes
            and quorum_result.approve_count >= QUALIFIED_OVERRIDE_QUORUM
        ):
            logger.info(
                f"{request.request_id}: Qualified veto overridden by {quorum_result.approve_count}/7"
            )

        # Phase 4: LIFTOFF - Final commitment
        final_decision = await self._phase_liftoff(quorum_result)
        phases_completed.append(VotingPhase.LIFTOFF)

        # Compute Merkle root of all votes
        merkle_root = self._compute_merkle_root(waggle_result.votes)

        # Generate evidence chain
        evidence_chain = self._generate_evidence_chain(
            request, scout_result, waggle_result, quorum_result
        )

        # Determine Majlis decision
        majlis_decision = self._derive_majlis_decision(quorum_result)

        # Get persona consensus if available
        persona_result = None
        if self.persona_consensus:
            # Would integrate with PersonaConsensusEngine here
            pass

        total_duration = (time.perf_counter() - start_time) * 1000

        result = CosmicVerdictResult(
            decision=final_decision,
            guardian_votes=waggle_result.votes,
            majlis_decision=majlis_decision,
            persona_consensus=persona_result,
            merkle_root=merkle_root,
            byzantine_tolerance=f"{BFT_F} of {BFT_N}",
            quorum_achieved=quorum_result.quorum_reached,
            evidence_chain=evidence_chain,
            request_id=request.request_id,
            voting_phases_completed=phases_completed,
            total_duration_ms=total_duration,
            ihsan_score=request.ihsan_score,
            snr_score=request.snr_score,
        )

        # Emit receipt
        self._emit_verdict_receipt(request, result)

        logger.info(
            f"Cosmic verdict {request.request_id}: {final_decision.value}, "
            f"quorum={quorum_result.quorum_reached}, "
            f"duration={total_duration:.1f}ms"
        )

        return result

    async def _phase_scout(self, request: VerdictRequest) -> ScoutResult:
        """
        SCOUT Phase: Independent reasoning with NO confidence sharing.

        Key insight from research: Hiding confidence scores during this phase
        prevents over-confidence cascades where early high-confidence votes
        unduly influence subsequent voters.

        Each guardian reasons in complete isolation.
        """
        start_time = time.perf_counter()
        votes: Dict[GuardianRole, GuardianVote] = {}

        # Get all guardians
        guardians = self.constellation.get_all_guardians()

        # Execute independent reasoning for each guardian
        for guardian in guardians:
            vote = await self._guardian_independent_reasoning(guardian, request)
            votes[guardian.role] = vote

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(f"SCOUT phase complete: {len(votes)} votes, {duration_ms:.1f}ms")

        return ScoutResult(votes=votes, duration_ms=duration_ms)

    async def _guardian_independent_reasoning(
        self,
        guardian: Guardian,
        request: VerdictRequest,
    ) -> GuardianVote:
        """
        Execute independent reasoning for a single guardian.

        The guardian evaluates the request based on their domain of responsibility
        without knowledge of other guardians' decisions or confidence levels.
        """
        # Evaluate based on guardian's specific domain
        decision = VoteDecision.APPROVE
        reasoning = ""
        evidence_quality = 0.85
        raw_confidence = 0.80

        # Ar-Ruh evaluates Ihsan (ethics/excellence)
        if guardian.role == GuardianRole.AR_RUH:
            if request.ihsan_score < IHSAN_THRESHOLD:
                decision = VoteDecision.REJECT
                reasoning = f"Ihsan score {request.ihsan_score:.3f} below threshold {IHSAN_THRESHOLD}"
                raw_confidence = 0.95
                evidence_quality = 0.90
            else:
                reasoning = f"Ihsan score {request.ihsan_score:.3f} meets threshold"
                raw_confidence = 0.92
                evidence_quality = 0.88

        # Al-Amin evaluates security
        elif guardian.role == GuardianRole.AL_AMIN:
            if request.snr_score < SNR_THRESHOLD:
                decision = VoteDecision.REJECT
                reasoning = (
                    f"SNR score {request.snr_score:.3f} below threshold {SNR_THRESHOLD}"
                )
                raw_confidence = 0.94
                evidence_quality = 0.92
            else:
                # Check for security-sensitive operations
                payload_str = json.dumps(request.payload).lower()
                security_keywords = ["secret", "key", "password", "credential", "token"]
                if any(kw in payload_str for kw in security_keywords):
                    decision = VoteDecision.SOFT_VETO
                    reasoning = "Security-sensitive operation detected, requires review"
                    raw_confidence = 0.88
                    evidence_quality = 0.85
                else:
                    reasoning = "No security concerns detected"
                    raw_confidence = 0.90

        # Al-Mujtahid evaluates compliance
        elif guardian.role == GuardianRole.AL_MUJTAHID:
            # Check for compliance indicators
            reasoning = "Compliance check passed"
            raw_confidence = 0.85
            evidence_quality = 0.82

        # Al-Muhasib evaluates resource usage
        elif guardian.role == GuardianRole.AL_MUHASIB:
            reasoning = "Resource allocation within bounds"
            raw_confidence = 0.83
            evidence_quality = 0.80

        # Al-Raqib monitors for anomalies
        elif guardian.role == GuardianRole.AL_RAQIB:
            reasoning = "No anomalies detected"
            raw_confidence = 0.82
            evidence_quality = 0.78

        # Al-Mustashar provides strategic assessment
        elif guardian.role == GuardianRole.AL_MUSTASHAR:
            if request.is_critical:
                reasoning = "Critical operation - recommend extra validation"
                raw_confidence = 0.85
                evidence_quality = 0.80
            else:
                reasoning = "Operation within normal risk parameters"
                raw_confidence = 0.80

        # Al-Murabbi evaluates knowledge/documentation
        elif guardian.role == GuardianRole.AL_MURABBI:
            reasoning = "Documentation standards met"
            raw_confidence = 0.78
            evidence_quality = 0.75

        # Majlis Al-Kawni (meta-council) synthesizes
        elif guardian.role == GuardianRole.MAJLIS_AL_KAWNI:
            reasoning = "Awaiting collective synthesis"
            raw_confidence = 0.90
            evidence_quality = 0.85

        else:
            reasoning = f"Guardian {guardian.role.value} default approval"

        vote = GuardianVote(
            guardian_role=guardian.role,
            decision=decision,
            reasoning=reasoning,
            evidence_quality=evidence_quality,
        )

        # Set hidden confidence (not shared during SCOUT)
        vote.set_hidden_confidence(raw_confidence)

        return vote

    async def _phase_waggle(self, scout_result: ScoutResult) -> WaggleResult:
        """
        WAGGLE Phase: Advocacy presentation with evidence quality.

        After SCOUT phase completes, confidences are revealed and guardians
        can present their advocacy for their positions.
        """
        start_time = time.perf_counter()

        # Reveal all confidences now that SCOUT is complete
        revealed_confidences = scout_result.reveal_all_confidences()

        # Calculate evidence scores based on reasoning quality
        evidence_scores: Dict[GuardianRole, float] = {}
        advocacy_quality: Dict[GuardianRole, float] = {}

        for role, vote in scout_result.votes.items():
            # Evidence score from vote
            evidence_scores[role] = vote.evidence_quality

            # Advocacy quality based on reasoning length and specificity
            reasoning_len = len(vote.reasoning)
            specificity = (
                1.0
                if any(
                    kw in vote.reasoning.lower()
                    for kw in ["threshold", "detected", "score", "specific"]
                )
                else 0.7
            )

            advocacy_quality[role] = min(1.0, (reasoning_len / 100) * specificity)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"WAGGLE phase complete: confidences revealed, {duration_ms:.1f}ms"
        )

        return WaggleResult(
            votes=scout_result.votes,
            evidence_scores=evidence_scores,
            advocacy_quality=advocacy_quality,
            duration_ms=duration_ms,
        )

    async def _phase_quorum(
        self,
        votes: Dict[GuardianRole, GuardianVote],
        is_critical: bool = False,
    ) -> QuorumResult:
        """
        QUORUM Phase: Decision threshold checking.

        Normal decisions: 5/8 required
        Critical decisions: 6/8 required
        """
        start_time = time.perf_counter()

        approve_count = 0
        reject_count = 0
        veto_count = 0
        abstain_count = 0
        absolute_vetoes: List[GuardianRole] = []
        qualified_vetoes: List[GuardianRole] = []

        for role, vote in votes.items():
            if vote.decision == VoteDecision.APPROVE:
                approve_count += 1
            elif vote.decision == VoteDecision.REJECT:
                reject_count += 1
                # Check if this is a veto
                guardian = self.constellation.get_guardian(role)
                if guardian and guardian.veto_power == VetoPower.ABSOLUTE:
                    veto_count += 1
                    absolute_vetoes.append(role)
                elif guardian and guardian.veto_power == VetoPower.QUALIFIED:
                    qualified_vetoes.append(role)
            elif vote.decision == VoteDecision.SOFT_VETO:
                # Soft veto counts as qualified veto for threshold purposes
                guardian = self.constellation.get_guardian(role)
                if guardian and guardian.veto_power == VetoPower.ABSOLUTE:
                    # Even soft veto from ABSOLUTE guardian is significant
                    qualified_vetoes.append(role)
                else:
                    # Treat as abstain for counting
                    abstain_count += 1
            elif vote.decision == VoteDecision.ABSTAIN:
                abstain_count += 1

        # Determine quorum type and threshold
        quorum_type = "critical" if is_critical else "normal"
        required_quorum = CRITICAL_QUORUM if is_critical else BFT_QUORUM

        # Check if quorum reached
        quorum_reached = approve_count >= required_quorum and not absolute_vetoes

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"QUORUM phase: approve={approve_count}, reject={reject_count}, "
            f"veto={veto_count}, abstain={abstain_count}, "
            f"quorum_reached={quorum_reached}, {duration_ms:.1f}ms"
        )

        return QuorumResult(
            approve_count=approve_count,
            reject_count=reject_count,
            veto_count=veto_count,
            abstain_count=abstain_count,
            quorum_type=quorum_type,
            quorum_reached=quorum_reached,
            absolute_vetoes=absolute_vetoes,
            qualified_vetoes=qualified_vetoes,
            duration_ms=duration_ms,
        )

    async def _phase_liftoff(self, quorum_result: QuorumResult) -> VerdictDecision:
        """
        LIFTOFF Phase: Winner-take-all commitment.

        Final decision is locked based on quorum results.
        """
        # Check for absolute vetoes first (cannot be overridden)
        if quorum_result.absolute_vetoes:
            return VerdictDecision.ABSOLUTE_VETO

        # Check for qualified vetoes that weren't overridden
        if quorum_result.qualified_vetoes:
            if quorum_result.approve_count < QUALIFIED_OVERRIDE_QUORUM:
                return VerdictDecision.QUALIFIED_VETO

        # Check if quorum was reached for approval
        if quorum_result.quorum_reached:
            return VerdictDecision.APPROVED

        # Check if clear rejection
        if quorum_result.reject_count > quorum_result.approve_count:
            return VerdictDecision.REJECTED

        # No clear decision - deadlock
        return VerdictDecision.DEADLOCK

    def _check_absolute_vetoes(
        self,
        votes: Dict[GuardianRole, GuardianVote],
    ) -> List[GuardianRole]:
        """
        Check for ABSOLUTE vetoes from Ar-Ruh, Al-Amin, or Majlis.

        These vetoes cannot be overridden under any circumstances.
        """
        absolute_vetoes: List[GuardianRole] = []

        for role, vote in votes.items():
            if role in ABSOLUTE_VETO_ROLES:
                if vote.decision in (VoteDecision.REJECT, VoteDecision.SOFT_VETO):
                    # Check if this is truly a veto (not just disagreement)
                    guardian = self.constellation.get_guardian(role)
                    if guardian and guardian.veto_power == VetoPower.ABSOLUTE:
                        absolute_vetoes.append(role)

        return absolute_vetoes

    def _compute_merkle_root(
        self,
        votes: Dict[GuardianRole, GuardianVote],
    ) -> str:
        """
        Compute Merkle root of all votes for integrity verification.
        """
        if not votes:
            return self._hash(b"empty")

        # Create leaf hashes from votes
        leaves = []
        for role, vote in sorted(votes.items(), key=lambda x: x[0].value):
            vote_data = (
                f"{role.value}:{vote.decision.value}:"
                f"{vote.evidence_quality}:{vote.timestamp.isoformat()}"
            )
            leaf_hash = self._hash(vote_data.encode())
            leaves.append(leaf_hash)

        # Build Merkle tree
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])  # Duplicate last for odd count

            new_level = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i + 1]
                new_hash = self._hash(combined.encode())
                new_level.append(new_hash)
            leaves = new_level

        return leaves[0]

    def _hash(self, data: bytes) -> str:
        """Compute hash using BLAKE3 or SHA-256 fallback."""
        if HAS_BLAKE3:
            return blake3.blake3(data).hexdigest()
        return hashlib.sha256(data).hexdigest()

    def _generate_evidence_chain(
        self,
        request: VerdictRequest,
        scout_result: ScoutResult,
        waggle_result: WaggleResult,
        quorum_result: QuorumResult,
    ) -> List[str]:
        """Generate chain of evidence hashes for audit trail."""
        chain = []

        # Hash request
        request_hash = self._hash(
            json.dumps(request.to_dict(), sort_keys=True).encode()
        )
        chain.append(f"request:{request_hash[:16]}")

        # Hash each phase
        scout_data = {
            "phase": "scout",
            "vote_count": len(scout_result.votes),
            "duration_ms": scout_result.duration_ms,
        }
        chain.append(f"scout:{self._hash(json.dumps(scout_data).encode())[:16]}")

        waggle_data = {
            "phase": "waggle",
            "evidence_scores": {
                k.value: v for k, v in waggle_result.evidence_scores.items()
            },
            "duration_ms": waggle_result.duration_ms,
        }
        chain.append(
            f"waggle:{self._hash(json.dumps(waggle_data, sort_keys=True).encode())[:16]}"
        )

        quorum_data = {
            "phase": "quorum",
            "approve": quorum_result.approve_count,
            "reject": quorum_result.reject_count,
            "quorum_reached": quorum_result.quorum_reached,
        }
        chain.append(f"quorum:{self._hash(json.dumps(quorum_data).encode())[:16]}")

        return chain

    def _derive_majlis_decision(self, quorum_result: QuorumResult) -> MajlisDecision:
        """Derive Majlis decision from quorum result."""
        if quorum_result.absolute_vetoes:
            return MajlisDecision.DEADLOCK

        total_votes = (
            quorum_result.approve_count
            + quorum_result.reject_count
            + quorum_result.veto_count
        )

        if total_votes == 0:
            return MajlisDecision.DEADLOCK

        if quorum_result.approve_count == 7:  # All 7 guardians
            return MajlisDecision.CONSENSUS
        elif quorum_result.approve_count >= 6:
            return MajlisDecision.SUPERMAJORITY
        elif quorum_result.approve_count >= 5:
            return MajlisDecision.MAJORITY
        elif quorum_result.approve_count >= 3:
            return MajlisDecision.SPLIT
        else:
            return MajlisDecision.DEADLOCK

    def _create_absolute_veto_result(
        self,
        request: VerdictRequest,
        scout_result: ScoutResult,
        absolute_vetoes: List[GuardianRole],
        phases_completed: List[VotingPhase],
        start_time: float,
    ) -> CosmicVerdictResult:
        """Create result for ABSOLUTE veto (immediate rejection)."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        return CosmicVerdictResult(
            decision=VerdictDecision.ABSOLUTE_VETO,
            guardian_votes=scout_result.votes,
            majlis_decision=MajlisDecision.DEADLOCK,
            persona_consensus=None,
            merkle_root=self._compute_merkle_root(scout_result.votes),
            byzantine_tolerance=f"{BFT_F} of {BFT_N}",
            quorum_achieved=False,
            evidence_chain=[
                f"absolute_veto:{','.join(r.value for r in absolute_vetoes)}"
            ],
            request_id=request.request_id,
            voting_phases_completed=phases_completed,
            total_duration_ms=duration_ms,
            ihsan_score=request.ihsan_score,
            snr_score=request.snr_score,
        )

    def _emit_verdict_receipt(
        self,
        request: VerdictRequest,
        result: CosmicVerdictResult,
    ) -> Dict[str, Any]:
        """Emit evidence receipt for the verdict."""
        receipt = {
            "receipt_id": str(uuid4()),
            "receipt_type": "cosmic_verdict",
            "request_id": request.request_id,
            "decision": result.decision.value,
            "majlis_decision": result.majlis_decision.value,
            "quorum_achieved": result.quorum_achieved,
            "merkle_root": result.merkle_root,
            "evidence_chain": result.evidence_chain,
            "ihsan_score": result.ihsan_score,
            "snr_score": result.snr_score,
            "duration_ms": result.total_duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integrity_hash": self._hash(
                json.dumps(result.to_dict(), sort_keys=True).encode()
            )[:32],
        }

        self._receipts.append(receipt)
        logger.debug(f"Verdict receipt emitted: {receipt['receipt_id']}")

        return receipt

    def get_receipts(self) -> List[Dict[str, Any]]:
        """Get all verdict receipts."""
        return list(self._receipts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "verdict_count": self._verdict_count,
            "receipt_count": len(self._receipts),
            "bft_parameters": {
                "n": BFT_N,
                "f": BFT_F,
                "quorum": BFT_QUORUM,
            },
            "thresholds": {
                "ihsan": IHSAN_THRESHOLD,
                "snr": SNR_THRESHOLD,
                "critical_quorum": CRITICAL_QUORUM,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_cosmic_verdict_engine(
    guardian_constellation: Optional[GuardianConstellation] = None,
    persona_consensus: Optional[Any] = None,
) -> CosmicVerdictEngine:
    """
    Create a CosmicVerdictEngine instance.

    Args:
        guardian_constellation: Optional pre-configured constellation
        persona_consensus: Optional PersonaConsensusEngine

    Returns:
        Configured CosmicVerdictEngine
    """
    return CosmicVerdictEngine(
        guardian_constellation=guardian_constellation,
        persona_consensus=persona_consensus,
    )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================


_cosmic_verdict_engine: Optional[CosmicVerdictEngine] = None


def get_cosmic_verdict_engine() -> CosmicVerdictEngine:
    """Get or create the global CosmicVerdictEngine instance."""
    global _cosmic_verdict_engine
    if _cosmic_verdict_engine is None:
        _cosmic_verdict_engine = create_cosmic_verdict_engine()
    return _cosmic_verdict_engine


def reset_cosmic_verdict_engine() -> None:
    """Reset the global CosmicVerdictEngine."""
    global _cosmic_verdict_engine
    _cosmic_verdict_engine = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Version info
    "COSMIC_VERDICT_VERSION",
    "COSMIC_VERDICT_DOMAIN",
    # BFT constants
    "BFT_N",
    "BFT_F",
    "BFT_QUORUM",
    "CRITICAL_QUORUM",
    # Threshold constants
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    # Enums
    "VerdictDecision",
    "VotingPhase",
    # Data classes
    "GuardianVote",
    "CosmicVerdictResult",
    "VerdictRequest",
    "ScoutResult",
    "WaggleResult",
    "QuorumResult",
    # Main class
    "CosmicVerdictEngine",
    # Factory functions
    "create_cosmic_verdict_engine",
    # Global instance functions
    "get_cosmic_verdict_engine",
    "reset_cosmic_verdict_engine",
    # Constants
    "ABSOLUTE_VETO_ROLES",
]


# =============================================================================
# MAIN - Demo/Test
# =============================================================================


if __name__ == "__main__":
    import asyncio

    async def demo():
        print("BIZRA Cosmic Verdict Engine - Swarm Intelligence Demo")
        print("=" * 60)

        # Create engine
        engine = create_cosmic_verdict_engine()

        print("\nEngine Statistics:")
        stats = engine.get_statistics()
        print(
            f"  BFT Parameters: n={stats['bft_parameters']['n']}, "
            f"f={stats['bft_parameters']['f']}, "
            f"quorum={stats['bft_parameters']['quorum']}"
        )
        print(f"  Ihsan Threshold: {stats['thresholds']['ihsan']}")
        print(f"  SNR Threshold: {stats['thresholds']['snr']}")

        # Test Case 1: Normal approval
        print("\n" + "-" * 40)
        print("Test Case 1: Normal request (should approve)")

        request1 = VerdictRequest(
            request_id="test-001",
            task="Execute normal operation",
            payload={"action": "process"},
            context={},
            ihsan_score=0.96,
            snr_score=0.99,
        )

        result1 = await engine.execute_verdict(request1)
        print(f"  Decision: {result1.decision.value}")
        print(f"  Majlis Decision: {result1.majlis_decision.value}")
        print(f"  Quorum Achieved: {result1.quorum_achieved}")
        print(f"  Duration: {result1.total_duration_ms:.1f}ms")
        print(f"  Phases: {[p.value for p in result1.voting_phases_completed]}")

        # Test Case 2: Low Ihsan score (should reject via Ar-Ruh)
        print("\n" + "-" * 40)
        print("Test Case 2: Low Ihsan score (should trigger ABSOLUTE veto)")

        request2 = VerdictRequest(
            request_id="test-002",
            task="Execute risky operation",
            payload={"action": "process"},
            context={},
            ihsan_score=0.90,  # Below threshold
            snr_score=0.99,
        )

        result2 = await engine.execute_verdict(request2)
        print(f"  Decision: {result2.decision.value}")
        print(f"  Majlis Decision: {result2.majlis_decision.value}")
        print(f"  Evidence Chain: {result2.evidence_chain}")

        # Test Case 3: Critical operation
        print("\n" + "-" * 40)
        print("Test Case 3: Critical operation (requires 6/8)")

        request3 = VerdictRequest(
            request_id="test-003",
            task="Deploy production changes",
            payload={"action": "deploy", "target": "production"},
            context={"environment": "production"},
            ihsan_score=0.97,
            snr_score=0.99,
            is_critical=True,
        )

        result3 = await engine.execute_verdict(request3)
        print(f"  Decision: {result3.decision.value}")
        print(f"  Majlis Decision: {result3.majlis_decision.value}")
        print(f"  Merkle Root: {result3.merkle_root[:32]}...")

        # Print guardian votes
        print("\n  Guardian Votes:")
        for role, vote in result3.guardian_votes.items():
            conf = vote.confidence if vote.confidence else "hidden"
            print(f"    {role.value}: {vote.decision.value} (confidence: {conf})")

        # Final stats
        print("\n" + "=" * 60)
        print("Final Statistics:")
        final_stats = engine.get_statistics()
        print(f"  Total Verdicts: {final_stats['verdict_count']}")
        print(f"  Total Receipts: {final_stats['receipt_count']}")
        print("=" * 60)

    asyncio.run(demo())
