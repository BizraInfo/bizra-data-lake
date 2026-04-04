# core/federation/consensus.py - Byzantine Fault Tolerant Pattern Consensus
#
# Implements 3-of-5 (or N-of-M) Byzantine consensus for pattern acceptance.
# Based on PBFT (Practical Byzantine Fault Tolerance) simplified for patterns.
#
# Consensus Flow:
# 1. PRE-PREPARE: Leader proposes pattern for consensus
# 2. PREPARE: Validators verify and broadcast prepare messages
# 3. COMMIT: Upon 2f+1 prepares, broadcast commit
# 4. FINALIZE: Upon 2f+1 commits, pattern is accepted

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from core.federation.protocol import (
    ConsensusVote,
    ConsensusResult,
    PatternEnvelope,
    VoteDecision,
    MIN_IHSAN_SCORE,
    canonical_json,
    sign_message,
)

logger = logging.getLogger("federation.consensus")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Byzantine parameters: N = 5, f = 1, quorum = 2f + 1 = 3
CONSENSUS_N = 5  # Total validators
CONSENSUS_F = 1  # Max Byzantine failures
CONSENSUS_QUORUM = 2 * CONSENSUS_F + 1  # = 3

# Timeouts
CONSENSUS_TIMEOUT_SEC = 30.0  # Max time for consensus round
VOTE_TIMEOUT_SEC = 10.0  # Time to wait for votes


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS STATE
# ═══════════════════════════════════════════════════════════════════════════════


class ConsensusPhase(Enum):
    """Phases of consensus."""

    IDLE = "idle"
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass
class ConsensusState:
    """State of an ongoing consensus round."""

    pattern_id: str
    phase: ConsensusPhase = ConsensusPhase.IDLE
    started_at: float = field(default_factory=time.time)

    # Votes collected
    prepare_votes: Dict[str, ConsensusVote] = field(default_factory=dict)
    commit_votes: Dict[str, ConsensusVote] = field(default_factory=dict)

    # Result
    result: Optional[ConsensusResult] = None

    def is_expired(self) -> bool:
        """Check if consensus round has timed out."""
        return time.time() - self.started_at > CONSENSUS_TIMEOUT_SEC

    def prepare_count(self) -> int:
        """Count ACCEPT votes in prepare phase."""
        return sum(
            1 for v in self.prepare_votes.values() if v.decision == VoteDecision.ACCEPT
        )

    def commit_count(self) -> int:
        """Count ACCEPT votes in commit phase."""
        return sum(
            1 for v in self.commit_votes.values() if v.decision == VoteDecision.ACCEPT
        )

    def has_prepare_quorum(self) -> bool:
        """Check if prepare quorum reached."""
        return self.prepare_count() >= CONSENSUS_QUORUM

    def has_commit_quorum(self) -> bool:
        """Check if commit quorum reached."""
        return self.commit_count() >= CONSENSUS_QUORUM


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════


class PatternValidator:
    """
    Validates patterns for consensus.

    Checks:
    1. Signature validity (cryptographic)
    2. Ihsān score meets threshold
    3. Pattern structure is valid
    4. No known malicious patterns
    5. Node reputation check
    """

    def __init__(self):
        # Known malicious pattern hashes
        self.blacklisted_patterns: Set[str] = set()

        # Node reputation cache
        self.node_reputation: Dict[str, float] = {}

        # Validation history
        self.validation_history: List[Dict[str, Any]] = []

    def validate(
        self,
        envelope: PatternEnvelope,
        local_ihsan_score: Optional[float] = None,
    ) -> tuple[VoteDecision, str, float]:
        """
        Validate pattern and return vote decision.

        Returns:
            (decision, reason, computed_ihsan)
        """
        # 1. Check blacklist
        if envelope.metadata.pattern_id in self.blacklisted_patterns:
            return VoteDecision.REJECT, "Pattern blacklisted", 0.0

        # 2. Verify cryptographic integrity
        valid, reason = envelope.verify()
        if not valid:
            return VoteDecision.REJECT, f"Verification failed: {reason}", 0.0

        # 3. Check Ihsān score
        ihsan = envelope.metadata.ihsan_score
        if ihsan < MIN_IHSAN_SCORE:
            return VoteDecision.REJECT, f"Ihsān {ihsan:.2f} < {MIN_IHSAN_SCORE}", ihsan

        # 4. Check node reputation
        origin = envelope.metadata.origin_node_id
        if origin in self.node_reputation:
            if self.node_reputation[origin] < 0.5:
                return VoteDecision.REJECT, "Origin node reputation too low", ihsan

        # 5. Check impact score
        if envelope.metadata.impact_score < 0.5:
            return (
                VoteDecision.REJECT,
                f"Impact score too low: {envelope.metadata.impact_score}",
                ihsan,
            )

        # 6. Check repetition count
        if envelope.metadata.repetition_count < 3:
            return VoteDecision.REJECT, "Insufficient repetitions", ihsan

        # 7. If we have local evaluation, weight it
        if local_ihsan_score is not None:
            # Combine origin and local scores
            combined = 0.7 * ihsan + 0.3 * local_ihsan_score
            if combined < MIN_IHSAN_SCORE:
                return (
                    VoteDecision.REJECT,
                    f"Combined Ihsān {combined:.2f} too low",
                    combined,
                )
            ihsan = combined

        # All checks passed
        return VoteDecision.ACCEPT, "Valid pattern", ihsan

    def update_reputation(self, node_id: str, delta: float):
        """Update node reputation."""
        current = self.node_reputation.get(node_id, 1.0)
        new_rep = max(0.0, min(1.0, current + delta))
        self.node_reputation[node_id] = new_rep

    def blacklist_pattern(self, pattern_id: str):
        """Add pattern to blacklist."""
        self.blacklisted_patterns.add(pattern_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class PatternConsensus:
    """
    Byzantine Fault Tolerant consensus for pattern acceptance.

    Uses simplified PBFT with 3-of-5 quorum requirement.
    Ensures network-wide agreement on pattern validity.
    """

    def __init__(
        self,
        node_id: str,
        private_key: bytes,
        public_key: bytes,
    ):
        self.node_id = node_id
        self._private_key = private_key
        self._public_key = public_key

        # Active consensus rounds
        self.active_rounds: Dict[str, ConsensusState] = {}

        # Completed rounds (recent)
        self.completed_rounds: Dict[str, ConsensusResult] = {}

        # Validator
        self.validator = PatternValidator()

        # Callbacks
        self._on_pattern_accepted: Optional[Callable[[PatternEnvelope], None]] = None
        self._on_pattern_rejected: Optional[Callable[[str, str], None]] = None
        self._broadcast_vote: Optional[Callable[[ConsensusVote], None]] = None

        # Statistics
        self.stats = ConsensusStats()

    # ─────────────────────────────────────────────────────────────────────────
    # Consensus Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def propose_pattern(self, envelope: PatternEnvelope) -> ConsensusState:
        """
        Propose a new pattern for consensus (PRE-PREPARE phase).
        """
        pattern_id = envelope.metadata.pattern_id

        # Already in consensus?
        if pattern_id in self.active_rounds:
            return self.active_rounds[pattern_id]

        if pattern_id in self.completed_rounds:
            # Already decided
            state = ConsensusState(
                pattern_id=pattern_id, phase=ConsensusPhase.FINALIZED
            )
            state.result = self.completed_rounds[pattern_id]
            return state

        # Create new consensus round
        state = ConsensusState(pattern_id=pattern_id, phase=ConsensusPhase.PRE_PREPARE)
        self.active_rounds[pattern_id] = state

        logger.info(f"📋 Starting consensus for pattern {pattern_id[:16]}...")

        # Validate and cast our vote
        decision, reason, ihsan = self.validator.validate(envelope)

        our_vote = self._create_vote(pattern_id, decision, reason, ihsan)
        state.prepare_votes[self.node_id] = our_vote

        # Move to PREPARE phase
        state.phase = ConsensusPhase.PREPARE

        # Broadcast vote
        if self._broadcast_vote:
            self._broadcast_vote(our_vote)

        self.stats.rounds_started += 1

        return state

    async def receive_vote(
        self,
        vote: ConsensusVote,
        phase: str = "prepare",
    ) -> Optional[ConsensusResult]:
        """
        Receive a vote from another validator.

        Returns ConsensusResult if finalized.
        """
        pattern_id = vote.pattern_id

        # Already completed?
        if pattern_id in self.completed_rounds:
            return self.completed_rounds[pattern_id]

        # Get or create state
        if pattern_id not in self.active_rounds:
            self.active_rounds[pattern_id] = ConsensusState(
                pattern_id=pattern_id,
                phase=ConsensusPhase.PREPARE,
            )

        state = self.active_rounds[pattern_id]

        # Check timeout
        if state.is_expired():
            return self._finalize_timeout(state)

        # Record vote
        if phase == "prepare":
            state.prepare_votes[vote.voter_id] = vote

            # Check for prepare quorum
            if state.has_prepare_quorum() and state.phase == ConsensusPhase.PREPARE:
                state.phase = ConsensusPhase.COMMIT
                logger.debug(f"Pattern {pattern_id[:16]} reached prepare quorum")

                # Cast commit vote
                commit_vote = self._create_vote(
                    pattern_id,
                    VoteDecision.ACCEPT,
                    "Prepare quorum reached",
                    vote.ihsan_score,
                )
                state.commit_votes[self.node_id] = commit_vote

                if self._broadcast_vote:
                    self._broadcast_vote(commit_vote)

        elif phase == "commit":
            state.commit_votes[vote.voter_id] = vote

            # Check for commit quorum
            if state.has_commit_quorum() and state.phase == ConsensusPhase.COMMIT:
                return self._finalize_accept(state)

        return None

    def _create_vote(
        self,
        pattern_id: str,
        decision: VoteDecision,
        reason: str,
        ihsan: float,
    ) -> ConsensusVote:
        """Create a signed vote."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create signature content
        content = canonical_json(
            {
                "pattern_id": pattern_id,
                "voter_id": self.node_id,
                "decision": decision.value,
                "reason": reason,
                "ihsan_score": ihsan,
                "timestamp": timestamp,
            }
        )

        signature = sign_message(content.encode(), self._private_key)

        return ConsensusVote(
            pattern_id=pattern_id,
            voter_id=self.node_id,
            decision=decision,
            reason=reason,
            ihsan_score=ihsan,
            timestamp=timestamp,
            signature=signature.hex(),
        )

    def _finalize_accept(self, state: ConsensusState) -> ConsensusResult:
        """Finalize consensus as accepted."""
        state.phase = ConsensusPhase.FINALIZED

        all_votes = list(state.prepare_votes.values()) + list(
            state.commit_votes.values()
        )

        result = ConsensusResult(
            pattern_id=state.pattern_id,
            accepted=True,
            accept_votes=state.commit_count(),
            reject_votes=sum(
                1
                for v in state.commit_votes.values()
                if v.decision == VoteDecision.REJECT
            ),
            abstain_votes=sum(
                1
                for v in state.commit_votes.values()
                if v.decision == VoteDecision.ABSTAIN
            ),
            quorum_reached=True,
            finalized_at=datetime.now(timezone.utc).isoformat(),
            votes=all_votes,
        )

        state.result = result
        self.completed_rounds[state.pattern_id] = result
        del self.active_rounds[state.pattern_id]

        self.stats.rounds_accepted += 1

        logger.info(
            f"✅ Pattern {state.pattern_id[:16]} ACCEPTED by consensus ({result.accept_votes}/{CONSENSUS_QUORUM})"
        )

        if self._on_pattern_accepted:
            # Callback would receive the envelope
            pass

        return result

    def _finalize_timeout(self, state: ConsensusState) -> ConsensusResult:
        """Finalize consensus as failed due to timeout."""
        state.phase = ConsensusPhase.FAILED

        result = ConsensusResult(
            pattern_id=state.pattern_id,
            accepted=False,
            accept_votes=state.prepare_count(),
            reject_votes=sum(
                1
                for v in state.prepare_votes.values()
                if v.decision == VoteDecision.REJECT
            ),
            abstain_votes=sum(
                1
                for v in state.prepare_votes.values()
                if v.decision == VoteDecision.ABSTAIN
            ),
            quorum_reached=False,
            finalized_at=datetime.now(timezone.utc).isoformat(),
            votes=list(state.prepare_votes.values()),
        )

        state.result = result
        self.completed_rounds[state.pattern_id] = result
        del self.active_rounds[state.pattern_id]

        self.stats.rounds_timeout += 1

        logger.warning(
            f"⏱️ Pattern {state.pattern_id[:16]} TIMEOUT - no consensus reached"
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_pattern_accepted(self, callback: Callable[[PatternEnvelope], None]):
        """Register callback for accepted patterns."""
        self._on_pattern_accepted = callback

    def on_pattern_rejected(self, callback: Callable[[str, str], None]):
        """Register callback for rejected patterns."""
        self._on_pattern_rejected = callback

    def set_vote_broadcaster(self, callback: Callable[[ConsensusVote], None]):
        """Set callback to broadcast votes to peers."""
        self._broadcast_vote = callback

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def get_result(self, pattern_id: str) -> Optional[ConsensusResult]:
        """Get consensus result for a pattern."""
        return self.completed_rounds.get(pattern_id)

    def is_accepted(self, pattern_id: str) -> bool:
        """Check if pattern was accepted."""
        result = self.completed_rounds.get(pattern_id)
        return result is not None and result.accepted

    def cleanup_old_rounds(self, max_age_sec: float = 3600.0):
        """Clean up old completed rounds."""
        now = time.time()
        cutoff = datetime.fromtimestamp(now - max_age_sec, timezone.utc).isoformat()

        to_remove = [
            pid
            for pid, result in self.completed_rounds.items()
            if result.finalized_at < cutoff
        ]

        for pid in to_remove:
            del self.completed_rounds[pid]


@dataclass
class ConsensusStats:
    """Consensus statistics."""

    rounds_started: int = 0
    rounds_accepted: int = 0
    rounds_rejected: int = 0
    rounds_timeout: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds_started": self.rounds_started,
            "rounds_accepted": self.rounds_accepted,
            "rounds_rejected": self.rounds_rejected,
            "rounds_timeout": self.rounds_timeout,
            "acceptance_rate": (self.rounds_accepted / max(1, self.rounds_started)),
        }
