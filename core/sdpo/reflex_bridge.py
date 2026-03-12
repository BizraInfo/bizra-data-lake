"""
SDPO → Reflex Bridge — Training-to-Compilation Pathway
═══════════════════════════════════════════════════════════════════════════════

Bridges the SDPO training pipeline with the Reflex Compilation system.
When SDPO training produces consistently high-quality response patterns
(reproducibility ≥ 0.90, ihsan ≥ 0.95), this bridge promotes them
to reflex candidates for O(1) compilation.

Data Flow:
    BIZRASDPOTrainer.train()
        → TrainingResult (loss, ihsan_score, per-sample quality)
            → ReflexBridge.evaluate_for_compilation()
                → ReflexCandidate (pattern_id, reproducibility, ihsan)
                    → Constitutional Tick Step 11: reflex_compile_if_eligible()

This closes the gap identified in Blueprint P0 (Section 3.1):
"SDPO training results never feed back into reflex compilation."

Standing on Giants:
- Kahneman (System 1/System 2, 2011) — reflexes ARE compiled System-2 paths
- Shannon (Information Theory / SNR, 1948)
- Deming (PDCA quality ratchet, 1950)
- Anthropic (Constitutional AI / Ihsān, 2023)

Constitutional: Reflex compilation requires Ihsān ≥ 0.98 (ticker Step 11).
Impact score > 0.0 prevents no-op gaming (RLVR constraint).
30-day deny-list for flagged patterns.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Set

from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Reflex compilation thresholds (from constitutional tick Step 11)
REFLEX_IHSAN_THRESHOLD = SNR_THRESHOLD_T0_ELITE  # 0.98 — stricter than production 0.95
REFLEX_REPRODUCIBILITY_THRESHOLD = 0.90  # Must reproduce results consistently
REFLEX_MIN_OBSERVATIONS = 5  # Minimum training appearances
REFLEX_IMPACT_FLOOR = 0.01  # Impact must be > 0 (anti-gaming)
DENY_LIST_EXPIRY_DAYS = 30  # Flagged patterns expire after 30 days


@dataclass(frozen=True)
class ReflexCandidate:
    """A pattern eligible for reflex compilation.

    Represents a response pattern that SDPO training has validated
    as consistently high-quality and reproducible.
    """

    pattern_id: str  # BLAKE3 or SHA-256 hash of the pattern
    pattern_description: str
    source_task: str
    avg_ihsan: float
    avg_snr: float
    reproducibility: float  # Fraction of consistent outcomes
    observation_count: int
    impact_score: float  # Normalized impact (0.0–1.0)
    required_observations: int = REFLEX_MIN_OBSERVATIONS
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def eligible(self) -> bool:
        """Check if candidate meets ALL compilation gates."""
        return (
            self.avg_ihsan >= REFLEX_IHSAN_THRESHOLD
            and self.reproducibility >= REFLEX_REPRODUCIBILITY_THRESHOLD
            and self.observation_count >= self.required_observations
            and self.impact_score > REFLEX_IMPACT_FLOOR
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_description": self.pattern_description,
            "source_task": self.source_task,
            "avg_ihsan": self.avg_ihsan,
            "avg_snr": self.avg_snr,
            "reproducibility": self.reproducibility,
            "observation_count": self.observation_count,
            "required_observations": self.required_observations,
            "impact_score": self.impact_score,
            "eligible": self.eligible,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TrainingObservation:
    """A single SDPO training result observation for a pattern."""

    task_description: str
    ihsan_score: float
    snr_score: float
    loss: float
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SDPOReflexBridge:
    """Bridge between SDPO training and Reflex Compilation.

    Accumulates training observations per-pattern, computes
    reproducibility and impact scores, and emits ReflexCandidates
    when thresholds are met.

    Usage:
        bridge = SDPOReflexBridge()

        # After each SDPO training step
        bridge.observe(
            task_description="Synthesize research findings",
            ihsan_score=0.98,
            snr_score=0.96,
            loss=0.15,
            success=True,
        )

        # Check for reflex candidates
        candidates = bridge.get_eligible_candidates()
        for candidate in candidates:
            # Feed to constitutional tick Step 11
            reflex_compile_if_eligible(candidate)
    """

    def __init__(
        self,
        ihsan_threshold: float = REFLEX_IHSAN_THRESHOLD,
        reproducibility_threshold: float = REFLEX_REPRODUCIBILITY_THRESHOLD,
        min_observations: int = REFLEX_MIN_OBSERVATIONS,
    ):
        self._ihsan_threshold = ihsan_threshold
        self._repro_threshold = reproducibility_threshold
        self._min_obs = min_observations
        self._observations: Dict[str, List[TrainingObservation]] = {}
        self._deny_list: Dict[str, datetime] = {}  # pattern_id → expiry
        self._compiled: Set[str] = set()  # Already-compiled pattern IDs

    def observe(
        self,
        task_description: str,
        ihsan_score: float,
        snr_score: float,
        loss: float,
        success: bool,
    ) -> str:
        """Record a training observation.

        Returns the pattern_id (hash of the task description).
        """
        pattern_id = self._hash_pattern(task_description)

        obs = TrainingObservation(
            task_description=task_description,
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            loss=loss,
            success=success,
        )

        if pattern_id not in self._observations:
            self._observations[pattern_id] = []
        self._observations[pattern_id].append(obs)

        return pattern_id

    def get_eligible_candidates(self) -> List[ReflexCandidate]:
        """Evaluate all observed patterns and return eligible candidates.

        Filters out:
        - Patterns on the deny-list (not expired)
        - Patterns already compiled
        - Patterns below observation count threshold
        """
        self._clean_expired_denials()
        candidates = []

        for pattern_id, observations in self._observations.items():
            if pattern_id in self._deny_list:
                continue
            if pattern_id in self._compiled:
                continue
            if len(observations) < self._min_obs:
                continue

            candidate = self._evaluate_pattern(pattern_id, observations)
            if self._candidate_eligible(candidate):
                candidates.append(candidate)

        return sorted(candidates, key=lambda c: c.avg_ihsan, reverse=True)

    def _candidate_eligible(self, candidate: ReflexCandidate) -> bool:
        """Apply bridge-local thresholds to a candidate.

        ReflexCandidate.eligible encodes the default constitutional thresholds.
        The bridge must still honor its configured thresholds so tests, dry runs,
        and staged rollout policies can tighten or relax observation counts
        without drifting from the candidate dataclass defaults.
        """
        return (
            candidate.avg_ihsan >= self._ihsan_threshold
            and candidate.reproducibility >= self._repro_threshold
            and candidate.observation_count >= self._min_obs
            and candidate.impact_score > REFLEX_IMPACT_FLOOR
        )

    def mark_compiled(self, pattern_id: str) -> None:
        """Mark a pattern as compiled (prevents re-compilation)."""
        self._compiled.add(pattern_id)
        logger.info("Pattern %s marked as compiled reflex", pattern_id[:16])

    def deny(self, pattern_id: str, reason: str = "") -> None:
        """Add a pattern to the deny-list for 30 days."""
        expiry = datetime.now(timezone.utc) + timedelta(days=DENY_LIST_EXPIRY_DAYS)
        self._deny_list[pattern_id] = expiry
        logger.warning(
            "Pattern %s denied until %s: %s",
            pattern_id[:16],
            expiry.isoformat(),
            reason,
        )

    @property
    def pattern_count(self) -> int:
        return len(self._observations)

    @property
    def total_observations(self) -> int:
        return sum(len(obs) for obs in self._observations.values())

    @property
    def compiled_count(self) -> int:
        return len(self._compiled)

    @property
    def denied_count(self) -> int:
        self._clean_expired_denials()
        return len(self._deny_list)

    def get_status(self) -> Dict[str, Any]:
        """Return bridge status summary."""
        return {
            "patterns_tracked": self.pattern_count,
            "total_observations": self.total_observations,
            "compiled_reflexes": self.compiled_count,
            "active_denials": self.denied_count,
            "eligible_now": len(self.get_eligible_candidates()),
        }

    def _evaluate_pattern(
        self, pattern_id: str, observations: List[TrainingObservation]
    ) -> ReflexCandidate:
        """Evaluate a pattern's readiness for reflex compilation."""
        ihsan_scores = [o.ihsan_score for o in observations]
        snr_scores = [o.snr_score for o in observations]
        successes = [o for o in observations if o.success]

        avg_ihsan = sum(ihsan_scores) / len(ihsan_scores)
        avg_snr = sum(snr_scores) / len(snr_scores)
        reproducibility = len(successes) / len(observations)

        # Impact: how much better than baseline (UNIFIED_SNR_THRESHOLD)?
        impact_score = max(0.0, avg_snr - UNIFIED_SNR_THRESHOLD)

        return ReflexCandidate(
            pattern_id=pattern_id,
            pattern_description=observations[0].task_description,
            source_task=observations[0].task_description,
            avg_ihsan=avg_ihsan,
            avg_snr=avg_snr,
            reproducibility=reproducibility,
            observation_count=len(observations),
            required_observations=self._min_obs,
            impact_score=impact_score,
        )

    def _clean_expired_denials(self) -> None:
        """Remove expired entries from the deny-list."""
        now = datetime.now(timezone.utc)
        expired = [pid for pid, expiry in self._deny_list.items() if expiry <= now]
        for pid in expired:
            del self._deny_list[pid]
            logger.info("Deny-list entry expired for pattern %s", pid[:16])

    @staticmethod
    def _hash_pattern(task_description: str) -> str:
        """Generate a stable pattern ID from task description."""
        return hashlib.sha256(task_description.encode("utf-8")).hexdigest()
