"""
Hierarchical Reasoning Model — Cross-Level Integration Mechanisms

Implements the 5 sophisticated cross-level integration mechanisms that
transform how levels communicate and coordinate in the HRM-Autopoiesis fusion.

Traditional hierarchical reasoning implements simple information flow:
goals descend, evidence ascends. The autopoietic merger introduces
mechanisms that go far beyond this.

HRM PDF Table 4: Cross-Level Integration Mechanisms
  1. Hypothesis Propagation — Bidirectional (Expectations, Evidence)
  2. Validation Cascade     — Bidirectional (Requests, Responses)
  3. Integration Sync       — All-to-All (Knowledge states)
  4. Attention Allocation   — Top-Down (Priority signals)
  5. Surprise Reporting     — Bottom-Up (Anomaly signals)

Standing on Giants:
  - Maturana & Varela (1980) — Structural coupling across scales
  - Friston (2010) — Hierarchical predictive coding
  - Simon (1962) — Near-decomposability in hierarchical systems
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.hrm.abstraction_levels import (
    AbstractionLevel,
    LevelBoundary,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class MessageType(str, Enum):
    """Types of cross-level messages."""

    HYPOTHESIS = "hypothesis"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    INTEGRATION_SYNC = "integration_sync"
    ATTENTION_SIGNAL = "attention_signal"
    SURPRISE_REPORT = "surprise_report"


class PropagationDirection(str, Enum):
    """Direction of message propagation."""

    UPWARD = "upward"  # Evidence ascending (bottom-up)
    DOWNWARD = "downward"  # Goals descending (top-down)
    BOTH = "both"  # Bidirectional


@dataclass
class CrossLevelMessage:
    """
    A message crossing between hierarchical levels.

    Carries metadata for intelligent routing: confidence, source level,
    abstraction requirements, and provenance chain.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_level: AbstractionLevel = AbstractionLevel.PERCEPTUAL
    target_level: AbstractionLevel = AbstractionLevel.OPERATIONAL
    message_type: MessageType = MessageType.HYPOTHESIS
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    provenance: List[str] = field(default_factory=list)
    requires_transform: bool = True

    @property
    def direction(self) -> str:
        """Infer direction from level comparison."""
        if self.target_level > self.source_level:
            return "upward"
        elif self.target_level < self.source_level:
            return "downward"
        return "lateral"

    @property
    def level_distance(self) -> int:
        """Number of levels the message must traverse."""
        return abs(int(self.target_level) - int(self.source_level))


@dataclass
class CascadeResult:
    """Result of a validation cascade across levels."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    requesting_level: AbstractionLevel = AbstractionLevel.TACTICAL
    responses: Dict[AbstractionLevel, Dict[str, Any]] = field(default_factory=dict)
    consensus_reached: bool = False
    aggregate_confidence: float = 0.0
    cascade_depth: int = 0

    @property
    def responding_levels(self) -> List[AbstractionLevel]:
        """Levels that provided validation responses."""
        return sorted(self.responses.keys())


@dataclass
class SyncResult:
    """Result of an integration synchronization across all levels."""

    sync_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    participating_levels: Set[AbstractionLevel] = field(default_factory=set)
    contradictions_found: int = 0
    gaps_identified: int = 0
    transfers_discovered: int = 0
    resolution_actions: List[Dict[str, Any]] = field(default_factory=list)
    sync_quality: float = 0.0

    @property
    def sync_complete(self) -> bool:
        """All 5 levels participated."""
        return len(self.participating_levels) >= len(AbstractionLevel)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-LEVEL BRIDGE — The Integration Engine
# ═══════════════════════════════════════════════════════════════════════════════


class CrossLevelBridge:
    """
    Orchestrates the 5 cross-level integration mechanisms.

    Golden Gem: The Permeable Boundary Principle
      Level boundaries are not walls but membranes — selectively
      permeable and actively regulating what crosses. The membrane
      learns which information should cross, require transformation,
      or be blocked.

    The bridge maintains telemetry on all cross-level traffic to
    enable Level N meta-autopoietic optimization.
    """

    def __init__(
        self,
        boundaries: Optional[List[LevelBoundary]] = None,
        cascade_timeout_ms: float = 5000.0,
        sync_interval_cycles: int = 5,
    ):
        from core.hrm.abstraction_levels import default_boundaries

        self._boundaries = {
            (b.source_level, b.target_level): b
            for b in (boundaries or default_boundaries())
        }
        self._cascade_timeout_ms = cascade_timeout_ms
        self._sync_interval_cycles = sync_interval_cycles

        # Telemetry
        self._message_log: List[CrossLevelMessage] = []
        self._total_messages = 0
        self._blocked_messages = 0
        self._cascade_count = 0
        self._sync_count = 0
        self._resonance_events = 0

    # ─── Mechanism 1: Hypothesis Propagation ───────────────────────────

    def propagate_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        source_level: AbstractionLevel,
        direction: PropagationDirection = PropagationDirection.BOTH,
        confidence: float = 0.5,
    ) -> List[CrossLevelMessage]:
        """
        Propagate a hypothesis to adjacent levels for consideration.

        Lower levels receive hypotheses as EXPECTATIONS that guide perception.
        Upper levels receive hypotheses as EVIDENCE for higher-order reasoning.

        The hypothesis carries metadata: confidence, source level, and
        abstraction mapping requirements. Receiving levels can accept,
        reject, or transform according to their own autopoietic criteria.
        """
        messages = []
        levels = list(AbstractionLevel)
        src_idx = int(source_level)

        targets = []
        if direction in (PropagationDirection.UPWARD, PropagationDirection.BOTH):
            if src_idx + 1 < len(levels):
                targets.append(levels[src_idx + 1])
        if direction in (
            PropagationDirection.DOWNWARD,
            PropagationDirection.BOTH,
        ):
            if src_idx - 1 >= 0:
                targets.append(levels[src_idx - 1])

        for target in targets:
            boundary_key = (source_level, target)
            boundary = self._boundaries.get(boundary_key)

            if boundary and not boundary.should_pass(confidence):
                boundary.record_crossing(passed=False)
                self._blocked_messages += 1
                logger.debug(
                    "Hypothesis blocked at boundary %s→%s (conf=%.2f)",
                    source_level.name,
                    target.name,
                    confidence,
                )
                continue

            if boundary:
                boundary.record_crossing(passed=True)

            msg = CrossLevelMessage(
                source_level=source_level,
                target_level=target,
                message_type=MessageType.HYPOTHESIS,
                payload={
                    "hypothesis": hypothesis,
                    "propagation_type": (
                        "expectation" if target < source_level else "evidence"
                    ),
                },
                confidence=confidence,
                provenance=[f"{source_level.name}→{target.name}"],
            )
            messages.append(msg)
            self._message_log.append(msg)
            self._total_messages += 1

        return messages

    # ─── Mechanism 2: Validation Cascade ───────────────────────────────

    def request_validation(
        self,
        hypothesis: Dict[str, Any],
        requesting_level: AbstractionLevel,
        validation_type: str = "evidence",
        level_states: Optional[Dict[AbstractionLevel, Any]] = None,
    ) -> CascadeResult:
        """
        Request validation from adjacent and extended levels.

        The request specifies what kind of validation is needed:
          - "evidence": Lower levels provide empirical support
          - "coherence": Upper levels check logical consistency
          - "resource": Tactical levels assess resource feasibility

        The cascade propagates until a level can provide validation.
        Lower levels provide evidence-based validation;
        upper levels provide coherence-based validation.
        """
        result = CascadeResult(requesting_level=requesting_level)
        levels = list(AbstractionLevel)
        req_idx = int(requesting_level)

        # Request from lower levels (evidence)
        for i in range(req_idx - 1, -1, -1):
            level = levels[i]
            score = self._simulate_validation(
                level, hypothesis, "evidence", level_states
            )
            result.responses[level] = {
                "type": "evidence",
                "score": score,
                "level": level.name,
            }
            result.cascade_depth += 1
            if score >= 0.8:
                break  # Sufficient evidence found

        # Request from upper levels (coherence)
        for i in range(req_idx + 1, len(levels)):
            level = levels[i]
            score = self._simulate_validation(
                level, hypothesis, "coherence", level_states
            )
            result.responses[level] = {
                "type": "coherence",
                "score": score,
                "level": level.name,
            }
            result.cascade_depth += 1
            if score >= 0.9:
                break  # Strong coherence confirmed

        # Aggregate
        if result.responses:
            scores = [r["score"] for r in result.responses.values()]
            result.aggregate_confidence = sum(scores) / len(scores)
            result.consensus_reached = result.aggregate_confidence >= 0.7

        self._cascade_count += 1
        return result

    # ─── Mechanism 3: Integration Synchronization ─────────────────────

    def synchronize_integration(
        self,
        level_states: Dict[AbstractionLevel, Dict[str, Any]],
    ) -> SyncResult:
        """
        Synchronize integrated knowledge across all levels.

        Not a simple merge — a NEGOTIATION where each level's learned
        patterns must be reconciled with others'. Contradictions are flagged
        for resolution; gaps identified for hypothesis generation;
        opportunities for cross-level insight transfer discovered.

        Synchronization is itself autopoietic — the mechanism learns
        to synchronize better.
        """
        result = SyncResult(
            participating_levels=set(level_states.keys()),
        )

        levels = sorted(level_states.keys())
        if len(levels) < 2:
            result.sync_quality = 1.0
            self._sync_count += 1
            return result

        # Detect contradictions between adjacent levels
        for i in range(len(levels) - 1):
            lower = level_states.get(levels[i], {})
            upper = level_states.get(levels[i + 1], {})

            lower_scores = lower.get("snr_scores", [])
            upper_scores = upper.get("snr_scores", [])

            # Contradiction: lower level high-confidence, upper low
            if lower_scores and upper_scores:
                lower_avg = sum(lower_scores) / len(lower_scores)
                upper_avg = sum(upper_scores) / len(upper_scores)
                if abs(lower_avg - upper_avg) > 0.2:
                    result.contradictions_found += 1
                    result.resolution_actions.append(
                        {
                            "type": "contradiction",
                            "between": (levels[i].name, levels[i + 1].name),
                            "delta": abs(lower_avg - upper_avg),
                        }
                    )

            # Gap: level has no recent hypotheses
            if not lower.get("active_hypotheses"):
                result.gaps_identified += 1

        # Discover transfer opportunities
        for i, level_a in enumerate(levels):
            for level_b in levels[i + 1 :]:
                state_a = level_states.get(level_a, {})
                state_b = level_states.get(level_b, {})
                if state_a.get("insights") and state_b.get("insights"):
                    result.transfers_discovered += 1

        # Quality score: lower contradictions = higher quality
        max_contradictions = max(len(levels) - 1, 1)
        contradiction_penalty = result.contradictions_found / max_contradictions
        result.sync_quality = max(0.0, 1.0 - contradiction_penalty * 0.5)

        self._sync_count += 1
        return result

    # ─── Mechanism 4: Attention Allocation ─────────────────────────────

    def allocate_attention(
        self,
        priority_level: AbstractionLevel,
        target_levels: Optional[List[AbstractionLevel]] = None,
        priority_signal: Dict[str, Any] = None,
    ) -> List[CrossLevelMessage]:
        """
        Top-down attention allocation: higher levels focus lower levels.

        When a strategic level identifies a priority, it sends attention
        signals downward to focus computational resources on relevant
        perceptual and operational processing.
        """
        messages = []
        if target_levels is None:
            # Default: all levels below the priority level
            target_levels = [
                l for l in AbstractionLevel if int(l) < int(priority_level)
            ]

        for target in target_levels:
            msg = CrossLevelMessage(
                source_level=priority_level,
                target_level=target,
                message_type=MessageType.ATTENTION_SIGNAL,
                payload={
                    "priority": priority_signal or {},
                    "focus_type": "strategic_directive",
                },
                confidence=0.9,
                provenance=[f"attention:{priority_level.name}→{target.name}"],
            )
            messages.append(msg)
            self._message_log.append(msg)
            self._total_messages += 1

        return messages

    # ─── Mechanism 5: Surprise Reporting ───────────────────────────────

    def report_surprise(
        self,
        anomaly: Dict[str, Any],
        source_level: AbstractionLevel,
        surprise_magnitude: float = 0.5,
    ) -> List[CrossLevelMessage]:
        """
        Bottom-up surprise reporting: lower levels alert upper levels.

        When a perceptual or operational level detects an anomaly, it
        propagates upward as a surprise signal. The surprise triggers
        hypothesis revision at higher levels.

        Hidden Pattern (HRM PDF): Surprise signals can invert the SNR
        gradient in crisis situations — lower levels may need HIGHER SNR
        to act autonomously while higher levels accept LOWER SNR to
        maintain situational awareness.
        """
        messages = []
        levels = list(AbstractionLevel)
        src_idx = int(source_level)

        # Propagate upward to all higher levels
        for i in range(src_idx + 1, len(levels)):
            target = levels[i]
            distance = i - src_idx

            # Confidence attenuates with distance
            attenuated_confidence = max(0.3, surprise_magnitude * (0.9**distance))

            msg = CrossLevelMessage(
                source_level=source_level,
                target_level=target,
                message_type=MessageType.SURPRISE_REPORT,
                payload={
                    "anomaly": anomaly,
                    "magnitude": surprise_magnitude,
                    "attenuated_magnitude": attenuated_confidence,
                },
                confidence=attenuated_confidence,
                provenance=[f"surprise:{source_level.name}→{target.name}"],
            )
            messages.append(msg)
            self._message_log.append(msg)
            self._total_messages += 1

        return messages

    # ─── Telemetry ─────────────────────────────────────────────────────

    def get_bridge_metrics(self) -> Dict[str, Any]:
        """Return bridge telemetry for Level N observation."""
        return {
            "total_messages": self._total_messages,
            "blocked_messages": self._blocked_messages,
            "pass_rate": (
                (self._total_messages - self._blocked_messages)
                / max(self._total_messages, 1)
            ),
            "cascade_count": self._cascade_count,
            "sync_count": self._sync_count,
            "resonance_events": self._resonance_events,
            "message_type_distribution": self._get_type_distribution(),
            "boundary_health": self._get_boundary_health(),
        }

    def _get_type_distribution(self) -> Dict[str, int]:
        """Count messages by type."""
        dist: Dict[str, int] = {}
        for msg in self._message_log:
            key = msg.message_type.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def _get_boundary_health(self) -> List[Dict[str, Any]]:
        """Report health of each boundary."""
        health = []
        for (src, tgt), boundary in self._boundaries.items():
            total = boundary.message_count + boundary.blocked_count
            health.append(
                {
                    "boundary": f"{src.name}→{tgt.name}",
                    "permeability": boundary.permeability,
                    "messages_passed": boundary.message_count,
                    "messages_blocked": boundary.blocked_count,
                    "utilization": total,
                }
            )
        return health

    def _simulate_validation(
        self,
        level: AbstractionLevel,
        hypothesis: Dict[str, Any],
        validation_type: str,
        level_states: Optional[Dict[AbstractionLevel, Any]],
    ) -> float:
        """
        Simulate validation from a level (placeholder for real integration).

        In production, this would invoke the level's AutopoieticLoop.validate().
        For now, produces a heuristic score based on level and type.
        """
        # Base score from level (higher levels = stricter)
        base = 0.85 - (int(level) * 0.05)

        # Adjust for validation type
        if validation_type == "evidence":
            base += 0.05  # Lower levels have more evidence
        elif validation_type == "coherence":
            base += 0.10  # Upper levels check coherence well

        # Adjust for hypothesis confidence
        hyp_conf = hypothesis.get("confidence", 0.5)
        base = (base + hyp_conf) / 2

        return min(1.0, max(0.0, base))
