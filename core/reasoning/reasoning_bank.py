"""
ReasoningBank Intelligence Engine — Adaptive Learning for Constitutional Agents
═══════════════════════════════════════════════════════════════════════════════════

The intelligence layer that sits between raw experience recording and reflex
compilation. Implements pattern recognition, strategy optimization, meta-learning,
and transfer learning — all gated by constitutional invariants.

Architecture:
    EventBus/Node0 → record_experience() → pattern_recognizer → strategy_optimizer
                                                                     ↓
    ReflexBridge ← get_recommendations() ← meta_learner ← strategy_scores

Data Flow:
    1. record_experience(task, approach, outcome, context, ihsan)
    2. _update_pattern_frequency(task, approach) — frequency tracking
    3. _update_strategy_score(task, approach, outcome) — UCB1 bandit
    4. _check_meta_learning() — learn about learning effectiveness
    5. recommend_strategy(task, context) — best strategy for context
    6. get_precipitable_patterns() → feed to SDPOReflexBridge

Constitutional Gates (§4):
    - Experience with Ihsān < 0.85 → recorded but FLAGGED, never promoted
    - Strategy recommendation requires Ihsān ≥ 0.90 (precipitation floor)
    - Pattern elevation to reflex requires Ihsān ≥ 0.98 (elite gate)
    - Gini check: no single strategy may dominate > 65% of recommendations

Standing on Giants:
    - Prophet Muhammad ﷺ (Ihsān, Hadith Jibril) — excellence as gate
    - Al-Ghazali (intent gate, 1096) — action must be grounded in purpose
    - Shannon (Information Theory, 1948) — SNR as quality gradient
    - Auer et al. (UCB1, 2002) — exploration-exploitation in strategy selection
    - Kahneman (System 1/2, 2011) — reflexes from verified deliberation only
    - Deming (PDCA, 1950) — continuous improvement ratchet
    - Boyd (OODA, 1976) — Observe-Orient-Decide-Act learning loop
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.integration.constants import (
    REFLEX_PRECIPITATION_IHSAN,
    SNR_THRESHOLD_T0_ELITE,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constitutional Thresholds (from constants.py — single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIENCE_MIN_IHSAN = 0.85  # Below this → flagged, never promoted
RECOMMENDATION_MIN_IHSAN = REFLEX_PRECIPITATION_IHSAN  # 0.90
PATTERN_ELEVATION_IHSAN = SNR_THRESHOLD_T0_ELITE  # 0.98
STRATEGY_GINI_CEILING = 0.65  # No single strategy > 65% of recommendations
MIN_EXPERIENCES_FOR_RECOMMENDATION = 3  # Need at least 3 data points
META_LEARNING_WINDOW = 50  # Evaluate meta-learning every N experiences
UCB1_EXPLORATION = 1.41  # sqrt(2) — classic UCB1 exploration constant


# ═══════════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Experience:
    """A single recorded task experience with constitutional scoring."""

    experience_id: str
    task_type: str
    approach: str
    success: bool
    ihsan_score: float
    snr_score: float
    duration_ms: float
    context: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    flagged: bool = False  # True if ihsan < EXPERIENCE_MIN_IHSAN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "task_type": self.task_type,
            "approach": self.approach,
            "success": self.success,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "duration_ms": self.duration_ms,
            "context": self.context,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "flagged": self.flagged,
        }


@dataclass(frozen=True)
class StrategyRecommendation:
    """A recommended strategy for a given task type and context."""

    task_type: str
    approach: str
    score: float  # UCB1 or mean quality score
    confidence: float  # 0.0–1.0 based on observation count
    avg_ihsan: float
    avg_duration_ms: float
    success_rate: float
    observation_count: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "approach": self.approach,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "avg_ihsan": round(self.avg_ihsan, 4),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "observation_count": self.observation_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PatternMatch:
    """A recognized recurring pattern eligible for elevation."""

    pattern_id: str
    task_type: str
    approach: str
    frequency: int
    avg_ihsan: float
    reproducibility: float
    eligible_for_reflex: bool
    evidence: List[str]  # experience_ids that anchor this pattern


@dataclass
class MetaLearningInsight:
    """A meta-learning observation about strategy effectiveness over time."""

    insight_id: str
    observation: str
    confidence: float
    improvement_rate: float  # Positive = getting better
    window_size: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Statistics (mutable, internal)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class _StrategyStats:
    """Internal mutable statistics for a task-type/approach pair."""

    total_count: int = 0
    success_count: int = 0
    ihsan_sum: float = 0.0
    snr_sum: float = 0.0
    duration_sum: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    experience_ids: List[str] = field(default_factory=list)
    recommendation_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def avg_ihsan(self) -> float:
        return self.ihsan_sum / self.total_count if self.total_count > 0 else 0.0

    @property
    def avg_snr(self) -> float:
        return self.snr_sum / self.total_count if self.total_count > 0 else 0.0

    @property
    def avg_duration(self) -> float:
        return self.duration_sum / self.total_count if self.total_count > 0 else 0.0

    @property
    def mean_quality(self) -> float:
        return statistics.mean(self.quality_scores) if self.quality_scores else 0.0

    @property
    def reproducibility(self) -> float:
        """Fraction of outcomes above Ihsān threshold."""
        if self.total_count == 0:
            return 0.0
        above = sum(1 for q in self.quality_scores if q >= RECOMMENDATION_MIN_IHSAN)
        return above / self.total_count


# ═══════════════════════════════════════════════════════════════════════════════
# ReasoningBank Intelligence Engine
# ═══════════════════════════════════════════════════════════════════════════════


class ReasoningBankEngine:
    """Constitutional adaptive learning engine for BIZRA agents.

    Records experiences, recognizes patterns, recommends strategies,
    and tracks meta-learning — all gated by Ihsān invariants.

    Usage:
        engine = ReasoningBankEngine()

        # Record an experience
        exp = engine.record_experience(
            task_type="code_review",
            approach="static_analysis_first",
            success=True,
            ihsan_score=0.96,
            snr_score=0.93,
            duration_ms=150.0,
            context={"language": "python"},
            metrics={"bugs_found": 3},
        )

        # Get recommendation
        rec = engine.recommend_strategy("code_review", {"language": "python"})

        # Check precipitable patterns for reflex bridge
        patterns = engine.get_precipitable_patterns()
    """

    def __init__(
        self,
        exploration_constant: float = UCB1_EXPLORATION,
        meta_window: int = META_LEARNING_WINDOW,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._exploration_c = exploration_constant
        self._meta_window = meta_window
        self._event_bus = event_bus

        # Core state
        self._experiences: List[Experience] = []
        self._strategies: Dict[str, Dict[str, _StrategyStats]] = defaultdict(
            lambda: defaultdict(_StrategyStats)
        )
        self._total_recommendations: int = 0
        self._meta_insights: List[MetaLearningInsight] = []

        # Counters
        self._total_flagged: int = 0
        self._total_promoted: int = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Experience Recording (§4 constitutional gate)
    # ═══════════════════════════════════════════════════════════════════════

    def record_experience(
        self,
        task_type: str,
        approach: str,
        success: bool,
        ihsan_score: float,
        snr_score: float = 0.0,
        duration_ms: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Experience:
        """Record a task experience with constitutional scoring.

        §4: Experiences with Ihsān < 0.85 are recorded but FLAGGED and
        never promoted to strategy recommendations or reflex candidates.
        """
        ctx = context or {}
        mets = metrics or {}
        flagged = ihsan_score < EXPERIENCE_MIN_IHSAN

        exp_id = self._generate_id(task_type, approach, len(self._experiences))
        exp = Experience(
            experience_id=exp_id,
            task_type=task_type,
            approach=approach,
            success=success,
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            duration_ms=duration_ms,
            context=ctx,
            metrics=mets,
            flagged=flagged,
        )

        self._experiences.append(exp)
        if flagged:
            self._total_flagged += 1
            logger.debug(
                "Experience %s flagged (Ihsān %.3f < %.3f)",
                exp_id,
                ihsan_score,
                EXPERIENCE_MIN_IHSAN,
            )
        else:
            self._update_strategy_stats(exp)

        # Meta-learning check every N experiences
        if len(self._experiences) % self._meta_window == 0:
            self._run_meta_learning()

        self._emit_event(
            "reasoning.experience_recorded",
            {
                "experience_id": exp_id,
                "task_type": task_type,
                "approach": approach,
                "ihsan_score": ihsan_score,
                "flagged": flagged,
            },
        )

        return exp

    # ═══════════════════════════════════════════════════════════════════════
    # Strategy Recommendation (UCB1 bandit + Ihsān gate)
    # ═══════════════════════════════════════════════════════════════════════

    def recommend_strategy(
        self,
        task_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategyRecommendation]:
        """Recommend the best strategy for a task type using UCB1.

        §4: Only recommends strategies with avg Ihsān ≥ 0.90.
        Applies Gini ceiling: no single strategy can be recommended
        more than 65% of the time (prevents monoculture).

        Standing on Giants: Auer et al. (UCB1, 2002) — optimal
        exploration-exploitation under uncertainty.
        """
        if task_type not in self._strategies:
            return None

        candidates = self._strategies[task_type]
        if not candidates:
            return None

        # Filter: must have minimum observations
        eligible = {
            approach: stats
            for approach, stats in candidates.items()
            if stats.total_count >= MIN_EXPERIENCES_FOR_RECOMMENDATION
        }

        if not eligible:
            return None

        total_pulls = sum(s.total_count for s in eligible.values())

        # Compute UCB1 score for each approach
        scored: List[Tuple[str, float, _StrategyStats]] = []
        for approach, stats in eligible.items():
            # Quality score: weighted combination of success rate + ihsān
            quality = 0.6 * stats.success_rate + 0.4 * stats.avg_ihsan

            # UCB1 exploration bonus
            if total_pulls > 0 and stats.total_count > 0:
                exploration = self._exploration_c * math.sqrt(
                    math.log(total_pulls) / stats.total_count
                )
            else:
                exploration = self._exploration_c

            ucb_score = quality + exploration
            scored.append((approach, ucb_score, stats))

        # Sort by UCB1 score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Ihsān gate: best approach must meet minimum
        best_approach, best_score, best_stats = scored[0]
        if best_stats.avg_ihsan < RECOMMENDATION_MIN_IHSAN:
            logger.debug(
                "Best strategy %s/%s has Ihsān %.3f < %.3f — suppressed",
                task_type,
                best_approach,
                best_stats.avg_ihsan,
                RECOMMENDATION_MIN_IHSAN,
            )
            return None

        # Gini ceiling check
        if self._total_recommendations > 0:
            rec_ratio = best_stats.recommendation_count / max(
                self._total_recommendations, 1
            )
            if rec_ratio > STRATEGY_GINI_CEILING and len(scored) > 1:
                # Promote second-best to maintain diversity
                best_approach, best_score, best_stats = scored[1]
                if best_stats.avg_ihsan < RECOMMENDATION_MIN_IHSAN:
                    return None

        # Update counters
        best_stats.recommendation_count += 1
        self._total_recommendations += 1

        confidence = min(1.0, best_stats.total_count / 20.0)

        rec = StrategyRecommendation(
            task_type=task_type,
            approach=best_approach,
            score=best_score,
            confidence=confidence,
            avg_ihsan=best_stats.avg_ihsan,
            avg_duration_ms=best_stats.avg_duration,
            success_rate=best_stats.success_rate,
            observation_count=best_stats.total_count,
            reason=self._build_recommendation_reason(
                best_approach, best_stats, confidence
            ),
        )

        self._emit_event("reasoning.strategy_recommended", rec.to_dict())
        return rec

    def compare_strategies(
        self,
        task_type: str,
    ) -> List[StrategyRecommendation]:
        """Compare all strategies for a task type, sorted by quality.

        Returns all strategies with sufficient data, regardless of Ihsān gate.
        Useful for diagnostic reporting.
        """
        if task_type not in self._strategies:
            return []

        results: List[StrategyRecommendation] = []
        total_pulls = sum(s.total_count for s in self._strategies[task_type].values())

        for approach, stats in self._strategies[task_type].items():
            if stats.total_count == 0:
                continue

            quality = 0.6 * stats.success_rate + 0.4 * stats.avg_ihsan
            if total_pulls > 0 and stats.total_count > 0:
                exploration = self._exploration_c * math.sqrt(
                    math.log(total_pulls) / stats.total_count
                )
            else:
                exploration = 0.0

            confidence = min(1.0, stats.total_count / 20.0)

            results.append(
                StrategyRecommendation(
                    task_type=task_type,
                    approach=approach,
                    score=quality + exploration,
                    confidence=confidence,
                    avg_ihsan=stats.avg_ihsan,
                    avg_duration_ms=stats.avg_duration,
                    success_rate=stats.success_rate,
                    observation_count=stats.total_count,
                    reason=f"Compared: {stats.total_count} obs, "
                    f"quality={quality:.3f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Pattern Recognition (frequency + reproducibility)
    # ═══════════════════════════════════════════════════════════════════════

    def get_precipitable_patterns(self) -> List[PatternMatch]:
        """Find patterns eligible for reflex precipitation.

        §2 Helix 3: Pattern must have:
          - Frequency ≥ MIN_EXPERIENCES_FOR_RECOMMENDATION
          - Avg Ihsān ≥ PATTERN_ELEVATION_IHSAN (0.98)
          - Reproducibility ≥ 0.90

        Standing on Giants: Kahneman (2011) — only verified System-2
        judgments may compile to System-1 reflexes.
        """
        patterns: List[PatternMatch] = []

        for task_type, approaches in self._strategies.items():
            for approach, stats in approaches.items():
                if stats.total_count < MIN_EXPERIENCES_FOR_RECOMMENDATION:
                    continue

                eligible = (
                    stats.avg_ihsan >= PATTERN_ELEVATION_IHSAN
                    and stats.reproducibility >= 0.90
                    and stats.success_rate >= 0.90
                )

                if eligible or stats.total_count >= 10:
                    pattern_id = self._pattern_hash(task_type, approach)
                    patterns.append(
                        PatternMatch(
                            pattern_id=pattern_id,
                            task_type=task_type,
                            approach=approach,
                            frequency=stats.total_count,
                            avg_ihsan=stats.avg_ihsan,
                            reproducibility=stats.reproducibility,
                            eligible_for_reflex=eligible,
                            evidence=stats.experience_ids[-10:],
                        )
                    )

        # Sort: eligible first, then by frequency
        patterns.sort(
            key=lambda p: (p.eligible_for_reflex, p.frequency),
            reverse=True,
        )
        return patterns

    def match_patterns(
        self,
        task_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[PatternMatch]:
        """Match known patterns for a specific task type.

        Returns all recognized patterns, ordered by strength.
        """
        all_patterns = self.get_precipitable_patterns()
        return [p for p in all_patterns if p.task_type == task_type]

    # ═══════════════════════════════════════════════════════════════════════
    # Transfer Learning
    # ═══════════════════════════════════════════════════════════════════════

    def transfer_knowledge(
        self,
        from_task: str,
        to_task: str,
        similarity: float = 0.5,
    ) -> int:
        """Transfer strategy knowledge from one task type to another.

        Copies strategy statistics with a similarity discount factor.
        Only transfers strategies that meet Ihsān threshold.

        Returns number of strategies transferred.
        """
        if from_task not in self._strategies:
            return 0

        if similarity <= 0.0 or similarity > 1.0:
            raise ValueError(f"similarity must be in (0.0, 1.0], got {similarity}")

        transferred = 0
        for approach, source_stats in self._strategies[from_task].items():
            if source_stats.avg_ihsan < RECOMMENDATION_MIN_IHSAN:
                continue
            if source_stats.total_count < MIN_EXPERIENCES_FOR_RECOMMENDATION:
                continue

            target_stats = self._strategies[to_task][approach]
            # Discount by similarity factor
            discounted_count = max(1, int(source_stats.total_count * similarity))
            discounted_success = int(source_stats.success_count * similarity)

            target_stats.total_count += discounted_count
            target_stats.success_count += discounted_success
            target_stats.ihsan_sum += source_stats.ihsan_sum * similarity
            target_stats.snr_sum += source_stats.snr_sum * similarity
            target_stats.duration_sum += source_stats.duration_sum * similarity

            # Transfer quality scores with discount
            for qs in source_stats.quality_scores[-discounted_count:]:
                target_stats.quality_scores.append(qs * similarity)

            transferred += 1
            self._total_promoted += 1

        if transferred > 0:
            self._emit_event(
                "reasoning.knowledge_transferred",
                {
                    "from_task": from_task,
                    "to_task": to_task,
                    "similarity": similarity,
                    "strategies_transferred": transferred,
                },
            )

        return transferred

    # ═══════════════════════════════════════════════════════════════════════
    # Meta-Learning
    # ═══════════════════════════════════════════════════════════════════════

    def get_meta_insights(self) -> List[MetaLearningInsight]:
        """Return all meta-learning insights accumulated so far."""
        return list(self._meta_insights)

    def _run_meta_learning(self) -> Optional[MetaLearningInsight]:
        """Analyze recent experiences to learn about learning effectiveness.

        Standing on Giants: Boyd (OODA, 1976) — observe the observation loop.
        """
        window = self._experiences[-self._meta_window :]
        if len(window) < self._meta_window:
            return None

        first_half = window[: self._meta_window // 2]
        second_half = window[self._meta_window // 2 :]

        avg_ihsan_first = statistics.mean(e.ihsan_score for e in first_half)
        avg_ihsan_second = statistics.mean(e.ihsan_score for e in second_half)
        improvement = avg_ihsan_second - avg_ihsan_first

        success_first = sum(1 for e in first_half if e.success) / len(first_half)
        success_second = sum(1 for e in second_half if e.success) / len(second_half)

        confidence = min(1.0, len(window) / 100.0)

        if improvement > 0.01:
            observation = (
                f"Ihsān improving: {avg_ihsan_first:.3f} → {avg_ihsan_second:.3f} "
                f"(+{improvement:.3f}). Success rate: "
                f"{success_first:.2f} → {success_second:.2f}"
            )
        elif improvement < -0.01:
            observation = (
                f"Ihsān declining: {avg_ihsan_first:.3f} → {avg_ihsan_second:.3f} "
                f"({improvement:.3f}). Investigate strategy drift."
            )
        else:
            observation = (
                f"Ihsān stable at ~{avg_ihsan_second:.3f}. "
                f"Success rate: {success_second:.2f}"
            )

        insight = MetaLearningInsight(
            insight_id=f"meta_{len(self._meta_insights):04d}",
            observation=observation,
            confidence=confidence,
            improvement_rate=improvement,
            window_size=self._meta_window,
        )
        self._meta_insights.append(insight)

        self._emit_event(
            "reasoning.meta_learning",
            {
                "insight_id": insight.insight_id,
                "improvement_rate": improvement,
                "confidence": confidence,
            },
        )

        return insight

    # ═══════════════════════════════════════════════════════════════════════
    # Health & Metrics
    # ═══════════════════════════════════════════════════════════════════════

    def health(self) -> Dict[str, Any]:
        """Return engine health summary for observability."""
        total = len(self._experiences)
        task_types = len(self._strategies)
        total_strategies = sum(
            len(approaches) for approaches in self._strategies.values()
        )
        precipitable = len(
            [p for p in self.get_precipitable_patterns() if p.eligible_for_reflex]
        )

        avg_ihsan = 0.0
        if self._experiences:
            avg_ihsan = statistics.mean(e.ihsan_score for e in self._experiences)

        return {
            "total_experiences": total,
            "total_flagged": self._total_flagged,
            "total_promoted": self._total_promoted,
            "task_types": task_types,
            "total_strategies": total_strategies,
            "precipitable_patterns": precipitable,
            "total_recommendations": self._total_recommendations,
            "meta_insights": len(self._meta_insights),
            "avg_ihsan": round(avg_ihsan, 4),
            "constitutional_gates": {
                "experience_min_ihsan": EXPERIENCE_MIN_IHSAN,
                "recommendation_min_ihsan": RECOMMENDATION_MIN_IHSAN,
                "pattern_elevation_ihsan": PATTERN_ELEVATION_IHSAN,
                "strategy_gini_ceiling": STRATEGY_GINI_CEILING,
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return detailed performance metrics."""
        h = self.health()

        # Per-task breakdown
        task_breakdown: Dict[str, Dict[str, Any]] = {}
        for task_type, approaches in self._strategies.items():
            best = max(
                approaches.values(),
                key=lambda s: s.mean_quality,
                default=None,
            )
            task_breakdown[task_type] = {
                "strategies": len(approaches),
                "total_observations": sum(s.total_count for s in approaches.values()),
                "best_quality": best.mean_quality if best else 0.0,
                "best_approach": next(
                    (a for a, s in approaches.items() if s is best), "none"
                ),
            }

        return {
            **h,
            "task_breakdown": task_breakdown,
            "improvement_trend": self._compute_improvement_trend(),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _update_strategy_stats(self, exp: Experience) -> None:
        """Update internal strategy statistics from a non-flagged experience."""
        stats = self._strategies[exp.task_type][exp.approach]
        stats.total_count += 1
        if exp.success:
            stats.success_count += 1
        stats.ihsan_sum += exp.ihsan_score
        stats.snr_sum += exp.snr_score
        stats.duration_sum += exp.duration_ms

        quality = 0.5 * (1.0 if exp.success else 0.0) + 0.5 * exp.ihsan_score
        stats.quality_scores.append(quality)
        stats.experience_ids.append(exp.experience_id)

    def _generate_id(self, task_type: str, approach: str, seq: int) -> str:
        """Generate a deterministic experience ID."""
        raw = f"{task_type}:{approach}:{seq}:{time.monotonic_ns()}"
        return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()

    def _pattern_hash(self, task_type: str, approach: str) -> str:
        """Generate a pattern ID from task + approach."""
        raw = f"pattern:{task_type}:{approach}"
        return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()

    def _build_recommendation_reason(
        self, approach: str, stats: _StrategyStats, confidence: float
    ) -> str:
        """Build human-readable reason for a recommendation."""
        parts = [
            f"UCB1 selected '{approach}'",
            f"({stats.total_count} obs",
            f"success={stats.success_rate:.0%}",
            f"Ihsān={stats.avg_ihsan:.3f}",
            f"confidence={confidence:.2f})",
        ]
        return " ".join(parts)

    def _compute_improvement_trend(self) -> str:
        """Compute overall improvement trend from meta insights."""
        if not self._meta_insights:
            return "insufficient_data"

        recent = self._meta_insights[-5:]
        rates = [i.improvement_rate for i in recent]
        avg_rate = statistics.mean(rates)

        if avg_rate > 0.01:
            return "improving"
        elif avg_rate < -0.01:
            return "declining"
        else:
            return "stable"

    def _emit_event(self, topic: str, payload: Dict[str, Any]) -> None:
        """Emit an event to the EventBus (if wired)."""
        if self._event_bus is None:
            return
        try:
            if hasattr(self._event_bus, "publish"):
                self._event_bus.publish(topic, payload)
            elif hasattr(self._event_bus, "emit"):
                self._event_bus.emit(topic, payload)
        except (RuntimeError, AttributeError, TypeError) as exc:
            logger.debug("EventBus emit failed: %s", exc)
