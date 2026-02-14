"""
Improvement Hypothesis Generator — Self-Improvement Through Informed Speculation
===============================================================================

Generates potential system improvements based on observed metrics and patterns.
Each hypothesis includes predicted impact, confidence, risk assessment, and
rollback plans to ensure safe experimentation.

Categories:
- PERFORMANCE: Speed/latency improvements (caching, batching, parallelization)
- QUALITY: Ihsan/SNR improvements (prompts, verification, constraints)
- EFFICIENCY: Resource optimization (memory, tokens, compute scheduling)
- CAPABILITY: New features (pattern recognition, skills, tools)
- RESILIENCE: Fault tolerance (error handling, recovery, redundancy)

Standing on Giants:
- Maturana & Varela (Autopoiesis)
- Deming (Continuous Improvement)
- Shannon (Information Theory)
- Anthropic (Constitutional AI)

Genesis Strict Synthesis v2.2.2
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class HypothesisCategory(str, Enum):
    """Categories of improvement hypotheses."""

    PERFORMANCE = "performance"  # Speed/latency improvements
    QUALITY = "quality"  # Ihsan/SNR improvements
    EFFICIENCY = "efficiency"  # Resource optimization
    CAPABILITY = "capability"  # New features/skills
    RESILIENCE = "resilience"  # Fault tolerance


class RiskLevel(str, Enum):
    """Risk level for hypothesis implementation."""

    LOW = "low"  # Safe to auto-apply
    MEDIUM = "medium"  # Requires review
    HIGH = "high"  # Requires explicit approval


class HypothesisStatus(str, Enum):
    """Status of a hypothesis in its lifecycle."""

    GENERATED = "generated"  # Just created
    VALIDATED = "validated"  # Passed initial checks
    SCHEDULED = "scheduled"  # Scheduled for testing
    TESTING = "testing"  # Currently being tested
    SUCCESSFUL = "successful"  # Proved beneficial
    FAILED = "failed"  # Did not improve
    REJECTED = "rejected"  # Rejected due to constraints
    ROLLED_BACK = "rolled_back"  # Was applied but rolled back


# =============================================================================
# SYSTEM OBSERVATION
# =============================================================================


@dataclass
class SystemObservation:
    """
    Snapshot of system state for hypothesis generation.

    Captures metrics across multiple dimensions that inform
    improvement opportunities.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Performance metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    cache_hit_rate: float = 0.0

    # Quality metrics
    ihsan_score: float = UNIFIED_IHSAN_THRESHOLD
    snr_score: float = UNIFIED_SNR_THRESHOLD
    error_rate: float = 0.0
    verification_failure_rate: float = 0.0

    # Efficiency metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    token_usage_avg: float = 0.0
    batch_utilization: float = 0.0

    # Capability metrics
    skill_coverage: float = 0.0
    pattern_recognition_accuracy: float = 0.0
    tool_success_rate: float = 0.0

    # Resilience metrics
    uptime_percent: float = 100.0
    recovery_time_avg_ms: float = 0.0
    retry_rate: float = 0.0
    circuit_breaker_trips: int = 0

    # Trend indicators (-1 to 1, negative = declining)
    latency_trend: float = 0.0
    quality_trend: float = 0.0
    efficiency_trend: float = 0.0
    error_trend: float = 0.0

    # Context
    observation_window_seconds: float = 60.0
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "performance": {
                "avg_latency_ms": self.avg_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "throughput_rps": self.throughput_rps,
                "cache_hit_rate": self.cache_hit_rate,
            },
            "quality": {
                "ihsan_score": self.ihsan_score,
                "snr_score": self.snr_score,
                "error_rate": self.error_rate,
                "verification_failure_rate": self.verification_failure_rate,
            },
            "efficiency": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "gpu_percent": self.gpu_percent,
                "token_usage_avg": self.token_usage_avg,
                "batch_utilization": self.batch_utilization,
            },
            "capability": {
                "skill_coverage": self.skill_coverage,
                "pattern_recognition_accuracy": self.pattern_recognition_accuracy,
                "tool_success_rate": self.tool_success_rate,
            },
            "resilience": {
                "uptime_percent": self.uptime_percent,
                "recovery_time_avg_ms": self.recovery_time_avg_ms,
                "retry_rate": self.retry_rate,
                "circuit_breaker_trips": self.circuit_breaker_trips,
            },
            "trends": {
                "latency": self.latency_trend,
                "quality": self.quality_trend,
                "efficiency": self.efficiency_trend,
                "error": self.error_trend,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemObservation":
        """Reconstruct from dictionary."""
        perf = data.get("performance", {})
        qual = data.get("quality", {})
        eff = data.get("efficiency", {})
        cap = data.get("capability", {})
        res = data.get("resilience", {})
        trends = data.get("trends", {})

        return cls(
            avg_latency_ms=perf.get("avg_latency_ms", 0.0),
            p95_latency_ms=perf.get("p95_latency_ms", 0.0),
            p99_latency_ms=perf.get("p99_latency_ms", 0.0),
            throughput_rps=perf.get("throughput_rps", 0.0),
            cache_hit_rate=perf.get("cache_hit_rate", 0.0),
            ihsan_score=qual.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD),
            snr_score=qual.get("snr_score", UNIFIED_SNR_THRESHOLD),
            error_rate=qual.get("error_rate", 0.0),
            verification_failure_rate=qual.get("verification_failure_rate", 0.0),
            cpu_percent=eff.get("cpu_percent", 0.0),
            memory_percent=eff.get("memory_percent", 0.0),
            gpu_percent=eff.get("gpu_percent", 0.0),
            token_usage_avg=eff.get("token_usage_avg", 0.0),
            batch_utilization=eff.get("batch_utilization", 0.0),
            skill_coverage=cap.get("skill_coverage", 0.0),
            pattern_recognition_accuracy=cap.get("pattern_recognition_accuracy", 0.0),
            tool_success_rate=cap.get("tool_success_rate", 0.0),
            uptime_percent=res.get("uptime_percent", 100.0),
            recovery_time_avg_ms=res.get("recovery_time_avg_ms", 0.0),
            retry_rate=res.get("retry_rate", 0.0),
            circuit_breaker_trips=res.get("circuit_breaker_trips", 0),
            latency_trend=trends.get("latency", 0.0),
            quality_trend=trends.get("quality", 0.0),
            efficiency_trend=trends.get("efficiency", 0.0),
            error_trend=trends.get("error", 0.0),
        )


# =============================================================================
# HYPOTHESIS
# =============================================================================


@dataclass
class Hypothesis:
    """
    A proposed system improvement with predicted outcomes.

    Each hypothesis represents a testable prediction about how
    a specific change might improve the system, along with
    risk assessment and rollback procedures.
    """

    id: str
    category: HypothesisCategory
    description: str
    predicted_improvement: dict[str, float]  # metric -> delta (positive = better)
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    implementation_plan: list[str]
    rollback_plan: list[str]
    ihsan_impact: float  # Estimated impact on Ihsan score (-1 to 1)
    dependencies: list[str] = field(default_factory=list)

    # Lifecycle tracking
    status: HypothesisStatus = HypothesisStatus.GENERATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validated_at: Optional[datetime] = None
    tested_at: Optional[datetime] = None
    outcome: Optional[dict[str, Any]] = None

    # Pattern matching
    trigger_pattern: str = ""  # What observation pattern triggered this
    similar_past_hypotheses: list[str] = field(default_factory=list)

    def expected_value(self) -> float:
        """
        Calculate expected value of implementing this hypothesis.

        EV = sum(improvements) * confidence - risk_penalty
        """
        improvement_sum = sum(self.predicted_improvement.values())

        # Risk penalty based on level
        risk_penalties = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.1,
            RiskLevel.HIGH: 0.3,
        }
        risk_penalty = risk_penalties.get(self.risk_level, 0.2)

        # Ihsan constraint: negative Ihsan impact is heavily penalized
        ihsan_penalty = 0.0
        if self.ihsan_impact < 0:
            ihsan_penalty = abs(self.ihsan_impact) * 2.0

        return (improvement_sum * self.confidence) - risk_penalty - ihsan_penalty

    def is_safe(self) -> bool:
        """Check if hypothesis is safe to implement without review."""
        return (
            self.risk_level == RiskLevel.LOW
            and self.confidence >= 0.7
            and self.ihsan_impact >= 0
            and self.status in (HypothesisStatus.GENERATED, HypothesisStatus.VALIDATED)
        )

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "description": self.description,
            "predicted_improvement": self.predicted_improvement,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "implementation_plan": self.implementation_plan,
            "rollback_plan": self.rollback_plan,
            "ihsan_impact": self.ihsan_impact,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expected_value": self.expected_value(),
            "is_safe": self.is_safe(),
            "trigger_pattern": self.trigger_pattern,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        """Reconstruct from dictionary."""
        return cls(
            id=data["id"],
            category=HypothesisCategory(data["category"]),
            description=data["description"],
            predicted_improvement=data["predicted_improvement"],
            confidence=data["confidence"],
            risk_level=RiskLevel(data["risk_level"]),
            implementation_plan=data["implementation_plan"],
            rollback_plan=data["rollback_plan"],
            ihsan_impact=data["ihsan_impact"],
            dependencies=data.get("dependencies", []),
            status=HypothesisStatus(data.get("status", "generated")),
            trigger_pattern=data.get("trigger_pattern", ""),
        )


# =============================================================================
# IMPROVEMENT PATTERNS
# =============================================================================


@dataclass
class ImprovementPattern:
    """
    A known pattern that triggers hypothesis generation.

    Patterns encode domain knowledge about what improvements
    typically work for specific problem signatures.
    """

    name: str
    category: HypothesisCategory
    condition: Callable[[SystemObservation], bool]
    hypothesis_template: Callable[[SystemObservation], Hypothesis]
    success_rate: float = 0.5  # Historical success rate
    application_count: int = 0

    def matches(self, observation: SystemObservation) -> bool:
        """Check if this pattern applies to the observation."""
        try:
            return self.condition(observation)
        except Exception as e:
            logger.debug(f"Pattern {self.name} condition error: {e}")
            return False


# =============================================================================
# HYPOTHESIS GENERATOR
# =============================================================================


class HypothesisGenerator:
    """
    Generates improvement hypotheses based on system observations.

    Uses pattern matching against known improvement opportunities,
    augmented by heuristic generation for novel situations. Learns
    from outcomes to improve future hypothesis quality.

    Usage:
        generator = HypothesisGenerator(memory_path=Path("./hypothesis_memory"))

        # Generate hypotheses from observation
        hypotheses = generator.generate(observation)

        # Rank by expected value
        ranked = generator.rank_hypotheses(hypotheses)

        # After testing, learn from outcome
        generator.learn_from_outcome(hypothesis, success=True)
    """

    def __init__(
        self,
        memory_path: Path,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ):
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)

        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold

        # Pattern library
        self._patterns: list[ImprovementPattern] = []
        self._initialize_patterns()

        # Learning from outcomes
        self._successful_patterns: list[Hypothesis] = []
        self._failed_patterns: list[Hypothesis] = []

        # Statistics
        self._total_generated: int = 0
        self._total_tested: int = 0
        self._total_successful: int = 0

        # Load persisted state
        self._load_state()

    def _initialize_patterns(self) -> None:
        """Initialize built-in improvement patterns."""

        # -------------------------------------------------------------------------
        # PERFORMANCE PATTERNS
        # -------------------------------------------------------------------------

        # High latency -> Caching hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="high_latency_caching",
                category=HypothesisCategory.PERFORMANCE,
                condition=lambda obs: obs.avg_latency_ms > 500
                and obs.cache_hit_rate < 0.7,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("perf_cache"),
                    category=HypothesisCategory.PERFORMANCE,
                    description=f"Improve cache strategy to reduce latency (current: {obs.avg_latency_ms:.0f}ms, hit rate: {obs.cache_hit_rate:.1%})",
                    predicted_improvement={
                        "latency_reduction_pct": 0.3,
                        "throughput_increase_pct": 0.2,
                    },
                    confidence=0.75,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=[
                        "Analyze cache miss patterns to identify hot keys",
                        "Increase cache size by 50%",
                        "Switch to adaptive LRU with frequency tracking",
                        "Add cache warming for predictable queries",
                    ],
                    rollback_plan=[
                        "Revert cache configuration to previous settings",
                        "Clear cache and let it rebuild naturally",
                    ],
                    ihsan_impact=0.05,
                    trigger_pattern="high_latency_caching",
                ),
            )
        )

        # Low throughput -> Batching hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="low_throughput_batching",
                category=HypothesisCategory.PERFORMANCE,
                condition=lambda obs: obs.throughput_rps < 10
                and obs.batch_utilization < 0.5,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("perf_batch"),
                    category=HypothesisCategory.PERFORMANCE,
                    description=f"Optimize request batching (current utilization: {obs.batch_utilization:.1%})",
                    predicted_improvement={
                        "throughput_increase_pct": 0.4,
                        "resource_efficiency_pct": 0.25,
                    },
                    confidence=0.8,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=[
                        "Increase batch accumulation window to 50ms",
                        "Implement dynamic batch sizing based on queue depth",
                        "Add batch priority for latency-sensitive requests",
                    ],
                    rollback_plan=[
                        "Reset batch size to default",
                        "Disable dynamic batching",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="low_throughput_batching",
                ),
            )
        )

        # CPU bottleneck -> Parallelization hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="cpu_bottleneck_parallel",
                category=HypothesisCategory.PERFORMANCE,
                condition=lambda obs: obs.cpu_percent > 80 and obs.latency_trend > 0.2,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("perf_parallel"),
                    category=HypothesisCategory.PERFORMANCE,
                    description=f"Increase parallelization to address CPU bottleneck ({obs.cpu_percent:.0f}% usage)",
                    predicted_improvement={
                        "cpu_utilization_efficiency": 0.2,
                        "latency_reduction_pct": 0.15,
                    },
                    confidence=0.65,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Profile CPU-bound operations to identify hotspots",
                        "Implement worker pool with configurable size",
                        "Add async processing for I/O-bound operations",
                        "Enable concurrent request processing up to core count",
                    ],
                    rollback_plan=[
                        "Reduce worker count to previous level",
                        "Revert to synchronous processing if instability occurs",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="cpu_bottleneck_parallel",
                ),
            )
        )

        # -------------------------------------------------------------------------
        # QUALITY PATTERNS
        # -------------------------------------------------------------------------

        # Low Ihsan -> Constraint tightening hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="low_ihsan_constraints",
                category=HypothesisCategory.QUALITY,
                condition=lambda obs: obs.ihsan_score < UNIFIED_IHSAN_THRESHOLD,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("qual_ihsan"),
                    category=HypothesisCategory.QUALITY,
                    description=f"Tighten quality constraints to improve Ihsan score ({obs.ihsan_score:.3f} < {UNIFIED_IHSAN_THRESHOLD})",
                    predicted_improvement={
                        "ihsan_score_delta": UNIFIED_IHSAN_THRESHOLD
                        - obs.ihsan_score
                        + 0.02,
                        "verification_accuracy_pct": 0.1,
                    },
                    confidence=0.7,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=[
                        "Enable stricter verification gates",
                        "Add secondary validation pass for edge cases",
                        "Increase constitutional check depth",
                        "Enable Ihsan audit logging",
                    ],
                    rollback_plan=[
                        "Restore previous verification settings",
                        "Disable secondary validation if latency impact > 20%",
                    ],
                    ihsan_impact=0.1,
                    trigger_pattern="low_ihsan_constraints",
                ),
            )
        )

        # Low SNR -> Signal enhancement hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="low_snr_enhancement",
                category=HypothesisCategory.QUALITY,
                condition=lambda obs: obs.snr_score < UNIFIED_SNR_THRESHOLD,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("qual_snr"),
                    category=HypothesisCategory.QUALITY,
                    description=f"Enhance signal quality (SNR: {obs.snr_score:.3f} < {UNIFIED_SNR_THRESHOLD})",
                    predicted_improvement={
                        "snr_score_delta": UNIFIED_SNR_THRESHOLD - obs.snr_score + 0.05,
                        "noise_reduction_pct": 0.2,
                    },
                    confidence=0.75,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=[
                        "Enable noise filtering in preprocessing",
                        "Add relevance scoring to prioritize signal",
                        "Implement adaptive grounding checks",
                        "Enable diversity deduplication",
                    ],
                    rollback_plan=[
                        "Disable noise filtering",
                        "Revert to default relevance weights",
                    ],
                    ihsan_impact=0.08,
                    trigger_pattern="low_snr_enhancement",
                ),
            )
        )

        # High verification failure -> Prompt improvement hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="verification_failure_prompts",
                category=HypothesisCategory.QUALITY,
                condition=lambda obs: obs.verification_failure_rate > 0.1,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("qual_prompts"),
                    category=HypothesisCategory.QUALITY,
                    description=f"Improve prompts to reduce verification failures ({obs.verification_failure_rate:.1%} failure rate)",
                    predicted_improvement={
                        "verification_success_pct": obs.verification_failure_rate * 0.5,
                        "output_quality_pct": 0.15,
                    },
                    confidence=0.6,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Analyze verification failures for common patterns",
                        "Add explicit constraint reminders to prompts",
                        "Implement chain-of-thought verification",
                        "Enable structured output enforcement",
                    ],
                    rollback_plan=[
                        "Revert to previous prompt templates",
                        "Disable chain-of-thought if latency impact > 30%",
                    ],
                    ihsan_impact=0.12,
                    trigger_pattern="verification_failure_prompts",
                ),
            )
        )

        # -------------------------------------------------------------------------
        # EFFICIENCY PATTERNS
        # -------------------------------------------------------------------------

        # Memory pressure -> Garbage collection hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="memory_pressure_gc",
                category=HypothesisCategory.EFFICIENCY,
                condition=lambda obs: obs.memory_percent > 85,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("eff_memory"),
                    category=HypothesisCategory.EFFICIENCY,
                    description=f"Optimize memory usage (current: {obs.memory_percent:.0f}%)",
                    predicted_improvement={
                        "memory_reduction_pct": 0.2,
                        "gc_pause_reduction_pct": 0.15,
                    },
                    confidence=0.7,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=[
                        "Reduce cache sizes by 30%",
                        "Enable LRU eviction for all caches",
                        "Add memory pressure monitoring",
                        "Implement object pooling for frequent allocations",
                    ],
                    rollback_plan=[
                        "Restore cache sizes",
                        "Disable aggressive eviction if hit rate drops",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="memory_pressure_gc",
                ),
            )
        )

        # High token usage -> Token optimization hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="high_token_usage",
                category=HypothesisCategory.EFFICIENCY,
                condition=lambda obs: obs.token_usage_avg > 2000,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("eff_tokens"),
                    category=HypothesisCategory.EFFICIENCY,
                    description=f"Reduce token usage (avg: {obs.token_usage_avg:.0f} tokens/request)",
                    predicted_improvement={
                        "token_reduction_pct": 0.25,
                        "cost_reduction_pct": 0.25,
                    },
                    confidence=0.65,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Analyze prompts for redundant context",
                        "Implement context compression",
                        "Add semantic caching for repeated patterns",
                        "Enable early stopping for confident responses",
                    ],
                    rollback_plan=[
                        "Restore full context if quality degrades",
                        "Disable compression if verification failures increase",
                    ],
                    ihsan_impact=-0.02,  # Slight risk to quality
                    trigger_pattern="high_token_usage",
                ),
            )
        )

        # Low GPU utilization -> Compute scheduling hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="low_gpu_utilization",
                category=HypothesisCategory.EFFICIENCY,
                condition=lambda obs: obs.gpu_percent < 50 and obs.throughput_rps > 0,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("eff_gpu"),
                    category=HypothesisCategory.EFFICIENCY,
                    description=f"Optimize GPU scheduling (utilization: {obs.gpu_percent:.0f}%)",
                    predicted_improvement={
                        "gpu_utilization_pct": 0.3,
                        "throughput_increase_pct": 0.2,
                    },
                    confidence=0.6,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Enable continuous batching for inference",
                        "Implement kernel fusion for common operations",
                        "Add speculative execution for predictable patterns",
                        "Enable pipelined model loading",
                    ],
                    rollback_plan=[
                        "Disable continuous batching",
                        "Revert to sequential execution",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="low_gpu_utilization",
                ),
            )
        )

        # -------------------------------------------------------------------------
        # CAPABILITY PATTERNS
        # -------------------------------------------------------------------------

        # Low skill coverage -> Skill acquisition hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="low_skill_coverage",
                category=HypothesisCategory.CAPABILITY,
                condition=lambda obs: obs.skill_coverage < 0.7,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("cap_skills"),
                    category=HypothesisCategory.CAPABILITY,
                    description=f"Expand skill coverage (current: {obs.skill_coverage:.1%})",
                    predicted_improvement={
                        "skill_coverage_pct": 0.15,
                        "task_success_pct": 0.1,
                    },
                    confidence=0.5,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Analyze task failures for missing capabilities",
                        "Implement skill discovery from successful patterns",
                        "Add tool integration for common gaps",
                        "Enable skill transfer from related domains",
                    ],
                    rollback_plan=[
                        "Disable new skills if error rate increases",
                        "Revert to core skill set",
                    ],
                    ihsan_impact=0.05,
                    trigger_pattern="low_skill_coverage",
                ),
            )
        )

        # Low pattern accuracy -> Pattern learning hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="low_pattern_accuracy",
                category=HypothesisCategory.CAPABILITY,
                condition=lambda obs: obs.pattern_recognition_accuracy < 0.8,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("cap_patterns"),
                    category=HypothesisCategory.CAPABILITY,
                    description=f"Improve pattern recognition (accuracy: {obs.pattern_recognition_accuracy:.1%})",
                    predicted_improvement={
                        "pattern_accuracy_pct": 0.15,
                        "prediction_quality_pct": 0.1,
                    },
                    confidence=0.55,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Expand pattern library with recent examples",
                        "Implement ensemble pattern matching",
                        "Add contextual pattern weighting",
                        "Enable pattern feedback loop",
                    ],
                    rollback_plan=[
                        "Revert to previous pattern library",
                        "Disable new pattern types",
                    ],
                    ihsan_impact=0.03,
                    trigger_pattern="low_pattern_accuracy",
                ),
            )
        )

        # -------------------------------------------------------------------------
        # RESILIENCE PATTERNS
        # -------------------------------------------------------------------------

        # High error rate -> Retry mechanism hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="high_error_rate_retry",
                category=HypothesisCategory.RESILIENCE,
                condition=lambda obs: obs.error_rate > 0.05 or obs.error_trend > 0.3,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("res_retry"),
                    category=HypothesisCategory.RESILIENCE,
                    description=f"Implement retry mechanisms (error rate: {obs.error_rate:.1%}, trend: {obs.error_trend:+.2f})",
                    predicted_improvement={
                        "error_reduction_pct": 0.3,
                        "success_rate_pct": 0.2,
                    },
                    confidence=0.8,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=[
                        "Add exponential backoff for transient errors",
                        "Implement circuit breaker pattern",
                        "Enable fallback to alternative backends",
                        "Add error classification for targeted retries",
                    ],
                    rollback_plan=[
                        "Disable retries if latency impact > 50%",
                        "Revert circuit breaker thresholds",
                    ],
                    ihsan_impact=0.02,
                    trigger_pattern="high_error_rate_retry",
                ),
            )
        )

        # Slow recovery -> Recovery optimization hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="slow_recovery_optimization",
                category=HypothesisCategory.RESILIENCE,
                condition=lambda obs: obs.recovery_time_avg_ms > 5000,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("res_recovery"),
                    category=HypothesisCategory.RESILIENCE,
                    description=f"Optimize recovery procedures (avg: {obs.recovery_time_avg_ms:.0f}ms)",
                    predicted_improvement={
                        "recovery_time_reduction_pct": 0.4,
                        "availability_pct": 0.05,
                    },
                    confidence=0.65,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Implement warm standby for critical components",
                        "Add health check pre-warming",
                        "Enable partial degradation modes",
                        "Implement checkpoint-based recovery",
                    ],
                    rollback_plan=[
                        "Disable warm standby if resource overhead > 20%",
                        "Revert to cold start recovery",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="slow_recovery_optimization",
                ),
            )
        )

        # Circuit breaker trips -> Redundancy hypothesis
        self._patterns.append(
            ImprovementPattern(
                name="circuit_breaker_redundancy",
                category=HypothesisCategory.RESILIENCE,
                condition=lambda obs: obs.circuit_breaker_trips > 3,
                hypothesis_template=lambda obs: Hypothesis(
                    id=self._generate_id("res_redundancy"),
                    category=HypothesisCategory.RESILIENCE,
                    description=f"Add redundancy for circuit-breaker-prone components ({obs.circuit_breaker_trips} trips)",
                    predicted_improvement={
                        "availability_pct": 0.1,
                        "circuit_breaker_trips_reduction": 0.5,
                    },
                    confidence=0.7,
                    risk_level=RiskLevel.HIGH,
                    implementation_plan=[
                        "Deploy secondary instances for critical paths",
                        "Implement load balancing with health-aware routing",
                        "Add request shadowing for hot standby",
                        "Enable automatic failover",
                    ],
                    rollback_plan=[
                        "Remove secondary instances",
                        "Disable automatic failover",
                        "Revert to single-instance mode",
                    ],
                    ihsan_impact=0.0,
                    dependencies=["infrastructure_capacity"],
                    trigger_pattern="circuit_breaker_redundancy",
                ),
            )
        )

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique hypothesis ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{prefix}_{timestamp}_{self._total_generated}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]

    def generate(self, observation: SystemObservation) -> list[Hypothesis]:
        """
        Generate hypotheses based on system observation.

        Process:
        1. Identify bottlenecks from observation
        2. Match against known improvement patterns
        3. Generate novel hypotheses via heuristics
        4. Filter by Ihsan constraints
        5. Return ranked by expected value

        Args:
            observation: Current system state snapshot

        Returns:
            list of generated hypotheses, ranked by expected value
        """
        hypotheses: list[Hypothesis] = []

        # Step 1: Match known patterns
        for pattern in self._patterns:
            if pattern.matches(observation):
                try:
                    hypothesis = pattern.hypothesis_template(observation)

                    # Adjust confidence based on pattern success rate
                    hypothesis.confidence *= 0.5 + 0.5 * pattern.success_rate

                    # Find similar past hypotheses
                    hypothesis.similar_past_hypotheses = self._find_similar(hypothesis)

                    hypotheses.append(hypothesis)
                    logger.debug(f"Pattern matched: {pattern.name}")

                except Exception as e:
                    logger.warning(
                        f"Pattern {pattern.name} hypothesis generation failed: {e}"
                    )

        # Step 2: Generate novel hypotheses via heuristics
        novel_hypotheses = self._generate_novel_hypotheses(observation)
        hypotheses.extend(novel_hypotheses)

        # Step 3: Filter by Ihsan constraints
        hypotheses = [h for h in hypotheses if h.ihsan_impact >= -0.05]

        # Step 4: Rank by expected value
        hypotheses.sort(key=lambda h: h.expected_value(), reverse=True)

        # Update statistics
        self._total_generated += len(hypotheses)

        logger.info(f"Generated {len(hypotheses)} hypotheses from observation")

        return hypotheses

    def _generate_novel_hypotheses(
        self, observation: SystemObservation
    ) -> list[Hypothesis]:
        """
        Generate novel hypotheses using heuristics.

        Used when no known patterns match but opportunities exist.
        """
        novel: list[Hypothesis] = []

        # Heuristic: Declining trends deserve attention
        if observation.quality_trend < -0.2:
            novel.append(
                Hypothesis(
                    id=self._generate_id("novel_quality"),
                    category=HypothesisCategory.QUALITY,
                    description=f"Address declining quality trend ({observation.quality_trend:+.2f})",
                    predicted_improvement={
                        "quality_stabilization": 0.2,
                        "trend_reversal": 0.3,
                    },
                    confidence=0.5,  # Lower confidence for novel hypotheses
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Analyze recent changes that may have caused decline",
                        "Enable additional quality monitoring",
                        "Consider reverting recent configuration changes",
                    ],
                    rollback_plan=[
                        "Revert to last known good configuration",
                    ],
                    ihsan_impact=0.05,
                    trigger_pattern="novel_quality_trend",
                )
            )

        if observation.efficiency_trend < -0.2:
            novel.append(
                Hypothesis(
                    id=self._generate_id("novel_efficiency"),
                    category=HypothesisCategory.EFFICIENCY,
                    description=f"Address declining efficiency trend ({observation.efficiency_trend:+.2f})",
                    predicted_improvement={
                        "efficiency_stabilization": 0.2,
                        "resource_optimization": 0.15,
                    },
                    confidence=0.45,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=[
                        "Profile resource usage patterns",
                        "Identify newly introduced inefficiencies",
                        "Enable resource leak detection",
                    ],
                    rollback_plan=[
                        "Restart services to clear potential memory leaks",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="novel_efficiency_trend",
                )
            )

        # Heuristic: Compound issues may need combined solutions
        if (
            observation.error_rate > 0.03
            and observation.memory_percent > 80
            and observation.latency_trend > 0.1
        ):
            novel.append(
                Hypothesis(
                    id=self._generate_id("novel_compound"),
                    category=HypothesisCategory.RESILIENCE,
                    description="Address compound stress indicators (errors + memory + latency)",
                    predicted_improvement={
                        "system_stability": 0.25,
                        "error_reduction": 0.15,
                        "memory_reduction": 0.1,
                    },
                    confidence=0.4,
                    risk_level=RiskLevel.HIGH,
                    implementation_plan=[
                        "Reduce load by enabling request rate limiting",
                        "Clear caches to relieve memory pressure",
                        "Enable graceful degradation mode",
                        "Investigate root cause correlation",
                    ],
                    rollback_plan=[
                        "Disable rate limiting",
                        "Restore caches",
                        "Exit degradation mode",
                    ],
                    ihsan_impact=0.0,
                    trigger_pattern="novel_compound_stress",
                )
            )

        return novel

    def _find_similar(self, hypothesis: Hypothesis) -> list[str]:
        """Find similar past hypotheses for learning."""
        similar: list[str] = []

        for past in self._successful_patterns + self._failed_patterns:
            if (
                past.category == hypothesis.category
                and past.trigger_pattern == hypothesis.trigger_pattern
            ):
                similar.append(past.id)

        return similar[:5]  # Limit to 5 most relevant

    def rank_hypotheses(
        self, hypotheses: list[Hypothesis], top_k: Optional[int] = None
    ) -> list[Hypothesis]:
        """
        Rank hypotheses by expected value.

        Args:
            hypotheses: list of hypotheses to rank
            top_k: Optional limit on returned hypotheses

        Returns:
            Hypotheses sorted by expected value (descending)
        """
        ranked = sorted(hypotheses, key=lambda h: h.expected_value(), reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def learn_from_outcome(
        self,
        hypothesis: Hypothesis,
        success: bool,
        actual_improvement: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Update pattern library based on hypothesis outcome.

        Args:
            hypothesis: The tested hypothesis
            success: Whether the hypothesis improved the system
            actual_improvement: Optional actual improvement measurements
        """
        # Update hypothesis status
        hypothesis.tested_at = datetime.now(timezone.utc)
        hypothesis.status = (
            HypothesisStatus.SUCCESSFUL if success else HypothesisStatus.FAILED
        )
        hypothesis.outcome = {
            "success": success,
            "actual_improvement": actual_improvement,
            "predicted_improvement": hypothesis.predicted_improvement,
        }

        # Store in appropriate list
        if success:
            self._successful_patterns.append(hypothesis)
            self._total_successful += 1
        else:
            self._failed_patterns.append(hypothesis)

        self._total_tested += 1

        # Update pattern success rate
        for pattern in self._patterns:
            if pattern.name == hypothesis.trigger_pattern:
                # Exponential moving average for success rate
                pattern.application_count += 1
                alpha = 0.3  # Learning rate
                outcome_value = 1.0 if success else 0.0
                pattern.success_rate = (
                    alpha * outcome_value + (1 - alpha) * pattern.success_rate
                )
                break

        # Persist state
        self._save_state()

        logger.info(
            f"Learned from hypothesis {hypothesis.id}: "
            f"success={success}, pattern={hypothesis.trigger_pattern}"
        )

    def get_successful_patterns(self) -> list[Hypothesis]:
        """Get list of successful hypotheses."""
        return list(self._successful_patterns)

    def get_failed_patterns(self) -> list[Hypothesis]:
        """Get list of failed hypotheses."""
        return list(self._failed_patterns)

    def get_statistics(self) -> dict[str, Any]:
        """Get generator statistics."""
        success_rate = (
            self._total_successful / self._total_tested
            if self._total_tested > 0
            else 0.0
        )

        pattern_stats = {}
        for pattern in self._patterns:
            pattern_stats[pattern.name] = {
                "category": pattern.category.value,
                "success_rate": pattern.success_rate,
                "application_count": pattern.application_count,
            }

        return {
            "total_generated": self._total_generated,
            "total_tested": self._total_tested,
            "total_successful": self._total_successful,
            "success_rate": success_rate,
            "patterns": pattern_stats,
            "successful_patterns_stored": len(self._successful_patterns),
            "failed_patterns_stored": len(self._failed_patterns),
        }

    def _save_state(self) -> None:
        """Persist generator state to disk."""
        state_path = self.memory_path / "hypothesis_generator_state.json"

        state = {
            "statistics": {
                "total_generated": self._total_generated,
                "total_tested": self._total_tested,
                "total_successful": self._total_successful,
            },
            "pattern_states": [
                {
                    "name": p.name,
                    "success_rate": p.success_rate,
                    "application_count": p.application_count,
                }
                for p in self._patterns
            ],
            "successful_patterns": [
                h.to_dict() for h in self._successful_patterns[-100:]
            ],
            "failed_patterns": [h.to_dict() for h in self._failed_patterns[-50:]],
        }

        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save hypothesis generator state: {e}")

    def _load_state(self) -> None:
        """Load generator state from disk."""
        state_path = self.memory_path / "hypothesis_generator_state.json"

        if not state_path.exists():
            return

        try:
            with open(state_path, "r") as f:
                state = json.load(f)

            # Restore statistics
            stats = state.get("statistics", {})
            self._total_generated = stats.get("total_generated", 0)
            self._total_tested = stats.get("total_tested", 0)
            self._total_successful = stats.get("total_successful", 0)

            # Restore pattern states
            pattern_states = {p["name"]: p for p in state.get("pattern_states", [])}
            for pattern in self._patterns:
                if pattern.name in pattern_states:
                    ps = pattern_states[pattern.name]
                    pattern.success_rate = ps.get("success_rate", 0.5)
                    pattern.application_count = ps.get("application_count", 0)

            # Restore hypothesis history
            self._successful_patterns = [
                Hypothesis.from_dict(h) for h in state.get("successful_patterns", [])
            ]
            self._failed_patterns = [
                Hypothesis.from_dict(h) for h in state.get("failed_patterns", [])
            ]

            logger.info(
                f"Loaded hypothesis generator state: {self._total_tested} tested, "
                f"{self._total_successful} successful"
            )

        except Exception as e:
            logger.warning(f"Failed to load hypothesis generator state: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_hypothesis_generator(
    memory_path: Optional[Path] = None,
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    snr_threshold: float = UNIFIED_SNR_THRESHOLD,
) -> HypothesisGenerator:
    """
    Create a configured hypothesis generator.

    Args:
        memory_path: Path for persisting state (default: ./hypothesis_memory)
        ihsan_threshold: Minimum Ihsan threshold
        snr_threshold: Minimum SNR threshold

    Returns:
        Configured HypothesisGenerator instance
    """
    if memory_path is None:
        memory_path = Path("./hypothesis_memory")

    return HypothesisGenerator(
        memory_path=memory_path,
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HypothesisCategory",
    "RiskLevel",
    "HypothesisStatus",
    # Data classes
    "SystemObservation",
    "Hypothesis",
    "ImprovementPattern",
    # Main class
    "HypothesisGenerator",
    # Factory
    "create_hypothesis_generator",
]
