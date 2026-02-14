"""
Runtime type Definitions — Protocols and TypedDicts
===================================================
type definitions for Sovereign Runtime components ensuring strict typing.

Standing on Giants: type Theory + Protocol Pattern + Python Typing
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)

# =============================================================================
# TYPED DICTS
# =============================================================================


class ReasoningResult(TypedDict, total=False):
    """type definition for reasoning results from GraphOfThoughts."""

    thoughts: list[str]
    conclusion: str
    confidence: float
    depth_reached: int


class SNRResult(TypedDict, total=False):
    """type definition for SNR optimization results."""

    original_length: int
    snr_score: float
    meets_threshold: bool
    optimized: Optional[str]  # RFC-04: Optimized content from SNR pipeline


class ValidationResult(TypedDict, total=False):
    """type definition for Guardian Council validation results."""

    is_valid: bool
    confidence: float
    issues: list[str]


class AutonomousCycleResult(TypedDict, total=False):
    """type definition for autonomous loop cycle results."""

    cycle: int
    decisions: int
    actions: int


class LoopStatus(TypedDict, total=False):
    """type definition for autonomous loop status."""

    running: bool
    cycle: int


# =============================================================================
# PROTOCOLS (Interfaces)
# =============================================================================


@runtime_checkable
class GraphReasonerProtocol(Protocol):
    """Protocol for Graph-of-Thoughts reasoner."""

    async def reason(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        max_depth: int = 5,
    ) -> ReasoningResult: ...


@runtime_checkable
class SNROptimizerProtocol(Protocol):
    """Protocol for SNR maximizer."""

    async def optimize(self, text: str) -> SNRResult: ...


@runtime_checkable
class GuardianProtocol(Protocol):
    """Protocol for Guardian Council validator."""

    async def validate(
        self,
        content: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult: ...


@runtime_checkable
class AutonomousLoopProtocol(Protocol):
    """Protocol for autonomous OODA loop."""

    async def start(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> LoopStatus: ...


@runtime_checkable
class ImpactTrackerProtocol(Protocol):
    """Protocol for impact tracking and sovereignty growth."""

    @property
    def node_id(self) -> str: ...
    @property
    def sovereignty_score(self) -> float: ...
    @property
    def sovereignty_tier(self) -> Any: ...
    @property
    def total_bloom(self) -> float: ...
    @property
    def achievements(self) -> list[str]: ...

    def record_event(
        self,
        category: str,
        action: str,
        bloom: float,
        uers: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any: ...

    def get_progress(self) -> Any: ...
    def flush(self) -> None: ...


# =============================================================================
# ENUMS
# =============================================================================


class RuntimeMode(Enum):
    """Operating mode of the runtime."""

    MINIMAL = auto()  # Basic reasoning only
    STANDARD = auto()  # Reasoning + SNR + Guardian
    AUTONOMOUS = auto()  # Full autonomous operation
    DEBUG = auto()  # Verbose debugging mode
    DEVELOPMENT = auto()  # Development/testing mode


class HealthStatus(Enum):
    """Health status of runtime components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RuntimeConfig:
    """Configuration for the Sovereign Runtime."""

    # Identity
    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}")

    # Thresholds
    ihsan_threshold: float = 0.95
    snr_threshold: float = 0.85

    # Mode
    mode: RuntimeMode = RuntimeMode.STANDARD

    # LLM Backend (env vars override defaults)
    lm_studio_url: str = field(
        default_factory=lambda: os.getenv(
            "LMSTUDIO_URL",
            f"http://{os.getenv('LMSTUDIO_HOST', '192.168.56.1')}:{os.getenv('LMSTUDIO_PORT', '1234')}",
        )
    )
    ollama_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
    )
    default_model: str = "qwen2.5:7b"

    # Reasoning
    max_reasoning_depth: int = 5
    reasoning_timeout: float = 60.0

    # Autonomous loop
    autonomous_cycle_interval: float = 10.0
    max_autonomous_cycles: int = 100
    loop_interval_seconds: float = 10.0

    # Validation
    guardian_quorum: int = 3
    validation_timeout: float = 30.0

    # Persistence
    state_dir: Path = field(default_factory=lambda: Path("sovereign_state"))
    checkpoint_interval: float = 300.0
    enable_persistence: bool = True

    # Cache
    enable_cache: bool = True
    max_cache_entries: int = 1000
    query_timeout_ms: int = (
        300000  # 5 min — reasoning models (R1/QwQ) need extended think time
    )

    # Feature flags
    enable_graph_reasoning: bool = True
    enable_snr_optimization: bool = True
    enable_guardian_validation: bool = True
    enable_autonomous_loop: bool = False
    autonomous_enabled: bool = False  # Alias for enable_autonomous_loop

    # Proactive Execution Kernel (PEK)
    enable_proactive_kernel: bool = False
    proactive_kernel_cycle_seconds: float = 5.0
    proactive_kernel_min_confidence: float = 0.58
    proactive_kernel_min_auto_confidence: float = 0.74
    proactive_kernel_base_tau: float = 0.55
    proactive_kernel_auto_execute_tau: float = 0.75
    proactive_kernel_queue_silent_tau: float = 0.35
    proactive_kernel_attention_budget_capacity: float = 8.0
    proactive_kernel_attention_recovery_per_cycle: float = 0.75
    proactive_kernel_emit_events: bool = False
    proactive_kernel_event_topic: str = "pek.proof.block"

    # Zero Point Kernel (trusted bootstrap preflight)
    enable_zpk_preflight: bool = False
    zpk_manifest_uri: str = ""
    zpk_release_public_key: str = ""
    zpk_allowed_versions: list[str] = field(default_factory=list)
    zpk_min_policy_version: int = 1
    zpk_min_ihsan_policy: float = 0.95
    zpk_emit_bootstrap_events: bool = False
    zpk_event_topic: str = "zpk.bootstrap.receipt"

    @classmethod
    def minimal(cls) -> "RuntimeConfig":
        """Create minimal configuration."""
        return cls(
            mode=RuntimeMode.MINIMAL,
            enable_graph_reasoning=False,
            enable_snr_optimization=False,
            enable_guardian_validation=False,
            enable_autonomous_loop=False,
            enable_proactive_kernel=False,
            enable_zpk_preflight=False,
        )

    @classmethod
    def standard(cls) -> "RuntimeConfig":
        """Create standard configuration."""
        return cls(mode=RuntimeMode.STANDARD)

    @classmethod
    def observer(cls) -> "RuntimeConfig":
        """Create observer configuration — PEK monitors but never auto-executes."""
        return cls(
            mode=RuntimeMode.STANDARD,
            enable_proactive_kernel=True,
            proactive_kernel_cycle_seconds=10.0,
            proactive_kernel_min_confidence=0.74,
            proactive_kernel_min_auto_confidence=0.95,  # Very high → proposals only
            proactive_kernel_auto_execute_tau=0.99,  # Effectively disables auto-execute
            proactive_kernel_base_tau=0.55,
            proactive_kernel_queue_silent_tau=0.35,
        )

    @classmethod
    def autonomous(cls) -> "RuntimeConfig":
        """Create autonomous configuration."""
        return cls(
            mode=RuntimeMode.AUTONOMOUS,
            enable_autonomous_loop=True,
            enable_proactive_kernel=True,
        )


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class RuntimeMetrics:
    """Metrics for runtime operations."""

    # Counters
    queries_processed: int = 0
    queries_succeeded: int = 0
    queries_failed: int = 0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0

    # Reasoning
    reasoning_calls: int = 0
    reasoning_avg_depth: float = 0.0

    # SNR
    snr_optimizations: int = 0
    snr_avg_improvement: float = 0.0

    # Validation
    validations: int = 0
    validation_pass_rate: float = 0.0

    # Autonomous
    autonomous_cycles: int = 0
    autonomous_decisions: int = 0
    autonomous_actions: int = 0

    # Timing
    avg_query_time_ms: float = 0.0
    total_uptime_seconds: float = 0.0
    started_at: Optional[datetime] = None  # RFC-02 FIX: Declared so runtime can set it

    # Quality
    current_ihsan_score: float = 0.0
    current_snr_score: float = 0.0

    # ── Computed properties ───────────────────────────────────
    # Standing on: Dijkstra — single source of truth; derived values
    # MUST NOT be stored, only computed from canonical counters.

    @property
    def total_queries(self) -> int:
        """Alias for queries_processed (backward-compat)."""
        return self.queries_processed

    def success_rate(self) -> float:
        """Fraction of queries that succeeded, 0.0 when no queries."""
        if self.queries_processed == 0:
            return 0.0
        return self.queries_succeeded / self.queries_processed

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of cache lookups that were hits, 0.0 when no lookups."""
        total_cache_lookups = self.cache_hits + self.cache_misses
        if total_cache_lookups == 0:
            return 0.0
        return self.cache_hits / total_cache_lookups

    @property
    def health_score(self) -> float:
        """Composite health: 40% SNR + 40% Ihsan + 20% success rate.

        Standing on: Shannon (SNR) + Ihsan (constitutional quality).
        """
        return (
            0.4 * self.current_snr_score
            + 0.4 * self.current_ihsan_score
            + 0.2 * self.success_rate()
        )

    def to_prometheus(self, include_help: bool = True) -> str:
        """Render runtime metrics in Prometheus exposition format.

        This is the single source of truth for API metrics serialization.
        """
        series = [
            (
                "sovereign_queries_total",
                "counter",
                "Total queries processed",
                f"{self.total_queries}",
            ),
            (
                "sovereign_query_success_rate",
                "gauge",
                "Query success rate",
                f"{self.success_rate():.4f}",
            ),
            (
                "sovereign_snr_score",
                "gauge",
                "Current SNR score",
                f"{self.current_snr_score:.4f}",
            ),
            (
                "sovereign_ihsan_score",
                "gauge",
                "Current Ihsan score",
                f"{self.current_ihsan_score:.4f}",
            ),
            (
                "sovereign_avg_query_time_ms",
                "gauge",
                "Average query time",
                f"{self.avg_query_time_ms:.2f}",
            ),
            (
                "sovereign_health_score",
                "gauge",
                "System health score",
                f"{self.health_score:.4f}",
            ),
            (
                "sovereign_reasoning_calls_total",
                "counter",
                "Total Graph-of-Thought reasoning calls",
                f"{self.reasoning_calls}",
            ),
            (
                "sovereign_reasoning_avg_depth",
                "gauge",
                "Average Graph-of-Thought depth",
                f"{self.reasoning_avg_depth:.2f}",
            ),
            (
                "sovereign_validation_pass_rate",
                "gauge",
                "Guardian validation pass rate",
                f"{self.validation_pass_rate:.4f}",
            ),
            (
                "sovereign_autonomous_cycles_total",
                "counter",
                "Total autonomous cycles",
                f"{self.autonomous_cycles}",
            ),
            (
                "sovereign_autonomous_decisions_total",
                "counter",
                "Total autonomous decisions",
                f"{self.autonomous_decisions}",
            ),
            (
                "sovereign_autonomous_actions_total",
                "counter",
                "Total autonomous actions",
                f"{self.autonomous_actions}",
            ),
            (
                "sovereign_cache_hits_total",
                "counter",
                "Total cache hits",
                f"{self.cache_hits}",
            ),
            (
                "sovereign_cache_misses_total",
                "counter",
                "Total cache misses",
                f"{self.cache_misses}",
            ),
            (
                "sovereign_cache_hit_rate",
                "gauge",
                "Cache hit rate",
                f"{self.cache_hit_rate:.4f}",
            ),
        ]

        lines: list[str] = []
        for idx, (name, metric_type, help_text, value) in enumerate(series):
            if include_help:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value}")
            if include_help and idx < len(series) - 1:
                lines.append("")

        return "\n".join(lines)

    def update_query_stats(self, success: bool, duration_ms: float) -> None:
        """Update query statistics."""
        self.queries_processed += 1
        if success:
            self.queries_succeeded += 1
        else:
            self.queries_failed += 1

        # Rolling average
        n = self.queries_processed
        self.avg_query_time_ms = (self.avg_query_time_ms * (n - 1) + duration_ms) / n

    def update_reasoning_stats(self, depth: int) -> None:
        """Update reasoning statistics."""
        self.reasoning_calls += 1
        n = self.reasoning_calls
        self.reasoning_avg_depth = (self.reasoning_avg_depth * (n - 1) + depth) / n

    def update_snr_stats(self, improvement: float) -> None:
        """Update SNR statistics."""
        self.snr_optimizations += 1
        n = self.snr_optimizations
        self.snr_avg_improvement = (
            self.snr_avg_improvement * (n - 1) + improvement
        ) / n

    def update_validation_stats(self, passed: bool) -> None:
        """Update validation statistics."""
        self.validations += 1
        # Rolling pass rate
        if passed:
            new_passes = self.validation_pass_rate * (self.validations - 1) + 1
        else:
            new_passes = self.validation_pass_rate * (self.validations - 1)
        self.validation_pass_rate = new_passes / self.validations

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "queries": {
                "processed": self.queries_processed,
                "succeeded": self.queries_succeeded,
                "failed": self.queries_failed,
                "avg_time_ms": round(self.avg_query_time_ms, 2),
            },
            "reasoning": {
                "calls": self.reasoning_calls,
                "avg_depth": round(self.reasoning_avg_depth, 2),
            },
            "snr": {
                "optimizations": self.snr_optimizations,
                "avg_improvement": round(self.snr_avg_improvement, 3),
            },
            "validation": {
                "count": self.validations,
                "pass_rate": round(self.validation_pass_rate, 3),
            },
            "autonomous": {
                "cycles": self.autonomous_cycles,
                "decisions": self.autonomous_decisions,
                "actions": self.autonomous_actions,
            },
            "quality": {
                "ihsan_score": round(self.current_ihsan_score, 3),
                "snr_score": round(self.current_snr_score, 3),
            },
            "uptime_seconds": round(self.total_uptime_seconds, 1),
        }


# =============================================================================
# QUERY/RESULT TYPES
# =============================================================================


@dataclass
class SovereignQuery:
    """A query to the Sovereign Runtime."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    require_reasoning: bool = True
    require_snr: bool = True
    require_validation: bool = False
    timeout: float = 60.0
    created_at: datetime = field(default_factory=datetime.now)

    # Phase 21: Multi-user identity
    user_id: str = ""  # empty = anonymous / single-user mode


@dataclass
class SovereignResult:
    """Result from the Sovereign Runtime."""

    query_id: str = ""
    success: bool = False
    response: str = ""

    # Reasoning
    reasoning_used: bool = False
    reasoning_depth: int = 0
    thoughts: list[str] = field(default_factory=list)

    # Quality scores
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    snr_ok: bool = False

    # Validation
    validated: bool = False
    validation_passed: bool = False
    validation_issues: list[str] = field(default_factory=list)

    # Timing
    processing_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    # Model attribution
    model_used: Optional[str] = None  # set during query processing; None = no LLM

    # Spearpoint: content-addressed graph artifact + claim provenance
    graph_hash: Optional[str] = None
    claim_tags: dict[str, str] = field(default_factory=dict)

    # Phase 21: Multi-user identity
    user_id: str = ""  # populated when auth middleware resolves user

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "query_id": self.query_id,
            "success": self.success,
            "response": self.response,
            "reasoning": {
                "used": self.reasoning_used,
                "depth": self.reasoning_depth,
                "thoughts": self.thoughts,
            },
            "quality": {
                "ihsan_score": round(self.ihsan_score, 3),
                "snr_score": round(self.snr_score, 3),
                "snr_ok": self.snr_ok,
            },
            "validation": {
                "performed": self.validated,
                "passed": self.validation_passed,
                "issues": self.validation_issues,
            },
            "processing_time_ms": round(self.processing_time_ms, 2),
            "graph_hash": self.graph_hash,
            "user_id": self.user_id,
            "error": self.error,
        }


# Need uuid for SovereignQuery

__all__ = [
    # TypedDicts
    "ReasoningResult",
    "SNRResult",
    "ValidationResult",
    "AutonomousCycleResult",
    "LoopStatus",
    # Protocols
    "GraphReasonerProtocol",
    "SNROptimizerProtocol",
    "GuardianProtocol",
    "AutonomousLoopProtocol",
    # Enums
    "RuntimeMode",
    "HealthStatus",
    # Config
    "RuntimeConfig",
    "RuntimeMetrics",
    # Query/Result
    "SovereignQuery",
    "SovereignResult",
]
