"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗   ██╗███╗   ██╗████████╗██╗███╗   ███╗███████╗                ║
║   ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██║████╗ ████║██╔════╝                ║
║   ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║██╔████╔██║█████╗                  ║
║   ██╔══██╗██║   ██║██║╚██╗██║   ██║   ██║██║╚██╔╝██║██╔══╝                  ║
║   ██║  ██║╚██████╔╝██║ ╚████║   ██║   ██║██║ ╚═╝ ██║███████╗                ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝                ║
║                                                                              ║
║                    SOVEREIGN UNIFIED RUNTIME v1.0                            ║
║         The Apex Integration — All Components, One Interface                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   "The whole is greater than the sum of its parts." — Aristotle              ║
║                                                                              ║
║   This runtime unifies:                                                      ║
║   • SovereignEngine (Core Reasoning)                                         ║
║   • GraphOfThoughts (Multi-Path Exploration)                                 ║
║   • SNRMaximizer (Signal Quality Enforcement)                                ║
║   • GuardianCouncil (Byzantine Validation)                                   ║
║   • AutonomousLoop (OODA Cycle)                                              ║
║   • SovereignOrchestrator (Task Decomposition)                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sovereign.runtime")

# Type variables
T = TypeVar("T")
QueryResult = TypeVar("QueryResult")


# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

class RuntimeMode(Enum):
    """Operating modes for the runtime."""
    DEVELOPMENT = auto()    # Verbose logging, relaxed thresholds
    PRODUCTION = auto()     # Strict thresholds, optimized performance
    AUTONOMOUS = auto()     # Full autonomous operation with decision loop
    SUPERVISED = auto()     # Human-in-the-loop for critical decisions
    FEDERATED = auto()      # Connected to P2P network


class HealthStatus(Enum):
    """System health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class RuntimeConfig:
    """
    Unified configuration for the Sovereign Runtime.
    All thresholds aligned with Ihsān principles.
    """
    # Identity
    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}")
    node_name: str = "BIZRA-Sovereign"
    version: str = "1.0.0"

    # Operating Mode
    mode: RuntimeMode = RuntimeMode.PRODUCTION

    # Quality Thresholds (Ihsān)
    snr_threshold: float = 0.95
    ihsan_threshold: float = 0.95
    confidence_threshold: float = 0.85
    consensus_threshold: float = 0.67  # 2/3 majority

    # Performance Limits
    max_concurrent_queries: int = 10
    max_thought_depth: int = 5
    max_reasoning_time_ms: int = 30000
    query_timeout_ms: int = 60000

    # Autonomous Loop
    autonomous_enabled: bool = True
    loop_interval_seconds: float = 5.0
    max_decisions_per_cycle: int = 3
    require_rollback_plans: bool = True

    # Guardian Council
    guardian_count: int = 8
    min_guardian_agreement: int = 5  # 5/8 = 62.5%

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    max_cache_entries: int = 1000

    # Persistence
    state_dir: Path = field(default_factory=lambda: Path("./sovereign_state"))
    enable_persistence: bool = True
    checkpoint_interval_seconds: int = 60

    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = True
    log_level: str = "INFO"


# =============================================================================
# RUNTIME METRICS
# =============================================================================

@dataclass
class RuntimeMetrics:
    """Real-time metrics for the runtime."""
    # Counters
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Autonomous Loop
    autonomous_cycles: int = 0
    decisions_made: int = 0
    decisions_approved: int = 0
    decisions_rejected: int = 0

    # Guardian Council
    council_invocations: int = 0
    council_approvals: int = 0
    council_rejections: int = 0

    # Timing (milliseconds)
    avg_query_time_ms: float = 0.0
    avg_reasoning_time_ms: float = 0.0
    avg_validation_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0

    # Quality
    current_snr: float = 0.0
    current_ihsan: float = 0.0
    health_score: float = 1.0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    last_query_at: Optional[datetime] = None
    last_checkpoint_at: Optional[datetime] = None

    def success_rate(self) -> float:
        """Calculate query success rate."""
        total = self.successful_queries + self.failed_queries
        return self.successful_queries / total if total > 0 else 1.0

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def decision_approval_rate(self) -> float:
        """Calculate autonomous decision approval rate."""
        total = self.decisions_approved + self.decisions_rejected
        return self.decisions_approved / total if total > 0 else 1.0

    def uptime_seconds(self) -> float:
        """Calculate runtime uptime."""
        return (datetime.now() - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "queries": {
                "total": self.total_queries,
                "successful": self.successful_queries,
                "failed": self.failed_queries,
                "success_rate": f"{self.success_rate():.2%}",
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": f"{self.cache_hit_rate():.2%}",
            },
            "autonomous": {
                "cycles": self.autonomous_cycles,
                "decisions_made": self.decisions_made,
                "approval_rate": f"{self.decision_approval_rate():.2%}",
            },
            "timing": {
                "avg_query_ms": f"{self.avg_query_time_ms:.1f}",
                "avg_reasoning_ms": f"{self.avg_reasoning_time_ms:.1f}",
                "p99_query_ms": f"{self.p99_query_time_ms:.1f}",
            },
            "quality": {
                "snr": f"{self.current_snr:.3f}",
                "ihsan": f"{self.current_ihsan:.3f}",
                "health": f"{self.health_score:.3f}",
            },
            "uptime_seconds": self.uptime_seconds(),
        }


# =============================================================================
# QUERY & RESPONSE
# =============================================================================

@dataclass
class SovereignQuery:
    """A query to the Sovereign Runtime."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Query options
    require_reasoning: bool = True
    require_validation: bool = True
    max_depth: int = 3
    timeout_ms: int = 30000

    # Metadata
    source: str = "api"
    priority: int = 5  # 1-10, higher = more important
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SovereignResult:
    """Result from the Sovereign Runtime."""
    query_id: str = ""
    success: bool = False

    # Response
    answer: str = ""
    confidence: float = 0.0
    reasoning_path: List[str] = field(default_factory=list)

    # Quality metrics
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    guardian_verdict: str = ""

    # Timing
    total_time_ms: float = 0.0
    reasoning_time_ms: float = 0.0
    validation_time_ms: float = 0.0

    # Metadata
    model_used: str = ""
    cached: bool = False
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)

    def meets_ihsan(self, threshold: float = 0.95) -> bool:
        """Check if result meets Ihsān threshold."""
        return self.ihsan_score >= threshold and self.snr_score >= threshold


# =============================================================================
# COMPONENT STUBS (for standalone operation)
# =============================================================================

class ComponentStub:
    """Base stub for components when full implementation unavailable."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"sovereign.{name}")

    async def initialize(self) -> bool:
        self.logger.info(f"{self.name} initialized (stub mode)")
        return True

    async def shutdown(self) -> None:
        self.logger.info(f"{self.name} shutdown")


class GraphReasonerStub(ComponentStub):
    """Stub for GraphOfThoughts when numpy unavailable."""

    def __init__(self):
        super().__init__("graph_reasoner")

    async def reason(
        self,
        query: str,
        context: Dict[str, Any],
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Simplified reasoning without full graph."""
        thoughts = [
            f"Understanding query: {query[:50]}...",
            "Analyzing context and constraints",
            "Synthesizing response with Ihsān principles",
        ]
        return {
            "thoughts": thoughts,
            "conclusion": f"Reasoned response for: {query}",
            "confidence": 0.88,
            "depth_reached": min(max_depth, len(thoughts)),
        }


class SNROptimizerStub(ComponentStub):
    """Stub for SNRMaximizer when numpy unavailable."""

    def __init__(self, threshold: float = 0.95):
        super().__init__("snr_optimizer")
        self.threshold = threshold

    async def optimize(self, content: str) -> Dict[str, Any]:
        """Simplified SNR optimization."""
        # Heuristic SNR based on content characteristics
        word_count = len(content.split())
        unique_ratio = len(set(content.lower().split())) / max(word_count, 1)

        snr = min(0.98, 0.7 + unique_ratio * 0.28)

        return {
            "original_length": len(content),
            "snr_score": snr,
            "meets_threshold": snr >= self.threshold,
            "optimized": content,
        }


class GuardianStub(ComponentStub):
    """Stub for GuardianCouncil."""

    def __init__(self, guardian_count: int = 8):
        super().__init__("guardian_council")
        self.guardian_count = guardian_count

    async def validate(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplified validation."""
        # Simulate guardian votes
        approvals = self.guardian_count - 1  # Typically approve

        return {
            "approved": approvals >= (self.guardian_count * 2 // 3),
            "votes": {
                "approve": approvals,
                "reject": self.guardian_count - approvals,
            },
            "consensus_score": approvals / self.guardian_count,
            "verdict": "APPROVED" if approvals >= 5 else "REJECTED",
        }


class AutonomousLoopStub(ComponentStub):
    """Stub for AutonomousLoop."""

    def __init__(self, interval: float = 5.0):
        super().__init__("autonomous_loop")
        self.interval = interval
        self.cycle_count = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one autonomous cycle."""
        self.cycle_count += 1
        return {
            "cycle": self.cycle_count,
            "state": "REFLECTING",
            "decisions": 0,
            "health": 0.95,
        }

    async def _loop(self):
        """Main loop."""
        while self._running:
            await self.run_cycle()
            await asyncio.sleep(self.interval)

    def start(self) -> asyncio.Task:
        """Start the loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        return self._task

    def stop(self):
        """Stop the loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "cycles": self.cycle_count,
        }


# =============================================================================
# SOVEREIGN RUNTIME
# =============================================================================

class SovereignRuntime:
    """
    The Unified Sovereign Runtime.

    Integrates all sovereign components into a cohesive system with:
    - Lifecycle management (init, run, shutdown)
    - Query processing with full reasoning pipeline
    - Autonomous operation loop
    - Real-time metrics and health monitoring
    - Graceful degradation when components unavailable

    Usage:
        async with SovereignRuntime.create() as runtime:
            result = await runtime.query("What is the meaning of sovereignty?")
            print(result.answer)
    """

    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.metrics = RuntimeMetrics()
        self.logger = logging.getLogger("sovereign.runtime")

        # State
        self._initialized = False
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialized lazily)
        self._graph_reasoner: Optional[Any] = None
        self._snr_optimizer: Optional[Any] = None
        self._guardian_council: Optional[Any] = None
        self._autonomous_loop: Optional[Any] = None
        self._orchestrator: Optional[Any] = None

        # Timing data for percentile calculations
        self._query_times: List[float] = []

        # Cache
        self._cache: Dict[str, SovereignResult] = {}

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        config: Optional[RuntimeConfig] = None
    ) -> AsyncIterator["SovereignRuntime"]:
        """
        Create and manage runtime lifecycle.

        Usage:
            async with SovereignRuntime.create() as runtime:
                result = await runtime.query("Hello")
        """
        runtime = cls(config)
        try:
            await runtime.initialize()
            yield runtime
        finally:
            await runtime.shutdown()

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME INITIALIZING")
        self.logger.info("=" * 60)
        self.logger.info(f"Node ID: {self.config.node_id}")
        self.logger.info(f"Mode: {self.config.mode.name}")
        self.logger.info(f"Ihsān Threshold: {self.config.ihsan_threshold}")

        # Try to load full components, fall back to stubs
        await self._init_components()

        # Start autonomous loop if enabled
        if self.config.autonomous_enabled:
            await self._start_autonomous_loop()

        # Setup signal handlers
        self._setup_signal_handlers()

        self._initialized = True
        self._running = True
        self.metrics.started_at = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME READY")
        self.logger.info("=" * 60)

    async def _init_components(self) -> None:
        """Initialize components with graceful fallback."""

        # Try full GraphOfThoughts
        try:
            from .graph_reasoner import GraphOfThoughts
            self._graph_reasoner = GraphOfThoughts()
            self.logger.info("✓ GraphOfThoughts loaded (full)")
        except ImportError:
            self._graph_reasoner = GraphReasonerStub()
            self.logger.warning("⚠ GraphOfThoughts unavailable, using stub")

        # Try full SNRMaximizer
        try:
            from .snr_maximizer import SNRMaximizer
            self._snr_optimizer = SNRMaximizer(threshold=self.config.snr_threshold)
            self.logger.info("✓ SNRMaximizer loaded (full)")
        except ImportError:
            self._snr_optimizer = SNROptimizerStub(self.config.snr_threshold)
            self.logger.warning("⚠ SNRMaximizer unavailable, using stub")

        # Try full GuardianCouncil
        try:
            from .guardian_council import GuardianCouncil
            self._guardian_council = GuardianCouncil()
            self.logger.info("✓ GuardianCouncil loaded (full)")
        except ImportError:
            self._guardian_council = GuardianStub(self.config.guardian_count)
            self.logger.warning("⚠ GuardianCouncil unavailable, using stub")

        # Try full AutonomousLoop
        try:
            from .autonomy import AutonomousLoop, DecisionGate
            gate = DecisionGate(ihsan_threshold=self.config.ihsan_threshold)
            self._autonomous_loop = AutonomousLoop(
                decision_gate=gate,
                snr_threshold=self.config.snr_threshold,
                ihsan_threshold=self.config.ihsan_threshold,
                cycle_interval=self.config.loop_interval_seconds,
            )
            self.logger.info("✓ AutonomousLoop loaded (full)")
        except ImportError:
            self._autonomous_loop = AutonomousLoopStub(
                self.config.loop_interval_seconds
            )
            self.logger.warning("⚠ AutonomousLoop unavailable, using stub")

    async def _start_autonomous_loop(self) -> None:
        """Start the autonomous operation loop."""
        if self._autonomous_loop:
            self._autonomous_loop.start()
            self.logger.info("Autonomous loop started")

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.shutdown())
                )
        except (NotImplementedError, RuntimeError):
            # Windows doesn't support add_signal_handler
            pass

    async def shutdown(self) -> None:
        """Gracefully shutdown the runtime."""
        if not self._running:
            return

        self.logger.info("Initiating graceful shutdown...")
        self._running = False

        # Stop autonomous loop
        if self._autonomous_loop:
            self._autonomous_loop.stop()

        # Checkpoint state if persistence enabled
        if self.config.enable_persistence:
            await self._checkpoint()

        self._shutdown_event.set()
        self.logger.info("Sovereign Runtime shutdown complete")

    async def wait_for_shutdown(self) -> None:
        """Wait until shutdown is complete."""
        await self._shutdown_event.wait()

    # -------------------------------------------------------------------------
    # QUERY PROCESSING
    # -------------------------------------------------------------------------

    async def query(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        **options
    ) -> SovereignResult:
        """
        Process a query through the full sovereign pipeline.

        Pipeline:
        1. Cache check
        2. Graph-of-Thoughts reasoning
        3. SNR optimization
        4. Guardian Council validation
        5. Result synthesis

        Args:
            content: The query content
            context: Optional context dictionary
            **options: Query options (require_reasoning, max_depth, etc.)

        Returns:
            SovereignResult with answer, confidence, and quality metrics
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized. Call initialize() first.")

        # Create query object
        query = SovereignQuery(
            content=content,
            context=context or {},
            require_reasoning=options.get("require_reasoning", True),
            require_validation=options.get("require_validation", True),
            max_depth=options.get("max_depth", self.config.max_thought_depth),
            timeout_ms=options.get("timeout_ms", self.config.query_timeout_ms),
        )

        start_time = time.perf_counter()
        self.metrics.total_queries += 1
        self.metrics.last_query_at = datetime.now()

        # Check cache
        cache_key = self._cache_key(query)
        if self.config.enable_cache and cache_key in self._cache:
            self.metrics.cache_hits += 1
            cached = self._cache[cache_key]
            cached.cached = True
            return cached

        self.metrics.cache_misses += 1

        try:
            result = await asyncio.wait_for(
                self._process_query(query, start_time),
                timeout=query.timeout_ms / 1000,
            )

            # Cache successful results
            if result.success and self.config.enable_cache:
                self._update_cache(cache_key, result)

            self.metrics.successful_queries += 1
            return result

        except asyncio.TimeoutError:
            self.metrics.failed_queries += 1
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=f"Query timeout after {query.timeout_ms}ms",
            )
        except Exception as e:
            self.metrics.failed_queries += 1
            self.logger.error(f"Query error: {e}")
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=str(e),
            )

    async def _process_query(
        self,
        query: SovereignQuery,
        start_time: float
    ) -> SovereignResult:
        """Internal query processing pipeline."""

        result = SovereignResult(query_id=query.id)
        reasoning_start = time.perf_counter()

        # STAGE 1: Graph-of-Thoughts Reasoning
        if query.require_reasoning and self._graph_reasoner:
            reasoning_result = await self._graph_reasoner.reason(
                query=query.content,
                context=query.context,
                max_depth=query.max_depth,
            )
            result.answer = reasoning_result.get("conclusion", "")
            result.confidence = reasoning_result.get("confidence", 0.0)
            result.reasoning_path = reasoning_result.get("thoughts", [])
        else:
            result.answer = f"Direct response to: {query.content}"
            result.confidence = 0.75

        result.reasoning_time_ms = (time.perf_counter() - reasoning_start) * 1000
        self.metrics.avg_reasoning_time_ms = (
            self.metrics.avg_reasoning_time_ms * 0.9 +
            result.reasoning_time_ms * 0.1
        )

        # STAGE 2: SNR Optimization
        if self._snr_optimizer:
            snr_result = await self._snr_optimizer.optimize(result.answer)
            result.snr_score = snr_result.get("snr_score", 0.0)
            if snr_result.get("optimized"):
                result.answer = snr_result["optimized"]

        self.metrics.current_snr = result.snr_score

        # STAGE 3: Guardian Council Validation
        validation_start = time.perf_counter()

        if query.require_validation and self._guardian_council:
            validation = await self._guardian_council.validate(
                content=result.answer,
                context=query.context,
            )
            result.guardian_verdict = validation.get("verdict", "UNKNOWN")
            result.ihsan_score = validation.get("consensus_score", 0.0)

            self.metrics.council_invocations += 1
            if validation.get("approved"):
                self.metrics.council_approvals += 1
            else:
                self.metrics.council_rejections += 1
        else:
            result.ihsan_score = result.snr_score  # Use SNR as proxy
            result.guardian_verdict = "SKIPPED"

        result.validation_time_ms = (time.perf_counter() - validation_start) * 1000
        self.metrics.avg_validation_time_ms = (
            self.metrics.avg_validation_time_ms * 0.9 +
            result.validation_time_ms * 0.1
        )
        self.metrics.current_ihsan = result.ihsan_score

        # STAGE 4: Finalize Result
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        result.success = True
        result.completed_at = datetime.now()

        # Update timing metrics
        self._query_times.append(result.total_time_ms)
        if len(self._query_times) > 100:
            self._query_times = self._query_times[-100:]

        self.metrics.avg_query_time_ms = sum(self._query_times) / len(self._query_times)
        self.metrics.p99_query_time_ms = sorted(self._query_times)[
            int(len(self._query_times) * 0.99)
        ] if self._query_times else 0

        # Update health score
        self.metrics.health_score = self._calculate_health()

        return result

    def _cache_key(self, query: SovereignQuery) -> str:
        """Generate cache key for a query."""
        import hashlib
        content = f"{query.content}:{query.max_depth}:{query.require_reasoning}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _update_cache(self, key: str, result: SovereignResult) -> None:
        """Update cache with new result."""
        if len(self._cache) >= self.config.max_cache_entries:
            # Simple LRU: remove oldest entries
            oldest_keys = list(self._cache.keys())[:100]
            for k in oldest_keys:
                del self._cache[k]
        self._cache[key] = result

    def _calculate_health(self) -> float:
        """Calculate overall system health score."""
        factors = [
            min(1.0, self.metrics.current_snr / self.config.snr_threshold),
            min(1.0, self.metrics.current_ihsan / self.config.ihsan_threshold),
            self.metrics.success_rate(),
            max(0, 1 - self.metrics.avg_query_time_ms / self.config.max_reasoning_time_ms),
        ]
        return sum(factors) / len(factors)

    # -------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # -------------------------------------------------------------------------

    async def think(self, question: str) -> str:
        """
        Simple thinking interface.

        Usage:
            answer = await runtime.think("What is consciousness?")
        """
        result = await self.query(question)
        return result.answer if result.success else f"Error: {result.error}"

    async def validate(self, content: str) -> bool:
        """
        Validate content against Ihsān standards.

        Usage:
            is_valid = await runtime.validate("Some content to check")
        """
        result = await self.query(
            content,
            require_reasoning=False,
            require_validation=True,
        )
        return result.meets_ihsan(self.config.ihsan_threshold)

    async def reason(
        self,
        question: str,
        depth: int = 3
    ) -> List[str]:
        """
        Get reasoning path for a question.

        Usage:
            steps = await runtime.reason("Why is the sky blue?")
        """
        result = await self.query(question, max_depth=depth)
        return result.reasoning_path

    # -------------------------------------------------------------------------
    # STATUS & METRICS
    # -------------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get comprehensive runtime status."""
        loop_status = (
            self._autonomous_loop.status()
            if self._autonomous_loop else {"running": False}
        )

        return {
            "identity": {
                "node_id": self.config.node_id,
                "node_name": self.config.node_name,
                "version": self.config.version,
            },
            "state": {
                "initialized": self._initialized,
                "running": self._running,
                "mode": self.config.mode.name,
            },
            "health": {
                "status": self._health_status().value,
                "score": f"{self.metrics.health_score:.3f}",
                "snr": f"{self.metrics.current_snr:.3f}",
                "ihsan": f"{self.metrics.current_ihsan:.3f}",
            },
            "autonomous": loop_status,
            "metrics": self.metrics.to_dict(),
        }

    def _health_status(self) -> HealthStatus:
        """Determine health status from metrics."""
        score = self.metrics.health_score
        if score >= 0.9:
            return HealthStatus.HEALTHY
        elif score >= 0.7:
            return HealthStatus.DEGRADED
        elif score > 0:
            return HealthStatus.CRITICAL
        return HealthStatus.UNKNOWN

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    async def _checkpoint(self) -> None:
        """Save runtime state to disk."""
        if not self.config.enable_persistence:
            return

        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)

            import json
            state = {
                "metrics": self.metrics.to_dict(),
                "config": {
                    "node_id": self.config.node_id,
                    "mode": self.config.mode.name,
                },
                "timestamp": datetime.now().isoformat(),
            }

            state_file = self.config.state_dir / "checkpoint.json"
            state_file.write_text(json.dumps(state, indent=2))

            self.metrics.last_checkpoint_at = datetime.now()
            self.logger.debug("Checkpoint saved")

        except Exception as e:
            self.logger.warning(f"Checkpoint failed: {e}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def cli_main():
    """Command-line interface for the Sovereign Runtime."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BIZRA Sovereign Runtime CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.sovereign.runtime query "What is sovereignty?"
  python -m core.sovereign.runtime status
  python -m core.sovereign.runtime --mode AUTONOMOUS run
        """
    )

    parser.add_argument(
        "command",
        choices=["query", "status", "run", "version"],
        help="Command to execute"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Command arguments"
    )
    parser.add_argument(
        "--mode",
        choices=["DEVELOPMENT", "PRODUCTION", "AUTONOMOUS", "SUPERVISED"],
        default="PRODUCTION",
        help="Runtime mode"
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.95,
        help="SNR threshold"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Configure
    config = RuntimeConfig(
        mode=RuntimeMode[args.mode],
        snr_threshold=args.snr,
        autonomous_enabled=(args.mode == "AUTONOMOUS"),
    )

    if args.command == "version":
        print(f"Sovereign Runtime v{config.version}")
        return

    async with SovereignRuntime.create(config) as runtime:
        if args.command == "query":
            query_text = " ".join(args.args) if args.args else "Hello, Sovereign"
            result = await runtime.query(query_text)

            if args.json:
                import json
                print(json.dumps({
                    "success": result.success,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "snr": result.snr_score,
                    "ihsan": result.ihsan_score,
                    "time_ms": result.total_time_ms,
                }, indent=2))
            else:
                print(f"\n{'─' * 60}")
                print(f"Query: {query_text}")
                print(f"{'─' * 60}")
                print(f"Answer: {result.answer}")
                print(f"{'─' * 60}")
                print(f"Confidence: {result.confidence:.2%}")
                print(f"SNR: {result.snr_score:.3f}")
                print(f"Ihsān: {result.ihsan_score:.3f}")
                print(f"Time: {result.total_time_ms:.1f}ms")
                print(f"{'─' * 60}\n")

        elif args.command == "status":
            status = runtime.status()
            if args.json:
                import json
                print(json.dumps(status, indent=2))
            else:
                print(f"\n{'═' * 60}")
                print("SOVEREIGN RUNTIME STATUS")
                print(f"{'═' * 60}")
                print(f"Node: {status['identity']['node_name']}")
                print(f"ID: {status['identity']['node_id']}")
                print(f"Mode: {status['state']['mode']}")
                print(f"Health: {status['health']['status']} ({status['health']['score']})")
                print(f"SNR: {status['health']['snr']}")
                print(f"Ihsān: {status['health']['ihsan']}")
                print(f"{'═' * 60}\n")

        elif args.command == "run":
            print("Sovereign Runtime running. Press Ctrl+C to stop.")
            await runtime.wait_for_shutdown()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "SovereignRuntime",
    "RuntimeConfig",
    "RuntimeMode",
    "RuntimeMetrics",
    # Query/Response
    "SovereignQuery",
    "SovereignResult",
    # Health
    "HealthStatus",
    # CLI
    "cli_main",
]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    asyncio.run(cli_main())
