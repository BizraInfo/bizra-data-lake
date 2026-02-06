"""
Runtime Core — Main SovereignRuntime Implementation
====================================================
The core runtime class with lifecycle management, query processing,
and system orchestration. Uses types and stubs from companion modules.

Standing on Giants: Besta (GoT) + Shannon (SNR) + Anthropic (Constitutional AI)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Dict,
    List,
    Optional,
)

from .runtime_stubs import (
    StubFactory,
)
from .runtime_types import (
    AutonomousLoopProtocol,
    GraphReasonerProtocol,
    GuardianProtocol,
    HealthStatus,
    RuntimeConfig,
    RuntimeMetrics,
    SNROptimizerProtocol,
    SovereignQuery,
    SovereignResult,
)

logger = logging.getLogger("sovereign.runtime")


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

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config: RuntimeConfig = config or RuntimeConfig()
        self.metrics: RuntimeMetrics = RuntimeMetrics()
        self.logger: logging.Logger = logging.getLogger("sovereign.runtime")

        # State
        self._initialized: bool = False
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Components (initialized lazily) - using Protocol types for type safety
        self._graph_reasoner: Optional[GraphReasonerProtocol] = None
        self._snr_optimizer: Optional[SNROptimizerProtocol] = None
        self._guardian_council: Optional[GuardianProtocol] = None
        self._autonomous_loop: Optional[AutonomousLoopProtocol] = None
        self._orchestrator: Optional[object] = None

        # Omega Point Integration (v2.2.3)
        self._gateway: Optional[object] = None  # InferenceGateway
        self._omega: Optional[object] = None  # OmegaEngine

        # PERF FIX: Use deque for O(1) bounded storage
        self._query_times: Deque[float] = deque(maxlen=100)

        # Cache
        self._cache: Dict[str, SovereignResult] = {}

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def create(
        cls, config: Optional[RuntimeConfig] = None
    ) -> AsyncIterator["SovereignRuntime"]:
        """Create and manage runtime lifecycle."""
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
        self.logger.info(f"Ihsan Threshold: {self.config.ihsan_threshold}")

        await self._init_components()

        if self.config.autonomous_enabled:
            await self._start_autonomous_loop()

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
            self._graph_reasoner = StubFactory.create_graph_reasoner("Import failed")
            self.logger.warning("⚠ GraphOfThoughts unavailable, using stub")

        # Try full SNRMaximizer
        try:
            from .snr_maximizer import SNRMaximizer

            self._snr_optimizer = SNRMaximizer(
                ihsan_threshold=self.config.snr_threshold
            )
            self.logger.info("✓ SNRMaximizer loaded (full)")
        except ImportError:
            self._snr_optimizer = StubFactory.create_snr_optimizer("Import failed")
            self.logger.warning("⚠ SNRMaximizer unavailable, using stub")

        # Try full GuardianCouncil
        try:
            from .guardian_council import GuardianCouncil

            self._guardian_council = GuardianCouncil()
            self.logger.info("✓ GuardianCouncil loaded (full)")
        except ImportError:
            self._guardian_council = StubFactory.create_guardian("Import failed")
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
            self._autonomous_loop = StubFactory.create_autonomous_loop("Import failed")
            self.logger.warning("⚠ AutonomousLoop unavailable, using stub")

        # Omega Point Integration
        await self._init_omega_components()

    async def _init_omega_components(self) -> None:
        """Initialize Omega Point components (InferenceGateway, OmegaEngine)."""
        # InferenceGateway - Real LLM backends
        try:
            from core.inference.gateway import InferenceGateway

            self._gateway = InferenceGateway()
            try:
                await asyncio.wait_for(self._gateway.initialize(), timeout=5.0)
                self.logger.info("✓ InferenceGateway loaded and initialized")
            except (asyncio.TimeoutError, Exception) as init_err:
                self.logger.warning(f"⚠ InferenceGateway init timeout/error: {init_err}, gateway available but uninitialized")
        except ImportError as e:
            self._gateway = None
            self.logger.warning(f"⚠ InferenceGateway unavailable: {e}")

        # OmegaEngine - Constitutional enforcement
        try:
            from .omega_engine import OmegaEngine

            self._omega = OmegaEngine()
            self.logger.info("✓ OmegaEngine loaded (Constitutional Core)")
        except ImportError as e:
            self._omega = None
            self.logger.warning(f"⚠ OmegaEngine unavailable: {e}")

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
                    sig, lambda: asyncio.create_task(self.shutdown())
                )
        except (NotImplementedError, RuntimeError):
            pass  # Windows doesn't support add_signal_handler

    async def shutdown(self) -> None:
        """Gracefully shutdown the runtime."""
        if not self._running:
            return

        self.logger.info("Initiating graceful shutdown...")
        self._running = False

        if self._autonomous_loop:
            self._autonomous_loop.stop()

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
        self, content: str, context: Optional[Dict[str, Any]] = None, **options
    ) -> SovereignResult:
        """Process a query through the full sovereign pipeline."""
        if not self._initialized:
            raise RuntimeError("Runtime not initialized. Call initialize() first.")

        query = SovereignQuery(
            text=content,
            context=context or {},
            require_reasoning=options.get("require_reasoning", True),
            require_validation=options.get("require_validation", False),
            timeout=options.get("timeout_ms", self.config.query_timeout_ms) / 1000,
        )

        start_time = time.perf_counter()
        self.metrics.queries_processed += 1

        # Check cache
        cache_key = self._cache_key(query)
        if self.config.enable_cache and cache_key in self._cache:
            self.metrics.cache_hits += 1
            cached = self._cache[cache_key]
            return cached

        self.metrics.cache_misses += 1

        try:
            result = await asyncio.wait_for(
                self._process_query(query, start_time),
                timeout=query.timeout,
            )

            if result.success and self.config.enable_cache:
                self._update_cache(cache_key, result)

            self.metrics.queries_succeeded += 1
            return result

        except asyncio.TimeoutError:
            self.metrics.queries_failed += 1
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=f"Query timeout after {query.timeout}s",
            )
        except Exception as e:
            self.metrics.queries_failed += 1
            self.logger.error(f"Query error: {e}")
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=str(e),
            )

    async def _process_query(
        self, query: SovereignQuery, start_time: float
    ) -> SovereignResult:
        """Internal query processing pipeline."""
        result = SovereignResult(query_id=query.id)
        reasoning_start = time.perf_counter()

        # STAGE 0: Select compute tier
        compute_tier = await self._select_compute_tier(query)

        # STAGE 1: Execute reasoning (GoT)
        reasoning_path, confidence, thought_prompt = (
            await self._execute_reasoning_stage(query)
        )
        result.thoughts = reasoning_path
        result.reasoning_depth = len(reasoning_path)

        # STAGE 2: Perform LLM inference
        answer, model_used = await self._perform_llm_inference(
            thought_prompt, compute_tier, query
        )
        result.response = answer

        # Update reasoning metrics
        (time.perf_counter() - reasoning_start) * 1000
        self.metrics.update_reasoning_stats(result.reasoning_depth)

        # STAGE 3: Optimize SNR
        optimized_content, snr_score = await self._optimize_snr(result.response)
        result.response = optimized_content
        result.snr_score = snr_score

        # STAGE 4: Constitutional validation
        ihsan_score, guardian_verdict = await self._validate_constitutionally(
            result.response, query.context, query, result.snr_score
        )
        result.ihsan_score = ihsan_score
        result.validated = query.require_validation
        result.validation_passed = ihsan_score >= self.config.ihsan_threshold

        # STAGE 5: Finalize result
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.success = True
        result.reasoning_used = query.require_reasoning

        # Update timing metrics
        self._query_times.append(result.processing_time_ms)
        self.metrics.update_query_stats(True, result.processing_time_ms)

        return result

    async def _select_compute_tier(self, query: SovereignQuery) -> Optional[object]:
        """STAGE 0: Treasury Mode to Compute Tier selection."""
        if not self._omega:
            return None

        mode = getattr(self._omega, "get_operational_mode", lambda: None)()
        if mode is None:
            return None
        return self._mode_to_tier(mode)

    async def _execute_reasoning_stage(
        self, query: SovereignQuery
    ) -> tuple[List[str], float, str]:
        """STAGE 1: Graph-of-Thoughts exploration."""
        thought_prompt: str = query.text
        reasoning_path: List[str] = []
        confidence: float = 0.75

        if query.require_reasoning and self._graph_reasoner:
            reasoning_result = await self._graph_reasoner.reason(
                query=query.text,
                context=query.context,
                max_depth=self.config.max_reasoning_depth,
            )
            reasoning_path = reasoning_result.get("thoughts", [])
            confidence = reasoning_result.get("confidence", 0.0)

            conclusion = reasoning_result.get("conclusion")
            if conclusion:
                thought_prompt = conclusion

        return reasoning_path, confidence, thought_prompt

    async def _perform_llm_inference(
        self, thought_prompt: str, compute_tier: Optional[object], query: SovereignQuery
    ) -> tuple[str, str]:
        """STAGE 2: LLM inference via gateway."""
        if self._gateway:
            try:
                infer_method = getattr(self._gateway, "infer", None)
                if infer_method is not None:
                    inference_result = await infer_method(
                        thought_prompt,
                        tier=compute_tier,
                        max_tokens=1024,
                    )
                    answer = getattr(inference_result, "content", str(inference_result))
                    model_used = getattr(inference_result, "model", "unknown")
                    return answer, model_used
            except Exception as e:
                self.logger.warning(f"Gateway inference failed: {e}, using stub")

        return f"Reasoned response for: {query.text}", "stub"

    async def _optimize_snr(self, content: str) -> tuple[str, float]:
        """STAGE 3: SNR optimization."""
        from core.integration.constants import UNIFIED_SNR_THRESHOLD

        optimized_content = content
        snr_score = UNIFIED_SNR_THRESHOLD

        if self._snr_optimizer:
            snr_result = self._snr_optimizer.optimize(content)
            snr_score = snr_result.get("snr_score", UNIFIED_SNR_THRESHOLD)

        self.metrics.current_snr_score = snr_score
        return optimized_content, snr_score

    async def _validate_constitutionally(
        self,
        content: str,
        context: Dict[str, Any],
        query: SovereignQuery,
        snr_score: float,
    ) -> tuple[float, str]:
        """STAGE 4: Constitutional validation."""
        ihsan_score = snr_score
        guardian_verdict = "SKIPPED"

        if self._omega:
            try:
                ihsan_vector = self._extract_ihsan_from_response(content, context)
                evaluate_ihsan = getattr(self._omega, "evaluate_ihsan", None)
                if evaluate_ihsan is not None and ihsan_vector is not None:
                    result = evaluate_ihsan(ihsan_vector)
                    if isinstance(result, tuple) and len(result) >= 2:
                        ihsan_score = result[0]
                    else:
                        ihsan_score = float(result) if result else snr_score
                guardian_verdict = "OMEGA_ONLY"
            except Exception as e:
                self.logger.warning(f"Omega Ihsan evaluation failed: {e}")
                ihsan_score = snr_score

        if query.require_validation and self._guardian_council:
            validation = await self._guardian_council.validate(
                content=content,
                context=context,
            )
            guardian_verdict = "VALIDATED" if validation.get("is_valid") else "REJECTED"
            guardian_score = validation.get("confidence", 0.0)

            if self._omega:
                ihsan_score = (ihsan_score + guardian_score) / 2
            else:
                ihsan_score = guardian_score

            self.metrics.validations += 1
            self.metrics.update_validation_stats(validation.get("is_valid", False))

        self.metrics.current_ihsan_score = ihsan_score
        return ihsan_score, guardian_verdict

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _cache_key(self, query: SovereignQuery) -> str:
        """Generate cache key for a query."""
        import hashlib

        content = f"{query.text}:{query.require_reasoning}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _update_cache(self, key: str, result: SovereignResult) -> None:
        """Update cache with new result."""
        if len(self._cache) >= self.config.max_cache_entries:
            oldest_keys = list(self._cache.keys())[:100]
            for k in oldest_keys:
                del self._cache[k]
        self._cache[key] = result

    def _mode_to_tier(self, mode: object) -> Optional[object]:
        """Map TreasuryMode to ComputeTier."""
        try:
            from core.inference.gateway import ComputeTier

            from .omega_engine import TreasuryMode

            mapping = {
                TreasuryMode.ETHICAL: ComputeTier.LOCAL,
                TreasuryMode.HIBERNATION: ComputeTier.EDGE,
                TreasuryMode.EMERGENCY: ComputeTier.EDGE,
            }
            return mapping.get(mode, ComputeTier.LOCAL)
        except ImportError:
            return None

    def _extract_ihsan_from_response(
        self, content: str, context: Dict[str, Any]
    ) -> Optional[object]:
        """Extract Ihsan vector from response content."""
        try:
            from .omega_engine import ihsan_from_scores

            word_count = len(content.split())
            has_harmful = any(
                w in content.lower()
                for w in ["kill", "harm", "destroy", "attack", "illegal"]
            )

            correctness = min(0.98, 0.85 + (word_count / 1000) * 0.1)
            safety = 0.50 if has_harmful else 0.98
            user_benefit = float(context.get("benefit_score", 0.92))
            efficiency = min(0.96, 1.0 - (word_count / 5000))

            return ihsan_from_scores(
                correctness=correctness,
                safety=safety,
                user_benefit=user_benefit,
                efficiency=efficiency,
            )
        except ImportError:
            return None

    # -------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # -------------------------------------------------------------------------

    async def think(self, question: str) -> str:
        """Simple thinking interface."""
        result = await self.query(question)
        return result.response if result.success else f"Error: {result.error}"

    async def validate(self, content: str) -> bool:
        """Validate content against Ihsan standards."""
        result = await self.query(
            content,
            require_reasoning=False,
            require_validation=True,
        )
        return result.ihsan_score >= self.config.ihsan_threshold

    async def reason(self, question: str, depth: int = 3) -> List[str]:
        """Get reasoning path for a question."""
        result = await self.query(question, max_depth=depth)
        return result.thoughts

    # -------------------------------------------------------------------------
    # STATUS & METRICS
    # -------------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get comprehensive runtime status."""
        loop_status = (
            self._autonomous_loop.status()
            if self._autonomous_loop
            else {"running": False}
        )

        omega_status = {"version": "2.2.3"}
        if self._omega:
            try:
                omega_status.update(self._omega.get_status() or {})
            except Exception:
                omega_status["connected"] = True

        # Always ensure version is present
        omega_status.setdefault("version", "2.2.3")

        # Include gateway info in omega_point status
        if self._gateway:
            omega_status["gateway"] = {
                "connected": True,
                "status": getattr(self._gateway, "status", "unknown"),
            }
        else:
            omega_status.setdefault("gateway", {"connected": False})

        return {
            "identity": {
                "node_id": self.config.node_id,
                "version": "1.0.0",
            },
            "state": {
                "initialized": self._initialized,
                "running": self._running,
                "mode": self.config.mode.name,
            },
            "health": {
                "status": self._health_status().value,
                "score": self._calculate_health(),
            },
            "autonomous": loop_status,
            "omega_point": omega_status,
            "metrics": self.metrics.to_dict(),
        }

    def _health_status(self) -> HealthStatus:
        """Determine health status from metrics."""
        score = self._calculate_health()
        if score >= 0.9:
            return HealthStatus.HEALTHY
        elif score >= 0.7:
            return HealthStatus.DEGRADED
        elif score > 0:
            return HealthStatus.UNHEALTHY
        return HealthStatus.UNKNOWN

    def _calculate_health(self) -> float:
        """Calculate overall system health score."""
        snr_factor = min(
            1.0, self.metrics.current_snr_score / self.config.snr_threshold
        )
        ihsan_factor = min(
            1.0, self.metrics.current_ihsan_score / self.config.ihsan_threshold
        )
        success_factor = self.metrics.queries_succeeded / max(
            1, self.metrics.queries_processed
        )
        return (snr_factor + ihsan_factor + success_factor) / 3

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

            self.logger.debug("Checkpoint saved")

        except Exception as e:
            self.logger.warning(f"Checkpoint failed: {e}")


__all__ = [
    "SovereignRuntime",
]
