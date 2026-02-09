"""
Runtime Core — Main SovereignRuntime Implementation
====================================================
The core runtime class with lifecycle management, query processing,
and system orchestration. Uses types and stubs from companion modules.

Standing on Giants: Besta (GoT) + Shannon (SNR) + Anthropic (Constitutional AI)
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
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

from .genesis_identity import GenesisState, load_and_validate_genesis
from .memory_coordinator import (
    MemoryCoordinator,
    MemoryCoordinatorConfig,
    RestorePriority,
)
from .runtime_stubs import (
    StubFactory,
)
from .runtime_types import (
    AutonomousLoopProtocol,
    GraphReasonerProtocol,
    GuardianProtocol,
    HealthStatus,
    ImpactTrackerProtocol,
    RuntimeConfig,
    RuntimeMetrics,
    SNROptimizerProtocol,
    SovereignQuery,
    SovereignResult,
)
from .user_context import UserContextManager, select_pat_agent

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

        # Genesis Identity (persistent across restarts)
        self._genesis: Optional[GenesisState] = None

        # Unified Memory Coordinator (auto-save + persistence)
        self._memory_coordinator: Optional[MemoryCoordinator] = None

        # Impact Tracker (sovereignty growth engine)
        self._impact_tracker: Optional[ImpactTrackerProtocol] = None

        # Omega Point Integration (v2.2.3)
        self._gateway: Optional[object] = None  # InferenceGateway
        self._omega: Optional[object] = None  # OmegaEngine
        self._living_memory: Optional[object] = None  # LivingMemoryCore
        self._pek: Optional[object] = None  # ProactiveExecutionKernel
        self._zpk_bootstrap_result: Optional[object] = None

        # PERF FIX: Use deque for O(1) bounded storage
        self._query_times: Deque[float] = deque(maxlen=100)

        # Cache
        self._cache: Dict[str, SovereignResult] = {}

        # User Context (the system knows its human)
        self._user_context: Optional[UserContextManager] = None

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    def _load_env_vars(self) -> None:
        """Load environment variables from sovereign_state/.env if present."""
        import os
        env_file = self.config.state_dir / ".env"
        if env_file.exists():
            for line in env_file.read_text().strip().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and not os.getenv(key):
                        os.environ[key] = value
            self.logger.info(f"✓ Loaded env vars from {env_file}")

    @staticmethod
    def _parse_env_bool(value: str, default: bool = False) -> bool:
        """Parse a boolean environment value with a safe default."""
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _apply_env_overrides(self) -> None:
        """Apply runtime config overrides from environment variables."""
        import os

        def _set_float(env_name: str, attr_name: str) -> None:
            raw = os.getenv(env_name)
            if not raw:
                return
            try:
                setattr(self.config, attr_name, float(raw))
            except ValueError:
                self.logger.warning("Invalid %s: %s", env_name, raw)

        def _set_bool(env_name: str, attr_name: str) -> None:
            raw = os.getenv(env_name)
            if raw is None:
                return
            current = getattr(self.config, attr_name)
            setattr(self.config, attr_name, self._parse_env_bool(raw, current))

        manifest_uri = os.getenv("ZPK_MANIFEST_URI")
        if manifest_uri:
            self.config.zpk_manifest_uri = manifest_uri

        release_pubkey = os.getenv("ZPK_RELEASE_PUBLIC_KEY")
        if release_pubkey:
            self.config.zpk_release_public_key = release_pubkey

        enabled = os.getenv("ZPK_PREFLIGHT_ENABLED")
        if enabled is not None:
            self.config.enable_zpk_preflight = self._parse_env_bool(
                enabled, self.config.enable_zpk_preflight
            )

        emit_events = os.getenv("ZPK_EMIT_BOOTSTRAP_EVENTS")
        if emit_events is not None:
            self.config.zpk_emit_bootstrap_events = self._parse_env_bool(
                emit_events, self.config.zpk_emit_bootstrap_events
            )

        event_topic = os.getenv("ZPK_EVENT_TOPIC")
        if event_topic:
            self.config.zpk_event_topic = event_topic

        allowed_versions = os.getenv("ZPK_ALLOWED_VERSIONS")
        if allowed_versions:
            self.config.zpk_allowed_versions = [
                part.strip()
                for part in allowed_versions.split(",")
                if part.strip()
            ]

        min_policy_version = os.getenv("ZPK_MIN_POLICY_VERSION")
        if min_policy_version:
            try:
                self.config.zpk_min_policy_version = int(min_policy_version)
            except ValueError:
                self.logger.warning(
                    "Invalid ZPK_MIN_POLICY_VERSION: %s", min_policy_version
                )

        min_ihsan_policy = os.getenv("ZPK_MIN_IHSAN_POLICY")
        if min_ihsan_policy:
            try:
                self.config.zpk_min_ihsan_policy = float(min_ihsan_policy)
            except ValueError:
                self.logger.warning(
                    "Invalid ZPK_MIN_IHSAN_POLICY: %s", min_ihsan_policy
                )

        # Proactive Execution Kernel (PEK) overrides
        _set_bool("PEK_ENABLED", "enable_proactive_kernel")
        _set_bool("PEK_EMIT_PROOF_EVENTS", "proactive_kernel_emit_events")

        pek_topic = os.getenv("PEK_PROOF_EVENT_TOPIC")
        if pek_topic:
            self.config.proactive_kernel_event_topic = pek_topic

        _set_float("PEK_CYCLE_SECONDS", "proactive_kernel_cycle_seconds")
        _set_float("PEK_MIN_CONFIDENCE", "proactive_kernel_min_confidence")
        _set_float("PEK_MIN_AUTO_CONFIDENCE", "proactive_kernel_min_auto_confidence")
        _set_float("PEK_BASE_TAU", "proactive_kernel_base_tau")
        _set_float("PEK_AUTO_EXECUTE_TAU", "proactive_kernel_auto_execute_tau")
        _set_float("PEK_QUEUE_SILENT_TAU", "proactive_kernel_queue_silent_tau")
        _set_float(
            "PEK_ATTENTION_BUDGET_CAPACITY",
            "proactive_kernel_attention_budget_capacity",
        )
        _set_float(
            "PEK_ATTENTION_BUDGET_RECOVERY_PER_CYCLE",
            "proactive_kernel_attention_recovery_per_cycle",
        )

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

        # Load env vars from sovereign_state/.env (API keys, endpoints)
        self._load_env_vars()
        self._apply_env_overrides()

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME INITIALIZING")
        self.logger.info("=" * 60)

        # Load genesis identity (persistent node_id from ceremony)
        self._load_genesis_identity()

        self.logger.info(f"Node ID: {self.config.node_id}")
        self.logger.info(f"Mode: {self.config.mode.name}")
        self.logger.info(f"Ihsan Threshold: {self.config.ihsan_threshold}")

        if self._genesis:
            self.logger.info(f"Node Name: {self._genesis.node_name}")
            self.logger.info(f"Location: {self._genesis.identity.location}")
            self.logger.info(
                f"PAT Team: {len(self._genesis.pat_team)} agents — "
                f"{', '.join(a.role for a in self._genesis.pat_team)}"
            )
            self.logger.info(
                f"SAT Team: {len(self._genesis.sat_team)} agents — "
                f"{', '.join(a.role for a in self._genesis.sat_team)}"
            )

        # Trusted bootstrap gate (optional fail-closed preflight)
        await self._run_zpk_preflight()

        await self._init_components()

        if self.config.autonomous_enabled:
            await self._start_autonomous_loop()

        # Initialize user context (the system knows its human)
        self._init_user_context()

        # Initialize unified memory coordinator with auto-save
        await self._init_memory_coordinator()

        # Initialize impact tracker (sovereignty growth engine)
        self._init_impact_tracker()

        self._setup_signal_handlers()

        self._initialized = True
        self._running = True
        self.metrics.started_at = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME READY")
        self.logger.info("=" * 60)

    def _init_user_context(self) -> None:
        """Initialize user context — the system knows its human."""
        self._user_context = UserContextManager(self.config.state_dir)
        self._user_context.load()

        # Wire genesis identity into user profile
        if self._genesis and not self._user_context.profile.node_id:
            self._user_context.profile.node_id = self._genesis.node_id
            self._user_context.profile.node_name = self._genesis.node_name

        if self._user_context.profile.is_populated():
            self.logger.info(
                f"User context loaded: {self._user_context.profile.name} "
                f"({self._user_context.conversation.get_turn_count()} turns)"
            )
        else:
            self.logger.info("User context: new session (profile not yet populated)")

        # Register with memory coordinator for auto-save
        if self._memory_coordinator:
            self._memory_coordinator.register_state_provider(
                "user_context",
                self._user_context.get_persistable_state,
                priority=RestorePriority.CORE,
            )

    def _load_genesis_identity(self) -> None:
        """Load persistent genesis identity if available."""
        try:
            genesis = load_and_validate_genesis(self.config.state_dir)
            if genesis is not None:
                self._genesis = genesis
                self.config.node_id = genesis.node_id
                self.logger.info(
                    f"Genesis identity loaded: {genesis.node_id} ({genesis.node_name})"
                )
            else:
                self.logger.info("No genesis — running as ephemeral node")
        except ValueError as e:
            self.logger.error(f"Genesis identity corrupted: {e}")

    async def _init_components(self) -> None:
        """Initialize components with graceful fallback.

        RFC-01 FIX: Respects feature flags from RuntimeConfig.
        """
        # Try full GraphOfThoughts (only if flag enabled)
        if self.config.enable_graph_reasoning:
            try:
                from .graph_reasoner import GraphOfThoughts

                self._graph_reasoner = GraphOfThoughts()
                self.logger.info("✓ GraphOfThoughts loaded (full)")
            except ImportError:
                self._graph_reasoner = StubFactory.create_graph_reasoner(
                    "Import failed"
                )
                self.logger.warning("⚠ GraphOfThoughts unavailable, using stub")
        else:
            self._graph_reasoner = StubFactory.create_graph_reasoner(
                "Disabled by config"
            )
            self.logger.info("○ GraphOfThoughts disabled by config")

        # Try full SNRMaximizer (only if flag enabled)
        if self.config.enable_snr_optimization:
            try:
                from .snr_maximizer import SNRMaximizer

                self._snr_optimizer = SNRMaximizer(
                    ihsan_threshold=self.config.snr_threshold
                )
                self.logger.info("✓ SNRMaximizer loaded (full)")
            except ImportError:
                self._snr_optimizer = StubFactory.create_snr_optimizer("Import failed")
                self.logger.warning("⚠ SNRMaximizer unavailable, using stub")
        else:
            self._snr_optimizer = StubFactory.create_snr_optimizer("Disabled by config")
            self.logger.info("○ SNRMaximizer disabled by config")

        # Try full GuardianCouncil (only if flag enabled)
        if self.config.enable_guardian_validation:
            try:
                from .guardian_council import GuardianCouncil

                self._guardian_council = GuardianCouncil()
                self.logger.info("✓ GuardianCouncil loaded (full)")
            except ImportError:
                self._guardian_council = StubFactory.create_guardian("Import failed")
                self.logger.warning("⚠ GuardianCouncil unavailable, using stub")
        else:
            self._guardian_council = StubFactory.create_guardian("Disabled by config")
            self.logger.info("○ GuardianCouncil disabled by config")

        # Try full AutonomousLoop (only if flag enabled)
        if self.config.enable_autonomous_loop:
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
                self._autonomous_loop = StubFactory.create_autonomous_loop(
                    "Import failed"
                )
                self.logger.warning("⚠ AutonomousLoop unavailable, using stub")
        else:
            self._autonomous_loop = StubFactory.create_autonomous_loop(
                "Disabled by config"
            )
            self.logger.info("○ AutonomousLoop disabled by config")

        # Omega Point Integration
        await self._init_omega_components()

        # PEK Integration (optional proactive kernel)
        await self._init_proactive_execution_kernel()

    async def _run_zpk_preflight(self) -> None:
        """Run Zero Point Kernel bootstrap preflight when enabled.

        Fail-closed: if enabled and preflight fails, runtime initialization aborts.
        """
        if not self.config.enable_zpk_preflight:
            self._zpk_bootstrap_result = None
            return

        if not self.config.zpk_manifest_uri or not self.config.zpk_release_public_key:
            raise RuntimeError(
                "ZPK preflight enabled but zpk_manifest_uri/zpk_release_public_key missing"
            )

        try:
            from core.zpk import ZeroPointKernel, ZPKPolicy
        except Exception as e:
            raise RuntimeError(f"ZPK preflight unavailable: {e}") from e

        allowed_versions = (
            set(self.config.zpk_allowed_versions)
            if self.config.zpk_allowed_versions
            else None
        )
        policy = ZPKPolicy(
            allowed_versions=allowed_versions,
            min_policy_version=self.config.zpk_min_policy_version,
            min_ihsan_policy=self.config.zpk_min_ihsan_policy,
        )

        event_bus = None
        if self.config.zpk_emit_bootstrap_events:
            try:
                from .event_bus import get_event_bus

                event_bus = get_event_bus()
            except Exception as e:
                self.logger.warning("ZPK event bus unavailable: %s", e)

        zpk = ZeroPointKernel(
            state_dir=self.config.state_dir,
            release_public_key_hex=self.config.zpk_release_public_key,
            event_bus=event_bus,
            event_topic=self.config.zpk_event_topic,
        )
        result = await zpk.bootstrap(
            self.config.zpk_manifest_uri,
            policy=policy,
        )
        self._zpk_bootstrap_result = result

        if not getattr(result, "success", False):
            reason = getattr(result, "reason", "unknown")
            raise RuntimeError(f"ZPK preflight failed: {reason}")

        self.logger.info(
            "✓ ZPK preflight passed (version=%s, rollback=%s)",
            getattr(result, "executed_version", "unknown"),
            getattr(result, "rollback_used", False),
        )

    async def _init_omega_components(self) -> None:
        """Initialize Omega Point components (InferenceGateway, OmegaEngine)."""
        # InferenceGateway - Real LLM backends
        try:
            from core.inference.gateway import CircuitBreakerConfig, InferenceConfig, InferenceGateway

            self._gateway = InferenceGateway(
                config=InferenceConfig(
                    require_local=False,
                    circuit_breaker=CircuitBreakerConfig(
                        request_timeout=180.0,  # Local models need time for long prompts
                        failure_threshold=3,
                        recovery_timeout=30.0,
                    ),
                )
            )
            try:
                await asyncio.wait_for(self._gateway.initialize(), timeout=30.0)
                self.logger.info("✓ InferenceGateway loaded and initialized")
            except (asyncio.TimeoutError, Exception) as init_err:
                self.logger.warning(
                    f"⚠ InferenceGateway init timeout/error: {init_err}, gateway available but uninitialized"
                )
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

    async def _init_proactive_execution_kernel(self) -> None:
        """Initialize Proactive Execution Kernel (PEK) when enabled."""
        if not self.config.enable_proactive_kernel:
            self._pek = None
            self.logger.info("○ ProactiveExecutionKernel disabled by config")
            return

        try:
            from core.pek.kernel import (
                ProactiveExecutionKernel,
                ProactiveExecutionKernelConfig,
            )
            from .opportunity_pipeline import OpportunityPipeline

            pipeline = OpportunityPipeline(
                snr_threshold=self.config.snr_threshold,
                ihsan_threshold=self.config.ihsan_threshold,
            )
            await pipeline.start()

            event_bus = None
            if self.config.proactive_kernel_emit_events:
                try:
                    from .event_bus import get_event_bus

                    event_bus = get_event_bus()
                except Exception as event_err:
                    self.logger.warning("⚠ PEK event bus unavailable: %s", event_err)

            pek_config = ProactiveExecutionKernelConfig(
                cycle_interval_seconds=self.config.proactive_kernel_cycle_seconds,
                min_confidence=self.config.proactive_kernel_min_confidence,
                min_auto_confidence=self.config.proactive_kernel_min_auto_confidence,
                base_tau=self.config.proactive_kernel_base_tau,
                auto_execute_tau=self.config.proactive_kernel_auto_execute_tau,
                queue_silent_tau=self.config.proactive_kernel_queue_silent_tau,
                attention_budget_capacity=(
                    self.config.proactive_kernel_attention_budget_capacity
                ),
                attention_budget_recovery_per_cycle=(
                    self.config.proactive_kernel_attention_recovery_per_cycle
                ),
                emit_proof_events=self.config.proactive_kernel_emit_events,
                proof_event_topic=self.config.proactive_kernel_event_topic,
            )
            self._pek = ProactiveExecutionKernel(
                opportunity_pipeline=pipeline,
                inference_gateway=self._gateway,
                living_memory=self._living_memory,
                state_dir=self.config.state_dir,
                config=pek_config,
                event_bus=event_bus,
            )

            # Optional formal verification hook (soft fallback when unavailable).
            try:
                from .z3_fate_gate import Z3FATEGate, Z3_AVAILABLE

                if Z3_AVAILABLE:
                    self._pek.set_fate_gate(Z3FATEGate())
                    self.logger.info("✓ PEK FATE gate enabled (Z3)")
            except Exception as fate_err:
                self.logger.warning(f"⚠ PEK FATE gate unavailable: {fate_err}")

            await self._pek.start()
            self.logger.info("✓ ProactiveExecutionKernel started")
        except Exception as e:
            self._pek = None
            self.logger.warning(f"⚠ ProactiveExecutionKernel init failed: {e}")

    async def _init_memory_coordinator(self) -> None:
        """Initialize the unified memory coordinator with auto-save."""
        try:
            config = MemoryCoordinatorConfig(
                state_dir=self.config.state_dir,
                auto_save_interval=120.0,
            )
            self._memory_coordinator = MemoryCoordinator(config)
            self._memory_coordinator.initialize(
                node_id=self.config.node_id,
                node_name=self._genesis.node_name if self._genesis else None,
            )

            # Register runtime state provider
            self._memory_coordinator.register_state_provider(
                "runtime", self._get_runtime_state, RestorePriority.CORE
            )

            # Register proactive component providers (if available)
            self._register_proactive_providers()

            # Register living memory if available
            try:
                from core.living_memory.core import LivingMemoryCore

                living_memory = LivingMemoryCore(
                    storage_path=self.config.state_dir / "living_memory",
                )
                await living_memory.initialize()
                self._living_memory = living_memory
                self._memory_coordinator.register_living_memory(living_memory)
                if self._pek and hasattr(self._pek, "set_living_memory"):
                    self._pek.set_living_memory(living_memory)
                self.logger.info("✓ LivingMemory connected to auto-save")
            except ImportError:
                self.logger.warning("⚠ LivingMemory unavailable")
            except Exception as e:
                self.logger.warning(f"⚠ LivingMemory init failed: {e}")

            # Start auto-save background loop
            if self.config.enable_persistence:
                await self._memory_coordinator.start_auto_save()
                self.logger.info("✓ MemoryCoordinator auto-save active")

        except Exception as e:
            self.logger.warning(f"⚠ MemoryCoordinator init failed: {e}")

    def _init_impact_tracker(self) -> None:
        """Initialize the impact tracker for sovereignty progression."""
        try:
            from core.pat.impact_tracker import ImpactTracker

            self._impact_tracker = ImpactTracker(
                node_id=self.config.node_id,
                state_dir=self.config.state_dir,
            )

            # Register as memory coordinator state provider
            if self._memory_coordinator:
                self._memory_coordinator.register_state_provider(
                    "impact_tracker",
                    self._get_impact_state,
                    RestorePriority.QUALITY,
                )

            self.logger.info(
                f"✓ ImpactTracker active "
                f"(tier: {self._impact_tracker.sovereignty_tier.value}, "
                f"score: {self._impact_tracker.sovereignty_score:.4f})"
            )
        except ImportError:
            self.logger.warning("⚠ ImpactTracker unavailable")
        except Exception as e:
            self.logger.warning(f"⚠ ImpactTracker init failed: {e}")

    def _get_impact_state(self) -> Dict[str, Any]:
        """Provide impact tracker state for memory coordinator."""
        if not self._impact_tracker:
            return {}
        try:
            progress = self._impact_tracker.get_progress()
            return progress.to_dict()
        except Exception:
            return {}

    def _record_query_impact(self, result: "SovereignResult") -> None:
        """Record a successful query as an impact event (fire-and-forget)."""
        if not self._impact_tracker:
            return
        try:
            from core.pat.impact_tracker import UERSScore, compute_query_bloom

            # Bloom from single source of truth (DRY)
            bloom = compute_query_bloom(
                processing_time_ms=result.processing_time_ms,
                reasoning_depth=result.reasoning_depth,
                validated=getattr(result, "validation_passed", False),
            )

            # Derive UERS from query quality signals
            uers = UERSScore(
                utility=min(1.0, len(result.response or "") / 500),
                efficiency=min(1.0, 1.0 - (result.processing_time_ms / 10000)),
                resilience=result.snr_score,
                sustainability=0.5,  # Base for runtime queries
                ethics=result.ihsan_score,
            )

            self._impact_tracker.record_event(
                category="computation",
                action="sovereign_query",
                bloom=bloom,
                uers=uers,
                metadata={
                    "query_id": result.query_id,
                    "processing_time_ms": result.processing_time_ms,
                    "reasoning_depth": result.reasoning_depth,
                    "snr_score": result.snr_score,
                    "ihsan_score": result.ihsan_score,
                },
            )
        except Exception as e:
            # Impact recording should never break queries
            self.logger.debug(f"Impact recording failed: {e}")

    def _get_runtime_state(self) -> Dict[str, Any]:
        """Provide runtime state snapshot for memory coordinator."""
        state: Dict[str, Any] = {
            "metrics": self.metrics.to_dict(),
            "config": {
                "node_id": self.config.node_id,
                "mode": self.config.mode.name,
            },
            "components": {
                "graph_reasoner": self._graph_reasoner is not None,
                "snr_optimizer": self._snr_optimizer is not None,
                "guardian_council": self._guardian_council is not None,
                "autonomous_loop": self._autonomous_loop is not None,
                "gateway": self._gateway is not None,
                "omega": self._omega is not None,
                "pek": self._pek is not None,
            },
            "cache_size": len(self._cache),
        }
        if self._zpk_bootstrap_result is not None:
            state["zpk_preflight"] = {
                "success": bool(getattr(self._zpk_bootstrap_result, "success", False)),
                "executed_version": getattr(
                    self._zpk_bootstrap_result, "executed_version", None
                ),
                "rollback_used": bool(
                    getattr(self._zpk_bootstrap_result, "rollback_used", False)
                ),
                "reason": getattr(self._zpk_bootstrap_result, "reason", ""),
            }
        if self._genesis:
            state["genesis"] = self._genesis.summary()
        return state

    def _register_proactive_providers(self) -> None:
        """Register proactive component state providers for persistence.

        Wraps each provider in try/except so unavailable components
        don't block the memory coordinator.
        """
        # PEK (kernel state + proof counters) — SAFETY priority
        if self._pek and hasattr(self._pek, "get_persistable_state"):
            try:
                self._memory_coordinator.register_state_provider(
                    "pek",
                    self._pek.get_persistable_state,
                    RestorePriority.SAFETY,
                )
                self.logger.debug("Registered PEK state provider")
            except Exception:
                pass

        # OpportunityPipeline — SAFETY priority (rate limiter must survive restarts)
        try:
            from .opportunity_pipeline import OpportunityPipeline

            pipeline = OpportunityPipeline()
            self._memory_coordinator.register_state_provider(
                "opportunity_pipeline",
                pipeline.get_persistable_state,
                RestorePriority.SAFETY,
            )
            self.logger.debug("Registered opportunity_pipeline state provider")
        except (ImportError, AttributeError):
            pass

        # ProactiveScheduler — QUALITY priority (job stats are nice-to-have)
        try:
            from .proactive_scheduler import ProactiveScheduler

            scheduler = ProactiveScheduler()
            self._memory_coordinator.register_state_provider(
                "scheduler",
                scheduler.get_persistable_state,
                RestorePriority.QUALITY,
            )
            self.logger.debug("Registered scheduler state provider")
        except (ImportError, AttributeError):
            pass

        # PredictiveMonitor — QUALITY priority (trend baselines)
        try:
            from .predictive_monitor import PredictiveMonitor

            monitor = PredictiveMonitor()
            self._memory_coordinator.register_state_provider(
                "predictive_monitor",
                monitor.get_persistable_state,
                RestorePriority.QUALITY,
            )
            self.logger.debug("Registered predictive_monitor state provider")
        except (ImportError, AttributeError):
            pass

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

        if self._pek and hasattr(self._pek, "stop"):
            try:
                await self._pek.stop()
            except Exception:
                pass

        # Save user context (conversation history + profile)
        if self._user_context:
            try:
                self._user_context.save()
            except Exception:
                pass

        # Flush impact tracker dirty state before memory coordinator stop
        if self._impact_tracker and hasattr(self._impact_tracker, "flush"):
            try:
                self._impact_tracker.flush()
            except Exception:
                pass

        # Stop memory coordinator (performs final save including all providers)
        # LCT-01 FIX: MemoryCoordinator.stop() already checkpoints all state.
        # The old _checkpoint() was a redundant second save of the same data.
        if self._memory_coordinator:
            await self._memory_coordinator.stop()

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
        # RFC-03 FIX: Don't manually increment here — update_query_stats() is
        # the single source of truth for all query counters.

        # Check cache
        cache_key = self._cache_key(query)
        if self.config.enable_cache and cache_key in self._cache:
            self.metrics.cache_hits += 1
            cached = self._cache[cache_key]
            return cached

        self.metrics.cache_misses += 1

        # Record human turn in conversation memory
        if self._user_context:
            self._user_context.conversation.add_human_turn(content)

        try:
            result = await asyncio.wait_for(
                self._process_query(query, start_time),
                timeout=query.timeout,
            )

            if result.success and self.config.enable_cache:
                self._update_cache(cache_key, result)

            # Record PAT response in conversation memory
            if self._user_context and result.success:
                agent_role = query.context.get("_responding_agent")
                self._user_context.conversation.add_pat_turn(
                    content=result.response or "",
                    agent_role=agent_role,
                    snr_score=result.snr_score,
                    ihsan_score=result.ihsan_score,
                )

            return result

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_query_stats(False, duration_ms)
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=f"Query timeout after {query.timeout}s",
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_query_stats(False, duration_ms)
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

        # Record impact for sovereignty progression (fire-and-forget)
        self._record_query_impact(result)

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

    async def _build_contextual_prompt(
        self, thought_prompt: str, query: SovereignQuery
    ) -> str:
        """Build a prompt enriched with user context, PAT identity, and memory retrieval."""
        if not self._user_context:
            return thought_prompt

        # Build PAT team info from genesis
        pat_info = ""
        selected_agent = None
        if self._genesis and self._genesis.pat_team:
            roles = [a.role for a in self._genesis.pat_team]
            pat_info = f"Available agents: {', '.join(roles)}"

            # Route to best agent
            selected_agent = select_pat_agent(query.text, self._genesis.pat_team)
            if selected_agent:
                pat_info += f"\nResponding as: {selected_agent.upper()}"

        # RAG retrieval from living memory
        memory_context = ""
        living_memory = getattr(self, "_living_memory", None)
        if living_memory:
            try:
                # Retrieve memories relevant to the query
                memories = await living_memory.retrieve(
                    query=query.text, top_k=5, min_score=0.15
                )
                if memories:
                    parts = []
                    for mem in memories:
                        label = mem.memory_type.value.upper()
                        # Truncate long memories to keep prompt manageable
                        content = mem.content
                        if len(content) > 800:
                            content = content[:800] + "..."
                        parts.append(f"[{label}] {content}")
                    memory_context = "\n\n".join(parts)
                    self.logger.debug(
                        f"RAG: retrieved {len(memories)} memories for query"
                    )
            except Exception as e:
                self.logger.warning(f"Memory retrieval failed: {e}")
                # Fall back to working context
                memory_context = living_memory.get_working_context(max_entries=5)

        # Build system prompt
        system_prompt = self._user_context.build_system_prompt(
            pat_team_info=pat_info,
            memory_context=memory_context,
        )

        # Store agent routing in query context for downstream use
        if selected_agent:
            query.context["_responding_agent"] = selected_agent

        return f"{system_prompt}\n\n--- QUERY ---\n{thought_prompt}"

    async def _perform_llm_inference(
        self, thought_prompt: str, compute_tier: Optional[object], query: SovereignQuery
    ) -> tuple[str, str]:
        """STAGE 2: LLM inference via gateway with user context."""
        # Build contextual prompt with user profile, memory, and PAT routing
        contextual_prompt = await self._build_contextual_prompt(thought_prompt, query)

        if self._gateway:
            try:
                infer_method = getattr(self._gateway, "infer", None)
                if infer_method is not None:
                    inference_result = await infer_method(
                        contextual_prompt,
                        tier=compute_tier,
                        max_tokens=512,
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
            result_or_coro = self._snr_optimizer.optimize(content)
            snr_result = (
                await result_or_coro
                if inspect.isawaitable(result_or_coro)
                else result_or_coro
            )
            snr_score = snr_result.get("snr_score", UNIFIED_SNR_THRESHOLD)
            # RFC-04 FIX: Actually use the optimized content from SNR pipeline
            optimized_content = snr_result.get("optimized") or content
            # Track SNR improvement
            if optimized_content != content:
                original_len = len(content)
                improvement = (original_len - len(optimized_content)) / max(
                    1, original_len
                )
                self.metrics.update_snr_stats(improvement)

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

        identity_info: Dict[str, Any] = {
            "node_id": self.config.node_id,
            "version": "1.0.0",
        }
        if self._genesis:
            identity_info["node_name"] = self._genesis.node_name
            identity_info["location"] = self._genesis.identity.location
            identity_info["public_key"] = self._genesis.identity.public_key[:16] + "..."
            identity_info["pat_agents"] = len(self._genesis.pat_team)
            identity_info["sat_agents"] = len(self._genesis.sat_team)
            identity_info["genesis_hash"] = (
                self._genesis.genesis_hash.hex()[:16] + "..."
                if self._genesis.genesis_hash
                else "none"
            )

        memory_status = (
            self._memory_coordinator.stats()
            if self._memory_coordinator
            else {"running": False}
        )

        # Impact / sovereignty progression
        sovereignty_info: Dict[str, Any] = {"tracking": False}
        if self._impact_tracker:
            try:
                sovereignty_info = {
                    "tracking": True,
                    "score": self._impact_tracker.sovereignty_score,
                    "tier": self._impact_tracker.sovereignty_tier.value,
                    "total_bloom": self._impact_tracker.total_bloom,
                    "achievements": len(self._impact_tracker.achievements),
                }
            except Exception:
                pass

        return {
            "identity": identity_info,
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
            "memory": memory_status,
            "sovereignty": sovereignty_info,
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

            state: Dict[str, Any] = {
                "metrics": self.metrics.to_dict(),
                "config": {
                    "node_id": self.config.node_id,
                    "mode": self.config.mode.name,
                },
                "timestamp": datetime.now().isoformat(),
            }
            if self._genesis:
                state["genesis"] = self._genesis.summary()

            state_file = self.config.state_dir / "checkpoint.json"
            state_file.write_text(json.dumps(state, indent=2))

            self.logger.debug("Checkpoint saved")

        except Exception as e:
            self.logger.warning(f"Checkpoint failed: {e}")


__all__ = [
    "SovereignRuntime",
]
