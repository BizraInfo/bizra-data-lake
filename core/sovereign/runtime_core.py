"""
Runtime Core — Main SovereignRuntime Implementation
====================================================
The core runtime class with lifecycle management, query processing,
and system orchestration. Uses types and stubs from companion modules.

Standing on Giants: Besta (GoT) + Shannon (SNR) + Anthropic (Constitutional AI)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import signal
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
)

from .genesis_identity import GenesisState, load_and_validate_genesis
from .memory_coordinator import (
    MemoryCoordinator,
    MemoryCoordinatorConfig,
    RestorePriority,
)
from .origin_guard import (
    NODE_ROLE_ENV,
    enforce_node0_fail_closed,
    normalize_node_role,
    resolve_origin_snapshot,
)
from .runtime_stubs import (
    StubFactory,
)
from .runtime_types import (
    AutonomousLoopProtocol,
    GoTNodeSnapshot,
    GraphReasonerProtocol,
    GuardianProtocol,
    HealthStatus,
    ImpactTrackerProtocol,
    ReasoningSummary,
    RuntimeConfig,
    RuntimeMetrics,
    SNROptimizerProtocol,
    SovereignQuery,
    SovereignResult,
)
from .user_context import UserContextManager, select_pat_agent

logger = logging.getLogger("sovereign.runtime")

# Elite version — single source of truth (deferred import to avoid circular deps)
try:
    from core.elite import ELITE_VERSION as _ELITE_VERSION
except ImportError:
    _ELITE_VERSION = "1.2.0"

# PERF: Module-level import eliminates 0.5ms deferred import per cache key computation
try:
    from core.proof_engine.canonical import hex_digest as _hex_digest
except ImportError:
    import hashlib as _hashlib

    def _hex_digest(data: bytes) -> str:  # type: ignore[misc]
        return _hashlib.blake2b(data, digest_size=16).hexdigest()


def _conservative_fallback_check(ctx: dict[str, Any]) -> bool:
    """Conservative fallback verification when Z3 solver is unavailable.

    DESIGN PRINCIPLE (α4): When the formal verifier (Z3) is unavailable,
    the fallback must be STRICTER, not weaker.  Default-deny: only actions
    matching a known-safe pattern are approved.  Unknown = Reject.

    Known-safe pattern (all must hold):
      - ihsan >= UNIFIED_IHSAN_THRESHOLD (constitutional floor)
      - snr   >= UNIFIED_SNR_THRESHOLD   (quality floor)
      - risk_level <= 0.5                (conservative: lower than Z3's 0.7)
      - cost  <= autonomy_limit          (budget constraint)
      - reversible is True OR human_approved is True (for any risk > 0.3)

    Any missing field defaults to the UNSAFE value (fail-closed).
    Standing on: Lamport (verify, don't trust).
    """
    try:
        from core.integration.constants import (
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )
    except ImportError:
        # If constants are also unavailable, fail closed.
        return False

    ihsan = ctx.get("ihsan", 0.0)
    snr = ctx.get("snr", 0.0)
    cost = ctx.get("cost", float("inf"))  # Missing cost → infinite (reject)
    autonomy_limit = ctx.get("autonomy_limit", 0.0)
    risk_level = ctx.get("risk_level", 1.0)  # Missing risk → max (reject)
    reversible = ctx.get("reversible", False)
    human_approved = ctx.get("human_approved", False)

    # Constitutional floors
    if ihsan < UNIFIED_IHSAN_THRESHOLD:
        return False
    if snr < UNIFIED_SNR_THRESHOLD:
        return False

    # Conservative risk gate: stricter threshold than Z3 (0.5 vs 0.7)
    if risk_level > 0.5 and not reversible and not human_approved:
        return False

    # Moderate risk still requires reversibility or approval
    if risk_level > 0.3 and not reversible and not human_approved:
        return False

    # Budget constraint
    if cost > autonomy_limit:
        return False

    return True


class _RuntimeInferenceBackend:
    """Inference adapter that lets the runtime-owned organism use the gateway."""

    def __init__(self, runtime: SovereignRuntime) -> None:
        self._runtime = runtime

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        gateway = getattr(self._runtime, "_gateway", None)
        if gateway is not None and hasattr(gateway, "infer"):
            result = await gateway.infer(
                prompt,
                max_tokens=kwargs.get("max_tokens"),
                temperature=float(kwargs.get("temperature", 0.3)),
            )
            return str(getattr(result, "content", result))
        return (
            "[runtime-degraded] Canonical organism inference backend unavailable. "
            f"Prompt summary: {prompt[:240]}"
        )


class _MissionPreflightChannel:
    """No-op ActionBus channel used to enforce TeleScript + FATE before mission run."""

    async def execute(self, action: Any) -> Any:
        from core.bus.channels import ChannelResult

        outcome_hash = _hex_digest(
            f"{action.action_id}:{action.kind}:{action.channel}".encode()
        )
        return ChannelResult(
            success=True,
            outcome_hash=outcome_hash,
            ihsan_score=float(action.payload.get("ihsan", 1.0)),
        )


class _RuntimeFATEGateAdapter:
    """Mission-level FATE adapter built on top of the runtime's current guardrails."""

    _HIGH_RISK_TERMS = (
        "rm -rf",
        "drop table",
        "delete production",
        "format disk",
        "shutdown",
        "wipe",
        "credential",
        "secret",
        "password",
        "token exfiltration",
    )

    def __init__(self, *, canonical_mode: bool) -> None:
        self._canonical_mode = canonical_mode
        self._z3_gate: Any | None = None
        self.enforced = False
        try:
            from core.sovereign.z3_fate_gate import Z3_AVAILABLE, Z3FATEGate

            if Z3_AVAILABLE:
                self._z3_gate = Z3FATEGate()
                self.enforced = True
        except (ImportError, AttributeError, OSError):
            self._z3_gate = None
            self.enforced = False

    def evaluate(self, action: Any) -> Any:
        from core.bus.action_bus import FATEResult

        payload = dict(getattr(action, "payload", {}) or {})
        ctx = self._build_context(payload)

        if self._z3_gate is not None:
            proof = self._z3_gate.generate_proof(ctx)
            if proof.satisfiable:
                return FATEResult(denied=False)
            reason = proof.counterexample or "fate_veto"
            codes = tuple(
                code.strip().replace(" ", "_")
                for code in reason.split(";")
                if code.strip()
            ) or ("fate_veto",)
            return FATEResult(
                denied=True,
                reason=reason,
                reason_codes=codes,
            )

        allowed = _conservative_fallback_check(ctx)
        return FATEResult(
            denied=not allowed,
            reason="" if allowed else "conservative_fallback_denied",
            reason_codes=() if allowed else ("conservative_fallback_denied",),
        )

    def _build_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        description = str(payload.get("description", "") or "").lower()
        risk_level = float(payload.get("risk_level", self._classify_risk(description)))
        time_budget_s = float(payload.get("time_budget_seconds", 900.0))
        autonomy_limit = float(payload.get("autonomy_limit", time_budget_s))
        cost = float(payload.get("cost", min(time_budget_s, autonomy_limit)))
        return {
            "ihsan": float(payload.get("ihsan", 1.0)),
            "snr": float(payload.get("snr", 1.0)),
            "risk_level": risk_level,
            "reversible": bool(payload.get("reversible", risk_level <= 0.3)),
            "human_approved": bool(payload.get("human_approved", False)),
            "cost": cost,
            "autonomy_limit": autonomy_limit,
        }

    def _classify_risk(self, description: str) -> float:
        if any(term in description for term in self._HIGH_RISK_TERMS):
            return 0.9
        if any(term in description for term in ("deploy", "publish", "push", "write")):
            return 0.45
        return 0.2


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

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config: RuntimeConfig = config or RuntimeConfig()
        self.metrics: RuntimeMetrics = RuntimeMetrics()
        self.logger: logging.Logger = logging.getLogger("sovereign.runtime")

        # State
        self._initialized: bool = False
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._strict_gate_passed: bool = True
        self._strict_gate_reason_codes: list[str] = []
        self._stub_components: list[str] = []

        # Components (initialized lazily) - using Protocol types for type safety
        self._graph_reasoner: GraphReasonerProtocol | None = None
        self._snr_optimizer: SNROptimizerProtocol | None = None
        self._guardian_council: GuardianProtocol | None = None
        self._autonomous_loop: AutonomousLoopProtocol | None = None
        self._orchestrator: object | None = None
        self._event_bus: object | None = None
        self._event_bus_task: asyncio.Task[Any] | None = None

        # Genesis Identity (persistent across restarts)
        self._genesis: GenesisState | None = None
        self._node_role: str = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        self._origin_snapshot: dict[str, Any] = resolve_origin_snapshot(
            self.config.state_dir, self._node_role
        )

        # Unified Memory Coordinator (auto-save + persistence)
        self._memory_coordinator: MemoryCoordinator | None = None

        # AgentDB (V3 unified memory with HNSW indexing)
        self._agent_db: object | None = None
        self._agent_db_bridge: object | None = None  # AgentDBBridge
        self._agent_db_health: object | None = None  # AgentDBHealthChecker

        # Impact Tracker (sovereignty growth engine)
        self._impact_tracker: ImpactTrackerProtocol | None = None

        # Evidence Ledger (append-only, hash-chained audit trail)
        self._evidence_ledger: object | None = None  # EvidenceLedger

        # Graph Artifact Store (query_id → schema-compliant GoT artifact)
        self._graph_artifacts: dict[str, dict[str, Any]] = {}

        # Last SNR trace from authoritative SNREngine v1 (for receipt embedding)
        self._last_snr_trace: dict[str, Any] | None = None

        # 6-Gate Chain — fail-closed execution pipeline (Golden Gem #1)
        self._gate_chain: object | None = None  # GateChain

        # Proof-of-Impact Engine — 4-stage PoI scoring pipeline
        self._poi_orchestrator: object | None = None  # PoIOrchestrator

        # SAT Controller — ecosystem homeostasis engine
        self._sat_controller: object | None = None  # SATController

        # Sovereign Experience Ledger (content-addressed episodic memory)
        self._experience_ledger: object | None = None  # ExperienceLedger

        # Unified Node0 Signer (Ed25519) — single identity for all subsystems
        self._node_signer: object | None = None  # Ed25519Signer
        self._organism: object | None = None  # SovereignOrganism
        self._node0: object | None = None  # Node0Heartbeat
        self._canonical_mode = False
        self._identity_mode = "placeholder_degraded"
        self._signer_public_key_prefix = ""
        self._genesis_backed_identity = False
        self._fate_mode = "degraded"
        self._fate_gate: object | None = None

        # IHSAN_FLOOR Watchdog — governance invariant enforcer (MCG Layer 7)
        self._ihsan_watchdog: object | None = None  # IhsanFloorWatchdog

        # Self-Evolving Judgment Engine — observation telemetry (Phase A)
        self._judgment_telemetry: object | None = None  # JudgmentTelemetry

        # Spearpoint Orchestrator (reproduce / improve / heartbeat)
        self._spearpoint_orchestrator: object | None = None

        # Omega Point Integration (v2.2.3)
        self._gateway: object | None = None  # InferenceGateway
        self._omega: object | None = None  # OmegaEngine
        self._living_memory: object | None = None  # LivingMemoryCore
        self._pek: object | None = None  # ProactiveExecutionKernel
        self._zpk_bootstrap_result: object | None = None
        self._autopoietic_loop: object | None = None  # AutopoieticLoop
        self._learning_loop: object | None = None  # LearningLoopOrchestrator
        self._autopoiesis_task: asyncio.Task[Any] | None = None
        self._autopoiesis_learning_task: asyncio.Task[Any] | None = None
        self._autopoiesis_learning_source: str = "disabled"

        # Phase 58: Equalizer Agent + Unified Model Router
        self._equalizer_agent: object | None = None  # EqualizerAgent
        self._unified_model_router: object | None = None  # UnifiedModelRouter

        # Phase 70: Bus Infrastructure (ActionBus, TopicRegistry, Config, Capsules)
        self._action_bus: object | None = None  # ActionBus
        self._topic_registry: object | None = None  # TopicRegistry
        self._telescript_engine: object | None = None  # TeleScriptEngine
        self._config_loader: object | None = None  # ConfigLoader
        self._capsule_registry: object | None = None  # CapsuleRegistry
        self._capsule_runtime: object | None = None  # CapsuleRuntime
        self._omega_controller: object | None = None  # OmegaLoopController
        self._bus_wiring_state: object | None = None  # BusWiringState

        # Phase 71: Seed Engine (DDAGI growth trajectory + self-RLVR)
        self._seed_engine: object | None = None  # SeedEngine

        # Phase 80: Runtime Daemons (PAT + SAT + DEMA + FATE Boundary)
        self._pat_runtime: object | None = None  # PATRuntime
        self._sat_runtime: object | None = None  # SATRuntime
        self._dema_router: object | None = None  # DEMARouter
        self._fate_boundary: object | None = None  # FATEBoundary
        self._urp_service: object | None = None  # URPService
        self._proactive_scheduler: object | None = None  # ProactiveScheduler
        self._proactive_scheduler_task: asyncio.Task[Any] | None = None

        # PERF FIX: Use deque for O(1) bounded storage
        self._query_times: deque[float] = deque(maxlen=100)

        # Cache
        self._cache: OrderedDict[str, SovereignResult] = OrderedDict()

        # User Context (the system knows its human)
        self._user_context: UserContextManager | None = None

        # Phase 31: Cognitive Fusion Engine
        self._hypergraph_store: object | None = None  # HyperGraphStore
        self._cognitive_fusion: object | None = None  # CognitiveFusionEngine
        self._memory_synthesizer: object | None = None  # MemorySynthesizer
        self._pattern_codebook: object | None = None  # PatternCodebook

        # Phase 32: Embedding Service + NTU Adapter
        self._embedding_service: object | None = None  # EmbeddingService
        self._embedding_gate: object | None = None  # EmbeddingQualityGate
        self._ntu_adapter: object | None = None  # NTUFusionAdapter

        # Phase 33: RDVE Engine (Recursive Discovery & Verification)
        self._rdve_engine: object | None = None  # RDVEOrchestrator

        # SpearPoint Pipeline — unified post-query cockpit
        self._spearpoint: object | None = None  # SpearPointPipeline

        # α7 Tiered Verification + α9 Performance Attestation (NODE0 integration)
        self._performance_attestor: object | None = None  # PerformanceAttestor
        self._tiered_verification_enabled: bool = False

        # Phase 25-28: Ecosystem subsystems
        self._hrm_engine: object | None = None  # HierarchicalReasoningModel
        self._northstar_engine: object | None = None  # NorthStarEngine
        self._guild_registry: object | None = None  # GuildRegistry
        self._quest_engine: object | None = None  # QuestEngine

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
                part.strip() for part in allowed_versions.split(",") if part.strip()
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
        _set_bool("BIZRA_AUTOPOIESIS_ENABLED", "enable_autopoiesis")

        pek_topic = os.getenv("PEK_PROOF_EVENT_TOPIC")
        if pek_topic:
            self.config.proactive_kernel_event_topic = pek_topic

        _set_float("PEK_CYCLE_SECONDS", "proactive_kernel_cycle_seconds")
        _set_float("PEK_MIN_CONFIDENCE", "proactive_kernel_min_confidence")
        _set_float("PEK_MIN_AUTO_CONFIDENCE", "proactive_kernel_min_auto_confidence")
        _set_float("PEK_BASE_TAU", "proactive_kernel_base_tau")
        _set_float("PEK_AUTO_EXECUTE_TAU", "proactive_kernel_auto_execute_tau")
        _set_float("PEK_QUEUE_SILENT_TAU", "proactive_kernel_queue_silent_tau")
        _set_float("BIZRA_AUTOPOIESIS_CYCLE_SECONDS", "autopoiesis_cycle_seconds")
        _set_float(
            "PEK_ATTENTION_BUDGET_CAPACITY",
            "proactive_kernel_attention_budget_capacity",
        )
        _set_float(
            "PEK_ATTENTION_BUDGET_RECOVERY_PER_CYCLE",
            "proactive_kernel_attention_recovery_per_cycle",
        )

    def _resolve_canonical_mode(self) -> bool:
        """Return whether runtime-canonical mode is explicitly enabled."""
        explicit = os.getenv("BIZRA_CANONICAL_MODE", "").strip().lower()
        if explicit in {"1", "true", "yes", "on"}:
            return True
        return os.getenv("BIZRA_ENV", "").strip().lower() == "production"

    def _load_identity_credentials(self) -> dict[str, str] | None:
        """Load genesis-backed Ed25519 credentials if available."""
        creds_path = self.config.state_dir / "identity" / "credentials.json"
        if not creds_path.exists():
            return None

        with open(creds_path, encoding="utf-8") as handle:
            creds = json.load(handle)

        private_hex = str(creds.get("private_key", "") or "")
        public_hex = str(creds.get("public_key", "") or "")
        node_id = str(creds.get("node_id", "") or "")
        if len(private_hex) != 64 or len(public_hex) != 64:
            raise ValueError(
                "identity/credentials.json is missing a valid Ed25519 keypair"
            )
        derived_node_id = self._derive_node_id_from_public_key(public_hex)
        if node_id and node_id != derived_node_id:
            if self._canonical_mode:
                raise ValueError(
                    "identity/credentials.json node_id does not match the Ed25519 public key"
                )
        else:
            node_id = derived_node_id
        return {
            "private_key_hex": private_hex,
            "public_key_hex": public_hex,
            "node_id": node_id,
        }

    def _signer_public_key_prefix_for(self, signer: object | None) -> str:
        if signer is None:
            return ""
        if hasattr(signer, "public_key_hex"):
            value = str(getattr(signer, "public_key_hex", "") or "")
            return value[:16]
        if hasattr(signer, "public_key_bytes"):
            try:
                return getattr(signer, "public_key_bytes")().hex()[:16]
            except (AttributeError, TypeError, ValueError):
                return ""
        return ""

    def _signer_public_key_hex_for(self, signer: object | None) -> str:
        if signer is None:
            return ""
        if hasattr(signer, "public_key_hex"):
            return str(getattr(signer, "public_key_hex", "") or "").lower()
        if hasattr(signer, "public_key_bytes"):
            try:
                return getattr(signer, "public_key_bytes")().hex()
            except (AttributeError, TypeError, ValueError):
                return ""
        return ""

    @staticmethod
    def _derive_node_id_from_public_key(public_key_hex: str) -> str:
        from core.pat.identity_card import _generate_node_id

        return _generate_node_id(public_key_hex)

    def _configure_canonical_action_bus(self) -> None:
        """Rewire ActionBus for canonical mission preflight semantics."""
        try:
            from core.bus.sovereign_wiring import (
                wire_action_bus,
                wire_telescript_engine,
            )

            telescript = self._telescript_engine or wire_telescript_engine()
            self._telescript_engine = telescript
            self._fate_gate = _RuntimeFATEGateAdapter(
                canonical_mode=self._canonical_mode
            )
            self._fate_mode = (
                "enforced"
                if getattr(self._fate_gate, "enforced", False)
                else "degraded"
            )
            self._action_bus = wire_action_bus(
                event_bus=self._event_bus,
                telescript=telescript,
                channels={"mission_gate": _MissionPreflightChannel()},
                fate_gate=self._fate_gate,
            )
            if self._canonical_mode and self._fate_mode != "enforced":
                raise RuntimeError(
                    "Canonical mode requires an enforced FATE gate on the mission spine"
                )
        except (ImportError, AttributeError, RuntimeError, OSError):
            self._fate_mode = "degraded"
            if self._canonical_mode:
                raise
            self.logger.warning(
                "Canonical ActionBus mission preflight unavailable; degraded mode active",
                exc_info=True,
            )

    async def _init_canonical_organism_stack(self) -> None:
        """Boot one runtime-owned organism + Node0 stack."""
        try:
            from core.sovereign.organism import SovereignOrganism

            organism = await SovereignOrganism.boot(
                inference=_RuntimeInferenceBackend(self),
                persistence_dir=self.config.state_dir / "node0",
                event_bus=self._event_bus,
                start_heartbeat=False,
                node_id=self.config.node_id,
                identity_mode=self._identity_mode,
                signer_public_key_prefix=self._signer_public_key_prefix,
                signer_public_key_hex=self._signer_public_key_hex_for(
                    self._node_signer
                ),
            )
            self._organism = organism
            self._node0 = getattr(organism, "_node0", None)
        except (ImportError, AttributeError, RuntimeError, OSError):
            self._organism = None
            self._node0 = None
            if self._canonical_mode:
                raise
            self.logger.warning(
                "Runtime-owned organism stack unavailable; mission authority degraded",
                exc_info=True,
            )

    def _init_autopoiesis_stack(self) -> None:
        """Wire the opt-in autopoiesis loop into the runtime lifecycle."""
        self._autopoietic_loop = None
        self._learning_loop = None
        self._autopoiesis_learning_source = "disabled"

        if not self.config.enable_autopoiesis:
            self.logger.info("○ Autopoiesis disabled by config")
            return

        try:
            from core.autopoiesis.loop_engine import (
                ActivationGuardrails,
                AutopoieticLoop,
            )

            node0_learning_loop = (
                getattr(self._node0, "_learning_loop", None)
                if self._node0 is not None
                else None
            )
            node0_reflex_bridge = (
                getattr(self._node0, "_reflex_bridge", None)
                if self._node0 is not None
                else None
            )

            learning_loop = node0_learning_loop
            learning_source = "node0_shared"

            if learning_loop is None:
                from core.orchestration.learning_loop import LearningLoopOrchestrator

                learning_loop = LearningLoopOrchestrator(
                    reflex_bridge=node0_reflex_bridge,
                )
                learning_source = "standalone"

            on_integration = None
            if learning_loop is not None and hasattr(learning_loop, "on_candidate"):
                on_integration = getattr(learning_loop, "on_candidate")

            z3_fate_gate = getattr(self._fate_gate, "_z3_gate", None)
            guardrails = ActivationGuardrails(
                require_live_sensors=True,
                allow_mock_sensors=False,
                require_fate_gate=True,
                allow_mock_fate_gate=not self._canonical_mode,
                min_ihsan_score=self.config.ihsan_threshold,
                min_snr_score=self.config.snr_threshold,
            )
            self._autopoietic_loop = AutopoieticLoop(
                fate_gate=z3_fate_gate,
                ihsan_floor=self.config.ihsan_threshold,
                snr_floor=self.config.snr_threshold,
                cycle_interval_s=self.config.autopoiesis_cycle_seconds,
                activation_guardrails=guardrails,
                on_integration=on_integration,
            )
            self._learning_loop = learning_loop
            self._autopoiesis_learning_source = learning_source
            self.logger.info(
                "✓ Autopoiesis wired (learning_source=%s, learning_loop=%s, cycle=%.1fs)",
                learning_source,
                learning_loop is not None,
                self.config.autopoiesis_cycle_seconds,
            )
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
        ):
            self._autopoietic_loop = None
            self._learning_loop = None
            self._autopoiesis_learning_source = "unavailable"
            self.logger.warning(
                "Autopoiesis stack unavailable; runtime continuing without self-improvement",
                exc_info=True,
            )

    def _task_is_running(self, task: asyncio.Task[Any] | None) -> bool:
        """Return True when an asyncio task exists and has not completed."""
        return task is not None and not task.done()

    def _record_autopoiesis_receipt(
        self,
        payload: dict[str, Any],
        *,
        receipt_kind: str,
    ) -> None:
        """Feed canonical runtime receipts into the autopoiesis observe plane."""
        loop_obj = self._autopoietic_loop
        recorder = getattr(loop_obj, "record_receipt_observation", None)
        if loop_obj is None or not callable(recorder):
            return
        try:
            recorder(payload, receipt_kind=receipt_kind)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            self.logger.debug(
                "Autopoiesis receipt observation failed for %s",
                receipt_kind,
                exc_info=True,
            )

    def _observe_autopoiesis_mission_receipt(self, receipt: Any) -> None:
        """Normalize an organism mission receipt into autopoiesis truth."""
        payload = {
            "mission_id": str(getattr(receipt, "mission_id", "") or ""),
            "ihsan_score": float(getattr(receipt, "ihsan_score", 0.0) or 0.0),
            "snr_score": float(getattr(receipt, "snr_score", 0.0) or 0.0),
            "duration_ms": float(getattr(receipt, "duration_ms", 0.0) or 0.0),
            "gate_passed": bool(getattr(receipt, "gate_passed", False)),
            "fate_verdict": str(getattr(receipt, "fate_verdict", "unknown") or ""),
        }
        self._record_autopoiesis_receipt(payload, receipt_kind="mission")

    def _observe_autopoiesis_breath_receipt(self, breath: Any) -> None:
        """Normalize a heartbeat receipt into autopoiesis truth."""
        helix_result = getattr(breath, "helix_result", {}) or {}
        payload = {
            "tick_number": int(getattr(breath, "tick_number", 0) or 0),
            "ihsan_composite": float(getattr(breath, "ihsan_composite", 0.0) or 0.0),
            "snr_score": float(getattr(breath, "ihsan_composite", 0.0) or 0.0),
            "duration_ms": float(getattr(breath, "duration_ms", 0.0) or 0.0),
            "gini_ok": bool(getattr(breath, "gini_ok", False)),
            "missions_processed": int(getattr(breath, "missions_processed", 0) or 0),
            "approved_count": int(helix_result.get("approved_count", 0) or 0),
            "rejected_count": int(helix_result.get("rejected_count", 0) or 0),
            "reflexes_precipitated": int(
                getattr(breath, "reflexes_precipitated", 0) or 0
            ),
        }
        self._record_autopoiesis_receipt(payload, receipt_kind="heartbeat")

    async def _run_autopoiesis_learning_loop(self) -> None:
        """Periodically flush learning-loop training + compilation."""
        cycle_seconds = max(1.0, float(self.config.autopoiesis_cycle_seconds))

        while True:
            try:
                learning_loop = self._learning_loop
                if learning_loop is not None and hasattr(
                    learning_loop, "run_full_cycle"
                ):
                    cycle_result = getattr(learning_loop, "run_full_cycle")()
                    if inspect.isawaitable(cycle_result):
                        await cycle_result
                await asyncio.sleep(cycle_seconds)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                self.logger.warning("Autopoiesis learning cycle error", exc_info=True)
                await asyncio.sleep(cycle_seconds)

    def _start_autopoiesis_tasks(self) -> None:
        """Start tracked background tasks for autopoiesis and learning."""
        if not self.config.enable_autopoiesis or self._autopoietic_loop is None:
            return
        if self._task_is_running(self._autopoiesis_task):
            return

        self._autopoiesis_task = asyncio.create_task(
            self._autopoietic_loop.start(),
            name="runtime_autopoiesis_loop",
        )

        if self._learning_loop is not None and not self._task_is_running(
            self._autopoiesis_learning_task
        ):
            self._autopoiesis_learning_task = asyncio.create_task(
                self._run_autopoiesis_learning_loop(),
                name="runtime_autopoiesis_learning",
            )

        self.logger.info(
            "Autopoiesis background tasks started (learning_source=%s)",
            self._autopoiesis_learning_source,
        )

    async def _stop_autopoiesis_tasks(self) -> None:
        """Stop and await any running autopoiesis background tasks."""
        loop_obj = self._autopoietic_loop
        if loop_obj is not None and hasattr(loop_obj, "stop"):
            try:
                stop_result = getattr(loop_obj, "stop")()
                if inspect.isawaitable(stop_result):
                    await stop_result
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                self.logger.debug("Autopoiesis stop hook failed", exc_info=True)

        for attr_name in ("_autopoiesis_learning_task", "_autopoiesis_task"):
            task = getattr(self, attr_name)
            if task is None:
                continue
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            setattr(self, attr_name, None)

    async def _preflight_mission(
        self,
        description: str,
        *,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the canonical TeleScript + FATE preflight for a mission."""
        context = dict(context or {})
        action_bus = self._action_bus
        if action_bus is None:
            if self._canonical_mode:
                raise RuntimeError("Canonical mission authority requires ActionBus")
            return {
                "allow_execution": True,
                "fate_verdict": "degraded",
                "fate_reason_codes": ["action_bus_unavailable"],
                "fate_mode": "degraded",
                "action_receipt_refs": [],
            }

        from core.bus.telescript import Capability
        from core.bus.types import ActionBudget, ActionEnvelope, ActionStatus
        from core.proof_engine.canonical import canonical_bytes

        mission_id = _hex_digest(
            canonical_bytes(
                {
                    "description": description,
                    "source": source,
                    "context": context,
                }
            )
        )
        actor_id = b""
        if self._node_signer is not None and hasattr(
            self._node_signer, "public_key_bytes"
        ):
            try:
                actor_id = getattr(self._node_signer, "public_key_bytes")()
            except (AttributeError, TypeError):
                actor_id = b""

        time_budget_seconds = float(context.get("time_budget_seconds", 900.0))
        payload = {
            "description": description,
            "source": source,
            "context": context,
            "ihsan": 1.0,
            "snr": 1.0,
            "time_budget_seconds": time_budget_seconds,
            "autonomy_limit": time_budget_seconds,
            "cost": min(time_budget_seconds, float(context.get("estimated_cost", 1.0))),
            "risk_level": context.get("risk_level"),
            "reversible": context.get("reversible", True),
            "human_approved": context.get("human_approved", False),
        }
        if payload["risk_level"] is None:
            payload.pop("risk_level")

        receipt = await action_bus.propose(
            ActionEnvelope(
                action_id=mission_id,
                kind="mission.execute",
                channel="mission_gate",
                payload=payload,
                capabilities=(Capability.LLM_QUERY.value,),
                budget=ActionBudget(time_ms=int(time_budget_seconds * 1000)),
                correlation_id=mission_id,
                actor_id=actor_id,
                timestamp=int(time.time() * 1000),
            )
        )
        status_value = getattr(getattr(receipt, "status", None), "value", "")
        denied = status_value in {
            ActionStatus.DENIED.value,
            ActionStatus.FAILED.value,
        }
        if denied:
            reason = str(getattr(receipt, "error_message", "") or "fate_veto")
            return {
                "allow_execution": False,
                "mission_id": mission_id,
                "fate_verdict": "rejected",
                "fate_reason_codes": [reason],
                "fate_mode": self._fate_mode,
                "action_receipt_refs": [str(getattr(receipt, "receipt_id", ""))],
            }
        return {
            "allow_execution": True,
            "mission_id": mission_id,
            "fate_verdict": (
                "approved" if self._fate_mode == "enforced" else "degraded"
            ),
            "fate_reason_codes": [],
            "fate_mode": self._fate_mode,
            "action_receipt_refs": [str(getattr(receipt, "receipt_id", ""))],
        }

    @classmethod
    @asynccontextmanager
    async def create(
        cls, config: RuntimeConfig | None = None
    ) -> AsyncIterator[SovereignRuntime]:
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
        self._canonical_mode = self._resolve_canonical_mode()
        self._node_role = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        enforce_node0_fail_closed(self.config.state_dir, self._node_role)
        self._origin_snapshot = resolve_origin_snapshot(
            self.config.state_dir, self._node_role
        )

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME INITIALIZING")
        self.logger.info("=" * 60)

        # Initialize sovereign event bus for cross-component pub/sub.
        self._init_event_bus()
        self._start_event_bus_task()

        # Initialize Phase 70 bus infrastructure (ActionBus, TopicRegistry, Config, Capsules)
        self._init_bus_infrastructure()

        # Load genesis identity (persistent node_id from ceremony)
        self._load_genesis_identity()

        self.logger.info(f"Node ID: {self.config.node_id}")
        self.logger.info(f"Mode: {self.config.mode.name}")
        self.logger.info(f"Ihsan Threshold: {self.config.ihsan_threshold}")
        self.logger.info(f"Node Role: {self._node_role}")

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

        # Initialize Evidence Ledger (append-only, hash-chained audit trail)
        self._init_evidence_ledger()

        # Initialize Sovereign Experience Ledger (content-addressed episodic memory)
        self._init_experience_ledger()

        # Initialize Self-Evolving Judgment Engine (observation telemetry)
        self._init_judgment_telemetry()

        # Initialize unified Node0 signer (Ed25519 identity)
        self._init_node_signer()

        # Rewire ActionBus so canonical missions pass through TeleScript + FATE once.
        self._configure_canonical_action_bus()

        # Initialize IHSAN_FLOOR watchdog (MCG Layer 7 governance)
        try:
            from core.proof_engine.ihsan_gate import IhsanFloorWatchdog

            self._ihsan_watchdog = IhsanFloorWatchdog(
                max_consecutive_failures=3,
                floor=0.90,
            )
            self.logger.info(
                "IhsanFloor watchdog initialized (floor=0.90, max_failures=3)"
            )
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.warning(f"IhsanFloor watchdog init failed: {e}")

        # Initialize 6-Gate Chain (fail-closed execution pipeline)
        self._init_gate_chain()

        # Initialize Proof-of-Impact Engine (4-stage scoring pipeline)
        self._init_poi_engine()

        # Trusted bootstrap gate (optional fail-closed preflight)
        await self._run_zpk_preflight()

        await self._init_components()
        self._enforce_stub_budget_gate()

        if self.config.autonomous_enabled:
            await self._start_autonomous_loop()

        # Initialize user context (the system knows its human)
        self._init_user_context()

        # Initialize unified memory coordinator with auto-save
        await self._init_memory_coordinator()

        # Initialize Phase 31: Cognitive Fusion (HyperGraph + Fusion Engine + Memory Coder)
        self._init_cognitive_fusion()

        # Initialize Phase 32: Embedding Service + NTU Adapter
        self._init_embedding_service()

        # Initialize Phase 25-28: Ecosystem subsystems (HRM + NorthStar + Guild + Quest)
        self._init_ecosystem_subsystems()

        # Initialize Phase 33: RDVE Engine (Recursive Discovery & Verification)
        self._init_rdve_engine()

        # Initialize impact tracker (sovereignty growth engine)
        self._init_impact_tracker()

        self._setup_signal_handlers()

        # Initialize SpearPoint Pipeline — the unified cockpit
        self._init_spearpoint_pipeline()

        # Initialize Spearpoint Orchestrator (reproduce / improve / heartbeat)
        self._init_spearpoint_orchestrator()

        # Phase 58: Initialize Equalizer Agent + Unified Model Router
        self._init_equalizer_and_router()

        # Runtime-canonical organism authority — one organism, one Node0, one signer.
        await self._init_canonical_organism_stack()
        self._init_autopoiesis_stack()

        # Phase 80: Runtime Daemons — PAT loop, SAT loop, DEMA routing, FATE boundary
        self._init_urp_service()
        self._init_fate_boundary()
        await self._init_pat_runtime()
        await self._init_sat_runtime()
        self._init_dema_router()
        self._activate_genesis_agents()
        self._init_proactive_scheduler()

        self._initialized = True
        self._running = True
        self.metrics.started_at = datetime.now()
        self._start_autopoiesis_tasks()

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME READY")
        self.logger.info("=" * 60)

    def _init_equalizer_and_router(self) -> None:
        """Initialize Equalizer Agent and Unified Model Router."""
        # Equalizer Agent — cognitive-debt homeostasis control loop
        try:
            from core.sovereign.equalizer_agent import EqualizerAgent

            self._equalizer_agent = EqualizerAgent(
                ihsan_target=self.config.ihsan_threshold,
            )
            self.logger.info(
                "EqualizerAgent initialized (ihsan_target=%.2f)",
                self.config.ihsan_threshold,
            )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning("EqualizerAgent init skipped: %s", e)

        # Unified Model Router — auto-failover LM Studio / Ollama
        try:
            from tools.engines.unified_model_router import UnifiedModelRouter

            self._unified_model_router = UnifiedModelRouter()
            self.logger.info(
                "UnifiedModelRouter registered (deferred init on first query)"
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.warning("UnifiedModelRouter init skipped: %s", e)

    def _init_event_bus(self) -> None:
        """Initialize sovereign event bus used by runtime side-channels."""
        try:
            from .event_bus import get_event_bus

            self._event_bus = get_event_bus()
            self.logger.info("✓ Sovereign EventBus initialized")
        except (ImportError, RuntimeError, AttributeError) as e:
            self._event_bus = None
            self.logger.warning("⚠ Sovereign EventBus unavailable: %s", e)

    def _start_event_bus_task(self) -> None:
        """Start the sovereign async event bus when it exposes a run loop."""
        if self._task_is_running(self._event_bus_task):
            return
        if self._event_bus is None or not hasattr(self._event_bus, "start"):
            return
        stats = getattr(self._event_bus, "stats", None)
        if callable(stats):
            try:
                if bool(stats().get("running", False)):
                    return
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                self.logger.debug(
                    "Event bus stats unavailable during start", exc_info=True
                )
        start = getattr(self._event_bus, "start", None)
        if not callable(start):
            return
        self._event_bus_task = asyncio.create_task(
            start(),
            name="runtime_event_bus",
        )

    async def _stop_event_bus_task(self) -> None:
        """Stop and await the sovereign event bus loop if running."""
        task = self._event_bus_task
        if task is None:
            return

        bus = self._event_bus
        if bus is not None and hasattr(bus, "stop"):
            stop = getattr(bus, "stop", None)
            if callable(stop):
                try:
                    stop()
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                    self.logger.debug("Event bus stop hook failed", exc_info=True)

        if not self._task_is_running(task):
            self._event_bus_task = None
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            self.logger.debug("Event bus shutdown failed", exc_info=True)
        self._event_bus_task = None

    def _init_bus_infrastructure(self) -> None:
        """Initialize Phase 70 bus infrastructure with graceful fallback.

        Wires: TopicRegistry, TeleScript, ActionBus, ConfigLoader,
        CapsuleRuntime, OmegaLoopController. Each component fails
        independently — runtime continues at reduced capability.
        """
        try:
            from core.bus.sovereign_wiring import wire_all

            components, wiring_state = wire_all(
                event_bus=self._event_bus,
                state_dir=self.config.state_dir,
            )

            self._action_bus = components.get("action_bus")
            self._topic_registry = components.get("topic_registry")
            self._telescript_engine = components.get("telescript_engine")
            self._config_loader = components.get("config_loader")
            self._capsule_registry = components.get("capsule_registry")
            self._capsule_runtime = components.get("capsule_runtime")
            self._omega_controller = components.get("omega_controller")
            self._bus_wiring_state = wiring_state

            if wiring_state.all_ok:
                self.logger.info("✓ Bus infrastructure fully wired (5/5)")
            else:
                self.logger.warning(
                    "⚠ Bus infrastructure partially wired: %s",
                    wiring_state.summary,
                )
        except (
            ImportError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            self.logger.warning("⚠ Bus infrastructure unavailable: %s", exc)

        # Phase 71: Seed Engine (DDAGI growth trajectory)
        self._init_seed_engine()

    def _init_seed_engine(self) -> None:
        """Initialize Phase 71 Seed Potential Engine with graceful fallback."""
        try:
            from core.sovereign.seed_engine import create_seed_engine

            self._seed_engine = create_seed_engine(runtime=self)
            self.logger.info("✓ Seed Engine initialized")
        except (
            ImportError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            self.logger.warning("⚠ Seed Engine unavailable: %s", exc)

    async def _dispatch_equalizer_command(self, eq_cmd: object) -> None:
        """Act on an EqualizerAgent command instead of just logging it.

        Dispatches the command to the event bus so that listeners (including
        the AutoModelRouter and any future consumers) can react.
        """
        try:
            from core.sovereign.equalizer_agent import EqualizerCommandKind

            kind = eq_cmd.kind  # type: ignore[attr-defined]
            reason = eq_cmd.reason  # type: ignore[attr-defined]

            if self._event_bus is not None and hasattr(self._event_bus, "emit"):
                from .event_bus import EventPriority

                await self._event_bus.emit(
                    topic="equalizer.command",
                    payload={
                        "kind": kind.value,
                        "reason": reason,
                        "batch_scale": getattr(eq_cmd, "batch_scale", 1),
                    },
                    priority=EventPriority.HIGH,
                    source="sovereign.runtime.equalizer",
                )

            # Direct action on the runtime when possible
            if kind == EqualizerCommandKind.HALT:
                self.logger.warning(
                    "Equalizer HALT: ihsan critical — pausing non-essential queries"
                )
            elif kind == EqualizerCommandKind.ESCALATE:
                self.logger.info("Equalizer ESCALATE: requesting larger model variant")
            elif kind == EqualizerCommandKind.RESUME:
                self.logger.info(
                    "Equalizer RESUME: recovery detected, resuming normal ops"
                )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.debug("Equalizer dispatch error: %s", e)

    @staticmethod
    def _is_stub_component(component: object | None) -> bool:
        """Return True when a component is a stub/fallback implementation."""
        if component is None:
            return True
        if getattr(component, "is_stub", False):
            return True
        return "stub" in type(component).__name__.lower()

    def _enforce_stub_budget_gate(self) -> None:
        """Fail-closed startup gate for strict runtime profiles."""
        self._stub_components = []
        self._strict_gate_reason_codes = []

        tracked_components = [
            (
                "graph_reasoner",
                self.config.enable_graph_reasoning,
                self._graph_reasoner,
            ),
            ("snr_optimizer", self.config.enable_snr_optimization, self._snr_optimizer),
            (
                "guardian_council",
                self.config.enable_guardian_validation,
                self._guardian_council,
            ),
            (
                "autonomous_loop",
                self.config.enable_autonomous_loop,
                self._autonomous_loop,
            ),
        ]

        for name, enabled, component in tracked_components:
            if enabled and self._is_stub_component(component):
                self._stub_components.append(name)

        if not self.config.strict_stub_budget:
            self._strict_gate_passed = True
            return

        if self.config.reject_stub_inference and any(
            name in {"graph_reasoner", "snr_optimizer", "guardian_council"}
            for name in self._stub_components
        ):
            self._strict_gate_reason_codes.append("STRICT_STUB_INFERENCE_COMPONENT")

        if len(self._stub_components) > self.config.stub_budget_max:
            self._strict_gate_reason_codes.append("STRICT_STUB_BUDGET_EXCEEDED")

        if self._strict_gate_reason_codes:
            self._strict_gate_passed = False
            reasons = ",".join(self._strict_gate_reason_codes)
            components = ",".join(self._stub_components)
            raise RuntimeError(
                "Strict startup gate failed "
                f"(reasons={reasons}, stub_components={components}, "
                f"budget={self.config.stub_budget_max})"
            )

        self._strict_gate_passed = True

    def _init_evidence_ledger(self) -> None:
        """Initialize the Evidence Ledger — append-only, hash-chained audit trail.

        Every query and verification call emits a receipt into this ledger.
        Standing on: Lamport (event ordering), Merkle (hash chains).
        """
        try:
            from core.proof_engine.evidence_ledger import EvidenceLedger

            ledger_path = self.config.state_dir / "evidence.jsonl"
            self._evidence_ledger = EvidenceLedger(ledger_path, validate_on_append=True)
            self.logger.info(
                f"Evidence Ledger initialized: {ledger_path} "
                f"(seq={self._evidence_ledger.sequence})"
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"Evidence Ledger init failed (non-fatal): {e}")
            self._evidence_ledger = None

    def _init_experience_ledger(self) -> None:
        """Initialize the Sovereign Experience Ledger — episodic memory.

        Content-addressed, hash-chained episodic memory store.
        Auto-commits episodes on every SNR_OK query verdict.

        Standing on: Tulving (episodic memory), Besta (GoT artifacts),
        Park et al. (generative agent memory).
        """
        try:
            from core.sovereign.experience_ledger import SovereignExperienceLedger

            self._experience_ledger = SovereignExperienceLedger()
            self.logger.info("Sovereign Experience Ledger initialized")
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.debug(f"Experience Ledger init skipped (non-fatal): {e}")
            self._experience_ledger = None

    def _init_judgment_telemetry(self) -> None:
        """Initialize the Self-Evolving Judgment Engine — observation telemetry.

        Phase A: Observation mode only. Records verdict distributions and
        computes Shannon entropy. NO policy mutation. NO threshold changes.

        Standing on: Shannon (1948), Aristotle (Nicomachean Ethics).
        """
        try:
            from core.sovereign.judgment_telemetry import JudgmentTelemetry

            self._judgment_telemetry = JudgmentTelemetry()
            self.logger.info("Judgment Telemetry (SJE Phase A) initialized")
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.debug(f"Judgment Telemetry init skipped (non-fatal): {e}")
            self._judgment_telemetry = None

    def _observe_judgment(self, result: SovereignResult) -> None:
        """Observe a verdict for the SJE based on query result quality.

        Verdict classification (observation only — no policy mutation):
          PROMOTE: ihsan >= 0.95 and snr_ok (excellence)
          NEUTRAL: snr_ok and ihsan >= ihsan_threshold (acceptable)
          DEMOTE:  not snr_ok (below SNR floor)
          FORBID:  validation explicitly failed

        Fire-and-forget: SJE failures never block query responses.
        """
        if self._judgment_telemetry is None:
            return
        try:
            from core.sovereign.judgment_telemetry import JudgmentVerdict

            if not result.success or (
                result.validated and not result.validation_passed
            ):
                verdict = JudgmentVerdict.FORBID
            elif not result.snr_ok:
                verdict = JudgmentVerdict.DEMOTE
            elif result.ihsan_score >= 0.95:
                verdict = JudgmentVerdict.PROMOTE
            else:
                verdict = JudgmentVerdict.NEUTRAL

            self._judgment_telemetry.observe(verdict)
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            self.logger.debug(f"SJE observe skipped (non-fatal): {e}")

    def _commit_experience_episode(
        self, result: SovereignResult, query: SovereignQuery
    ) -> None:
        """Auto-commit a query episode to the SEL on SNR_OK verdict.

        Standing on: Tulving (episodic encoding), Shannon (SNR gating).
        Fire-and-forget: SEL failures never block query responses.
        """
        if self._experience_ledger is None:
            return
        if not result.success or not result.snr_ok:
            return
        try:
            from core.proof_engine.canonical import hex_digest

            # Build graph hash from GoT thoughts (if available)
            graph_hash = ""
            graph_node_count = 0
            if result.thoughts:
                graph_hash = hex_digest(
                    "|".join(result.thoughts).encode("utf-8")
                )  # SEC-001: BLAKE3 for Rust interop
                graph_node_count = len(result.thoughts)

            # Build action log
            actions = []
            model_used = result.model_used
            if model_used:
                actions.append(
                    (
                        "inference",
                        f"LLM: {model_used}",
                        True,
                        int(result.processing_time_ms * 1_000),
                    )
                )
            if result.snr_ok:
                actions.append(
                    (
                        "snr_gate",
                        f"SNR={result.snr_score:.3f}",
                        True,
                        0,
                    )
                )

            # Truncate response for storage
            response_summary = (result.response or "")[:500] or None

            self._experience_ledger.commit(
                context=query.text[:500],
                graph_hash=graph_hash,
                graph_node_count=graph_node_count,
                actions=actions,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                snr_ok=result.snr_ok,
                response_summary=response_summary,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.debug(f"SEL commit skipped (non-fatal): {e}")

    def _init_node_signer(self) -> None:
        """Initialize the unified Node0 Ed25519 signer.

        All subsystems (GateChain, PoI, Evidence) use this single identity.
        Standing on: Bernstein (Ed25519, 2011).
        """
        try:
            from core.proof_engine.receipt import Ed25519Signer

            signer_material = self._load_identity_credentials()
            if signer_material is not None:
                self._node_signer = Ed25519Signer(
                    private_key_hex=signer_material["private_key_hex"],
                    public_key_hex=signer_material["public_key_hex"],
                )
                self._identity_mode = "genesis_ed25519"
                self._genesis_backed_identity = True
                if signer_material["node_id"]:
                    self.config.node_id = signer_material["node_id"]
            else:
                if self._canonical_mode:
                    raise RuntimeError(
                        "Canonical mode requires sovereign_state/identity/credentials.json"
                    )
                from core.sovereign.mission import _load_or_create_node_signer

                private_hex, public_hex = _load_or_create_node_signer(
                    {"sovereign_state_dir": str(self.config.state_dir)}
                )
                self._node_signer = Ed25519Signer(
                    private_key_hex=private_hex,
                    public_key_hex=public_hex,
                )
                self._identity_mode = "placeholder_degraded"
                self._genesis_backed_identity = False
            self._signer_public_key_prefix = self._signer_public_key_prefix_for(
                self._node_signer
            )
            self.logger.info(
                "Node0 signer initialized: %s... (mode=%s)",
                self._signer_public_key_prefix,
                self._identity_mode,
            )
        except (
            ImportError,
            RuntimeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as e:
            if self._canonical_mode:
                raise
            self.logger.warning(
                f"Ed25519 signer init degraded, falling back to HMAC: {e}"
            )
            from core.proof_engine.receipt import SimpleSigner

            self._node_signer = SimpleSigner(
                secret=self.config.node_id.encode("utf-8") + b"_node0_v1"
            )
            self._identity_mode = "placeholder_degraded"
            self._genesis_backed_identity = False
            self._signer_public_key_prefix = self._signer_public_key_prefix_for(
                self._node_signer
            )

    def _init_gate_chain(self) -> None:
        """Initialize the 6-Gate Chain — fail-closed execution pipeline.

        The GateChain runs as a pre-flight check before query processing.
        If any gate fails, the query is rejected with a signed receipt.

        Standing on: Lamport (fail-closed), Dijkstra (structured decomposition).
        """
        try:
            from core.proof_engine.gates import GateChain

            # Use the unified Node0 signer for all receipts
            if self._node_signer is None:
                self._init_node_signer()
            self._gate_chain = GateChain(signer=self._node_signer)
            self.logger.info(
                f"GateChain initialized: " f"{[g.name for g in self._gate_chain.gates]}"
            )
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            # CRITICAL-1 FIX (Saltzer & Schroeder 1975): Fail-CLOSED, not fail-OPEN.
            # When GateChain can't initialize, ALL queries must be rejected,
            # not silently bypassed. Previously set self._gate_chain = None
            # which caused line 578 to return None (pass-through).
            self.logger.error(
                f"GateChain init FAILED — all queries will be REJECTED until resolved: {e}"
            )
            self._gate_chain = None  # _run_gate_chain_preflight now rejects when None

    async def _run_gate_chain_preflight(
        self, query: SovereignQuery, result: SovereignResult
    ) -> SovereignResult | None:
        """Run the 6-Gate Chain as a pre-flight check.

        If any gate fails, returns a rejection SovereignResult immediately.
        If all gates pass (or gate chain is disabled), returns None to continue.

        Standing on: Lamport (fail-closed), BIZRA Spearpoint (6-gate chain).
        """
        if self._gate_chain is None:
            # CRITICAL-1 FIX: Reject ALL queries when gate chain unavailable.
            # Previously returned None (pass-through), violating IHSAN_FLOOR.
            self.logger.warning("GateChain unavailable — REJECTING query (fail-closed)")
            result.success = False
            result.response = (
                "Query rejected: Gate chain unavailable. "
                "Constitutional invariants cannot be verified."
            )
            result.validation_passed = False
            return result

        try:
            from core.proof_engine.canonical import CanonPolicy, CanonQuery

            canon_query = CanonQuery(
                user_id=(
                    query.user_id
                    if (hasattr(query, "user_id") and query.user_id)
                    else "anonymous"
                ),
                user_state=(
                    query.context.get("user_state", "active")
                    if query.context
                    else "active"
                ),
                intent=query.text,
            )
            canon_policy = CanonPolicy(
                policy_id="sovereign_v1",
                version="1.0.0",
                rules={"snr_min": 0.95, "ihsan_min": self.config.ihsan_threshold},
                thresholds={
                    "snr": 0.95,
                    "ihsan": self.config.ihsan_threshold,
                },
            )
            # Bootstrap Ihsan: At cold start (no queries processed yet), the system
            # IS constitutionally compliant — all gates are active, all invariants
            # are enforced. Use the configured threshold as the initial score.
            # After first query, measured Ihsan takes over.
            ihsan_for_gate = self.metrics.current_ihsan_score
            if ihsan_for_gate is None or (
                ihsan_for_gate == 0.0 and self.metrics.total_queries == 0
            ):
                ihsan_for_gate = (
                    self.config.ihsan_threshold
                )  # System IS compliant at boot

            # CRITICAL-3 FIX: Compute Z3 satisfiability instead of assuming True.
            # Standing on: ZANN_ZERO ("no assumptions"), Lamport (verify, don't trust).
            z3_sat = False  # Fail-closed default
            z3_action_ctx = {
                "ihsan": ihsan_for_gate,
                "snr": 0.85,  # Pre-inference minimum SNR gate
                "cost": 0.0,
                "autonomy_limit": 10.0,  # Default limit
                "risk_level": 0.3,  # Read-only query = low risk
                "reversible": True,
                "human_approved": False,
            }
            try:
                from core.sovereign.z3_fate_gate import Z3FATEGate

                z3_gate = Z3FATEGate()
                z3_proof = z3_gate.generate_proof(z3_action_ctx)
                z3_sat = z3_proof.satisfiable
            except (ImportError, RuntimeError, ValueError, OSError) as z3_err:
                # Z3 unavailable — degrade to conservative fallback module (α4).
                # ImportError: z3-solver not installed (graceful degradation).
                # Richer than the inline _conservative_fallback_check: returns
                # FallbackVerdict with reason codes, action type gating, and
                # Z3 re-validation flags. Standing on: Lamport (verify).
                self.logger.debug(
                    "Z3 solver unavailable, using conservative fallback "
                    f"(default-deny, stricter thresholds): {z3_err}"
                )
                try:
                    from .conservative_fallback import conservative_fallback_check

                    # Enrich context with action_type for the module's safe-set check
                    enriched_ctx = {**z3_action_ctx, "action_type": "query"}
                    verdict = conservative_fallback_check(enriched_ctx)
                    z3_sat = verdict.approved
                    if not verdict.approved:
                        self.logger.info(
                            "Conservative fallback REJECTED: %s",
                            verdict.reason_detail,
                        )
                except ImportError:
                    # Module unavailable — fall back to inline check
                    z3_sat = _conservative_fallback_check(z3_action_ctx)

            # Risk assessment: read-only queries are low risk.
            # State-mutating ops or cloud API would score higher.
            base_risk = 0.1  # Read-only query default

            context = {
                "trust_score": 0.6,  # Local system has earned base trust
                "ihsan_score": ihsan_for_gate,
                "z3_satisfiable": z3_sat,
                "risk_score": base_risk,
                "source_trust_score": 0.6,
                "prediction_accuracy": 0.5,
                "context_fit_score": 0.5,
            }

            chain_result, receipt = self._gate_chain.evaluate(
                canon_query, canon_policy, context
            )

            if chain_result.passed:
                self.logger.debug(
                    f"GateChain PASSED: all {len(chain_result.gate_results)} gates"
                )
                return None

            # Gate chain failed — build rejection result
            self.logger.warning(
                f"GateChain REJECTED at gate '{chain_result.last_gate_passed}': "
                f"{chain_result.rejection_reason}"
            )
            result.success = False
            result.response = (
                f"Query rejected by gate chain: {chain_result.rejection_reason}"
            )
            result.snr_score = chain_result.snr
            result.snr_ok = chain_result.snr >= self.config.snr_threshold
            result.ihsan_score = chain_result.ihsan_score
            result.validation_passed = False
            result.claim_tags = {"gate_chain": "measured"}
            return result

        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as e:
            # CRITICAL-2 FIX (Saltzer & Schroeder 1975): Fail-CLOSED on gate errors.
            # Previously returned None (pass-through), allowing queries to bypass
            # ALL constitutional gates on ANY exception.
            self.logger.error(f"GateChain preflight FAILED — REJECTING query: {e}")
            result.success = False
            result.response = f"Query rejected: Gate chain error ({e})"
            result.validation_passed = False
            return result

    def _emit_query_receipt(
        self, result: SovereignResult, query: SovereignQuery
    ) -> None:
        """Emit a receipt for a completed query into the Evidence Ledger.

        CRITICAL-10 FIX: Failures are now LOGGED at WARNING level (visible in metrics).
        Non-blocking, but no longer invisible.
        """
        if self._evidence_ledger is None:
            return
        try:
            from core.proof_engine.canonical import hex_digest
            from core.proof_engine.evidence_ledger import emit_receipt

            decision, status, reason_codes = self._receipt_outcome(result)

            query_digest = hex_digest(
                query.text.encode("utf-8")
            )  # SEC-001: BLAKE3 for Rust interop

            seal_digest = hex_digest(
                (result.response or "").encode("utf-8")
            )  # SEC-001: BLAKE3 for Rust interop

            receipt_id = result.query_id.replace("-", "")[:32]
            entry = emit_receipt(
                self._evidence_ledger,
                receipt_id=receipt_id,
                node_id=self.config.node_id,
                policy_version="1.0.0",
                status=status,
                decision=decision,
                reason_codes=reason_codes,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                ihsan_threshold=self.config.ihsan_threshold,
                seal_digest=seal_digest,
                query_digest=query_digest,
                graph_hash=result.graph_hash,
                payload_digest=(
                    hex_digest("|".join(result.thoughts).encode("utf-8"))
                    if result.thoughts
                    else None
                ),  # SEC-001: BLAKE3 for Rust interop
                gate_passed="commit" if decision == "APPROVED" else "ihsan_gate",
                duration_ms=result.processing_time_ms,
                claim_tags=(
                    {
                        "measured": sum(
                            1 for v in result.claim_tags.values() if v == "measured"
                        ),
                        "design": sum(
                            1 for v in result.claim_tags.values() if v == "design"
                        ),
                        "implemented": sum(
                            1 for v in result.claim_tags.values() if v == "implemented"
                        ),
                        "target": sum(
                            1 for v in result.claim_tags.values() if v == "target"
                        ),
                    }
                    if result.claim_tags
                    else None
                ),
                snr_trace=self._last_snr_trace,
                origin=self._origin_snapshot,
                critical_decision=True,
                node_role=self._node_role,
                state_dir=self.config.state_dir,
            )
            if isinstance(query.context, dict):
                query.context["_last_receipt_id"] = receipt_id
                query.context["_last_receipt_decision"] = decision
                query.context["_last_receipt_entry_hash"] = getattr(
                    entry, "entry_hash", None
                )
            # Clear trace after emission
            self._last_snr_trace = None
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"Receipt emission failed (non-fatal): {e}")

    @staticmethod
    def _receipt_outcome(
        result: SovereignResult,
    ) -> tuple[str, str, list[str]]:
        """Compute canonical receipt decision/status/reason codes.

        Delegates to the single source of truth in receipt_outcome module.
        """
        from .receipt_outcome import receipt_outcome

        return receipt_outcome(result)

    async def _apply_receipt_memory_feedback(
        self, result: SovereignResult, query: SovereignQuery
    ) -> None:
        """
        SEL SENSE stage feedback: reinforce/flag source memories from receipt outcome.

        Success path reinforces memory IDs used in contextual retrieval.
        Rejected/quarantined path marks those IDs for healing revalidation.
        """
        if self._living_memory is None or not isinstance(query.context, dict):
            return
        source_ids = query.context.get("_source_memory_ids")
        if not isinstance(source_ids, list) or not source_ids:
            return

        apply_feedback = getattr(self._living_memory, "apply_execution_feedback", None)
        if apply_feedback is None:
            return

        decision, _, reason_codes = self._receipt_outcome(result)
        success = decision == "APPROVED"
        reason = ",".join(reason_codes) if reason_codes else decision
        receipt_ref = query.context.get("_last_receipt_id") or result.query_id
        try:
            feedback_or_coro = apply_feedback(
                source_ids,
                success=success,
                reason=reason,
                receipt_ref=receipt_ref,
            )
            feedback = (
                await feedback_or_coro
                if inspect.isawaitable(feedback_or_coro)
                else feedback_or_coro
            )
            if isinstance(feedback, dict):
                action = "reinforced" if success else "flagged"
                self.logger.debug(
                    "Memory receipt feedback %s=%s decision=%s receipt=%s",
                    action,
                    feedback.get(action, 0),
                    decision,
                    receipt_ref,
                )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"Memory receipt feedback skipped (non-fatal): {e}")

    def _schedule_receipt_memory_feedback(
        self, result: SovereignResult, query: SovereignQuery
    ) -> None:
        """Schedule non-blocking SEL SENSE feedback wiring."""
        try:
            asyncio.ensure_future(self._apply_receipt_memory_feedback(result, query))
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(
                f"Failed to schedule receipt memory feedback (non-fatal): {e}"
            )

    def _register_poi_contribution(
        self, result: SovereignResult, query: SovereignQuery
    ) -> None:
        """Register a successful query as a PoI contribution.

        Fire-and-forget: PoI failures never block query responses.

        Standing on: Nakamoto (PoW), Shannon (SNR as quality),
        Al-Ghazali (proportional justice).
        """
        if self._poi_orchestrator is None:
            return
        if not result.success:
            return

        try:
            from core.proof_engine.poi_engine import (
                ContributionMetadata,
                ContributionType,
            )

            content_hash = result.graph_hash or result.query_id
            metadata = ContributionMetadata(
                contributor_id=self.config.node_id,
                contribution_type=ContributionType.DATA,
                content_hash=content_hash,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                timestamp=datetime.now(),
            )
            self._poi_orchestrator.register_contribution(metadata)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            # CRITICAL-10 FIX: PoI failures must be VISIBLE, not silent.
            self.logger.warning(f"PoI contribution registration failed: {e}")

    def _encode_query_memory(
        self, result: SovereignResult, query: SovereignQuery
    ) -> None:
        """Encode successful query experience into Living Memory.

        Standing on: Tulving (1972) — episodic memory as experiential encoding.
        Fire-and-forget: memory failures never block query responses.
        """
        if self._living_memory is None or not result.success:
            return
        if not result.response:
            return
        try:
            from core.living_memory.core import MemoryType

            # Truncate for memory efficiency (keep first 500 chars of each)
            q_text = query.text[:500]
            r_text = (result.response or "")[:500]

            content = (
                f"Query: {q_text}\n"
                f"Response: {r_text}\n"
                f"SNR: {result.snr_score:.3f} | Ihsan: {result.ihsan_score:.3f}"
            )

            # Schedule encoding as background task (non-blocking)
            import asyncio

            asyncio.ensure_future(
                self._living_memory.encode(
                    content=content,
                    memory_type=MemoryType.EPISODIC,
                    source="query_pipeline",
                    importance=result.ihsan_score,
                    emotional_weight=max(result.snr_score, 0.5),
                )
            )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.debug(f"Memory encoding skipped (non-fatal): {e}")

    def _store_graph_artifact(self, query_id: str, graph_hash: str | None) -> None:
        """Store the GoT graph artifact for later retrieval via API.

        Standing on: Besta (GoT, 2024) — graph artifacts are first-class,
        Merkle (1979) — content-addressed integrity.

        Fire-and-forget: exceptions are caught and logged.
        """
        try:
            if not self._graph_reasoner:
                return
            # The GraphOfThoughts instance has to_artifact()
            to_artifact = getattr(self._graph_reasoner, "to_artifact", None)
            if to_artifact is None:
                return
            artifact = to_artifact(build_id=query_id)
            self._graph_artifacts[query_id] = artifact
            # Bound storage to prevent unbounded memory growth
            if len(self._graph_artifacts) > 100:
                oldest = next(iter(self._graph_artifacts))
                del self._graph_artifacts[oldest]
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"Graph artifact storage failed (non-fatal): {e}")

    def get_graph_artifact(self, query_id: str) -> dict[str, Any] | None:
        """Retrieve a stored graph artifact by query ID."""
        return self._graph_artifacts.get(query_id)

    def get_gate_chain_stats(self) -> dict[str, Any] | None:
        """Get GateChain evaluation statistics."""
        if self._gate_chain is None:
            return None
        return self._gate_chain.get_stats()

    def _init_poi_engine(self) -> None:
        """Initialize the Proof-of-Impact Engine — 4-stage scoring pipeline.

        Standing on: Nakamoto (PoW), Page & Brin (PageRank), Gini (inequality),
        Al-Ghazali (proportional justice), Shannon (SNR as quality).
        """
        try:
            from core.proof_engine.poi_engine import PoIConfig, PoIOrchestrator

            config = PoIConfig()
            self._poi_orchestrator = PoIOrchestrator(config)
            # Wire unified Node0 signer into PoI for receipt signing
            if self._node_signer is not None and hasattr(
                self._poi_orchestrator, "_signer"
            ):
                self._poi_orchestrator._signer = self._node_signer
            self.logger.info(
                f"PoI Engine initialized: "
                f"alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}"
            )

            # Initialize SAT Controller with the PoI orchestrator
            from core.sovereign.sat_controller import SATController

            self._sat_controller = SATController(
                poi_orchestrator=self._poi_orchestrator,
                config=config,
            )
            self.logger.info("SAT Controller initialized")
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"PoI Engine init failed (non-fatal): {e}")
            self._poi_orchestrator = None
            self._sat_controller = None

    def get_poi_stats(self) -> dict[str, Any] | None:
        """Get Proof-of-Impact engine statistics."""
        if self._poi_orchestrator is None:
            return None
        return self._poi_orchestrator.get_stats()

    def get_contributor_poi(self, contributor_id: str) -> dict[str, Any] | None:
        """Get most recent PoI for a contributor."""
        if self._poi_orchestrator is None:
            return None
        poi = self._poi_orchestrator.get_contributor_poi(contributor_id)
        if poi is None:
            return None
        return poi.to_dict()

    def compute_poi_epoch(self, epoch_id: str | None = None) -> dict[str, Any] | None:
        """Run a full PoI computation epoch.

        Returns the audit trail as a dict, or None if engine is unavailable.
        """
        if self._poi_orchestrator is None:
            return None
        audit = self._poi_orchestrator.compute_epoch(epoch_id)
        return audit.to_dict()

    def get_sat_stats(self) -> dict[str, Any] | None:
        """Get SAT Controller statistics."""
        if self._sat_controller is None:
            return None
        return self._sat_controller.get_stats()

    def finalize_sat_epoch(self, epoch_reward: float = 1000.0) -> dict[str, Any] | None:
        """Finalize a PoI epoch via SAT Controller.

        Computes scores, distributes tokens, checks Gini, rebalances if needed.
        """
        if self._sat_controller is None:
            return None
        return self._sat_controller.finalize_epoch(epoch_reward)

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
                if self._node_role == "node0":
                    raise RuntimeError(
                        "Node0 role requires validated genesis identity; none found"
                    )
                self.logger.info("No genesis — running as ephemeral node")
        except ValueError as e:
            if self._node_role == "node0":
                raise RuntimeError(f"Genesis identity corrupted: {e}") from e
            self.logger.error(f"Genesis identity corrupted: {e}")

    async def _init_components(self) -> None:
        """Initialize components with graceful fallback.

        RFC-01 FIX: Respects feature flags from RuntimeConfig.
        """
        # Try full GraphOfThoughts (only if flag enabled)
        if self.config.enable_graph_reasoning:
            try:
                from .graph_reasoner import GraphOfThoughts

                self._graph_reasoner = GraphOfThoughts()  # type: ignore[assignment]
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

                self._snr_optimizer = SNRMaximizer(  # type: ignore[assignment]
                    ihsan_threshold=self.config.snr_threshold
                )
                self.logger.info("✓ SNRMaximizer loaded (full)")
            except ImportError:
                self._snr_optimizer = StubFactory.create_snr_optimizer("Import failed")  # type: ignore[assignment]
                self.logger.warning("⚠ SNRMaximizer unavailable, using stub")
        else:
            self._snr_optimizer = StubFactory.create_snr_optimizer("Disabled by config")  # type: ignore[assignment]
            self.logger.info("○ SNRMaximizer disabled by config")

        # Try full GuardianCouncil (only if flag enabled)
        if self.config.enable_guardian_validation:
            try:
                from .guardian_council import GuardianCouncil

                self._guardian_council = GuardianCouncil()  # type: ignore[assignment]
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
                self._autonomous_loop = AutonomousLoop(  # type: ignore[assignment]
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

        # CMN Runtime — Constitutional Membrane Network invariant harness
        try:
            from core.sovereign.cmn_runtime import CMNRuntime

            seed_ledger = getattr(self, "_seed_ledger_path", None)
            self._cmn_runtime = CMNRuntime(
                data_dir=self.config.data_dir,
                node_id=self.config.node_id,
                seed_ledger_path=seed_ledger,
            )
            cmn_report = self._cmn_runtime.boot()
            ok_count = sum(1 for v in cmn_report.values() if v == "ok")
            self.logger.info(
                "CMN runtime booted: %d/%d components ok", ok_count, len(cmn_report)
            )
        except Exception as exc:
            self._cmn_runtime = None
            self.logger.warning("CMN runtime unavailable: %s", exc)

        # TRUE SPEARPOINT: Wire InferenceGateway into GraphOfThoughts post-hoc.
        # GoT is initialized before the gateway (which lives in omega components),
        # so we inject the gateway after both are ready.
        if (
            self._gateway is not None
            and self._graph_reasoner is not None
            and hasattr(self._graph_reasoner, "_inference_gateway")
        ):
            self._graph_reasoner._inference_gateway = self._gateway  # type: ignore[union-attr]
            self.logger.info(
                "✓ SPEARPOINT: InferenceGateway wired into GraphOfThoughts — "
                "GoT will use real LLM for hypothesis generation and conclusions"
            )

        # Wire InferenceGateway into Guardian Council for LLM-backed evaluation
        if self._guardian_council and self._gateway:
            if hasattr(self._guardian_council, "set_inference_gateway"):
                self._guardian_council.set_inference_gateway(self._gateway)
                self.logger.info(
                    "✓ SPEARPOINT: InferenceGateway wired into GuardianCouncil — "
                    "Guardians can use LLM for proposal evaluation"
                )

        # α9 Performance Attestation — thermodynamic anomaly detection
        try:
            from .performance_attestation import PerformanceAttestor

            self._performance_attestor = PerformanceAttestor()
            self.logger.info("✓ PerformanceAttestor loaded (α9)")
        except ImportError:
            self.logger.debug("○ PerformanceAttestor unavailable")

        # α7 Tiered Verification — multi-speed constitutional verification
        try:
            from .tiered_verification import tier_1_precheck  # noqa: F401

            self._tiered_verification_enabled = True
            self.logger.info("✓ TieredVerification loaded (α7)")
        except ImportError:
            self.logger.debug("○ TieredVerification unavailable")

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
        except (RuntimeError, ValueError, TypeError, OSError) as e:
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
            except (ImportError, RuntimeError, AttributeError) as e:
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
            from core.inference.gateway import (  # type: ignore[attr-defined]
                CircuitBreakerConfig,
                InferenceConfig,
                InferenceGateway,
            )

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
            except (TimeoutError, OSError, RuntimeError) as init_err:  # SEC-003
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

        # SovereignOrchestrator — task decomposition + agent routing
        try:
            from .orchestrator import RoutingStrategy, SovereignOrchestrator

            orch = SovereignOrchestrator(routing_strategy=RoutingStrategy.ADAPTIVE)
            orch.register_default_agents()
            if self._gateway:
                orch.set_gateway(self._gateway)
            self._orchestrator = orch
            self.logger.info("✓ SovereignOrchestrator loaded (Adaptive routing)")
        except ImportError as e:
            self._orchestrator = None
            self.logger.warning(f"⚠ SovereignOrchestrator unavailable: {e}")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self._orchestrator = None
            self.logger.warning(f"⚠ SovereignOrchestrator init failed: {e}")

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
                except (ImportError, RuntimeError, AttributeError) as event_err:
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
                from .z3_fate_gate import Z3_AVAILABLE, Z3FATEGate

                if Z3_AVAILABLE:
                    self._pek.set_fate_gate(Z3FATEGate())
                    self.logger.info("✓ PEK FATE gate enabled (Z3)")
            except (ImportError, RuntimeError, ValueError, OSError) as fate_err:
                self.logger.warning(f"⚠ PEK FATE gate unavailable: {fate_err}")

            await self._pek.start()

            # Wire PEK into validation pipeline (SNR + Guardian + Evidence)
            if self._snr_optimizer and hasattr(self._pek, "set_snr_optimizer"):
                self._pek.set_snr_optimizer(self._snr_optimizer)
            if self._guardian_council and hasattr(self._pek, "set_guardian_council"):
                self._pek.set_guardian_council(self._guardian_council)
            if self._evidence_ledger and hasattr(self._pek, "set_evidence_ledger"):
                self._pek.set_evidence_ledger(self._evidence_ledger)

            self.logger.info("✓ ProactiveExecutionKernel started")
        except (RuntimeError, ValueError, TypeError, OSError) as e:
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
                # Wire memory into orchestrator for context-aware task execution
                if self._orchestrator and hasattr(self._orchestrator, "set_memory"):
                    self._orchestrator.set_memory(living_memory)
                self.logger.info("✓ LivingMemory connected to auto-save")
            except ImportError:
                self.logger.warning("⚠ LivingMemory unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ LivingMemory init failed: {e}")

            # Initialize AgentDB (V3 unified memory with HNSW indexing)
            try:
                from core.memory import AgentDB, MemoryConfig
                from core.memory.coordinator_bridge import AgentDBBridge
                from core.memory.health import AgentDBHealthChecker
                from core.memory.orchestrator import MigrationOrchestrator

                agent_db_config = MemoryConfig(
                    data_dir=self.config.state_dir / "agent_db",
                    living_memory_db=self.config.state_dir
                    / "living_memory"
                    / "memory.db",
                )
                self._agent_db = AgentDB(agent_db_config)
                self._agent_db.initialize()

                # V3 Bridge: register AgentDB with MemoryCoordinator
                # (replaces manual state_provider + adds HNSW flush on save)
                self._agent_db_bridge = AgentDBBridge(
                    self._agent_db, self._memory_coordinator
                )
                self._agent_db_bridge.register()

                # V3 Health: wire health checker for monitoring
                self._agent_db_health = AgentDBHealthChecker(self._agent_db)

                self.logger.info(
                    f"✓ AgentDB initialized: {self._agent_db.count} records, "
                    f"{self._agent_db.hnsw.count} vectors"
                )

                # Wire embedding function into AgentDB (if service already up)
                if (
                    hasattr(self, "_embedding_service")
                    and self._embedding_service is not None
                ):
                    try:
                        self._agent_db.set_embedding_fn(self._embedding_service.embed)
                        self.logger.debug("AgentDB embedding function wired (early)")
                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                        AttributeError,
                        OSError,
                    ) as emb_err:
                        self.logger.debug(f"AgentDB embedding fn not wired: {emb_err}")

                # V3 Migration Orchestrator: import from all legacy sources
                try:
                    orch = MigrationOrchestrator(self._agent_db)
                    if self._living_memory is not None:
                        orch.set_living_memory(self._living_memory)
                    if self._experience_ledger is not None:
                        orch.set_experience_ledger(self._experience_ledger)
                    result = orch.run()
                    if result.total_imported > 0:
                        self.logger.info(
                            f"AgentDB migration: {result.total_imported} records "
                            f"imported ({result.total_errors} errors)"
                        )
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    OSError,
                ) as mig_err:
                    self.logger.debug(f"AgentDB migration skipped: {mig_err}")

            except ImportError:
                self.logger.warning("⚠ AgentDB unavailable (core.memory not installed)")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ AgentDB init failed: {e}")

            # Start auto-save background loop
            if self.config.enable_persistence:
                await self._memory_coordinator.start_auto_save()
                self.logger.info("✓ MemoryCoordinator auto-save active")

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"⚠ MemoryCoordinator init failed: {e}")

    def _init_cognitive_fusion(self) -> None:
        """Initialize Phase 31 cognitive fusion subsystems.

        Wires HyperGraphStore + CognitiveFusionEngine + MemorySynthesizer
        into the sovereign runtime. All components are optional — graceful
        degradation if any import fails.

        Standing on: Berge (hypergraph) + Simon (hierarchy) + Shannon (SNR)
        """
        # 1. HyperGraph Store
        try:
            from core.hypergraph import HyperGraphStore

            self._hypergraph_store = HyperGraphStore()
            self.logger.info("✓ HyperGraphStore initialized")
        except ImportError:
            self.logger.warning("⚠ HyperGraphStore unavailable")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"⚠ HyperGraphStore init failed: {e}")

        # 2. Cognitive Fusion Engine (requires HyperGraph + AgentDB)
        if self.config.enable_cognitive_fusion:
            try:
                from core.cognitive_fusion import CognitiveFusionEngine
                from core.hypergraph import HyperGraphRAGFusion

                rag_fusion = None
                if self._hypergraph_store is not None:
                    rag_fusion = HyperGraphRAGFusion(
                        store=self._hypergraph_store,
                        agent_db=self._agent_db,
                    )

                self._cognitive_fusion = CognitiveFusionEngine(
                    hypergraph_rag=rag_fusion,
                )
                self.logger.info("✓ CognitiveFusionEngine initialized")
            except ImportError:
                self.logger.warning("⚠ CognitiveFusionEngine unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ CognitiveFusionEngine init failed: {e}")
        else:
            self.logger.info("○ CognitiveFusionEngine disabled by config")

        # 3. Memory Synthesizer + Pattern Codebook
        if self.config.enable_memory_synthesizer:
            try:
                from core.memory_coder import (
                    MemorySynthesizer,
                    PatternCodebook,
                )

                self._pattern_codebook = PatternCodebook(
                    agent_db=self._agent_db,
                )
                self._memory_synthesizer = MemorySynthesizer(
                    agent_db=self._agent_db,
                    codebook=self._pattern_codebook,
                )
                self.logger.info(
                    f"✓ MemorySynthesizer initialized "
                    f"(codebook: {self._pattern_codebook.size} patterns)"
                )
            except ImportError:
                self.logger.warning("⚠ MemorySynthesizer unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ MemorySynthesizer init failed: {e}")
        else:
            self.logger.info("○ MemorySynthesizer disabled by config")

    def _init_embedding_service(self) -> None:
        """Initialize Phase 32 embedding service and NTU fusion adapter.

        Provides real embeddings for CognitiveFusion (replacing dummy [0.0]*768)
        and temporal context enrichment from NTU pattern detection.

        Standing on: Reimers & Gurevych (sentence-BERT) + Takens (temporal patterns)
        """
        # 1. Embedding Service (tiered: sentence-transformers → Ollama)
        try:
            from core.embedding import (
                EmbeddingConfig,
                EmbeddingQualityGate,
                EmbeddingService,
            )

            self._embedding_service = EmbeddingService(EmbeddingConfig.from_env())
            self._embedding_gate = EmbeddingQualityGate()
            self.logger.info("✓ EmbeddingService initialized (tiered fallback)")

            # Late-bind embedding function into AgentDB (initialized earlier)
            if self._agent_db is not None:
                try:
                    self._agent_db.set_embedding_fn(self._embedding_service.embed)
                    self.logger.info(
                        "AgentDB embedding function wired via EmbeddingService"
                    )
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    OSError,
                ) as wire_err:
                    self.logger.debug(
                        f"AgentDB embedding fn late-wire failed: {wire_err}"
                    )
        except ImportError:
            self.logger.warning("⚠ EmbeddingService unavailable")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"⚠ EmbeddingService init failed: {e}")

        # 2. NTU Fusion Adapter
        try:
            from core.ntu import NTUBridge, NTUFusionAdapter

            bridge = NTUBridge()
            self._ntu_adapter = NTUFusionAdapter(ntu_bridge=bridge)
            self.logger.info("✓ NTUFusionAdapter initialized")
        except ImportError:
            self.logger.warning("⚠ NTUFusionAdapter unavailable (numpy required)")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"⚠ NTUFusionAdapter init failed: {e}")

    def _init_ecosystem_subsystems(self) -> None:
        """Initialize Phase 25-28 ecosystem subsystems.

        Wires HRM cognitive engine + NorthStar flagship + Guild membership +
        Quest mission engine. All optional — graceful degradation.

        Standing on: Simon (hierarchy) + Csikszentmihalyi (flow) + Ostrom (commons)
        """
        # 1. Hierarchical Reasoning Model
        if self.config.enable_hrm:
            try:
                from core.hrm import HierarchicalReasoningModel, HRMConfig

                self._hrm_engine = HierarchicalReasoningModel(HRMConfig())
                num_levels = len(self._hrm_engine._config.active_levels)
                self.logger.info(f"✓ HRM initialized ({num_levels} levels)")
            except ImportError:
                self.logger.warning("⚠ HRM unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ HRM init failed: {e}")
        else:
            self.logger.info("○ HRM disabled by config")

        # 2. NorthStar Engine (cognitive flagship)
        if self.config.enable_northstar:
            try:
                from core.northstar import NorthStarEngine

                self._northstar_engine = NorthStarEngine()
                self.logger.info("✓ NorthStar Engine initialized")
            except ImportError:
                self.logger.warning("⚠ NorthStar unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ NorthStar init failed: {e}")
        else:
            self.logger.info("○ NorthStar disabled by config")

        # 3. Guild Registry (collaborative communities)
        if self.config.enable_guild_system:
            try:
                from core.guild import GuildRegistry

                self._guild_registry = GuildRegistry()
                guild_count = len(self._guild_registry._guilds)
                self.logger.info(f"✓ GuildRegistry initialized ({guild_count} guilds)")
            except ImportError:
                self.logger.warning("⚠ GuildRegistry unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ GuildRegistry init failed: {e}")
        else:
            self.logger.info("○ GuildRegistry disabled by config")

        # 4. Quest Engine (impact missions)
        if self.config.enable_quest_system:
            try:
                from core.quest import QuestEngine

                self._quest_engine = QuestEngine()
                quest_count = len(self._quest_engine._quests)
                self.logger.info(f"✓ QuestEngine initialized ({quest_count} quests)")
            except ImportError:
                self.logger.warning("⚠ QuestEngine unavailable")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"⚠ QuestEngine init failed: {e}")
        else:
            self.logger.info("○ QuestEngine disabled by config")

    def _init_rdve_engine(self) -> None:
        """Initialize Phase 33: RDVE Engine (Recursive Discovery & Verification).

        Wires the RDVEOrchestrator into the sovereign runtime as a background
        campaign runner. RDVE is NOT in the per-query pipeline — it is an async
        "slower scientist" that runs discovery cycles via run_campaign().

        The orchestrator self-wires its subcomponents (HypothesisGenerator,
        GoTHypothesisExplorer, SNRMaximizer, AutopoieticLoop) using sensible
        defaults, so no explicit dependency injection is needed here.

        Standing on: Shannon (SNR) + Besta (GoT) + Maturana (autopoiesis) +
                     Boyd (OODA) + Deming (PDCA) + Al-Ghazali (Ihsan)
        """
        try:
            from core.rdve import RDVEConfig, RDVEOrchestrator

            self._rdve_engine = RDVEOrchestrator(config=RDVEConfig())
            self.logger.info(
                f"✓ RDVE Engine initialized "
                f"(SNR floor={self._rdve_engine.config.snr_floor:.2f}, "
                f"Ihsan floor={self._rdve_engine.config.ihsan_floor:.2f})"
            )
        except ImportError:
            self.logger.warning("⚠ RDVE Engine unavailable (missing dependencies)")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"⚠ RDVE Engine init failed: {e}")

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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            self.logger.warning(f"⚠ ImpactTracker init failed: {e}")

    def _init_spearpoint_pipeline(self) -> None:
        """Initialize the SpearPoint Pipeline — unified post-query cockpit.

        Consolidates 7 fire-and-forget operations into one observable,
        error-isolated pipeline. Each step tracks success/failure independently.

        Standing on: Lamport (fail-closed), Shannon (SNR gating).
        """
        try:
            from .spearpoint_pipeline import SpearPointPipeline

            # SNR trace is passed via mutable single-element list reference
            self._snr_trace_slot: list = [self._last_snr_trace]
            self._spearpoint = SpearPointPipeline(
                evidence_ledger=self._evidence_ledger,
                graph_reasoner=self._graph_reasoner,
                graph_artifacts=self._graph_artifacts,
                living_memory=self._living_memory,
                experience_ledger=self._experience_ledger,
                poi_orchestrator=self._poi_orchestrator,
                judgment_telemetry=self._judgment_telemetry,
                impact_tracker=self._impact_tracker,
                sat_controller=self._sat_controller,
                config=self.config,
                snr_trace_ref=self._snr_trace_slot,
            )
            self.logger.info("SpearPoint Pipeline (cockpit) initialized")
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"SpearPoint Pipeline init failed (non-fatal): {e}")
            self._spearpoint = None

    def _init_spearpoint_orchestrator(self) -> None:
        """Initialize the Spearpoint Orchestrator — mission router for
        reproduce (evaluation) and improve (research) operations.

        Shares the evidence ledger with the runtime so receipts flow
        into the same append-only chain.

        Standing on: Boyd (OODA loop), Goldratt (Theory of Constraints).
        """
        try:
            from core.spearpoint.config import SpearpointConfig
            from core.spearpoint.orchestrator import SpearpointOrchestrator

            config = SpearpointConfig.from_env()

            # Share the runtime's evidence ledger path if available
            if self._evidence_ledger is not None:
                ledger_path = getattr(self._evidence_ledger, "path", None)
                if ledger_path is not None:
                    config.evidence_ledger_path = ledger_path

            self._spearpoint_orchestrator = SpearpointOrchestrator(config=config)
            self.logger.info(
                f"Spearpoint Orchestrator initialized "
                f"(ihsan={config.ihsan_threshold}, snr={config.snr_threshold})"
            )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.warning(f"Spearpoint Orchestrator init failed (non-fatal): {e}")
            self._spearpoint_orchestrator = None

    def _get_impact_state(self) -> dict[str, Any]:
        """Provide impact tracker state for memory coordinator."""
        if not self._impact_tracker:
            return {}
        try:
            progress = self._impact_tracker.get_progress()
            return progress.to_dict()
        except (RuntimeError, ValueError, TypeError, OSError):
            return {}

    def _record_query_impact(self, result: SovereignResult) -> None:
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
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            # CRITICAL-10 FIX: Impact failures must be VISIBLE, not silent.
            self.logger.warning(f"Impact recording failed: {e}")

    def _get_runtime_state(self) -> dict[str, Any]:
        """Provide runtime state snapshot for memory coordinator."""
        state: dict[str, Any] = {
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
                "hypergraph_store": self._hypergraph_store is not None,
                "cognitive_fusion": self._cognitive_fusion is not None,
                "memory_synthesizer": self._memory_synthesizer is not None,
                "hrm_engine": self._hrm_engine is not None,
                "northstar_engine": self._northstar_engine is not None,
                "guild_registry": self._guild_registry is not None,
                "quest_engine": self._quest_engine is not None,
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
                if self._memory_coordinator is None:
                    return
                self._memory_coordinator.register_state_provider(
                    "pek",
                    self._pek.get_persistable_state,
                    RestorePriority.SAFETY,
                )
                self.logger.debug("Registered PEK state provider")
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                self.logger.warning(
                    "Failed to register PEK state provider", exc_info=True
                )

        # OpportunityPipeline — SAFETY priority (rate limiter must survive restarts)
        try:
            from .opportunity_pipeline import OpportunityPipeline

            pipeline = OpportunityPipeline()
            if self._memory_coordinator is None:
                return
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
            if self._memory_coordinator is None:
                return
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
            if self._memory_coordinator is None:
                return
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
            await self._autonomous_loop.start()
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

    # ─── Phase 80: Runtime Daemon Init Methods ─────────────────────────

    def _init_urp_service(self) -> None:
        """Initialize URP Service as in-process singleton."""
        try:
            from core.urp.service import URPService

            self._urp_service = URPService()
            self.logger.info("✓ URPService initialized (in-process)")
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
            self.logger.warning(f"URPService init skipped: {e}")

    def _init_fate_boundary(self) -> None:
        """Initialize FATE boundary membrane (PAT↔URP crossing gate)."""
        try:
            from core.sovereign.fate_boundary import FATEBoundary

            self._fate_boundary = FATEBoundary(
                fate_gate=self._fate_gate,
                ihsan_threshold=self.config.ihsan_threshold,
                receipt_dir=self.config.state_dir / "receipts",
            )
            self.logger.info(
                "✓ FATE Boundary initialized "
                f"(ihsan≥{self.config.ihsan_threshold}, "
                f"fate_gate={'online' if self._fate_gate else 'degraded'})"
            )
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
            self.logger.warning(f"FATE Boundary init skipped: {e}")

    async def _init_pat_runtime(self) -> None:
        """Initialize PAT-7 agent runtime daemon."""
        try:
            from core.pat.runtime import PATRuntime

            pat_team = []
            if self._genesis and hasattr(self._genesis, "pat_team"):
                pat_team = list(self._genesis.pat_team)

            self._pat_runtime = PATRuntime(
                agents=pat_team,
                query_fn=self.query if hasattr(self, "query") else None,
                receipt_dir=self.config.state_dir / "receipts",
                fate_boundary=self._fate_boundary,
            )
            await self._pat_runtime.start()
            self.logger.info(
                f"✓ PAT Runtime started ({self._pat_runtime.agent_count} agents, "
                f"{self._pat_runtime.active_count} active)"
            )
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
            self.logger.warning(f"PAT Runtime init skipped: {e}")

    async def _init_sat_runtime(self) -> None:
        """Initialize SAT-5 validation runtime daemon."""
        try:
            from core.sat.runtime import SATRuntime

            sat_team = []
            if self._genesis and hasattr(self._genesis, "sat_team"):
                sat_team = list(self._genesis.sat_team)

            self._sat_runtime = SATRuntime(
                agents=sat_team,
                receipt_dir=self.config.state_dir / "receipts",
                ihsan_threshold=self.config.ihsan_threshold,
            )
            await self._sat_runtime.start()
            self.logger.info(
                f"✓ SAT Runtime started ({len(sat_team)} agents, "
                f"{len(self._sat_runtime._gates_loaded)} gates)"
            )
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
            self.logger.warning(f"SAT Runtime init skipped: {e}")

    def _init_dema_router(self) -> None:
        """Initialize DEMA Router (user directive → PAT agent routing)."""
        try:
            from core.sovereign.dema_router import DEMARouter

            pat_team = []
            if self._genesis and hasattr(self._genesis, "pat_team"):
                pat_team = list(self._genesis.pat_team)

            select_fn = None
            if self._user_context and hasattr(self._user_context, "select_pat_agent"):
                select_fn = self._user_context.select_pat_agent

            self._dema_router = DEMARouter(
                pat_runtime=self._pat_runtime,
                select_agent_fn=select_fn,
                pat_team=pat_team,
            )
            self.logger.info("✓ DEMA Router initialized")
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
            self.logger.warning(f"DEMA Router init skipped: {e}")

    def _activate_genesis_agents(self) -> None:
        """Activate all genesis PAT + SAT agents (DORMANT → ACTIVE)."""
        if not self._genesis:
            return
        activated = 0
        for team_name, team in [
            ("PAT", self._genesis.pat_team),
            ("SAT", self._genesis.sat_team),
        ]:
            for agent_id in team:
                agent = agent_id  # AgentIdentity objects
                if hasattr(agent, "activate") and callable(agent.activate):
                    try:
                        if agent.activate():
                            activated += 1
                    except (RuntimeError, TypeError, ValueError):
                        pass
        if activated > 0:
            self.logger.info(
                f"✓ {activated} genesis agents activated (DORMANT → ACTIVE)"
            )

    def _init_proactive_scheduler(self) -> None:
        """Initialize ProactiveScheduler with morning briefing mission."""
        try:
            from core.sovereign.proactive_scheduler import (
                ProactiveScheduler,
                ScheduleType,
            )

            self._proactive_scheduler = ProactiveScheduler(
                check_interval=30.0,
                max_concurrent=3,
            )

            # Register morning briefing (Dubai 07:00 GST = 03:00 UTC, daily)
            async def morning_briefing() -> str:
                """Generate morning briefing via DEMA router."""
                if self._dema_router:
                    self._dema_router.route(
                        "Generate morning briefing: summarize yesterday's activity, "
                        "active missions, node health, and priority items for today.",
                        requester_id="scheduler",
                    )
                return "Morning briefing dispatched"

            self._proactive_scheduler.schedule(
                name="morning_briefing",
                handler=morning_briefing,
                schedule_type=ScheduleType.RECURRING,
                interval_seconds=86400,
                priority=2,
            )

            # Start scheduler as background task
            self._proactive_scheduler_task = asyncio.create_task(
                self._proactive_scheduler.start(),
                name="proactive_scheduler",
            )
            self.logger.info(
                "✓ ProactiveScheduler started (morning_briefing registered, 24h cycle)"
            )
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as e:
            self.logger.warning(f"ProactiveScheduler init skipped: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown the runtime."""
        if not self._running:
            return

        self.logger.info("Initiating graceful shutdown...")
        self._running = False

        if self._autonomous_loop:
            self._autonomous_loop.stop()

        await self._stop_autopoiesis_tasks()
        await self._stop_event_bus_task()

        # Stop Phase 80 runtime daemons (PAT → SAT → Scheduler)
        if self._pat_runtime and hasattr(self._pat_runtime, "stop"):
            try:
                await self._pat_runtime.stop()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
                self.logger.debug("PAT Runtime stop failed", exc_info=True)
        if self._sat_runtime and hasattr(self._sat_runtime, "stop"):
            try:
                await self._sat_runtime.stop()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
                self.logger.debug("SAT Runtime stop failed", exc_info=True)
        if self._proactive_scheduler and hasattr(self._proactive_scheduler, "stop"):
            try:
                self._proactive_scheduler.stop()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
                self.logger.debug("Scheduler stop failed", exc_info=True)
        if self._proactive_scheduler_task:
            self._proactive_scheduler_task.cancel()
            try:
                await self._proactive_scheduler_task
            except asyncio.CancelledError:
                pass

        if self._pek and hasattr(self._pek, "stop"):
            try:
                await self._pek.stop()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
                self.logger.debug("PEK stop failed during shutdown", exc_info=True)

        # Save user context (conversation history + profile)
        if self._user_context:
            try:
                self._user_context.save()
            except (OSError, RuntimeError, ValueError):
                self.logger.warning(
                    "Failed to save user context during shutdown", exc_info=True
                )

        # Flush impact tracker dirty state before memory coordinator stop
        if self._impact_tracker and hasattr(self._impact_tracker, "flush"):
            try:
                self._impact_tracker.flush()
            except (RuntimeError, ValueError, TypeError, OSError):
                self.logger.warning(
                    "Failed to flush impact tracker during shutdown", exc_info=True
                )

        # Stop memory coordinator (performs final save including all providers)
        # LCT-01 FIX: MemoryCoordinator.stop() already checkpoints all state.
        # The old _checkpoint() was a redundant second save of the same data.
        if self._memory_coordinator:
            await self._memory_coordinator.stop()

        if self._organism and hasattr(self._organism, "shutdown"):
            try:
                await self._organism.shutdown()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
                self.logger.warning(
                    "Failed to shutdown runtime-owned organism cleanly",
                    exc_info=True,
                )

        self._shutdown_event.set()
        self.logger.info("Sovereign Runtime shutdown complete")

    async def wait_for_shutdown(self) -> None:
        """Wait until shutdown is complete."""
        await self._shutdown_event.wait()

    async def mission(
        self,
        description: str,
        *,
        source: str,
        context: Any | None = None,
        proof_mode: str = "verified",
    ) -> Any:
        """Execute one canonical mission through the runtime-owned organism."""
        del proof_mode  # Proof selection remains an API/UI concern; authority stays here.

        if not self._initialized:
            raise RuntimeError("Runtime not initialized. Call initialize() first.")
        if self._organism is None:
            raise RuntimeError("Canonical organism mission authority unavailable")

        mission_context = dict(context or {})
        preflight = await self._preflight_mission(
            description,
            source=source,
            context=mission_context,
        )
        receipt = await self._organism.mission(description, preflight=preflight)
        self._observe_autopoiesis_mission_receipt(receipt)
        try:
            breath = await self._organism.tick()
            self._observe_autopoiesis_breath_receipt(breath)
        except (RuntimeError, AttributeError, TypeError, OSError):
            if self._canonical_mode:
                raise
            self.logger.warning(
                "Canonical organism breath failed after mission execution",
                exc_info=True,
            )
        return receipt

    # -------------------------------------------------------------------------
    # QUERY PROCESSING
    # -------------------------------------------------------------------------

    async def query(
        self, content: str, context: dict[str, Any] | None = None, **options
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
            user_id=options.get("user_id", ""),
        )

        start_time = time.perf_counter()
        # RFC-03 FIX: Don't manually increment here — update_query_stats() is
        # the single source of truth for all query counters.

        # Check cache
        cache_key = self._cache_key(query)
        if self.config.enable_cache and cache_key in self._cache:
            self.metrics.cache_hits += 1
            self._cache.move_to_end(cache_key)  # LRU refresh
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

            # Equalizer Agent: observe Ihsan homeostasis after each query
            if self._equalizer_agent is not None:
                try:
                    self._equalizer_agent.observe(
                        layer=self.metrics.total_queries % 256,
                        ihsan_score=result.ihsan_score,
                        backlog=len(self._cache),
                        presence=255 if query.user_id else 0,
                    )
                    eq_cmd = self._equalizer_agent.next_command()
                    if eq_cmd is not None:
                        self.logger.info(
                            "Equalizer: %s reason=%s", eq_cmd.kind.value, eq_cmd.reason
                        )
                        # Act on equalizer command via event bus
                        await self._dispatch_equalizer_command(eq_cmd)
                except (ImportError, RuntimeError, AttributeError) as eq_err:
                    self.logger.debug("Equalizer observe error: %s", eq_err)

            return result

        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_query_stats(False, duration_ms)
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=f"Query timeout after {query.timeout}s",
                user_id=query.user_id,
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_query_stats(False, duration_ms)
            self.logger.error(f"Query error: {e}")
            return SovereignResult(
                query_id=query.id,
                success=False,
                error=str(e),
                user_id=query.user_id,
            )

    def _estimate_complexity(self, query: SovereignQuery) -> float:
        """Estimate query complexity on 0.0-1.0 scale for orchestrator routing.

        Standing on: DSPy (Stanford, 2024) — self-optimizing prompt complexity.
        Signals: word count, sub-question markers, domain breadth, explicit hints.
        """
        text = query.text
        words = text.split()
        word_count = len(words)

        # Length signal (long queries tend to be complex)
        length_score = min(word_count / 80, 1.0)

        # Sub-question markers
        sub_q_keywords = {
            "and also",
            "additionally",
            "furthermore",
            "then",
            "compare",
            "contrast",
            "analyze",
            "evaluate",
            "step by step",
            "multi",
            "comprehensive",
            "full",
        }
        sub_q_score = sum(0.15 for kw in sub_q_keywords if kw in text.lower())

        # Question count
        q_count = text.count("?")
        q_score = min(q_count * 0.2, 0.6)

        # Explicit complexity hint from context
        hint = query.context.get("complexity_hint", 0.0)

        score = min(
            1.0,
            0.3 * length_score
            + 0.3 * min(sub_q_score, 1.0)
            + 0.2 * q_score
            + 0.2 * float(hint),
        )
        return score

    async def _orchestrate_complex_query(
        self, query: SovereignQuery, start_time: float
    ) -> SovereignResult:
        """Route complex queries through orchestrator for task decomposition.

        Standing on: Crew AI (2024) — role-based agent collaboration.
        The orchestrator decomposes the query into sub-tasks, routes each to
        a specialized agent, executes them (with real LLM or heuristic fallback),
        and synthesizes the results.
        """
        result = SovereignResult(query_id=query.id, user_id=query.user_id)

        try:
            from .orchestrator import TaskNode

            plan = await self._orchestrator.decomposer.decompose(  # type: ignore[union-attr]
                TaskNode(
                    title=query.text[:120],
                    description=query.text,
                )
            )
            for task in plan.subtasks:
                await self._orchestrator.execute_task(task)  # type: ignore[union-attr]

            # Collect all task outputs
            parts = []
            for task in plan.subtasks:
                task_result = self._orchestrator.task_results.get(task.id, {})  # type: ignore[union-attr]
                content = task_result.get("content", "")
                if content:
                    parts.append(content)

            combined = (
                "\n\n".join(parts)
                if parts
                else f"Orchestrated analysis of: {query.text}"
            )
            result.response = combined

            # Run through SNR + Constitutional stages
            optimized, snr_score, claim_tags = await self._optimize_snr(result.response)
            result.response = optimized
            result.snr_score = snr_score
            result.snr_ok = snr_score >= self.config.snr_threshold
            result.claim_tags = claim_tags

            ihsan_score, verdict = await self._validate_constitutionally(
                result.response, query.context, query, snr_score
            )
            result.ihsan_score = ihsan_score
            result.validated = query.require_validation
            result.validation_passed = ihsan_score >= self.config.ihsan_threshold

            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            result.success = True
            result.reasoning_used = True
            result.reasoning_depth = len(plan.subtasks)
            result.thoughts = [t.title for t in plan.subtasks]

            self._query_times.append(result.processing_time_ms)
            self.metrics.update_query_stats(True, result.processing_time_ms)

            # SPEARPOINT COCKPIT: Execute unified post-query pipeline
            if self._spearpoint is not None:
                if hasattr(self, "_snr_trace_slot"):
                    self._snr_trace_slot[0] = self._last_snr_trace
                sp_result = await self._spearpoint.execute(result, query)
                if hasattr(self, "_snr_trace_slot"):
                    self._last_snr_trace = self._snr_trace_slot[0]
                result.spearpoint = sp_result.to_dict()  # type: ignore[attr-defined]
            else:
                self._record_query_impact(result)
                self._emit_query_receipt(result, query)
                self._schedule_receipt_memory_feedback(result, query)
                self._encode_query_memory(result, query)
                self._commit_experience_episode(result, query)
                self._observe_judgment(result)

            return result

        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            self.logger.warning(
                f"Orchestrator path failed ({e}), falling back to direct pipeline"
            )
            return await self._process_query_direct(query, start_time)

    async def _process_query(
        self, query: SovereignQuery, start_time: float
    ) -> SovereignResult:
        """Internal query processing — routes to orchestrator or direct pipeline.

        Standing on: Besta (GoT, 2024) + Shannon (SNR) + Anthropic (Constitutional AI).
        Complexity ≥ 0.6 and orchestrator available → decompose via agent swarm.
        Otherwise → direct 5-stage pipeline (GoT → LLM → SNR → Guardian → Finalize).
        """
        complexity = self._estimate_complexity(query)
        if complexity >= 0.6 and self._orchestrator is not None:
            self.logger.info(
                f"Query complexity={complexity:.2f} — routing to orchestrator"
            )
            return await self._orchestrate_complex_query(query, start_time)

        return await self._process_query_direct(query, start_time)

    async def _process_query_direct(
        self, query: SovereignQuery, start_time: float
    ) -> SovereignResult:
        """Direct 5-stage query pipeline (bypasses orchestrator)."""
        result = SovereignResult(query_id=query.id, user_id=query.user_id)

        # α7 TIER 1: Instant pattern pre-check (< 50ms, before gate chain)
        if self._tiered_verification_enabled:
            try:
                from .tiered_verification import TierDecision, tier_1_precheck

                t1 = tier_1_precheck(
                    action_type="query",
                    content=query.text,
                    category=query.context.get("category", ""),
                )
                if t1.decision == TierDecision.BLOCK:
                    self.logger.warning(
                        "α7 Tier 1 BLOCKED query: %s (%.1fms)",
                        t1.reason,
                        t1.elapsed_ms,
                    )
                    result.success = False
                    result.response = f"Query blocked by safety pre-check: {t1.reason}"
                    result.validation_passed = False
                    result.processing_time_ms = (
                        time.perf_counter() - start_time
                    ) * 1000
                    self.metrics.update_query_stats(False, result.processing_time_ms)
                    return result
            except ImportError:
                pass  # Graceful degradation — skip Tier 1 if unavailable

        # PRE-FLIGHT: 6-Gate Chain (fail-closed)
        gate_rejection = await self._run_gate_chain_preflight(query, result)
        if gate_rejection is not None:
            gate_rejection.processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000
            self._query_times.append(gate_rejection.processing_time_ms)
            self.metrics.update_query_stats(False, gate_rejection.processing_time_ms)
            self._emit_query_receipt(gate_rejection, query)
            self._schedule_receipt_memory_feedback(gate_rejection, query)
            return gate_rejection

        # STAGE 0: Select compute tier
        compute_tier = await self._select_compute_tier(query)

        # STAGE 1: Execute reasoning (GoT)
        reasoning_path, confidence, thought_prompt, graph_hash = (
            await self._execute_reasoning_stage(query)
        )
        result.thoughts = reasoning_path
        result.reasoning_depth = len(reasoning_path)
        result.graph_hash = graph_hash

        # SPEARPOINT: Store graph artifact for retrieval (fire-and-forget)
        self._store_graph_artifact(query.id, graph_hash)

        # STAGE 1.5: Cognitive Fusion (MoE → HRM → RAG → NorthStar)
        # Enriches the LLM prompt with complexity-scaled RAG context
        # when the CognitiveFusionEngine is available. Falls through
        # transparently when absent — zero disruption to existing pipeline.
        fusion_result = None
        if self._cognitive_fusion is not None:
            fusion_result = self._run_cognitive_fusion(query, thought_prompt)
            if fusion_result is not None:
                thought_prompt = self._enrich_prompt_with_fusion(
                    thought_prompt, fusion_result
                )
                result.fusion_report = {  # type: ignore[attr-defined]
                    "complexity": fusion_result.routing.complexity_class,
                    "expert_tier": fusion_result.expert_tier,
                    "target_level": fusion_result.target_level,
                    "hrm_snr": round(fusion_result.compound_snr, 4),
                    "retrieval_count": len(fusion_result.retrieval),
                    "fusion_snr": round(fusion_result.snr_score, 4),
                    "fusion_ihsan": round(fusion_result.ihsan_score, 4),
                    "passes_gate": fusion_result.passes_gate,
                }

        # STAGE 2: Perform LLM inference (α9: measured by PerformanceAttestor)
        _llm_start = time.perf_counter()
        answer, model_used = await self._perform_llm_inference(
            thought_prompt, compute_tier, query
        )
        result.response = answer
        if self._performance_attestor is not None:
            _llm_ms = (time.perf_counter() - _llm_start) * 1000
            _attest = self._performance_attestor.record_measurement(
                "sovereign_runtime", "llm_inference", _llm_ms
            )
            if hasattr(_attest, "is_suspicious") and _attest.is_suspicious:
                self.logger.warning(
                    "α9 LLM inference anomaly: %s (%.1fms, %.1fσ)",
                    _attest.detail,
                    _attest.measured_time_ms,
                    _attest.deviation_sigma,
                )

        # TRUE SPEARPOINT: Detect template/stub output and degrade result
        is_real_inference = model_used not in ("NO_LLM", "stub", "template")
        if not is_real_inference:
            self.logger.info(
                f"SPEARPOINT: Pipeline running without LLM (model={model_used}). "
                f"Result will be tagged as degraded."
            )

        # Update reasoning metrics
        self.metrics.update_reasoning_stats(result.reasoning_depth)

        # STAGE 2.5: FATE Bridge — evidence auditing + SAT validation
        # Runs only when evidence_refs are available (e.g., from RAG/retrieval).
        # Fail-open: if FATE gate is unavailable, pipeline continues.
        try:
            from core.sovereign.fate_bridge import run_fate_bridge

            evidence_refs = query.context.get("evidence_refs", [])
            if evidence_refs and is_real_inference:
                fate_bridge_result = run_fate_bridge(
                    answer=result.response or "",
                    evidence_refs=evidence_refs,
                    confidence="high" if confidence > 0.7 else "medium",
                )
                if fate_bridge_result.enabled:
                    result.fate_verdict = fate_bridge_result.verdict
                    result.fate_evidence_valid = fate_bridge_result.evidence_valid
                    if not fate_bridge_result.passed:
                        self.logger.warning(
                            "FATE bridge BLOCKED: %s (ihsan=%.2f, evidence_valid=%s)",
                            fate_bridge_result.reason,
                            fate_bridge_result.ihsan_score,
                            fate_bridge_result.evidence_valid,
                        )
                        result.success = False
                        result.response = f"Response blocked by FATE gate: {fate_bridge_result.reason}"
                        result.validation_passed = False
                        result.processing_time_ms = (
                            time.perf_counter() - start_time
                        ) * 1000
                        self.metrics.update_query_stats(
                            False, result.processing_time_ms
                        )
                        self._emit_query_receipt(result, query)
                        return result
        except ImportError:
            pass  # Graceful degradation — FATE bridge not available

        # STAGE 3: Optimize SNR
        optimized_content, snr_score, claim_tags = await self._optimize_snr(
            result.response
        )
        result.response = optimized_content
        result.snr_score = snr_score
        result.snr_ok = snr_score >= self.config.snr_threshold
        result.claim_tags = claim_tags

        # STAGE 4: Constitutional validation
        ihsan_score, guardian_verdict = await self._validate_constitutionally(
            result.response, query.context, query, result.snr_score
        )
        result.ihsan_score = ihsan_score
        result.validated = query.require_validation
        result.validation_passed = ihsan_score >= self.config.ihsan_threshold

        # α7 TIER 3: Post-execution attestation (flags if quality drifted)
        if self._tiered_verification_enabled:
            try:
                from .tiered_verification import TierDecision, tier_3_attestation

                t3_ctx = {
                    "ihsan": ihsan_score,
                    "snr": snr_score,
                    "action_type": "query",
                    "risk_level": 0.1,
                }
                t3 = await tier_3_attestation(t3_ctx, execution_result=result)
                if t3.decision == TierDecision.FLAG:
                    self.logger.warning(
                        "α7 Tier 3 FLAGGED result: %s (%.1fms)",
                        t3.reason,
                        t3.elapsed_ms,
                    )
                    result.flagged_for_review = True  # type: ignore[attr-defined]
                    result.flag_reason = t3.reason  # type: ignore[attr-defined]
            except ImportError:
                pass  # Graceful degradation

        # STAGE 4.5: Build reasoning summary (Week 3 — transparent decision trace)
        # Standing on Giants: Besta (GoT graph), Shannon (SNR per node),
        # Al-Ghazali (auditable intention), Boyd (visible orient phase)
        reasoning_ms = (time.perf_counter() - start_time) * 1000
        got_nodes = []
        for i, thought in enumerate(reasoning_path):
            got_nodes.append(
                GoTNodeSnapshot(
                    node_id=f"got_{query.id}_{i}",
                    content=thought[:256],
                    score=snr_score if i == len(reasoning_path) - 1 else confidence,
                    depth=i,
                    is_conclusion=(i == len(reasoning_path) - 1),
                    parent_id=f"got_{query.id}_{i - 1}" if i > 0 else None,
                )
            )
        result.reasoning_summary = ReasoningSummary(
            got_nodes=got_nodes,
            agent_scores={},  # Populated by orchestrator path when agents are used
            alternatives_considered=max(1, len(reasoning_path) - 1),
            convergence_reason=(
                f"GoT depth {len(reasoning_path)}, "
                f"confidence {confidence:.2f}, "
                f"SNR {snr_score:.3f}, "
                f"Ihsān {ihsan_score:.3f}"
            ),
            total_reasoning_ms=reasoning_ms,
            confidence=confidence,
            guardian_verdicts=(
                {"constitutional": guardian_verdict} if guardian_verdict else {}
            ),
            model_used=model_used,
        )

        # STAGE 5: Finalize result
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.success = True
        result.reasoning_used = query.require_reasoning

        # TRUE SPEARPOINT: Tag model source for observability.
        # When no real LLM was used, the output is template-based.
        # Mark it clearly so consumers know the quality level.
        result.model_used = model_used
        if not is_real_inference:
            result.degraded = True  # type: ignore[attr-defined]
            result.degraded_reason = (  # type: ignore[attr-defined]
                f"No LLM backend available (model={model_used}). "
                "Response is template/GoT-derived, not LLM-grounded."
            )

        # Update timing metrics
        self._query_times.append(result.processing_time_ms)
        self.metrics.update_query_stats(True, result.processing_time_ms)

        # SPEARPOINT COCKPIT: Execute unified post-query pipeline
        if self._spearpoint is not None:
            # Sync SNR trace into the pipeline's shared slot
            if hasattr(self, "_snr_trace_slot"):
                self._snr_trace_slot[0] = self._last_snr_trace
            sp_result = await self._spearpoint.execute(result, query)
            # Sync trace back (cleared after receipt emission)
            if hasattr(self, "_snr_trace_slot"):
                self._last_snr_trace = self._snr_trace_slot[0]
            # Attach pipeline diagnostics to result metadata
            result.spearpoint = sp_result.to_dict()  # type: ignore[attr-defined]
        else:
            # Fallback: original fire-and-forget calls (pre-pipeline)
            self._record_query_impact(result)
            self._register_poi_contribution(result, query)
            self._emit_query_receipt(result, query)
            self._schedule_receipt_memory_feedback(result, query)
            self._encode_query_memory(result, query)
            self._commit_experience_episode(result, query)
            self._observe_judgment(result)

        return result

    async def _select_compute_tier(self, query: SovereignQuery) -> object | None:
        """STAGE 0: Treasury Mode to Compute Tier selection."""
        if not self._omega:
            return None

        mode = getattr(self._omega, "get_operational_mode", lambda: None)()
        if mode is None:
            return None
        return self._mode_to_tier(mode)

    def _run_cognitive_fusion(
        self, query: SovereignQuery, thought_prompt: str
    ) -> object | None:
        """STAGE 1.5: Cognitive Fusion — MoE → HRM → RAG → NorthStar.

        Runs the CognitiveFusionEngine synchronously (all subsystems are CPU-bound).
        Uses real embeddings from EmbeddingService when available, with quality
        gate validation. Falls back to zero vector only as last resort.

        Standing on: Vaswani (MoE) + Simon (hierarchy) + Shannon (SNR)
                   + Reimers (sentence-BERT) + Takens (NTU temporal)
        """
        try:
            # Step 1: Generate real embedding (Phase 32)
            embedding: list[float] | None = None
            if self._embedding_service is not None:
                try:
                    embedding = self._embedding_service.embed(query.text)  # type: ignore[union-attr]

                    # Step 1a: Quality gate
                    if self._embedding_gate is not None and embedding is not None:
                        gate_result = self._embedding_gate.validate(embedding)  # type: ignore[union-attr]
                        if not gate_result.passed:
                            self.logger.warning(
                                f"Embedding quality gate failed: {gate_result.reason}"
                            )
                            embedding = None
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    OSError,
                ) as e:
                    self.logger.debug(f"Embedding generation failed: {e}")

            # Fallback: zero vector (degraded — RAG retrieval will be empty)
            if embedding is None:
                embedding = [0.0] * 768

            # Step 2: Enrich context with NTU temporal state
            context = dict(query.context)
            if self._ntu_adapter is not None:
                try:
                    context = self._ntu_adapter.enrich_context(context)  # type: ignore[union-attr]
                except (
                    ImportError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as e:
                    self.logger.debug(f"NTU enrichment skipped: {e}")

            # Step 3: Run fusion pipeline
            return self._cognitive_fusion.process(  # type: ignore[union-attr]
                query=query.text,
                query_embedding=embedding,
                context=context,
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            self.logger.warning(f"Cognitive fusion skipped: {e}")
            return None

    @staticmethod
    def _enrich_prompt_with_fusion(thought_prompt: str, fusion_result: object) -> str:
        """Augment the GoT prompt with RAG context from cognitive fusion.

        Prepends retrieved context chunks (if any) so the LLM has grounded
        knowledge to reason over. Keeps the original GoT prompt intact as
        the primary instruction.
        """
        retrieval = getattr(fusion_result, "retrieval", [])
        if not retrieval:
            return thought_prompt

        # Build context block from retrieved chunks (max 5 for prompt budget)
        chunks = []
        for item in retrieval[:5]:
            if hasattr(item, "content"):
                chunks.append(str(item.content)[:500])
            elif isinstance(item, dict) and "content" in item:
                chunks.append(str(item["content"])[:500])
            elif isinstance(item, str):
                chunks.append(item[:500])

        if not chunks:
            return thought_prompt

        context_block = "\n---\n".join(chunks)
        return (
            f"[Retrieved Context ({len(chunks)} sources)]\n"
            f"{context_block}\n\n"
            f"[Query + Reasoning]\n{thought_prompt}"
        )

    async def _execute_reasoning_stage(
        self, query: SovereignQuery
    ) -> tuple[list[str], float, str, str | None]:
        """STAGE 1: Graph-of-Thoughts exploration.

        Returns (reasoning_path, confidence, thought_prompt, graph_hash).
        """
        thought_prompt: str = query.text
        reasoning_path: list[str] = []
        confidence: float = 0.75
        graph_hash: str | None = None

        if query.require_reasoning and self._graph_reasoner:
            reasoning_result = await self._graph_reasoner.reason(
                query=query.text,
                context=query.context,
                max_depth=self.config.max_reasoning_depth,
            )
            reasoning_path = reasoning_result.get("thoughts", [])
            confidence = reasoning_result.get("confidence", 0.0)
            graph_hash = reasoning_result.get("graph_hash")

            conclusion = reasoning_result.get("conclusion")
            if conclusion:
                thought_prompt = conclusion

        return reasoning_path, confidence, thought_prompt, graph_hash

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
        query.context.pop("_source_memory_ids", None)
        living_memory = getattr(self, "_living_memory", None)
        if living_memory:
            try:
                # Retrieve memories relevant to the query
                memories = await living_memory.retrieve(
                    query=query.text, top_k=5, min_score=0.15
                )
                if memories:
                    parts = []
                    source_ids: list[str] = []
                    for mem in memories:
                        label = mem.memory_type.value.upper()
                        # Truncate long memories to keep prompt manageable
                        content = mem.content
                        if len(content) > 800:
                            content = content[:800] + "..."
                        parts.append(f"[{label}] {content}")
                        mem_id = getattr(mem, "id", None)
                        if isinstance(mem_id, str) and mem_id:
                            source_ids.append(mem_id)
                    memory_context = "\n\n".join(parts)
                    if source_ids:
                        query.context["_source_memory_ids"] = source_ids
                    self.logger.debug(
                        f"RAG: retrieved {len(memories)} memories for query"
                    )
            except (RuntimeError, ValueError, TypeError, OSError) as e:
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
        self, thought_prompt: str, compute_tier: object | None, query: SovereignQuery
    ) -> tuple[str, str]:
        """STAGE 2: LLM inference via gateway with user context.

        TRUE SPEARPOINT: Fail-loud when no LLM is available.
        Returns (answer, model_used) where model_used is NEVER silently "stub".
        When gateway fails, the model_used is tagged "NO_LLM" so downstream
        stages can detect and act on it (instead of blindly validating fake output).
        """
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
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                ValueError,
            ) as e:
                self.logger.warning(f"Gateway inference failed: {e}")

        # FAIL-LOUD: Tag as NO_LLM so pipeline can reject/degrade gracefully
        self.logger.warning(
            "SPEARPOINT: No LLM backend available — returning template output tagged 'NO_LLM'"
        )
        return thought_prompt, "NO_LLM"

    async def _optimize_snr(self, content: str) -> tuple[str, float, dict[str, str]]:
        """STAGE 3: SNR optimization — dual engine (maximizer + authoritative scorer).

        The SNRMaximizer handles text optimization (noise removal, content cleaning).
        The SNREngine v1 computes the authoritative, auditable SNR score with trace.

        Standing on: Shannon (1948) — SNR as information quality.

        Returns (optimized_content, snr_score, claim_tags).
        """
        from core.integration.constants import UNIFIED_SNR_THRESHOLD

        optimized_content = content
        snr_score = UNIFIED_SNR_THRESHOLD
        claim_tags: dict[str, str] = {}

        # Phase 1: SNRMaximizer — text optimization (noise removal)
        if self._snr_optimizer:
            result_or_coro = self._snr_optimizer.optimize(content)
            snr_result = (
                await result_or_coro
                if inspect.isawaitable(result_or_coro)
                else result_or_coro
            )
            snr_score = snr_result.get("snr_score", UNIFIED_SNR_THRESHOLD)
            claim_tags = snr_result.get("claim_tags", {})
            # RFC-04 FIX: Actually use the optimized content from SNR pipeline
            optimized_content = snr_result.get("optimized") or content
            # Track SNR improvement
            if optimized_content != content:
                original_len = len(content)
                improvement = (original_len - len(optimized_content)) / max(
                    1, original_len
                )
                self.metrics.update_snr_stats(improvement)

        # Phase 2: SNREngine v1 — authoritative scorer with audit trace
        # Produces receipt-compatible output + SNRTrace artifact.
        try:
            from core.proof_engine.snr import SNREngine, SNRInput

            engine = SNREngine()
            inputs = SNRInput(
                source_trust_score=snr_score,
                ihsan_score=self.metrics.current_ihsan_score or 0.95,
                z3_satisfiable=True,
            )
            authoritative = engine.snr_score(inputs)

            # Use the authoritative score; merge claim tags
            snr_score = authoritative["score"]
            for k, v in authoritative.get("claim_tags", {}).items():
                claim_tags.setdefault(k, v)

            # Store the last SNR trace for receipt embedding
            self._last_snr_trace = authoritative
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            self.logger.debug(f"SNREngine v1 scoring skipped: {e}")

        self.metrics.current_snr_score = snr_score
        return optimized_content, snr_score, claim_tags

    async def _validate_constitutionally(
        self,
        content: str,
        context: dict[str, Any],
        query: SovereignQuery,
        snr_score: float,
    ) -> tuple[float, str]:
        """STAGE 4: Constitutional validation — IhsanGate + Omega + Guardian.

        Standing on: Anthropic (Constitutional AI, 2022), Islamic ethics (Ihsan).

        Evaluation order:
        1. IhsanGate v1 (authoritative, fail-closed) — proof_engine gate
        2. Omega engine (if available) — deep ihsan evaluation
        3. Guardian Council (if requested) — multi-perspective validation

        The final score is the authoritative IhsanGate result, optionally
        enriched by Omega/Guardian signals.
        """
        ihsan_score = snr_score
        guardian_verdict = "SKIPPED"

        # Phase 1: IhsanGate v1 — authoritative fail-closed gate
        ihsan_gate_result = None
        try:
            from core.proof_engine.ihsan_gate import IhsanComponents, IhsanGate

            gate = IhsanGate(threshold=self.config.ihsan_threshold)
            components: Any = None

            # Primary path: compute components from actual content/query/context.
            try:
                from core.proof_engine.ihsan_computer import IhsanComputer

                components = IhsanComputer().compute(
                    content=content,
                    snr_score=snr_score,
                    query_text=query.text,
                    context=context,
                )
            except (
                ImportError,
                RuntimeError,
                ValueError,
                TypeError,
            ) as ihsan_computer_err:
                self.logger.debug(
                    "IhsanComputer unavailable, using legacy component projection: %s",
                    ihsan_computer_err,
                )

            # Fallback path: preserve historical behavior if IhsanComputer unavailable.
            if components is None:
                components = IhsanComponents(
                    correctness=min(snr_score * 1.02, 1.0),
                    safety=0.95,
                    efficiency=min(snr_score, 1.0),
                    user_benefit=min(snr_score * 0.98, 1.0),
                )

            ihsan_gate_result = gate.ihsan_score(components)
            ihsan_score = ihsan_gate_result["score"]
            guardian_verdict = ihsan_gate_result["decision"]

            # Optional thermodynamic gate (Lyapunov + thermal Ihsan).
            # Off by default to preserve current runtime contract.
            use_thermal_gate = os.getenv(
                "BIZRA_ENABLE_THERMODYNAMIC_GATE", "0"
            ).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if use_thermal_gate:
                try:
                    from core.proof_engine.thermodynamic_gate import (
                        ThermodynamicIhsanGate,
                    )

                    thermal_gate = ThermodynamicIhsanGate(
                        threshold=self.config.ihsan_threshold
                    )
                    thermal_step = context.get("thermal_step", 0)
                    prev_energy = context.get("previous_total_energy")
                    thermal_decision = thermal_gate.evaluate(
                        content=content,
                        snr_score=snr_score,
                        query_text=query.text,
                        context=context,
                        previous_energy=(
                            float(prev_energy) if prev_energy is not None else None
                        ),
                        step=thermal_step,
                    )
                    context["thermodynamic_ihsan"] = thermal_decision.to_dict()
                    ihsan_score = min(
                        ihsan_score, thermal_decision.profile.composite_ihsan
                    )
                    if not thermal_decision.approved:
                        guardian_verdict = "THERMODYNAMIC_REJECTED"
                        ihsan_gate_result["decision"] = "REJECTED"
                        ihsan_gate_result["passed"] = False
                        reason_codes = ihsan_gate_result.setdefault("reason_codes", [])
                        if "THERMODYNAMIC_GATE_REJECTED" not in reason_codes:
                            reason_codes.append("THERMODYNAMIC_GATE_REJECTED")
                except (
                    ImportError,
                    RuntimeError,
                    ValueError,
                    TypeError,
                ) as thermal_gate_err:
                    self.logger.debug(
                        "Thermodynamic gate unavailable, continuing without it: %s",
                        thermal_gate_err,
                    )

            # Record in IHSAN_FLOOR watchdog (MCG governance invariant)
            if self._ihsan_watchdog is not None:
                healthy = self._ihsan_watchdog.record(ihsan_score)
                if not healthy:
                    self.logger.warning(
                        "IHSAN_FLOOR BREACH: System entering DEGRADED mode — "
                        f"{self._ihsan_watchdog.consecutive_failures} consecutive failures"
                    )
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            self.logger.debug(f"IhsanGate v1 evaluation skipped: {e}")

        # Phase 2: Omega engine — deep ihsan evaluation (enriches gate result)
        if self._omega:
            try:
                ihsan_vector = self._extract_ihsan_from_response(content, context)
                evaluate_ihsan = getattr(self._omega, "evaluate_ihsan", None)
                if evaluate_ihsan is not None and ihsan_vector is not None:
                    result = evaluate_ihsan(ihsan_vector)
                    omega_score = 0.0
                    if isinstance(result, tuple) and len(result) >= 2:
                        omega_score = result[0]
                    else:
                        omega_score = float(result) if result else snr_score
                    # Blend: IhsanGate is authoritative (70%), Omega enriches (30%)
                    ihsan_score = 0.7 * ihsan_score + 0.3 * omega_score
                    guardian_verdict = "IHSAN_GATE+OMEGA"
            except (RuntimeError, ValueError, TypeError) as e:
                self.logger.warning(f"Omega Ihsan evaluation failed: {e}")

        # Phase 3: Guardian Council — multi-perspective validation
        if query.require_validation and self._guardian_council:
            validation = await self._guardian_council.validate(
                content=content,
                context=context,
            )
            guardian_score = validation.get("confidence", 0.0)
            is_valid = validation.get("is_valid", False)

            # Guardian safety signal enriches the ihsan score
            # but IhsanGate remains the authoritative decision maker
            ihsan_score = 0.6 * ihsan_score + 0.4 * guardian_score
            guardian_verdict = (
                f"IHSAN_GATE+GUARDIAN({'VALID' if is_valid else 'INVALID'})"
            )

            self.metrics.validations += 1
            self.metrics.update_validation_stats(is_valid)

        self.metrics.current_ihsan_score = ihsan_score
        return ihsan_score, guardian_verdict

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _cache_key(self, query: SovereignQuery) -> str:
        """Generate cache key for a query (SEC-001: BLAKE3, module-level import)."""
        content = f"{query.text}:{query.require_reasoning}"
        return _hex_digest(content.encode())[:16]

    def _update_cache(self, key: str, result: SovereignResult) -> None:
        """Update cache with new result (O(1) LRU via OrderedDict)."""
        # Move to end if exists (LRU refresh)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = result
            return
        # Evict oldest entries if at capacity — O(1) per eviction via popitem(last=False)
        while len(self._cache) >= self.config.max_cache_entries:
            self._cache.popitem(last=False)
        self._cache[key] = result

    def _mode_to_tier(self, mode: object) -> object | None:
        """Map TreasuryMode to ComputeTier."""
        try:
            from core.inference.gateway import ComputeTier  # type: ignore[attr-defined]

            from .omega_engine import TreasuryMode

            mapping = {
                TreasuryMode.ETHICAL: ComputeTier.LOCAL,
                TreasuryMode.HIBERNATION: ComputeTier.EDGE,
                TreasuryMode.EMERGENCY: ComputeTier.EDGE,
            }
            if isinstance(mode, TreasuryMode):
                return mapping.get(mode, ComputeTier.LOCAL)
            return None
        except ImportError:
            return None

    def _extract_ihsan_from_response(
        self, content: str, context: dict[str, Any]
    ) -> object | None:
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

    async def reason(self, question: str, depth: int = 3) -> list[str]:
        """Get reasoning path for a question."""
        result = await self.query(question, max_depth=depth)
        return result.thoughts

    def _collect_pat_sat_receipt_chain_status(self) -> dict[str, Any]:
        """Collect PAT↔SAT negotiation receipt-chain telemetry from evidence ledger."""
        status: dict[str, Any] = {
            "available": False,
            "total_entries": 0,
            "total_negotiation_receipts": 0,
            "chain_valid": None,
            "chain_error_count": 0,
            "chain_error": None,
            "verified_end_to_end": False,
            "latest_receipt_id": None,
            "latest_sequence": None,
            "latest_entry_hash": None,
            "latest_payload_digest": None,
            "latest_signed": False,
            "latest_decision": None,
            "latest_timestamp": None,
            "ledger_last_hash": None,
        }

        ledger = self._evidence_ledger
        if ledger is None:
            return status

        status["available"] = True
        sequence = getattr(ledger, "sequence", 0)
        status["total_entries"] = sequence if isinstance(sequence, int) else 0

        last_hash = getattr(ledger, "last_hash", None)
        if isinstance(last_hash, str) and last_hash:
            status["ledger_last_hash"] = last_hash

        verify_chain = getattr(ledger, "verify_chain", None)
        if callable(verify_chain) and status["total_entries"] > 0:
            try:
                chain_valid, chain_errors = verify_chain()
                status["chain_valid"] = bool(chain_valid)
                if isinstance(chain_errors, list):
                    status["chain_error_count"] = len(chain_errors)
                else:
                    status["chain_error_count"] = 0
            except (RuntimeError, AttributeError, TypeError) as exc:
                status["chain_valid"] = False
                status["chain_error"] = str(exc)

        entries_fn = getattr(ledger, "entries", None)
        if not callable(entries_fn) or status["total_entries"] <= 0:
            return status

        try:
            entries = entries_fn()
        except (RuntimeError, AttributeError, TypeError) as exc:
            status["chain_error"] = str(exc)
            return status

        if not isinstance(entries, list):
            return status

        latest_entry = None
        receipt_count = 0
        for entry in entries:
            receipt = getattr(entry, "receipt", None)
            if not isinstance(receipt, dict):
                continue
            origin = receipt.get("origin", {})
            if not isinstance(origin, dict) or origin.get("channel") != "pat_sat":
                continue
            receipt_count += 1
            latest_entry = entry

        status["total_negotiation_receipts"] = receipt_count
        if latest_entry is None:
            status["verified_end_to_end"] = bool(
                status["chain_valid"] is True and receipt_count > 0
            )
            return status

        latest_receipt = (
            latest_entry.receipt if isinstance(latest_entry.receipt, dict) else {}
        )
        status["latest_receipt_id"] = latest_receipt.get("receipt_id")
        latest_sequence = getattr(latest_entry, "sequence", None)
        status["latest_sequence"] = (
            latest_sequence if isinstance(latest_sequence, int) else None
        )
        latest_hash = getattr(latest_entry, "entry_hash", None)
        status["latest_entry_hash"] = (
            latest_hash if isinstance(latest_hash, str) else None
        )
        status["latest_timestamp"] = getattr(latest_entry, "timestamp", None)
        status["latest_decision"] = latest_receipt.get("decision")

        outputs = latest_receipt.get("outputs", {})
        if isinstance(outputs, dict):
            status["latest_payload_digest"] = outputs.get("payload_digest")

        signature = latest_receipt.get("signature", {})
        status["latest_signed"] = (
            isinstance(signature, dict)
            and isinstance(signature.get("value"), str)
            and bool(signature["value"])
        )
        status["verified_end_to_end"] = bool(
            status["chain_valid"] is True
            and receipt_count > 0
            and status["latest_signed"]
        )
        return status

    # -------------------------------------------------------------------------
    # STATUS & METRICS
    # -------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Get comprehensive runtime status."""
        loop_status = (
            self._autonomous_loop.status()
            if self._autonomous_loop
            else {"running": False}
        )

        omega_status: dict[str, Any] = {"version": "2.2.3"}
        if self._omega:
            try:
                _get_status = getattr(self._omega, "get_status", None)
                if _get_status is not None:
                    omega_status.update(_get_status() or {})
            except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError):
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
            omega_status.setdefault("gateway", {"connected": False})  # type: ignore[arg-type]

        identity_info: dict[str, Any] = {
            "node_id": self.config.node_id,
            "version": _ELITE_VERSION,
            "origin": dict(self._origin_snapshot),
            "identity_mode": self._identity_mode,
            "genesis_backed": self._genesis_backed_identity,
        }
        if self._signer_public_key_prefix:
            identity_info["signer_public_key_prefix"] = (
                self._signer_public_key_prefix + "..."
            )
        if self._node_signer and hasattr(self._node_signer, "public_key_hex"):
            identity_info["signer_public_key"] = (
                self._node_signer.public_key_hex[:16] + "..."
            )
        if self._genesis:
            identity_info["node_name"] = self._genesis.node_name
            identity_info["location"] = self._genesis.identity.location
            identity_info["public_key"] = self._genesis.identity.public_key[:16] + "..."
            identity_info["pat_agents"] = len(self._genesis.pat_team)
            identity_info["sat_agents"] = len(self._genesis.sat_team)
            identity_info["sat_mode"] = (
                "full49" if len(self._genesis.sat_team) >= 49 else "mini5"
            )
            identity_info["genesis_hash"] = (
                self._genesis.genesis_hash.hex()[:16] + "..."
                if self._genesis.genesis_hash
                else "none"
            )
        else:
            identity_info["sat_mode"] = self.config.sat_mode

        memory_status = (
            self._memory_coordinator.stats()
            if self._memory_coordinator
            else {"running": False}
        )

        autopoiesis_status: dict[str, Any] = {
            "enabled": self.config.enable_autopoiesis,
            "wired": self._autopoietic_loop is not None,
            "task_running": self._task_is_running(self._autopoiesis_task),
            "learning_task_running": self._task_is_running(
                self._autopoiesis_learning_task
            ),
            "learning_source": self._autopoiesis_learning_source,
        }
        if self._autopoietic_loop is not None and hasattr(
            self._autopoietic_loop, "get_status"
        ):
            try:
                autopoiesis_status["loop"] = self._autopoietic_loop.get_status()
            except (RuntimeError, ValueError, TypeError, OSError):
                self.logger.debug(
                    "Failed to collect autopoiesis loop status", exc_info=True
                )
        if self._learning_loop is not None and hasattr(
            self._learning_loop, "get_status"
        ):
            try:
                autopoiesis_status["learning_loop"] = self._learning_loop.get_status()
            except (RuntimeError, ValueError, TypeError, OSError):
                self.logger.debug(
                    "Failed to collect autopoiesis learning status", exc_info=True
                )

        # Impact / sovereignty progression
        sovereignty_info: dict[str, Any] = {"tracking": False}
        if self._impact_tracker:
            try:
                sovereignty_info = {
                    "tracking": True,
                    "score": self._impact_tracker.sovereignty_score,
                    "tier": self._impact_tracker.sovereignty_tier.value,
                    "total_bloom": self._impact_tracker.total_bloom,
                    "achievements": len(self._impact_tracker.achievements),
                }
            except (RuntimeError, ValueError, TypeError, OSError):
                self.logger.debug("Failed to collect sovereignty info", exc_info=True)

        canonical_loop_status: dict[str, Any] = {
            "truth_label": "CANONICAL_LOOP: UNWIRED",
            "mission_state_authority": self._organism is not None,
            "authority_path": (
                "runtime->organism->node0"
                if self._organism is not None and self._node0 is not None
                else ""
            ),
            "node0": None,
        }
        if self._node0 is not None:
            try:
                node0_health = self._node0.health()
                canonical_loop_status["node0"] = node0_health.get(
                    "canonical_loop_status"
                )
                if canonical_loop_status["node0"] is not None:
                    canonical_loop_status["truth_label"] = str(
                        canonical_loop_status["node0"].get(
                            "truth_label",
                            "CANONICAL_LOOP: WIRED",
                        )
                    )
            except (RuntimeError, ValueError, TypeError, OSError):
                self.logger.debug(
                    "Failed to collect canonical loop status from Node0",
                    exc_info=True,
                )

        return {
            "identity": identity_info,
            "state": {
                "initialized": self._initialized,
                "running": self._running,
                "mode": self.config.mode.name,
                "sat_mode": self.config.sat_mode,
                "strict_gate_passed": self._strict_gate_passed,
                "mission_authority": (
                    "organism" if self._organism is not None else "legacy"
                ),
                "fate_mode": self._fate_mode,
                "event_bus_running": self._task_is_running(self._event_bus_task),
            },
            "canonical": {
                "enabled": self._canonical_mode,
                "mission_authority": (
                    "organism" if self._organism is not None else "legacy"
                ),
                "authority_path": (
                    "runtime->organism->node0"
                    if self._organism is not None and self._node0 is not None
                    else ""
                ),
                "runtime_owned_organism": self._organism is not None,
                "runtime_owned_node0": self._node0 is not None,
                "identity_mode": self._identity_mode,
                "signer_public_key_prefix": self._signer_public_key_prefix,
                "fate_mode": self._fate_mode,
                "loop": canonical_loop_status,
            },
            "health": {
                "status": self._health_status().value,
                "score": self._calculate_health(),
                "ihsan_watchdog": (
                    self._ihsan_watchdog.status() if self._ihsan_watchdog else None
                ),
                "strict_gate": {
                    "enabled": self.config.strict_stub_budget,
                    "passed": self._strict_gate_passed,
                    "reason_codes": list(self._strict_gate_reason_codes),
                    "stub_components": list(self._stub_components),
                    "stub_budget_max": self.config.stub_budget_max,
                },
            },
            "autonomous": loop_status,
            "autopoiesis": autopoiesis_status,
            "omega_point": omega_status,
            "memory": memory_status,
            "sovereignty": sovereignty_info,
            "pat_sat": {
                "negotiation_receipt_chain": self._collect_pat_sat_receipt_chain_status()
            },
            "equalizer": {
                "active": self._equalizer_agent is not None,
                "mode": (
                    self._equalizer_agent.detect_mode().value
                    if self._equalizer_agent and self._equalizer_agent.history
                    else "uninitialized"
                ),
                "observations": (
                    len(self._equalizer_agent.history) if self._equalizer_agent else 0
                ),
            },
            "unified_model_router": {
                "active": self._unified_model_router is not None,
            },
            "metrics": self.metrics.to_dict(),
        }

    def _health_status(self) -> HealthStatus:
        """Determine health status from metrics."""
        # IHSAN_FLOOR invariant: if watchdog is degraded, force DEGRADED
        if self._ihsan_watchdog is not None and self._ihsan_watchdog.is_degraded:
            return HealthStatus.DEGRADED

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

            state: dict[str, Any] = {
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

        except (OSError, RuntimeError, ValueError) as e:
            self.logger.warning(f"Checkpoint failed: {e}")


__all__ = [
    "SovereignRuntime",
]
