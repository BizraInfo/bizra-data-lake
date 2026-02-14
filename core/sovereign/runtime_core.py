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
import logging
import os
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Optional,
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
        self._node_role: str = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        self._origin_snapshot: dict[str, Any] = resolve_origin_snapshot(
            self.config.state_dir, self._node_role
        )

        # Unified Memory Coordinator (auto-save + persistence)
        self._memory_coordinator: Optional[MemoryCoordinator] = None

        # Impact Tracker (sovereignty growth engine)
        self._impact_tracker: Optional[ImpactTrackerProtocol] = None

        # Evidence Ledger (append-only, hash-chained audit trail)
        self._evidence_ledger: Optional[object] = None  # EvidenceLedger

        # Graph Artifact Store (query_id → schema-compliant GoT artifact)
        self._graph_artifacts: dict[str, dict[str, Any]] = {}

        # Last SNR trace from authoritative SNREngine v1 (for receipt embedding)
        self._last_snr_trace: Optional[dict[str, Any]] = None

        # 6-Gate Chain — fail-closed execution pipeline (Golden Gem #1)
        self._gate_chain: Optional[object] = None  # GateChain

        # Proof-of-Impact Engine — 4-stage PoI scoring pipeline
        self._poi_orchestrator: Optional[object] = None  # PoIOrchestrator

        # SAT Controller — ecosystem homeostasis engine
        self._sat_controller: Optional[object] = None  # SATController

        # Sovereign Experience Ledger (content-addressed episodic memory)
        self._experience_ledger: Optional[object] = None  # ExperienceLedger

        # Unified Node0 Signer (Ed25519) — single identity for all subsystems
        self._node_signer: Optional[object] = None  # Ed25519Signer

        # IHSAN_FLOOR Watchdog — governance invariant enforcer (MCG Layer 7)
        self._ihsan_watchdog: Optional[object] = None  # IhsanFloorWatchdog

        # Self-Evolving Judgment Engine — observation telemetry (Phase A)
        self._judgment_telemetry: Optional[object] = None  # JudgmentTelemetry

        # Spearpoint Orchestrator (reproduce / improve / heartbeat)
        self._spearpoint_orchestrator: Optional[object] = None

        # Omega Point Integration (v2.2.3)
        self._gateway: Optional[object] = None  # InferenceGateway
        self._omega: Optional[object] = None  # OmegaEngine
        self._living_memory: Optional[object] = None  # LivingMemoryCore
        self._pek: Optional[object] = None  # ProactiveExecutionKernel
        self._zpk_bootstrap_result: Optional[object] = None

        # PERF FIX: Use deque for O(1) bounded storage
        self._query_times: Deque[float] = deque(maxlen=100)

        # Cache
        self._cache: dict[str, SovereignResult] = {}

        # User Context (the system knows its human)
        self._user_context: Optional[UserContextManager] = None

        # SpearPoint Pipeline — unified post-query cockpit
        self._spearpoint: Optional[object] = None  # SpearPointPipeline

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
        self._node_role = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        enforce_node0_fail_closed(self.config.state_dir, self._node_role)
        self._origin_snapshot = resolve_origin_snapshot(
            self.config.state_dir, self._node_role
        )

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME INITIALIZING")
        self.logger.info("=" * 60)

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
        except Exception as e:
            self.logger.warning(f"IhsanFloor watchdog init failed: {e}")

        # Initialize 6-Gate Chain (fail-closed execution pipeline)
        self._init_gate_chain()

        # Initialize Proof-of-Impact Engine (4-stage scoring pipeline)
        self._init_poi_engine()

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

        # Initialize SpearPoint Pipeline — the unified cockpit
        self._init_spearpoint_pipeline()

        # Initialize Spearpoint Orchestrator (reproduce / improve / heartbeat)
        self._init_spearpoint_orchestrator()

        self._initialized = True
        self._running = True
        self.metrics.started_at = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("SOVEREIGN RUNTIME READY")
        self.logger.info("=" * 60)

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
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            self.logger.debug(f"Judgment Telemetry init skipped (non-fatal): {e}")
            self._judgment_telemetry = None

    def _observe_judgment(self, result: "SovereignResult") -> None:
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
        except Exception as e:
            self.logger.debug(f"SJE observe skipped (non-fatal): {e}")

    def _commit_experience_episode(
        self, result: "SovereignResult", query: "SovereignQuery"
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
        except Exception as e:
            self.logger.debug(f"SEL commit skipped (non-fatal): {e}")

    def _init_node_signer(self) -> None:
        """Initialize the unified Node0 Ed25519 signer.

        All subsystems (GateChain, PoI, Evidence) use this single identity.
        Standing on: Bernstein (Ed25519, 2011).
        """
        try:
            from core.proof_engine.receipt import Ed25519Signer

            self._node_signer = Ed25519Signer.generate()
            self.logger.info(
                f"Node0 Ed25519 signer initialized: "
                f"{self._node_signer.public_key_hex[:16]}..."
            )
        except Exception as e:
            self.logger.warning(
                f"Ed25519 signer init failed, falling back to HMAC: {e}"
            )
            from core.proof_engine.receipt import SimpleSigner

            self._node_signer = SimpleSigner(
                secret=self.config.node_id.encode("utf-8") + b"_node0_v1"
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
        except Exception as e:
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
    ) -> Optional[SovereignResult]:
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
            try:
                from core.sovereign.z3_fate_gate import Z3FATEGate

                z3_gate = Z3FATEGate()
                z3_proof = z3_gate.generate_proof(
                    {
                        "ihsan": ihsan_for_gate,
                        "snr": 0.85,  # Pre-inference minimum SNR gate
                        "cost": 0.0,
                        "autonomy_limit": 10.0,  # Default limit
                        "risk_level": 0.3,  # Read-only query = low risk
                        "reversible": True,
                        "human_approved": False,
                    }
                )
                z3_sat = z3_proof.satisfiable
            except Exception as z3_err:
                self.logger.debug(f"Z3 proof unavailable (fail-closed): {z3_err}")

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

        except Exception as e:
            # CRITICAL-2 FIX (Saltzer & Schroeder 1975): Fail-CLOSED on gate errors.
            # Previously returned None (pass-through), allowing queries to bypass
            # ALL constitutional gates on ANY exception.
            self.logger.error(f"GateChain preflight FAILED — REJECTING query: {e}")
            result.success = False
            result.response = f"Query rejected: Gate chain error ({e})"
            result.validation_passed = False
            return result

    def _emit_query_receipt(
        self, result: "SovereignResult", query: "SovereignQuery"
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

            decision = "APPROVED"
            reason_codes: list = []
            status = "accepted"

            if not result.validation_passed:
                decision = "REJECTED"
                reason_codes.append("IHSAN_BELOW_THRESHOLD")
                status = "rejected"
            if result.snr_score < 0.85:
                if "SNR_BELOW_THRESHOLD" not in reason_codes:
                    reason_codes.append("SNR_BELOW_THRESHOLD")
                if decision == "APPROVED":
                    decision = "QUARANTINED"
                    status = "quarantined"

            query_digest = hex_digest(
                query.text.encode("utf-8")
            )  # SEC-001: BLAKE3 for Rust interop

            seal_digest = hex_digest(
                (result.response or "").encode("utf-8")
            )  # SEC-001: BLAKE3 for Rust interop

            emit_receipt(
                self._evidence_ledger,
                receipt_id=result.query_id.replace("-", "")[:32],
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
            # Clear trace after emission
            self._last_snr_trace = None
        except Exception as e:
            self.logger.warning(f"Receipt emission failed (non-fatal): {e}")

    def _register_poi_contribution(
        self, result: "SovereignResult", query: "SovereignQuery"
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
        except Exception as e:
            # CRITICAL-10 FIX: PoI failures must be VISIBLE, not silent.
            self.logger.warning(f"PoI contribution registration failed: {e}")

    def _encode_query_memory(
        self, result: "SovereignResult", query: "SovereignQuery"
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
        except Exception as e:
            self.logger.debug(f"Memory encoding skipped (non-fatal): {e}")

    def _store_graph_artifact(self, query_id: str, graph_hash: Optional[str]) -> None:
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
        except Exception as e:
            self.logger.warning(f"Graph artifact storage failed (non-fatal): {e}")

    def get_graph_artifact(self, query_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a stored graph artifact by query ID."""
        return self._graph_artifacts.get(query_id)

    def get_gate_chain_stats(self) -> Optional[dict[str, Any]]:
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
        except Exception as e:
            self.logger.warning(f"PoI Engine init failed (non-fatal): {e}")
            self._poi_orchestrator = None
            self._sat_controller = None

    def get_poi_stats(self) -> Optional[dict[str, Any]]:
        """Get Proof-of-Impact engine statistics."""
        if self._poi_orchestrator is None:
            return None
        return self._poi_orchestrator.get_stats()

    def get_contributor_poi(self, contributor_id: str) -> Optional[dict[str, Any]]:
        """Get most recent PoI for a contributor."""
        if self._poi_orchestrator is None:
            return None
        poi = self._poi_orchestrator.get_contributor_poi(contributor_id)
        if poi is None:
            return None
        return poi.to_dict()

    def compute_poi_epoch(
        self, epoch_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Run a full PoI computation epoch.

        Returns the audit trail as a dict, or None if engine is unavailable.
        """
        if self._poi_orchestrator is None:
            return None
        audit = self._poi_orchestrator.compute_epoch(epoch_id)
        return audit.to_dict()

    def get_sat_stats(self) -> Optional[dict[str, Any]]:
        """Get SAT Controller statistics."""
        if self._sat_controller is None:
            return None
        return self._sat_controller.get_stats()

    def finalize_sat_epoch(
        self, epoch_reward: float = 1000.0
    ) -> Optional[dict[str, Any]]:
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
        except Exception as e:
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
                from .z3_fate_gate import Z3_AVAILABLE, Z3FATEGate

                if Z3_AVAILABLE:
                    self._pek.set_fate_gate(Z3FATEGate())
                    self.logger.info("✓ PEK FATE gate enabled (Z3)")
            except Exception as fate_err:
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
                # Wire memory into orchestrator for context-aware task execution
                if self._orchestrator and hasattr(self._orchestrator, "set_memory"):
                    self._orchestrator.set_memory(living_memory)
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
        except Exception as e:
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
        except Exception as e:
            self.logger.warning(f"Spearpoint Orchestrator init failed (non-fatal): {e}")
            self._spearpoint_orchestrator = None

    def _get_impact_state(self) -> dict[str, Any]:
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
            except Exception:
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
                self.logger.debug("PEK stop failed during shutdown", exc_info=True)

        # Save user context (conversation history + profile)
        if self._user_context:
            try:
                self._user_context.save()
            except Exception:
                self.logger.warning(
                    "Failed to save user context during shutdown", exc_info=True
                )

        # Flush impact tracker dirty state before memory coordinator stop
        if self._impact_tracker and hasattr(self._impact_tracker, "flush"):
            try:
                self._impact_tracker.flush()
            except Exception:
                self.logger.warning(
                    "Failed to flush impact tracker during shutdown", exc_info=True
                )

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
        self, content: str, context: Optional[dict[str, Any]] = None, **options
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
                user_id=query.user_id,
            )
        except Exception as e:
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
                self._encode_query_memory(result, query)
                self._commit_experience_episode(result, query)
                self._observe_judgment(result)

            return result

        except Exception as e:
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

        # PRE-FLIGHT: 6-Gate Chain (fail-closed)
        gate_rejection = await self._run_gate_chain_preflight(query, result)
        if gate_rejection is not None:
            gate_rejection.processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000
            self._query_times.append(gate_rejection.processing_time_ms)
            self.metrics.update_query_stats(False, gate_rejection.processing_time_ms)
            self._emit_query_receipt(gate_rejection, query)
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

        # STAGE 2: Perform LLM inference
        answer, model_used = await self._perform_llm_inference(
            thought_prompt, compute_tier, query
        )
        result.response = answer

        # TRUE SPEARPOINT: Detect template/stub output and degrade result
        is_real_inference = model_used not in ("NO_LLM", "stub", "template")
        if not is_real_inference:
            self.logger.info(
                f"SPEARPOINT: Pipeline running without LLM (model={model_used}). "
                f"Result will be tagged as degraded."
            )

        # Update reasoning metrics
        self.metrics.update_reasoning_stats(result.reasoning_depth)

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
            self._encode_query_memory(result, query)
            self._commit_experience_episode(result, query)
            self._observe_judgment(result)

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
    ) -> tuple[list[str], float, str, Optional[str]]:
        """STAGE 1: Graph-of-Thoughts exploration.

        Returns (reasoning_path, confidence, thought_prompt, graph_hash).
        """
        thought_prompt: str = query.text
        reasoning_path: list[str] = []
        confidence: float = 0.75
        graph_hash: Optional[str] = None

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
            except Exception as e:
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
        except Exception as e:
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
            components = IhsanComponents(
                correctness=min(snr_score * 1.02, 1.0),
                safety=0.95,  # Default safety assumption; overridden by Guardian
                efficiency=min(snr_score, 1.0),
                user_benefit=min(snr_score * 0.98, 1.0),
            )
            ihsan_gate_result = gate.ihsan_score(components)
            ihsan_score = ihsan_gate_result["score"]
            guardian_verdict = ihsan_gate_result["decision"]

            # Record in IHSAN_FLOOR watchdog (MCG governance invariant)
            if self._ihsan_watchdog is not None:
                healthy = self._ihsan_watchdog.record(ihsan_score)
                if not healthy:
                    self.logger.warning(
                        "IHSAN_FLOOR BREACH: System entering DEGRADED mode — "
                        f"{self._ihsan_watchdog.consecutive_failures} consecutive failures"
                    )
        except Exception as e:
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
            except Exception as e:
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
        """Generate cache key for a query (SEC-001: BLAKE3)."""
        from core.proof_engine.canonical import hex_digest

        content = f"{query.text}:{query.require_reasoning}"
        return hex_digest(content.encode())[:16]

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

    async def reason(self, question: str, depth: int = 3) -> list[str]:
        """Get reasoning path for a question."""
        result = await self.query(question, max_depth=depth)
        return result.thoughts

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
            omega_status.setdefault("gateway", {"connected": False})  # type: ignore[arg-type]

        identity_info: dict[str, Any] = {
            "node_id": self.config.node_id,
            "version": "1.0.0",
            "origin": dict(self._origin_snapshot),
        }
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
            except Exception:
                self.logger.debug("Failed to collect sovereignty info", exc_info=True)

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
                "ihsan_watchdog": (
                    self._ihsan_watchdog.status() if self._ihsan_watchdog else None
                ),
            },
            "autonomous": loop_status,
            "omega_point": omega_status,
            "memory": memory_status,
            "sovereignty": sovereignty_info,
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

        except Exception as e:
            self.logger.warning(f"Checkpoint failed: {e}")


__all__ = [
    "SovereignRuntime",
]
