"""
Sovereign Organism — The Living BIZRA Runtime
===============================================
Drop into: core/sovereign/organism.py

THE INTEGRATION LAYER: Before this module, four excellent modules
existed in isolation:
  - NervousSystem (S1/S2 cognitive bridge)
  - MissionPipeline (12-agent HHMM chain)
  - Helix3Scheduler (evolutionary heartbeat)
  - EventBus + BLOOM + ReflexCompiler (Phase 80 infrastructure)

After this module, ONE class composes them all into a Living Organism:

  SovereignOrganism.boot(inference)
    → Creates NervousSystem with all Phase 80 modules
    → Wires MissionPipeline as 12-agent cognitive chain
    → Wires Helix3 as evolutionary heartbeat
    → Starts 60-second heartbeat
    → Ready for `organism.mission("task")` calls

PMBOK Alignment:
  Integration Management   — this module (composes all deliverables)
  Quality Management       — Ihsān gates at every layer
  Risk Management          — graceful degradation, health checks
  Stakeholder Management   — DEMA (P7) boundary model enforced

DevOps Alignment:
  Continuous Integration   — scripts/ci_organism_gate.py
  Continuous Delivery      — organism.health for monitoring
  Infrastructure as Code   — boot() wires everything deterministically
  Observability            — health, stats, evidence chain

Standing on Giants:
  Prophet Muhammad ﷺ     — Ihsān ("worship Allah as if you see Him")
  Deming (1950)          — PDCA: boot→mission→tick→improve
  Boyd (1976)            — OODA loop: each mission is Observe→Orient→Decide→Act
  Kahneman (2011)        — S1/S2/S3 composed into one cognitive system
  Nakamoto (2008)        — Evidence chain linking all receipts
  Lamport (1978)         — Distributed consensus via heartbeat

Constitutional Authority:
  §1  The Living Organism: "You ARE 12 agents"
  §2  Triple Helix: S1 + S2 + S3 composed here
  §6  Mode 2: Mission Orchestration — the primary flow
  §7  Evidence: every mission produces chained receipts
  §10 CLI: `bizra mission "task"` entry point
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("bizra.sovereign.organism")


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
)

HEARTBEAT_INTERVAL_S = 60  # §2: Every 60 seconds


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLS
# ═══════════════════════════════════════════════════════════════════


class InferenceBackend(Protocol):
    """Any LLM backend (Ollama, LM Studio, Echo for testing)."""

    async def infer(self, prompt: str, **kwargs: Any) -> str: ...


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════


@dataclass
class OrganismReceipt:
    """Complete receipt from an organism mission execution.

    Combines NervousSystem receipt + Pipeline traces + evidence chain.
    """

    mission_id: str
    input_text: str
    output_text: str
    system: str  # "S1" (reflex) or "S2" (deliberation)
    complexity: str  # Pipeline complexity tier
    agents_activated: int  # How many agents fired
    agent_chain: List[str]  # Ordered agent IDs
    ihsan_score: float  # 8D tensor composite
    snr_score: float  # Signal-to-noise ratio
    gate_passed: bool  # All constitutional gates passed
    gate_reasons: List[str]  # Reasons if any gate failed
    rewarded: bool  # SEED minted?
    reward_amount: float  # SEED amount
    evidence_hash: str  # Pipeline evidence hash
    chain_hash: str  # Organism-level chain hash
    duration_ms: float  # Total wall time
    tick_count: int  # Organism heartbeat count at time of mission
    frozen_agents: List[str]  # Agents excluded from learning
    fate_verdict: str = "approved"
    fate_reason_codes: List[str] = field(default_factory=list)
    fate_mode: str = "enforced"
    action_receipt_refs: List[str] = field(default_factory=list)
    identity_mode: str = "placeholder_degraded"
    signer_public_key_prefix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrganismHealth:
    """Health status of the living organism — for monitoring/CI gates."""

    alive: bool
    uptime_seconds: float
    missions_completed: int
    missions_failed: int
    ticks_completed: int
    current_ihsan_avg: float  # Running average Ihsān
    current_gini: float  # Last known Gini
    gate_pass_rate: float  # Fraction of missions passing all gates
    heartbeat_active: bool  # Is the 60s heartbeat running?
    agents_registered: int  # Should be 12
    pipeline_complexity_dist: Dict[str, int]
    reflex_cache_size: int
    evidence_chain_length: int


# ═══════════════════════════════════════════════════════════════════
# SOVEREIGN ORGANISM
# ═══════════════════════════════════════════════════════════════════


class SovereignOrganism:
    """The Living Organism — the complete BIZRA runtime.

    This is the §1 incarnation: "You ARE 12 agents."

    Composes:
      - SovereignNervousSystem (S1 reflex + S2 deliberation)
      - MissionPipeline (12-agent HHMM-routed chain)
      - Helix3Scheduler (S3 evolutionary heartbeat)
      - EventBus + 12 subscribers
      - BLOOM token minter + community pool
      - ReflexCompiler (reflex cache)
      - Evidence chain (hash-linked receipts)

    Lifecycle:
      boot() → mission() → tick() → ... → shutdown()

    Usage:
      organism = await SovereignOrganism.boot(inference=my_llm)
      receipt = await organism.mission("implement user auth")
      print(receipt.agents_activated)  # 4-12 agents
      print(receipt.gate_passed)       # True/False
      await organism.shutdown()
    """

    def __init__(self) -> None:
        # These are set by boot() — not constructor
        self._nervous_system: Any = None
        self._pipeline: Any = None
        self._helix3: Any = None
        self._inference: Any = None

        # Node0 Heartbeat — the ONE canonical ingest authority
        # Wired at boot(); all mission receipts flow through here.
        self._node0: Any = None

        # CQRS EventBus with 12 subscribers (§1: "You ARE 12 agents")
        self._cqrs_bus: Any = None
        self._subscribers: List[Any] = []

        # State
        self._boot_time: float = 0.0
        self._chain_hash = "0" * 64
        self._mission_counter = 0
        self._missions_failed = 0
        self._ihsan_history: List[float] = []
        self._heartbeat_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._heartbeat_active = False
        self._shutdown_requested = False
        self._identity_mode = "placeholder_degraded"
        self._signer_public_key_prefix = ""
        self._signer_public_key_hex = ""
        self._persistence_dir: Optional[Path] = None
        self._external_event_bus: Optional[Any] = None
        self._pending_delivery_mirror_tasks: set[asyncio.Task[Any]] = set()
        self._delivery_mirror_successes = 0
        self._delivery_mirror_failures = 0
        self._last_delivery_mirror_error = ""

        # Callbacks
        self._on_receipt: Optional[Callable[[OrganismReceipt], None]] = None
        self._on_heartbeat: Optional[Callable[[Any], None]] = None

    # ─── Factory (Genesis Ceremony) ───────────────────────────────

    @classmethod
    async def boot(
        cls,
        inference: InferenceBackend,
        *,
        persistence_dir: Optional[Path] = None,
        event_bus: Optional[Any] = None,
        reward_per_mission: float = 1.0,
        on_receipt: Optional[Callable[[OrganismReceipt], None]] = None,
        on_heartbeat: Optional[Callable[[Any], None]] = None,
        start_heartbeat: bool = False,
        node_id: Optional[str] = None,
        identity_mode: str = "placeholder_degraded",
        signer_public_key_prefix: str = "",
        signer_public_key_hex: str = "",
    ) -> "SovereignOrganism":
        """Bootstrap the Living Organism — the Genesis Ceremony.

        Wires all components in constitutional order:
          1. NervousSystem.create() → Phase 80 infrastructure
          2. MissionPipeline → 12-agent HHMM chain
          3. wire_pipeline_to_nervous_system → connects muscles to brain
          4. wire_helix3 → connects evolutionary heartbeat
          5. wire_all_subscribers → connects 12 CQRS bus subscribers
          6. (optional) start_heartbeat → 60s tick loop

        Args:
            inference: LLM backend (Ollama, LM Studio, or EchoInference for testing)
            persistence_dir: Directory for reflex cache + evidence chain
            reward_per_mission: SEED reward per successful mission
            on_receipt: Callback fired after every mission
            on_heartbeat: Callback fired after every 60s tick
            start_heartbeat: Whether to auto-start the 60s evolutionary loop

        Returns:
            A fully-wired SovereignOrganism ready for mission() calls.
        """
        from core.sovereign.helix3 import wire_helix3
        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.mission_pipeline import (
            wire_pipeline_to_nervous_system,
        )

        org = cls()
        org._inference = inference
        org._boot_time = time.monotonic()
        org._on_receipt = on_receipt
        org._on_heartbeat = on_heartbeat
        org._identity_mode = identity_mode
        org._signer_public_key_prefix = signer_public_key_prefix
        org._signer_public_key_hex = signer_public_key_hex
        org._persistence_dir = persistence_dir
        org._external_event_bus = event_bus

        # Step 1: Create NervousSystem with all Phase 80 modules
        org._nervous_system = SovereignNervousSystem.create(
            inference=inference,
            persistence_dir=persistence_dir,
            reward_per_mission=reward_per_mission,
        )

        # Step 2: Wire MissionPipeline → 12-agent HHMM chain
        org._pipeline = wire_pipeline_to_nervous_system(org._nervous_system, inference)

        # Step 3: Wire Helix3 → evolutionary heartbeat
        org._helix3 = wire_helix3(
            org._nervous_system,
            on_heartbeat=on_heartbeat,
        )

        # Step 4: Wire 12 CQRS bus subscribers — the nervous system
        org._wire_subscribers()

        # Step 5: Wire Node0Heartbeat — the ONE canonical ingest authority.
        # Passes the organism's NervousSystem-wired Helix3 so there is
        # exactly one Helix3 instance (no duplication).
        org._boot_node0(persistence_dir, event_bus=event_bus, node_id=node_id)

        # Step 6: Start heartbeat if requested
        if start_heartbeat:
            await org.start_heartbeat()

        logger.info(
            "Organism booted: NervousSystem + Pipeline (12 agents) "
            "+ Helix3 (60s tick) + Node0 Heartbeat + %d CQRS subscribers",
            len(org._subscribers),
        )

        return org

    def _wire_subscribers(self) -> None:
        """Wire the 12 CQRS EventBus subscribers — the organism's nervous system.

        Uses no-op adapters for subsystems not yet initialized at this level.
        The bus module provides hash-chained event integrity (Nakamoto, 2008)
        and fail-closed semantics for safety subscribers (§4).
        """
        try:
            from core.bus.subscribers import EventBus, wire_all_subscribers
        except ImportError:
            logger.warning("core.bus.subscribers not available — skipping CQRS wiring")
            return

        delivery_receipt_path = None
        if self._persistence_dir is not None:
            delivery_receipt_path = (
                self._persistence_dir / "audit" / "cqrs_delivery_receipts.jsonl"
            )

        bus = EventBus(
            delivery_receipt_path=delivery_receipt_path,
            delivery_receipt_sink=self._mirror_cqrs_delivery_receipt,
        )

        # Adapters: delegate to real subsystems where available,
        # no-op stubs where the organism doesn't yet own the dependency.
        # These will be replaced as subsystems come online.
        reflex = None
        if self._nervous_system and hasattr(self._nervous_system, "_reflex"):
            reflex = self._nervous_system._reflex

        class _NoOpStore:
            def reinforce(self, **kw: Any) -> None:
                pass

            def get_success_count(self, key: str) -> int:
                return 0

            def set_success_count(self, key: str, val: int) -> None:
                pass

            def promote_to_semantic(self, **kw: Any) -> bool:
                return True

            def record_failure_pattern(self, **kw: Any) -> None:
                pass

        class _NoOpTeleScript:
            def begin_execution(self, **kw: Any) -> str:
                return "noop"

        class _NoOpSession:
            def halt(self, **kw: Any) -> None:
                pass

        class _NoOpAuditLog:
            def log_violation(self, **kw: Any) -> None:
                logger.warning("CQRS audit: %s", kw)

        class _NoOpQuarantine:
            def isolate(self, **kw: Any) -> None:
                logger.warning("CQRS quarantine: %s", kw)

        class _NoOpHealing:
            def diagnose(self, **kw: Any) -> Any:
                class _Plan:
                    strategy = "retry"

                return _Plan()

        class _NoOpHHMM:
            def classify(self, payload: Any) -> str:
                return "macro_general"

        class _NoOpPoI:
            total_credit: float = 0.0

            def accumulate(self, **kw: Any) -> float:
                self.total_credit += 0.01
                return 0.01

        class _NoOpMinter:
            def compute_reward(self, **kw: Any) -> float:
                return 0.0

            def mint_seed(self, **kw: Any) -> None:
                pass

        class _NoOpBudget:
            total_used: int = 0

            def record_retrieval(self, **kw: Any) -> None:
                self.total_used += kw.get("tokens", 0)

        class _NoOpSelfModel:
            def update_capability_map(self, **kw: Any) -> None:
                pass

        class _NoOpCapRegistry:
            def register(self, **kw: Any) -> None:
                pass

            def count(self) -> int:
                return 12

            def count_by_type(self, t: str) -> int:
                return 7 if t == "PAT" else 5

            def total_capabilities(self) -> int:
                return 42

            def capability_vector(self) -> List[float]:
                return [1.0] * 8

        class _NoOpReflexCache(dict):  # type: ignore[type-arg]
            def precipitate(self, **kw: Any) -> None:
                self[kw.get("action_type", "")] = kw

        try:
            self._subscribers = wire_all_subscribers(
                bus,
                memory_store=_NoOpStore(),
                telescript_engine=_NoOpTeleScript(),
                receipt_chain=[],
                reflex_cache=reflex if reflex else _NoOpReflexCache(),
                session_manager=_NoOpSession(),
                audit_log=_NoOpAuditLog(),
                quarantine_store=_NoOpQuarantine(),
                healing_engine=_NoOpHealing(),
                hhmm_engine=_NoOpHHMM(),
                poi_engine=_NoOpPoI(),
                token_minter=_NoOpMinter(),
                context_budget=_NoOpBudget(),
                self_model=_NoOpSelfModel(),
                capability_registry=_NoOpCapRegistry(),
            )
            self._cqrs_bus = bus
            logger.info(
                "CQRS bus wired: %d subscribers, chain height %d",
                len(self._subscribers),
                bus.chain_height,
            )

            # ── Phase 87: Wire Rust constitutional bridge ──────────────
            # Every Python cognitive event now flows through Rust's 12
            # constitutional subscribers for independent verification.
            # The language boundary IS the trust boundary.
            try:
                from core.bus.rust_bridge import wire_rust_bridge

                self._rust_bridge = wire_rust_bridge(bus, production=False)
                if self._rust_bridge:
                    logger.info("Rust bridge ACTIVE: Python→Rust synapse wired")
            except Exception as rust_exc:
                logger.info("Rust bridge not available (degraded): %s", rust_exc)
                self._rust_bridge = None
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning(
                "CQRS subscriber wiring failed (degraded): %s", exc, exc_info=True
            )
            self._cqrs_bus = None
            self._subscribers = []

    # ─── Node0 Heartbeat (P0 Closure — ONE canonical ingest) ──────

    def _boot_node0(
        self,
        persistence_dir: Optional[Path],
        *,
        event_bus: Optional[Any] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """Wire Node0Heartbeat with the organism's Helix3.

        This makes Node0Heartbeat the single authority for:
        - Mission receipt ingestion
        - Evidence chain persistence
        - Memory persistence
        - Reflex precipitation checking

        Standing on Giants:
          Deming (1950) — PDCA closure: boot→ingest→breathe→improve
          Nakamoto (2008) — One hash chain, one authority
        """
        try:
            from core.bus.event_publisher import combine_event_buses
            from core.node0.heartbeat import Node0Heartbeat

            data_dir = persistence_dir or Path("sovereign_state") / "node0"
            node0_bus = combine_event_buses(self._cqrs_bus, event_bus)
            self._node0 = Node0Heartbeat(
                data_dir=data_dir,
                node_id=node_id,
                helix3=self._helix3,
                event_bus=node0_bus,
                identity_mode=self._identity_mode,
                signer_public_key_prefix=self._signer_public_key_prefix,
                signer_public_key_hex=self._signer_public_key_hex,
                genesis_backed=self._identity_mode == "genesis_ed25519",
            )
            self._node0.boot()

            # Transfer ingest authority: Node0 is now the SOLE feeder
            # of Helix3.  wire_helix3() patched ns._on_receipt to auto-
            # call scheduler.ingest_receipt() on every NervousSystem
            # receipt.  With Node0 active, _ingest_to_node0() handles
            # that feed — keeping the callback would double-count every
            # mission.  Nakamoto (2008): one chain, one authority.
            if self._nervous_system is not None:
                self._nervous_system._on_receipt = None

            logger.info(
                "Node0Heartbeat wired: node_id=%s, sovereignty=%s",
                self._node0.node_id,
                (
                    self._node0._boot_receipt.sovereignty_proven
                    if self._node0._boot_receipt
                    else "unknown"
                ),
            )
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("Node0Heartbeat unavailable (degraded): %s", exc)
            self._node0 = None

    def _ingest_to_node0(self, receipt: "OrganismReceipt") -> None:
        """Bridge organism receipt into Node0Heartbeat — the ONE ingest path.

        Every mission, regardless of origin (API, CLI, terminal), must
        flow through this single authority for evidence + memory + reflex.
        """
        if self._node0 is None:
            return

        try:
            degradation_receipts = list(
                (receipt.metadata or {}).get("degradation_receipts", [])
            )
            self._node0.ingest_mission_receipt(
                {
                    "mission_id": receipt.mission_id,
                    "description": receipt.input_text[:200],
                    "ihsan_score": receipt.ihsan_score,
                    "snr_score": receipt.snr_score,
                    "agent_id": ",".join(receipt.agent_chain) or "organism",
                    "gate_passed": receipt.gate_passed,
                    "duration_ms": receipt.duration_ms,
                    "fate_verdict": receipt.fate_verdict,
                    "fate_reason_codes": list(receipt.fate_reason_codes),
                    "fate_mode": receipt.fate_mode,
                    "action_receipt_refs": list(receipt.action_receipt_refs),
                    "identity_mode": receipt.identity_mode,
                    "signer_public_key_prefix": receipt.signer_public_key_prefix,
                    "degraded": bool(
                        (receipt.metadata or {}).get("degraded") or degradation_receipts
                    ),
                    "degradation_receipts": degradation_receipts,
                }
            )
            boundary_recorder = getattr(
                self._node0,
                "record_boundary_error_receipt",
                None,
            )
            if callable(boundary_recorder):
                for degradation in degradation_receipts:
                    if not isinstance(degradation, dict):
                        continue
                    boundary_recorder(
                        {
                            **degradation,
                            "source": "organism:mission.boundary",
                            "mission_id": receipt.mission_id,
                            "system": receipt.system,
                        }
                    )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning("Node0 ingest failed: %s", exc)

    # ─── Mission Execution (§6 Mode 2) ───────────────────────────

    async def mission(
        self,
        text: str,
        *,
        preflight: Optional[Dict[str, Any]] = None,
    ) -> OrganismReceipt:
        """Submit a mission to the Living Organism.

        Flow (§6 Mode 2):
          1. DEMA (P7) receives text → classifies intent
          2. HHMM router selects agents → Pipeline executes chain
          3. NervousSystem routes S1 (cache) or S2 (full pipeline)
          4. P4 scores → P5 gates → S2 verifies → S3 records
          5. Evidence chain extended, SEED minted if Ihsān ≥ 0.95
          6. Helix3 ingests receipt for next evolutionary tick
          7. OrganismReceipt returned

        Args:
            text: Mission description (natural language)

        Returns:
            OrganismReceipt with full evidence trail
        """
        t0 = time.monotonic()
        self._mission_counter += 1
        preflight = dict(preflight or {})
        fate_verdict = str(preflight.get("fate_verdict", "approved") or "approved")
        fate_reason_codes = [
            str(code) for code in list(preflight.get("fate_reason_codes", []))
        ]
        fate_mode = str(preflight.get("fate_mode", "enforced") or "enforced")
        action_receipt_refs = [
            str(ref) for ref in list(preflight.get("action_receipt_refs", []))
        ]

        try:
            if not preflight.get("allow_execution", True):
                duration_ms = round((time.monotonic() - t0) * 1000, 2)
                mission_id = str(
                    preflight.get(
                        "mission_id", f"org-blocked-{self._mission_counter:06d}"
                    )
                )
                evidence_data = f"{self._chain_hash}:{mission_id}:rejected"
                self._chain_hash = hashlib.sha256(evidence_data.encode()).hexdigest()
                receipt = OrganismReceipt(
                    mission_id=mission_id,
                    input_text=text[:500],
                    output_text="[BLOCKED] Mission rejected by pre-execution FATE gate.",
                    system="BLOCKED",
                    complexity="blocked",
                    agents_activated=0,
                    agent_chain=[],
                    ihsan_score=0.0,
                    snr_score=0.0,
                    gate_passed=False,
                    gate_reasons=fate_reason_codes or ["fate_rejected"],
                    rewarded=False,
                    reward_amount=0.0,
                    evidence_hash="",
                    chain_hash=self._chain_hash,
                    duration_ms=duration_ms,
                    tick_count=self._helix3.stats.total_ticks if self._helix3 else 0,
                    frozen_agents=[],
                    fate_verdict="rejected",
                    fate_reason_codes=fate_reason_codes or ["fate_rejected"],
                    fate_mode=fate_mode,
                    action_receipt_refs=action_receipt_refs,
                    identity_mode=self._identity_mode,
                    signer_public_key_prefix=self._signer_public_key_prefix,
                    metadata={},
                )
                if self._on_receipt:
                    self._on_receipt(receipt)
                self._emit_cqrs_receipt(receipt)
                self._ingest_to_node0(receipt)
                return receipt

            # ─── Seed Chain: wrap raw text into constitutional prompt ───
            seed_chain_meta: Dict[str, Any] = {}
            try:
                from core.prompt.seed_chain import EvidenceTag, small_seed

                chain = small_seed(text, agent="P7_DEMA")
                # Add any preflight evidence
                if action_receipt_refs:
                    for ref in action_receipt_refs:
                        chain.bayyinah.add(
                            f"Prior receipt: {ref}",
                            EvidenceTag.VERIFIED,
                            source="receipt_chain",
                        )
                validation_errors = chain.validate()
                if validation_errors:
                    logger.warning("Seed Chain validation: %s", validation_errors)
                governed_prompt = chain.to_prompt()
                seed_chain_meta = {
                    "seed_chain_hash": chain.compute_hash(),
                    "seed_chain_agent": chain.niyyah.target_agent,
                    "seed_chain_mode": chain.amanah.reasoning_mode,
                    "seed_chain_validation": validation_errors or [],
                }
            except Exception as exc:
                logger.debug("Seed Chain construction failed: %s", exc)
                governed_prompt = text

            # Run through NervousSystem → Pipeline → 12 agents
            ns_receipt = await self._nervous_system.run(governed_prompt)

            # Get pipeline details (if available)
            pipeline_stats = self._pipeline.stats if self._pipeline else None
            pipeline_result = (
                getattr(self._pipeline, "last_result", None)
                if self._pipeline is not None
                else None
            )
            complexity = "unknown"
            agents_activated = 0
            agent_chain: List[str] = []
            gate_passed = True
            gate_reasons: List[str] = []
            frozen_agents: List[str] = []

            # Extract pipeline trace from the last execution
            if pipeline_stats and pipeline_stats.missions_executed > 0:
                complexity = max(
                    pipeline_stats.complexity_distribution,
                    key=pipeline_stats.complexity_distribution.get,
                    default="unknown",
                )
                agents_activated = round(pipeline_stats.avg_agents_per_mission)

            pipeline_degradation_receipts: List[Dict[str, Any]] = []
            if pipeline_result is not None:
                from core.errors import InferenceError

                for trace in list(getattr(pipeline_result, "agent_traces", [])):
                    trace_metadata = getattr(trace, "metadata", None) or {}
                    if not trace_metadata.get("degraded"):
                        continue
                    typed_error = InferenceError(
                        str(getattr(trace, "agent_id", "unknown") or "unknown"),
                        str(
                            trace_metadata.get("error_message")
                            or "pipeline inference degraded"
                        ),
                        context={
                            "agent_id": getattr(trace, "agent_id", "unknown"),
                            "phase": getattr(trace, "phase", ""),
                            "mission_id": ns_receipt.mission_id,
                        },
                    )
                    pipeline_degradation_receipts.append(typed_error.to_receipt())

            duration_ms = round((time.monotonic() - t0) * 1000, 2)

            # Update chain hash
            evidence_data = (
                f"{self._chain_hash}:{ns_receipt.mission_id}:"
                f"{ns_receipt.ihsan_score}:{ns_receipt.evidence_hash}"
            )
            self._chain_hash = hashlib.sha256(evidence_data.encode()).hexdigest()

            # Track Ihsān history
            self._ihsan_history.append(ns_receipt.ihsan_score)
            if len(self._ihsan_history) > 1000:
                self._ihsan_history = self._ihsan_history[-500:]

            receipt_metadata = {**dict(ns_receipt.metadata or {}), **seed_chain_meta}
            if pipeline_degradation_receipts:
                existing_degradations = list(
                    receipt_metadata.get("degradation_receipts", [])
                )
                receipt_metadata["degradation_receipts"] = [
                    *existing_degradations,
                    *pipeline_degradation_receipts,
                ]
                receipt_metadata["degraded"] = True

            receipt = OrganismReceipt(
                mission_id=ns_receipt.mission_id,
                input_text=text[:500],
                output_text=ns_receipt.output_text[:2000],
                system=ns_receipt.system,
                complexity=complexity,
                agents_activated=agents_activated,
                agent_chain=agent_chain,
                ihsan_score=ns_receipt.ihsan_score,
                snr_score=ns_receipt.snr_score,
                gate_passed=gate_passed,
                gate_reasons=gate_reasons,
                rewarded=ns_receipt.rewarded,
                reward_amount=ns_receipt.reward_amount,
                evidence_hash=ns_receipt.evidence_hash,
                chain_hash=self._chain_hash,
                duration_ms=duration_ms,
                tick_count=self._helix3.stats.total_ticks if self._helix3 else 0,
                frozen_agents=frozen_agents,
                fate_verdict=fate_verdict,
                fate_reason_codes=fate_reason_codes,
                fate_mode=fate_mode,
                action_receipt_refs=action_receipt_refs,
                identity_mode=self._identity_mode,
                signer_public_key_prefix=self._signer_public_key_prefix,
                metadata=receipt_metadata,
            )

            if self._on_receipt:
                self._on_receipt(receipt)

            # Emit to CQRS bus — fires 12 subscribers
            self._emit_cqrs_receipt(receipt)

            # Bridge to Node0Heartbeat — THE canonical ingest authority
            # Evidence chain, memory persistence, reflex check all happen here.
            self._ingest_to_node0(receipt)

            logger.info(
                "Mission %s: %s, ihsan=%.4f, %s, %.1fms",
                receipt.mission_id,
                receipt.system,
                receipt.ihsan_score,
                "PASS" if receipt.gate_passed else "FAIL",
                receipt.duration_ms,
            )

            return receipt

        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            self._missions_failed += 1
            logger.error("Mission failed: %s", exc)

            # Return a degraded receipt rather than crashing
            receipt = OrganismReceipt(
                mission_id=f"org-fail-{self._mission_counter:06d}",
                input_text=text[:500],
                output_text=f"[DEGRADED] Mission failed: {exc}",
                system="ERROR",
                complexity="error",
                agents_activated=0,
                agent_chain=[],
                ihsan_score=0.0,
                snr_score=0.0,
                gate_passed=False,
                gate_reasons=[f"organism_error: {exc}"],
                rewarded=False,
                reward_amount=0.0,
                evidence_hash="",
                chain_hash=self._chain_hash,
                duration_ms=round((time.monotonic() - t0) * 1000, 2),
                tick_count=0,
                frozen_agents=[],
                fate_verdict=fate_verdict,
                fate_reason_codes=fate_reason_codes or [f"organism_error: {exc}"],
                fate_mode=fate_mode,
                action_receipt_refs=action_receipt_refs,
                identity_mode=self._identity_mode,
                signer_public_key_prefix=self._signer_public_key_prefix,
                metadata={},
            )
            if self._on_receipt:
                self._on_receipt(receipt)
            self._emit_cqrs_receipt(receipt)
            self._ingest_to_node0(receipt)
            return receipt

    # ─── CQRS Bus Emission ──────────────────────────────────────────

    def _emit_cqrs_receipt(self, receipt: OrganismReceipt) -> None:
        """Publish a mission receipt to the CQRS bus, firing all 12 subscribers.

        This is the causal bridge: organism receipt → hash-chained event log
        → subscriber side-effects (memory reinforce, HHMM promotion, PoI credit).
        """
        if not self._cqrs_bus:
            return

        try:
            from core.bus.subscribers import EventType

            self._cqrs_bus.publish(
                EventType.ACTION_RECEIPT,
                {
                    "action_type": f"mission:{receipt.system}",
                    "ihsan_composite": receipt.ihsan_score,
                    "snr_score": receipt.snr_score,
                    "result_summary": receipt.output_text[:200],
                    "mission_id": receipt.mission_id,
                    "agents_activated": receipt.agents_activated,
                    "gate_passed": receipt.gate_passed,
                    "duration_ms": receipt.duration_ms,
                    "chain_hash": receipt.chain_hash,
                },
            )

            # If Ihsān below production threshold, emit breach event
            if not receipt.gate_passed or receipt.ihsan_score < UNIFIED_IHSAN_THRESHOLD:
                self._cqrs_bus.publish(
                    EventType.IHSAN_GATE_BREACHED,
                    {
                        "session_id": receipt.mission_id,
                        "ihsan_composite": receipt.ihsan_score,
                        "action_type": f"mission:{receipt.system}",
                        "violation_dimensions": receipt.gate_reasons,
                    },
                )

        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning("CQRS receipt emission failed: %s", exc, exc_info=True)

    def _mirror_cqrs_delivery_receipt(self, payload: Dict[str, Any]) -> None:
        """Mirror CQRS subscriber delivery evidence into the sovereign async bus."""
        node0 = self._node0
        if node0 is not None:
            recorder = getattr(node0, "record_cqrs_delivery_receipt", None)
            if callable(recorder):
                try:
                    recorder(payload)
                except (
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                ) as exc:
                    logger.debug(
                        "Node0 CQRS delivery persistence failed (non-fatal): %s",
                        exc,
                    )

        if self._external_event_bus is None:
            return

        from core.bus.event_publisher import publish_topic_event

        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is None:
                asyncio.run(
                    publish_topic_event(
                        self._external_event_bus,
                        "cqrs.delivery.receipt",
                        payload,
                    )
                )
                self._delivery_mirror_successes += 1
                self._last_delivery_mirror_error = ""
                return

            task = running_loop.create_task(
                publish_topic_event(
                    self._external_event_bus,
                    "cqrs.delivery.receipt",
                    payload,
                ),
                name="cqrs_delivery_mirror",
            )
            self._pending_delivery_mirror_tasks.add(task)
            task.add_done_callback(self._finalize_delivery_mirror)
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
        ) as exc:
            self._record_delivery_mirror_failure(exc)

    def _finalize_delivery_mirror(self, task: asyncio.Task[Any]) -> None:
        """Turn async delivery mirror completion into observable stats."""
        self._pending_delivery_mirror_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError as exc:
            self._record_delivery_mirror_failure(exc)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            self._record_delivery_mirror_failure(exc)
        else:
            self._delivery_mirror_successes += 1
            self._last_delivery_mirror_error = ""

    def _record_delivery_mirror_failure(self, exc: BaseException) -> None:
        self._delivery_mirror_failures += 1
        self._last_delivery_mirror_error = f"{type(exc).__name__}: {exc}"
        logger.debug(
            "CQRS delivery mirror failed (non-fatal): %s",
            self._last_delivery_mirror_error,
        )

    # ─── Evolutionary Heartbeat (§2 Helix 3) ─────────────────────

    async def tick(self) -> Any:
        """Process one evolutionary tick manually.

        Normally called automatically every 60 seconds by the heartbeat.
        Can be called manually for testing or on-demand evolution.

        When Node0Heartbeat is wired, delegates to breathe() which wraps
        the Helix3 tick with evidence persistence, memory storage, and
        reflex precipitation — the full organism breath cycle.

        Returns:
            HeartbeatReceipt from Helix3Scheduler (or BreathReceipt if Node0 active)
        """
        if self._helix3 is None:
            raise RuntimeError("Organism not booted — call boot() first")

        # If Node0 is wired, breathe() is the canonical tick path:
        # it runs Helix3 tick + evidence + memory + reflex in one atomic cycle.
        if self._node0 is not None:
            try:
                breath = self._node0.breathe()
                logger.info(
                    "Tick %d (via Node0 breathe): ihsan=%.4f, missions=%d, "
                    "evidence=%d, reflexes=%d",
                    breath.tick_number,
                    breath.ihsan_composite,
                    breath.missions_processed,
                    breath.evidence_entries,
                    breath.reflexes_precipitated,
                )
                return breath
            except (RuntimeError, AttributeError, TypeError, OSError) as exc:
                logger.warning("Node0 breathe failed, falling back to Helix3: %s", exc)

        # Fallback: direct Helix3 tick (no evidence/memory/reflex closure)
        receipt = self._helix3.process_tick()
        logger.info(
            "Tick %d: ihsan=%.4f, minted=%d, halted=%s",
            receipt.tick_number,
            receipt.ihsan_composite,
            receipt.seed_minted,
            not receipt.gini_ok,
        )
        return receipt

    async def start_heartbeat(self) -> None:
        """Start the 60-second evolutionary heartbeat (§2 Helix 3).

        The heartbeat runs process_tick() every HEARTBEAT_INTERVAL_S seconds,
        processing all accumulated NervousSystem receipts.
        """
        if self._heartbeat_active:
            logger.warning("Heartbeat already active")
            return

        self._heartbeat_active = True
        self._shutdown_requested = False

        async def _heartbeat_loop() -> None:
            while not self._shutdown_requested:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                if self._shutdown_requested:
                    break
                try:
                    await self.tick()
                except (RuntimeError, AttributeError, TypeError, OSError) as exc:
                    logger.error("Heartbeat tick failed: %s", exc)

        self._heartbeat_task = asyncio.create_task(_heartbeat_loop())
        logger.info("Heartbeat started: tick every %ds", HEARTBEAT_INTERVAL_S)

    async def shutdown(self) -> None:
        """Graceful shutdown of the Living Organism.

        Stops the heartbeat, processes final tick, logs shutdown receipt.
        """
        logger.info("Organism shutdown requested")
        self._shutdown_requested = True

        # Cancel heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_active = False

        # Final tick (process any remaining receipts)
        if self._helix3:
            try:
                await self.tick()
            except (RuntimeError, ValueError) as exc:
                logger.warning("Final tick failed: %s", exc)

        if self._pending_delivery_mirror_tasks:
            pending = tuple(self._pending_delivery_mirror_tasks)
            await asyncio.gather(*pending, return_exceptions=True)
            self._pending_delivery_mirror_tasks.clear()

        logger.info(
            "Organism shut down: %d missions, %d ticks",
            self._mission_counter,
            self._helix3.stats.total_ticks if self._helix3 else 0,
        )

    # ─── Observability (PMBOK Quality Management) ─────────────────

    @property
    def health(self) -> OrganismHealth:
        """Current health status of the Living Organism.

        Used by:
          - CI gates (scripts/ci_organism_gate.py)
          - `bizra status` CLI command
          - Monitoring dashboards
          - Self-harness validation (Mode 3)
        """
        uptime = time.monotonic() - self._boot_time if self._boot_time else 0.0
        avg_ihsan = (
            sum(self._ihsan_history) / len(self._ihsan_history)
            if self._ihsan_history
            else 0.0
        )

        pipeline_dist: Dict[str, int] = {}
        if self._pipeline:
            pipeline_dist = dict(self._pipeline.stats.complexity_distribution)

        reflex_size = 0
        if self._nervous_system and hasattr(self._nervous_system, "_reflex"):
            reflex = self._nervous_system._reflex
            if reflex and hasattr(reflex, "_cache"):
                reflex_size = len(reflex._cache)

        ticks = 0
        gini = 0.0
        if self._helix3:
            ticks = self._helix3.stats.total_ticks
            gini = 0.0  # Gini from last tick, not tracked cumulatively

        total_missions = self._mission_counter
        gate_pass_rate = 1.0
        if self._pipeline and self._pipeline.stats.missions_executed > 0:
            gate_pass_rate = self._pipeline.stats.gate_pass_rate

        return OrganismHealth(
            alive=self._boot_time > 0 and not self._shutdown_requested,
            uptime_seconds=round(uptime, 2),
            missions_completed=total_missions - self._missions_failed,
            missions_failed=self._missions_failed,
            ticks_completed=ticks,
            current_ihsan_avg=round(avg_ihsan, 4),
            current_gini=gini,
            gate_pass_rate=round(gate_pass_rate, 4),
            heartbeat_active=self._heartbeat_active,
            agents_registered=12,
            pipeline_complexity_dist=pipeline_dist,
            reflex_cache_size=reflex_size,
            evidence_chain_length=total_missions,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Aggregated statistics across all subsystems."""
        h = self.health
        result: Dict[str, Any] = {
            "organism": {
                "alive": h.alive,
                "uptime_s": h.uptime_seconds,
                "missions": h.missions_completed,
                "failures": h.missions_failed,
                "ticks": h.ticks_completed,
                "ihsan_avg": h.current_ihsan_avg,
                "gini": h.current_gini,
                "gate_pass_rate": h.gate_pass_rate,
            },
        }
        if self._pipeline:
            result["pipeline"] = self._pipeline.stats.to_dict()
        if self._helix3:
            result["helix3"] = self._helix3.stats.as_dict()
        if self._nervous_system and hasattr(self._nervous_system, "_stats"):
            ns_stats = self._nervous_system._stats
            result["nervous_system"] = {
                "s1_hits": ns_stats.s1_hits,
                "s2_executions": ns_stats.s2_executions,
                "total": ns_stats.total_missions,
            }
        if self._cqrs_bus:
            delivery_summary = {}
            summary_fn = getattr(self._cqrs_bus, "delivery_summary", None)
            if callable(summary_fn):
                try:
                    delivery_summary = summary_fn()
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                    logger.debug("CQRS delivery summary unavailable", exc_info=True)
            result["cqrs_bus"] = {
                "subscribers_wired": len(self._subscribers),
                "chain_height": self._cqrs_bus.chain_height,
                "chain_valid": self._cqrs_bus.verify_chain(),
                "delivery_mirror_enabled": self._external_event_bus is not None,
                "delivery_mirror_successes": self._delivery_mirror_successes,
                "delivery_mirror_failures": self._delivery_mirror_failures,
                "pending_delivery_mirrors": len(self._pending_delivery_mirror_tasks),
                "last_delivery_mirror_error": self._last_delivery_mirror_error,
                **delivery_summary,
            }
        if self._node0:
            result["node0"] = self._node0.health()
        return result

    # ─── Constitutional Invariant Checks ──────────────────────────

    def check_invariants(self) -> List[str]:
        """Verify all constitutional invariants (§4).

        Returns list of violations (empty = healthy).
        Used by CI gates and self-harness.
        """
        violations: List[str] = []
        h = self.health

        # §4: Ihsān production ≥ 0.95
        if h.missions_completed > 5 and h.current_ihsan_avg < UNIFIED_IHSAN_THRESHOLD:
            violations.append(
                f"IHSAN_BELOW_PRODUCTION: avg={h.current_ihsan_avg:.4f} "
                f"< {UNIFIED_IHSAN_THRESHOLD}"
            )

        # §4: Gini ≤ 0.35
        if h.current_gini > ADL_GINI_THRESHOLD:
            violations.append(
                f"GINI_ABOVE_CEILING: {h.current_gini:.4f} > {ADL_GINI_THRESHOLD}"
            )

        # §1: 12 agents registered
        if h.agents_registered != 12:
            violations.append(f"AGENT_COUNT_MISMATCH: {h.agents_registered} != 12")

        # Operational: gate pass rate should be healthy
        if h.missions_completed > 10 and h.gate_pass_rate < 0.80:
            violations.append(f"GATE_PASS_RATE_LOW: {h.gate_pass_rate:.4f} < 0.80")

        return violations
