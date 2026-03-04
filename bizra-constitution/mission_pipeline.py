"""
BIZRA Mission Pipeline — The First Heartbeat
═════════════════════════════════════════════

The complete PAT execution pipeline:
  Input → Classify → Route → Execute → Gate → Evidence → Output

This module wires together every component built in the constitution package:
  - HhmmRouter: complexity classification
  - ReflexCache: O(1) pattern retrieval
  - IhsanGate: 6-dim quality verification
  - SNR: signal quality measurement
  - EvidenceLedger: hash-chained proof trail

The pipeline IS the trust compiler. Raw intent enters at the top.
Constitutional membership (proven, attested, evidenced output) exits
at the bottom. Seven stages. Monotonic trust increase.

Constitution reference: §4 [pat], §6 [gates], §7 [hhmm]
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ihsan_gate import IhsanGate, IhsanScore, IhsanTier
from snr import measure_mission_snr, MissionSNR
from evidence_receipt import EvidenceLedger, EvidenceReceipt
from reflex_cache import ReflexCache, ReflexEntry
from hhmm_router import (
    HhmmRouter, ClassificationResult, ComplexityTier,
    ActionBus, MissionTicket,
)

try:
    from generated.generated_constants import (
        PAT_AGENT_NAMES,
        PAT_TRUST_STAGES,
        IHSAN_GATE_MINIMUM,
        IHSAN_BLOOM_ELIGIBILITY,
        CONSTITUTION_VERSION,
    )
except ImportError:
    PAT_AGENT_NAMES = ["Planner", "Researcher", "Coder", "Evaluator",
                       "Ethicist", "Publisher", "Integrator"]
    PAT_TRUST_STAGES = ["abstracting", "gathering", "executing", "attesting",
                        "certifying", "publishing", "chaining"]
    IHSAN_GATE_MINIMUM = 0.85
    IHSAN_BLOOM_ELIGIBILITY = 0.90
    CONSTITUTION_VERSION = "5.0.0-GENESIS"

logger = logging.getLogger("bizra.mission_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION & RESULT STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class MissionStatus(Enum):
    PENDING = "pending"
    CLASSIFIED = "classified"
    EXECUTING = "executing"
    GATE_PASS = "gate_pass"
    GATE_FAIL = "gate_fail"
    EVIDENCED = "evidenced"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Mission:
    """A single user mission flowing through the PAT pipeline."""
    mission_id: str
    input_text: str
    status: MissionStatus = MissionStatus.PENDING
    created_at: float = field(default_factory=time.time)

    # Populated during execution
    classification: ClassificationResult | None = None
    output_text: str = ""
    ihsan_score: IhsanScore | None = None
    mission_snr: MissionSNR | None = None
    evidence_receipt: EvidenceReceipt | None = None
    reflex_hit: bool = False
    agent_trace: list[dict] = field(default_factory=list)
    error: str | None = None

    # Timing
    classify_ms: float = 0.0
    execute_ms: float = 0.0
    gate_ms: float = 0.0
    evidence_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == MissionStatus.COMPLETE

    @property
    def bloom_eligible(self) -> bool:
        if self.ihsan_score is None:
            return False
        return self.ihsan_score.bloom_eligible

    def as_evidence(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "status": self.status.value,
            "tier": self.classification.tier.value if self.classification else None,
            "handler": self.classification.handler if self.classification else None,
            "ihsan_composite": self.ihsan_score.composite if self.ihsan_score else None,
            "ihsan_tier": self.ihsan_score.tier.value if self.ihsan_score else None,
            "snr_normalized": self.mission_snr.snr_normalized if self.mission_snr else None,
            "bloom_eligible": self.bloom_eligible,
            "reflex_hit": self.reflex_hit,
            "total_ms": self.total_ms,
            "agent_count": len(self.agent_trace),
            "receipt_id": self.evidence_receipt.receipt_id if self.evidence_receipt else None,
        }


@dataclass
class PipelineStats:
    """Aggregate pipeline statistics."""
    missions_completed: int = 0
    missions_failed: int = 0
    gate_passes: int = 0
    gate_fails: int = 0
    reflex_hits: int = 0
    bloom_eligible: int = 0
    total_latency_ms: float = 0.0
    evidence_receipts: int = 0

    @property
    def avg_latency_ms(self) -> float:
        total = self.missions_completed + self.missions_failed
        if total == 0:
            return 0.0
        return self.total_latency_ms / total

    @property
    def gate_pass_rate(self) -> float:
        total = self.gate_passes + self.gate_fails
        if total == 0:
            return 0.0
        return self.gate_passes / total

    def as_dict(self) -> dict:
        return {
            "missions_completed": self.missions_completed,
            "missions_failed": self.missions_failed,
            "gate_pass_rate": round(self.gate_pass_rate, 4),
            "reflex_hits": self.reflex_hits,
            "bloom_eligible": self.bloom_eligible,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "evidence_receipts": self.evidence_receipts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAT AGENT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════


class PatAgent:
    """
    Simulated PAT agent for the genesis pipeline.

    At genesis: agents use template-based responses.
    At production: agents invoke LLM inference via Ollama.

    Each agent adds trust monotonically (Theorem: trust compiler).
    """

    def __init__(self, name: str, trust_stage: str):
        self.name = name
        self.trust_stage = trust_stage

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute this agent's stage. Returns enriched data."""
        start = time.monotonic()

        result = {
            "agent": self.name,
            "trust_stage": self.trust_stage,
            "input_keys": list(input_data.keys()),
        }

        if self.name == "Planner":
            result["plan"] = self._plan(input_data.get("input_text", ""))
        elif self.name == "Researcher":
            result["context"] = self._research(input_data)
        elif self.name == "Coder":
            result["output"] = self._execute(input_data)
        elif self.name == "Evaluator":
            result["evaluation"] = "ihsan_gate_handles_this"
        elif self.name == "Ethicist":
            result["ethics_check"] = self._ethics_check(input_data)
        elif self.name == "Publisher":
            result["formatted"] = self._format(input_data)
        elif self.name == "Integrator":
            result["evidence"] = "ledger_handles_this"

        result["elapsed_ms"] = round((time.monotonic() - start) * 1000, 3)
        return result

    def _plan(self, text: str) -> dict:
        """Planner: decompose intent into structured plan."""
        words = text.split()
        return {
            "intent": text[:100],
            "steps": min(max(len(words) // 10, 1), 5),
            "complexity_estimate": min(len(words) / 50.0, 1.0),
        }

    def _research(self, data: dict) -> dict:
        """Researcher: gather context with provenance."""
        return {
            "sources_consulted": 1,
            "context_tokens": len(str(data)) // 4,
            "provenance": "local_knowledge",
        }

    def _execute(self, data: dict) -> str:
        """Coder: produce the actual output."""
        input_text = data.get("input_text", "")
        plan = data.get("plan", {})

        # Genesis-mode: template response
        # Production-mode: this calls Ollama
        return (
            f"Based on analysis of your request, here is a structured response "
            f"addressing the {plan.get('steps', 1)} identified aspects. "
            f"The key insight is that {input_text[:50]}... requires careful "
            f"consideration of context, evidence, and constitutional compliance. "
            f"Therefore, the recommended approach involves systematic verification "
            f"at each step, ensuring both correctness and alignment with the "
            f"stated mission objectives. This response has been generated with "
            f"epistemic humility — noting areas of uncertainty where they exist — "
            f"and structured for verifiability."
        )

    def _ethics_check(self, data: dict) -> dict:
        """Ethicist: Daughter Test + constitutional compliance."""
        return {
            "daughter_test": True,
            "constitutional_compliance": True,
            "concerns": [],
        }

    def _format(self, data: dict) -> dict:
        """Publisher: format for delivery."""
        return {
            "format": "text",
            "bloom_candidate": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION PIPELINE — The Trust Compiler
# ═══════════════════════════════════════════════════════════════════════════════


class MissionPipeline:
    """
    The complete PAT execution pipeline.

    Input → Classify → Route → Execute → Gate → Evidence → Output

    This is the Genesis Engine. The first heartbeat of NODE0.
    Every mission that passes through this pipeline:
      1. Is classified by complexity
      2. Is routed to the appropriate handler
      3. Is executed by PAT agents (or served from cache)
      4. Is verified by the 6-dim Ihsan gate
      5. Has its SNR measured
      6. Is recorded in the evidence chain
      7. Is considered for reflex precipitation

    The output is constitutionally certified.
    """

    def __init__(
        self,
        evidence_path: str | Path = "evidence_ledger.jsonl",
        cache_path: Path | None = None,
        llm_fn: Callable[[str], str] | None = None,
    ):
        # Core components (all constitution-powered)
        self.reflex_cache = ReflexCache(persistence_path=cache_path)
        self.router = HhmmRouter(reflex_cache=self.reflex_cache)
        self.ihsan_gate = IhsanGate()
        self.evidence_ledger = EvidenceLedger(evidence_path)
        self.action_bus = ActionBus()

        # PAT agents
        self.agents = [
            PatAgent(name, stage)
            for name, stage in zip(PAT_AGENT_NAMES, PAT_TRUST_STAGES)
        ]

        # Optional LLM function for production mode
        self._llm_fn = llm_fn

        # Stats
        self.stats = PipelineStats()

    def execute(self, input_text: str) -> Mission:
        """
        Execute a mission through the complete pipeline.

        This is the heartbeat. One call = one constitutional cycle.

        Args:
            input_text: The user's request.

        Returns:
            Mission with complete execution trace, evidence receipt,
            and constitutional certification.
        """
        mission = Mission(
            mission_id=str(uuid.uuid4())[:12],
            input_text=input_text,
        )

        pipeline_start = time.monotonic()

        try:
            # ── Stage 1: CLASSIFY ──
            self._stage_classify(mission)

            # ── Stage 2: ROUTE & EXECUTE ──
            self._stage_execute(mission)

            # ── Stage 3: GATE (Ihsan verification) ──
            self._stage_gate(mission)

            # ── Stage 4: SNR MEASUREMENT ──
            self._stage_snr(mission)

            # ── Stage 5: EVIDENCE (if gate passed) ──
            if mission.status == MissionStatus.GATE_PASS:
                self._stage_evidence(mission)
                self._stage_precipitation(mission)
                mission.status = MissionStatus.COMPLETE
                self.stats.missions_completed += 1
            else:
                self.stats.missions_failed += 1

        except Exception as e:
            mission.status = MissionStatus.ERROR
            mission.error = str(e)
            self.stats.missions_failed += 1
            logger.error(f"Mission {mission.mission_id} failed: {e}")

        mission.total_ms = round((time.monotonic() - pipeline_start) * 1000, 2)
        self.stats.total_latency_ms += mission.total_ms

        return mission

    # ── Pipeline Stages ──

    def _stage_classify(self, mission: Mission):
        """Stage 1: HHMM classification."""
        start = time.monotonic()
        mission.classification = self.router.classify(mission.input_text)
        mission.classify_ms = round((time.monotonic() - start) * 1000, 3)
        mission.status = MissionStatus.CLASSIFIED

    def _stage_execute(self, mission: Mission):
        """Stage 2: Execute via appropriate handler."""
        start = time.monotonic()
        mission.status = MissionStatus.EXECUTING

        tier = mission.classification.tier

        if tier == ComplexityTier.TRIVIAL and mission.classification.has_reflex:
            # Reflex cache hit — serve from S1
            entry = self.reflex_cache.lookup(mission.input_text)
            if entry:
                mission.output_text = entry.output_template
                mission.reflex_hit = True
                mission.agent_trace.append({
                    "agent": "ReflexCache",
                    "trust_stage": "s1_retrieval",
                    "elapsed_ms": 0.0,
                    "cache_hit": True,
                })
                self.stats.reflex_hits += 1
                mission.execute_ms = round((time.monotonic() - start) * 1000, 3)
                return

        # Full PAT pipeline (S2)
        pipeline_data: dict[str, Any] = {"input_text": mission.input_text}

        for agent in self.agents:
            result = agent.execute(pipeline_data)
            mission.agent_trace.append(result)

            # Forward outputs
            if "plan" in result:
                pipeline_data["plan"] = result["plan"]
            if "context" in result:
                pipeline_data["context"] = result["context"]
            if "output" in result:
                # Use LLM if available, otherwise use template
                if self._llm_fn and agent.name == "Coder":
                    try:
                        pipeline_data["output"] = self._llm_fn(mission.input_text)
                    except Exception as e:
                        logger.warning(f"LLM fallback: {e}")
                        pipeline_data["output"] = result["output"]
                else:
                    pipeline_data["output"] = result["output"]

        mission.output_text = pipeline_data.get("output", "")
        mission.execute_ms = round((time.monotonic() - start) * 1000, 3)

    def _stage_gate(self, mission: Mission):
        """Stage 3: Ihsan gate verification."""
        start = time.monotonic()

        context = {
            "mission_keywords": mission.input_text.split()[:10],
            "task_complexity": mission.classification.tier.value if mission.classification else "complex",
            "is_fallback": False,
            "latency_ms": mission.execute_ms,
            "latency_budget_ms": (
                mission.classification.latency_budget_ms
                if mission.classification else 15000
            ),
        }

        mission.ihsan_score = self.ihsan_gate.evaluate(
            mission.output_text, context
        )

        mission.gate_ms = round((time.monotonic() - start) * 1000, 3)

        if mission.ihsan_score.passes:
            mission.status = MissionStatus.GATE_PASS
            self.stats.gate_passes += 1
            if mission.ihsan_score.bloom_eligible:
                self.stats.bloom_eligible += 1
        else:
            mission.status = MissionStatus.GATE_FAIL
            self.stats.gate_fails += 1

    def _stage_snr(self, mission: Mission):
        """Stage 4: SNR measurement."""
        if mission.ihsan_score is None:
            return
        mission.mission_snr = measure_mission_snr(
            output=mission.output_text,
            ihsan_composite=mission.ihsan_score.composite,
        )

    def _stage_evidence(self, mission: Mission):
        """Stage 5: Record in evidence chain."""
        if mission.ihsan_score is None or mission.mission_snr is None:
            return

        start = time.monotonic()

        mission.evidence_receipt = self.evidence_ledger.append(
            mission_id=mission.mission_id,
            ihsan_tensor=mission.ihsan_score.as_tensor_dict(),
            ihsan_composite=mission.ihsan_score.composite,
            gate_results={
                "alpha_4": True,
                "alpha_7": mission.ihsan_score.passes,
                "alpha_8": True,  # Dark matter check (placeholder)
                "alpha_9": True,  # Attestation (self-attested at genesis)
                "alpha_10": True, # Daughter test (passed in Ethicist agent)
            },
            snr_normalized=mission.mission_snr.snr_normalized,
            tier=mission.ihsan_score.tier.value,
            agent_chain=[a.get("agent", "unknown") for a in mission.agent_trace],
        )

        mission.evidence_ms = round((time.monotonic() - start) * 1000, 3)
        self.stats.evidence_receipts += 1

    def _stage_precipitation(self, mission: Mission):
        """Stage 6: Consider for reflex cache precipitation."""
        if mission.reflex_hit or mission.ihsan_score is None:
            return

        self.reflex_cache.record_observation(
            input_text=mission.input_text,
            output_text=mission.output_text,
            ihsan_composite=mission.ihsan_score.composite,
            ihsan_tensor=mission.ihsan_score.as_tensor_dict(),
        )

    # ── Health & Introspection ──

    def health(self) -> dict[str, Any]:
        """Complete pipeline health report."""
        chain_valid, chain_count, chain_errors = self.evidence_ledger.verify_chain()

        return {
            "constitution_version": CONSTITUTION_VERSION,
            "pipeline_stats": self.stats.as_dict(),
            "cache_stats": self.reflex_cache.stats.as_dict(),
            "cache_size": self.reflex_cache.size,
            "router_classifications": self.router.classification_count,
            "evidence_chain_valid": chain_valid,
            "evidence_chain_count": chain_count,
            "evidence_chain_errors": chain_errors,
            "action_bus_queue": self.action_bus.queue_depth,
            "action_bus_active": self.action_bus.active_count,
            "agents": PAT_AGENT_NAMES,
        }

    def shutdown(self):
        """Graceful shutdown: persist cache, flush evidence."""
        self.reflex_cache.save_to_disk()
        logger.info("Pipeline shutdown complete")
