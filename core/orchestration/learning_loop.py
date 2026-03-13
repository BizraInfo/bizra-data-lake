"""
Learning Loop Orchestrator — Closes the Autopoiesis → SDPO → Reflex Pipeline
═══════════════════════════════════════════════════════════════════════════════

The keystone integration that connects three self-improvement engines into a
single closed-loop learning pipeline:

    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │Autopoiesis│───▶│   SDPO    │───▶│  Reflex   │
    │  discover │    │  distill  │    │  compile  │
    └─────▲─────┘    └───────────┘    └─────┬─────┘
          │                                  │
          └──────── feedback ────────────────┘

Data Flow:
    AutopoieticLoop.on_integration → IntegrationCandidate
        → AutopoiesisSDPOBridge.collect(EvolutionTrace)
            → flush_to_training_data() → TrainingBatch
                → BIZRASDPOTrainer.train(batches) → TrainingResult
                    → SDPOReflexBridge.observe(task, ihsan, snr, loss, success)
                        → get_eligible_candidates() → ReflexCandidate[]
                            → compile_reflex(pattern, chain, confidence) → Reflex

Feature flag: BIZRA_CLOSED_LOOP_ENABLED (default=False, opt-in).

Standing on Giants:
- Deming (PDCA, 1950) — Plan-Do-Check-Act as a closed loop
- Holland (Genetic Algorithms, 1975) — evolutionary discovery
- Kahneman (System 1/2, 2011) — reflex = compiled System-2 → System-1
- Shannon (Information Theory, 1948) — SNR as quality gradient
- Anthropic (Constitutional AI) — Ihsān as hard constraint

Blueprint Reference: Elite Implementation Blueprint v1.0 — P0 Close Learning Loop
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.autopoiesis.loop import IntegrationCandidate
from core.autopoiesis.sdpo_bridge import AutopoiesisSDPOBridge, EvolutionTrace
from core.constitutional.algorithms import compile_reflex
from core.constitutional.fixed_point import fp
from core.constitutional.types import Reflex
from core.hashtable.cognitive_hash_table import CognitiveHashTable
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.prediction.hierarchical_hmm import HierarchicalHMMEngine
from core.sdpo.reflex_bridge import ReflexCandidate, SDPOReflexBridge
from core.sdpo.training.bizra_sdpo_trainer import (
    BIZRASDPOTrainer,
    TrainingBatch,
    TrainingResult,
)

logger = logging.getLogger(__name__)

# Feature flag: off by default for safety
CLOSED_LOOP_ENABLED = os.environ.get("BIZRA_CLOSED_LOOP_ENABLED", "0") == "1"


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator Events
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LoopEvent:
    """A learning loop lifecycle event for observability."""

    event_type: str  # CANDIDATE_RECEIVED, TRAINING_STARTED, REFLEX_COMPILED, etc.
    source: str  # Which stage emitted this event
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LoopMetrics:
    """Aggregate metrics for the learning loop."""

    candidates_received: int = 0
    candidates_accepted: int = 0
    candidates_filtered: int = 0
    training_runs: int = 0
    training_batches_total: int = 0
    reflexes_compiled: int = 0
    reflexes_denied: int = 0
    total_observations: int = 0
    avg_training_ihsan: float = 0.0
    loop_cycles: int = 0
    last_cycle_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_received": self.candidates_received,
            "candidates_accepted": self.candidates_accepted,
            "candidates_filtered": self.candidates_filtered,
            "training_runs": self.training_runs,
            "training_batches_total": self.training_batches_total,
            "reflexes_compiled": self.reflexes_compiled,
            "reflexes_denied": self.reflexes_denied,
            "total_observations": self.total_observations,
            "avg_training_ihsan": round(self.avg_training_ihsan, 4),
            "loop_cycles": self.loop_cycles,
            "last_cycle_at": (
                self.last_cycle_at.isoformat() if self.last_cycle_at else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Learning Loop Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class LearningLoopOrchestrator:
    """Connects autopoiesis → SDPO training → reflex compilation.

    Three integration points:
    1. ``on_candidate()`` — called by autopoiesis when a genome is integration-ready
    2. ``run_training_cycle()`` — called periodically to flush buffered traces → SDPO
    3. ``run_compilation_cycle()`` — called periodically to promote patterns → reflexes

    All operations are gated by Ihsān thresholds and the BIZRA_CLOSED_LOOP_ENABLED
    feature flag. When disabled, the orchestrator accepts events but does not
    execute training or compilation (dry-run mode for telemetry).
    """

    def __init__(
        self,
        sdpo_trainer: Optional[BIZRASDPOTrainer] = None,
        reflex_cache: Optional[Dict[bytes, Reflex]] = None,
        evolution_bridge: Optional[AutopoiesisSDPOBridge] = None,
        reflex_bridge: Optional[SDPOReflexBridge] = None,
        hmm_engine: Optional[HierarchicalHMMEngine] = None,
        context_cache: Optional[CognitiveHashTable] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._trainer = sdpo_trainer
        self._reflex_cache = reflex_cache if reflex_cache is not None else {}
        self._evo_bridge = evolution_bridge or AutopoiesisSDPOBridge()
        self._reflex_bridge = reflex_bridge or SDPOReflexBridge()
        self._hmm = hmm_engine or HierarchicalHMMEngine()
        self._context_cache = context_cache or CognitiveHashTable()

        self._enabled = enabled if enabled is not None else CLOSED_LOOP_ENABLED
        self._metrics = LoopMetrics()
        self._events: List[LoopEvent] = []
        self._training_ihsan_history: List[float] = []

        logger.info(
            "LearningLoopOrchestrator initialized — enabled=%s, "
            "trainer=%s, reflex_cache_size=%d",
            self._enabled,
            self._trainer is not None,
            len(self._reflex_cache),
        )

    # ─── Stage 1: Autopoiesis → Evolution Bridge ────────────────────────────

    def on_candidate(self, candidate: IntegrationCandidate) -> bool:
        """Handle an integration candidate from the autopoietic loop.

        Converts the candidate to an EvolutionTrace and feeds it to the
        evolution→SDPO bridge. Returns True if accepted, False if filtered.

        This method is designed to be passed as the ``on_integration``
        callback to ``AutopoieticLoop.__init__()``.
        """
        self._metrics.candidates_received += 1
        self._emit(
            "CANDIDATE_RECEIVED",
            "autopoiesis",
            {
                "genome_id": getattr(candidate.genome, "genome_id", "unknown"),
                "fitness": candidate.fitness,
                "ihsan_score": candidate.ihsan_score,
                "recommendation": candidate.recommendation,
            },
        )

        # Gate: fitness and ihsan
        if candidate.fitness < 0.90:
            self._metrics.candidates_filtered += 1
            logger.debug(
                "Candidate filtered: fitness %.3f < 0.90",
                candidate.fitness,
            )
            return False

        if candidate.ihsan_score < UNIFIED_IHSAN_THRESHOLD:
            self._metrics.candidates_filtered += 1
            logger.debug(
                "Candidate filtered: ihsan %.3f < %.3f",
                candidate.ihsan_score,
                UNIFIED_IHSAN_THRESHOLD,
            )
            return False

        # Convert IntegrationCandidate → EvolutionTrace
        genome = candidate.genome
        trace = EvolutionTrace(
            genome_id=getattr(genome, "genome_id", "unknown"),
            fitness=candidate.fitness,
            ihsan_score=candidate.ihsan_score,
            snr_score=getattr(genome, "snr_score", UNIFIED_SNR_THRESHOLD),
            task_description=getattr(
                genome,
                "task_description",
                f"Evolved genome ({candidate.recommendation})",
            ),
            task_output=getattr(
                genome,
                "task_output",
                f"Genome output (fitness={candidate.fitness:.3f})",
            ),
            reasoning_steps=getattr(genome, "reasoning_steps", []),
            quality_feedback=candidate.recommendation,
            improvement_suggestions=getattr(
                genome,
                "improvement_suggestions",
                [],
            ),
        )

        accepted = self._evo_bridge.collect(trace)
        if accepted:
            self._metrics.candidates_accepted += 1
            self._emit(
                "CANDIDATE_ACCEPTED",
                "evolution_bridge",
                {
                    "genome_id": trace.genome_id,
                    "pending_count": self._evo_bridge.pending_count,
                    "ready_for_training": self._evo_bridge.ready_for_training,
                },
            )
        else:
            self._metrics.candidates_filtered += 1

        return accepted

    # ─── Stage 2: Evolution Bridge → SDPO Training ──────────────────────────

    async def run_training_cycle(self) -> Optional[TrainingResult]:
        """Flush buffered evolution traces to SDPO training.

        Returns the TrainingResult if training was executed, None otherwise.
        Called periodically (e.g., every 5 autopoiesis cycles).
        """
        if not self._evo_bridge.ready_for_training:
            logger.debug(
                "Training cycle skipped — %d traces pending (need %d)",
                self._evo_bridge.pending_count,
                self._evo_bridge._min_samples,
            )
            return None

        # Flush traces → training data
        training_data = self._evo_bridge.flush_to_training_data()
        if training_data is None:
            return None

        self._emit(
            "TRAINING_DATA_READY",
            "evolution_bridge",
            {
                "batch_size": len(training_data["questions"]),
            },
        )

        if not self._enabled:
            logger.info(
                "Learning loop disabled — training data flushed but not executed "
                "(set BIZRA_CLOSED_LOOP_ENABLED=1 to enable)",
            )
            self._emit(
                "TRAINING_SKIPPED_DISABLED",
                "orchestrator",
                {
                    "batch_size": len(training_data["questions"]),
                },
            )
            return None

        if self._trainer is None:
            logger.warning(
                "No SDPO trainer configured — cannot execute training",
            )
            self._emit("TRAINING_SKIPPED_NO_TRAINER", "orchestrator", {})
            return None

        # Build TrainingBatch
        batch = TrainingBatch(
            questions=training_data["questions"],
            failed_attempts=training_data["failed_attempts"],
            feedbacks=training_data["feedbacks"],
            corrected_attempts=training_data["corrected_attempts"],
            quality_scores=training_data["quality_scores"],
        )

        self._emit(
            "TRAINING_STARTED",
            "sdpo_trainer",
            {
                "batch_size": len(batch),
            },
        )

        t0 = time.monotonic()
        result = await self._trainer.train([batch])
        elapsed = time.monotonic() - t0

        # Update metrics
        self._metrics.training_runs += 1
        self._metrics.training_batches_total += 1
        self._training_ihsan_history.append(result.final_ihsan_score)
        self._metrics.avg_training_ihsan = sum(self._training_ihsan_history) / len(
            self._training_ihsan_history
        )

        self._emit(
            "TRAINING_COMPLETED",
            "sdpo_trainer",
            {
                "final_loss": result.final_loss,
                "final_ihsan": result.final_ihsan_score,
                "epochs": result.total_epochs_completed,
                "steps": result.total_steps,
                "elapsed_seconds": round(elapsed, 2),
            },
        )

        # Feed training results → reflex bridge observations
        self._observe_training_results(training_data, result)

        logger.info(
            "Training cycle completed: loss=%.4f, ihsan=%.3f, elapsed=%.1fs",
            result.final_loss,
            result.final_ihsan_score,
            elapsed,
        )

        return result

    def _observe_training_results(
        self,
        training_data: Dict[str, List],
        result: TrainingResult,
    ) -> None:
        """Pipe training results into the reflex bridge as observations."""
        questions = training_data["questions"]
        quality_scores = training_data["quality_scores"]
        snr_scores = training_data.get("snr_scores") or []

        for i, question in enumerate(questions):
            ihsan = quality_scores[i] if i < len(quality_scores) else 0.0
            snr = snr_scores[i] if i < len(snr_scores) else 0.0
            # Use training result's overall ihsan as a floor
            effective_ihsan = max(ihsan, result.final_ihsan_score)
            effective_snr = max(0.0, min(1.0, snr))

            self._reflex_bridge.observe(
                task_description=question,
                ihsan_score=effective_ihsan,
                snr_score=effective_snr,
                loss=result.final_loss,
                success=(
                    effective_ihsan >= UNIFIED_IHSAN_THRESHOLD
                    and effective_snr >= UNIFIED_SNR_THRESHOLD
                ),
            )
            self._metrics.total_observations += 1

    # ─── Stage 3: Reflex Bridge → Constitutional Compilation ────────────────

    def run_compilation_cycle(self) -> List[ReflexCandidate]:
        """Check for eligible reflex candidates and compile them.

        Returns the list of candidates that were successfully compiled.
        Called periodically (e.g., every 30 seconds or every tick).
        """
        self._metrics.loop_cycles += 1
        self._metrics.last_cycle_at = datetime.now(timezone.utc)

        candidates = self._reflex_bridge.get_eligible_candidates()
        if not candidates:
            return []

        self._emit(
            "COMPILATION_CANDIDATES_FOUND",
            "reflex_bridge",
            {
                "candidate_count": len(candidates),
            },
        )

        if not self._enabled:
            logger.info(
                "Learning loop disabled — %d candidates eligible but not compiled",
                len(candidates),
            )
            return []

        compiled = []
        for candidate in candidates:
            reflex = self._compile_candidate(candidate)
            if reflex is not None:
                compiled.append(candidate)

        if compiled:
            logger.info(
                "Compilation cycle: %d/%d candidates compiled to reflexes",
                len(compiled),
                len(candidates),
            )

        return compiled

    def _compile_candidate(self, candidate: ReflexCandidate) -> Optional[Reflex]:
        """Compile a single reflex candidate via constitutional algorithm.

        Converts the floating-point ihsan score to fixed-point and delegates
        to ``compile_reflex()`` from ``core/constitutional/algorithms.py``.
        """
        # Convert float → fixed-point for constitutional algorithm
        confidence = fp(candidate.avg_ihsan)

        # Action chain: the pattern description becomes the action sequence
        action_chain = [candidate.pattern_description]

        reflex = compile_reflex(
            pattern=candidate.pattern_description,
            action_chain=action_chain,
            confidence=confidence,
        )

        if reflex is None:
            # compile_reflex returns None if confidence < IHSAN_FLOOR
            self._metrics.reflexes_denied += 1
            self._reflex_bridge.deny(
                candidate.pattern_id,
                reason=f"Rejected by compile_reflex (confidence={confidence})",
            )
            self._emit(
                "REFLEX_DENIED",
                "compiler",
                {
                    "pattern_id": candidate.pattern_id[:16],
                    "avg_ihsan": candidate.avg_ihsan,
                    "confidence_fp": confidence,
                },
            )
            return None

        # Store in reflex cache
        self._reflex_cache[reflex.pattern_hash] = reflex
        self._context_cache.put(candidate.pattern_id, reflex)  # High-perf O(1) access
        self._reflex_bridge.mark_compiled(candidate.pattern_id)
        self._metrics.reflexes_compiled += 1

        self._emit(
            "REFLEX_COMPILED",
            "compiler",
            {
                "pattern_id": candidate.pattern_id[:16],
                "pattern_hash": reflex.pattern_hash.hex()[:16],
                "avg_ihsan": candidate.avg_ihsan,
                "reproducibility": candidate.reproducibility,
                "observations": candidate.observation_count,
                "cache_size": len(self._reflex_cache),
            },
        )

        return reflex

    # ─── Full Cycle (convenience) ───────────────────────────────────────────

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Execute one complete learning loop cycle.

        Runs training (if traces are buffered) then compilation.
        Returns a summary dict.
        """
        training_result = await self.run_training_cycle()
        compiled = self.run_compilation_cycle()

        return {
            "training_executed": training_result is not None,
            "training_ihsan": (
                training_result.final_ihsan_score if training_result else None
            ),
            "reflexes_compiled": len(compiled),
            "reflex_cache_size": len(self._reflex_cache),
            "metrics": self._metrics.to_dict(),
        }

    # ─── Observability ──────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def metrics(self) -> LoopMetrics:
        return self._metrics

    @property
    def reflex_cache_size(self) -> int:
        return len(self._reflex_cache)

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive orchestrator status."""
        return {
            "enabled": self._enabled,
            "evolution_bridge": {
                "pending_traces": self._evo_bridge.pending_count,
                "ready_for_training": self._evo_bridge.ready_for_training,
                "bridge_history": len(self._evo_bridge.get_bridge_history()),
            },
            "reflex_bridge": self._reflex_bridge.get_status(),
            "reflex_cache_size": len(self._reflex_cache),
            "metrics": self._metrics.to_dict(),
            "recent_events": [e.to_dict() for e in self._events[-10:]],
        }

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent loop events."""
        return [e.to_dict() for e in self._events[-limit:]]

    def _emit(self, event_type: str, source: str, payload: Dict[str, Any]) -> None:
        """Emit a loop lifecycle event."""
        event = LoopEvent(event_type=event_type, source=source, payload=payload)
        self._events.append(event)
        # Cap event history at 500 entries
        if len(self._events) > 500:
            self._events = self._events[-500:]
        logger.debug("LoopEvent: %s from %s — %s", event_type, source, payload)
