"""
Autopoiesis → SDPO Bridge — Evolutionary Learning Integration
═══════════════════════════════════════════════════════════════════════════════

Bridges the Autopoietic Loop (genetic evolution) with the SDPO training
pipeline (self-distillation). When autopoiesis produces high-fitness genomes,
this bridge converts their behavioral traces into SDPO training batches.

Data Flow:
    AutopoieticLoop.evolve()
        → FitnessResult (score, metadata, behavioral trace)
            → SDPOBridge.convert_to_training_batch()
                → TrainingBatch (questions, attempts, feedback, corrections)
                    → BIZRASDPOTrainer.train()

This closes the gap identified in Blueprint P0 (Section 3.1):
"Autopoiesis and SDPO exist in parallel but never cross-pollinate."

Standing on Giants:
- Holland (Genetic Algorithms, 1975)
- Maturana & Varela (Autopoiesis, 1972)
- Shannon (Information Theory / SNR, 1948)
- Anthropic (Constitutional AI / Ihsān, 2023)
- SDPO Paper (Self-Distillation with Rich Feedback)

Constitutional: All bridge operations gated by Ihsān ≥ 0.95.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Bridge-specific thresholds
EVOLUTION_TO_TRAINING_MIN_FITNESS = 0.80  # Minimum fitness to be worth training on
EVOLUTION_TO_TRAINING_MIN_SAMPLES = 3  # Minimum evolved genomes per batch
REPRODUCIBILITY_THRESHOLD = 0.90  # Must be reproducible before training
MAX_BATCH_SIZE = 16  # Prevent memory overload in training pipeline


@dataclass
class EvolutionTrace:
    """Behavioral trace from an evolved genome.

    Captures what an agent genome DID during evaluation — the raw
    material that SDPO can learn from.
    """

    genome_id: str
    fitness: float
    ihsan_score: float
    snr_score: float
    task_description: str
    task_output: str
    reasoning_steps: List[str]
    quality_feedback: str  # From fitness evaluator
    improvement_suggestions: List[str]
    evaluation_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passes_ihsan(self) -> bool:
        return self.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    @property
    def passes_snr(self) -> bool:
        return self.snr_score >= UNIFIED_SNR_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "task_description": self.task_description,
            "task_output": self.task_output,
            "reasoning_steps": self.reasoning_steps,
            "quality_feedback": self.quality_feedback,
            "improvement_suggestions": self.improvement_suggestions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BridgeResult:
    """Result of an evolution-to-training bridge operation."""

    traces_received: int
    traces_filtered: int  # Below fitness/ihsan thresholds
    traces_converted: int
    batch_size: int
    avg_fitness: float
    avg_ihsan: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def conversion_ratio(self) -> float:
        if self.traces_received == 0:
            return 0.0
        return self.traces_converted / self.traces_received

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces_received": self.traces_received,
            "traces_filtered": self.traces_filtered,
            "traces_converted": self.traces_converted,
            "batch_size": self.batch_size,
            "conversion_ratio": self.conversion_ratio,
            "avg_fitness": self.avg_fitness,
            "avg_ihsan": self.avg_ihsan,
            "timestamp": self.timestamp.isoformat(),
        }


class AutopoiesisSDPOBridge:
    """Bridge between Autopoietic evolution and SDPO training.

    Collects behavioral traces from evolved genomes, filters by
    constitutional thresholds, and converts to SDPO training batches.

    Usage:
        bridge = AutopoiesisSDPOBridge()

        # During autopoietic evaluation
        trace = EvolutionTrace(
            genome_id="gen_42_elite_3",
            fitness=0.91,
            ihsan_score=0.96,
            snr_score=0.93,
            task_description="Synthesize research findings",
            task_output="The key insight is...",
            reasoning_steps=["Gathered sources", "Cross-referenced", "Synthesized"],
            quality_feedback="Strong synthesis, missed secondary source",
            improvement_suggestions=["Include secondary source analysis"],
        )
        bridge.collect(trace)

        # When ready to train
        batch_data = bridge.flush_to_training_data()
    """

    def __init__(
        self,
        min_fitness: float = EVOLUTION_TO_TRAINING_MIN_FITNESS,
        min_samples: int = EVOLUTION_TO_TRAINING_MIN_SAMPLES,
        max_batch: int = MAX_BATCH_SIZE,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self._min_fitness = min_fitness
        self._min_samples = min_samples
        self._max_batch = max_batch
        self._ihsan_threshold = ihsan_threshold
        self._traces: List[EvolutionTrace] = []
        self._bridge_results: List[BridgeResult] = []

    def collect(self, trace: EvolutionTrace) -> bool:
        """Collect an evolution trace for future training conversion.

        Returns True if the trace was accepted, False if filtered.
        """
        if trace.fitness < self._min_fitness:
            logger.debug(
                "Filtered trace %s: fitness %.3f < %.3f",
                trace.genome_id,
                trace.fitness,
                self._min_fitness,
            )
            return False

        if trace.ihsan_score < self._ihsan_threshold:
            logger.debug(
                "Filtered trace %s: ihsan %.3f < %.3f",
                trace.genome_id,
                trace.ihsan_score,
                self._ihsan_threshold,
            )
            return False

        self._traces.append(trace)
        return True

    @property
    def pending_count(self) -> int:
        return len(self._traces)

    @property
    def ready_for_training(self) -> bool:
        return len(self._traces) >= self._min_samples

    def flush_to_training_data(self) -> Optional[Dict[str, List]]:
        """Convert collected traces to SDPO-compatible training data.

        Returns a dict with keys matching TrainingBatch fields:
        - questions: task descriptions
        - failed_attempts: lower-quality outputs (from lower-fitness traces)
        - feedbacks: quality feedback strings
        - corrected_attempts: higher-quality outputs (from higher-fitness traces)
        - quality_scores: ihsan scores

        Returns None if insufficient traces collected.
        """
        if not self.ready_for_training:
            logger.info(
                "Not enough traces for training: %d < %d",
                len(self._traces),
                self._min_samples,
            )
            return None

        # Sort by fitness descending — top traces become "corrected" attempts
        sorted_traces = sorted(self._traces, key=lambda t: t.fitness, reverse=True)

        # Cap at max batch size
        batch_traces = sorted_traces[: self._max_batch]

        # Build training data using contrastive pairs:
        # High-fitness traces provide corrected_attempts
        # Their feedback provides the improvement signal
        questions = []
        failed_attempts = []
        feedbacks = []
        corrected_attempts = []
        quality_scores = []

        for trace in batch_traces:
            questions.append(trace.task_description)
            corrected_attempts.append(trace.task_output)
            feedbacks.append(trace.quality_feedback)
            quality_scores.append(trace.ihsan_score)

            # Construct "failed" version from improvement suggestions
            failed_version = self._construct_counterfactual(trace)
            failed_attempts.append(failed_version)

        # Record bridge result
        all_fitness = [t.fitness for t in batch_traces]
        all_ihsan = [t.ihsan_score for t in batch_traces]
        result = BridgeResult(
            traces_received=len(self._traces),
            traces_filtered=len(self._traces) - len(batch_traces),
            traces_converted=len(batch_traces),
            batch_size=len(batch_traces),
            avg_fitness=sum(all_fitness) / len(all_fitness),
            avg_ihsan=sum(all_ihsan) / len(all_ihsan),
        )
        self._bridge_results.append(result)

        logger.info(
            "Bridge flushed: %d traces → %d training samples (avg fitness: %.3f)",
            result.traces_received,
            result.traces_converted,
            result.avg_fitness,
        )

        # Clear buffer
        self._traces.clear()

        return {
            "questions": questions,
            "failed_attempts": failed_attempts,
            "feedbacks": feedbacks,
            "corrected_attempts": corrected_attempts,
            "quality_scores": quality_scores,
        }

    def get_bridge_history(self) -> List[Dict[str, Any]]:
        """Return history of all bridge operations."""
        return [r.to_dict() for r in self._bridge_results]

    @staticmethod
    def _construct_counterfactual(trace: EvolutionTrace) -> str:
        """Build a plausible 'failed' version from improvement suggestions.

        Uses the improvement suggestions to describe what was missing,
        creating a contrastive pair for SDPO training.
        """
        if not trace.improvement_suggestions:
            return f"[Incomplete attempt for: {trace.task_description}]"

        gaps = "; ".join(trace.improvement_suggestions)
        return (
            f"[Attempt missing: {gaps}] " f"Partial output: {trace.task_output[:200]}"
        )
