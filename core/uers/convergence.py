"""
Convergence Loop — The Engine of Entropy Reduction

The iterative feedback system that drives the manifold toward singularity:
1. Ingestion & Entropy Measurement
2. Hypothesis Generation
3. Vector Interaction & Probing
4. Delta E Evaluation
5. Model Update
6. Termination (Singularity)

The loop continues until H(residual) → 0
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.uers import CONVERGENCE_STATES, PROOF_OF_IMPACT
from core.uers.entropy import EntropyCalculator, ManifoldState
from core.uers.vectors import AnalyticalManifold, VectorType, Probe, ProbeResult

logger = logging.getLogger(__name__)


class LoopStage(str, Enum):
    """Stages of the convergence loop."""
    INGESTION = "ingestion"
    HYPOTHESIS = "hypothesis"
    PROBING = "probing"
    EVALUATION = "evaluation"
    UPDATE = "update"
    TERMINATION = "termination"


class ConvergenceState(str, Enum):
    """State of the convergence process."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    SINGULARITY = "singularity"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Hypothesis:
    """A hypothesis about the target system."""
    id: str
    description: str
    source_vector: VectorType
    target_vector: VectorType
    confidence: float
    predicted_delta_e: float
    probe_operations: List[str] = field(default_factory=list)
    validated: Optional[bool] = None
    actual_delta_e: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description[:100],
            "source": self.source_vector.value,
            "target": self.target_vector.value,
            "confidence": self.confidence,
            "predicted_delta_e": self.predicted_delta_e,
            "validated": self.validated,
            "actual_delta_e": self.actual_delta_e,
        }


@dataclass
class LoopIteration:
    """Record of a single loop iteration."""
    iteration: int
    stage: LoopStage
    entropy_before: float
    entropy_after: float
    delta_e: float
    hypotheses_tested: int
    probes_executed: int
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def was_productive(self) -> bool:
        """Iteration was productive if entropy reduced."""
        return self.delta_e > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "stage": self.stage.value,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "delta_e": self.delta_e,
            "was_productive": self.was_productive,
            "hypotheses": self.hypotheses_tested,
            "probes": self.probes_executed,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ConvergenceResult:
    """Complete result of a convergence run."""
    id: str
    initial_entropy: float
    final_entropy: float
    total_delta_e: float
    iterations: int
    state: ConvergenceState
    singularity_achieved: bool
    iteration_history: List[LoopIteration] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def convergence_rate(self) -> float:
        """Average entropy reduction per iteration."""
        if self.iterations <= 0:
            return 0.0
        return self.total_delta_e / self.iterations

    @property
    def efficiency(self) -> float:
        """Ratio of productive iterations."""
        if not self.iteration_history:
            return 0.0
        productive = sum(1 for i in self.iteration_history if i.was_productive)
        return productive / len(self.iteration_history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "initial_entropy": self.initial_entropy,
            "final_entropy": self.final_entropy,
            "total_delta_e": self.total_delta_e,
            "iterations": self.iterations,
            "state": self.state.value,
            "singularity_achieved": self.singularity_achieved,
            "convergence_rate": self.convergence_rate,
            "efficiency": self.efficiency,
            "duration_ms": self.duration_ms,
            "iteration_history": [i.to_dict() for i in self.iteration_history[-10:]],
        }


class ConvergenceLoop:
    """
    The Convergence Loop — Engine of the UERS Framework.

    Orchestrates the iterative reduction of entropy across the
    5-Dimensional Analytical Manifold until singularity is achieved.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        singularity_threshold: float = 0.01,
        stagnation_limit: int = 5,
        min_delta_e: float = 0.001,
    ):
        self.max_iterations = max_iterations
        self.singularity_threshold = singularity_threshold
        self.stagnation_limit = stagnation_limit
        self.min_delta_e = min_delta_e

        # Components
        self._manifold = AnalyticalManifold()
        self._entropy_calc = EntropyCalculator()

        # State
        self._state = ConvergenceState.INITIALIZING
        self._current_iteration = 0
        self._stagnation_count = 0
        self._results: List[ConvergenceResult] = []

        # Hypothesis generator (can be overridden)
        self._hypothesis_generator: Optional[Callable] = None

        # Probe executor (can be overridden)
        self._probe_executor: Optional[Callable] = None

    # =========================================================================
    # LOOP STAGES
    # =========================================================================

    async def _ingest(
        self,
        surface_data: Optional[bytes] = None,
        structural_data: Optional[Dict] = None,
        behavioral_data: Optional[List[str]] = None,
        hypothetical_data: Optional[Dict] = None,
        contextual_data: Optional[Dict] = None,
    ) -> float:
        """
        Stage 1: Ingestion & Entropy Measurement.

        Ingest data into the manifold and measure initial entropy.
        """
        if surface_data:
            self._manifold.update_surface(surface_data)

        if structural_data:
            self._manifold.update_structural(
                nodes=structural_data.get("nodes", 1),
                edges=structural_data.get("edges", 0),
                components=structural_data.get("components", 1),
            )

        if behavioral_data:
            self._manifold.update_behavioral(behavioral_data)

        if hypothetical_data:
            self._manifold.update_hypothetical(
                explored_paths=hypothetical_data.get("explored", 0),
                total_paths=hypothetical_data.get("total", 1),
                feasible_paths=hypothetical_data.get("feasible", 0),
            )

        if contextual_data:
            self._manifold.update_contextual(
                text=contextual_data.get("text", ""),
                intent_score=contextual_data.get("intent", 0.5),
                alignment_score=contextual_data.get("alignment", 0.5),
            )

        return self._manifold.get_average_entropy()

    async def _generate_hypotheses(self) -> List[Hypothesis]:
        """
        Stage 2: Hypothesis Generation.

        Generate hypotheses about how to reduce entropy.
        """
        hypotheses = []

        # Use custom generator if provided
        if self._hypothesis_generator:
            return self._hypothesis_generator(self._manifold)

        # Default: Generate hypotheses based on probe suggestions
        suggestions = self._manifold.suggest_probes()

        for source, target, operation in suggestions:
            source_entropy = self._manifold._vectors[source].entropy.normalized
            target_entropy = self._manifold._vectors[target].entropy.normalized

            # Estimate potential entropy reduction
            predicted_delta_e = target_entropy * 0.1 * (1 - source_entropy)

            hypothesis = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                description=f"Probe {source.value}→{target.value} via {operation}",
                source_vector=source,
                target_vector=target,
                confidence=1 - source_entropy,  # Higher confidence if source is resolved
                predicted_delta_e=predicted_delta_e,
                probe_operations=[operation],
            )
            hypotheses.append(hypothesis)

        return hypotheses

    async def _execute_probes(
        self,
        hypotheses: List[Hypothesis],
    ) -> Tuple[List[Probe], float]:
        """
        Stage 3: Vector Interaction & Probing.

        Execute probes to test hypotheses and gather information.
        """
        probes = []
        total_delta_e = 0.0

        for hypothesis in hypotheses:
            for operation in hypothesis.probe_operations:
                probe = self._manifold.create_probe(
                    source=hypothesis.source_vector,
                    target=hypothesis.target_vector,
                    operation=operation,
                )

                # Execute with custom executor or default
                if self._probe_executor:
                    probe = self._probe_executor(probe)
                else:
                    probe = self._manifold.execute_probe(probe)

                probes.append(probe)

                if probe.result == ProbeResult.SUCCESS:
                    total_delta_e += probe.delta_e
                    hypothesis.actual_delta_e += probe.delta_e

        return probes, total_delta_e

    async def _evaluate(
        self,
        hypotheses: List[Hypothesis],
        probes: List[Probe],
        delta_e: float,
    ) -> bool:
        """
        Stage 4: Delta E Evaluation.

        Evaluate whether entropy was reduced and update model.
        """
        # Validate hypotheses
        for hypothesis in hypotheses:
            if hypothesis.actual_delta_e > 0:
                hypothesis.validated = True
            else:
                hypothesis.validated = False

        # Determine if progress was made
        if delta_e >= self.min_delta_e:
            self._stagnation_count = 0
            self._state = ConvergenceState.CONVERGING
            return True
        else:
            self._stagnation_count += 1
            if self._stagnation_count >= self.stagnation_limit:
                self._state = ConvergenceState.DIVERGING
            return False

    async def _check_termination(self) -> bool:
        """
        Stage 5: Termination Check.

        Check if singularity has been achieved.
        """
        current_entropy = self._manifold.get_average_entropy()

        if current_entropy <= self.singularity_threshold:
            self._state = ConvergenceState.SINGULARITY
            return True

        if self._current_iteration >= self.max_iterations:
            self._state = ConvergenceState.STOPPED
            return True

        if self._state == ConvergenceState.DIVERGING:
            return True

        return False

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def converge(
        self,
        surface_data: Optional[bytes] = None,
        structural_data: Optional[Dict] = None,
        behavioral_data: Optional[List[str]] = None,
        hypothetical_data: Optional[Dict] = None,
        contextual_data: Optional[Dict] = None,
    ) -> ConvergenceResult:
        """
        Execute the convergence loop.

        Drives the manifold toward singularity through iterative
        hypothesis generation, probing, and entropy reduction.
        """
        result_id = f"conv_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        self._state = ConvergenceState.RUNNING
        self._current_iteration = 0
        self._stagnation_count = 0

        iteration_history: List[LoopIteration] = []
        all_hypotheses: List[Hypothesis] = []

        # Initial ingestion
        initial_entropy = await self._ingest(
            surface_data=surface_data,
            structural_data=structural_data,
            behavioral_data=behavioral_data,
            hypothetical_data=hypothetical_data,
            contextual_data=contextual_data,
        )

        logger.info(f"Convergence started: initial_entropy={initial_entropy:.4f}")

        # Main loop
        while not await self._check_termination():
            self._current_iteration += 1
            iter_start = time.time()

            entropy_before = self._manifold.get_average_entropy()

            # Generate hypotheses
            hypotheses = await self._generate_hypotheses()
            all_hypotheses.extend(hypotheses)

            # Execute probes
            probes, delta_e = await self._execute_probes(hypotheses)

            # Evaluate results
            productive = await self._evaluate(hypotheses, probes, delta_e)

            entropy_after = self._manifold.get_average_entropy()

            # Record iteration
            iteration = LoopIteration(
                iteration=self._current_iteration,
                stage=LoopStage.EVALUATION,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                delta_e=entropy_before - entropy_after,
                hypotheses_tested=len(hypotheses),
                probes_executed=len(probes),
                duration_ms=(time.time() - iter_start) * 1000,
            )
            iteration_history.append(iteration)

            logger.info(
                f"Iteration {self._current_iteration}: "
                f"ΔE={iteration.delta_e:.4f}, "
                f"entropy={entropy_after:.4f}, "
                f"state={self._state.value}"
            )

        final_entropy = self._manifold.get_average_entropy()
        total_delta_e = initial_entropy - final_entropy
        duration_ms = (time.time() - start_time) * 1000

        result = ConvergenceResult(
            id=result_id,
            initial_entropy=initial_entropy,
            final_entropy=final_entropy,
            total_delta_e=total_delta_e,
            iterations=self._current_iteration,
            state=self._state,
            singularity_achieved=self._state == ConvergenceState.SINGULARITY,
            iteration_history=iteration_history,
            hypotheses=all_hypotheses,
            duration_ms=duration_ms,
        )

        self._results.append(result)

        logger.info(
            f"Convergence complete: "
            f"ΔE_total={total_delta_e:.4f}, "
            f"iterations={self._current_iteration}, "
            f"singularity={result.singularity_achieved}"
        )

        return result

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_hypothesis_generator(
        self,
        generator: Callable[[AnalyticalManifold], List[Hypothesis]],
    ) -> None:
        """Set custom hypothesis generator."""
        self._hypothesis_generator = generator

    def set_probe_executor(
        self,
        executor: Callable[[Probe], Probe],
    ) -> None:
        """Set custom probe executor."""
        self._probe_executor = executor

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    def get_manifold(self) -> AnalyticalManifold:
        """Get the analytical manifold."""
        return self._manifold

    def get_state(self) -> ConvergenceState:
        """Get current convergence state."""
        return self._state

    def get_current_entropy(self) -> float:
        """Get current manifold entropy."""
        return self._manifold.get_average_entropy()

    def get_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent convergence results."""
        return [r.to_dict() for r in self._results[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get convergence statistics."""
        if not self._results:
            return {
                "total_runs": 0,
                "state": self._state.value,
            }

        return {
            "total_runs": len(self._results),
            "singularities_achieved": sum(
                1 for r in self._results if r.singularity_achieved
            ),
            "avg_delta_e": sum(r.total_delta_e for r in self._results) / len(self._results),
            "avg_iterations": sum(r.iterations for r in self._results) / len(self._results),
            "avg_efficiency": sum(r.efficiency for r in self._results) / len(self._results),
            "current_state": self._state.value,
            "current_entropy": self.get_current_entropy(),
        }

    def reset(self) -> None:
        """Reset the convergence loop."""
        self._manifold = AnalyticalManifold()
        self._state = ConvergenceState.INITIALIZING
        self._current_iteration = 0
        self._stagnation_count = 0
        logger.info("Convergence loop reset")
