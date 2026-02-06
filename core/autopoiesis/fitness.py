"""
Fitness Evaluator — Ihsān-Constrained Agent Fitness
===============================================================================

Evaluates agent fitness using multi-objective optimization:
- Ihsān score (excellence/quality)
- SNR score (signal clarity)
- Novelty score (behavioral diversity)
- Efficiency score (resource usage)

Constitutional constraint: Agents must maintain Ihsān ≥ 0.95 to survive.

Standing on Giants: Pareto (multi-objective) + Shannon + Anthropic
Genesis Strict Synthesis v2.2.2
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.autopoiesis import (
    FITNESS_EFFICIENCY_WEIGHT,
    FITNESS_IHSAN_WEIGHT,
    FITNESS_NOVELTY_WEIGHT,
    FITNESS_SNR_WEIGHT,
)
from core.autopoiesis.genome import AgentGenome
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


class FitnessComponent(Enum):
    """Components of the fitness function."""

    IHSAN = "ihsan"
    SNR = "snr"
    NOVELTY = "novelty"
    EFFICIENCY = "efficiency"
    TASK_SUCCESS = "task_success"
    COLLABORATION = "collaboration"


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""

    genome_id: str
    overall_fitness: float
    component_scores: Dict[FitnessComponent, float]
    ihsan_compliant: bool
    is_elite: bool
    rank: int = 0
    crowding_distance: float = 0.0  # For NSGA-II style selection
    evaluation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "overall_fitness": self.overall_fitness,
            "component_scores": {k.value: v for k, v in self.component_scores.items()},
            "ihsan_compliant": self.ihsan_compliant,
            "is_elite": self.is_elite,
            "rank": self.rank,
        }


@dataclass
class EvaluationContext:
    """Context for fitness evaluation."""

    task_history: List[Dict[str, Any]] = field(default_factory=list)
    peer_genomes: List[AgentGenome] = field(default_factory=list)
    environment_metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0


class FitnessEvaluator:
    """
    Multi-objective fitness evaluator for agent genomes.

    Implements Ihsān-constrained optimization where agents must
    maintain constitutional thresholds to be considered viable.

    Usage:
        evaluator = FitnessEvaluator()

        # Evaluate single genome
        result = await evaluator.evaluate(genome, context)
        if result.ihsan_compliant:
            print(f"Fitness: {result.overall_fitness}")

        # Rank population
        ranked = await evaluator.rank_population(population, context)
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        weights: Optional[Dict[FitnessComponent, float]] = None,
        task_evaluator: Optional[Callable[[AgentGenome, Dict], float]] = None,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.task_evaluator = task_evaluator

        # Default weights
        self.weights = weights or {
            FitnessComponent.IHSAN: FITNESS_IHSAN_WEIGHT,
            FitnessComponent.SNR: FITNESS_SNR_WEIGHT,
            FitnessComponent.NOVELTY: FITNESS_NOVELTY_WEIGHT,
            FitnessComponent.EFFICIENCY: FITNESS_EFFICIENCY_WEIGHT,
        }

    async def evaluate(
        self,
        genome: AgentGenome,
        context: EvaluationContext,
    ) -> FitnessResult:
        """Evaluate fitness of a single genome."""
        start = datetime.now(timezone.utc)
        component_scores: Dict[FitnessComponent, float] = {}

        # Evaluate each component
        component_scores[FitnessComponent.IHSAN] = self._evaluate_ihsan(genome, context)
        component_scores[FitnessComponent.SNR] = self._evaluate_snr(genome, context)
        component_scores[FitnessComponent.NOVELTY] = self._evaluate_novelty(
            genome, context
        )
        component_scores[FitnessComponent.EFFICIENCY] = self._evaluate_efficiency(
            genome, context
        )

        # Task success (if evaluator provided)
        if self.task_evaluator and context.task_history:
            task_score = self.task_evaluator(genome, context.task_history[-1])
            component_scores[FitnessComponent.TASK_SUCCESS] = task_score

        # Calculate overall fitness (weighted sum)
        overall = sum(
            self.weights.get(comp, 0.1) * score
            for comp, score in component_scores.items()
        )

        # Ihsān compliance check (hard constraint)
        ihsan_score = component_scores[FitnessComponent.IHSAN]
        ihsan_compliant = ihsan_score >= self.ihsan_threshold

        # Non-compliant agents get severe fitness penalty
        if not ihsan_compliant:
            overall *= 0.1  # 90% penalty

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        # Update genome's fitness
        genome.fitness = overall

        return FitnessResult(
            genome_id=genome.id,
            overall_fitness=overall,
            component_scores=component_scores,
            ihsan_compliant=ihsan_compliant,
            is_elite=False,  # Set during ranking
            evaluation_time_ms=elapsed_ms,
        )

    def _evaluate_ihsan(
        self,
        genome: AgentGenome,
        context: EvaluationContext,
    ) -> float:
        """Evaluate Ihsān (excellence) score."""
        # Base score from genome configuration
        ihsan_gene = genome.get_gene("ihsan_threshold")
        base_score = ihsan_gene.value if ihsan_gene else 0.95

        # Adjust based on task history
        if context.task_history:
            recent_scores = [
                t.get("ihsan_score", 0.95) for t in context.task_history[-10:]
            ]
            if recent_scores:
                base_score = (base_score + sum(recent_scores) / len(recent_scores)) / 2

        return min(1.0, base_score)

    def _evaluate_snr(
        self,
        genome: AgentGenome,
        context: EvaluationContext,
    ) -> float:
        """Evaluate signal-to-noise ratio."""
        snr_gene = genome.get_gene("snr_threshold")
        base_snr = snr_gene.value if snr_gene else 0.85

        # Reasoning depth contributes to signal quality
        depth_gene = genome.get_gene("reasoning_depth")
        if depth_gene:
            depth_bonus = min(0.1, depth_gene.value * 0.01)
            base_snr = min(1.0, base_snr + depth_bonus)

        return base_snr

    def _evaluate_novelty(
        self,
        genome: AgentGenome,
        context: EvaluationContext,
    ) -> float:
        """Evaluate behavioral novelty (diversity from population)."""
        if not context.peer_genomes:
            return 1.0  # First agent is maximally novel

        # Calculate average distance from peers
        distances = [
            genome.distance(peer)
            for peer in context.peer_genomes
            if peer.id != genome.id
        ]

        if not distances:
            return 1.0

        # k-nearest neighbors novelty (use 5 nearest)
        k = min(5, len(distances))
        k_nearest = sorted(distances)[:k]
        novelty = sum(k_nearest) / k

        return min(1.0, novelty)

    def _evaluate_efficiency(
        self,
        genome: AgentGenome,
        context: EvaluationContext,
    ) -> float:
        """Evaluate resource efficiency."""
        efficiency = 0.0

        # Batch size efficiency
        batch_gene = genome.get_gene("batch_size")
        if batch_gene:
            # Sweet spot around 8-16
            batch_val = batch_gene.value
            batch_eff = 1.0 - abs(12 - batch_val) / 52  # Normalized
            efficiency += batch_eff * 0.3

        # Cache strategy
        cache_gene = genome.get_gene("cache_strategy")
        if cache_gene:
            cache_scores = {"none": 0.2, "lru": 0.7, "lfu": 0.6, "adaptive": 0.9}
            efficiency += cache_scores.get(cache_gene.value, 0.5) * 0.3

        # Parallelism
        parallel_gene = genome.get_gene("parallel_tasks")
        if parallel_gene:
            # Sweet spot around 4-8
            parallel_val = parallel_gene.value
            parallel_eff = 1.0 - abs(6 - parallel_val) / 10
            efficiency += max(0, parallel_eff) * 0.4

        return min(1.0, efficiency)

    async def rank_population(
        self,
        population: List[AgentGenome],
        context: EvaluationContext,
    ) -> List[FitnessResult]:
        """Rank entire population by fitness."""
        # Update context with peer genomes
        context.peer_genomes = population

        # Evaluate all genomes
        results = []
        for genome in population:
            result = await self.evaluate(genome, context)
            results.append(result)

        # Sort by fitness (descending)
        results.sort(key=lambda r: r.overall_fitness, reverse=True)

        # Assign ranks and mark elites
        elite_count = max(1, int(len(results) * 0.1))  # Top 10%
        for i, result in enumerate(results):
            result.rank = i + 1
            result.is_elite = i < elite_count

        # Calculate crowding distance for NSGA-II style diversity preservation
        self._calculate_crowding_distances(results)

        return results

    def _calculate_crowding_distances(self, results: List[FitnessResult]):
        """Calculate crowding distance for diversity preservation."""
        if len(results) < 3:
            for r in results:
                r.crowding_distance = float("inf")
            return

        # For each objective, calculate contribution to crowding
        for component in FitnessComponent:
            # Sort by this component
            sorted_by_comp = sorted(
                results, key=lambda r: r.component_scores.get(component, 0)
            )

            # Boundary points get infinite distance
            sorted_by_comp[0].crowding_distance = float("inf")
            sorted_by_comp[-1].crowding_distance = float("inf")

            # Calculate range
            min_val = sorted_by_comp[0].component_scores.get(component, 0)
            max_val = sorted_by_comp[-1].component_scores.get(component, 0)
            range_val = max_val - min_val if max_val > min_val else 1.0

            # Calculate distances
            for i in range(1, len(sorted_by_comp) - 1):
                prev_val = sorted_by_comp[i - 1].component_scores.get(component, 0)
                next_val = sorted_by_comp[i + 1].component_scores.get(component, 0)
                sorted_by_comp[i].crowding_distance += (next_val - prev_val) / range_val


class SelectionPressure:
    """
    Selection mechanisms for evolutionary pressure.

    Implements tournament selection with Ihsān as a hard constraint.
    """

    def __init__(
        self,
        tournament_size: int = 5,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.tournament_size = tournament_size
        self.ihsan_threshold = ihsan_threshold

    def tournament_select(
        self,
        results: List[FitnessResult],
        n_select: int,
    ) -> List[FitnessResult]:
        """Select n individuals via tournament selection."""
        selected = []

        # Filter to compliant only
        compliant = [r for r in results if r.ihsan_compliant]

        if not compliant:
            # Emergency: relax constraint if no compliant individuals
            compliant = results[: max(1, len(results) // 2)]

        for _ in range(n_select):
            # Random tournament
            tournament = random.sample(
                compliant, min(self.tournament_size, len(compliant))
            )

            # Select best from tournament
            winner = max(tournament, key=lambda r: r.overall_fitness)
            selected.append(winner)

        return selected

    def elitist_select(
        self,
        results: List[FitnessResult],
        n_elite: int,
    ) -> List[FitnessResult]:
        """Select top n individuals (elitism)."""
        # Filter to compliant first
        compliant = [r for r in results if r.ihsan_compliant]

        if not compliant:
            compliant = results

        # Sort by fitness and take top n
        sorted_results = sorted(
            compliant, key=lambda r: r.overall_fitness, reverse=True
        )

        return sorted_results[:n_elite]

    def diversity_select(
        self,
        results: List[FitnessResult],
        n_select: int,
    ) -> List[FitnessResult]:
        """Select with crowding distance for diversity."""
        # Filter compliant
        compliant = [r for r in results if r.ihsan_compliant]

        if not compliant:
            compliant = results

        # Sort by crowding distance (higher = more diverse)
        sorted_results = sorted(
            compliant, key=lambda r: r.crowding_distance, reverse=True
        )

        return sorted_results[:n_select]


# Import random at module level
import random
