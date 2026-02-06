"""
Evolution Engine — Genetic Algorithm for Agent Evolution
===============================================================================

Implements the core evolutionary loop:
1. Initialize population
2. Evaluate fitness
3. Select parents
4. Crossover and mutation
5. Replace population
6. Repeat until convergence

Constitutional constraint: All offspring must maintain Ihsān compliance.

Standing on Giants: Holland (GA) + Darwin + Anthropic
Genesis Strict Synthesis v2.2.2
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import asyncio
import random

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.autopoiesis import (
    POPULATION_SIZE,
    ELITE_RATIO,
    MUTATION_RATE,
    CROSSOVER_RATE,
    GENERATION_LIMIT,
)
from core.autopoiesis.genome import AgentGenome, GenomeFactory
from core.autopoiesis.fitness import (
    FitnessEvaluator,
    FitnessResult,
    EvaluationContext,
    SelectionPressure,
)


class EvolutionStatus(Enum):
    """Status of the evolution process."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    CONVERGED = "converged"
    STALLED = "stalled"
    COMPLETED = "completed"


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    min_fitness: float
    fitness_std: float
    ihsan_compliance_rate: float
    elite_count: int
    mutations: int
    crossovers: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "ihsan_compliance_rate": self.ihsan_compliance_rate,
            "elite_count": self.elite_count,
        }


@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine."""
    population_size: int = POPULATION_SIZE
    max_generations: int = GENERATION_LIMIT
    elite_ratio: float = ELITE_RATIO
    mutation_rate: float = MUTATION_RATE
    crossover_rate: float = CROSSOVER_RATE
    tournament_size: int = 5
    convergence_threshold: float = 0.001  # Fitness change below this = converged
    stall_generations: int = 10  # Generations without improvement before stall
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD


@dataclass
class EvolutionResult:
    """Result of an evolution run."""
    status: EvolutionStatus
    generations_completed: int
    best_genome: Optional[AgentGenome]
    best_fitness: float
    final_population: List[AgentGenome]
    generation_history: List[GenerationStats]
    total_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "generations": self.generations_completed,
            "best_fitness": self.best_fitness,
            "best_genome_id": self.best_genome.id if self.best_genome else None,
            "population_size": len(self.final_population),
            "history_length": len(self.generation_history),
            "total_time": self.total_time_seconds,
        }


class EvolutionEngine:
    """
    Genetic algorithm engine for evolving agent populations.

    Implements Ihsān-constrained evolution where all agents must
    maintain constitutional compliance to survive.

    Usage:
        engine = EvolutionEngine()

        # Run evolution
        result = await engine.evolve()

        # Get best agent
        best = result.best_genome
        print(f"Best fitness: {result.best_fitness}")
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        initial_population: Optional[List[AgentGenome]] = None,
    ):
        self.config = config or EvolutionConfig()
        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()
        self.selector = SelectionPressure(
            tournament_size=self.config.tournament_size,
            ihsan_threshold=self.config.ihsan_threshold,
        )

        # Population
        self.population = initial_population or []
        self.fitness_results: Dict[str, FitnessResult] = {}

        # State
        self.generation = 0
        self.status = EvolutionStatus.INITIALIZING
        self.history: List[GenerationStats] = []
        self._running = False
        self._best_fitness_ever = 0.0
        self._stall_counter = 0

        # Callbacks
        self._on_generation: Optional[Callable[[GenerationStats], None]] = None

    def initialize(self, size: Optional[int] = None):
        """Initialize the population."""
        size = size or self.config.population_size

        if not self.population:
            self.population = GenomeFactory.create_population(size)

        self.generation = 0
        self.status = EvolutionStatus.INITIALIZING
        self.history = []
        self._best_fitness_ever = 0.0
        self._stall_counter = 0

    async def evolve(
        self,
        max_generations: Optional[int] = None,
        callback: Optional[Callable[[GenerationStats], None]] = None,
    ) -> EvolutionResult:
        """Run the evolution process."""
        start_time = datetime.now(timezone.utc)
        max_gen = max_generations or self.config.max_generations
        self._on_generation = callback

        # Initialize if needed
        if not self.population:
            self.initialize()

        self.status = EvolutionStatus.RUNNING
        self._running = True

        best_genome = None
        best_fitness = 0.0

        while self._running and self.generation < max_gen:
            # Run one generation
            stats = await self._run_generation()

            # Track best
            if stats.best_fitness > best_fitness:
                best_fitness = stats.best_fitness
                best_genome = self._get_best_genome()

            # Check for convergence
            if self._check_convergence(stats):
                self.status = EvolutionStatus.CONVERGED
                break

            # Check for stall
            if self._stall_counter >= self.config.stall_generations:
                self.status = EvolutionStatus.STALLED
                break

            # Callback
            if self._on_generation:
                self._on_generation(stats)

            self.generation += 1

        if self._running:
            self.status = EvolutionStatus.COMPLETED

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return EvolutionResult(
            status=self.status,
            generations_completed=self.generation,
            best_genome=best_genome,
            best_fitness=best_fitness,
            final_population=self.population.copy(),
            generation_history=self.history.copy(),
            total_time_seconds=elapsed,
        )

    async def _run_generation(self) -> GenerationStats:
        """Run a single generation."""
        context = EvaluationContext(
            peer_genomes=self.population,
            generation=self.generation,
        )

        # Evaluate fitness
        results = await self.fitness_evaluator.rank_population(
            self.population, context
        )
        self.fitness_results = {r.genome_id: r for r in results}

        # Calculate statistics
        stats = self._calculate_stats(results)
        self.history.append(stats)

        # Check improvement
        if stats.best_fitness > self._best_fitness_ever + self.config.convergence_threshold:
            self._best_fitness_ever = stats.best_fitness
            self._stall_counter = 0
        else:
            self._stall_counter += 1

        # Selection and reproduction
        new_population = await self._reproduce(results)
        self.population = new_population

        return stats

    async def _reproduce(self, results: List[FitnessResult]) -> List[AgentGenome]:
        """Create next generation through selection and reproduction."""
        new_population: List[AgentGenome] = []

        # Elitism: keep top performers
        n_elite = max(1, int(self.config.population_size * self.config.elite_ratio))
        elite = self.selector.elitist_select(results, n_elite)

        for elite_result in elite:
            genome = self._get_genome_by_id(elite_result.genome_id)
            if genome:
                new_population.append(genome)

        # Fill rest with offspring
        mutation_count = 0
        crossover_count = 0

        while len(new_population) < self.config.population_size:
            # Tournament selection for parents
            parents = self.selector.tournament_select(results, 2)

            if len(parents) < 2:
                # Not enough parents, create random
                child = GenomeFactory.create_random()
            else:
                parent1 = self._get_genome_by_id(parents[0].genome_id)
                parent2 = self._get_genome_by_id(parents[1].genome_id)

                if not parent1 or not parent2:
                    child = GenomeFactory.create_random()
                elif random.random() < self.config.crossover_rate:
                    # Crossover
                    child1, child2 = parent1.crossover(parent2)
                    child = child1
                    crossover_count += 1
                else:
                    # Clone parent
                    child = parent1.mutate(self.config.mutation_rate)
                    if child.mutations_applied:
                        mutation_count += 1

            # Apply mutation
            if random.random() < self.config.mutation_rate:
                child = child.mutate(self.config.mutation_rate)
                mutation_count += 1

            # Ensure Ihsān compliance (repair if needed)
            child = self._ensure_ihsan_compliance(child)

            new_population.append(child)

        return new_population[:self.config.population_size]

    def _ensure_ihsan_compliance(self, genome: AgentGenome) -> AgentGenome:
        """Repair genome to ensure Ihsān compliance."""
        if genome.is_ihsan_compliant():
            return genome

        # Repair constitution genes
        ihsan_gene = genome.get_gene("ihsan_threshold")
        if ihsan_gene and ihsan_gene.value < UNIFIED_IHSAN_THRESHOLD:
            ihsan_gene.value = UNIFIED_IHSAN_THRESHOLD

        snr_gene = genome.get_gene("snr_threshold")
        if snr_gene and snr_gene.value < UNIFIED_SNR_THRESHOLD:
            snr_gene.value = UNIFIED_SNR_THRESHOLD

        fate_gene = genome.get_gene("fate_compliance")
        if fate_gene:
            fate_gene.value = True

        return genome

    def _calculate_stats(self, results: List[FitnessResult]) -> GenerationStats:
        """Calculate generation statistics."""
        fitnesses = [r.overall_fitness for r in results]
        compliant = [r for r in results if r.ihsan_compliant]

        import statistics

        return GenerationStats(
            generation=self.generation,
            population_size=len(results),
            best_fitness=max(fitnesses) if fitnesses else 0.0,
            avg_fitness=statistics.mean(fitnesses) if fitnesses else 0.0,
            min_fitness=min(fitnesses) if fitnesses else 0.0,
            fitness_std=statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            ihsan_compliance_rate=len(compliant) / len(results) if results else 0.0,
            elite_count=sum(1 for r in results if r.is_elite),
            mutations=0,  # Tracked in reproduce
            crossovers=0,
        )

    def _check_convergence(self, stats: GenerationStats) -> bool:
        """Check if evolution has converged."""
        if len(self.history) < 5:
            return False

        # Check if fitness variance is very low
        if stats.fitness_std < self.config.convergence_threshold:
            return True

        # Check if best fitness hasn't improved
        recent = self.history[-5:]
        fitness_changes = [
            abs(recent[i].best_fitness - recent[i - 1].best_fitness)
            for i in range(1, len(recent))
        ]

        return all(c < self.config.convergence_threshold for c in fitness_changes)

    def _get_genome_by_id(self, genome_id: str) -> Optional[AgentGenome]:
        """Get genome by ID from population."""
        for genome in self.population:
            if genome.id == genome_id:
                return genome
        return None

    def _get_best_genome(self) -> Optional[AgentGenome]:
        """Get the best genome in current population."""
        if not self.fitness_results:
            return self.population[0] if self.population else None

        best_result = max(
            self.fitness_results.values(),
            key=lambda r: r.overall_fitness
        )
        return self._get_genome_by_id(best_result.genome_id)

    def stop(self):
        """Stop the evolution process."""
        self._running = False

    def pause(self):
        """Pause evolution (can resume)."""
        self.status = EvolutionStatus.PAUSED
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get current evolution statistics."""
        return {
            "status": self.status.value,
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self._best_fitness_ever,
            "stall_counter": self._stall_counter,
            "history_length": len(self.history),
        }


class CoevolutionEngine:
    """
    Coevolution engine for multiple interacting populations.

    Enables different agent types to coevolve, creating emergent
    collaborative behaviors.
    """

    def __init__(
        self,
        populations: Dict[str, List[AgentGenome]],
        config: Optional[EvolutionConfig] = None,
    ):
        self.config = config or EvolutionConfig()
        self.engines: Dict[str, EvolutionEngine] = {}

        for name, population in populations.items():
            self.engines[name] = EvolutionEngine(
                config=self.config,
                initial_population=population,
            )

        self.generation = 0
        self.interaction_history: List[Dict[str, Any]] = []

    async def coevolve(
        self,
        max_generations: int = 50,
        interaction_fn: Optional[Callable[[Dict[str, AgentGenome]], Dict[str, float]]] = None,
    ) -> Dict[str, EvolutionResult]:
        """
        Run coevolution across all populations.

        Args:
            max_generations: Maximum generations to run
            interaction_fn: Function to evaluate inter-population interactions

        Returns:
            Dict of evolution results per population
        """
        results: Dict[str, EvolutionResult] = {}

        for gen in range(max_generations):
            self.generation = gen

            # Run one generation for each population
            for name, engine in self.engines.items():
                await engine._run_generation()

            # Evaluate interactions between populations
            if interaction_fn:
                representatives = {
                    name: engine._get_best_genome()
                    for name, engine in self.engines.items()
                }
                interaction_scores = interaction_fn(representatives)

                # Apply interaction bonuses to fitness
                for name, bonus in interaction_scores.items():
                    if name in self.engines:
                        best = self.engines[name]._get_best_genome()
                        if best:
                            best.fitness += bonus

                self.interaction_history.append({
                    "generation": gen,
                    "scores": interaction_scores,
                })

            # Increment generation counters
            for engine in self.engines.values():
                engine.generation += 1

        # Compile results
        for name, engine in self.engines.items():
            best = engine._get_best_genome()
            results[name] = EvolutionResult(
                status=engine.status,
                generations_completed=engine.generation,
                best_genome=best,
                best_fitness=best.fitness if best else 0.0,
                final_population=engine.population,
                generation_history=engine.history,
                total_time_seconds=0.0,
            )

        return results
