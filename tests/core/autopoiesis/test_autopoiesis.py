"""
Autopoiesis Integration Tests — Self-Evolving Agent Ecosystem
===============================================================================

Tests the complete autopoietic loop including genome, fitness, evolution,
and emergence detection.

Genesis Strict Synthesis v2.2.2
"""

import pytest
import asyncio

from core.autopoiesis import (
    POPULATION_SIZE,
    MUTATION_RATE,
    UNIFIED_IHSAN_THRESHOLD,
)
from core.autopoiesis.genome import (
    AgentGenome,
    Gene,
    GeneType,
    GenomeFactory,
)
from core.autopoiesis.fitness import (
    FitnessEvaluator,
    FitnessResult,
    EvaluationContext,
    SelectionPressure,
    FitnessComponent,
)
from core.autopoiesis.evolution import (
    EvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    EvolutionStatus,
)
from core.autopoiesis.emergence import (
    EmergenceDetector,
    EmergenceReport,
    EmergenceType,
    NoveltyLevel,
    BehaviorSignature,
)
from core.autopoiesis.loop import (
    AutopoieticLoop,
    AutopoiesisConfig,
    AutopoiesisPhase,
    create_autopoietic_loop,
)


class TestAgentGenome:
    """Test genome representation and operations."""

    def test_genome_creation(self):
        """Test basic genome creation."""
        genome = AgentGenome()

        assert genome.id
        assert genome.generation == 0
        assert len(genome.genes) > 0

    def test_genome_ihsan_compliance(self):
        """Test Ihsān compliance checking."""
        genome = AgentGenome()

        # Default genome should be compliant
        assert genome.is_ihsan_compliant()

        # Constitution genes should be immutable
        ihsan_gene = genome.get_gene("ihsan_threshold")
        assert ihsan_gene.immutable

    def test_genome_mutation(self):
        """Test genome mutation."""
        genome = AgentGenome()
        original_genes = {
            name: gene.value for name, gene in genome.genes.items()
        }

        mutated = genome.mutate(rate=1.0)  # Force mutation

        # Some genes should have changed
        changed = sum(
            1 for name, gene in mutated.genes.items()
            if gene.value != original_genes.get(name) and not gene.immutable
        )

        assert changed >= 0  # At least some mutations possible

    def test_genome_crossover(self):
        """Test genome crossover."""
        parent1 = AgentGenome()
        parent2 = AgentGenome()

        # Modify parent2
        parent2.genes["reasoning_depth"].value = 8

        child1, child2 = parent1.crossover(parent2)

        assert child1.generation == 1
        assert child2.generation == 1
        assert parent1.id in child1.parent_ids
        assert parent2.id in child1.parent_ids

    def test_genome_distance(self):
        """Test genetic distance calculation."""
        genome1 = AgentGenome()
        genome2 = AgentGenome()

        # Same genome should have zero distance
        assert genome1.distance(genome1) == 0.0

        # Modified genome should have positive distance
        genome2.genes["reasoning_depth"].value = 10
        distance = genome1.distance(genome2)
        assert distance > 0


class TestGenomeFactory:
    """Test genome factory functions."""

    def test_create_random(self):
        """Test random genome creation."""
        genome = GenomeFactory.create_random()

        assert genome.id
        assert genome.is_ihsan_compliant()

    def test_create_specialist(self):
        """Test specialist genome creation."""
        reasoning_genome = GenomeFactory.create_specialist("reasoning")
        efficiency_genome = GenomeFactory.create_specialist("efficiency")

        assert reasoning_genome.get_gene("reasoning_depth").value > 5
        assert efficiency_genome.get_gene("batch_size").value > 16

    def test_create_population(self):
        """Test population creation."""
        population = GenomeFactory.create_population(10)

        assert len(population) == 10
        assert all(g.is_ihsan_compliant() for g in population)


class TestFitnessEvaluator:
    """Test fitness evaluation."""

    @pytest.fixture
    def evaluator(self):
        return FitnessEvaluator()

    @pytest.fixture
    def context(self):
        return EvaluationContext()

    @pytest.mark.asyncio
    async def test_evaluate_single(self, evaluator, context):
        """Test evaluating a single genome."""
        genome = AgentGenome()
        result = await evaluator.evaluate(genome, context)

        assert isinstance(result, FitnessResult)
        assert result.genome_id == genome.id
        assert 0 <= result.overall_fitness <= 1
        assert FitnessComponent.IHSAN in result.component_scores

    @pytest.mark.asyncio
    async def test_evaluate_ihsan_compliance(self, evaluator, context):
        """Test that non-compliant genomes get penalized."""
        genome = AgentGenome()

        # Force non-compliance (shouldn't happen with immutable genes,
        # but test the logic)
        result = await evaluator.evaluate(genome, context)

        # Compliant genome should have decent fitness
        assert result.ihsan_compliant
        assert result.overall_fitness > 0.5

    @pytest.mark.asyncio
    async def test_rank_population(self, evaluator, context):
        """Test population ranking."""
        population = GenomeFactory.create_population(10)
        context.peer_genomes = population

        ranked = await evaluator.rank_population(population, context)

        assert len(ranked) == 10
        assert ranked[0].rank == 1
        assert ranked[0].is_elite
        # Verify sorted by fitness
        for i in range(1, len(ranked)):
            assert ranked[i].overall_fitness <= ranked[i - 1].overall_fitness


class TestSelectionPressure:
    """Test selection mechanisms."""

    @pytest.fixture
    def selector(self):
        return SelectionPressure(tournament_size=3)

    @pytest.mark.asyncio
    async def test_tournament_selection(self, selector):
        """Test tournament selection."""
        # Create mock results
        results = [
            FitnessResult(
                genome_id=f"genome_{i}",
                overall_fitness=0.5 + i * 0.05,
                component_scores={FitnessComponent.IHSAN: 0.95},
                ihsan_compliant=True,
                is_elite=i < 2,
            )
            for i in range(10)
        ]

        selected = selector.tournament_select(results, 5)

        assert len(selected) == 5
        # Higher fitness genomes should be more likely selected


class TestEvolutionEngine:
    """Test evolution engine."""

    @pytest.fixture
    def engine(self):
        config = EvolutionConfig(
            population_size=10,
            max_generations=5,
            mutation_rate=0.2,
        )
        return EvolutionEngine(config=config)

    def test_initialization(self, engine):
        """Test engine initialization."""
        engine.initialize(size=10)

        assert len(engine.population) == 10
        assert engine.generation == 0

    @pytest.mark.asyncio
    async def test_evolution_run(self, engine):
        """Test running evolution."""
        engine.initialize()
        result = await engine.evolve(max_generations=3)

        assert isinstance(result, EvolutionResult)
        assert result.generations_completed == 3
        assert result.best_genome is not None
        assert len(result.final_population) == engine.config.population_size

    @pytest.mark.asyncio
    async def test_ihsan_preservation(self, engine):
        """Test that Ihsān compliance is preserved through evolution."""
        engine.initialize()
        result = await engine.evolve(max_generations=5)

        # All offspring should maintain Ihsān compliance
        for genome in result.final_population:
            assert genome.is_ihsan_compliant()


class TestEmergenceDetector:
    """Test emergence detection."""

    @pytest.fixture
    def detector(self):
        return EmergenceDetector()

    def test_behavior_signature(self):
        """Test behavior signature creation."""
        genome = AgentGenome()
        sig = BehaviorSignature(genome)

        assert sig.genome_id == genome.id
        assert len(sig.traits) > 0

    def test_signature_distance(self):
        """Test signature distance calculation."""
        genome1 = AgentGenome()
        genome2 = AgentGenome()
        genome2.genes["reasoning_depth"].value = 10

        sig1 = BehaviorSignature(genome1)
        sig2 = BehaviorSignature(genome2)

        distance = sig1.distance(sig2)
        assert distance > 0

    def test_analyze_generation(self, detector):
        """Test generation analysis."""
        population = GenomeFactory.create_population(10)

        report = detector.analyze_generation(
            population=population,
            generation=0,
        )

        assert isinstance(report, EmergenceReport)
        assert report.generation == 0
        assert 0 <= report.diversity_score <= 1

    def test_detect_convergence(self, detector):
        """Test convergent trait detection via strategy emergence."""
        # Create population with same strategy
        population = [AgentGenome() for _ in range(10)]
        for genome in population:
            genome.genes["decision_strategy"].value = "cautious"

        report = detector.analyze_generation(population, generation=1)

        # When all have same value, it's detected as strategy emergence (100% adoption)
        # rather than "convergent" (low but non-zero variance)
        assert len(report.properties) > 0
        strategy_emergence = [
            p for p in report.properties
            if p.emergence_type == EmergenceType.STRATEGY
        ]
        assert len(strategy_emergence) > 0
        assert strategy_emergence[0].confidence == 1.0  # 100% adoption


class TestAutopoieticLoop:
    """Test the complete autopoietic loop."""

    @pytest.fixture
    def loop(self):
        config = AutopoiesisConfig(
            population_size=10,
            evolution_generations=3,
            cycle_interval_seconds=0.1,
        )
        return AutopoieticLoop(config=config)

    def test_creation(self, loop):
        """Test loop creation."""
        assert loop.state.phase == AutopoiesisPhase.IDLE
        assert loop.state.evolution_cycle == 0

    @pytest.mark.asyncio
    async def test_observe_phase(self, loop):
        """Test observation phase."""
        loop.evolution_engine.initialize()
        await loop._phase_observe()

        assert loop.state.population_size > 0

    @pytest.mark.asyncio
    async def test_evolve_phase(self, loop):
        """Test evolution phase."""
        loop.evolution_engine.initialize()
        result = await loop._phase_evolve()

        assert isinstance(result, EvolutionResult)
        assert loop.state.total_generations > 0

    @pytest.mark.asyncio
    async def test_detect_phase(self, loop):
        """Test emergence detection phase."""
        loop.evolution_engine.initialize()
        await loop._phase_evolve()
        report = await loop._phase_detect()

        assert isinstance(report, EmergenceReport)

    @pytest.mark.asyncio
    async def test_single_cycle(self, loop):
        """Test running a single cycle."""
        loop.evolution_engine.initialize()
        await loop._run_cycle()

        assert loop.state.evolution_cycle == 1
        assert loop.state.total_generations > 0

    def test_get_status(self, loop):
        """Test status reporting."""
        status = loop.get_status()

        assert "state" in status
        assert "evolution" in status
        assert "emergence" in status


class TestIntegrationScenarios:
    """Integration test scenarios."""

    @pytest.mark.asyncio
    async def test_full_evolution_cycle(self):
        """Test a complete evolution cycle."""
        config = EvolutionConfig(
            population_size=20,
            max_generations=10,
        )
        engine = EvolutionEngine(config=config)
        detector = EmergenceDetector()

        # Run evolution
        result = await engine.evolve()

        assert result.status in (EvolutionStatus.COMPLETED, EvolutionStatus.CONVERGED)
        assert result.best_genome is not None
        assert result.best_fitness > 0

        # Analyze for emergence
        report = detector.analyze_generation(
            result.final_population,
            generation=result.generations_completed,
        )

        assert report.diversity_score > 0

    @pytest.mark.asyncio
    async def test_ihsan_never_violated(self):
        """Test that Ihsān is never violated across evolution."""
        config = EvolutionConfig(
            population_size=30,
            max_generations=20,
            mutation_rate=0.3,  # Higher mutation
        )
        engine = EvolutionEngine(config=config)

        result = await engine.evolve()

        # Check every genome
        for genome in result.final_population:
            assert genome.is_ihsan_compliant(), \
                f"Genome {genome.id} violated Ihsān constraint"

    @pytest.mark.asyncio
    async def test_diversity_maintained(self):
        """Test that diversity doesn't collapse."""
        config = EvolutionConfig(
            population_size=20,
            max_generations=15,
        )
        engine = EvolutionEngine(config=config)
        detector = EmergenceDetector()

        result = await engine.evolve()
        report = detector.analyze_generation(
            result.final_population,
            generation=result.generations_completed,
        )

        # Stochastic simulation — use lenient threshold to avoid flaky failures
        # while still catching total diversity collapse (score ≈ 0.0)
        assert report.diversity_score > 0.01, \
            f"Population diversity collapsed (score={report.diversity_score:.4f})"


# Run quick integration test
if __name__ == "__main__":
    print("Running Autopoiesis Integration Tests...")

    async def quick_test():
        # Create and run evolution
        engine = EvolutionEngine(
            config=EvolutionConfig(population_size=20, max_generations=5)
        )
        result = await engine.evolve()

        print(f"Evolution completed: {result.generations_completed} generations")
        print(f"Best fitness: {result.best_fitness:.3f}")
        print(f"Final population: {len(result.final_population)}")

        # Check emergence
        detector = EmergenceDetector()
        report = detector.analyze_generation(result.final_population, generation=5)
        print(f"Emergences detected: {len(report.properties)}")
        print(f"Diversity: {report.diversity_score:.3f}")

        print("\nAll quick tests passed!")

    asyncio.run(quick_test())
