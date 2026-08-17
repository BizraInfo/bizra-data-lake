"""
Autopoietic Loop — Self-Evolving Agent Ecosystem Controller
===============================================================================

The central orchestrator for BIZRA's self-evolving agent ecosystem:
1. OBSERVE: Monitor agent population performance
2. EVOLVE: Apply genetic algorithms for improvement
3. EMERGE: Detect novel capabilities
4. INTEGRATE: Merge successful traits into production agents
5. REFLECT: Learn from evolution history

This loop runs continuously, ensuring agents improve while
maintaining strict Ihsān (excellence) constraints.

Standing on Giants: Maturana & Varela + Holland + Shannon + Anthropic
Genesis Strict Synthesis v2.2.2
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from core.autopoiesis import GENERATION_LIMIT, POPULATION_SIZE
from core.autopoiesis.emergence import EmergenceDetector, EmergenceReport
from core.autopoiesis.evolution import EvolutionConfig, EvolutionEngine, EvolutionResult
from core.autopoiesis.fitness import EvaluationContext, FitnessEvaluator
from core.autopoiesis.genome import AgentGenome, GenomeFactory
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


class AutopoiesisPhase(Enum):
    """Phases of the autopoietic loop."""

    IDLE = "idle"
    OBSERVING = "observing"
    EVOLVING = "evolving"
    DETECTING = "detecting"
    INTEGRATING = "integrating"
    REFLECTING = "reflecting"
    EMERGENCY = "emergency"


@dataclass
class AutopoiesisConfig:
    """Configuration for the autopoietic loop."""

    population_size: int = POPULATION_SIZE
    evolution_generations: int = 20  # Generations per evolution cycle
    max_evolution_cycles: int = GENERATION_LIMIT
    cycle_interval_seconds: float = 60.0  # Time between cycles
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    integration_threshold: float = 0.9  # Minimum fitness for integration
    emergency_diversity_threshold: float = 0.1  # Below this = emergency


@dataclass
class AutopoiesisState:
    """State of the autopoietic system."""

    phase: AutopoiesisPhase = AutopoiesisPhase.IDLE
    evolution_cycle: int = 0
    total_generations: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 1.0
    ihsan_compliance_rate: float = 1.0
    recent_receipt_count: int = 0
    recent_receipt_success_rate: float = 1.0
    recent_receipt_ihsan: float = 0.0
    recent_receipt_snr: float = 0.0
    recent_receipt_latency_ms: float = 0.0
    recent_receipt_trend: str = "unknown"
    emergences_detected: int = 0
    integrations_performed: int = 0
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "cycle": self.evolution_cycle,
            "generations": self.total_generations,
            "population": self.population_size,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "diversity": self.diversity,
            "ihsan_rate": self.ihsan_compliance_rate,
            "receipt_count": self.recent_receipt_count,
            "receipt_success_rate": self.recent_receipt_success_rate,
            "receipt_ihsan": self.recent_receipt_ihsan,
            "receipt_snr": self.recent_receipt_snr,
            "receipt_latency_ms": self.recent_receipt_latency_ms,
            "receipt_trend": self.recent_receipt_trend,
            "emergences": self.emergences_detected,
            "integrations": self.integrations_performed,
        }


@dataclass
class IntegrationCandidate:
    """A genome candidate for production integration."""

    genome: AgentGenome
    fitness: float
    novelty_score: float
    ihsan_score: float
    recommendation: str
    approved: bool = False


class AutopoieticLoop:
    """
    The self-evolving agent ecosystem controller.

    Orchestrates continuous evolution of agent populations,
    detecting emergent capabilities and integrating successful
    traits into production systems.

    Usage:
        loop = AutopoieticLoop()

        # Start the loop
        await loop.start()

        # Check status
        print(loop.get_status())

        # Get best evolved agent
        best = loop.get_best_genome()
        if best:
            print(f"Best fitness: {best.fitness}")

        # Stop
        await loop.stop()
    """

    def __init__(
        self,
        config: Optional[AutopoiesisConfig] = None,
        on_emergence: Optional[Callable[[EmergenceReport], None]] = None,
        on_integration: Optional[Callable[[IntegrationCandidate], Any]] = None,
        authorize_integration: Optional[Callable[[IntegrationCandidate], bool]] = None,
    ):
        self.config = config or AutopoiesisConfig()
        self.on_emergence = on_emergence
        # Historical compatibility hook: observation / learning intake only.
        # Its return value is intentionally non-authoritative.
        self.on_integration = on_integration
        # Production promotion is a separate authority surface and fails closed
        # when no authority provider is configured.
        self.authorize_integration = authorize_integration

        # Components
        self.evolution_engine = EvolutionEngine(
            config=EvolutionConfig(
                population_size=self.config.population_size,
                max_generations=self.config.evolution_generations,
                ihsan_threshold=self.config.ihsan_threshold,
            )
        )
        self.emergence_detector = EmergenceDetector(
            ihsan_threshold=self.config.ihsan_threshold
        )
        self.fitness_evaluator = FitnessEvaluator()

        # State
        self.state = AutopoiesisState()
        self._running = False
        self._paused = False

        # Population history
        self._population_history: List[List[AgentGenome]] = []
        self._integration_history: List[IntegrationCandidate] = []
        self._production_agents: Dict[str, AgentGenome] = {}
        self._receipt_history: Deque[Dict[str, Any]] = deque(maxlen=128)

    def record_receipt_observation(
        self,
        receipt: Dict[str, Any],
        *,
        receipt_kind: str = "mission",
    ) -> None:
        """Record a canonical mission/heartbeat receipt for the OBSERVE phase.

        The autopoietic loop already monitors its internal population.
        This method extends that observation plane with production-truth
        receipts emitted by the canonical runtime path.
        """
        ihsan = self._coerce_float(
            receipt.get("ihsan_score", receipt.get("ihsan_composite", 0.0))
        )
        snr = self._coerce_float(receipt.get("snr_score", ihsan))
        latency_ms = self._coerce_float(receipt.get("duration_ms", 0.0))

        if receipt_kind == "heartbeat":
            success = (
                self._coerce_bool(receipt.get("gini_ok", True))
                and ihsan >= self.config.ihsan_threshold
            )
        else:
            success = (
                self._coerce_bool(receipt.get("gate_passed", True))
                and str(receipt.get("fate_verdict", "approved")) != "rejected"
                and ihsan >= self.config.ihsan_threshold
            )

        normalized = {
            "kind": receipt_kind,
            "timestamp": datetime.now(timezone.utc),
            "ihsan_score": ihsan,
            "snr_score": snr,
            "latency_ms": latency_ms,
            "success": success,
            "mission_id": str(receipt.get("mission_id", "")),
            "tick_number": int(receipt.get("tick_number", 0) or 0),
            "approved_count": int(receipt.get("approved_count", 0) or 0),
            "rejected_count": int(receipt.get("rejected_count", 0) or 0),
            "missions_processed": int(receipt.get("missions_processed", 0) or 0),
            "reflexes_precipitated": int(receipt.get("reflexes_precipitated", 0) or 0),
        }
        self._receipt_history.append(normalized)

    async def start(self):
        """Start the autopoietic loop."""
        logger.info("Starting autopoietic loop")
        self._running = True

        # Initialize population if needed
        if not self.evolution_engine.population:
            self.evolution_engine.initialize()

        while self._running:
            if self._paused:
                await asyncio.sleep(1)
                continue

            try:
                await self._run_cycle()
            except (
                asyncio.CancelledError,
                RuntimeError,
                OSError,
            ) as e:  # SEC-003 — async boundary
                logger.error(f"Autopoiesis cycle error: {e}")
                self.state.phase = AutopoiesisPhase.EMERGENCY

            await asyncio.sleep(self.config.cycle_interval_seconds)

        logger.info("Autopoietic loop stopped")

    async def stop(self):
        """Stop the autopoietic loop."""
        self._running = False

    def pause(self):
        """Pause the loop."""
        self._paused = True
        self.state.phase = AutopoiesisPhase.IDLE

    def resume(self):
        """Resume the loop."""
        self._paused = False

    async def _run_cycle(self):
        """Run one complete autopoietic cycle."""
        self.state.evolution_cycle += 1
        self.state.last_update = datetime.now(timezone.utc)

        # Phase 1: OBSERVE
        await self._phase_observe()

        # Phase 2: EVOLVE
        evolution_result = await self._phase_evolve()

        # Phase 3: DETECT emergence
        emergence_report = await self._phase_detect()

        # Phase 4: INTEGRATE successful agents
        await self._phase_integrate(evolution_result, emergence_report)

        # Phase 5: REFLECT and adapt
        await self._phase_reflect()

    async def _phase_observe(self):
        """Observe current population state."""
        self.state.phase = AutopoiesisPhase.OBSERVING

        population = self.evolution_engine.population
        self.state.population_size = len(population)

        results = []
        if population:
            # Evaluate current fitness
            context = EvaluationContext(peer_genomes=population)
            results = await self.fitness_evaluator.rank_population(population, context)

        if results:
            self.state.best_fitness = results[0].overall_fitness
            self.state.avg_fitness = sum(r.overall_fitness for r in results) / len(
                results
            )
            self.state.ihsan_compliance_rate = sum(
                1 for r in results if r.ihsan_compliant
            ) / len(results)

        receipt_summary = self._summarize_receipt_window()
        self.state.recent_receipt_count = receipt_summary["count"]
        self.state.recent_receipt_success_rate = receipt_summary["success_rate"]
        self.state.recent_receipt_ihsan = receipt_summary["avg_ihsan"]
        self.state.recent_receipt_snr = receipt_summary["avg_snr"]
        self.state.recent_receipt_latency_ms = receipt_summary["avg_latency_ms"]
        self.state.recent_receipt_trend = receipt_summary["trend"]

        if receipt_summary["count"] > 0 and not results:
            # When the internal population is still cold, let production-truth
            # receipts seed the observation state instead of reporting zeros.
            self.state.best_fitness = max(
                self.state.best_fitness,
                receipt_summary["avg_ihsan"],
            )
            self.state.avg_fitness = receipt_summary["avg_snr"]
            self.state.ihsan_compliance_rate = receipt_summary["success_rate"]

        logger.debug(
            f"Observed: pop={self.state.population_size}, "
            f"best={self.state.best_fitness:.3f}, "
            f"ihsan_rate={self.state.ihsan_compliance_rate:.2%}, "
            f"receipt_count={receipt_summary['count']}"
        )

    async def _phase_evolve(self) -> EvolutionResult:
        """Run evolution for configured generations."""
        self.state.phase = AutopoiesisPhase.EVOLVING

        result = await self.evolution_engine.evolve(
            max_generations=self.config.evolution_generations
        )

        self.state.total_generations += result.generations_completed

        # Save population snapshot
        self._population_history.append([genome for genome in result.final_population])
        if len(self._population_history) > 10:
            self._population_history = self._population_history[-10:]

        logger.info(
            f"Evolved {result.generations_completed} generations, "
            f"best fitness: {result.best_fitness:.3f}"
        )

        return result

    async def _phase_detect(self) -> EmergenceReport:
        """Detect emergent properties in evolved population."""
        self.state.phase = AutopoiesisPhase.DETECTING

        current_pop = self.evolution_engine.population
        previous_pop = (
            self._population_history[-2] if len(self._population_history) >= 2 else None
        )

        report = self.emergence_detector.analyze_generation(
            population=current_pop,
            generation=self.state.total_generations,
            previous_population=previous_pop,
        )

        self.state.emergences_detected += len(report.properties)
        self.state.diversity = report.diversity_score

        # Check for emergency (diversity collapse)
        if report.diversity_score < self.config.emergency_diversity_threshold:
            logger.warning(f"Diversity collapse detected: {report.diversity_score:.3f}")
            self.state.phase = AutopoiesisPhase.EMERGENCY
            await self._handle_emergency()

        # Callback for emergence
        if self.on_emergence and report.properties:
            self.on_emergence(report)

        logger.debug(
            f"Detected {len(report.properties)} emergences, "
            f"diversity: {report.diversity_score:.3f}"
        )

        return report

    async def _phase_integrate(
        self,
        evolution_result: EvolutionResult,
        emergence_report: EmergenceReport,
    ):
        """Integrate successful agents into production."""
        self.state.phase = AutopoiesisPhase.INTEGRATING

        candidates: List[IntegrationCandidate] = []

        # Find candidates meeting integration threshold. Fitness and Ihsan establish
        # eligibility for review; they never manufacture authority to promote.
        for genome in evolution_result.final_population:
            if genome.fitness >= self.config.integration_threshold:
                # Calculate novelty from emergence data
                is_novel = genome.id in emergence_report.novel_genomes
                novelty_score = 0.8 if is_novel else 0.3

                candidate = IntegrationCandidate(
                    genome=genome,
                    fitness=genome.fitness,
                    novelty_score=novelty_score,
                    ihsan_score=(
                        genome.get_gene("ihsan_threshold").value  # type: ignore[union-attr]
                        if genome.get_gene("ihsan_threshold")
                        else 0.95
                    ),
                    recommendation=(
                        "Eligible for integration review"
                        if genome.is_ihsan_compliant()
                        else "Review required"
                    ),
                    approved=False,
                )
                candidates.append(candidate)

        # Learning/notification and production authority are intentionally separate.
        # An observer may ingest a candidate, but its return value never authorizes
        # a production state transition.
        for candidate in candidates:
            if self.on_integration:
                self.on_integration(candidate)

            if self.authorize_integration:
                candidate.approved = self.authorize_integration(candidate) is True

            if candidate.approved:
                self._production_agents[candidate.genome.id] = candidate.genome
                self._integration_history.append(candidate)
                self.state.integrations_performed += 1

                logger.info(
                    f"Integrated genome {candidate.genome.id} "
                    f"with fitness {candidate.fitness:.3f}"
                )

    async def _phase_reflect(self):
        """Reflect on evolution progress and adapt parameters."""
        self.state.phase = AutopoiesisPhase.REFLECTING

        # Analyze trends
        if len(self._population_history) >= 3:
            recent_diversity = [
                self._calculate_diversity(pop) for pop in self._population_history[-3:]
            ]

            # If diversity is declining, increase mutation rate
            if all(d < recent_diversity[0] for d in recent_diversity[1:]):
                current_rate = self.evolution_engine.config.mutation_rate
                new_rate = min(0.3, current_rate * 1.2)
                self.evolution_engine.config.mutation_rate = new_rate
                logger.info(
                    f"Increased mutation rate to {new_rate:.2f} due to declining diversity"
                )

            # If diversity is very high but fitness not improving, decrease mutation
            elif (
                self.state.diversity > 0.8 and self.evolution_engine._stall_counter > 5
            ):
                current_rate = self.evolution_engine.config.mutation_rate
                new_rate = max(0.05, current_rate * 0.8)
                self.evolution_engine.config.mutation_rate = new_rate
                logger.info(
                    f"Decreased mutation rate to {new_rate:.2f} for exploitation"
                )

        self._adapt_to_external_receipt_signal()

        # Update state
        self.state.phase = AutopoiesisPhase.IDLE

    async def _handle_emergency(self):
        """Handle emergency situations like diversity collapse."""
        logger.warning("Emergency: Injecting diversity")

        # Inject random genomes
        new_randoms = GenomeFactory.create_population(self.config.population_size // 4)
        self.evolution_engine.population.extend(new_randoms)

        # Trim to population size
        self.evolution_engine.population = self.evolution_engine.population[
            : self.config.population_size
        ]

        # Reset stall counter
        self.evolution_engine._stall_counter = 0

    def _calculate_diversity(self, population: List[AgentGenome]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 1.0

        total_distance = 0
        count = 0

        for i, g1 in enumerate(population):
            for g2 in population[i + 1 :]:
                total_distance += g1.distance(g2)  # type: ignore[assignment]
                count += 1

        return total_distance / count if count > 0 else 0.0

    def get_best_genome(self) -> Optional[AgentGenome]:
        """Get the best genome from current population."""
        if not self.evolution_engine.population:
            return None

        return max(self.evolution_engine.population, key=lambda g: g.fitness)

    def get_production_agents(self) -> List[AgentGenome]:
        """Get all integrated production agents."""
        return list(self._production_agents.values())

    def get_status(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "state": self.state.to_dict(),
            "evolution": self.evolution_engine.get_stats(),
            "emergence": self.emergence_detector.get_stats(),
            "receipt_observations": self._summarize_receipt_window(),
            "production_agents": len(self._production_agents),
            "running": self._running,
            "paused": self._paused,
        }

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no"}
        return bool(value)

    def _summarize_receipt_window(self) -> Dict[str, Any]:
        """Summarize recent runtime receipts into one observation window."""
        if not self._receipt_history:
            return {
                "count": 0,
                "mission_count": 0,
                "heartbeat_count": 0,
                "avg_ihsan": 0.0,
                "avg_snr": 0.0,
                "avg_latency_ms": 0.0,
                "success_rate": 1.0,
                "trend": "unknown",
            }

        window = list(self._receipt_history)[-16:]
        mission_count = sum(1 for item in window if item["kind"] == "mission")
        heartbeat_count = len(window) - mission_count
        avg_ihsan = sum(item["ihsan_score"] for item in window) / len(window)
        avg_snr = sum(item["snr_score"] for item in window) / len(window)
        avg_latency = sum(item["latency_ms"] for item in window) / len(window)
        success_rate = sum(1 for item in window if item["success"]) / len(window)

        trend = "stable"
        if len(window) >= 4:
            midpoint = len(window) // 2
            older = window[:midpoint]
            newer = window[midpoint:]
            older_ihsan = sum(item["ihsan_score"] for item in older) / len(older)
            newer_ihsan = sum(item["ihsan_score"] for item in newer) / len(newer)
            delta = newer_ihsan - older_ihsan
            if delta > 0.01:
                trend = "improving"
            elif delta < -0.01:
                trend = "degrading"

        return {
            "count": len(window),
            "mission_count": mission_count,
            "heartbeat_count": heartbeat_count,
            "avg_ihsan": round(avg_ihsan, 4),
            "avg_snr": round(avg_snr, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": round(success_rate, 4),
            "trend": trend,
        }

    def _adapt_to_external_receipt_signal(self) -> None:
        """Use recent production receipts to bias exploration/exploitation."""
        summary = self._summarize_receipt_window()
        if summary["count"] < 3:
            return

        current_rate = self.evolution_engine.config.mutation_rate
        new_rate = current_rate

        if (
            summary["trend"] == "degrading"
            or summary["success_rate"] < 0.85
            or summary["avg_ihsan"] < self.config.ihsan_threshold
        ):
            new_rate = min(0.3, current_rate * 1.1)
            reason = "receipt_degradation"
        elif (
            summary["trend"] == "improving"
            and summary["avg_ihsan"] >= self.config.ihsan_threshold
            and summary["avg_snr"] >= self.config.snr_threshold
        ):
            new_rate = max(0.05, current_rate * 0.95)
            reason = "receipt_stability"
        else:
            return

        if abs(new_rate - current_rate) > 1e-9:
            self.evolution_engine.config.mutation_rate = new_rate
            logger.info(
                "Adjusted mutation rate to %.3f from receipt signal (%s, count=%d)",
                new_rate,
                reason,
                summary["count"],
            )


# Factory function
def create_autopoietic_loop(
    population_size: int = POPULATION_SIZE,
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
) -> AutopoieticLoop:
    """Create a configured autopoietic loop."""
    config = AutopoiesisConfig(
        population_size=population_size,
        ihsan_threshold=ihsan_threshold,
    )
    return AutopoieticLoop(config=config)


# Exports
__all__ = [
    "AutopoiesisPhase",
    "AutopoiesisConfig",
    "AutopoiesisState",
    "IntegrationCandidate",
    "AutopoieticLoop",
    "create_autopoietic_loop",
]
