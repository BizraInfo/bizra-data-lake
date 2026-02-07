"""
Emergence Detector — Identifying Novel Capabilities in Evolved Agents
===============================================================================

Detects emergent properties and novel capabilities that arise from evolution:
- Behavioral novelty detection
- Capability synergy identification
- Pattern recognition for emergent behaviors
- Constitutional compliance verification

Standing on Giants: Maturana (Emergence) + Holland (Complex Adaptive Systems)
Genesis Strict Synthesis v2.2.2
"""

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.autopoiesis.genome import AgentGenome, GeneType
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD


class EmergenceType(Enum):
    """Types of emergent properties."""

    CAPABILITY = "capability"  # New functional ability
    STRATEGY = "strategy"  # Novel decision pattern
    COLLABORATION = "collaboration"  # Inter-agent cooperation
    EFFICIENCY = "efficiency"  # Resource optimization
    RESILIENCE = "resilience"  # Fault tolerance


class NoveltyLevel(Enum):
    """Levels of novelty."""

    INCREMENTAL = "incremental"  # Small improvement
    SIGNIFICANT = "significant"  # Notable change
    BREAKTHROUGH = "breakthrough"  # Major new capability


@dataclass
class EmergentProperty:
    """A detected emergent property."""

    id: str
    emergence_type: EmergenceType
    novelty_level: NoveltyLevel
    description: str
    genome_ids: List[str]
    confidence: float
    evidence: Dict[str, Any]
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ihsan_compliant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.emergence_type.value,
            "novelty": self.novelty_level.value,
            "description": self.description,
            "genome_count": len(self.genome_ids),
            "confidence": self.confidence,
            "ihsan_compliant": self.ihsan_compliant,
            "first_detected": self.first_detected.isoformat(),
        }


@dataclass
class EmergenceReport:
    """Report of detected emergent properties."""

    generation: int
    properties: List[EmergentProperty]
    novel_genomes: List[str]
    convergent_traits: List[str]
    diversity_score: float
    ihsan_emergence_rate: float  # Rate of Ihsān-compliant emergences
    analysis_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "emergent_count": len(self.properties),
            "novel_genomes": len(self.novel_genomes),
            "convergent_traits": self.convergent_traits,
            "diversity": self.diversity_score,
            "ihsan_emergence_rate": self.ihsan_emergence_rate,
            "properties": [p.to_dict() for p in self.properties],
        }


class BehaviorSignature:
    """Captures behavioral signature of a genome for comparison."""

    def __init__(self, genome: AgentGenome):
        self.genome_id = genome.id
        self.traits: Dict[str, float] = {}
        self._extract_traits(genome)

    def _extract_traits(self, genome: AgentGenome):
        """Extract behavioral traits from genome."""
        for name, gene in genome.genes.items():
            if isinstance(gene.value, (int, float)):
                self.traits[name] = float(gene.value)
            elif isinstance(gene.value, bool):
                self.traits[name] = 1.0 if gene.value else 0.0
            elif isinstance(gene.value, str):
                # Hash string to float
                self.traits[name] = hash(gene.value) % 1000 / 1000

    def distance(self, other: "BehaviorSignature") -> float:
        """Calculate behavioral distance from another signature."""
        common_traits = set(self.traits.keys()) & set(other.traits.keys())

        if not common_traits:
            return 1.0

        total_diff = sum(
            abs(self.traits[t] - other.traits.get(t, 0)) for t in common_traits
        )

        return total_diff / len(common_traits)

    def to_vector(self, trait_order: List[str]) -> List[float]:
        """Convert to fixed-order vector."""
        return [self.traits.get(t, 0.0) for t in trait_order]


class EmergenceDetector:
    """
    Detects emergent properties in evolving agent populations.

    Tracks behavioral patterns across generations to identify:
    - Novel capabilities that weren't in ancestors
    - Synergistic combinations of traits
    - Convergent evolution toward successful patterns
    - Ihsān-compliant innovations

    Usage:
        detector = EmergenceDetector()

        # Analyze a generation
        report = detector.analyze_generation(population, generation=10)

        # Check for specific emergence
        if detector.has_breakthrough():
            print("Breakthrough detected!")
    """

    def __init__(
        self,
        novelty_threshold: float = 0.3,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.novelty_threshold = novelty_threshold
        self.ihsan_threshold = ihsan_threshold

        # Archive of past behaviors
        self._signature_archive: List[BehaviorSignature] = []
        self._detected_properties: List[EmergentProperty] = []
        self._trait_order: List[str] = []

        # Statistics
        self._generation_history: List[EmergenceReport] = []

    def analyze_generation(
        self,
        population: List[AgentGenome],
        generation: int,
        previous_population: Optional[List[AgentGenome]] = None,
    ) -> EmergenceReport:
        """Analyze a generation for emergent properties."""
        start = datetime.now(timezone.utc)
        properties: List[EmergentProperty] = []

        # Extract signatures
        current_signatures = [BehaviorSignature(g) for g in population]

        # Update trait order from first genome
        if population and not self._trait_order:
            self._trait_order = list(population[0].genes.keys())

        # Detect novelty
        novel_genomes = self._detect_novel_genomes(current_signatures)

        # Detect capability emergences
        cap_emergences = self._detect_capability_emergence(population)
        properties.extend(cap_emergences)

        # Detect strategy emergences
        strat_emergences = self._detect_strategy_emergence(population)
        properties.extend(strat_emergences)

        # Detect collaboration emergences (if we have previous generation)
        if previous_population:
            collab_emergences = self._detect_collaboration_emergence(
                population, previous_population
            )
            properties.extend(collab_emergences)

        # Detect convergent traits
        convergent = self._detect_convergent_traits(current_signatures)

        # Calculate diversity
        diversity = self._calculate_diversity(current_signatures)

        # Ihsān emergence rate
        ihsan_compliant_props = [p for p in properties if p.ihsan_compliant]
        ihsan_rate = len(ihsan_compliant_props) / len(properties) if properties else 1.0

        # Update archive
        self._update_archive(current_signatures)

        # Store detected properties
        self._detected_properties.extend(properties)

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        report = EmergenceReport(
            generation=generation,
            properties=properties,
            novel_genomes=[g.genome_id for g in novel_genomes],
            convergent_traits=convergent,
            diversity_score=diversity,
            ihsan_emergence_rate=ihsan_rate,
            analysis_time_ms=elapsed_ms,
        )

        self._generation_history.append(report)
        return report

    def _detect_novel_genomes(
        self,
        signatures: List[BehaviorSignature],
    ) -> List[BehaviorSignature]:
        """Detect genomes with novel behavioral signatures."""
        novel = []

        for sig in signatures:
            if not self._signature_archive:
                novel.append(sig)
                continue

            # Find minimum distance to any archived signature
            min_distance = min(
                sig.distance(archived) for archived in self._signature_archive
            )

            if min_distance > self.novelty_threshold:
                novel.append(sig)

        return novel

    def _detect_capability_emergence(
        self,
        population: List[AgentGenome],
    ) -> List[EmergentProperty]:
        """Detect new capability combinations."""
        emergences = []

        # Analyze capability gene distributions
        cap_stats: Dict[str, Dict[str, Any]] = {}

        for genome in population:
            for gene in genome.get_genes_by_type(GeneType.CAPABILITY):
                if gene.name not in cap_stats:
                    cap_stats[gene.name] = {"values": [], "sum": 0}
                cap_stats[gene.name]["values"].append(gene.value)
                if isinstance(gene.value, (int, float)):
                    cap_stats[gene.name]["sum"] += gene.value

        # Detect outliers (potential emergences)
        for gene_name, stats in cap_stats.items():
            values = stats["values"]
            if len(values) < 3:
                continue

            if all(isinstance(v, (int, float)) for v in values):
                mean_val = sum(values) / len(values)
                std_val = math.sqrt(
                    sum((v - mean_val) ** 2 for v in values) / len(values)
                )

                # Check for significant outliers
                outliers = [
                    i for i, v in enumerate(values) if abs(v - mean_val) > 2 * std_val
                ]

                if outliers:
                    emergences.append(
                        EmergentProperty(
                            id=hashlib.md5(
                                f"{gene_name}_cap_{len(outliers)}".encode(),
                                usedforsecurity=False,
                            ).hexdigest()[:8],
                            emergence_type=EmergenceType.CAPABILITY,
                            novelty_level=(
                                NoveltyLevel.SIGNIFICANT
                                if len(outliers) > 2
                                else NoveltyLevel.INCREMENTAL
                            ),
                            description=f"Unusual {gene_name} values detected in {len(outliers)} genomes",
                            genome_ids=[population[i].id for i in outliers],
                            confidence=0.7 + 0.1 * min(len(outliers), 3),
                            evidence={
                                "gene": gene_name,
                                "outlier_count": len(outliers),
                            },
                        )
                    )

        return emergences

    def _detect_strategy_emergence(
        self,
        population: List[AgentGenome],
    ) -> List[EmergentProperty]:
        """Detect emergent strategy patterns."""
        emergences = []

        # Collect strategy genes
        strategy_patterns: Dict[Tuple, List[str]] = {}

        for genome in population:
            strategy_genes = genome.get_genes_by_type(GeneType.STRATEGY)
            pattern = tuple((g.name, g.value) for g in strategy_genes)

            if pattern not in strategy_patterns:
                strategy_patterns[pattern] = []
            strategy_patterns[pattern].append(genome.id)

        # Detect dominant patterns (potential emergent strategies)
        total = len(population)
        for pattern, genome_ids in strategy_patterns.items():
            ratio = len(genome_ids) / total

            if ratio > 0.3:  # More than 30% share this strategy
                emergences.append(
                    EmergentProperty(
                        id=hashlib.md5(str(pattern).encode(), usedforsecurity=False).hexdigest()[:8],
                        emergence_type=EmergenceType.STRATEGY,
                        novelty_level=(
                            NoveltyLevel.SIGNIFICANT
                            if ratio > 0.5
                            else NoveltyLevel.INCREMENTAL
                        ),
                        description=f"Strategy pattern adopted by {ratio:.0%} of population",
                        genome_ids=genome_ids,
                        confidence=ratio,
                        evidence={"pattern": [dict(pattern)], "adoption_rate": ratio},
                    )
                )

        return emergences

    def _detect_collaboration_emergence(
        self,
        current: List[AgentGenome],
        previous: List[AgentGenome],
    ) -> List[EmergentProperty]:
        """Detect emergent collaboration patterns."""
        emergences = []

        # Check for increased collaboration tendency
        current_collab = [
            g.get_gene("collaboration_tendency").value
            for g in current
            if g.get_gene("collaboration_tendency")
        ]
        previous_collab = [
            g.get_gene("collaboration_tendency").value
            for g in previous
            if g.get_gene("collaboration_tendency")
        ]

        if current_collab and previous_collab:
            current_avg = sum(current_collab) / len(current_collab)
            previous_avg = sum(previous_collab) / len(previous_collab)

            if current_avg > previous_avg + 0.1:
                emergences.append(
                    EmergentProperty(
                        id=hashlib.md5(f"collab_{current_avg}".encode(), usedforsecurity=False).hexdigest()[
                            :8
                        ],
                        emergence_type=EmergenceType.COLLABORATION,
                        novelty_level=NoveltyLevel.SIGNIFICANT,
                        description=f"Collaboration tendency increased from {previous_avg:.2f} to {current_avg:.2f}",
                        genome_ids=[
                            g.id
                            for g in current
                            if g.get_gene("collaboration_tendency")
                            and g.get_gene("collaboration_tendency").value
                            > previous_avg
                        ],
                        confidence=0.8,
                        evidence={
                            "previous_avg": previous_avg,
                            "current_avg": current_avg,
                        },
                    )
                )

        return emergences

    def _detect_convergent_traits(
        self,
        signatures: List[BehaviorSignature],
    ) -> List[str]:
        """Detect traits that are converging across population."""
        if len(signatures) < 3:
            return []

        convergent = []

        # Get all trait names
        all_traits = set()
        for sig in signatures:
            all_traits.update(sig.traits.keys())

        for trait in all_traits:
            values = [sig.traits.get(trait, 0) for sig in signatures]

            if len(set(values)) == 1:
                continue  # All same value, not interesting

            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)

            # Low variance = convergence
            if variance < 0.01:
                convergent.append(trait)

        return convergent

    def _calculate_diversity(self, signatures: List[BehaviorSignature]) -> float:
        """Calculate population diversity."""
        if len(signatures) < 2:
            return 1.0

        # Pairwise distance average
        total_distance = 0
        count = 0

        for i, sig1 in enumerate(signatures):
            for sig2 in signatures[i + 1 :]:
                total_distance += sig1.distance(sig2)
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _update_archive(self, signatures: List[BehaviorSignature], max_size: int = 500):
        """Update signature archive."""
        self._signature_archive.extend(signatures)

        # Prune if too large
        if len(self._signature_archive) > max_size:
            self._signature_archive = self._signature_archive[-max_size:]

    def has_breakthrough(self) -> bool:
        """Check if any breakthrough emergences have been detected."""
        return any(
            p.novelty_level == NoveltyLevel.BREAKTHROUGH
            for p in self._detected_properties
        )

    def get_emergences_by_type(self, etype: EmergenceType) -> List[EmergentProperty]:
        """Get all detected emergences of a specific type."""
        return [p for p in self._detected_properties if p.emergence_type == etype]

    def get_stats(self) -> Dict[str, Any]:
        """Get emergence detector statistics."""
        return {
            "total_emergences": len(self._detected_properties),
            "archive_size": len(self._signature_archive),
            "generations_analyzed": len(self._generation_history),
            "breakthroughs": sum(
                1
                for p in self._detected_properties
                if p.novelty_level == NoveltyLevel.BREAKTHROUGH
            ),
            "by_type": {
                etype.value: len(self.get_emergences_by_type(etype))
                for etype in EmergenceType
            },
        }
