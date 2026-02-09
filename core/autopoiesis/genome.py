"""
Agent Genome — Genetic Representation of Agent Behavior
===============================================================================

Encodes agent configuration and behavior as an evolvable genome:
- Capability genes (what the agent can do)
- Strategy genes (how the agent decides)
- Constitution genes (ethical constraints — immutable)
- Efficiency genes (resource usage patterns)

Standing on Giants: Holland (GA) + Maturana (Autopoiesis)
Genesis Strict Synthesis v2.2.2
"""

import copy
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.autopoiesis import MUTATION_RATE
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


class GeneType(Enum):
    """Types of genes in the agent genome."""

    CAPABILITY = "capability"  # What agent can do
    STRATEGY = "strategy"  # Decision-making approach
    CONSTITUTION = "constitution"  # Ethical constraints (immutable)
    EFFICIENCY = "efficiency"  # Resource optimization
    COMMUNICATION = "communication"  # Inter-agent protocols


class MutationType(Enum):
    """Types of mutations."""

    POINT = "point"  # Single value change
    INSERTION = "insertion"  # Add new capability
    DELETION = "deletion"  # Remove capability
    SWAP = "swap"  # Exchange two genes
    INVERSION = "inversion"  # Reverse gene order


@dataclass
class Gene:
    """A single gene in the genome."""

    name: str
    gene_type: GeneType
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    immutable: bool = False  # Constitution genes are immutable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mutate(self, rate: float = MUTATION_RATE) -> "Gene":
        """Return a mutated copy of this gene."""
        if self.immutable or random.random() > rate:
            return copy.deepcopy(self)

        new_gene = copy.deepcopy(self)

        if isinstance(self.value, float):
            # Gaussian mutation for floats
            delta = random.gauss(0, 0.1)
            new_value = self.value + delta
            if self.min_value is not None:
                new_value = max(self.min_value, new_value)
            if self.max_value is not None:
                new_value = min(self.max_value, new_value)
            new_gene.value = new_value

        elif isinstance(self.value, int):
            # Integer mutation
            delta = random.randint(-1, 1)
            new_value = self.value + delta
            if self.min_value is not None:
                new_value = max(int(self.min_value), new_value)
            if self.max_value is not None:
                new_value = min(int(self.max_value), new_value)
            new_gene.value = new_value

        elif isinstance(self.value, bool):
            # Flip boolean
            new_gene.value = not self.value

        elif isinstance(self.value, str) and self.metadata.get("choices"):
            # Select from choices
            choices = self.metadata["choices"]
            new_gene.value = random.choice(choices)

        return new_gene

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.gene_type.value,
            "value": self.value,
            "immutable": self.immutable,
        }


@dataclass
class AgentGenome:
    """
    Complete genetic representation of an agent.

    The genome encodes all evolvable aspects of agent behavior,
    while constitutional constraints remain immutable.
    """

    id: str = ""
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    genes: Dict[str, Gene] = field(default_factory=dict)
    fitness: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mutations_applied: List[MutationType] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.genes:
            self._initialize_default_genes()

    def _generate_id(self) -> str:
        """Generate unique genome ID (CSPRNG)."""
        import secrets
        return secrets.token_hex(6)  # 12 hex chars, unpredictable

    def _initialize_default_genes(self):
        """Initialize with default gene set."""
        # Constitution genes (IMMUTABLE - Ihsān constraints)
        self.genes["ihsan_threshold"] = Gene(
            name="ihsan_threshold",
            gene_type=GeneType.CONSTITUTION,
            value=UNIFIED_IHSAN_THRESHOLD,
            min_value=0.95,  # Never below 0.95
            max_value=1.0,
            immutable=True,  # Cannot be mutated
        )
        self.genes["snr_threshold"] = Gene(
            name="snr_threshold",
            gene_type=GeneType.CONSTITUTION,
            value=UNIFIED_SNR_THRESHOLD,
            min_value=0.85,
            max_value=1.0,
            immutable=True,
        )
        self.genes["fate_compliance"] = Gene(
            name="fate_compliance",
            gene_type=GeneType.CONSTITUTION,
            value=True,
            immutable=True,
        )

        # Capability genes (evolvable)
        self.genes["reasoning_depth"] = Gene(
            name="reasoning_depth",
            gene_type=GeneType.CAPABILITY,
            value=3,
            min_value=1,
            max_value=10,
        )
        self.genes["context_window"] = Gene(
            name="context_window",
            gene_type=GeneType.CAPABILITY,
            value=4096,
            min_value=512,
            max_value=32768,
        )
        self.genes["tool_proficiency"] = Gene(
            name="tool_proficiency",
            gene_type=GeneType.CAPABILITY,
            value=0.7,
            min_value=0.0,
            max_value=1.0,
        )

        # Strategy genes (evolvable)
        self.genes["exploration_rate"] = Gene(
            name="exploration_rate",
            gene_type=GeneType.STRATEGY,
            value=0.2,
            min_value=0.0,
            max_value=0.5,
        )
        self.genes["risk_tolerance"] = Gene(
            name="risk_tolerance",
            gene_type=GeneType.STRATEGY,
            value=0.3,
            min_value=0.0,
            max_value=0.5,
        )
        self.genes["decision_strategy"] = Gene(
            name="decision_strategy",
            gene_type=GeneType.STRATEGY,
            value="balanced",
            metadata={"choices": ["cautious", "balanced", "aggressive", "adaptive"]},
        )

        # Efficiency genes (evolvable)
        self.genes["batch_size"] = Gene(
            name="batch_size",
            gene_type=GeneType.EFFICIENCY,
            value=8,
            min_value=1,
            max_value=64,
        )
        self.genes["cache_strategy"] = Gene(
            name="cache_strategy",
            gene_type=GeneType.EFFICIENCY,
            value="lru",
            metadata={"choices": ["none", "lru", "lfu", "adaptive"]},
        )
        self.genes["parallel_tasks"] = Gene(
            name="parallel_tasks",
            gene_type=GeneType.EFFICIENCY,
            value=4,
            min_value=1,
            max_value=16,
        )

        # Communication genes (evolvable)
        self.genes["verbosity"] = Gene(
            name="verbosity",
            gene_type=GeneType.COMMUNICATION,
            value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        self.genes["collaboration_tendency"] = Gene(
            name="collaboration_tendency",
            gene_type=GeneType.COMMUNICATION,
            value=0.7,
            min_value=0.0,
            max_value=1.0,
        )

    def get_gene(self, name: str) -> Optional[Gene]:
        """Get a gene by name."""
        return self.genes.get(name)

    def get_genes_by_type(self, gene_type: GeneType) -> List[Gene]:
        """Get all genes of a specific type."""
        return [g for g in self.genes.values() if g.gene_type == gene_type]

    def mutate(self, rate: float = MUTATION_RATE) -> "AgentGenome":
        """Create a mutated copy of this genome."""
        new_genome = AgentGenome(
            generation=self.generation + 1,
            parent_ids=[self.id],
        )

        mutations_applied = []

        # Mutate each gene
        for name, gene in self.genes.items():
            original_value = gene.value
            new_gene = gene.mutate(rate)
            new_genome.genes[name] = new_gene

            if new_gene.value != original_value:
                mutations_applied.append(MutationType.POINT)

        # Possibility of structural mutations (lower probability)
        if random.random() < rate * 0.1:
            # Could add/remove optional genes here
            mutations_applied.append(MutationType.SWAP)

        new_genome.mutations_applied = mutations_applied
        return new_genome

    def crossover(self, other: "AgentGenome") -> Tuple["AgentGenome", "AgentGenome"]:
        """Perform crossover with another genome."""
        child1 = AgentGenome(
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.id, other.id],
        )
        child2 = AgentGenome(
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.id, other.id],
        )

        # Uniform crossover
        for name in self.genes:
            if name in other.genes:
                if random.random() < 0.5:
                    child1.genes[name] = copy.deepcopy(self.genes[name])
                    child2.genes[name] = copy.deepcopy(other.genes[name])
                else:
                    child1.genes[name] = copy.deepcopy(other.genes[name])
                    child2.genes[name] = copy.deepcopy(self.genes[name])
            else:
                # Gene only in self
                child1.genes[name] = copy.deepcopy(self.genes[name])

        # Include genes only in other
        for name in other.genes:
            if name not in self.genes:
                child2.genes[name] = copy.deepcopy(other.genes[name])

        return child1, child2

    def distance(self, other: "AgentGenome") -> float:
        """Calculate genetic distance from another genome."""
        if not other:
            return 1.0

        total_diff = 0.0
        gene_count = 0

        for name, gene in self.genes.items():
            if name in other.genes:
                other_gene = other.genes[name]
                if isinstance(gene.value, (int, float)) and isinstance(
                    other_gene.value, (int, float)
                ):
                    # Normalized difference
                    max_val = max(abs(gene.value), abs(other_gene.value), 1.0)
                    total_diff += abs(gene.value - other_gene.value) / max_val
                elif gene.value != other_gene.value:
                    total_diff += 1.0
                gene_count += 1

        return total_diff / gene_count if gene_count > 0 else 1.0

    def is_ihsan_compliant(self) -> bool:
        """Check if genome maintains Ihsān constraints."""
        ihsan_gene = self.get_gene("ihsan_threshold")
        snr_gene = self.get_gene("snr_threshold")
        fate_gene = self.get_gene("fate_compliance")

        return (
            ihsan_gene
            and ihsan_gene.value >= UNIFIED_IHSAN_THRESHOLD
            and snr_gene
            and snr_gene.value >= UNIFIED_SNR_THRESHOLD
            and fate_gene
            and fate_gene.value is True
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "genes": {name: gene.to_dict() for name, gene in self.genes.items()},
            "fitness": self.fitness,
            "created_at": self.created_at.isoformat(),
            "ihsan_compliant": self.is_ihsan_compliant(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGenome":
        """Reconstruct genome from dictionary."""
        genome = cls(
            id=data.get("id", ""),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            fitness=data.get("fitness", 0.0),
        )

        for name, gene_data in data.get("genes", {}).items():
            genome.genes[name] = Gene(
                name=gene_data["name"],
                gene_type=GeneType(gene_data["type"]),
                value=gene_data["value"],
                immutable=gene_data.get("immutable", False),
            )

        return genome


class GenomeFactory:
    """Factory for creating agent genomes."""

    @staticmethod
    def create_random() -> AgentGenome:
        """Create a genome with random (but valid) genes."""
        genome = AgentGenome()

        # Randomize evolvable genes
        for name, gene in genome.genes.items():
            if not gene.immutable:
                genome.genes[name] = gene.mutate(rate=1.0)  # Force mutation

        return genome

    @staticmethod
    def create_specialist(specialty: str) -> AgentGenome:
        """Create a genome specialized for a task type."""
        genome = AgentGenome()

        if specialty == "reasoning":
            genome.genes["reasoning_depth"].value = 8
            genome.genes["exploration_rate"].value = 0.3
            genome.genes["risk_tolerance"].value = 0.2

        elif specialty == "efficiency":
            genome.genes["batch_size"].value = 32
            genome.genes["parallel_tasks"].value = 12
            genome.genes["cache_strategy"].value = "adaptive"

        elif specialty == "collaboration":
            genome.genes["collaboration_tendency"].value = 0.9
            genome.genes["verbosity"].value = 0.7
            genome.genes["risk_tolerance"].value = 0.1

        return genome

    @staticmethod
    def create_population(size: int) -> List[AgentGenome]:
        """Create a diverse initial population."""
        population = []

        # Mix of specialists and random
        specialists = ["reasoning", "efficiency", "collaboration"]
        for i in range(size):
            if i < len(specialists):
                genome = GenomeFactory.create_specialist(specialists[i])
            else:
                genome = GenomeFactory.create_random()
            population.append(genome)

        return population
