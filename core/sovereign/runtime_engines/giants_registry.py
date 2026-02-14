"""
Standing on the Shoulders of Giants — Knowledge Attribution Protocol
═══════════════════════════════════════════════════════════════════════════════

"If I have seen further, it is by standing on the shoulders of giants."
    — Isaac Newton (1675), after Bernard of Chartres (1159)

This module implements the BIZRA Giants Registry — a formal protocol for
attributing foundational knowledge that powers the sovereign system.

Every significant algorithm, pattern, and principle in BIZRA traces back
to seminal work by pioneers. This registry ensures proper attribution
and enables the system to explain its reasoning foundations.

Registry Categories:
1. INFORMATION_THEORY — Shannon, Kolmogorov, Huffman
2. DISTRIBUTED_SYSTEMS — Lamport, Brewer, Verma
3. SOCIAL_NETWORKS — Granovetter, Dunbar, Barabási
4. DECISION_THEORY — Boyd, Kahneman, Nash
5. MACHINE_LEARNING — Vaswani, Shazeer, Anthropic
6. ECONOMICS — Markowitz, Black, Fama
7. PHILOSOPHY — Al-Ghazali, Aristotle, Kant

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        GIANTS REGISTRY                                   │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │  Giant → { name, year, work, contribution, citation, applications }│ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                │                                        │
    │                                ▼                                        │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │  Application → { module, method, giants_used, explanation }        │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘

Created: 2026-02-04 | BIZRA Sovereign Runtime v1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GiantCategory(str, Enum):
    """Categories of foundational knowledge."""

    INFORMATION_THEORY = "information_theory"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    SOCIAL_NETWORKS = "social_networks"
    DECISION_THEORY = "decision_theory"
    MACHINE_LEARNING = "machine_learning"
    ECONOMICS = "economics"
    PHILOSOPHY = "philosophy"
    MATHEMATICS = "mathematics"
    OPERATIONS = "operations"
    COGNITIVE_SCIENCE = "cognitive_science"


@dataclass
class Giant:
    """
    A foundational contributor to human knowledge.

    Each Giant represents a seminal contribution that BIZRA builds upon.
    """

    name: str
    year: int
    work: str
    contribution: str
    category: GiantCategory
    citation: str
    key_insight: str
    applications_in_bizra: list[str] = field(default_factory=list)
    related_giants: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.name, self.year))

    def __eq__(self, other):
        if not isinstance(other, Giant):
            return False
        return self.name == other.name and self.year == other.year

    def format_attribution(self) -> str:
        """Format attribution string."""
        return f"{self.name} ({self.year}): {self.work}"

    def format_full_citation(self) -> str:
        """Format full academic citation."""
        return self.citation


@dataclass
class GiantApplication:
    """
    Records where a Giant's work is applied in BIZRA.
    """

    module: str
    method: str
    giants: list[Giant]
    explanation: str
    performance_impact: Optional[str] = None


class GiantsRegistry:
    """
    The Standing on Giants Registry.

    Maintains a comprehensive registry of all foundational work
    that BIZRA builds upon, enabling:
    - Proper attribution in logs and documentation
    - Explainability of algorithm choices
    - Knowledge provenance tracking

    "We are like dwarfs sitting on the shoulders of giants. We see more,
    and things that are more distant, than they did, not because our sight
    is superior or because we are taller than they, but because they raise
    us up, and by their great stature add to ours."
        — Bernard of Chartres (c. 1159)
    """

    def __init__(self):
        self._giants: dict[str, Giant] = {}
        self._applications: list[GiantApplication] = []
        self._category_index: dict[GiantCategory, list[str]] = {
            cat: [] for cat in GiantCategory
        }

        # Initialize with foundational giants
        self._register_foundational_giants()

        logger.info(f"GiantsRegistry initialized with {len(self._giants)} giants")

    def _register_foundational_giants(self) -> None:
        """Register all foundational giants that BIZRA builds upon."""

        # ═══════════════════════════════════════════════════════════════════
        # INFORMATION THEORY
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="Claude Shannon",
                year=1948,
                work="A Mathematical Theory of Communication",
                contribution="Information theory, entropy, channel capacity, SNR",
                category=GiantCategory.INFORMATION_THEORY,
                citation="Shannon, C.E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27(3), 379-423.",
                key_insight="Information can be quantified. The fundamental limits of communication are mathematical, not physical.",
                applications_in_bizra=[
                    "SNR Maximizer (signal-to-noise optimization)",
                    "Entropy-based uncertainty quantification",
                    "Channel capacity for inference pipelines",
                    "Information-theoretic filtering thresholds",
                ],
                related_giants=["Kolmogorov", "Huffman"],
            )
        )

        self.register(
            Giant(
                name="Andrey Kolmogorov",
                year=1965,
                work="Three Approaches to the Quantitative Definition of Information",
                contribution="Algorithmic information theory, complexity",
                category=GiantCategory.INFORMATION_THEORY,
                citation="Kolmogorov, A.N. (1965). Three Approaches to the Quantitative Definition of Information. Problems of Information Transmission, 1(1), 1-7.",
                key_insight="The complexity of an object is the length of its shortest description.",
                applications_in_bizra=[
                    "Compression-based pattern detection",
                    "Complexity bounds for reasoning",
                ],
            )
        )

        # ═══════════════════════════════════════════════════════════════════
        # DISTRIBUTED SYSTEMS
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="Leslie Lamport",
                year=1982,
                work="The Byzantine Generals Problem",
                contribution="Byzantine fault tolerance, distributed consensus",
                category=GiantCategory.DISTRIBUTED_SYSTEMS,
                citation="Lamport, L., Shostak, R., & Pease, M. (1982). The Byzantine Generals Problem. ACM Transactions on Programming Languages and Systems, 4(3), 382-401.",
                key_insight="Consensus is possible with f < n/3 Byzantine failures. Safety and liveness are distinct properties.",
                applications_in_bizra=[
                    "PBFT consensus in bizra-federation",
                    "SAT validator Byzantine voting (3/5 threshold)",
                    "Guardian Council veto mechanism",
                ],
                related_giants=["Brewer", "Burns"],
            )
        )

        self.register(
            Giant(
                name="Eric Brewer",
                year=2000,
                work="CAP Theorem",
                contribution="Consistency-Availability-Partition tolerance tradeoff",
                category=GiantCategory.DISTRIBUTED_SYSTEMS,
                citation="Brewer, E.A. (2000). Towards Robust Distributed Systems. PODC Keynote.",
                key_insight="In a distributed system, you can have at most 2 of 3: Consistency, Availability, Partition tolerance.",
                applications_in_bizra=[
                    "Federation network design (AP with eventual C)",
                    "Swarm orchestration tradeoffs",
                ],
            )
        )

        self.register(
            Giant(
                name="Abhishek Verma",
                year=2015,
                work="Large-scale cluster management at Google with Borg",
                contribution="Borg cluster manager, large-scale orchestration",
                category=GiantCategory.OPERATIONS,
                citation="Verma, A., et al. (2015). Large-scale cluster management at Google with Borg. EuroSys '15.",
                key_insight="Declarative job specifications with automatic bin-packing achieves high utilization at scale.",
                applications_in_bizra=[
                    "SwarmOrchestrator scaling algorithms",
                    "Resource allocation patterns",
                ],
                related_giants=["Burns", "Hamilton"],
            )
        )

        self.register(
            Giant(
                name="Brendan Burns",
                year=2016,
                work="Design Patterns for Container-based Distributed Systems",
                contribution="Kubernetes design principles, sidecar pattern",
                category=GiantCategory.OPERATIONS,
                citation="Burns, B., & Oppenheimer, D. (2016). Design Patterns for Container-based Distributed Systems. HotCloud '16.",
                key_insight="Composable container patterns enable reusable distributed system building blocks.",
                applications_in_bizra=[
                    "HybridSwarmOrchestrator patterns",
                    "Self-healing loop design",
                ],
            )
        )

        self.register(
            Giant(
                name="James Hamilton",
                year=2007,
                work="On Designing and Deploying Internet-Scale Services",
                contribution="Operations at scale, availability engineering",
                category=GiantCategory.OPERATIONS,
                citation="Hamilton, J. (2007). On Designing and Deploying Internet-Scale Services. LISA '07.",
                key_insight="Design for failure. Automate everything. 99.99% availability requires different thinking than 99%.",
                applications_in_bizra=[
                    "99.9% availability target",
                    "Self-healing with exponential backoff",
                    "Health check patterns",
                ],
            )
        )

        # ═══════════════════════════════════════════════════════════════════
        # SOCIAL NETWORKS
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="Mark Granovetter",
                year=1973,
                work="The Strength of Weak Ties",
                contribution="Weak ties theory, information bridges",
                category=GiantCategory.SOCIAL_NETWORKS,
                citation="Granovetter, M.S. (1973). The Strength of Weak Ties. American Journal of Sociology, 78(6), 1360-1380.",
                key_insight="Weak ties (acquaintances) are more important than strong ties for accessing novel information.",
                applications_in_bizra=[
                    "SocialGraph diverse routing bonus",
                    "Collaboration discovery preferring weak ties",
                ],
                related_giants=["Dunbar", "Barabási"],
            )
        )

        self.register(
            Giant(
                name="Robin Dunbar",
                year=1992,
                work="Neocortex size as a constraint on group size in primates",
                contribution="Dunbar's number (150 stable relationships)",
                category=GiantCategory.COGNITIVE_SCIENCE,
                citation="Dunbar, R.I.M. (1992). Neocortex size as a constraint on group size in primates. Journal of Human Evolution, 22(6), 469-493.",
                key_insight="Cognitive limits constrain the number of stable social relationships to approximately 150.",
                applications_in_bizra=[
                    "SocialGraph relationship cap",
                    "Agent network scaling limits",
                ],
            )
        )

        self.register(
            Giant(
                name="Larry Page & Sergey Brin",
                year=1998,
                work="The PageRank Citation Ranking",
                contribution="PageRank algorithm for link analysis",
                category=GiantCategory.MATHEMATICS,
                citation="Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. Stanford InfoLab.",
                key_insight="Link structure encodes importance. Recursive definition: a page is important if important pages link to it.",
                applications_in_bizra=[
                    "Trust propagation in SocialGraph",
                    "Agent reputation scoring",
                ],
            )
        )

        self.register(
            Giant(
                name="Albert-László Barabási",
                year=2002,
                work="Linked: The New Science of Networks",
                contribution="Scale-free networks, preferential attachment",
                category=GiantCategory.SOCIAL_NETWORKS,
                citation="Barabási, A.-L. (2002). Linked: The New Science of Networks. Perseus Books.",
                key_insight="Real networks are scale-free with power-law degree distribution. Hubs emerge naturally.",
                applications_in_bizra=[
                    "Network topology analysis",
                    "Hub agent identification",
                ],
            )
        )

        # ═══════════════════════════════════════════════════════════════════
        # DECISION THEORY
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="John Boyd",
                year=1995,
                work="OODA Loop",
                contribution="Observe-Orient-Decide-Act decision cycle",
                category=GiantCategory.DECISION_THEORY,
                citation="Boyd, J.R. (1995). The Essence of Winning and Losing. Unpublished briefing.",
                key_insight="Decision speed matters. Faster OODA loops disrupt opponents. Orientation is the schwerpunkt.",
                applications_in_bizra=[
                    "Extended OODA loop (8 states)",
                    "Cycle-based proactive execution",
                ],
                related_giants=["Kahneman"],
            )
        )

        self.register(
            Giant(
                name="John Nash",
                year=1950,
                work="Equilibrium Points in N-Person Games",
                contribution="Nash equilibrium, game theory",
                category=GiantCategory.ECONOMICS,
                citation="Nash, J. (1950). Equilibrium Points in N-Person Games. PNAS, 36(1), 48-49.",
                key_insight="Non-cooperative games have equilibrium points where no player benefits from unilateral deviation.",
                applications_in_bizra=[
                    "Negotiation engine (Nash bargaining)",
                    "Multi-agent coordination",
                ],
            )
        )

        # ═══════════════════════════════════════════════════════════════════
        # MACHINE LEARNING
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="Ashish Vaswani",
                year=2017,
                work="Attention Is All You Need",
                contribution="Transformer architecture, self-attention",
                category=GiantCategory.MACHINE_LEARNING,
                citation="Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.",
                key_insight="Self-attention enables parallel sequence processing. Position encodings replace recurrence.",
                applications_in_bizra=[
                    "Inference gateway model selection",
                    "Attention-based context management",
                ],
                related_giants=["Shazeer"],
            )
        )

        self.register(
            Giant(
                name="Noam Shazeer",
                year=2017,
                work="Outrageously Large Neural Networks (Mixture of Experts)",
                contribution="Sparse MoE, conditional computation",
                category=GiantCategory.MACHINE_LEARNING,
                citation="Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.",
                key_insight="Conditional computation via learned gating enables scaling without proportional compute increase.",
                applications_in_bizra=[
                    "PAT agent routing (gated expert selection)",
                    "Multi-model inference tiering",
                ],
            )
        )

        self.register(
            Giant(
                name="Maciej Besta",
                year=2024,
                work="Graph of Thoughts",
                contribution="Non-linear multi-path reasoning with graphs",
                category=GiantCategory.MACHINE_LEARNING,
                citation="Besta, M., et al. (2024). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. AAAI 2024.",
                key_insight="Reasoning as graph construction. Aggregate, refine, prune thought nodes for better solutions.",
                applications_in_bizra=[
                    "ThoughtGraph in bizra-core",
                    "Multi-path goal exploration",
                    "Collaboration discovery GoT",
                ],
            )
        )

        self.register(
            Giant(
                name="Anthropic",
                year=2022,
                work="Constitutional AI",
                contribution="AI alignment via constitutional principles",
                category=GiantCategory.MACHINE_LEARNING,
                citation="Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. Anthropic.",
                key_insight="Self-critique against a constitution enables harmless helpfulness without human labeling.",
                applications_in_bizra=[
                    "Ihsan constraint (≥0.95 threshold)",
                    "Constitutional gate validation",
                    "Daughter Test compliance",
                ],
            )
        )

        # ═══════════════════════════════════════════════════════════════════
        # ECONOMICS
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="Harry Markowitz",
                year=1952,
                work="Portfolio Selection",
                contribution="Modern portfolio theory, risk-return optimization",
                category=GiantCategory.ECONOMICS,
                citation="Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.",
                key_insight="Diversification reduces risk. Optimal portfolios lie on the efficient frontier.",
                applications_in_bizra=[
                    "OpportunityEngine position sizing",
                    "Risk-adjusted signal evaluation",
                ],
                related_giants=["Black", "Fama", "Lo"],
            )
        )

        self.register(
            Giant(
                name="Andrew Lo",
                year=2004,
                work="The Adaptive Markets Hypothesis",
                contribution="Evolution-based market dynamics",
                category=GiantCategory.ECONOMICS,
                citation="Lo, A.W. (2004). The Adaptive Markets Hypothesis. Journal of Portfolio Management, 30(5), 15-29.",
                key_insight="Markets evolve. Efficiency varies with market ecology. Adapt or die.",
                applications_in_bizra=[
                    "MarketAnalyzer adaptive patterns",
                    "Dynamic signal thresholds",
                ],
            )
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHILOSOPHY
        # ═══════════════════════════════════════════════════════════════════

        self.register(
            Giant(
                name="Abu Hamid Al-Ghazali",
                year=1095,
                work="Ihya Ulum al-Din (Revival of Religious Sciences)",
                contribution="Muraqabah (vigilance), Ihsan (excellence)",
                category=GiantCategory.PHILOSOPHY,
                citation="Al-Ghazali (1095). Ihya Ulum al-Din. Various translations available.",
                key_insight="Muraqabah: constant awareness. Ihsan: excellence as if observed. Inner states matter.",
                applications_in_bizra=[
                    "MuraqabahEngine (24/7 vigilance)",
                    "Ihsan threshold (0.95 excellence)",
                    "Constitutional spirituality",
                ],
            )
        )

        self.register(
            Giant(
                name="Thomas Malone",
                year=2018,
                work="Superminds",
                contribution="Collective intelligence factor",
                category=GiantCategory.COGNITIVE_SCIENCE,
                citation="Malone, T.W. (2018). Superminds: The Surprising Power of People and Computers Thinking Together. Little, Brown.",
                key_insight="Group intelligence > sum of individual intelligences when communication and collaboration are optimized.",
                applications_in_bizra=[
                    "CollectiveIntelligence synthesis",
                    "Team synergy scoring",
                    "PAT + SAT emergent intelligence",
                ],
            )
        )

    def register(self, giant: Giant) -> None:
        """Register a giant in the registry."""
        key = f"{giant.name}:{giant.year}"
        self._giants[key] = giant
        self._category_index[giant.category].append(key)

    def get(self, name: str, year: Optional[int] = None) -> Optional[Giant]:
        """Get a giant by name (and optionally year)."""
        if year:
            key = f"{name}:{year}"
            return self._giants.get(key)

        # Search by name only
        for key, giant in self._giants.items():
            if giant.name == name:
                return giant
        return None

    def get_by_category(self, category: GiantCategory) -> list[Giant]:
        """Get all giants in a category."""
        keys = self._category_index.get(category, [])
        return [self._giants[k] for k in keys]

    def record_application(
        self,
        module: str,
        method: str,
        giant_names: list[str],
        explanation: str,
        performance_impact: Optional[str] = None,
    ) -> None:
        """Record an application of giants' work."""
        giants = [self.get(name) for name in giant_names]
        giants = [g for g in giants if g is not None]

        app = GiantApplication(
            module=module,
            method=method,
            giants=giants,
            explanation=explanation,
            performance_impact=performance_impact,
        )
        self._applications.append(app)

    def get_applications_for(self, giant_name: str) -> list[GiantApplication]:
        """Get all applications of a giant's work."""
        return [
            app
            for app in self._applications
            if any(g.name == giant_name for g in app.giants)
        ]

    def format_attribution_header(self, module: str) -> str:
        """Format an attribution header for a module."""
        apps = [a for a in self._applications if a.module == module]
        if not apps:
            return ""

        giants_used = set()
        for app in apps:
            for g in app.giants:
                giants_used.add(g)

        lines = ["Standing on the Shoulders of Giants:"]
        for g in sorted(giants_used, key=lambda x: x.year):
            lines.append(f"- {g.format_attribution()}")

        return "\n".join(lines)

    def explain_algorithm(self, method_name: str) -> str:
        """Explain the foundational basis of an algorithm."""
        apps = [a for a in self._applications if method_name in a.method]
        if not apps:
            return f"No recorded giants for {method_name}"

        app = apps[0]
        lines = [f"Algorithm: {method_name}"]
        lines.append(f"Explanation: {app.explanation}")
        lines.append("Based on:")
        for g in app.giants:
            lines.append(f"  - {g.format_attribution()}")
            lines.append(f"    Key insight: {g.key_insight}")

        return "\n".join(lines)

    def summary(self) -> dict[str, int]:
        """Get registry summary."""
        return {
            "total_giants": len(self._giants),
            "categories": {  # type: ignore[dict-item]
                cat.value: len(keys)
                for cat, keys in self._category_index.items()
                if keys
            },
            "applications": len(self._applications),
        }


# Global registry instance
_registry: Optional[GiantsRegistry] = None


def get_giants_registry() -> GiantsRegistry:
    """Get the global giants registry."""
    global _registry
    if _registry is None:
        _registry = GiantsRegistry()
    return _registry


def attribute(giant_names: list[str]) -> str:
    """Quick attribution string for given giants."""
    registry = get_giants_registry()
    giants = [registry.get(name) for name in giant_names]
    giants = [g for g in giants if g is not None]
    return " | ".join(g.format_attribution() for g in giants)
