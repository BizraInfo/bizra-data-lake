"""
RDVE Interdisciplinary Transfer Engine — Cross-Domain Pattern Library

Implements the cross-domain knowledge transfer mechanism from the RDVE whitepaper.
Scientific concepts are represented as transferable pattern templates that can be
instantiated in new contexts. This enables systematic rather than serendipitous
cross-disciplinary innovation.

Pattern Template Structure:
    - Source domain (where the pattern originated)
    - Core principle (the transferable insight)
    - Transfer conditions (when the pattern applies)
    - Instantiation recipe (how to apply it)
    - Historical examples (prior successful transfers)

Standing on Giants:
    Shannon (information theory → SNR quality) ·
    Maturana (biology → autopoietic computing) ·
    Gini (economics → fairness metrics) ·
    Lamport (distributed systems → consensus) ·
    Kahneman (psychology → cognitive budgets) ·
    Deming (manufacturing → software quality)

Artifact: core/rdve/interdisciplinary.py
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAINS & PATTERN TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════


class Domain(str, Enum):
    """Scientific domains for cross-disciplinary transfer."""

    INFORMATION_THEORY = "information_theory"
    BIOLOGY = "biology"
    PHYSICS = "physics"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    MANUFACTURING = "manufacturing"
    MILITARY = "military"
    MATHEMATICS = "mathematics"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    NEUROSCIENCE = "neuroscience"
    GAME_THEORY = "game_theory"
    ETHICS = "ethics"


class TransferConfidence(str, Enum):
    """Confidence level of a cross-domain transfer."""

    PROVEN = "proven"  # Transfer validated in production
    HIGH = "high"  # Strong theoretical + empirical support
    MEDIUM = "medium"  # Theoretical support, limited empirical
    SPECULATIVE = "speculative"  # Novel hypothesis, needs validation


@dataclass
class DomainPattern:
    """
    A transferable pattern template from one domain to another.

    This is the atomic unit of interdisciplinary innovation —
    a distilled principle that can be instantiated in new contexts.
    """

    id: str
    name: str
    source_domain: Domain
    core_principle: str
    transfer_conditions: List[str]
    instantiation_recipe: List[str]
    historical_examples: List[Dict[str, str]]
    target_domains: List[Domain]
    confidence: TransferConfidence
    giant: str  # Who discovered/formalized this principle

    # Metadata
    bizra_implementation: Optional[str] = None  # File path if already implemented
    tags: List[str] = field(default_factory=list)

    def matches_context(self, context_tags: Set[str]) -> float:
        """
        Compute match score between this pattern and a problem context.

        Returns 0.0-1.0 indicating how applicable this pattern is.
        """
        if not self.tags:
            return 0.0
        overlap = context_tags & set(self.tags)
        return len(overlap) / max(len(self.tags), 1)


@dataclass
class TransferResult:
    """Result of attempting a cross-domain transfer."""

    pattern: DomainPattern
    target_domain: Domain
    applicability_score: float
    instantiation: str  # How to apply it
    risks: List[str]
    expected_benefit: str


# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

# These patterns encode the actual cross-domain transfers already proven
# in the BIZRA codebase, plus promising new transfers from the RDVE whitepaper.

CANONICAL_PATTERNS: Final[List[DomainPattern]] = [
    DomainPattern(
        id="shannon_snr",
        name="Signal-to-Noise Ratio as Quality Metric",
        source_domain=Domain.INFORMATION_THEORY,
        core_principle=(
            "Quality = Signal / Noise. Geometric mean for signal ensures "
            "uniformly strong dimensions. Arithmetic mean for noise allows "
            "single-dimension mitigation."
        ),
        transfer_conditions=[
            "System produces outputs with variable quality",
            "Quality has multiple measurable dimensions",
            "Need to distinguish genuine insight from noise",
        ],
        instantiation_recipe=[
            "Define signal dimensions (relevance, novelty, groundedness, etc.)",
            "Define noise dimensions (redundancy, inconsistency, ambiguity, etc.)",
            "Use geometric mean for signal power (demands uniform strength)",
            "Use weighted arithmetic mean for noise (allows targeted reduction)",
            "Gate: SNR must exceed threshold before output is accepted",
        ],
        historical_examples=[
            {
                "from": "Telecommunications",
                "to": "AI output quality",
                "result": "SNRMaximizer in BIZRA sovereign engine",
            },
            {
                "from": "Audio engineering",
                "to": "LLM response filtering",
                "result": "7-noise-type model with Ihsan coupling",
            },
        ],
        target_domains=[Domain.PSYCHOLOGY, Domain.ECONOMICS, Domain.BIOLOGY],
        confidence=TransferConfidence.PROVEN,
        giant="Shannon (1948)",
        bizra_implementation="core/sovereign/snr_maximizer.py",
        tags=["quality", "filtering", "scoring", "multi-dimensional"],
    ),
    DomainPattern(
        id="maturana_autopoiesis",
        name="Autopoietic Self-Production",
        source_domain=Domain.BIOLOGY,
        core_principle=(
            "A system that continuously produces and regenerates the components "
            "that constitute it. The system produces itself through its own "
            "operation, maintaining identity while changing structure."
        ),
        transfer_conditions=[
            "System needs to improve itself autonomously",
            "Self-modification must preserve system identity/constraints",
            "Rollback capability is available for failed modifications",
        ],
        instantiation_recipe=[
            "Define the system's constitutional identity (invariants)",
            "Create observe-hypothesize-validate-implement loop",
            "Gate all modifications through constitutional checks",
            "Maintain rollback stack for safety",
            "Feed outcomes back to improve hypothesis generation",
        ],
        historical_examples=[
            {
                "from": "Cell biology (autopoiesis)",
                "to": "Self-improving AI systems",
                "result": "AutopoieticLoop in BIZRA with Z3 FATE gates",
            },
        ],
        target_domains=[
            Domain.DISTRIBUTED_SYSTEMS,
            Domain.ECONOMICS,
            Domain.GAME_THEORY,
        ],
        confidence=TransferConfidence.PROVEN,
        giant="Maturana (1972)",
        bizra_implementation="core/autopoiesis/loop_engine.py",
        tags=["self-improvement", "identity", "regeneration", "recursive"],
    ),
    DomainPattern(
        id="gini_fairness",
        name="Gini Coefficient as Fairness Hard Gate",
        source_domain=Domain.ECONOMICS,
        core_principle=(
            "Inequality can be measured by the Gini coefficient (0=perfect equality, "
            "1=total inequality). Setting a hard threshold prevents plutocratic "
            "concentration of resources."
        ),
        transfer_conditions=[
            "System distributes resources among participants",
            "Fairness is a hard constraint, not an optimization target",
            "Need to prevent concentration of power/resources",
        ],
        instantiation_recipe=[
            "Measure resource distribution across participants",
            "Compute Gini coefficient from distribution",
            "REJECT any transaction that would push Gini above threshold",
            "Combine with Harberger tax for continuous redistribution",
            "Apply Universal Basic Compute (UBC) floor",
        ],
        historical_examples=[
            {
                "from": "Income inequality (Gini, 1912)",
                "to": "Compute resource fairness",
                "result": "ADL_GINI_THRESHOLD=0.40 in BIZRA constitution",
            },
        ],
        target_domains=[Domain.DISTRIBUTED_SYSTEMS, Domain.GAME_THEORY],
        confidence=TransferConfidence.PROVEN,
        giant="Gini (1912), Harberger (1962), Rawls (1971)",
        bizra_implementation="core/integration/constants.py",
        tags=["fairness", "distribution", "inequality", "hard-gate"],
    ),
    DomainPattern(
        id="boyd_ooda",
        name="OODA Loop for Rapid Decision-Making",
        source_domain=Domain.MILITARY,
        core_principle=(
            "Observe-Orient-Decide-Act cycle. Speed of iteration matters more "
            "than quality of individual decisions. The entity that completes "
            "OODA faster gains temporal advantage."
        ),
        transfer_conditions=[
            "System operates in dynamic, changing environment",
            "Speed of adaptation matters",
            "Feedback from actions is available",
        ],
        instantiation_recipe=[
            "Observe: Gather raw state (non-mutating sensors)",
            "Orient: Form multiple hypotheses (GoT divergence)",
            "Decide: Score hypotheses through quality gates",
            "Act: Implement best hypothesis with rollback safety",
            "Minimize cycle time while maintaining quality floor",
        ],
        historical_examples=[
            {
                "from": "Fighter pilot decision loops",
                "to": "Autonomous agent decision cycles",
                "result": "Node0 Proactive Pilot OODA implementation",
            },
        ],
        target_domains=[Domain.DISTRIBUTED_SYSTEMS, Domain.GAME_THEORY],
        confidence=TransferConfidence.PROVEN,
        giant="Boyd (1976)",
        bizra_implementation="scripts/node0_activate.py",
        tags=["speed", "adaptation", "decision", "loop", "tempo"],
    ),
    DomainPattern(
        id="deming_pdca",
        name="PDCA Quality Cycle",
        source_domain=Domain.MANUFACTURING,
        core_principle=(
            "Plan-Do-Check-Act cycle for continuous quality improvement. "
            "Quality is not an event but a process. Each cycle tightens "
            "the quality spiral upward."
        ),
        transfer_conditions=[
            "System produces measurable outputs",
            "Quality can be quantified and compared across iterations",
            "Iterative improvement is possible",
        ],
        instantiation_recipe=[
            "Plan: Define quality targets (SNR/Ihsan thresholds)",
            "Do: Execute the process (generate/explore/filter)",
            "Check: Measure results against targets (constitutional gate)",
            "Act: Integrate improvements, adjust targets if needed",
            "Repeat with tightening quality spiral",
        ],
        historical_examples=[
            {
                "from": "Manufacturing quality control",
                "to": "CI/CD pipeline quality gates",
                "result": "ElitePipeline with constitutional validation",
            },
        ],
        target_domains=[Domain.BIOLOGY, Domain.PSYCHOLOGY],
        confidence=TransferConfidence.PROVEN,
        giant="Deming (1950)",
        bizra_implementation="core/elite/pipeline.py",
        tags=["quality", "iteration", "improvement", "measurement"],
    ),
    DomainPattern(
        id="kahneman_cognitive_budget",
        name="Cognitive Budget Allocation",
        source_domain=Domain.PSYCHOLOGY,
        core_principle=(
            "Cognitive resources are finite. Allocate them proportionally "
            "to task complexity. System 1 (fast/cheap) for trivial tasks, "
            "System 2 (slow/expensive) for complex tasks."
        ),
        transfer_conditions=[
            "System has variable-cost inference options",
            "Tasks vary in complexity",
            "Need to minimize cost while maintaining quality",
        ],
        instantiation_recipe=[
            "Classify task complexity (trivial/simple/moderate/complex/frontier)",
            "Route to appropriate model tier (nano/fast/reasoning/frontier)",
            "Set token budget proportional to complexity",
            "Validate output quality regardless of tier used",
        ],
        historical_examples=[
            {
                "from": "Dual-process theory (Kahneman, 2011)",
                "to": "Model routing in BIZRA",
                "result": "7-3-6-9 DNA cognitive budget tiers",
            },
        ],
        target_domains=[Domain.ECONOMICS, Domain.DISTRIBUTED_SYSTEMS],
        confidence=TransferConfidence.PROVEN,
        giant="Kahneman (2011)",
        bizra_implementation="core/elite/cognitive_budget.py",
        tags=["resources", "routing", "efficiency", "tiered"],
    ),
    DomainPattern(
        id="lamport_bft",
        name="Byzantine Fault Tolerance for Trust",
        source_domain=Domain.DISTRIBUTED_SYSTEMS,
        core_principle=(
            "In a system with n nodes, up to f = (n-1)/3 can be Byzantine "
            "(malicious/faulty) and the system still reaches consensus. "
            "Trust is computational, not assumed."
        ),
        transfer_conditions=[
            "Multiple agents/sources must agree",
            "Some agents may be unreliable or adversarial",
            "Need guaranteed correctness despite failures",
        ],
        instantiation_recipe=[
            "Define quorum requirement (2f+1 out of 3f+1 agents)",
            "Each agent signs its output cryptographically",
            "Consensus requires supermajority agreement",
            "Dissenting agents are flagged but not expelled",
        ],
        historical_examples=[
            {
                "from": "Byzantine Generals Problem (1982)",
                "to": "Guardian Council consensus in BIZRA",
                "result": "Multi-agent validation with signed verdicts",
            },
        ],
        target_domains=[Domain.ECONOMICS, Domain.GAME_THEORY, Domain.ETHICS],
        confidence=TransferConfidence.PROVEN,
        giant="Lamport (1982)",
        bizra_implementation="core/pci/gates.py",
        tags=["consensus", "trust", "fault-tolerance", "adversarial"],
    ),
    DomainPattern(
        id="thermodynamic_free_energy",
        name="Free Energy Minimization for Learning",
        source_domain=Domain.PHYSICS,
        core_principle=(
            "Systems evolve toward states of minimum free energy. Learning can "
            "be framed as minimizing the surprise (prediction error) between "
            "model predictions and observations."
        ),
        transfer_conditions=[
            "System has an internal model of its environment",
            "Prediction errors are measurable",
            "Model can be updated based on observations",
        ],
        instantiation_recipe=[
            "Define an internal generative model",
            "Compute prediction error (surprise) on new observations",
            "Update model to minimize prediction error",
            "Balance model complexity vs. prediction accuracy",
        ],
        historical_examples=[
            {
                "from": "Thermodynamics (Helmholtz, 1882)",
                "to": "Active inference / free energy principle",
                "result": "Friston's free energy principle for brain modeling",
            },
        ],
        target_domains=[Domain.NEUROSCIENCE, Domain.PSYCHOLOGY, Domain.BIOLOGY],
        confidence=TransferConfidence.MEDIUM,
        giant="Helmholtz (1882), Friston (2006)",
        tags=["prediction", "surprise", "learning", "model-updating"],
    ),
    DomainPattern(
        id="nash_equilibrium",
        name="Nash Equilibrium for Multi-Agent Coordination",
        source_domain=Domain.GAME_THEORY,
        core_principle=(
            "A stable state where no agent can improve its outcome by "
            "unilaterally changing its strategy. In multi-agent systems, "
            "design incentives so that the Nash equilibrium aligns with "
            "the system's desired behavior."
        ),
        transfer_conditions=[
            "Multiple agents with potentially conflicting objectives",
            "Agents can observe and respond to each other's actions",
            "System needs stable coordination without central control",
        ],
        instantiation_recipe=[
            "Define agent utility functions (include Ihsan as constraint)",
            "Identify the Nash equilibrium of the agent game",
            "Verify equilibrium aligns with system goals",
            "If misaligned, adjust incentive structure",
            "Add constitutional constraints as hard gates on strategy space",
        ],
        historical_examples=[
            {
                "from": "Game theory (Nash, 1950)",
                "to": "Multi-agent AI coordination",
                "result": "GANs, self-play RL, token economy design",
            },
        ],
        target_domains=[Domain.ECONOMICS, Domain.DISTRIBUTED_SYSTEMS],
        confidence=TransferConfidence.HIGH,
        giant="Nash (1950)",
        tags=["coordination", "incentives", "equilibrium", "multi-agent"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# INTERDISCIPLINARY TRANSFER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class InterdisciplinaryTransfer:
    """
    Cross-domain pattern transfer engine.

    Given a problem context (described by tags and target domain),
    retrieves applicable patterns from the canonical library and
    generates instantiation proposals.

    Standing on Giants:
        All giants in the pattern library — this engine is itself a
        transfer of the "pattern language" concept from Christopher
        Alexander's architecture theory (1977) to AI system design.

    Usage:
        >>> engine = InterdisciplinaryTransfer()
        >>> results = engine.find_transfers(
        ...     context_tags={"quality", "filtering", "multi-dimensional"},
        ...     target_domain=Domain.BIOLOGY,
        ... )
        >>> for r in results:
        ...     print(f"{r.pattern.name}: {r.applicability_score:.2f}")
    """

    def __init__(
        self,
        patterns: Optional[List[DomainPattern]] = None,
        min_applicability: float = 0.3,
    ):
        self._patterns = patterns or list(CANONICAL_PATTERNS)
        self._min_applicability = min_applicability
        self._transfer_history: List[TransferResult] = []

        logger.info(
            f"InterdisciplinaryTransfer initialized with "
            f"{len(self._patterns)} patterns from "
            f"{len(set(p.source_domain for p in self._patterns))} domains"
        )

    def add_pattern(self, pattern: DomainPattern) -> None:
        """Add a new pattern to the library."""
        self._patterns.append(pattern)
        logger.info(f"Added pattern: {pattern.name} ({pattern.source_domain.value})")

    def find_transfers(
        self,
        context_tags: Set[str],
        target_domain: Optional[Domain] = None,
        min_confidence: TransferConfidence = TransferConfidence.MEDIUM,
    ) -> List[TransferResult]:
        """
        Find applicable cross-domain transfers for a given context.

        Args:
            context_tags: Tags describing the problem context
            target_domain: Optional target domain filter
            min_confidence: Minimum confidence level

        Returns:
            List of TransferResult sorted by applicability score
        """
        confidence_order = {
            TransferConfidence.PROVEN: 4,
            TransferConfidence.HIGH: 3,
            TransferConfidence.MEDIUM: 2,
            TransferConfidence.SPECULATIVE: 1,
        }
        min_conf_level = confidence_order[min_confidence]

        results: List[TransferResult] = []

        for pattern in self._patterns:
            # Filter by confidence
            if confidence_order[pattern.confidence] < min_conf_level:
                continue

            # Filter by target domain
            if target_domain and target_domain not in pattern.target_domains:
                # Also check if pattern's source domain is the target
                # (same-domain patterns always applicable)
                if pattern.source_domain != target_domain:
                    continue

            # Compute applicability score
            score = pattern.matches_context(context_tags)

            # Boost score for proven patterns
            confidence_boost = confidence_order[pattern.confidence] * 0.1
            score = min(1.0, score + confidence_boost)

            if score >= self._min_applicability:
                result = TransferResult(
                    pattern=pattern,
                    target_domain=target_domain or Domain.DISTRIBUTED_SYSTEMS,
                    applicability_score=score,
                    instantiation="\n".join(
                        f"  {i+1}. {step}"
                        for i, step in enumerate(pattern.instantiation_recipe)
                    ),
                    risks=[
                        f"Transfer from {pattern.source_domain.value} may not "
                        f"fully generalize to new context",
                        "Requires empirical validation after transfer",
                    ],
                    expected_benefit=pattern.core_principle[:200],
                )
                results.append(result)

        # Sort by applicability score (descending)
        results.sort(key=lambda r: r.applicability_score, reverse=True)

        return results

    def get_patterns_by_domain(self, domain: Domain) -> List[DomainPattern]:
        """Get all patterns from a specific source domain."""
        return [p for p in self._patterns if p.source_domain == domain]

    def get_proven_patterns(self) -> List[DomainPattern]:
        """Get all patterns with PROVEN confidence (validated in BIZRA)."""
        return [
            p for p in self._patterns
            if p.confidence == TransferConfidence.PROVEN
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern library statistics."""
        domains = {}
        for p in self._patterns:
            d = p.source_domain.value
            if d not in domains:
                domains[d] = 0
            domains[d] += 1

        return {
            "total_patterns": len(self._patterns),
            "patterns_by_domain": domains,
            "proven_count": sum(
                1 for p in self._patterns
                if p.confidence == TransferConfidence.PROVEN
            ),
            "implemented_count": sum(
                1 for p in self._patterns
                if p.bizra_implementation is not None
            ),
            "transfers_executed": len(self._transfer_history),
        }
