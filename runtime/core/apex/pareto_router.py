"""
BIZRA Apex - Pareto-Optimal Multi-Objective Router
===================================================
Implements Pareto-optimal routing for multi-objective decision making
with Thompson sampling-based selection from the Pareto front.

The ParetoOptimalRouter manages multiple competing objectives:
1. SNR (Signal-to-Noise Ratio): Quality of reasoning output
2. Novelty: Semantic distance from known patterns
3. Alignment: Task-persona alignment score
4. Ethics: Ihsan ethical compliance score
5. Latency_inv: Inverse latency (1/latency for minimization)

Key Features:
    - Dirichlet sampling for diverse weight vector generation
    - Pareto dominance checking for front identification
    - Thompson sampling for selection from Pareto front
    - Bayesian posterior updates for continuous learning
    - Integration with PersonaDefinition for persona-aware routing

Algorithm:
    1. Generate candidate lambda vectors via Dirichlet sampling
    2. Score each lambda against all objectives
    3. Identify Pareto-optimal points (non-dominated)
    4. Apply Thompson sampling with preference bias
    5. Update posteriors based on execution outcomes

Domain: bizra-pci-v1:apex:pareto
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

# Optional numpy for enhanced performance
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Import PersonaDefinition for integration
from core.personaplex.persona import (
    PersonaDefinition,
    create_default_registry,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (imported from unified constants.py)
# =============================================================================

from core.constants import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    NOVELTY_THRESHOLD_STANDARD,
)

DOMAIN_PREFIX = "bizra-pci-v1:apex:pareto"
VERSION = "1.0.0"

# Default thresholds from BIZRA constitution (aliased for backward compatibility)
DEFAULT_SNR_THRESHOLD = SNR_THRESHOLD_T0_ELITE  # 0.98
DEFAULT_IHSAN_THRESHOLD = IHSAN_THRESHOLD  # 0.95
DEFAULT_NOVELTY_THRESHOLD = NOVELTY_THRESHOLD_STANDARD  # 0.75

# Dirichlet sampling parameters
DEFAULT_DIRICHLET_ALPHA = 1.0  # Uniform prior over simplex
DEFAULT_NUM_SAMPLES = 50

# Thompson sampling parameters
DEFAULT_THOMPSON_ALPHA = 1.0
DEFAULT_THOMPSON_BETA = 1.0


# =============================================================================
# STDLIB FALLBACKS (when numpy unavailable)
# =============================================================================


def _stdlib_dirichlet_sample(
    alphas: Sequence[float],
    rng: Optional[random.Random] = None,
) -> List[float]:
    """
    Sample from Dirichlet distribution using stdlib.

    Uses the relationship: Dirichlet(alpha) can be sampled by normalizing
    independent Gamma(alpha_i, 1) samples.

    Args:
        alphas: Concentration parameters
        rng: Random number generator

    Returns:
        Sample from Dirichlet(alphas) as list of floats summing to 1
    """
    if rng is None:
        rng = random.Random()

    # Sample from Gamma distributions
    samples = [rng.gammavariate(a, 1.0) for a in alphas]
    total = sum(samples)

    if total <= 0:
        # Fallback to uniform
        n = len(alphas)
        return [1.0 / n] * n

    return [s / total for s in samples]


def _stdlib_beta_sample(
    alpha: float,
    beta: float,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Sample from Beta distribution using stdlib.

    Args:
        alpha: Shape parameter alpha
        beta: Shape parameter beta
        rng: Random number generator

    Returns:
        Sample from Beta(alpha, beta)
    """
    if rng is None:
        rng = random.Random()

    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)

    if x + y <= 0:
        return 0.5

    return x / (x + y)


def _stdlib_log(x: float) -> float:
    """Logarithm with protection against zero."""
    return math.log(x) if x > 0 else float("-inf")


# =============================================================================
# OBJECTIVE VECTOR
# =============================================================================


class ObjectiveName(str, Enum):
    """Names of optimization objectives."""

    SNR = "snr"
    NOVELTY = "novelty"
    ALIGNMENT = "alignment"
    ETHICS = "ethics"
    LATENCY_INV = "latency_inv"


@dataclass(frozen=True)
class ObjectiveVector:
    """
    Vector of optimization objectives for Pareto analysis.

    All objectives are normalized to [0, 1] range where higher is better.
    For latency, we use inverse latency so higher = faster = better.

    Attributes:
        snr: Signal-to-Noise Ratio (quality of reasoning output)
        novelty: Semantic distance from known patterns (0=repetitive, 1=novel)
        alignment: Task-persona alignment score (domain/capability match)
        ethics: Ihsan ethical compliance score (8-dimension weighted score)
        latency_inv: Inverse latency (1 / normalized_latency), higher = faster

    Example:
        >>> obj = ObjectiveVector(snr=0.98, novelty=0.75, alignment=0.85,
        ...                       ethics=0.95, latency_inv=0.8)
        >>> obj.to_array()
        [0.98, 0.75, 0.85, 0.95, 0.8]
    """

    snr: float
    novelty: float
    alignment: float
    ethics: float
    latency_inv: float

    def __post_init__(self) -> None:
        """Validate all objectives are in valid range."""
        for name in ObjectiveName:
            value = getattr(self, name.value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Objective {name.value} must be in [0, 1], got {value}"
                )

    @property
    def dimension(self) -> int:
        """Number of objectives."""
        return 5

    def to_array(self) -> List[float]:
        """Convert to array representation for computation."""
        return [
            self.snr,
            self.novelty,
            self.alignment,
            self.ethics,
            self.latency_inv,
        ]

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "snr": self.snr,
            "novelty": self.novelty,
            "alignment": self.alignment,
            "ethics": self.ethics,
            "latency_inv": self.latency_inv,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ObjectiveVector":
        """Deserialize from dictionary."""
        return cls(
            snr=data.get("snr", 0.0),
            novelty=data.get("novelty", 0.0),
            alignment=data.get("alignment", 0.0),
            ethics=data.get("ethics", 0.0),
            latency_inv=data.get("latency_inv", 0.0),
        )

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "ObjectiveVector":
        """Create from array representation."""
        if len(arr) != 5:
            raise ValueError(f"Expected 5 objectives, got {len(arr)}")
        return cls(
            snr=arr[0],
            novelty=arr[1],
            alignment=arr[2],
            ethics=arr[3],
            latency_inv=arr[4],
        )

    def weighted_sum(self, lambdas: Sequence[float]) -> float:
        """
        Compute weighted sum of objectives.

        Args:
            lambdas: Weight vector (must sum to 1)

        Returns:
            Weighted sum of objectives
        """
        if len(lambdas) != self.dimension:
            raise ValueError(f"Expected {self.dimension} weights, got {len(lambdas)}")

        values = self.to_array()
        return sum(w * v for w, v in zip(lambdas, values))

    def satisfies_thresholds(
        self,
        snr_threshold: float = DEFAULT_SNR_THRESHOLD,
        ethics_threshold: float = DEFAULT_IHSAN_THRESHOLD,
        novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    ) -> bool:
        """
        Check if objectives meet BIZRA quality thresholds.

        Args:
            snr_threshold: Minimum SNR (default: 0.98)
            ethics_threshold: Minimum Ihsan score (default: 0.95)
            novelty_threshold: Minimum novelty score (default: 0.75)

        Returns:
            True if all thresholds are satisfied
        """
        return (
            self.snr >= snr_threshold
            and self.ethics >= ethics_threshold
            and self.novelty >= novelty_threshold
        )


# =============================================================================
# PARETO POINT
# =============================================================================


@dataclass
class ParetoPoint:
    """
    A point on the Pareto front with associated metadata.

    Attributes:
        lambdas: Weight vector that generated this point
        objectives: Objective values at this point
        persona_id: Associated persona identifier
        thompson_alpha: Thompson sampling alpha parameter
        thompson_beta: Thompson sampling beta parameter
        selection_count: Number of times this point was selected
        success_count: Number of successful outcomes
        created_at: Timestamp of creation
    """

    lambdas: Tuple[float, ...]
    objectives: ObjectiveVector
    persona_id: str
    thompson_alpha: float = DEFAULT_THOMPSON_ALPHA
    thompson_beta: float = DEFAULT_THOMPSON_BETA
    selection_count: int = 0
    success_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def point_id(self) -> str:
        """Generate unique identifier for this point."""
        # Hash the lambdas and persona_id for uniqueness
        content = f"{self.lambdas}:{self.persona_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def thompson_mean(self) -> float:
        """Expected value from Beta distribution."""
        return self.thompson_alpha / (self.thompson_alpha + self.thompson_beta)

    @property
    def thompson_variance(self) -> float:
        """Variance of Beta distribution (uncertainty measure)."""
        total = self.thompson_alpha + self.thompson_beta
        return (self.thompson_alpha * self.thompson_beta) / (
            total * total * (total + 1)
        )

    def thompson_sample(self, rng: Optional[Any] = None) -> float:
        """
        Sample from Thompson posterior.

        Args:
            rng: Random number generator (numpy or stdlib)

        Returns:
            Sample from Beta(alpha, beta)
        """
        if NUMPY_AVAILABLE and rng is None:
            rng = np.random.default_rng()

        if NUMPY_AVAILABLE and hasattr(rng, "beta"):
            return rng.beta(self.thompson_alpha, self.thompson_beta)
        else:
            return _stdlib_beta_sample(
                self.thompson_alpha,
                self.thompson_beta,
                rng if isinstance(rng, random.Random) else None,
            )

    def update_from_outcome(self, success: bool, snr_bonus: float = 0.0) -> None:
        """
        Update Thompson parameters from execution outcome.

        Args:
            success: Whether the execution was successful
            snr_bonus: Additional alpha boost based on actual SNR
        """
        self.selection_count += 1

        if success:
            self.success_count += 1
            self.thompson_alpha += 1.0 + snr_bonus
        else:
            self.thompson_beta += 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "point_id": self.point_id,
            "lambdas": list(self.lambdas),
            "objectives": self.objectives.to_dict(),
            "persona_id": self.persona_id,
            "thompson_alpha": self.thompson_alpha,
            "thompson_beta": self.thompson_beta,
            "selection_count": self.selection_count,
            "success_count": self.success_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoPoint":
        """Deserialize from dictionary."""
        return cls(
            lambdas=tuple(data["lambdas"]),
            objectives=ObjectiveVector.from_dict(data["objectives"]),
            persona_id=data["persona_id"],
            thompson_alpha=data.get("thompson_alpha", DEFAULT_THOMPSON_ALPHA),
            thompson_beta=data.get("thompson_beta", DEFAULT_THOMPSON_BETA),
            selection_count=data.get("selection_count", 0),
            success_count=data.get("success_count", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


# =============================================================================
# ROUTING PREFERENCE
# =============================================================================


@dataclass
class RoutingPreference:
    """
    User/task preference for objective weighting.

    Preferences bias selection from the Pareto front toward
    points that better match the specified objective priorities.

    Attributes:
        snr_weight: Preference for quality (higher = prioritize quality)
        novelty_weight: Preference for novelty
        alignment_weight: Preference for task alignment
        ethics_weight: Preference for ethical compliance
        latency_weight: Preference for speed (higher = prioritize speed)

    Note:
        Weights are normalized internally - raw values indicate relative importance.
    """

    snr_weight: float = 1.0
    novelty_weight: float = 1.0
    alignment_weight: float = 1.0
    ethics_weight: float = 1.0
    latency_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate weights are non-negative."""
        weights = [
            self.snr_weight,
            self.novelty_weight,
            self.alignment_weight,
            self.ethics_weight,
            self.latency_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All preference weights must be non-negative")

    def to_normalized_array(self) -> List[float]:
        """
        Convert to normalized weight array.

        Returns:
            Weights normalized to sum to 1
        """
        weights = [
            self.snr_weight,
            self.novelty_weight,
            self.alignment_weight,
            self.ethics_weight,
            self.latency_weight,
        ]
        total = sum(weights)
        if total <= 0:
            # Uniform if all zero
            return [0.2] * 5
        return [w / total for w in weights]

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "snr_weight": self.snr_weight,
            "novelty_weight": self.novelty_weight,
            "alignment_weight": self.alignment_weight,
            "ethics_weight": self.ethics_weight,
            "latency_weight": self.latency_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "RoutingPreference":
        """Deserialize from dictionary."""
        return cls(
            snr_weight=data.get("snr_weight", 1.0),
            novelty_weight=data.get("novelty_weight", 1.0),
            alignment_weight=data.get("alignment_weight", 1.0),
            ethics_weight=data.get("ethics_weight", 1.0),
            latency_weight=data.get("latency_weight", 1.0),
        )

    @classmethod
    def quality_focused(cls) -> "RoutingPreference":
        """Preference that prioritizes SNR and ethics."""
        return cls(snr_weight=3.0, ethics_weight=2.0)

    @classmethod
    def speed_focused(cls) -> "RoutingPreference":
        """Preference that prioritizes latency."""
        return cls(latency_weight=3.0, alignment_weight=1.5)

    @classmethod
    def novelty_focused(cls) -> "RoutingPreference":
        """Preference that prioritizes novelty."""
        return cls(novelty_weight=3.0, snr_weight=1.5)

    @classmethod
    def balanced(cls) -> "RoutingPreference":
        """Balanced preference (default)."""
        return cls()


# =============================================================================
# PARETO SELECTION RESULT
# =============================================================================


@dataclass
class ParetoSelectionResult:
    """
    Result of Pareto-optimal selection.

    Attributes:
        selected_point: The chosen Pareto point
        pareto_front_size: Size of the Pareto front
        thompson_sample: The Thompson sample value that won
        preference_score: Score based on user preference
        alternatives: Top alternative points considered
        reasoning: Human-readable explanation of selection
        timestamp: ISO timestamp of selection
    """

    selected_point: ParetoPoint
    pareto_front_size: int
    thompson_sample: float
    preference_score: float
    alternatives: List[ParetoPoint]
    reasoning: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "selected_point": self.selected_point.to_dict(),
            "pareto_front_size": self.pareto_front_size,
            "thompson_sample": self.thompson_sample,
            "preference_score": self.preference_score,
            "alternatives": [p.to_dict() for p in self.alternatives[:3]],
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


# =============================================================================
# OBJECTIVE EVALUATOR PROTOCOL
# =============================================================================


class ObjectiveEvaluator(Protocol):
    """Protocol for evaluating objectives for a given task."""

    def evaluate(
        self,
        task_domains: List[str],
        persona: PersonaDefinition,
        lambdas: Tuple[float, ...],
    ) -> ObjectiveVector:
        """
        Evaluate objectives for a task-persona-lambda combination.

        Args:
            task_domains: List of domain keywords for the task
            persona: The persona being evaluated
            lambdas: Weight vector for scalarization

        Returns:
            ObjectiveVector with evaluated scores
        """
        ...


# =============================================================================
# DEFAULT OBJECTIVE EVALUATOR
# =============================================================================


class DefaultObjectiveEvaluator:
    """
    Default implementation of objective evaluation.

    Uses heuristic scoring based on persona attributes and domain matching.
    Production systems may override with model-based evaluation.
    """

    def __init__(
        self,
        base_snr: float = 0.90,
        base_novelty: float = 0.70,
        base_ethics: float = 0.92,
        base_latency_inv: float = 0.75,
    ):
        """
        Initialize with base scores.

        Args:
            base_snr: Base SNR score for all personas
            base_novelty: Base novelty score
            base_ethics: Base ethics score
            base_latency_inv: Base inverse latency
        """
        self.base_snr = base_snr
        self.base_novelty = base_novelty
        self.base_ethics = base_ethics
        self.base_latency_inv = base_latency_inv

    def evaluate(
        self,
        task_domains: List[str],
        persona: PersonaDefinition,
        lambdas: Tuple[float, ...],
    ) -> ObjectiveVector:
        """
        Evaluate objectives using heuristic scoring.

        Scoring logic:
        - SNR: base + alignment bonus + vote_weight bonus
        - Novelty: base adjusted by persona creativity indicators
        - Alignment: Direct computation from persona.compute_task_alignment
        - Ethics: Higher for ethics/security personas
        - Latency: Lower for complex (high vote_weight) personas
        """
        # Compute alignment
        alignment = (
            persona.compute_task_alignment(task_domains) if task_domains else 0.5
        )

        # SNR: Higher alignment and vote_weight = better quality
        snr = min(
            1.0, self.base_snr + alignment * 0.05 + persona.base_vote_weight * 0.03
        )

        # Novelty: Creative personas score higher
        creative_domains = {
            "creativity",
            "synthesis",
            "ideation",
            "cross-domain-transfer",
        }
        persona_domains_lower = set(d.lower() for d in persona.expertise_domains)
        creative_overlap = len(creative_domains & persona_domains_lower)
        novelty = min(1.0, self.base_novelty + creative_overlap * 0.05)

        # Ethics: Security and ethics personas score higher
        if persona.is_security_persona or persona.is_ethics_persona:
            ethics = min(1.0, self.base_ethics + 0.06)
        else:
            ethics = self.base_ethics

        # Latency: Inverse relationship with complexity (vote_weight as proxy)
        # Higher vote_weight = more sophisticated = potentially slower
        latency_inv = max(0.1, self.base_latency_inv - persona.base_vote_weight * 0.1)

        return ObjectiveVector(
            snr=snr,
            novelty=novelty,
            alignment=alignment,
            ethics=ethics,
            latency_inv=latency_inv,
        )


# =============================================================================
# PARETO OPTIMAL ROUTER
# =============================================================================


class ParetoOptimalRouter:
    """
    Multi-objective router using Pareto optimization and Thompson sampling.

    This router combines:
    1. Dirichlet sampling to generate diverse weight vectors
    2. Pareto dominance to identify optimal trade-off points
    3. Thompson sampling to balance exploration and exploitation
    4. Preference-based bias for user-specified priorities

    Algorithm:
        1. compute_pareto_front(): Generate candidates via Dirichlet sampling,
           evaluate objectives, and identify non-dominated points
        2. select_from_pareto(): Use Thompson sampling with preference bias
           to select from the Pareto front
        3. update_from_outcome(): Bayesian update of Thompson parameters

    Attributes:
        personas: Dictionary mapping persona_id to PersonaDefinition
        evaluator: ObjectiveEvaluator for scoring
        pareto_front: Current Pareto front
        history: Selection history for analysis

    Example:
        >>> router = ParetoOptimalRouter(personas=registry.list_all())
        >>> router.compute_pareto_front(task_domains=["reasoning", "planning"])
        >>> result = router.select_from_pareto(preference=RoutingPreference.quality_focused())
        >>> # After execution...
        >>> router.update_from_outcome(result.selected_point.lambdas, success=True, actual_snr=0.97)
    """

    def __init__(
        self,
        personas: List[PersonaDefinition],
        evaluator: Optional[ObjectiveEvaluator] = None,
        persistence_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Pareto-optimal router.

        Args:
            personas: List of PersonaDefinition objects
            evaluator: Custom objective evaluator (uses default if None)
            persistence_path: Path for saving/loading state
            random_seed: Random seed for reproducibility

        Raises:
            ValueError: If no personas provided
        """
        if not personas:
            raise ValueError("At least one persona must be provided")

        self.personas: Dict[str, PersonaDefinition] = {
            p.persona_id: p for p in personas
        }

        self.evaluator: ObjectiveEvaluator = evaluator or DefaultObjectiveEvaluator()

        self.persistence_path = persistence_path or Path(
            "docs/evidence/apex/pareto_state.json"
        )

        # Initialize random number generator
        if random_seed is not None:
            if NUMPY_AVAILABLE:
                self.rng = np.random.default_rng(random_seed)
            else:
                self.rng = random.Random(random_seed)
        else:
            if NUMPY_AVAILABLE:
                self.rng = np.random.default_rng()
            else:
                self.rng = random.Random()

        # Pareto front state
        self.pareto_front: List[ParetoPoint] = []
        self._all_evaluated_points: List[ParetoPoint] = []

        # Selection history
        self._selection_history: List[ParetoSelectionResult] = []
        self._max_history = 500

        # Try to load existing state
        self._load_state()

        logger.info(
            f"ParetoOptimalRouter initialized with {len(personas)} personas, "
            f"numpy={'enabled' if NUMPY_AVAILABLE else 'disabled'}"
        )

    def compute_pareto_front(
        self,
        task_domains: List[str],
        num_samples: int = DEFAULT_NUM_SAMPLES,
        dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    ) -> List[ParetoPoint]:
        """
        Compute Pareto front for given task domains.

        Algorithm:
        1. Generate num_samples weight vectors via Dirichlet sampling
        2. For each persona and weight vector, evaluate objectives
        3. Identify Pareto-optimal (non-dominated) points
        4. Store front for subsequent selection

        Args:
            task_domains: List of domain keywords characterizing the task
            num_samples: Number of Dirichlet samples per persona
            dirichlet_alpha: Concentration parameter (lower = more extreme)

        Returns:
            List of ParetoPoint objects forming the Pareto front
        """
        logger.debug(
            f"Computing Pareto front for domains={task_domains}, "
            f"samples={num_samples}"
        )

        # Generate all candidate points
        candidates: List[ParetoPoint] = []

        # Dirichlet concentration parameters (5 objectives)
        alphas = [dirichlet_alpha] * 5

        for persona in self.personas.values():
            for _ in range(num_samples):
                # Sample weight vector from Dirichlet
                if NUMPY_AVAILABLE:
                    lambdas = tuple(self.rng.dirichlet(alphas))
                else:
                    lambdas = tuple(_stdlib_dirichlet_sample(alphas, self.rng))

                # Evaluate objectives for this persona-lambda pair
                objectives = self.evaluator.evaluate(
                    task_domains=task_domains,
                    persona=persona,
                    lambdas=lambdas,
                )

                # Create point with existing Thompson state if available
                point = self._get_or_create_point(
                    lambdas=lambdas,
                    objectives=objectives,
                    persona_id=persona.persona_id,
                )

                candidates.append(point)

        # Store all evaluated points for analysis
        self._all_evaluated_points = candidates

        # Identify Pareto front (non-dominated points)
        self.pareto_front = self._extract_pareto_front(candidates)

        logger.info(
            f"Pareto front computed: {len(self.pareto_front)} points "
            f"from {len(candidates)} candidates"
        )

        return self.pareto_front

    def select_from_pareto(
        self,
        preference: Optional[RoutingPreference] = None,
        require_threshold: bool = True,
    ) -> ParetoSelectionResult:
        """
        Select a point from the Pareto front using Thompson sampling.

        Algorithm:
        1. Filter points that meet thresholds (if required)
        2. Compute preference-weighted scores
        3. Sample from Thompson posterior for each point
        4. Combine Thompson sample with preference score
        5. Select point with highest combined score

        Args:
            preference: User preference for objective weighting
            require_threshold: If True, filter points not meeting BIZRA thresholds

        Returns:
            ParetoSelectionResult with selected point and metadata

        Raises:
            ValueError: If Pareto front is empty
        """
        if not self.pareto_front:
            raise ValueError(
                "Pareto front is empty. Call compute_pareto_front() first."
            )

        preference = preference or RoutingPreference.balanced()
        pref_weights = preference.to_normalized_array()

        # Filter by thresholds if required
        viable_points = self.pareto_front
        if require_threshold:
            viable_points = [
                p for p in self.pareto_front if p.objectives.satisfies_thresholds()
            ]

            if not viable_points:
                # Relax to just ethics threshold
                viable_points = [
                    p
                    for p in self.pareto_front
                    if p.objectives.ethics >= DEFAULT_IHSAN_THRESHOLD
                ]

                if not viable_points:
                    # Use full front if nothing passes
                    logger.warning("No points meet thresholds, using full Pareto front")
                    viable_points = self.pareto_front

        # Score each point
        scored_points: List[Tuple[ParetoPoint, float, float]] = []

        for point in viable_points:
            # Thompson sample
            thompson_sample = point.thompson_sample(self.rng)

            # Preference-weighted objective score
            obj_array = point.objectives.to_array()
            preference_score = sum(w * v for w, v in zip(pref_weights, obj_array))

            # Combined score: Thompson sample weighted by preference
            # This balances exploration (Thompson) with exploitation (preference)
            combined_score = 0.6 * thompson_sample + 0.4 * preference_score

            scored_points.append((point, combined_score, thompson_sample))

        # Sort by combined score
        scored_points.sort(key=lambda x: x[1], reverse=True)

        # Select winner
        winner, combined_score, thompson_sample = scored_points[0]

        # Get alternatives
        alternatives = [p for p, _, _ in scored_points[1:4]]

        # Build reasoning
        reasoning = self._build_selection_reasoning(
            winner=winner,
            thompson_sample=thompson_sample,
            preference=preference,
            viable_count=len(viable_points),
            total_front=len(self.pareto_front),
        )

        # Create result
        result = ParetoSelectionResult(
            selected_point=winner,
            pareto_front_size=len(self.pareto_front),
            thompson_sample=thompson_sample,
            preference_score=combined_score,
            alternatives=alternatives,
            reasoning=reasoning,
        )

        # Update selection history
        self._selection_history.append(result)
        if len(self._selection_history) > self._max_history:
            self._selection_history = self._selection_history[-self._max_history :]

        logger.info(
            f"Selected persona={winner.persona_id}, "
            f"thompson={thompson_sample:.3f}, "
            f"preference_score={combined_score:.3f}"
        )

        return result

    def update_from_outcome(
        self,
        lambdas: Tuple[float, ...],
        success: bool,
        actual_snr: float = 0.0,
    ) -> None:
        """
        Update Thompson posteriors based on execution outcome.

        Args:
            lambdas: Weight vector of the executed point
            success: Whether execution was successful
            actual_snr: Actual SNR achieved (used for alpha bonus)
        """
        # Find the point with matching lambdas
        target_point: Optional[ParetoPoint] = None

        for point in self._all_evaluated_points:
            if point.lambdas == lambdas:
                target_point = point
                break

        if target_point is None:
            logger.warning(f"Point with lambdas={lambdas} not found for update")
            return

        # Calculate SNR bonus (higher SNR = more alpha boost)
        snr_bonus = 0.0
        if success and actual_snr > DEFAULT_SNR_THRESHOLD:
            snr_bonus = (actual_snr - DEFAULT_SNR_THRESHOLD) * 5.0

        # Update the point
        target_point.update_from_outcome(success=success, snr_bonus=snr_bonus)

        logger.debug(
            f"Updated point for persona={target_point.persona_id}: "
            f"success={success}, snr_bonus={snr_bonus:.3f}, "
            f"new_mean={target_point.thompson_mean:.3f}"
        )

        # Persist periodically
        if len(self._selection_history) % 10 == 0:
            self._save_state()

    def _dominates(self, a: ObjectiveVector, b: ObjectiveVector) -> bool:
        """
        Check if point a Pareto-dominates point b.

        a dominates b if:
        - a is at least as good as b in all objectives
        - a is strictly better than b in at least one objective

        Args:
            a: First objective vector
            b: Second objective vector

        Returns:
            True if a dominates b
        """
        a_arr = a.to_array()
        b_arr = b.to_array()

        at_least_as_good = all(ai >= bi for ai, bi in zip(a_arr, b_arr))
        strictly_better = any(ai > bi for ai, bi in zip(a_arr, b_arr))

        return at_least_as_good and strictly_better

    def _extract_pareto_front(
        self,
        candidates: List[ParetoPoint],
    ) -> List[ParetoPoint]:
        """
        Extract Pareto-optimal points from candidates.

        A point is Pareto-optimal if no other point dominates it.

        Args:
            candidates: List of candidate points

        Returns:
            List of non-dominated (Pareto-optimal) points
        """
        if not candidates:
            return []

        pareto_front: List[ParetoPoint] = []

        for candidate in candidates:
            is_dominated = False

            # Check if candidate is dominated by any existing front member
            for front_member in pareto_front:
                if self._dominates(front_member.objectives, candidate.objectives):
                    is_dominated = True
                    break

            if not is_dominated:
                # Remove front members dominated by candidate
                pareto_front = [
                    m
                    for m in pareto_front
                    if not self._dominates(candidate.objectives, m.objectives)
                ]
                pareto_front.append(candidate)

        return pareto_front

    def _get_or_create_point(
        self,
        lambdas: Tuple[float, ...],
        objectives: ObjectiveVector,
        persona_id: str,
    ) -> ParetoPoint:
        """
        Get existing point with Thompson state or create new one.

        This preserves learned Thompson parameters across front computations.
        """
        # Generate point ID for lookup
        content = f"{lambdas}:{persona_id}"
        point_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Check existing points
        for existing in self._all_evaluated_points:
            if existing.point_id == point_id:
                # Update objectives but keep Thompson state
                return ParetoPoint(
                    lambdas=lambdas,
                    objectives=objectives,
                    persona_id=persona_id,
                    thompson_alpha=existing.thompson_alpha,
                    thompson_beta=existing.thompson_beta,
                    selection_count=existing.selection_count,
                    success_count=existing.success_count,
                    created_at=existing.created_at,
                )

        # Create new point
        return ParetoPoint(
            lambdas=lambdas,
            objectives=objectives,
            persona_id=persona_id,
        )

    def _build_selection_reasoning(
        self,
        winner: ParetoPoint,
        thompson_sample: float,
        preference: RoutingPreference,
        viable_count: int,
        total_front: int,
    ) -> str:
        """Build human-readable selection reasoning."""
        persona = self.personas.get(winner.persona_id)
        persona_name = persona.name if persona else winner.persona_id

        parts = [
            f"Selected '{persona_name}' from Pareto front "
            f"({viable_count}/{total_front} viable points).",
        ]

        # Describe winning objectives
        obj = winner.objectives
        parts.append(
            f"Objectives: SNR={obj.snr:.3f}, novelty={obj.novelty:.3f}, "
            f"alignment={obj.alignment:.3f}, ethics={obj.ethics:.3f}."
        )

        # Thompson sampling info
        parts.append(
            f"Thompson sample={thompson_sample:.3f} "
            f"(mean={winner.thompson_mean:.3f}, "
            f"selections={winner.selection_count})."
        )

        # Preference impact
        pref_weights = preference.to_normalized_array()
        max_weight_idx = pref_weights.index(max(pref_weights))
        objective_names = ["SNR", "novelty", "alignment", "ethics", "speed"]
        parts.append(f"Preference emphasized: {objective_names[max_weight_idx]}.")

        return " ".join(parts)

    def get_front_statistics(self) -> Dict[str, Any]:
        """Get statistics about current Pareto front."""
        if not self.pareto_front:
            return {"status": "empty", "front_size": 0}

        # Aggregate statistics
        personas_in_front = set(p.persona_id for p in self.pareto_front)

        obj_arrays = [p.objectives.to_array() for p in self.pareto_front]

        if NUMPY_AVAILABLE:
            arr = np.array(obj_arrays)
            means = arr.mean(axis=0).tolist()
            stds = arr.std(axis=0).tolist()
        else:
            means = [
                sum(obj[i] for obj in obj_arrays) / len(obj_arrays) for i in range(5)
            ]
            stds = [0.0] * 5  # Skip std without numpy

        return {
            "status": "computed",
            "front_size": len(self.pareto_front),
            "personas_represented": list(personas_in_front),
            "objective_means": {
                "snr": means[0],
                "novelty": means[1],
                "alignment": means[2],
                "ethics": means[3],
                "latency_inv": means[4],
            },
            "objective_stds": {
                "snr": stds[0],
                "novelty": stds[1],
                "alignment": stds[2],
                "ethics": stds[3],
                "latency_inv": stds[4],
            },
            "selection_history_size": len(self._selection_history),
        }

    def to_json(self) -> str:
        """Serialize state to JSON."""
        state = {
            "version": VERSION,
            "domain": DOMAIN_PREFIX,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pareto_front": [p.to_dict() for p in self.pareto_front],
            "all_points_count": len(self._all_evaluated_points),
            "selection_history_count": len(self._selection_history),
            # Store Thompson state for all evaluated points
            "point_states": [
                {
                    "point_id": p.point_id,
                    "lambdas": list(p.lambdas),
                    "persona_id": p.persona_id,
                    "thompson_alpha": p.thompson_alpha,
                    "thompson_beta": p.thompson_beta,
                    "selection_count": p.selection_count,
                    "success_count": p.success_count,
                }
                for p in self._all_evaluated_points
            ],
        }
        return json.dumps(state, indent=2)

    def _save_state(self) -> None:
        """Save state to persistence path."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_path.write_text(self.to_json(), encoding="utf-8")
            logger.debug(f"Saved Pareto router state to {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to save Pareto router state: {e}")

    def _load_state(self) -> None:
        """Load state from persistence path."""
        if not self.persistence_path.exists():
            return

        try:
            json_str = self.persistence_path.read_text(encoding="utf-8")
            data = json.loads(json_str)

            # Restore point states (Thompson parameters)
            for state in data.get("point_states", []):
                point = ParetoPoint(
                    lambdas=tuple(state["lambdas"]),
                    objectives=ObjectiveVector(0, 0, 0, 0, 0),  # Placeholder
                    persona_id=state["persona_id"],
                    thompson_alpha=state.get("thompson_alpha", DEFAULT_THOMPSON_ALPHA),
                    thompson_beta=state.get("thompson_beta", DEFAULT_THOMPSON_BETA),
                    selection_count=state.get("selection_count", 0),
                    success_count=state.get("success_count", 0),
                )
                self._all_evaluated_points.append(point)

            logger.info(
                f"Loaded Pareto router state from {self.persistence_path}: "
                f"{len(self._all_evaluated_points)} points restored"
            )
        except Exception as e:
            logger.warning(f"Failed to load Pareto router state: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_bizra_pareto_router(
    random_seed: Optional[int] = None,
    evaluator: Optional[ObjectiveEvaluator] = None,
    persistence_path: Optional[Path] = None,
) -> ParetoOptimalRouter:
    """
    Create a ParetoOptimalRouter pre-configured with standard BIZRA personas.

    This is the recommended entry point for creating a Pareto router with
    the full PAT/SAT persona set from the default registry.

    Args:
        random_seed: Optional seed for deterministic sampling
        evaluator: Custom objective evaluator (uses default if None)
        persistence_path: Custom persistence path (uses default if None)

    Returns:
        ParetoOptimalRouter instance with standard BIZRA personas

    Example:
        >>> router = create_bizra_pareto_router(random_seed=42)
        >>> router.compute_pareto_front(task_domains=["reasoning", "security"])
        >>> result = router.select_from_pareto(
        ...     preference=RoutingPreference.quality_focused()
        ... )
        >>> print(f"Selected: {result.selected_point.persona_id}")
        >>> print(f"SNR: {result.selected_point.objectives.snr:.3f}")
    """
    registry = create_default_registry()
    personas = registry.list_all()

    return ParetoOptimalRouter(
        personas=personas,
        evaluator=evaluator,
        persistence_path=persistence_path,
        random_seed=random_seed,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "DOMAIN_PREFIX",
    "VERSION",
    "DEFAULT_SNR_THRESHOLD",
    "DEFAULT_IHSAN_THRESHOLD",
    "DEFAULT_NOVELTY_THRESHOLD",
    # Enums
    "ObjectiveName",
    # Core Data Classes
    "ObjectiveVector",
    "ParetoPoint",
    "RoutingPreference",
    "ParetoSelectionResult",
    # Router
    "ParetoOptimalRouter",
    # Evaluator
    "ObjectiveEvaluator",
    "DefaultObjectiveEvaluator",
    # Factory
    "create_bizra_pareto_router",
]


# =============================================================================
# CLI / TESTING
# =============================================================================


def main() -> None:
    """Test Pareto-optimal router functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Pareto-Optimal Router Testing")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["reasoning", "planning"],
        help="Task domains to test",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of Dirichlet samples per persona",
    )
    parser.add_argument(
        "--preference",
        choices=["balanced", "quality", "speed", "novelty"],
        default="balanced",
        help="Selection preference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--simulate",
        type=int,
        default=0,
        help="Simulate N selection-outcome cycles",
    )

    args = parser.parse_args()

    # Create router
    router = create_bizra_pareto_router(random_seed=args.seed)

    # Compute Pareto front
    print(f"\nComputing Pareto front for domains: {args.domains}")
    print(f"Samples per persona: {args.samples}")

    front = router.compute_pareto_front(
        task_domains=args.domains,
        num_samples=args.samples,
    )

    print(f"\nPareto front size: {len(front)}")

    # Show statistics
    stats = router.get_front_statistics()
    print(f"Personas in front: {stats['personas_represented']}")
    print(f"Objective means: {stats['objective_means']}")

    # Select with preference
    preference_map = {
        "balanced": RoutingPreference.balanced(),
        "quality": RoutingPreference.quality_focused(),
        "speed": RoutingPreference.speed_focused(),
        "novelty": RoutingPreference.novelty_focused(),
    }
    preference = preference_map[args.preference]

    print(f"\nSelecting with preference: {args.preference}")
    result = router.select_from_pareto(preference=preference)

    print("\nSelection Result:")
    print(f"  Persona: {result.selected_point.persona_id}")
    print(f"  Thompson sample: {result.thompson_sample:.4f}")
    print(f"  Preference score: {result.preference_score:.4f}")
    print("  Objectives:")
    obj = result.selected_point.objectives
    print(f"    SNR: {obj.snr:.4f}")
    print(f"    Novelty: {obj.novelty:.4f}")
    print(f"    Alignment: {obj.alignment:.4f}")
    print(f"    Ethics: {obj.ethics:.4f}")
    print(f"    Latency_inv: {obj.latency_inv:.4f}")
    print(f"\nReasoning: {result.reasoning}")

    # Simulate if requested
    if args.simulate > 0:
        print(f"\n{'='*60}")
        print(f"Simulating {args.simulate} selection-outcome cycles...")

        for i in range(args.simulate):
            # Compute fresh front
            router.compute_pareto_front(
                task_domains=args.domains,
                num_samples=args.samples,
            )

            # Select
            result = router.select_from_pareto(preference=preference)

            # Simulate outcome (biased by objective quality)
            obj = result.selected_point.objectives
            success_prob = (obj.snr + obj.ethics) / 2
            if NUMPY_AVAILABLE:
                success = router.rng.random() < success_prob
            else:
                success = router.rng.random() < success_prob

            actual_snr = obj.snr + (0.02 if success else -0.02)
            actual_snr = max(0.0, min(1.0, actual_snr))

            # Update
            router.update_from_outcome(
                lambdas=result.selected_point.lambdas,
                success=success,
                actual_snr=actual_snr,
            )

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{args.simulate} cycles")

        print("\nFinal front statistics:")
        final_stats = router.get_front_statistics()
        print(f"  Objective means: {final_stats['objective_means']}")

        # Show Thompson state for top points
        print("\nTop points by Thompson mean:")
        top_points = sorted(
            router._all_evaluated_points,
            key=lambda p: p.thompson_mean,
            reverse=True,
        )[:5]
        for p in top_points:
            print(
                f"  {p.persona_id}: mean={p.thompson_mean:.3f}, "
                f"selections={p.selection_count}, "
                f"successes={p.success_count}"
            )


if __name__ == "__main__":
    main()
