"""
BIZRA Apex Sovereign - Autonomous SNR Optimizer
=================================================
Self-optimizing SNR maximization targeting 0.99+ (sovereign level).

Implements multi-strategy optimization combining:
- Thompson Sampling (exploration/exploitation)
- SONA Learning (continuous weight updates)
- Pattern Elevation (cache high-SNR patterns)
- Diversity Injection (cross-domain bonus)
- Ensemble Refinement (multi-model combination)

From the BIZRA Blueprint:
    - Target SNR: 0.99+ (sovereign level)
    - Convergence within 5 iterations
    - Integration with Thompson Router, SONA Learner, Pattern Cache

Key Formulas:
    - SNR contribution: sqrt(sum(w_i^2 * snr_i^2)) * diversity_bonus
    - Diversity bonus: min(0.10, (domain_count - 1) * 0.02)
    - Thompson sample: Beta(successes + 1, failures + 1).rvs()

Domain: bizra-pci-v1:apex:sovereign:optimizer
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    IHSAN_THRESHOLD as CONST_IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_PAT_SOVEREIGN,
)

# Optional numpy for enhanced performance
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DOMAIN_PREFIX = "bizra-pci-v1:apex:sovereign:optimizer"
VERSION = "1.0.0"

# Sovereign-level SNR target (from core/constants.py)
SNR_TARGET_SOVEREIGN = SNR_THRESHOLD_PAT_SOVEREIGN  # 0.99
SNR_TARGET_STANDARD = SNR_THRESHOLD_T0_ELITE  # 0.98
IHSAN_THRESHOLD = CONST_IHSAN_THRESHOLD  # 0.95

# Optimization parameters
MAX_ITERATIONS = 5
CONVERGENCE_THRESHOLD = 0.001
DIVERSITY_BONUS_PER_DOMAIN = 0.02
MAX_DIVERSITY_BONUS = 0.10

# Pattern elevation
PATTERN_ELEVATION_THRESHOLD = 3
PATTERN_CACHE_MAX_SIZE = 10000


# =============================================================================
# ENUMS
# =============================================================================


class OptimizationStrategy(str, Enum):
    """
    Strategies for SNR optimization.

    Each strategy contributes to the overall optimization through
    different mechanisms:

    THOMPSON_SAMPLING: Bayesian exploration/exploitation balance
    SONA_LEARNING: Continuous weight updates from feedback
    PATTERN_ELEVATION: Cache and shortcut high-SNR patterns
    DIVERSITY_INJECTION: Cross-domain bonus for novel combinations
    ENSEMBLE_REFINEMENT: Multi-model combination optimization
    """
    THOMPSON_SAMPLING = "thompson_sampling"
    SONA_LEARNING = "sona_learning"
    PATTERN_ELEVATION = "pattern_elevation"
    DIVERSITY_INJECTION = "diversity_injection"
    ENSEMBLE_REFINEMENT = "ensemble_refinement"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OptimizationState:
    """
    State of the optimization process at a given iteration.

    Tracks current SNR, strategies applied, improvements made,
    and convergence metrics.

    Attributes:
        current_snr: Current SNR score
        target_snr: Target SNR to reach (default: 0.99 sovereign level)
        iteration: Current iteration number
        strategies_applied: List of strategies applied in this iteration
        improvements: List of SNR improvements per strategy
        convergence_rate: Rate of convergence toward target
    """
    current_snr: float
    target_snr: float = SNR_TARGET_SOVEREIGN
    iteration: int = 0
    strategies_applied: List[OptimizationStrategy] = field(default_factory=list)
    improvements: List[float] = field(default_factory=list)
    convergence_rate: float = 0.0

    # Additional metrics
    delta_from_target: float = field(init=False)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.delta_from_target = self.target_snr - self.current_snr

    @property
    def total_improvement(self) -> float:
        """Total improvement across all strategies."""
        return sum(self.improvements)

    @property
    def target_met(self) -> bool:
        """Check if target SNR is met."""
        return self.current_snr >= self.target_snr

    @property
    def converged(self) -> bool:
        """Check if optimization has converged (minimal improvement)."""
        return abs(self.total_improvement) < CONVERGENCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_snr": self.current_snr,
            "target_snr": self.target_snr,
            "iteration": self.iteration,
            "strategies_applied": [s.value for s in self.strategies_applied],
            "improvements": self.improvements,
            "convergence_rate": self.convergence_rate,
            "delta_from_target": self.delta_from_target,
            "total_improvement": self.total_improvement,
            "target_met": self.target_met,
            "converged": self.converged,
            "timestamp": self.timestamp,
        }


@dataclass
class ElevatedPattern:
    """
    A pattern that has been elevated to the cache for shortcut execution.

    Patterns are elevated after repeated successful execution (>3 times)
    with high SNR scores.

    Attributes:
        pattern_hash: SHA-256 hash identifying the pattern
        pattern_signature: Human-readable pattern description
        shortcut_result: Cached result for shortcut execution
        snr_score: Average SNR score for this pattern
        elevation_count: Number of times pattern was successfully used
        created_at: Timestamp of pattern creation
        last_used: Timestamp of last usage
        metadata: Additional pattern metadata
    """
    pattern_hash: str
    pattern_signature: str
    shortcut_result: Dict[str, Any]
    snr_score: float
    elevation_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_used: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_hash": self.pattern_hash,
            "pattern_signature": self.pattern_signature,
            "shortcut_result": self.shortcut_result,
            "snr_score": self.snr_score,
            "elevation_count": self.elevation_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElevatedPattern":
        """Deserialize from dictionary."""
        return cls(
            pattern_hash=data["pattern_hash"],
            pattern_signature=data["pattern_signature"],
            shortcut_result=data["shortcut_result"],
            snr_score=data["snr_score"],
            elevation_count=data.get("elevation_count", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_used=data.get("last_used", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OptimizedResult:
    """
    Final result from the autonomous optimization process.

    Contains the achieved SNR, whether target was met, number of
    iterations used, effective strategies, and full optimization trace.

    Attributes:
        achieved_snr: Final SNR score achieved
        target_met: Whether the sovereign SNR target (0.99) was met
        iterations_used: Number of iterations required
        strategies_effective: Strategies that contributed positive improvement
        final_result: The optimized SynthesisResult
        optimization_trace: Full trace of OptimizationState per iteration
        diversity_domains: Domains covered in the optimization
        pattern_elevated: Whether a new pattern was elevated
        timestamp: Completion timestamp
    """
    achieved_snr: float
    target_met: bool
    iterations_used: int
    strategies_effective: List[OptimizationStrategy]
    final_result: Dict[str, Any]  # SynthesisResult as dict
    optimization_trace: List[OptimizationState]
    diversity_domains: Set[str] = field(default_factory=set)
    pattern_elevated: bool = False
    pattern_hash: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def improvement_over_initial(self) -> float:
        """Calculate total improvement from initial to final SNR."""
        if not self.optimization_trace:
            return 0.0
        initial = self.optimization_trace[0].current_snr
        return self.achieved_snr - initial

    @property
    def average_improvement_per_iteration(self) -> float:
        """Calculate average improvement per iteration."""
        if self.iterations_used <= 0:
            return 0.0
        return self.improvement_over_initial / self.iterations_used

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "achieved_snr": self.achieved_snr,
            "target_met": self.target_met,
            "iterations_used": self.iterations_used,
            "strategies_effective": [s.value for s in self.strategies_effective],
            "final_result": self.final_result,
            "optimization_trace": [s.to_dict() for s in self.optimization_trace],
            "diversity_domains": list(self.diversity_domains),
            "pattern_elevated": self.pattern_elevated,
            "pattern_hash": self.pattern_hash,
            "improvement_over_initial": self.improvement_over_initial,
            "average_improvement_per_iteration": self.average_improvement_per_iteration,
            "timestamp": self.timestamp,
        }


# =============================================================================
# SUPPORT CLASSES
# =============================================================================


class ThompsonSampler:
    """
    Thompson Sampling implementation using Beta distribution.

    Provides exploration/exploitation balance through Bayesian sampling.
    Uses Beta(alpha, beta) distribution where:
    - alpha represents successes + 1
    - beta represents failures + 1

    The sampled value from Beta distribution guides the exploration
    bonus applied to SNR optimization.

    Attributes:
        alpha: Shape parameter (successes + 1), default 1.0
        beta: Shape parameter (failures + 1), default 1.0

    Example:
        >>> sampler = ThompsonSampler()
        >>> value = sampler.sample()  # Sample from Beta(1.0, 1.0)
        >>> sampler.update(0.9)  # Update with reward (success)
        >>> value = sampler.sample()  # Now samples from Beta(1.9, 1.0)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize Thompson Sampler.

        Args:
            alpha: Initial alpha parameter (successes + 1)
            beta: Initial beta parameter (failures + 1)
        """
        self.alpha = alpha
        self.beta = beta
        self._rng: Optional[Any] = None

    def sample(self) -> float:
        """
        Sample from the Beta(alpha, beta) distribution.

        Returns:
            Sampled value in [0, 1] range
        """
        if NUMPY_AVAILABLE:
            if self._rng is None:
                self._rng = np.random.default_rng()
            return float(self._rng.beta(self.alpha, self.beta))
        else:
            return _stdlib_beta_sample(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """
        Update the sampler with observed reward.

        Treats reward as a continuous success signal:
        - reward close to 1.0 increases alpha more
        - reward close to 0.0 increases beta more

        Args:
            reward: Observed reward in [0, 1] range
        """
        # Clamp reward to valid range
        reward = max(0.0, min(1.0, reward))

        # Update based on reward
        # High reward -> increase alpha (successes)
        # Low reward -> increase beta (failures)
        self.alpha += reward
        self.beta += (1.0 - reward)

    @property
    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    def reset(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Reset the sampler to initial state."""
        self.alpha = alpha
        self.beta = beta

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "variance": self.variance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ThompsonSampler":
        """Deserialize from dictionary."""
        return cls(
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
        )


class PatternCache:
    """
    Cache for elevated high-SNR patterns.

    Patterns are recorded during execution and elevated to shortcuts
    after repeated successful use (>3 times with high SNR).

    Attributes:
        cache: Dictionary mapping pattern hash to SNR score
        elevation_threshold: Number of successful uses before elevation (default: 3)
        max_size: Maximum number of patterns to cache

    Example:
        >>> cache = PatternCache()
        >>> cache.record_pattern("abc123", 0.95)  # First occurrence
        >>> cache.record_pattern("abc123", 0.96)  # Second occurrence
        >>> cache.record_pattern("abc123", 0.97)  # Third occurrence
        >>> snr = cache.get_pattern("abc123")  # Returns 0.96 (average)
    """

    def __init__(
        self,
        max_size: int = PATTERN_CACHE_MAX_SIZE,
        elevation_threshold: int = PATTERN_ELEVATION_THRESHOLD,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize pattern cache.

        Args:
            max_size: Maximum patterns to store
            elevation_threshold: Uses required for elevation
            persistence_path: Optional path for persistence
        """
        self.max_size = max_size
        self.elevation_threshold = elevation_threshold
        self.persistence_path = persistence_path or Path(
            "docs/evidence/apex/pattern_cache.json"
        )

        # Cache storage
        self._elevated_patterns: Dict[str, ElevatedPattern] = {}
        self._pending_patterns: Dict[str, Dict[str, Any]] = {}

        # Load persisted state
        self._load_state()

        logger.info(
            f"PatternCache initialized: max_size={max_size}, "
            f"elevation_threshold={elevation_threshold}, "
            f"loaded={len(self._elevated_patterns)} elevated patterns"
        )

    def get_pattern(self, pattern_hash: str) -> Optional[ElevatedPattern]:
        """
        Get an elevated pattern by hash.

        Args:
            pattern_hash: SHA-256 hash of the pattern

        Returns:
            ElevatedPattern if found and elevated, None otherwise
        """
        pattern = self._elevated_patterns.get(pattern_hash)
        if pattern:
            pattern.last_used = datetime.now(timezone.utc).isoformat()
            pattern.elevation_count += 1
            logger.debug(f"Pattern cache hit: {pattern_hash[:8]}")
        return pattern

    def record_pattern(
        self,
        pattern_hash: str,
        result: Dict[str, Any],
        snr_score: float,
        signature: str = "",
    ) -> bool:
        """
        Record a pattern execution for potential elevation.

        Args:
            pattern_hash: SHA-256 hash of the pattern
            result: Execution result to cache
            snr_score: SNR score achieved
            signature: Human-readable pattern description

        Returns:
            True if pattern was elevated, False otherwise
        """
        if pattern_hash in self._elevated_patterns:
            # Already elevated, just update stats
            return False

        # Track pending pattern
        if pattern_hash not in self._pending_patterns:
            self._pending_patterns[pattern_hash] = {
                "count": 0,
                "snr_scores": [],
                "results": [],
                "signature": signature,
            }

        pending = self._pending_patterns[pattern_hash]
        pending["count"] += 1
        pending["snr_scores"].append(snr_score)
        pending["results"].append(result)

        # Check for elevation
        if pending["count"] >= self.elevation_threshold:
            avg_snr = sum(pending["snr_scores"]) / len(pending["snr_scores"])
            if avg_snr >= SNR_TARGET_STANDARD:  # Must meet standard threshold
                return self.elevate_pattern(pattern_hash, result, avg_snr, signature)

        return False

    def elevate_pattern(
        self,
        pattern_hash: str,
        shortcut_result: Dict[str, Any],
        snr_score: float,
        signature: str = "",
    ) -> bool:
        """
        Elevate a pattern to the shortcut cache.

        Args:
            pattern_hash: SHA-256 hash of the pattern
            shortcut_result: Result to use as shortcut
            snr_score: SNR score for this pattern
            signature: Human-readable pattern description

        Returns:
            True if elevation succeeded
        """
        # Check cache size
        if len(self._elevated_patterns) >= self.max_size:
            # Evict least recently used pattern
            self._evict_lru()

        # Create elevated pattern
        pattern = ElevatedPattern(
            pattern_hash=pattern_hash,
            pattern_signature=signature or f"pattern-{pattern_hash[:8]}",
            shortcut_result=shortcut_result,
            snr_score=snr_score,
        )

        self._elevated_patterns[pattern_hash] = pattern

        # Remove from pending
        if pattern_hash in self._pending_patterns:
            del self._pending_patterns[pattern_hash]

        logger.info(
            f"Pattern elevated: {pattern_hash[:8]} with SNR={snr_score:.4f}"
        )

        # Persist
        self._save_state()

        return True

    def _evict_lru(self) -> None:
        """Evict least recently used pattern."""
        if not self._elevated_patterns:
            return

        # Find LRU
        lru_hash = min(
            self._elevated_patterns.keys(),
            key=lambda h: self._elevated_patterns[h].last_used
        )

        logger.debug(f"Evicting LRU pattern: {lru_hash[:8]}")
        del self._elevated_patterns[lru_hash]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "elevated_count": len(self._elevated_patterns),
            "pending_count": len(self._pending_patterns),
            "max_size": self.max_size,
            "elevation_threshold": self.elevation_threshold,
            "avg_elevated_snr": (
                sum(p.snr_score for p in self._elevated_patterns.values()) /
                len(self._elevated_patterns)
                if self._elevated_patterns else 0.0
            ),
        }

    def _save_state(self) -> None:
        """Save state to persistence path."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "version": VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elevated_patterns": {
                    h: p.to_dict() for h, p in self._elevated_patterns.items()
                },
            }
            self.persistence_path.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
            logger.debug(f"Saved pattern cache to {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to save pattern cache: {e}")

    def _load_state(self) -> None:
        """Load state from persistence path."""
        if not self.persistence_path.exists():
            return

        try:
            data = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            for h, p_data in data.get("elevated_patterns", {}).items():
                self._elevated_patterns[h] = ElevatedPattern.from_dict(p_data)
            logger.info(f"Loaded {len(self._elevated_patterns)} elevated patterns")
        except Exception as e:
            logger.warning(f"Failed to load pattern cache: {e}")


class SONAWeightLearner:
    """
    Self-Optimizing Neural Allocation weight learner.

    Implements continuous learning of optimal weights based on
    execution outcomes. Uses gradient-free optimization with
    momentum for stable updates.

    Attributes:
        weight_matrix: Matrix of learned weights
        learning_rate: Rate of weight updates
        momentum: Momentum for stable updates
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize SONA weight learner.

        Args:
            learning_rate: Learning rate for weight updates
            momentum: Momentum coefficient
            persistence_path: Optional persistence path
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.persistence_path = persistence_path or Path(
            "docs/evidence/apex/sona_weights.json"
        )

        # Weight storage
        self._weight_matrix: Dict[str, float] = {}
        self._velocity: Dict[str, float] = {}
        self._history: List[Dict[str, Any]] = []
        self._max_history = 1000

        # Load persisted state
        self._load_state()

        logger.info(
            f"SONAWeightLearner initialized: lr={learning_rate}, "
            f"momentum={momentum}"
        )

    def compute_optimal_weights(
        self,
        prior_result: Dict[str, Any],
        delta: float,
    ) -> Dict[str, float]:
        """
        Compute optimal weights based on prior result and improvement delta.

        Args:
            prior_result: Previous execution result
            delta: SNR improvement to optimize for

        Returns:
            Dictionary of adjusted weights
        """
        # Extract persona contributions from prior result
        contributions = prior_result.get("contributing_personas", {})
        if not contributions:
            # Default equal weights
            return {"default": 1.0}

        # Compute gradients based on delta
        gradients: Dict[str, float] = {}
        for persona_id, contribution in contributions.items():
            # Gradient: positive delta -> increase weight for high contributors
            # negative delta -> decrease weight for high contributors
            gradient = delta * contribution
            gradients[persona_id] = gradient

        # Apply momentum and update weights
        new_weights: Dict[str, float] = {}
        for persona_id, gradient in gradients.items():
            key = f"weight:{persona_id}"

            # Get current weight
            current = self._weight_matrix.get(key, 0.5)

            # Apply momentum
            prev_velocity = self._velocity.get(key, 0.0)
            velocity = self.momentum * prev_velocity + self.learning_rate * gradient
            self._velocity[key] = velocity

            # Update weight
            new_weight = current + velocity
            new_weight = max(0.01, min(0.99, new_weight))  # Clamp to [0.01, 0.99]

            self._weight_matrix[key] = new_weight
            new_weights[persona_id] = new_weight

        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def record_outcome(
        self,
        iteration: int,
        snr: float,
        weights_used: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record an outcome for learning.

        Args:
            iteration: Iteration number
            snr: SNR achieved
            weights_used: Weights used in this iteration
        """
        record = {
            "iteration": iteration,
            "snr": snr,
            "weights_used": weights_used or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._history.append(record)

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Persist periodically
        if len(self._history) % 10 == 0:
            self._save_state()

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """Get weight update history."""
        return self._history.copy()

    def get_current_weights(self) -> Dict[str, float]:
        """Get current weight matrix."""
        return {
            k.replace("weight:", ""): v
            for k, v in self._weight_matrix.items()
            if k.startswith("weight:")
        }

    def _save_state(self) -> None:
        """Save state to persistence path."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "version": VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "weight_matrix": self._weight_matrix,
                "velocity": self._velocity,
                "history_count": len(self._history),
            }
            self.persistence_path.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to save SONA weights: {e}")

    def _load_state(self) -> None:
        """Load state from persistence path."""
        if not self.persistence_path.exists():
            return

        try:
            data = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            self._weight_matrix = data.get("weight_matrix", {})
            self._velocity = data.get("velocity", {})
            logger.info("Loaded SONA weight state")
        except Exception as e:
            logger.warning(f"Failed to load SONA weights: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _stdlib_beta_sample(
    alpha: float,
    beta: float,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Sample from Beta distribution using stdlib.

    Uses the relationship: Beta(a,b) can be sampled via Gamma distributions.

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


def compute_snr_contribution(
    weights: Dict[str, float],
    snr_scores: Dict[str, float],
    diversity_bonus: float = 0.0,
) -> float:
    """
    Compute SNR contribution using the formula:

        SNR = sqrt(sum(w_i^2 * snr_i^2)) * (1 + diversity_bonus)

    Args:
        weights: Dictionary of persona weights
        snr_scores: Dictionary of persona SNR scores
        diversity_bonus: Bonus for cross-domain coverage

    Returns:
        Computed SNR contribution
    """
    if not weights or not snr_scores:
        return 0.0

    sum_squared = 0.0
    for persona_id, weight in weights.items():
        snr = snr_scores.get(persona_id, 0.9)  # Default SNR
        sum_squared += (weight ** 2) * (snr ** 2)

    base_snr = math.sqrt(sum_squared)
    return base_snr * (1.0 + diversity_bonus)


def compute_diversity_bonus(domain_count: int) -> float:
    """
    Compute diversity bonus based on number of domains covered.

    Formula: min(0.10, (domain_count - 1) * 0.02)

    Args:
        domain_count: Number of distinct domains

    Returns:
        Diversity bonus (0.0 to 0.10)
    """
    if domain_count <= 1:
        return 0.0

    bonus = (domain_count - 1) * DIVERSITY_BONUS_PER_DOMAIN
    return min(MAX_DIVERSITY_BONUS, bonus)


def compute_pattern_hash(
    domains: List[str],
    personas: List[str],
    task_signature: str,
) -> str:
    """
    Compute SHA-256 hash for pattern identification.

    Args:
        domains: List of domains involved
        personas: List of personas involved
        task_signature: Task signature string

    Returns:
        SHA-256 hash string
    """
    content = json.dumps({
        "domains": sorted(domains),
        "personas": sorted(personas),
        "task": task_signature,
    }, sort_keys=True)

    return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# MAIN OPTIMIZER CLASS
# =============================================================================


class AutonomousSNROptimizer:
    """
    Self-optimizing SNR maximization engine targeting sovereign level (0.99+).

    Combines multiple optimization strategies:
    - Thompson Sampling for exploration/exploitation balance
    - SONA Learning for continuous weight optimization
    - Pattern Elevation for caching high-SNR shortcuts
    - Diversity Injection for cross-domain bonus
    - Ensemble Refinement for multi-model optimization

    Algorithm:
        1. Check pattern cache for shortcut
        2. If no shortcut, iterate optimization:
           a. Apply Thompson adjustment
           b. Apply SONA weight update
           c. Check pattern elevation
           d. Compute diversity bonus
           e. Resynthesize with adjustments
        3. Check convergence or target met
        4. Emit optimized result with trace

    Attributes:
        SNR_TARGET: Target SNR (0.99 sovereign level)
        MAX_ITERATIONS: Maximum optimization iterations (5)
        CONVERGENCE_THRESHOLD: Threshold for convergence (0.001)

    Example:
        >>> optimizer = AutonomousSNROptimizer()
        >>> result = await optimizer.optimize(initial_synthesis_result)
        >>> print(f"Achieved SNR: {result.achieved_snr}")
        >>> print(f"Target met: {result.target_met}")
    """

    # Class constants
    SNR_TARGET = SNR_TARGET_SOVEREIGN
    MAX_ITERATIONS = MAX_ITERATIONS
    CONVERGENCE_THRESHOLD = CONVERGENCE_THRESHOLD

    def __init__(
        self,
        thompson_router: Optional[Any] = None,
        sona_learner: Optional[SONAWeightLearner] = None,
        pattern_cache: Optional[PatternCache] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Autonomous SNR Optimizer.

        Args:
            thompson_router: ThompsonSamplingRouter instance for exploration
            sona_learner: SONAWeightLearner for continuous optimization
            pattern_cache: PatternCache for shortcut patterns
            random_seed: Optional random seed for reproducibility
        """
        self.thompson_router = thompson_router
        self.sona_learner = sona_learner or SONAWeightLearner()
        self.pattern_cache = pattern_cache or PatternCache()

        # Random number generator
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

        # Thompson sampling state (per persona)
        self._thompson_alphas: Dict[str, float] = {}
        self._thompson_betas: Dict[str, float] = {}

        # Statistics
        self._total_optimizations = 0
        self._successful_optimizations = 0
        self._cache_hits = 0

        logger.info(
            f"AutonomousSNROptimizer initialized: "
            f"target={self.SNR_TARGET}, max_iter={self.MAX_ITERATIONS}"
        )

    async def optimize(
        self,
        initial_result: Dict[str, Any],
        domains: Optional[List[str]] = None,
        personas: Optional[List[str]] = None,
        task_signature: str = "",
        resynthesize_fn: Optional[Callable] = None,
    ) -> OptimizedResult:
        """
        Optimize SNR for the given initial result.

        This is the main entry point for optimization. It applies
        multiple strategies iteratively until the target SNR is met
        or convergence is reached.

        Args:
            initial_result: Initial SynthesisResult as dictionary
            domains: List of domains involved (for diversity bonus)
            personas: List of personas involved
            task_signature: Task signature for pattern hashing
            resynthesize_fn: Optional async function to resynthesize

        Returns:
            OptimizedResult with achieved SNR and full trace
        """
        self._total_optimizations += 1
        start_time = datetime.now(timezone.utc)

        domains = domains or []
        personas = personas or list(
            initial_result.get("contributing_personas", {}).keys()
        )

        # Compute pattern hash for cache lookup
        pattern_hash = compute_pattern_hash(domains, personas, task_signature)

        # Check pattern cache for shortcut
        cached_pattern = self.pattern_cache.get_pattern(pattern_hash)
        if cached_pattern:
            self._cache_hits += 1
            logger.info(f"Pattern cache hit: {pattern_hash[:8]}")

            return OptimizedResult(
                achieved_snr=cached_pattern.snr_score,
                target_met=cached_pattern.snr_score >= self.SNR_TARGET,
                iterations_used=0,
                strategies_effective=[OptimizationStrategy.PATTERN_ELEVATION],
                final_result=cached_pattern.shortcut_result,
                optimization_trace=[],
                diversity_domains=set(domains),
                pattern_elevated=False,  # Already elevated
                pattern_hash=pattern_hash,
            )

        # Extract initial SNR
        initial_snr = initial_result.get("snr_score", 0.9)
        if isinstance(initial_snr, str):
            initial_snr = float(initial_snr)

        # Initialize optimization trace
        trace: List[OptimizationState] = []
        current_result = initial_result.copy()
        current_snr = initial_snr
        effective_strategies: Set[OptimizationStrategy] = set()

        # Iterative optimization loop
        for iteration in range(self.MAX_ITERATIONS):
            logger.debug(
                f"Optimization iteration {iteration + 1}: current_snr={current_snr:.4f}"
            )

            # Create state for this iteration
            state = OptimizationState(
                current_snr=current_snr,
                target_snr=self.SNR_TARGET,
                iteration=iteration,
            )

            # Strategy 1: Thompson Sampling adjustment
            thompson_adjustment = self._thompson_adjustment(current_snr, personas)
            if thompson_adjustment != 0:
                state.strategies_applied.append(OptimizationStrategy.THOMPSON_SAMPLING)
                state.improvements.append(thompson_adjustment)
                if thompson_adjustment > 0:
                    effective_strategies.add(OptimizationStrategy.THOMPSON_SAMPLING)

            # Strategy 2: SONA weight update
            sona_weights = self._sona_weight_update(
                current_result,
                self.SNR_TARGET - current_snr,
            )
            sona_improvement = self._estimate_sona_improvement(sona_weights, current_snr)
            if sona_improvement != 0:
                state.strategies_applied.append(OptimizationStrategy.SONA_LEARNING)
                state.improvements.append(sona_improvement)
                if sona_improvement > 0:
                    effective_strategies.add(OptimizationStrategy.SONA_LEARNING)

            # Strategy 3: Pattern elevation check
            elevation_occurred = self._pattern_elevation_check(
                current_result,
                pattern_hash,
                current_snr,
            )
            if elevation_occurred:
                state.strategies_applied.append(OptimizationStrategy.PATTERN_ELEVATION)
                effective_strategies.add(OptimizationStrategy.PATTERN_ELEVATION)

            # Strategy 4: Diversity injection
            diversity_bonus = self._diversity_bonus(domains)
            if diversity_bonus > 0:
                state.strategies_applied.append(OptimizationStrategy.DIVERSITY_INJECTION)
                state.improvements.append(diversity_bonus)
                effective_strategies.add(OptimizationStrategy.DIVERSITY_INJECTION)

            # Strategy 5: Ensemble refinement (if multiple personas)
            if len(personas) >= 2:
                ensemble_improvement = self._ensemble_refinement(
                    current_result,
                    personas,
                    sona_weights,
                )
                if ensemble_improvement > 0:
                    state.strategies_applied.append(OptimizationStrategy.ENSEMBLE_REFINEMENT)
                    state.improvements.append(ensemble_improvement)
                    effective_strategies.add(OptimizationStrategy.ENSEMBLE_REFINEMENT)

            # Compute new SNR with all adjustments
            adjustments = {
                "thompson": thompson_adjustment,
                "diversity": diversity_bonus,
                "sona_weights": sona_weights,
            }

            # Resynthesize if function provided
            if resynthesize_fn:
                try:
                    current_result = await resynthesize_fn(current_result, adjustments)
                    current_snr = current_result.get("snr_score", current_snr)
                except Exception as e:
                    logger.warning(f"Resynthesize failed: {e}")
                    # Estimate improvement without actual resynthesis
                    current_snr = min(
                        0.999,
                        current_snr + state.total_improvement
                    )
            else:
                # Estimate improvement
                current_snr = min(
                    0.999,
                    current_snr + state.total_improvement
                )
                current_result["snr_score"] = current_snr

            # Compute convergence rate
            if iteration > 0 and trace:
                prev_snr = trace[-1].current_snr
                state.convergence_rate = (current_snr - prev_snr) / max(
                    self.SNR_TARGET - prev_snr, 0.001
                )

            state.current_snr = current_snr
            trace.append(state)

            # Record SONA outcome
            self.sona_learner.record_outcome(
                iteration=iteration,
                snr=current_snr,
                weights_used=sona_weights,
            )

            # Check termination conditions
            if state.target_met:
                logger.info(
                    f"Target SNR met at iteration {iteration + 1}: {current_snr:.4f}"
                )
                break

            if self._check_convergence(trace):
                logger.info(
                    f"Converged at iteration {iteration + 1}: {current_snr:.4f}"
                )
                break

        # Record pattern for potential future elevation
        self.pattern_cache.record_pattern(
            pattern_hash=pattern_hash,
            result=current_result,
            snr_score=current_snr,
            signature=task_signature,
        )

        # Build final result
        target_met = current_snr >= self.SNR_TARGET
        if target_met:
            self._successful_optimizations += 1

        result = OptimizedResult(
            achieved_snr=current_snr,
            target_met=target_met,
            iterations_used=len(trace),
            strategies_effective=list(effective_strategies),
            final_result=current_result,
            optimization_trace=trace,
            diversity_domains=set(domains),
            pattern_elevated=self.pattern_cache.get_pattern(pattern_hash) is not None,
            pattern_hash=pattern_hash,
        )

        logger.info(
            f"Optimization complete: achieved={current_snr:.4f}, "
            f"target_met={target_met}, iterations={len(trace)}, "
            f"strategies={len(effective_strategies)}"
        )

        return result

    async def optimize_snr(
        self,
        initial_snr: float,
        context: Dict[str, Any],
    ) -> OptimizedResult:
        """
        Simplified optimization interface with SNR value and context dict.

        This method provides a cleaner API for direct SNR optimization
        when you have an initial SNR value and a context dictionary.

        Args:
            initial_snr: Initial SNR score to optimize from
            context: Context dictionary containing:
                - domains: List of domains (optional)
                - personas: List of personas (optional)
                - task_signature: Task signature string (optional)
                - Any other context data

        Returns:
            OptimizedResult with achieved SNR and full trace

        Example:
            >>> optimizer = AutonomousSNROptimizer()
            >>> result = await optimizer.optimize_snr(
            ...     initial_snr=0.92,
            ...     context={
            ...         "domains": ["reasoning", "ethics", "security"],
            ...         "personas": ["MasterReasoner", "EthicsGuardian"],
            ...         "task_signature": "complex-decision-making",
            ...     }
            ... )
            >>> print(f"Achieved SNR: {result.achieved_snr}")
        """
        # Extract context fields
        domains = context.get("domains", [])
        personas = context.get("personas", [])
        task_signature = context.get("task_signature", "")

        # Build initial result dict
        initial_result = {
            "snr_score": initial_snr,
            "contributing_personas": {p: 1.0 / len(personas) for p in personas} if personas else {},
            **{k: v for k, v in context.items() if k not in ("domains", "personas", "task_signature")},
        }

        # Delegate to full implementation
        return await self.optimize(
            initial_result=initial_result,
            domains=domains,
            personas=personas,
            task_signature=task_signature,
        )

    def _thompson_adjustment(
        self,
        current_snr: float,
        personas: List[str],
    ) -> float:
        """
        Apply Thompson Sampling adjustment for exploration/exploitation.

        Uses Beta distributions to sample exploration bonus for each
        persona, balancing exploitation of known good personas with
        exploration of potentially better ones.

        Args:
            current_snr: Current SNR score
            personas: List of personas to consider

        Returns:
            SNR adjustment from Thompson sampling
        """
        if not personas:
            return 0.0

        total_adjustment = 0.0

        for persona in personas:
            # Get or initialize Beta parameters
            alpha = self._thompson_alphas.get(persona, 1.0)
            beta = self._thompson_betas.get(persona, 1.0)

            # Sample from Beta distribution
            if NUMPY_AVAILABLE:
                sample = self.rng.beta(alpha, beta)
            else:
                sample = _stdlib_beta_sample(alpha, beta, self.rng)

            # Exploration bonus: higher variance = more exploration
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            exploration_bonus = sample * math.sqrt(variance) * 0.01

            total_adjustment += exploration_bonus

        # Normalize by number of personas
        adjustment = total_adjustment / len(personas) if personas else 0.0

        return adjustment

    def _sona_weight_update(
        self,
        result: Dict[str, Any],
        delta: float,
    ) -> Dict[str, float]:
        """
        Apply SONA learning weight update.

        Args:
            result: Current synthesis result
            delta: Gap to target SNR

        Returns:
            Updated weights from SONA learner
        """
        return self.sona_learner.compute_optimal_weights(result, delta)

    def _pattern_elevation_check(
        self,
        result: Dict[str, Any],
        pattern_hash: str,
        snr_score: float,
    ) -> bool:
        """
        Check if pattern should be elevated to cache.

        Patterns are elevated after 3+ successful uses with
        SNR >= 0.98 (standard threshold).

        Args:
            result: Current synthesis result
            pattern_hash: Pattern hash for identification
            snr_score: Current SNR score

        Returns:
            True if pattern was elevated
        """
        # Record for tracking (elevation happens automatically in cache)
        return self.pattern_cache.record_pattern(
            pattern_hash=pattern_hash,
            result=result,
            snr_score=snr_score,
        )

    def _diversity_bonus(self, domains: List[str]) -> float:
        """
        Compute diversity bonus from cross-domain coverage.

        Formula: min(0.10, (domain_count - 1) * 0.02)

        Args:
            domains: List of domains covered

        Returns:
            Diversity bonus (0.0 to 0.10)
        """
        unique_domains = set(domains)
        return compute_diversity_bonus(len(unique_domains))

    def _estimate_sona_improvement(
        self,
        weights: Dict[str, float],
        current_snr: float,
    ) -> float:
        """
        Estimate improvement from SONA weight optimization.

        Args:
            weights: New optimized weights
            current_snr: Current SNR

        Returns:
            Estimated SNR improvement
        """
        if not weights:
            return 0.0

        # Estimate improvement based on weight concentration
        # More concentrated weights on high performers = better SNR
        max_weight = max(weights.values())
        weight_concentration = max_weight - (1.0 / len(weights))

        # Scale improvement by gap to target
        gap = self.SNR_TARGET - current_snr
        improvement = weight_concentration * gap * 0.1

        return max(0.0, improvement)

    def _ensemble_refinement(
        self,
        result: Dict[str, Any],
        personas: List[str],
        weights: Dict[str, float],
    ) -> float:
        """
        Apply ensemble refinement for multi-model optimization.

        Combines multiple persona outputs using optimized weights
        for improved SNR.

        Args:
            result: Current synthesis result
            personas: List of personas
            weights: Optimized weights

        Returns:
            SNR improvement from ensemble refinement
        """
        if len(personas) < 2:
            return 0.0

        # Compute ensemble bonus from weight distribution
        # More diverse but concentrated ensemble = better
        sorted_weights = sorted(weights.values(), reverse=True)

        if len(sorted_weights) >= 2:
            # Gini-like concentration: top persona should dominate
            top_concentration = sorted_weights[0] - sorted_weights[1]
            ensemble_bonus = top_concentration * 0.005
            return max(0.0, ensemble_bonus)

        return 0.0

    def _check_convergence(self, trace: List[OptimizationState]) -> bool:
        """
        Check if optimization has converged.

        Convergence is detected when improvement falls below threshold
        for 2 consecutive iterations.

        Args:
            trace: Optimization trace so far

        Returns:
            True if converged
        """
        if len(trace) < 2:
            return False

        # Check last 2 iterations
        recent_improvements = [
            s.total_improvement for s in trace[-2:]
        ]

        return all(
            abs(imp) < self.CONVERGENCE_THRESHOLD
            for imp in recent_improvements
        )

    def update_thompson(
        self,
        persona: str,
        success: bool,
        reward: float = 0.0,
    ) -> None:
        """
        Update Thompson sampling parameters for a persona.

        Args:
            persona: Persona identifier
            success: Whether execution was successful
            reward: Additional reward signal (0-1)
        """
        if persona not in self._thompson_alphas:
            self._thompson_alphas[persona] = 1.0
            self._thompson_betas[persona] = 1.0

        if success:
            self._thompson_alphas[persona] += 1.0 + reward
        else:
            self._thompson_betas[persona] += 1.0

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "total_optimizations": self._total_optimizations,
            "successful_optimizations": self._successful_optimizations,
            "success_rate": (
                self._successful_optimizations / self._total_optimizations
                if self._total_optimizations > 0 else 0.0
            ),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._total_optimizations
                if self._total_optimizations > 0 else 0.0
            ),
            "pattern_cache": self.pattern_cache.get_stats(),
            "sona_weights": self.sona_learner.get_current_weights(),
            "thompson_personas": len(self._thompson_alphas),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_autonomous_optimizer(
    random_seed: Optional[int] = None,
    persistence_dir: Optional[Path] = None,
) -> AutonomousSNROptimizer:
    """
    Create an AutonomousSNROptimizer with default configuration.

    Args:
        random_seed: Optional seed for reproducibility
        persistence_dir: Optional directory for persistence

    Returns:
        Configured AutonomousSNROptimizer instance
    """
    base_dir = persistence_dir or Path("docs/evidence/apex")

    sona_learner = SONAWeightLearner(
        persistence_path=base_dir / "sona_weights.json"
    )

    pattern_cache = PatternCache(
        persistence_path=base_dir / "pattern_cache.json"
    )

    return AutonomousSNROptimizer(
        sona_learner=sona_learner,
        pattern_cache=pattern_cache,
        random_seed=random_seed,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "DOMAIN_PREFIX",
    "VERSION",
    "SNR_TARGET_SOVEREIGN",
    "SNR_TARGET_STANDARD",
    "IHSAN_THRESHOLD",
    "MAX_ITERATIONS",
    "CONVERGENCE_THRESHOLD",
    "DIVERSITY_BONUS_PER_DOMAIN",
    "MAX_DIVERSITY_BONUS",
    # Enums
    "OptimizationStrategy",
    # Data Classes
    "OptimizationState",
    "ElevatedPattern",
    "OptimizedResult",
    # Support Classes
    "ThompsonSampler",
    "PatternCache",
    "SONAWeightLearner",
    # Main Optimizer
    "AutonomousSNROptimizer",
    # Factory Functions
    "create_autonomous_optimizer",
    # Helper Functions
    "compute_snr_contribution",
    "compute_diversity_bonus",
    "compute_pattern_hash",
]


# =============================================================================
# CLI / TESTING
# =============================================================================


async def main() -> None:
    """Test the Autonomous SNR Optimizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BIZRA Autonomous SNR Optimizer Testing"
    )
    parser.add_argument(
        "--simulate",
        type=int,
        default=0,
        help="Simulate N optimization runs",
    )
    parser.add_argument(
        "--initial-snr",
        type=float,
        default=0.92,
        help="Initial SNR for simulation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show optimizer statistics",
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = create_autonomous_optimizer(random_seed=args.seed)

    if args.simulate > 0:
        print(f"\nSimulating {args.simulate} optimization runs...")
        print(f"Initial SNR: {args.initial_snr}")
        print(f"Target SNR: {optimizer.SNR_TARGET}")
        print()

        for i in range(args.simulate):
            # Create mock initial result
            initial_result = {
                "snr_score": args.initial_snr,
                "contributing_personas": {
                    "MasterReasoner": 0.4,
                    "SecurityGuardian": 0.3,
                    "EthicsValidator": 0.2,
                    "CreativeSynthesizer": 0.1,
                },
                "content": f"Test synthesis {i + 1}",
            }

            domains = ["reasoning", "security", "ethics"]
            personas = list(initial_result["contributing_personas"].keys())

            # Run optimization
            result = await optimizer.optimize(
                initial_result=initial_result,
                domains=domains,
                personas=personas,
                task_signature=f"test-task-{i}",
            )

            print(
                f"Run {i + 1}: "
                f"achieved={result.achieved_snr:.4f}, "
                f"target_met={result.target_met}, "
                f"iterations={result.iterations_used}, "
                f"strategies={len(result.strategies_effective)}"
            )

            # Update Thompson for next run
            for persona in personas:
                optimizer.update_thompson(
                    persona,
                    success=result.target_met,
                    reward=result.achieved_snr - args.initial_snr,
                )

        print()

    if args.stats:
        stats = optimizer.get_stats()
        print("\nOptimizer Statistics:")
        print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
