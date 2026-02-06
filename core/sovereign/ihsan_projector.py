"""
IhsanProjector - Mathematical Bridge from Constitutional AI to NTU Cognitive Units

Solves GAP-C1: O(1) projection from 8D Ihsan constitutional space to 3D NTU state.

Mathematical Foundation:
    NTU = W @ Ihsan_8D + b
    where W in R^(3x8), b in R^3

The projection preserves constitutional invariants:
    - If any dimension < 0.5, output belief must reflect doubt
    - Justice (adl) has highest weight on potential (action bias)
    - Excellence (ihsan) maps primarily to belief (confidence)

Standing on Giants:
    - Shannon (1948): Information-theoretic grounding
    - Anthropic (2023): Constitutional AI principles
    - Friston (2010): Active Inference / Free Energy Principle
    - Islamic Ethics: 8 dimensions from Prophetic tradition

8D Ihsan Vector:
    [0] truthfulness   (sidq)      - Aligns with belief
    [1] trustworthiness (amanah)   - Aligns with belief
    [2] justice        (adl)       - Aligns with potential (action)
    [3] excellence     (ihsan)     - Primary belief contributor
    [4] wisdom         (hikmah)    - Reduces entropy
    [5] compassion     (rahmah)    - Modulates potential
    [6] patience       (sabr)      - Reduces entropy variance
    [7] gratitude      (shukr)     - Stabilizes belief

NTU State (3D):
    belief:    B in [0, 1]   - Confidence in current state
    entropy:   H in [0, 1]   - Uncertainty (normalized Shannon)
    potential: Phi in [-1, 1] - Action potential (bidirectional)

Complexity: O(1) - Single matrix multiply + bias addition
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

import numpy as np

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    CONFIDENCE_MINIMUM,
)
from core.ntu import NTUState

logger = logging.getLogger(__name__)


class IhsanDimension(IntEnum):
    """
    8 dimensions of Ihsan ethical scoring.

    Each dimension represents a fundamental ethical quality from Islamic tradition.
    The ordering is significant for the projection matrix design.
    """
    TRUTHFULNESS = 0      # sidq (صدق)
    TRUSTWORTHINESS = 1   # amanah (أمانة)
    JUSTICE = 2           # adl (عدل)
    EXCELLENCE = 3        # ihsan (إحسان)
    WISDOM = 4            # hikmah (حكمة)
    COMPASSION = 5        # rahmah (رحمة)
    PATIENCE = 6          # sabr (صبر)
    GRATITUDE = 7         # shukr (شكر)


# Arabic names for documentation and display
IHSAN_ARABIC_NAMES: Dict[IhsanDimension, str] = {
    IhsanDimension.TRUTHFULNESS: "صدق",
    IhsanDimension.TRUSTWORTHINESS: "أمانة",
    IhsanDimension.JUSTICE: "عدل",
    IhsanDimension.EXCELLENCE: "إحسان",
    IhsanDimension.WISDOM: "حكمة",
    IhsanDimension.COMPASSION: "رحمة",
    IhsanDimension.PATIENCE: "صبر",
    IhsanDimension.GRATITUDE: "شكر",
}


@dataclass(frozen=True)
class IhsanVector:
    """
    Immutable 8-dimensional Ihsan constitutional vector.

    Each dimension is in [0, 1] representing the degree of ethical alignment.
    The vector is frozen to ensure constitutional immutability.
    """
    truthfulness: float = 0.5
    trustworthiness: float = 0.5
    justice: float = 0.5
    excellence: float = 0.5
    wisdom: float = 0.5
    compassion: float = 0.5
    patience: float = 0.5
    gratitude: float = 0.5

    def __post_init__(self) -> None:
        """Validate all dimensions are in [0, 1]."""
        for dim in IhsanDimension:
            value = self._get_by_index(dim.value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Ihsan dimension {dim.name} must be in [0, 1], got {value}"
                )

    def _get_by_index(self, idx: int) -> float:
        """Get dimension value by index."""
        return [
            self.truthfulness,
            self.trustworthiness,
            self.justice,
            self.excellence,
            self.wisdom,
            self.compassion,
            self.patience,
            self.gratitude,
        ][idx]

    @property
    def as_array(self) -> np.ndarray:
        """Convert to numpy array for matrix operations."""
        return np.array([
            self.truthfulness,
            self.trustworthiness,
            self.justice,
            self.excellence,
            self.wisdom,
            self.compassion,
            self.patience,
            self.gratitude,
        ], dtype=np.float64)

    @property
    def min_dimension(self) -> Tuple[IhsanDimension, float]:
        """Return the weakest dimension (constitutional bottleneck)."""
        arr = self.as_array
        min_idx = int(np.argmin(arr))
        return IhsanDimension(min_idx), arr[min_idx]

    @property
    def aggregate_score(self) -> float:
        """
        Compute aggregate Ihsan score using geometric mean.

        Geometric mean penalizes imbalance - all dimensions must be strong.
        """
        arr = self.as_array
        # Avoid log(0) by clamping
        arr = np.clip(arr, 1e-10, 1.0)
        return float(np.exp(np.mean(np.log(arr))))

    @property
    def has_constitutional_violation(self) -> bool:
        """Check if any dimension is below minimum threshold (0.5)."""
        return bool(np.any(self.as_array < 0.5))

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "truthfulness": self.truthfulness,
            "trustworthiness": self.trustworthiness,
            "justice": self.justice,
            "excellence": self.excellence,
            "wisdom": self.wisdom,
            "compassion": self.compassion,
            "patience": self.patience,
            "gratitude": self.gratitude,
            "aggregate": self.aggregate_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "IhsanVector":
        """Construct from dictionary."""
        return cls(
            truthfulness=data.get("truthfulness", 0.5),
            trustworthiness=data.get("trustworthiness", 0.5),
            justice=data.get("justice", 0.5),
            excellence=data.get("excellence", 0.5),
            wisdom=data.get("wisdom", 0.5),
            compassion=data.get("compassion", 0.5),
            patience=data.get("patience", 0.5),
            gratitude=data.get("gratitude", 0.5),
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "IhsanVector":
        """Construct from numpy array (must be length 8)."""
        if len(arr) != 8:
            raise ValueError(f"Expected 8-dimensional array, got {len(arr)}")
        arr = np.clip(arr, 0.0, 1.0)
        return cls(
            truthfulness=float(arr[0]),
            trustworthiness=float(arr[1]),
            justice=float(arr[2]),
            excellence=float(arr[3]),
            wisdom=float(arr[4]),
            compassion=float(arr[5]),
            patience=float(arr[6]),
            gratitude=float(arr[7]),
        )

    @classmethod
    def perfect(cls) -> "IhsanVector":
        """Create a perfect Ihsan vector (all 1.0)."""
        return cls(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    @classmethod
    def neutral(cls) -> "IhsanVector":
        """Create a neutral Ihsan vector (all 0.5)."""
        return cls(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


@dataclass
class ProjectorConfig:
    """
    Configuration for the IhsanProjector.

    The projection weights are designed based on ethical mappings:
    - Belief: Dominated by truthfulness, excellence, trustworthiness
    - Entropy: Inversely related to wisdom, patience (knowledge reduces uncertainty)
    - Potential: Dominated by justice (action bias), compassion (positive action)
    """
    # Constitutional invariant thresholds
    doubt_threshold: float = 0.5          # Below this triggers doubt
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD  # 0.95

    # Output clamping
    belief_range: Tuple[float, float] = (0.0, 1.0)
    entropy_range: Tuple[float, float] = (0.0, 1.0)
    potential_range: Tuple[float, float] = (-1.0, 1.0)

    # Learned vs fixed weights
    use_learned_weights: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.doubt_threshold < 1.0:
            raise ValueError("doubt_threshold must be in (0, 1)")


class IhsanProjector:
    """
    O(1) Projection from 8D Ihsan Space to 3D NTU State.

    Mathematical Form:
        [belief  ]     [W_00 ... W_07] [truthfulness ]   [b_0]
        [entropy ] = ( [W_10 ... W_17] [trustworthiness] + [b_1] ) + adjustments
        [potential]    [W_20 ... W_27] [...]             [b_2]

    Constitutional Invariants:
        1. If min(ihsan_vector) < 0.5, belief is penalized
        2. Justice has highest weight on potential
        3. Excellence is the primary belief contributor
        4. Wisdom and patience reduce entropy

    Complexity: O(1) - fixed 3x8 matrix multiply + 3-element bias
    """

    # Weight matrix W in R^(3x8)
    # Rows: [belief, entropy, potential]
    # Cols: [truthfulness, trustworthiness, justice, excellence, wisdom, compassion, patience, gratitude]
    #
    # Design rationale:
    # - Belief row: Excellence (0.25), Truthfulness (0.20), Trustworthiness (0.15), others moderate
    # - Entropy row: Negative for wisdom (-0.25), patience (-0.15) - these reduce uncertainty
    # - Potential row: Justice (0.30), Compassion (0.20) - action-oriented virtues

    DEFAULT_WEIGHT_MATRIX = np.array([
        # belief weights:    sidq   aman   adl    ihsn   hikm   rahm   sabr   shukr
        [                    0.20,  0.15,  0.08,  0.25,  0.12,  0.08,  0.06,  0.06],
        # entropy weights (negated - high wisdom/patience = low entropy):
        [                    0.05,  0.05,  0.10,  0.05, -0.30,  0.15, -0.25,  0.15],
        # potential weights:
        [                    0.05,  0.10,  0.30,  0.15,  0.10,  0.20,  0.05,  0.05],
    ], dtype=np.float64)

    # Bias vector b in R^3
    # Small positive bias for belief (optimistic prior)
    # Moderate positive bias for entropy (uncertainty prior)
    # Zero bias for potential (neutral action prior)
    DEFAULT_BIAS = np.array([0.05, 0.30, 0.0], dtype=np.float64)

    def __init__(
        self,
        config: Optional[ProjectorConfig] = None,
        weight_matrix: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the projector.

        Args:
            config: Projector configuration
            weight_matrix: Custom 3x8 weight matrix (default: designed weights)
            bias: Custom 3-element bias vector (default: designed bias)
        """
        self.config = config or ProjectorConfig()

        # Initialize projection parameters
        self._W = weight_matrix if weight_matrix is not None else self.DEFAULT_WEIGHT_MATRIX.copy()
        self._b = bias if bias is not None else self.DEFAULT_BIAS.copy()

        # Validate dimensions
        if self._W.shape != (3, 8):
            raise ValueError(f"Weight matrix must be 3x8, got {self._W.shape}")
        if self._b.shape != (3,):
            raise ValueError(f"Bias must be 3-element, got {self._b.shape}")

        logger.info(
            f"IhsanProjector initialized: doubt_threshold={self.config.doubt_threshold}, "
            f"ihsan_threshold={self.config.ihsan_threshold}"
        )

    def project(self, ihsan: IhsanVector) -> NTUState:
        """
        Project 8D Ihsan vector to 3D NTU state.

        This is the core O(1) operation:
            raw = W @ ihsan + b
            output = apply_invariants(raw, ihsan)

        Args:
            ihsan: 8-dimensional Ihsan constitutional vector

        Returns:
            NTUState with belief, entropy, potential
        """
        # O(1) affine transformation
        ihsan_arr = ihsan.as_array
        raw = self._W @ ihsan_arr + self._b

        # Apply constitutional invariants
        belief, entropy, potential = self._apply_invariants(raw, ihsan)

        # Clamp to valid ranges
        belief = float(np.clip(belief, *self.config.belief_range))
        entropy = float(np.clip(entropy, *self.config.entropy_range))
        potential = float(np.clip(potential, *self.config.potential_range))

        # Create NTU state (note: NTUState expects entropy in [0,1], potential in [0,1])
        # We remap potential from [-1, 1] to [0, 1] for NTU compatibility
        ntu_potential = (potential + 1.0) / 2.0

        return NTUState(
            belief=belief,
            entropy=entropy,
            potential=ntu_potential,
        )

    def _apply_invariants(
        self,
        raw: np.ndarray,
        ihsan: IhsanVector,
    ) -> Tuple[float, float, float]:
        """
        Apply constitutional invariants to raw projection.

        Invariant 1: If any dimension < 0.5, belief is penalized
        Invariant 2: Excellence (ihsan) dominates belief
        Invariant 3: Justice (adl) dominates potential

        Args:
            raw: Raw [belief, entropy, potential] from affine transform
            ihsan: Original Ihsan vector for invariant checks

        Returns:
            Adjusted (belief, entropy, potential)
        """
        belief, entropy, potential = raw[0], raw[1], raw[2]

        # Invariant 1: Constitutional violation penalty
        if ihsan.has_constitutional_violation:
            min_dim, min_val = ihsan.min_dimension
            # Penalty scales with severity of violation
            penalty = (self.config.doubt_threshold - min_val) * 0.5
            belief = belief - penalty

            logger.debug(
                f"Constitutional violation: {min_dim.name}={min_val:.3f}, "
                f"belief penalty={penalty:.3f}"
            )

        # Invariant 2: Excellence floor for belief
        # If excellence is very high but belief is low, boost belief
        if ihsan.excellence > 0.9 and belief < 0.7:
            belief = 0.5 * belief + 0.5 * ihsan.excellence

        # Invariant 3: Justice dominance for potential
        # Ensure justice has outsized impact on action potential
        justice_contribution = (ihsan.justice - 0.5) * 0.4
        potential = potential + justice_contribution

        # Normalize entropy to be information-theoretic
        # High wisdom + patience should yield low entropy
        wisdom_patience_effect = (ihsan.wisdom + ihsan.patience) / 2.0
        entropy = entropy * (1.5 - wisdom_patience_effect)

        return belief, entropy, potential

    def project_batch(
        self,
        ihsan_vectors: List[IhsanVector],
    ) -> List[NTUState]:
        """
        Project multiple Ihsan vectors efficiently.

        Uses vectorized operations for O(n) total, O(1) per vector.

        Args:
            ihsan_vectors: List of Ihsan vectors

        Returns:
            List of NTU states
        """
        return [self.project(v) for v in ihsan_vectors]

    def inverse_project(
        self,
        ntu: NTUState,
        prior: Optional[IhsanVector] = None,
    ) -> IhsanVector:
        """
        Approximate inverse projection from NTU to Ihsan.

        This is an underdetermined system (3 equations, 8 unknowns).
        We use a prior (default: neutral) and minimum-norm solution.

        Args:
            ntu: NTU state to invert
            prior: Prior Ihsan vector for regularization (default: neutral)

        Returns:
            Approximate Ihsan vector
        """
        prior = prior or IhsanVector.neutral()

        # Target NTU values (remap potential back to [-1, 1])
        target = np.array([
            ntu.belief,
            ntu.entropy,
            ntu.potential * 2.0 - 1.0,  # [0,1] -> [-1,1]
        ])

        # Subtract bias
        target = target - self._b

        # Minimum-norm solution using pseudoinverse
        W_pinv = np.linalg.pinv(self._W)
        delta = W_pinv @ target

        # Regularize toward prior
        ihsan_arr = prior.as_array + 0.5 * delta
        ihsan_arr = np.clip(ihsan_arr, 0.0, 1.0)

        return IhsanVector.from_array(ihsan_arr)

    def get_jacobian(self) -> np.ndarray:
        """
        Return the Jacobian of the projection.

        For an affine map, the Jacobian is simply the weight matrix W.
        This is useful for sensitivity analysis.

        Returns:
            3x8 Jacobian matrix
        """
        return self._W.copy()

    def sensitivity_analysis(
        self,
        ihsan: IhsanVector,
        dimension: IhsanDimension,
        delta: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute sensitivity of NTU outputs to a single Ihsan dimension.

        Uses finite differences for accuracy.

        Args:
            ihsan: Base Ihsan vector
            dimension: Dimension to perturb
            delta: Perturbation size

        Returns:
            Dict with d_belief/d_dim, d_entropy/d_dim, d_potential/d_dim
        """
        arr = ihsan.as_array.copy()
        idx = dimension.value

        # Forward and backward perturbations
        arr_plus = arr.copy()
        arr_plus[idx] = min(1.0, arr[idx] + delta)

        arr_minus = arr.copy()
        arr_minus[idx] = max(0.0, arr[idx] - delta)

        ntu_plus = self.project(IhsanVector.from_array(arr_plus))
        ntu_minus = self.project(IhsanVector.from_array(arr_minus))

        actual_delta = arr_plus[idx] - arr_minus[idx]
        if actual_delta < 1e-10:
            return {"d_belief": 0.0, "d_entropy": 0.0, "d_potential": 0.0}

        return {
            "d_belief": (ntu_plus.belief - ntu_minus.belief) / actual_delta,
            "d_entropy": (ntu_plus.entropy - ntu_minus.entropy) / actual_delta,
            "d_potential": (ntu_plus.potential - ntu_minus.potential) / actual_delta,
        }

    def calibrate_from_examples(
        self,
        examples: List[Tuple[IhsanVector, NTUState]],
        learning_rate: float = 0.01,
        iterations: int = 100,
    ) -> float:
        """
        Calibrate projection weights from labeled examples.

        Uses gradient descent to minimize MSE between projected and target NTU states.

        Args:
            examples: List of (ihsan_vector, target_ntu_state) pairs
            learning_rate: Gradient descent step size
            iterations: Number of optimization iterations

        Returns:
            Final MSE loss
        """
        if not examples:
            return 0.0

        # Prepare training data
        X = np.array([ex[0].as_array for ex in examples])  # (n, 8)
        Y = np.array([
            [ex[1].belief, ex[1].entropy, ex[1].potential * 2.0 - 1.0]
            for ex in examples
        ])  # (n, 3)

        n = len(examples)

        for _ in range(iterations):
            # Forward pass
            predictions = X @ self._W.T + self._b  # (n, 3)

            # Loss
            errors = predictions - Y  # (n, 3)
            mse = float(np.mean(errors ** 2))

            # Gradients
            grad_W = (2.0 / n) * (errors.T @ X)  # (3, 8)
            grad_b = (2.0 / n) * np.sum(errors, axis=0)  # (3,)

            # Update
            self._W -= learning_rate * grad_W
            self._b -= learning_rate * grad_b

        # Final loss
        predictions = X @ self._W.T + self._b
        final_mse = float(np.mean((predictions - Y) ** 2))

        logger.info(f"Calibration complete: final MSE = {final_mse:.6f}")
        return final_mse

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the projector."""
        return {
            "config": {
                "doubt_threshold": self.config.doubt_threshold,
                "ihsan_threshold": self.config.ihsan_threshold,
                "belief_range": self.config.belief_range,
                "entropy_range": self.config.entropy_range,
                "potential_range": self.config.potential_range,
            },
            "weight_matrix": self._W.tolist(),
            "bias": self._b.tolist(),
            "weight_norms": {
                "belief_row": float(np.linalg.norm(self._W[0])),
                "entropy_row": float(np.linalg.norm(self._W[1])),
                "potential_row": float(np.linalg.norm(self._W[2])),
            },
            "dominant_dimensions": {
                "belief": IhsanDimension(int(np.argmax(self._W[0]))).name,
                "entropy": IhsanDimension(int(np.argmax(np.abs(self._W[1])))).name,
                "potential": IhsanDimension(int(np.argmax(self._W[2]))).name,
            },
        }


def project_ihsan_to_ntu(ihsan: IhsanVector) -> NTUState:
    """
    Convenience function for one-shot projection.

    Creates a default projector and projects the given Ihsan vector.
    For repeated projections, instantiate IhsanProjector directly.

    Args:
        ihsan: 8-dimensional Ihsan constitutional vector

    Returns:
        NTUState with belief, entropy, potential
    """
    projector = IhsanProjector()
    return projector.project(ihsan)


def create_ihsan_from_scores(
    correctness: float = 0.5,
    safety: float = 0.5,
    helpfulness: float = 0.5,
    efficiency: float = 0.5,
) -> IhsanVector:
    """
    Map common LLM evaluation scores to Ihsan dimensions.

    This provides compatibility with existing scoring systems:
    - correctness -> truthfulness, excellence
    - safety -> trustworthiness, justice
    - helpfulness -> compassion, gratitude
    - efficiency -> wisdom, patience

    Args:
        correctness: Factual accuracy score [0, 1]
        safety: Safety/alignment score [0, 1]
        helpfulness: User benefit score [0, 1]
        efficiency: Resource efficiency score [0, 1]

    Returns:
        IhsanVector mapped from these scores
    """
    return IhsanVector(
        truthfulness=correctness,
        trustworthiness=safety,
        justice=0.5 * safety + 0.5 * helpfulness,  # Blend
        excellence=0.7 * correctness + 0.3 * efficiency,  # Weighted blend
        wisdom=efficiency,
        compassion=helpfulness,
        patience=0.5 * efficiency + 0.5,  # Bias toward patience
        gratitude=helpfulness,
    )
