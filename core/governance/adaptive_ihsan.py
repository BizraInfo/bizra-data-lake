"""
Adaptive Ihsān with Dirichlet Posterior — Constitutional Learning Engine

╔══════════════════════════════════════════════════════════════════════════════╗
║   HP-01: Constitutional Convergence Isomorphism (SNR 0.97)                   ║
║   Discovery: IHSAN_WEIGHTS forms a Dirichlet simplex.                        ║
║   Insight: System can LEARN optimal ethical emphasis via Bayesian update.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Dirichlet (1839): Dirichlet distribution on probability simplices
- Bayes (1763): Posterior ∝ Likelihood × Prior
- Shannon (1948): Information-theoretic divergence measures
- Al-Ghazali (1095): Ihsān as dynamic ethical excellence
- Anthropic (2023): Constitutional AI with learnable constraints
- Ferguson (1973): Dirichlet Process for nonparametric Bayesian models

Mathematical Foundation:
    Prior:     α₀ = [α₁, α₂, ..., α₈]  (concentration parameters)
    Weights:   w ~ Dir(α₀)               (current IHSAN_WEIGHTS as mode)
    Observe:   x = (dimension_id, outcome) (which dimension contributed to success)
    Posterior: α_n = α₀ + Σ counts        (conjugate update)

    Mode of Dir(α): w_i = (α_i - 1) / (Σα - 8)  when α_i > 1

Constitutional Invariants:
    INV-1: Weights MUST sum to 1.0 (simplex constraint)
    INV-2: No weight may drop below FLOOR (0.02) — all virtues matter
    INV-3: No weight may exceed CEILING (0.35) — no single virtue dominates
    INV-4: Safety + Correctness weights ≥ 0.30 — non-negotiable minimum
    INV-5: KL-divergence from canonical weights ≤ MAX_DRIFT — bounded evolution
    INV-6: All updates produce cryptographic audit receipt

Complexity: O(8) per update — constant time, bounded by dimension count
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

from core.integration.constants import (
    IHSAN_WEIGHTS,
    UNIFIED_IHSAN_THRESHOLD,
)
from core.proof_engine.canonical import hex_digest

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUTIONAL BOUNDS — These are HARD GATES, not suggestions
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum weight for any dimension — all virtues are essential
WEIGHT_FLOOR: Final[float] = 0.02

# Maximum weight for any dimension — prevent tyranny of one virtue
WEIGHT_CEILING: Final[float] = 0.35

# Safety + Correctness minimum combined weight
SAFETY_CORRECTNESS_FLOOR: Final[float] = 0.30

# Maximum KL-divergence from canonical weights (bounded evolution)
MAX_KL_DRIFT: Final[float] = 0.15

# Concentration scale — higher = stronger prior (slower adaptation)
# At 100, roughly 100 observations needed to shift weights meaningfully
DEFAULT_CONCENTRATION: Final[float] = 100.0

# Minimum concentration to prevent degenerate distributions
MIN_CONCENTRATION: Final[float] = 10.0

# Canonical dimension ordering (must match IHSAN_WEIGHTS keys)
DIMENSION_ORDER: Final[Tuple[str, ...]] = (
    "correctness",
    "safety",
    "user_benefit",
    "efficiency",
    "auditability",
    "anti_centralization",
    "robustness",
    "adl_fairness",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class UpdateOutcome(str, Enum):
    """Outcome of an observation for Bayesian update."""
    SUCCESS = "success"       # Dimension contributed to good outcome
    FAILURE = "failure"       # Dimension was bottleneck
    NEUTRAL = "neutral"       # No signal


@dataclass(frozen=True)
class DirichletObservation:
    """
    A single observation for Dirichlet posterior update.

    Records which dimension contributed to which outcome,
    enabling the system to learn which ethical emphases
    produce the best results in practice.
    """
    dimension: str                    # Which dimension was observed
    outcome: UpdateOutcome            # What happened
    magnitude: float = 1.0            # Strength of observation [0, 1]
    context: str = ""                 # Execution context for audit
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSION_ORDER:
            raise ValueError(
                f"Unknown dimension '{self.dimension}'. "
                f"Valid: {DIMENSION_ORDER}"
            )
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError(f"Magnitude must be in [0, 1], got {self.magnitude}")


@dataclass
class DirichletState:
    """
    Current state of the Dirichlet posterior.

    The concentration parameters α fully characterize the distribution.
    Mode (most likely weights) = (α_i - 1) / (Σα - K) for α_i > 1.
    """
    alphas: Dict[str, float]          # Concentration parameters
    observation_count: int = 0        # Total observations processed
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self.created_at = now
            self.last_updated = now

    @property
    def total_concentration(self) -> float:
        """Sum of all concentration parameters."""
        return sum(self.alphas.values())

    @property
    def mode(self) -> Dict[str, float]:
        """
        Mode of the Dirichlet distribution (MAP estimate).

        For Dir(α), mode_i = (α_i - 1) / (Σα - K) when all α_i > 1.
        This is the most likely weight vector.
        """
        K = len(self.alphas)
        total = self.total_concentration
        denominator = total - K

        if denominator <= 0:
            # Fallback to mean when concentration is too low
            return self.mean

        mode = {}
        for dim, alpha in self.alphas.items():
            if alpha <= 1.0:
                # Mode is at boundary; use mean instead
                return self.mean
            mode[dim] = (alpha - 1.0) / denominator

        return mode

    @property
    def mean(self) -> Dict[str, float]:
        """
        Mean of the Dirichlet distribution.

        E[w_i] = α_i / Σα — always valid, even for small α.
        """
        total = self.total_concentration
        if total <= 0:
            K = len(self.alphas)
            return {dim: 1.0 / K for dim in self.alphas}
        return {dim: alpha / total for dim, alpha in self.alphas.items()}

    @property
    def variance(self) -> Dict[str, float]:
        """
        Variance of each weight under the Dirichlet.

        Var[w_i] = α_i(Σα - α_i) / (Σα²(Σα + 1))
        Lower variance = more confident in the weight.
        """
        total = self.total_concentration
        denominator = total * total * (total + 1.0)
        if denominator <= 0:
            return {dim: 0.0 for dim in self.alphas}

        return {
            dim: (alpha * (total - alpha)) / denominator
            for dim, alpha in self.alphas.items()
        }

    @property
    def effective_sample_size(self) -> float:
        """
        Effective sample size — how many observations the prior is worth.

        ESS ≈ Σα for Dirichlet. Higher = stronger prior, slower adaptation.
        """
        return self.total_concentration

    def entropy(self) -> float:
        """
        Entropy of the Dirichlet distribution.

        H[Dir(α)] = ln B(α) - (K-1)ψ(Σα) + Σ(α_i-1)ψ(α_i)

        where B is the multivariate Beta function and ψ is digamma.

        Higher entropy = more uncertainty about weights.
        """
        from scipy.special import digamma, gammaln

        alphas_arr = np.array(list(self.alphas.values()))
        alpha_sum = np.sum(alphas_arr)
        K = len(alphas_arr)

        # Log multivariate Beta
        log_B = np.sum(gammaln(alphas_arr)) - gammaln(alpha_sum)

        # Entropy
        H = log_B - (K - 1) * digamma(alpha_sum)
        H += np.sum((alphas_arr - 1.0) * digamma(alphas_arr))

        return float(H)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence and audit."""
        return {
            "alphas": {k: round(v, 6) for k, v in self.alphas.items()},
            "observation_count": self.observation_count,
            "total_concentration": round(self.total_concentration, 6),
            "mode": {k: round(v, 6) for k, v in self.mode.items()},
            "mean": {k: round(v, 6) for k, v in self.mean.items()},
            "variance": {k: round(v, 8) for k, v in self.variance.items()},
            "effective_sample_size": round(self.effective_sample_size, 2),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirichletState":
        """Reconstruct from serialized form."""
        return cls(
            alphas=data["alphas"],
            observation_count=data.get("observation_count", 0),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class UpdateReceipt:
    """
    Cryptographic receipt for every Ihsān weight update.

    INV-6: All updates produce audit receipt — no silent mutations.
    """
    observation: DirichletObservation
    prior_weights: Dict[str, float]
    posterior_weights: Dict[str, float]
    kl_divergence: float
    invariant_checks: Dict[str, bool]
    accepted: bool
    rejection_reason: Optional[str]
    receipt_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": {
                "dimension": self.observation.dimension,
                "outcome": self.observation.outcome.value,
                "magnitude": self.observation.magnitude,
                "context": self.observation.context,
            },
            "prior_weights": {k: round(v, 6) for k, v in self.prior_weights.items()},
            "posterior_weights": {k: round(v, 6) for k, v in self.posterior_weights.items()},
            "kl_divergence": round(self.kl_divergence, 8),
            "invariant_checks": self.invariant_checks,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "receipt_hash": self.receipt_hash,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE IHSĀN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptiveIhsan:
    """
    Dirichlet-Posterior Adaptive Ihsān Engine.

    Transforms the static IHSAN_WEIGHTS into a learnable Dirichlet posterior
    that adapts to observed outcomes while preserving constitutional invariants.

    The engine observes which ethical dimensions contribute to successful
    outcomes and which are bottlenecks, then updates the weight emphasis
    accordingly — within strict constitutional bounds.

    Standing on Giants:
    - Dirichlet (1839): Conjugate prior for categorical/multinomial
    - Bayes (1763): P(θ|data) ∝ P(data|θ) × P(θ)
    - Al-Ghazali (1095): Excellence (Ihsān) as continuous refinement
    - Shannon (1948): KL-divergence for distribution comparison

    Usage:
        engine = AdaptiveIhsan()
        obs = DirichletObservation("safety", UpdateOutcome.SUCCESS, magnitude=1.0)
        receipt = engine.update(obs)
        if receipt.accepted:
            current_weights = engine.current_weights
    """

    def __init__(
        self,
        concentration: float = DEFAULT_CONCENTRATION,
        canonical_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize with canonical weights as Dirichlet prior.

        Args:
            concentration: Prior strength (higher = slower adaptation)
            canonical_weights: Starting weights (default: IHSAN_WEIGHTS)
        """
        self._canonical = canonical_weights or dict(IHSAN_WEIGHTS)
        self._concentration = max(concentration, MIN_CONCENTRATION)

        # Initialize alphas from canonical weights × concentration
        # This encodes the canonical weights as the mode of our prior
        # For Dir(α), mode_i = (α_i - 1)/(Σα - K)
        # So α_i = mode_i × (Σα - K) + 1
        # With Σα = concentration, K = 8:
        K = len(self._canonical)
        alphas = {}
        for dim, weight in self._canonical.items():
            alphas[dim] = weight * (self._concentration - K) + 1.0

        self._state = DirichletState(alphas=alphas)
        self._receipts: List[UpdateReceipt] = []

        # Validate initial state
        self._verify_all_invariants(self.current_weights)

        logger.info(
            f"AdaptiveIhsan initialized: concentration={self._concentration:.0f}, "
            f"K={K}, ESS={self._state.effective_sample_size:.0f}"
        )

    @property
    def current_weights(self) -> Dict[str, float]:
        """
        Current adaptive weights (mode of Dirichlet posterior).

        These replace the static IHSAN_WEIGHTS for scoring.
        """
        return self._state.mode

    @property
    def canonical_weights(self) -> Dict[str, float]:
        """Original canonical weights (the prior)."""
        return dict(self._canonical)

    @property
    def state(self) -> DirichletState:
        """Current Dirichlet state for inspection."""
        return self._state

    @property
    def observation_count(self) -> int:
        """Total observations processed."""
        return self._state.observation_count

    # ─────────────────────────────────────────────────────────────────────────
    # BAYESIAN UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, observation: DirichletObservation) -> UpdateReceipt:
        """
        Bayesian update of Ihsān weights from a single observation.

        Process:
        1. Record prior weights
        2. Compute posterior alphas (conjugate update)
        3. Extract posterior mode (new weights)
        4. Verify ALL constitutional invariants
        5. Accept or reject update
        6. Generate cryptographic receipt

        Args:
            observation: What was observed

        Returns:
            UpdateReceipt with full audit trail
        """
        prior_weights = self.current_weights.copy()
        prior_alphas = {k: v for k, v in self._state.alphas.items()}

        # Compute update magnitude based on outcome
        if observation.outcome == UpdateOutcome.SUCCESS:
            # Success: increase emphasis on this dimension
            delta = observation.magnitude
        elif observation.outcome == UpdateOutcome.FAILURE:
            # Failure as bottleneck: ALSO increase emphasis (it needs more weight)
            # This is key insight: if a dimension fails, we need MORE of it, not less
            delta = observation.magnitude * 0.5
        else:
            # Neutral: no update
            delta = 0.0

        if delta == 0.0:
            return self._make_receipt(
                observation, prior_weights, prior_weights,
                accepted=True, reason=None,
            )

        # Conjugate update: α_new = α_old + count
        new_alphas = dict(prior_alphas)
        new_alphas[observation.dimension] += delta

        # Compute candidate posterior weights
        candidate_state = DirichletState(
            alphas=new_alphas,
            observation_count=self._state.observation_count + 1,
        )
        candidate_weights = candidate_state.mode

        # ── INVARIANT VERIFICATION ──────────────────────────────────────
        invariants = self._check_all_invariants(candidate_weights)
        all_passed = all(invariants.values())

        if all_passed:
            # Accept update
            self._state = candidate_state
            self._state.last_updated = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            receipt = self._make_receipt(
                observation, prior_weights, candidate_weights,
                accepted=True, reason=None, invariants=invariants,
            )
        else:
            # Reject update — constitutional violation
            failed = [k for k, v in invariants.items() if not v]
            reason = f"Constitutional violation: {', '.join(failed)}"
            receipt = self._make_receipt(
                observation, prior_weights, prior_weights,
                accepted=False, reason=reason, invariants=invariants,
            )
            logger.warning(f"Ihsān update REJECTED: {reason}")

        self._receipts.append(receipt)
        return receipt

    def update_batch(
        self, observations: List[DirichletObservation]
    ) -> List[UpdateReceipt]:
        """
        Process multiple observations sequentially.

        Each observation is applied independently with its own invariant check.
        """
        return [self.update(obs) for obs in observations]

    # ─────────────────────────────────────────────────────────────────────────
    # INVARIANT VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def _check_all_invariants(self, weights: Dict[str, float]) -> Dict[str, bool]:
        """
        Verify all constitutional invariants against candidate weights.

        Returns dict mapping invariant name to pass/fail.
        """
        return {
            "INV-1_simplex": self._check_simplex(weights),
            "INV-2_floor": self._check_floor(weights),
            "INV-3_ceiling": self._check_ceiling(weights),
            "INV-4_safety_correctness": self._check_safety_correctness(weights),
            "INV-5_kl_drift": self._check_kl_drift(weights),
        }

    def _verify_all_invariants(self, weights: Dict[str, float]) -> None:
        """Verify invariants and raise if any fail (for initialization)."""
        results = self._check_all_invariants(weights)
        failed = [k for k, v in results.items() if not v]
        if failed:
            raise ValueError(f"Constitutional invariant violation: {failed}")

    def _check_simplex(self, weights: Dict[str, float]) -> bool:
        """INV-1: Weights must sum to 1.0."""
        return abs(sum(weights.values()) - 1.0) < 1e-6

    def _check_floor(self, weights: Dict[str, float]) -> bool:
        """INV-2: No weight below WEIGHT_FLOOR."""
        return all(w >= WEIGHT_FLOOR - 1e-8 for w in weights.values())

    def _check_ceiling(self, weights: Dict[str, float]) -> bool:
        """INV-3: No weight above WEIGHT_CEILING."""
        return all(w <= WEIGHT_CEILING + 1e-8 for w in weights.values())

    def _check_safety_correctness(self, weights: Dict[str, float]) -> bool:
        """INV-4: Safety + Correctness >= SAFETY_CORRECTNESS_FLOOR."""
        combined = weights.get("safety", 0) + weights.get("correctness", 0)
        return combined >= SAFETY_CORRECTNESS_FLOOR - 1e-8

    def _check_kl_drift(self, weights: Dict[str, float]) -> bool:
        """INV-5: KL(posterior || canonical) <= MAX_KL_DRIFT."""
        kl = self._kl_divergence(weights, self._canonical)
        return kl <= MAX_KL_DRIFT

    @staticmethod
    def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        KL(P || Q) — Kullback-Leibler divergence.

        Standing on: Shannon (1948), Kullback & Leibler (1951).
        Measures how far P has drifted from Q.
        """
        kl = 0.0
        for dim in p:
            p_i = max(p[dim], 1e-10)
            q_i = max(q.get(dim, 1e-10), 1e-10)
            kl += p_i * math.log(p_i / q_i)
        return kl

    # ─────────────────────────────────────────────────────────────────────────
    # RECEIPT GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    def _make_receipt(
        self,
        observation: DirichletObservation,
        prior_weights: Dict[str, float],
        posterior_weights: Dict[str, float],
        accepted: bool,
        reason: Optional[str],
        invariants: Optional[Dict[str, bool]] = None,
    ) -> UpdateReceipt:
        """Generate cryptographic receipt for the update."""
        kl = self._kl_divergence(posterior_weights, self._canonical)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Compute receipt hash
        receipt_data = json.dumps({
            "observation": observation.dimension,
            "outcome": observation.outcome.value,
            "prior": {k: round(v, 8) for k, v in prior_weights.items()},
            "posterior": {k: round(v, 8) for k, v in posterior_weights.items()},
            "kl": round(kl, 8),
            "accepted": accepted,
            "timestamp": now,
        }, sort_keys=True, separators=(",", ":"))
        receipt_hash = hex_digest(receipt_data.encode())

        return UpdateReceipt(
            observation=observation,
            prior_weights=prior_weights,
            posterior_weights=posterior_weights,
            kl_divergence=kl,
            invariant_checks=invariants or {},
            accepted=accepted,
            rejection_reason=reason,
            receipt_hash=receipt_hash,
            timestamp=now,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS & DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    def convergence_report(self) -> Dict[str, Any]:
        """
        Report on how weights have evolved from the canonical prior.

        Includes drift analysis, convergence metrics, and recommendations.
        """
        current = self.current_weights
        canonical = self._canonical

        drift_per_dim = {
            dim: round(current.get(dim, 0) - canonical.get(dim, 0), 6)
            for dim in DIMENSION_ORDER
        }

        return {
            "observation_count": self._state.observation_count,
            "effective_sample_size": round(self._state.effective_sample_size, 2),
            "kl_divergence_from_canonical": round(
                self._kl_divergence(current, canonical), 8
            ),
            "max_kl_allowed": MAX_KL_DRIFT,
            "drift_budget_remaining": round(
                MAX_KL_DRIFT - self._kl_divergence(current, canonical), 8
            ),
            "canonical_weights": {k: round(v, 4) for k, v in canonical.items()},
            "current_weights": {k: round(v, 4) for k, v in current.items()},
            "drift_per_dimension": drift_per_dim,
            "variance": {k: round(v, 8) for k, v in self._state.variance.items()},
            "most_increased": max(drift_per_dim, key=lambda k: drift_per_dim[k]),
            "most_decreased": min(drift_per_dim, key=lambda k: drift_per_dim[k]),
            "accepted_updates": sum(1 for r in self._receipts if r.accepted),
            "rejected_updates": sum(1 for r in self._receipts if not r.accepted),
        }

    def reset_to_canonical(self) -> None:
        """Reset weights to canonical prior (emergency constitutional reset)."""
        K = len(self._canonical)
        alphas = {}
        for dim, weight in self._canonical.items():
            alphas[dim] = weight * (self._concentration - K) + 1.0

        self._state = DirichletState(alphas=alphas)
        logger.warning("AdaptiveIhsan RESET to canonical weights")

    def get_receipts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent update receipts for audit."""
        return [r.to_dict() for r in self._receipts[-limit:]]

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization for persistence."""
        return {
            "concentration": self._concentration,
            "canonical_weights": self._canonical,
            "state": self._state.to_dict(),
            "receipts_count": len(self._receipts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveIhsan":
        """Reconstruct from serialized form."""
        engine = cls(
            concentration=data.get("concentration", DEFAULT_CONCENTRATION),
            canonical_weights=data.get("canonical_weights"),
        )
        if "state" in data:
            engine._state = DirichletState.from_dict(data["state"])
        return engine


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_adaptive_ihsan(
    concentration: float = DEFAULT_CONCENTRATION,
) -> AdaptiveIhsan:
    """Create an AdaptiveIhsan engine with default constitutional weights."""
    return AdaptiveIhsan(concentration=concentration)


__all__ = [
    "AdaptiveIhsan",
    "DirichletObservation",
    "DirichletState",
    "UpdateReceipt",
    "UpdateOutcome",
    "create_adaptive_ihsan",
    # Constants
    "WEIGHT_FLOOR",
    "WEIGHT_CEILING",
    "SAFETY_CORRECTNESS_FLOOR",
    "MAX_KL_DRIFT",
    "DEFAULT_CONCENTRATION",
    "DIMENSION_ORDER",
]
