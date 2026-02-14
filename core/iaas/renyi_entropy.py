"""
Rényi Entropy & Rate-Distortion Optimal Compression

╔══════════════════════════════════════════════════════════════════════════════╗
║   Ω-4: Rényi Entropy + Rate-Distortion Compression (P1-PERFORMANCE)         ║
║   Goal: Add Rényi-2 entropy to SNR diversity + optimal SDPO compression.     ║
║   Impact: Theoretical floor for context compression at 0.7 target.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Rényi (1961): "On measures of entropy and information"
  - H_α(X) = (1/(1-α)) × log(Σ p_i^α)
  - α=1 recovers Shannon entropy (limit)
  - α=2 gives collision entropy — more sensitive to dominant terms
  - α→∞ gives min-entropy — worst-case measure
- Shannon (1948): Rate-distortion theory
  - R(D) = min_{p(x̂|x): E[d(x,x̂)]≤D} I(X;X̂)
  - Minimum bits to reconstruct at distortion ≤ D
- Cover & Thomas (2006): "Elements of Information Theory" (2nd ed.)
- Blahut (1972): Blahut-Arimoto algorithm for R(D) computation

Why Rényi-2 instead of Shannon for SNR:
    Shannon entropy: H₁ = -Σ pᵢ log pᵢ
    Rényi-2 entropy: H₂ = -log(Σ pᵢ²) = -log(collision probability)

    H₂ is more sensitive to concentration:
    - Uniform: H₂ = H₁ = log(n)
    - One dominant: H₂ << H₁ (detects dominance faster)

    For SNR diversity, this means Rényi-2 catches "one source dominates"
    earlier than Shannon, which is exactly the redundancy signal we need.

Rate-Distortion for SDPO:
    SDPO targets 70% compression (0.7 ratio). Rate-distortion theory gives
    the theoretical minimum bits needed:

    For Gaussian source: R(D) = max(0, ½ log(σ²/D))
    For BIZRA contexts: R(D) ≈ H(X) - H(X|X̂ at D=0.3)

    This tells us HOW MUCH information we can safely discard.

Complexity: O(N) for entropy, O(N²) for rate-distortion estimation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

from core.integration.constants import UNIFIED_SNR_THRESHOLD

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default Rényi order for SNR diversity (α=2 → collision entropy)
DEFAULT_RENYI_ALPHA: Final[float] = 2.0

# SDPO compression target (from sdpo/__init__.py)
SDPO_COMPRESSION_TARGET: Final[float] = 0.7

# Rate-distortion estimation parameters
RD_MAX_DISTORTION: Final[float] = 0.5  # Maximum tolerable distortion
RD_NUM_POINTS: Final[int] = 50  # Points on R(D) curve


# ═══════════════════════════════════════════════════════════════════════════════
# RÉNYI ENTROPY — Generalized entropy family
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RenyiResult:
    """Result of Rényi entropy computation."""
    alpha: float               # Order parameter
    entropy: float             # H_α in bits
    shannon_entropy: float     # H₁ for comparison
    min_entropy: float         # H_∞ (worst-case)
    max_entropy: float         # log₂(N) — uniform distribution
    normalized: float          # H_α / H_max ∈ [0, 1]
    collision_probability: float  # Σ pᵢ² (for α=2)
    effective_support: float   # 2^H_α — effective number of categories

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def renyi_entropy(
    probs: np.ndarray,
    alpha: float = DEFAULT_RENYI_ALPHA,
) -> RenyiResult:
    """
    Compute Rényi entropy of order α.

    H_α(X) = (1/(1-α)) × log₂(Σ pᵢ^α)

    Special cases:
        α → 1: Shannon entropy H₁ = -Σ pᵢ log₂ pᵢ
        α = 2: Collision entropy H₂ = -log₂(Σ pᵢ²)
        α → ∞: Min-entropy H_∞ = -log₂(max pᵢ)

    Args:
        probs: Probability distribution (must sum to 1.0)
        alpha: Rényi order parameter (default: 2.0)

    Returns:
        RenyiResult with entropy values and diagnostics
    """
    # Validate and clean
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]  # Remove zeros

    if len(probs) == 0:
        return RenyiResult(
            alpha=alpha, entropy=0.0, shannon_entropy=0.0,
            min_entropy=0.0, max_entropy=0.0, normalized=0.0,
            collision_probability=1.0, effective_support=1.0,
        )

    # Normalize
    probs = probs / probs.sum()
    n = len(probs)
    max_entropy = math.log2(n) if n > 1 else 0.0

    # Shannon entropy (H₁)
    shannon = float(-np.sum(probs * np.log2(probs)))

    # Min-entropy (H_∞)
    min_entropy = float(-math.log2(np.max(probs)))

    # Collision probability
    collision_prob = float(np.sum(probs ** 2))

    # Rényi entropy (H_α)
    if abs(alpha - 1.0) < 1e-10:
        # α → 1: Shannon limit
        entropy = shannon
    elif alpha == float('inf'):
        entropy = min_entropy
    else:
        sum_p_alpha = float(np.sum(probs ** alpha))
        if sum_p_alpha > 0:
            entropy = float((1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha))
        else:
            entropy = 0.0

    # Effective support: 2^H_α
    effective_support = 2.0 ** entropy if entropy >= 0 else 1.0

    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return RenyiResult(
        alpha=alpha,
        entropy=entropy,
        shannon_entropy=shannon,
        min_entropy=min_entropy,
        max_entropy=max_entropy,
        normalized=normalized,
        collision_probability=collision_prob,
        effective_support=effective_support,
    )


def renyi_divergence(
    p: np.ndarray,
    q: np.ndarray,
    alpha: float = DEFAULT_RENYI_ALPHA,
) -> float:
    """
    Rényi divergence D_α(P || Q).

    D_α(P||Q) = (1/(α-1)) × log(Σ pᵢ^α × qᵢ^(1-α))

    This generalizes KL-divergence (α→1).
    Used for comparing distributions (e.g., current vs. canonical Ihsān weights).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Clean
    mask = (p > 0) & (q > 0)
    p, q = p[mask], q[mask]
    if len(p) == 0:
        return float('inf')

    p, q = p / p.sum(), q / q.sum()

    if abs(alpha - 1.0) < 1e-10:
        # KL divergence (Shannon limit)
        return float(np.sum(p * np.log2(p / q)))

    sum_term = float(np.sum(p ** alpha * q ** (1 - alpha)))
    if sum_term <= 0:
        return float('inf')

    return float((1.0 / (alpha - 1.0)) * math.log2(sum_term))


# ═══════════════════════════════════════════════════════════════════════════════
# RATE-DISTORTION — Theoretical compression limits
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RateDistortionPoint:
    """Single point on the rate-distortion curve R(D)."""
    distortion: float   # D — expected distortion
    rate: float         # R(D) — minimum bits needed
    compression_ratio: float  # 1 - D (approximate)


@dataclass
class RateDistortionResult:
    """Complete rate-distortion analysis."""
    source_entropy: float              # H(X) — source entropy
    rd_curve: List[RateDistortionPoint]  # R(D) curve
    optimal_distortion: float          # D* for target compression
    optimal_rate: float                # R(D*) — bits needed
    achievable_compression: float      # Maximum compression at tolerance
    target_compression: float          # User's target (default 0.7)
    is_feasible: bool                  # Can we achieve target?
    efficiency: float                  # How close to theoretical limit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_entropy": self.source_entropy,
            "optimal_distortion": self.optimal_distortion,
            "optimal_rate": self.optimal_rate,
            "achievable_compression": self.achievable_compression,
            "target_compression": self.target_compression,
            "is_feasible": self.is_feasible,
            "efficiency": self.efficiency,
            "rd_curve_points": len(self.rd_curve),
        }


def estimate_rate_distortion(
    source_entropy: float,
    variance: float = 1.0,
    target_compression: float = SDPO_COMPRESSION_TARGET,
    num_points: int = RD_NUM_POINTS,
) -> RateDistortionResult:
    """
    Estimate rate-distortion curve for source compression.

    For Gaussian source (common assumption for LLM embedding spaces):
        R(D) = max(0, ½ × log₂(σ² / D))

    For discrete source with entropy H:
        R(D) ≈ H × (1 - D/D_max) for small D

    Args:
        source_entropy: Entropy of source in bits
        variance: Source variance (for Gaussian model)
        target_compression: Target compression ratio (0.7 = keep 70%)
        num_points: Points on R(D) curve

    Returns:
        RateDistortionResult with curve and optimal operating point
    """
    rd_curve: List[RateDistortionPoint] = []

    # Target distortion corresponding to compression target
    # Compression ratio = 1 - (bits_kept / bits_total)
    # So target distortion D ≈ 1 - compression_ratio
    target_distortion = 1.0 - target_compression

    # Compute R(D) curve
    distortions = np.linspace(0.001, RD_MAX_DISTORTION, num_points)

    for d in distortions:
        # Gaussian R(D)
        rate_gaussian = max(0.0, 0.5 * math.log2(variance / d)) if d > 0 else source_entropy

        # Discrete approximation
        rate_discrete = max(0.0, source_entropy * (1.0 - d / RD_MAX_DISTORTION))

        # Blended estimate (practical sources are between Gaussian and discrete)
        rate = 0.6 * rate_gaussian + 0.4 * rate_discrete

        rd_curve.append(RateDistortionPoint(
            distortion=float(d),
            rate=float(rate),
            compression_ratio=float(1.0 - d),
        ))

    # Find optimal operating point for target
    optimal_rate = max(0.0, 0.5 * math.log2(variance / target_distortion)) if target_distortion > 0 else source_entropy
    optimal_rate_discrete = max(0.0, source_entropy * (1.0 - target_distortion / RD_MAX_DISTORTION))
    optimal_rate = 0.6 * optimal_rate + 0.4 * optimal_rate_discrete

    # Achievable compression
    achievable = 1.0 - (optimal_rate / source_entropy) if source_entropy > 0 else 0.0

    # Feasibility
    is_feasible = achievable >= (1.0 - target_compression) * 0.8  # 80% of target is feasible

    # Efficiency: how close to theoretical limit
    efficiency = min(1.0, target_compression / max(achievable, 0.01)) if achievable > 0 else 0.0

    return RateDistortionResult(
        source_entropy=source_entropy,
        rd_curve=rd_curve,
        optimal_distortion=target_distortion,
        optimal_rate=optimal_rate,
        achievable_compression=achievable,
        target_compression=target_compression,
        is_feasible=is_feasible,
        efficiency=efficiency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SNR ENHANCEMENT — Plug Rényi-2 into SNR v2 diversity component
# ═══════════════════════════════════════════════════════════════════════════════


def enhanced_diversity_score(
    texts: List[str],
    alpha: float = DEFAULT_RENYI_ALPHA,
    shannon_weight: float = 0.4,
    renyi_weight: float = 0.4,
    anti_redundancy_weight: float = 0.2,
) -> Tuple[float, Dict[str, Any]]:
    """
    Enhanced diversity score using both Shannon and Rényi-2 entropy.

    This replaces the single Shannon entropy in SNR v2 with a blended
    measure that catches dominance patterns earlier.

    diversity = w₁ × H₁_norm + w₂ × H₂_norm + w₃ × (1 - redundancy)

    Args:
        texts: Input texts to measure diversity over
        alpha: Rényi order (default 2.0)
        shannon_weight: Weight for Shannon entropy
        renyi_weight: Weight for Rényi entropy
        anti_redundancy_weight: Weight for anti-redundancy

    Returns:
        (diversity_score, detail_dict)
    """
    if not texts:
        return 0.0, {"error": "empty_input"}

    # Build word frequency distribution
    all_words: List[str] = []
    for text in texts:
        all_words.extend(text.lower().split())

    if not all_words:
        return 0.0, {"error": "no_words"}

    word_counts: Dict[str, int] = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    total = len(all_words)
    probs = np.array(list(word_counts.values()), dtype=np.float64) / total

    # Compute Rényi entropy
    renyi_result = renyi_entropy(probs, alpha=alpha)

    # Compute inter-text redundancy (pairwise word overlap)
    redundancy = 0.0
    if len(texts) > 1:
        word_sets = [set(t.lower().split()) for t in texts]
        overlaps = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                if union:
                    overlap = len(word_sets[i] & word_sets[j]) / len(union)
                    overlaps.append(overlap)
        redundancy = float(np.mean(overlaps)) if overlaps else 0.0

    # Blended diversity score
    diversity = (
        shannon_weight * renyi_result.normalized +
        renyi_weight * renyi_result.normalized +  # H₂ always ≤ H₁, so more conservative
        anti_redundancy_weight * (1.0 - redundancy)
    )

    # Use Rényi-2 specifically for the Rényi component
    renyi_2 = renyi_entropy(probs, alpha=2.0)
    diversity = (
        shannon_weight * (renyi_result.shannon_entropy / max(renyi_result.max_entropy, 1.0)) +
        renyi_weight * renyi_2.normalized +
        anti_redundancy_weight * (1.0 - redundancy)
    )

    details = {
        "shannon_entropy": renyi_result.shannon_entropy,
        "renyi_2_entropy": renyi_2.entropy,
        "max_entropy": renyi_result.max_entropy,
        "shannon_normalized": renyi_result.shannon_entropy / max(renyi_result.max_entropy, 1.0),
        "renyi_2_normalized": renyi_2.normalized,
        "collision_probability": renyi_2.collision_probability,
        "effective_support": renyi_2.effective_support,
        "redundancy": redundancy,
        "diversity_score": diversity,
        "vocab_size": len(word_counts),
        "total_words": total,
    }

    return float(diversity), details


# ═══════════════════════════════════════════════════════════════════════════════
# SDPO COMPRESSION ADVISOR — Rate-distortion guided compression
# ═══════════════════════════════════════════════════════════════════════════════


class SDPOCompressionAdvisor:
    """
    Rate-distortion guided compression advisor for SDPO.

    Given source content, estimates:
    1. Theoretical minimum bits needed after compression
    2. Whether 70% compression target is achievable
    3. Which content should be prioritized for retention
    4. Expected quality loss at different compression levels

    Standing on: Shannon (1948), Blahut (1972), Cover & Thomas (2006)
    """

    def __init__(
        self,
        target_compression: float = SDPO_COMPRESSION_TARGET,
        quality_floor: float = UNIFIED_SNR_THRESHOLD,
    ):
        self.target_compression = target_compression
        self.quality_floor = quality_floor

    def analyze_compressibility(
        self,
        texts: List[str],
        importance_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze compressibility of a text collection.

        Returns rate-distortion analysis with practical recommendations.
        """
        if not texts:
            return {"error": "empty_input"}

        # Compute source entropy
        all_words = []
        for t in texts:
            all_words.extend(t.lower().split())

        if not all_words:
            return {"error": "no_words"}

        word_counts: Dict[str, int] = {}
        for w in all_words:
            word_counts[w] = word_counts.get(w, 0) + 1

        total = len(all_words)
        probs = np.array(list(word_counts.values()), dtype=np.float64) / total

        # Source entropy
        renyi_result = renyi_entropy(probs, alpha=2.0)
        source_entropy = renyi_result.shannon_entropy
        source_variance = float(np.var(probs))

        # Rate-distortion analysis
        rd_result = estimate_rate_distortion(
            source_entropy=source_entropy,
            variance=max(source_variance, 0.001),
            target_compression=self.target_compression,
        )

        # Per-text importance (Rényi-2 based)
        text_importance = []
        for i, text in enumerate(texts):
            words = text.lower().split()
            if not words:
                text_importance.append(0.0)
                continue

            # Information content: -log₂(p(text)) estimated via word entropy
            text_word_counts: Dict[str, int] = {}
            for w in words:
                text_word_counts[w] = text_word_counts.get(w, 0) + 1
            text_probs = np.array(list(text_word_counts.values()), dtype=np.float64)
            text_probs = text_probs / text_probs.sum()
            text_renyi = renyi_entropy(text_probs, alpha=2.0)

            # Higher entropy = more unique information = higher importance
            importance = text_renyi.normalized
            if importance_scores and i < len(importance_scores):
                importance = 0.5 * importance + 0.5 * importance_scores[i]
            text_importance.append(float(importance))

        # Rank texts by importance
        ranked_indices = sorted(range(len(texts)), key=lambda i: -text_importance[i])

        # Determine how many texts to keep at target compression
        keep_count = max(1, int(len(texts) * self.target_compression))

        return {
            "source_entropy_bits": source_entropy,
            "renyi_2_entropy_bits": renyi_result.entropy,
            "collision_probability": renyi_result.collision_probability,
            "rate_distortion": rd_result.to_dict(),
            "is_feasible": rd_result.is_feasible,
            "efficiency": rd_result.efficiency,
            "text_count": len(texts),
            "keep_count": keep_count,
            "compression_ratio": self.target_compression,
            "ranked_indices": ranked_indices[:keep_count],
            "text_importance": text_importance,
            "recommendation": (
                "compression_safe" if rd_result.is_feasible
                else "compression_risky"
            ),
        }


__all__ = [
    # Rényi entropy
    "renyi_entropy",
    "renyi_divergence",
    "RenyiResult",
    "DEFAULT_RENYI_ALPHA",
    # Rate-distortion
    "estimate_rate_distortion",
    "RateDistortionResult",
    "RateDistortionPoint",
    # SNR enhancement
    "enhanced_diversity_score",
    # SDPO compression
    "SDPOCompressionAdvisor",
]
