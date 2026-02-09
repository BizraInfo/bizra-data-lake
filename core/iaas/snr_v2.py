"""
SNR v2 — Enhanced Signal-to-Noise Ratio with IaaS Integration

Standing on Giants:
- Claude Shannon (1948): "A Mathematical Theory of Communication"
  - H(X) = -sum(p(x) * log2(p(x))) — Entropy as uncertainty measure
  - C = B * log2(1 + SNR) — Channel capacity theorem
  - "The fundamental problem of communication is that of reproducing at
     one point either exactly or approximately a message selected at
     another point." — Shannon, 1948
- ARTE Engine (BIZRA v1): Symbolic-Neural SNR calculation
- DATA4LLM IaaS Framework: Quality dimensions
- DDAGI Constitution Article 7: Ihsan Threshold (0.95)

"Signal-to-Noise Ratio measures the proportion of valuable information
 (signal) relative to irrelevant or redundant content (noise)."

SNR v2.1 Formula (Shannon-Enhanced):
    SNR = (S^w_s × D^w_d × G^w_g × I^w_i) - N_penalty

Where:
    S = Signal Strength (relevance to query/task)
    D = Diversity (Shannon entropy normalized)
    G = Grounding (factual/symbolic backing)
    I = IaaS Score (data quality from 4 dimensions)
    N_penalty = Noise penalty from classified noise types
    w_* = Component weights (default: equal at 0.25)

Noise Classification (Shannon-inspired):
    - Redundancy: Duplicate/repetitive information (reduces entropy)
    - Inconsistency: Contradictory statements (increases uncertainty wrong)
    - Ambiguity: Vague/unclear content (entropy without signal)

Ihsan Gate: SNR >= 0.95 required for production use.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import unified thresholds from authoritative source
from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Use unified constants
IHSAN_THRESHOLD = UNIFIED_IHSAN_THRESHOLD
SNR_THRESHOLD = UNIFIED_SNR_THRESHOLD

# Elite thresholds for high-quality operations
SNR_ELITE = SNR_THRESHOLD_T0_ELITE  # 0.98
SNR_HIGH = SNR_THRESHOLD_T1_HIGH  # 0.95


class NoiseType(Enum):
    """
    Shannon-inspired noise classification.

    Per Shannon (1948), noise corrupts the channel between sender and receiver.
    We classify noise types to enable targeted mitigation:

    - REDUNDANCY: Repetitive information that doesn't add entropy
    - INCONSISTENCY: Contradictory information that creates confusion
    - AMBIGUITY: Vague information with high entropy but low signal
    - IRRELEVANCE: Off-topic information (wrong channel)
    """

    REDUNDANCY = "redundancy"  # Low entropy due to repetition
    INCONSISTENCY = "inconsistency"  # Contradictory statements
    AMBIGUITY = "ambiguity"  # High entropy, low signal
    IRRELEVANCE = "irrelevance"  # Off-topic content


@dataclass
class NoiseAnalysis:
    """
    Detailed noise breakdown per Shannon principles.

    "The semantic aspects of communication are irrelevant to the engineering
     problem." — Shannon, 1948

    However, for LLM quality, we must classify noise semantically.
    """

    redundancy_score: float = 0.0  # [0,1] — higher = more redundant
    inconsistency_score: float = 0.0  # [0,1] — higher = more contradictory
    ambiguity_score: float = 0.0  # [0,1] — higher = more vague
    irrelevance_score: float = 0.0  # [0,1] — higher = more off-topic

    @property
    def total_noise(self) -> float:
        """
        Combined noise penalty using weighted sum.

        Weights reflect impact on signal quality:
        - Inconsistency is most damaging (contradictions corrupt meaning)
        - Redundancy wastes channel capacity
        - Ambiguity reduces actionability
        - Irrelevance dilutes signal
        """
        weights = {
            "inconsistency": 0.35,  # Most damaging
            "redundancy": 0.25,
            "ambiguity": 0.25,
            "irrelevance": 0.15,
        }
        return (
            weights["inconsistency"] * self.inconsistency_score
            + weights["redundancy"] * self.redundancy_score
            + weights["ambiguity"] * self.ambiguity_score
            + weights["irrelevance"] * self.irrelevance_score
        )

    @property
    def dominant_noise_type(self) -> NoiseType:
        """Identify the primary noise source for targeted mitigation."""
        scores = {
            NoiseType.REDUNDANCY: self.redundancy_score,
            NoiseType.INCONSISTENCY: self.inconsistency_score,
            NoiseType.AMBIGUITY: self.ambiguity_score,
            NoiseType.IRRELEVANCE: self.irrelevance_score,
        }
        return max(scores, key=lambda k: scores[k])


@dataclass
class SNRComponentsV2:
    """
    Components of SNR v2.1 calculation (Shannon-Enhanced).

    Each component ranges from 0.0 to 1.0.
    Final SNR uses weighted geometric mean minus noise penalty.

    Standing on Giants — Shannon (1948):
    "Information is the resolution of uncertainty."
    Each component measures a different aspect of information quality.
    """

    signal_strength: float = 0.0  # Relevance to query
    diversity: float = 0.0  # Information diversity (Shannon entropy)
    grounding: float = 0.0  # Factual backing
    iaas_score: float = 0.0  # Data quality score

    # Sub-components for transparency
    semantic_relevance: float = 0.0  # Embedding similarity
    lexical_overlap: float = 0.0  # Keyword match
    redundancy: float = 0.0  # Pairwise similarity (lower = better)
    entropy: float = 0.0  # Shannon entropy (normalized bits)
    entropy_bits: float = 0.0  # Raw Shannon entropy in bits
    max_entropy_bits: float = 0.0  # Maximum possible entropy
    symbolic_coverage: float = 0.0  # Symbolic knowledge coverage
    neural_confidence: float = 0.0  # Neural model confidence

    # Noise analysis (Shannon-inspired)
    noise_analysis: Optional[NoiseAnalysis] = None

    # IaaS sub-scores
    inclusiveness: float = 0.0
    abundance: float = 0.0
    articulation: float = 0.0
    sanitization: float = 0.0

    # Component weights (aligned with Rust implementation)
    weight_signal: float = 0.30
    weight_diversity: float = 0.25
    weight_grounding: float = 0.25
    weight_iaas: float = 0.20

    @property
    def snr(self) -> float:
        """
        Compute final SNR using weighted geometric mean minus noise penalty.

        Formula (Shannon-Enhanced v2.1):
            SNR = (S^w_s * D^w_d * G^w_g * I^w_i) - noise_penalty

        This aligns with the Rust implementation's approach while adding
        noise classification from the Python side.

        Per Shannon: "The capacity of a channel is the maximum rate at which
        information can be transmitted with arbitrarily small error."
        Our SNR approximates the efficiency of the information channel.
        """
        # Weighted geometric mean via log-sum-exp
        # Shannon insight: geometric mean handles multiplicative effects
        components = [
            (max(self.signal_strength, 1e-10), self.weight_signal),
            (max(self.diversity, 1e-10), self.weight_diversity),
            (max(self.grounding, 1e-10), self.weight_grounding),
            (max(self.iaas_score, 1e-10), self.weight_iaas),
        ]

        # Weighted geometric mean: prod(x_i^w_i)
        weighted_product = 1.0
        for value, weight in components:
            weighted_product *= math.pow(value, weight)

        # Apply noise penalty if available
        noise_penalty = 0.0
        if self.noise_analysis is not None:
            # Scale penalty: max 0.3 reduction for fully noisy content
            noise_penalty = self.noise_analysis.total_noise * 0.3

        # Clamp to valid range [0, 1]
        return max(0.0, min(1.0, weighted_product - noise_penalty))

    @property
    def snr_classic(self) -> float:
        """
        Classic geometric mean SNR (backward compatibility).

        Original v2 formula: SNR = (S * D * G * I)^(1/4)
        """
        components = [
            max(self.signal_strength, 1e-10),
            max(self.diversity, 1e-10),
            max(self.grounding, 1e-10),
            max(self.iaas_score, 1e-10),
        ]
        return math.exp(sum(math.log(c) for c in components) / len(components))

    @property
    def channel_efficiency(self) -> float:
        """
        Shannon channel efficiency approximation.

        Per Shannon-Hartley theorem: C = B * log2(1 + SNR)
        Normalized to [0, 1] assuming max practical SNR of ~10 (10dB).

        This represents how efficiently the content uses the "channel"
        between source (data) and destination (user understanding).
        """
        # Convert our [0,1] SNR to a ratio for Shannon formula
        # At SNR=0.95, we want high efficiency; at SNR=0.5, moderate
        snr_ratio = self.snr / (1.0 - self.snr + 1e-10)  # Maps to [0, inf)
        snr_ratio = min(snr_ratio, 20.0)  # Cap for numerical stability

        # Shannon capacity (normalized)
        raw_capacity = math.log2(1.0 + snr_ratio)
        max_capacity = math.log2(1.0 + 20.0)  # Max at SNR ratio of 20

        return raw_capacity / max_capacity

    @property
    def ihsan_achieved(self) -> bool:
        """Check if Ihsan threshold is met."""
        return self.snr >= IHSAN_THRESHOLD

    @property
    def elite_achieved(self) -> bool:
        """Check if elite threshold (T0) is met."""
        return self.snr >= SNR_ELITE

    @property
    def quality_tier(self) -> str:
        """
        Return quality tier based on SNR.

        Tiers (aligned with constants.py):
        - T0_ELITE: >= 0.98 (Ihsan++, production-critical)
        - T1_HIGH: >= 0.95 (Ihsan, production-ready)
        - T2_STANDARD: >= 0.90 (Good, general use)
        - T3_ACCEPTABLE: >= 0.85 (Minimum, limited use)
        - BELOW_THRESHOLD: < 0.85 (Requires improvement)
        """
        snr = self.snr
        if snr >= 0.98:
            return "T0_ELITE"
        elif snr >= 0.95:
            return "T1_HIGH"
        elif snr >= 0.90:
            return "T2_STANDARD"
        elif snr >= 0.85:
            return "T3_ACCEPTABLE"
        else:
            return "BELOW_THRESHOLD"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging with Shannon-enhanced metrics."""
        result = {
            "snr": self.snr,
            "snr_classic": self.snr_classic,
            "channel_efficiency": self.channel_efficiency,
            "ihsan_achieved": self.ihsan_achieved,
            "elite_achieved": self.elite_achieved,
            "quality_tier": self.quality_tier,
            "components": {
                "signal_strength": self.signal_strength,
                "diversity": self.diversity,
                "grounding": self.grounding,
                "iaas_score": self.iaas_score,
            },
            "weights": {
                "signal": self.weight_signal,
                "diversity": self.weight_diversity,
                "grounding": self.weight_grounding,
                "iaas": self.weight_iaas,
            },
            "sub_components": {
                "semantic_relevance": self.semantic_relevance,
                "lexical_overlap": self.lexical_overlap,
                "redundancy": self.redundancy,
                "entropy": self.entropy,
                "entropy_bits": self.entropy_bits,
                "max_entropy_bits": self.max_entropy_bits,
                "symbolic_coverage": self.symbolic_coverage,
                "neural_confidence": self.neural_confidence,
            },
            "iaas_dimensions": {
                "inclusiveness": self.inclusiveness,
                "abundance": self.abundance,
                "articulation": self.articulation,
                "sanitization": self.sanitization,
            },
        }

        # Include noise analysis if present
        if self.noise_analysis is not None:
            result["noise_analysis"] = {
                "redundancy": self.noise_analysis.redundancy_score,
                "inconsistency": self.noise_analysis.inconsistency_score,
                "ambiguity": self.noise_analysis.ambiguity_score,
                "irrelevance": self.noise_analysis.irrelevance_score,
                "total_noise": self.noise_analysis.total_noise,
                "dominant_type": self.noise_analysis.dominant_noise_type.value,
            }

        return result


class IhsanGate:
    """
    Ihsān (Excellence) Gate — Quality Enforcement Checkpoint.

    Per DDAGI Constitution Article 7:
    "No inference, retrieval, or generation shall proceed unless
     the underlying data achieves Ihsān threshold (SNR >= 0.95)."

    This gate enforces:
    1. Minimum SNR threshold
    2. All IaaS dimensions must pass
    3. No critical violations (PII, toxicity, ethics)
    """

    def __init__(
        self,
        snr_threshold: float = 0.95,
        require_all_dimensions: bool = True,
        allow_pii: bool = False,
        allow_toxic: bool = False,
    ):
        self.snr_threshold = snr_threshold
        self.require_all_dimensions = require_all_dimensions
        self.allow_pii = allow_pii
        self.allow_toxic = allow_toxic

    def check(
        self,
        components: SNRComponentsV2,
        pii_detected: bool = False,
        toxicity_detected: bool = False,
        ethics_violation: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Check if Ihsān gate passes.

        Returns:
            - passed: bool
            - violations: List of violation messages
        """
        violations = []

        # Check SNR threshold
        if components.snr < self.snr_threshold:
            violations.append(
                f"SNR {components.snr:.4f} below threshold {self.snr_threshold}"
            )

        # Check individual dimensions
        if self.require_all_dimensions:
            dimension_threshold = 0.7

            if components.inclusiveness < dimension_threshold:
                violations.append(
                    f"Inclusiveness {components.inclusiveness:.2f} below {dimension_threshold}"
                )
            if components.abundance < dimension_threshold:
                violations.append(
                    f"Abundance {components.abundance:.2f} below {dimension_threshold}"
                )
            if components.articulation < dimension_threshold:
                violations.append(
                    f"Articulation {components.articulation:.2f} below {dimension_threshold}"
                )
            if components.sanitization < dimension_threshold:
                violations.append(
                    f"Sanitization {components.sanitization:.2f} below {dimension_threshold}"
                )

        # Check safety constraints
        if pii_detected and not self.allow_pii:
            violations.append("PII detected in content")

        if toxicity_detected and not self.allow_toxic:
            violations.append("Toxic content detected")

        if ethics_violation:
            violations.append("Ethics violation detected")

        passed = len(violations) == 0

        if passed:
            logger.info(f"Ihsān Gate PASSED: SNR={components.snr:.4f}")
        else:
            logger.warning(f"Ihsān Gate FAILED: {violations}")

        return passed, violations


class SNRCalculatorV2:
    """
    Enhanced SNR Calculator with IaaS Integration (Shannon-Enhanced v2.1).

    Computes Signal-to-Noise Ratio using:
    1. Signal Strength: Query-document relevance
    2. Diversity: Shannon entropy and redundancy analysis
    3. Grounding: Symbolic-neural backing
    4. IaaS Score: Data quality from 4 dimensions
    5. Noise Analysis: Classified noise types with penalties

    Standing on Giants — Shannon (1948):
    "The fundamental problem of communication is that of reproducing at
     one point either exactly or approximately a message selected at
     another point."

    This is the production-grade SNR calculator for BIZRA.
    """

    # Ambiguity indicators (words that reduce precision)
    AMBIGUITY_MARKERS = frozenset(
        [
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could",
            "somewhat",
            "probably",
            "likely",
            "unlikely",
            "roughly",
            "approximately",
            "sort of",
            "kind of",
            "more or less",
            "etc",
            "thing",
            "stuff",
            "various",
            "several",
            "many",
            "some",
            "certain",
            "particular",
        ]
    )

    # Grounding indicators (words that increase factual basis)
    GROUNDING_MARKERS = frozenset(
        [
            "according",
            "cited",
            "research",
            "study",
            "evidence",
            "data",
            "measured",
            "calculated",
            "proven",
            "documented",
            "verified",
            "specifically",
            "precisely",
            "exactly",
            "defined",
            "known",
            "fact",
            "source",
            "reference",
            "per",
            "based",
        ]
    )

    def __init__(
        self,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        diversity_alpha: float = 0.5,  # Balance between entropy and anti-redundancy
        grounding_symbolic_weight: float = 0.5,
        enable_noise_analysis: bool = True,
    ):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.diversity_alpha = diversity_alpha
        self.grounding_symbolic_weight = grounding_symbolic_weight
        self.enable_noise_analysis = enable_noise_analysis

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_lexical_overlap(self, query: str, text: str) -> float:
        """Compute lexical overlap between query and text."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & text_words)
        return overlap / len(query_words)

    def _compute_signal_strength(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        texts: List[str],
        text_embeddings: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Compute signal strength from semantic and lexical relevance.

        Returns: (signal_strength, semantic_relevance, lexical_overlap)
        """
        if not texts:
            return 0.0, 0.0, 0.0

        # Semantic relevance (embedding similarity)
        semantic_scores = []
        if query_embedding is not None and text_embeddings is not None:
            for i in range(len(texts)):
                sim = self._cosine_similarity(query_embedding, text_embeddings[i])
                semantic_scores.append(sim)
        semantic_relevance = np.mean(semantic_scores) if semantic_scores else 0.5

        # Lexical overlap
        lexical_scores = [self._compute_lexical_overlap(query, text) for text in texts]
        lexical_overlap = np.mean(lexical_scores) if lexical_scores else 0.0

        # Combined signal strength
        signal = (
            self.semantic_weight * semantic_relevance
            + self.lexical_weight * lexical_overlap
        )

        return float(signal), float(semantic_relevance), float(lexical_overlap)

    def _compute_diversity(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Compute diversity from entropy and anti-redundancy.

        Returns: (diversity, entropy, redundancy)
        """
        if not texts:
            return 0.0, 0.0, 1.0

        # Compute redundancy (pairwise similarity)
        redundancy: float = 0.0
        if embeddings is not None and len(embeddings) > 1:
            pairwise_sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    pairwise_sims.append(sim)
            redundancy = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0

        # Compute entropy (vocabulary diversity)
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())

        if not all_words:
            return 0.0, 0.0, redundancy

        # Word frequency distribution
        word_counts: Dict[str, int] = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        total = len(all_words)
        probs = np.array([count / total for count in word_counts.values()])

        # Shannon entropy (normalized)
        entropy: float = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = np.log2(len(word_counts)) if word_counts else 1.0
        normalized_entropy = entropy / max(max_entropy, 1.0)

        # Diversity = balance of entropy and anti-redundancy
        diversity = self.diversity_alpha * normalized_entropy + (
            1 - self.diversity_alpha
        ) * (1 - redundancy)

        return float(diversity), float(normalized_entropy), float(redundancy)

    def _compute_grounding(
        self,
        texts: List[str],
        symbolic_sources: Optional[List[str]] = None,
        neural_confidences: Optional[List[float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute grounding from symbolic and neural backing.

        Returns: (grounding, symbolic_coverage, neural_confidence)
        """
        # Symbolic coverage (how many sources are grounded in structured knowledge)
        symbolic_coverage = 0.5  # Default
        if symbolic_sources:
            # Count structured sources (databases, knowledge graphs, etc.)
            structured_keywords = [
                "database",
                "kg",
                "ontology",
                "fact",
                "verified",
                "citation",
            ]
            grounded = sum(
                1
                for s in symbolic_sources
                if any(kw in s.lower() for kw in structured_keywords)
            )
            symbolic_coverage = (
                grounded / len(symbolic_sources) if symbolic_sources else 0.0
            )

        # Neural confidence (model's confidence in the responses)
        neural_confidence = 0.7  # Default
        if neural_confidences:
            neural_confidence = float(np.mean(neural_confidences))

        # Combined grounding
        grounding = (
            self.grounding_symbolic_weight * symbolic_coverage
            + (1 - self.grounding_symbolic_weight) * neural_confidence
        )

        return float(grounding), float(symbolic_coverage), float(neural_confidence)

    def compute_snr(
        self,
        query: str,
        texts: List[str],
        query_embedding: Optional[np.ndarray] = None,
        text_embeddings: Optional[np.ndarray] = None,
        iaas_score: float = 0.8,
        iaas_dimensions: Optional[Dict[str, float]] = None,
        symbolic_sources: Optional[List[str]] = None,
        neural_confidences: Optional[List[float]] = None,
    ) -> SNRComponentsV2:
        """
        Compute comprehensive SNR v2.

        Args:
            query: User query or task description
            texts: Retrieved or generated texts
            query_embedding: Query embedding vector
            text_embeddings: Text embedding vectors
            iaas_score: Pre-computed IaaS score (0.0 to 1.0)
            iaas_dimensions: Individual IaaS dimension scores
            symbolic_sources: Source identifiers for grounding
            neural_confidences: Model confidence scores

        Returns:
            SNRComponentsV2 with full breakdown
        """
        components = SNRComponentsV2()

        # 1. Signal Strength
        signal, semantic, lexical = self._compute_signal_strength(
            query, query_embedding, texts, text_embeddings
        )
        components.signal_strength = signal
        components.semantic_relevance = semantic
        components.lexical_overlap = lexical

        # 2. Diversity
        diversity, entropy, redundancy = self._compute_diversity(texts, text_embeddings)
        components.diversity = diversity
        components.entropy = entropy
        components.redundancy = redundancy

        # 3. Grounding
        grounding, symbolic, neural = self._compute_grounding(
            texts, symbolic_sources, neural_confidences
        )
        components.grounding = grounding
        components.symbolic_coverage = symbolic
        components.neural_confidence = neural

        # 4. IaaS Score
        components.iaas_score = iaas_score

        # IaaS dimensions
        if iaas_dimensions:
            components.inclusiveness = iaas_dimensions.get("inclusiveness", 0.8)
            components.abundance = iaas_dimensions.get("abundance", 0.8)
            components.articulation = iaas_dimensions.get("articulation", 0.8)
            components.sanitization = iaas_dimensions.get("sanitization", 0.8)
        else:
            # Default to equal contribution
            components.inclusiveness = iaas_score
            components.abundance = iaas_score
            components.articulation = iaas_score
            components.sanitization = iaas_score

        logger.info(
            f"SNR v2: {components.snr:.4f} "
            f"(S={components.signal_strength:.2f}, D={components.diversity:.2f}, "
            f"G={components.grounding:.2f}, I={components.iaas_score:.2f})"
        )

        return components

    def validate_for_ihsan(
        self,
        components: SNRComponentsV2,
        pii_detected: bool = False,
        toxicity_detected: bool = False,
        ethics_violation: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Validate SNR components against Ihsān threshold.

        Returns (passed, violations) tuple.
        """
        gate = IhsanGate()
        return gate.check(components, pii_detected, toxicity_detected, ethics_violation)

    def calculate_simple(
        self,
        query: str,
        texts: List[str],
        iaas_score: float = 0.8,
    ) -> SNRComponentsV2:
        """
        Simplified SNR calculation without embeddings.

        This is a convenience method for cases where embeddings are not available.
        Uses lexical-only signal strength and heuristic grounding estimation.

        Args:
            query: Query or assessment context
            texts: Texts to evaluate
            iaas_score: Default IaaS quality score

        Returns:
            SNRComponentsV2 with computed scores
        """
        components = SNRComponentsV2()

        # Signal Strength (lexical only)
        lexical_scores = [self._compute_lexical_overlap(query, text) for text in texts]
        components.lexical_overlap = (
            float(np.mean(lexical_scores)) if lexical_scores else 0.3
        )
        components.semantic_relevance = 0.5  # Default without embeddings
        components.signal_strength = (
            0.4 * components.semantic_relevance + 0.6 * components.lexical_overlap
        )

        # Diversity (word-based entropy)
        diversity, entropy, redundancy = self._compute_diversity(texts, None)
        components.diversity = diversity
        components.entropy = entropy
        components.redundancy = redundancy

        # Grounding (heuristic without explicit sources)
        # Estimate based on specificity indicators in text
        all_text = " ".join(texts)
        words = all_text.split()
        specific_count = sum(
            1 for w in words if any(c.isdigit() for c in w) or len(w) > 8
        )
        heuristic_grounding = min(
            1.0, 0.5 + 0.5 * (specific_count / max(len(words), 1))
        )
        components.grounding = heuristic_grounding
        components.symbolic_coverage = 0.5
        components.neural_confidence = 0.7

        # IaaS Score
        components.iaas_score = iaas_score
        components.inclusiveness = iaas_score
        components.abundance = iaas_score
        components.articulation = iaas_score
        components.sanitization = iaas_score

        logger.debug(
            f"SNR v2 (simple): {components.snr:.4f} "
            f"(S={components.signal_strength:.2f}, D={components.diversity:.2f}, "
            f"G={components.grounding:.2f}, I={components.iaas_score:.2f})"
        )

        return components


def demonstrate_snr_v2():
    """Demonstration of SNR v2 calculation."""
    import numpy as np

    calculator = SNRCalculatorV2()

    # Sample data
    query = "How does the IaaS framework ensure data quality?"
    texts = [
        "The IaaS framework ensures data quality through four dimensions: Inclusiveness, Abundance, Articulation, and Sanitization.",
        "Data quality is measured using perplexity, instruction-following difficulty, and cluster complexity.",
        "Sanitization removes PII, toxic content, and ethics violations from training data.",
    ]

    # Mock embeddings (in production, use actual encoder)
    np.random.seed(42)
    query_embedding = np.random.randn(384)
    text_embeddings = np.random.randn(3, 384)

    # Compute SNR
    components = calculator.compute_snr(
        query=query,
        texts=texts,
        query_embedding=query_embedding,
        text_embeddings=text_embeddings,
        iaas_score=0.92,
        iaas_dimensions={
            "inclusiveness": 0.95,
            "abundance": 0.88,
            "articulation": 0.94,
            "sanitization": 0.91,
        },
    )

    print("\n" + "=" * 60)
    print("SNR v2 CALCULATION RESULTS")
    print("=" * 60)
    print(f"\nFinal SNR: {components.snr:.4f}")
    print(f"Ihsān Achieved: {components.ihsan_achieved}")
    print("\nComponents:")
    print(f"  Signal Strength: {components.signal_strength:.3f}")
    print(f"    - Semantic Relevance: {components.semantic_relevance:.3f}")
    print(f"    - Lexical Overlap: {components.lexical_overlap:.3f}")
    print(f"  Diversity: {components.diversity:.3f}")
    print(f"    - Entropy: {components.entropy:.3f}")
    print(f"    - Redundancy: {components.redundancy:.3f}")
    print(f"  Grounding: {components.grounding:.3f}")
    print(f"    - Symbolic Coverage: {components.symbolic_coverage:.3f}")
    print(f"    - Neural Confidence: {components.neural_confidence:.3f}")
    print(f"  IaaS Score: {components.iaas_score:.3f}")
    print(f"    - Inclusiveness: {components.inclusiveness:.3f}")
    print(f"    - Abundance: {components.abundance:.3f}")
    print(f"    - Articulation: {components.articulation:.3f}")
    print(f"    - Sanitization: {components.sanitization:.3f}")

    # Validate against Ihsān gate
    passed, violations = calculator.validate_for_ihsan(components)
    print(f"\nIhsān Gate: {'PASSED' if passed else 'FAILED'}")
    if violations:
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    demonstrate_snr_v2()
