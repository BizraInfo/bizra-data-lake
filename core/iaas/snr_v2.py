"""
SNR v2 — Enhanced Signal-to-Noise Ratio with IaaS Integration

Standing on Giants:
- ARTE Engine (BIZRA v1): Symbolic-Neural SNR calculation
- DATA4LLM IaaS Framework: Quality dimensions
- DDAGI Constitution Article 7: Ihsān Threshold (0.95)
- Information Theory: Shannon entropy for diversity

"Signal-to-Noise Ratio measures the proportion of valuable information
 (signal) relative to irrelevant or redundant content (noise)."

SNR v2 Formula:
    SNR = (S × D × G × I) ^ (1/4)

Where:
    S = Signal Strength (relevance to query/task)
    D = Diversity (information entropy)
    G = Grounding (factual/symbolic backing)
    I = IaaS Score (data quality from 4 dimensions)

Ihsān Gate: SNR >= 0.95 required for production use.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SNRComponentsV2:
    """
    Components of SNR v2 calculation.

    Each component ranges from 0.0 to 1.0.
    Final SNR is the geometric mean of all components.
    """
    signal_strength: float = 0.0      # Relevance to query
    diversity: float = 0.0            # Information diversity
    grounding: float = 0.0            # Factual backing
    iaas_score: float = 0.0           # Data quality score

    # Sub-components for transparency
    semantic_relevance: float = 0.0   # Embedding similarity
    lexical_overlap: float = 0.0      # Keyword match
    redundancy: float = 0.0           # Pairwise similarity (lower = better)
    entropy: float = 0.0              # Shannon entropy of topics
    symbolic_coverage: float = 0.0    # Symbolic knowledge coverage
    neural_confidence: float = 0.0    # Neural model confidence

    # IaaS sub-scores
    inclusiveness: float = 0.0
    abundance: float = 0.0
    articulation: float = 0.0
    sanitization: float = 0.0

    @property
    def snr(self) -> float:
        """Compute final SNR as geometric mean."""
        components = [
            max(self.signal_strength, 1e-10),
            max(self.diversity, 1e-10),
            max(self.grounding, 1e-10),
            max(self.iaas_score, 1e-10),
        ]
        return math.exp(sum(math.log(c) for c in components) / len(components))

    @property
    def ihsan_achieved(self) -> bool:
        """Check if Ihsān threshold is met."""
        return self.snr >= 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "snr": self.snr,
            "ihsan_achieved": self.ihsan_achieved,
            "components": {
                "signal_strength": self.signal_strength,
                "diversity": self.diversity,
                "grounding": self.grounding,
                "iaas_score": self.iaas_score,
            },
            "sub_components": {
                "semantic_relevance": self.semantic_relevance,
                "lexical_overlap": self.lexical_overlap,
                "redundancy": self.redundancy,
                "entropy": self.entropy,
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
    Enhanced SNR Calculator with IaaS Integration.

    Computes Signal-to-Noise Ratio using:
    1. Signal Strength: Query-document relevance
    2. Diversity: Information entropy and redundancy
    3. Grounding: Symbolic-neural backing
    4. IaaS Score: Data quality from 4 dimensions

    This is the production-grade SNR calculator for BIZRA.
    """

    def __init__(
        self,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        diversity_alpha: float = 0.5,  # Balance between entropy and anti-redundancy
        grounding_symbolic_weight: float = 0.5,
    ):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.diversity_alpha = diversity_alpha
        self.grounding_symbolic_weight = grounding_symbolic_weight

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
        lexical_scores = [
            self._compute_lexical_overlap(query, text) for text in texts
        ]
        lexical_overlap = np.mean(lexical_scores) if lexical_scores else 0.0

        # Combined signal strength
        signal = (
            self.semantic_weight * semantic_relevance +
            self.lexical_weight * lexical_overlap
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
        redundancy = 0.0
        if embeddings is not None and len(embeddings) > 1:
            pairwise_sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    pairwise_sims.append(sim)
            redundancy = np.mean(pairwise_sims) if pairwise_sims else 0.0

        # Compute entropy (vocabulary diversity)
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())

        if not all_words:
            return 0.0, 0.0, redundancy

        # Word frequency distribution
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        total = len(all_words)
        probs = np.array([count / total for count in word_counts.values()])

        # Shannon entropy (normalized)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(word_counts)) if word_counts else 1.0
        normalized_entropy = entropy / max(max_entropy, 1.0)

        # Diversity = balance of entropy and anti-redundancy
        diversity = (
            self.diversity_alpha * normalized_entropy +
            (1 - self.diversity_alpha) * (1 - redundancy)
        )

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
            structured_keywords = ['database', 'kg', 'ontology', 'fact', 'verified', 'citation']
            grounded = sum(
                1 for s in symbolic_sources
                if any(kw in s.lower() for kw in structured_keywords)
            )
            symbolic_coverage = grounded / len(symbolic_sources) if symbolic_sources else 0.0

        # Neural confidence (model's confidence in the responses)
        neural_confidence = 0.7  # Default
        if neural_confidences:
            neural_confidence = float(np.mean(neural_confidences))

        # Combined grounding
        grounding = (
            self.grounding_symbolic_weight * symbolic_coverage +
            (1 - self.grounding_symbolic_weight) * neural_confidence
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
            components.inclusiveness = iaas_dimensions.get('inclusiveness', 0.8)
            components.abundance = iaas_dimensions.get('abundance', 0.8)
            components.articulation = iaas_dimensions.get('articulation', 0.8)
            components.sanitization = iaas_dimensions.get('sanitization', 0.8)
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
            'inclusiveness': 0.95,
            'abundance': 0.88,
            'articulation': 0.94,
            'sanitization': 0.91,
        },
    )

    print("\n" + "=" * 60)
    print("SNR v2 CALCULATION RESULTS")
    print("=" * 60)
    print(f"\nFinal SNR: {components.snr:.4f}")
    print(f"Ihsān Achieved: {components.ihsan_achieved}")
    print(f"\nComponents:")
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
