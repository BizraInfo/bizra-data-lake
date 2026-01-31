"""
SNR Maximizer — Signal Amplification & Noise Reduction Engine

Standing on Giants:
- Information Theory (Shannon, 1948)
- Signal Processing (Wiener, 1949)
- DATA4LLM IaaS Framework (Tsinghua, 2024)
- BIZRA SNR v2 (Node0 Genesis)

"In the noise of infinite information, the sovereign agent
 distinguishes signal from noise with mathematical precision."

SNR Maximization Strategy:
═══════════════════════════

    Input Stream           Signal Processing           Output
    ───────────    ───────────────────────────    ──────────
                   ┌─────────────────────────┐
    [Raw Data] ──► │  1. NOISE DETECTION     │
                   │     - Redundancy        │
                   │     - Inconsistency     │
                   │     - Low relevance     │
                   ├─────────────────────────┤
                   │  2. SIGNAL EXTRACTION   │ ──► [High SNR]
                   │     - Key insights      │
                   │     - Novel patterns    │
                   │     - Grounded facts    │
                   ├─────────────────────────┤
                   │  3. AMPLIFICATION       │
                   │     - Semantic boost    │
                   │     - Cross-validation  │
                   │     - Source weighting  │
                   └─────────────────────────┘

Formula:
    SNR = 10 × log₁₀(Signal_Power / Noise_Power)

    Where:
    Signal_Power = Relevance × Novelty × Groundedness × Coherence
    Noise_Power = Redundancy × Inconsistency × Ambiguity + ε
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Categories of noise to filter."""
    REDUNDANCY = "redundancy"           # Duplicate information
    INCONSISTENCY = "inconsistency"     # Contradictory claims
    AMBIGUITY = "ambiguity"             # Unclear statements
    IRRELEVANCE = "irrelevance"         # Off-topic content
    HALLUCINATION = "hallucination"     # Ungrounded claims
    VERBOSITY = "verbosity"             # Excessive wordiness
    BIAS = "bias"                       # Systematic distortion


class SignalType(Enum):
    """Categories of valuable signal."""
    INSIGHT = "insight"                 # Novel understanding
    EVIDENCE = "evidence"               # Grounded facts
    PATTERN = "pattern"                 # Recurring structures
    CAUSATION = "causation"             # Cause-effect relationships
    SYNTHESIS = "synthesis"             # Combined knowledge
    PREDICTION = "prediction"           # Future implications
    ACTIONABLE = "actionable"           # Concrete next steps


@dataclass
class NoiseProfile:
    """Profile of noise in a content piece."""
    redundancy: float = 0.0
    inconsistency: float = 0.0
    ambiguity: float = 0.0
    irrelevance: float = 0.0
    hallucination: float = 0.0
    verbosity: float = 0.0
    bias: float = 0.0

    @property
    def total_noise(self) -> float:
        """Compute total noise power."""
        return (
            self.redundancy * 0.20 +
            self.inconsistency * 0.25 +
            self.ambiguity * 0.15 +
            self.irrelevance * 0.15 +
            self.hallucination * 0.10 +
            self.verbosity * 0.05 +
            self.bias * 0.10
        )

    def to_dict(self) -> dict:
        return {
            "redundancy": self.redundancy,
            "inconsistency": self.inconsistency,
            "ambiguity": self.ambiguity,
            "irrelevance": self.irrelevance,
            "hallucination": self.hallucination,
            "verbosity": self.verbosity,
            "bias": self.bias,
            "total": self.total_noise,
        }


@dataclass
class SignalProfile:
    """Profile of signal strength in content."""
    relevance: float = 0.5
    novelty: float = 0.5
    groundedness: float = 0.5
    coherence: float = 0.5
    actionability: float = 0.5
    specificity: float = 0.5

    @property
    def total_signal(self) -> float:
        """Compute total signal power (geometric mean)."""
        values = [
            max(self.relevance, 1e-10),
            max(self.novelty, 1e-10),
            max(self.groundedness, 1e-10),
            max(self.coherence, 1e-10),
            max(self.actionability, 1e-10),
            max(self.specificity, 1e-10),
        ]
        return math.exp(sum(math.log(v) for v in values) / len(values))

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "novelty": self.novelty,
            "groundedness": self.groundedness,
            "coherence": self.coherence,
            "actionability": self.actionability,
            "specificity": self.specificity,
            "total": self.total_signal,
        }


@dataclass
class SNRAnalysis:
    """Complete SNR analysis result."""
    signal: SignalProfile
    noise: NoiseProfile
    snr_linear: float = 0.0
    snr_db: float = 0.0
    ihsan_achieved: bool = False
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.snr_linear = self.signal.total_signal / (self.noise.total_noise + 1e-10)
        self.snr_db = 10 * math.log10(max(self.snr_linear, 1e-10))
        self.ihsan_achieved = self.snr_linear >= 0.95

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.to_dict(),
            "noise": self.noise.to_dict(),
            "snr_linear": self.snr_linear,
            "snr_db": self.snr_db,
            "ihsan_achieved": self.ihsan_achieved,
            "recommendations": self.recommendations,
        }


class NoiseFilter:
    """
    Multi-stage noise filtering engine.

    Implements cascading filters to progressively remove noise
    while preserving signal integrity.
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.7,
        consistency_check: bool = True,
        verbosity_limit: float = 0.3,
    ):
        self.redundancy_threshold = redundancy_threshold
        self.consistency_check = consistency_check
        self.verbosity_limit = verbosity_limit

        # Seen content for redundancy detection
        self._seen_hashes: Set[int] = set()
        self._seen_concepts: Dict[str, int] = defaultdict(int)

    def _compute_hash(self, text: str) -> int:
        """Compute content hash for redundancy detection."""
        # Normalize and hash
        normalized = " ".join(text.lower().split())
        return hash(normalized)

    def _detect_redundancy(self, text: str) -> float:
        """Detect redundancy with previously seen content."""
        text_hash = self._compute_hash(text)

        # Exact duplicate
        if text_hash in self._seen_hashes:
            return 1.0

        # Concept-level redundancy
        words = set(text.lower().split())
        overlap_scores = []
        for concept, count in self._seen_concepts.items():
            if concept in words:
                overlap_scores.append(count / (count + 1))

        self._seen_hashes.add(text_hash)
        for word in words:
            if len(word) > 4:  # Skip short words
                self._seen_concepts[word] += 1

        if overlap_scores:
            return min(sum(overlap_scores) / len(overlap_scores), 1.0)
        return 0.0

    def _detect_ambiguity(self, text: str) -> float:
        """Detect ambiguous or vague language."""
        ambiguous_markers = [
            "maybe", "perhaps", "might", "could be", "possibly",
            "sort of", "kind of", "somewhat", "seems like",
            "i think", "i guess", "not sure", "unclear",
        ]

        text_lower = text.lower()
        matches = sum(1 for marker in ambiguous_markers if marker in text_lower)
        word_count = len(text.split())

        return min(matches / max(word_count / 10, 1), 1.0)

    def _detect_verbosity(self, text: str) -> float:
        """Detect excessive wordiness."""
        words = text.split()
        word_count = len(words)

        if word_count < 10:
            return 0.0

        # Check for filler phrases
        fillers = [
            "in order to", "due to the fact that", "at this point in time",
            "it is important to note that", "as a matter of fact",
            "for all intents and purposes", "in the event that",
        ]

        filler_count = sum(1 for f in fillers if f in text.lower())
        unique_ratio = len(set(words)) / word_count

        verbosity = (filler_count * 0.1) + (1 - unique_ratio) * 0.5
        return min(verbosity, 1.0)

    def analyze(self, text: str, context: Optional[str] = None) -> NoiseProfile:
        """Analyze text for noise components."""
        return NoiseProfile(
            redundancy=self._detect_redundancy(text),
            ambiguity=self._detect_ambiguity(text),
            verbosity=self._detect_verbosity(text),
            # These require more context/external validation
            inconsistency=0.0,
            irrelevance=0.0,
            hallucination=0.0,
            bias=0.0,
        )

    def filter(self, text: str, threshold: float = 0.5) -> Tuple[str, NoiseProfile]:
        """Filter noise from text, returning cleaned version."""
        noise = self.analyze(text)

        if noise.total_noise < threshold:
            return text, noise

        # Apply filtering strategies
        filtered = text

        # Remove verbose phrases
        if noise.verbosity > self.verbosity_limit:
            verbose_phrases = [
                "in order to", "due to the fact that", "at this point in time",
            ]
            for phrase in verbose_phrases:
                filtered = filtered.replace(phrase, "")

        return filtered.strip(), noise

    def reset(self):
        """Reset seen content tracking."""
        self._seen_hashes.clear()
        self._seen_concepts.clear()


class SignalAmplifier:
    """
    Signal amplification engine.

    Enhances valuable signals through:
    1. Cross-reference validation
    2. Source authority weighting
    3. Semantic enrichment
    4. Pattern reinforcement
    """

    def __init__(
        self,
        relevance_weight: float = 0.25,
        novelty_weight: float = 0.20,
        groundedness_weight: float = 0.25,
        coherence_weight: float = 0.15,
        actionability_weight: float = 0.15,
    ):
        self.weights = {
            "relevance": relevance_weight,
            "novelty": novelty_weight,
            "groundedness": groundedness_weight,
            "coherence": coherence_weight,
            "actionability": actionability_weight,
        }

        # Knowledge base for grounding checks
        self._known_facts: Set[str] = set()
        self._source_authority: Dict[str, float] = {}

    def _compute_relevance(self, text: str, query: str) -> float:
        """Compute relevance to query/context."""
        if not query:
            return 0.5

        text_words = set(text.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5

        overlap = len(text_words & query_words)
        return min(overlap / len(query_words), 1.0)

    def _compute_novelty(self, text: str) -> float:
        """Estimate information novelty."""
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        # Check for novel patterns
        novel_indicators = [
            "novel", "new", "first", "discover", "breakthrough",
            "insight", "finding", "reveal", "emerge",
        ]

        indicator_count = sum(1 for i in novel_indicators if i in text.lower())
        novelty_boost = min(indicator_count * 0.1, 0.3)

        return min(unique_ratio * 0.7 + novelty_boost + 0.3, 1.0)

    def _compute_groundedness(self, text: str, sources: Optional[List[str]] = None) -> float:
        """Compute factual groundedness."""
        # Check for citation markers
        citation_patterns = ["according to", "research shows", "study found", "data indicates"]
        citation_count = sum(1 for p in citation_patterns if p in text.lower())

        # Check against known facts
        grounded_in_facts = 0
        for fact in self._known_facts:
            if fact.lower() in text.lower():
                grounded_in_facts += 1

        base_score = 0.5
        citation_boost = min(citation_count * 0.1, 0.3)
        fact_boost = min(grounded_in_facts * 0.05, 0.2)

        return min(base_score + citation_boost + fact_boost, 1.0)

    def _compute_coherence(self, text: str) -> float:
        """Compute internal coherence."""
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.7

        # Check for logical connectors
        connectors = [
            "therefore", "thus", "hence", "because", "since",
            "however", "moreover", "furthermore", "consequently",
        ]

        connector_count = sum(1 for c in connectors if c in text.lower())

        # Penalize very long sentences (harder to follow)
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        length_penalty = max(0, (avg_sentence_len - 30) * 0.01)

        base_score = 0.6
        connector_boost = min(connector_count * 0.05, 0.2)

        return max(min(base_score + connector_boost - length_penalty, 1.0), 0.3)

    def _compute_actionability(self, text: str) -> float:
        """Compute actionability score."""
        action_patterns = [
            "should", "must", "need to", "can be", "try",
            "implement", "use", "apply", "consider", "ensure",
            "step", "first", "then", "finally", "next",
        ]

        text_lower = text.lower()
        action_count = sum(1 for p in action_patterns if p in text_lower)

        # Check for concrete examples
        example_patterns = ["example", "such as", "for instance", "e.g.", "like"]
        example_count = sum(1 for p in example_patterns if p in text_lower)

        base_score = 0.4
        action_boost = min(action_count * 0.05, 0.4)
        example_boost = min(example_count * 0.1, 0.2)

        return min(base_score + action_boost + example_boost, 1.0)

    def analyze(
        self,
        text: str,
        query: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> SignalProfile:
        """Analyze text for signal strength."""
        return SignalProfile(
            relevance=self._compute_relevance(text, query or ""),
            novelty=self._compute_novelty(text),
            groundedness=self._compute_groundedness(text, sources),
            coherence=self._compute_coherence(text),
            actionability=self._compute_actionability(text),
            specificity=self._compute_novelty(text) * 0.8,  # Related to novelty
        )

    def amplify(
        self,
        text: str,
        boost_factor: float = 1.2,
        query: Optional[str] = None,
    ) -> Tuple[str, SignalProfile]:
        """Amplify signal in text."""
        signal = self.analyze(text, query)

        # For now, return original with analysis
        # Future: Apply semantic enrichment
        return text, signal

    def add_known_fact(self, fact: str):
        """Add a verified fact to the knowledge base."""
        self._known_facts.add(fact.lower())

    def set_source_authority(self, source: str, authority: float):
        """Set authority weight for a source."""
        self._source_authority[source] = min(max(authority, 0.0), 1.0)


class SNRMaximizer:
    """
    Unified SNR Maximization Engine.

    Combines noise filtering and signal amplification to
    achieve maximum signal-to-noise ratio.

    Per DDAGI Constitution Article 7:
    "No inference shall proceed unless SNR ≥ 0.95 (Ihsān threshold)"
    """

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        auto_filter: bool = True,
        auto_amplify: bool = True,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.auto_filter = auto_filter
        self.auto_amplify = auto_amplify

        self.noise_filter = NoiseFilter()
        self.signal_amplifier = SignalAmplifier()

        # Statistics
        self.stats = {
            "analyses": 0,
            "ihsan_passes": 0,
            "ihsan_fails": 0,
            "avg_snr": 0.0,
        }

    def analyze(
        self,
        text: str,
        query: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> SNRAnalysis:
        """Perform complete SNR analysis."""
        noise = self.noise_filter.analyze(text)
        signal = self.signal_amplifier.analyze(text, query, sources)

        analysis = SNRAnalysis(signal=signal, noise=noise)

        # Generate recommendations
        if not analysis.ihsan_achieved:
            if noise.redundancy > 0.3:
                analysis.recommendations.append("Reduce redundant information")
            if noise.ambiguity > 0.3:
                analysis.recommendations.append("Clarify ambiguous statements")
            if signal.groundedness < 0.6:
                analysis.recommendations.append("Add citations or evidence")
            if signal.coherence < 0.6:
                analysis.recommendations.append("Improve logical flow")

        # Update stats
        self.stats["analyses"] += 1
        if analysis.ihsan_achieved:
            self.stats["ihsan_passes"] += 1
        else:
            self.stats["ihsan_fails"] += 1
        self.stats["avg_snr"] = (
            (self.stats["avg_snr"] * (self.stats["analyses"] - 1) + analysis.snr_linear)
            / self.stats["analyses"]
        )

        return analysis

    def maximize(
        self,
        text: str,
        query: Optional[str] = None,
        max_iterations: int = 3,
    ) -> Tuple[str, SNRAnalysis]:
        """
        Iteratively maximize SNR through filtering and amplification.

        Returns optimized text and final analysis.
        """
        current_text = text
        best_analysis = self.analyze(current_text, query)

        for i in range(max_iterations):
            if best_analysis.ihsan_achieved:
                break

            # Apply noise filtering
            if self.auto_filter:
                current_text, _ = self.noise_filter.filter(current_text)

            # Re-analyze
            analysis = self.analyze(current_text, query)

            if analysis.snr_linear > best_analysis.snr_linear:
                best_analysis = analysis
            else:
                break  # No improvement

        logger.info(
            f"SNR Maximization: {best_analysis.snr_linear:.3f} "
            f"({'PASS' if best_analysis.ihsan_achieved else 'FAIL'})"
        )

        return current_text, best_analysis

    def gate(self, text: str, query: Optional[str] = None) -> Tuple[bool, SNRAnalysis]:
        """
        Ihsān Gate: Check if content passes SNR threshold.

        Returns (passed, analysis).
        """
        analysis = self.analyze(text, query)
        passed = analysis.snr_linear >= self.ihsan_threshold

        if not passed:
            logger.warning(
                f"Ihsān Gate FAILED: SNR {analysis.snr_linear:.3f} < {self.ihsan_threshold}"
            )

        return passed, analysis

    def reset(self):
        """Reset internal state."""
        self.noise_filter.reset()
        self.stats = {
            "analyses": 0,
            "ihsan_passes": 0,
            "ihsan_fails": 0,
            "avg_snr": 0.0,
        }
