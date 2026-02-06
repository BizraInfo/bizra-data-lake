"""
SDPO Advantage Calculator — Token-Level Credit Assignment for BIZRA
═══════════════════════════════════════════════════════════════════════════════

Implements SDPO advantage function for token-level optimization:
A_i,t(ŷ_i,t) = log(π_θ(ŷ_i,t | x, f_i, y_i,<t) / π_θ(ŷ_i,t | x, y_i,<t))

This module provides the core SDPO advantage calculation that enables:
- Token-level credit assignment
- Rich feedback integration
- Self-teaching capability

Standing on Giants: Shannon (information theory) + SDPO Paper
Genesis Strict Synthesis v2.2.2
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Use numpy if available, otherwise fallback to pure Python
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo import SDPO_ADVANTAGE_THRESHOLD


@dataclass
class SDPOAdvantage:
    """Result of SDPO advantage calculation."""

    token_advantages: List[float]
    overall_advantage: float
    advantage_variance: float
    max_advantage: float
    min_advantage: float
    positive_ratio: float  # Ratio of tokens with positive advantage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_advantages": self.token_advantages,
            "overall_advantage": self.overall_advantage,
            "advantage_variance": self.advantage_variance,
            "max_advantage": self.max_advantage,
            "min_advantage": self.min_advantage,
            "positive_ratio": self.positive_ratio,
        }

    @property
    def is_beneficial(self) -> bool:
        """Returns True if feedback provides net positive advantage."""
        return self.overall_advantage > SDPO_ADVANTAGE_THRESHOLD


@dataclass
class SDPOFeedback:
    """Rich feedback for SDPO self-teaching."""

    text: str
    source: str  # "fate", "ihsan", "snr", "user", etc.
    confidence: float
    improvement_areas: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "improvement_areas": self.improvement_areas,
        }


class TokenProbabilityProvider(ABC):
    """Abstract interface for getting token probabilities from a model."""

    @abstractmethod
    async def get_logits(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> List[List[float]]:
        """Get logits for each token position."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass


class SDPOAdvantageCalculator:
    """
    SDPO Advantage Calculator for BIZRA.

    Calculates token-level advantages that represent how much each token
    benefits from feedback-guided generation vs. context-only generation.

    Formula: A_i,t = log(π_θ(ŷ_i,t | x, f_i, y_i,<t) / π_θ(ŷ_i,t | x, y_i,<t))

    Usage:
        calculator = SDPOAdvantageCalculator(model_provider)

        advantage = await calculator.calculate_advantages(
            question="What is sovereignty?",
            failed_attempt="Sovereignty is about control...",
            feedback="Your response lacks precision. Be more specific about...",
            corrected_attempt="Sovereignty is the supreme authority..."
        )

        if advantage.is_beneficial:
            # Apply SDPO gradient update
            ...
    """

    def __init__(
        self,
        model_provider: Optional[TokenProbabilityProvider] = None,
        epsilon: float = 1e-10,
    ):
        self.model_provider = model_provider
        self.epsilon = epsilon  # For numerical stability

    async def calculate_advantages(
        self,
        question: str,
        failed_attempt: str,
        feedback: str,
        corrected_attempt: str,
    ) -> SDPOAdvantage:
        """
        Calculate token-level advantages for SDPO update.

        Args:
            question: Original question/task
            failed_attempt: Initial unsuccessful response
            feedback: Rich feedback explaining the failure
            corrected_attempt: Improved response after feedback

        Returns:
            SDPOAdvantage with token-level and aggregate metrics
        """
        if self.model_provider is None:
            # Fallback to heuristic-based estimation
            return self._estimate_advantages_heuristic(
                question, failed_attempt, feedback, corrected_attempt
            )

        # Get token IDs for corrected attempt
        corrected_tokens = self.model_provider.tokenize(corrected_attempt)

        # Create prompts
        base_prompt = self._create_base_prompt(question, failed_attempt)
        feedback_prompt = self._create_feedback_prompt(
            question, failed_attempt, feedback
        )

        # Get logits for both contexts
        base_logits = await self.model_provider.get_logits(base_prompt)
        feedback_logits = await self.model_provider.get_logits(feedback_prompt)

        # Calculate softmax probabilities and advantages
        advantages = []
        for t in range(
            min(len(corrected_tokens), len(base_logits), len(feedback_logits))
        ):
            token_id = corrected_tokens[t]

            # Get probabilities (softmax)
            p_base = self._softmax_prob(base_logits[t], token_id)
            p_feedback = self._softmax_prob(feedback_logits[t], token_id)

            # Calculate log advantage
            if p_base > self.epsilon and p_feedback > self.epsilon:
                advantage = math.log(p_feedback / p_base)
            else:
                advantage = 0.0

            advantages.append(advantage)

        return self._compile_advantage_result(advantages)

    def _estimate_advantages_heuristic(
        self,
        question: str,
        failed_attempt: str,
        feedback: str,
        corrected_attempt: str,
    ) -> SDPOAdvantage:
        """
        Estimate advantages using heuristics when model unavailable.

        Uses text similarity and feedback analysis to estimate
        which parts of the corrected attempt benefited from feedback.
        """
        # Simple word-level analysis
        failed_words = set(failed_attempt.lower().split())
        corrected_words = corrected_attempt.lower().split()
        feedback_words = set(feedback.lower().split())

        advantages = []
        for word in corrected_words:
            word_lower = word.lower()

            # Heuristic: New words that appear in feedback get positive advantage
            if word_lower in feedback_words and word_lower not in failed_words:
                advantages.append(1.0)  # Strong positive
            elif word_lower not in failed_words:
                advantages.append(0.3)  # Moderate positive
            else:
                advantages.append(0.0)  # Neutral

        # Normalize by feedback quality indicators
        quality_keywords = ["improve", "correct", "instead", "should", "better"]
        quality_bonus = (
            sum(1 for kw in quality_keywords if kw in feedback.lower()) * 0.1
        )

        if advantages:
            if HAS_NUMPY:
                advantages = list(np.array(advantages) + quality_bonus)
            else:
                advantages = [a + quality_bonus for a in advantages]

        return self._compile_advantage_result(advantages)

    def _softmax_prob(self, logits: List[float], token_id: int) -> float:
        """Calculate softmax probability for a specific token."""
        if not logits or token_id >= len(logits):
            return self.epsilon

        if HAS_NUMPY:
            exp_logits = np.exp(np.array(logits) - np.max(logits))
            softmax = exp_logits / np.sum(exp_logits)
            return float(softmax[token_id])
        else:
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            total = sum(exp_logits)
            return exp_logits[token_id] / total if total > 0 else self.epsilon

    def _compile_advantage_result(self, advantages: List[float]) -> SDPOAdvantage:
        """Compile advantage list into structured result."""
        if not advantages:
            return SDPOAdvantage(
                token_advantages=[],
                overall_advantage=0.0,
                advantage_variance=0.0,
                max_advantage=0.0,
                min_advantage=0.0,
                positive_ratio=0.0,
            )

        if HAS_NUMPY:
            arr = np.array(advantages)
            overall = float(np.mean(arr))
            variance = float(np.var(arr))
            max_adv = float(np.max(arr))
            min_adv = float(np.min(arr))
            positive_ratio = float(np.sum(arr > 0) / len(arr))
        else:
            overall = sum(advantages) / len(advantages)
            mean = overall
            variance = sum((a - mean) ** 2 for a in advantages) / len(advantages)
            max_adv = max(advantages)
            min_adv = min(advantages)
            positive_ratio = sum(1 for a in advantages if a > 0) / len(advantages)

        return SDPOAdvantage(
            token_advantages=advantages,
            overall_advantage=overall,
            advantage_variance=variance,
            max_advantage=max_adv,
            min_advantage=min_adv,
            positive_ratio=positive_ratio,
        )

    def _create_base_prompt(self, question: str, attempt: str) -> str:
        """Create base prompt without feedback (for baseline probability)."""
        return f"""Question: {question}

Solution attempt:
{attempt}

Next token:"""

    def _create_feedback_prompt(
        self,
        question: str,
        attempt: str,
        feedback: str,
    ) -> str:
        """Create Self-Teacher prompt with feedback."""
        return f"""Question: {question}

Previous unsuccessful attempt:
{attempt}

Feedback on why it failed:
{feedback}

Correct solution:"""


class BIZRAFeedbackGenerator:
    """
    Generate SDPO-compatible feedback from BIZRA quality gates.

    Integrates with:
    - Ihsān scoring (excellence threshold)
    - SNR analysis (signal quality)
    - FATE verification (ethical compliance)
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold

    def generate_feedback(
        self,
        quality_check: Dict[str, Any],
    ) -> SDPOFeedback:
        """
        Generate SDPO-compatible feedback from BIZRA quality gate results.

        Args:
            quality_check: Dict with ihsan_score, snr, fate_compliant, etc.

        Returns:
            SDPOFeedback for self-teaching
        """
        feedback_parts = []
        improvement_areas = []
        confidence_scores = []

        # Ihsān feedback
        ihsan_score = quality_check.get("ihsan_score", 1.0)
        if ihsan_score < self.ihsan_threshold:
            issues = quality_check.get("ihsan_issues", ["general quality"])
            feedback_parts.append(
                f"Ihsān score too low ({ihsan_score:.3f} < {self.ihsan_threshold}). "
                f"Improve excellence in: {', '.join(issues)}"
            )
            improvement_areas.extend(issues)
            confidence_scores.append(0.9)

        # FATE compliance feedback
        if not quality_check.get("fate_compliant", True):
            violation = quality_check.get("fate_violation", "ethical constraint")
            correction = quality_check.get(
                "fate_correction", "ensure ethical compliance"
            )
            feedback_parts.append(
                f"FATE violation: {violation}. Correct by: {correction}"
            )
            improvement_areas.append("ethical_compliance")
            confidence_scores.append(0.95)

        # SNR feedback
        snr = quality_check.get("snr", 1.0)
        if snr < self.snr_threshold:
            feedback_parts.append(
                f"SNR too low ({snr:.3f} < {self.snr_threshold}). "
                f"Increase signal clarity and reduce noise."
            )
            improvement_areas.append("signal_quality")
            confidence_scores.append(0.85)

        # Compile feedback
        text = (
            "\n".join(feedback_parts)
            if feedback_parts
            else "Response meets quality standards."
        )
        confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 1.0
        )
        source = "bizra_quality_gates"

        return SDPOFeedback(
            text=text,
            source=source,
            confidence=confidence,
            improvement_areas=improvement_areas,
        )


# Convenience functions for direct use
async def calculate_sdpo_advantage(
    question: str,
    failed_attempt: str,
    feedback: str,
    corrected_attempt: str,
    model_provider: Optional[TokenProbabilityProvider] = None,
) -> SDPOAdvantage:
    """
    Convenience function for calculating SDPO advantages.

    Usage:
        advantage = await calculate_sdpo_advantage(
            question="What is 2+2?",
            failed_attempt="2+2 equals 5",
            feedback="Basic arithmetic error. 2+2=4, not 5.",
            corrected_attempt="2+2 equals 4"
        )
    """
    calculator = SDPOAdvantageCalculator(model_provider)
    return await calculator.calculate_advantages(
        question, failed_attempt, feedback, corrected_attempt
    )


def generate_bizra_feedback(quality_check: Dict[str, Any]) -> SDPOFeedback:
    """
    Convenience function for generating BIZRA-compatible SDPO feedback.

    Usage:
        feedback = generate_bizra_feedback({
            "ihsan_score": 0.85,
            "snr": 0.90,
            "fate_compliant": True,
        })
    """
    generator = BIZRAFeedbackGenerator()
    return generator.generate_feedback(quality_check)
