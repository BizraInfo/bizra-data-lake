"""
Embedding Quality Gate — Validates embedding vectors before retrieval.

Rejects:
- Zero vectors (norm < min_norm)
- Uniform distributions (entropy_ratio > max_entropy_ratio)

Standing on Giants: Shannon (1948, entropy as quality signal)
Artifact: core/embedding/quality_gate.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GateResult:
    """Result of embedding quality validation."""

    passed: bool
    reason: str
    score: float


class EmbeddingQualityGate:
    """
    Validates embedding vectors before they enter the retrieval pipeline.

    Two checks:
    1. L2 norm must exceed min_norm (rejects zero/near-zero vectors)
    2. Shannon entropy ratio must be below max_entropy_ratio
       (rejects uniform distributions that carry no information)
    """

    def __init__(
        self,
        min_norm: float = 0.1,
        max_entropy_ratio: float = 0.98,
    ) -> None:
        self.min_norm = min_norm
        self.max_entropy_ratio = max_entropy_ratio

    def validate(self, embedding: list[float]) -> GateResult:
        """Validate an embedding vector."""
        if not embedding:
            return GateResult(passed=False, reason="empty_embedding", score=0.0)

        # Check 1: L2 norm
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm < self.min_norm:
            return GateResult(
                passed=False,
                reason="embedding_norm_too_low",
                score=norm,
            )

        # Check 2: Shannon entropy of absolute value distribution
        abs_values = [abs(x) for x in embedding]
        total = sum(abs_values)
        if total < 1e-10:
            return GateResult(
                passed=False,
                reason="embedding_norm_too_low",
                score=0.0,
            )

        probs = [v / total for v in abs_values]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(embedding))
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0

        if entropy_ratio > self.max_entropy_ratio:
            return GateResult(
                passed=False,
                reason="embedding_too_uniform",
                score=entropy_ratio,
            )

        # Passed — score is 1.0 minus entropy_ratio (higher = more informative)
        return GateResult(
            passed=True,
            reason="ok",
            score=1.0 - entropy_ratio,
        )
