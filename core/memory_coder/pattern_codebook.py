"""
Pattern Codebook: indexed collection of synthesized cognitive patterns.

A codebook is an entropy-compressed dictionary of reusable reasoning patterns
distilled from accumulated agent memory. Each entry is a SynthesizedPattern
with an embedding centroid, keyword signature, and quality score.

The codebook supports two retrieval modes:
  1. Agent-DB backed -- delegates cosine search to the vector database.
  2. Linear scan     -- pure-Python fallback when no database is available.

Standing on Giants: Shannon (codebooks) + Deming (continuous improvement)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.integration.constants import SNR_THRESHOLD_T1_HIGH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numpy import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# SynthesizedPattern
# ---------------------------------------------------------------------------
@dataclass
class SynthesizedPattern:
    """A reusable cognitive pattern distilled from clustered memories."""

    pattern_id: str
    embedding: List[float]
    keywords: List[str]
    snr: float
    source_count: int
    access_count: int
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_strong(self) -> bool:
        """A pattern is *strong* when it exceeds T1 SNR and has broad support."""
        return self.snr >= SNR_THRESHOLD_T1_HIGH and self.source_count >= 10


# ---------------------------------------------------------------------------
# Cosine-similarity helpers
# ---------------------------------------------------------------------------


def _cosine_similarity_np(a: List[float], b: List[float]) -> float:
    """Cosine similarity using numpy."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _cosine_similarity_pure(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity fallback."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity, preferring numpy when available."""
    if _HAS_NUMPY:
        return _cosine_similarity_np(a, b)
    return _cosine_similarity_pure(a, b)


# ---------------------------------------------------------------------------
# PatternCodebook
# ---------------------------------------------------------------------------
class PatternCodebook:
    """Indexed collection of synthesized cognitive patterns.

    Parameters
    ----------
    agent_db : optional
        Any object implementing an ``AgentDBProtocol``-compatible interface
        (``search`` and ``store`` methods). When provided, patterns are
        persisted to the vector database and lookups use its ANN index.
    """

    def __init__(self, agent_db: Optional[Any] = None) -> None:
        self._agent_db = agent_db
        self._patterns: Dict[str, SynthesizedPattern] = {}

    # -- Mutation -----------------------------------------------------------

    def add(self, pattern: SynthesizedPattern) -> None:
        """Store a pattern in the codebook (and optionally in agent_db)."""
        self._patterns[pattern.pattern_id] = pattern
        if self._agent_db is not None:
            try:
                self._agent_db.store(
                    content=json.dumps(
                        {
                            "pattern_id": pattern.pattern_id,
                            "keywords": pattern.keywords,
                            "snr": pattern.snr,
                            "source_count": pattern.source_count,
                        }
                    ),
                    embedding=pattern.embedding,
                    metadata={"type": "synthesized_pattern", **pattern.metadata},
                )
            except Exception:
                logger.warning(
                    "Failed to persist pattern %s to agent_db", pattern.pattern_id
                )

    # -- Query --------------------------------------------------------------

    def lookup(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SynthesizedPattern]:
        """Return the *top_k* most similar patterns to *query_embedding*.

        Uses agent_db.search when available; falls back to linear scan.
        """
        if self._agent_db is not None:
            return self._lookup_via_agent_db(query_embedding, top_k)
        return self._lookup_linear(query_embedding, top_k)

    def contains_similar(
        self, pattern: SynthesizedPattern, threshold: float = 0.90
    ) -> bool:
        """Return ``True`` if a pattern with cosine similarity >= *threshold* exists."""
        for existing in self._patterns.values():
            sim = cosine_similarity(existing.embedding, pattern.embedding)
            if sim >= threshold:
                return True
        return False

    # -- Properties ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of patterns stored in the codebook."""
        return len(self._patterns)

    @property
    def strong_patterns(self) -> List[SynthesizedPattern]:
        """Subset of patterns that meet the *is_strong* criterion."""
        return [p for p in self._patterns.values() if p.is_strong]

    # -- Internal helpers ---------------------------------------------------

    def _lookup_linear(
        self, query_embedding: List[float], top_k: int
    ) -> List[SynthesizedPattern]:
        """Brute-force cosine scan over all stored patterns."""
        scored: List[Tuple[float, SynthesizedPattern]] = []
        for pattern in self._patterns.values():
            sim = cosine_similarity(pattern.embedding, query_embedding)
            scored.append((sim, pattern))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def _lookup_via_agent_db(
        self, query_embedding: List[float], top_k: int
    ) -> List[SynthesizedPattern]:
        """Delegate search to the agent database vector index."""
        try:
            results = self._agent_db.search(embedding=query_embedding, top_k=top_k)
            # Map results back to our in-memory patterns where possible
            matched: List[SynthesizedPattern] = []
            for result in results:
                content = getattr(result, "content", "") or ""
                try:
                    data = json.loads(content)
                    pid = data.get("pattern_id", "")
                except (json.JSONDecodeError, TypeError):
                    pid = ""
                if pid in self._patterns:
                    matched.append(self._patterns[pid])
            return matched[:top_k]
        except Exception:
            logger.warning("agent_db.search failed; falling back to linear scan")
            return self._lookup_linear(query_embedding, top_k)
