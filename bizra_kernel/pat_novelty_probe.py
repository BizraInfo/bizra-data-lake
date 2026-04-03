"""
PAT Novelty Probe — Semantic Distance Detection
================================================
Measures semantic novelty of insights relative to known patterns.

Constitution: constitution/pat_enforcement_v1.yaml
Threshold: novelty_score >= 0.75 (semantic distance from known patterns)

Integration:
- Vector embeddings for semantic comparison
- SAPE elevation system
- Pattern database
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("pat.novelty_probe")


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

NOVELTY_MINIMUM = 0.75
PATTERN_SIMILARITY_THRESHOLD = 0.85  # Above this = too similar to existing pattern


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnownPattern:
    """Represents a known pattern from the database."""
    pattern_id: str
    content: str
    embedding: Optional[List[float]]
    domain: str
    frequency: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "content": self.content,
            "domain": self.domain,
            "frequency": self.frequency,
            "timestamp": self.timestamp,
        }


@dataclass
class NoveltyResult:
    """Result of novelty analysis."""
    novelty_score: float
    passed: bool
    closest_patterns: List[KnownPattern]
    semantic_distances: List[float]
    evidence: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "novelty_score": self.novelty_score,
            "passed": self.passed,
            "closest_patterns": [p.to_dict() for p in self.closest_patterns],
            "semantic_distances": self.semantic_distances,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAT NOVELTY PROBE
# ═══════════════════════════════════════════════════════════════════════════════

class PATNoveltyProbe:
    """
    Measures semantic novelty of insights.

    Computes semantic distance from known patterns using embeddings.
    """

    def __init__(
        self,
        novelty_threshold: float = NOVELTY_MINIMUM,
        pattern_db_path: Optional[str] = None,
    ):
        """Initialize novelty probe."""
        self.novelty_threshold = novelty_threshold
        self.pattern_db_path = pattern_db_path

        # In-memory pattern cache (would be loaded from DB)
        self.known_patterns: List[KnownPattern] = []

        logger.info(
            f"PATNoveltyProbe initialized: threshold={novelty_threshold}"
        )

    async def probe(
        self,
        insight: str,
        embedding: Optional[List[float]] = None,
        domain: Optional[str] = None,
    ) -> NoveltyResult:
        """
        Probe novelty of an insight.

        Args:
            insight: The insight text to evaluate
            embedding: Optional pre-computed embedding
            domain: Optional domain context

        Returns:
            NoveltyResult with novelty score and evidence
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self._generate_embedding(insight)

        # Load known patterns (mock for now)
        if not self.known_patterns:
            await self._load_known_patterns(domain)

        # Compute semantic distances to known patterns
        distances = []
        closest_patterns = []

        for pattern in self.known_patterns:
            if pattern.embedding is None:
                continue

            # Compute cosine distance
            distance = await self._compute_cosine_distance(embedding, pattern.embedding)
            distances.append(distance)

            # Track closest patterns (top 3)
            if len(closest_patterns) < 3 or distance > min(
                d for d, _ in zip(distances[-3:], closest_patterns[-3:])
            ):
                closest_patterns.append(pattern)

        # Sort closest patterns by distance (descending)
        closest_patterns = sorted(
            zip(distances, closest_patterns), key=lambda x: x[0], reverse=True
        )[:3]

        semantic_distances = [d for d, _ in closest_patterns]
        closest_patterns = [p for _, p in closest_patterns]

        # Novelty score is the average distance to closest patterns
        # Higher distance = more novel
        if semantic_distances:
            novelty_score = np.mean(semantic_distances)
        else:
            # No known patterns = maximally novel
            novelty_score = 1.0

        passed = novelty_score >= self.novelty_threshold

        evidence = [
            f"Novelty score: {novelty_score:.4f} (threshold: {self.novelty_threshold})",
            f"Closest patterns: {len(closest_patterns)}",
        ]

        if closest_patterns:
            evidence.append(
                f"Most similar pattern: '{closest_patterns[0].content[:60]}...' "
                f"(distance: {semantic_distances[0]:.4f})"
            )

        if not passed:
            evidence.append(
                f"FAIL: Insight too similar to known patterns "
                f"(novelty {novelty_score:.4f} < {self.novelty_threshold})"
            )

        return NoveltyResult(
            novelty_score=novelty_score,
            passed=passed,
            closest_patterns=closest_patterns,
            semantic_distances=semantic_distances,
            evidence=evidence,
        )

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Integration point: Would use embedding model (e.g., nomic-embed-text).
        """
        # Mock implementation: Simple hash-based embedding
        # Real implementation would use Ollama embeddings API
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))

        # Generate 384-dimensional embedding (common size)
        embedding = np.random.randn(384)

        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    async def _load_known_patterns(self, domain: Optional[str] = None) -> None:
        """
        Load known patterns from database.

        Integration point: Would query pattern database or SAPE elevation cache.
        """
        # Mock implementation: Generate some known patterns
        mock_patterns = [
            "Use caching to improve performance",
            "Implement parallel processing for faster execution",
            "Add error handling for robustness",
            "Use indexing for faster database queries",
            "Apply load balancing for scalability",
        ]

        for i, content in enumerate(mock_patterns):
            pattern_id = hashlib.sha256(content.encode()).hexdigest()[:16]
            embedding = await self._generate_embedding(content)

            pattern = KnownPattern(
                pattern_id=pattern_id,
                content=content,
                embedding=embedding,
                domain=domain or "general",
                frequency=10 - i,  # Mock frequency
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            self.known_patterns.append(pattern)

        logger.info(f"Loaded {len(self.known_patterns)} known patterns")

    async def _compute_cosine_distance(
        self, embedding_a: List[float], embedding_b: List[float]
    ) -> float:
        """
        Compute cosine distance between two embeddings.

        Returns:
            Distance from 0.0 (identical) to 2.0 (opposite)
            In practice, most distances are 0.0-1.0
        """
        vec_a = np.array(embedding_a)
        vec_b = np.array(embedding_b)

        # Cosine similarity
        similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

        # Convert to distance (0 = identical, 1 = orthogonal, 2 = opposite)
        distance = 1.0 - similarity

        return float(distance)

    async def register_pattern(
        self,
        content: str,
        domain: str,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Register a new pattern in the database.

        Used when a novel insight is accepted and should be tracked.

        Args:
            content: Pattern content
            domain: Domain context
            embedding: Optional pre-computed embedding

        Returns:
            Pattern ID
        """
        pattern_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        if embedding is None:
            embedding = await self._generate_embedding(content)

        pattern = KnownPattern(
            pattern_id=pattern_id,
            content=content,
            embedding=embedding,
            domain=domain,
            frequency=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.known_patterns.append(pattern)

        logger.info(f"Registered new pattern: {pattern_id}")

        return pattern_id

    async def boost_novelty_with_hypergraph(
        self,
        insight: str,
        hypergraph_data: Dict[str, Any],
    ) -> float:
        """
        Boost novelty score using hypergraph relationships.

        Integration point: pat_hypergraph_booster.py

        Args:
            insight: The insight text
            hypergraph_data: Hypergraph relationship data

        Returns:
            Boosted novelty score
        """
        # Mock implementation: Would query hypergraph for relationship novelty
        base_result = await self.probe(insight)
        base_novelty = base_result.novelty_score

        # Check for novel hypergraph connections
        novel_connections = hypergraph_data.get("novel_connections", 0)

        # Boost factor based on novel connections (up to +0.15)
        boost_factor = min(novel_connections * 0.05, 0.15)

        boosted_novelty = min(base_novelty + boost_factor, 1.0)

        logger.info(
            f"Hypergraph boost: {base_novelty:.4f} → {boosted_novelty:.4f} "
            f"(+{boost_factor:.4f})"
        )

        return boosted_novelty


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Test novelty probe."""
    probe = PATNoveltyProbe()

    # Test case 1: Novel insight
    novel_insight = "Use quantum entanglement for zero-latency distributed consensus"
    result = await probe.probe(novel_insight, domain="Distributed Systems")

    print("Test 1 - Novel Insight:")
    print(f"  Insight: {novel_insight}")
    print(f"  Novelty Score: {result.novelty_score:.4f}")
    print(f"  Passed: {result.passed}")
    print(f"  Evidence: {result.evidence[0]}")
    print()

    # Test case 2: Common pattern
    common_pattern = "Use caching to improve performance"
    result = await probe.probe(common_pattern, domain="Performance")

    print("Test 2 - Common Pattern:")
    print(f"  Insight: {common_pattern}")
    print(f"  Novelty Score: {result.novelty_score:.4f}")
    print(f"  Passed: {result.passed}")
    print(f"  Evidence: {result.evidence[0]}")
    print()

    # Test case 3: Register new pattern
    new_pattern = "Apply blockchain consensus for distributed state synchronization"
    pattern_id = await probe.register_pattern(new_pattern, domain="Distributed Systems")

    print("Test 3 - Register Pattern:")
    print(f"  Pattern: {new_pattern}")
    print(f"  Pattern ID: {pattern_id}")
    print(f"  Total patterns: {len(probe.known_patterns)}")
    print()

    # Test case 4: Hypergraph boost
    boosted = await probe.boost_novelty_with_hypergraph(
        novel_insight,
        hypergraph_data={"novel_connections": 3},
    )

    print("Test 4 - Hypergraph Boost:")
    print(f"  Original novelty: {result.novelty_score:.4f}")
    print(f"  Boosted novelty: {boosted:.4f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
