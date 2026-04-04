"""
Pattern Extractor - Success/Failure Pattern Mining
===================================================
Extracts patterns from execution history for SONA learning
and SAPE elevation with deduplication and similarity scoring.

From the Blueprint:
    - Extract success patterns from positive outcomes
    - Extract failure patterns for error analysis
    - Compute pattern hashes for deduplication
    - Pattern similarity scoring using embeddings

Key Features:
    - Pattern type classification (success/failure/mixed)
    - Sequence-based pattern detection
    - Hash-based deduplication
    - Embedding similarity for pattern clustering
    - Integration with SAPE elevation threshold (>3 repetitions)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class PatternType(str, Enum):
    """Types of execution patterns."""

    SUCCESS = "success"  # Consistently successful execution path
    FAILURE = "failure"  # Consistently failing execution path
    MIXED = "mixed"  # Variable outcomes
    RECOVERY = "recovery"  # Failure followed by success
    DEGRADATION = "degradation"  # Success followed by failure


class PatternScope(str, Enum):
    """Scope of the pattern."""

    AGENT = "agent"  # Single agent behavior
    CATEGORY = "category"  # Task category pattern
    SEQUENCE = "sequence"  # Multi-step sequence
    GLOBAL = "global"  # System-wide pattern


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION PATTERN
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExecutionPattern:
    """
    A detected execution pattern.

    Attributes:
        pattern_id: Unique identifier (hash of signature)
        pattern_type: Success, failure, or mixed
        scope: Agent, category, sequence, or global
        signature: The pattern signature (sequence of elements)
        occurrence_count: Number of times pattern occurred
        success_count: Number of successful occurrences
        failure_count: Number of failed occurrences
        avg_quality: Average quality score across occurrences
        avg_latency_ms: Average latency across occurrences
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of last occurrence
        related_patterns: IDs of similar/related patterns
        metadata: Additional pattern metadata
    """

    pattern_id: str
    pattern_type: PatternType
    scope: PatternScope
    signature: List[str]

    occurrence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0

    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    related_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Embedding for similarity computation
    _embedding: Optional[List[float]] = field(default=None, repr=False)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.occurrence_count == 0:
            return 0.0
        return self.success_count / self.occurrence_count

    @property
    def is_consistent(self) -> bool:
        """Check if pattern is consistently success or failure."""
        rate = self.success_rate
        return rate >= 0.9 or rate <= 0.1

    @property
    def should_elevate(self) -> bool:
        """Check if pattern should be elevated to SAPE (>3 repetitions, high success)."""
        return (
            self.occurrence_count > 3
            and self.success_rate >= 0.7
            and self.pattern_type == PatternType.SUCCESS
        )

    def update_metrics(
        self,
        success: bool,
        quality: float,
        latency_ms: float,
    ) -> None:
        """Update pattern metrics with new occurrence."""
        self.occurrence_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Running average
        n = self.occurrence_count
        self.avg_quality = (self.avg_quality * (n - 1) + quality) / n
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1) + latency_ms) / n
        self.last_seen = datetime.now(timezone.utc).isoformat()

        # Update pattern type based on new success rate
        rate = self.success_rate
        if rate >= 0.8:
            self.pattern_type = PatternType.SUCCESS
        elif rate <= 0.2:
            self.pattern_type = PatternType.FAILURE
        else:
            self.pattern_type = PatternType.MIXED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "scope": self.scope.value,
            "signature": self.signature,
            "occurrence_count": self.occurrence_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "avg_quality": self.avg_quality,
            "avg_latency_ms": self.avg_latency_ms,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "related_patterns": self.related_patterns,
            "should_elevate": self.should_elevate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPattern":
        """Deserialize from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            scope=PatternScope(data["scope"]),
            signature=data["signature"],
            occurrence_count=data.get("occurrence_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            avg_quality=data.get("avg_quality", 0.0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
            related_patterns=data.get("related_patterns", []),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION RECORD (INPUT TYPE)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExecutionRecord:
    """Record of a single execution for pattern extraction."""

    task_id: str
    task_category: str
    agent_name: str
    success: bool
    quality_score: float
    latency_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════


class PatternExtractor:
    """
    Extracts execution patterns from history for learning and SAPE elevation.

    Features:
        - Success/failure pattern detection
        - Sequence-based pattern discovery
        - Hash-based deduplication
        - Embedding-based similarity scoring
        - SAPE elevation support (>3 repetitions)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        similarity_threshold: float = 0.8,
        max_sequence_length: int = 5,
    ):
        """
        Initialize Pattern Extractor.

        Args:
            embedding_dim: Dimension for pattern embeddings
            similarity_threshold: Threshold for pattern similarity
            max_sequence_length: Maximum length for sequence patterns
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_sequence_length = max_sequence_length

        # Pattern storage
        self._patterns: Dict[str, ExecutionPattern] = {}

        # Hash to embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Sequence buffer for sequence pattern detection
        self._sequence_buffer: List[ExecutionRecord] = []

        logger.info("PatternExtractor initialized")

    def extract_success_patterns(
        self,
        history: Sequence[ExecutionRecord],
    ) -> List[ExecutionPattern]:
        """
        Extract success patterns from execution history.

        Args:
            history: List of execution records

        Returns:
            List of success patterns sorted by occurrence count
        """
        # Filter successful executions
        successes = [r for r in history if r.success]

        # Extract patterns
        for record in successes:
            self._process_record(record)

        # Return success patterns
        patterns = [
            p for p in self._patterns.values() if p.pattern_type == PatternType.SUCCESS
        ]
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)

        logger.debug(f"Extracted {len(patterns)} success patterns")
        return patterns

    def extract_failure_patterns(
        self,
        history: Sequence[ExecutionRecord],
    ) -> List[ExecutionPattern]:
        """
        Extract failure patterns for error analysis.

        Args:
            history: List of execution records

        Returns:
            List of failure patterns sorted by occurrence count
        """
        # Filter failed executions
        failures = [r for r in history if not r.success]

        # Extract patterns
        for record in failures:
            self._process_record(record)

        # Return failure patterns
        patterns = [
            p for p in self._patterns.values() if p.pattern_type == PatternType.FAILURE
        ]
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)

        logger.debug(f"Extracted {len(patterns)} failure patterns")
        return patterns

    def extract_all_patterns(
        self,
        history: Sequence[ExecutionRecord],
    ) -> List[ExecutionPattern]:
        """
        Extract all patterns from execution history.

        Args:
            history: List of execution records

        Returns:
            List of all patterns sorted by occurrence count
        """
        # Process all records
        for record in history:
            self._process_record(record)

        # Extract sequence patterns
        self._extract_sequence_patterns(history)

        # Sort by occurrence
        patterns = list(self._patterns.values())
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)

        logger.debug(f"Extracted {len(patterns)} total patterns")
        return patterns

    def _process_record(self, record: ExecutionRecord) -> None:
        """Process a single execution record."""
        # Agent pattern
        agent_hash = self.compute_pattern_hash([f"agent:{record.agent_name}"])
        self._update_or_create_pattern(
            agent_hash,
            [f"agent:{record.agent_name}"],
            PatternScope.AGENT,
            record,
        )

        # Category pattern
        category_hash = self.compute_pattern_hash([f"category:{record.task_category}"])
        self._update_or_create_pattern(
            category_hash,
            [f"category:{record.task_category}"],
            PatternScope.CATEGORY,
            record,
        )

        # Agent-category combination
        combo_hash = self.compute_pattern_hash(
            [f"agent:{record.agent_name}", f"category:{record.task_category}"]
        )
        self._update_or_create_pattern(
            combo_hash,
            [f"agent:{record.agent_name}", f"category:{record.task_category}"],
            PatternScope.AGENT,
            record,
        )

        # Add to sequence buffer
        self._sequence_buffer.append(record)
        if len(self._sequence_buffer) > 100:
            self._sequence_buffer = self._sequence_buffer[-100:]

    def _update_or_create_pattern(
        self,
        pattern_id: str,
        signature: List[str],
        scope: PatternScope,
        record: ExecutionRecord,
    ) -> None:
        """Update existing pattern or create new one."""
        if pattern_id not in self._patterns:
            # Determine initial type based on success
            initial_type = (
                PatternType.SUCCESS if record.success else PatternType.FAILURE
            )

            self._patterns[pattern_id] = ExecutionPattern(
                pattern_id=pattern_id,
                pattern_type=initial_type,
                scope=scope,
                signature=signature,
            )

        pattern = self._patterns[pattern_id]
        pattern.update_metrics(
            record.success,
            record.quality_score,
            record.latency_ms,
        )

    def _extract_sequence_patterns(
        self,
        history: Sequence[ExecutionRecord],
    ) -> None:
        """Extract sequence patterns from history."""
        if len(history) < 2:
            return

        # Sliding window for sequence detection
        for window_size in range(
            2, min(self.max_sequence_length + 1, len(history) + 1)
        ):
            for i in range(len(history) - window_size + 1):
                window = history[i : i + window_size]

                # Create signature from window
                signature = [
                    f"{r.agent_name}:{r.task_category}:{1 if r.success else 0}"
                    for r in window
                ]

                # Compute hash
                pattern_hash = self.compute_pattern_hash(signature)

                # Check if pattern already exists
                if pattern_hash in self._patterns:
                    # Update existing pattern
                    pattern = self._patterns[pattern_hash]
                    success = all(r.success for r in window)
                    avg_quality = sum(r.quality_score for r in window) / len(window)
                    avg_latency = sum(r.latency_ms for r in window) / len(window)
                    pattern.update_metrics(success, avg_quality, avg_latency)
                else:
                    # Create new sequence pattern
                    success = all(r.success for r in window)
                    initial_type = (
                        PatternType.SUCCESS if success else PatternType.FAILURE
                    )

                    pattern = ExecutionPattern(
                        pattern_id=pattern_hash,
                        pattern_type=initial_type,
                        scope=PatternScope.SEQUENCE,
                        signature=signature,
                    )
                    pattern.update_metrics(
                        success,
                        sum(r.quality_score for r in window) / len(window),
                        sum(r.latency_ms for r in window) / len(window),
                    )
                    self._patterns[pattern_hash] = pattern

    def compute_pattern_hash(self, signature: Sequence[str]) -> str:
        """
        Compute deterministic hash for pattern deduplication.

        Args:
            signature: List of pattern elements

        Returns:
            16-character hex hash
        """
        # Sort for order-independent hashing (for non-sequence patterns)
        if len(signature) > 0 and not any(
            ":" in s and s.count(":") > 1 for s in signature
        ):
            # Agent/category patterns - order doesn't matter
            canonical = "|".join(sorted(signature))
        else:
            # Sequence patterns - order matters
            canonical = "|".join(signature)

        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def compute_pattern_similarity(
        self,
        pattern_a: ExecutionPattern,
        pattern_b: ExecutionPattern,
    ) -> float:
        """
        Compute similarity between two patterns using embeddings.

        Args:
            pattern_a: First pattern
            pattern_b: Second pattern

        Returns:
            Similarity score (0-1)
        """
        # Get or compute embeddings
        emb_a = self._get_pattern_embedding(pattern_a)
        emb_b = self._get_pattern_embedding(pattern_b)

        # Cosine similarity
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def _get_pattern_embedding(self, pattern: ExecutionPattern) -> np.ndarray:
        """Get or compute embedding for pattern."""
        if pattern._embedding is not None:
            return np.array(pattern._embedding)

        if pattern.pattern_id in self._embedding_cache:
            return self._embedding_cache[pattern.pattern_id]

        # Compute simple embedding from signature
        # In production, use sentence transformer or similar
        embedding = self._compute_simple_embedding(pattern.signature)

        self._embedding_cache[pattern.pattern_id] = embedding
        pattern._embedding = embedding.tolist()

        return embedding

    def _compute_simple_embedding(self, signature: Sequence[str]) -> np.ndarray:
        """
        Compute simple embedding from signature.

        Uses hash-based projection for deterministic embeddings.
        Replace with proper embedding model for better similarity.
        """
        embedding = np.zeros(self.embedding_dim)

        for i, element in enumerate(signature):
            # Hash each element to get deterministic values
            h = hashlib.sha256(element.encode()).digest()

            # Use hash bytes to set embedding values
            for j in range(min(8, self.embedding_dim)):
                idx = (i * 8 + j) % self.embedding_dim
                embedding[idx] += h[j % len(h)] / 255.0 - 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def find_similar_patterns(
        self,
        pattern: ExecutionPattern,
        top_k: int = 5,
    ) -> List[Tuple[ExecutionPattern, float]]:
        """
        Find patterns similar to the given pattern.

        Args:
            pattern: Pattern to find similars for
            top_k: Number of similar patterns to return

        Returns:
            List of (pattern, similarity) tuples
        """
        similarities = []

        for other in self._patterns.values():
            if other.pattern_id == pattern.pattern_id:
                continue

            sim = self.compute_pattern_similarity(pattern, other)
            if sim >= self.similarity_threshold:
                similarities.append((other, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Update related patterns
        pattern.related_patterns = [p.pattern_id for p, _ in similarities[:top_k]]

        return similarities[:top_k]

    def get_elevation_candidates(self) -> List[ExecutionPattern]:
        """
        Get patterns that are candidates for SAPE elevation.

        Returns:
            List of patterns with >3 repetitions and high success rate
        """
        candidates = [p for p in self._patterns.values() if p.should_elevate]
        candidates.sort(key=lambda p: p.occurrence_count, reverse=True)
        return candidates

    def get_pattern(self, pattern_id: str) -> Optional[ExecutionPattern]:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_all_patterns(self) -> List[ExecutionPattern]:
        """Get all tracked patterns."""
        return list(self._patterns.values())

    def clear_patterns(self) -> None:
        """Clear all tracked patterns."""
        self._patterns.clear()
        self._embedding_cache.clear()
        self._sequence_buffer.clear()

    def to_json(self) -> str:
        """Serialize patterns to JSON."""
        data = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patterns": {pid: p.to_dict() for pid, p in self._patterns.items()},
            "config": {
                "embedding_dim": self.embedding_dim,
                "similarity_threshold": self.similarity_threshold,
                "max_sequence_length": self.max_sequence_length,
            },
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PatternExtractor":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        config = data.get("config", {})

        extractor = cls(
            embedding_dim=config.get("embedding_dim", 64),
            similarity_threshold=config.get("similarity_threshold", 0.8),
            max_sequence_length=config.get("max_sequence_length", 5),
        )

        # Load patterns
        for pid, p_data in data.get("patterns", {}).items():
            extractor._patterns[pid] = ExecutionPattern.from_dict(p_data)

        return extractor


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Test Pattern Extractor."""
    import argparse

    parser = argparse.ArgumentParser(description="Pattern Extractor")
    parser.add_argument("--simulate", type=int, default=0, help="Simulate N executions")
    parser.add_argument(
        "--elevation", action="store_true", help="Show elevation candidates"
    )
    args = parser.parse_args()

    extractor = PatternExtractor()

    if args.simulate > 0:
        print(f"\nSimulating {args.simulate} executions...\n")

        # Generate sample history
        agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]
        categories = ["reasoning", "creative", "analysis"]

        history = []
        for i in range(args.simulate):
            agent = np.random.choice(agents)
            category = np.random.choice(categories)

            # Bias success toward matching agent-category
            match_bonus = (
                0.3
                if (
                    (agent == "MasterReasoner" and category == "reasoning")
                    or (agent == "CreativeSynthesizer" and category == "creative")
                    or (agent == "DataAnalyzer" and category == "analysis")
                )
                else 0.0
            )

            success = np.random.random() < (0.5 + match_bonus)

            record = ExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category=category,
                agent_name=agent,
                success=success,
                quality_score=(
                    np.random.uniform(0.7, 1.0)
                    if success
                    else np.random.uniform(0.3, 0.7)
                ),
                latency_ms=np.random.uniform(500, 2000),
            )
            history.append(record)

        # Extract patterns
        patterns = extractor.extract_all_patterns(history)

        print(f"Patterns extracted: {len(patterns)}")
        print("\nTop Success Patterns:")
        for p in [p for p in patterns if p.pattern_type == PatternType.SUCCESS][:5]:
            print(
                f"  {p.pattern_id[:8]}: {p.signature} "
                f"(count={p.occurrence_count}, rate={p.success_rate:.2%})"
            )

        print("\nTop Failure Patterns:")
        for p in [p for p in patterns if p.pattern_type == PatternType.FAILURE][:5]:
            print(
                f"  {p.pattern_id[:8]}: {p.signature} "
                f"(count={p.occurrence_count}, rate={p.success_rate:.2%})"
            )

    if args.elevation:
        candidates = extractor.get_elevation_candidates()
        print(f"\nElevation Candidates ({len(candidates)}):")
        for p in candidates:
            print(f"  {p.pattern_id[:8]}: {p.signature}")
            print(f"    count={p.occurrence_count}, success_rate={p.success_rate:.2%}")


if __name__ == "__main__":
    main()
