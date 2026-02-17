"""
Memory Auto-Coder: Distills raw memories into reusable cognitive patterns.

Does NOT generate Python source code. Synthesizes reusable cognitive patterns
from accumulated memory -- distilling experience into a codebook that accelerates
future reasoning.

The synthesize cycle follows a PDCA loop:
  Plan   -- retrieve high-frequency recent memories
  Do     -- cluster by embedding similarity
  Check  -- validate SNR and novelty
  Act    -- promote novel patterns to the codebook

Standing on Giants: Deming (PDCA) + Shannon (compression) + Kauffman (adjacent possible)
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from core.integration.constants import UNIFIED_SNR_THRESHOLD
from core.memory_coder.pattern_codebook import (
    PatternCodebook,
    SynthesizedPattern,
    cosine_similarity,
)

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
# AgentDB Protocol (structural typing -- no concrete import required)
# ---------------------------------------------------------------------------
class AgentDBProtocol(Protocol):
    """Minimal interface expected from an agent memory database."""

    def search(
        self, embedding: List[float], top_k: int = 10, **kwargs: Any
    ) -> List[Any]: ...

    def store(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> str: ...


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------
@dataclass
class MemoryRecord:
    """A memory record retrieved from AgentDB.

    Attributes
    ----------
    record_id : str
        Unique identifier for the memory.
    content : str
        Textual content of the memory.
    embedding : List[float]
        Dense vector representation.
    keywords : List[str]
        Extracted keyword tags.
    snr : float
        Signal-to-noise quality score (must meet UNIFIED_SNR_THRESHOLD).
    access_count : int
        Number of times this memory has been retrieved.
    created_at : str
        ISO-8601 timestamp of creation.
    metadata : Dict[str, Any]
        Arbitrary extra fields.
    """

    record_id: str
    content: str
    embedding: List[float]
    keywords: List[str] = field(default_factory=list)
    snr: float = UNIFIED_SNR_THRESHOLD
    access_count: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MemorySynthesizer
# ---------------------------------------------------------------------------
class MemorySynthesizer:
    """Distills raw agent memories into reusable cognitive patterns.

    Parameters
    ----------
    agent_db : optional
        A vector database implementing :class:`AgentDBProtocol`.
        When ``None``, the synthesizer operates in *offline* mode using
        only the in-memory codebook.
    codebook : PatternCodebook
        The destination codebook that receives newly discovered patterns.
    """

    def __init__(
        self,
        agent_db: Optional[Any] = None,
        codebook: Optional[PatternCodebook] = None,
    ) -> None:
        self._agent_db = agent_db
        self._codebook = codebook or PatternCodebook(agent_db=agent_db)

    @property
    def codebook(self) -> PatternCodebook:
        """Expose the underlying pattern codebook."""
        return self._codebook

    # -- Public API ---------------------------------------------------------

    def synthesize_cycle(
        self,
        window_hours: int = 24,
        min_access_count: int = 5,
        min_cluster_size: int = 3,
    ) -> List[SynthesizedPattern]:
        """Run one PDCA synthesis cycle.

        1. Retrieve recent high-frequency memories.
        2. Cluster by cosine similarity.
        3. Extract a candidate pattern from each cluster.
        4. Validate SNR >= ``UNIFIED_SNR_THRESHOLD``.
        5. Check novelty against codebook (threshold 0.90).
        6. Add novel patterns to codebook.

        Returns
        -------
        List[SynthesizedPattern]
            Patterns that were newly added to the codebook in this cycle.
        """
        records = self._retrieve_recent_memories(
            window_hours=window_hours,
            min_access_count=min_access_count,
        )

        if not records:
            logger.info("No high-frequency memories found; cycle yields 0 patterns")
            return []

        clusters = self._cluster_by_embedding(
            records,
            min_cluster_size=min_cluster_size,
            threshold=UNIFIED_SNR_THRESHOLD,
        )

        novel_patterns: List[SynthesizedPattern] = []
        for cluster in clusters:
            pattern = self._extract_pattern(cluster)

            # Gate 1: SNR quality
            if pattern.snr < UNIFIED_SNR_THRESHOLD:
                logger.debug(
                    "Pattern %s rejected: SNR %.3f < %.3f",
                    pattern.pattern_id,
                    pattern.snr,
                    UNIFIED_SNR_THRESHOLD,
                )
                continue

            # Gate 2: Novelty (avoid duplicates in codebook)
            if self._codebook.contains_similar(pattern, threshold=0.90):
                logger.debug(
                    "Pattern %s rejected: too similar to existing codebook entry",
                    pattern.pattern_id,
                )
                continue

            self._codebook.add(pattern)
            novel_patterns.append(pattern)
            logger.info(
                "Pattern %s added (SNR=%.3f, sources=%d, keywords=%s)",
                pattern.pattern_id,
                pattern.snr,
                pattern.source_count,
                pattern.keywords[:3],
            )

        return novel_patterns

    # -- Clustering ---------------------------------------------------------

    def _cluster_by_embedding(
        self,
        records: List[MemoryRecord],
        min_cluster_size: int = 3,
        threshold: float = UNIFIED_SNR_THRESHOLD,
    ) -> List[List[MemoryRecord]]:
        """Greedy single-pass clustering by cosine similarity.

        For each unassigned record, find all other unassigned records with
        cosine similarity >= *threshold*. If the resulting group meets
        *min_cluster_size*, it forms a cluster.
        """
        assigned: set[int] = set()
        clusters: List[List[MemoryRecord]] = []

        for i, anchor in enumerate(records):
            if i in assigned:
                continue

            cluster_indices = [i]
            for j, candidate in enumerate(records):
                if j in assigned or j == i:
                    continue
                sim = cosine_similarity(anchor.embedding, candidate.embedding)
                if sim >= threshold:
                    cluster_indices.append(j)

            if len(cluster_indices) >= min_cluster_size:
                cluster = [records[idx] for idx in cluster_indices]
                clusters.append(cluster)
                assigned.update(cluster_indices)

        return clusters

    # -- Pattern extraction -------------------------------------------------

    def _extract_pattern(self, cluster: List[MemoryRecord]) -> SynthesizedPattern:
        """Extract a single pattern from a cluster of similar memories.

        - Embedding = centroid (element-wise mean).
        - Keywords  = top-5 most frequent across the cluster.
        - SNR       = arithmetic mean of member SNRs.
        """
        centroid = self._compute_centroid([r.embedding for r in cluster])

        all_keywords: List[str] = []
        for record in cluster:
            all_keywords.extend(record.keywords)
        top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]

        mean_snr = sum(r.snr for r in cluster) / len(cluster)

        # Deterministic ID from centroid hash
        centroid_bytes = ",".join(f"{v:.6f}" for v in centroid).encode("utf-8")
        pattern_id = hashlib.sha256(centroid_bytes).hexdigest()[:16]

        return SynthesizedPattern(
            pattern_id=pattern_id,
            embedding=centroid,
            keywords=top_keywords,
            snr=mean_snr,
            source_count=len(cluster),
            access_count=0,
            metadata={
                "source_ids": [r.record_id for r in cluster],
            },
        )

    # -- Internal helpers ---------------------------------------------------

    def _retrieve_recent_memories(
        self,
        window_hours: int,
        min_access_count: int,
    ) -> List[MemoryRecord]:
        """Retrieve high-frequency recent memories from agent_db.

        Falls back gracefully when agent_db is ``None`` or does not support
        time-windowed queries.
        """
        if self._agent_db is None:
            return []

        try:
            # Attempt a time-windowed search if the DB supports it.
            # We use a zero-vector query (semantic wildcard) with filters.
            dummy_embedding = [0.0] * 128
            raw_results = self._agent_db.search(
                embedding=dummy_embedding,
                top_k=200,
                window_hours=window_hours,
                min_access_count=min_access_count,
            )
            return self._raw_to_records(raw_results)
        except TypeError:
            # agent_db.search does not accept window_hours / min_access_count
            try:
                raw_results = self._agent_db.search(
                    embedding=[0.0] * 128,
                    top_k=200,
                )
                return self._raw_to_records(raw_results)
            except Exception:
                logger.warning("agent_db.search failed; returning empty list")
                return []
        except Exception:
            logger.warning("Memory retrieval failed; returning empty list")
            return []

    @staticmethod
    def _raw_to_records(raw_results: List[Any]) -> List[MemoryRecord]:
        """Convert raw agent_db results into MemoryRecord objects."""
        records: List[MemoryRecord] = []
        for item in raw_results:
            if isinstance(item, MemoryRecord):
                records.append(item)
                continue
            # Duck-type conversion for arbitrary result objects
            try:
                records.append(
                    MemoryRecord(
                        record_id=getattr(item, "record_id", getattr(item, "id", "")),
                        content=getattr(item, "content", ""),
                        embedding=getattr(item, "embedding", []),
                        keywords=getattr(item, "keywords", []),
                        snr=getattr(item, "snr", UNIFIED_SNR_THRESHOLD),
                        access_count=getattr(item, "access_count", 1),
                        created_at=getattr(
                            item,
                            "created_at",
                            datetime.now(timezone.utc).isoformat(),
                        ),
                        metadata=getattr(item, "metadata", {}),
                    )
                )
            except Exception:
                logger.debug("Skipping unconvertible result: %s", type(item))
        return records

    @staticmethod
    def _compute_centroid(embeddings: List[List[float]]) -> List[float]:
        """Compute the element-wise mean of a list of embedding vectors."""
        if not embeddings:
            return []

        if _HAS_NUMPY:
            matrix = np.array(embeddings, dtype=np.float64)
            return np.mean(matrix, axis=0).tolist()

        # Pure-Python fallback
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for d in range(dim):
                centroid[d] += emb[d]
        n = len(embeddings)
        return [c / n for c in centroid]
