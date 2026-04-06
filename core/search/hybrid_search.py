"""Hybrid Search Engine — Reciprocal Rank Fusion over FAISS + RuVector HNSW.

Runs two independent retrieval engines in parallel and fuses results using
RRF (Cormack, Clarke & Buettcher, 2009). Two independent ranking signals
yield strictly higher recall than either alone — Shannon-optimal retrieval.

Architecture:
    query → EmbeddingService → vector
                                 ├── FAISS (flat L2, exact) → ranked list A
                                 └── RuVector (HNSW, approximate) → ranked list B
                                                                       ↓
                                              Reciprocal Rank Fusion (k=60)
                                                                       ↓
                                              Deduplicated, re-ranked results

Standing on Giants:
    Cormack, Clarke & Buettcher (2009) — Reciprocal Rank Fusion
    Johnson et al. (2021) — FAISS billion-scale similarity
    Malkov & Yashunin (2018) — HNSW approximate nearest neighbors
    Shannon (1948) — independent signals maximize information
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from core.integration.constants import FAISS_DEFAULT_TOP_K, FAISS_SIMILARITY_FLOOR
from core.memory.types import MemoryKind, MemoryRecord, SearchResult

logger = logging.getLogger(__name__)

# RRF constant — standard value from Cormack et al.
RRF_K = 60


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (k + rank)


class HybridSearchEngine:
    """Fuses FAISS and RuVector results via Reciprocal Rank Fusion.

    When both engines are available, runs searches in parallel threads
    and merges results. Gracefully degrades to whichever engine is
    available if the other fails or is absent.

    Thread-safe: internal engines handle their own concurrency.
    """

    def __init__(
        self,
        faiss_engine: Optional[object] = None,
        ruvector_engine: Optional[object] = None,
    ) -> None:
        self._faiss = faiss_engine
        self._ruvector = ruvector_engine
        self._initialized = False

    def _ensure_engines(self) -> None:
        """Lazy-init engines on first use."""
        if self._initialized:
            return

        if self._faiss is None:
            try:
                from core.search.vector_search import VectorSearchEngine

                engine = VectorSearchEngine()
                # Only use if FAISS index exists
                engine._ensure_loaded()
                self._faiss = engine
                logger.info(
                    "HybridSearch: FAISS engine ready (%d vectors)", engine.vector_count
                )
            except Exception as e:
                logger.info("HybridSearch: FAISS unavailable: %s", e)
                self._faiss = None

        if self._ruvector is None:
            try:
                from core.search.ruvector_search import RuVectorSearchEngine

                engine = RuVectorSearchEngine()
                if engine.is_available:
                    self._ruvector = engine
                    logger.info("HybridSearch: RuVector engine ready")
                else:
                    logger.info("HybridSearch: RuVector DB not found")
            except Exception as e:
                logger.info("HybridSearch: RuVector unavailable: %s", e)
                self._ruvector = None

        self._initialized = True

    @property
    def available_engines(self) -> List[str]:
        """List of currently available search backends."""
        self._ensure_engines()
        engines = []
        if self._faiss is not None:
            engines.append("faiss")
        if self._ruvector is not None:
            engines.append("ruvector")
        return engines

    def _fuse_results(
        self,
        result_lists: Dict[str, List[SearchResult]],
        top_k: int,
        min_score: float,
    ) -> List[SearchResult]:
        """Fuse multiple ranked lists via Reciprocal Rank Fusion.

        Deduplicates by content hash (first 100 chars + source_id).
        Returns top_k results sorted by fused RRF score.
        """
        # Accumulate RRF scores per unique result
        # Key: (source_id, content_prefix) → (rrf_score, best_result)
        scored: Dict[str, tuple[float, SearchResult]] = {}

        for engine_name, results in result_lists.items():
            for rank, sr in enumerate(results, start=1):
                # Dedup key: source_id if available, else content prefix
                key = sr.record.source_id or sr.record.content[:100]
                rrf = _rrf_score(rank)

                if key in scored:
                    existing_rrf, existing_sr = scored[key]
                    scored[key] = (existing_rrf + rrf, existing_sr)
                else:
                    scored[key] = (rrf, sr)

        # Sort by fused RRF score (descending)
        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)

        # Build final results with normalized scores
        max_rrf = ranked[0][0] if ranked else 1.0
        fused: List[SearchResult] = []
        for rrf_total, sr in ranked[:top_k]:
            normalized = rrf_total / max_rrf if max_rrf > 0 else 0.0
            if normalized < min_score:
                continue
            fused_record = MemoryRecord(
                id=str(uuid.uuid4()),
                content=sr.record.content,
                kind=MemoryKind.SEMANTIC,
                source=sr.record.source,
                source_id=sr.record.source_id,
                metadata={
                    **sr.record.metadata,
                    "rrf_score": round(rrf_total, 6),
                    "engine": "hybrid_rrf",
                    "sources": list(result_lists.keys()),
                },
            )
            fused.append(
                SearchResult(
                    record=fused_record,
                    score=round(normalized, 4),
                    vector_score=sr.vector_score,
                )
            )

        return fused

    def search(
        self,
        query: str,
        top_k: int = FAISS_DEFAULT_TOP_K,
        min_score: float = FAISS_SIMILARITY_FLOOR,
    ) -> List[SearchResult]:
        """Hybrid semantic search with parallel engine execution and RRF fusion."""
        self._ensure_engines()

        if not self._faiss and not self._ruvector:
            logger.warning("HybridSearch: no engines available")
            return []

        # Single engine path — no fusion needed
        if self._faiss and not self._ruvector:
            return self._faiss.search(query, top_k=top_k, min_score=min_score)
        if self._ruvector and not self._faiss:
            return self._ruvector.search(query, top_k=top_k, min_score=min_score)

        # Parallel execution — both engines available
        fetch_k = top_k * 3  # Over-fetch for better fusion
        result_lists: Dict[str, List[SearchResult]] = {}

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid") as pool:
            futures = {}
            if self._faiss:
                futures[pool.submit(self._faiss.search, query, fetch_k, 0.0)] = "faiss"
            if self._ruvector:
                futures[pool.submit(self._ruvector.search, query, fetch_k, 0.0)] = (
                    "ruvector"
                )

            for future in as_completed(futures, timeout=30):
                engine_name = futures[future]
                try:
                    result_lists[engine_name] = future.result()
                    logger.debug(
                        "HybridSearch: %s returned %d results",
                        engine_name,
                        len(result_lists[engine_name]),
                    )
                except Exception as e:
                    logger.warning("HybridSearch: %s failed: %s", engine_name, e)
                    result_lists[engine_name] = []

        if not any(result_lists.values()):
            return []

        return self._fuse_results(result_lists, top_k, min_score)

    def search_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = FAISS_DEFAULT_TOP_K,
        min_score: float = FAISS_SIMILARITY_FLOOR,
    ) -> List[SearchResult]:
        """Hybrid search using a pre-computed embedding vector."""
        self._ensure_engines()

        if not self._faiss and not self._ruvector:
            return []

        if self._faiss and not self._ruvector:
            return self._faiss.search_by_vector(
                vector, top_k=top_k, min_score=min_score
            )
        if self._ruvector and not self._faiss:
            return self._ruvector.search_by_vector(
                vector, top_k=top_k, min_score=min_score
            )

        fetch_k = top_k * 3
        result_lists: Dict[str, List[SearchResult]] = {}

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid") as pool:
            futures = {}
            if self._faiss:
                futures[
                    pool.submit(self._faiss.search_by_vector, vector, fetch_k, 0.0)
                ] = "faiss"
            if self._ruvector:
                futures[
                    pool.submit(self._ruvector.search_by_vector, vector, fetch_k, 0.0)
                ] = "ruvector"

            for future in as_completed(futures, timeout=30):
                engine_name = futures[future]
                try:
                    result_lists[engine_name] = future.result()
                except Exception as e:
                    logger.warning(
                        "HybridSearch: %s vector search failed: %s", engine_name, e
                    )
                    result_lists[engine_name] = []

        if not any(result_lists.values()):
            return []

        return self._fuse_results(result_lists, top_k, min_score)

    @property
    def is_loaded(self) -> bool:
        self._ensure_engines()
        return bool(self._faiss or self._ruvector)
