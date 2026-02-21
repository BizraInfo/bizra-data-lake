"""BIZRA Vector Search Engine — Phase 46: Cognitive Resonance.

Bridges the 152 MB FAISS IVF index (102 714 vectors, 384-dim) into the
sovereign runtime.  Lazy-loads on first query, hydrates from parquet,
returns SearchResult/MemoryRecord.  Thread-safe after load.

Standing on Giants: Johnson et al. (FAISS, 2021) · Shannon (1948)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.integration.constants import (
    FAISS_DEFAULT_TOP_K,
    FAISS_EMBEDDING_DIM,
    FAISS_GOLD_DIR,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    FAISS_SIMILARITY_FLOOR,
)
from core.memory.types import MemoryKind, MemoryRecord, SearchResult

logger = logging.getLogger(__name__)

# Feature flag — callers check this; the engine itself works regardless.
PHASE46_ENABLED: bool = os.getenv("BIZRA_PHASE46_SEARCH_ENABLED", "0").lower() in {
    "1",
    "true",
    "yes",
}


def _resolve_root() -> Path:
    """Resolve the BIZRA-DATA-LAKE project root."""
    if env_root := os.getenv("BIZRA_DATA_LAKE_ROOT"):
        return Path(env_root)
    return Path(__file__).resolve().parent.parent.parent


class VectorSearchEngine:
    """FAISS-backed semantic search with parquet content hydration.

    Thread-safe for concurrent reads after first query triggers lazy init.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        embedding_service: Optional[Any] = None,
    ) -> None:
        self._root = root or _resolve_root()
        self._embedding_service = embedding_service
        self._index: Optional[Any] = None  # faiss.Index (lazy)
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._chunk_ids: List[str] = []
        self._meta: Optional[Dict[str, Any]] = None
        self._init_lock = threading.Lock()
        self._loaded = False

    # ---- Lazy initialisation ----

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._init_lock:
            if self._loaded:
                return
            self._load_index()
            self._load_texts()
            self._loaded = True
            logger.info(
                "VectorSearchEngine ready — %d vectors, dim=%d",
                self._index.ntotal,
                FAISS_EMBEDDING_DIM,  # type: ignore[union-attr]
            )

    def _load_index(self) -> None:
        import faiss  # type: ignore[import-untyped]

        index_path = self._root / FAISS_INDEX_PATH
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. Run vector_engine.py to build it."
            )
        self._index = faiss.read_index(str(index_path))
        logger.info(
            "Loaded FAISS index: %s (%d vectors)", index_path, self._index.ntotal
        )

        meta_path = self._root / FAISS_META_PATH
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if self._index.d != FAISS_EMBEDDING_DIM:
            raise ValueError(
                f"Index dimension mismatch: d={self._index.d}, expected {FAISS_EMBEDDING_DIM}"
            )

    def _load_texts(self) -> None:
        """Load chunk texts from all parquet sources in index build order."""
        import pandas as pd

        gold_dir = self._root / FAISS_GOLD_DIR

        # Derive ordered filenames from meta.json source strings.
        # Each entry: "chunks.parquet (84795 vectors)"
        ordered_files: List[str] = []
        if self._meta and "sources" in self._meta:
            for entry in self._meta["sources"]:
                ordered_files.append(entry.split("(")[0].strip())
        else:
            ordered_files = sorted(f.name for f in gold_dir.glob("*chunks*.parquet"))
            if (
                "chunks.parquet" not in ordered_files
                and (gold_dir / "chunks.parquet").exists()
            ):
                ordered_files.insert(0, "chunks.parquet")

        texts: List[str] = []
        sources: List[str] = []
        chunk_ids: List[str] = []

        for fname in ordered_files:
            fpath = gold_dir / fname
            if not fpath.exists():
                logger.warning("Parquet source missing, skipping: %s", fpath)
                continue
            # Read only chunk_text and chunk_id (if present) — skip embeddings
            try:
                df = pd.read_parquet(fpath, columns=["chunk_text", "chunk_id"])
            except Exception:
                df = pd.read_parquet(fpath, columns=["chunk_text"])
            t = df["chunk_text"].fillna("").astype(str).tolist()
            ids = (
                df["chunk_id"].astype(str).tolist()
                if "chunk_id" in df.columns
                else [f"{fname}:{i}" for i in range(len(df))]
            )
            texts.extend(t)
            sources.extend([fname] * len(t))
            chunk_ids.extend(ids)
            logger.info("Loaded %d texts from %s", len(t), fname)

        if self._index is not None and len(texts) != self._index.ntotal:
            logger.warning(
                "Text count (%d) != index vectors (%d) — content may be misaligned.",
                len(texts),
                self._index.ntotal,
            )
        self._texts = texts
        self._sources = sources
        self._chunk_ids = chunk_ids

    # ---- Embedding ----

    def _get_embedding_service(self) -> Any:
        if self._embedding_service is None:
            from core.embedding.service import EmbeddingService

            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _normalise_vector(self, arr: np.ndarray) -> np.ndarray:
        """L2-normalise for cosine via inner-product."""
        norm = np.linalg.norm(arr)
        return (arr / norm if norm > 0 else arr).reshape(1, -1)

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self._get_embedding_service().embed(text)
        arr = np.array(vec, dtype=np.float32)
        if arr.shape[0] != FAISS_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim {arr.shape[0]} != {FAISS_EMBEDDING_DIM}. "
                "Check that EmbeddingService uses all-MiniLM-L6-v2."
            )
        return self._normalise_vector(arr)

    # ---- Core search ----

    def _raw_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        min_score: float,
    ) -> List[SearchResult]:
        """Run FAISS search and hydrate results."""
        import faiss as _faiss  # type: ignore[import-untyped]

        fetch_k = min(top_k * 3, self._index.ntotal)  # type: ignore[union-attr]
        scores, indices = self._index.search(query_vec, fetch_k)  # type: ignore[union-attr]
        # L2 on normalised vecs: cos = 1 - L2_sq/2; IP: score IS cosine
        is_l2 = self._index.metric_type == _faiss.METRIC_L2  # type: ignore[union-attr]
        results: List[SearchResult] = []
        for raw, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = (1.0 - float(raw) / 2.0) if is_l2 else float(raw)
            if sim < min_score:
                continue
            text = self._texts[idx] if idx < len(self._texts) else ""
            source = self._sources[idx] if idx < len(self._sources) else "unknown"
            cid = self._chunk_ids[idx] if idx < len(self._chunk_ids) else str(idx)
            record = MemoryRecord(
                id=str(uuid.uuid4()),
                content=text,
                kind=MemoryKind.SEMANTIC,
                source=source,
                source_id=cid,
                metadata={"faiss_idx": int(idx), "cosine_similarity": sim},
            )
            results.append(SearchResult(record=record, score=sim, vector_score=sim))
            if len(results) >= top_k:
                break
        return results

    def search(
        self,
        query: str,
        top_k: int = FAISS_DEFAULT_TOP_K,
        min_score: float = FAISS_SIMILARITY_FLOOR,
    ) -> List[SearchResult]:
        """Semantic search: encode *query* and return top-k results above *min_score*."""
        self._ensure_loaded()
        return self._raw_search(self._encode_query(query), top_k, min_score)

    def search_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = FAISS_DEFAULT_TOP_K,
        min_score: float = FAISS_SIMILARITY_FLOOR,
    ) -> List[SearchResult]:
        """Search using a pre-computed embedding vector."""
        self._ensure_loaded()
        arr = np.array(vector, dtype=np.float32)
        if arr.shape[0] != FAISS_EMBEDDING_DIM:
            raise ValueError(f"Vector dim {arr.shape[0]} != {FAISS_EMBEDDING_DIM}")
        return self._raw_search(self._normalise_vector(arr), top_k, min_score)

    # ---- Diagnostics ----

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vector_count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._meta
