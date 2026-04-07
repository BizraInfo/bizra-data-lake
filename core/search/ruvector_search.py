"""RuVector HNSW Search Engine — self-learning vector search via native NAPI.

Bridges Python queries to RuVector's HNSW index (84K+ vectors, 384-dim cosine)
via a lightweight Node.js subprocess. Same interface as VectorSearchEngine.

Standing on Giants: Malkov & Yashunin (2018) — HNSW approximate nearest neighbors.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from core.integration.constants import FAISS_DEFAULT_TOP_K, FAISS_SIMILARITY_FLOOR
from core.memory.types import MemoryKind, MemoryRecord, SearchResult

logger = logging.getLogger(__name__)

_NODE_PATH = "/usr/lib/node_modules"


def _resolve_root() -> Path:
    """Resolve BIZRA-DATA-LAKE project root."""
    import os

    if env_root := os.getenv("BIZRA_DATA_LAKE_ROOT"):
        return Path(env_root)
    return Path(__file__).resolve().parent.parent.parent


class RuVectorSearchEngine:
    """HNSW-backed semantic search via RuVector native NAPI binding.

    Thread-safe: each search spawns an isolated Node.js subprocess.
    No persistent server required — subprocess overhead is ~50ms,
    search itself is <35ms for 84K vectors.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        embedding_service: Optional[Any] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self._root = root or _resolve_root()
        self._embedding_service = embedding_service
        self._db_path = db_path or str(self._root / "04_GOLD" / "ruvector_bizra")
        self._query_script = str(self._root / "scripts" / "ruvector_query.mjs")
        self._available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Check if RuVector DB and Node.js runtime are available."""
        if self._available is not None:
            return self._available
        self._available = (
            Path(self._db_path).exists() and Path(self._query_script).exists()
        )
        if not self._available:
            logger.info(
                "RuVector not available: db=%s, script=%s",
                self._db_path,
                self._query_script,
            )
        return self._available

    def _get_embedding_service(self) -> Any:
        if self._embedding_service is None:
            from core.embedding.service import EmbeddingService

            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self._get_embedding_service().embed(text)
        return np.array(vec, dtype=np.float32)

    def _call_ruvector(self, vector: np.ndarray, k: int) -> list:
        """Call RuVector via Node.js subprocess."""
        import os

        payload = json.dumps({"vector": vector.tolist(), "k": k})
        env = os.environ.copy()
        env["NODE_PATH"] = _NODE_PATH
        env["RUVECTOR_DB"] = self._db_path
        try:
            result = subprocess.run(
                ["node", self._query_script],
                input=payload.encode(),
                capture_output=True,
                timeout=30,
                cwd=str(self._root),
                env=env,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace").strip()
                logger.warning("RuVector query failed: %s", stderr)
                return []
            return json.loads(result.stdout.decode())
        except subprocess.TimeoutExpired:
            logger.warning("RuVector query timed out (30s)")
            return []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("RuVector query error: %s", e)
            return []

    def search(
        self,
        query: str,
        top_k: int = FAISS_DEFAULT_TOP_K,
        min_score: float = FAISS_SIMILARITY_FLOOR,
    ) -> List[SearchResult]:
        """Semantic search: encode query and return top-k results."""
        if not self.is_available:
            return []

        vector = self._encode_query(query)
        raw = self._call_ruvector(vector, top_k * 2)

        results: List[SearchResult] = []
        for item in raw:
            # RuVector returns cosine distance (0=identical); convert to similarity
            distance = float(item.get("score", 1.0))
            similarity = 1.0 - distance
            if similarity < min_score:
                continue
            record = MemoryRecord(
                id=str(uuid.uuid4()),
                content=item.get("text", ""),
                kind=MemoryKind.SEMANTIC,
                source="ruvector_hnsw",
                source_id=item.get("id", ""),
                metadata={
                    "ruvector_distance": distance,
                    "cosine_similarity": similarity,
                    "engine": "ruvector_hnsw",
                },
            )
            results.append(
                SearchResult(record=record, score=similarity, vector_score=similarity)
            )
            if len(results) >= top_k:
                break

        return results

    def search_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = FAISS_DEFAULT_TOP_K,
        min_score: float = FAISS_SIMILARITY_FLOOR,
    ) -> List[SearchResult]:
        """Search using a pre-computed embedding vector."""
        if not self.is_available:
            return []

        arr = np.array(vector, dtype=np.float32)
        raw = self._call_ruvector(arr, top_k * 2)

        results: List[SearchResult] = []
        for item in raw:
            distance = float(item.get("score", 1.0))
            similarity = 1.0 - distance
            if similarity < min_score:
                continue
            record = MemoryRecord(
                id=str(uuid.uuid4()),
                content=item.get("text", ""),
                kind=MemoryKind.SEMANTIC,
                source="ruvector_hnsw",
                source_id=item.get("id", ""),
                metadata={
                    "ruvector_distance": distance,
                    "cosine_similarity": similarity,
                    "engine": "ruvector_hnsw",
                },
            )
            results.append(
                SearchResult(record=record, score=similarity, vector_score=similarity)
            )
            if len(results) >= top_k:
                break

        return results
