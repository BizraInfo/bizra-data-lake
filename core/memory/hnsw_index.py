"""
HNSW Vector Index — Sub-linear nearest-neighbor search.

Wraps hnswlib with save/load, incremental add/remove, and
automatic fallback to numpy cosine scan if hnswlib is unavailable.

Performance targets (cosine, dim=768, ef_search=100):
  1K entries:   ~0.035ms per query  (140x vs linear)
  10K entries:  ~0.04ms             (1,250x)
  100K entries: ~0.05ms             (10,000x)
  1M entries:   ~0.4ms              (12,500x)

Standing on Giants: Malkov & Yashunin (2016) — HNSW algorithm
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import HNSWConfig

logger = logging.getLogger(__name__)

# Try to import hnswlib; fall back to numpy brute-force
try:
    import hnswlib

    _HAS_HNSWLIB = True
except ImportError:
    _HAS_HNSWLIB = False
    logger.warning("hnswlib not installed — falling back to numpy cosine scan")


class HNSWIndex:
    """HNSW vector index with disk persistence.

    Thread-safe for concurrent reads. Writes should be serialized
    by the caller (AgentDB handles this).
    """

    def __init__(self, config: Optional[HNSWConfig] = None) -> None:
        self._config = config or HNSWConfig()
        self._index: object = None  # hnswlib.Index or None
        self._id_map: dict[int, str] = {}  # internal_id -> record_id
        self._reverse_map: dict[str, int] = {}  # record_id -> internal_id
        self._next_internal_id: int = 0
        self._initialized = False
        self._use_hnswlib = _HAS_HNSWLIB

        # Fallback storage (numpy brute-force)
        self._fallback_vectors: dict[str, np.ndarray] = {}

    @property
    def count(self) -> int:
        if self._use_hnswlib and self._index is not None:
            return self._index.get_current_count()  # type: ignore[union-attr]
        return len(self._fallback_vectors)

    @property
    def capacity(self) -> int:
        if self._use_hnswlib and self._index is not None:
            return self._index.get_max_elements()  # type: ignore[union-attr]
        return self._config.max_elements

    def initialize(self) -> None:
        """Create or re-create the index in memory."""
        if self._use_hnswlib:
            idx = hnswlib.Index(space=self._config.space, dim=self._config.dimensions)  # type: ignore[name-defined]
            idx.init_index(
                max_elements=self._config.max_elements,
                M=self._config.m,
                ef_construction=self._config.ef_construction,
            )
            idx.set_ef(self._config.ef_search)
            self._index = idx
        self._initialized = True
        logger.info(
            f"HNSW index initialized: dim={self._config.dimensions}, "
            f"backend={'hnswlib' if self._use_hnswlib else 'numpy'}"
        )

    def add(self, record_id: str, vector: Sequence[float]) -> None:
        """Add a vector for a record. Replaces if record_id already exists."""
        if not self._initialized:
            self.initialize()

        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self._config.dimensions,):
            raise ValueError(
                f"Vector dim {vec.shape[0]} != index dim {self._config.dimensions}"
            )

        if self._use_hnswlib and self._index is not None:
            # Remove old entry if updating
            if record_id in self._reverse_map:
                old_id = self._reverse_map[record_id]
                try:
                    self._index.mark_deleted(old_id)  # type: ignore[union-attr]
                except RuntimeError:
                    pass  # Already deleted
                del self._id_map[old_id]

            # Auto-resize if at capacity
            if self._next_internal_id >= self._index.get_max_elements():  # type: ignore[union-attr]
                new_cap = self._index.get_max_elements() * 2  # type: ignore[union-attr]
                self._index.resize_index(new_cap)  # type: ignore[union-attr]
                logger.info(f"HNSW index resized to {new_cap}")

            internal_id = self._next_internal_id
            self._next_internal_id += 1

            self._index.add_items(  # type: ignore[union-attr]
                vec.reshape(1, -1), np.array([internal_id])
            )
            self._id_map[internal_id] = record_id
            self._reverse_map[record_id] = internal_id
        else:
            self._fallback_vectors[record_id] = vec

    def remove(self, record_id: str) -> bool:
        """Mark a vector as deleted. Returns True if found."""
        if self._use_hnswlib and self._index is not None:
            if record_id not in self._reverse_map:
                return False
            internal_id = self._reverse_map.pop(record_id)
            del self._id_map[internal_id]
            try:
                self._index.mark_deleted(internal_id)  # type: ignore[union-attr]
            except RuntimeError:
                pass
            return True
        else:
            return self._fallback_vectors.pop(record_id, None) is not None

    def search(
        self, query_vector: Sequence[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for nearest neighbors.

        Returns list of (record_id, distance) sorted by distance ascending.
        For cosine space, distance = 1 - cosine_similarity.
        """
        if self.count == 0:
            return []

        vec = np.asarray(query_vector, dtype=np.float32)
        k = min(top_k, self.count)

        if self._use_hnswlib and self._index is not None:
            # Clamp k to number of live (non-deleted) entries
            live_count = len(self._id_map)
            effective_k = min(k, live_count)
            if effective_k <= 0:
                return []
            try:
                labels, distances = self._index.knn_query(vec.reshape(1, -1), k=effective_k)  # type: ignore[union-attr]
            except RuntimeError:
                # hnswlib can error if too many deletions; fall back gracefully
                return []
            results = []
            for label, dist in zip(labels[0], distances[0]):
                rid = self._id_map.get(int(label))
                if rid is not None:
                    results.append((rid, float(dist)))
            return results
        else:
            return self._numpy_search(vec, k)

    def _numpy_search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Brute-force cosine search (fallback when hnswlib unavailable)."""
        if not self._fallback_vectors:
            return []

        ids = list(self._fallback_vectors.keys())
        matrix = np.stack([self._fallback_vectors[i] for i in ids])

        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_norm = matrix / norms

        # Cosine distance = 1 - dot product of normalized vectors
        similarities = matrix_norm @ query_norm
        distances = 1.0 - similarities

        top_indices = np.argsort(distances)[:top_k]
        return [(ids[i], float(distances[i])) for i in top_indices]

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_hnswlib and self._index is not None:
            self._index.save_index(str(path))  # type: ignore[union-attr]
            # Save ID maps alongside
            import json

            meta_path = path.with_suffix(".meta.json")
            meta = {
                "id_map": {str(k): v for k, v in self._id_map.items()},
                "next_internal_id": self._next_internal_id,
                "config": {
                    "dimensions": self._config.dimensions,
                    "space": self._config.space,
                    "m": self._config.m,
                    "ef_construction": self._config.ef_construction,
                    "ef_search": self._config.ef_search,
                    "max_elements": self._config.max_elements,
                },
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            logger.info(f"HNSW index saved: {path} ({self.count} vectors)")
        else:
            # Save numpy fallback as npz
            npz_path = path.with_suffix(".npz")
            if self._fallback_vectors:
                ids = list(self._fallback_vectors.keys())
                vecs = np.stack([self._fallback_vectors[i] for i in ids])
                np.savez(str(npz_path), ids=np.array(ids), vectors=vecs)
                logger.info(f"Numpy index saved: {npz_path} ({len(ids)} vectors)")

    def load(self, path: Path) -> bool:
        """Load index from disk. Returns True if loaded successfully."""
        if self._use_hnswlib:
            import json

            meta_path = path.with_suffix(".meta.json")
            if not path.exists() or not meta_path.exists():
                return False

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cfg = meta.get("config", {})
                self._config.max_elements = max(
                    cfg.get("max_elements", self._config.max_elements),
                    len(meta.get("id_map", {})) + 1000,
                )

                idx = hnswlib.Index(  # type: ignore[name-defined]
                    space=self._config.space, dim=self._config.dimensions
                )
                idx.load_index(str(path), max_elements=self._config.max_elements)
                idx.set_ef(self._config.ef_search)
                self._index = idx

                self._id_map = {int(k): v for k, v in meta["id_map"].items()}
                self._reverse_map = {v: int(k) for k, v in meta["id_map"].items()}
                self._next_internal_id = meta.get("next_internal_id", 0)
                self._initialized = True

                logger.info(f"HNSW index loaded: {path} ({self.count} vectors)")
                return True
            except Exception as e:
                logger.error(f"Failed to load HNSW index: {e}")
                return False
        else:
            npz_path = path.with_suffix(".npz")
            if not npz_path.exists():
                return False
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                ids = data["ids"]
                vecs = data["vectors"]
                self._fallback_vectors = {str(ids[i]): vecs[i] for i in range(len(ids))}
                self._initialized = True
                logger.info(f"Numpy index loaded: {npz_path} ({len(ids)} vectors)")
                return True
            except Exception as e:
                logger.error(f"Failed to load numpy index: {e}")
                return False

    def clear(self) -> None:
        """Clear all vectors and reinitialize."""
        self._id_map.clear()
        self._reverse_map.clear()
        self._fallback_vectors.clear()
        self._next_internal_id = 0
        self._initialized = False
        self._index = None
        self.initialize()
