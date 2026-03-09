# Phase 2: HNSW Vector Index

> ADR-009 | Hybrid Memory Backend — Vector Layer
> Standing on Giants: Malkov & Yashunin (HNSW, 2018) · Johnson (hnswlib)

## 2.1 — `core/memory/hnsw_index.py`

### Requirements
- Wrap `hnswlib` with a clean Python interface
- Support: add, search, remove, save, load, resize
- Graceful fallback to numpy cosine scan if hnswlib unavailable
- Thread-safe for concurrent reads (hnswlib supports this natively)
- Cosine similarity, 768 dimensions, M=16, ef_construction=200

### Pseudocode

```
IMPORT logging, threading FROM stdlib
IMPORT numpy as np
IMPORT Path FROM pathlib
IMPORT HNSWConfig FROM core.memory.config
IMPORT Optional, List, Tuple FROM typing

logger = GET_LOGGER(__name__)

# ── Fallback Detection ──

TRY:
    IMPORT hnswlib
    HNSW_AVAILABLE = True
EXCEPT ImportError:
    HNSW_AVAILABLE = False
    logger.warning("hnswlib not installed — falling back to numpy cosine scan")

# ── Interface ──

CLASS HNSWVectorIndex:
    """
    HNSW-backed approximate nearest neighbor index.

    Wraps hnswlib for O(log N) vector search.
    Falls back to O(N) numpy cosine scan if hnswlib unavailable.
    """

    FUNCTION __init__(config: HNSWConfig = HNSWConfig()):
        self._config = config
        self._lock = threading.Lock()
        self._id_map: Dict[str, int] = {}     # external_id -> internal_label
        self._reverse_map: Dict[int, str] = {} # internal_label -> external_id
        self._next_label: int = 0
        self._index = None     # hnswlib.Index or None
        self._fallback: Dict[str, np.ndarray] = {}  # fallback linear store

        IF HNSW_AVAILABLE:
            self._index = hnswlib.Index(space=config.space, dim=config.dim)
            self._index.init_index(
                max_elements=config.max_elements,
                ef_construction=config.ef_construction,
                M=config.M,
            )
            self._index.set_ef(config.ef_search)

    # ── Core Operations ──

    FUNCTION add(id: str, embedding: np.ndarray) -> None:
        """Add or update a vector in the index."""
        VALIDATE embedding.shape == (self._config.dim,)

        WITH self._lock:
            IF id IN self._id_map:
                # Update: remove old, add new
                self._remove_internal(id)

            IF self._index IS NOT None:
                label = self._next_label
                self._next_label += 1
                self._index.add_items(
                    data=embedding.reshape(1, -1).astype(np.float32),
                    ids=np.array([label]),
                )
                self._id_map[id] = label
                self._reverse_map[label] = id
            ELSE:
                # Fallback: store in dict
                self._fallback[id] = embedding.astype(np.float32)

    FUNCTION search(query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find top_k nearest neighbors. Returns [(id, score), ...]."""
        VALIDATE query.shape == (self._config.dim,)

        IF self._index IS NOT None AND len(self._id_map) > 0:
            k = min(top_k, len(self._id_map))
            labels, distances = self._index.knn_query(
                data=query.reshape(1, -1).astype(np.float32),
                k=k,
            )
            results = []
            FOR label, distance IN zip(labels[0], distances[0]):
                IF label IN self._reverse_map:
                    # hnswlib cosine returns 1 - cosine_sim, so score = 1 - distance
                    score = 1.0 - distance
                    results.append((self._reverse_map[label], score))
            RETURN sorted(results, key=lambda x: x[1], reverse=True)
        ELSE:
            # Fallback: numpy cosine scan
            RETURN self._linear_search(query, top_k)

    FUNCTION remove(id: str) -> bool:
        """Remove a vector from the index. Returns True if found."""
        WITH self._lock:
            IF id IN self._id_map:
                self._remove_internal(id)
                RETURN True
            IF id IN self._fallback:
                DEL self._fallback[id]
                RETURN True
            RETURN False

    FUNCTION __len__() -> int:
        IF self._index IS NOT None:
            RETURN len(self._id_map)
        RETURN len(self._fallback)

    FUNCTION __contains__(id: str) -> bool:
        RETURN id IN self._id_map OR id IN self._fallback

    # ── Persistence ──

    FUNCTION save(path: Optional[Path] = None) -> None:
        """Save index to disk."""
        path = path OR self._config.index_path
        path.parent.mkdir(parents=True, exist_ok=True)

        IF self._index IS NOT None:
            self._index.save_index(str(path))
            # Save ID maps alongside
            map_path = path.with_suffix(".idmap.json")
            WRITE_JSON(map_path, {
                "id_map": self._id_map,
                "next_label": self._next_label,
            })
        ELSE:
            # Fallback: save as numpy dict
            np.savez(str(path.with_suffix(".npz")), **self._fallback)

        logger.info(f"HNSW index saved: {len(self)} vectors -> {path}")

    FUNCTION load(path: Optional[Path] = None) -> None:
        """Load index from disk."""
        path = path OR self._config.index_path

        IF self._index IS NOT None AND path.exists():
            self._index.load_index(str(path), max_elements=self._config.max_elements)
            map_path = path.with_suffix(".idmap.json")
            IF map_path.exists():
                data = READ_JSON(map_path)
                self._id_map = {k: int(v) for k, v in data["id_map"].items()}
                self._reverse_map = {v: k for k, v in self._id_map.items()}
                self._next_label = data["next_label"]
            logger.info(f"HNSW index loaded: {len(self)} vectors from {path}")

    # ── Internal ──

    FUNCTION _remove_internal(id: str) -> None:
        """Remove from internal maps. hnswlib doesn't support true delete,
           so we remove from maps and mark label as stale."""
        IF id IN self._id_map:
            label = self._id_map.pop(id)
            self._reverse_map.pop(label, None)
            # Note: hnswlib.mark_deleted(label) available in v0.8+
            TRY:
                self._index.mark_deleted(label)
            EXCEPT:
                PASS  # Older hnswlib — stale labels filtered in search

    FUNCTION _linear_search(query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """O(N) fallback using numpy cosine similarity."""
        IF NOT self._fallback:
            RETURN []

        ids = list(self._fallback.keys())
        matrix = np.stack(list(self._fallback.values()))
        # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = matrix_norm @ query_norm

        top_indices = np.argsort(scores)[-top_k:][::-1]
        RETURN [(ids[i], float(scores[i])) FOR i IN top_indices]
```

### TDD Anchors

```
TEST test_add_and_search():
    idx = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100))
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    ASSERT results[0][0] == "a"   # Closest match
    ASSERT results[0][1] > 0.9    # High cosine similarity

TEST test_remove():
    idx = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100))
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    ASSERT len(idx) == 1
    ASSERT idx.remove("a") IS True
    ASSERT "a" NOT IN idx

TEST test_save_and_load(tmp_path):
    idx = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100, index_path=tmp_path/"test.index"))
    idx.add("x", np.array([0.5, 0.5, 0, 0], dtype=np.float32))
    idx.save()
    # Create new index and load
    idx2 = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100, index_path=tmp_path/"test.index"))
    idx2.load()
    ASSERT len(idx2) == 1
    ASSERT "x" IN idx2

TEST test_search_empty_index():
    idx = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100))
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32))
    ASSERT results == []

TEST test_update_existing():
    idx = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100))
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    idx.add("a", np.array([0, 1, 0, 0], dtype=np.float32))  # Update
    ASSERT len(idx) == 1
    results = idx.search(np.array([0, 1, 0, 0], dtype=np.float32), top_k=1)
    ASSERT results[0][0] == "a"

TEST test_fallback_without_hnswlib(monkeypatch):
    monkeypatch.setattr("core.memory.hnsw_index.HNSW_AVAILABLE", False)
    idx = HNSWVectorIndex(HNSWConfig(dim=4, max_elements=100))
    idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32))
    ASSERT len(results) >= 1
    ASSERT results[0][0] == "a"

@pytest.mark.slow
TEST test_performance_1k():
    """Benchmark: 1K vectors should search in < 1ms with HNSW."""
    idx = HNSWVectorIndex(HNSWConfig(dim=768, max_elements=2000))
    FOR i IN range(1000):
        idx.add(f"v{i}", np.random.randn(768).astype(np.float32))
    query = np.random.randn(768).astype(np.float32)
    start = time.perf_counter()
    results = idx.search(query, top_k=10)
    elapsed = time.perf_counter() - start
    ASSERT elapsed < 0.001  # < 1ms target
    ASSERT len(results) == 10
```

### Edge Cases

1. **Dimension mismatch** — `add()` and `search()` validate shape == (dim,)
2. **Duplicate ID** — `add()` removes old entry first, then inserts new
3. **Remove nonexistent** — returns False, no error
4. **hnswlib not installed** — seamless fallback to numpy O(N) scan
5. **Index full** — hnswlib raises; catch and resize or log error
6. **Concurrent access** — `_lock` serializes writes; hnswlib allows concurrent reads
7. **Zero-norm query** — epsilon guard in `_linear_search` prevents division by zero
