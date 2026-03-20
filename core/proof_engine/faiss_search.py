"""
FAISS Semantic Search — Query 84,795 vectors in <5ms.

Loads embeddings from 04_GOLD/chunks.parquet, builds FAISS index,
returns top-K relevant chunks for any query.

Standing on: Johnson et al. (2019) — FAISS billion-scale similarity search.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("bizra.faiss_search")

# Lazy globals — loaded once, reused
_INDEX = None
_TEXTS: List[str] = []
_ENCODER = None
_LOADED = False


def _get_encoder():
    """Lazy-load sentence-transformers encoder."""
    global _ENCODER
    if _ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            logger.info("Encoder loaded on %s", device)
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return None
    return _ENCODER


def load_index(
    parquet_path: str = "04_GOLD/chunks.parquet",
) -> Tuple[bool, int]:
    """
    Load FAISS index from parquet embeddings.
    Returns (success, num_vectors).
    """
    global _INDEX, _TEXTS, _LOADED

    if _LOADED and _INDEX is not None:
        return True, len(_TEXTS)

    try:
        import faiss
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("faiss or pyarrow not installed")
        return False, 0

    path = Path(parquet_path)
    if not path.exists():
        logger.warning("Parquet not found: %s", path)
        return False, 0

    t0 = time.time()
    table = pq.read_table(str(path), columns=["chunk_text", "embedding"])

    texts = table.column("chunk_text").to_pylist()
    embeddings_raw = table.column("embedding").to_pylist()

    # Parse embeddings (stored as JSON strings or lists)
    embeddings = []
    for emb in embeddings_raw:
        if isinstance(emb, str):
            embeddings.append(json.loads(emb))
        elif isinstance(emb, (list, np.ndarray)):
            embeddings.append(emb)
        else:
            continue

    if not embeddings:
        logger.warning("No valid embeddings found")
        return False, 0

    matrix = np.array(embeddings, dtype=np.float32)
    dim = matrix.shape[1]

    # Build FAISS index (flat L2 — fast enough for 84K vectors)
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(matrix)
    index.add(matrix)

    _INDEX = index
    _TEXTS = texts
    _LOADED = True

    elapsed = time.time() - t0
    logger.info(
        "FAISS index: %d vectors, dim=%d, loaded in %.2fs", len(texts), dim, elapsed
    )
    return True, len(texts)


def search(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Search for top-K relevant chunks.
    Returns list of (chunk_text, distance).
    """
    if _INDEX is None:
        ok, _ = load_index()
        if not ok:
            return []

    encoder = _get_encoder()
    if encoder is None:
        return []

    # Encode query
    q_vec = encoder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype=np.float32)

    # Search
    distances, indices = _INDEX.search(q_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(_TEXTS):
            results.append((_TEXTS[idx], float(distances[0][i])))

    return results


def format_context(results: List[Tuple[str, float]], max_chars: int = 2000) -> str:
    """Format search results as context for LLM prompt."""
    if not results:
        return ""

    lines = ["[Relevant knowledge from your sovereign data:]"]
    total = 0
    for text, dist in results:
        snippet = text[:400]
        if total + len(snippet) > max_chars:
            break
        lines.append(f"- {snippet}")
        total += len(snippet)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok, count = load_index()
    print(f"Index loaded: {ok}, vectors: {count}")
    if ok:
        results = search("What is BIZRA's constitutional architecture?")
        print("\nTop 5 results:")
        for text, dist in results:
            print(f"  [{dist:.4f}] {text[:80]}...")
