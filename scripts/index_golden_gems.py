#!/usr/bin/env python3
"""Index Golden Gems corpus into FAISS knowledge base.

Reads golden_gems_corpus.jsonl, chunks each gem, embeds with all-MiniLM-L6-v2,
creates golden_gems_chunks.parquet, and rebuilds the unified FAISS IVF index.

Standing on Giants: Shannon (information theory) · Besta (GoT) · Friston (Active Inference)
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "04_GOLD"
CORPUS_PATH = GOLD_DIR / "golden_gems_corpus.jsonl"
PARQUET_PATH = GOLD_DIR / "golden_gems_chunks.parquet"
INDEX_PATH = GOLD_DIR / "node0_faiss.index"
META_PATH = GOLD_DIR / "node0_faiss_meta.json"

# All parquet sources for the unified index
PARQUET_SOURCES = [
    "chunks.parquet",
    "conversations_chunks.parquet",
    "research_chunks.parquet",
    "golden_gems_chunks.parquet",
]

MAX_CHUNK_CHARS = 600  # ~150 tokens, well under 512 token limit


def chunk_gem(gem: dict) -> list[dict]:
    """Split a gem into retrievable chunks."""
    text = gem["content"]
    title = gem["title"]
    gem_id = gem["gem_id"]
    snr = gem["snr_score"]

    # Prefix each chunk with title for context
    prefix = f"[{title}] "

    chunks = []
    # Split on sentence boundaries
    sentences = text.replace(". ", ".\n").split("\n")
    current = prefix
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS and len(current) > len(prefix):
            chunk_id = hashlib.blake2b(current.encode(), digest_size=8).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": current.strip(),
                "gem_id": gem_id,
                "gem_title": title,
                "snr_score": snr,
            })
            current = prefix
        current += sentence + " "

    # Flush remaining
    if len(current) > len(prefix):
        chunk_id = hashlib.blake2b(current.encode(), digest_size=8).hexdigest()
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_text": current.strip(),
            "gem_id": gem_id,
            "gem_title": title,
            "snr_score": snr,
        })

    return chunks


def main():
    logger.info("=== Golden Gems Indexer ===")

    # 1. Read corpus
    if not CORPUS_PATH.exists():
        logger.error(f"Corpus not found: {CORPUS_PATH}")
        sys.exit(1)

    gems = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                gems.append(json.loads(line))

    logger.info(f"Loaded {len(gems)} Golden Gems")

    # 2. Chunk
    all_chunks = []
    for gem in gems:
        chunks = chunk_gem(gem)
        all_chunks.extend(chunks)
        logger.info(f"  {gem['gem_id']}: {gem['title']} → {len(chunks)} chunks")

    logger.info(f"Total chunks: {len(all_chunks)}")

    # 3. Create parquet
    df = pd.DataFrame(all_chunks)
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"Saved: {PARQUET_PATH} ({len(df)} rows, {PARQUET_PATH.stat().st_size:,} bytes)")

    # 4. Embed all chunks
    logger.info("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["chunk_text"].tolist()
    logger.info(f"Embedding {len(texts)} Golden Gem chunks...")
    gem_vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    gem_vectors = gem_vectors.astype(np.float32)
    logger.info(f"Golden Gem vectors shape: {gem_vectors.shape}")

    # 5. Load all existing parquet sources and build unified index
    logger.info("Loading all corpus parquets for unified index...")
    all_vectors = []
    total_rows = 0

    for pq_name in PARQUET_SOURCES:
        pq_path = GOLD_DIR / pq_name
        if not pq_path.exists():
            logger.warning(f"  SKIP: {pq_name} not found")
            continue

        pq_df = pd.read_parquet(pq_path, columns=["chunk_id", "chunk_text"])
        logger.info(f"  {pq_name}: {len(pq_df)} chunks")
        total_rows += len(pq_df)

        if pq_name == "golden_gems_chunks.parquet":
            # Already embedded above
            all_vectors.append(gem_vectors)
        else:
            # Embed existing chunks
            logger.info(f"  Embedding {pq_name}...")
            vecs = model.encode(
                pq_df["chunk_text"].tolist(),
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=256,
            )
            all_vectors.append(vecs.astype(np.float32))

    # Concatenate all vectors
    unified_vectors = np.vstack(all_vectors)
    logger.info(f"Unified vectors: {unified_vectors.shape} ({total_rows} total chunks)")

    # 6. Build IVF index
    dim = unified_vectors.shape[1]  # 384
    n_vectors = unified_vectors.shape[0]
    n_centroids = min(320, int(np.sqrt(n_vectors)))

    logger.info(f"Building IVF index: {n_vectors} vectors, {dim} dims, {n_centroids} centroids...")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_centroids, faiss.METRIC_INNER_PRODUCT)

    logger.info("Training index...")
    index.train(unified_vectors)

    logger.info("Adding vectors...")
    index.add(unified_vectors)
    index.nprobe = 18

    # Save
    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS index saved: {INDEX_PATH} ({INDEX_PATH.stat().st_size:,} bytes)")

    # 7. Update metadata
    meta = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_vectors": int(index.ntotal),
        "dim": dim,
        "n_centroids": n_centroids,
        "n_probe": 18,
        "metric": "cosine (L2-normalized + inner_product)",
        "sources": [
            f"{pq_name} ({len(pd.read_parquet(GOLD_DIR / pq_name))} vectors)"
            for pq_name in PARQUET_SOURCES
            if (GOLD_DIR / pq_name).exists()
        ],
        "embedding_model": "all-MiniLM-L6-v2",
        "index_size_bytes": INDEX_PATH.stat().st_size,
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Metadata saved: {META_PATH}")
    logger.info(f"=== COMPLETE: {index.ntotal} vectors indexed ===")

    return index.ntotal


if __name__ == "__main__":
    total = main()
    print(f"\nFinal vector count: {total}")
