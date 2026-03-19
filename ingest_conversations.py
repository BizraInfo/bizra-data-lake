#!/usr/bin/env python3
"""
Conversation JSON Ingestion Pipeline
=====================================
Reads ChatGPT conversation JSON archives from 00_INTAKE/2025-12-15-conversations-*/,
extracts user and assistant text turns, generates embeddings via all-MiniLM-L6-v2,
writes conversations_chunks.parquet to 04_GOLD/, and rebuilds the FAISS index from
both the existing chunks.parquet and the new conversations_chunks.parquet.

Usage:
    source .venv-linux/bin/activate
    python ingest_conversations.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIZRA_ROOT = Path("/mnt/c/BIZRA-DATA-LAKE")
INTAKE_DIR = BIZRA_ROOT / "00_INTAKE"
GOLD_DIR = BIZRA_ROOT / "04_GOLD"

CONVERSATION_DIRS = [
    INTAKE_DIR / "2025-12-15-conversations-1",
    INTAKE_DIR / "2025-12-15-conversations-2",
    INTAKE_DIR / "2025-12-15-conversations-3",
]

EXISTING_CHUNKS_PATH = GOLD_DIR / "chunks.parquet"
OUTPUT_CHUNKS_PATH = GOLD_DIR / "conversations_chunks.parquet"
FAISS_INDEX_PATH = GOLD_DIR / "node0_faiss.index"
FAISS_META_PATH = GOLD_DIR / "node0_faiss_meta.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 128
MAX_CHUNK_CHARS = 8000  # ~2000 tokens; truncate longer turns
MIN_CHUNK_CHARS = 20  # skip trivially short turns
ROLES_TO_EXTRACT = {"user", "assistant"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_conversations")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deterministic_id(text: str, namespace: str = "conv") -> str:
    """Generate a 16-hex-char deterministic ID from text content."""
    return hashlib.blake2b(f"{namespace}:{text}".encode(), digest_size=8).hexdigest()


def extract_turns(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract text turns from a ChatGPT conversation JSON.

    Returns a list of dicts with keys: role, text, create_time, node_id.
    Only includes user and assistant turns with non-trivial text content.
    """
    mapping = conversation.get("mapping", {})
    turns: list[dict[str, Any]] = []

    for node_id, node in mapping.items():
        msg = node.get("message")
        if msg is None:
            continue

        author = msg.get("author", {})
        role = author.get("role", "")
        if role not in ROLES_TO_EXTRACT:
            continue

        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue

        parts = content.get("parts", [])
        text_parts = [
            p for p in parts if isinstance(p, str) and len(p.strip()) >= MIN_CHUNK_CHARS
        ]
        if not text_parts:
            continue

        combined_text = "\n\n".join(text_parts)
        if len(combined_text) > MAX_CHUNK_CHARS:
            combined_text = combined_text[:MAX_CHUNK_CHARS]

        turns.append(
            {
                "role": role,
                "text": combined_text,
                "create_time": msg.get("create_time"),
                "node_id": node_id,
            }
        )

    # Sort by create_time to preserve conversation order
    turns.sort(key=lambda t: t.get("create_time") or 0)
    return turns


def load_conversations() -> list[dict[str, Any]]:
    """Load all conversation JSON files from the three intake directories."""
    conversations: list[dict[str, Any]] = []
    errors = 0

    for conv_dir in CONVERSATION_DIRS:
        if not conv_dir.exists():
            log.warning("Directory not found: %s", conv_dir)
            continue

        json_files = sorted(conv_dir.glob("*.json"))
        log.info("Found %d JSON files in %s", len(json_files), conv_dir.name)

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_source_file"] = str(jf)
                conversations.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("Failed to parse %s: %s", jf.name, exc)
                errors += 1

    log.info("Loaded %d conversations (%d parse errors)", len(conversations), errors)
    return conversations


def build_chunks(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert conversations into chunk records matching the chunks.parquet schema.

    Each conversation turn (user or assistant) becomes one chunk.
    The doc_id is derived from the conversation_id.
    """
    chunks: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for conv in conversations:
        conv_id = conv.get("conversation_id", "")
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time")
        model_slug = conv.get("default_model_slug", "unknown")
        source_file = conv.get("_source_file", "")

        doc_id = deterministic_id(conv_id, namespace="conv_doc")
        turns = extract_turns(conv)

        for idx, turn in enumerate(turns):
            # Prefix the text with role and conversation context
            role_label = "Human" if turn["role"] == "user" else "Assistant"
            chunk_text = f"[{role_label}] {turn['text']}"

            chunk_id = deterministic_id(
                f"{conv_id}:{turn['node_id']}:{idx}", namespace="conv_chunk"
            )

            token_est = len(chunk_text.split()) * 1.3  # rough token estimate

            metadata = {
                "source_type": "conversation",
                "conversation_id": conv_id,
                "conversation_title": title,
                "role": turn["role"],
                "turn_index": idx,
                "model_slug": model_slug,
                "source_file": Path(source_file).name if source_file else "",
            }

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "token_est": float(token_est),
                    "created_at": now_iso,
                    "chunk_metadata_json": json.dumps(metadata),
                    "embedding_model": EMBEDDING_MODEL,
                }
            )

    log.info("Built %d chunks from %d conversations", len(chunks), len(conversations))
    return chunks


def generate_embeddings(
    chunks: list[dict[str, Any]], model: SentenceTransformer
) -> np.ndarray:
    """
    Generate L2-normalized embeddings for all chunks in batches.
    Returns a float32 array of shape (n_chunks, 384).
    """
    texts = [c["chunk_text"] for c in chunks]
    n = len(texts)
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    log.info("Generating embeddings for %d chunks (batch_size=%d)...", n, BATCH_SIZE)
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_texts = texts[start:end]
        batch_emb = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalization built-in
            convert_to_numpy=True,
        )
        embeddings[start:end] = batch_emb.astype(np.float32)

        if (start // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - t0
            pct = end / n * 100
            log.info("  Progress: %d/%d (%.1f%%) — %.1fs elapsed", end, n, pct, elapsed)

    elapsed = time.time() - t0
    log.info(
        "Embedding generation complete in %.1fs (%.0f chunks/sec)", elapsed, n / elapsed
    )

    # Double-check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not L2-normalized"
    log.info("L2 normalization verified (mean norm: %.6f)", norms.mean())

    return embeddings


def save_conversations_parquet(
    chunks: list[dict[str, Any]], embeddings: np.ndarray
) -> pd.DataFrame:
    """Save the conversation chunks with embeddings to conversations_chunks.parquet."""
    df = pd.DataFrame(chunks)

    # Store embeddings as list-of-float (matching existing parquet schema)
    df["embedding"] = [embeddings[i] for i in range(len(embeddings))]

    # Enforce column order to match existing chunks.parquet
    column_order = [
        "chunk_id",
        "doc_id",
        "chunk_index",
        "chunk_text",
        "token_est",
        "created_at",
        "chunk_metadata_json",
        "embedding",
        "embedding_model",
    ]
    df = df[column_order]

    df.to_parquet(OUTPUT_CHUNKS_PATH, engine="pyarrow", index=False)
    log.info(
        "Saved %d chunks to %s (%.1f MB)",
        len(df),
        OUTPUT_CHUNKS_PATH.name,
        OUTPUT_CHUNKS_PATH.stat().st_size / 1e6,
    )
    return df


def rebuild_faiss_index(
    existing_parquet: Path, new_parquet: Path
) -> tuple[faiss.Index, int]:
    """
    Rebuild the FAISS IVF index from both existing and new parquet files.

    Strategy:
    1. Load embeddings from both parquets.
    2. Compute appropriate number of centroids for the combined dataset.
    3. Train a new IVFFlat index on the combined vectors.
    4. Add all vectors.
    5. Save the index and metadata.
    """
    log.info("Loading existing embeddings from %s...", existing_parquet.name)
    df_existing = pd.read_parquet(existing_parquet)
    existing_embs = np.array(df_existing["embedding"].tolist(), dtype=np.float32)
    n_existing = len(existing_embs)
    log.info("  Existing: %d vectors", n_existing)

    log.info("Loading new embeddings from %s...", new_parquet.name)
    df_new = pd.read_parquet(new_parquet)
    new_embs = np.array(df_new["embedding"].tolist(), dtype=np.float32)
    n_new = len(new_embs)
    log.info("  New: %d vectors", n_new)

    # Combine
    all_embs = np.vstack([existing_embs, new_embs])
    n_total = len(all_embs)
    log.info("Combined: %d vectors (%d + %d)", n_total, n_existing, n_new)

    # Verify normalization
    norms = np.linalg.norm(all_embs[:1000], axis=1)
    assert np.allclose(
        norms, 1.0, atol=1e-4
    ), f"Vectors not normalized: norms range [{norms.min():.4f}, {norms.max():.4f}]"

    # Calculate centroids: sqrt(n) is a good heuristic, minimum 1
    n_centroids = max(1, int(np.sqrt(n_total)))
    # IVF requires at least n_centroids * 39 training points
    min_training = n_centroids * 39
    if n_total < min_training:
        n_centroids = max(1, n_total // 39)
    n_probe = max(1, n_centroids // 10)

    log.info(
        "FAISS config: dim=%d, centroids=%d, nprobe=%d, metric=inner_product",
        EMBEDDING_DIM,
        n_centroids,
        n_probe,
    )

    # Build IVFFlat with inner product (cosine on L2-normalized vectors)
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(
        quantizer, EMBEDDING_DIM, n_centroids, faiss.METRIC_INNER_PRODUCT
    )

    log.info("Training FAISS index on %d vectors...", n_total)
    t0 = time.time()
    index.train(all_embs)
    train_time = time.time() - t0
    log.info("Training complete in %.1fs", train_time)

    log.info("Adding %d vectors to index...", n_total)
    t0 = time.time()
    index.add(all_embs)
    add_time = time.time() - t0
    log.info("Add complete in %.1fs", add_time)

    index.nprobe = n_probe

    # Save index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    index_size = FAISS_INDEX_PATH.stat().st_size
    log.info(
        "Saved FAISS index: %s (%.1f MB, %d vectors)",
        FAISS_INDEX_PATH.name,
        index_size / 1e6,
        index.ntotal,
    )

    # Build combined chunk_id list for meta (first and last 3 as sample)
    all_chunk_ids = pd.concat(
        [df_existing["chunk_id"], df_new["chunk_id"]], ignore_index=True
    ).tolist()
    sample_ids = all_chunk_ids[:3] + all_chunk_ids[-3:]

    # Save metadata
    meta = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_vectors": int(index.ntotal),
        "dim": EMBEDDING_DIM,
        "n_centroids": n_centroids,
        "n_probe": n_probe,
        "metric": "cosine (L2-normalized + inner_product)",
        "source": f"04_GOLD/chunks.parquet + 04_GOLD/{OUTPUT_CHUNKS_PATH.name}",
        "embedding_model": EMBEDDING_MODEL,
        "chunk_ids": sample_ids,
        "index_size_bytes": index_size,
        "n_existing_chunks": n_existing,
        "n_conversation_chunks": n_new,
    }
    with open(FAISS_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved FAISS metadata to %s", FAISS_META_PATH.name)

    return index, n_total


def verify_index(index: faiss.Index, df_new: pd.DataFrame) -> None:
    """Run a quick sanity check: query with a new embedding and verify we get a match."""
    sample_emb = np.array(df_new["embedding"].iloc[0], dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(sample_emb, 5)
    log.info("Verification search — top 5 distances: %s", distances[0].tolist())
    log.info("Verification search — top 5 indices: %s", indices[0].tolist())
    if distances[0][0] > 0.99:
        log.info("Self-match confirmed (distance %.4f >= 0.99)", distances[0][0])
    else:
        log.warning(
            "Self-match distance %.4f is lower than expected — check normalization",
            distances[0][0],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    log.info("=" * 70)
    log.info("BIZRA Conversation Ingestion Pipeline")
    log.info("=" * 70)

    # Step 1: Load conversations
    conversations = load_conversations()
    if not conversations:
        log.error("No conversations found. Exiting.")
        sys.exit(1)

    # Step 2: Build chunks
    chunks = build_chunks(conversations)
    if not chunks:
        log.error("No chunks extracted. Exiting.")
        sys.exit(1)

    # Step 3: Generate embeddings
    log.info("Loading SentenceTransformer model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = generate_embeddings(chunks, model)

    # Step 4: Save parquet
    df_new = save_conversations_parquet(chunks, embeddings)

    # Step 5: Rebuild FAISS index
    index, n_total = rebuild_faiss_index(EXISTING_CHUNKS_PATH, OUTPUT_CHUNKS_PATH)

    # Step 6: Verify
    verify_index(index, df_new)

    # Summary
    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("  Conversations processed: %d", len(conversations))
    log.info("  Chunks created:          %d", len(chunks))
    log.info("  Parquet saved:           %s", OUTPUT_CHUNKS_PATH.name)
    log.info("  FAISS index vectors:     %d (was %d)", n_total, n_total - len(chunks))
    log.info("  FAISS index file:        %s", FAISS_INDEX_PATH.name)
    log.info("  Total time:              %.1fs", elapsed)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
