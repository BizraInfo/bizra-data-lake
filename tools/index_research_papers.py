"""
Index 3,451 research papers from data/sci_reasoning/prior_works/ into the knowledge base.

Creates research_chunks.parquet in 04_GOLD/ and rebuilds the combined FAISS index
(node0_faiss.index) from all parquet files with embedding columns.

Schema:
  - Files WITH target_paper: chunk_text = "{title}\n\n{abstract}"
  - Files WITHOUT target_paper: chunk_text = synthesis_narrative
  - Venue/year derived from directory name (e.g., ICLR_2024 -> venue=ICLR, year=2024)
  - Paper ID derived from filename (e.g., 06lrITXVAx.json -> paper_id=06lrITXVAx)

Embedding: all-MiniLM-L6-v2 (384-dim), L2-normalized for cosine similarity.
FAISS: IndexIVFFlat with sqrt(N) centroids and nprobe=ceil(sqrt(ncentroids)).
"""

import hashlib
import json
import logging
import math
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
PRIOR_WORKS_DIR = BIZRA_ROOT / "data" / "sci_reasoning" / "prior_works"
GOLD_DIR = BIZRA_ROOT / "04_GOLD"
OUTPUT_PARQUET = GOLD_DIR / "research_chunks.parquet"
FAISS_INDEX_PATH = GOLD_DIR / "node0_faiss.index"
FAISS_META_PATH = GOLD_DIR / "node0_faiss_meta.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 96  # Balanced for memory on RTX 4090 / 128GB RAM
TOKEN_EST_RATIO = 0.75  # ~0.75 tokens per character for English text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_chunk_id(text: str) -> str:
    """Deterministic 16-hex-char chunk ID from content hash."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def make_doc_id(venue: str, year: str, paper_id: str) -> str:
    """Deterministic 16-hex-char doc ID from paper identity."""
    key = f"{venue}_{year}_{paper_id}"
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def parse_venue_year(dirname: str) -> tuple[str, int]:
    """Parse 'ICLR_2024' -> ('ICLR', 2024)."""
    parts = dirname.rsplit("_", 1)
    venue = parts[0]
    year = int(parts[1]) if len(parts) > 1 else 0
    return venue, year


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, int(len(text) / 4))


# ---------------------------------------------------------------------------
# Step 1: Load and parse all paper JSONs
# ---------------------------------------------------------------------------
def load_papers() -> list[dict[str, Any]]:
    """Load all paper JSONs and extract chunk text + metadata."""
    papers: list[dict[str, Any]] = []
    skipped = 0

    venue_dirs = sorted(d for d in PRIOR_WORKS_DIR.iterdir() if d.is_dir())
    log.info("Scanning %d venue directories in %s", len(venue_dirs), PRIOR_WORKS_DIR)

    for venue_dir in venue_dirs:
        venue, year = parse_venue_year(venue_dir.name)
        json_files = sorted(venue_dir.glob("*.json"))
        log.info("  %s: %d JSON files", venue_dir.name, len(json_files))

        for jf in json_files:
            paper_id = jf.stem  # filename without .json

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("Skipping %s: %s", jf, exc)
                skipped += 1
                continue

            # Extract chunk text
            target_paper = data.get("target_paper") or {}
            title = target_paper.get("title", "")
            abstract = target_paper.get("abstract", "")
            authors = target_paper.get("authors", "")
            keywords = target_paper.get("keywords", "")
            presentation_type = target_paper.get("presentation_type", "")
            synthesis = data.get("synthesis_narrative", "")

            # Determine the conference/year from target_paper if available,
            # otherwise fall back to directory name
            paper_venue = target_paper.get("conference", venue)
            paper_year = target_paper.get("year", year)

            # Build chunk text: prefer title+abstract, fall back to synthesis
            if title and abstract:
                chunk_text = f"{title}\n\n{abstract}"
            elif title:
                chunk_text = title
            elif synthesis:
                # For papers without target_paper, use synthesis narrative
                # Prefix with prior_works titles to give context
                prior_titles = [
                    pw.get("title", "")
                    for pw in data.get("prior_works", [])
                    if pw.get("title")
                ]
                if prior_titles:
                    context_header = (
                        f"Research synthesis connecting: "
                        f"{'; '.join(prior_titles[:5])}"
                    )
                    chunk_text = f"{context_header}\n\n{synthesis}"
                else:
                    chunk_text = synthesis
            else:
                log.warning("Skipping %s: no usable text", jf)
                skipped += 1
                continue

            # Truncate very long texts to ~2000 chars for embedding quality
            if len(chunk_text) > 2500:
                chunk_text = chunk_text[:2500]

            doc_id = make_doc_id(str(paper_venue), str(paper_year), paper_id)
            chunk_id = make_chunk_id(chunk_text)

            metadata = {
                "venue": str(paper_venue),
                "year": int(paper_year),
                "paper_id": paper_id,
                "source_dir": venue_dir.name,
                "source_type": "research_paper",
            }
            if title:
                metadata["title"] = title
            if authors:
                metadata["authors"] = authors
            if keywords:
                metadata["keywords"] = keywords
            if presentation_type:
                metadata["presentation_type"] = presentation_type
            if not title and synthesis:
                metadata["text_source"] = "synthesis_narrative"

            papers.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "chunk_text": chunk_text,
                    "token_est": float(estimate_tokens(chunk_text)),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "metadata": metadata,  # kept temporarily for reporting
                }
            )

    log.info(
        "Loaded %d papers (%d skipped) from %d venue directories",
        len(papers),
        skipped,
        len(venue_dirs),
    )
    return papers


# ---------------------------------------------------------------------------
# Step 2: Generate embeddings in batches
# ---------------------------------------------------------------------------
def generate_embeddings(papers: list[dict[str, Any]]) -> np.ndarray:
    """Generate L2-normalized embeddings for all papers using SentenceTransformer."""
    log.info("Loading SentenceTransformer model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [p["chunk_text"] for p in papers]
    n = len(texts)
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    log.info("Generating embeddings for %d papers (batch_size=%d)", n, BATCH_SIZE)
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_texts = texts[start:end]

        batch_emb = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )
        embeddings[start:end] = batch_emb

        if (start // BATCH_SIZE) % 5 == 0:
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            log.info(
                "  Embedded %d / %d (%.1f papers/sec)",
                end,
                n,
                rate,
            )

    elapsed = time.time() - t0
    log.info(
        "Embedding complete: %d vectors in %.1fs (%.1f papers/sec)",
        n,
        elapsed,
        n / elapsed if elapsed > 0 else 0,
    )

    # Verify L2-normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(
        norms, 1.0, atol=1e-5
    ), f"Embeddings not L2-normalized: min={norms.min():.6f}, max={norms.max():.6f}"

    return embeddings


# ---------------------------------------------------------------------------
# Step 3: Create research_chunks.parquet
# ---------------------------------------------------------------------------
def create_parquet(
    papers: list[dict[str, Any]], embeddings: np.ndarray
) -> pd.DataFrame:
    """Create and save research_chunks.parquet."""
    rows = []
    for i, paper in enumerate(papers):
        rows.append(
            {
                "chunk_id": paper["chunk_id"],
                "doc_id": paper["doc_id"],
                "chunk_index": paper["chunk_index"],
                "chunk_text": paper["chunk_text"],
                "token_est": paper["token_est"],
                "created_at": paper["created_at"],
                "chunk_metadata_json": paper["chunk_metadata_json"],
                "embedding": embeddings[i],
                "embedding_model": paper["embedding_model"],
            }
        )

    df = pd.DataFrame(rows)

    # Match the schema of existing chunks.parquet
    df["chunk_index"] = df["chunk_index"].astype("int64")
    df["token_est"] = df["token_est"].astype("float64")

    log.info("Writing %d rows to %s", len(df), OUTPUT_PARQUET)
    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")

    # Verify written file
    verify = pd.read_parquet(OUTPUT_PARQUET)
    assert len(verify) == len(df), "Parquet verification failed: row count mismatch"
    log.info(
        "Parquet verified: %d rows, columns: %s", len(verify), list(verify.columns)
    )

    return df


# ---------------------------------------------------------------------------
# Step 4: Rebuild combined FAISS index from all parquet files
# ---------------------------------------------------------------------------
def rebuild_faiss_index() -> int:
    """Rebuild node0_faiss.index from ALL parquets in 04_GOLD/ that have embeddings."""
    log.info("Rebuilding combined FAISS index from all parquets in %s", GOLD_DIR)

    all_embeddings: list[np.ndarray] = []
    all_chunk_ids: list[str] = []
    sources: list[str] = []

    for parquet_file in sorted(GOLD_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as exc:
            log.warning("Could not read %s: %s", parquet_file.name, exc)
            continue

        if "embedding" not in df.columns:
            log.info("  Skipping %s (no embedding column)", parquet_file.name)
            continue

        # Extract embeddings - handle both numpy arrays and lists
        emb_list = df["embedding"].tolist()
        n = len(emb_list)
        if n == 0:
            log.info("  Skipping %s (0 rows)", parquet_file.name)
            continue

        emb_matrix = np.array(emb_list, dtype=np.float32)

        # Verify dimension
        if emb_matrix.shape[1] != EMBEDDING_DIM:
            log.warning(
                "  Skipping %s: dimension %d != %d",
                parquet_file.name,
                emb_matrix.shape[1],
                EMBEDDING_DIM,
            )
            continue

        # Ensure L2-normalization
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        emb_matrix = emb_matrix / norms

        all_embeddings.append(emb_matrix)

        if "chunk_id" in df.columns:
            all_chunk_ids.extend(df["chunk_id"].tolist())
        else:
            all_chunk_ids.extend([f"{parquet_file.stem}_{i}" for i in range(n)])

        sources.append(f"{parquet_file.name} ({n} vectors)")
        log.info("  Loaded %s: %d vectors", parquet_file.name, n)

    if not all_embeddings:
        log.error("No embedding parquets found. Aborting FAISS rebuild.")
        return 0

    # Concatenate all embeddings
    combined = np.vstack(all_embeddings)
    n_total = combined.shape[0]
    log.info("Combined: %d vectors from %d parquet files", n_total, len(all_embeddings))

    # Build IndexIVFFlat: ncentroids = sqrt(N), nprobe = sqrt(ncentroids)
    n_centroids = max(4, int(math.sqrt(n_total)))
    n_probe = max(1, int(math.ceil(math.sqrt(n_centroids))))

    log.info(
        "Building IndexIVFFlat: dim=%d, n_centroids=%d, n_probe=%d",
        EMBEDDING_DIM,
        n_centroids,
        n_probe,
    )

    # Use inner product since vectors are L2-normalized (equivalent to cosine)
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(
        quantizer, EMBEDDING_DIM, n_centroids, faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = n_probe

    # Train the index
    t0 = time.time()
    log.info("Training FAISS index on %d vectors...", n_total)
    index.train(combined)
    log.info("Training completed in %.1fs", time.time() - t0)

    # Add vectors
    t0 = time.time()
    index.add(combined)
    log.info("Added %d vectors in %.1fs", n_total, time.time() - t0)

    # Save
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    index_size = FAISS_INDEX_PATH.stat().st_size
    log.info("Saved FAISS index: %s (%.1f MB)", FAISS_INDEX_PATH, index_size / 1e6)

    # Save metadata
    # Store first 5 and last 5 chunk_ids plus "..." for compactness
    if len(all_chunk_ids) > 10:
        sample_ids = all_chunk_ids[:5] + ["..."] + all_chunk_ids[-5:]
    else:
        sample_ids = all_chunk_ids

    meta = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_vectors": n_total,
        "dim": EMBEDDING_DIM,
        "n_centroids": n_centroids,
        "n_probe": n_probe,
        "metric": "cosine (L2-normalized + inner_product)",
        "sources": sources,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_ids": sample_ids,
        "index_size_bytes": index_size,
    }
    with open(FAISS_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved FAISS metadata: %s", FAISS_META_PATH)

    return n_total


# ---------------------------------------------------------------------------
# Step 5: Report
# ---------------------------------------------------------------------------
def report(papers: list[dict[str, Any]], faiss_total: int) -> None:
    """Print final summary report."""
    venues = {}
    text_sources = {"title_abstract": 0, "title_only": 0, "synthesis": 0}

    for p in papers:
        meta = p["metadata"]
        venue_key = f"{meta['venue']}_{meta['year']}"
        venues[venue_key] = venues.get(venue_key, 0) + 1

        if meta.get("text_source") == "synthesis_narrative":
            text_sources["synthesis"] += 1
        elif "\n\n" in p["chunk_text"]:
            text_sources["title_abstract"] += 1
        else:
            text_sources["title_only"] += 1

    print("\n" + "=" * 70)
    print("RESEARCH PAPER INDEXING REPORT")
    print("=" * 70)
    print(f"\nPapers processed:      {len(papers)}")
    print(f"  Title + Abstract:    {text_sources['title_abstract']}")
    print(f"  Title only:          {text_sources['title_only']}")
    print(f"  Synthesis narrative:  {text_sources['synthesis']}")
    print(f"\nChunks created:        {len(papers)} (1 chunk per paper)")
    print(f"Output parquet:        {OUTPUT_PARQUET}")
    print("\nVenue breakdown:")
    for vk in sorted(venues.keys()):
        print(f"  {vk:20s}: {venues[vk]:>5d}")
    print(f"\nFAISS index total:     {faiss_total} vectors")
    print(f"FAISS index path:      {FAISS_INDEX_PATH}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Starting research paper indexing pipeline")
    t_start = time.time()

    # Validate directories
    if not PRIOR_WORKS_DIR.exists():
        log.error("Prior works directory not found: %s", PRIOR_WORKS_DIR)
        sys.exit(1)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load papers
    papers = load_papers()
    if not papers:
        log.error("No papers loaded. Exiting.")
        sys.exit(1)

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(papers)

    # Step 3: Create parquet
    create_parquet(papers, embeddings)

    # Step 4: Rebuild FAISS index
    faiss_total = rebuild_faiss_index()

    # Step 5: Report
    report(papers, faiss_total)

    elapsed = time.time() - t_start
    log.info("Pipeline completed in %.1fs", elapsed)


if __name__ == "__main__":
    main()
