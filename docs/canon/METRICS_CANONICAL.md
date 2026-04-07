# METRICS_CANONICAL

Canonical benchmark results for the BIZRA system.
## no_std Dependency Audit (D4)

**Date:** 2026-04-07 | **Workspace:** bizra-omega (26 crates)

| Status | Count | Crates |
|--------|-------|--------|
| `no_std` | 2 | bizra-efi, bizra-hooks |
| `std` | 24 | All others |

**Candidates for `no_std` migration** (zero std-only references in src/):

| Crate | std::fs/net/thread refs | Migration feasibility |
|-------|------------------------|----------------------|
| bizra-sippar | 0 | HIGH — pure arithmetic, no I/O |
| bizra-mission | 0 | HIGH — state machine + receipts, no I/O |
| bizra-hypergraph | 0 | HIGH — graph data structures only |
| bizra-core | 11 | MEDIUM — has some fs/io, needs feature gates |

**Impact:** Migrating sippar + mission + hypergraph to `no_std` would bring 5/26 crates
(19%) to embedded-compatible, enabling UEFI and bare-metal deployment of the
constitutional spine (receipts, state machine, exact arithmetic).

## Recall@k Benchmark (D6)

**Date:** 2026-04-07 | **Method:** Known-item search | **N=100** | **Seed:** 42
**Corpus:** 84,795 chunks (384-dim embeddings) from 1,437 documents

### v2 — Native HNSW (optimized)

Replaced RuVector Node.js subprocess bridge (10,032ms/query) with native hnswlib
(40ms/query, 250x speedup). Tuned ef_search=200 (Malkov & Yashunin: ef >> k).

| Engine | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency |
|--------|----------|----------|-----------|-----|---------|-------------|
| FAISS (IVF) | 0.3700 | 0.5600 | 0.5800 | 0.4428 | 0.4815 | 826 ms |
| HNSW (native) | 0.3300 | 0.4700 | 0.5500 | 0.3867 | 0.4324 | 40 ms |
| Hybrid RRF | 0.2300 | 0.5200 | 0.6100 | 0.3500 | 0.4127 | 86 ms |

### v1 — Original (subprocess bridge)

| Engine | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency |
|--------|----------|----------|-----------|-----|---------|-------------|
| FAISS | 0.3700 | 0.5600 | 0.5800 | 0.4428 | 0.4815 | 921 ms |
| RuVector | 0.1900 | 0.3000 | 0.3700 | 0.2337 | 0.2654 | 10032 ms |
| Hybrid RRF | 0.2800 | 0.5000 | 0.6000 | 0.3660 | 0.4211 | 4886 ms |

**Methodology:** For each query, a chunk is randomly sampled from the GOLD corpus.
The first sentence (or first 200 chars) is used as the query. Ground truth: the source
chunk_id must appear in the top-k results. This is known-item search evaluation
(Voorhees, 2001). Queries are deterministic (seed=42) for reproducibility.

**Verdict: PARTIAL** — Hybrid RRF R@10=0.61, between 0.50 (overclaim) and 0.70
(verified). Both engines use the same embedding model (all-MiniLM-L6-v2, 384-dim),
producing correlated rankings that limit RRF fusion gain. To reach 0.70, the system
needs orthogonal retrieval signals (BM25 sparse + dense hybrid, or cross-encoder
re-ranking). No prior SNR claims invalidated — retrieval quality is honest and
measurable.

**Key improvement:** Native HNSW replaces RuVector subprocess bridge — 250x latency
reduction (10,032ms → 40ms) while improving HNSW recall from 0.37 to 0.55.
