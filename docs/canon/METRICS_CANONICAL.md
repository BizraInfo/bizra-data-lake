# METRICS_CANONICAL

Canonical benchmark results for the BIZRA system.

## Recall@k Benchmark (D6)

**Date:** 2026-04-07 | **Method:** Known-item search | **N=100** | **Seed:** 42
**Corpus:** 84795 chunks (384-dim, all-MiniLM-L6-v2) from 1,437 documents

| Engine | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency |
|--------|----------|----------|-----------|-----|---------|-------------|
| FAISS | 0.3700 | 0.5600 | 0.5800 | 0.4428 | 0.4815 | 974.7 ms |
| Hybrid (FAISS-only) | 0.2700 | 0.4400 | 0.5900 | 0.3533 | 0.4086 | 233.3 ms |

**RuVector Status:** UNAVAILABLE during this benchmark. Node.js subprocess bridge
times out at 30s/query. Hybrid RRF fusion could not be tested with both engines.

**Methodology:** Known-item search (Voorhees, 2001). For each of 100 randomly sampled
chunks (seed=42), the first sentence is used as the query. Ground truth: the source
chunk_id must appear in top-k results.

**Verdict: PARTIAL** — Recall@10 = 0.58 is between the OVERCLAIM threshold (0.50) and
the VERIFIED threshold (0.70). The retrieval pipeline is functional but not elite.
No prior SNR claims are invalidated. RuVector fusion (expected to boost recall via
independent HNSW signal) remains untested due to subprocess timeout.

**Honest gaps:**
- RuVector subprocess bridge needs timeout investigation
- Hybrid RRF with dual engines is the expected path to Recall@10 >= 0.70
- Embedding model ceiling (all-MiniLM-L6-v2, 384-dim) may limit single-engine recall

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
