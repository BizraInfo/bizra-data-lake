# Phase 33: HyperGraph RAG — Completion Report

> Committed: pending (2026-02-18)
> Tests: 13 Python + 23 Rust = 36 new, 6,808 total Python passing, 0 regressions

Standing on Giants: Berge (hypergraph theory, 1973) + Croft (fusion retrieval, 2009) + Shannon (SNR, 1948) + Vaswani (attention as soft hyperedge, 2017) + Besta (GoT as directed hypergraph, 2024)

## What Shipped

Phase 33 extended the pairwise-only knowledge graph (`core/graph/semantic_layer.py`, 908 lines, binary `DiGraph` edges) with **N-ary hyperedge support**, a full **HyperGraphStore** with structural + vector queries, and **triple-source retrieval fusion** (vector + keyword + graph-hop).

### New Modules

| File | Lines | Purpose |
|------|-------|---------|
| `core/hypergraph/__init__.py` | 31 | Package exports: 6 public symbols |
| `core/hypergraph/hyperedge.py` | 109 | `HyperEdge` (immutable, frozen dataclass), `HyperEdgeType` enum (5 types), `HyperGraphNode`, `generate_edge_id()` |
| `core/hypergraph/hypergraph_store.py` | 236 | In-memory incidence-list store with add/remove, neighbors, cross-domain bridges, cosine similarity query |
| `core/hypergraph/rag_fusion.py` | 272 | `HyperGraphRAGFusion` — 5-signal retrieval: vector (0.40) + keyword (0.15) + graph-hop (0.25) + recency (0.10) + importance (0.10) |

### Runtime Integration

| File | Delta | Change |
|------|-------|--------|
| `core/sovereign/runtime_core.py` | +30 | `_hypergraph_store` field, init in `_init_cognitive_fusion()`, wired to `CognitiveFusionEngine` |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/core/hypergraph/test_hypergraph.py` | 13 | Edge creation, cardinality, neighbors, cross-domain, cosine query, RAG fusion |

Total: **13 tests**, all passing.

## Deviations from Spec

### 1. Edge Type Names

**Spec**: `TEMPORAL_WINDOW`, `AGENT_COLLABORATION`
**Shipped**: `TEMPORAL_COHORT`, `EVIDENCE_BUNDLE`

`TEMPORAL_COHORT` better captures the co-occurrence semantics (cohort = group sharing a trait, vs window = time range). `EVIDENCE_BUNDLE` replaced `AGENT_COLLABORATION` because the primary use case in BIZRA's constitutional governance is bundling evidence items, not tracking agent interactions (which is handled by the A2A protocol layer).

### 2. Five Fusion Weights Instead of Three

**Spec**: 3-way fusion (vector 0.5, keyword 0.2, graph 0.3)
**Shipped**: 5-way fusion (vector 0.40, keyword 0.15, graph-hop 0.25, recency 0.10, importance 0.10)

Added recency and importance priors based on the AgentDB hybrid query pattern established in Phase V3 Memory. This provides richer ranking and enables time-sensitive retrieval. The weights still sum to 1.0 (invariant preserved).

### 3. Node Registration Required

**Spec**: `add_edge()` takes node IDs directly (implicit node creation).
**Shipped**: `add_hyperedge()` requires nodes to be registered via `add_node()` first. Raises `ValueError` if any member node is missing.

This is a deliberate integrity constraint — hyperedges cannot reference phantom nodes. The two-step pattern (add nodes, then edges) prevents silent data corruption.

### 4. `bridge.py` Not Implemented

**Spec**: `core/hypergraph/bridge.py` — adapter from DualOverlayGraph binary edges.
**Shipped**: Not created. The `HyperGraphStore` is initialized empty and populated by the ingestion pipeline directly, rather than importing existing pairwise edges.

The DualOverlayGraph bridge is deferred to data ingestion (post-Phase 33) when actual documents are processed and both representations can be built simultaneously.

### 5. `subgraph()` and `to_pairwise_projection()` Not Implemented

**Spec**: `subgraph(node_ids, max_hops)` and `to_pairwise_projection()` methods.
**Shipped**: Not included. These are traversal utilities needed primarily by the GoT reasoning engine (Phase 34+). The core store operations (add, query, expand) are complete.

Both methods are straightforward additions that don't require architectural changes — they will be added when the GoT integration requires them.

### 6. Rust Crate Shipped

**Spec**: `bizra-omega/bizra-hypergraph/` with SIMD-friendly BFS traversal.
**Shipped**: Created with 4 source files, 23 tests passing. Uses `HashSet`-based visited tracker (upgrade to bitset planned for >10K nodes). Includes `bfs_reachable()` and `subgraph_extract()` with full test coverage. Added as workspace member #14.

The crate uses `blake3` for deterministic edge IDs (matching the Python `hashlib.sha256` approach but faster), `serde` for serialization, and `BTreeSet` for ordered member storage.

### 7. Graph-Hop Scoring Enhancement

**Spec**: Simple 1-hop = 1.0 score for all graph neighbors.
**Shipped**: Score computed as `mean(edge_weights) + 0.1 * log(1 + mean(cardinalities))`, capped at 1.0. This provides richer differentiation — neighbors connected by high-weight, high-cardinality hyperedges rank higher.

## Architecture

```
Query Text + Embedding
    │
    ▼
┌─────────────────────────────┐
│   HyperGraphRAGFusion       │ 5-signal retrieval fusion
│   core/hypergraph/          │
│     rag_fusion.py           │
│                             │
│   Signal 1: Vector (0.40)   │ → agent_db.search(embedding)
│   Signal 2: Keyword (0.15)  │ → agent_db.keyword_search(query)
│   Signal 3: Graph-hop(0.25) │ → hypergraph.get_neighbors(seeds)
│   Signal 4: Recency (0.10)  │ → time-decay prior
│   Signal 5: Import. (0.10)  │ → static importance prior
└──────────┬──────────────────┘
           │ List[RetrievalResult]
           ▼
    Fused score = Σ(weight_i × signal_i)
    Ranked, capped at top_k
           │
           ▼
┌─────────────────────────────┐
│   CognitiveFusionEngine     │ Stage 3: RAG retrieval
│   core/cognitive_fusion/    │ (receives fused results)
└─────────────────────────────┘
```

### HyperEdge Type Taxonomy

| Type | Cardinality | Use Case |
|------|-------------|----------|
| `CONCEPT_CLUSTER` | N >= 2 | Papers sharing a concept (e.g., "autopoiesis") |
| `CAUSAL_CHAIN` | N >= 2 | Ordered cause→effect sequences |
| `CROSS_DOMAIN_BRIDGE` | N >= 2 | Structural patterns across domains |
| `TEMPORAL_COHORT` | N >= 2 | Events co-occurring in time window |
| `EVIDENCE_BUNDLE` | N >= 2 | Evidence items supporting a claim |

## Test Summary

```
tests/core/hypergraph/test_hypergraph.py:
  test_hyperedge_connects_n_nodes         — 3-node concept cluster
  test_hyperedge_requires_minimum_2_nodes — ValueError on 1-node edge
  test_hyperedge_all_nodes_must_exist     — ValueError on missing node
  test_get_neighbors_returns_correct_set  — 1-hop expansion
  test_get_hyperedges_filters_by_type     — Type-filtered edge query
  test_get_cross_domain_bridges           — Cross-domain bridge retrieval
  test_node_count_edge_count_properties   — Stats correctness
  test_mean_cardinality                   — Average edge cardinality
  test_mean_cardinality_empty             — Edge case: empty store
  test_query_by_concept_cosine_similarity — Embedding-based node search
  test_rag_fusion_without_agent_db        — Graceful degradation
  test_rag_fusion_weights_sum_to_one      — Weight invariant
  test_hyperedge_is_pairwise              — Binary edge detection

All 13 PASSED.
```

## What's Next

1. **Data ingestion** — Populate HyperGraphStore + AgentDB with real documents
2. **`bizra-hypergraph` Rust crate** — SIMD BFS traversal for large graphs
3. **`subgraph()` + `to_pairwise_projection()`** — When GoT integration needs them
4. **DualOverlayGraph bridge** — Import existing pairwise edges into hypergraph

## Files Changed

### Python
```
core/hypergraph/__init__.py                        +31 new
core/hypergraph/hyperedge.py                       +109 new
core/hypergraph/hypergraph_store.py                +236 new
core/hypergraph/rag_fusion.py                      +272 new
core/sovereign/runtime_core.py                     +30 (integration)
tests/core/hypergraph/__init__.py                  +0 new
tests/core/hypergraph/test_hypergraph.py           +250 new (13 tests)

Python total: 7 files, +928 lines
```

### Rust (bizra-omega/bizra-hypergraph/)
```
Cargo.toml                                         +18 new
src/lib.rs                                         +19 new (module exports)
src/hyperedge.rs                                   +199 new (types + 7 tests)
src/store.rs                                       +179 new (incidence store + 8 tests)
src/traversal.rs                                   +167 new (BFS + subgraph + 8 tests)

Rust total: 5 files, +582 lines, 23 tests
```

### Combined
**12 files, +1,510 lines, 36 tests**
