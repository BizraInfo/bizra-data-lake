# Module 06 — Knowledge Layer

> **Domain:** Memory, hypergraph, embeddings, living memory, RAG pipeline
> **Source Specs:** V3 Memory Unification, Phase 49 (refinement), Phase 56 (engine)
> **Rust Mirror:** `bizra-omega/bizra-memory/`

## 6.1 Living Memory Core

**Status:** [x] BUILT
**Path:** `core/living_memory/core.py`

Proactive retrieval system that surfaces relevant knowledge before being asked.
Integrates with FAISS for vector similarity and maintains temporal decay.

**Key class:** `LivingMemoryCore`
**Integration:** Used by MissionOrchestrator in SYNTHESIZE phase

**Tests:** `tests/core/living_memory/`

---

## 6.2 Hypergraph Knowledge Structure

**Status:** [x] BUILT
**Path:** `core/hypergraph/`

Hyperedge-based knowledge representation allowing N-ary relationships (beyond
simple binary edges). Used for complex concept mapping.

**Rust mirror:** `bizra-omega/bizra-core/` (hypergraph primitives)

**Tests:** `tests/core/hypergraph/`

---

## 6.3 Graph Structures

**Status:** [x] BUILT
**Path:** `core/graph/`

Standard graph operations, traversal, and query support. Interfaces with Neo4j
(port 7474/7687) for persistent wisdom graph storage.

**Tests:** `tests/core/graph/`

---

## 6.4 Vector Embedding Pipeline

**Status:** [x] BUILT
**Path:** `vector_engine.py` (root)

Layer 2 of data pipeline: generates embeddings from `04_GOLD/documents.parquet`,
produces `04_GOLD/chunks.parquet` with vector representations.

**Stack:** sentence-transformers, FAISS-CPU
**Storage:** ChromaDB (8001 internal, 8100 external), pgvector (5433)

---

## 6.5 FAISS Index Management

**Status:** [x] BUILT
**Path:** `core/living_memory/`, `vector_engine.py`

HNSW index for fast approximate nearest neighbor search. Similarity floor
at `FAISS_SIMILARITY_FLOOR = 0.35` from constants.py.

**Known flaky test:** `test_save_all_flushes_hnsw` — race condition under full suite load

---

## 6.6 Corpus Manager (Data Pipeline Layer 1)

**Status:** [x] BUILT
**Path:** `corpus_manager.py` (root)

Builds `04_GOLD/documents.parquet` from ingested files. Handles deduplication
via SHA-256, quarantine to `99_QUARANTINE/`.

**Pipeline:** `00_INTAKE/` -> `01_RAW/` -> `02_PROCESSED/` -> `03_INDEXED/` -> `04_GOLD/`

---

## 6.7 Language Extraction Engine

**Status:** [x] BUILT
**Path:** `langextract_engine.py` (root)

Layer 4: LLM-powered extraction producing `assertions.jsonl`. Uses tiered
inference (LM Studio -> Ollama -> Cloud fallback).

---

## 6.8 ARTE SNR Validation

**Status:** [x] BUILT
**Path:** `arte_engine.py` (root)

SNR validation layer for extracted knowledge. Filters low-quality assertions
below `UNIFIED_SNR_THRESHOLD`.

---

## 6.9 Memory Synthesis Pipeline (Rust)

**Status:** [x] BUILT
**Path:** `bizra-omega/bizra-memory/`

Rust-native memory synthesis with higher throughput than Python path.
Integrates with UCF EventBus for real-time memory updates.

**Tests:** `bizra-omega/bizra-memory/tests/` (cargo test)

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 6.1 Living Memory | BUILT | Full |
| 6.2 Hypergraph | BUILT | Full |
| 6.3 Graph Structures | BUILT | Full |
| 6.4 Vector Pipeline | BUILT | Integration |
| 6.5 FAISS Index | BUILT | Flaky edge |
| 6.6 Corpus Manager | BUILT | Pipeline |
| 6.7 LangExtract | BUILT | Full |
| 6.8 ARTE SNR | BUILT | Full |
| 6.9 Rust Memory | BUILT | Cargo tests |
| **TOTAL** | **9/9** | **100%** |
