# BIZRA Knowledge Base (Maestro)

**Purpose:** A living, high‑signal index of the BIZRA corpus for rapid retrieval, reasoning, and system activation.

---

## 1) Canonical Entry Points (Read First)
- `NODE0_GENESIS_COVENANT.md` — mission & covenant
- `ARCHITECTURE.md` — system architecture v2.0
- `HYPERGRAPH-RAG-VISION.md` — retrieval vision
- `TAXONOMY.md` — knowledge taxonomy
- `SAPE_IMPLEMENTATION_BLUEPRINT.md` — SAPE spec
- `SAPE_COMPREHENSIVE_ANALYSIS.md` — SAPE analysis
- `BIZRA-ECOSYSTEM-MAP.md` — ecosystem map
- `README.md` / `QUICK-START.md` — practical overview

---

## 2) Data Lake Topology
- `00_INTAKE/` → raw drop zone
- `01_RAW/` → immutable backups
- `02_PROCESSED/` → sorted by type
- `03_INDEXED/` → embeddings, graphs, indexes
- `04_GOLD/` → curated high‑value assets
- `99_QUARANTINE/` → duplicates/corrupt

---

## 3) Core Engines & Modules (by file)
- **Config:** `bizra_config.py`
- **Corpus:** `corpus_manager.py`
- **Vector:** `vector_engine.py`
- **Hypergraph:** `hypergraph_engine.py`
- **SNR/ARTE:** `arte_engine.py`
- **KEP Bridge:** `kep_bridge.py`
- **PAT Engine:** `pat_engine.py`
- **Orchestrator/Nexus:** `bizra_orchestrator.py`, `bizra_nexus.py`
- **Sovereign suite:** `sovereign_*`
- **Flywheel:** `flywheel.py` / `flywheel_api.py`
- **MCP:** `mcp_gateway.py`, `bizra_mcp.py`

---

## 4) Indexed Knowledge Assets (Key)
- `03_INDEXED/graph/nodes.jsonl`
- `03_INDEXED/graph/edges.jsonl`
- `03_INDEXED/embeddings/` (FAISS)
- `04_GOLD/chunks.parquet`
- `04_GOLD/documents.parquet`
- `04_GOLD/poi_ledger.jsonl`

---

## 5) Retrieval Modes
- **Semantic:** FAISS HNSW (single‑vector)
- **Structural:** Graph traversal
- **Hybrid:** vector + graph
- **Multi‑vector:** XTR‑WARP (ColBERT)

---

## 6) Live TODO (Next actions)
1. Build machine‑readable **Corpus Map** (file counts, types, sizes)
2. Generate **Module Index** (top‑level imports, dependency graph)
3. Create **Doc Cross‑Reference** (topics ↔ files)
4. Attach **SNR + Ihsān gating** to all new ingestion pipelines

---

## 7) Operating Principles
- **Ihsān gate:** SNR ≥ 0.99
- **Evidence‑first:** every insight must cite sources
- **No destructive changes** without explicit approval

---

## 8) Quick Jump
- SAPE: `SAPE_IMPLEMENTATION_BLUEPRINT.md`
- Taxonomy: `TAXONOMY.md`
- Architecture: `ARCHITECTURE.md`
- Recovery: `RECOVERY-REPORT.md`
- Strategy: `BIZRA_STRATEGY_DECK_2026.md`

---

*This file is the top‑level knowledge anchor. Update after every major discovery.*
