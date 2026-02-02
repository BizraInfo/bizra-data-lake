# Doc → Code Map (Truth‑State)

**Purpose:** eliminate doc drift by mapping claims to code locations.

## Core Docs
- `README.md` → `DataLakeProcessor.ps1`, `corpus_manager.py`, `vector_engine.py`, `hypergraph_engine.py`
- `ARCHITECTURE.md` → `bizra_config.py`, `corpus_manager.py`, `vector_engine.py`, `arte_engine.py`, `hypergraph_engine.py`, `pat_engine.py`, `kep_bridge.py`, `warp_bridge.py`
- `IHSAN_CONSTRAINTS.yaml` → `arte_engine.py` (Ihsān gate), `core/iaas/ingest_gates.py`
- `AUDIT_REPORT.md` → `core/federation/*`, `bizra_resilience.py`, `core/sovereign/*`
- `docs/knowledge/uers_overview.md` → `core/uers/*`

## Pipelines
- Ingestion → `DataLakeProcessor.ps1` → `corpus_manager.py`
- Chunking/Embedding → `vector_engine.py`
- Graph Index → `hypergraph_engine.py`
- Reasoning → `arte_engine.py`
- Orchestration → `bizra_orchestrator.py`, `bizra_nexus.py`

## Notes
- Federation networking claims are **aspirational** unless backed by executable P2P tests.
- Any doc section without a mapping should be marked **ASPIRATIONAL**.
