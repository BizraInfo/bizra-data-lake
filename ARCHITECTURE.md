# BIZRA Data Lake Architecture v2.0
**Doc Status:** Data pipeline architecture. For the full sovereign runtime architecture, see [ARCHITECTURE_BLUEPRINT_v2.3.0.md](docs/ARCHITECTURE_BLUEPRINT_v2.3.0.md).
## Comprehensive Technical Documentation

**Document:** BIZRA-ARCH-002
**Version:** 2.1.0
**Updated:** 2026-02-14
**Status:** Production

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Architecture](#2-core-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [SAPE Framework](#4-sape-framework)
5. [DDAGI Consciousness System](#5-ddagi-consciousness-system)
6. [Hypergraph RAG](#6-hypergraph-rag)
7. [PAT Engine](#7-pat-engine)
8. [Resilience Patterns](#8-resilience-patterns)
9. [Monitoring & Observability](#9-monitoring--observability)
10. [API Reference](#10-api-reference)

---

## 1. System Overview

### 1.1 Purpose

The BIZRA Data Lake is a unified knowledge repository that transforms scattered data (1.37TB+) into an organized, searchable knowledge base with:

- **Vector embeddings** for semantic search
- **Graph relationships** for structural reasoning
- **Multi-agent processing** for complex queries
- **Ihsān-grade quality** via SNR optimization

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BIZRA DATA LAKE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │00_INTAKE│───▶│ 01_RAW  │───▶│02_PROC. │───▶│03_INDEX │         │
│  │ (Drop)  │    │(Backup) │    │(Sorted) │    │(Vectors)│         │
│  └─────────┘    └─────────┘    └─────────┘    └────┬────┘         │
│                                                     │               │
│                                                     ▼               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     PROCESSING LAYER                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │ Corpus   │  │ Vector   │  │Hypergraph│  │ LangExtract  │  │  │
│  │  │ Manager  │  │ Engine   │  │ Engine   │  │   Engine     │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘  │  │
│  └───────┼─────────────┼─────────────┼──────────────────────────┘  │
│          │             │             │                              │
│          ▼             ▼             ▼                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     REASONING LAYER                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │  ARTE    │  │   SNR    │  │   KEP    │  │     PAT      │  │  │
│  │  │  Engine  │  │Optimizer │  │  Bridge  │  │   Engine     │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘  │  │
│  └───────┼─────────────┼─────────────┼──────────────────────────┘  │
│          │             │             │                              │
│          ▼             ▼             ▼                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   CONSCIOUSNESS LAYER                         │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │                   DDAGI System                           │ │  │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │ │  │
│  │  │  │Merkle-DAG  │  │Consciousness│  │  POI Attestation   │  │ │  │
│  │  │  │Verification│  │   Events   │  │      Ledger        │  │ │  │
│  │  │  └────────────┘  └────────────┘  └────────────────────┘  │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────┐                                                        │
│  │04_GOLD  │ ◀── Curated High-Value Data                           │
│  │(Parquet)│                                                        │
│  └─────────┘                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Metrics

| Metric | Current Value |
|--------|--------------|
| Total Nodes | 56,358 |
| Total Edges | 88,649 |
| Embedded Chunks | 84,795 |
| Embedding Dimension | 384 |
| POI Attestations | 21 |
| DDAGI Events | 15 |

---

## 2. Core Architecture

### 2.1 Directory Structure

```
C:\BIZRA-DATA-LAKE\
├── 00_INTAKE/              # Drop zone for new files
├── 01_RAW/                 # Immutable timestamped backups
├── 02_PROCESSED/           # Organized by type
│   ├── images/
│   ├── documents/
│   ├── code/
│   ├── text/
│   ├── data/
│   ├── models/
│   ├── media/
│   └── archives/
├── 03_INDEXED/             # Vector embeddings + metadata
│   ├── graph/              # Hypergraph structures
│   ├── embeddings/         # FAISS indices
│   ├── chat_history/       # Conversation graphs
│   ├── metrics/            # Operational metrics
│   └── validation/         # Validation reports
├── 04_GOLD/                # Curated Parquet tables
│   ├── documents.parquet
│   ├── chunks.parquet
│   └── poi_ledger.jsonl
└── 99_QUARANTINE/          # Duplicates, corrupted files
```

### 2.2 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Configuration | `bizra_config.py` | Central configuration |
| Corpus Manager | `corpus_manager.py` | Document parsing |
| Vector Engine | `vector_engine.py` | Embedding generation |
| ARTE Engine | `arte_engine.py` | SNR calculation, GoT |
| Hypergraph Engine | `tools/engines/hypergraph_engine.py` | Graph operations |
| PAT Engine | `tools/pat_engine.py` | Multi-agent coordination |
| KEP Bridge | `tools/bridges/kep_bridge.py` | Cross-domain synergies |
| **WARP Bridge** | `tools/bridges/warp_bridge.py` | **Multi-vector ColBERT retrieval** |
| Orchestrator | `bizra_orchestrator.py` | Unified query routing |
| Nexus | `tools/bizra_nexus.py` | Unified engine orchestration |

### 2.3 Retrieval Engines

| Engine | Type | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| **FAISS HNSW** | Single-vector (384-dim) | ⚡ Fast | ★★★☆ | High-throughput queries |
| **XTR-WARP** | Multi-vector (ColBERT) | ⚡ Fast | ★★★★ | High-accuracy semantic search |
| **Graph Traversal** | Structural | ⏱️ Medium | ★★★☆ | Relationship queries |
| **Hybrid** | Combined | ⏱️ Medium | ★★★★ | Complex reasoning |

---

## 3. Data Pipeline

### 3.1 Ingestion Flow

```
Files → INTAKE → Hash Check → Type Sort → PROCESSED → Parse → INDEXED
                     │
                     └──▶ QUARANTINE (duplicates)
```

### 3.2 Processing Stages

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1. Intake | DataLakeProcessor.ps1 | Raw files | Sorted files |
| 2. Parse | corpus_manager.py | Documents | documents.parquet |
| 3. Chunk | vector_engine.py | Documents | chunks.parquet |
| 4. Embed | vector_engine.py | Chunks | 384-dim vectors |
| 5. Index | hypergraph_engine.py | Embeddings | FAISS HNSW |
| 6. Graph | hypergraph_engine.py | Chunks | NetworkX graph |

### 3.3 Deduplication

SHA-256 hash-based deduplication at intake:

```python
def calculate_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
```

---

## 4. SAPE Framework

### 4.1 Overview

SAPE (Security, Architecture, Performance, Engineering) provides the analytical framework for system evaluation and optimization.

### 4.2 SNR Calculation

Signal-to-Noise Ratio using weighted geometric mean:

```python
SNR = exp(Σ wᵢ × log(componentᵢ))

Components:
- signal_strength:      0.35  # Retrieval relevance
- information_density:  0.25  # Content richness
- symbolic_grounding:   0.25  # Graph connectivity
- coverage_balance:     0.15  # Query coverage
```

### 4.3 Ihsan Threshold

Excellence gate. Production threshold is 0.95 (defined in `core/integration/constants.py`):

```python
IHSAN_CONSTRAINT = 0.95  # Production (UNIFIED_IHSAN_THRESHOLD)
STRICT_THRESHOLD = 0.99  # Strict/consensus mode

class IhsanGate:
    @classmethod
    def validate(cls, snr: float) -> IhsanResult:
        if snr >= IHSAN_CONSTRAINT:
            return IhsanResult(status="IHSAN_ACHIEVED")
        elif snr >= ACCEPTABLE_THRESHOLD:
            return IhsanResult(status="ACCEPTABLE", optimize=True)
        else:
            return IhsanResult(status="BELOW_STANDARD", proceed=False)
```

### 4.4 Graph-of-Thoughts

Multi-phase reasoning with typed thoughts:

```python
class ThoughtType(Enum):
    HYPOTHESIS = "hypothesis"      # Initial conjecture
    EVIDENCE = "evidence"          # Supporting data
    CONTRADICTION = "contradiction" # Conflicting info
    SYNTHESIS = "synthesis"        # Combined insight
    REFINEMENT = "refinement"      # Iterative improvement
    CONCLUSION = "conclusion"      # Final determination
```

---

## 5. DDAGI Consciousness System

### 5.1 Overview

The Distributed Decentralized Artificial General Intelligence (DDAGI) system provides:

- **Self-awareness** through consciousness event logging
- **Cryptographic verification** via Merkle-DAG
- **Knowledge attestation** through POI ledger

### 5.2 Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    DDAGI CONSCIOUSNESS                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Consciousness│    │  Merkle-DAG  │    │     POI      │  │
│  │    Events     │◀──▶│  Verification│◀──▶│   Ledger     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Cryptographic Attestation               │  │
│  │  Ed25519 Signatures + SHA-256 Content Hashes         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 5.3 Consciousness Events

Stored in `03_INDEXED/ddagi_consciousness.jsonl`:

```json
{
  "event_id": "consciousness_001",
  "event_type": "reflection",
  "timestamp": "2026-01-22T10:30:00Z",
  "snr": 1.0,
  "context": {
    "query": "What patterns exist across domains?",
    "insights_generated": 5,
    "synthesis_depth": 3
  },
  "merkle_hash": "sha256:abc123..."
}
```

### 5.4 Event Types

| Event | Description | Trigger |
|-------|-------------|---------|
| REFLECTION | Self-assessment | Periodic, post-query |
| LEARNING | Knowledge integration | Novel pattern detection |
| ADAPTATION | Behavioral adjustment | Performance degradation |
| SYNTHESIS | Cross-domain insight | Multi-hop completion |
| ATTESTATION | Cryptographic verification | POI checkpoint |

### 5.5 POI Ledger

Proof-of-Impact attestation in `04_GOLD/poi_ledger.jsonl`:

```json
{
  "entry_id": "poi_001",
  "type": "synthesis",
  "timestamp": "2026-01-22T10:30:00Z",
  "content_hash": "sha256:abc123...",
  "parent_hashes": ["sha256:def456..."],
  "genesis_merkle_root": "sha256:ghi789...",
  "attestation_hash": "ed25519:jkl012...",
  "snr_at_creation": 0.97,
  "ihsan_compliant": true
}
```

---

## 6. Hypergraph RAG

### 6.1 Overview

Hypergraph Retrieval-Augmented Generation combines:

- **FAISS HNSW** for vector similarity (M=32, efConstruction=200)
- **NetworkX MultiDiGraph** for structural relationships
- **Multi-hop reasoning** for complex queries

### 6.2 Retrieval Modes

```python
class RetrievalMode(Enum):
    SEMANTIC = "semantic"       # Vector similarity only
    STRUCTURAL = "structural"   # Graph traversal only
    HYBRID = "hybrid"          # Combined (default)
    MULTI_HOP = "multi_hop"    # N-hop graph expansion
```

### 6.3 Graph Structure

```
Node Types:
- document: Full document
- chunk: Text chunk
- concept: Extracted concept
- entity: Named entity

Edge Types:
- contains: Document → Chunk
- references: Chunk → Concept
- similar_to: Chunk ↔ Chunk
- synergy: Concept ↔ Concept (via KEP)
```

### 6.4 FAISS Configuration

```python
FAISS_CONFIG = {
    "index_type": "HNSW",
    "M": 32,                    # Connections per node
    "efConstruction": 200,      # Build-time accuracy
    "efSearch": 64,             # Query-time accuracy
    "dimension": 384,           # MiniLM embedding dim
    "metric": "L2"              # Distance metric
}
```

---

## 7. PAT Engine

### 7.1 Overview

Personal Agentic Team (PAT) provides multi-agent coordination:

### 7.2 Agent Roles

| Agent | Role | ThinkingMode |
|-------|------|--------------|
| Strategist | Goal decomposition | SYNTHESIS |
| Researcher | Information gathering | DEEP |
| Analyst | Pattern recognition | CRITICAL |
| Creator | Content generation | CREATIVE |
| Guardian | Quality assurance | CRITICAL |
| Coordinator | Orchestration | FAST |

### 7.3 Backend Support

```python
class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, messages, model, temperature, max_tokens) -> str:
        pass

# Implementations:
- OllamaBackend: Local inference (llama3.2, mistral, etc.)
- OpenAIBackend: OpenAI API compatible
- LMStudioBackend: Local LM Studio server
```

### 7.4 Task Processing Flow

```
1. Strategic Decomposition (Strategist)
2. Agent Selection (based on task keywords)
3. Parallel Agent Processing
4. Coordinator Synthesis
5. Guardian Validation
6. SNR Calculation & Ihsān Gate
```

---

## 8. Resilience Patterns

### 8.1 Circuit Breaker

Prevents cascade failures:

```python
class CircuitBreaker:
    States: CLOSED → OPEN → HALF_OPEN → CLOSED

    Configuration:
    - failure_threshold: 3-5 failures
    - success_threshold: 2-3 successes
    - timeout_seconds: 30-60s
```

### 8.2 Retry Logic

Exponential backoff with jitter:

```python
@retry(max_retries=3, base_delay=1.0, exponential_base=2.0, jitter=True)
async def flaky_operation():
    ...
```

### 8.3 Graceful Degradation

Fallback patterns:

```python
@with_fallback(fallback_value="Cached response")
async def get_llm_response():
    # If LLM fails, return cached response
    ...
```

---

## 9. Monitoring & Observability

### 9.1 Metrics Dashboard

Real-time monitoring via `metrics_dashboard.py`:

```
Metrics Collected:
- SNR values and statistics
- Latency (p50, p95, p99)
- Error rates
- Ihsān compliance rate
- Circuit breaker states
```

### 9.2 Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| SNR | < 0.99 | < 0.95 |
| Latency (p95) | > 3000ms | > 5000ms |
| Error Rate | > 1% | > 5% |
| Ihsān Compliance | < 95% | < 80% |

### 9.3 Self-Healing

Automated recovery via `self_healing.py`:

```python
class HealingAgent:
    critical_files = [
        "04_GOLD/documents.parquet",
        "03_INDEXED/chat_history/graph.json",
        "bizra_config.py"
    ]

    def attempt_repair(self, issues):
        if "documents.parquet" in issues:
            subprocess.run(["python", "corpus_manager.py"])
```

---

## 10. API Reference

### 10.1 BIZRAOrchestrator

```python
from bizra_orchestrator import BIZRAOrchestrator, BIZRAQuery

orchestrator = BIZRAOrchestrator()
await orchestrator.initialize()

response = await orchestrator.query(BIZRAQuery(
    text="Find documents about machine learning",
    complexity=QueryComplexity.RESEARCH,
    enable_hypergraph=True
))

print(response.synthesis)
print(f"SNR: {response.snr_score}")
```

### 10.2 SNREngine

```python
from arte_engine import SNREngine

engine = SNREngine()
result = engine.calculate_snr(
    query_embedding=np.array([...]),
    context_embeddings=[np.array([...]), ...],
    symbolic_facts=["fact1", "fact2"],
    neural_results=[{"text": "...", "score": 0.9}]
)

print(f"SNR: {result['snr']}")
print(f"Ihsān: {result['ihsan_achieved']}")
```

### 10.3 MetricsDashboard

```python
from metrics_dashboard import MetricsDashboard, record_snr

dashboard = MetricsDashboard()
dashboard.print_dashboard()

# Record metrics
record_snr(overall=0.97, signal=0.95, density=0.92,
           grounding=0.94, balance=0.91)
```

### 10.4 Resilience

```python
from tools.engines.bizra_resilience import CircuitBreaker, retry, with_fallback

breaker = CircuitBreaker("my_service")

@breaker
@retry(max_retries=3)
@with_fallback(fallback_value="default")
async def protected_operation():
    ...
```

---

## Appendices

### A. Configuration Reference

See `bizra_config.py` for all configurable parameters.

### B. File Formats

| File | Format | Schema |
|------|--------|--------|
| documents.parquet | Parquet | id, title, content, source, hash |
| chunks.parquet | Parquet | chunk_id, doc_id, text, embedding |
| poi_ledger.jsonl | JSONL | See POI Ledger section |
| graph.json | JSON | NetworkX JSON format |

### C. Dependencies

```
sentence-transformers>=2.2.0
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
faiss-cpu>=1.7.0 (or faiss-gpu)
networkx>=3.0
httpx>=0.24.0
unstructured>=0.10.0
```

---

*BIZRA Data Lake Architecture v2.1*
*Updated: 2026-02-14*
*Ihsan Compliance Target: >= 0.95 (production), >= 0.99 (strict)*
