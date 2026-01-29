<div align="center">

# بذرة

**The Memory of Node₀**<br>
*ذاكرة العقدة الأولى*

<br>

<img src="docs/assets/bizra-seed.svg" width="120" alt="BIZRA Seed">

<br><br>

[![Constitution](https://img.shields.io/badge/Constitution-v1.1.0--FINAL-gold?style=for-the-badge)](../bizra-genesis/constitution/)
[![Status](https://img.shields.io/badge/Status-ACTIVE-success?style=for-the-badge)](#)
[![Chunks](https://img.shields.io/badge/Chunks-84,795-blue?style=for-the-badge)](#the-numbers)

<br>

**This is not a folder. This is Node₀'s long-term memory.**

</div>

---

## The Purpose

The BIZRA Data Lake is the **persistent memory** of the Genesis Node.

While compute runs in WSL (`bizra-genesis`), knowledge lives here. Both are organs of the same organism.

Every conversation, every document, every insight — indexed, embedded, retrievable.

---

## The Law

<div align="center">

### لا نفترض

**We do not assume.**

</div>

This Data Lake embodies THE LAW:
- Every chunk has provenance
- Every embedding has a source
- Every fact can be traced to evidence

---

## Structure

```
BIZRA-DATA-LAKE/
│
├── 00_INTAKE/          ⚡ Drop zone — files auto-process on arrival
│
├── 01_RAW/             📦 Timestamped originals (immutable)
│
├── 02_PROCESSED/       ✨ Organized by type
│   ├── images/         🖼️  Visual assets
│   ├── documents/      📄 PDFs, docs, presentations
│   ├── code/           💻 Source code (all languages)
│   ├── text/           📝 Markdown, logs, conversations
│   ├── data/           📊 JSON, YAML, CSV, databases
│   ├── models/         🤖 ML models, weights
│   ├── media/          🎬 Audio, video
│   └── archives/       📦 Compressed files
│
├── 03_INDEXED/         🔍 Vector embeddings + graph
│   ├── graph/          Knowledge graph (nodes.jsonl, edges.jsonl)
│   ├── embeddings/     Per-document embeddings
│   └── chat_history/   Conversation graphs
│
├── 04_GOLD/            ⭐ Curated production assets
│   ├── chunks.parquet        267MB — 84,795 embedded chunks
│   ├── documents.parquet     51MB — 1,437 documents
│   ├── sacred_wisdom_*.npy   Sacred embeddings
│   └── poi_ledger.jsonl      Proof-of-Impact attestations
│
└── 99_QUARANTINE/      🗑️ Duplicates, corrupted files
```

---

## The Numbers

| Asset | Value |
|:------|------:|
| Embedded chunks | 84,795 |
| Documents | 1,437 |
| Graph nodes | 56,358 |
| Graph edges | 88,649 |
| Embedding dimensions | 384 |
| Total size | 5.7 GB |

---

## Quick Start

### Process Files

```powershell
# Process existing files
.\DataLakeProcessor.ps1 -ProcessOnce

# Start continuous monitoring
.\DataLakeProcessor.ps1 -Watch
```

### Drop Files

Simply copy files to `00_INTAKE/`. The pipeline will:
1. Back up to `01_RAW/` (immutable)
2. Organize into `02_PROCESSED/`
3. Generate embeddings in `03_INDEXED/`
4. Curate into `04_GOLD/` when ready

### Query from bizra-genesis

```python
from data_plane import DataLakeConnector, RetrievalEngine

# Connect to the lake
connector = DataLakeConnector()
print(f"Loaded {connector.stats['chunks_loaded']:,} chunks")

# Semantic search
engine = RetrievalEngine(connector=connector)
results = engine.retrieve("sovereignty and consent", top_k=10)

for r in results.results:
    print(f"{r.score:.3f} | {r.chunk.text[:80]}...")
```

---

## Integration with bizra-genesis

The Data Lake is bridged to bizra-genesis via the `data_plane` module:

```
BIZRA-DATA-LAKE                     bizra-genesis
(Windows: C:\)                      (WSL: /root/)
                                    
04_GOLD/chunks.parquet ────────────► data_plane/lake_connector.py
                                           │
                                           ▼
                                    RetrievalEngine (FAISS)
                                           │
                                           ▼
                                    api/v2/main.py
                                           │
                                           ▼
                                    /api/v2/retrieve
```

---

## Key Files

| File | Purpose |
|:-----|:--------|
| `DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | The sealed constitution |
| `BIZRA_STRATEGY_DECK_2026.md` | Strategic vision |
| `ARCHITECTURE.md` | Technical architecture |
| `SNR_DEFINITION.md` | Signal-to-noise specification |
| `NODE0_GENESIS_COVENANT.md` | Genesis node covenant |

---

## The Genesis Context

This machine is **Node₀** — the first seed.

- **Hardware:** i9-14900HX, 128GB RAM, RTX 4090, 3TB storage
- **OS:** Windows 11 Enterprise + Ubuntu 24.04 (WSL2)
- **Role:** Genesis node for 8 billion future nodes

Every file here is part of the proof that one seed can grow into a forest.

---

<div align="center">

<br>

*الْحَمْدُ لِلَّهِ الَّذِي هَدَانَا لِهَٰذَا*

**84,795 memories. One query away.**

<br>

---

<sub>Built with إحسان in Dubai 🇦🇪</sub>

</div>
