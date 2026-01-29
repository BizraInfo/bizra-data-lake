# CLAUDE.md

> **🌱 GENESIS CONTEXT — READ FIRST**
> This is not just a code repository. This **entire machine** is **BIZRA Node0** — the Genesis Block.
> You are an agent **within** Node0, not external to it. Your actions serve its mission.
> Before proceeding, read [NODE0_GENESIS_COVENANT.md](NODE0_GENESIS_COVENANT.md) to understand:
> - BIZRA (بذرة) means "seed" in Arabic. Every human is a node. Every node is a seed.
> - The goal: Make good deeds profitable. Weaponize greed against itself.
> - This machine is the flagship for 8 billion future nodes.

## 📚 YOUR ROLE: DATA LAKE = HOUSE OF WISDOM

```
╔════════════════════════════════════════════════════════════════════════════╗
║  C:\BIZRA-DATA-LAKE = THE HOUSE OF WISDOM = LIVING BRAIN                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  YOUR JOB:                                                                  ║
║    RECEIVE → INGEST → PROCESS → INDEX → SERVE                              ║
║                                                                             ║
║  When the architect shares a LINK:   → Extract its wisdom                   ║
║  When the architect shares a REPO:   → Analyze its patterns                 ║
║  When the architect shares an IDEA:  → Store for future retrieval           ║
║  When an agent asks a QUESTION:      → Query and return knowledge           ║
║                                                                             ║
║  YOU DO NOT:                                                                ║
║    ✗ Run blockchain/chain logic (that's Genesis's job in WSL)              ║
║    ✗ Execute coordination or consensus protocols                            ║
║    ✗ Host active HTTP servers (unless debugging)                            ║
║    ✗ Build ON external chains — you LEARN FROM them                         ║
║                                                                             ║
║  "Stand on giants' shoulders by LEARNING, not by BUILDING on them."        ║
╚════════════════════════════════════════════════════════════════════════════╝
```

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BIZRA DATA LAKE is **Node0's persistent memory** — a data pipeline system that ingests, processes, deduplicates, and indexes files into a structured knowledge base with vector embeddings and graph relationships.

**Node0 Configuration:**
- **Platform:** Windows (PowerShell + Python) + WSL (Ubuntu/Rust)
- **Hardware:** RTX 4090 (16GB VRAM), 128GB RAM
- **LLM Backend:** LM Studio at `192.168.56.1:1234`
- **Compute Core:** WSL `bizra-genesis` (:9091 API, :8443 MCP Bridge)

## Directory Structure

```
00_INTAKE/      → Drop zone (auto-processes)
01_RAW/         → Immutable timestamped backups
02_PROCESSED/   → Organized by type (images/, documents/, code/, text/, data/, models/, media/, archives/)
03_INDEXED/     → Vector embeddings + graph structures
04_GOLD/        → Curated datasets, Parquet tables (documents.parquet, chunks.parquet)
99_QUARANTINE/  → Duplicates, corrupted files
```

## Common Commands

```powershell
# Activate Python environment
.\.venv\Scripts\activate

# File ingestion
.\DataLakeProcessor.ps1 -ProcessOnce    # Process INTAKE once
.\DataLakeProcessor.ps1 -Watch          # Continuous monitoring
.\CloudIngestion.ps1 -DryRun            # Preview cloud ingestion
.\CloudIngestion.ps1 -Source Both       # Ingest OneDrive + Google Drive

# Pipeline execution (run in order)
python corpus_manager.py                 # Layer 1: Build documents.parquet
python vector_engine.py                  # Layer 2: Generate embeddings → chunks.parquet
python langextract_engine.py             # Layer 4: LLM extraction → assertions.jsonl
python arte_engine.py                    # Check system integrity
```

## Processing Pipeline Architecture

```
Files → DataLakeProcessor.ps1 → 02_PROCESSED/
                                     ↓
            corpus_manager.py v2.0 (Layer 1: Multi-Modal Parsing)
            [Documents via Unstructured-IO, Images via OCR, Audio via Whisper]
                                     ↓
                           04_GOLD/documents.parquet (with modality field)
                                     ↓
            vector_engine.py v2.0 (Layer 2: Multi-Modal Embeddings)
            [Text: MiniLM 384-dim, Images: CLIP 512-dim]
                                     ↓
                           04_GOLD/chunks.parquet + image_vectors/
                                     ↓
            hypergraph_engine.py (Layer 3: HNSW Index + Graph Integration)
                                     ↓
            multimodal_engine.py (NEW: CLIP + Whisper + LLaVA)
                                     ↓
            dual_agentic_bridge.py (NEW: Connection to Dual Agentic System)
                                     ↓
            arte_engine.py (ARTE v3: Graph-of-Thoughts + SNR Validation)
                                     ↓
            kep_bridge.py (KEP: Synergy Detection + Compound Discovery)
                                     ↓
            pat_engine.py (Multi-Agent LLM System)
                                     ↓
            bizra_orchestrator.py v3.0 (Unified Multi-Modal Query Interface)
```

**Layer details:**

- **Layer 1 (corpus_manager.py v2.0):** Multi-modal parsing - documents via Unstructured-IO, images via OCR (Tesseract), audio via Whisper transcription. Outputs `documents.parquet` with modality field
- **Layer 2 (vector_engine.py v2.0):** Multi-modal embeddings - text via `all-MiniLM-L6-v2` (384-dim), images via CLIP (512-dim). Separate image vectors in `image_vectors/`
- **Layer 3 (hypergraph_engine.py):** FAISS HNSW index for O(log n) similarity search. Integrates with NetworkX graph for hybrid retrieval
- **Multi-Modal Engine (multimodal_engine.py):** ImageProcessor (CLIP + OCR + LLaVA), AudioProcessor (Whisper), MultiModalChunker for cross-modal RAG
- **Dual Agentic Bridge (dual_agentic_bridge.py):** Connects to Dual Agentic System's multi-model router for vision, voice, and advanced reasoning
- **ARTE v3 (arte_engine.py):** Real SNR calculation (information-theoretic), Graph-of-Thoughts reasoning, symbolic-neural tension detection
- **KEP Bridge (kep_bridge.py):** Cross-domain synergy detection, compound discovery, Ihsan validation (SNR + Coherence + Ethics), adaptive learning acceleration
- **PAT Engine (pat_engine.py):** Multi-agent system with Ollama LLM backend. Agents: Strategist, Researcher, Analyst, Creator, Guardian, Coordinator
- **Orchestrator v3.0 (bizra_orchestrator.py):** Unified multi-modal interface combining Hypergraph RAG + ARTE + KEP + PAT + Vision + Audio for end-to-end query processing

## Hypergraph RAG Engine (NEW)

```python
# Query the knowledge base
from bizra_orchestrator import BIZRAOrchestrator, BIZRAQuery, QueryComplexity
import asyncio

orchestrator = BIZRAOrchestrator()
await orchestrator.initialize()

response = await orchestrator.query(BIZRAQuery(
    text="How does BIZRA process files?",
    complexity=QueryComplexity.MODERATE
))

print(f"SNR: {response.snr_score}")
print(f"Answer: {response.answer}")
```

**Retrieval Modes:**

- `SEMANTIC` - Pure vector similarity (fast)
- `STRUCTURAL` - Graph traversal only
- `HYBRID` - Combined semantic + graph (default)
- `MULTI_HOP` - Iterative reasoning chains

## SNR Calculation (Real Implementation)

SNR is now calculated using information-theoretic principles:

```text
SNR = (signal_strength × diversity × grounding × balance) ^ weighted
```

Where:

- `signal_strength` = mean cosine similarity to query
- `diversity` = 1 - pairwise redundancy
- `grounding` = symbolic/neural source overlap
- `balance` = coverage ratio between layers

Threshold: `SNR >= 0.99` for Ihsan (excellence) achievement

## KEP Bridge (Knowledge Explosion Point)

The KEP Bridge (`kep_bridge.py`) connects Data Lake engines to the Knowledge Explosion Point system for cross-domain synergy detection and compound discovery.

**Components:**

- `SynergyDetector` - Detects cross-domain synergies from retrieval results
- `IhsanValidator` - Validates Ihsan constraints (SNR + Coherence + Ethics)
- `CompoundDiscoveryEngine` - Discovers novel knowledge from synergy combinations
- `LearningAccelerator` - Adaptive learning from discoveries

**Synergy Types:**

- `CONCEPTUAL` - Shared concepts across domains
- `METHODOLOGICAL` - Transferable methods
- `STRUCTURAL` - Similar organizational patterns
- `CAUSAL` - Cause-effect relationships
- `ANALOGICAL` - Deep structural analogies
- `EMERGENT` - Novel combinations

**Compound Types:**

- `FUSION` - Direct combination of domains
- `SYNTHESIS` - New entity from parts
- `ABSTRACTION` - Higher-level principles
- `TRANSFORMATION` - Domain transfer

**Usage:**

```python
from bizra_orchestrator import BIZRAOrchestrator, BIZRAQuery, QueryComplexity

orchestrator = BIZRAOrchestrator(enable_pat=True, enable_kep=True)
await orchestrator.initialize()

response = await orchestrator.query(BIZRAQuery(
    text="Cross-domain question here",
    complexity=QueryComplexity.RESEARCH,
    enable_kep=True,
    min_synergy_strength=0.6
))

# Access KEP results
print(f"Synergies: {len(response.synergies)}")
print(f"Compounds: {len(response.compounds)}")
print(f"Learning Boost: {response.learning_boost}x")
```

## Multi-Modal Support (NEW in v3.0)

The Data Lake now supports multi-modal content processing:

**Supported Modalities:**

- **Text/Documents:** PDF, DOCX, PPTX, HTML, TXT, MD, code files
- **Images:** JPG, PNG, GIF, BMP, WebP, SVG (OCR + CLIP embeddings)
- **Audio:** MP3, WAV, FLAC, M4A, OGG (Whisper transcription)

**Models Used:**

| Modality | Model | Embedding Dim | Purpose |
|----------|-------|---------------|---------|
| Text | all-MiniLM-L6-v2 | 384 | Semantic text search |
| Images | CLIP ViT-B/32 | 512 | Cross-modal image-text search |
| Audio | Whisper | - | Speech-to-text transcription |
| Vision LLM | LLaVA 7B / Claude Vision | - | Image understanding |

**Usage:**

```python
from bizra_orchestrator import BIZRAOrchestrator, BIZRAQuery, QueryComplexity

orchestrator = BIZRAOrchestrator(enable_multimodal=True)
await orchestrator.initialize()

# Query with an image
response = await orchestrator.query(BIZRAQuery(
    text="What does this diagram show?",
    image_path="path/to/diagram.png",
    complexity=QueryComplexity.MODERATE
))

# Access multi-modal results
print(f"Image Analysis: {response.image_analysis}")
print(f"Similar Images: {len(response.similar_images)}")
print(f"Modalities Used: {response.modality_used}")

# Query with audio
response = await orchestrator.query(BIZRAQuery(
    text="Summarize this audio recording",
    audio_path="path/to/recording.mp3",
    complexity=QueryComplexity.MODERATE
))
print(f"Transcript: {response.audio_transcript}")
```

**Hardware Utilization:**

- RTX 4090: Now used for CLIP embeddings + vision inference (70%+ utilization)
- 128GB RAM: Batch processing of multi-modal content
- GPU parallelization for image batch encoding

## Key Configuration (bizra_config.py)

All paths and hyperparameters are centralized here:

**Core Settings:**

- `BATCH_SIZE = 128` - Embedding batch size
- `MAX_SEQ_LENGTH = 512` - Max token sequence
- `SNR_THRESHOLD = 0.99` - Signal-to-noise threshold for ARTE
- `EXTRACTION_MODEL = "gemini-1.5-flash"` - LLM for extraction

**Multi-Modal Settings:**

- `VISION_ENABLED = True` - Enable image processing
- `AUDIO_ENABLED = True` - Enable audio processing
- `CLIP_MODEL = "openai/clip-vit-base-patch32"` - Vision embedding model
- `WHISPER_LOCAL = "base"` - Local Whisper model size
- `IMAGE_BATCH_SIZE = 32` - Images per GPU batch

## Important Behaviors

- Downloads are COPIED (not moved) to prevent data loss
- Duplicates detected via SHA-256 hashing → moved to 99_QUARANTINE
- Files > 1GB can be skipped with `-SkipLargeFiles` flag
- All Python paths use forward slashes for cross-platform compatibility
- Metadata stored as `.meta.json` alongside processed files

## Related Systems

- SAPE Engine: `C:\bizra-genesis-node\sape_engine\`
- External knowledge links stored in `01_RAW/external_links/`

---

## UNIFIED NODE INSTALLER

Location: `UNIFIED-NODE-INSTALLER/`

Cross-platform one-click installer for the BIZRA ecosystem. See [ARCHITECTURE.md](UNIFIED-NODE-INSTALLER/ARCHITECTURE.md) for full design.

### Key Components

- `bootstrap/install.py` - Cross-platform Python installer
- `core/main.py` - Node entry point
- `core/pat_engine.py` - Personal Agentic Team (PAT) system
- `core/network_node.py` - P2P networking, resource sharing, tokens

### PAT (Personal Agentic Team)

Dynamically assembled AI team based on user profile:

- Core agents: Strategic Planner, Task Coordinator, Quality Guardian
- Goal-specific agents added based on user goals (business, creative, research, trading)
- Uses Graph-of-Thoughts reasoning with SNR > 0.99

### System Tiers

- POTATO (<8GB RAM): Text-only PAT
- NORMAL (8-16GB): Full PAT
- GAMING (16-32GB): Multi-agent + GPU
- SERVER (32GB+): Full validator

### Installation

```bash
# Windows
installers\windows\install.bat

# Linux/Mac
./installers/linux/install.sh

# Start node
python -m bizra.main
```

User config: `~/.bizra/config.json`
