# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**This is the BIZRA-DATA-LAKE repository** — the persistent memory and knowledge layer of the BIZRA ecosystem. This repo handles data ingestion, processing, deduplication, and indexing into a knowledge base with vector embeddings and graph relationships.

> **Context:** This repo is one component of BIZRA Node0. Other components (bizra-genesis-node, BIZRA-OS, etc.) live in separate repositories. See [NODE0_GENESIS_COVENANT.md](NODE0_GENESIS_COVENANT.md) for ecosystem context.

**Environment:** Windows (PowerShell + Python) + WSL (Ubuntu/Rust) | RTX 4090 (16GB VRAM), 128GB RAM | LM Studio at `192.168.56.1:1234`

## Common Commands

```powershell
# Activate Python environment
.\.venv\Scripts\activate

# File ingestion
.\DataLakeProcessor.ps1 -ProcessOnce    # Process INTAKE once
.\DataLakeProcessor.ps1 -Watch          # Continuous monitoring

# Pipeline execution (run in order)
python corpus_manager.py                 # Layer 1: Build documents.parquet
python vector_engine.py                  # Layer 2: Generate embeddings → chunks.parquet
python langextract_engine.py             # Layer 4: LLM extraction → assertions.jsonl
python arte_engine.py                    # Check system integrity

# Testing
pytest                                   # Run all tests
pytest tests/test_snr_engine.py          # Run single test file
pytest -m "not slow"                     # Skip slow tests
pytest -m "not requires_ollama"          # Skip tests needing Ollama

# Linting/Formatting
black .                                  # Format code
isort .                                  # Sort imports
mypy .                                   # Type checking
```

## Directory Structure

```
00_INTAKE/      → Drop zone (auto-processes)
01_RAW/         → Immutable timestamped backups
02_PROCESSED/   → Organized by type (images/, documents/, code/, text/, data/)
03_INDEXED/     → Vector embeddings + graph structures
04_GOLD/        → Curated datasets (documents.parquet, chunks.parquet)
99_QUARANTINE/  → Duplicates, corrupted files
core/           → Sovereignty infrastructure modules
tests/          → Pytest test suite
```

## Core Module Architecture

The `core/` package provides the sovereignty infrastructure:

```
core/
├── pci/          # Proof-Carrying Inference Protocol
│   ├── envelope.py    # PCI message envelopes with signatures
│   ├── crypto.py      # Ed25519 signing/verification
│   ├── gates.py       # Inference gate constraints
│   └── epigenome.py   # Runtime context adaptation
├── vault/        # Encryption at Rest
│   └── vault.py       # Secure credential storage
├── federation/   # P2P Network Layer
│   ├── gossip.py      # Gossip protocol for node discovery
│   ├── consensus.py   # Byzantine fault-tolerant consensus
│   ├── propagation.py # Pattern sharing across nodes
│   └── node.py        # Federation node implementation
├── inference/    # Embedded LLM Gateway
│   ├── gateway.py     # Core inference with tiered backends
│   ├── selector.py    # Adaptive model selection by task complexity
│   ├── unified.py     # Complete inference system
│   └── backends/      # LlamaCpp, Ollama backends
└── a2a/          # Agent-to-Agent Protocol
    ├── schema.py      # AgentCard, TaskCard definitions
    ├── engine.py      # A2A message routing
    └── transport.py   # P2P message transport
```

**Inference Tiers:**
- EDGE/NANO: Always-on, low-power (0.5B-1.5B models)
- LOCAL/MEDIUM: On-demand, high-power (7B models, RTX 4090)
- POOL/LARGE: Federated URP compute (70B+ models)

## Processing Pipeline

```
Files → DataLakeProcessor.ps1 → 02_PROCESSED/
                                     ↓
            corpus_manager.py (Layer 1: Multi-Modal Parsing)
                                     ↓
                           04_GOLD/documents.parquet
                                     ↓
            vector_engine.py (Layer 2: Embeddings)
            [Text: MiniLM 384-dim, Images: CLIP 512-dim]
                                     ↓
                           04_GOLD/chunks.parquet
                                     ↓
            arte_engine.py (ARTE: SNR Validation)
                                     ↓
            bizra_orchestrator.py (Unified Query Interface)
```

## Key Configuration (bizra_config.py)

All paths and hyperparameters are centralized here. Key settings:

- `BATCH_SIZE = 128` — Embedding batch size
- `SNR_THRESHOLD = 0.85` — Minimum signal quality
- `IHSAN_CONSTRAINT = 0.95` — Excellence target
- Paths resolve automatically for Windows/WSL/Linux via `BIZRA_DATA_LAKE_ROOT` env var

**LLM Backends (in priority order):**
1. LM Studio: `192.168.56.1:1234` (primary)
2. Ollama: `localhost:11434` (fallback)

## Test Markers

```python
@pytest.mark.slow           # Long-running tests
@pytest.mark.integration    # Integration tests
@pytest.mark.requires_ollama # Needs Ollama running
@pytest.mark.requires_gpu   # Needs GPU
```

## SNR Calculation

Signal-to-Noise Ratio enforces quality constraints:

```
SNR = (signal_strength × diversity × grounding × balance) ^ weighted
```

Threshold: `SNR >= 0.85` minimum, `>= 0.95` for Ihsan (excellence)

## Important Behaviors

- Downloads are COPIED (not moved) to prevent data loss
- Duplicates detected via SHA-256 → moved to 99_QUARANTINE
- All Python paths use forward slashes for cross-platform compatibility
- Metadata stored as `.meta.json` alongside processed files
