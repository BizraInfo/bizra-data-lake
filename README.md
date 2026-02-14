<div align="center">

# BIZRA

**Sovereign Agentic Infrastructure for the Next Internet**

<br>

<img src="docs/assets/bizra-seed.svg" width="120" alt="BIZRA Seed">

<br><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-stable-DEA584?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/Tests-CI_Verified-success?style=for-the-badge)](#testing)

<br>

BIZRA is a decentralized agentic system where every human is a node, every node is sovereign, and every action is ethically constrained. Built on **Proof-Carrying Inference**, **FATE gates**, and **constitutional AI governance**.

[Architecture](#architecture) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Contributing](CONTRIBUTING.md)

</div>

---

## What is BIZRA?

**BIZRA** (Arabic: بذرة, "seed") is an open-source framework for building sovereign AI agents that operate under constitutional constraints. It combines:

- **Proof-Carrying Inference (PCI):** Every AI inference carries a cryptographic proof of its inputs, model, and ethical compliance
- **FATE Gates:** Fairness, Accountability, Transparency, and Ethics gates that constrain all agent actions
- **Ihsan Constraint:** A quality threshold (Signal-to-Noise Ratio >= 0.95) that enforces excellence as a hard requirement, not a suggestion
- **Federation Protocol:** Byzantine fault-tolerant gossip protocol for peer-to-peer agent coordination
- **Constitutional Governance:** Agents are bound by an immutable constitution that cannot be overridden at runtime

### Design Philosophy

> **We do not assume.** Every claim has provenance. Every inference has proof. Every agent has a constitution.

BIZRA stands on the shoulders of giants: Shannon (information theory), Lamport (distributed consensus), Vaswani (attention mechanisms), Al-Ghazali (ethical reasoning), and General Magic (mobile agents, 1990).

---

## Architecture

```
                         BIZRA Sovereign Architecture
 ┌──────────────────────────────────────────────────────────────────────┐
 │                        Constitutional Layer                          │
 │   Immutable rules ─ FATE gates ─ Ihsan threshold ─ ADL invariants   │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │
 ┌──────────────────────────────┴───────────────────────────────────────┐
 │                        Sovereign Runtime                             │
 │   Graph-of-Thoughts ─ SNR Maximizer ─ Omega Engine ─ Treasury       │
 └──────────────┬───────────────────────────────────┬───────────────────┘
                │                                   │
 ┌──────────────┴───────────┐       ┌───────────────┴───────────────────┐
 │   Inference Gateway      │       │   Federation Layer                │
 │   Local-first ─ Tiered   │       │   Gossip ─ Consensus ─ PCI       │
 │   Edge/Local/Pool        │       │   BFT ─ Propagation              │
 └──────────────────────────┘       └───────────────────────────────────┘
```

### Core Modules

| Module | Purpose | Language |
|--------|---------|----------|
| `core/pci/` | Proof-Carrying Inference protocol (Ed25519 signatures, envelopes, gates) | Python |
| `core/proof_engine/` | Receipt builder, Ed25519 signer, BLAKE3 evidence ledger | Python |
| `core/federation/` | P2P federation (gossip, BFT consensus, secure transport) | Python |
| `core/inference/` | Tiered inference gateway (edge/local/pool backends) | Python |
| `core/sovereign/` | Sovereign runtime (Graph-of-Thoughts, treasury, autonomy) | Python |
| `core/spearpoint/` | Autonomous research engine (15 Sci-Reasoning patterns, RDVE) | Python |
| `core/bridges/` | Desktop Bridge (TCP 9742), Sci-Reasoning, Rust lifecycle | Python |
| `core/skills/` | Skill router + registry (Smart Files, RDVE, 43 skills) | Python |
| `core/token/` | Token ledger, minting, Ed25519-signed transactions | Python |
| `bizra-omega/` | High-performance core (14 Rust crates) | Rust |
| `bizra-omega/bizra-core/` | Constitution, identity, FATE gates, Islamic finance | Rust |
| `bizra-omega/bizra-federation/` | Federation protocol with gossip and signed messages | Rust |
| `bizra-omega/bizra-cli/` | Terminal UI with real-time dashboards | Rust |

---

## Quick Start

### Prerequisites

- Python 3.11+ with pip
- Rust stable toolchain (for the Omega workspace)
- An LLM backend: [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), or any OpenAI-compatible API

### Install

```bash
# Clone the repository
git clone https://github.com/BizraInfo/bizra-data-lake.git
cd bizra-data-lake

# Create a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -e ".[dev]"

# Build Rust workspace (optional, for high-performance features)
cd bizra-omega && cargo build --release && cd ..
```

### Configure

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your LLM backend settings:
#   LM_STUDIO_API_KEY=your_key
#   LM_STUDIO_BASE_URL=http://localhost:1234
```

### Run

```bash
# Start the sovereign runtime
python -m core.sovereign

# Run the CLI (Rust)
cd bizra-omega && cargo run --release --bin bizra
```

---

## Testing

```bash
# Run all tests (Python)
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow"

# Run Rust tests
cd bizra-omega && cargo test --workspace

# Run with coverage
pytest tests/ --cov=core --cov-report=term-missing
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Long-running tests (>30s) |
| `@pytest.mark.integration` | Requires external services |
| `@pytest.mark.requires_ollama` | Requires Ollama running |
| `@pytest.mark.requires_gpu` | Requires CUDA GPU |
| `@pytest.mark.requires_network` | Requires network access |

---

## Constitutional Thresholds

BIZRA enforces quality and ethics as hard constraints, not optional checks:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Ihsan (Excellence) | >= 0.95 | Minimum quality for any output |
| SNR (Signal-to-Noise) | >= 0.85 | Information quality filter |
| ADL (Justice) Gini | <= 0.40 | Resource distribution fairness |
| Harm Score | <= 0.30 | Maximum allowable harm potential |
| Confidence | >= 0.80 | Minimum inference confidence |

These thresholds are defined in [`core/integration/constants.py`](core/integration/constants.py) as the single source of truth.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Documentation Portal](docs/README.md) | Canonical docs entrypoint (role-based reading paths) |
| [Quick Start](docs/QUICK-START.md) | First-time setup and data ingestion |
| [Operations Runbook](docs/OPERATIONS_RUNBOOK.md) | Startup, health checks, incident handling |
| [Testing Guide](docs/TESTING.md) | Local gates, markers, coverage, CI alignment |
| [Architecture Blueprint](docs/ARCHITECTURE_BLUEPRINT_v2.3.0.md) | Full system architecture |
| [Desktop Bridge](docs/DESKTOP_BRIDGE.md) | AHK hotkey integration, JSON-RPC protocol |
| [Spearpoint (RDVE)](docs/SPEARPOINT.md) | Autonomous research engine, 15 thinking patterns |
| [DevOps Blueprint](docs/DEVOPS_BLUEPRINT.md) | CI/CD pipeline, K8s deployment, rollback |
| [Constitution](docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md) | Immutable constitutional rules |
| [Security Policy](SECURITY.md) | Vulnerability reporting and security architecture |
| [Contributing Guide](CONTRIBUTING.md) | How to contribute |

---

## Project Structure

```
bizra-data-lake/
├── core/                   # Python sovereign infrastructure
│   ├── pci/                # Proof-Carrying Inference protocol
│   ├── proof_engine/       # Receipt builder, Ed25519 signer, evidence ledger
│   ├── federation/         # P2P federation layer
│   ├── inference/          # Tiered inference gateway
│   ├── sovereign/          # Sovereign runtime engine
│   ├── spearpoint/         # Autonomous research (RDVE, 15 patterns)
│   ├── bridges/            # Desktop Bridge, Sci-Reasoning, Rust bridge
│   ├── skills/             # Skill router + Smart File Management
│   ├── token/              # Token ledger, minting, transactions
│   ├── benchmark/          # CLEAR framework, dominance loop, ablation
│   ├── integration/        # Cross-module constants and bridges
│   └── iaas/               # Information-as-a-Service (SNR)
├── bizra-omega/            # Rust high-performance workspace
│   ├── bizra-core/         # Constitution, identity, FATE
│   ├── bizra-federation/   # Gossip + consensus protocol
│   ├── bizra-cli/          # Terminal UI dashboard
│   ├── bizra-api/          # REST API server
│   ├── bizra-inference/    # Inference backends
│   └── 9 more crates...   # Telescript, proofspace, hunter, etc.
├── tests/                  # Comprehensive test suite
├── docs/                   # Architecture and specifications
├── deploy/                 # Docker and Kubernetes configs
└── .github/workflows/      # CI/CD pipelines
```

---

## License

[MIT](LICENSE) -- Copyright 2026 BIZRA Sovereign

---

<div align="center">
<br>

*Every seed carries within it the memory of the forest it will become.*

<br>

Built with Ihsan in Dubai

</div>
