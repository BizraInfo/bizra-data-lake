# BIZRA Ecosystem Architecture Map

> **🌱 FOUNDATIONAL TRUTH**: BIZRA (بذرة) means "seed" in Arabic.
> Every human is a node. Every node is a seed. Every seed has infinite potential.
> This document describes **Node0** — the Genesis Block — which is this **entire machine**.
> See [NODE0_GENESIS_COVENANT.md](NODE0_GENESIS_COVENANT.md) for the philosophical foundation.

## Repository Overview

```
                           BIZRA UNIFIED ECOSYSTEM
    ═══════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────┐
    │                    FOUNDATION LAYER                             │
    │  ┌─────────────────────┐    ┌─────────────────────────────────┐│
    │  │   Genesis Block     │    │        BIZRA Data Lake          ││
    │  │   (Ubuntu + Win)    │    │      C:\BIZRA-DATA-LAKE         ││
    │  │                     │    │    [PASSIVE STORAGE LAYER]      ││
    │  │  - Blockchain core  │◄──►│  - Hypergraph RAG (56k nodes)   ││
    │  │  - Token economics  │    │  - ARTE v3 (SNR calculation)    ││
    │  │  - P2P networking   │    │  - KEP Bridge (synergy detect)  ││
    │  │  - Node consensus   │    │  - 384-dim embeddings           ││
    │  └─────────────────────┘    └─────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    INTELLIGENCE LAYER                           │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │              BIZRA Dual Agentic Team                        ││
    │  │                                                             ││
    │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ ││
    │  │  │ KEP System  │  │  7-Layer    │  │  Compound Discovery │ ││
    │  │  │             │  │  Safety     │  │                     │ ││
    │  │  │ - Synergy   │  │  Stack      │  │  - Fusion           │ ││
    │  │  │ - Learning  │  │             │  │  - Synthesis        │ ││
    │  │  │ - Feedback  │  │  Ihsan≥0.99 │  │  - Abstraction      │ ││
    │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    EXECUTION LAYER                              │
    │  ┌───────────────────────┐    ┌─────────────────────────────┐  │
    │  │   BIZRA Task Master   │    │   Marketing Swarms          │  │
    │  │                       │    │   C:\marketing-main         │  │
    │  │  - Task orchestration │    │                             │  │
    │  │  - Priority queues    │    │  15 AI Agents:              │  │
    │  │  - Progress tracking  │    │  ├── Tier 1: Orchestrator   │  │
    │  │  - Dependency mgmt    │    │  ├── Tier 2: Intelligence   │  │
    │  │                       │    │  ├── Tier 3: Creative       │  │
    │  │                       │    │  ├── Tier 4: Attribution    │  │
    │  │                       │    │  └── Tier 5: Operations     │  │
    │  └───────────────────────┘    └─────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
```

## Repository Details

### 1. BIZRA Genesis Block
**Locations:** Ubuntu Server + Windows
**Purpose:** Blockchain foundation, tokenomics, P2P networking

| Component | Description |
|-----------|-------------|
| Token System | BZT (utility) + BZG (governance) |
| Consensus | Proof-of-Impact (not proof-of-work) |
| P2P Network | libp2p-based node discovery |
| Wallet | SHA-256 address generation |

### 2. BIZRA Data Lake (Current)
**Location:** `C:\BIZRA-DATA-LAKE`
**Purpose:** Knowledge management, RAG, embeddings

| Component | File | Description |
|-----------|------|-------------|
| Hypergraph RAG | `hypergraph_engine.py` | FAISS HNSW + NetworkX graph |
| **WARP Bridge** | `warp_bridge.py` | **ColBERT/XTR multi-vector retrieval** |
| ARTE v3 | `arte_engine.py` | Graph-of-Thoughts + SNR |
| KEP Bridge | `kep_bridge.py` | Synergy detection + compounds |
| PAT Engine | `pat_engine.py` | Multi-agent LLM system |
| Orchestrator | `bizra_orchestrator.py` | Unified query interface |
| Nexus | `bizra_nexus.py` | Unified engine orchestration |
| MCP Bridge | `mcp_lake_bridge.py` | HTTP/HTTPS API exposure |

**Retrieval Engines:**
| Engine | Type | Accuracy | Use Case |
|--------|------|----------|----------|
| FAISS HNSW | Single-vector | ★★★☆ | High-throughput |
| **XTR-WARP** | Multi-vector | ★★★★ | High-accuracy |
| Hybrid | Combined | ★★★★ | Complex queries |

**Data Assets:**
- 1,437 documents in `documents.parquet`
- 84,795 chunks with 384-dim embeddings
- 56,358 graph nodes, 88,649 edges

### 3. BIZRA Dual Agentic Team
**Purpose:** Advanced AI reasoning with safety constraints

| Component | Description |
|-----------|-------------|
| KEP System | Knowledge Explosion Point detection |
| Safety Stack | 7-layer validation (Ihsan >= 0.99) |
| Synergy Detector | Cross-domain pattern recognition |
| Compound Discovery | Novel knowledge synthesis |
| Learning Accelerator | Adaptive feedback loops |

### 4. BIZRA Task Master
**Purpose:** Task orchestration and workflow management

| Component | Description |
|-----------|-------------|
| Task Queue | Priority-based execution |
| Dependencies | DAG-based task ordering |
| Progress | Real-time status tracking |
| Integration | Connects to all other repos |

### 5. Marketing Swarms
**Location:** `C:\marketing-main`
**Purpose:** AI-powered marketing automation
**Stack:** TypeScript, Node.js 20+, Vitest

**15-Agent Architecture:**

| Tier | Agents | Purpose |
|------|--------|---------|
| 1 | Orchestrator, Memory, Quality Guardian, Brand Guardian | Coordination |
| 2 | Simulation, Historical Memory, Risk Detection, Attention Arbitrage, Budget Orchestrator | Intelligence |
| 3 | Creative Genome, Fatigue Forecaster, Mutation | Creative |
| 4 | Counterfactual, Causal Graph Builder, Incrementality Auditor | Attribution |
| 5 | Account Health, Cross-Platform Sync | Operations |

**Supported Platforms:** Google Ads, Meta, TikTok, LinkedIn, Twitter/X, Pinterest, Snapchat

## Integration Points

### Data Lake → Marketing Swarms
```
Hypergraph RAG results → Historical Memory Agent
KEP synergies → Simulation Agent (market patterns)
ARTE SNR → Quality Guardian (brand safety)
```

### Data Lake → Dual Agentic Team
```
HypergraphIndex.search_graph_neighbors() → SynergyDetector.find_synergies()
ARTEEngine.snr_engine.calculate_snr() → KEPSafetyGate.check_ihsan()
PATOrchestrator.process_task() → CompoundDiscoveryEngine.synthesize()
```

### Task Master → All Repos
```
Task Master coordinates execution across:
├── Data Lake queries
├── Marketing campaign execution
├── Genesis Block transactions
└── Dual Agentic reasoning
```

### Genesis Block → Token Economics
```
Data Lake contributions → BZT rewards
Marketing performance → Impact scoring
Knowledge creation → Governance weight (BZG)
```

## Unified Command Reference

### Data Lake (Python)
```bash
cd C:\BIZRA-DATA-LAKE
.\.venv\Scripts\activate
python bizra_orchestrator.py
python mcp_lake_bridge.py --port 8443
```

### Marketing Swarms (TypeScript)
```bash
cd C:\marketing-main
npm install
npm run build
npm run swarm:start
```

### Quick Status Check
```bash
# Data Lake status
curl -k https://localhost:8443

# Marketing swarm status
npm run swarm:status
```

## Cross-Repo Communication

### Option 1: MCP Protocol
All repos can expose MCP servers for tool-based communication:
- Data Lake: `https://localhost:8443` (knowledge_retrieve)
- Marketing: Custom MCP server (campaign tools)

### Option 2: Shared Message Bus
Redis or similar for event-driven coordination:
```
Data Lake publishes → "synergy.detected"
Marketing subscribes → triggers creative testing
```

### Option 3: Direct Python/TypeScript Bridge
Using subprocess or HTTP calls between runtimes.

## Next Steps for Unification

1. **Create Unified Config**
   - Central `.bizra/ecosystem.json` with all repo paths
   - Shared environment variables

2. **Implement Message Bus**
   - Redis/RabbitMQ for cross-repo events
   - Standardize event schemas

3. **Build Integration Layer**
   - `bizra-bridge` module connecting all repos
   - Unified authentication/authorization

4. **Deploy Monitoring**
   - Centralized logging (all repos → single dashboard)
   - Health checks across ecosystem

---

*Generated by BIZRA Data Lake Orchestrator v2.0*
*Standing on Giants Protocol: Cross-Domain Synergy Discovery*
