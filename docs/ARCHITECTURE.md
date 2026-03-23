# BIZRA Architecture Guide

## System Overview

BIZRA is a sovereign AI operating system where intelligence is defined by constitutional filtration — the refusal set IS the capability.

```
┌──────────────────────────────────────────────────────────────┐
│                    USER (scripts/bizra TUI)                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│              MISSION EXECUTOR (9 stages)                      │
│  FAISS → Amplify → Inference → Skill → SEED → Memory →      │
│  EventBus → Notify → Watcher                                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│              CONSTITUTIONAL SPINE                             │
│  Fail-closed gates │ BLAKE3 receipts │ Ed25519 signatures    │
│  Ihsan ≥ 0.95 │ Gini ≤ 0.35 │ Geometric mean (zero = zero) │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌─────▼──────────┐
│ Rust Binary  │  │ Python Runtime  │  │ Event Buses    │
│ bizra-node   │  │ sovereign/      │  │ CQRS+Sovereign │
│ (24 crates)  │  │ node0/          │  │ (FanoutBus)    │
└──────────────┘  └─────────────────┘  └────────────────┘
```

## Three-Layer Receipt Cascade

### Layer 1: Mission Execution (Rust, ~20s, request-time)

Every user request flows through the 14-state mission state machine:

```
Submitted → Queued → WarmingRetrieval → WarmingModel → Retrieving →
Routing → Running → Scoring → Persisting → Complete | Degraded | Failed
```

**Gates:** Preflight (capability check), State Machine (illegal transitions rejected), Guardian Veto (→ Degrade, not Fail), Ihsan Floor (< 0.95 → Degrade), Amanah (unsigned → reject).

Key insight: Guardian veto causes DEGRADATION, not failure. Receipt is preserved for measurement. Helix3 then filters degraded receipts from the evolutionary tensor.

### Layer 2: Evolutionary Learning (Python Helix3, 60s tick)

Every 60 seconds, Helix3 processes accumulated receipts:

1. **FATE Filter** — only `fate_verdict=="approved"` receipts enter tensor
2. **Geometric Mean** — 8D Ihsan tensor; any dimension = 0 → composite = 0
3. **Mint/Halt** — only excellence (≥ 0.95) earns SEED; Gini > 0.35 → HALT
4. **Reflex Precipitation** — high-confidence patterns compiled to System-1 cache
5. **Evidence Chain** — BLAKE3 hash links each tick to the previous

### Layer 3: Metacognition (Python Node0, opt-in)

Governed recursive self-improvement (autopoiesis):

- Activated via `BIZRA_AUTOPOIESIS_ENABLED=true`
- Z3 formal verification before any integration
- Shadow deployment before production merge
- Reversibility plan required for each candidate
- Feeds existing Node0 learning loop (not a parallel spine)

## 9-Stage Mission Pipeline

| Stage | Module | What It Does |
|-------|--------|-------------|
| 1 | `faiss_search.py` | 84,795-vector semantic search (0.5s cached) |
| 1.5 | `diffusion_reasoning_amplifier.py` | HMM→GoT depth/hypothesis hints (fail-closed) |
| 2 | `bizra-node` binary | Ollama inference with sovereign system prompt |
| 3 | `file_organizer.py` / skills | Skill dispatch (organize, browse) |
| 4 | `seed_calc.py` + `seed_ledger.py` | PoI reward calculation + JSONL persistence |
| 5 | `brain.py` | Living Memory (episodic + semantic + procedural) |
| 6 | `event_publisher.py` | FanoutEventBus → CQRS + sovereign buses |
| 7 | Desktop Bridge (port 9742) | AHK/toast notification |
| 8 | `proactive_watcher.py` | Filesystem change detection |

## Key Architectural Decisions

### Governance Above Cognition
The system's intelligence is defined by what it refuses, not what it generates. Constitutional gates are enforced at compile time (Rust newtypes) and runtime (fail-closed gates).

### Degradation Preserves Evidence
Guardian veto and Ihsan floor produce DEGRADED missions, not failures. Receipts are preserved for measurement. Helix3 filters them from learning. This creates an audit trail of failures while preventing them from corrupting the evolutionary process.

### Frozen Agents (Gödel Grounding)
P5 (Ethicist) and S2 (Oracle) derive their rules from external constitutional axioms (Maqasid al-Sharia), not from learned data. No agent evaluates its own ethical constraints.

### Arithmetic Densification
Three Rust types make illegal economic states unrepresentable:
- `IhsanScore(u16)` — fixed-point quality, no floating-point drift
- `ExactAmount(i64)` — micro-units, exact three-way splits
- `BoundedRatio(u32)` — `a + complement() == ONE` by construction

### Receipt Chain Integrity
Every receipt includes `previous_receipt_hash`. Modifying receipt N breaks receipt N+1. The chain is tamper-evident, cross-session persistent (chain_head file), and Ed25519 signed.

## Directory Structure

```
bizra-data-lake/
├── core/                    # Python sovereign runtime (58 subpackages)
│   ├── sovereign/           # Runtime core, API, organism, mission executor
│   ├── node0/               # Heartbeat, boot/breath receipts
│   ├── bus/                  # ActionBus, EventBus, event_publisher
│   ├── pci/                 # Proof-Carrying Inference gates
│   ├── proof_engine/        # FAISS, SEED calc, evidence ledger
│   ├── living_memory/       # 3-layer cognitive memory
│   ├── federation/          # Gossip, consensus, cross-node ops
│   ├── reasoning/           # DiffusionAmplifier, GoT, guardian council
│   ├── autopoiesis/         # Self-improvement loop (opt-in)
│   └── token/               # SEED/BLOOM dual-token economics
├── bizra-omega/             # Rust workspace (24 crates)
│   ├── bizra-core/          # Constitution, identity, canonical hashing
│   ├── bizra-node/          # Sovereign binary (2.8MB)
│   ├── bizra-mission/       # 14-state lifecycle + receipts
│   ├── bizra-hooks/         # Nervous system (13 subscribers)
│   ├── bizra-agent/         # OmniKernel cognitive cycle
│   ├── bizra-federation/    # Gossip + PBFT consensus
│   ├── bizra-inference/     # Tiered inference gateway
│   └── bizra-sippar/        # Babylonian exact arithmetic
├── scripts/bizra            # Sovereign TUI (500+ LOC bash)
├── frontend/                # React dashboard + prototypes
├── tests/                   # 12,680 tests (Python + Rust)
├── .github/workflows/       # 7 CI pipelines
└── docs/                    # Papers, specs, evidence
```

## Constitutional Thresholds

All defined in `core/integration/constants.py` (single source of truth):

| Threshold | Value | Purpose |
|-----------|-------|---------|
| IHSAN_THRESHOLD | 0.95 | Production quality floor |
| STRICT_IHSAN_THRESHOLD | 0.99 | Consensus-critical operations |
| SNR_THRESHOLD | 0.85 | Minimum signal quality |
| ADL_GINI_THRESHOLD | 0.35 | Distributive justice hard gate |
| ZAKAT_RATE | 0.025 | 2.5% automatic redistribution |
| ADL_HARBERGER_TAX_RATE | 0.05 | 5% annual continuous redistribution |

Cross-language sync enforced: Python `constants.py` ↔ Rust `bizra-core/src/lib.rs`.
