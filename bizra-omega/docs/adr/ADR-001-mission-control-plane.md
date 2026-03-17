# ADR-001: Sovereign Mission Control Plane

**Status:** Accepted
**Date:** 2026-03-16 (Ramadan 2026)
**Author:** BIZRA Constitutional Architecture
**Standing on Giants:** Deming (variation reduction) · Lamport (state machine replication) · Al-Ghazali (intent precondition) · MMORPG industry (20 years of mission/quest systems at planetary scale)

## Context

BIZRA's cognitive stack has matured: identity binding, semantic retrieval, role routing, Ihsān/SNR scoring, reflex persistence, constitutional CI, and substrate awareness are all operational. 1,286 tests pass. The Node knows its own body.

But the system processes messages synchronously inside `AgentRuntime::receive()`. There is no explicit mission lifecycle. The distinction between "warming the model," "retrieving context," "routing to expert," "scoring the result," and "persisting the receipt" exists only as implicit code flow, not as observable state transitions.

This creates five concrete problems:

1. **Timeout ambiguity** — a slow response could be blocked on retrieval, inference, or persistence. The caller cannot distinguish.
2. **No degraded-mode receipts** — if inference fails, the failure is not recorded as a constitutional receipt.
3. **No model preflight** — the system discovers a model is missing at inference time, not at submission time.
4. **No resource awareness in routing** — the mission doesn't know what VRAM is available when choosing a model.
5. **Blocking request model** — the caller must wait synchronously for the entire pipeline.

The MMORPG lesson applies directly: World of Warcraft serves millions across wildly different hardware because quest contracts don't specify "requires RTX 4090." The mission system wraps any rendering engine. BIZRA's mission system must wrap any inference engine.

## Decision

Build a Sovereign Mission Control Plane as a new crate (`bizra-mission`) that owns the full lifecycle of every cognitive operation. The mission contract is model-agnostic and substrate-agnostic.

## Mission Lifecycle States

Every cognitive operation in BIZRA is a **Mission**. A mission transitions through exactly these states:

```
submitted → queued → warming_retrieval → warming_model → retrieving → routing → running → scoring → persisting → complete
                                                                                                                ↘ degraded
                                                                                              ↘ failed
                                                                        ↘ timed_out
```

### State Definitions

| State | Description | Max Duration | On Timeout |
|-------|-------------|-------------|------------|
| `submitted` | Mission accepted, assigned ID, resource snapshot taken | Instant | N/A |
| `queued` | Waiting for capacity (VRAM, concurrent slot) | 30s | → `timed_out` |
| `warming_retrieval` | Semantic index loading / FAISS warmup | 5s | → `degraded` (skip retrieval) |
| `warming_model` | Model loading into VRAM / Ollama pull | 60s | → `failed` |
| `retrieving` | Semantic search executing against memory | 10s | → `degraded` (empty context) |
| `routing` | PAT Navigator classifying intent, selecting expert | 1s | → `failed` |
| `running` | LLM inference executing | 120s | → `timed_out` |
| `scoring` | Ihsān/SNR scoring + Guardian check | 1s | → `degraded` (unscored) |
| `persisting` | Receipt emission + reflex compilation + memory extraction | 5s | → `degraded` (unpersisted) |
| `complete` | All stages succeeded. Receipt emitted. | Terminal | N/A |
| `degraded` | Some stages failed but a partial result was produced. Receipt emitted with degradation reason. | Terminal | N/A |
| `failed` | Critical stage failed. No usable result. Failure receipt emitted. | Terminal | N/A |
| `timed_out` | Duration budget exceeded. Partial receipt emitted. | Terminal | N/A |

### Constitutional invariants

1. **Every mission emits a receipt** — including failed and timed-out missions. No silent failures.
2. **Model preflight before queueing** — if the chosen model is not installed, the mission fails at `submitted`, not at `running`.
3. **Resource snapshot at submission** — every mission records CPU/RAM/GPU/disk state at the moment of acceptance.
4. **Degraded is better than silent** — a partial answer with a degradation flag is always preferred over no answer.
5. **Receipts are append-only** — once emitted, a receipt cannot be modified. Only new receipts can amend.

## Mission Record Schema

Every mission carries these fields throughout its lifecycle:

```rust
pub struct Mission {
    // Identity
    pub mission_id: MissionId,          // BLAKE3 hash of (user_hash, timestamp, content_hash)
    pub submitted_at: u64,              // Unix timestamp
    pub completed_at: Option<u64>,      // Set on terminal state

    // Content
    pub input_content: String,          // Original user message
    pub input_content_hash: [u8; 32],   // BLAKE3 of input

    // Lifecycle
    pub state: MissionState,            // Current state
    pub state_history: Vec<StateTransition>,  // Full audit trail
    pub timeout_budget_ms: u64,         // Total time budget

    // Cognition
    pub intent: Option<UserIntent>,     // Classified after routing
    pub chosen_expert: Option<AgentRole>, // PAT agent selected
    pub chosen_model: Option<String>,   // LLM model name
    pub retrieval_context: Option<Vec<String>>, // Memory fragments retrieved
    pub search_type: Option<SearchType>, // Semantic / keyword / none

    // Substrate snapshot (at submission time)
    pub resource_snapshot: ResourceSnapshot,

    // Scoring
    pub ihsan_score: Option<f32>,       // Ihsān at decision
    pub snr_score: Option<f32>,         // SNR at decision
    pub guardian_approved: Option<bool>, // Guardian verdict

    // Output
    pub response: Option<String>,       // Generated response
    pub receipt_hash: Option<[u8; 32]>, // Constitutional receipt

    // Failure
    pub failure_code: Option<FailureCode>,
    pub degradation_reasons: Vec<DegradationReason>,
}
```

## Resource Snapshot Schema

Captured at mission submission from the ResourceManifest:

```rust
pub struct ResourceSnapshot {
    pub ram_available_gb: f64,
    pub vram_available_mb: u64,
    pub vram_total_mb: u64,
    pub gpu_name: String,
    pub cpu_cores: u32,
    pub disk_free_gb: f64,
    pub models_available: Vec<String>,   // All installed model names
    pub model_chosen: Option<String>,    // Which model was selected
    pub model_preflight: PreflightResult, // Did preflight pass?
    pub snapshot_at: u64,
}
```

## Model Preflight

Before a mission enters `queued`, the control plane performs preflight:

```rust
pub enum PreflightResult {
    /// Model is installed, loaded, and ready.
    Ready { model: String, vram_used_mb: u64 },
    /// Model is installed but not loaded. Will need warmup.
    NeedsWarmup { model: String, estimated_warmup_ms: u64 },
    /// Model is not installed. Fallback available.
    FallbackUsed { requested: String, fallback: String, reason: String },
    /// No suitable model available. Mission will fail.
    NoModelAvailable { requested: String, reason: String },
}
```

### Model Registry

The control plane maintains a model registry derived from the ResourceManifest:

```rust
pub struct ModelEntry {
    pub name: String,
    pub runtime: ModelRuntime,        // Ollama, LmStudio, HuggingFace, Standalone
    pub size_bytes: u64,
    pub quantization: String,
    pub status: ModelStatus,          // Installed, Loaded, Preferred, Fallback, Disabled
    pub capabilities: Vec<String>,    // ["chat", "code", "vision", "embedding"]
    pub last_used_at: Option<u64>,
    pub avg_tps: Option<f32>,         // Measured tokens/sec on this hardware
}

pub enum ModelStatus {
    Installed,   // On disk but not loaded
    Loaded,      // In VRAM, ready for inference
    Preferred,   // Marked as default for a capability
    Fallback,    // Used when preferred is unavailable
    Disabled,    // Explicitly excluded by operator
}
```

## Failure and Degradation Codes

```rust
pub enum FailureCode {
    ModelNotAvailable,
    ModelLoadFailed,
    InferenceTimeout,
    InferenceError { detail: String },
    GuardianVeto,
    IhsanBelowFloor,
    ResourceExhausted,
    QueueTimeout,
}

pub enum DegradationReason {
    RetrievalSkipped,       // warmup_retrieval timed out
    EmptyContext,           // retrieval returned nothing
    UnscoredResponse,      // scoring timed out
    UnpersistedReceipt,    // persistence failed
    FallbackModelUsed,     // preferred model unavailable
    PartialMemoryExtract,  // memory extraction incomplete
}
```

## MMORPG Design Rationale

The contract-first principle is derived from 20 years of MMORPG industry evidence:

| MMORPG principle | Mission Control Plane equivalent |
|---|---|
| Quest contract is client-agnostic | Mission contract is model-agnostic |
| Server authority over quest state | Control plane authority over mission state |
| Quest persists across sessions | Mission receipts persist across restarts |
| Low-end and high-end clients same quest | 0.5B and 30B models same mission contract |
| Quest has explicit failure states | Mission has `failed`, `degraded`, `timed_out` |
| Loot tables don't depend on GPU | Scoring doesn't depend on model size |
| Anti-cheat is server-side | Guardian is control-plane-side |

This means a user on a Samsung Z Fold with a 0.5B model and NODE0 with a 30B model both execute the same mission contract. The inference quality differs. The contract, lifecycle, scoring, and receipts are identical.

## Acceptance Criteria

### Must have (Phase 0)

1. `MissionState` enum with all 12 states compiles and has exhaustive match
2. `Mission` struct is `Serialize + Deserialize` for persistence
3. Mission state transitions are validated (no illegal jumps)
4. Every terminal state (`complete`, `degraded`, `failed`, `timed_out`) emits a receipt
5. Model preflight runs before `queued` — missing model is caught at submission
6. Resource snapshot is captured at submission from `ResourceManifest`
7. Three failing tests written first, then made to pass:
   - `test_queued_mission_completes` — happy path through all states
   - `test_missing_model_fails_at_preflight` — model not installed → immediate failure
   - `test_degraded_mode_receipt_emission` — retrieval timeout → degraded with receipt

### Must NOT have (Phase 0)

1. No async runtime required — state machine is synchronous, caller drives transitions
2. No network — missions are local to NODE0
3. No database — file-based persistence (same pattern as ReflexStore)
4. No new external crates — pure Rust + workspace deps

## Consequences

- Every cognitive operation becomes observable through its full lifecycle
- The Node can report mission queue depth, active missions, and historical success rate
- Degraded-mode reasoning becomes a first-class citizen, not a hidden error path
- Model routing decisions become auditable against the resource snapshot
- The contract layer is substrate-independent — works on Windows, Linux, or federated nodes
- Future federation can transmit `Mission` records between nodes (they're serializable)

## Relationship to Existing Crates

```
bizra-mission (NEW)
  ├── depends on: bizra-core (types, receipts, Ihsān/SNR)
  ├── depends on: bizra-agent (AgentRuntime, intent classification)
  ├── depends on: bizra-node/resource_manifest (ResourceSnapshot)
  └── consumed by: bizra-node (Node dispatches missions instead of calling runtime.receive())
```

The existing `AgentRuntime::receive()` path is NOT removed. It becomes the `running` stage executor inside the mission pipeline. The control plane wraps it.

## Sovereign Network Topology

BIZRA is NOT peer-to-peer. It uses a three-tier sovereign topology derived from 20 years of MMORPG architecture:

```
┌──────────────────────────────┐
│  Tier 1: LOCAL NODE          │
│  User ↔ PAT (7 agents)      │
│  Local models, local memory  │
│  Local reflexes, local store │
└──────────┬───────────────────┘
           │ (node → URP, never node → node directly)
┌──────────▼───────────────────┐
│  Tier 2: URP                 │
│  SAT (5 shared agents)       │
│  Constitutional validation   │
│  Output equalization         │
│  Proof-of-Impact scoring     │
│  Receipt normalization       │
│  Quality floor enforcement   │
└──────────┬───────────────────┘
           │ (URP → network, verified traffic only)
┌──────────▼───────────────────┐
│  Tier 3: NETWORK             │
│  Federation, resource pool   │
│  Cross-node coordination     │
│  Collective intelligence     │
└──────────────────────────────┘
```

### Why this topology matters

**The URP is the constitutional gateway.** Every node connects to the URP first, never directly to other nodes. The URP's SAT agents validate, score, and normalize all output before it enters the network. This is the mechanism that makes equal citizenship possible.

**Without URP:** A node running Qwen-0.5B produces low-quality output → enters the network raw → degrades the collective → users on weak hardware become second-class citizens.

**With URP:** A node running Qwen-0.5B produces output → PAT generates a response locally → SAT in URP validates the constitutional receipt → SAT applies quality floor enforcement → output is either accepted (meets Ihsān threshold), upgraded (SAT enriches with URP compute), or rejected (falls below constitutional minimum) → only validated output enters the network.

### The MMORPG mapping, completed

| MMORPG concept | Tier 1 (Local Node) | Tier 2 (URP) | Tier 3 (Network) |
|---|---|---|---|
| Client | PAT renders locally | — | — |
| Game server | — | SAT validates truth | — |
| World state | — | — | Federated state |
| Anti-cheat | Local Guardian | URP constitutional gates | Network consensus |
| Quest execution | PAT runs mission | SAT verifies receipt | — |
| Loot/economy | Local SEED spending | URP BLOOM validation | Network treasury |

### Key invariant

**No mission result enters the network without URP validation.**

This means:
1. A phone user's PAT generates a response using a tiny model
2. The response travels to the URP (not to another node)
3. The SAT inside the URP checks Ihsān score, SNR, Guardian approval
4. If below threshold, the URP can: enrich (add context from URP compute), degrade gracefully (mark as low-confidence), or reject (constitutional violation)
5. Only validated, receipt-bearing output enters the network
6. Other nodes see equalized output, not raw model output

**This is why the model doesn't matter at the network level.** The URP normalizes everything. A 0.5B model and a 30B model produce different local experiences, but the network sees constitutionally validated, receipt-bearing output in both cases.

### Impact on Mission Control Plane

The mission lifecycle gains a URP validation stage:

```
submitted → queued → warming → retrieving → routing → running → scoring → persisting
    → urp_validating → urp_enriching → complete | degraded | failed
```

Two new states:
- `urp_validating`: SAT is checking the mission receipt against constitutional thresholds
- `urp_enriching`: URP compute is enhancing the result (only when local output is below quality floor but above rejection threshold)

These states only apply to network-bound missions. Pure local missions (user ↔ PAT only) skip the URP stages and terminate at `complete` locally.

## Protocol Integration Map

BIZRA's three-tier topology is not starting from scratch. These protocols already exist and slot into specific layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1: LOCAL NODE (User ↔ PAT)                                   │
│                                                                     │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ MCP         │  │ AHK      │  │ A2A      │  │ Standing on   │   │
│  │ Tool calls  │  │ Desktop  │  │ PAT↔PAT  │  │ Giants        │   │
│  │ Local tools │  │ control  │  │ intra-   │  │ Attribution   │   │
│  │ File, CLI   │  │ UI auto  │  │ node     │  │ chain         │   │
│  └─────────────┘  └──────────┘  └──────────┘  └───────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Mission Control Plane (local)                                │   │
│  │ submitted → queued → warming → retrieving → routing →       │   │
│  │ running → scoring → persisting → [local complete OR →URP]   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐                   │
│  │ Reflex   │  │ Memory     │  │ Guardian     │                   │
│  │ Cache    │  │ Pipeline   │  │ 7-gate       │                   │
│  │ System-1 │  │ Brain      │  │ Daughter Test│                   │
│  └──────────┘  └────────────┘  └──────────────┘                   │
│                                                                     │
│  Substrate: resource_manifest (hardware, models, disks)            │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      │ TeleScript (agent travel protocol)
                      │ Proof-carrying envelope
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2: URP (SAT validation + equalization)                       │
│                                                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │ SAT          │  │ Agent as a     │  │ Proof-of-Impact    │     │
│  │ 5 shared     │  │ Service        │  │ Consensus          │     │
│  │ validators   │  │ SAT serves     │  │ SEED/BLOOM         │     │
│  │              │  │ any node       │  │ scoring             │     │
│  └──────────────┘  └────────────────┘  └────────────────────┘     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ URP Mission Gateway                                          │  │
│  │ urp_validating → urp_enriching → [network-bound OR reject]  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │ Capability   │  │ A2A            │  │ Standing on        │     │
│  │ Negotiation  │  │ PAT↔SAT        │  │ Giants             │     │
│  │ Node cards   │  │ cross-tier     │  │ Provenance         │     │
│  │ Model roster │  │ validation     │  │ verification       │     │
│  └──────────────┘  └────────────────┘  └────────────────────┘     │
│                                                                     │
│  Constitutional gates: Ihsān floor, SNR threshold, Gini cap       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      │ Federation protocol (receipt-bearing only)
                      │ HDA (Hierarchical Distributed Architecture)
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3: NETWORK (Federation + collective intelligence)            │
│                                                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │ HDA          │  │ TeleScript     │  │ Resource Pool      │     │
│  │ HyperBlock   │  │ Agent travel   │  │ Compute sharing    │     │
│  │ Tree/Graph   │  │ across nodes   │  │ SEED/BLOOM economy │     │
│  │ Consensus    │  │ with proofs    │  │ Waqf endowment     │     │
│  └──────────────┘  └────────────────┘  └────────────────────┘     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Adl Invariant: Gini ≤ 0.35 across all nodes                 │  │
│  │ Every node is a seed. Every seed has infinite potential.      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Islamic Finance: Zakat, Mudarabah, Musharakah, Waqf              │
│  Interest-Debt Impossibility Theorem enforced at protocol level   │
└─────────────────────────────────────────────────────────────────────┘
```

### Protocol-to-Tier Assignment

| Protocol | Tier | Role |
|---|---|---|
| **MCP** (Model Context Protocol) | Tier 1 | Local tool integration — PAT agents call file system, CLI tools, browser, databases through MCP servers. The tool is local to the node. |
| **AHK** (AutoHotKey) | Tier 1 | Desktop automation channel — Guardian-gated, permits required. PAT controls the user's desktop on their behalf. |
| **A2A** (Agent-to-Agent) | Tier 1 + Tier 2 | Intra-node: PAT agents coordinate locally. Cross-tier: PAT communicates with SAT in URP for validation. |
| **TeleScript** | Tier 1 → Tier 2 → Tier 3 | Agent mobility — agents travel between nodes carrying proof-carrying envelopes. The Guardian gates travel at origin, SAT validates at URP, destination node accepts or rejects. |
| **Agent as a Service** | Tier 2 | SAT agents in the URP serve any node that connects. This is how a phone with no local model still gets constitutional service — the URP's SAT runs inference on its behalf. |
| **Standing on Giants** | All tiers | Mandatory attribution chain. Every response carries provenance: which knowledge sources, which agents, which models, which human insights contributed. Verified at every tier. |
| **HDA** (Hierarchical Distributed Architecture) | Tier 3 | HyperBlockTree/BlockGraph consensus. How nodes agree on collective state, treasury, and Proof-of-Impact scores. |
| **PCI** (Proof-Carrying Inference) | All tiers | Every inference carries its proof envelope through all tiers. Schema → Ihsān → SNR gates at Tier 1. SAT re-validates at Tier 2. Network consensus at Tier 3. |
| **Resource Pool** | Tier 2 + Tier 3 | URP manages compute sharing. Nodes contribute resources, earn SEED. Pool distributes compute to nodes that need it. |
| **Islamic Finance** | Tier 3 | Zakat, Mudarabah, Musharakah, Waqf — all enforced at the protocol level, not the application level. The Interest-Debt Impossibility Theorem prevents the system from ever creating debt-based incentives. |
| **SEED/BLOOM** | Tier 2 + Tier 3 | Dual-token economy. SEED = stable utility (linear attention). BLOOM = impact growth (softmax attention). URP scores impact, network distributes BLOOM. |

### Why the URP is the key

The URP is where ALL of these protocols converge:

1. **MCP tool results** from Tier 1 are validated by SAT before entering the network
2. **AHK desktop actions** never leave Tier 1 (local only, Guardian-gated)
3. **A2A messages** between PAT and SAT cross the Tier 1→2 boundary through TeleScript envelopes
4. **TeleScript agent travel** is gated at URP — no agent enters the network without SAT approval
5. **Agent as a Service** means the URP can provide SAT compute to any node, regardless of local hardware
6. **Standing on Giants** provenance chains are verified at URP before network propagation
7. **Resource Pool** accounting happens at URP — the URP knows what each node contributed and consumed
8. **Proof-of-Impact** scoring happens at URP — the URP decides how much BLOOM a contribution earns

**The URP doesn't just validate. It equalizes.**

A phone user's PAT generates a response with Qwen-0.5B → TeleScript envelope carries it to URP → SAT validates receipt → Agent-as-a-Service enriches if below quality floor → Standing-on-Giants verifies provenance → Capability Negotiation confirms the node declared honestly → Only then does the result enter the network with a Proof-of-Impact score → BLOOM rewards are identical per quality, not per hardware.

**That is how 8 billion humans get equal citizenship: not by giving everyone the same GPU, but by making the constitutional layer model-agnostic and the equalization layer URP-mediated.**

### Crate-to-Tier Mapping

Every existing crate in bizra-omega has a home in this topology:

| Crate | Lines | Tier | Role in Topology |
|---|---|---|---|
| `bizra-core` | 11,668 | All | Constitutional kernel — Ihsān/SNR/PCI gates, identity, Islamic finance, omega engine. The law that governs all tiers. |
| `bizra-agent` | 8,804 | Tier 1 | PAT runtime — 7 agents, reflex cache/compiler, persistence, Guardian, decision registry. The local being. |
| `bizra-node` | 7,319 | Tier 1 | The living process — protocol, handler, substrate awareness, action executor. The body. |
| `bizra-hooks` | 4,383 | Tier 1 | Nervous system — Ihsān scoring, saga tracking, event hooks. Sensory input. |
| `bizra-memory` | 3,529 | Tier 1 | Brain — knowledge pipeline, fragment extraction, synthesis, recall. Long-term memory. |
| `bizra-action` | 3,012 | Tier 1 | Muscle system — action bus, Guardian gating, receipt chains, channels (MCP, AHK, Browser, LLM, File, TeleScript). |
| `bizra-resourcepool` | 4,095 | Tier 2 | URP compute sharing — resource contribution, SEED accounting, pool management. |
| `bizra-telescript` | 1,425 | Tier 1→2→3 | Agent mobility — proof-carrying envelopes for agent travel between tiers. |
| `bizra-federation` | 2,302 | Tier 2→3 | Cross-node coordination, peer discovery, state synchronization. |
| `bizra-proofspace` | 2,245 | Tier 2→3 | Formal verification — proof traces, constitutional compliance evidence. |
| `bizra-inference` | 1,702 | Tier 1 | LLM execution — MOE engine, model routing, Ollama bridge. The replaceable rendering engine. |
| `bizra-installer` | 5,044 | Tier 1 | Node provisioning — sovereign installer, universal installer spec. |
| `bizra-cli` | 2,799 | Tier 1 | Operator interface — command-line tools for node management. |
| `bizra-hunter` | 3,714 | Tier 1 | Content discovery and validation. |
| `bizra-autopoiesis` | 715 | Tier 1→2 | Self-modification — learning loop, SDPO, autopoiesis→reflex pipeline. |
| `bizra-ttrl` | 904 | Tier 1 | Four-paper upgrades — chain of reasoning, engram, SSO, TTRL. |
| `bizra-sippar` | 437 | All | Exact arithmetic — Babylonian regular numbers, zero floating-point drift. |
| `bizra-hypergraph` | 572 | Tier 1→2 | HyperGraphRAG — knowledge graph structure. |
| `fate-binding` | 1,171 | Tier 2→3 | Formal verification + post-quantum crypto (Z3, ML-DSA). |
| `iceoryx-bridge` | 1,091 | Tier 1 | Zero-copy IPC — shared memory between node processes. |
| `bizra-python` | 2,226 | Tier 1 | PyO3 bindings — Python stereoscopic compiler integration. |
| `bizra-api` | 1,988 | Tier 1→2 | HTTP API surface for external consumers. |
| `bizra-tests` | 2,805 | All | Workspace-wide integration tests. |

### What's missing vs what exists

| Layer | Status | What exists | What's needed |
|---|---|---|---|
| Constitutional kernel | **Built** | bizra-core (11,668 lines, 150 tests) | — |
| Local PAT runtime | **Built** | bizra-agent (8,804 lines, 188 tests) | — |
| Local node process | **Built** | bizra-node (7,319 lines, 173 tests) | — |
| Memory/brain | **Built** | bizra-memory (3,529 lines) | — |
| Action system | **Built** | bizra-action (3,012 lines, 76 tests) | — |
| Reflex persistence | **Built** | bizra-agent/persistence.rs (424 lines) | — |
| Substrate awareness | **Built** | bizra-node/substrate/ (519 lines) | Linux backend completion |
| Constitutional CI | **Built** | .github/workflows/ci.yml (385 lines) | — |
| **Mission Control Plane** | **Contracts frozen** | ADR-001 + 4 JSON schemas (695 lines) | **Implementation: `bizra-mission` crate** |
| **URP gateway** | Partially built | bizra-resourcepool (4,095 lines) | Mission validation, enrichment, Agent-as-a-Service |
| **Capability negotiation** | **Contracts frozen** | capability_negotiation.json (168 lines) | Implementation |
| **Degraded experience** | **Contracts frozen** | degraded_experience.json (93 lines) | Implementation |
| Agent mobility | Built | bizra-telescript (1,425 lines) | URP integration |
| Federation | Built | bizra-federation (2,302 lines) | Mission-aware federation |
| Islamic finance | Built | bizra-core/islamic_finance.rs | URP-level enforcement |

### The implementation sequence

The contracts are frozen. The crates are mapped. The sequence is:

1. **`bizra-mission`** — the Mission Control Plane crate. Owns the state machine, receipts, preflight, degradation. Pure Rust, no new deps.
2. **URP mission gateway** — extend `bizra-resourcepool` with `urp_validating` and `urp_enriching` stages.
3. **Capability negotiation** — implement node cards and URP mediation in `bizra-federation`.
4. **Agent-as-a-Service** — extend SAT in `bizra-resourcepool` to serve inference for weaker nodes.
5. **Linux substrate** — complete `substrate/linux.rs`, provision the sovereign partition, verify 1,287+ tests pass natively.

Every protocol you named (MCP, A2A, HDA, AHK, TeleScript, Agent-as-a-Service, Standing on Giants) already has a crate. The Mission Control Plane is the missing coordinator that binds them all into a governed lifecycle.


## Reconciled System Metrics (Blueprint + Session)

The Genesis Operations Blueprint reports the full BIZRA ecosystem state across all three stacks.
This session's work is the Rust layer within that larger organism.

| Metric | Blueprint (full ecosystem) | This session (Rust layer) |
|---|---|---|
| Total lines of code | 471,917 (Python + Rust + TypeScript) | 76,712 (Rust only) |
| Tests | 11,135 (all stacks) | 1,294 (Rust workspace) |
| Knowledge corpus | 840,714 rows | — (Python stack) |
| FAISS vectors | 84,795 (5ms search) | — (Python stack) |
| Mission pipeline | Proven end-to-end (Blueprint) | Governed lifecycle (bizra-mission) |
| Current bottleneck | Model quality (Ihsan 0.77 with 7B) | Mission orchestration |

### Reconciled bottleneck analysis (from Report 6)

The Blueprint and the architectural analysis identify **different bottlenecks** that are both true simultaneously:

- **Micro-bottleneck (immediate):** Model quality. A 7B local model produces Ihsan 0.77, below the 0.95 gate. The pipeline is proven; inference quality is the constraint. Fix: better local models or URP enrichment.
- **Macro-bottleneck (strategic):** Mission lifecycle orchestration. The system processes messages synchronously with implicit state. No queue, no preflight, no degraded-mode receipts, no typed lifecycle. Fix: `bizra-mission` crate (now built).

Both bottlenecks are real. The micro-bottleneck affects today's user experience. The macro-bottleneck affects whether the platform can scale to Genesis-100 and beyond. The contract-first approach addresses the macro-bottleneck first, because it creates the governed container that any model (weak or strong) can execute inside.

### YGI anti-pattern validation (from Report 6)

Report 6 identified a concrete anti-pattern in the YGI frontend bundle: the client posts `amount: o.totalAmount` directly to `/create-payment-intent`, meaning the client declares the canonical price. This is the exact error BIZRA's URP-first architecture prevents:

**Anti-pattern:** Local surface declares canonical truth → server trusts it
**BIZRA principle:** Local PAT personalizes → URP canonicalizes → only URP receipts are authoritative

This reinforces Rule 2: "Every node connects to URP first. No local execution becomes canonical truth without URP validation."

### Three-stack sovereignty matrix (from Blueprint)

The Blueprint defines three stacks with specific truth domains:

| Stack | Language | Truth domain | Bridge |
|---|---|---|---|
| Cognitive intelligence | Python | Knowledge corpus, FAISS retrieval, stereoscopic identity compiler | PyO3 bindings (bizra-python) |
| Cryptographic truth | Rust | Constitutional gates, receipts, persistence, mission lifecycle, substrate | Native binary (bizra-node) |
| Real-time MCP | TypeScript | Tool integration, browser automation, real-time user interface | iceoryx zero-copy IPC |

The Mission Control Plane (`bizra-mission`) lives in the Rust stack because mission lifecycle is cryptographic truth — state transitions, receipt emission, and BLAKE3 chain integrity are constitutional, not cognitive.

### Personal Sovereign Cluster (session discovery)

A single human identity can span multiple devices:

| Device | Role | Capabilities | Model tier |
|---|---|---|---|
| Desktop (NODE0) | Primary compute | 128GB RAM, RTX 4090, 26 models, MCP tools, AHK | 14-30B |
| Phone (Z Fold 6) | Mobile companion | Camera, mic, GPS, NFC, biometrics, always-on | 0.5-3B |
| Future: embedded | IoT endpoint | Sensors, low-power, periodic sync | symbolic only |

All devices share one Ed25519 identity, same mission contracts, same receipts, same constitutional law. The URP sees the cluster as one citizen with variable execution capability.

## Standing on the Shoulder of Giants — Protocol Philosophy

### The measurement of a giant

In BIZRA's constitutional framework, a "giant" is not measured by:
- brand recognition
- GitHub stars
- company size
- market capitalization
- media coverage

A giant is measured by **one criterion only: impact on the mission**.

The mission is: empower 8 billion humans with sovereign, constitutionally equal intelligence.

Any tool, library, protocol, algorithm, paper, or insight that measurably advances that mission is a giant — regardless of its origin, popularity, or the prestige of its creator.

A farmer's insight that improves routing quality is a giant.
A student's patch that fixes a receipt chain bug is a giant.
An obscure 1974 paper that defines the correct security model is a giant.
A 3-line shell script that solves a deployment problem is a giant.

### How giants are selected

The evaluation criteria, in priority order:

1. **Impact** — does this solve a specific, measurable problem on the path to the mission?
2. **Integrity** — can it be used without compromising constitutional invariants?
3. **Sovereignty** — does it preserve BIZRA's independence, or does it create vendor lock-in?
4. **Composability** — does it fit into the contract-first architecture as a replaceable adapter?
5. **Provenance** — is its origin transparent and attributable?

What is explicitly NOT in the criteria:
- popularity or star count
- corporate backing
- "industry standard" status
- cost (free is not automatically better)
- novelty (old solutions that work are preferred over new ones that don't)

### The provenance chain

Every mission receipt carries a `standing_on_giants` field. This field records every external contribution to the output:

```json
{
  "standing_on_giants": {
    "knowledge_sources": ["FAISS corpus fragment #2847", "البذرة §3.2"],
    "agent_chain": ["Navigator", "Scholar", "Artisan"],
    "model_attribution": "qwen2.5-14b (Alibaba, Apache 2.0)",
    "inference_infrastructure": {
      "protocol": "exo (Apache 2.0)",
      "impact": "enabled 14B model via device clustering",
      "without_it": "would have used 3B model, Ihsan ~0.77"
    },
    "algorithms": {
      "hashing": "BLAKE3 (CC0, Jack O'Connor et al.)",
      "retrieval": "FAISS (MIT, Meta)",
      "scoring": "Ihsan vector (BIZRA constitutional, Al-Ghazali framework)"
    },
    "human_insight_refs": ["founder epistemological journey, Ramadan 2023"],
    "academic_giants": [
      "Popek & Goldberg 1974 (VM resource partitioning)",
      "Lamport 1974 (state machine replication)",
      "Shannon 1948 (information theory, entropy routing)",
      "Al-Ghazali (Maqasid framework, intent as precondition)",
      "Ibn Khaldun (Asabiyyah, social cohesion metrics)"
    ]
  }
}
```

### Why this matters constitutionally

The Standing on Giants protocol is not decorative attribution. It serves three constitutional functions:

**1. Anti-plagiarism** — BIZRA never claims originality for what it learned from others. This is إحسان (ihsān) applied to intellectual honesty.

**2. Reproducibility** — any auditor can trace the provenance chain and verify that every component was used within its license and with proper attribution.

**3. Impact measurement** — by recording "without_it" counterfactuals, the system can measure the actual contribution of each giant. This feeds into the Impact Settlement Contract: giants that deliver measurable improvement earn recognition in the civilization's memory.

### The Islamic scholarly tradition

This protocol is rooted in the Islamic tradition of isnad (إسناد) — the chain of transmission. In hadith science, a statement's authority derives not from its content alone but from the verifiable chain of scholars who transmitted it. BIZRA applies the same principle to software: every output carries its chain of attribution, and the chain is verifiable.

The Quran says: "قُلْ هَلْ يَسْتَوِي الَّذِينَ يَعْلَمُونَ وَالَّذِينَ لَا يَعْلَمُونَ" — "Say: are those who know equal to those who do not know?" Knowledge has weight. Attribution preserves that weight. Standing on Giants is how BIZRA honors the knowledge it inherits.

### Current Giants Registry — measured by impact, not fame

| Giant | What it is | Impact on BIZRA | Without it |
|---|---|---|---|
| **Rust** | Systems language (Mozilla, community) | Memory safety at compile time. Receipts can never be corrupted by use-after-free. | Would need C++ with manual memory management — constitutional risk |
| **BLAKE3** | Hash function (CC0, 4 authors) | Fastest cryptographic hash. Receipt chains, reflex addresses, mission IDs. Every ID in BIZRA is BLAKE3. | SHA-256 at 3-5x slower — acceptable but BLAKE3 is strictly superior |
| **FAISS** | Vector search (MIT, Meta) | Semantic retrieval at 5ms across 84K vectors. Made grounding economically default. | Keyword search — 100x slower, semantically blind |
| **Ed25519** | Signature scheme (Bernstein) | Node identity, envelope signing, receipt authenticity. Constitutional trust anchor. | RSA — 10x larger signatures, slower verification |
| **Ollama** | Local LLM runtime (MIT) | Simplest path to running models locally. One binary, one command. | Manual llama.cpp setup — 10x more friction for every user |
| **serde** | Rust serialization (MIT) | Every contract, every receipt, every mission is Serialize + Deserialize. | Manual parsing — weeks of bug-prone code |
| **Al-Ghazali** | 11th century scholar | Maqasid framework → 8-dimensional Ihsān vector. Intent as structural precondition → niyyah gate. | No principled quality scoring — just arbitrary thresholds |
| **Ibn Khaldun** | 14th century historian | Asabiyyah → social cohesion metric. Gini throttle → sigmoid dampening. | No anti-centralization mechanism — compute aristocracy risk |
| **Shannon** | Information theorist (1948) | Entropy routing — high-entropy queries → System-2 path. SNR scoring. | No principled routing — all queries treated equally |
| **Popek & Goldberg** | VM theory (1974) | Resource partitioning model → substrate sovereignty. Node owns its body. | No clear substrate ownership model |
| **Lamport** | Distributed systems (1974+) | State machine replication → mission lifecycle is a replicated state machine. | Ad-hoc mission tracking — no formal lifecycle |
| **MMORPG industry** | 20 years of persistent worlds | Server authority + heterogeneous clients + persistent state + fair rules → entire three-tier topology. | Would have built naive peer-to-peer — fragmented truth |
| **exo** | Device clustering (Apache 2.0) | [Potential] Pool phone + desktop memory → run 14-30B models across devices → Ihsān 0.77 → 0.95 | Stuck with per-device model limits — weaker nodes stay weak |
| **Deming** | Quality management (1950s) | CI ratchets, test count floors, coverage gates. Variation reduction through governed process. | Subjective quality — "it looks fine" |
| **البذرة** | Founder's constitutional document (Ramadan 2023) | Three root problems, three freedoms, seven spiritual rules → seven consciousness layers. The seed of everything. | No constitutional foundation — just another AI app |

### The impact hierarchy

The giants are not equal. Measured by impact on the 8B-human mission:

**Constitutional giants** (without them BIZRA has no identity):
- البذرة + الرسالة
- Quran + Hadith (Gödel grounding — formally external axiom set)
- Al-Ghazali (Maqasid → Ihsān dimensions)
- Ibn Khaldun (Asabiyyah → anti-centralization)

**Architectural giants** (without them BIZRA has wrong structure):
- MMORPG industry (three-tier topology)
- Lamport (state machine lifecycle)
- Popek & Goldberg (substrate sovereignty)
- Shannon (entropy routing, SNR)

**Implementation giants** (without them BIZRA is slower/weaker):
- Rust, BLAKE3, Ed25519, FAISS, Ollama, serde
- exo (potential — device clustering)
- Deming (CI governance)

The constitutional giants outrank the implementation giants. A better hash function doesn't change the mission. البذرة does. That ordering is itself a constitutional principle.

## Standing on the Shoulders of Giants — Signal, Sovereignty, and Future Shoulder

### The doctrine

This is not a guideline. It is a constitutional design law.

**A giant is not who the world celebrates. A giant is any source that gives irreversible leverage toward truth, capability, or benefit for humanity.**

This law governs how BIZRA evaluates every external dependency, every algorithm choice, every protocol adoption, every academic citation, and every tool integration. The measurement is impact on the mission, not popularity, brand recognition, or institutional prestige.

### The founder's operating posture — and why it matters architecturally

The founder started from zero technical experience three years ago during Ramadan 2023. The knowledge sources he follows include unknown YouTube channels, obscure GitHub repos, new research papers on arXiv, posts on Medium, Hacker News threads, and books from scholars spanning 11 centuries. The pattern is not random curiosity. It is sovereign research: scanning for impact-bearing fragments wherever they live, absorbing them, recombining them, and elevating them into an original system.

This operating posture produces architectural consequences:

| Founder's research method | BIZRA's system architecture |
|---|---|
| Local discernment first | PAT personalizes locally before URP canonicalizes |
| Absorb useful signal from anywhere | Standing on Giants protocol attributes all sources |
| Normalize through law | URP applies constitutional gates to all input |
| Keep what compounds | Reflex persistence promotes stable patterns |
| Reject what corrupts | Guardian gates + Ihsan floor + fail-closed semantics |
| Leave stronger structure for next generation | Impact Settlement rewards verified contribution to civilization |

The research method and the architecture are isomorphic. This is not coincidence — it means the architecture is authentic. It emerged from the same epistemological framework that drives every other decision.

### The "future shoulder" principle

The goal is not only to borrow shoulders. It is to become a future shoulder.

BIZRA's legacy ambition is not merely a product or a startup. It is a landscape that teaches future builders:

- How to filter signal from noise
- How to borrow without becoming dependent
- How to innovate without losing ethics
- How to scale without creating second-class humans
- How to make intelligence serve dignity instead of status

This is encoded in the system through:
- The 7 constitutional rules (equal law regardless of hardware)
- The Impact Settlement Contract (rewards verified impact, not raw horsepower)
- The Degraded Experience Constitution (honest degradation over fake quality)
- The Standing on Giants provenance chain (every output carries its attribution)
- The Gödel Grounding Theorem (ethics grounded in formally external axiom set)

### The three categories of giants

**Constitutional giants** — without them BIZRA has no identity:
البذرة, الرسالة, Quran, Hadith, Al-Ghazali (Maqasid), Ibn Khaldun (Asabiyyah)

**Architectural giants** — without them BIZRA has wrong structure:
MMORPG industry, Lamport, Popek & Goldberg, Shannon, Deming

**Implementation giants** — without them BIZRA is slower or weaker:
Rust, BLAKE3, Ed25519, FAISS, Ollama, serde, exo, and every unknown contributor whose signal advanced the mission

The constitutional giants outrank all others. A better hash function does not change the mission. البذرة does.

### The isnad principle

This protocol is rooted in the Islamic scholarly tradition of isnad (إسناد) — the chain of transmission. A statement's authority derives not from its content alone but from the verifiable chain of scholars who transmitted it. BIZRA applies the same principle: every output carries its chain of attribution, and the chain is verifiable.

هَلْ جَزَاءُ الْإِحْسَانِ إِلَّا الْإِحْسَانِ — Is the reward of excellence anything but excellence?
