# Data Flow Architecture

How data moves through the BIZRA system.

## Table of Contents

1. [Overview](#overview)
2. [Message Flow](#message-flow)
3. [Inference Pipeline](#inference-pipeline)
4. [Memory System](#memory-system)
5. [FATE Validation](#fate-validation)
6. [Agent Communication](#agent-communication)
7. [Federation Data Flow](#federation-data-flow)
8. [Caching Strategy](#caching-strategy)
9. [Error Handling](#error-handling)

---

## Overview

BIZRA processes data through multiple interconnected pipelines.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIZRA DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   USER INPUT                                                                │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│   │  Parse  │ →  │ Enrich  │ →  │  Route  │ →  │ Process │                │
│   │ Command │    │ Context │    │  Agent  │    │  Task   │                │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘                │
│                                                       │                     │
│                                                       ▼                     │
│                                               ┌─────────────┐              │
│                                               │  Inference  │              │
│                                               │   Backend   │              │
│                                               └──────┬──────┘              │
│                                                      │                     │
│       ┌──────────────────────────────────────────────┤                     │
│       │                                              │                     │
│       ▼                                              ▼                     │
│   ┌─────────┐                                ┌─────────────┐              │
│   │  FATE   │ ← ─ ─ ─ ─ ─ Validation ─ ─ ─ ─ │   Result    │              │
│   │  Gates  │                                │  Synthesis  │              │
│   └────┬────┘                                └──────┬──────┘              │
│        │                                            │                     │
│        ▼ Pass                                       │                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐       │                     │
│   │ Memory  │ ←  │ Format  │ ←  │  Learn  │ ← ────┘                     │
│   │  Store  │    │ Output  │    │ Extract │                              │
│   └─────────┘    └────┬────┘    └─────────┘                              │
│                       │                                                   │
│                       ▼                                                   │
│                  USER OUTPUT                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Message Flow

### Input Processing

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INPUT PROCESSING PIPELINE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw Input                                                              │
│       │                                                                  │
│       ▼                                                                  │
│   ┌────────────────┐                                                    │
│   │ 1. TOKENIZE    │  Split into tokens, identify structure            │
│   └───────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│   ┌────────────────┐                                                    │
│   │ 2. CLASSIFY    │  Command? Question? Task? Conversation?           │
│   └───────┬────────┘                                                    │
│           │                                                              │
│           ├─────────────────────────────────────┐                       │
│           │ Slash Command                        │ Natural Language      │
│           ▼                                      ▼                       │
│   ┌────────────────┐                    ┌────────────────┐             │
│   │ 3a. PARSE CMD  │                    │ 3b. NLU PARSE  │             │
│   │    /cmd args   │                    │  Intent + Slots │             │
│   └───────┬────────┘                    └───────┬────────┘             │
│           │                                      │                       │
│           └──────────────┬───────────────────────┘                       │
│                          │                                               │
│                          ▼                                               │
│   ┌────────────────────────────────────┐                               │
│   │ 4. ENRICH CONTEXT                   │                               │
│   │    • Add recent history             │                               │
│   │    • Add relevant memories          │                               │
│   │    • Add current state              │                               │
│   │    • Add agent context              │                               │
│   └───────────────┬────────────────────┘                               │
│                   │                                                      │
│                   ▼                                                      │
│   ┌────────────────────────────────────┐                               │
│   │ 5. ROUTE TO AGENT                   │                               │
│   │    Match capability → Select agent  │                               │
│   └────────────────────────────────────┘                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Message Structure

```rust
struct Message {
    id: Uuid,
    timestamp: DateTime<Utc>,

    // Content
    content: MessageContent,

    // Routing
    sender: AgentId,
    recipient: Option<AgentId>,

    // Context
    context: MessageContext,

    // Metadata
    metadata: HashMap<String, Value>,
}

struct MessageContent {
    raw: String,
    parsed: ParsedContent,
    intent: Option<Intent>,
    entities: Vec<Entity>,
}

struct MessageContext {
    conversation_id: Uuid,
    parent_message_id: Option<Uuid>,
    session_context: SessionContext,
    memory_context: Vec<MemoryItem>,
    task_context: Option<TaskContext>,
}
```

---

## Inference Pipeline

### LLM Request Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Enriched Message                                                       │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────┐                                                  │
│   │ PROMPT BUILDER   │  Construct prompt from template + context        │
│   │                  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ System     │  │  Agent personality + FATE constraints           │
│   │  │ Prompt     │  │                                                  │
│   │  └────────────┘  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ Context    │  │  Memory + history + current state               │
│   │  │ Block      │  │                                                  │
│   │  └────────────┘  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ User       │  │  The actual request                             │
│   │  │ Message    │  │                                                  │
│   │  └────────────┘  │                                                  │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ BACKEND SELECTOR │  Choose: LM Studio → Ollama → Fallback         │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ INFERENCE ENGINE │                                                  │
│   │                  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ Rate Limit │  │  Respect backend limits                         │
│   │  └────────────┘  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ Request    │  │  Make API call with timeout                     │
│   │  └────────────┘  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ Streaming? │──┼──→ Stream tokens to UI                          │
│   │  └────────────┘  │                                                  │
│   │  ┌────────────┐  │                                                  │
│   │  │ Response   │  │  Parse and structure response                   │
│   │  └────────────┘  │                                                  │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ POST-PROCESS     │  Extract structured data, code blocks, etc.     │
│   └──────────────────┘                                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Backend Selection Logic

```rust
fn select_backend(task: &Task) -> Backend {
    // 1. Check primary backend (LM Studio)
    if lm_studio.is_available() && lm_studio.supports(task.model_requirements) {
        return Backend::LMStudio;
    }

    // 2. Fall back to Ollama
    if ollama.is_available() && ollama.supports(task.model_requirements) {
        return Backend::Ollama;
    }

    // 3. Check federation pool
    if task.allows_federation && federation.has_capacity() {
        return Backend::Federation;
    }

    // 4. Queue for later
    Backend::Queued
}
```

---

## Memory System

### Memory Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          MEMORY SYSTEM                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                      WORKING MEMORY                                │ │
│   │   • Current conversation context                                   │ │
│   │   • Active task state                                              │ │
│   │   • Recent files/edits                                             │ │
│   │   Capacity: 100K tokens │ TTL: Session                            │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼ Summarize                                │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                     SHORT-TERM MEMORY                              │ │
│   │   • Session summaries                                              │ │
│   │   • Recent learnings                                               │ │
│   │   • Temporary preferences                                          │ │
│   │   Capacity: 10K items │ TTL: 7 days                               │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼ Consolidate                              │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                      LONG-TERM MEMORY                              │ │
│   │                                                                    │ │
│   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │ │
│   │   │  Decisions   │   │  Learnings   │   │  Patterns    │        │ │
│   │   │  (Permanent) │   │  (Permanent) │   │  (1 year)    │        │ │
│   │   └──────────────┘   └──────────────┘   └──────────────┘        │ │
│   │                                                                    │ │
│   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │ │
│   │   │  Contacts    │   │   Projects   │   │ Preferences  │        │ │
│   │   │  (Permanent) │   │  (Permanent) │   │  (Permanent) │        │ │
│   │   └──────────────┘   └──────────────┘   └──────────────┘        │ │
│   │                                                                    │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                      VECTOR INDEX                                  │ │
│   │   Model: all-MiniLM-L6-v2 │ Dimensions: 384 │ Similarity: Cosine │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Memory Operations

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       MEMORY OPERATIONS                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STORE                           RECALL                                 │
│   ─────                           ──────                                 │
│                                                                          │
│   Content                         Query                                  │
│      │                               │                                   │
│      ▼                               ▼                                   │
│   ┌────────┐                    ┌────────┐                              │
│   │ Embed  │                    │ Embed  │                              │
│   └───┬────┘                    └───┬────┘                              │
│       │                             │                                    │
│       ▼                             ▼                                    │
│   ┌────────┐                    ┌────────┐                              │
│   │Classify│                    │ Search │                              │
│   │Category│                    │ Index  │                              │
│   └───┬────┘                    └───┬────┘                              │
│       │                             │                                    │
│       ▼                             ▼                                    │
│   ┌────────┐                    ┌────────┐                              │
│   │ Store  │                    │ Rank   │                              │
│   │  Data  │                    │Results │                              │
│   └───┬────┘                    └───┬────┘                              │
│       │                             │                                    │
│       ▼                             ▼                                    │
│   ┌────────┐                    ┌────────┐                              │
│   │ Index  │                    │ Filter │                              │
│   │ Vector │                    │ By Ctx │                              │
│   └────────┘                    └────────┘                              │
│                                      │                                   │
│                                      ▼                                   │
│                                  Memories                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## FATE Validation

### Validation Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       FATE VALIDATION PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Agent Output                                                           │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      PARALLEL EVALUATION                         │   │
│   │                                                                  │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │   │
│   │   │  Ihsān   │  │   Adl    │  │   Harm   │  │Confidence│      │   │
│   │   │Evaluator │  │Evaluator │  │Evaluator │  │Evaluator │      │   │
│   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │   │
│   │        │             │             │             │              │   │
│   │        ▼             ▼             ▼             ▼              │   │
│   │      0.97          0.28          0.12          0.91            │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      THRESHOLD CHECK                             │   │
│   │                                                                  │   │
│   │   Ihsān:      0.97 ≥ 0.95  ✓ PASS                              │   │
│   │   Adl:        0.28 ≤ 0.35  ✓ PASS                              │   │
│   │   Harm:       0.12 ≤ 0.30  ✓ PASS                              │   │
│   │   Confidence: 0.91 ≥ 0.80  ✓ PASS                              │   │
│   │                                                                  │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                        │
│               ┌────────────────┼────────────────┐                      │
│               │ All Pass       │ Any Fail       │                      │
│               ▼                ▼                │                      │
│        ┌──────────┐     ┌──────────┐          │                      │
│        │ Release  │     │ Handle   │          │                      │
│        │ Output   │     │ Failure  │          │                      │
│        └──────────┘     └────┬─────┘          │                      │
│                              │                 │                      │
│               ┌──────────────┼────────────────┼──────────────┐      │
│               │              │                │              │      │
│               ▼              ▼                ▼              ▼      │
│          ┌────────┐    ┌────────┐      ┌────────┐    ┌────────┐   │
│          │ Ihsān  │    │  Adl   │      │  Harm  │    │Confid. │   │
│          │ Retry  │    │Rebalance│      │ BLOCK  │    │Disclaim│   │
│          └────────┘    └────────┘      └────────┘    └────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### FATE Scores Calculation

```rust
struct FATEScores {
    ihsan: f64,      // Excellence (accuracy, completeness, coherence, etc.)
    adl: f64,        // Justice (Gini coefficient of resource distribution)
    harm: f64,       // Harm potential (max of harm categories)
    confidence: f64, // Certainty (model confidence, source reliability, etc.)
}

impl FATEScores {
    fn calculate_ihsan(output: &Output) -> f64 {
        let accuracy = evaluate_accuracy(output);      // 0.30 weight
        let completeness = evaluate_completeness(output); // 0.25 weight
        let coherence = evaluate_coherence(output);    // 0.20 weight
        let relevance = evaluate_relevance(output);    // 0.15 weight
        let clarity = evaluate_clarity(output);        // 0.10 weight

        accuracy * 0.30 + completeness * 0.25 + coherence * 0.20
            + relevance * 0.15 + clarity * 0.10
    }

    fn calculate_adl(context: &Context) -> f64 {
        gini_coefficient(&context.resource_distribution)
    }

    fn calculate_harm(output: &Output) -> f64 {
        let physical = assess_physical_harm(output);
        let psychological = assess_psychological_harm(output);
        let financial = assess_financial_harm(output);
        let privacy = assess_privacy_risk(output);
        let security = assess_security_risk(output);

        [physical, psychological, financial, privacy, security]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
    }

    fn calculate_confidence(output: &Output) -> f64 {
        let model_conf = output.model_confidence;      // 0.40 weight
        let source_rel = evaluate_sources(output);     // 0.30 weight
        let consistency = check_consistency(output);   // 0.20 weight
        let verification = verification_level(output); // 0.10 weight

        model_conf * 0.40 + source_rel * 0.30
            + consistency * 0.20 + verification * 0.10
    }
}
```

---

## Agent Communication

### A2A Message Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     AGENT-TO-AGENT COMMUNICATION                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐         ┌──────────────┐         ┌──────────┐           │
│   │ Agent A  │         │   A2A Bus    │         │ Agent B  │           │
│   │(Sender)  │         │              │         │(Receiver)│           │
│   └────┬─────┘         └──────┬───────┘         └────┬─────┘           │
│        │                      │                      │                  │
│        │  1. Create TaskCard  │                      │                  │
│        │─────────────────────>│                      │                  │
│        │                      │                      │                  │
│        │                      │ 2. Validate Schema   │                  │
│        │                      │ 3. Check ACL         │                  │
│        │                      │ 4. Route by Capability                 │
│        │                      │                      │                  │
│        │                      │  5. Deliver TaskCard │                  │
│        │                      │─────────────────────>│                  │
│        │                      │                      │                  │
│        │                      │                      │ 6. Process Task  │
│        │                      │                      │                  │
│        │                      │  7. Return Result    │                  │
│        │                      │<─────────────────────│                  │
│        │                      │                      │                  │
│        │                      │ 8. FATE Validate     │                  │
│        │                      │                      │                  │
│        │  9. Receive Result   │                      │                  │
│        │<─────────────────────│                      │                  │
│        │                      │                      │                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Execution

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE EXECUTION FLOW                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Pipeline: feature_development                                          │
│                                                                          │
│   Stage 1          Stage 2          Stage 3          Stage 4            │
│   ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐             │
│   │Rsrch │───────>│ Dev  │───────>│Review│───────>│Guard │             │
│   └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘             │
│      │               │               │               │                  │
│      ▼               ▼               ▼               ▼                  │
│   ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐             │
│   │Output│───────>│Output│───────>│Output│───────>│Output│             │
│   │  A   │        │  B   │        │  C   │        │ Final│             │
│   └──────┘        └──────┘        └──────┘        └──────┘             │
│                                                                          │
│   Each stage:                                                            │
│   1. Receives previous output as context                                 │
│   2. Processes according to task card                                    │
│   3. Validates output through FATE                                       │
│   4. Passes to next stage or returns final                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Federation Data Flow

### Node Discovery

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      FEDERATION NODE DISCOVERY                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐                                        ┌─────────┐        │
│   │ Node 0  │                                        │ Node N  │        │
│   │ (Self)  │                                        │ (Peer)  │        │
│   └────┬────┘                                        └────┬────┘        │
│        │                                                  │             │
│        │  1. Gossip: "I'm alive"                         │             │
│        │─────────────────────────────────────────────────>│             │
│        │                                                  │             │
│        │  2. Gossip: "I'm alive + known peers"           │             │
│        │<─────────────────────────────────────────────────│             │
│        │                                                  │             │
│        │  3. Capability Exchange                          │             │
│        │<─────────────────────────────────────────────────>             │
│        │                                                  │             │
│        │  4. Add to Peer Table                           │             │
│        │                                                  │             │
│                                                                          │
│   Peer Table:                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Node ID    │ Address        │ Capabilities    │ Last Seen     │   │
│   ├─────────────────────────────────────────────────────────────────┤   │
│   │ node_abc   │ 192.168.1.10   │ [inference]     │ 2s ago        │   │
│   │ node_def   │ 192.168.1.20   │ [storage]       │ 5s ago        │   │
│   │ node_ghi   │ 10.0.0.5       │ [compute]       │ 1s ago        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Consensus Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         PBFT CONSENSUS FLOW                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Client    Primary    Replica1   Replica2   Replica3                   │
│     │          │          │          │          │                       │
│     │ Request  │          │          │          │                       │
│     │─────────>│          │          │          │                       │
│     │          │          │          │          │                       │
│     │          │ Pre-prepare (with request)     │                       │
│     │          │─────────>│─────────>│─────────>│                       │
│     │          │          │          │          │                       │
│     │          │    Prepare (from all replicas) │                       │
│     │          │<─────────│<─────────│<─────────│                       │
│     │          │─────────>│─────────>│─────────>│                       │
│     │          │          │          │          │                       │
│     │          │     Commit (2f+1 prepares)     │                       │
│     │          │<─────────│<─────────│<─────────│                       │
│     │          │─────────>│─────────>│─────────>│                       │
│     │          │          │          │          │                       │
│     │ Reply    │          │          │          │                       │
│     │<─────────│<─────────│<─────────│<─────────│                       │
│     │          │          │          │          │                       │
│                                                                          │
│   f = 1 (tolerates 1 Byzantine failure with 4 nodes)                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Caching Strategy

### Cache Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CACHING LAYERS                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                         L1: IN-MEMORY                              │ │
│   │   • Recent embeddings      TTL: 5 min    Size: 1GB               │ │
│   │   • Prompt templates       TTL: 60 min   Size: 100MB             │ │
│   │   • Agent states           TTL: Session  Size: 10MB              │ │
│   │   Hit Rate Target: 80%                                            │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │ Miss                                     │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                         L2: DISK CACHE                             │ │
│   │   • Vector index chunks    TTL: 24h      Size: 10GB              │ │
│   │   • Research cache         TTL: 7d       Size: 5GB               │ │
│   │   • Model responses        TTL: 1h       Size: 2GB               │ │
│   │   Hit Rate Target: 60%                                            │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │ Miss                                     │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                       L3: FEDERATED                                │ │
│   │   • Shared knowledge       TTL: Varies   Size: Distributed       │ │
│   │   • Cross-node patterns    TTL: 30d      Size: Distributed       │ │
│   │   Hit Rate Target: 30%                                            │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │ Miss                                     │
│                              ▼                                          │
│                         ORIGIN (Compute)                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

### Error Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ERROR HANDLING FLOW                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Error Occurs                                                           │
│        │                                                                 │
│        ▼                                                                 │
│   ┌────────────────┐                                                    │
│   │ CLASSIFY ERROR │                                                    │
│   └───────┬────────┘                                                    │
│           │                                                              │
│   ┌───────┼───────────────────────────────────────┐                    │
│   │       │               │               │       │                    │
│   ▼       ▼               ▼               ▼       ▼                    │
│ Transient  Network      Backend       FATE      Fatal                  │
│   │         │             │           Fail        │                    │
│   │         │             │             │         │                    │
│   ▼         ▼             ▼             ▼         ▼                    │
│ ┌─────┐  ┌─────┐      ┌─────┐      ┌─────┐   ┌─────┐                 │
│ │Retry│  │Retry│      │Switch│     │Handle│   │Log  │                 │
│ │Exp. │  │With │      │Backend│    │Gate  │   │Alert│                 │
│ │Back.│  │Diff.│      │      │     │Fail  │   │Abort│                 │
│ └──┬──┘  │Peer │      └──┬───┘     └──┬───┘   └──┬──┘                 │
│    │     └──┬──┘         │            │          │                    │
│    │        │            │            │          │                    │
│    └────────┴────────────┴────────────┘          │                    │
│                   │                               │                    │
│                   ▼                               │                    │
│            ┌─────────────┐                       │                    │
│            │ Max Retries │                       │                    │
│            │  Exceeded?  │                       │                    │
│            └──────┬──────┘                       │                    │
│                   │ Yes                           │                    │
│                   ▼                               │                    │
│            ┌─────────────┐                       │                    │
│            │  Escalate   │<──────────────────────┘                    │
│            │  to Human   │                                            │
│            └─────────────┘                                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

**Data flows like water — design for the current.** 🌊
