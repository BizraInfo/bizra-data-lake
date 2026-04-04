# NODE0 ACTIVATION KERNEL — Formal Specification
## The Bridge from Architecture to Civilization-Engine
**Version:** 1.0.0
**Date:** 2026-03-29
**Truth Label:** VALIDATED
**Canonical Status:** DRAFT → targeting PROVEN on first end-to-end mission
**Authority Chain:** DECLARATION → SYSTEM_INSTRUCTION_CHAIN → BIZRA_KERNEL_SPEC → this document

---

## 0. Why This Exists

The kernel spec defines the constitutional checkpoint — what REFUSES.
This spec defines the sovereign operating heartbeat — what LIVES.

The real system is not:
```
chatbot → tools → response
```
The real system is:
```
human intent → PAT inside node → FATE/admission → SAT inside URP
  → evidence ledger → persistent character sheet
  → reflex/progression → network
```

That is the category jump. This document specifies the 7 modules that
make it real.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NODE0 ACTIVATION KERNEL                    │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   GENESIS    │  │  CHARACTER   │  │  TRUTH REGISTRY   │   │
│  │  ACTIVATION  │  │    SHEET     │  │  (claim status)   │   │
│  │  (one-time)  │  │ (persistent) │  │                   │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘   │
│         │                 │                    │              │
│  ┌──────▼─────────────────▼────────────────────▼──────────┐  │
│  │              MISSION STATE MACHINE                      │  │
│  │  IDLE→BRIEFED→DECOMPOSED→EXECUTING→GATED→EVIDENCED→RET │  │
│  └──────┬──────────────────────────────┬──────────────────┘  │
│         │                              │                     │
│  ┌──────▼───────┐  ┌──────────────────▼───────────────┐     │
│  │  EVALUATOR   │  │         ACTION BUS               │     │
│  │  ADMISSION   │  │  (permissioned execution)        │     │
│  │  (IHSAN gate)│  │  ┌───────────────────────────┐   │     │
│  └──────────────┘  │  │     EVENT BUS (observe)   │   │     │
│                    │  └───────────────────────────┘   │     │
│                    └──────────────┬────────────────────┘     │
│                                  │                           │
│                    ┌─────────────▼────────────────┐          │
│                    │    EVIDENCE & REPLAY          │          │
│                    │  (receipts, bundles, replay)  │          │
│                    └──────────────────────────────┘          │
│                                                               │
│  ═══════════════════════════════════════════════════════════  │
│  │              BIZRA-KERNEL (ICS) — underneath             │ │
│  ═══════════════════════════════════════════════════════════  │
└─────────────────────────────────────────────────────────────┘
```
---

# MODULE 1: genesis_activation

## Purpose
One-time, irreversible activation of Node0. Converts vision into state.
Once executed, genesis cannot be re-run. The node IS activated, permanently.

## Schema

```rust
struct GenesisActivation {
    // Immutable genesis record — written once, never modified
    genesis_id:          UUID,           // Unique activation identifier
    activated_at:        ISO8601,        // Timestamp of genesis moment
    activated_by:        NodeIdentity,   // First Architect's public key
    genesis_hash:        BLAKE3,         // Hash of this entire struct (self-referential seal)

    // Authority grants
    first_architect:     ArchitectGrant, // Momo's founder authority
    pat_roster:          [AgentID; 7],   // PAT-7 minted at genesis
    sat_roster:          [AgentID; 5],   // SAT-5 minted at genesis
    urp_id:              UUID,           // Universal Resource Plane created

    // Constitutional binding
    kernel_fingerprint:  BLAKE3,         // Hash of bizra-kernel binary at activation time
    constitution_hash:   BLAKE3,         // Hash of 00_CONSTITUTION/ directory at activation
    invariant_snapshot:  InvariantSet,   // Frozen anchors committed at genesis

    // Genesis evidence
    genesis_evidence:    EvidenceBinding, // First evidence entry in the ledger
    genesis_manifest:    ManifestRef,     // Links to MANIFEST_001
}

struct ArchitectGrant {
    identity:            NodeIdentity,   // Ed25519 public key
    title:               "First Architect",
    authority_level:     AuthorityLevel::Founder,
    exemptions:          Vec<FounderExemption>,  // Explicit, enumerated
    granted_at:          ISO8601,
    grant_seal:          Signature,      // Kernel-signed
}

enum FounderExemption {
    BypassPhaseGate { reason: String, expires: Option<ISO8601> },
    ManualTruthLabelOverride { scope: String },
    EmergencyKernelConfig { field: String },
    // Every exemption MUST have a reason and optional expiry
    // No blanket exemptions. No "founder can do anything."
}
```

## State Transitions

```
UNACTIVATED ──[genesis_activate()]──→ ACTIVATED
     │                                    │
     │  (no other path exists)            │  (irreversible)
     │                                    │
     └── CANNOT return to UNACTIVATED ────┘
```

## Activation Protocol

```
STEP 1: VERIFY PRECONDITIONS
  assert kernel.is_running()
  assert node.state == UNACTIVATED
  assert identity_key.exists()
  assert constitution_dir.is_complete()  // All canonical files present

STEP 2: MINT AGENTS
  pat_7 = mint_agents([
    Agent::Strategist,    // PAT-1: Goal decomposition, planning
    Agent::Analyst,       // PAT-2: Data analysis, pattern recognition
    Agent::Creative,      // PAT-3: Synthesis, innovation, alternatives
    Agent::Technical,     // PAT-4: Implementation, architecture
    Agent::Ethical,       // PAT-5: FATE gate, moral reasoning
    Agent::Social,        // PAT-6: Communication, empathy, UX
    Agent::Executive,     // PAT-7: Decision authority, final call
  ])
  sat_5 = mint_agents([
    Agent::Memory,        // SAT-1: Persistent state, recall, context
    Agent::Learning,      // SAT-2: Pattern extraction, skill building
    Agent::Communication, // SAT-3: Protocol handling, message routing
    Agent::Monitoring,    // SAT-4: Health, metrics, anomaly detection
    Agent::Integration,   // SAT-5: External systems, federation, URP
  ])
STEP 3: CREATE UNIVERSAL RESOURCE PLANE
  urp = URP::create(
    owner: first_architect.identity,
    pat_roster: pat_7,
    sat_roster: sat_5,
    kernel_ref: kernel.fingerprint(),
  )

STEP 4: STAMP GENESIS EVIDENCE
  genesis_evidence = EvidenceBinding {
    claim_id: uuid(),
    claim_text: "Node0 activated by First Architect",
    sources: [Source {
      uri: "local://genesis_activation",
      content_hash: blake3(activation_payload),
      method: ExtractionMethod::DirectRead,
    }],
    confidence: 1.0,  // Genesis is self-evident
    attester: kernel.process_id(),
    kernel_seal: kernel.sign(activation_payload),
  }

STEP 5: WRITE GENESIS RECORD
  genesis = GenesisActivation { /* all fields populated */ }
  genesis.genesis_hash = blake3(genesis)  // Self-seal
  persist(genesis, path: "node0/genesis.bin")
  // This file is append-only, read-many, write-ONCE

STEP 6: INITIALIZE CHARACTER SHEET
  character_sheet = CharacterSheet::from_genesis(genesis)
  persist(character_sheet, path: "node0/character.bin")

STEP 7: SET NODE STATE
  node.state = ACTIVATED
  audit_log.write(GenesisComplete { genesis_id, timestamp })
  mission_state_machine.transition(IDLE)
```

## Acceptance Criteria

- [ ] Genesis can execute exactly once. Second call returns `AlreadyActivated` error.
- [ ] PAT-7 agents are persisted with unique AgentIDs that survive restart.
- [ ] SAT-5 agents are persisted with unique AgentIDs that survive restart.
- [ ] URP is created with references to both rosters.
- [ ] Genesis evidence entry has kernel_seal signature.
- [ ] Genesis record is BLAKE3 self-sealed.
- [ ] `genesis.bin` is write-once: any modification attempt fails.
- [ ] All founder exemptions are explicit and enumerated (no wildcards).
- [ ] Node state transitions from UNACTIVATED → ACTIVATED irreversibly.
- [ ] Character sheet is initialized and persisted from genesis data.
---

# MODULE 2: character_sheet

## Purpose
Persistent node/user state that solves session amnesia. The character sheet
IS the node. When the node restarts, the character sheet restores identity.
When a year passes, the character sheet records progression. The current
context window is just the CAMERA. The character sheet is the WORLD.

## Schema

```rust
struct CharacterSheet {
    // === IDENTITY (immutable after genesis) ===
    node_id:             UUID,
    genesis_ref:         GenesisRef,        // Points to genesis.bin
    identity:            NodeIdentity,      // Ed25519 keypair reference
    first_architect:     ArchitectGrant,
    sovereignty_tier:    SovereigntyTier,   // Solo | Federated | Networked

    // === AGENT ROSTERS (minted at genesis, evolved over time) ===
    pat_roster:          PatRoster,         // 7 Primary Agents of Thought
    sat_roster:          SatRoster,         // 5 Support Agents of Thought

    // === PROGRESSION (updated after every mission) ===
    ihsan_score:         f64,               // Current composite IHSAN (0.0–1.0)
    snr_score:           f64,               // Signal-to-noise ratio of outputs
    reputation:          ReputationLedger,  // Per-domain reputation tracking
    missions_completed:  u64,               // Total missions returned
    missions_failed:     u64,               // Missions that didn't reach RETURNED
    evidence_count:      u64,               // Total evidence bindings created
    streaks:             StreakTracker,      // Consecutive successes, daily activity

    // === ECONOMIC STATE (placeholders for Phase 2) ===
    seed_balance:        u64,               // SEED tokens (transferable utility)
    bloom_balance:       u64,               // BLOOM tokens (soulbound governance)
    zakat_due:           f64,               // Accumulated purification obligation
    gini_contribution:   f64,               // This node's effect on network Gini

    // === MISSION STATE (live) ===
    active_mission:      Option<MissionRef>,  // Current mission if any
    mission_history:     Vec<MissionDigest>,  // Completed mission summaries
    last_mission_at:     Option<ISO8601>,

    // === MEMORY & CONTEXT ===
    memory_root:         BLAKE3,            // Hash of persistent memory store
    last_context_hash:   BLAKE3,            // Hash of last context snapshot
    known_skills:        Vec<SkillID>,      // Learned/registered capabilities
    active_quests:       Vec<QuestRef>,     // Long-running multi-mission arcs

    // === META ===
    created_at:          ISO8601,
    last_updated_at:     ISO8601,
    version:             u32,               // Schema version for migration
    checksum:            BLAKE3,            // Self-integrity hash
}

struct PatRoster {
    agents: [PatAgent; 7],
    authority_model: AuthorityModel::Democratic,  // PAT agents deliberate
}

struct PatAgent {
    id:                  AgentID,
    role:                PatRole,           // Strategist|Analyst|Creative|Technical|Ethical|Social|Executive
    activation_count:    u64,
    last_activated:      Option<ISO8601>,
    specialization_data: HashMap<String, f64>,  // Domain expertise scores
}

struct SatRoster {
    agents: [SatAgent; 5],
    authority_model: AuthorityModel::Delegated,  // SAT agents serve PAT
}

struct SatAgent {
    id:                  AgentID,
    role:                SatRole,           // Memory|Learning|Communication|Monitoring|Integration
    activation_count:    u64,
    last_activated:      Option<ISO8601>,
    health_status:       HealthStatus,      // Healthy | Degraded | Offline
}
struct ReputationLedger {
    domains: HashMap<String, DomainReputation>,
    // e.g., "technical_architecture": DomainReputation { score: 0.87, evidence_count: 42 }
    // e.g., "ethical_reasoning": DomainReputation { score: 0.93, evidence_count: 18 }
    global_reputation:   f64,              // Weighted average across domains
}

struct StreakTracker {
    current_daily:       u32,              // Consecutive days with ≥1 completed mission
    best_daily:          u32,              // All-time best daily streak
    current_quality:     u32,              // Consecutive missions with IHSAN ≥ 0.95
    best_quality:        u32,
}

enum SovereigntyTier {
    Solo,                // Node operates independently, no network
    Federated,           // Node participates in federation, retains sovereignty
    Networked,           // Node is part of full BIZRA network (Phase 3+)
}
```

## Persistence Contract

```
WRITE RULES:
  - Character sheet is persisted to `node0/character.bin` after EVERY state change
  - Before write: compute new checksum = blake3(sheet_without_checksum)
  - Write is atomic: write to temp file, fsync, rename (same pattern as kernel)
  - Old versions are kept as `node0/character.bin.{version}` for rollback

READ RULES:
  - On node boot: load character.bin, verify checksum
  - If checksum fails: attempt rollback to previous version
  - If all versions corrupt: PANIC — node identity compromised

SURVIVAL TEST:
  - Kill the process at any point during a mission
  - Restart the node
  - Character sheet must restore to last consistent state
  - Active mission should be in GATED or IDLE (never mid-write)
```

## Acceptance Criteria

- [ ] Character sheet survives node restart with zero data loss
- [ ] Checksum validation detects any tampering or corruption
- [ ] PAT-7 and SAT-5 rosters persist with all metadata across restarts
- [ ] Mission history accumulates — no session amnesia
- [ ] Progression metrics (IHSAN, SNR, reputation) update after each mission
- [ ] Economic placeholders exist but are inert until Phase 2 activation
- [ ] Schema version field enables future migration without data loss
- [ ] Streak tracker accurately counts consecutive activity
---

# MODULE 3: mission_state_machine

## Purpose
The OS heartbeat. Every user intent becomes a mission. Every mission flows
through a deterministic 7-state lifecycle. This is NOT a chat loop — it is
a quest-state machine. The mission state machine is what makes BIZRA a DDAGI
OS instead of a chatbot.

## States

```
┌──────┐    brief()     ┌─────────┐   decompose()   ┌──────────────┐
│ IDLE │───────────────→│ BRIEFED │────────────────→│ DECOMPOSED   │
└──┬───┘                └─────────┘                  └──────┬───────┘
   │                                                        │
   │  ┌─────────────────────────────────────────────────────┘
   │  │  execute()
   │  ▼
   │  ┌───────────┐    gate()      ┌────────┐   evidence()  ┌───────────┐
   │  │ EXECUTING │──────────────→│ GATED  │──────────────→│ EVIDENCED │
   │  └───────────┘               └────────┘               └─────┬─────┘
   │                                   │                         │
   │                          reject() │                  return()│
   │                                   ▼                         ▼
   │                           ┌──────────┐              ┌──────────┐
   │                           │ REJECTED │              │ RETURNED │
   │                           └────┬─────┘              └────┬─────┘
   │                                │                         │
   └────────────────────────────────┴─────────────────────────┘
                              reset to IDLE
                        (character sheet updated)
```

## State Definitions

```rust
enum MissionState {
    Idle,
    Briefed,
    Decomposed,
    Executing,
    Gated,
    Evidenced,
    Returned,
    Rejected,  // Terminal failure state (also returns to IDLE)
}

struct Mission {
    mission_id:       UUID,
    created_at:       ISO8601,
    state:            MissionState,
    state_history:    Vec<StateTransition>,  // Full audit trail of transitions

    // BRIEFED phase
    intent:           String,            // Human's original intent (natural language)
    intent_hash:      BLAKE3,            // Immutable hash of original intent
    pat_assignment:   Vec<AgentID>,      // Which PAT agents are assigned

    // DECOMPOSED phase
    task_dag:         TaskDAG,           // Directed acyclic graph of subtasks
    estimated_cost:   ResourceEstimate,  // Predicted resource consumption
    risk_assessment:  RiskLevel,         // Low | Medium | High | Critical

    // EXECUTING phase
    execution_log:    Vec<ActionRecord>, // Every action taken, receipted
    sat_assignment:   Vec<AgentID>,      // Which SAT agents are supporting

    // GATED phase
    gate_result:      Option<GateVerdict>,   // PASS | HOLD | REJECT
    evaluator_scores: Option<EvaluatorScores>,
    gate_evidence:    Vec<EvidenceBinding>,

    // EVIDENCED phase
    evidence_bundle:  Option<EvidenceBundle>,
    receipt:          Option<MissionReceipt>,

    // RETURNED phase
    output:           Option<MissionOutput>,     // Final deliverable to user
    replay_package:   Option<ReplayPackage>,      // Everything needed to replay
    character_delta:  Option<CharacterSheetDelta>, // Changes to character sheet
}
struct StateTransition {
    from:         MissionState,
    to:           MissionState,
    triggered_by: ProcessID,       // Which agent/process caused this
    timestamp:    ISO8601,
    reason:       String,
    receipt_id:   UUID,            // Every transition produces a receipt
}

struct TaskDAG {
    nodes:        Vec<Task>,
    edges:        Vec<(TaskID, TaskID)>,  // Dependency edges
    critical_path: Vec<TaskID>,          // Longest dependency chain
}

struct Task {
    task_id:      TaskID,
    description:  String,
    assigned_to:  AgentID,         // Which PAT/SAT agent owns this
    status:       TaskStatus,      // Pending | Running | Done | Failed
    requires:     Vec<Capability>, // Kernel capabilities needed
    evidence:     Vec<EvidenceBinding>,
}
```

## Transition Rules (The OS Heartbeat)

```
IDLE → BRIEFED:
  TRIGGER: Human submits intent
  ACTION:  Hash intent, assign PAT agents, record in mission
  GUARD:   Node must be ACTIVATED, no active mission (or allow queue)

BRIEFED → DECOMPOSED:
  TRIGGER: PAT-1 (Strategist) + PAT-4 (Technical) produce task DAG
  ACTION:  Validate DAG is acyclic, estimate resources, assess risk
  GUARD:   Task DAG must have ≥1 task, resource estimate within budget

DECOMPOSED → EXECUTING:
  TRIGGER: All tasks have assigned agents, kernel capabilities requested
  ACTION:  Begin task execution via Action Bus, assign SAT support
  GUARD:   All required capabilities granted by kernel (AUTH_GRANTED)

EXECUTING → GATED:
  TRIGGER: All tasks complete OR critical task fails OR timeout
  ACTION:  Submit execution results to evaluator_admission (Module 4)
  GUARD:   Execution log is non-empty, every action has receipt

GATED → EVIDENCED:
  TRIGGER: Evaluator returns PASS verdict
  ACTION:  Bundle all evidence, create mission receipt, build replay package
  GUARD:   IHSAN score ≥ 0.95, all claims evidence-bound

GATED → REJECTED:
  TRIGGER: Evaluator returns REJECT verdict
  ACTION:  Log rejection reason, attempt remediation or return to EXECUTING
  GUARD:   If max_retries exceeded, transition to REJECTED (terminal)
  NOTE:    REJECTED still updates character sheet (missions_failed++)

EVIDENCED → RETURNED:
  TRIGGER: Evidence bundle sealed, replay package complete
  ACTION:  Deliver output to user, update character sheet, persist all state
  GUARD:   Receipt exists, truth labels applied, character delta computed

RETURNED → IDLE:
  TRIGGER: Automatic after character sheet update
  ACTION:  Clear active_mission, increment missions_completed, update streaks
  GUARD:   Character sheet checksum valid after update

REJECTED → IDLE:
  TRIGGER: After rejection is logged and character sheet updated
  ACTION:  Increment missions_failed, reset streak if applicable
```

## Acceptance Criteria

- [ ] Every mission flows through the state machine — no shortcuts
- [ ] State transitions produce receipts (NO_SILENT_ACTION compliance)
- [ ] GATED state is mandatory — no output reaches user without evaluation
- [ ] State history is append-only audit trail of every transition
- [ ] Mission survives node restart in any state (resume from last transition)
- [ ] REJECTED missions still produce evidence (negative evidence is evidence)
- [ ] Task DAG prevents circular dependencies (validated at DECOMPOSED transition)
- [ ] Character sheet is updated atomically at RETURNED/REJECTED → IDLE
---

# MODULE 4: evaluator_admission

## Purpose
The symbolic-neural hinge. Until runtime admission depends on the evaluator
path, the constitutional system is split between theory and operation. This
module makes the evaluator the SOLE authoritative IHSAN source for live
admission. No output passes without evaluator verdict.

## Architecture

```
Mission output (from EXECUTING)
        │
        ▼
┌───────────────────────────────────────────────┐
│            EVALUATOR ADMISSION                │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │       Provider Registry                  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────┐ │  │
│  │  │Primary   │ │Secondary │ │Fallback │ │  │
│  │  │Evaluator │ │Evaluator │ │(kernel  │ │  │
│  │  │(LLM-as-  │ │(rule-    │ │default) │ │  │
│  │  │ judge)   │ │ based)   │ │         │ │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬────┘ │  │
│  │       │             │            │       │  │
│  │       ▼             ▼            ▼       │  │
│  │  ┌──────────────────────────────────┐    │  │
│  │  │    Verdict Precedence Engine     │    │  │
│  │  │  (resolves conflicting scores)   │    │  │
│  │  └──────────────┬───────────────────┘    │  │
│  └─────────────────┼───────────────────────┘  │
│                    │                           │
│           ┌────────▼────────┐                  │
│           │ PASS|HOLD|REJECT│                  │
│           └─────────────────┘                  │
└───────────────────────────────────────────────┘
```

## Schema

```rust
struct EvaluatorAdmission {
    provider_registry:  ProviderRegistry,
    fallback_hierarchy: Vec<ProviderID>,     // Ordered: try 1st, then 2nd, etc.
    verdict_precedence: VerdictPrecedenceRule,
    timeout_ms:         u64,                 // Default: 5000
    timeout_behavior:   TimeoutBehavior,     // Reject (fail-closed)
}

struct ProviderRegistry {
    providers:  Vec<EvaluatorProvider>,
    active:     Vec<ProviderID>,             // Currently available
}

struct EvaluatorProvider {
    id:              ProviderID,
    name:            String,
    provider_type:   ProviderType,           // LLMJudge | RuleBased | HumanReview | Hybrid
    endpoint:        ProviderEndpoint,       // IPC address or function reference
    capabilities:    Vec<ScoringDimension>,  // Which dimensions it can score
    trust_level:     f64,                    // 0.0–1.0, used in verdict precedence
    avg_latency_ms:  f64,                    // For routing decisions
    health:          HealthStatus,
}

enum ProviderType {
    LLMJudge,       // LLM-as-judge (planner/critic pattern)
    RuleBased,      // Deterministic rule engine (fast, limited)
    HumanReview,    // Human in the loop (slow, highest trust)
    Hybrid,         // Combination of above
}

struct EvaluatorScores {
    provider_id:     ProviderID,
    dimensions:      Vec<DimensionScore>,
    composite:       f64,                    // Weighted composite IHSAN score
    reasoning:       String,                 // Why these scores (evidence-bound)
    evidence_refs:   Vec<EvidenceRef>,       // What evidence was evaluated
    scored_at:       ISO8601,
    verdict:         GateVerdict,
}
struct DimensionScore {
    dimension:   ScoringDimension,
    score:       f64,                        // 0.0–1.0
    weight:      f64,                        // From kernel IHSAN config
    reasoning:   String,                     // Per-dimension justification
}

enum ScoringDimension {
    Truthfulness,    // 0.30 weight — alignment with bound evidence
    HarmAvoidance,   // 0.25 weight — could output cause harm
    Fairness,        // 0.20 weight — equitable across affected parties
    Transparency,    // 0.15 weight — reasoning visible and traceable
    Beneficence,     // 0.10 weight — actively helps user's stated goal
}

enum GateVerdict {
    Pass,            // Composite ≥ 0.95, no dimension < 0.60
    Hold,            // 0.80 ≤ composite < 0.95 — needs improvement or human review
    Reject,          // Composite < 0.80 OR any dimension < 0.60
}

enum VerdictPrecedenceRule {
    StrictestWins,        // If ANY provider says Reject, it's Reject
    WeightedConsensus,    // Weighted by provider trust_level
    PrimaryWithFallback,  // Use primary unless unavailable, then fallback
}
```

## Verdict Precedence Logic

```
fn resolve_verdict(scores: Vec<EvaluatorScores>, rule: VerdictPrecedenceRule) -> GateVerdict {
    match rule {
        StrictestWins => {
            if scores.any(|s| s.verdict == Reject) { return Reject }
            if scores.any(|s| s.verdict == Hold)   { return Hold }
            return Pass
        }
        PrimaryWithFallback => {
            let primary = scores.find(|s| s.provider_id == primary_id);
            match primary {
                Some(s) => s.verdict,
                None    => {
                    // Primary unavailable — use fallback chain
                    for fallback_id in fallback_hierarchy {
                        if let Some(s) = scores.find(|s| s.provider_id == fallback_id) {
                            return s.verdict
                        }
                    }
                    Reject  // No provider available — fail-closed
                }
            }
        }
    }
}
```

## Critical Design Decision: Policy vs Kernel Reasons

```rust
enum RejectionReason {
    // KERNEL reasons — frozen invariant violations, non-negotiable
    KernelInvariantViolation { invariant: InvariantID, detail: String },

    // POLICY reasons — evaluator-determined, may evolve
    PolicyViolation { policy: String, dimension: ScoringDimension, score: f64 },

    // TIMEOUT reasons — provider unavailable
    EvaluatorTimeout { provider: ProviderID, elapsed_ms: u64 },
}
// Kernel reasons ALWAYS take precedence over policy reasons.
// If the kernel says Reject (invariant violation), evaluator scores are irrelevant.
// If the kernel says Pass (no invariant violation), evaluator scores determine admission.
```

## Acceptance Criteria

- [ ] 100% of mission outputs pass through evaluator_admission — zero bypass
- [ ] Provider registry supports hot-swap (add/remove providers without restart)
- [ ] Fallback hierarchy activates automatically when primary is unavailable
- [ ] Timeout defaults to Reject (fail-closed), not Pass
- [ ] Kernel invariant violations override evaluator verdicts
- [ ] Every verdict includes reasoning that is itself evidence-bound
- [ ] Scoring dimensions and weights match kernel IHSAN config exactly
- [ ] Human review provider can be registered for HOLD queue processing
---

# MODULE 5: action_bus

## Purpose
The critical separation that every mature system needs but most AI systems
lack. The Event Bus and Action Bus serve fundamentally different purposes
and MUST NOT be conflated.

**Event Bus** = observation, telemetry, orchestration, status. Read-only intent.
Things that HAPPENED. Subscriptions, notifications, metrics.

**Action Bus** = permissioned execution intents, ordered commits, audit-grade
side effects. Write intent. Things that WILL HAPPEN, subject to authorization.

Collapsing them means you can't distinguish "I saw X happen" from
"I caused X to happen." That distinction IS constitutional enforcement.

## Dual Bus Architecture

```
┌────────────────────────────────────────────────────────┐
│                    ACTION BUS                           │
│  (write intent — every message requires kernel auth)    │
│                                                         │
│  ActionIntent → Kernel AUTH_REQUEST → Permit/Deny       │
│       │                                                 │
│       ▼ (if permitted)                                  │
│  ActionCommit → Execute → ActionReceipt                 │
│       │                                                 │
│       ▼ (always)                                        │
│  ActionReceipt → Evidence Ledger + Event Bus            │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    EVENT BUS                            │
│  (read intent — no authorization required to observe)   │
│                                                         │
│  Sources: ActionReceipts, StateTransitions, Metrics,    │
│           Heartbeats, ExternalSignals                   │
│                                                         │
│  Consumers: Monitoring (SAT-4), Learning (SAT-2),       │
│             Mission State Machine, Dashboard, Logs      │
└────────────────────────────────────────────────────────┘
```

## Schema

```rust
// === ACTION BUS ===

struct ActionIntent {
    intent_id:      UUID,
    mission_id:     UUID,                // Which mission this action belongs to
    task_id:        TaskID,              // Which task in the DAG
    requested_by:   AgentID,             // Which PAT/SAT agent wants this
    action_type:    ActionType,
    parameters:     HashMap<String, Value>,
    required_caps:  Vec<Capability>,     // Capabilities needed from kernel
    timestamp:      ISO8601,
    ordering:       SequenceNumber,      // Monotonic — actions have total ordering
}

enum ActionType {
    FileRead { path: String },
    FileWrite { path: String, content: Bytes },
    NetworkRequest { url: String, method: HttpMethod },
    BrowserNavigate { url: String },
    LLMInvoke { model: String, prompt: String },
    AgentMessage { target: AgentID, payload: Bytes },
    EconomicTransaction { tx_type: EconTxType, amount: u64 },
    UserOutput { content: String, format: OutputFormat },
    CustomAction { name: String, payload: Bytes },
}

struct ActionPermit {
    permit_id:      UUID,
    intent_id:      UUID,                // References the ActionIntent
    granted_by:     ProcessID,           // Always the kernel
    capabilities:   Vec<Capability>,     // Granted capabilities
    expires_at:     ISO8601,             // TTL — permit is time-limited
    constraints:    Vec<Constraint>,     // Additional restrictions
}

struct ActionCommit {
    commit_id:      UUID,
    intent_id:      UUID,
    permit_id:      UUID,
    executed_at:    ISO8601,
    result:         ActionResult,        // Success(output) | Failure(error)
    side_effects:   Vec<SideEffect>,     // Observable state changes
    resource_used:  ResourceUsage,       // CPU, memory, network consumed
}

struct ActionReceipt {
    receipt_id:     UUID,
    commit_id:      UUID,
    mission_id:     UUID,
    action_type:    ActionType,
    result_summary: String,
    evidence:       Option<EvidenceBinding>,  // If action produced a claim
    receipt_hash:   BLAKE3,              // Self-integrity seal
    kernel_seal:    Signature,           // Kernel signs every receipt
}
// === EVENT BUS ===

struct Event {
    event_id:       UUID,
    event_type:     EventType,
    source:         ProcessID,
    timestamp:      ISO8601,
    payload:        Bytes,               // MessagePack-encoded
    // Events are fire-and-forget. No authorization. No ordering guarantee.
    // They are OBSERVATIONS, not COMMANDS.
}

enum EventType {
    ActionCompleted { receipt_id: UUID },
    StateTransition { mission_id: UUID, from: MissionState, to: MissionState },
    MetricUpdate { metric: String, value: f64 },
    Heartbeat { process_id: ProcessID },
    EvaluatorVerdict { mission_id: UUID, verdict: GateVerdict },
    CharacterSheetUpdated { version: u32 },
    ExternalSignal { source: String, data: Bytes },
}
```

## Action Bus Flow (The Commit Protocol)

```
1. AGENT creates ActionIntent
     │
2.   ▼ Action Bus validates: is intent well-formed? mission_id valid?
     │
3.   ▼ Action Bus sends AUTH_REQUEST to kernel (via kernel IPC)
     │
4.   ▼ Kernel checks: capabilities granted? invariants respected?
     │
5a.  ▼ AUTH_GRANTED → ActionPermit issued (time-limited)
     │
5b.  ▼ AUTH_DENIED → ActionReceipt(DENIED) → Event Bus → Mission log
     │
6.   ▼ Agent executes action using permit
     │
7.   ▼ Agent reports ActionCommit (result + side effects)
     │
8.   ▼ Action Bus creates ActionReceipt (kernel-sealed)
     │
9.   ▼ ActionReceipt → Evidence Ledger + Event Bus
     │
10.  ▼ Mission execution log updated
```

## Ordering Guarantee

Actions within a mission have **total ordering** via SequenceNumber.
The Action Bus processes intents in sequence order. This prevents race
conditions where two agents try to write the same file, or where an
action depends on the output of a previous action.

Cross-mission actions are **partially ordered** — missions are independent
unless explicitly linked.

## Acceptance Criteria

- [ ] Every side effect in the system flows through the Action Bus
- [ ] Every action produces a kernel-sealed receipt (NO_SILENT_ACTION)
- [ ] Action Bus and Event Bus are completely separate channels
- [ ] Actions have total ordering within a mission
- [ ] Denied actions produce receipts (negative evidence)
- [ ] Action permits have TTL and expire if not used
- [ ] Event Bus subscribers cannot inject actions (read-only)
- [ ] Resource usage is tracked per action for budget enforcement
---

# MODULE 6: evidence_and_replay

## Purpose
Closes the deepest audit gap. Every completed mission must produce artifacts
that answer two questions:
1. "What happened and why?" (Evidence)
2. "Can we do it again and get the same result?" (Replay)

This is what separates a sovereign intelligence system from a stochastic
black box. If you can't replay a mission and verify the verdict matches,
you don't have constitutional enforcement — you have constitutional aspiration.

## Evidence Bundle Schema

```rust
struct EvidenceBundle {
    bundle_id:       UUID,
    mission_id:      UUID,
    created_at:      ISO8601,

    // Every claim made during the mission
    claims:          Vec<BoundClaim>,

    // Every action receipt from the Action Bus
    receipts:        Vec<ActionReceipt>,

    // Every state transition in the mission
    transitions:     Vec<StateTransition>,

    // Evaluator verdict and reasoning
    evaluator_data:  EvaluatorScores,

    // Mission-level summary
    summary:         MissionSummary,

    // Integrity
    bundle_hash:     BLAKE3,             // Hash of entire bundle
    kernel_seal:     Signature,          // Kernel attests this bundle
    manifest_ref:    Option<ManifestRef>, // Link to proof manifest chain
}

struct BoundClaim {
    claim_id:        UUID,
    claim_text:      String,
    truth_label:     TruthLabel,         // From truth_registry (Module 7)
    evidence:        EvidenceBinding,    // From kernel evidence system
    produced_by:     AgentID,            // Which agent made this claim
    produced_at:     ISO8601,
    in_task:         TaskID,             // Which DAG task
}

struct MissionSummary {
    intent:          String,             // Original human intent
    outcome:         MissionOutcome,     // Completed | PartiallyCompleted | Failed
    total_actions:   u32,
    total_claims:    u32,
    evidence_coverage: f64,              // claims_with_evidence / total_claims
    ihsan_score:     f64,                // Final composite
    duration_ms:     u64,
    resource_total:  ResourceUsage,
}
```

## Replay System Schema

```rust
struct ReplayPackage {
    package_id:      UUID,
    mission_id:      UUID,
    created_at:      ISO8601,

    // INPUT CAPTURE — everything needed to re-run
    original_intent: String,
    intent_hash:     BLAKE3,
    character_snapshot: CharacterSheetDigest,  // State at mission start
    task_dag:        TaskDAG,                 // The decomposition
    action_sequence: Vec<ActionIntent>,       // Ordered action intents

    // EXPECTED OUTPUT
    expected_evidence: EvidenceBundle,
    expected_verdict:  GateVerdict,
    expected_output:   MissionOutput,

    // REPLAY METADATA
    replay_hash:     BLAKE3,                  // Hash of this package
    determinism_level: DeterminismLevel,
}

enum DeterminismLevel {
    FullyDeterministic,   // Same inputs → same outputs guaranteed
    SemiDeterministic,    // LLM calls may vary; structure should match
    NonDeterministic,     // External API calls, live data — structure matches, values may differ
}
```
## Replay Verification Protocol

```
fn verify_replay(package: ReplayPackage) -> ReplayVerdict {
    // 1. Restore character sheet to snapshot state
    let replay_state = CharacterSheet::from_digest(package.character_snapshot);

    // 2. Re-execute mission from intent through state machine
    let replay_mission = MissionStateMachine::new(replay_state);
    replay_mission.brief(package.original_intent);
    replay_mission.decompose();  // Should produce equivalent DAG

    // 3. Execute actions in original order
    for action in package.action_sequence {
        let result = replay_mission.execute_action(action);
        // Record replay result
    }

    // 4. Gate the replay output
    let replay_verdict = evaluator_admission.evaluate(replay_mission.output());

    // 5. Compare replay verdict with original verdict
    let verdict_match = (replay_verdict == package.expected_verdict);

    // 6. Compare evidence bundles structurally
    let evidence_parity = structural_compare(
        replay_mission.evidence_bundle(),
        package.expected_evidence,
    );

    // 7. Produce replay diff
    ReplayVerdict {
        verdict_match,
        evidence_parity,     // 0.0–1.0 structural similarity
        divergences: find_divergences(replay_mission, package),
        replay_hash: blake3(replay_output),
    }
}

struct ReplayVerdict {
    verdict_match:    bool,              // Did replay produce same gate verdict?
    evidence_parity:  f64,               // Structural similarity of evidence bundles
    divergences:      Vec<Divergence>,   // Where replay differed from original
    replay_hash:      BLAKE3,
}

struct Divergence {
    location:     String,                // Which action/claim diverged
    original:     String,                // What the original produced
    replay:       String,                // What the replay produced
    severity:     DivergenceSeverity,    // Cosmetic | Structural | Verdict-Affecting
}
```

## Acceptance Criteria

- [ ] Every completed mission produces an EvidenceBundle (100% receipt coverage)
- [ ] Every EvidenceBundle is kernel-sealed with BLAKE3 + Ed25519 signature
- [ ] Every completed mission produces a ReplayPackage
- [ ] Replay of deterministic missions produces identical verdicts (100% parity)
- [ ] Replay of semi-deterministic missions produces structurally equivalent evidence (≥ 95%)
- [ ] Divergences are categorized by severity and logged
- [ ] Evidence bundles chain to proof manifests (IISAL linkage)
- [ ] Replay can be triggered at any time after mission completion
---

# MODULE 7: truth_registry

## Purpose
Single source of truth for the truth status of EVERY claim surfaced by the
system. No UI, no document, no runtime output should speak outside this
registry. The truth label is not decoration — it is part of the product.
It is part of the moat. It is what makes BIZRA's outputs uniquely trustworthy.

## Schema

```rust
struct TruthRegistry {
    claims:         HashMap<ClaimID, RegisteredClaim>,
    label_counts:   HashMap<TruthLabel, u64>,           // Running totals
    last_updated:   ISO8601,
    registry_hash:  BLAKE3,                             // Integrity seal
}

struct RegisteredClaim {
    claim_id:       ClaimID,
    claim_text:     String,
    label:          TruthLabel,
    label_history:  Vec<LabelTransition>,  // Full audit trail of label changes
    evidence:       Vec<EvidenceBinding>,   // Supporting evidence (may grow)
    produced_by:    AgentID,
    mission_id:     UUID,
    registered_at:  ISO8601,
    last_reviewed:  ISO8601,
    confidence:     f64,                    // Current composite confidence
    exposure:       ClaimExposure,          // Where this claim is visible
}

enum TruthLabel {
    Live,       // Running in production, empirically verified by continuous monitoring
    Verified,   // Tested and confirmed correct at a point in time
    Validated,  // Reviewed by evaluator, evidence-bound, not yet in production
    Wired,      // Code exists, integrations connected, not yet validated
    Planned,    // Specified, designed, not yet implemented
    Vision,     // Conceptual, aspirational, no implementation exists
}

enum ClaimExposure {
    UserFacing,     // Visible to end user — highest scrutiny
    OperatorFacing, // Visible to system operator
    Internal,       // Agent-to-agent only
    Documentary,    // In specification documents
}

struct LabelTransition {
    from:           TruthLabel,
    to:             TruthLabel,
    reason:         String,
    evidence_delta: Vec<EvidenceBinding>,  // What evidence justified this change
    transitioned_by: ProcessID,
    timestamp:       ISO8601,
}
```

## Label Transition Rules

```
Valid transitions (forward — building toward truth):
  VISION → PLANNED → WIRED → VALIDATED → VERIFIED → LIVE

Valid transitions (backward — honesty about regression):
  LIVE → VERIFIED  (monitoring shows drift)
  VERIFIED → VALIDATED  (point-in-time verification expired)
  Any → VISION  (fundamental rearchitecture)

INVALID transitions (skipping is lying):
  VISION → LIVE  ✗ (cannot jump from concept to production)
  PLANNED → VERIFIED  ✗ (cannot verify without implementation)
  VISION → VERIFIED  ✗ (cannot verify a concept)

RULE: Every forward transition requires NEW evidence not present in
the previous label's evidence set. You cannot promote a claim by
repeating the same evidence.

RULE: Every backward transition requires a REASON. Moving a claim
backwards without documenting why is a NO_SILENT_ACTION violation.
```
## Registry Operations

```rust
impl TruthRegistry {
    /// Register a new claim (always starts at the label supported by evidence)
    fn register(&mut self, claim: RegisteredClaim) -> Result<ClaimID> {
        // Validate: label is consistent with evidence quality
        assert!(label_matches_evidence(claim.label, &claim.evidence));
        // Validate: claim_text is non-empty
        assert!(!claim.claim_text.is_empty());
        // Insert and update counts
        self.claims.insert(claim.claim_id, claim);
        self.label_counts[claim.label] += 1;
        self.update_hash();
        Ok(claim.claim_id)
    }

    /// Promote a claim to a higher truth label
    fn promote(&mut self, id: ClaimID, to: TruthLabel, new_evidence: Vec<EvidenceBinding>) -> Result<()> {
        let claim = self.claims.get_mut(id)?;
        // Validate: transition is valid (no skipping)
        assert!(is_valid_forward_transition(claim.label, to));
        // Validate: new evidence exists and is not a repeat
        assert!(!new_evidence.is_empty());
        assert!(new_evidence.iter().all(|e| !claim.evidence.contains(e)));
        // Transition
        claim.label_history.push(LabelTransition { from: claim.label, to, ... });
        claim.label = to;
        claim.evidence.extend(new_evidence);
        self.update_hash();
        Ok(())
    }

    /// Demote a claim (honesty about regression)
    fn demote(&mut self, id: ClaimID, to: TruthLabel, reason: String) -> Result<()> {
        let claim = self.claims.get_mut(id)?;
        assert!(!reason.is_empty(), "Demotion requires a reason");
        claim.label_history.push(LabelTransition { from: claim.label, to, reason, ... });
        claim.label = to;
        self.update_hash();
        Ok(())
    }

    /// Query: what claims does this mission surface?
    fn claims_for_mission(&self, mission_id: UUID) -> Vec<&RegisteredClaim> {
        self.claims.values().filter(|c| c.mission_id == mission_id).collect()
    }

    /// Query: what is the system-wide truth distribution?
    fn truth_distribution(&self) -> HashMap<TruthLabel, u64> {
        self.label_counts.clone()
    }

    /// Audit: any user-facing claims below VALIDATED?
    fn audit_user_facing(&self) -> Vec<&RegisteredClaim> {
        self.claims.values()
            .filter(|c| c.exposure == ClaimExposure::UserFacing)
            .filter(|c| c.label < TruthLabel::Validated)
            .collect()
        // This should ALWAYS return empty. If not, it's a violation.
    }
}
```

## Integration with Other Modules

- **Mission State Machine**: At EVIDENCED → RETURNED, all claims from the mission
  are registered in the truth registry with appropriate labels
- **Evidence & Replay**: Replay verification can trigger label transitions
  (if replay diverges, affected claims may be demoted)
- **Evaluator Admission**: Evaluator checks that user-facing claims are ≥ VALIDATED
- **Character Sheet**: truth_distribution contributes to SNR score
- **Action Bus**: Claims produced by actions inherit the action's evidence binding

## Acceptance Criteria

- [ ] Every claim surfaced by the system is registered in the truth registry
- [ ] No user-facing claim exists below VALIDATED label (enforced, not advisory)
- [ ] Label transitions follow valid paths — no skipping (programmatically enforced)
- [ ] Forward transitions require new evidence not already in the claim's evidence set
- [ ] Backward transitions require a non-empty reason string
- [ ] Registry integrity hash is recomputed on every mutation
- [ ] Label history is append-only (complete audit trail)
- [ ] System-wide truth distribution is queryable for dashboarding
---

# END-TO-END ACCEPTANCE TEST: The Masterpiece Gate

The Node0 Activation Kernel is DONE if and only if ALL of these pass:

## Test 1: Genesis Integrity
```
GIVEN a fresh, unactivated node
WHEN  genesis_activate() is called with valid First Architect identity
THEN  Node transitions to ACTIVATED
AND   PAT-7 and SAT-5 are persisted with unique IDs
AND   URP is created and referenced
AND   genesis.bin is written with BLAKE3 self-seal
AND   Character sheet is initialized from genesis data
AND   Second call to genesis_activate() returns AlreadyActivated error
```

## Test 2: Mission Lifecycle Completion
```
GIVEN an activated node with character sheet
WHEN  a user submits intent "Summarize the competitive landscape for BIZRA"
THEN  Mission enters BRIEFED (PAT agents assigned)
AND   transitions to DECOMPOSED (task DAG with ≥1 task)
AND   transitions to EXECUTING (actions via Action Bus with kernel auth)
AND   transitions to GATED (evaluator produces IHSAN score)
AND   if IHSAN ≥ 0.95: transitions to EVIDENCED (bundle + receipt created)
AND   transitions to RETURNED (output delivered, character sheet updated)
AND   returns to IDLE with missions_completed incremented
AND   EVERY state transition has a receipt in the audit log
```

## Test 3: Persistence Survival
```
GIVEN a node mid-mission (state = EXECUTING, 3 actions completed)
WHEN  the node process is killed (SIGKILL)
AND   the node is restarted
THEN  character sheet loads from last consistent checkpoint
AND   genesis data is intact
AND   mission can resume from last receipted action
AND   no data is lost from completed actions
```

## Test 4: Evidence and Replay
```
GIVEN a completed mission with 5 claims and 8 actions
WHEN  evidence_bundle is inspected
THEN  all 5 claims have evidence bindings with ≥1 source each
AND   all 8 actions have kernel-sealed receipts
AND   bundle_hash is valid BLAKE3 of bundle contents
WHEN  replay is executed from the ReplayPackage
THEN  replay produces structurally equivalent evidence bundle
AND   replay verdict matches original verdict
AND   evidence_parity ≥ 0.95
```

## Test 5: Evaluator Authority
```
GIVEN a mission with output that scores IHSAN = 0.91
WHEN  evaluator_admission processes the output
THEN  verdict is HOLD (not PASS — 0.91 < 0.95)
AND   output does NOT reach the user
AND   human review notification is generated
GIVEN the evaluator is unavailable (timeout)
THEN  verdict defaults to REJECT (fail-closed)
AND   output does NOT reach the user
```

## Test 6: Action/Event Separation
```
GIVEN a mission in EXECUTING state
WHEN  an agent submits an ActionIntent for FileWrite
THEN  the intent goes through Action Bus → kernel AUTH_REQUEST
AND   an ActionReceipt is produced (kernel-sealed)
AND   the receipt is published on the Event Bus
AND   Event Bus subscribers can observe but CANNOT inject new actions
AND   the FileWrite side effect is logged in the mission's execution log
```

## Test 7: Truth Label Accuracy
```
GIVEN a mission that produces 3 user-facing claims
WHEN  the claims are registered in truth_registry
THEN  all 3 have labels ≥ VALIDATED (enforced, not advisory)
AND   each has ≥1 evidence binding with confidence ≥ 0.50
WHEN  an agent attempts to register a user-facing claim as PLANNED
THEN  the registration is REJECTED (user-facing requires ≥ VALIDATED)
WHEN  a claim is demoted from VERIFIED to VALIDATED
THEN  a non-empty reason is required and recorded
AND   the demotion appears in label_history
```
---

# KPI VERIFICATION MATRIX

These are the ONLY success metrics for the Activation Kernel.
Anything not on this list is not measured. Anything measured must pass.

| KPI | Target | Measurement Method | Enforcement |
|-----|--------|-------------------|-------------|
| Genesis Activation Integrity | 100% one-time, zero duplicate genesis | Unit test: call genesis twice, second must fail | HARD — test suite |
| Mission Lifecycle Completion | ≥ 95% for bounded templates | `missions_completed / (missions_completed + missions_failed)` | Dashboard + alerting |
| Receipt Coverage | 100% — every action has a kernel-sealed receipt | `receipted_actions / total_actions` | HARD — kernel enforces |
| Replay Parity | ≥ 95% initially, target 100% | `replay_evidence_parity` across all replayed missions | Replay test suite |
| Persistence Survival | 100% character sheet restoration after restart | Kill-restart test: no data loss | HARD — integration test |
| Evaluator Authority Coverage | 100% of admissions use evaluator path | `evaluated_outputs / total_outputs` | HARD — mission state machine enforces |
| Action/Event Separation | 100% of side effects through Action Bus | `action_bus_commits / total_side_effects` | HARD — architectural |
| Truth Label Accuracy | 100% on user-facing surfaces (≥ VALIDATED) | `truth_registry.audit_user_facing().len() == 0` | HARD — registry enforces |

---

# IMPLEMENTATION PRIORITY ORDER

Build in this order. Each module depends on the ones before it.

```
Week 1:  character_sheet + genesis_activation
         (Node can be activated, state persists)

Week 2:  mission_state_machine + truth_registry
         (Missions flow through states, claims are labeled)

Week 3:  action_bus (with kernel integration)
         (Actions are authorized, receipted, separated from events)

Week 4:  evaluator_admission
         (Outputs are scored, gated, held/rejected)

Week 5:  evidence_and_replay
         (Missions produce evidence bundles, replay works)

Week 6:  End-to-end integration + acceptance tests
         (All 7 tests pass, all 8 KPIs green)
```

---

# DEFINITION OF MASTERPIECE DONE

The Activation Kernel achieves masterpiece status when:

- [x] Node0 can be activated once and only once
- [ ] PAT-7 and SAT-5 are persisted, not reimagined per session
- [ ] URP is created and referenced by state, not just described
- [ ] One user mission flows through PAT → gate → SAT/URP → evidence
- [ ] The result survives restart
- [ ] Replay can reproduce the verdict
- [ ] Action side effects are separated from event chatter
- [ ] All exposed claims are truth-labeled
- [ ] The system returns to IDLE with updated character sheet after mission

If any of these are missing, it is not the masterpiece yet.

---

*NODE0 ACTIVATION KERNEL SPECIFICATION — COMPLETE*
*7 modules. 8 KPIs. 7 acceptance tests. 6-week build order.*
*Constitutional authority: DECLARATION → SYSTEM_INSTRUCTION_CHAIN → BIZRA_KERNEL_SPEC → this document*
*This is the bridge from architecture to civilization-engine.*
*Autopoietic Cycle #2 — Phase 4 (Amanah) Complete*