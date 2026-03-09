# Node0 Terminal Mission Loop v1.0

> **Locked:** 2026-03-07
> **Scope:** The complete runtime behavior from mission submission to reflex compilation
> **Rule:** Two-touch interaction. Interrupt only for constitutional, permission, or irreversible reasons.

---

## The Golden Path

```
USER                    TERMINAL                  EVENTBUS                  SYSTEM
  │                        │                         │                        │
  │  "organize invoices"   │                         │                        │
  │───────────────────────>│                         │                        │
  │                        │  mission.submitted      │                        │
  │                        │────────────────────────>│                        │
  │                        │                         │  mission.classified    │
  │                        │                         │───────────────────────>│
  │                        │                         │                        │
  │                        │                         │  [REFLEX CHECK]        │
  │                        │                         │  Is pattern in cache?  │
  │                        │                         │                        │
  │                        │                         │  YES → S1 (50ms)       │
  │                        │                         │  NO  → S2 (1800ms)     │
  │                        │                         │                        │
  │                        │                         │  mission.routed        │
  │                        │                         │───────────────────────>│
  │                        │                         │                        │
  │                        │  [LIVE STREAM]           │                        │
  │  "⚙ Atlas planning..." │<────────────────────────│                        │
  │  "🔍 Oracle researching"│<────────────────────────│                        │
  │  "🛠 Forge executing..."│<────────────────────────│                        │
  │                        │                         │                        │
  │                        │                         │  [GATE CHECK]          │
  │                        │                         │  TeleScript: ✓         │
  │                        │                         │  Skill Tier: ✓         │
  │                        │                         │  FATE Gate: ✓          │
  │                        │                         │  Ihsān: 0.96 ✓        │
  │                        │                         │                        │
  │                        │                         │  action.executed       │
  │                        │                         │  action.verified       │
  │                        │                         │  receipt.emitted       │
  │                        │                         │  receipt.signed        │
  │                        │                         │                        │
  │                        │                         │  economy.seed_minted   │
  │                        │                         │  economy.zakat_collected│
  │                        │                         │                        │
  │                        │                         │  memory.episode_stored │
  │                        │                         │  [reflex candidate?]   │
  │                        │                         │                        │
  │                        │                         │  mission.completed     │
  │                        │                         │───────────────────────>│
  │                        │                         │                        │
  │  RESULT + RECEIPT      │                         │                        │
  │<───────────────────────│                         │                        │
  │                        │                         │                        │
  │  "⚡ REFLEX compiled!"  │  (if 3rd success)       │                        │
  │<───────────────────────│                         │                        │
  │                        │                         │                        │
```

---

## Phase 1: Mission Submission

**User action:** Type a mission in natural language.

```
bizra> organize my invoices by vendor and month
```

**System response:** Acknowledge and classify.

```
📋 Mission accepted: organize invoices
   Route: S2 (novel pattern, no reflex match)
   Agents: Atlas → Oracle → Forge → Judge → Crown
   Estimated: ~12 seconds
```

**Events emitted:**
1. `mission.submitted` — intent captured, context attached
2. `mission.classified` — novelty=0.82, route=planned, est=12000ms
3. `mission.routed` — agents=[Atlas, Oracle, Forge, Judge, Crown]
4. `mission.started` — execution clock begins

---

## Phase 2: Bounded Cognition

Before routing to agents, the system performs bounded estimation:

| Check | Question | Time |
|-------|----------|------|
| Reflex lookup | Is this pattern in System-1 cache? | <1ms |
| Novelty estimation | How novel is this mission vs. history? | <10ms |
| Risk classification | Does this require confirmation? | <10ms |
| Route selection | Reflex (S1), bounded (S1.5), or planned (S2)? | <10ms |

**If reflex hit:** Skip to Phase 5 (50ms total).
**If novel:** Continue to Phase 3.

---

## Phase 3: Constitutional Gate Check

Every mission passes through three gates before execution:

| Gate | Check | Pass Condition | Fail Action |
|------|-------|----------------|-------------|
| TeleScript | Permission boundary | Action in permit list | `action.blocked` event |
| Skill Tier | Capability check | Actor tier ≥ required tier | `action.blocked` event |
| FATE | Constitutional invariants | Ihsān ≥ 0.85 + no invariant violation | `mission.rejected` event |

**If any gate fails:**
- `mission.rejected` event emitted with reason
- Terminal shows rejection with explanation
- No action executed, no SEED minted
- The Daughter Test enforced

**Live stream to terminal:**
```
🔐 TeleScript: ✓ (path:read, path:write permitted)
📊 Skill Tier: ✓ (Adept — tier sufficient)
⚖️ FATE Gate: ✓ (Ihsān=0.96, no violations)
```

---

## Phase 4: Agent Execution

The PAT-7 team executes the mission plan:

| Agent | Role | What It Does |
|-------|------|-------------|
| P1 Atlas | Planner | Decomposes mission into steps |
| P2 Oracle | Researcher | Gathers context (RAG, web, memory) |
| P3 Forge | Coder | Executes technical steps |
| P4 Judge | Evaluator | Scores output quality (SNR) |
| P5 Crown | Ethicist | Constitutional compliance check |
| P6 Herald | Publisher | Formats result for human |
| P7 Nexus/DEMA | Integrator | Orchestrates the team, presents as one voice |

**Events emitted per step:**
- `action.requested` (with tier, scope, reversibility)
- `action.executed` (with pre/post state hashes)
- `action.verified` (with verification method)
- `telescript.step` → triggers Receipt Append subscriber

**Live stream to terminal:**
```
⚙ Atlas: Decomposed into 4 steps
🔍 Oracle: Found 50 PDFs in ~/Invoices/
🛠 Forge: Extracted metadata from 47/50 files
🛠 Forge: Created 12 vendor folders
🛠 Forge: Moved 47 files, 3 to Unknown/
📊 Judge: Quality score 0.96 (accuracy=0.94, completeness=0.97)
⚖️ Crown: Constitutional compliance ✓
📢 Herald: Summary report generated (328 words)
```

---

## Phase 5: Receipt Generation

Every completed mission produces a Proof-of-Impact receipt:

```
╔════════════════════════════════════════════════════════════╗
║  📜 RECEIPT                                                ║
║                                                            ║
║  Mission:   organize invoices by vendor and month          ║
║  Status:    COMPLETED                                      ║
║  Duration:  8.523s                                         ║
║                                                            ║
║  Ihsān:     0.9587                                         ║
║    accuracy      0.94  ████████████████████░░  94%          ║
║    safety        1.00  ██████████████████████  100%         ║
║    fairness      1.00  ██████████████████████  100%         ║
║    transparency  0.97  ████████████████████░░  97%          ║
║    privacy       0.98  ████████████████████░░  98%          ║
║    accountability 0.97 ████████████████████░░  97%          ║
║    sustainability 0.96 ████████████████████░░  96%          ║
║    beneficence   0.95  ████████████████████░░  95%          ║
║                                                            ║
║  SEED:      +2.38 (node: 1.19, pool: 1.19)                ║
║  Zakat:     0.06 (2.5%)                                    ║
║  Receipt:   d3e4f5a6...a0b1c2d3                            ║
║  Chain:     ← a1b2c3d4...d9e0f1a2                          ║
║  Signed:    Ed25519 ✓                                      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Events emitted:**
- `receipt.emitted` — full Ihsān 8-dim, CPVA, SEED yield
- `receipt.signed` — Ed25519 signature with node's key
- `economy.seed_minted` — amount, node_share, pool_share
- `economy.zakat_collected` — 2.5% deducted

---

## Phase 6: Memory Update

After receipt generation, memory is updated:

| Memory Type | Update | Subscriber |
|-------------|--------|-----------|
| Episodic | Store mission outcome + receipt | SUB-1 (ActionReceiptMemoryReinforce) |
| Semantic | Update user model (preferred domains, quality trend) | SUB-8 (ActionReceiptHHMMPromotion) |
| Procedural | Score reflex candidate | SUB-4 (SessionEndGenesisCompile) |

**Events emitted:**
- `memory.episode_stored`
- `memory.semantic_updated` (if user model changed)
- `memory.reflex_candidate_raised` (if pattern repeats)

---

## Phase 7: Reflex Precipitation Check

After every mission, check if a reflex should compile:

**Rule:** 3 successful executions of the same pattern with Ihsān ≥ 0.90.

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Execution count | ≥ 3 | Check procedural memory |
| Avg Ihsān | ≥ 0.90 | Compute from last 3 |
| Pattern match | Same macro-state | HHMM classification |

**If reflex compiles:**
```
⚡ REFLEX COMPILED: "invoice organization"
   Executions: 3 (all Ihsān ≥ 0.90)
   Avg Ihsān: 0.943
   Next time: System-1 (50ms instead of 8500ms)
   Speedup: 170×
```

**Events emitted:**
- `memory.reflex_compiled` — pattern, ihsan, count, latency reduction
- The next time this pattern is encountered, Phase 2 routes to S1

---

## Interrupt Conditions

The Two-Touch Principle allows interruption ONLY when:

| Condition | Terminal Display | User Action Required |
|-----------|-----------------|---------------------|
| Constitutional risk | `⚖️ CONSTITUTIONAL GATE: This action may violate [invariant]. Proceed? [y/n]` | Confirm or cancel |
| Permission boundary | `🔐 PERMISSION: This requires [tier]. You are [current_tier]. Denied.` | None (informational) |
| Irreversible action | `⚠️ IRREVERSIBLE: This will [description]. Confirm? [y/n]` | Confirm or cancel |
| Missing data | `❓ MISSING: I need [information] to proceed. Please provide:` | Provide data |
| Identity confirmation | `🆔 IDENTITY: This action requires your confirmation. [y/n]` | Confirm |

**All other missions execute autonomously.** The system is trusted until it proves otherwise.

---

## Error Recovery

| Error Type | System Response | Event |
|------------|----------------|-------|
| Action failed | Attempt rollback, report to user | `action.failed` + `mission.rolled_back` |
| Gate rejected | Show reason, suggest alternative | `mission.rejected` |
| Verification failed | Re-execute with different approach | `action.verified` (verified=false) |
| Timeout | Show progress, offer to continue or cancel | `mission.escalated` |
| Constitutional breach | Halt session, quarantine, log | `ihsan.gate.breached` → SUB-5 |

---

## Metrics Tracked Per Mission

| Metric | What It Measures |
|--------|-----------------|
| Duration (ms) | Total wall-clock time |
| Ihsān composite | Constitutional quality (8 dimensions) |
| SNR | Signal-to-noise ratio |
| CPVA | Cost per verified action |
| SEED earned | Economic reward |
| Pool share | Community contribution (50%) |
| Zakat | Constitutional redistribution (2.5%) |
| Reflex candidate | Whether pattern qualifies for compilation |
| Route | S1 (reflex), S1.5 (bounded), S2 (planned) |
| Agents used | Which PAT agents were invoked |
| Gates passed | TeleScript, Tier, FATE results |

---

## The Flywheel

```
Mission → Receipt → Memory → Reflex → Faster Mission → More Receipts → ...

Each cycle:
- System gets faster (more reflexes)
- System gets cheaper (S1 vs S2)
- System gets smarter (richer memory)
- System gets safer (more verified patterns)
- User gets wealthier (SEED minting)
- Network gets stronger (reflexes shared)
```

This is the loop that makes BIZRA a sovereign operating system,
not a chatbot with extra steps.

---

## The Final Test

A user opens the terminal. Types one sentence. Gets a result.
The result is receipted, constitutional, and economically rewarded.
The next time they ask the same thing, it's 170× faster.
They never had to think about agents, gates, events, or tokens.
They just got their work done.

That is the Node0 standard.