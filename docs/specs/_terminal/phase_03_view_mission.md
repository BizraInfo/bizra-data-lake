# Phase 03 — View 2: MISSION (Console)

> **Purpose:** One-line mission submission, autonomous execution, result return.
> **Status:** PARTIAL — Python REPL has stub, Rust TUI has Chat tab (close but not wired).

## 3.1 Interaction Model: Two-Touch Principle

```
Touch 1: User types mission → "organize my invoices by vendor and month"
Touch 2: System returns result + receipt + SEED earned

Interrupt ONLY for:
  - CONSTITUTIONAL_RISK → "This action may violate [invariant]. Proceed? [y/n]"
  - PERMISSION_BOUNDARY → "Requires [tier]. You are [current]. Denied."
  - IRREVERSIBLE_ACTION → "This will [description]. Confirm? [y/n]"
  - MISSING_DATA → "Need [info] to proceed. Please provide:"
```

## 3.2 Execution Pipeline (from node0_terminal_mission_loop.md)

```pseudocode
function execute_mission(intent: str, session: Session) -> MissionResult:
    // Phase 1: Submit
    event = bus.publish("mission.submitted", {intent, session_id})

    // Phase 2: Bounded Cognition
    reflex = reflex_cache.lookup(intent)
    IF reflex:
        route = "S1"  // 50ms — System-1 cache hit
        estimated_ms = 50
    ELSE:
        novelty = estimate_novelty(intent, memory)
        route = "S2"  // 1800ms — System-2 planned
        estimated_ms = novelty * 15_000  // Scale with novelty

    bus.publish("mission.classified", {novelty, route, estimated_ms})

    // Phase 3: Constitutional Gates (3-gate pipeline)
    gate_results = []
    FOR gate IN [telescript_gate, skill_tier_gate, fate_gate]:
        result = gate.check(intent, session)
        gate_results.append(result)
        IF NOT result.passed:
            bus.publish("mission.rejected", {gate: gate.name, reason: result.reason})
            RETURN MissionResult(rejected=True, reason=result.reason)

    // Phase 4: Agent Execution
    bus.publish("mission.routed", {agents: select_agents(intent)})

    // Live stream: each agent step emits events
    plan = atlas.plan(intent)           // P1: Planner
    context = oracle.research(plan)     // P2: Researcher
    result = forge.execute(plan, context) // P3: Coder
    score = judge.evaluate(result)      // P4: Evaluator
    compliance = crown.check(result)    // P5: Ethicist
    output = herald.format(result)      // P6: Publisher
    // P7 Nexus orchestrates all above

    // Phase 5: Receipt Generation
    receipt = generate_receipt(result, score, compliance)
    bus.publish("receipt.emitted", receipt)
    sign(receipt, node_signer)
    bus.publish("receipt.signed", {receipt_id, signature})

    // Phase 6: Economy
    IF score.ihsan >= MINTING_FLOOR (0.95):
        seed = mint_seed(score, receipt)
        bus.publish("economy.seed_minted", seed)

    // Phase 7: Memory + Reflex Check
    store_episode(result, receipt)
    check_reflex_candidate(intent, score.ihsan)

    RETURN MissionResult(
        result=output,
        receipt=receipt,
        seed_earned=seed,
        reflex_compiled=check_compiled(),
    )
```

## 3.3 Terminal Display (live stream)

```
bizra> organize my invoices by vendor and month

  Mission accepted: organize invoices
  Route: S2 (novel pattern) | Est: ~12s
  Agents: Atlas > Oracle > Forge > Judge > Crown

  [Atlas]  Decomposed into 4 steps
  [Oracle] Found 50 PDFs in ~/Invoices/
  [Forge]  Extracted metadata from 47/50 files
  [Forge]  Created 12 vendor folders
  [Forge]  Moved 47 files, 3 to Unknown/
  [Judge]  Quality: 0.96 (accuracy=0.94, completeness=0.97)
  [Crown]  Constitutional compliance: PASS

  ┌─ RECEIPT ──────────────────────────────────────────┐
  │  Ihsan: 0.9587                                     │
  │  SEED:  +2.38 (node: 1.19, pool: 1.19)           │
  │  Zakat: 0.06 (2.5%)                               │
  │  Hash:  d3e4f5a6...a0b1c2d3                       │
  │  Chain: <- a1b2c3d4...d9e0f1a2                    │
  └────────────────────────────────────────────────────┘

  Reflex candidate: "invoice organization" (2/3 toward compilation)
```

## 3.4 Existing Implementation

**Python (sovereign_terminal.py:577-594):** `mission <task>` command — STUB only. Returns hardcoded result. Not wired to API.

**Python (__main__.py):** `run_query()` — wired to sovereign API for single queries, but not mission pipeline.

**Rust (app.rs):** Chat tab exists — has input buffer, message history, agent routing display. Needs: receipt rendering, gate visibility, SEED display.

## 3.5 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| Wire mission to `/v1/plan` API | Python | 60 | P0 |
| Live event stream (WebSocket or SSE) | Both | 150 | P0 |
| Receipt rendering widget | Rust | 100 | P0 |
| Gate pass/fail display | Both | 40 | P1 |
| Reflex candidate notification | Both | 30 | P1 |
| Interrupt confirmation dialog | Both | 50 | P1 |

## 3.6 TDD Anchors

```
TEST: mission_two_touch_no_interrupt
  GIVEN mission with no risk flags
  WHEN mission executes
  THEN only 2 user interactions: submit + receive result

TEST: mission_constitutional_interrupt
  GIVEN mission that would violate FATE gate
  WHEN gate check runs
  THEN user prompted for confirmation before execution

TEST: mission_receipt_hash_chain
  GIVEN completed mission
  WHEN receipt generated
  THEN receipt.prev_hash == previous receipt's event_hash

TEST: mission_seed_not_minted_below_floor
  GIVEN mission with Ihsan = 0.90
  WHEN receipt generated
  THEN seed_earned == 0 (below 0.95 minting floor)

TEST: mission_reflex_hit_routes_s1
  GIVEN pattern in reflex cache
  WHEN mission matches pattern
  THEN route == "S1" and duration < 100ms
```
