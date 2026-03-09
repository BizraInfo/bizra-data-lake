# Phase 08 — EventBus Wiring & Subscriber Integration

> **Purpose:** Wire the 12 EventBus subscribers to the terminal for live event display.
> **Status:** PARTIAL — subscribers.py defines all 12. Not wired to terminal rendering.

## 8.1 Subscriber Map

From `terminal/subscribers.py`, the complete EventBus nervous system:

### Phase 1: Learning Loop (SUB 1-4)

| # | Subscriber | Event | Action | Terminal Effect |
|---|-----------|-------|--------|----------------|
| 1 | `ActionReceiptMemoryReinforce` | `action.receipt` | Reinforce memory trace | Mission view: "[Memory] Pattern reinforced" |
| 2 | `ActionIntentTeleScriptBegin` | `action.intent` | Begin TeleScript workflow | Mission view: "TeleScript started" |
| 3 | `TeleScriptStepReceiptAppend` | `telescript.step` | Append step receipt | Receipts view: new entry |
| 4 | `SessionEndGenesisCompile` | `session.end` | Check reflex compilation | Skills view: reflex notification |

### Phase 2: Safety (SUB 5-7)

| # | Subscriber | Event | Action | Terminal Effect |
|---|-----------|-------|--------|----------------|
| 5 | `IhsanGateBreachHandler` | `ihsan.gate.breached` | HALT session (fail-closed) | RED ALERT: "SESSION HALTED" |
| 6 | `FailedActionQuarantine` | `action.receipt.failed` | Quarantine + decrement count | Mission view: "QUARANTINED" |
| 7 | `TeleScriptRollbackHealing` | `telescript.rolled_back` | Self-repair diagnosis | Mission view: "Healing in progress" |

### Phase 3: Economics (SUB 8-12)

| # | Subscriber | Event | Action | Terminal Effect |
|---|-----------|-------|--------|----------------|
| 8 | `ActionReceiptHHMMPromotion` | `action.receipt` | Promote to semantic memory | Memory view: "Pattern promoted" |
| 9 | `MemoryPromotedPoICredit` | `memory.promoted` | Accumulate PoI credit | Wallet view: PoI credit |
| 10 | `TeleScriptCompletedPoIAccumulate` | `telescript.completed` | Mint SEED if >= 0.95 | Wallet view: "+X.XX SEED" |
| 11 | `MemoryRetrievedBudgetReport` | `memory.retrieved` | Track context budget | System view: budget % |
| 12 | `AgentRegisteredSelfModelUpdate` | `agent.registered` | Update self-model | System view: capability count |

## 8.2 Event Schema (40 event types)

From `terminal/event_schema_v1.json`:

| Category | Events | Count |
|----------|--------|-------|
| Identity | genesis.created, node.unlocked, signer.rotated | 3 |
| Mission | submitted, classified, routed, started, completed, rejected, escalated, rolled_back | 8 |
| Action | requested, blocked, executed, verified, failed | 5 |
| Receipt | emitted, signed, audited | 3 |
| Memory | semantic_updated, episode_stored, reflex_candidate_raised, reflex_compiled, reflex_pruned | 5 |
| Economy | seed_minted, bloom_accrued, zakat_collected, supply_cap_hit | 4 |
| Social | attestation_given, skill_published, skill_purchased, reflex_imported | 4 |
| **Total** | | **32** |

Plus 8 internal types from subscribers.py EventType enum = **40 total**.

## 8.3 Terminal Integration Pattern

```pseudocode
// The terminal subscribes to EventBus as a passive observer
class TerminalSubscriber:
    event_types = [ALL]  // Listen to everything

    def handle(self, event: Event):
        // Route to correct view
        MATCH event.event_type:
            "mission.*"         -> mission_view.update(event)
            "action.*"          -> mission_view.update(event)
            "receipt.*"         -> receipts_view.update(event)
            "memory.reflex_*"   -> skills_view.update(event)
            "memory.*"          -> memory_view.update(event)
            "economy.*"         -> wallet_view.update(event)
            "ihsan.gate.*"      -> system_view.alert(event)
            _                   -> status_bar.flash(event)

// For Rust TUI: events arrive via channel (mpsc)
// For Python REPL: events arrive via callback
// For Web Terminal: events arrive via WebSocket
```

## 8.4 Action Schema Integration

From `terminal/action_schema_v1.json`:

| Category | Actions | Required Tier | Gate Pipeline |
|----------|---------|--------------|--------------|
| Local | file.read, file.write, file.move, clipboard.*, window.*, process.*, script.* | novice-master | TeleScript > SkillTier > FATE |
| External | api.request, mcp.invoke, browser.task, a2a.message, marketplace.publish | adept-master | TeleScript > SkillTier > FATE |
| Meta | memory.update, proof.generate, reflex.compile, reflex.prune, rollback.*, self_check.* | novice | Internal only |

## 8.5 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| TerminalSubscriber (Python) | Python | 60 | P0 |
| Event channel (Rust mpsc) | Rust | 80 | P0 |
| Event-to-view routing | Both | 40 | P0 |
| Status bar event flash | Both | 30 | P1 |
| Event log scrollback | Both | 50 | P2 |

## 8.6 TDD Anchors

```
TEST: event_routes_to_correct_view
  GIVEN event type "economy.seed_minted"
  WHEN TerminalSubscriber handles
  THEN wallet_view.update() called

TEST: ihsan_breach_triggers_alert
  GIVEN event type "ihsan.gate.breached"
  WHEN TerminalSubscriber handles
  THEN system_view.alert() called with RED priority

TEST: all_40_event_types_have_handler
  GIVEN every event type from schema
  WHEN routed through TerminalSubscriber
  THEN no unhandled events (all have a target view)

TEST: event_chain_hash_continuity
  GIVEN 100 sequential events
  THEN event[n].prev_hash == event[n-1].event_hash for all n
```
