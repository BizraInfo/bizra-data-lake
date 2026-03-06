# Phase 68 — Bus Architecture: MMORPG-Grade Nervous System

## Status: SPEC-READY
## Date: 2026-03-06
## Architect: Node0 Sovereignty Team

---

## 1. Vision

Make BIZRA feel like an MMORPG engine: deterministic, persistent, replayable.
LLMs are "voice acting" — the buses are the game loop.

**CQRS + Event Sourcing:**
- **Action Bus** = command pipeline (proposals to cause change)
- **Event Bus** = nervous system (facts that happened)
- **Event Log** = disk truth (immutable history)

**Rule:** Actions produce events. Only after verification do we emit receipts.

---

## 2. What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| Python EventBus | `core/sovereign/event_bus.py` | Complete (priority queue, wildcards, correlation) |
| Rust EventBus | `bizra-hooks/src/event_bus.rs` | Complete (8-shard FNV-1a, 512 slots) |
| Rust ActionBus | `bizra-agent/src/action_bus.rs` | Complete (permit-guarded, receipt-chained) |
| Rust Dispatcher | `bizra-action/src/dispatcher.rs` | Complete (OODA loop, 8 channels) |
| MissionOrchestrator | `core/sovereign/mission.py` | Complete (6-phase, 4 degradation levels) |
| ChannelDispatcher | `core/bridges/channel_dispatcher.py` | Complete (4-channel routing) |
| RustEventBridge | `core/sovereign/event_bus.py` | Complete (PyO3 facade) |
| 12 Subscribers | `bizra-hooks/src/subscribers.rs` | Wired at boot |

---

## 3. What Phase 68 Adds

| Component | Spec File | New Code Location |
|-----------|-----------|-------------------|
| Python ActionBus | `phase_68_01_action_bus.md` | `core/bus/action_bus.py` |
| Omega Loop Controller | `phase_68_02_omega_loop.md` | `core/bus/omega_loop.py` |
| Unified Config System | `phase_68_03_config_system.md` | `core/config/` |
| Capsule Runtime | `phase_68_04_capsule_runtime.md` | `core/bus/capsule_runtime.py` |
| TeleScript Python | `phase_68_05_telescript_python.md` | `core/bus/telescript.py` |
| Topic Registry | `phase_68_06_topic_registry.md` | `core/bus/topics.py` |

---

## 4. The 6 Primitives (Legend Stack)

```
1. Gates      — Pre-execution veto (FATE + TeleScript)
2. Hooks      — Post-execution automation (format/lint/test)
3. Capsules   — Skills: packaged workflows + assets
4. Workers    — Subagents: parallel specialists with quotas
5. Bridges    — External integrations (MCP pattern)
6. Packs      — Signed bundles of the above
```

**Omega Loop Controller** — Iterative self-improvement with proof-based stop.

---

## 5. Canonical Flow (One Action Lifecycle)

```
USER INTENT
  |
  v
PAT-7 (plan/spec)
  |
  v
ActionBus.propose(action)
  |
  v
TeleScript.check(capabilities)
  |-- deny --> EventBus.emit("policy.telescript.denied")
  +-- allow
        |
        v
FATE Gate.evaluate(action)
  |-- deny --> EventBus.emit("policy.fate.vetoed")
  +-- allow
        |
        v
ActionBus.execute(action)
  |
  v
Channel.dispatch(action)  [HDA / File / Browser / LLM]
  |
  v
Verifier.check(pre_state, post_state)
  |
  v
Receipt = sign(action_id, outcome_hash, ihsan_score)
  |
  v
EventBus.emit("action.receipt", receipt)
  |
  v
EventLog.append(receipt_event)       [L1 disk truth]
  |
  v
Post-Receipt Hooks
  |-- reflex_compile_if_eligible()   [myelination]
  |-- update_economy()               [SEED mint if ihsan >= 0.95]
  +-- diffuse_if_high_quality()      [federation broadcast]
```

---

## 6. Topic Map (Extended from 12 Rust Topics)

### Tier 0: Constitutional (always active)
- `action.intent` — Mission to execute
- `action.receipt` — Action completed with proof
- `action.receipt.failed` — Action failed
- `ihsan.breach` — Excellence threshold violated
- `poi.credit` — Proof-of-Intent credit awarded

### Tier 1: Cognitive (active at degradation level >= 2)
- `memory.promoted` — Memory elevated to cache
- `memory.retrieved` — Memory lookup hit
- `reflex.compiled` — Pattern myelinated
- `reflex.cache_hit` — System-1 hit
- `reflex.pruned` — Low-quality reflex removed

### Tier 2: Lifecycle (always active)
- `node.lifecycle.*` — boot, shutdown, upgrade
- `session.end` — Session cleanup
- `system.lifecycle` — Agent lifecycle events

### Tier 3: Economic (active during ticker)
- `economy.seed_minted` — SEED tokens created
- `economy.bloom_accrued` — Governance weight gained
- `economy.zakat` — Purification deduction
- `economy.demurrage` — Idle tax applied
- `economy.asabiyyah` — Cohesion score updated

### Tier 4: Federation (active when peers > 0)
- `federation.peer_seen` — New peer discovered
- `federation.attestation.*` — sent/received/reciprocal
- `federation.diffusion` — Reflex shared to network

### Tier 5: Policy (always active)
- `policy.fate.vetoed` — FATE gate denied action
- `policy.telescript.denied` — Capability mask blocked
- `policy.invariant.violation` — Constitutional invariant broken

### Tier 6: Mission (active during orchestration)
- `mission.created` — New mission started
- `mission.planned` — Decomposition complete
- `mission.executed` — All channels done
- `mission.verified` — SNR + Ihsan gate passed
- `mission.failed` — Mission did not meet threshold

---

## 7. Implementation Order

### Phase A: Foundation (core/bus/ module)
1. `topics.py` — Topic registry + validation
2. `action_bus.py` — Python CQRS action pipeline
3. `telescript.py` — Capability masks

### Phase B: Control Loop
4. `omega_loop.py` — Proof-based iteration controller

### Phase C: Extensibility
5. `capsule_runtime.py` — Skill/workflow execution
6. Config system (`core/config/`)

### Phase D: Wire into existing systems
7. MissionOrchestrator uses ActionBus (not direct channel calls)
8. Ticker emits economy.* events via EventBus
9. Asabiyyah-Gini coupling (Phase 67.03) wired

---

## 8. Design Principles

1. **Events are facts.** Safe to replay. Character save built from them.
2. **Actions are proposals.** Can be vetoed, retried, rolled back.
3. **Only receipts become truth.** LLM output is noise until verified.
4. **Python-first, Rust-portable.** All schemas JSON-serializable.
5. **Fixed-point economics.** No floats cross the bus boundary.
6. **At-least-once delivery.** Idempotency keys prevent double-mint.
7. **Backpressure built in.** Bounded queues, not unbounded growth.

---

## 9. Standing on Giants

- Nakamoto (2008): Event log as consensus substrate
- Lamport (1978): Logical clocks for event ordering
- Kahneman (2002): System-1 cache (reflex) vs System-2 reasoning (LLM)
- Fowler (2005): CQRS + Event Sourcing patterns
- Gray & Reuter (1993): Two-phase commit for action lifecycle
- Ibn Khaldun (1377): Asabiyyah as network health metric
