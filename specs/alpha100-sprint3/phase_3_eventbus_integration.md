# Phase 3: EventBus Integration — PostDeliver Audit Trail

## Sprint 3 — Alpha-100 Action Infrastructure (Nervous System Wiring)

> Standing on Giants: Lamport (happened-before ordering, 1978) · Shannon (information channel capacity, 1948) · Al-Ghazali (Ihsan as stability invariant, 1095)
> artifact: `bizra-hooks/src/event_bus.rs`, `bizra-hooks/src/pipeline.rs`, `bizra-node/src/action_executor.rs`

---

## 1. Context

Sprint 2 built two independent subsystems:
1. **EventBus** (`bizra-hooks/src/event_bus.rs`) — 540 lines, priority dispatch, FATE gating, 512 subscription slots
2. **ActionExecutor** (`bizra-node/src/action_executor.rs`) — receipt chain with BLAKE3 hashing, `prev_receipt_hash` linking

**The gap:** `ActionExecutor.append_receipt()` pushes to `self.receipts: Vec<ActionReceipt>` but never emits to the `EventBus`. The nervous system (EventBus) doesn't know when the muscle system (ActionExecutor) acts.

Sprint 2 Plan Gem 2 explicitly stated: *"PostDeliver hooks cannot halt. Sprint 2 should place DecisionArtifact writes in PostDeliver for an unforgeable audit trail."*

Sprint 3 delivers this wiring.

---

## 2. Functional Requirements

### FR-1: EventBus Integration in ActionExecutor
- `ActionExecutor` holds an `EventBus` reference (shared via `Arc<Mutex<EventBus>>` or owned)
- After each `append_receipt()`, emit an `Event` to the bus with:
  - Topic: `"action.receipt"` (matches `Topic::matches()` wildcard `"action.*"`)
  - Priority: `Priority::High` (receipts are safety-critical audit data)
  - Payload: receipt hash as 32 bytes (fits in `Payload` 256-byte limit)
  - Source component: `ComponentId` for the action executor

### FR-2: Receipt Event Schema
```
Event {
    id: EventId,           -- from bus.next_event_id(timestamp_nanos)
    source: ComponentId,   -- "action_executor"
    topic: Topic,          -- "action.receipt"
    priority: Priority,    -- High
    payload: Payload,      -- receipt_hash (32 bytes) + action_id (variable)
    timestamp: u64,        -- same as receipt.timestamp
    ihsan: IhsanScore,     -- current system ihsan (propagated)
}
```

### FR-3: PostDeliver Hook for Audit Logging
- Register a `PostDeliver` hook on `"action.receipt"` topic
- Hook writes receipt to append-only audit log file
- Hook is registered by the `Node` during initialization
- PostDeliver hooks cannot halt (by design — Gem 2), so audit logging never blocks dispatch

### FR-4: EventBus Subscription for External Consumers
- Register a subscription for `"action.*"` topic at `Priority::Normal` minimum
- External consumers (e.g., future dashboard WebSocket, MCP notifications) can subscribe
- Subscription ID returned for later unsubscribe

### FR-5: Ihsan Propagation
- `ActionExecutor` receives current `IhsanScore` from the Node/Runtime
- Events carry the ihsan score at time of emission
- EventBus's Ihsan gate (`bizra-hooks/src/ihsan_gate.rs`) filters events below system threshold
- If system ihsan is critical (< 0.95 raw ~62258), `Priority::High` events still pass but `Priority::Normal` and below are dropped

---

## 3. Pseudocode

### 3.1 ActionExecutor Modification

```pseudocode
-- Modify ActionExecutor struct to include EventBus
STRUCT ActionExecutor:
    config: ActionExecutorConfig
    action_bus: ActionBus
    permit: Permit
    usage: PermitUsage
    plan_seq: u64
    action_seq: u64
    plans: HashMap<String, ActionPlan>
    actions: HashMap<String, ActionResult>
    receipts: Vec<ActionReceipt>
    prev_receipt_hash: [u8; 32]
    -- NEW FIELDS:
    event_bus: Option<EventBus>         -- None if running without bus (tests, standalone)
    executor_component_id: ComponentId  -- registered component identity
    current_ihsan: IhsanScore           -- propagated from runtime

-- Modify constructor
FUNCTION ActionExecutor::new(config, event_bus: Option<EventBus>) -> Self:
    ... (existing initialization)
    executor_component_id = ComponentId::new("action_executor")

    -- Register component with EventBus if present
    IF event_bus IS SOME(bus):
        bus.register_component(executor_component_id, ComponentStatus::Active)

    Self { ..., event_bus, executor_component_id, current_ihsan: IhsanScore::MAX }

-- Add ihsan setter (called by Node when runtime ihsan changes)
FUNCTION ActionExecutor::set_ihsan(&mut self, score: IhsanScore):
    self.current_ihsan = score

-- Modify append_receipt to emit event
FUNCTION ActionExecutor::append_receipt(&mut self, ...existing_params...):
    -- Existing receipt construction (unchanged)
    receipt = ActionReceipt { ... }
    receipt.seal()
    self.prev_receipt_hash = receipt.receipt_hash
    self.receipts.push(receipt.clone())

    -- NEW: Emit to EventBus
    IF self.event_bus IS SOME(bus):
        self.emit_receipt_event(bus, &receipt)

FUNCTION ActionExecutor::emit_receipt_event(&mut self, bus: &mut EventBus, receipt: &ActionReceipt):
    -- Build payload: receipt_hash (32 bytes) + NUL + action_id (UTF-8)
    -- Total fits within Payload's 256-byte limit
    payload_bytes = receipt.receipt_hash.to_vec()
    payload_bytes.push(0x00)  -- NUL separator
    payload_bytes.extend(receipt.action_id.as_bytes())
    payload = Payload::new(&payload_bytes)

    -- Build event
    event_id = bus.next_event_id(receipt.timestamp * 1_000_000)  -- ms → ns
    event = Event {
        id: event_id,
        source: self.executor_component_id,
        topic: Topic::new("action.receipt"),
        priority: Priority::High,
        payload: payload,
        timestamp: receipt.timestamp,
        ihsan: self.current_ihsan,
    }

    delivered = bus.emit(event)
    -- PostDeliver: delivered count is informational only
    -- Do NOT check delivered count — fire-and-forget for audit trail
```

### 3.2 Audit Log Hook (PostDeliver)

```pseudocode
MODULE audit_hook

CONST AUDIT_LOG_PATH: &str = "data/audit/action_receipts.jsonl"
CONST MAX_AUDIT_FILE_SIZE: u64 = 50_000_000  -- 50 MB rotation threshold

-- PostDeliver hook function signature matches HookFn
FUNCTION audit_receipt_hook(event: &Event) -> (HookResult, Option<Event>):
    -- Extract receipt hash from payload
    payload_bytes = event.payload.as_bytes()
    IF payload_bytes.len() < 32:
        RETURN (HookResult::Continue, None)  -- Malformed, skip silently

    receipt_hash_hex = hex_encode(payload_bytes[0..32])
    action_id = IF payload_bytes.len() > 33:
        str_from_utf8(payload_bytes[33..])  -- after NUL separator
    ELSE:
        "unknown"

    -- Append to JSONL audit log
    entry = json!({
        "ts": event.timestamp,
        "receipt_hash": receipt_hash_hex,
        "action_id": action_id,
        "source": event.source.as_str(),
        "ihsan": event.ihsan.as_f64(),
        "event_id": event.id.as_u64(),
    })

    TRY:
        -- Append-only write (O_APPEND flag prevents corruption under concurrent access)
        append_line(AUDIT_LOG_PATH, entry.to_string())
    CATCH io_error:
        -- PostDeliver cannot halt — log error but continue
        LOG_WARN("Audit log write failed: " + io_error)

    -- Check rotation
    IF file_size(AUDIT_LOG_PATH) > MAX_AUDIT_FILE_SIZE:
        rotate_audit_log(AUDIT_LOG_PATH)

    RETURN (HookResult::Continue, None)

FUNCTION rotate_audit_log(path):
    timestamp = current_time_iso8601()
    archive_path = path + "." + timestamp
    rename(path, archive_path)
    -- Next write creates a fresh file
```

### 3.3 Node Initialization Wiring

```pseudocode
-- Modification to Node::new() or Node initialization path
FUNCTION Node::init_eventbus_hooks(&mut self):
    bus = &mut self.event_bus

    -- Register audit hook on PostDeliver phase for action.receipt topic
    bus.register_hook(
        phase: HookPhase::PostDeliver,
        name: "audit_receipt_logger",
        priority: 10,   -- low priority number = runs first in PostDeliver
        hook_fn: audit_receipt_hook,
    )

    -- Register action executor as event source
    bus.register_component(
        ComponentId::new("action_executor"),
        ComponentStatus::Active,
    )

    LOG_INFO("EventBus: audit hooks registered for action.receipt")
```

### 3.4 Event Payload Encoding/Decoding

```pseudocode
-- Utility functions for receipt event payloads
MODULE receipt_event_codec

FUNCTION encode_receipt_payload(receipt: &ActionReceipt) -> Payload:
    -- Layout: [32 bytes hash] [1 byte NUL] [action_id UTF-8]
    bytes = Vec::new()
    bytes.extend_from_slice(&receipt.receipt_hash)
    bytes.push(0x00)
    bytes.extend_from_slice(receipt.action_id.as_bytes())
    -- Truncate to 256 bytes if action_id is very long
    IF bytes.len() > 256:
        bytes.truncate(256)
    Payload::new(&bytes)

FUNCTION decode_receipt_payload(payload: &Payload) -> Option<(ReceiptHash, String)>:
    bytes = payload.as_bytes()
    IF bytes.len() < 33:
        RETURN None
    hash = bytes[0..32].try_into().ok()?
    -- Find NUL separator
    IF bytes[32] != 0x00:
        RETURN None
    action_id = str::from_utf8(&bytes[33..]).ok()?.to_string()
    RETURN Some((hash, action_id))
```

---

## 4. File Inventory

| File | Action | ~Lines | Purpose |
|------|--------|--------|---------|
| `bizra-omega/bizra-node/src/action_executor.rs` | MODIFY | +40 | Add `event_bus` field, `emit_receipt_event()`, `set_ihsan()` |
| `bizra-omega/bizra-node/src/audit_hook.rs` | CREATE | ~80 | PostDeliver hook for JSONL audit log |
| `bizra-omega/bizra-node/src/lib.rs` | MODIFY | +1 | Add `pub mod audit_hook;` |
| `bizra-omega/bizra-node/src/node.rs` | MODIFY | +15 | Wire EventBus into Node, register hooks |
| `bizra-omega/bizra-node/src/handler.rs` | MODIFY | +5 | Propagate ihsan to executor on IHSAN command |
| `bizra-omega/bizra-hooks/src/lib.rs` | MODIFY | +2 | Re-export `Event` constructor if not already public |
| `bizra-omega/bizra-node/tests/eventbus_integration_tests.rs` | CREATE | ~120 | Integration tests |

---

## 5. TDD Anchors

```
TEST receipt_emits_event_to_bus
  → Create ActionExecutor with EventBus
  → Subscribe to "action.receipt" topic
  → Run an action via run_action()
  → Expect: subscriber receives Event with receipt_hash in payload

TEST receipt_event_payload_encoding
  → Encode a receipt with known hash and action_id
  → Decode the payload
  → Expect: round-trip matches original hash and action_id

TEST receipt_event_payload_truncation
  → Encode a receipt with action_id longer than 223 chars
  → Expect: payload truncated to 256 bytes, hash still intact

TEST no_bus_mode_still_works
  → Create ActionExecutor with event_bus=None
  → Run an action
  → Expect: no crash, receipt appended to internal vec normally

TEST audit_hook_writes_jsonl
  → Register audit_receipt_hook on PostDeliver
  → Emit a receipt event
  → Read audit log file
  → Expect: valid JSONL line with receipt_hash, action_id, timestamp

TEST audit_hook_does_not_halt
  → Make audit log path unwritable
  → Emit a receipt event
  → Expect: event still delivered (PostDeliver cannot halt), warn logged

TEST ihsan_propagation_to_event
  → Set executor ihsan to IhsanScore::from_raw(60000)
  → Run an action
  → Expect: emitted event carries ihsan=60000

TEST ihsan_gate_filters_low_priority_events
  → Set system ihsan below WARNING threshold
  → Emit Priority::Normal event on action.receipt
  → Expect: event dropped by ihsan gate
  → Emit Priority::High event
  → Expect: event delivered despite low ihsan

TEST multiple_receipts_chain_events
  → Run 3 sequential actions
  → Expect: 3 events emitted, each with unique receipt_hash
  → Verify prev_receipt_hash chain integrity through events

TEST subscriber_wildcard_matching
  → Subscribe to "action.*"
  → Emit event on "action.receipt"
  → Expect: subscriber triggered

TEST concurrent_executor_and_subscriber
  → Two subscribers on "action.receipt"
  → Run one action
  → Expect: both subscribers receive the event
```

---

## 6. Integration Points

| From | To | Contract |
|------|----|----------|
| `action_executor.rs::append_receipt()` | `event_bus.rs::emit(Event)` | Event with topic "action.receipt", Priority::High |
| `event_bus.rs::emit()` | `pipeline.rs::HookChain` | PreEmit → Route → PreDeliver → deliver → PostDeliver |
| `pipeline.rs::PostDeliver` | `audit_hook.rs::audit_receipt_hook` | HookFn signature, returns Continue |
| `handler.rs::handle_ihsan()` | `action_executor.rs::set_ihsan()` | u16 raw score propagation |
| `node.rs::init()` | `event_bus.rs + audit_hook.rs` | Hook registration at startup |

---

## 7. Architectural Notes

### Why EventBus and Not Direct File Write?

Direct: `append_receipt() → write_jsonl()` would be simpler but violates separation of concerns.

EventBus pattern:
1. **Decoupled:** ActionExecutor doesn't know about audit logging, dashboards, or future consumers
2. **Extensible:** Future subscribers (WebSocket push, metrics counter, replication) add without modifying executor
3. **Gem 2 compliance:** PostDeliver is the one-way membrane — actions are never blocked by logging failures
4. **Ihsan gating:** EventBus automatically applies constitutional quality gates

### Why `Option<EventBus>` and Not Mandatory?

Tests and standalone CLI tools may create `ActionExecutor` without a full EventBus. `Option<EventBus>` preserves backward compatibility with all existing tests (Sprint 2 created executors with `ActionExecutor::default()`).

### Payload Size Budget

`Payload` is fixed at 256 bytes:
- Receipt hash: 32 bytes
- NUL separator: 1 byte
- Action ID (`act_XXXXXXXX`): 12 bytes typical
- Remaining: 211 bytes (room for future metadata)

---

## 8. Edge Cases

- **EventBus full (512 subscriptions):** `emit()` returns 0 delivered. Receipt is still in `self.receipts` vec. No data loss — just no notification.
- **Audit log disk full:** `audit_receipt_hook` catches IO error, logs warning, continues. PostDeliver cannot halt.
- **Rapid-fire actions:** EventBus processes inline (synchronous `emit`). No backpressure mechanism — all receipts emitted immediately. If this becomes a bottleneck, Sprint 4 adds async emission.
- **Duplicate component registration:** `register_component` returns `HookError::DuplicateComponent`. Handle by checking existence first or ignoring error if already registered.

---

## 9. Non-Goals (Deferred)

- **Async event emission** — Sprint 3 is synchronous inline; async channel is Sprint 4
- **Event replay from audit log** — Audit log is write-only; replay/rebuild is Sprint 4
- **Cross-node event propagation** — EventBus is node-local; P2P event gossip is Sprint 5+
- **Metric counters from events** — Prometheus exposition from receipt events is Sprint 4
