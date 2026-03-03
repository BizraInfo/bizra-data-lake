# Unified Concurrency Fabric — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unlock concurrency across the BIZRA nervous system by sharding the EventBus, splitting the OmniKernel read/write paths, and bridging Rust↔Python events via PyO3.

**Architecture:** Three interdependent changes forming one coherent "Unified Concurrency Fabric": (1) Namespace-sharded EventBus with 8 shards replacing flat 512-slot array, (2) Two-phase OmniKernel with `try_cache_hit(&self)` for concurrent reads and `run_cycle(&mut self)` preserved for backward compat, (3) PyEventBridge PyO3 class forwarding Python events into Rust BizraSystem.

**Tech Stack:** Pure Rust (no new deps for bizra-hooks), std::sync::atomic (bizra-agent), PyO3 0.24 (bizra-python), Python asyncio (core/)

**Design doc:** `docs/plans/2026-02-27-unified-concurrency-fabric-design.md`

---

## Phase 1: Sharded EventBus (bizra-hooks)

### Task 1.1: Add shard_index helper + test

**Files:**
- Modify: `bizra-omega/bizra-hooks/src/event_bus.rs`

**Step 1: Write the failing test**

Add at the bottom of the `#[cfg(test)] mod tests` block in `event_bus.rs`:

```rust
#[test]
fn shard_index_groups_same_namespace() {
    let t1 = Topic::new("action.receipt");
    let t2 = Topic::new("action.intent");
    let t3 = Topic::new("memory.promoted");
    assert_eq!(shard_index(&t1), shard_index(&t2));
    assert_ne!(shard_index(&t1), shard_index(&t3));
}

#[test]
fn shard_index_within_bounds() {
    for topic in &["action.x", "memory.x", "telescript.x", "session.x",
                    "system.x", "ihsan.x", "poi.x", "unknown.x"] {
        let idx = shard_index(&Topic::new(topic));
        assert!(idx < NUM_SHARDS, "shard_index({topic}) = {idx} >= {NUM_SHARDS}");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd bizra-omega && cargo test -p bizra-hooks -- shard_index 2>&1 | tail -5`
Expected: FAIL — `shard_index` not found

**Step 3: Write minimal implementation**

Add above the `EventBus` struct in `event_bus.rs`:

```rust
/// Number of topic-namespace shards (power of 2 for fast modulo).
const NUM_SHARDS: usize = 8;

/// Capacity per shard. NUM_SHARDS * SHARD_CAPACITY = 512 (same total).
const SHARD_CAPACITY: usize = 64;

/// Map a topic to its shard index via FNV-1a hash of the namespace prefix.
/// The namespace is everything before the first `.` in the topic string.
fn shard_index(topic: &Topic) -> usize {
    let s = topic.as_str();
    let prefix_end = s.as_bytes().iter().position(|&b| b == b'.').unwrap_or(s.len());
    // FNV-1a 32-bit hash of the prefix bytes
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in &s.as_bytes()[..prefix_end] {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    (hash as usize) & (NUM_SHARDS - 1) // Bitwise AND since NUM_SHARDS is power of 2
}
```

**Step 4: Run test to verify it passes**

Run: `cd bizra-omega && cargo test -p bizra-hooks -- shard_index -v 2>&1 | tail -10`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-hooks/src/event_bus.rs
git commit -m "feat(hooks): add shard_index namespace router for EventBus"
```

---

### Task 1.2: Implement EventShard struct

**Files:**
- Modify: `bizra-omega/bizra-hooks/src/event_bus.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn event_shard_subscribe_and_dispatch() {
    let mut shard = EventShard::new();
    let comp = ComponentId::from_name("test", "1.0.0");
    let sub_id = shard.subscribe(comp, "action.receipt", Priority::Low, noop_handler).unwrap();
    assert_eq!(shard.sub_count(), 1);

    let event = Event {
        id: EventId::new(1000, 0),
        source: make_source(),
        topic: Topic::new("action.receipt"),
        priority: Priority::Normal,
        payload: Payload::empty(),
        ihsan_score: IhsanScore::MAX,
    };
    let (delivered, _) = shard.dispatch(&event);
    assert_eq!(delivered, 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cd bizra-omega && cargo test -p bizra-hooks -- event_shard_subscribe 2>&1 | tail -5`
Expected: FAIL — `EventShard` not found

**Step 3: Write implementation**

Add `EventShard` struct and methods in `event_bus.rs` (above the `EventBus` struct):

```rust
/// A single shard of the EventBus. Owns handlers for one topic namespace.
#[derive(Clone)]
struct EventShard {
    handlers: [Option<HandlerEntry>; SHARD_CAPACITY],
    sub_count: usize,
}

impl EventShard {
    const fn new() -> Self {
        EventShard {
            handlers: [None; SHARD_CAPACITY],
            sub_count: 0,
        }
    }

    fn subscribe(
        &mut self,
        component: ComponentId,
        topic_filter: &str,
        min_priority: Priority,
        handler: EventHandler,
        sub_id: SubscriptionId,
    ) -> Result<SubscriptionId, HookError> {
        let slot = self
            .handlers
            .iter()
            .position(|h| h.is_none())
            .ok_or(HookError::SubscribersFull)?;

        self.handlers[slot] = Some(HandlerEntry {
            sub: Subscription {
                id: sub_id,
                component,
                topic_filter: Topic::new(topic_filter),
                min_priority,
                active: true,
            },
            handler,
        });
        self.sub_count += 1;
        Ok(sub_id)
    }

    fn unsubscribe(&mut self, sub_id: SubscriptionId) -> bool {
        for entry in self.handlers.iter_mut() {
            if let Some(h) = entry {
                if h.sub.id == sub_id {
                    *entry = None;
                    self.sub_count -= 1;
                    return true;
                }
            }
        }
        false
    }

    fn unsubscribe_all(&mut self, component: &ComponentId) -> usize {
        let mut removed = 0;
        for entry in self.handlers.iter_mut() {
            if let Some(h) = entry {
                if h.sub.component == *component {
                    *entry = None;
                    removed += 1;
                }
            }
        }
        self.sub_count -= removed;
        removed
    }

    fn set_active(&mut self, sub_id: SubscriptionId, active: bool) -> bool {
        for h in self.handlers.iter_mut().flatten() {
            if h.sub.id == sub_id {
                h.sub.active = active;
                return true;
            }
        }
        false
    }

    fn set_active_for_component(&mut self, id: &ComponentId, active: bool) {
        for h in self.handlers.iter_mut().flatten() {
            if h.sub.component == *id {
                h.sub.active = active;
            }
        }
    }

    /// Dispatch an event to all matching handlers in this shard.
    /// Returns (delivered_count, should_halt).
    fn dispatch(&self, event: &Event) -> (usize, bool) {
        let mut delivered = 0;
        for h in self.handlers.iter().flatten() {
            if h.sub.matches(event) {
                let result = (h.handler)(event);
                delivered += 1;
                match result {
                    HookResult::Continue => continue,
                    HookResult::Skip => break,
                    HookResult::Halt => return (delivered, true),
                    HookResult::Transform => continue,
                }
            }
        }
        (delivered, false)
    }

    fn sub_count(&self) -> usize {
        self.sub_count
    }
}
```

Note: `subscribe` takes `sub_id` as a parameter (allocated by the parent bus). Adjust the test to pass a `SubscriptionId`:

```rust
#[test]
fn event_shard_subscribe_and_dispatch() {
    let mut shard = EventShard::new();
    let comp = ComponentId::from_name("test", "1.0.0");
    shard.subscribe(comp, "action.receipt", Priority::Low, noop_handler, SubscriptionId(1)).unwrap();
    assert_eq!(shard.sub_count(), 1);

    let event = Event {
        id: EventId::new(1000, 0),
        source: make_source(),
        topic: Topic::new("action.receipt"),
        priority: Priority::Normal,
        payload: Payload::empty(),
        ihsan_score: IhsanScore::MAX,
    };
    let (delivered, halted) = shard.dispatch(&event);
    assert_eq!(delivered, 1);
    assert!(!halted);
}
```

**Step 4: Run test**

Run: `cd bizra-omega && cargo test -p bizra-hooks -- event_shard 2>&1 | tail -10`
Expected: PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-hooks/src/event_bus.rs
git commit -m "feat(hooks): add EventShard struct for namespace-isolated dispatch"
```

---

### Task 1.3: Replace EventBus internals with ShardedEventBus

**Files:**
- Modify: `bizra-omega/bizra-hooks/src/event_bus.rs`

**Step 1: Write the failing test (shard isolation)**

```rust
#[test]
fn sharded_bus_isolates_namespaces() {
    let mut bus = EventBus::new();
    let action_comp = ComponentId::from_name("action-handler", "1.0.0");
    let memory_comp = ComponentId::from_name("memory-handler", "1.0.0");

    // Subscribe to different namespaces
    bus.subscribe(action_comp, "action.receipt", Priority::Low, noop_handler).unwrap();
    bus.subscribe(memory_comp, "memory.promoted", Priority::Low, noop_handler).unwrap();

    // Emit to action namespace — should reach only action handler
    let delivered = bus.emit_simple(
        make_source(), "action.receipt", Payload::empty(), Priority::Normal, 1000,
    );
    assert_eq!(delivered, 1);

    // Emit to memory namespace — should reach only memory handler
    let delivered = bus.emit_simple(
        make_source(), "memory.promoted", Payload::empty(), Priority::Normal, 1001,
    );
    assert_eq!(delivered, 1);
}
```

**Step 2: Run test — should pass with current flat bus too (baseline)**

Run: `cd bizra-omega && cargo test -p bizra-hooks -- sharded_bus_isolates 2>&1 | tail -5`

**Step 3: Rewrite EventBus struct to use shards**

Replace the `EventBus` struct and its `impl` block:

```rust
/// The Event Bus — routes events between components via namespace sharding.
///
/// Topics are sharded by their namespace prefix (text before the first `.`).
/// Each shard dispatches independently, eliminating cross-namespace contention.
pub struct EventBus {
    /// Per-namespace shards
    shards: [EventShard; NUM_SHARDS],
    /// Global shard for subscriptions without a namespace prefix
    global: EventShard,
    /// Next subscription ID (monotonic counter across all shards)
    next_sub_id: u32,

    /// Pending event queue (priority-ordered deferred dispatch)
    pending: [Option<Event>; MAX_PENDING],
    pending_count: usize,

    /// Telemetry counters
    total_emitted: u64,
    total_delivered: u64,
    total_dropped: u64,

    /// Event ID sequence counter
    sequence: u16,
    last_timestamp: u64,
}

impl EventBus {
    pub fn new() -> Self {
        const EMPTY_SHARD: EventShard = EventShard::new();
        EventBus {
            shards: [EMPTY_SHARD; NUM_SHARDS],
            global: EventShard::new(),
            next_sub_id: 1,
            pending: [None; MAX_PENDING],
            pending_count: 0,
            total_emitted: 0,
            total_delivered: 0,
            total_dropped: 0,
            sequence: 0,
            last_timestamp: 0,
        }
    }

    // ━━━ Subscribe / Unsubscribe ━━━

    pub fn subscribe(
        &mut self,
        component: ComponentId,
        topic_filter: &str,
        min_priority: Priority,
        handler: EventHandler,
    ) -> Result<SubscriptionId, HookError> {
        let sub_id = SubscriptionId(self.next_sub_id);
        self.next_sub_id += 1;

        let topic = Topic::new(topic_filter);
        let shard = self.shard_for_topic_mut(&topic);
        shard.subscribe(component, topic_filter, min_priority, handler, sub_id)
    }

    pub fn unsubscribe(&mut self, sub_id: SubscriptionId) -> bool {
        // Search all shards (subscription could be anywhere)
        for shard in self.shards.iter_mut() {
            if shard.unsubscribe(sub_id) { return true; }
        }
        self.global.unsubscribe(sub_id)
    }

    pub fn unsubscribe_all(&mut self, component: &ComponentId) -> usize {
        let mut removed = 0;
        for shard in self.shards.iter_mut() {
            removed += shard.unsubscribe_all(component);
        }
        removed += self.global.unsubscribe_all(component);
        removed
    }

    pub fn set_active(&mut self, sub_id: SubscriptionId, active: bool) -> bool {
        for shard in self.shards.iter_mut() {
            if shard.set_active(sub_id, active) { return true; }
        }
        self.global.set_active(sub_id, active)
    }

    // ━━━ Emit ━━━

    pub fn next_event_id(&mut self, timestamp_nanos: u64) -> EventId {
        if timestamp_nanos != self.last_timestamp {
            self.sequence = 0;
            self.last_timestamp = timestamp_nanos;
        } else {
            self.sequence = self.sequence.wrapping_add(1);
        }
        EventId::new(timestamp_nanos, self.sequence)
    }

    pub fn emit(&mut self, event: Event) -> usize {
        self.total_emitted += 1;

        // Dispatch to the target shard
        let shard = self.shard_for_topic_mut(&event.topic);
        let (shard_delivered, shard_halted) = shard.dispatch(&event);

        // Also dispatch to global shard (cross-namespace wildcards)
        let global_delivered = if !shard_halted {
            let (gd, _) = self.global.dispatch(&event);
            gd
        } else {
            0
        };

        let delivered = shard_delivered + global_delivered;
        if delivered == 0 {
            self.total_dropped += 1;
        } else {
            self.total_delivered += delivered as u64;
        }
        delivered
    }

    pub fn emit_simple(
        &mut self,
        source: ComponentId,
        topic: &str,
        payload: Payload,
        priority: Priority,
        timestamp_nanos: u64,
    ) -> usize {
        let id = self.next_event_id(timestamp_nanos);
        let event = Event {
            id, source,
            topic: Topic::new(topic),
            priority, payload,
            ihsan_score: IhsanScore::MAX,
        };
        self.emit(event)
    }

    // ━━━ Queue (unchanged) ━━━

    pub fn enqueue(&mut self, event: Event) -> bool {
        if self.pending_count >= MAX_PENDING { return false; }
        let insert_pos = self.pending[..self.pending_count]
            .iter()
            .position(|e| {
                e.as_ref()
                    .map(|existing| event.priority > existing.priority)
                    .unwrap_or(true)
            })
            .unwrap_or(self.pending_count);
        if insert_pos < self.pending_count {
            for i in (insert_pos..self.pending_count).rev() {
                self.pending[i + 1] = self.pending[i];
            }
        }
        self.pending[insert_pos] = Some(event);
        self.pending_count += 1;
        true
    }

    pub fn flush(&mut self) -> usize {
        let mut dispatched = 0;
        let count = self.pending_count;
        self.pending_count = 0;
        for i in 0..count {
            if let Some(event) = self.pending[i].take() {
                self.emit(event);
                dispatched += 1;
            }
        }
        dispatched
    }

    // ━━━ Telemetry (unchanged API) ━━━

    pub fn subscription_count(&self) -> usize {
        let mut total = self.global.sub_count();
        for shard in &self.shards {
            total += shard.sub_count();
        }
        total
    }

    pub fn total_emitted(&self) -> u64 { self.total_emitted }
    pub fn total_delivered(&self) -> u64 { self.total_delivered }
    pub fn total_dropped(&self) -> u64 { self.total_dropped }
    pub fn pending_count(&self) -> usize { self.pending_count }

    pub fn set_active_for_component(&mut self, id: &ComponentId, active: bool) {
        for shard in self.shards.iter_mut() {
            shard.set_active_for_component(id, active);
        }
        self.global.set_active_for_component(id, active);
    }

    pub fn delivery_ratio(&self) -> f64 {
        if self.total_emitted == 0 { 1.0 }
        else { self.total_delivered as f64 / self.total_emitted as f64 }
    }

    // ━━━ Internal routing ━━━

    fn shard_for_topic_mut(&mut self, topic: &Topic) -> &mut EventShard {
        let s = topic.as_str();
        if s.contains('.') {
            &mut self.shards[shard_index(topic)]
        } else {
            &mut self.global
        }
    }
}
```

**Step 4: Run ALL bizra-hooks tests**

Run: `cd bizra-omega && cargo test -p bizra-hooks 2>&1 | tail -20`
Expected: ALL tests pass (including existing tests — the API is identical)

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-hooks/src/event_bus.rs
git commit -m "feat(hooks): replace flat EventBus with namespace-sharded dispatch

Shards events by topic prefix (action, memory, telescript, etc.).
Each shard dispatches independently: O(N/8) instead of O(N).
Zero new dependencies. All types remain Copy + no_std compatible."
```

---

### Task 1.4: Run full workspace tests (regression check)

**Files:** None (verification only)

**Step 1: Run bizra-hooks tests**

Run: `cd bizra-omega && cargo test -p bizra-hooks 2>&1 | tail -5`
Expected: All pass

**Step 2: Run subscriber tests (most likely to break)**

Run: `cd bizra-omega && cargo test -p bizra-hooks -- subscribers 2>&1 | tail -10`
Expected: All 12-subscriber tests pass (wire_all, lifecycle flow, etc.)

**Step 3: Run full workspace (catch any downstream breakage)**

Run: `cd bizra-omega && cargo test --workspace 2>&1 | tail -10`
Expected: 610+ tests pass

---

## Phase 2: OmniKernel Read/Write Split (bizra-agent)

### Task 2.1: Add `lookup_readonly` to ReflexCache

**Files:**
- Modify: `bizra-omega/bizra-agent/src/reflex_cache.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn lookup_readonly_finds_active_rule() {
    let mut cache = ReflexCache::new(2048);
    cache.load_bootstrap_rules();
    let trigger = TriggerHash(*blake3::hash(b"test").as_bytes());
    // Insert a test rule
    cache.insert_rule(ReflexRule::test_rule(trigger));

    // Read-only lookup should find it without mutating stats
    let stats_before = cache.stats().clone();
    let result = cache.lookup_readonly(ReflexMode::Active, &trigger, Some(BOOTSTRAP_POLICY_HASH));
    assert!(result.is_some());
    // Stats should NOT change (read-only)
    assert_eq!(cache.stats().hits, stats_before.hits);
}
```

**Step 2: Run test to verify it fails**

Run: `cd bizra-omega && cargo test -p bizra-agent -- lookup_readonly 2>&1 | tail -5`
Expected: FAIL — `lookup_readonly` not found

**Step 3: Implement `lookup_readonly`**

Add to the `impl ReflexCache` block in `reflex_cache.rs`:

```rust
/// Read-only cache lookup. Does NOT update hit/miss counters.
/// Use this for concurrent read access paths.
pub fn lookup_readonly(
    &self,
    mode: ReflexMode,
    trigger: &TriggerHash,
    current_policy_hash: Option<[u8; 32]>,
) -> Option<ReflexRule> {
    if mode != ReflexMode::Active {
        return None;
    }
    let policy_hash = current_policy_hash?;
    let rule = self.by_trigger.get(trigger)?;
    if rule.quarantined || rule.policy_hash != policy_hash {
        return None;
    }
    Some(rule.clone())
}
```

**Step 4: Run test**

Run: `cd bizra-omega && cargo test -p bizra-agent -- lookup_readonly 2>&1 | tail -5`
Expected: PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-agent/src/reflex_cache.rs
git commit -m "feat(agent): add ReflexCache::lookup_readonly for concurrent read access"
```

---

### Task 2.2: Add `lookup_readonly` to EngramCache

**Files:**
- Modify: `bizra-omega/bizra-ttrl/src/engram.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn lookup_readonly_returns_hit_without_mutating() {
    let mut cache = EngramCache::new();
    cache.insert(b"intent_bytes", "Paris", 0.99, 1000);

    let hits_before = cache.hits;
    let result = cache.lookup_readonly(b"intent_bytes", 0.95);
    assert!(matches!(result, EngramResult::Hit { .. }));
    // hits counter should NOT change
    assert_eq!(cache.hits, hits_before);
}
```

**Step 2: Run test to verify it fails**

Run: `cd bizra-omega && cargo test -p bizra-ttrl -- lookup_readonly 2>&1 | tail -5`
Expected: FAIL — `lookup_readonly` not found

**Step 3: Implement**

Add to `impl EngramCache` in `engram.rs`:

```rust
/// Read-only lookup. Does NOT update hit/miss counters or entry.hit_count.
/// Use for concurrent read access in the OmniKernel fast path.
pub fn lookup_readonly(&self, intent_canonical: &[u8], min_confidence: f64) -> EngramResult {
    let key = Self::key(intent_canonical);
    match self.store.get(&key) {
        Some(entry) if entry.confidence >= min_confidence => {
            EngramResult::Hit {
                value: entry.value.clone(),
                confidence: entry.confidence,
            }
        }
        _ => EngramResult::Miss,
    }
}
```

**Step 4: Run test**

Run: `cd bizra-omega && cargo test -p bizra-ttrl -- lookup_readonly 2>&1 | tail -5`
Expected: PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-ttrl/src/engram.rs
git commit -m "feat(ttrl): add EngramCache::lookup_readonly for concurrent read access"
```

---

### Task 2.3: Add `try_cache_hit` to OmniKernel

**Files:**
- Modify: `bizra-omega/bizra-agent/src/omni_kernel.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_try_cache_hit_returns_engram_hit() {
    let mut k = make_kernel();
    let c = cycle("what is the capital of France");

    // Pre-populate Engram cache
    k.engram_cache_mut().insert(c.intent_bytes.as_slice(), "Paris", 0.99, c.now_ms);

    // try_cache_hit should find it (read-only)
    let result = k.try_cache_hit(&c, &[]);
    assert!(result.is_some());
    let hit = result.unwrap();
    assert_eq!(hit.path, CyclePath::EngramHit);
    assert_eq!(hit.response, "Paris");
}

#[test]
fn test_try_cache_hit_returns_none_on_miss() {
    let k = make_kernel();
    let c = cycle("novel question");
    let result = k.try_cache_hit(&c, &[]);
    assert!(result.is_none());
}
```

**Step 2: Run test to verify it fails**

Run: `cd bizra-omega && cargo test -p bizra-agent -- try_cache_hit 2>&1 | tail -5`
Expected: FAIL — `try_cache_hit` not found

**Step 3: Implement `try_cache_hit`**

Add to `impl OmniKernel` in `omni_kernel.rs`:

```rust
/// Attempt a read-only cache hit (Tier-1 Reflex or Tier-2 Engram).
///
/// This method takes `&self`, enabling concurrent access when the kernel
/// is wrapped in an `RwLock`. Returns `Some(CacheHitResult)` on hit,
/// `None` on miss (caller should fall through to `run_cycle` for full inference).
///
/// **Important:** This does NOT mint PoI or update telemetry counters.
/// Call `complete_cache_hit()` to finalize the receipt with PoI minting.
pub fn try_cache_hit(
    &self,
    cycle: &OmniCycle,
    level_scores: &[(HhmmLevel, f64)],
) -> Option<CacheHitResult> {
    // Line 1: Chain of Reasoning — check pivots (read-only)
    let reasoning_chain = self.build_reasoning_chain(&cycle.intent, level_scores);
    for pivot in reasoning_chain.decision_pivots() {
        if !pivot.passes(self.config.ihsan_threshold) {
            return None; // Pivot failed — caller should use run_cycle for full receipt
        }
    }

    // Line 2: Tier-1 Reflex Cache (read-only)
    let state_hash = TriggerHash(*blake3::hash(&cycle.intent_bytes).as_bytes());
    if let Some(rule) = self.reflex_cache.lookup_readonly(
        self.reflex_mode,
        &state_hash,
        Some(self.policy_hash),
    ) {
        return Some(CacheHitResult {
            path: CyclePath::ReflexHit,
            ihsan_score: rule.compile_ihsan as f64,
            pivot_chain_hash: reasoning_chain.tail_hash(),
            response: rule.action_template.route_signature.clone(),
        });
    }

    // Line 2b: Tier-2 Engram Cache (read-only)
    match self.engram_cache.lookup_readonly(
        &cycle.intent_bytes,
        self.config.engram_min_confidence,
    ) {
        EngramResult::Hit { value, .. } => {
            Some(CacheHitResult {
                path: CyclePath::EngramHit,
                ihsan_score: self.config.ihsan_threshold, // conservative estimate
                pivot_chain_hash: reasoning_chain.tail_hash(),
                response: value,
            })
        }
        EngramResult::Miss => None,
    }
}
```

And the result struct:

```rust
/// Result from a read-only cache hit. Must be finalized via `complete_cache_hit()`.
#[derive(Debug, Clone)]
pub struct CacheHitResult {
    pub path: CyclePath,
    pub ihsan_score: f64,
    pub pivot_chain_hash: [u8; 32],
    pub response: String,
}
```

**Step 4: Run test**

Run: `cd bizra-omega && cargo test -p bizra-agent -- try_cache_hit 2>&1 | tail -10`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-agent/src/omni_kernel.rs
git commit -m "feat(agent): add OmniKernel::try_cache_hit for concurrent read-only cycles"
```

---

### Task 2.4: Add `complete_cache_hit` to OmniKernel

**Files:**
- Modify: `bizra-omega/bizra-agent/src/omni_kernel.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_complete_cache_hit_mints_poi() {
    let mut k = make_kernel();
    let c = cycle("what is the capital of France");
    k.engram_cache_mut().insert(c.intent_bytes.as_slice(), "Paris", 0.99, c.now_ms);

    let hit = k.try_cache_hit(&c, &[]).unwrap();
    let receipt = k.complete_cache_hit(hit, &c);

    assert_eq!(receipt.path, CyclePath::EngramHit);
    assert!(receipt.poi_yield.is_some());
    assert!(receipt.gate_passed);
    assert_eq!(receipt.response, "Paris");
}
```

**Step 2: Verify failure**

Run: `cd bizra-omega && cargo test -p bizra-agent -- complete_cache_hit 2>&1 | tail -5`

**Step 3: Implement**

```rust
/// Finalize a cache hit by minting PoI and producing a full CycleReceipt.
/// This requires `&mut self` because it updates the MetabolicLedger.
pub fn complete_cache_hit(
    &mut self,
    hit: CacheHitResult,
    cycle: &OmniCycle,
) -> CycleReceipt {
    let is_reflex = matches!(hit.path, CyclePath::ReflexHit);
    let poi = self.metabolic_ledger.mint_poi_yield(
        is_reflex,
        self.config.network_size,
        cycle.now_ms,
    );

    // Update mutable cache stats to keep telemetry accurate
    if is_reflex {
        self.reflex_cache.record_hit();
    } else {
        self.engram_cache.record_hit();
    }

    tracing::debug!(
        path = ?hit.path,
        ihsan = hit.ihsan_score,
        "Omni-Kernel: cache hit completed with PoI mint"
    );

    CycleReceipt {
        path: hit.path,
        ihsan_score: hit.ihsan_score,
        pivot_chain_hash: hit.pivot_chain_hash,
        gate_passed: hit.ihsan_score >= self.config.ihsan_threshold,
        poi_yield: Some(poi),
        ttrl_queued: false,
        response: hit.response,
    }
}
```

Note: `record_hit()` methods need to be added to ReflexCache and EngramCache (simple counter increments). Add:

In `reflex_cache.rs`:
```rust
pub fn record_hit(&mut self) { self.stats.hits += 1; }
```

In `engram.rs`:
```rust
pub fn record_hit(&mut self) { self.hits += 1; }
```

**Step 4: Run test**

Run: `cd bizra-omega && cargo test -p bizra-agent -- complete_cache_hit 2>&1 | tail -5`
Expected: PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-agent/src/omni_kernel.rs bizra-agent/src/reflex_cache.rs bizra-ttrl/src/engram.rs
git commit -m "feat(agent): add OmniKernel::complete_cache_hit for two-phase read/write cycles"
```

---

### Task 2.5: Regression test — existing run_cycle unchanged

**Step 1: Run all existing OmniKernel tests**

Run: `cd bizra-omega && cargo test -p bizra-agent -- omni_kernel 2>&1 | tail -15`
Expected: All 5 existing tests + 4 new tests pass (9 total)

**Step 2: Run full workspace**

Run: `cd bizra-omega && cargo test --workspace 2>&1 | tail -10`
Expected: All pass

---

## Phase 3: PyO3 Event Bridge (bizra-python)

### Task 3.1: Add PyEventBridge class

**Files:**
- Modify: `bizra-omega/bizra-python/src/lib.rs`

**Step 1: Write the failing test (Rust-side unit test)**

Add a test module in `lib.rs` or a new file `bridge.rs`:

```rust
#[cfg(test)]
mod bridge_tests {
    use bizra_hooks::BizraSystem;
    use bizra_hooks::subscribers::wire_all;

    #[test]
    fn bridge_wire_and_emit() {
        let mut system = BizraSystem::new();
        let (wired, errors) = wire_all(&mut system, 1000);
        assert_eq!(wired, 12);
        assert!(errors.is_empty());

        // Register a source component (simulating Python caller)
        let src = system.register_component("python-bridge", "1.0.0", 1500).unwrap();
        system.activate_component(&src).unwrap();

        // Emit an action intent (like Python would)
        let delivered = system.emit(
            src,
            "action.intent",
            bizra_hooks::types::Payload::from_text("organize_invoices"),
            bizra_hooks::types::Priority::Normal,
            2000,
        ).unwrap();
        assert!(delivered >= 1);

        let health = system.health();
        assert!(health.events_emitted >= 1);
    }
}
```

**Step 2: Verify it compiles and passes**

Run: `cd bizra-omega && cargo test -p bizra-python -- bridge 2>&1 | tail -10`

**Step 3: Add the PyEventBridge pyclass**

Add to `lib.rs`:

```rust
/// Python-callable event bridge into the Rust nervous system.
///
/// Usage from Python:
///   from bizra import PyEventBridge
///   bridge = PyEventBridge(production=False)
///   bridge.wire_subscribers()
///   delivered = bridge.emit("action.intent", "organize_invoices", 1)
///   health = bridge.health()
#[pyclass]
pub struct PyEventBridge {
    system: bizra_hooks::BizraSystem,
    source: Option<bizra_hooks::types::ComponentId>,
}

#[pymethods]
impl PyEventBridge {
    #[new]
    #[pyo3(signature = (production = false))]
    fn new(production: bool) -> Self {
        let system = if production {
            bizra_hooks::BizraSystem::production()
        } else {
            bizra_hooks::BizraSystem::new()
        };
        PyEventBridge { system, source: None }
    }

    /// Wire all 12 constitutional subscribers. Returns count wired.
    fn wire_subscribers(&mut self) -> PyResult<usize> {
        let (wired, errors) = bizra_hooks::subscribers::wire_all(&mut self.system, 0);
        if !errors.is_empty() {
            return Err(PyRuntimeError::new_err(
                format!("Failed to wire {} subscribers: {:?}", errors.len(), errors),
            ));
        }

        // Register Python bridge as a source component
        let src = self.system.register_component("python-bridge", "1.0.0", 1)
            .map_err(|e| PyRuntimeError::new_err(format!("Registration failed: {e}")))?;
        self.system.activate_component(&src)
            .map_err(|e| PyRuntimeError::new_err(format!("Activation failed: {e}")))?;
        self.source = Some(src);

        Ok(wired)
    }

    /// Emit an event from Python into the Rust nervous system.
    /// priority: 0=Low, 1=Normal, 2=High, 3=Critical, 4=Emergency
    fn emit(&mut self, topic: &str, payload: &str, priority: u8) -> PyResult<usize> {
        let src = self.source.ok_or_else(||
            PyRuntimeError::new_err("Call wire_subscribers() first")
        )?;
        let prio = match priority {
            0 => bizra_hooks::types::Priority::Low,
            1 => bizra_hooks::types::Priority::Normal,
            2 => bizra_hooks::types::Priority::High,
            3 => bizra_hooks::types::Priority::Critical,
            4 => bizra_hooks::types::Priority::Emergency,
            _ => return Err(PyValueError::new_err("priority must be 0-4")),
        };
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.system.emit(
            src, topic,
            bizra_hooks::types::Payload::from_text(payload),
            prio, now_ns,
        ).map_err(|e| PyRuntimeError::new_err(format!("Emit failed: {e}")))
    }

    /// Get system health as a Python dict.
    fn health(&self, py: Python<'_>) -> PyResult<PyObject> {
        let h = self.system.health();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("events_emitted", h.events_emitted)?;
        dict.set_item("events_delivered", h.events_delivered)?;
        dict.set_item("events_dropped", h.events_dropped)?;
        dict.set_item("delivery_ratio", h.delivery_ratio)?;
        dict.set_item("active_subscriptions", h.active_subscriptions)?;
        dict.set_item("system_ihsan", h.system_ihsan.as_f64())?;
        dict.set_item("gate_evaluations", h.gate_evaluations)?;
        dict.set_item("gate_violations", h.gate_violations)?;
        dict.set_item("gate_stability", h.gate_stability)?;
        Ok(dict.into())
    }
}
```

And register the class in the module init:

Find the `#[pymodule]` function and add: `m.add_class::<PyEventBridge>()?;`

**Step 4: Run test**

Run: `cd bizra-omega && cargo test -p bizra-python -- bridge 2>&1 | tail -10`
Expected: PASS

**Step 5: Commit**

```bash
cd bizra-omega && git add bizra-python/src/lib.rs
git commit -m "feat(python): add PyEventBridge for Python→Rust event forwarding

Exposes BizraSystem to Python via PyO3. Supports:
- wire_subscribers(): Wire all 12 constitutional subscribers
- emit(topic, payload, priority): Forward events into Rust nervous system
- health(): Get system health snapshot as Python dict"
```

---

### Task 3.2: Add Python-side RustBridge wrapper

**Files:**
- Modify: `core/sovereign/event_bus.py`

**Step 1: Write the test**

Create or modify test file:

```python
# tests/core/test_rust_bridge.py
import pytest

def test_rust_bridge_import():
    """Verify PyEventBridge can be imported from bizra."""
    try:
        from bizra import PyEventBridge
    except ImportError:
        pytest.skip("bizra native module not built (run: cd bizra-omega/bizra-python && maturin develop)")

def test_rust_bridge_wire_and_emit():
    """Verify event bridge can wire subscribers and emit."""
    try:
        from bizra import PyEventBridge
    except ImportError:
        pytest.skip("bizra native module not built")

    bridge = PyEventBridge(production=False)
    wired = bridge.wire_subscribers()
    assert wired == 12

    delivered = bridge.emit("action.intent", "test_payload", 1)
    assert delivered >= 1

    health = bridge.health()
    assert health["events_emitted"] >= 1
    assert health["active_subscriptions"] >= 12
```

**Step 2: Run test (may skip if not built)**

Run: `pytest tests/core/test_rust_bridge.py -v 2>&1 | tail -10`

**Step 3: Add RustBridge class to event_bus.py**

Add at the bottom of `core/sovereign/event_bus.py`:

```python
class RustBridge:
    """Forward events to the Rust BizraSystem via PyO3 bindings.

    This connects the Python orchestration layer to the Rust nervous system,
    enabling Python-originated events (proactive suggestions, opportunity
    detection) to flow through the constitutional enforcement pipeline.

    Usage:
        bridge = RustBridge(production=False)
        bridge.emit("action.intent", "organize_invoices")
        health = bridge.health()
    """

    def __init__(self, production: bool = False):
        try:
            from bizra import PyEventBridge
            self._bridge = PyEventBridge(production=production)
            self._wired = self._bridge.wire_subscribers()
        except ImportError:
            self._bridge = None
            self._wired = 0

    @property
    def available(self) -> bool:
        return self._bridge is not None

    def emit(self, topic: str, payload: str, priority: int = 1) -> int:
        if self._bridge is None:
            return 0
        return self._bridge.emit(topic, payload, priority)

    def health(self) -> dict:
        if self._bridge is None:
            return {"available": False}
        h = self._bridge.health()
        h["available"] = True
        return h
```

**Step 4: Run test**

Run: `pytest tests/core/test_rust_bridge.py -v 2>&1 | tail -10`

**Step 5: Commit**

```bash
git add core/sovereign/event_bus.py tests/core/test_rust_bridge.py
git commit -m "feat(sovereign): add RustBridge for Python→Rust event forwarding"
```

---

## Phase 4: Final Regression + Verification

### Task 4.1: Full Rust workspace test

Run: `cd bizra-omega && cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass (612+ existing + new UCF tests)

### Task 4.2: Clippy + fmt check

Run: `cd bizra-omega && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -10`
Expected: Clean

### Task 4.3: Python test suite

Run: `pytest tests/ -m "not requires_ollama and not requires_gpu and not slow" 2>&1 | tail -10`
Expected: All pass

### Task 4.4: Final commit (tag)

```bash
git add -A
git commit -m "feat: Unified Concurrency Fabric (UCF) — Phase 1 complete

Three interdependent optimizations forming the P0 Trinity:

1. Namespace-Sharded EventBus (bizra-hooks)
   - 8 shards by topic prefix, eliminating cross-namespace contention
   - O(N/8) dispatch instead of O(N) per emit
   - Zero new dependencies, no_std compatible

2. Two-Phase OmniKernel (bizra-agent)
   - try_cache_hit(&self) for concurrent Tier-1/Tier-2 reads
   - complete_cache_hit(&mut self) for brief PoI minting
   - run_cycle unchanged for backward compatibility

3. PyO3 Event Bridge (bizra-python → core/)
   - PyEventBridge class: wire_subscribers(), emit(), health()
   - RustBridge Python wrapper with graceful fallback

Standing on: Shannon, Lamport, Hoare, Al-Ghazali, Boyd"
```
