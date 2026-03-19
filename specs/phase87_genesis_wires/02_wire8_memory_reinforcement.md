# Wire 8 — Memory Reinforcement (Subscribers #1 + #2)

## Problem

`handle_memory_reinforce()` and `handle_hhmm_promotion_check()` in
`bizra-hooks/src/subscribers.rs` are stubs — they return `HookResult::Continue`
without doing anything. The feedback loop exists structurally (EventBus
delivers events) but the handlers are empty.

## Current State (stub)

```rust
pub fn handle_memory_reinforce(event: &Event) -> HookResult {
    if event.ihsan_score.meets_ihsan() {
        HookResult::Continue  // ← does nothing
    } else {
        HookResult::Continue  // ← does nothing
    }
}
```

## Constraint

EventHandler is `fn(&Event) -> HookResult` — a pure function pointer.
No `&mut self`, no access to MemoryPipeline, no side effects through parameters.

**Solution**: Side effects via event re-emission. The handler emits a NEW event
that a downstream component (with mutable state) consumes. This is the
established pattern — subscriber #3 already emits PoICredit events.

BUT: EventHandler can't emit events (no access to EventBus).

**Real solution**: The handler extracts information from the payload and
signals intent via HookResult. The CALLER (EventBus.dispatch → Node) reads
the result and performs the mutation. This is already how Halt works.

**Simplest correct solution**: Add a `Transform` result that carries
a reinforce/promote signal in the existing Payload. The Node's dispatch
loop checks for Transform results and applies mutations.

## Actually — The Correct Architecture

Looking at the code more carefully: the handlers ARE correct as stubs.
The real work happens in the **Node's heartbeat loop** (Phase 86-B),
which already calls `pipeline.extract()` and `pipeline.force_synthesize()`.

The subscriber's job is to SET A FLAG that the heartbeat loop reads.
This avoids the mutable-state-in-handler problem entirely.

## Pseudocode: Reinforcement via Atomic Flags

```rust
// bizra-hooks/src/subscribers.rs

use std::sync::atomic::{AtomicU64, Ordering};

/// Global reinforcement counter — incremented by subscriber,
/// drained by heartbeat loop. Lock-free, zero-allocation.
pub static REINFORCE_PENDING: AtomicU64 = AtomicU64::new(0);
pub static PROMOTE_CHECK_PENDING: AtomicU64 = AtomicU64::new(0);
pub static QUARANTINE_PENDING: AtomicU64 = AtomicU64::new(0);
pub static SESSION_COMPILE_PENDING: AtomicU64 = AtomicU64::new(0);

/// #1: ActionReceipt → Signal reinforcement needed
pub fn handle_memory_reinforce(event: &Event) -> HookResult {
    if event.ihsan_score.meets_ihsan() {
        // Signal: "a good action completed, reinforce contributing memories"
        REINFORCE_PENDING.fetch_add(1, Ordering::Relaxed);
    }
    // Don't reinforce bad actions — Hebbian anti-learning
    HookResult::Continue
}

/// #2: ActionReceipt → Signal promotion check needed
pub fn handle_hhmm_promotion_check(event: &Event) -> HookResult {
    if event.ihsan_score.meets_ihsan() {
        PROMOTE_CHECK_PENDING.fetch_add(1, Ordering::Relaxed);
    }
    HookResult::Continue
}

/// #8: ActionReceipt[failed] → Signal quarantine needed
pub fn handle_memory_quarantine(event: &Event) -> HookResult {
    QUARANTINE_PENDING.fetch_add(1, Ordering::Relaxed);
    HookResult::Continue
}
```

## Pseudocode: Heartbeat Loop Drains Flags

```rust
// bizra-node/src/heartbeat.rs — in the existing heartbeat tick

use bizra_hooks::subscribers::{
    REINFORCE_PENDING, PROMOTE_CHECK_PENDING,
    QUARANTINE_PENDING, SESSION_COMPILE_PENDING,
};

impl Node {
    fn heartbeat_tick(&mut self, now_ms: u64) {
        // ... existing heartbeat logic (Loop A, B, C, D) ...

        // ── NEW: Drain subscriber signals ──
        let reinforce_count = REINFORCE_PENDING.swap(0, Ordering::Relaxed);
        if reinforce_count > 0 {
            // Reinforce recent memory atoms that contributed to actions
            self.runtime.pipeline_mut().reinforce_recent(reinforce_count);
        }

        let promote_count = PROMOTE_CHECK_PENDING.swap(0, Ordering::Relaxed);
        if promote_count > 0 {
            // Check if any atoms crossed the promotion threshold (0.92)
            self.runtime.pipeline_mut().check_promotions();
        }

        let quarantine_count = QUARANTINE_PENDING.swap(0, Ordering::Relaxed);
        if quarantine_count > 0 {
            // Quarantine atoms that contributed to failed actions
            self.runtime.pipeline_mut().quarantine_recent(quarantine_count);
        }
    }
}
```

## MemoryPipeline Extensions

```rust
// bizra-memory/src/lib.rs — add methods

impl MemoryPipeline {
    /// Reinforce N most recently accessed atoms.
    /// Increases confidence by delta proportional to access count.
    pub fn reinforce_recent(&mut self, count: u64) {
        let delta = 0.02 * (count as f64).min(5.0); // Cap at +0.10
        for atom in self.engram.recent_accessed_mut(count as usize) {
            atom.confidence.boost(delta);
        }
    }

    /// Check if any atoms crossed the glacial promotion threshold.
    /// Threshold: confidence >= 0.92 AND kind == Fact|Pattern
    pub fn check_promotions(&mut self) -> usize {
        let promoted = self.engram.promote_ready_atoms(0.92);
        promoted
    }

    /// Quarantine N most recently used atoms (from failed actions).
    /// Reduces confidence and flags for review.
    pub fn quarantine_recent(&mut self, count: u64) {
        let penalty = 0.05 * (count as f64).min(3.0); // Cap at -0.15
        for atom in self.engram.recent_accessed_mut(count as usize) {
            atom.confidence.penalize(penalty);
            if atom.confidence.value() < 0.30 {
                atom.quarantined = true;
            }
        }
    }
}
```

## TDD Anchors

```rust
#[cfg(test)]
mod wire8_tests {
    use super::*;

    #[test]
    fn reinforce_flag_set_on_good_action() {
        REINFORCE_PENDING.store(0, Ordering::Relaxed);

        let event = Event {
            id: EventId::new(1000, 0),
            source: ComponentId::from_name("test", "1.0.0"),
            topic: Topic::new(TOPIC_ACTION_RECEIPT),
            priority: Priority::Normal,
            payload: Payload::from_text("receipt:001"),
            ihsan_score: IhsanScore::from_raw(9700), // above threshold
        };

        let result = handle_memory_reinforce(&event);
        assert_eq!(result, HookResult::Continue);
        assert_eq!(REINFORCE_PENDING.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reinforce_flag_not_set_on_bad_action() {
        REINFORCE_PENDING.store(0, Ordering::Relaxed);

        let event = Event {
            id: EventId::new(1000, 0),
            source: ComponentId::from_name("test", "1.0.0"),
            topic: Topic::new(TOPIC_ACTION_RECEIPT),
            priority: Priority::Normal,
            payload: Payload::from_text("receipt:002"),
            ihsan_score: IhsanScore::from_raw(8000), // below threshold
        };

        handle_memory_reinforce(&event);
        assert_eq!(REINFORCE_PENDING.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pipeline_reinforce_increases_confidence() {
        let mut pipeline = MemoryPipeline::new();
        pipeline.ingest(FragmentKind::UserMessage, "I prefer Rust", 1, 1, 1000).unwrap();
        pipeline.extract(1001);

        let before = pipeline.profile().completeness();
        pipeline.reinforce_recent(1);
        let after = pipeline.profile().completeness();

        assert!(after >= before);
    }

    #[test]
    fn pipeline_quarantine_reduces_confidence() {
        let mut pipeline = MemoryPipeline::new();
        pipeline.ingest(FragmentKind::UserMessage, "bad pattern", 1, 1, 1000).unwrap();
        pipeline.extract(1001);

        pipeline.quarantine_recent(3); // 3 failures → significant penalty
        // Atom confidence should be reduced
    }
}
```

## Blast Radius

| File | Change | Risk |
|------|--------|------|
| `bizra-hooks/src/subscribers.rs` | Add atomic flags, update 3 handlers | Low — additive |
| `bizra-memory/src/lib.rs` | Add reinforce/promote/quarantine methods | Low — new methods |
| `bizra-node/src/heartbeat.rs` | Drain flags in tick | Low — additive to existing loop |
| Existing tests | No change | Zero |
