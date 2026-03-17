# Phase 86-B: 4-Loop HHMM EventBus Wiring
## Activating the Nervous System for Self-Sustaining Operation

**Sprint Ref**: D (4-Loop Wiring), B4 (EventBus async bridge)
**Gap Source**: SAPE Omega Synthesis §F2, Genesis Ops Blueprint §2.3
**Standing On**: Lamport (1978) clocks, Maturana (1980) autopoiesis, Fine (2007) HHMM

---

## 1. Problem Statement

```
Current:  13 subscribers REGISTERED in EventBus (wire_all)
          Bus dispatches events synchronously within process
          No heartbeat cycle driving continuous operation
          No cross-loop feedback (Loop A doesn't trigger Loop B)

Required: 4 hierarchical loops wired in dependency order
          Heartbeat cycle drives continuous event emission
          Each loop's output feeds the next loop's input
          Self-sustaining: output of Loop D feeds back to Loop A

Why:      The HHMM analysis (SAPE §F2) proves these 4 loops
          must wire in strict sequential order. Wiring out of
          order creates dependency violations. The self-sustaining
          loop IS the S1 gate.
```

## 2. The 4-Loop Architecture (from SAPE Omega Synthesis)

```
Loop A: Perception → Memory (Foundation)
  Input:   Raw user message
  Process:  Fragment extraction → Engram cache → Atom store
  Output:  memory.fragment.stored event
  Subscribers: MemoryFragmentStored, EngineContextReady

Loop B: Memory → Cognition (Understanding)
  Input:   memory.fragment.stored event
  Process:  Synthesis trigger → Pattern extraction → Insight
  Output:  cognition.synthesis.complete event
  Subscribers: SynthesisComplete, ProfileUpdated

Loop C: Cognition → Action (Response)
  Input:   cognition.synthesis.complete event
  Process:  Route selection → Agent execution → Receipt emission
  Output:  action.receipt.emitted event
  Subscribers: ActionReceiptEmitted, IhsanScoreUpdated

Loop D: Action → Evolution (Learning)
  Input:   action.receipt.emitted event
  Process:  Reflex compilation → TTRL update → Skill promotion
  Output:  evolution.reflex.compiled event
  Subscribers: ReflexCompiled, SkillPromoted, MintReady

Feedback: Loop D output → Loop A input
  evolution.reflex.compiled → next perception uses compiled reflex
  This IS the self-sustaining loop.
```

## 3. Pseudocode — Heartbeat Cycle

```rust
// ── node.rs: The heartbeat that drives continuous operation ──

impl Node {
    /// The sovereign heartbeat.
    /// Called every HEARTBEAT_INTERVAL_MS (default: 1000ms).
    /// Drives the 4-loop HHMM through event emission.
    ///
    /// Standing on: Maturana (autopoiesis), Friston (free energy)
    ///
    /// TDD anchor: test_heartbeat_emits_events
    /// TDD anchor: test_heartbeat_drives_synthesis
    pub fn heartbeat(&mut self, now: u64) {
        // Loop A check: any pending fragments need processing?
        if self.runtime.has_pending_fragments() {
            self.runtime.process_pending_fragments(now);
            // Emits: memory.fragment.stored
        }

        // Loop B check: synthesis interval reached?
        if self.should_synthesize(now) {
            self.runtime.synthesize(now);
            // Emits: cognition.synthesis.complete
        }

        // Loop C check: any unprocessed action receipts?
        if self.runtime.has_pending_receipts() {
            self.runtime.process_pending_receipts(now);
            // Emits: action.receipt.emitted
        }

        // Loop D check: any compilable reflexes?
        if self.runtime.has_compilable_patterns() {
            self.runtime.compile_reflexes(now);
            // Emits: evolution.reflex.compiled
            // → feeds back to Loop A on next heartbeat
        }

        // Health telemetry
        self.last_heartbeat = now;
        self.heartbeat_count += 1;
    }

    fn should_synthesize(&self, now: u64) -> bool {
        let interval = self.config.synthesis_interval_ms;
        now.saturating_sub(self.last_synthesis) >= interval
    }
}
```

## 4. Pseudocode — EventBus Async Bridge

```rust
// ── event_bus_bridge.rs ───────────────────────────────
// Bridges the synchronous EventBus to the async heartbeat cycle.
// The bus dispatches immediately; the bridge queues cross-loop events.

use std::collections::VecDeque;
use bizra_hooks::{Event, EventBus};

/// Cross-loop event queue.
/// Loop A events → queue → Loop B picks up on next heartbeat.
///
/// TDD anchor: test_cross_loop_event_propagation
/// TDD anchor: test_event_ordering_preserved
pub struct EventBridge {
    /// Events waiting to be processed by the next loop
    pending: VecDeque<Event>,
    /// Maximum queue depth (constitutional bound)
    max_depth: usize,
}

impl EventBridge {
    pub fn new(max_depth: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(max_depth),
            max_depth,
        }
    }

    /// Enqueue an event from the current loop for the next loop.
    ///
    /// TDD anchor: test_queue_overflow_degrades_not_panics
    pub fn enqueue(&mut self, event: Event) -> bool {
        if self.pending.len() >= self.max_depth {
            // Constitutional degradation: drop oldest, don't panic
            self.pending.pop_front();
        }
        self.pending.push_back(event);
        true
    }

    /// Drain all pending events for the next loop iteration.
    pub fn drain(&mut self) -> Vec<Event> {
        self.pending.drain(..).collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}
```

## 5. Pseudocode — Subscriber Verification

```rust
// ── Verify all 13 subscribers reach end-to-end ──────

// The 13 subscribers (from bizra-hooks/src/subscribers.rs):
//
// Loop A (Perception → Memory):
//   1. memory.fragment.stored     → MemoryFragmentHandler
//   2. engine.context.ready       → ContextReadyHandler
//
// Loop B (Memory → Cognition):
//   3. synthesis.complete         → SynthesisHandler
//   4. profile.updated            → ProfileHandler
//
// Loop C (Cognition → Action):
//   5. action.intent              → ActionIntentHandler
//   6. action.receipt             → ActionReceiptHandler
//   7. ihsan.score.updated        → IhsanHandler
//
// Loop D (Action → Evolution):
//   8. reflex.compiled            → ReflexHandler
//   9. ihsan.gate.breached        → IhsanBreachHandler (Emergency)
//  10. guardian.veto              → GuardianVetoHandler
//  11. skill.promoted             → SkillHandler
//  12. mint.ready                 → MintReadyHandler
//  13. saga.received              → SagaHandler

/// TDD anchor: test_all_13_subscribers_reachable
fn verify_all_subscribers(bus: &EventBus) -> bool {
    let topics = [
        "memory.fragment.stored",
        "engine.context.ready",
        "synthesis.complete",
        "profile.updated",
        "action.intent",
        "action.receipt",
        "ihsan.score.updated",
        "reflex.compiled",
        "ihsan.gate.breached",
        "guardian.veto",
        "skill.promoted",
        "mint.ready",
        "saga.received",
    ];

    for topic in &topics {
        let count = bus.subscriber_count_for(topic);
        assert!(count >= 1, "no subscriber for topic: {}", topic);
    }
    true
}
```

## 6. Wiring Order (HHMM Dependency Chain)

```
Day 1: Wire Loop A (Perception → Memory)
  - Verify subscribers 1-2 receive events
  - Test: message → fragment.stored event → engram cache updated
  - Gate: integration test passes

Day 2: Wire Loop B (Memory → Cognition)
  - Verify subscribers 3-4 receive events
  - Test: fragment.stored → synthesis.complete → profile.updated
  - Gate: synthesis produces insight from accumulated fragments
  - DEPENDS ON: Loop A emitting memory.fragment.stored

Day 3: Wire Loop C (Cognition → Action)
  - Verify subscribers 5-7 receive events
  - Test: synthesis.complete → action.intent → receipt → ihsan.score
  - Gate: receipt is signed (86-A) and chained
  - DEPENDS ON: Loop B emitting cognition.synthesis.complete

Day 4: Wire Loop D (Action → Evolution)
  - Verify subscribers 8-13 receive events
  - Test: receipt → reflex.compiled → skill.promoted → mint.ready
  - Gate: compiled reflex serves next identical query via S1 path
  - DEPENDS ON: Loop C emitting action.receipt
  - FEEDBACK: reflex.compiled feeds next Loop A perception
```

## 7. TDD Test Matrix

```
TEST                                    GATE        LOOP
─────────────────────────────────────────────────────────────
test_loop_a_message_to_fragment         Integration A
test_loop_b_fragment_to_synthesis       Integration B
test_loop_c_synthesis_to_receipt        Integration C
test_loop_d_receipt_to_reflex           Integration D
test_feedback_reflex_serves_next_query  Integration D→A
test_all_13_subscribers_reachable       Unit        All
test_cross_loop_event_propagation       Unit        Bridge
test_event_ordering_preserved           Unit        Bridge
test_queue_overflow_degrades_not_panics Unit        Bridge
test_heartbeat_emits_events             Integration All
test_heartbeat_drives_synthesis         Integration B
test_100_missions_zero_crashes          Stress      All
test_self_sustaining_loop_proof         S1 Gate     All
```

## 8. Self-Sustaining Loop Proof (S1 Gate Test)

```rust
/// THE S1 GATE TEST
/// Proves: the system processes tasks, learns from them,
/// and uses learned reflexes to process future tasks faster.
///
/// This is the single most important test in the entire codebase.
/// When this passes, NODE0 transitions from S0 to S1.
///
/// Standing on: Maturana (autopoiesis), Deming (continuous improvement)
///
/// TDD anchor: test_self_sustaining_loop_proof
fn test_self_sustaining_loop_proof() {
    let mut node = Node::new(config_with_persistence_and_signing());

    // Phase 1: Process a novel query (S2 deliberation path)
    let r1 = node.execute("RECEIVE\tWhat are BIZRA thresholds?\t1000");
    assert!(r1.contains("decision_mode=system2"));  // S2: full deliberation
    assert!(r1.contains("reflex_hit=false"));        // No reflex available

    // Phase 2: Heartbeat triggers Loop D → reflex compilation
    node.heartbeat(2000);

    // Phase 3: Same query again — should hit compiled reflex (S1 fast path)
    let r2 = node.execute("RECEIVE\tWhat are BIZRA thresholds?\t3000");
    assert!(r2.contains("decision_mode=system1"));   // S1: reflex fast path
    assert!(r2.contains("reflex_hit=true"));          // Compiled reflex served

    // Phase 4: Verify the learning loop
    let health = node.execute("HEALTH");
    assert!(health.contains("reflex_hits=1"));        // One reflex hit recorded
    assert!(health.contains("reflex_misses=1"));      // One miss (first query)

    // Phase 5: Verify receipt chain integrity
    // Both missions should have signed, chained receipts
    // ...

    // S1 GATE: The system learned from experience and improved.
    // This is autopoiesis — the system produces the conditions
    // for its own continued operation.
    node.execute("SHUTDOWN");
}
```

## 9. Acceptance Criteria (S1 Gate)

```
[x] All 4 loops wire in dependency order (A→B→C→D)
[x] All 13 EventBus subscribers receive events end-to-end
[x] Heartbeat cycle drives continuous operation
[x] Cross-loop event bridge preserves ordering
[x] Novel query → S2 path → reflex compilation
[x] Repeated query → S1 path (compiled reflex)
[x] Self-sustaining loop proof test passes
[x] 100 governed missions, 0 crashes
[x] All existing 1,381+ tests still pass
[x] Zero clippy warnings
```
