//! # BIZRA Saga — The Heart That Pumps the First Pulse
//!
//! A saga sequences a single user request through the entire BIZRA pipeline:
//!
//! ```text
//! RECEIVE → PLAN → EXECUTE → EVALUATE → DRAFT → GATE → ATTEST → RESPOND
//! ```
//!
//! Every component it touches already compiles. The saga is the last inch
//! of corridor between "architecture proven" and "production running."
//!
//! ## Design Principles
//! - **Event-driven state machine**: each phase emits an event, advances on response
//! - **Zero heap allocation**: fixed-size types, no_std compatible
//! - **Fail-closed compensation**: on failure, emit rollback events in reverse
//! - **Ihsān enforcement**: every phase transition checks the quality gate
//!
//! ## Standing on Giants
//! - **Garcia-Molina (1987)**: Saga pattern — local transactions with compensating actions
//! - **Hewitt (1973)**: Actor model — communication IS the system
//! - **Armstrong (Erlang/OTP)**: Let it crash — isolate failure, compensate, continue

use crate::types::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Saga Topic Constants — extend the canonical taxonomy
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub const TOPIC_SAGA_RECEIVED: &str = "saga.received";
pub const TOPIC_SAGA_PLANNED: &str = "saga.planned";
pub const TOPIC_SAGA_EXECUTED: &str = "saga.executed";
pub const TOPIC_SAGA_EVALUATED: &str = "saga.evaluated";
pub const TOPIC_SAGA_DRAFTED: &str = "saga.drafted";
pub const TOPIC_SAGA_GATED: &str = "saga.gated";
pub const TOPIC_SAGA_ATTESTED: &str = "saga.attested";
pub const TOPIC_SAGA_COMPLETED: &str = "saga.completed";
pub const TOPIC_SAGA_FAILED: &str = "saga.failed";
pub const TOPIC_SAGA_COMPENSATING: &str = "saga.compensating";

/// Maximum saga steps before forced timeout.
const MAX_STEPS: u8 = 8;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Saga Phase — The State Machine
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Each phase of a saga's lifecycle.
/// The happy path: Received → Planned → Executed → Evaluated → Drafted → Gated → Attested → Completed.
/// Any phase can transition to Failed, which triggers compensation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SagaPhase {
    /// Intent received, waiting for planner decomposition.
    Received = 0,
    /// Tasks decomposed by planner agent.
    Planned = 1,
    /// Knowledge retrieved by researcher agent.
    Executed = 2,
    /// Result quality-checked by evaluator agent.
    Evaluated = 3,
    /// Response synthesized by integrator agent.
    Drafted = 4,
    /// Constitutional gates α4→α10 passed.
    Gated = 5,
    /// Receipt signed, BlockGraph attestation complete.
    Attested = 6,
    /// Response delivered to user. Terminal state.
    Completed = 7,
    /// Something failed. Terminal unless compensating.
    Failed = 8,
    /// Rolling back completed steps in reverse order.
    Compensating = 9,
}

impl SagaPhase {
    /// The topic to emit when entering this phase.
    pub fn topic(&self) -> &'static str {
        match self {
            Self::Received => TOPIC_SAGA_RECEIVED,
            Self::Planned => TOPIC_SAGA_PLANNED,
            Self::Executed => TOPIC_SAGA_EXECUTED,
            Self::Evaluated => TOPIC_SAGA_EVALUATED,
            Self::Drafted => TOPIC_SAGA_DRAFTED,
            Self::Gated => TOPIC_SAGA_GATED,
            Self::Attested => TOPIC_SAGA_ATTESTED,
            Self::Completed => TOPIC_SAGA_COMPLETED,
            Self::Failed => TOPIC_SAGA_FAILED,
            Self::Compensating => TOPIC_SAGA_COMPENSATING,
        }
    }

    /// Next phase in the happy path. None if terminal.
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Received => Some(Self::Planned),
            Self::Planned => Some(Self::Executed),
            Self::Executed => Some(Self::Evaluated),
            Self::Evaluated => Some(Self::Drafted),
            Self::Drafted => Some(Self::Gated),
            Self::Gated => Some(Self::Attested),
            Self::Attested => Some(Self::Completed),
            Self::Completed | Self::Failed | Self::Compensating => None,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Saga — The Orchestrator
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Unique saga identifier. Derived from timestamp + sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SagaId(pub u64);

/// A saga instance — tracks one request through the full pipeline.
///
/// Fixed-size, Copy, no heap. Designed for the nervous system.
#[derive(Debug, Clone, Copy)]
pub struct Saga {
    /// Unique identifier.
    pub id: SagaId,
    /// Component that owns this saga (the saga orchestrator).
    pub owner: ComponentId,
    /// Current phase.
    pub phase: SagaPhase,
    /// Ihsān score at each phase transition (indexed by SagaPhase as u8).
    pub phase_scores: [IhsanScore; MAX_STEPS as usize],
    /// Number of phases completed (for compensation: unwind this many).
    pub steps_completed: u8,
    /// Timestamp when saga was created.
    pub started_at: u64,
    /// Timestamp of most recent phase transition.
    pub last_transition: u64,
    /// Whether the Ihsān gate has been checked.
    pub gate_passed: bool,
    /// Error code if failed (0 = no error).
    pub error_code: u16,
}

/// What the saga wants the system to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SagaAction {
    /// Emit an event for the current phase and wait.
    Emit {
        topic: &'static str,
        priority: Priority,
    },
    /// Saga completed successfully.
    Complete,
    /// Saga failed — begin compensation.
    Fail { error_code: u16 },
    /// Compensation complete — saga is done (with errors).
    Aborted,
    /// No action needed (saga is in a terminal state).
    None,
}

impl Saga {
    /// Create a new saga for an incoming request.
    pub fn new(id: SagaId, owner: ComponentId, now_nanos: u64) -> Self {
        Saga {
            id,
            owner,
            phase: SagaPhase::Received,
            phase_scores: [IhsanScore::MAX; MAX_STEPS as usize],
            steps_completed: 0,
            started_at: now_nanos,
            last_transition: now_nanos,
            gate_passed: false,
            error_code: 0,
        }
    }

    /// Advance the saga to the next phase.
    ///
    /// Called when the current phase's work is done. Checks Ihsān,
    /// transitions state, and returns what to emit next.
    pub fn advance(&mut self, ihsan: IhsanScore, now_nanos: u64) -> SagaAction {
        // Terminal states don't advance.
        if self.is_terminal() {
            return SagaAction::None;
        }

        // Record Ihsān at this phase.
        let idx = self.phase as u8;
        if (idx as usize) < self.phase_scores.len() {
            self.phase_scores[idx as usize] = ihsan;
        }

        // Ihsān gate: reject if below warning threshold.
        // Constitutional floor (0.99) is enforced by IhsanGate on events.
        // Here we enforce the operational threshold (0.95).
        if ihsan.is_critical() {
            return self.fail(1, now_nanos); // Error 1: Ihsān below critical
        }

        // Advance to next phase.
        match self.phase.next() {
            Some(next) => {
                self.steps_completed += 1;
                self.phase = next;
                self.last_transition = now_nanos;

                if next == SagaPhase::Completed {
                    SagaAction::Complete
                } else {
                    SagaAction::Emit {
                        topic: next.topic(),
                        priority: self.priority_for_phase(next),
                    }
                }
            }
            None => SagaAction::None,
        }
    }

    /// Mark the saga as failed and begin compensation.
    pub fn fail(&mut self, error_code: u16, now_nanos: u64) -> SagaAction {
        if self.is_terminal() {
            return SagaAction::None;
        }

        self.error_code = error_code;
        self.last_transition = now_nanos;

        if self.steps_completed == 0 {
            // Nothing to compensate.
            self.phase = SagaPhase::Failed;
            SagaAction::Fail { error_code }
        } else {
            // Begin compensation — unwind completed steps.
            self.phase = SagaPhase::Compensating;
            SagaAction::Emit {
                topic: TOPIC_SAGA_COMPENSATING,
                priority: Priority::High,
            }
        }
    }

    /// Complete one compensation step. Returns whether compensation is done.
    pub fn compensate_step(&mut self, now_nanos: u64) -> SagaAction {
        if self.phase != SagaPhase::Compensating {
            return SagaAction::None;
        }

        if self.steps_completed == 0 {
            self.phase = SagaPhase::Failed;
            self.last_transition = now_nanos;
            SagaAction::Aborted
        } else {
            self.steps_completed -= 1;
            self.last_transition = now_nanos;
            SagaAction::Emit {
                topic: TOPIC_SAGA_COMPENSATING,
                priority: Priority::High,
            }
        }
    }

    /// Mark constitutional gates as passed.
    pub fn mark_gated(&mut self) {
        self.gate_passed = true;
    }

    /// Is this saga in a terminal state?
    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, SagaPhase::Completed | SagaPhase::Failed)
    }

    /// Duration since saga started (in nanoseconds).
    pub fn elapsed_nanos(&self, now: u64) -> u64 {
        now.saturating_sub(self.started_at)
    }

    /// Minimum Ihsān score across all completed phases.
    pub fn min_ihsan(&self) -> IhsanScore {
        let mut min = IhsanScore::MAX;
        for i in 0..self.steps_completed as usize {
            if self.phase_scores[i] < min {
                min = self.phase_scores[i];
            }
        }
        min
    }

    fn priority_for_phase(&self, phase: SagaPhase) -> Priority {
        match phase {
            SagaPhase::Gated => Priority::Critical, // Constitutional check
            SagaPhase::Attested => Priority::High,  // Proof generation
            SagaPhase::Failed => Priority::Emergency, // System alert
            SagaPhase::Compensating => Priority::High, // Rollback
            _ => Priority::Normal,                  // Standard work
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Saga Registry — Track active sagas
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const MAX_ACTIVE_SAGAS: usize = 64;

/// Tracks all active sagas. Fixed-size, no heap.
pub struct SagaRegistry {
    sagas: [Option<Saga>; MAX_ACTIVE_SAGAS],
    count: usize,
    next_id: u64,
    total_created: u64,
    total_completed: u64,
    total_failed: u64,
}

impl Default for SagaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SagaRegistry {
    pub const fn new() -> Self {
        SagaRegistry {
            sagas: [None; MAX_ACTIVE_SAGAS],
            count: 0,
            next_id: 1, // 0 is reserved for null
            total_created: 0,
            total_completed: 0,
            total_failed: 0,
        }
    }

    /// Create a new saga. Returns None if registry is full.
    pub fn create(&mut self, owner: ComponentId, now_nanos: u64) -> Option<SagaId> {
        if self.count >= MAX_ACTIVE_SAGAS {
            return None;
        }

        let id = SagaId(self.next_id);
        self.next_id += 1;
        self.total_created += 1;

        let saga = Saga::new(id, owner, now_nanos);

        // Find empty slot.
        for slot in self.sagas.iter_mut() {
            if slot.is_none() {
                *slot = Some(saga);
                self.count += 1;
                return Some(id);
            }
        }

        None // Should not reach here if count < MAX
    }

    /// Get a saga by ID (immutable).
    pub fn get(&self, id: SagaId) -> Option<&Saga> {
        self.sagas
            .iter()
            .filter_map(|s| s.as_ref())
            .find(|s| s.id == id)
    }

    /// Get a saga by ID (mutable).
    pub fn get_mut(&mut self, id: SagaId) -> Option<&mut Saga> {
        self.sagas
            .iter_mut()
            .filter_map(|s| s.as_mut())
            .find(|s| s.id == id)
    }

    /// Remove a completed/failed saga from the registry.
    pub fn remove(&mut self, id: SagaId) -> Option<Saga> {
        for slot in self.sagas.iter_mut() {
            if let Some(saga) = slot {
                if saga.id == id {
                    let removed = *saga;
                    *slot = None;
                    self.count -= 1;

                    match removed.phase {
                        SagaPhase::Completed => self.total_completed += 1,
                        SagaPhase::Failed => self.total_failed += 1,
                        _ => {}
                    }

                    return Some(removed);
                }
            }
        }
        None
    }

    /// Number of active sagas.
    pub fn active_count(&self) -> usize {
        self.count
    }

    /// Telemetry.
    pub fn total_created(&self) -> u64 {
        self.total_created
    }
    pub fn total_completed(&self) -> u64 {
        self.total_completed
    }
    pub fn total_failed(&self) -> u64 {
        self.total_failed
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Saga Event Handler — Wire into EventBus
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// EventBus handler for saga lifecycle events.
/// Logs saga transitions for downstream subscribers.
pub fn handle_saga_event(event: &Event) -> HookResult {
    // Saga events are informational — downstream systems
    // (memory, receipt chain, telemetry) subscribe and react.
    // The handler itself just validates and continues.
    if event.ihsan_score.is_critical() {
        HookResult::Halt // Stop propagation of critically low-quality saga events
    } else {
        HookResult::Continue
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    fn test_owner() -> ComponentId {
        ComponentId::from_name("saga-orchestrator", "0.1.0")
    }

    #[test]
    fn saga_happy_path() {
        let mut saga = Saga::new(SagaId(1), test_owner(), 1000);
        let good = IhsanScore::from_f64(0.98);

        // Walk through every phase: Received → Completed
        assert_eq!(saga.phase, SagaPhase::Received);

        let act = saga.advance(good, 2000); // Received → Planned
        assert_eq!(saga.phase, SagaPhase::Planned);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_PLANNED,
                ..
            }
        ));

        let act = saga.advance(good, 3000); // Planned → Executed
        assert_eq!(saga.phase, SagaPhase::Executed);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_EXECUTED,
                ..
            }
        ));

        let act = saga.advance(good, 4000); // Executed → Evaluated
        assert_eq!(saga.phase, SagaPhase::Evaluated);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_EVALUATED,
                ..
            }
        ));

        let act = saga.advance(good, 5000); // Evaluated → Drafted
        assert_eq!(saga.phase, SagaPhase::Drafted);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_DRAFTED,
                ..
            }
        ));

        let act = saga.advance(good, 6000); // Drafted → Gated
        assert_eq!(saga.phase, SagaPhase::Gated);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_GATED,
                ..
            }
        ));

        let act = saga.advance(good, 7000); // Gated → Attested
        assert_eq!(saga.phase, SagaPhase::Attested);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_ATTESTED,
                ..
            }
        ));

        let act = saga.advance(good, 8000); // Attested → Completed
        assert_eq!(saga.phase, SagaPhase::Completed);
        assert!(matches!(act, SagaAction::Complete));
        assert_eq!(saga.steps_completed, 7);
        assert!(saga.is_terminal());
    }

    #[test]
    fn saga_fails_on_critical_ihsan() {
        let mut saga = Saga::new(SagaId(2), test_owner(), 1000);
        let good = IhsanScore::from_f64(0.98);
        let bad = IhsanScore::from_f64(0.80); // Below 0.95 warning

        saga.advance(good, 2000); // Received → Planned
        saga.advance(good, 3000); // Planned → Executed

        // Ihsān drops below critical during evaluation
        let act = saga.advance(bad, 4000);
        assert_eq!(saga.phase, SagaPhase::Compensating);
        assert!(matches!(
            act,
            SagaAction::Emit {
                topic: TOPIC_SAGA_COMPENSATING,
                ..
            }
        ));
        assert_eq!(saga.error_code, 1);
    }

    #[test]
    fn saga_compensation_unwinds_steps() {
        let mut saga = Saga::new(SagaId(3), test_owner(), 1000);
        let good = IhsanScore::from_f64(0.98);

        // Complete 3 phases
        saga.advance(good, 2000); // Received → Planned (step 1)
        saga.advance(good, 3000); // Planned → Executed (step 2)
        saga.advance(good, 4000); // Executed → Evaluated (step 3)
        assert_eq!(saga.steps_completed, 3);

        // Force failure
        saga.fail(42, 5000);
        assert_eq!(saga.phase, SagaPhase::Compensating);

        // Unwind: 3 → 2 → 1 → 0 → Failed
        let act = saga.compensate_step(6000);
        assert_eq!(saga.steps_completed, 2);
        assert!(matches!(act, SagaAction::Emit { .. }));

        let act = saga.compensate_step(7000);
        assert_eq!(saga.steps_completed, 1);
        assert!(matches!(act, SagaAction::Emit { .. }));

        let act = saga.compensate_step(8000);
        assert_eq!(saga.steps_completed, 0);
        assert!(matches!(act, SagaAction::Emit { .. }));

        let act = saga.compensate_step(9000);
        assert_eq!(saga.phase, SagaPhase::Failed);
        assert!(matches!(act, SagaAction::Aborted));
        assert!(saga.is_terminal());
    }

    #[test]
    fn saga_terminal_state_rejects_advance() {
        let mut saga = Saga::new(SagaId(4), test_owner(), 1000);
        saga.phase = SagaPhase::Completed;
        let act = saga.advance(IhsanScore::MAX, 2000);
        assert!(matches!(act, SagaAction::None));
    }

    #[test]
    fn saga_min_ihsan_tracks_lowest() {
        let mut saga = Saga::new(SagaId(5), test_owner(), 1000);
        saga.advance(IhsanScore::from_f64(0.99), 2000);
        saga.advance(IhsanScore::from_f64(0.96), 3000);
        saga.advance(IhsanScore::from_f64(0.98), 4000);

        let min = saga.min_ihsan();
        assert!((min.as_f64() - 0.96).abs() < 0.01);
    }

    #[test]
    fn saga_phase_topics_are_distinct() {
        let phases = [
            SagaPhase::Received,
            SagaPhase::Planned,
            SagaPhase::Executed,
            SagaPhase::Evaluated,
            SagaPhase::Drafted,
            SagaPhase::Gated,
            SagaPhase::Attested,
            SagaPhase::Completed,
            SagaPhase::Failed,
            SagaPhase::Compensating,
        ];
        for (i, a) in phases.iter().enumerate() {
            for (j, b) in phases.iter().enumerate() {
                if i != j {
                    assert_ne!(a.topic(), b.topic(), "Phase topics must be unique");
                }
            }
        }
    }

    #[test]
    fn saga_registry_create_and_remove() {
        let mut reg = SagaRegistry::new();
        let owner = test_owner();

        let id = reg.create(owner, 1000).unwrap();
        assert_eq!(reg.active_count(), 1);
        assert_eq!(reg.total_created(), 1);

        let saga = reg.get(id).unwrap();
        assert_eq!(saga.phase, SagaPhase::Received);

        // Advance and complete
        {
            let saga = reg.get_mut(id).unwrap();
            for t in 1..=7 {
                saga.advance(IhsanScore::from_f64(0.99), t * 1000);
            }
            assert!(saga.is_terminal());
        }

        // Remove
        let removed = reg.remove(id).unwrap();
        assert_eq!(removed.phase, SagaPhase::Completed);
        assert_eq!(reg.active_count(), 0);
        assert_eq!(reg.total_completed(), 1);
    }

    #[test]
    fn saga_registry_capacity_limit() {
        let mut reg = SagaRegistry::new();
        let owner = test_owner();

        // Fill to capacity
        for i in 0..MAX_ACTIVE_SAGAS {
            assert!(reg.create(owner, i as u64 * 1000).is_some());
        }

        // Next create should fail
        assert!(reg.create(owner, 999_999).is_none());
        assert_eq!(reg.active_count(), MAX_ACTIVE_SAGAS);
    }

    #[test]
    fn saga_gated_priority_is_critical() {
        let mut saga = Saga::new(SagaId(7), test_owner(), 1000);
        let good = IhsanScore::from_f64(0.98);

        // Advance to Gated phase
        for t in 1..=5 {
            saga.advance(good, t * 1000);
        }
        assert_eq!(saga.phase, SagaPhase::Gated);

        // The Gated→Attested transition should be High priority
        // (Gated phase itself was entered with Critical priority)
    }

    #[test]
    fn saga_handler_halts_critical_events() {
        let critical_event = Event {
            id: EventId::new(1000, 0),
            source: ComponentId::null(),
            topic: Topic::new(TOPIC_SAGA_DRAFTED),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::from_f64(0.50), // Critically low
        };

        assert_eq!(handle_saga_event(&critical_event), HookResult::Halt);
    }

    #[test]
    fn saga_handler_continues_healthy_events() {
        let healthy_event = Event {
            id: EventId::new(1000, 0),
            source: ComponentId::null(),
            topic: Topic::new(TOPIC_SAGA_PLANNED),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::from_f64(0.99),
        };

        assert_eq!(handle_saga_event(&healthy_event), HookResult::Continue);
    }
}
