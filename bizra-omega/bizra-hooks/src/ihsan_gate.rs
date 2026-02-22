//! # إحسان Gate — RSI Pillar V: Stable Iteration
//!
//! The إحسان Gate is the Lyapunov certificate enforcer. It ensures that
//! system quality never degrades below the constitutional floor (99.0%).
//!
//! ## RSI Mapping
//! - **Lyapunov function**: إحسان score (scalar, monotonically bounded)
//! - **Constitutional floor**: 0.990 (must not cross downward without human approval)
//! - **Mutation bound**: Score delta per change bounded by configurable epsilon
//! - **Compositional stability**: Aggregate score across all components
//!
//! ## Integration
//! The Gate operates as a PreEmit hook on the EventBus. Events from
//! components with degraded إحسان are either flagged, throttled, or halted.

use crate::types::*;

/// Gate enforcement policy — what happens when إحسان is violated?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatePolicy {
    /// Log warning but allow event (observation mode)
    #[default]
    Observe,
    /// Attach warning flag to event, still deliver
    Flag,
    /// Throttle: allow 1 in N events from degraded component
    Throttle(u32),
    /// Hard reject: halt events from components below floor
    Reject,
}

/// Configuration for the إحسان Gate.
#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Constitutional floor score (default: 0.990)
    pub floor: IhsanScore,
    /// Warning threshold (default: 0.950)
    pub warning: IhsanScore,
    /// Maximum allowed score drop per mutation (Lyapunov epsilon)
    pub max_delta: IhsanScore,
    /// Enforcement policy
    pub policy: GatePolicy,
    /// Whether to emit إحسان events on score changes
    pub emit_events: bool,
}

impl GateConfig {
    /// Production configuration: **enforcement mode**.
    ///
    /// Events from components below the constitutional floor (0.990) are
    /// hard-rejected. This transforms the Ihsan floor from a monitoring
    /// metric into an active guardrail.
    ///
    /// Standing on Giants: Al-Ghazali — Ihsan is not optional in production.
    pub fn production() -> Self {
        GateConfig {
            floor: IhsanScore::IHSAN_FLOOR,
            warning: IhsanScore::WARNING,
            max_delta: IhsanScore::from_f64(0.01),
            policy: GatePolicy::Reject,
            emit_events: true,
        }
    }

    /// Development configuration: **observation mode**.
    ///
    /// Events from degraded components are logged but allowed through.
    /// Use this for local development and testing where enforcement
    /// would block iteration velocity.
    pub fn development() -> Self {
        GateConfig {
            floor: IhsanScore::IHSAN_FLOOR,
            warning: IhsanScore::WARNING,
            max_delta: IhsanScore::from_f64(0.01),
            policy: GatePolicy::Observe,
            emit_events: true,
        }
    }

    /// Staged rollout configuration: **throttle mode**.
    ///
    /// Allows 1 in N events from degraded components through.
    /// Use this as a transition step between Observe and Reject.
    pub fn staged(throttle_n: u32) -> Self {
        GateConfig {
            floor: IhsanScore::IHSAN_FLOOR,
            warning: IhsanScore::WARNING,
            max_delta: IhsanScore::from_f64(0.01),
            policy: GatePolicy::Throttle(throttle_n),
            emit_events: true,
        }
    }
}

impl Default for GateConfig {
    fn default() -> Self {
        // Default is development mode for backward compatibility.
        // Production deployments MUST use GateConfig::production().
        Self::development()
    }
}

/// Evaluation result from the gate.
#[derive(Debug, Clone, Copy)]
pub struct GateVerdict {
    /// The score that was evaluated
    pub score: IhsanScore,
    /// Previous score (for delta calculation)
    pub previous: IhsanScore,
    /// Whether the score meets the constitutional floor
    pub meets_floor: bool,
    /// Whether the delta exceeds allowed bounds
    pub delta_exceeded: bool,
    /// The action taken
    pub action: GateAction,
}

/// What the gate decided to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAction {
    /// Event passes through normally
    Allow,
    /// Event passes with warning flag
    Flagged,
    /// Event was throttled (not delivered this time)
    Throttled,
    /// Event was rejected
    Rejected,
}

/// إحسان Gate — the quality enforcer.
///
/// Tracks per-component إحسان scores, enforces constitutional floor,
/// bounds mutation deltas, and gates event flow based on quality.
pub struct IhsanGate {
    /// Gate configuration
    config: GateConfig,

    /// Per-component score tracking (component_id hash → score slot)
    /// Using a simple hash-addressed array for zero-allocation
    scores: [(ComponentId, IhsanScore, u64); 256], // id, score, last_check_timestamp
    score_count: usize,

    /// Per-component throttle counters
    throttle_counters: [u32; 256],

    /// Total evaluations performed
    total_evaluations: u64,
    /// Total violations detected
    total_violations: u64,
    /// Total events rejected
    total_rejections: u64,

    /// Lyapunov stability tracking: consecutive evaluations meeting floor
    consecutive_stable: u64,
    /// Maximum consecutive stable count (high water mark)
    max_consecutive_stable: u64,
}

impl IhsanGate {
    /// Create a new gate with default (development) configuration.
    pub fn new() -> Self {
        Self::with_config(GateConfig::default())
    }

    /// Create a production gate: Reject policy, enforcement active.
    ///
    /// This is the gate configuration that transforms the Ihsan floor
    /// from a monitoring metric into an enforcement gate. Events from
    /// components below 0.990 are hard-rejected.
    pub fn production() -> Self {
        Self::with_config(GateConfig::production())
    }

    /// Create a development gate: Observe policy, logging only.
    pub fn development() -> Self {
        Self::with_config(GateConfig::development())
    }

    /// Create with custom configuration.
    pub fn with_config(config: GateConfig) -> Self {
        IhsanGate {
            config,
            scores: [(ComponentId::null(), IhsanScore::MAX, 0); 256],
            score_count: 0,
            throttle_counters: [0; 256],
            total_evaluations: 0,
            total_violations: 0,
            total_rejections: 0,
            consecutive_stable: 0,
            max_consecutive_stable: 0,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Core Evaluation — The Lyapunov Check
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Evaluate an event against the إحسان gate.
    /// This is the core function — called as a PreEmit hook.
    pub fn evaluate(&mut self, event: &Event) -> GateVerdict {
        self.total_evaluations += 1;

        let current = event.ihsan_score;
        let previous = self.get_score(&event.source);

        // Check constitutional floor
        let meets_floor = current >= self.config.floor;

        // Check delta bound (Lyapunov epsilon)
        let delta_exceeded = if previous > current {
            let delta_raw = previous.raw().saturating_sub(current.raw());
            delta_raw > self.config.max_delta.raw()
        } else {
            false // Score improved or unchanged — always OK
        };

        // Update tracked score
        self.set_score(event.source, current, event.id.timestamp_nanos());

        // Determine action based on policy
        let action = if meets_floor && !delta_exceeded {
            // All clear — update stability tracking
            self.consecutive_stable += 1;
            if self.consecutive_stable > self.max_consecutive_stable {
                self.max_consecutive_stable = self.consecutive_stable;
            }
            GateAction::Allow
        } else {
            // Violation detected
            self.total_violations += 1;
            self.consecutive_stable = 0;

            match self.config.policy {
                GatePolicy::Observe => GateAction::Allow,
                GatePolicy::Flag => GateAction::Flagged,
                GatePolicy::Throttle(n) => {
                    let slot = self.find_score_slot(&event.source);
                    self.throttle_counters[slot] += 1;
                    if self.throttle_counters[slot].is_multiple_of(n) {
                        GateAction::Allow // Let through every Nth event
                    } else {
                        self.total_rejections += 1;
                        GateAction::Throttled
                    }
                }
                GatePolicy::Reject => {
                    self.total_rejections += 1;
                    GateAction::Rejected
                }
            }
        };

        GateVerdict {
            score: current,
            previous,
            meets_floor,
            delta_exceeded,
            action,
        }
    }

    /// Quick check: should this event be allowed through?
    pub fn should_allow(&mut self, event: &Event) -> bool {
        let verdict = self.evaluate(event);
        verdict.action != GateAction::Rejected && verdict.action != GateAction::Throttled
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Score Tracking
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get tracked score for a component.
    pub fn get_score(&self, id: &ComponentId) -> IhsanScore {
        let slot = self.find_score_slot(id);
        if self.scores[slot].0 == *id {
            self.scores[slot].1
        } else {
            IhsanScore::MAX // Unknown components start at perfection
        }
    }

    /// Manually set a component's score (e.g., from external health check).
    pub fn set_score(&mut self, id: ComponentId, score: IhsanScore, timestamp: u64) {
        let slot = self.find_score_slot(&id);
        if self.scores[slot].0.is_null() {
            self.score_count += 1;
        }
        self.scores[slot] = (id, score, timestamp);
    }

    fn find_score_slot(&self, id: &ComponentId) -> usize {
        // Simple hash-addressing with linear probing
        let hash = {
            let bytes = id.0;
            let mut h: usize = 0;
            for &b in &bytes {
                h = h.wrapping_mul(31).wrapping_add(b as usize);
            }
            h % 256
        };

        let mut slot = hash;
        for _ in 0..256 {
            if self.scores[slot].0 == *id || self.scores[slot].0.is_null() {
                return slot;
            }
            slot = (slot + 1) % 256;
        }
        hash // Fallback: overwrite original slot
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Configuration
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get current gate configuration.
    pub fn config(&self) -> &GateConfig {
        &self.config
    }

    /// Update gate policy.
    pub fn set_policy(&mut self, policy: GatePolicy) {
        self.config.policy = policy;
    }

    /// Update constitutional floor.
    pub fn set_floor(&mut self, floor: IhsanScore) {
        self.config.floor = floor;
    }

    /// Update maximum delta bound.
    pub fn set_max_delta(&mut self, delta: IhsanScore) {
        self.config.max_delta = delta;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Telemetry — Lyapunov Stability Metrics
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn total_evaluations(&self) -> u64 {
        self.total_evaluations
    }

    pub fn total_violations(&self) -> u64 {
        self.total_violations
    }

    pub fn total_rejections(&self) -> u64 {
        self.total_rejections
    }

    /// How many consecutive evaluations have passed the floor.
    /// This is the Lyapunov stability indicator.
    pub fn consecutive_stable(&self) -> u64 {
        self.consecutive_stable
    }

    /// High water mark for consecutive stability.
    pub fn max_consecutive_stable(&self) -> u64 {
        self.max_consecutive_stable
    }

    /// Violation rate: violations / total evaluations.
    pub fn violation_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.total_violations as f64 / self.total_evaluations as f64
        }
    }

    /// Stability score: 1.0 - violation_rate (higher = more stable).
    pub fn stability_score(&self) -> f64 {
        1.0 - self.violation_rate()
    }

    /// Number of tracked components.
    pub fn tracked_count(&self) -> usize {
        self.score_count
    }
}

impl Default for IhsanGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(source: ComponentId, score: f64) -> Event {
        Event {
            id: EventId::new(1000, 0),
            source,
            topic: Topic::new("test.event"),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::from_f64(score),
        }
    }

    #[test]
    fn perfect_score_passes() {
        let mut gate = IhsanGate::new();
        let comp = ComponentId::from_name("perfect", "1.0.0");
        let event = make_event(comp, 1.0);

        let verdict = gate.evaluate(&event);
        assert!(verdict.meets_floor);
        assert!(!verdict.delta_exceeded);
        assert_eq!(verdict.action, GateAction::Allow);
    }

    #[test]
    fn below_floor_detected() {
        let mut gate = IhsanGate::with_config(GateConfig {
            policy: GatePolicy::Reject,
            ..Default::default()
        });

        let comp = ComponentId::from_name("degraded", "1.0.0");
        let event = make_event(comp, 0.95); // Below 0.99 floor

        let verdict = gate.evaluate(&event);
        assert!(!verdict.meets_floor);
        assert_eq!(verdict.action, GateAction::Rejected);
        assert_eq!(gate.total_violations(), 1);
        assert_eq!(gate.total_rejections(), 1);
    }

    #[test]
    fn observe_mode_allows_violations() {
        let mut gate = IhsanGate::new(); // Default: Observe mode
        let comp = ComponentId::from_name("test", "1.0.0");
        let event = make_event(comp, 0.50); // Very low score

        let verdict = gate.evaluate(&event);
        assert!(!verdict.meets_floor);
        assert_eq!(verdict.action, GateAction::Allow); // Observe allows
    }

    #[test]
    fn delta_detection() {
        let mut gate = IhsanGate::with_config(GateConfig {
            policy: GatePolicy::Flag,
            max_delta: IhsanScore::from_f64(0.01),
            ..Default::default()
        });

        let comp = ComponentId::from_name("volatile", "1.0.0");

        // First event: score 0.995 (above floor, first reading)
        let e1 = make_event(comp, 0.995);
        let v1 = gate.evaluate(&e1);
        assert_eq!(v1.action, GateAction::Allow);

        // Second event: score drops to 0.980 (delta = 0.015 > 0.01 limit)
        let mut e2 = make_event(comp, 0.980);
        e2.id = EventId::new(2000, 0);
        let v2 = gate.evaluate(&e2);
        assert!(v2.delta_exceeded);
        assert_eq!(v2.action, GateAction::Flagged);
    }

    #[test]
    fn throttle_policy() {
        let mut gate = IhsanGate::with_config(GateConfig {
            policy: GatePolicy::Throttle(3), // Allow every 3rd event
            ..Default::default()
        });

        let comp = ComponentId::from_name("throttled", "1.0.0");

        // 3 events with low score — only 3rd should pass
        for i in 0..3 {
            let mut event = make_event(comp, 0.50);
            event.id = EventId::new(1000 + i, 0);
            let verdict = gate.evaluate(&event);

            if i == 2 {
                assert_eq!(verdict.action, GateAction::Allow);
            } else {
                assert_eq!(verdict.action, GateAction::Throttled);
            }
        }
    }

    #[test]
    fn production_gate_rejects_below_floor() {
        let mut gate = IhsanGate::production();
        let comp = ComponentId::from_name("degraded", "1.0.0");
        let event = make_event(comp, 0.95); // Below 0.990 floor

        let verdict = gate.evaluate(&event);
        assert!(!verdict.meets_floor);
        assert_eq!(verdict.action, GateAction::Rejected);
        assert_eq!(gate.total_rejections(), 1);
    }

    #[test]
    fn production_gate_allows_above_floor() {
        let mut gate = IhsanGate::production();
        let comp = ComponentId::from_name("excellent", "1.0.0");
        let event = make_event(comp, 0.995); // Above 0.990 floor

        let verdict = gate.evaluate(&event);
        assert!(verdict.meets_floor);
        assert_eq!(verdict.action, GateAction::Allow);
        assert_eq!(gate.total_rejections(), 0);
    }

    #[test]
    fn production_gate_rejects_at_exact_boundary() {
        let mut gate = IhsanGate::production();
        let comp = ComponentId::from_name("edge", "1.0.0");
        // 0.989 is below 0.990 floor
        let event = make_event(comp, 0.989);

        let verdict = gate.evaluate(&event);
        assert!(!verdict.meets_floor);
        assert_eq!(verdict.action, GateAction::Rejected);
    }

    #[test]
    fn development_gate_observes_violations() {
        let mut gate = IhsanGate::development();
        let comp = ComponentId::from_name("testing", "1.0.0");
        let event = make_event(comp, 0.50); // Way below floor

        let verdict = gate.evaluate(&event);
        assert!(!verdict.meets_floor);
        // Development mode: violation detected but event allowed through
        assert_eq!(verdict.action, GateAction::Allow);
        assert_eq!(gate.total_violations(), 1);
        assert_eq!(gate.total_rejections(), 0);
    }

    #[test]
    fn staged_gate_throttles_degraded() {
        let mut gate = IhsanGate::with_config(GateConfig::staged(3));
        let comp = ComponentId::from_name("degraded", "1.0.0");

        // 3 events below floor — only every 3rd allowed
        for i in 0..6 {
            let mut event = make_event(comp, 0.50);
            event.id = EventId::new(1000 + i, 0);
            let verdict = gate.evaluate(&event);
            if (i + 1) % 3 == 0 {
                assert_eq!(verdict.action, GateAction::Allow);
            } else {
                assert_eq!(verdict.action, GateAction::Throttled);
            }
        }
    }

    #[test]
    fn config_constructors_are_consistent() {
        let prod = GateConfig::production();
        let dev = GateConfig::development();
        let staged = GateConfig::staged(5);

        // Same floor across all configs
        assert_eq!(prod.floor, dev.floor);
        assert_eq!(dev.floor, staged.floor);

        // Policy differs
        assert_eq!(prod.policy, GatePolicy::Reject);
        assert_eq!(dev.policy, GatePolicy::Observe);
        assert_eq!(staged.policy, GatePolicy::Throttle(5));

        // Default is development
        let default = GateConfig::default();
        assert_eq!(default.policy, GatePolicy::Observe);
    }

    #[test]
    fn stability_tracking() {
        let mut gate = IhsanGate::new();
        let comp = ComponentId::from_name("stable", "1.0.0");

        // 10 consecutive good events
        for i in 0..10 {
            let mut event = make_event(comp, 0.999);
            event.id = EventId::new(1000 + i, 0);
            gate.evaluate(&event);
        }

        assert_eq!(gate.consecutive_stable(), 10);
        assert_eq!(gate.max_consecutive_stable(), 10);
        assert!(gate.stability_score() > 0.99);

        // One bad event resets consecutive counter
        let bad = make_event(comp, 0.50);
        gate.evaluate(&bad);
        assert_eq!(gate.consecutive_stable(), 0);
        assert_eq!(gate.max_consecutive_stable(), 10); // High water mark preserved
    }
}
