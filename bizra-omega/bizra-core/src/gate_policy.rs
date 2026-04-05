//! Constitutional Gate Policy — unified enforcement for threshold violations.
//!
//! This module defines the canonical enforcement protocol for all constitutional
//! threshold checks (Ihsan, SNR, ADL, etc.) across the BIZRA ecosystem.
//!
//! ## The Problem
//! The same threshold check (`ihsan < 0.95`) previously triggered 5 different
//! behaviors across modules: silent drop, hard halt, zero reward, policy-dependent,
//! and APPROVED/REJECTED decisions. This module unifies the decision point.
//!
//! ## Usage
//! ```rust
//! use bizra_core::gate_policy::{apply_gate, env_gate_policy, GateAction, GatePolicy};
//!
//! let policy = env_gate_policy(); // Observe in dev, Reject in prod
//! let verdict = apply_gate(0.93, 0.95, policy);
//! match verdict.action {
//!     GateAction::Allow => { /* proceed */ }
//!     GateAction::AllowWithWarning => { /* log and proceed */ }
//!     GateAction::Rejected => { /* fail-closed */ }
//!     _ => { /* handle Flag/Throttle */ }
//! }
//! ```
//!
//! Standing on Giants: Al-Ghazali (Ihsan as obligation) · Deming (quality at source)

/// What happens when a constitutional threshold is violated.
///
/// Ordered from most permissive to most restrictive:
/// Observe → Flag → Throttle(n) → Reject
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatePolicy {
    /// Log warning but allow (development/observation mode)
    #[default]
    Observe,
    /// Attach warning flag to the event, still deliver
    Flag,
    /// Allow 1 in N events from the violating component
    Throttle(u32),
    /// Hard reject — fail-closed (production mode)
    Reject,
}

/// The action taken after policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAction {
    /// Score passed threshold — proceed normally
    Allow,
    /// Score failed but policy is Observe — proceed with logged warning
    AllowWithWarning,
    /// Score failed, policy is Flag — proceed with flag attached
    Flagged,
    /// Score failed, policy is Throttle — suppressed this time
    Throttled,
    /// Score failed, policy is Reject — hard stop
    Rejected,
}

/// The result of applying a gate policy to a score.
#[derive(Debug, Clone)]
pub struct GateVerdict {
    /// The score that was evaluated
    pub score: f64,
    /// The threshold it was measured against
    pub threshold: f64,
    /// Whether the score met the threshold
    pub passed: bool,
    /// The policy that was applied
    pub policy: GatePolicy,
    /// The resulting action
    pub action: GateAction,
}

/// Resolve the active gate policy from the `BIZRA_ENV` environment variable.
///
/// - `BIZRA_ENV=prod` or `BIZRA_ENV=production` → `GatePolicy::Reject`
/// - All other values or unset → `GatePolicy::Observe`
///
/// Individual modules MAY override this with explicit configuration.
pub fn env_gate_policy() -> GatePolicy {
    match std::env::var("BIZRA_ENV").as_deref() {
        Ok("prod") | Ok("production") => GatePolicy::Reject,
        _ => GatePolicy::Observe,
    }
}

/// Apply a gate policy to a score/threshold pair.
///
/// This is the canonical function all enforcement surfaces should call.
///
/// - If `score >= threshold`, returns `GateAction::Allow` regardless of policy.
/// - If `score < threshold`, the policy determines the action.
///
/// **Throttle counter state** is the caller's responsibility — this function
/// always returns `Throttled` for the `Throttle` policy on failure. The caller
/// decides whether this particular invocation is the 1-in-N that gets through.
pub fn apply_gate(score: f64, threshold: f64, policy: GatePolicy) -> GateVerdict {
    let passed = score >= threshold;
    let action = if passed {
        GateAction::Allow
    } else {
        match policy {
            GatePolicy::Observe => GateAction::AllowWithWarning,
            GatePolicy::Flag => GateAction::Flagged,
            GatePolicy::Throttle(_) => GateAction::Throttled,
            GatePolicy::Reject => GateAction::Rejected,
        }
    };
    GateVerdict {
        score,
        threshold,
        passed,
        policy,
        action,
    }
}

// ─── Wire 5: Gate Maturation ─────────────────────────────────────────────────

/// Maturation thresholds — cycle counts at which the policy auto-promotes.
///
/// Standing on Giants: Deming (PDCA maturation) · Lamport (safety liveness)
#[derive(Debug, Clone, Copy)]
pub struct MaturationThresholds {
    /// Cycles before Observe → Flag
    pub observe_to_flag: u64,
    /// Cycles before Flag → Throttle(5)
    pub flag_to_throttle: u64,
    /// Cycles before Throttle → Reject
    pub throttle_to_reject: u64,
}

impl Default for MaturationThresholds {
    fn default() -> Self {
        Self {
            observe_to_flag: 100,
            flag_to_throttle: 500,
            throttle_to_reject: 1000,
        }
    }
}

/// Auto-promoting gate policy that hardens with accumulated evidence.
///
/// Starts at `Observe` and promotes through the GatePolicy ladder:
///   Observe → Flag → Throttle(5) → Reject
///
/// Each `tick()` increments the cycle counter. When a threshold is crossed,
/// the policy promotes to the next level. Promotion is monotonic — a gate
/// never softens once hardened.
#[derive(Debug, Clone)]
pub struct GateMaturationPolicy {
    thresholds: MaturationThresholds,
    cycle_count: u64,
    current: GatePolicy,
}

impl GateMaturationPolicy {
    /// Create a maturation policy with the given thresholds, starting at Observe.
    pub fn new(thresholds: MaturationThresholds) -> Self {
        Self {
            thresholds,
            cycle_count: 0,
            current: GatePolicy::Observe,
        }
    }

    /// Record one cycle and auto-promote if a threshold is crossed.
    /// Returns the (possibly new) active policy.
    pub fn tick(&mut self) -> GatePolicy {
        self.cycle_count += 1;
        self.current = match self.current {
            GatePolicy::Observe if self.cycle_count >= self.thresholds.observe_to_flag => {
                GatePolicy::Flag
            }
            GatePolicy::Flag if self.cycle_count >= self.thresholds.flag_to_throttle => {
                GatePolicy::Throttle(5)
            }
            GatePolicy::Throttle(_) if self.cycle_count >= self.thresholds.throttle_to_reject => {
                GatePolicy::Reject
            }
            other => other,
        };
        self.current
    }

    /// Current active policy.
    pub fn current(&self) -> GatePolicy {
        self.current
    }

    /// Total cycles recorded.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Whether the gate has reached its terminal (Reject) state.
    pub fn is_mature(&self) -> bool {
        matches!(self.current, GatePolicy::Reject)
    }
}

impl Default for GateMaturationPolicy {
    fn default() -> Self {
        Self::new(MaturationThresholds::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env-var tests mutate global state — serialize them to avoid thread races.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_passing_score_always_allows() {
        let v = apply_gate(0.96, 0.95, GatePolicy::Reject);
        assert!(v.passed);
        assert_eq!(v.action, GateAction::Allow);
    }

    #[test]
    fn test_failing_observe_allows_with_warning() {
        let v = apply_gate(0.90, 0.95, GatePolicy::Observe);
        assert!(!v.passed);
        assert_eq!(v.action, GateAction::AllowWithWarning);
    }

    #[test]
    fn test_failing_flag_returns_flagged() {
        let v = apply_gate(0.90, 0.95, GatePolicy::Flag);
        assert!(!v.passed);
        assert_eq!(v.action, GateAction::Flagged);
    }

    #[test]
    fn test_failing_throttle_returns_throttled() {
        let v = apply_gate(0.90, 0.95, GatePolicy::Throttle(3));
        assert!(!v.passed);
        assert_eq!(v.action, GateAction::Throttled);
    }

    #[test]
    fn test_failing_reject_returns_rejected() {
        let v = apply_gate(0.90, 0.95, GatePolicy::Reject);
        assert!(!v.passed);
        assert_eq!(v.action, GateAction::Rejected);
    }

    #[test]
    fn test_exact_threshold_passes() {
        let v = apply_gate(0.95, 0.95, GatePolicy::Reject);
        assert!(v.passed);
        assert_eq!(v.action, GateAction::Allow);
    }

    #[test]
    fn test_env_gate_default_is_observe() {
        let _lock = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("BIZRA_ENV").ok();
        std::env::remove_var("BIZRA_ENV");
        let p = env_gate_policy();
        // Restore
        match prev {
            Some(v) => std::env::set_var("BIZRA_ENV", v),
            None => std::env::remove_var("BIZRA_ENV"),
        }
        assert_eq!(p, GatePolicy::Observe);
    }

    #[test]
    fn test_env_gate_prod_is_reject() {
        let _lock = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("BIZRA_ENV").ok();
        std::env::set_var("BIZRA_ENV", "prod");
        let p = env_gate_policy();
        // Restore
        match prev {
            Some(v) => std::env::set_var("BIZRA_ENV", v),
            None => std::env::remove_var("BIZRA_ENV"),
        }
        assert_eq!(p, GatePolicy::Reject);
    }

    #[test]
    fn test_env_gate_production_long_form() {
        let _lock = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("BIZRA_ENV").ok();
        std::env::set_var("BIZRA_ENV", "production");
        let p = env_gate_policy();
        // Restore
        match prev {
            Some(v) => std::env::set_var("BIZRA_ENV", v),
            None => std::env::remove_var("BIZRA_ENV"),
        }
        assert_eq!(p, GatePolicy::Reject);
    }

    #[test]
    fn test_verdict_fields_populated() {
        let v = apply_gate(0.93, 0.95, GatePolicy::Reject);
        assert!((v.score - 0.93).abs() < f64::EPSILON);
        assert!((v.threshold - 0.95).abs() < f64::EPSILON);
        assert!(!v.passed);
        assert_eq!(v.policy, GatePolicy::Reject);
        assert_eq!(v.action, GateAction::Rejected);
    }

    // ── Wire 5: Maturation tests ─────────────────────────────

    #[test]
    fn test_maturation_starts_at_observe() {
        let m = GateMaturationPolicy::default();
        assert_eq!(m.current(), GatePolicy::Observe);
        assert_eq!(m.cycle_count(), 0);
        assert!(!m.is_mature());
    }

    #[test]
    fn test_maturation_promotes_through_all_stages() {
        let thresholds = MaturationThresholds {
            observe_to_flag: 3,
            flag_to_throttle: 6,
            throttle_to_reject: 10,
        };
        let mut m = GateMaturationPolicy::new(thresholds);

        // Cycles 1-2: still Observe
        for _ in 0..2 {
            assert_eq!(m.tick(), GatePolicy::Observe);
        }

        // Cycle 3: promotes to Flag
        assert_eq!(m.tick(), GatePolicy::Flag);

        // Cycles 4-5: still Flag
        for _ in 0..2 {
            assert_eq!(m.tick(), GatePolicy::Flag);
        }

        // Cycle 6: promotes to Throttle(5)
        assert_eq!(m.tick(), GatePolicy::Throttle(5));
        assert!(!m.is_mature());

        // Cycles 7-9: still Throttle
        for _ in 0..3 {
            assert_eq!(m.tick(), GatePolicy::Throttle(5));
        }

        // Cycle 10: promotes to Reject (terminal)
        assert_eq!(m.tick(), GatePolicy::Reject);
        assert!(m.is_mature());
        assert_eq!(m.cycle_count(), 10);

        // Further ticks stay at Reject (monotonic)
        assert_eq!(m.tick(), GatePolicy::Reject);
    }

    #[test]
    fn test_maturation_never_softens() {
        let thresholds = MaturationThresholds {
            observe_to_flag: 1,
            flag_to_throttle: 2,
            throttle_to_reject: 3,
        };
        let mut m = GateMaturationPolicy::new(thresholds);

        // Rapid promotion to Reject
        for _ in 0..5 {
            m.tick();
        }
        assert_eq!(m.current(), GatePolicy::Reject);
        assert!(m.is_mature());

        // 100 more ticks — still Reject
        for _ in 0..100 {
            assert_eq!(m.tick(), GatePolicy::Reject);
        }
    }

    #[test]
    fn test_default_thresholds() {
        let t = MaturationThresholds::default();
        assert_eq!(t.observe_to_flag, 100);
        assert_eq!(t.flag_to_throttle, 500);
        assert_eq!(t.throttle_to_reject, 1000);
    }
}
