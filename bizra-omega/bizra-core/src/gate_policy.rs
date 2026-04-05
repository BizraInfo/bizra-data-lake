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
}
