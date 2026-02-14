//! Critical Cascade Gate Enforcement
//!
//! TRICK 7: Fail-safe gate system that pauses operations on threshold breach.
//!
//! Gates:
//! - Ethics: 1 failure = pause (Ihsān enforcement)
//! - Legal: 1 failure = pause (compliance)
//! - Technical: 10 failures = pause (quality control)
//!
//! Auto-resume disabled for critical gates (manual intervention required).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Gate types in the critical cascade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GateType {
    /// Ethics gate (Ihsān ≥ 0.95)
    Ethics = 0,
    /// Legal compliance gate
    Legal = 1,
    /// Technical quality gate
    Technical = 2,
}

/// Configuration for a single gate
#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Failure threshold before pause
    pub threshold: u32,
    /// Whether auto-resume is allowed
    pub auto_resume: bool,
    /// Reset period in seconds (0 = never)
    pub reset_period_secs: u64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            threshold: 10,
            auto_resume: true,
            reset_period_secs: 3600, // 1 hour
        }
    }
}

/// State of a single gate
struct GateState {
    /// Current failure count
    failures: AtomicU32,
    /// Whether gate is open (allowing operations)
    open: AtomicBool,
    /// Timestamp of last failure (nanoseconds)
    last_failure: AtomicU32, // Store as seconds for simplicity
}

impl GateState {
    fn new() -> Self {
        Self {
            failures: AtomicU32::new(0),
            open: AtomicBool::new(true),
            last_failure: AtomicU32::new(0),
        }
    }
}

/// Critical Cascade controller
///
/// Manages three gates with configurable thresholds.
/// All gates must be open for operations to proceed.
pub struct CriticalCascade {
    /// Gate states
    ethics: GateState,
    legal: GateState,
    technical: GateState,
    /// Gate configurations
    ethics_config: GateConfig,
    legal_config: GateConfig,
    technical_config: GateConfig,
}

impl CriticalCascade {
    /// Create new cascade with default configurations
    pub fn new() -> Self {
        Self {
            ethics: GateState::new(),
            legal: GateState::new(),
            technical: GateState::new(),
            // Critical gates: 1 failure, no auto-resume
            ethics_config: GateConfig {
                threshold: 1,
                auto_resume: false,
                reset_period_secs: 0,
            },
            legal_config: GateConfig {
                threshold: 1,
                auto_resume: false,
                reset_period_secs: 0,
            },
            // Technical gate: 10 failures, auto-resume after 1 hour
            technical_config: GateConfig {
                threshold: 10,
                auto_resume: true,
                reset_period_secs: 3600,
            },
        }
    }

    /// Configure a specific gate
    pub fn configure(&mut self, gate: GateType, config: GateConfig) {
        match gate {
            GateType::Ethics => self.ethics_config = config,
            GateType::Legal => self.legal_config = config,
            GateType::Technical => self.technical_config = config,
        }
    }

    /// Get state for a gate
    fn get_state(&self, gate: GateType) -> &GateState {
        match gate {
            GateType::Ethics => &self.ethics,
            GateType::Legal => &self.legal,
            GateType::Technical => &self.technical,
        }
    }

    /// Get config for a gate
    fn get_config(&self, gate: GateType) -> &GateConfig {
        match gate {
            GateType::Ethics => &self.ethics_config,
            GateType::Legal => &self.legal_config,
            GateType::Technical => &self.technical_config,
        }
    }

    /// Check if a gate is open
    pub fn is_open(&self, gate: GateType) -> bool {
        let state = self.get_state(gate);
        let config = self.get_config(gate);

        // Check if gate is marked open
        if state.open.load(Ordering::Relaxed) {
            return true;
        }

        // Check auto-resume
        if config.auto_resume && config.reset_period_secs > 0 {
            let last_failure = state.last_failure.load(Ordering::Relaxed);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);

            if now > last_failure + config.reset_period_secs as u32 {
                // Auto-resume: reset gate
                state.open.store(true, Ordering::Relaxed);
                state.failures.store(0, Ordering::Relaxed);
                return true;
            }
        }

        false
    }

    /// Check if all gates are open
    pub fn all_open(&self) -> bool {
        self.is_open(GateType::Ethics)
            && self.is_open(GateType::Legal)
            && self.is_open(GateType::Technical)
    }

    /// Record a failure for a gate
    pub fn record_failure(&self, gate: GateType) {
        let state = self.get_state(gate);
        let config = self.get_config(gate);

        // Increment failure count
        let failures = state.failures.fetch_add(1, Ordering::Relaxed) + 1;

        // Update last failure timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        state.last_failure.store(now, Ordering::Relaxed);

        // Check threshold
        if failures >= config.threshold {
            state.open.store(false, Ordering::Relaxed);
            tracing::warn!(
                gate = ?gate,
                failures = failures,
                threshold = config.threshold,
                "Gate closed due to threshold breach"
            );
        }
    }

    /// Record a success (reset failure count for auto-reset gates)
    pub fn record_success(&self, gate: GateType) {
        let state = self.get_state(gate);
        let config = self.get_config(gate);

        if config.auto_resume {
            state.failures.store(0, Ordering::Relaxed);
        }
    }

    /// Manually reset a gate (requires explicit action)
    pub fn reset(&self, gate: GateType) {
        let state = self.get_state(gate);
        state.failures.store(0, Ordering::Relaxed);
        state.open.store(true, Ordering::Relaxed);
        tracing::info!(gate = ?gate, "Gate manually reset");
    }

    /// Get failure count for a gate
    pub fn failure_count(&self, gate: GateType) -> u32 {
        self.get_state(gate).failures.load(Ordering::Relaxed)
    }

    /// Get cascade status snapshot
    pub fn status(&self) -> CascadeStatus {
        CascadeStatus {
            ethics_open: self.is_open(GateType::Ethics),
            ethics_failures: self.failure_count(GateType::Ethics),
            legal_open: self.is_open(GateType::Legal),
            legal_failures: self.failure_count(GateType::Legal),
            technical_open: self.is_open(GateType::Technical),
            technical_failures: self.failure_count(GateType::Technical),
            all_open: self.all_open(),
        }
    }
}

impl Default for CriticalCascade {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of cascade status
#[derive(Debug, Clone, serde::Serialize)]
pub struct CascadeStatus {
    pub ethics_open: bool,
    pub ethics_failures: u32,
    pub legal_open: bool,
    pub legal_failures: u32,
    pub technical_open: bool,
    pub technical_failures: u32,
    pub all_open: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_new() {
        let cascade = CriticalCascade::new();
        assert!(cascade.all_open());
    }

    #[test]
    fn test_ethics_gate_closes_on_single_failure() {
        let cascade = CriticalCascade::new();

        cascade.record_failure(GateType::Ethics);
        assert!(!cascade.is_open(GateType::Ethics));
        assert!(!cascade.all_open());
    }

    #[test]
    fn test_technical_gate_tolerates_failures() {
        let cascade = CriticalCascade::new();

        // 9 failures should not close technical gate
        for _ in 0..9 {
            cascade.record_failure(GateType::Technical);
        }
        assert!(cascade.is_open(GateType::Technical));

        // 10th failure should close it
        cascade.record_failure(GateType::Technical);
        assert!(!cascade.is_open(GateType::Technical));
    }

    #[test]
    fn test_manual_reset() {
        let cascade = CriticalCascade::new();

        cascade.record_failure(GateType::Ethics);
        assert!(!cascade.is_open(GateType::Ethics));

        cascade.reset(GateType::Ethics);
        assert!(cascade.is_open(GateType::Ethics));
    }

    #[test]
    fn test_success_resets_auto_resume_gates() {
        let cascade = CriticalCascade::new();

        cascade.record_failure(GateType::Technical);
        assert_eq!(cascade.failure_count(GateType::Technical), 1);

        cascade.record_success(GateType::Technical);
        assert_eq!(cascade.failure_count(GateType::Technical), 0);
    }

    #[test]
    fn test_success_does_not_reset_critical_gates() {
        let cascade = CriticalCascade::new();

        cascade.record_failure(GateType::Ethics);
        assert_eq!(cascade.failure_count(GateType::Ethics), 1);

        cascade.record_success(GateType::Ethics);
        // Ethics gate doesn't auto-reset
        assert_eq!(cascade.failure_count(GateType::Ethics), 1);
    }
}
