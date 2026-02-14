//! Self-Evolving Judgment Engine (SJE) — Observation Telemetry
//!
//! Phase A: Observation Mode Only.
//!
//! Records verdict distributions and computes entropy to measure
//! judgment stability. NO policy mutation. NO threshold changes.
//!
//! # Standing on Giants
//!
//! - **Shannon** (1948): Entropy as uncertainty measure
//! - **Aristotle** (Nicomachean Ethics): Practical wisdom via observation

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ═══════════════════════════════════════════════════════════════════════════════
// Judgment Verdicts
// ═══════════════════════════════════════════════════════════════════════════════

/// Possible judgment outcomes for an episode.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JudgmentVerdict {
    /// Episode quality warrants promotion in RIR ranking
    Promote,
    /// Episode quality is acceptable, no action needed
    Neutral,
    /// Episode quality is below threshold, reduce RIR weight
    Demote,
    /// Episode violates safety constraints, exclude from retrieval
    Forbid,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Judgment Telemetry
// ═══════════════════════════════════════════════════════════════════════════════

/// Observation-mode telemetry for the SJE.
///
/// Tracks verdict distribution and computes Shannon entropy
/// to measure judgment stability over time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JudgmentTelemetry {
    /// Count per verdict type
    pub verdict_counts: BTreeMap<JudgmentVerdict, u64>,
    /// Total observations recorded
    pub total_observations: u64,
}

impl JudgmentTelemetry {
    /// Create a new empty telemetry instance.
    pub fn new() -> Self {
        let mut counts = BTreeMap::new();
        counts.insert(JudgmentVerdict::Promote, 0);
        counts.insert(JudgmentVerdict::Neutral, 0);
        counts.insert(JudgmentVerdict::Demote, 0);
        counts.insert(JudgmentVerdict::Forbid, 0);
        Self {
            verdict_counts: counts,
            total_observations: 0,
        }
    }

    /// Record a verdict observation.
    pub fn observe(&mut self, verdict: JudgmentVerdict) {
        *self.verdict_counts.entry(verdict).or_insert(0) += 1;
        self.total_observations += 1;
    }

    /// Compute Shannon entropy of the verdict distribution.
    ///
    /// H = -sum(p_i * log2(p_i)) for all verdicts with p_i > 0.
    /// Returns 0.0 for zero or one observations.
    /// Max entropy = log2(4) = 2.0 (uniform over 4 verdicts).
    pub fn entropy(&self) -> f64 {
        if self.total_observations <= 1 {
            return 0.0;
        }

        let total = self.total_observations as f64;
        let mut h = 0.0f64;

        for &count in self.verdict_counts.values() {
            if count > 0 {
                let p = count as f64 / total;
                h -= p * p.log2();
            }
        }

        h
    }

    /// Return the most frequent verdict, or None if empty.
    pub fn dominant_verdict(&self) -> Option<&JudgmentVerdict> {
        if self.total_observations == 0 {
            return None;
        }
        self.verdict_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(v, _)| v)
    }

    /// Check if judgment is stable (low entropy = high consensus).
    pub fn is_stable(&self, entropy_threshold: f64) -> bool {
        self.entropy() < entropy_threshold
    }
}

impl Default for JudgmentTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Epoch Distribution Simulator
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulate proportional epoch allocation. No tokens emitted.
///
/// Pure mathematical rehearsal for genesis economy modeling.
/// Each node receives: floor(impact_i * epoch_cap / total_impact).
pub fn simulate_epoch_distribution(impacts: &[u64], epoch_cap: u64) -> Vec<u64> {
    if impacts.is_empty() {
        return Vec::new();
    }

    let total: u64 = impacts.iter().sum();
    if total == 0 {
        return vec![0; impacts.len()];
    }

    impacts
        .iter()
        .map(|&i| (i as u128 * epoch_cap as u128 / total as u128) as u64)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_telemetry() {
        let t = JudgmentTelemetry::new();
        assert_eq!(t.total_observations, 0);
        assert_eq!(t.entropy(), 0.0);
        assert!(t.dominant_verdict().is_none());
        assert!(t.is_stable(0.5));
    }

    #[test]
    fn test_observe_single_verdict() {
        let mut t = JudgmentTelemetry::new();
        t.observe(JudgmentVerdict::Promote);
        assert_eq!(t.total_observations, 1);
        assert_eq!(t.entropy(), 0.0); // Single observation = 0 entropy
    }

    #[test]
    fn test_entropy_uniform() {
        let mut t = JudgmentTelemetry::new();
        for _ in 0..25 {
            t.observe(JudgmentVerdict::Promote);
            t.observe(JudgmentVerdict::Neutral);
            t.observe(JudgmentVerdict::Demote);
            t.observe(JudgmentVerdict::Forbid);
        }
        // Uniform distribution: H = log2(4) = 2.0
        assert!((t.entropy() - 2.0).abs() < 0.01);
        assert!(!t.is_stable(0.5));
    }

    #[test]
    fn test_entropy_dominant() {
        let mut t = JudgmentTelemetry::new();
        for _ in 0..95 {
            t.observe(JudgmentVerdict::Promote);
        }
        for _ in 0..5 {
            t.observe(JudgmentVerdict::Neutral);
        }
        // Heavily skewed: low entropy
        assert!(t.entropy() < 0.5);
        assert!(t.is_stable(0.5));
        assert_eq!(t.dominant_verdict(), Some(&JudgmentVerdict::Promote));
    }

    #[test]
    fn test_dominant_verdict() {
        let mut t = JudgmentTelemetry::new();
        t.observe(JudgmentVerdict::Promote);
        t.observe(JudgmentVerdict::Promote);
        t.observe(JudgmentVerdict::Demote);
        assert_eq!(t.dominant_verdict(), Some(&JudgmentVerdict::Promote));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Epoch Distribution Tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_epoch_distribution_empty() {
        assert!(simulate_epoch_distribution(&[], 1000).is_empty());
    }

    #[test]
    fn test_epoch_distribution_zero_total() {
        let result = simulate_epoch_distribution(&[0, 0, 0], 1000);
        assert_eq!(result, vec![0, 0, 0]);
    }

    #[test]
    fn test_epoch_distribution_single_node() {
        let result = simulate_epoch_distribution(&[100], 1000);
        assert_eq!(result, vec![1000]);
    }

    #[test]
    fn test_epoch_distribution_proportional() {
        let result = simulate_epoch_distribution(&[100, 200, 300], 600);
        assert_eq!(result, vec![100, 200, 300]);
    }

    #[test]
    fn test_epoch_distribution_rounding() {
        // 1/3 of 100 = 33 (floor)
        let result = simulate_epoch_distribution(&[1, 1, 1], 100);
        assert_eq!(result, vec![33, 33, 33]);
        // Total allocated <= epoch_cap (dust remains)
        let total: u64 = result.iter().sum();
        assert!(total <= 100);
    }

    #[test]
    fn test_epoch_distribution_large_values() {
        // Test with large values that could overflow u64 without u128
        let result = simulate_epoch_distribution(&[1_000_000_000, 2_000_000_000], 10_000_000_000);
        assert_eq!(result[0], 3_333_333_333);
        assert_eq!(result[1], 6_666_666_666);
    }
}
