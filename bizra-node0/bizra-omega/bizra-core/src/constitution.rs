//! Constitution — Ihsan Governance Framework

use crate::{IHSAN_THRESHOLD, SNR_THRESHOLD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Excellence threshold configuration for the Ihsan governance framework.
///
/// Defines the minimum acceptable and optimal target scores for Ihsan
/// (pursuit of excellence) across all system operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IhsanThreshold {
    /// Minimum acceptable Ihsan score (hard floor, e.g. 0.95).
    pub minimum: f64,
    /// Target Ihsan score for optimal operation (aspirational, e.g. 0.99).
    pub target: f64,
}

impl Default for IhsanThreshold {
    fn default() -> Self {
        Self {
            minimum: IHSAN_THRESHOLD,
            target: 0.99,
        }
    }
}

/// A single constitutional rule with enforcement policy and penalty.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    /// Unique rule identifier (e.g. `"IHSAN_001"`).
    pub id: String,
    /// Human-readable rule name.
    pub name: String,
    /// Full description of the rule's intent and scope.
    pub description: String,
    /// How strictly this rule is enforced.
    pub enforcement: Enforcement,
    /// Consequence applied when the rule is violated.
    pub penalty: Penalty,
}

/// Enforcement level for a constitutional rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Enforcement {
    /// Hard enforcement — violations cause immediate rejection.
    Strict,
    /// Soft enforcement — violations produce warnings but allow processing.
    Advisory,
    /// Observation only — violations are logged for analysis.
    Monitoring,
}

/// Penalty applied when a constitutional rule is violated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Penalty {
    /// Immediate rejection of the operation.
    Reject,
    /// Trust score reduction by the specified amount.
    TrustPenalty(f64),
    /// Temporary cooldown period in seconds.
    Cooldown(u64),
    /// No penalty (informational only).
    None,
}

/// The BIZRA Constitution — root governance document.
///
/// Defines all operational rules, thresholds, and enforcement policies.
/// Every PCI envelope and Omega operation is validated against this.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constitution {
    /// Semantic version of this constitution (e.g. `"1.0.0"`).
    pub version: String,
    /// Ihsan excellence thresholds.
    pub ihsan: IhsanThreshold,
    /// Minimum acceptable Signal-to-Noise Ratio.
    pub snr_threshold: f64,
    /// Active governance rules keyed by rule ID.
    pub rules: HashMap<String, Rule>,
    /// Whether this constitution is currently enforced.
    pub active: bool,
}

impl Default for Constitution {
    fn default() -> Self {
        let mut rules = HashMap::new();
        rules.insert(
            "IHSAN_001".into(),
            Rule {
                id: "IHSAN_001".into(),
                name: "Excellence Threshold".into(),
                description: "All outputs must meet Ihsan threshold (>=0.95)".into(),
                enforcement: Enforcement::Strict,
                penalty: Penalty::Reject,
            },
        );
        rules.insert(
            "SNR_001".into(),
            Rule {
                id: "SNR_001".into(),
                name: "Signal Quality".into(),
                description: "All inputs must meet SNR threshold (>=0.85)".into(),
                enforcement: Enforcement::Strict,
                penalty: Penalty::Reject,
            },
        );
        rules.insert(
            "CRYPTO_001".into(),
            Rule {
                id: "CRYPTO_001".into(),
                name: "Signature Validity".into(),
                description: "All PCI envelopes must have valid Ed25519 signatures".into(),
                enforcement: Enforcement::Strict,
                penalty: Penalty::Reject,
            },
        );
        Self {
            version: "1.0.0".into(),
            ihsan: IhsanThreshold::default(),
            snr_threshold: SNR_THRESHOLD,
            rules,
            active: true,
        }
    }
}

impl Constitution {
    /// Returns `true` if the given score meets the Ihsan minimum threshold.
    pub fn check_ihsan(&self, score: f64) -> bool {
        score >= self.ihsan.minimum
    }
    /// Returns `true` if the given SNR meets the signal quality threshold.
    pub fn check_snr(&self, snr: f64) -> bool {
        snr >= self.snr_threshold
    }
    /// Looks up a rule by its unique identifier.
    pub fn get_rule(&self, id: &str) -> Option<&Rule> {
        self.rules.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ihsan_check() {
        let c = Constitution::default();
        assert!(c.check_ihsan(0.95));
        assert!(!c.check_ihsan(0.94));
    }
}
