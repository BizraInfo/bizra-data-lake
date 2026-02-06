//! Constitution — Ihsan Governance Framework

use crate::{IHSAN_THRESHOLD, SNR_THRESHOLD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IhsanThreshold {
    pub minimum: f64,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub enforcement: Enforcement,
    pub penalty: Penalty,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Enforcement {
    Strict,
    Advisory,
    Monitoring,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Penalty {
    Reject,
    TrustPenalty(f64),
    Cooldown(u64),
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constitution {
    pub version: String,
    pub ihsan: IhsanThreshold,
    pub snr_threshold: f64,
    pub rules: HashMap<String, Rule>,
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
    pub fn check_ihsan(&self, score: f64) -> bool {
        score >= self.ihsan.minimum
    }
    pub fn check_snr(&self, snr: f64) -> bool {
        snr >= self.snr_threshold
    }
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
