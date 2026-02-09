//! Gate Chain - Constitutional Enforcement
//!
//! Every inference output passes through the gate chain.
//! Gates are applied in order: Schema → SNR → Ihsān → License
//! If any gate fails, the output is rejected.

use napi::bindgen_prelude::*;
use serde::{Deserialize, Serialize};

use crate::{GateResult, IHSAN_THRESHOLD, SNR_THRESHOLD};

/// Individual gate in the chain
pub trait Gate {
    fn name(&self) -> &str;
    fn check(&self, output: &InferenceOutput) -> GateResult;
}

/// Parsed inference output for gate validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub content: String,
    pub model_id: String,
    pub ihsan_score: Option<f64>,
    pub snr_score: Option<f64>,
    pub schema_valid: Option<bool>,
    pub capability_card_valid: Option<bool>,
}

/// Schema Gate - Validates JSON structure
pub struct SchemaGate;

impl Gate for SchemaGate {
    fn name(&self) -> &str {
        "SCHEMA"
    }

    fn check(&self, output: &InferenceOutput) -> GateResult {
        // Check if output has valid structure
        let passed = !output.content.is_empty() && !output.model_id.is_empty();

        GateResult {
            passed,
            gate_name: self.name().to_string(),
            score: if passed { 1.0 } else { 0.0 },
            reason: if passed {
                None
            } else {
                Some("Invalid output schema: missing required fields".to_string())
            },
        }
    }
}

/// SNR Gate - Shannon signal quality threshold
pub struct SnrGate {
    pub threshold: f64,
}

impl Default for SnrGate {
    fn default() -> Self {
        Self {
            threshold: SNR_THRESHOLD,
        }
    }
}

impl SnrGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Gate for SnrGate {
    fn name(&self) -> &str {
        "SNR"
    }

    fn check(&self, output: &InferenceOutput) -> GateResult {
        match output.snr_score {
            Some(score) => {
                let passed = score >= self.threshold;
                GateResult {
                    passed,
                    gate_name: self.name().to_string(),
                    score,
                    reason: if passed {
                        None
                    } else {
                        Some(format!(
                            "SNR score {:.3} below threshold {:.2}",
                            score, self.threshold
                        ))
                    },
                }
            }
            None => GateResult {
                passed: false,
                gate_name: self.name().to_string(),
                score: 0.0,
                reason: Some("SNR score not provided".to_string()),
            },
        }
    }
}

/// Ihsān Gate - Z3-verified excellence threshold
pub struct IhsanGate {
    pub threshold: f64,
}

impl Default for IhsanGate {
    fn default() -> Self {
        Self {
            threshold: IHSAN_THRESHOLD,
        }
    }
}

impl IhsanGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Gate for IhsanGate {
    fn name(&self) -> &str {
        "IHSAN"
    }

    fn check(&self, output: &InferenceOutput) -> GateResult {
        match output.ihsan_score {
            Some(score) => {
                let passed = score >= self.threshold;
                GateResult {
                    passed,
                    gate_name: self.name().to_string(),
                    score,
                    reason: if passed {
                        None
                    } else {
                        Some(format!(
                            "Ihsān score {:.3} below threshold {:.2}",
                            score, self.threshold
                        ))
                    },
                }
            }
            None => GateResult {
                passed: false,
                gate_name: self.name().to_string(),
                score: 0.0,
                reason: Some("Ihsān score not provided".to_string()),
            },
        }
    }
}

/// License Gate - Validates CapabilityCard
pub struct LicenseGate;

impl Gate for LicenseGate {
    fn name(&self) -> &str {
        "LICENSE"
    }

    fn check(&self, output: &InferenceOutput) -> GateResult {
        match output.capability_card_valid {
            Some(valid) => GateResult {
                passed: valid,
                gate_name: self.name().to_string(),
                score: if valid { 1.0 } else { 0.0 },
                reason: if valid {
                    None
                } else {
                    Some("Invalid or expired CapabilityCard".to_string())
                },
            },
            None => GateResult {
                passed: false,
                gate_name: self.name().to_string(),
                score: 0.0,
                reason: Some("Model not licensed. Run Constitution Challenge first.".to_string()),
            },
        }
    }
}

/// The complete gate chain
pub struct GateChain {
    gates: Vec<Box<dyn Gate + Send + Sync>>,
}

impl Default for GateChain {
    fn default() -> Self {
        Self {
            gates: vec![
                Box::new(SchemaGate),
                Box::new(SnrGate::new()),
                Box::new(IhsanGate::new()),
                Box::new(LicenseGate),
            ],
        }
    }
}

impl GateChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a minimal chain (SCHEMA + SNR + IHSAN only, no LICENSE)
    pub fn minimal() -> Self {
        Self {
            gates: vec![
                Box::new(SchemaGate),
                Box::new(SnrGate::new()),
                Box::new(IhsanGate::new()),
            ],
        }
    }

    /// Validate an output through all gates
    pub fn validate(&self, output_json: &str) -> Result<GateResult> {
        let output: InferenceOutput = serde_json::from_str(output_json)
            .map_err(|e| Error::from_reason(format!("Invalid output JSON: {}", e)))?;

        for gate in &self.gates {
            let result = gate.check(&output);
            if !result.passed {
                return Ok(result);
            }
        }

        // All gates passed
        Ok(GateResult {
            passed: true,
            gate_name: "ALL".to_string(),
            score: output.ihsan_score.unwrap_or(0.0),
            reason: None,
        })
    }

    /// Get detailed results from all gates
    pub fn validate_detailed(&self, output_json: &str) -> Result<Vec<GateResult>> {
        let output: InferenceOutput = serde_json::from_str(output_json)
            .map_err(|e| Error::from_reason(format!("Invalid output JSON: {}", e)))?;

        Ok(self.gates.iter().map(|gate| gate.check(&output)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_output() -> InferenceOutput {
        InferenceOutput {
            content: "Test response content".to_string(),
            model_id: "test-model".to_string(),
            ihsan_score: Some(0.97),
            snr_score: Some(0.90),
            schema_valid: Some(true),
            capability_card_valid: Some(true),
        }
    }

    #[test]
    fn test_schema_gate_pass() {
        let gate = SchemaGate;
        let output = make_valid_output();
        let result = gate.check(&output);
        assert!(result.passed);
    }

    #[test]
    fn test_schema_gate_fail() {
        let gate = SchemaGate;
        let output = InferenceOutput {
            content: "".to_string(),
            model_id: "".to_string(),
            ..make_valid_output()
        };
        let result = gate.check(&output);
        assert!(!result.passed);
    }

    #[test]
    fn test_snr_gate_pass() {
        let gate = SnrGate::new();
        let output = make_valid_output();
        let result = gate.check(&output);
        assert!(result.passed);
    }

    #[test]
    fn test_snr_gate_fail() {
        let gate = SnrGate::new();
        let output = InferenceOutput {
            snr_score: Some(0.50),
            ..make_valid_output()
        };
        let result = gate.check(&output);
        assert!(!result.passed);
    }

    #[test]
    fn test_ihsan_gate_pass() {
        let gate = IhsanGate::new();
        let output = make_valid_output();
        let result = gate.check(&output);
        assert!(result.passed);
    }

    #[test]
    fn test_ihsan_gate_fail() {
        let gate = IhsanGate::new();
        let output = InferenceOutput {
            ihsan_score: Some(0.90),
            ..make_valid_output()
        };
        let result = gate.check(&output);
        assert!(!result.passed);
    }

    #[test]
    fn test_gate_chain_all_pass() {
        let chain = GateChain::new();
        let output = serde_json::to_string(&make_valid_output()).unwrap();
        let result = chain.validate(&output).unwrap();
        assert!(result.passed);
        assert_eq!(result.gate_name, "ALL");
    }

    #[test]
    fn test_gate_chain_fails_at_ihsan() {
        let chain = GateChain::new();
        let output = InferenceOutput {
            ihsan_score: Some(0.90),
            ..make_valid_output()
        };
        let output_json = serde_json::to_string(&output).unwrap();
        let result = chain.validate(&output_json).unwrap();
        assert!(!result.passed);
        assert_eq!(result.gate_name, "IHSAN");
    }
}
