//! PCI Gate Chain — Tiered verification

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::RejectCode;
use crate::constitution::Constitution;

#[derive(Clone, Debug)]
pub struct GateResult {
    pub gate: String,
    pub passed: bool,
    pub code: RejectCode,
    pub duration: Duration,
}

impl GateResult {
    pub fn pass(gate: &str, duration: Duration) -> Self {
        Self {
            gate: gate.into(),
            passed: true,
            code: RejectCode::Success,
            duration,
        }
    }
    pub fn fail(gate: &str, code: RejectCode, duration: Duration) -> Self {
        Self {
            gate: gate.into(),
            passed: false,
            code,
            duration,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateTier {
    Cheap,     // <10ms
    Medium,    // <150ms
    Expensive, // <2000ms
}

impl GateTier {
    pub fn max_duration(&self) -> Duration {
        match self {
            Self::Cheap => Duration::from_millis(10),
            Self::Medium => Duration::from_millis(150),
            Self::Expensive => Duration::from_millis(2000),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GateContext {
    pub sender_id: String,
    pub envelope_id: String,
    pub content: Vec<u8>,
    pub constitution: Constitution,
    pub snr_score: Option<f64>,
    pub ihsan_score: Option<f64>,
}

pub trait Gate: Send + Sync {
    fn name(&self) -> &'static str;
    fn tier(&self) -> GateTier;
    fn verify(&self, ctx: &GateContext) -> GateResult;
}

pub struct GateChain {
    gates: Vec<Box<dyn Gate>>,
    /// Reserved for deduplication in future
    #[allow(dead_code)]
    seen_ids: Arc<RwLock<HashSet<String>>>,
}

impl GateChain {
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            seen_ids: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn add<G: Gate + 'static>(&mut self, gate: G) -> &mut Self {
        self.gates.push(Box::new(gate));
        self
    }

    pub fn verify(&self, ctx: &GateContext) -> Vec<GateResult> {
        let mut results = Vec::new();
        for gate in &self.gates {
            let result = gate.verify(ctx);
            let passed = result.passed;
            results.push(result);
            if !passed {
                break;
            }
        }
        results
    }

    pub fn all_passed(results: &[GateResult]) -> bool {
        results.iter().all(|r| r.passed)
    }
}

impl Default for GateChain {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in Gates
pub struct SchemaGate;
impl Gate for SchemaGate {
    fn name(&self) -> &'static str {
        "Schema"
    }
    fn tier(&self) -> GateTier {
        GateTier::Cheap
    }
    fn verify(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        if let Ok(s) = std::str::from_utf8(&ctx.content) {
            if serde_json::from_str::<serde_json::Value>(s).is_ok() {
                return GateResult::pass(self.name(), start.elapsed());
            }
        }
        GateResult::fail(self.name(), RejectCode::RejectGateSchema, start.elapsed())
    }
}

pub struct SNRGate;
impl Gate for SNRGate {
    fn name(&self) -> &'static str {
        "SNR"
    }
    fn tier(&self) -> GateTier {
        GateTier::Medium
    }
    fn verify(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        // SECURITY: Fail-closed semantics - missing SNR score = rejection
        // (Genesis Strict Synthesis: No assumptions, only verified excellence)
        match ctx.snr_score {
            Some(snr) if ctx.constitution.check_snr(snr) => {
                GateResult::pass(self.name(), start.elapsed())
            }
            Some(_) => GateResult::fail(self.name(), RejectCode::RejectGateSNR, start.elapsed()),
            None => {
                // Missing score fails-closed (SEC-020: Shannon signal quality)
                GateResult::fail(self.name(), RejectCode::RejectGateSNR, start.elapsed())
            }
        }
    }
}

pub struct IhsanGate;
impl Gate for IhsanGate {
    fn name(&self) -> &'static str {
        "Ihsan"
    }
    fn tier(&self) -> GateTier {
        GateTier::Expensive
    }
    fn verify(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();
        // SECURITY: Fail-closed semantics - missing Ihsān score = rejection
        // (Genesis Strict Synthesis: Runtime requires proof of excellence)
        match ctx.ihsan_score {
            Some(ihsan) if ctx.constitution.check_ihsan(ihsan) => {
                GateResult::pass(self.name(), start.elapsed())
            }
            Some(_) => GateResult::fail(self.name(), RejectCode::RejectGateIhsan, start.elapsed()),
            None => {
                // Missing score fails-closed (Ihsān = 1.0 for Runtime tier)
                GateResult::fail(self.name(), RejectCode::RejectGateIhsan, start.elapsed())
            }
        }
    }
}

pub fn default_gate_chain() -> GateChain {
    let mut chain = GateChain::new();
    chain.add(SchemaGate).add(SNRGate).add(IhsanGate);
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_chain() {
        let chain = default_gate_chain();
        let ctx = GateContext {
            sender_id: "test".into(),
            envelope_id: "pci_123".into(),
            content: br#"{"test":true}"#.to_vec(),
            constitution: Constitution::default(),
            snr_score: Some(0.9),
            ihsan_score: Some(0.96),
        };
        let results = chain.verify(&ctx);
        assert!(GateChain::all_passed(&results));
    }

    #[test]
    fn test_fail_closed_missing_snr() {
        // Genesis Strict Synthesis: Missing SNR score = fail-closed
        let gate = SNRGate;
        let ctx = GateContext {
            sender_id: "test".into(),
            envelope_id: "pci_123".into(),
            content: br#"{"test":true}"#.to_vec(),
            constitution: Constitution::default(),
            snr_score: None, // Missing score
            ihsan_score: Some(0.96),
        };
        let result = gate.verify(&ctx);
        assert!(!result.passed, "Missing SNR score must fail-closed");
        assert_eq!(result.code, RejectCode::RejectGateSNR);
    }

    #[test]
    fn test_fail_closed_missing_ihsan() {
        // Genesis Strict Synthesis: Missing Ihsān score = fail-closed
        let gate = IhsanGate;
        let ctx = GateContext {
            sender_id: "test".into(),
            envelope_id: "pci_123".into(),
            content: br#"{"test":true}"#.to_vec(),
            constitution: Constitution::default(),
            snr_score: Some(0.9),
            ihsan_score: None, // Missing score
        };
        let result = gate.verify(&ctx);
        assert!(!result.passed, "Missing Ihsān score must fail-closed");
        assert_eq!(result.code, RejectCode::RejectGateIhsan);
    }

    #[test]
    fn test_low_snr_rejected() {
        let gate = SNRGate;
        let ctx = GateContext {
            sender_id: "test".into(),
            envelope_id: "pci_123".into(),
            content: br#"{"test":true}"#.to_vec(),
            constitution: Constitution::default(),
            snr_score: Some(0.5), // Below threshold (0.85)
            ihsan_score: Some(0.96),
        };
        let result = gate.verify(&ctx);
        assert!(!result.passed, "Low SNR score must be rejected");
    }

    #[test]
    fn test_low_ihsan_rejected() {
        let gate = IhsanGate;
        let ctx = GateContext {
            sender_id: "test".into(),
            envelope_id: "pci_123".into(),
            content: br#"{"test":true}"#.to_vec(),
            constitution: Constitution::default(),
            snr_score: Some(0.9),
            ihsan_score: Some(0.8), // Below threshold (0.95)
        };
        let result = gate.verify(&ctx);
        assert!(!result.passed, "Low Ihsān score must be rejected");
    }
}
