//! PCI Gate Chain — Tiered verification

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::RejectCode;
use crate::constitution::Constitution;

/// Outcome of a single gate verification pass.
#[derive(Clone, Debug)]
pub struct GateResult {
    /// Name of the gate that produced this result.
    pub gate: String,
    /// Whether the gate check passed.
    pub passed: bool,
    /// Reject code (`Success` when passed).
    pub code: RejectCode,
    /// Wall-clock time spent in this gate.
    pub duration: Duration,
}

impl GateResult {
    /// Creates a passing `GateResult` for the named gate.
    pub fn pass(gate: &str, duration: Duration) -> Self {
        Self {
            gate: gate.into(),
            passed: true,
            code: RejectCode::Success,
            duration,
        }
    }
    /// Creates a failing `GateResult` with the given reject code.
    pub fn fail(gate: &str, code: RejectCode, duration: Duration) -> Self {
        Self {
            gate: gate.into(),
            passed: false,
            code,
            duration,
        }
    }
}

/// Cost tier governing a gate's maximum allowed execution time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateTier {
    /// Fast gate, must complete in <10 ms.
    Cheap,     // <10ms
    /// Moderate gate, must complete in <150 ms.
    Medium,    // <150ms
    /// Heavyweight gate, must complete in <2000 ms.
    Expensive, // <2000ms
}

impl GateTier {
    /// Returns the maximum wall-clock duration allowed for this tier.
    pub fn max_duration(&self) -> Duration {
        match self {
            Self::Cheap => Duration::from_millis(10),
            Self::Medium => Duration::from_millis(150),
            Self::Expensive => Duration::from_millis(2000),
        }
    }
}

/// Input context supplied to every gate in the chain.
#[derive(Clone, Debug)]
pub struct GateContext {
    /// Identity of the envelope sender.
    pub sender_id: String,
    /// Unique PCI envelope identifier.
    pub envelope_id: String,
    /// Raw payload bytes to validate.
    pub content: Vec<u8>,
    /// Active constitution for threshold lookups.
    pub constitution: Constitution,
    /// Pre-computed Signal-to-Noise Ratio (if available).
    pub snr_score: Option<f64>,
    /// Pre-computed Ihsan excellence score (if available).
    pub ihsan_score: Option<f64>,
}

/// A verification gate in the PCI pipeline.
///
/// Implementors perform a single validation check (e.g. schema, SNR, Ihsan)
/// and return a [`GateResult`] indicating pass or fail.
pub trait Gate: Send + Sync {
    /// Human-readable gate name used in [`GateResult::gate`].
    fn name(&self) -> &'static str;
    /// Cost tier governing the gate's execution budget.
    fn tier(&self) -> GateTier;
    /// Execute the gate check against the given context.
    fn verify(&self, ctx: &GateContext) -> GateResult;
}

/// Ordered sequence of gates executed in fail-fast order.
///
/// The chain short-circuits on the first failing gate, ensuring cheap
/// checks (e.g. schema) run before expensive ones (e.g. Ihsan ML).
pub struct GateChain {
    gates: Vec<Box<dyn Gate>>,
    /// Reserved for deduplication in future
    #[allow(dead_code)]
    seen_ids: Arc<RwLock<HashSet<String>>>,
}

impl GateChain {
    /// Creates an empty gate chain.
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            seen_ids: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Appends a gate to the end of the chain.
    pub fn add<G: Gate + 'static>(&mut self, gate: G) -> &mut Self {
        self.gates.push(Box::new(gate));
        self
    }

    /// Runs all gates in order, stopping at the first failure.
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

    /// Returns `true` if every result in the slice is a pass.
    pub fn all_passed(results: &[GateResult]) -> bool {
        results.iter().all(|r| r.passed)
    }
}

impl Default for GateChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in gate: validates that the payload is well-formed JSON.
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

/// Built-in gate: checks Signal-to-Noise Ratio against the constitution.
///
/// Fail-closed: missing SNR score results in rejection.
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

/// Built-in gate: checks Ihsan excellence score against the constitution.
///
/// Fail-closed: missing Ihsan score results in rejection.
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

/// Returns the default gate chain: `Schema → Ihsan → SNR`.
///
/// Ethics before efficiency: Ihsan rejects unethical content before
/// wasting compute on SNR analysis (fail-fast-on-ethics).
pub fn default_gate_chain() -> GateChain {
    let mut chain = GateChain::new();
    // Ethics before efficiency: Ihsan gate rejects unethical content
    // before wasting compute on SNR analysis (fail-fast-on-ethics).
    // See also: core/pci/gates.py line 10 (Python PCI specification).
    chain.add(SchemaGate).add(IhsanGate).add(SNRGate);
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
