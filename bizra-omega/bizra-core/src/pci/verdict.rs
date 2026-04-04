//! # GateVerdict — Cross-Layer Contract #2
//!
//! The authoritative output of the gate chain evaluation.
//! Links a mission envelope to its gate results with a canonical hash
//! for cross-language verification and tamper detection.
//!
//! Flow: MissionEnvelope → GateChain → GateVerdict → ReceiptArtifact

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use super::gates::GateResult;
use super::RejectCode;

/// Domain prefix for verdict hashing.
pub const DOMAIN_VERDICT: &str = "bizra-verdict-v1";

const FIXED_POINT_P: f64 = 1_000_000.0;

/// Overall mission admission status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictStatus {
    /// All gates passed — mission admitted.
    Admitted,
    /// One or more gates failed — mission rejected.
    Rejected,
    /// Evaluation deferred (dependency unavailable).
    Deferred,
}

/// Proof verification status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// Proof verified by all gates.
    Verified,
    /// Proof cannot be verified (missing data).
    Unverifiable,
    /// Proof verification pending.
    Pending,
}

/// The GateVerdict — canonical cross-layer contract #2.
///
/// Produced by the gate chain, consumed by the receipt system.
/// The `verdict_hash` enables cross-language verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateVerdict {
    /// Links to the MissionEnvelope that was evaluated.
    pub mission_id: String,
    /// Ordered results from each gate in the chain.
    pub gate_results: Vec<VerdictGateEntry>,
    /// Overall admission status.
    pub status: VerdictStatus,
    /// Proof verification status.
    pub proof_status: ProofStatus,
    /// Measured Ihsan score (from IhsanGate).
    pub ihsan_score: f64,
    /// Measured SNR score (from SNRGate).
    pub snr_score: f64,
    /// Reject codes (empty if admitted).
    pub reject_codes: Vec<RejectCode>,
    /// Policy version used for evaluation.
    pub policy_version: String,
    /// Evaluation timestamp (Unix ms).
    pub evaluated_at: u64,
    /// Canonical hash (BLAKE3, domain-separated).
    pub verdict_hash: [u8; 32],
}

/// Individual gate entry in the verdict (serializable version of GateResult).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerdictGateEntry {
    /// Name of the gate that produced this entry.
    pub gate_name: String,
    /// Whether the gate check passed.
    pub passed: bool,
    /// Wall-clock time spent in this gate (nanoseconds).
    pub latency_ns: u64,
    /// Reject code if gate failed; None if passed.
    pub reject_code: Option<RejectCode>,
}

impl VerdictGateEntry {
    /// Convert from a GateResult.
    pub fn from_gate_result(result: &GateResult) -> Self {
        Self {
            gate_name: result.gate.clone(),
            passed: result.passed,
            latency_ns: result.duration.as_nanos() as u64,
            reject_code: if result.passed {
                None
            } else {
                Some(result.code)
            },
        }
    }
}

impl GateVerdict {
    /// Create a verdict from gate chain results.
    pub fn from_gate_results(
        mission_id: String,
        results: &[GateResult],
        ihsan_score: f64,
        snr_score: f64,
        policy_version: String,
        now_ms: u64,
    ) -> Self {
        let all_passed = results.iter().all(|r| r.passed);
        let reject_codes: Vec<RejectCode> = results
            .iter()
            .filter(|r| !r.passed)
            .map(|r| r.code)
            .collect();

        let gate_results: Vec<VerdictGateEntry> = results
            .iter()
            .map(VerdictGateEntry::from_gate_result)
            .collect();

        let status = if all_passed {
            VerdictStatus::Admitted
        } else {
            VerdictStatus::Rejected
        };

        let proof_status = if all_passed {
            ProofStatus::Verified
        } else {
            ProofStatus::Unverifiable
        };

        let mut verdict = Self {
            mission_id,
            gate_results,
            status,
            proof_status,
            ihsan_score,
            snr_score,
            reject_codes,
            policy_version,
            evaluated_at: now_ms,
            verdict_hash: [0; 32],
        };
        verdict.verdict_hash = verdict.compute_hash();
        verdict
    }

    /// Domain-separated BLAKE3 hash (golden-vector protocol).
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(128);

        // mission_id
        let mid = self.mission_id.as_bytes();
        buf.extend_from_slice(&(mid.len() as u32).to_le_bytes());
        buf.extend_from_slice(mid);

        // status as u8
        buf.push(match self.status {
            VerdictStatus::Admitted => 0,
            VerdictStatus::Rejected => 1,
            VerdictStatus::Deferred => 2,
        });

        // ihsan + snr as fixed-point
        buf.extend_from_slice(&((self.ihsan_score * FIXED_POINT_P).round() as u64).to_le_bytes());
        buf.extend_from_slice(&((self.snr_score * FIXED_POINT_P).round() as u64).to_le_bytes());

        // gate count + pass/fail per gate
        buf.extend_from_slice(&(self.gate_results.len() as u32).to_le_bytes());
        for g in &self.gate_results {
            buf.push(if g.passed { 1 } else { 0 });
        }

        // policy_version
        let pv = self.policy_version.as_bytes();
        buf.extend_from_slice(&(pv.len() as u32).to_le_bytes());
        buf.extend_from_slice(pv);

        // timestamp
        buf.extend_from_slice(&self.evaluated_at.to_le_bytes());

        let mut hasher = Hasher::new();
        hasher.update(DOMAIN_VERDICT.as_bytes());
        hasher.update(b":");
        hasher.update(&buf);
        hasher.finalize().into()
    }

    /// Check if the mission was admitted.
    pub fn is_admitted(&self) -> bool {
        self.status == VerdictStatus::Admitted
    }

    /// Total gate evaluation latency.
    pub fn total_latency_ns(&self) -> u64 {
        self.gate_results.iter().map(|g| g.latency_ns).sum()
    }

    /// Returns the primary reject code, ordered by constitutional precedence.
    ///
    /// Precedence (highest to lowest):
    ///   1. RIBA — exploitation is the worst constitutional violation
    ///   2. ZANN — speculative/ungrounded content undermines trust
    ///   3. FATE — fairness/ethics violation
    ///   4. Ihsan — excellence threshold not met
    ///   5. SNR — signal quality below threshold
    ///   6. Everything else (schema, signature, rate, quota, etc.)
    ///
    /// When multiple gates fail, this surfaces the most constitutionally
    /// significant cause — not just the first gate that ran.
    pub fn primary_reject_code(&self) -> Option<RejectCode> {
        if self.reject_codes.is_empty() {
            return None;
        }
        self.reject_codes
            .iter()
            .copied()
            .max_by_key(|c| reject_precedence(*c))
    }
}

/// Constitutional precedence ordering for reject codes.
/// Higher value = higher priority = surfaces first in verdicts.
fn reject_precedence(code: RejectCode) -> u8 {
    match code {
        RejectCode::RejectRiba => 100,        // Exploitation — worst violation
        RejectCode::RejectZann => 90,         // Speculative content — truth violation
        RejectCode::RejectGateFATE => 80,     // Fairness/ethics
        RejectCode::RejectGateIhsan => 70,    // Excellence threshold
        RejectCode::RejectGateSNR => 60,      // Signal quality
        RejectCode::RejectSignature => 50,    // Crypto integrity
        RejectCode::RejectHashMismatch => 50, // Content integrity
        RejectCode::RejectReplay => 45,       // Security
        RejectCode::RejectGateSchema => 30,   // Schema validation
        RejectCode::RejectGateSig => 30,      // Gate-level signature
        RejectCode::RejectGateRate => 20,     // Rate limiting
        RejectCode::RejectSchema => 15,       // Basic schema
        RejectCode::RejectMissingField => 15, // Missing field
        RejectCode::RejectExpired => 10,      // TTL
        RejectCode::RejectQuota => 10,        // Quota
        RejectCode::RejectTimeout => 5,       // Timeout
        RejectCode::RejectInternal => 1,      // Internal error
        RejectCode::Success => 0,             // Not a rejection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constitution::Constitution;
    use crate::pci::gates::{default_gate_chain, GateContext, GateResult};
    use std::time::Duration;

    fn mock_passing_results() -> Vec<GateResult> {
        vec![
            GateResult::pass("Schema", Duration::from_micros(50)),
            GateResult::pass("Ihsan", Duration::from_micros(200)),
            GateResult::pass("SNR", Duration::from_micros(100)),
        ]
    }

    fn mock_failing_results() -> Vec<GateResult> {
        vec![
            GateResult::pass("Schema", Duration::from_micros(50)),
            GateResult::fail(
                "Ihsan",
                RejectCode::RejectGateIhsan,
                Duration::from_micros(200),
            ),
        ]
    }

    #[test]
    fn test_verdict_admitted() {
        let v = GateVerdict::from_gate_results(
            "m1".into(),
            &mock_passing_results(),
            0.97,
            0.92,
            "0.89.1".into(),
            1000,
        );
        assert!(v.is_admitted());
        assert_eq!(v.status, VerdictStatus::Admitted);
        assert_eq!(v.proof_status, ProofStatus::Verified);
        assert!(v.reject_codes.is_empty());
        assert_eq!(v.gate_results.len(), 3);
    }

    #[test]
    fn test_verdict_rejected() {
        let v = GateVerdict::from_gate_results(
            "m2".into(),
            &mock_failing_results(),
            0.40,
            0.92,
            "0.89.1".into(),
            1000,
        );
        assert!(!v.is_admitted());
        assert_eq!(v.status, VerdictStatus::Rejected);
        assert_eq!(v.reject_codes, vec![RejectCode::RejectGateIhsan]);
    }

    #[test]
    fn test_verdict_hash_deterministic() {
        let v1 = GateVerdict::from_gate_results(
            "m1".into(),
            &mock_passing_results(),
            0.97,
            0.92,
            "0.89.1".into(),
            1000,
        );
        let v2 = GateVerdict::from_gate_results(
            "m1".into(),
            &mock_passing_results(),
            0.97,
            0.92,
            "0.89.1".into(),
            1000,
        );
        assert_eq!(v1.verdict_hash, v2.verdict_hash);
    }

    #[test]
    fn test_verdict_hash_changes_on_rejection() {
        let admitted = GateVerdict::from_gate_results(
            "m1".into(),
            &mock_passing_results(),
            0.97,
            0.92,
            "0.89.1".into(),
            1000,
        );
        let rejected = GateVerdict::from_gate_results(
            "m1".into(),
            &mock_failing_results(),
            0.40,
            0.92,
            "0.89.1".into(),
            1000,
        );
        assert_ne!(admitted.verdict_hash, rejected.verdict_hash);
    }

    #[test]
    fn test_verdict_total_latency() {
        let v = GateVerdict::from_gate_results(
            "m1".into(),
            &mock_passing_results(),
            0.97,
            0.92,
            "0.89.1".into(),
            1000,
        );
        assert_eq!(v.total_latency_ns(), 350_000); // 50+200+100 us = 350us = 350_000ns
    }

    #[test]
    fn test_verdict_precedence_riba_surfaces_over_snr() {
        // THE ONE CONSTITUTIONAL TEST: when a mission triggers both
        // Riba (exploitation) and SNR (low quality) rejections,
        // the verdict's primary cause MUST be Riba — not SNR.
        // Exploitation is a constitutional invariant violation;
        // low quality is just a threshold miss.
        let results = vec![
            GateResult::pass("Schema", Duration::from_micros(30)),
            GateResult::fail("SNR", RejectCode::RejectGateSNR, Duration::from_micros(100)),
            GateResult::fail("Riba", RejectCode::RejectRiba, Duration::from_micros(200)),
        ];
        let v = GateVerdict::from_gate_results(
            "exploitative-mission".into(),
            &results,
            0.30,
            0.20,
            "0.90.0".into(),
            2000,
        );
        assert!(!v.is_admitted());
        assert_eq!(v.reject_codes.len(), 2);
        // Primary cause MUST be Riba, not SNR
        assert_eq!(v.primary_reject_code(), Some(RejectCode::RejectRiba));
    }

    #[test]
    fn test_verdict_precedence_zann_over_ihsan() {
        // Speculative content (zann) outranks excellence threshold (ihsan)
        let results = vec![
            GateResult::fail(
                "Ihsan",
                RejectCode::RejectGateIhsan,
                Duration::from_micros(150),
            ),
            GateResult::fail("Zann", RejectCode::RejectZann, Duration::from_micros(100)),
        ];
        let v = GateVerdict::from_gate_results(
            "speculative-mission".into(),
            &results,
            0.50,
            0.80,
            "0.90.0".into(),
            3000,
        );
        assert_eq!(v.primary_reject_code(), Some(RejectCode::RejectZann));
    }

    #[test]
    fn test_verdict_precedence_full_ordering() {
        // All constitutional codes present — Riba must surface
        let results = vec![
            GateResult::fail(
                "Gate1",
                RejectCode::RejectGateSNR,
                Duration::from_micros(10),
            ),
            GateResult::fail(
                "Gate2",
                RejectCode::RejectGateIhsan,
                Duration::from_micros(10),
            ),
            GateResult::fail(
                "Gate3",
                RejectCode::RejectGateFATE,
                Duration::from_micros(10),
            ),
            GateResult::fail("Gate4", RejectCode::RejectZann, Duration::from_micros(10)),
            GateResult::fail("Gate5", RejectCode::RejectRiba, Duration::from_micros(10)),
        ];
        let v = GateVerdict::from_gate_results(
            "worst-case".into(),
            &results,
            0.0,
            0.0,
            "0.90.0".into(),
            4000,
        );
        assert_eq!(v.primary_reject_code(), Some(RejectCode::RejectRiba));
    }

    #[test]
    fn test_verdict_precedence_admitted_returns_none() {
        let v = GateVerdict::from_gate_results(
            "clean-mission".into(),
            &mock_passing_results(),
            0.97,
            0.92,
            "0.90.0".into(),
            5000,
        );
        assert_eq!(v.primary_reject_code(), None);
    }

    #[test]
    fn test_real_gate_chain_to_verdict() {
        let chain = default_gate_chain();
        let ctx = GateContext {
            sender_id: "node0".into(),
            envelope_id: "env-001".into(),
            content: b"{\"task\": \"hello\"}".to_vec(),
            constitution: Constitution::default(),
            snr_score: Some(0.95),
            ihsan_score: Some(0.97),
        };
        let results = chain.verify(&ctx);
        let verdict = GateVerdict::from_gate_results(
            "mission-001".into(),
            &results,
            0.97,
            0.95,
            "0.89.1".into(),
            1000,
        );
        assert!(verdict.is_admitted());
        assert_ne!(verdict.verdict_hash, [0; 32]);
    }
}
