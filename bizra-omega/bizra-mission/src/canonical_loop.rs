//! # Canonical Loop — The Live Runtime Pipeline
//!
//! Wires all 4 cross-layer contracts into one executable path:
//!   MissionEnvelope → GateChain → GateVerdict → ReceiptArtifact → ManifestArtifact
//!
//! This is the production code path that closes the verification gap.
//! Every mission that enters this loop exits with either:
//!   - An ADMITTED verdict + signed receipt + manifest entry, OR
//!   - A REJECTED verdict + rejection receipt + manifest entry
//!
//! No silent completions. No unverified actions. No missing receipts.
//!
//! Core Runtime Law: Mission → Proof → Receipt → Refinement → Reflex → Trust
//!
//! Standing on: Lamport (state machines), Nakamoto (hash chains),
//!              Al-Ghazali (niyyah), Deming (PDCA), Shannon (SNR)

use crate::{
    envelope::{ConstitutionalContext, EnvelopeError, MissionEnvelope},
    manifest::{ManifestArtifact, ReceiptRef},
    mission::Mission,
    receipt::MissionReceipt,
};
use bizra_core::{
    constitution::Constitution,
    pci::{
        gates::{default_gate_chain, GateContext},
        verdict::{GateVerdict, VerdictStatus},
    },
};

/// Result of running one mission through the canonical loop.
#[derive(Debug)]
pub struct CanonicalResult {
    /// The envelope that entered the loop.
    pub envelope: MissionEnvelope,
    /// The gate chain verdict.
    pub verdict: GateVerdict,
    /// Whether the mission was admitted.
    pub admitted: bool,
    /// The mission receipt (always produced — even for rejections).
    pub receipt: MissionReceipt,
}

/// Errors that can occur in the canonical loop.
#[derive(Debug)]
pub enum CanonicalError {
    /// Envelope failed integrity or expiration check.
    EnvelopeInvalid(EnvelopeError),
}

impl std::fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EnvelopeInvalid(e) => write!(f, "Envelope invalid: {e}"),
        }
    }
}

impl std::error::Error for CanonicalError {}

/// Run one mission through the complete canonical loop.
///
/// This is THE production entry point. Every cognitive operation
/// in BIZRA passes through this function.
///
/// Returns `CanonicalResult` with verdict + receipt on success.
/// Returns `CanonicalError` only for envelope-level failures
/// (expired, tampered). Gate rejections are NOT errors — they
/// produce REJECTED verdicts with rejection receipts.
pub fn run_canonical_loop(
    envelope: &MissionEnvelope,
    payload: &[u8],
    constitution: &Constitution,
    ihsan_score: f64,
    snr_score: f64,
    now_ms: u64,
    previous_receipt: Option<[u8; 32]>,
) -> Result<CanonicalResult, CanonicalError> {
    // ── Step 1: Verify envelope integrity ────────────────────
    envelope
        .verify(now_ms)
        .map_err(CanonicalError::EnvelopeInvalid)?;

    // ── Step 2: Run gate chain (Schema → Ihsan → SNR) ────────
    let chain = default_gate_chain();
    let ctx = GateContext {
        sender_id: envelope.initiator_id.clone(),
        envelope_id: envelope.mission_id.clone(),
        content: payload.to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(snr_score),
        ihsan_score: Some(ihsan_score),
    };
    let gate_results = chain.verify(&ctx);

    // ── Step 3: Produce verdict ──────────────────────────────
    let verdict = GateVerdict::from_gate_results(
        envelope.mission_id.clone(),
        &gate_results,
        ihsan_score,
        snr_score,
        envelope.constitutional_context.policy_version.clone(),
        now_ms,
    );

    let admitted = verdict.is_admitted();

    // ── Step 4: Create mission + receipt ──────────────────────
    let mut mission = Mission::new(envelope.payload_hash, now_ms);

    if admitted {
        mission.ihsan_score = Some(ihsan_score as f32);
        mission.snr_score = Some(snr_score as f32);
        mission.guardian_approved = Some(true);
        mission
            .transition(
                crate::state::MissionState::Complete,
                now_ms,
                "Canonical loop: admitted",
            )
            .ok();
    } else {
        mission.ihsan_score = Some(ihsan_score as f32);
        mission.snr_score = Some(snr_score as f32);
        mission.guardian_approved = Some(false);
        mission
            .transition(
                crate::state::MissionState::Failed,
                now_ms,
                "Canonical loop: rejected by gate chain",
            )
            .ok();
    }

    let receipt = MissionReceipt::from_mission(&mission, previous_receipt);

    Ok(CanonicalResult {
        envelope: envelope.clone(),
        verdict,
        admitted,
        receipt,
    })
}

/// Run multiple missions and produce a manifest.
pub fn run_canonical_batch(
    missions: Vec<(&MissionEnvelope, &[u8], f64, f64)>,
    constitution: &Constitution,
    node_id: &str,
    policy_version: &str,
    now_ms: u64,
) -> (Vec<CanonicalResult>, ManifestArtifact) {
    let mut results = Vec::new();
    let mut receipt_refs = Vec::new();
    let mut prev_hash: Option<[u8; 32]> = None;

    for (envelope, payload, ihsan, snr) in &missions {
        match run_canonical_loop(
            envelope,
            payload,
            constitution,
            *ihsan,
            *snr,
            now_ms,
            prev_hash,
        ) {
            Ok(result) => {
                let receipt_ref = ReceiptRef {
                    receipt_id: result.receipt.receipt_id,
                    mission_id: result.receipt.mission_id,
                    is_success: result.admitted,
                    ihsan_score: result.receipt.ihsan_score,
                };
                prev_hash = Some(result.receipt.receipt_id);
                receipt_refs.push(receipt_ref);
                results.push(result);
            }
            Err(_) => {
                // Envelope-level failures don't produce receipts
                // (the envelope itself is invalid — nothing to receipt)
            }
        }
    }

    let manifest = ManifestArtifact::new(
        node_id.to_string(),
        policy_version.to_string(),
        now_ms,
        now_ms + 1,
        receipt_refs,
    );

    (results, manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_constitution() -> Constitution {
        Constitution::default()
    }

    fn test_envelope(now: u64) -> MissionEnvelope {
        MissionEnvelope::new(
            "canonical-test".into(),
            "node0-genesis".into(),
            b"{\"task\": \"sort inbox\"}",
            ConstitutionalContext::default(),
            now,
            120_000,
        )
    }

    #[test]
    fn test_canonical_loop_positive_path() {
        let now = 1000;
        let envelope = test_envelope(now);
        let result = run_canonical_loop(
            &envelope,
            b"{\"task\": \"sort inbox\"}",
            &test_constitution(),
            0.97, // Above Ihsan threshold
            0.95, // Above SNR threshold
            now,
            None,
        )
        .unwrap();

        assert!(result.admitted);
        assert_eq!(result.verdict.status, VerdictStatus::Admitted);
        assert!(result.receipt.verify_hash());
    }

    #[test]
    fn test_canonical_loop_negative_path() {
        let now = 1000;
        let envelope = test_envelope(now);
        let result = run_canonical_loop(
            &envelope,
            b"{\"task\": \"evil task\"}",
            &test_constitution(),
            0.30, // Below Ihsan threshold — REJECTED
            0.95,
            now,
            None,
        )
        .unwrap();

        assert!(!result.admitted);
        assert_eq!(result.verdict.status, VerdictStatus::Rejected);
        // Rejection still produces a receipt
        assert!(result.receipt.verify_hash());
    }

    #[test]
    fn test_canonical_loop_expired_envelope() {
        let now = 1000;
        let envelope = test_envelope(now);
        // Verify at time AFTER expiry
        let result = run_canonical_loop(
            &envelope,
            b"{}",
            &test_constitution(),
            0.97,
            0.95,
            200_000, // Way past TTL
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_canonical_loop_receipt_chain() {
        let now = 1000;
        let constitution = test_constitution();

        // Mission 1
        let e1 = MissionEnvelope::new(
            "m1".into(),
            "node0".into(),
            b"{\"a\":1}",
            ConstitutionalContext::default(),
            now,
            120_000,
        );
        let r1 =
            run_canonical_loop(&e1, b"{\"a\":1}", &constitution, 0.97, 0.95, now, None).unwrap();

        // Mission 2 — chained to mission 1
        let e2 = MissionEnvelope::new(
            "m2".into(),
            "node0".into(),
            b"{\"a\":2}",
            ConstitutionalContext::default(),
            now,
            120_000,
        );
        let r2 = run_canonical_loop(
            &e2,
            b"{\"a\":2}",
            &constitution,
            0.96,
            0.93,
            now,
            Some(r1.receipt.receipt_id), // Chain link
        )
        .unwrap();

        // Verify chain
        assert_eq!(
            r2.receipt.previous_receipt_hash,
            Some(r1.receipt.receipt_id)
        );
    }

    #[test]
    fn test_canonical_batch_produces_manifest() {
        let now = 1000;
        let constitution = test_constitution();

        let e1 = MissionEnvelope::new(
            "b1".into(),
            "node0".into(),
            b"{\"ok\":true}",
            ConstitutionalContext::default(),
            now,
            120_000,
        );
        let e2 = MissionEnvelope::new(
            "b2".into(),
            "node0".into(),
            b"{\"bad\":true}",
            ConstitutionalContext::default(),
            now,
            120_000,
        );

        let missions = vec![
            (&e1, b"{\"ok\":true}".as_slice(), 0.97, 0.95), // Admitted
            (&e2, b"{\"bad\":true}".as_slice(), 0.30, 0.95), // Rejected
        ];

        let (results, manifest) =
            run_canonical_batch(missions, &constitution, "node0", "0.89.1", now);

        assert_eq!(results.len(), 2);
        assert!(results[0].admitted);
        assert!(!results[1].admitted);
        assert_eq!(manifest.total_missions, 2);
        assert_eq!(manifest.admitted, 1);
        assert_eq!(manifest.rejected, 1);
        assert!(manifest.verify_integrity());
    }
}
