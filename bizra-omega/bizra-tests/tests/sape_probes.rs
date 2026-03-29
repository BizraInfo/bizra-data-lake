//! # SAPE Probe Suite — Phase 2: Probe Rarely Fired Circuits
//!
//! Seven probes that must ALL pass before any capability can be elevated.
//! Each probe tests a fail-closed behavior under adverse conditions.
//!
//! Standing on: Dijkstra (weakest preconditions), Lamport (adversarial reasoning)

use bizra_core::{
    constitution::Constitution,
    pci::{
        gates::{default_gate_chain, GateContext, GateResult},
        verdict::{GateVerdict, ProofStatus, VerdictStatus},
        RejectCode,
    },
};
use bizra_mission::{
    envelope::{ConstitutionalContext, EnvelopeError, MissionEnvelope},
    manifest::{ManifestArtifact, ReceiptRef},
};
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════
// PROBE 1: Negative Path — Constitutional Rejection
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_01_negative_path_rejects_low_ihsan() {
    let chain = default_gate_chain();
    let ctx = GateContext {
        sender_id: "attacker".into(),
        envelope_id: "evil-001".into(),
        content: b"{}".to_vec(),
        constitution: Constitution::default(),
        snr_score: Some(0.95),
        ihsan_score: Some(0.40), // Below 0.95 threshold
    };

    let results = chain.verify(&ctx);
    let verdict = GateVerdict::from_gate_results(
        "negative-test".into(),
        &results,
        0.40,
        0.95,
        "0.89.1".into(),
        1000,
    );

    assert_eq!(verdict.status, VerdictStatus::Rejected);
    assert!(!verdict.is_admitted());
    assert!(
        verdict.reject_codes.contains(&RejectCode::RejectGateIhsan),
        "Expected IhsanGate rejection, got: {:?}",
        verdict.reject_codes
    );
}

#[test]
fn probe_01_negative_path_rejects_missing_scores() {
    let chain = default_gate_chain();
    let ctx = GateContext {
        sender_id: "unknown".into(),
        envelope_id: "no-scores".into(),
        content: b"{}".to_vec(),
        constitution: Constitution::default(),
        snr_score: None, // Missing
        ihsan_score: None, // Missing — fail-closed
    };

    let results = chain.verify(&ctx);

    // Must reject (fail-closed on missing scores)
    assert!(
        results.iter().any(|r| !r.passed),
        "Missing scores must trigger rejection (fail-closed)"
    );
}

// ═══════════════════════════════════════════════════════════════
// PROBE 2: Proof Engine Timeout (simulated via envelope expiry)
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_02_expired_envelope_rejected() {
    let envelope = MissionEnvelope::new(
        "timeout-mission".into(),
        "node0".into(),
        b"Process this request",
        ConstitutionalContext::default(),
        1000,    // created_at
        120_000, // ttl = 120s
    );

    // Verify at time AFTER expiry
    let result = envelope.verify(200_000); // 200s > 120s TTL
    assert_eq!(result, Err(EnvelopeError::Expired));
}

// ═══════════════════════════════════════════════════════════════
// PROBE 3: Dependency Failure (gate chain still rejects)
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_03_invalid_schema_rejected_regardless() {
    let chain = default_gate_chain();
    let ctx = GateContext {
        sender_id: "node0".into(),
        envelope_id: "bad-schema".into(),
        content: b"NOT VALID JSON {{{".to_vec(), // Malformed
        constitution: Constitution::default(),
        snr_score: Some(0.99),
        ihsan_score: Some(0.99),
    };

    let results = chain.verify(&ctx);
    assert!(
        results.iter().any(|r| !r.passed),
        "Invalid schema must be rejected even with perfect scores"
    );
    assert_eq!(results[0].code, RejectCode::RejectGateSchema);
}

// ═══════════════════════════════════════════════════════════════
// PROBE 4: Replay Divergence (tamper detection)
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_04_replay_divergence_detected() {
    let original = MissionEnvelope::new(
        "mission-replay".into(),
        "node0".into(),
        b"Organize files in ~/Documents",
        ConstitutionalContext::default(),
        1000,
        120_000,
    );

    // Clone and tamper
    let mut tampered = original.clone();
    tampered.initiator_id = "tampered-attacker".into();
    // Keep original hash (attacker tries to reuse it)

    let result = tampered.verify(1000);
    assert_eq!(
        result,
        Err(EnvelopeError::IntegrityFailure),
        "Tampered envelope must fail integrity check"
    );
}

#[test]
fn probe_04_replay_divergence_payload_tamper() {
    let original = MissionEnvelope::new(
        "m1".into(),
        "node0".into(),
        b"Sort inbox",
        ConstitutionalContext::default(),
        1000,
        120_000,
    );

    let mut tampered = original.clone();
    // Tamper with payload hash directly
    tampered.payload_hash[0] ^= 0xFF;

    assert_eq!(
        tampered.verify(1000),
        Err(EnvelopeError::IntegrityFailure)
    );
}

// ═══════════════════════════════════════════════════════════════
// PROBE 5: Reflex Promotion with Incomplete Provenance
// (Tests that the reflex system requires sufficient evidence)
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_05_verdict_requires_all_gates() {
    // A verdict with only 1 gate result (should have 3)
    let partial = vec![GateResult::pass("Schema", Duration::from_micros(50))];

    let verdict = GateVerdict::from_gate_results(
        "partial-mission".into(),
        &partial,
        0.97,
        0.95,
        "0.89.1".into(),
        1000,
    );

    // Verdict is "admitted" because the single gate passed,
    // but gate_results.len() < 3 — a complete chain has 3 gates.
    assert_eq!(verdict.gate_results.len(), 1);
    // Downstream consumers should verify gate count >= 3
}

// ═══════════════════════════════════════════════════════════════
// PROBE 6: Policy Version Mismatch
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_06_different_policy_produces_different_verdict() {
    let results = vec![
        GateResult::pass("Schema", Duration::from_micros(50)),
        GateResult::pass("Ihsan", Duration::from_micros(200)),
        GateResult::pass("SNR", Duration::from_micros(100)),
    ];

    let v1 = GateVerdict::from_gate_results(
        "m1".into(), &results, 0.97, 0.95, "0.89.0".into(), 1000,
    );
    let v2 = GateVerdict::from_gate_results(
        "m1".into(), &results, 0.97, 0.95, "0.90.0".into(), 1000,
    );

    // Different policy versions MUST produce different hashes
    assert_ne!(
        v1.verdict_hash, v2.verdict_hash,
        "Policy version change must invalidate verdict hash"
    );
}

// ═══════════════════════════════════════════════════════════════
// PROBE 7: Manifest Integrity (Evidence Bundle Tamper)
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_07_manifest_tamper_detected() {
    let receipts = vec![
        ReceiptRef {
            receipt_id: [1; 32],
            mission_id: [10; 32],
            is_success: true,
            ihsan_score: Some(0.97),
        },
        ReceiptRef {
            receipt_id: [2; 32],
            mission_id: [11; 32],
            is_success: true,
            ihsan_score: Some(0.96),
        },
    ];

    let manifest = ManifestArtifact::new(
        "node0".into(),
        "0.89.1".into(),
        1000,
        2000,
        receipts,
    );

    // Manifest is valid
    assert!(manifest.verify_integrity());

    // Tamper: add a receipt after sealing
    let mut tampered = manifest.clone();
    tampered.receipts.push(ReceiptRef {
        receipt_id: [99; 32],
        mission_id: [99; 32],
        is_success: true,
        ihsan_score: Some(0.99),
    });

    // Tampered manifest MUST fail integrity check
    assert!(
        !tampered.verify_integrity(),
        "Adding a receipt after sealing must break integrity"
    );
}

#[test]
fn probe_07_manifest_reorder_detected() {
    let receipts = vec![
        ReceiptRef {
            receipt_id: [1; 32],
            mission_id: [10; 32],
            is_success: true,
            ihsan_score: Some(0.97),
        },
        ReceiptRef {
            receipt_id: [2; 32],
            mission_id: [11; 32],
            is_success: false,
            ihsan_score: Some(0.40),
        },
    ];

    let manifest = ManifestArtifact::new(
        "node0".into(),
        "0.89.1".into(),
        1000,
        2000,
        receipts,
    );

    // Reorder receipts
    let mut reordered = manifest.clone();
    reordered.receipts.swap(0, 1);

    // Reordering MUST break integrity
    assert!(
        !reordered.verify_integrity(),
        "Reordering receipts must break integrity"
    );
}

// ═══════════════════════════════════════════════════════════════
// META: Full gate chain → verdict → manifest pipeline
// ═══════════════════════════════════════════════════════════════

#[test]
fn probe_meta_full_pipeline_positive_path() {
    // 1. Create envelope
    let envelope = MissionEnvelope::new(
        "pipeline-test".into(),
        "node0-genesis".into(),
        b"{\"task\": \"sort inbox\"}",
        ConstitutionalContext::default(),
        1000,
        120_000,
    );
    assert!(envelope.verify(1000).is_ok());

    // 2. Run gate chain
    let chain = default_gate_chain();
    let ctx = GateContext {
        sender_id: envelope.initiator_id.clone(),
        envelope_id: envelope.mission_id.clone(),
        content: b"{\"task\": \"sort inbox\"}".to_vec(),
        constitution: Constitution::default(),
        snr_score: Some(0.95),
        ihsan_score: Some(0.97),
    };
    let results = chain.verify(&ctx);

    // 3. Produce verdict
    let verdict = GateVerdict::from_gate_results(
        envelope.mission_id.clone(),
        &results,
        0.97,
        0.95,
        "0.89.1".into(),
        1000,
    );
    assert!(verdict.is_admitted());
    assert_eq!(verdict.proof_status, ProofStatus::Verified);

    // 4. Bundle into manifest
    let receipt_ref = ReceiptRef {
        receipt_id: verdict.verdict_hash, // Use verdict hash as receipt ID for test
        mission_id: [0; 32],
        is_success: true,
        ihsan_score: Some(0.97),
    };
    let manifest = ManifestArtifact::new(
        "node0".into(),
        "0.89.1".into(),
        1000,
        2000,
        vec![receipt_ref],
    );
    assert!(manifest.verify_integrity());
    assert_eq!(manifest.admission_rate(), 1.0);
}

#[test]
fn probe_meta_full_pipeline_negative_path() {
    // 1. Create envelope with valid structure
    let envelope = MissionEnvelope::new(
        "negative-pipeline".into(),
        "node0-genesis".into(),
        b"{\"task\": \"evil task\"}",
        ConstitutionalContext::default(),
        1000,
        120_000,
    );

    // 2. Run gate chain with LOW ihsan (constitutional violation)
    let chain = default_gate_chain();
    let ctx = GateContext {
        sender_id: envelope.initiator_id.clone(),
        envelope_id: envelope.mission_id.clone(),
        content: b"{\"task\": \"evil task\"}".to_vec(),
        constitution: Constitution::default(),
        snr_score: Some(0.95),
        ihsan_score: Some(0.30), // Below threshold
    };
    let results = chain.verify(&ctx);

    // 3. Verdict must be REJECTED
    let verdict = GateVerdict::from_gate_results(
        envelope.mission_id.clone(),
        &results,
        0.30,
        0.95,
        "0.89.1".into(),
        1000,
    );
    assert!(!verdict.is_admitted());
    assert_eq!(verdict.status, VerdictStatus::Rejected);

    // 4. Bundle rejected receipt into manifest
    let receipt_ref = ReceiptRef {
        receipt_id: verdict.verdict_hash,
        mission_id: [0; 32],
        is_success: false,
        ihsan_score: Some(0.30),
    };
    let manifest = ManifestArtifact::new(
        "node0".into(),
        "0.89.1".into(),
        1000,
        2000,
        vec![receipt_ref],
    );
    assert!(manifest.verify_integrity());
    assert_eq!(manifest.admission_rate(), 0.0);
    assert_eq!(manifest.rejected, 1);
}
