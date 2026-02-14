//! Integration Tests for BIZRA Omega
//!
//! These tests verify the complete flow across all modules.

use bizra_core::{
    NodeIdentity, Constitution, PCIEnvelope, GateChain, GateContext,
    domain_separated_digest, IHSAN_THRESHOLD, SNR_THRESHOLD,
};

/// Test full PCI envelope lifecycle
#[test]
fn test_pci_envelope_lifecycle() {
    // 1. Create identity
    let identity = NodeIdentity::generate();
    assert!(!identity.node_id().0.is_empty());

    // 2. Create payload
    let payload = serde_json::json!({
        "action": "inference",
        "model": "qwen2.5-7b",
        "prompt": "Hello, BIZRA!"
    });

    // 3. Create envelope
    let envelope = PCIEnvelope::create(&identity, payload.clone(), 3600, vec![])
        .expect("Failed to create envelope");

    // 4. Verify envelope
    assert!(envelope.verify().is_ok());
    assert!(envelope.id.starts_with("pci_"));
    assert_eq!(envelope.ttl, 3600);

    // 5. Verify signature is correct
    let pub_key = identity.public_key_hex();
    assert_eq!(envelope.public_key, pub_key);
}

/// Test constitution validation
#[test]
fn test_constitution_validation() {
    let constitution = Constitution::default();

    // Ihsan checks
    assert!(constitution.check_ihsan(0.95));
    assert!(constitution.check_ihsan(0.99));
    assert!(!constitution.check_ihsan(0.94));
    assert!(!constitution.check_ihsan(0.0));

    // SNR checks
    assert!(constitution.check_snr(0.85));
    assert!(constitution.check_snr(0.95));
    assert!(!constitution.check_snr(0.84));
}

/// Test cryptographic operations
#[test]
fn test_crypto_operations() {
    let identity = NodeIdentity::generate();
    let message = b"BIZRA sovereignty test";

    // Sign
    let signature = identity.sign(message);
    assert!(!signature.is_empty());

    // Verify
    assert!(NodeIdentity::verify(message, &signature, identity.verifying_key()));

    // Tampered message fails
    assert!(!NodeIdentity::verify(b"tampered", &signature, identity.verifying_key()));
}

/// Test domain separation
#[test]
fn test_domain_separation() {
    let msg1 = b"test message";
    let msg2 = b"test message";

    // Same message = same digest
    let d1 = domain_separated_digest(msg1);
    let d2 = domain_separated_digest(msg2);
    assert_eq!(d1, d2);

    // Different from raw hash
    let raw = blake3::hash(msg1).to_hex().to_string();
    assert_ne!(d1, raw);

    // Prefix is applied
    let with_prefix = {
        let mut h = blake3::Hasher::new();
        h.update(b"bizra-pci-v1:");
        h.update(msg1);
        h.finalize().to_hex().to_string()
    };
    assert_eq!(d1, with_prefix);
}

/// Test identity restoration
#[test]
fn test_identity_persistence() {
    let original = NodeIdentity::generate();
    let secret = original.secret_bytes();

    let restored = NodeIdentity::from_secret_bytes(&secret);

    assert_eq!(original.node_id(), restored.node_id());
    assert_eq!(original.public_key_hex(), restored.public_key_hex());

    // Signatures should be identical
    let msg = b"persistence test";
    let sig1 = original.sign(msg);
    let sig2 = restored.sign(msg);
    assert_eq!(sig1, sig2);
}

/// Test gate chain with passing scores
#[test]
fn test_gate_chain_pass() {
    use bizra_core::pci::gates::{default_gate_chain, GateChain};

    let chain = default_gate_chain();
    let constitution = Constitution::default();

    let ctx = GateContext {
        sender_id: "test_node".into(),
        envelope_id: "pci_test123456".into(),
        content: br#"{"valid": "json"}"#.to_vec(),
        constitution,
        snr_score: Some(0.90),
        ihsan_score: Some(0.96),
    };

    let results = chain.verify(&ctx);
    assert!(GateChain::all_passed(&results));
}

/// Test gate chain with failing Ihsan
#[test]
fn test_gate_chain_ihsan_fail() {
    use bizra_core::pci::gates::{default_gate_chain, GateChain};
    use bizra_core::RejectCode;

    let chain = default_gate_chain();
    let constitution = Constitution::default();

    let ctx = GateContext {
        sender_id: "test_node".into(),
        envelope_id: "pci_test789".into(),
        content: br#"{"valid": "json"}"#.to_vec(),
        constitution,
        snr_score: Some(0.90),
        ihsan_score: Some(0.80), // Below threshold
    };

    let results = chain.verify(&ctx);
    assert!(!GateChain::all_passed(&results));

    // Find the failing gate
    let failed = results.iter().find(|r| !r.passed).unwrap();
    assert_eq!(failed.code, RejectCode::RejectGateIhsan);
}

/// Test constants are correct
#[test]
fn test_constants() {
    assert!((IHSAN_THRESHOLD - 0.95).abs() < 0.001);
    assert!((SNR_THRESHOLD - 0.85).abs() < 0.001);
}
