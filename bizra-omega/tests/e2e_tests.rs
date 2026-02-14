//! End-to-End Integration Tests
//!
//! These tests verify the complete BIZRA flow:
//! Identity → PCI Envelope → Gates → Inference → Federation
//!
//! Run with: cargo test --test e2e_tests

use bizra_core::{
    NodeIdentity, Constitution, PCIEnvelope,
    domain_separated_digest, IHSAN_THRESHOLD, SNR_THRESHOLD,
    pci::gates::{default_gate_chain, GateChain, GateContext},
    pci::RejectCode,
};
use bizra_inference::{
    gateway::{InferenceRequest, InferenceGateway},
    selector::{ModelSelector, ModelTier, TaskComplexity},
};
use bizra_federation::{
    gossip::{GossipProtocol, GossipConfig, Member, NodeState},
    consensus::{ConsensusEngine, Proposal, Vote},
};

/// Test complete identity lifecycle
#[test]
fn e2e_identity_lifecycle() {
    // 1. Generate identity
    let identity = NodeIdentity::generate();
    assert!(!identity.node_id().0.is_empty());
    println!("✓ Generated node: {}", identity.node_id().0);

    // 2. Sign message
    let message = b"BIZRA sovereignty test";
    let signature = identity.sign(message);
    assert!(!signature.is_empty());
    println!("✓ Signed message: {}...", &signature[..16]);

    // 3. Verify signature
    assert!(NodeIdentity::verify(message, &signature, identity.verifying_key()));
    println!("✓ Signature verified");

    // 4. Tampered message fails
    assert!(!NodeIdentity::verify(b"tampered", &signature, identity.verifying_key()));
    println!("✓ Tampered message rejected");

    // 5. Persist and restore
    let secret = identity.secret_bytes();
    let restored = NodeIdentity::from_secret_bytes(&secret);
    assert_eq!(identity.node_id(), restored.node_id());
    println!("✓ Identity persistence verified");
}

/// Test complete PCI envelope flow
#[test]
fn e2e_pci_envelope_flow() {
    let identity = NodeIdentity::generate();

    // 1. Create envelope with JSON payload
    let payload = serde_json::json!({
        "action": "inference",
        "model": "qwen2.5-7b",
        "prompt": "Explain quantum computing",
        "max_tokens": 512
    });

    let envelope = PCIEnvelope::create(&identity, payload.clone(), 3600, vec![])
        .expect("Failed to create envelope");

    println!("✓ Created envelope: {}", envelope.id);
    assert!(envelope.id.starts_with("pci_"));

    // 2. Verify envelope
    assert!(envelope.verify().is_ok());
    println!("✓ Envelope verified");

    // 3. Check content hash
    let expected_hash = domain_separated_digest(
        serde_json::to_string(&payload).unwrap().as_bytes()
    );
    assert_eq!(envelope.content_hash, expected_hash);
    println!("✓ Content hash matches");

    // 4. Provenance chain
    let derived_payload = serde_json::json!({
        "action": "summary",
        "source": envelope.id.clone()
    });
    let derived = PCIEnvelope::create(&identity, derived_payload, 3600, vec![envelope.id.clone()])
        .expect("Failed to create derived envelope");

    assert_eq!(derived.provenance.len(), 1);
    assert_eq!(derived.provenance[0], envelope.id);
    println!("✓ Provenance chain established");
}

/// Test gate chain validation
#[test]
fn e2e_gate_chain_validation() {
    let chain = default_gate_chain();
    let constitution = Constitution::default();

    // 1. Valid content passes all gates
    let valid_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_abc123".into(),
        content: br#"{"valid": "json", "data": 42}"#.to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(0.90),
        ihsan_score: Some(0.96),
    };

    let results = chain.verify(&valid_ctx);
    assert!(GateChain::all_passed(&results));
    println!("✓ Valid content passed all {} gates", results.len());

    // 2. Invalid JSON fails schema gate
    let invalid_json_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_def456".into(),
        content: b"not valid json {{{".to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(0.90),
        ihsan_score: Some(0.96),
    };

    let results = chain.verify(&invalid_json_ctx);
    assert!(!GateChain::all_passed(&results));
    assert_eq!(results[0].code, RejectCode::RejectGateSchema);
    println!("✓ Invalid JSON rejected at schema gate");

    // 3. Low SNR fails SNR gate
    let low_snr_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_ghi789".into(),
        content: br#"{"valid": "json"}"#.to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(0.50), // Below 0.85 threshold
        ihsan_score: Some(0.96),
    };

    let results = chain.verify(&low_snr_ctx);
    assert!(!GateChain::all_passed(&results));
    println!("✓ Low SNR rejected at SNR gate");

    // 4. Low Ihsan fails Ihsan gate
    let low_ihsan_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_jkl012".into(),
        content: br#"{"valid": "json"}"#.to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(0.90),
        ihsan_score: Some(0.80), // Below 0.95 threshold
    };

    let results = chain.verify(&low_ihsan_ctx);
    assert!(!GateChain::all_passed(&results));
    println!("✓ Low Ihsan rejected at Ihsan gate");
}

/// Test model tier selection
#[test]
fn e2e_model_tier_selection() {
    let selector = ModelSelector::default();

    // 1. Simple task → Edge tier
    let simple = TaskComplexity::estimate("What is 2+2?", 50);
    assert_eq!(simple, TaskComplexity::Simple);
    assert_eq!(selector.select_tier(&simple), ModelTier::Edge);
    println!("✓ Simple task → Edge tier");

    // 2. Medium task → Edge tier
    let medium = TaskComplexity::estimate(
        "Write a function to calculate fibonacci numbers up to n",
        200
    );
    assert_eq!(medium, TaskComplexity::Medium);
    assert_eq!(selector.select_tier(&medium), ModelTier::Edge);
    println!("✓ Medium task → Edge tier");

    // 3. Complex task → Local tier
    let complex = TaskComplexity::estimate(
        "Explain the mathematical foundations of quantum computing and \
         how Shor's algorithm works, including the quantum Fourier transform",
        1000
    );
    assert_eq!(complex, TaskComplexity::Complex);
    assert_eq!(selector.select_tier(&complex), ModelTier::Local);
    println!("✓ Complex task → Local tier");

    // 4. Expert task → Pool tier
    let expert = TaskComplexity::estimate(
        &"Explain ".repeat(100) + "```python\ndef complex_algo(): pass```",
        3000
    );
    assert_eq!(expert, TaskComplexity::Expert);
    assert_eq!(selector.select_tier(&expert), ModelTier::Pool);
    println!("✓ Expert task → Pool tier");
}

/// Test constitution thresholds
#[test]
fn e2e_constitution_thresholds() {
    let constitution = Constitution::default();

    // 1. Verify default thresholds
    assert!((IHSAN_THRESHOLD - 0.95).abs() < 0.001);
    assert!((SNR_THRESHOLD - 0.85).abs() < 0.001);
    println!("✓ Default thresholds: Ihsan={}, SNR={}", IHSAN_THRESHOLD, SNR_THRESHOLD);

    // 2. Ihsan checks
    assert!(constitution.check_ihsan(0.95));
    assert!(constitution.check_ihsan(0.99));
    assert!(constitution.check_ihsan(1.0));
    assert!(!constitution.check_ihsan(0.94));
    assert!(!constitution.check_ihsan(0.0));
    println!("✓ Ihsan threshold enforced");

    // 3. SNR checks
    assert!(constitution.check_snr(0.85));
    assert!(constitution.check_snr(0.95));
    assert!(constitution.check_snr(1.0));
    assert!(!constitution.check_snr(0.84));
    assert!(!constitution.check_snr(0.0));
    println!("✓ SNR threshold enforced");
}

/// Test domain separation
#[test]
fn e2e_domain_separation() {
    let message = b"test message for hashing";

    // 1. Domain-separated hash
    let digest = domain_separated_digest(message);
    assert_eq!(digest.len(), 64); // 32 bytes hex
    println!("✓ Domain digest: {}...", &digest[..16]);

    // 2. Deterministic
    let digest2 = domain_separated_digest(message);
    assert_eq!(digest, digest2);
    println!("✓ Deterministic hashing");

    // 3. Different from raw hash
    let raw = blake3::hash(message).to_hex().to_string();
    assert_ne!(digest, raw);
    println!("✓ Domain separation applied");

    // 4. Correct prefix
    let expected = {
        let mut h = blake3::Hasher::new();
        h.update(b"bizra-pci-v1:");
        h.update(message);
        h.finalize().to_hex().to_string()
    };
    assert_eq!(digest, expected);
    println!("✓ Prefix verified: bizra-pci-v1:");
}

/// Test gossip protocol membership
#[tokio::test]
async fn e2e_gossip_membership() {
    let config = GossipConfig::default();
    let identity = NodeIdentity::generate();
    let mut gossip = GossipProtocol::new(config, identity.node_id().0.clone());

    // 1. Initially empty
    assert_eq!(gossip.members().len(), 0);
    println!("✓ Initial members: 0");

    // 2. Add members
    gossip.add_member(Member {
        node_id: "peer_1".into(),
        address: "192.168.1.1:7946".parse().unwrap(),
        state: NodeState::Alive,
        incarnation: 1,
        last_updated: std::time::Instant::now(),
    });

    gossip.add_member(Member {
        node_id: "peer_2".into(),
        address: "192.168.1.2:7946".parse().unwrap(),
        state: NodeState::Alive,
        incarnation: 1,
        last_updated: std::time::Instant::now(),
    });

    assert_eq!(gossip.members().len(), 2);
    println!("✓ Added 2 members");

    // 3. Get alive members
    let alive = gossip.alive_members();
    assert_eq!(alive.len(), 2);
    println!("✓ 2 members alive");

    // 4. Mark one as suspect
    gossip.mark_suspect("peer_1");
    let alive = gossip.alive_members();
    assert_eq!(alive.len(), 1);
    println!("✓ 1 member alive after suspect");
}

/// Test consensus voting
#[tokio::test]
async fn e2e_consensus_voting() {
    let engine = ConsensusEngine::new(5); // 5 nodes, f=2, quorum=5

    // 1. Create proposal
    let proposal = Proposal {
        id: "prop_001".into(),
        pattern_id: "pattern_xyz".into(),
        proposer: "node_1".into(),
        data: vec![0.1, 0.2, 0.3],
    };

    engine.submit_proposal(proposal.clone());
    println!("✓ Proposal submitted: {}", proposal.id);

    // 2. Not enough votes (need 5)
    for i in 1..=4 {
        engine.submit_vote(Vote {
            proposal_id: "prop_001".into(),
            voter: format!("node_{}", i),
            accept: true,
        });
    }

    assert!(engine.check_consensus("prop_001").is_none());
    println!("✓ No consensus with 4 votes");

    // 3. Quorum reached with 5th vote
    engine.submit_vote(Vote {
        proposal_id: "prop_001".into(),
        voter: "node_5".into(),
        accept: true,
    });

    assert!(engine.check_consensus("prop_001").is_some());
    println!("✓ Consensus reached with 5 votes");
}

/// Test full inference request flow (without actual backend)
#[test]
fn e2e_inference_request() {
    // 1. Create request
    let request = InferenceRequest {
        id: "req_001".into(),
        prompt: "What is the meaning of life?".into(),
        system: Some("You are a philosophical AI.".into()),
        max_tokens: 512,
        temperature: 0.7,
        complexity: TaskComplexity::Medium,
        preferred_tier: Some(ModelTier::Local),
    };

    assert_eq!(request.id, "req_001");
    println!("✓ Created inference request: {}", request.id);

    // 2. Verify defaults
    let default = InferenceRequest::default();
    assert!(default.id.len() > 0);
    assert_eq!(default.max_tokens, 1024);
    assert_eq!(default.temperature, 0.7);
    println!("✓ Default request validated");
}

/// Integration: Full PCI + Gate flow
#[test]
fn e2e_full_pci_gate_flow() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let chain = default_gate_chain();

    // 1. Create high-quality envelope
    let payload = serde_json::json!({
        "action": "sovereign_inference",
        "quality_score": 0.97,
        "snr": 0.92
    });

    let envelope = PCIEnvelope::create(&identity, payload, 3600, vec![])
        .expect("Failed to create envelope");

    // 2. Verify envelope signature
    assert!(envelope.verify().is_ok());

    // 3. Check gates with scores
    let ctx = GateContext {
        sender_id: identity.node_id().0.clone(),
        envelope_id: envelope.id.clone(),
        content: serde_json::to_vec(&envelope.payload).unwrap(),
        constitution,
        snr_score: Some(0.92),
        ihsan_score: Some(0.97),
    };

    let results = chain.verify(&ctx);
    assert!(GateChain::all_passed(&results));

    println!("✓ Full PCI + Gate flow passed");
    println!("  • Envelope: {}", envelope.id);
    println!("  • Sender: {}", identity.node_id().0);
    println!("  • Gates passed: {}", results.len());
}

/// Performance benchmark: Identity operations
#[test]
fn benchmark_identity_ops() {
    use std::time::Instant;

    let identity = NodeIdentity::generate();
    let message = b"Benchmark message for BIZRA";

    // Sign 100 times
    let start = Instant::now();
    for _ in 0..100 {
        identity.sign(message);
    }
    let sign_elapsed = start.elapsed();

    // Verify 100 times
    let signature = identity.sign(message);
    let start = Instant::now();
    for _ in 0..100 {
        NodeIdentity::verify(message, &signature, identity.verifying_key());
    }
    let verify_elapsed = start.elapsed();

    // Hash 1000 times
    let start = Instant::now();
    for i in 0..1000 {
        domain_separated_digest(format!("message {}", i).as_bytes());
    }
    let hash_elapsed = start.elapsed();

    println!("\n📊 Performance Benchmarks:");
    println!("   100 signatures: {:?} ({:.0}/sec)", sign_elapsed, 100.0 / sign_elapsed.as_secs_f64());
    println!("   100 verifies:   {:?} ({:.0}/sec)", verify_elapsed, 100.0 / verify_elapsed.as_secs_f64());
    println!("   1000 hashes:    {:?} ({:.0}/sec)", hash_elapsed, 1000.0 / hash_elapsed.as_secs_f64());

    // Assert reasonable performance
    assert!(sign_elapsed.as_millis() < 5000); // < 5s for 100 sigs
    assert!(hash_elapsed.as_millis() < 1000); // < 1s for 1000 hashes
}
