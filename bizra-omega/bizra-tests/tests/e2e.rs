//! End-to-End Integration Tests
//!
//! These tests verify the complete BIZRA flow:
//! Identity → PCI Envelope → Gates → Inference → Federation
//!
//! Run with: cargo test -p bizra-tests --test e2e

use bizra_core::{
    domain_separated_digest,
    pci::gates::{default_gate_chain, GateChain, GateContext},
    pci::RejectCode,
    Constitution, NodeId, NodeIdentity, PCIEnvelope, IHSAN_THRESHOLD, SNR_THRESHOLD,
};
use bizra_federation::{
    consensus::ConsensusEngine,
    gossip::{GossipMessage, GossipProtocol, Member, SignedGossipMessage},
};
use bizra_inference::{
    gateway::InferenceRequest,
    selector::{ModelSelector, ModelTier, TaskComplexity},
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
    assert!(NodeIdentity::verify(
        message,
        &signature,
        identity.verifying_key()
    ));
    println!("✓ Signature verified");

    // 4. Tampered message fails
    assert!(!NodeIdentity::verify(
        b"tampered",
        &signature,
        identity.verifying_key()
    ));
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
    let expected_hash =
        domain_separated_digest(serde_json::to_string(&payload).unwrap().as_bytes());
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

    // 2. Medium or Simple task → Edge tier
    let medium = TaskComplexity::estimate(
        "Write a function to calculate fibonacci numbers up to n",
        200,
    );
    // Complexity depends on word count and max_tokens
    assert!(matches!(
        medium,
        TaskComplexity::Simple | TaskComplexity::Medium
    ));
    assert_eq!(selector.select_tier(&medium), ModelTier::Edge);
    println!("✓ Medium/Simple task → Edge tier");

    // 3. Complex task → Local tier
    let complex = TaskComplexity::estimate(
        "Explain the mathematical foundations of quantum computing and \
         how Shor's algorithm works, including the quantum Fourier transform",
        1000,
    );
    assert_eq!(complex, TaskComplexity::Complex);
    assert_eq!(selector.select_tier(&complex), ModelTier::Local);
    println!("✓ Complex task → Local tier");

    // 4. Expert task → Pool tier
    let expert = TaskComplexity::estimate(
        &("Explain ".repeat(100) + "```python\ndef complex_algo(): pass```"),
        3000,
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
    println!(
        "✓ Default thresholds: Ihsan={}, SNR={}",
        IHSAN_THRESHOLD, SNR_THRESHOLD
    );

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
    let node_id = NodeId("node_local".into());
    let local_addr: std::net::SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let gossip = GossipProtocol::new_with_generated_key(node_id.clone(), local_addr);

    // 1. Initially has self
    assert_eq!(gossip.member_count().await, 1);
    println!("✓ Initial member count: 1 (self)");

    // 2. Add seed nodes
    let peer1_id = NodeId("peer_1".into());
    let peer1_addr: std::net::SocketAddr = "192.168.1.1:7946".parse().unwrap();
    gossip.add_seed(peer1_id.clone(), peer1_addr).await;

    let peer2_id = NodeId("peer_2".into());
    let peer2_addr: std::net::SocketAddr = "192.168.1.2:7946".parse().unwrap();
    gossip.add_seed(peer2_id.clone(), peer2_addr).await;

    assert_eq!(gossip.member_count().await, 3);
    println!("✓ Added 2 seed nodes, total: 3");

    // 3. Get alive members
    let alive = gossip.alive_members().await;
    assert_eq!(alive.len(), 3);
    println!("✓ 3 members alive");

    // 4. Handle signed join message (secure API)
    let peer3_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    let peer3_id = NodeId("peer_3".into());
    gossip
        .register_peer_pubkey(peer3_id.clone(), peer3_key.verifying_key().to_bytes())
        .await;
    let new_member = Member::new(peer3_id, "192.168.1.3:7946".parse().unwrap());
    let join_msg = SignedGossipMessage::sign(
        GossipMessage::Join { member: new_member },
        &peer3_key,
    );
    gossip.handle_signed_message(join_msg).await.unwrap();
    assert_eq!(gossip.member_count().await, 4);
    println!("✓ Handled signed join, total: 4");

    // 5. Handle signed leave message
    let peer1_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    gossip
        .register_peer_pubkey(peer1_id.clone(), peer1_key.verifying_key().to_bytes())
        .await;
    let leave_msg = SignedGossipMessage::sign(
        GossipMessage::Leave { node_id: peer1_id },
        &peer1_key,
    );
    gossip.handle_signed_message(leave_msg).await.unwrap();
    let alive = gossip.alive_members().await;
    assert_eq!(alive.len(), 3); // peer_1 is now Left, not Alive
    println!("✓ Handled signed leave, alive: 3");
}

/// Test consensus voting
#[test]
fn e2e_consensus_voting() {
    let identity = NodeIdentity::generate();
    let mut engine = ConsensusEngine::new(identity);
    engine.set_node_count(5); // 5 nodes, quorum = 4 (2/3 + 1)

    // 1. Create proposal with sufficient Ihsan
    let pattern = serde_json::json!({
        "pattern_id": "pattern_xyz",
        "embedding": [0.1, 0.2, 0.3]
    });

    let proposal = engine.propose(pattern, 0.97).expect("Failed to propose");
    println!("✓ Proposal submitted: {}", proposal.id);

    // 2. Self-vote
    let vote = engine
        .vote(&proposal.id, true, 0.96)
        .expect("Failed to vote");
    let consensus = engine.receive_vote(vote).expect("Failed to receive vote");
    assert!(!consensus); // Need more votes
    println!("✓ Self-vote received, no consensus yet");

    // 3. Add more votes (simulating other nodes)
    // Note: In real scenario, votes would come from other nodes
    // For testing, we just verify the API works

    // 4. Verify proposal below Ihsan threshold is rejected
    let low_ihsan = engine.propose(serde_json::json!({}), 0.80);
    assert!(low_ihsan.is_err());
    println!("✓ Low Ihsan proposal rejected");
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
    assert!(!default.id.is_empty());
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

    let envelope =
        PCIEnvelope::create(&identity, payload, 3600, vec![]).expect("Failed to create envelope");

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
    println!(
        "   100 signatures: {:?} ({:.0}/sec)",
        sign_elapsed,
        100.0 / sign_elapsed.as_secs_f64()
    );
    println!(
        "   100 verifies:   {:?} ({:.0}/sec)",
        verify_elapsed,
        100.0 / verify_elapsed.as_secs_f64()
    );
    println!(
        "   1000 hashes:    {:?} ({:.0}/sec)",
        hash_elapsed,
        1000.0 / hash_elapsed.as_secs_f64()
    );

    // Assert reasonable performance
    assert!(sign_elapsed.as_millis() < 5000); // < 5s for 100 sigs
    assert!(hash_elapsed.as_millis() < 1000); // < 1s for 1000 hashes
}
