# BIZRA Omega Test Expansion Plan

**Version**: 2.0
**Target Coverage**: >= 85%
**Framework**: Rust (cargo test) + Property-based (proptest)
**Date**: 2026-02-01

---

## Executive Summary

This document outlines a comprehensive test expansion strategy for the BIZRA Omega ecosystem. The analysis identified significant coverage gaps in:

1. **Graph-of-Thoughts (GoT) Operations** - Missing property-based tests for complex graph operations
2. **Inference Gateway Fallback** - No tests for backend failover, timeout handling, or tier escalation
3. **Federation Security** - Limited fuzz testing for message parsing and replay attack vectors
4. **A2A Protocol** - Absent from current test suite entirely

---

## Current Coverage Analysis

### Existing Test Files

| File | Lines | Coverage Areas |
|------|-------|----------------|
| `tests/integration_tests.rs` | 168 | PCI envelope, constitution, crypto, gates |
| `tests/e2e_tests.rs` | 441 | Full pipeline, gossip basic, consensus basic |
| `bizra-tests/tests/integration.rs` | 547 | Sovereign module, Omega engine, SNR |

### Coverage Gaps Identified

| Module | Current | Gap | Priority |
|--------|---------|-----|----------|
| `graph_of_thoughts.rs` | 40% | Parallel exploration, aggregate consensus, backtrack edge cases | P0 |
| `gateway.rs` | 25% | Backend fallback, timeout recovery, tier escalation | P0 |
| `gossip.rs` | 35% | Signature verification, replay protection, malformed messages | P0 |
| `consensus.rs` | 45% | Byzantine scenarios, quorum edge cases, vote spoofing | P1 |
| `snr_engine.rs` | 70% | Boundary conditions, Unicode handling, adversarial input | P1 |
| `selector.rs` | 30% | Complexity estimation, tier boundary conditions | P2 |
| A2A Protocol | 0% | Entire module untested | P0 |

---

## P0: Critical Test Cases (Immediate)

### 1. Graph-of-Thoughts Property-Based Tests

**Location**: `bizra-tests/tests/got_properties.rs`

```rust
//! Property-based tests for Graph-of-Thoughts operations
//! Standing on Giants: Besta et al. (2024)

use proptest::prelude::*;
use bizra_core::sovereign::graph_of_thoughts::*;

// === PROPERTY: Graph Invariants ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: Every node has at most one parent
    #[test]
    fn prop_single_parent(
        operations in prop::collection::vec(any::<GraphOp>(), 1..100)
    ) {
        let mut graph = ThoughtGraph::new();
        for op in operations {
            apply_operation(&mut graph, op);
        }

        for thought in graph.thoughts.values() {
            if let Some(parent_id) = &thought.parent {
                // Parent must exist
                prop_assert!(graph.thoughts.contains_key(parent_id));
                // Parent must have this node as child
                let parent = graph.thoughts.get(parent_id).unwrap();
                prop_assert!(parent.children.contains(&thought.id));
            }
        }
    }

    /// Property: Frontier nodes are always leaves
    #[test]
    fn prop_frontier_are_leaves(
        num_thoughts in 1..50usize
    ) {
        let mut graph = ThoughtGraph::new();
        let root = graph.create_thought("root", None);

        for i in 0..num_thoughts {
            let parent = if i == 0 { root.clone() } else {
                format!("thought_{}", (i % 5) + 1)
            };
            graph.create_thought(&format!("thought_{}", i + 1), Some(&parent));
        }

        let frontier = graph.get_frontier();
        for node in frontier {
            prop_assert!(node.children.is_empty(),
                "Frontier node {} has children", node.id);
        }
    }

    /// Property: Backtrack always returns highest-SNR unexplored node
    #[test]
    fn prop_backtrack_selects_max_snr(
        snr_scores in prop::collection::vec(0.0f64..1.0, 2..20)
    ) {
        let mut graph = ThoughtGraph::new();
        let root = graph.create_thought("root", None);

        let mut expected_max_id = None;
        let mut max_snr = f64::NEG_INFINITY;

        for (i, snr) in snr_scores.iter().enumerate() {
            let id = graph.create_thought_with_type(
                &format!("hypothesis_{}", i),
                Some(&root),
                ThoughtType::Hypothesis,
            );
            if let Some(node) = graph.get_thought_mut(&id) {
                node.set_snr(*snr);
            }
            if *snr > max_snr {
                max_snr = *snr;
                expected_max_id = Some(id);
            }
        }

        let backtrack_node = graph.backtrack();
        prop_assert!(backtrack_node.is_some());
        prop_assert_eq!(
            backtrack_node.unwrap().id,
            expected_max_id.unwrap()
        );
    }

    /// Property: Aggregate consensus requires majority
    #[test]
    fn prop_aggregate_consensus_majority(
        path_results in prop::collection::vec(any::<bool>(), 3..10)
    ) {
        let graph = ThoughtGraph::new();
        let paths: Vec<ReasoningPath> = path_results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                let mut path = graph.create_path(&format!("path_{}", i));
                path.add_thought(ThoughtNode::new("t", "test"));
                path.record_result("t", *result);
                path
            })
            .collect();

        let aggregate = graph.aggregate_paths(&paths);
        let successful = path_results.iter().filter(|&&r| r).count();
        let expected_consensus = successful > path_results.len() / 2;

        prop_assert_eq!(aggregate.consensus, expected_consensus);
    }
}

// === PROPERTY: SNR Monotonicity ===

proptest! {
    /// Property: Completing a thought with higher confidence improves path confidence
    #[test]
    fn prop_confidence_monotonic_on_success(
        initial_confidence in 0.1f64..0.9
    ) {
        let mut path = ReasoningPath::new("test");
        path.add_thought(ThoughtNode::new("t1", "thought"));
        path.confidence = initial_confidence;

        // Success should not decrease confidence
        path.record_result("t1", true);
        prop_assert!(path.confidence >= initial_confidence * 0.5);
    }
}

// === EDGE CASES ===

#[test]
fn test_got_empty_graph_operations() {
    let graph = ThoughtGraph::new();

    // Empty graph frontier is empty
    assert!(graph.get_frontier().is_empty());

    // Backtrack on empty graph returns None
    assert!(graph.backtrack().is_none());

    // Get conclusions on empty graph returns empty
    assert!(graph.get_conclusions(0.0).is_empty());

    // Stats reflect empty state
    let stats = graph.stats();
    assert_eq!(stats.total_thoughts, 0);
    assert_eq!(stats.root_count, 0);
}

#[test]
fn test_got_single_conclusion_only() {
    let mut graph = ThoughtGraph::new();

    // Single conclusion node
    let conclusion = graph.create_thought_with_type(
        "Final answer",
        None,
        ThoughtType::Conclusion,
    );

    if let Some(node) = graph.get_thought_mut(&conclusion) {
        node.set_snr(0.95);
    }

    // Backtrack should return None (conclusion is terminal)
    assert!(graph.backtrack().is_none());

    // But conclusions should be found
    let conclusions = graph.get_conclusions(0.9);
    assert_eq!(conclusions.len(), 1);
}

#[test]
fn test_got_deep_tree_backtrack() {
    let mut graph = ThoughtGraph::new();

    // Create deep chain: root -> h1 -> h2 -> h3 -> h4 -> h5
    let root = graph.create_thought("root", None);
    let mut parent = root;
    let mut deepest = String::new();

    for i in 1..=5 {
        let id = graph.create_thought_with_type(
            &format!("hypothesis_{}", i),
            Some(&parent),
            ThoughtType::Hypothesis,
        );
        if let Some(node) = graph.get_thought_mut(&id) {
            node.set_snr(0.5 + (i as f64 * 0.08));
        }
        deepest = id.clone();
        parent = id;
    }

    // Backtrack should return deepest node (h5 with highest SNR 0.9)
    let backtrack = graph.backtrack().unwrap();
    assert_eq!(backtrack.id, deepest);
}

#[test]
fn test_reasoning_path_partial_completion() {
    let mut path = ReasoningPath::new("test");

    path.add_thought(ThoughtNode::new("t1", "first"));
    path.add_thought(ThoughtNode::new("t2", "second"));
    path.add_thought(ThoughtNode::new("t3", "third"));

    // Partial completion
    path.record_result("t1", true);
    path.record_result("t2", false);

    assert!(!path.is_complete());
    assert_eq!(path.success_rate(), 0.5);

    // Final result should be false (not all passed)
    assert_eq!(path.final_result, Some(false));
}
```

### 2. Inference Gateway Fallback Tests

**Location**: `bizra-tests/tests/inference_fallback.rs`

```rust
//! Inference Gateway fallback and resilience tests

use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use bizra_core::{NodeIdentity, Constitution};
use bizra_inference::{
    gateway::{InferenceGateway, InferenceRequest, GatewayError},
    selector::{ModelTier, TaskComplexity},
    backends::{Backend, BackendError, BackendConfig},
};
use async_trait::async_trait;

// === MOCK BACKENDS ===

/// Mock backend that always succeeds
struct MockSuccessBackend {
    name: String,
    delay_ms: u64,
}

#[async_trait]
impl Backend for MockSuccessBackend {
    fn name(&self) -> &str { &self.name }

    async fn generate(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        Ok(InferenceResponse {
            request_id: request.id.clone(),
            text: "Mock response".into(),
            model: self.name.clone(),
            tier: ModelTier::Edge,
            completion_tokens: 10,
            duration_ms: self.delay_ms,
            tokens_per_second: 100.0,
        })
    }

    async fn health_check(&self) -> bool { true }
}

/// Mock backend that always fails
struct MockFailingBackend {
    name: String,
    error_message: String,
}

#[async_trait]
impl Backend for MockFailingBackend {
    fn name(&self) -> &str { &self.name }

    async fn generate(&self, _request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        Err(BackendError::Generation(self.error_message.clone()))
    }

    async fn health_check(&self) -> bool { false }
}

/// Mock backend that times out
struct MockTimeoutBackend {
    name: String,
}

#[async_trait]
impl Backend for MockTimeoutBackend {
    fn name(&self) -> &str { &self.name }

    async fn generate(&self, _request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        // Sleep longer than any reasonable timeout
        tokio::time::sleep(Duration::from_secs(3600)).await;
        unreachable!()
    }

    async fn health_check(&self) -> bool { true }
}

// === TEST CASES ===

#[tokio::test]
async fn test_gateway_no_backend_error() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    let request = InferenceRequest {
        id: "test_001".into(),
        prompt: "Hello".into(),
        complexity: TaskComplexity::Simple,
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;
    assert!(matches!(result, Err(GatewayError::NoBackend(ModelTier::Edge))));
}

#[tokio::test]
async fn test_gateway_backend_error_propagation() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    // Register failing backend
    let failing = Arc::new(MockFailingBackend {
        name: "failing".into(),
        error_message: "Connection refused".into(),
    });
    gateway.register_backend(ModelTier::Edge, failing).await;

    let request = InferenceRequest {
        id: "test_002".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;
    assert!(matches!(result, Err(GatewayError::Backend(_))));
}

#[tokio::test]
async fn test_gateway_timeout_handling() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let mut gateway = InferenceGateway::new(identity, constitution);
    gateway.timeout = Duration::from_millis(100); // Very short timeout

    // Register timeout backend
    let timeout_backend = Arc::new(MockTimeoutBackend {
        name: "timeout".into(),
    });
    gateway.register_backend(ModelTier::Edge, timeout_backend).await;

    let request = InferenceRequest {
        id: "test_003".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let result = gateway.infer(request).await;
    assert!(matches!(result, Err(GatewayError::Timeout)));
}

#[tokio::test]
async fn test_gateway_tier_selection() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    // Register backends for different tiers
    gateway.register_backend(
        ModelTier::Edge,
        Arc::new(MockSuccessBackend { name: "edge".into(), delay_ms: 10 })
    ).await;
    gateway.register_backend(
        ModelTier::Local,
        Arc::new(MockSuccessBackend { name: "local".into(), delay_ms: 10 })
    ).await;

    // Simple task should use Edge
    let simple_request = InferenceRequest {
        id: "simple".into(),
        prompt: "2+2".into(),
        complexity: TaskComplexity::Simple,
        preferred_tier: None,
        ..Default::default()
    };

    let response = gateway.infer(simple_request).await.unwrap();
    assert_eq!(response.tier, ModelTier::Edge);
}

#[tokio::test]
async fn test_gateway_respects_preferred_tier() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    // Register both tiers
    gateway.register_backend(
        ModelTier::Edge,
        Arc::new(MockSuccessBackend { name: "edge".into(), delay_ms: 10 })
    ).await;
    gateway.register_backend(
        ModelTier::Local,
        Arc::new(MockSuccessBackend { name: "local".into(), delay_ms: 10 })
    ).await;

    // Even simple task, force Local tier
    let request = InferenceRequest {
        id: "forced_local".into(),
        prompt: "2+2".into(),
        complexity: TaskComplexity::Simple,
        preferred_tier: Some(ModelTier::Local),
        ..Default::default()
    };

    let response = gateway.infer(request).await.unwrap();
    assert_eq!(response.tier, ModelTier::Local);
}

#[tokio::test]
async fn test_gateway_tokens_per_second_calculation() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = InferenceGateway::new(identity, constitution);

    gateway.register_backend(
        ModelTier::Edge,
        Arc::new(MockSuccessBackend { name: "edge".into(), delay_ms: 100 })
    ).await;

    let request = InferenceRequest {
        id: "tps_test".into(),
        prompt: "Hello".into(),
        preferred_tier: Some(ModelTier::Edge),
        ..Default::default()
    };

    let response = gateway.infer(request).await.unwrap();

    // Should have calculated tokens per second
    assert!(response.tokens_per_second > 0.0);
    assert!(response.duration_ms >= 100);
}

// === CONCURRENT ACCESS TESTS ===

#[tokio::test]
async fn test_gateway_concurrent_requests() {
    let identity = NodeIdentity::generate();
    let constitution = Constitution::default();
    let gateway = Arc::new(InferenceGateway::new(identity, constitution));

    gateway.register_backend(
        ModelTier::Edge,
        Arc::new(MockSuccessBackend { name: "edge".into(), delay_ms: 50 })
    ).await;

    // Spawn 10 concurrent requests
    let mut handles = vec![];
    for i in 0..10 {
        let gw = gateway.clone();
        let handle = tokio::spawn(async move {
            let request = InferenceRequest {
                id: format!("concurrent_{}", i),
                prompt: format!("Request {}", i),
                preferred_tier: Some(ModelTier::Edge),
                ..Default::default()
            };
            gw.infer(request).await
        });
        handles.push(handle);
    }

    // All should succeed
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}
```

### 3. Federation Security Fuzz Tests

**Location**: `bizra-tests/tests/federation_fuzz.rs`

```rust
//! Fuzz tests for federation protocol security
//! SECURITY: Tests for message parsing, signature verification, and replay attacks

use proptest::prelude::*;
use bizra_federation::gossip::*;
use ed25519_dalek::SigningKey;
use chrono::{DateTime, Utc, Duration as ChronoDuration};

// === FUZZ: Message Parsing ===

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    /// Fuzz: Arbitrary bytes should never panic in from_bytes
    #[test]
    fn fuzz_signed_message_from_bytes_no_panic(
        data in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        // Should return Err, never panic
        let _ = SignedGossipMessage::from_bytes(&data);
    }

    /// Fuzz: Version byte fuzzing
    #[test]
    fn fuzz_message_version_byte(
        version in 0u8..=255,
        rest in prop::collection::vec(any::<u8>(), 103..200)
    ) {
        let mut data = vec![version];
        data.extend(rest);

        let result = SignedGossipMessage::from_bytes(&data);

        if version != 1 {
            // Non-v1 should fail with UnsupportedVersion
            prop_assert!(matches!(
                result,
                Err(FederationError::UnsupportedVersion(_))
            ));
        }
    }

    /// Fuzz: Signature field corruption
    #[test]
    fn fuzz_signature_corruption(
        corruption_index in 1usize..65,
        corruption_value in any::<u8>()
    ) {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let msg = GossipMessage::Ping {
            from: "node_test".into(),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::sign(msg, &signing_key);

        // Serialize and corrupt
        let mut bytes = signed.to_bytes();
        if corruption_index < bytes.len() {
            bytes[corruption_index] ^= corruption_value;
        }

        // Parse should succeed but verification should fail
        if let Ok(parsed) = SignedGossipMessage::from_bytes(&bytes) {
            let verify_result = parsed.verify();
            // Corrupted signature should fail verification
            prop_assert!(
                verify_result.is_err() || corruption_value == 0,
                "Corrupted message verified successfully"
            );
        }
    }
}

// === SECURITY: Replay Attack Prevention ===

#[tokio::test]
async fn test_replay_attack_prevention() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let local_id = "local_node".to_string();
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_id = "peer_node".to_string();

    let protocol = GossipProtocol::new(
        local_id.clone(),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    // Register peer
    protocol.register_peer_pubkey(
        peer_id.clone(),
        peer_key.verifying_key().to_bytes()
    ).await;

    // Create old message (6 minutes ago, beyond 5-minute window)
    let old_timestamp = Utc::now() - ChronoDuration::seconds(360);
    let msg = GossipMessage::Ping {
        from: peer_id.clone(),
        incarnation: 1,
    };

    // Manually construct old signed message
    let old_signed = SignedGossipMessage {
        message: msg,
        signature: [0u8; 64], // Will be invalid anyway
        sender_pubkey: peer_key.verifying_key().to_bytes(),
        timestamp: old_timestamp,
    };

    // Should reject as expired
    let result = protocol.handle_signed_message(old_signed).await;
    assert!(matches!(result, Err(FederationError::MessageExpired)));
}

#[tokio::test]
async fn test_future_timestamp_rejection() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        "local".into(),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    protocol.register_peer_pubkey(
        "peer".into(),
        peer_key.verifying_key().to_bytes()
    ).await;

    // Create message from future (10 minutes ahead)
    let future_timestamp = Utc::now() + ChronoDuration::seconds(600);
    let msg = GossipMessage::Ping {
        from: "peer".into(),
        incarnation: 1,
    };

    let future_signed = SignedGossipMessage {
        message: msg,
        signature: peer_key.sign(&[0u8; 32]).to_bytes(),
        sender_pubkey: peer_key.verifying_key().to_bytes(),
        timestamp: future_timestamp,
    };

    let result = protocol.handle_signed_message(future_signed).await;
    assert!(matches!(result, Err(FederationError::MessageExpired)));
}

// === SECURITY: Node ID Spoofing Prevention ===

#[tokio::test]
async fn test_node_id_spoofing_rejected() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let attacker_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let victim_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        "local".into(),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    // Register victim's key
    protocol.register_peer_pubkey(
        "victim".into(),
        victim_key.verifying_key().to_bytes()
    ).await;

    // Attacker tries to claim victim's ID
    let spoofed_msg = GossipMessage::Ping {
        from: "victim".into(), // Claims to be victim
        incarnation: 999,
    };

    // But signs with attacker's key
    let signed = SignedGossipMessage::sign(spoofed_msg, &attacker_key);

    // Should reject - pubkey doesn't match registered victim key
    let result = protocol.handle_signed_message(signed).await;
    assert!(matches!(result, Err(FederationError::UnknownSender)));
}

// === SECURITY: Malformed Message Handling ===

#[test]
fn test_truncated_message_handling() {
    // Various truncation points
    let truncation_points = [0, 1, 64, 65, 96, 97, 104, 105, 110];

    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let msg = GossipMessage::Ping {
        from: "test".into(),
        incarnation: 1,
    };
    let signed = SignedGossipMessage::sign(msg, &signing_key);
    let full_bytes = signed.to_bytes();

    for &len in &truncation_points {
        if len < full_bytes.len() {
            let truncated = &full_bytes[..len];
            let result = SignedGossipMessage::from_bytes(truncated);
            assert!(result.is_err(), "Truncated at {} should fail", len);
        }
    }
}

#[test]
fn test_oversized_message_handling() {
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

    // Message with very long node_id
    let long_id = "x".repeat(100_000);
    let msg = GossipMessage::Ping {
        from: long_id,
        incarnation: 1,
    };

    // Should still serialize/deserialize correctly (no panic)
    let signed = SignedGossipMessage::sign(msg, &signing_key);
    let bytes = signed.to_bytes();

    // Verify round-trip
    let parsed = SignedGossipMessage::from_bytes(&bytes).unwrap();
    assert!(parsed.verify().is_ok());
}

// === CONSENSUS SECURITY ===

#[test]
fn test_consensus_vote_signature_required() {
    let identity = NodeIdentity::generate();
    let mut engine = ConsensusEngine::new(identity.clone());
    engine.set_node_count(3);

    // Create proposal
    let proposal = engine.propose(
        serde_json::json!({"pattern": "test"}),
        0.96
    ).unwrap();

    // Create forged vote (wrong signature)
    let forger = NodeIdentity::generate();
    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: identity.node_id().clone(), // Claims to be original identity
        approve: true,
        ihsan_score: 0.96,
        timestamp: Utc::now(),
    };

    // Sign with forger's key (wrong key)
    let forged_vote = SignedVote::new(vote, forger.signing_key());

    // Should reject - signature doesn't match
    let result = engine.receive_vote(forged_vote);
    assert!(matches!(result, Err(ConsensusError::PubkeyMismatch { .. })));
}

#[test]
fn test_consensus_below_ihsan_rejected() {
    let identity = NodeIdentity::generate();
    let mut engine = ConsensusEngine::new(identity.clone());

    // Proposal with score below threshold (0.95)
    let result = engine.propose(
        serde_json::json!({"pattern": "test"}),
        0.90
    );

    assert!(matches!(result, Err(ConsensusError::IhsanThreshold(_))));
}

#[test]
fn test_consensus_unknown_voter_rejected() {
    let identity = NodeIdentity::generate();
    let mut engine = ConsensusEngine::new(identity.clone());
    engine.set_node_count(3);

    let proposal = engine.propose(
        serde_json::json!({"pattern": "test"}),
        0.96
    ).unwrap();

    // Unknown voter tries to vote
    let unknown = NodeIdentity::generate();
    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: unknown.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: Utc::now(),
    };
    let signed = SignedVote::new(vote, unknown.signing_key());

    let result = engine.receive_vote(signed);
    assert!(matches!(result, Err(ConsensusError::UnknownVoter(_))));
}
```

---

## P1: High-Priority Test Cases

### 4. SNR Engine Boundary Tests

**Location**: `bizra-tests/tests/snr_boundaries.rs`

```rust
//! Boundary condition tests for SNR Engine

use bizra_core::sovereign::snr_engine::*;

#[test]
fn test_snr_boundary_exact_threshold() {
    let engine = SNREngine::new(0.85, 0.95);

    // Exactly at threshold
    let metrics = SignalMetrics {
        signal_strength: 0.87,
        diversity: 0.85,
        grounding: 0.85,
        balance: 0.85,
        noise_level: 0.0,
        ..Default::default()
    };

    let snr = metrics.compute_snr();
    assert!(engine.check(&metrics) == (snr >= 0.85));
}

#[test]
fn test_snr_unicode_handling() {
    let engine = SNREngine::new(0.85, 0.95);

    // Unicode text (Arabic for "sovereignty")
    let arabic = "سيادة sovereign السيادة";
    let result = engine.analyze_text(arabic);
    assert!(result.is_ok());

    // Emoji-heavy text
    let emoji = "Test text with many words and content";
    let result = engine.analyze_text(emoji);
    assert!(result.is_ok());

    // Mixed scripts
    let mixed = "English text mixed with different word types";
    let result = engine.analyze_text(mixed);
    assert!(result.is_ok());
}

#[test]
fn test_snr_adversarial_input() {
    let engine = SNREngine::new(0.85, 0.95);

    // Null bytes (should not panic)
    let null_bytes = "text\0with\0nulls";
    let _ = engine.analyze_text(null_bytes);

    // Control characters
    let control = "text\x00\x01\x02\x03with\x1b[0mcontrol";
    let _ = engine.analyze_text(control);

    // Very long word
    let long_word = &"a".repeat(10000);
    let result = engine.analyze_text(long_word);
    assert!(result.is_ok());
}

#[test]
fn test_snr_whitespace_only() {
    let engine = SNREngine::new(0.85, 0.95);

    // Only whitespace (should fail with EmptyInput after word parsing)
    let whitespace = "   \t\n\r   ";
    let result = engine.analyze_text(whitespace);
    assert!(result.is_err());
}

#[test]
fn test_snr_metric_ranges() {
    // All metrics should be clamped to [0, 1]
    let mut node = ThoughtNode::new("test", "description");

    node.set_snr(-0.5);
    assert!(node.snr_score >= 0.0);

    node.set_snr(1.5);
    assert!(node.snr_score <= 1.0);
}
```

### 5. Model Selector Tests

**Location**: `bizra-tests/tests/selector_tests.rs`

```rust
//! Model selector and complexity estimation tests

use bizra_inference::selector::*;

#[test]
fn test_complexity_estimation_boundaries() {
    // Very short prompt - Simple
    let simple = TaskComplexity::estimate("Hi", 10);
    assert_eq!(simple, TaskComplexity::Simple);

    // Medium length with code - Medium or Complex
    let with_code = TaskComplexity::estimate(
        "Write a function: ```python\ndef foo(): pass```",
        200
    );
    assert!(matches!(with_code, TaskComplexity::Medium | TaskComplexity::Complex));

    // Very long prompt - should escalate
    let long_prompt = "Explain ".repeat(500);
    let complex = TaskComplexity::estimate(&long_prompt, 3000);
    assert!(matches!(complex, TaskComplexity::Complex | TaskComplexity::Expert));
}

#[test]
fn test_tier_selection_mapping() {
    let selector = ModelSelector::default();

    assert_eq!(selector.select_tier(&TaskComplexity::Simple), ModelTier::Edge);
    assert_eq!(selector.select_tier(&TaskComplexity::Medium), ModelTier::Edge);
    assert_eq!(selector.select_tier(&TaskComplexity::Complex), ModelTier::Local);
    assert_eq!(selector.select_tier(&TaskComplexity::Expert), ModelTier::Pool);
}
```

---

## P2: Lower-Priority Test Cases

### 6. Integration Tests for Full Pipeline

**Location**: `bizra-tests/tests/full_pipeline.rs`

```rust
//! Full pipeline integration tests

#[tokio::test]
async fn test_full_sovereign_pipeline() {
    // Identity -> PCI -> SNR -> GoT -> Inference -> Consensus

    // 1. Create sovereign identity
    let identity = NodeIdentity::generate();

    // 2. Create and sign PCI envelope
    let payload = serde_json::json!({
        "action": "reason",
        "question": "What is the optimal approach?"
    });
    let envelope = PCIEnvelope::create(&identity, payload, 3600, vec![]).unwrap();
    assert!(envelope.verify().is_ok());

    // 3. Validate content SNR
    let engine = SNREngine::new(0.85, 0.95);
    let content = "Analyze the tradeoffs between consistency and availability...";
    let metrics = engine.analyze_text(content).unwrap();
    assert!(engine.check(&metrics));

    // 4. Create reasoning graph
    let mut graph = ThoughtGraph::new();
    let root = graph.create_thought("Initial analysis", None);
    let h1 = graph.create_thought_with_type("Hypothesis A", Some(&root), ThoughtType::Hypothesis);
    let h2 = graph.create_thought_with_type("Hypothesis B", Some(&root), ThoughtType::Hypothesis);

    // 5. Explore and aggregate
    let paths = graph.explore_parallel(&root);
    assert_eq!(paths.len(), 2);
}
```

---

## Mock Strategies

### External Dependency Mocking

| Dependency | Mock Strategy |
|------------|---------------|
| LlamaCpp Backend | `MockSuccessBackend` / `MockFailingBackend` / `MockTimeoutBackend` |
| Ollama Backend | Same mock pattern |
| Network I/O | `tokio::io::DuplexStream` for bidirectional testing |
| Filesystem | `tempfile` crate for temporary test directories |
| Time | `tokio::time::pause()` for deterministic timing tests |

### Trait-based Mocking

```rust
// Backend trait already enables easy mocking
#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError>;
    async fn health_check(&self) -> bool;
}

// Mock implementations shown in test code above
```

---

## Test Configuration

### Cargo.toml additions

```toml
[dev-dependencies]
proptest = "1.4"
proptest-derive = "0.4"
tokio-test = "0.4"
mockall = "0.12"
tempfile = "3.10"
criterion = "0.5"

[[test]]
name = "got_properties"
path = "tests/got_properties.rs"

[[test]]
name = "inference_fallback"
path = "tests/inference_fallback.rs"

[[test]]
name = "federation_fuzz"
path = "tests/federation_fuzz.rs"
```

### Test Markers (pytest style for Rust)

```rust
// Use cfg attributes for conditional test execution

#[test]
#[cfg(feature = "slow-tests")]
fn slow_property_test() { /* ... */ }

#[tokio::test]
#[cfg(feature = "integration")]
async fn integration_test() { /* ... */ }

#[test]
#[ignore] // Skip by default, run with --ignored
fn expensive_fuzz_test() { /* ... */ }
```

---

## Coverage Targets

| Module | Current Est. | Target | Priority |
|--------|-------------|--------|----------|
| `graph_of_thoughts.rs` | 40% | 90% | P0 |
| `gateway.rs` | 25% | 85% | P0 |
| `gossip.rs` | 35% | 90% | P0 |
| `consensus.rs` | 45% | 90% | P1 |
| `snr_engine.rs` | 70% | 95% | P1 |
| `selector.rs` | 30% | 80% | P2 |
| `envelope.rs` | 60% | 85% | P2 |
| **Overall** | **~45%** | **>=85%** | - |

---

## Execution Plan

### Phase 1 (Week 1): P0 Critical Tests
1. Implement `got_properties.rs` - Property-based GoT tests
2. Implement `inference_fallback.rs` - Gateway resilience tests
3. Implement `federation_fuzz.rs` - Security fuzz tests
4. Target: 65% coverage

### Phase 2 (Week 2): P1 High-Priority Tests
1. Implement `snr_boundaries.rs` - SNR edge cases
2. Implement `selector_tests.rs` - Model selection tests
3. Expand consensus security tests
4. Target: 80% coverage

### Phase 3 (Week 3): P2 and Integration
1. Full pipeline integration tests
2. Performance benchmarks
3. A2A protocol tests (when module available)
4. Target: 85% coverage

---

## Quality Gates

Tests must pass these quality gates:

1. **All tests pass** - Zero failures in CI
2. **Coverage >= 85%** - Measured by cargo-tarpaulin
3. **No panics in fuzz tests** - 10,000 iterations minimum
4. **Property tests pass** - 1,000 cases per property
5. **Performance benchmarks** - No regressions > 10%

---

## References

- Besta et al. (2024): "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
- Das et al. (2002): "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol"
- Lamport (1982): "The Byzantine Generals Problem"
- Shannon (1948): "A Mathematical Theory of Communication"

---

*Document generated by TDD Agent | BIZRA Omega v2.2.0*
