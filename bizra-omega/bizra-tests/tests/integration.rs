//! Integration Tests for BIZRA Omega
//!
//! These tests verify the complete flow across all modules,
//! including the sovereign orchestration layer.

use bizra_core::{
    NodeIdentity, Constitution, PCIEnvelope, GateChain, GateContext,
    domain_separated_digest, IHSAN_THRESHOLD, SNR_THRESHOLD,
    // Sovereign module imports
    sovereign::{
        OmegaEngine, OmegaConfig, OmegaMetrics, CircuitState,
        SNREngine, SNRConfig, SignalMetrics,
        ThoughtGraph, ThoughtNode, ReasoningPath,
        SovereignError, SovereignResult,
        GiantRegistry,
    },
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

// ═══════════════════════════════════════════════════════════════════════════════
// SOVEREIGN INTEGRATION TESTS
// Standing on Giants: Shannon (SNR) • Besta (GoT) • Lamport (Consensus)
// ═══════════════════════════════════════════════════════════════════════════════

/// Test OmegaEngine full workflow with production configuration
#[tokio::test]
async fn test_omega_engine_production_workflow() {
    let engine = OmegaEngine::production();

    // Execute multiple operations
    for i in 0..5 {
        let result = engine.execute(|| i * 2).await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.value, i * 2);
        assert!(!res.operation_id.is_empty());
    }

    // Verify metrics
    let metrics = engine.metrics().await;
    assert_eq!(metrics.total_operations, 5);
    assert_eq!(metrics.successful_operations, 5);
    assert_eq!(metrics.failed_operations, 0);
    assert_eq!(metrics.circuit_state, CircuitState::Closed);
}

/// Test OmegaEngine with identity binding
#[tokio::test]
async fn test_omega_with_identity() {
    let identity = NodeIdentity::generate();
    let engine = OmegaEngine::new(OmegaConfig::default())
        .with_identity(identity)
        .await;

    let result = engine.execute(|| "sovereign operation").await;
    assert!(result.is_ok());

    let res = result.unwrap();
    assert_eq!(res.value, "sovereign operation");
}

/// Test SNR Engine integration with Omega
#[tokio::test]
async fn test_snr_engine_omega_integration() {
    let config = OmegaConfig::development();
    let engine = OmegaEngine::new(config);

    // High-quality content should pass
    let good_content = "The quantum entanglement research demonstrates \
                        significant advancements in computational efficiency \
                        through novel algorithmic approaches validated by data.";

    let result = engine.validate_with_reasoning(good_content).await;
    assert!(result.is_ok(), "High-quality content should pass validation");

    let (metrics, path) = result.unwrap();
    assert!(metrics.compute_snr() > 0.5);
    assert!(!path.thoughts.is_empty());
}

/// Test SNR Engine rejects low-quality content
#[tokio::test]
async fn test_snr_rejects_noise() {
    let config = OmegaConfig::production();
    let engine = OmegaEngine::new(config);

    // Low-quality repetitive content
    let noisy_content = "um um um uh uh uh like like like yeah yeah";

    let snr_engine = engine.snr_engine();
    let result = snr_engine.analyze_text(noisy_content);

    assert!(result.is_ok());
    let metrics = result.unwrap();
    // Repetitive content should have low diversity
    assert!(metrics.diversity < 0.5, "Repetitive content should have low diversity");
}

/// Test Graph-of-Thoughts reasoning path creation
#[tokio::test]
async fn test_got_reasoning_path() {
    let graph = ThoughtGraph::new();
    let path = graph.create_path("test_reasoning");

    // Add thoughts
    let mut path = path;
    path.add_thought(ThoughtNode::new("hypothesis", "Form initial hypothesis"));
    path.add_thought(ThoughtNode::new("analysis", "Analyze evidence"));
    path.add_thought(ThoughtNode::new("synthesis", "Synthesize conclusions"));

    assert_eq!(path.thoughts.len(), 3);
    assert!(path.thoughts.iter().any(|t| t.id == "hypothesis"));
    assert!(path.thoughts.iter().any(|t| t.id == "analysis"));
    assert!(path.thoughts.iter().any(|t| t.id == "synthesis"));
}

/// Test GoT parallel exploration
#[tokio::test]
async fn test_got_parallel_exploration() {
    let graph = ThoughtGraph::new();

    // Create multiple reasoning paths
    let path1 = graph.create_path("approach_a");
    let path2 = graph.create_path("approach_b");
    let path3 = graph.create_path("approach_c");

    // All paths should be independent
    assert_ne!(path1.id, path2.id);
    assert_ne!(path2.id, path3.id);
}

/// Test error propagation through sovereign stack
#[tokio::test]
async fn test_sovereign_error_propagation() {
    // Test SNR threshold error
    let err = SovereignError::SNRBelowThreshold {
        actual: 0.72,
        threshold: 0.85,
    };
    assert!(err.is_quality_violation());
    assert!(!err.is_recoverable());

    // Test Ihsan violation error
    let err = SovereignError::IhsanViolation {
        actual: 0.90,
        threshold: 0.95,
    };
    assert!(err.is_quality_violation());
    assert_eq!(err.severity(), 0.8);

    // Test recoverable error
    let err = SovereignError::Timeout { duration_ms: 5000 };
    assert!(err.is_recoverable());
    assert!(!err.is_quality_violation());
}

/// Test circuit breaker behavior
#[tokio::test]
async fn test_circuit_breaker_integration() {
    let config = OmegaConfig {
        circuit_breaker_threshold: 2,
        circuit_breaker_recovery_ms: 100,
        ..OmegaConfig::default()
    };
    let engine = OmegaEngine::new(config);

    // Circuit should start closed
    let metrics = engine.metrics().await;
    assert_eq!(metrics.circuit_state, CircuitState::Closed);

    // Successful operations keep it closed
    engine.execute(|| 1 + 1).await.unwrap();
    let metrics = engine.metrics().await;
    assert_eq!(metrics.circuit_state, CircuitState::Closed);
}

/// Test metrics accumulation
#[tokio::test]
async fn test_omega_metrics_accumulation() {
    let engine = OmegaEngine::new(OmegaConfig::development());

    // Execute operations
    for _ in 0..10 {
        engine.execute(|| {
            // Simulate some work
            let mut sum = 0u64;
            for i in 0..1000 {
                sum += i;
            }
            sum
        }).await.unwrap();
    }

    let metrics = engine.metrics().await;

    // Verify accumulation
    assert_eq!(metrics.total_operations, 10);
    assert_eq!(metrics.successful_operations, 10);
    assert!(metrics.avg_latency_us > 0, "Average latency should be recorded");
}

/// Test health check integration
#[tokio::test]
async fn test_omega_health_check() {
    let engine = OmegaEngine::new(OmegaConfig::default());

    // Fresh engine should be healthy
    let health = engine.health_check().await;
    assert!(health.is_ok());

    // Execute some operations
    for _ in 0..5 {
        engine.execute(|| "healthy").await.unwrap();
    }

    // Should still be healthy
    let health = engine.health_check().await;
    assert!(health.is_ok());
}

/// Test Giants registry attribution
#[test]
fn test_giants_attribution_integration() {
    let registry = GiantRegistry::new();
    let attribution = registry.attribution();

    // Verify key giants are attributed
    assert!(attribution.contains("Shannon"));
    assert!(attribution.contains("Lamport"));
    assert!(attribution.contains("Besta"));
    assert!(attribution.contains("Vaswani"));
    assert!(attribution.contains("Torvalds"));

    // Test domain queries
    let information = registry.by_domain("Information Theory");
    assert!(!information.is_empty());

    let distributed = registry.by_domain("Distributed Systems");
    assert!(!distributed.is_empty());
}

/// Test SNR configuration presets
#[test]
fn test_snr_config_presets() {
    let production = SNRConfig::default();
    let edge = SNRConfig::edge();

    // Edge should be more lenient
    assert!(edge.snr_floor <= production.snr_floor);
    assert!(edge.max_input_size < production.max_input_size);
}

/// Test OmegaConfig presets
#[test]
fn test_omega_config_presets() {
    let prod = OmegaConfig::production();
    let dev = OmegaConfig::development();
    let edge = OmegaConfig::edge();

    // Production has telemetry enabled
    assert!(prod.enable_telemetry);

    // Development is more lenient
    assert!(!dev.enable_telemetry);
    assert!(dev.snr_floor < prod.snr_floor);

    // Edge has reduced features
    assert!(!edge.enable_got);
    assert!(!edge.enable_adaptive);
    assert!(edge.max_concurrency < prod.max_concurrency);
}

/// Test full sovereign pipeline: Identity → PCI → SNR → Omega
#[tokio::test]
async fn test_full_sovereign_pipeline() {
    // 1. Create identity
    let identity = NodeIdentity::generate();

    // 2. Create Omega engine with identity
    let engine = OmegaEngine::new(OmegaConfig::development())
        .with_identity(identity)
        .await;

    // 3. Create validated content
    let content = "The distributed consensus algorithm demonstrates \
                   Byzantine fault tolerance through rigorous verification.";

    // 4. Validate through SNR
    let snr_result = engine.snr_engine().analyze_text(content);
    assert!(snr_result.is_ok());
    let metrics = snr_result.unwrap();
    assert!(metrics.compute_snr() > 0.5);

    // 5. Execute through Omega
    let omega_result = engine.execute(|| {
        // Simulate sovereign operation
        "operation_complete"
    }).await;

    assert!(omega_result.is_ok());
    let res = omega_result.unwrap();
    assert_eq!(res.value, "operation_complete");

    // 6. Verify attribution
    let attribution = engine.print_attribution();
    assert!(attribution.contains("Shannon"));
}

/// Test concurrent operations through Omega
#[tokio::test]
async fn test_omega_concurrent_operations() {
    use std::sync::Arc;

    let engine = Arc::new(OmegaEngine::new(OmegaConfig::default()));

    // Spawn multiple concurrent operations
    let mut handles = vec![];
    for i in 0..10 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            engine_clone.execute(|| i * 2).await
        });
        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // Verify metrics
    let metrics = engine.metrics().await;
    assert_eq!(metrics.total_operations, 10);
    assert_eq!(metrics.successful_operations, 10);
}

/// Test SNR text analysis with various content types
#[test]
fn test_snr_content_type_analysis() {
    let engine = SNREngine::with_config(SNRConfig::default());

    // Technical content (should have high SNR)
    let technical = "The cryptographic hash function provides \
                     collision resistance through algorithmic complexity.";
    let tech_result = engine.analyze_text(technical);
    assert!(tech_result.is_ok());
    let tech_snr = tech_result.unwrap().compute_snr();

    // Filler content (should have lower SNR)
    let filler = "So basically like you know what I mean right \
                  it's kind of sort of maybe probably I guess.";
    let filler_result = engine.analyze_text(filler);
    assert!(filler_result.is_ok());
    let filler_snr = filler_result.unwrap().compute_snr();

    // Technical should score higher than filler
    assert!(tech_snr > filler_snr,
            "Technical content ({}) should score higher than filler ({})",
            tech_snr, filler_snr);
}

/// Test input validation bounds
#[test]
fn test_input_validation_bounds() {
    let config = SNRConfig {
        max_input_size: 100,
        ..SNRConfig::default()
    };
    let engine = SNREngine::with_config(config);

    // Small input should pass
    let small = "Hello world";
    assert!(engine.analyze_text(small).is_ok());

    // Large input should be rejected
    let large = "x".repeat(200);
    let result = engine.analyze_text(&large);
    assert!(result.is_err());

    match result.unwrap_err() {
        SovereignError::InputTooLarge { size, max_size } => {
            assert_eq!(size, 200);
            assert_eq!(max_size, 100);
        }
        _ => panic!("Expected InputTooLarge error"),
    }
}
