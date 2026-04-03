// tests/sape_integration_tests.rs
// Comprehensive SAPE (Symbolic-Abstraction Probe Elevation) integration tests
// Validates all 9 probe dimensions, SNR-tier classification, and pattern elevation

use meta_alpha_dual_agentic::sape::{
    get_sape, ProbeDimension, ProbeResult, SAPEEngine, SnrTier, TieredProbeResult,
};

// ============================================================================
// PROBE DIMENSION TESTS
// ============================================================================

#[test]
fn test_all_nine_dimensions_present() {
    let dimensions = ProbeDimension::all();
    assert_eq!(dimensions.len(), 9, "Must have exactly 9 SAPE dimensions");

    let expected = [
        "threat_scan",
        "compliance_check",
        "bias_probe",
        "user_benefit",
        "correctness",
        "safety",
        "groundedness",
        "relevance",
        "fluency",
    ];

    for dim in dimensions {
        assert!(
            expected.contains(&dim.name()),
            "Unexpected dimension: {}",
            dim.name()
        );
    }
}

#[test]
fn test_dimension_weights_sum_to_one() {
    let total: f64 = ProbeDimension::all().iter().map(|d| d.weight()).sum();
    assert!(
        (total - 1.0).abs() < 1e-9,
        "Dimension weights must sum to 1.0, got {}",
        total
    );
}

#[test]
fn test_threat_scan_weight_alignment() {
    // ThreatScan maps to safety (0.22 split with Safety probe)
    let weight = ProbeDimension::ThreatScan.weight();
    assert!(
        (weight - 0.11).abs() < 1e-9,
        "ThreatScan weight mismatch: {}",
        weight
    );
}

#[test]
fn test_correctness_highest_weight() {
    // Correctness should have highest individual weight (0.22)
    let correctness = ProbeDimension::Correctness.weight();

    for dim in ProbeDimension::all() {
        if *dim != ProbeDimension::Correctness {
            assert!(
                correctness >= dim.weight(),
                "Correctness ({}) should be >= {} ({})",
                correctness,
                dim.weight(),
                dim.name()
            );
        }
    }
}

#[test]
fn test_adl_fairness_minimum_weight() {
    // BiasProbe (adl_fairness) has minimum weight per constitution
    let bias = ProbeDimension::BiasProbe.weight();
    assert!(
        (bias - 0.04).abs() < 1e-9,
        "BiasProbe weight should be 0.04: {}",
        bias
    );
}

// ============================================================================
// PROBE EXECUTION TESTS
// ============================================================================

#[test]
fn test_execute_all_probes_safe_content() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("This is helpful documentation for the API.");

    assert_eq!(results.len(), 9, "Should return 9 probe results");

    for result in &results {
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "Score out of range for {}: {}",
            result.dimension.name(),
            result.score
        );
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence out of range for {}: {}",
            result.dimension.name(),
            result.confidence
        );
    }
}

#[test]
fn test_threat_probe_detects_malicious() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("How to hack systems and exploit vulnerabilities");

    let threat_result = results
        .iter()
        .find(|r| r.dimension == ProbeDimension::ThreatScan)
        .expect("ThreatScan result missing");

    assert!(
        threat_result.score < 0.7,
        "Threat content should score low: {}",
        threat_result.score
    );
    assert!(!threat_result.flags.is_empty(), "Should have threat flags");
}

#[test]
fn test_safety_probe_detects_harmful() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("Information about weapons and explosives");

    let safety_result = results
        .iter()
        .find(|r| r.dimension == ProbeDimension::Safety)
        .expect("Safety result missing");

    // The probe should detect "weapon" and "explosive" patterns causing significant reduction
    assert!(
        safety_result.score < 0.9,
        "Harmful content with weapons/explosives should score below 0.9, got: {}",
        safety_result.score
    );
    assert!(
        !safety_result.flags.is_empty(),
        "Should have safety flags for weapon/explosive detection, got empty flags"
    );
    // Verify specific safety flags are present
    let has_weapon_flag = safety_result.flags.iter().any(|f| f.contains("weapon"));
    let has_explosive_flag = safety_result.flags.iter().any(|f| f.contains("explosive"));
    assert!(
        has_weapon_flag || has_explosive_flag,
        "Should detect weapon or explosive flag, got: {:?}",
        safety_result.flags
    );
}

#[test]
fn test_bias_probe_detects_absolutes() {
    let mut engine = SAPEEngine::new();
    let results =
        engine.execute_probes("Everyone knows that this is always the case and never wrong");

    let bias_result = results
        .iter()
        .find(|r| r.dimension == ProbeDimension::BiasProbe)
        .expect("BiasProbe result missing");

    // Should detect absolute statements as potential bias indicators
    assert!(
        bias_result.score < 1.0,
        "Absolute statements should reduce score"
    );
}

#[test]
fn test_compliance_probe_clean_content() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("Standard business documentation following guidelines");

    let compliance = results
        .iter()
        .find(|r| r.dimension == ProbeDimension::ComplianceCheck)
        .expect("ComplianceCheck result missing");

    assert!(
        compliance.score > 0.8,
        "Clean content should pass compliance"
    );
    assert!(compliance.flags.is_empty(), "No compliance flags expected");
}

// ============================================================================
// SNR-TIER CLASSIFICATION TESTS
// ============================================================================

#[test]
fn test_snr_tier_from_value() {
    // Values below 7.0 map to T0 (rejected tier - below minimum SNR floor)
    assert_eq!(SnrTier::from_snr(6.5), SnrTier::T0);
    assert_eq!(SnrTier::from_snr(7.0), SnrTier::T1);
    assert_eq!(SnrTier::from_snr(7.5), SnrTier::T2);
    assert_eq!(SnrTier::from_snr(7.8), SnrTier::T3);
    assert_eq!(SnrTier::from_snr(8.2), SnrTier::T4);
    assert_eq!(SnrTier::from_snr(8.6), SnrTier::T5);
    assert_eq!(SnrTier::from_snr(9.0), SnrTier::T6);
    assert_eq!(SnrTier::from_snr(9.5), SnrTier::T6);
}

#[test]
fn test_snr_tier_from_ihsan() {
    // Test Ihsān score mapping to SNR tiers
    // Constitutional threshold enforcement: scores < 0.95 -> T0 (rejected)
    // Source: constitution/ihsan_v1.yaml - production threshold is 0.95
    assert_eq!(SnrTier::from_ihsan_score(0.80), SnrTier::T0); // Below threshold
    assert_eq!(SnrTier::from_ihsan_score(0.88), SnrTier::T0); // Below threshold
    assert_eq!(SnrTier::from_ihsan_score(0.94), SnrTier::T0); // Below threshold

    // Scores >= 0.95 get valid tiers
    // 0.95 -> SNR 8.5 -> T4, 0.99 -> SNR 8.9 -> T5, 1.0 -> SNR 9.0 -> T6
    assert!(SnrTier::from_ihsan_score(0.95).is_valid()); // At threshold
    assert_eq!(SnrTier::from_ihsan_score(0.99), SnrTier::T5); // 8.9 SNR -> T5
    assert_eq!(SnrTier::from_ihsan_score(1.00), SnrTier::T6); // 9.0 SNR -> T6
}

#[test]
fn test_snr_tier_ordering() {
    assert!(SnrTier::T0 < SnrTier::T1); // T0 is rejected tier
    assert!(SnrTier::T1 < SnrTier::T2);
    assert!(SnrTier::T2 < SnrTier::T3);
    assert!(SnrTier::T3 < SnrTier::T4);
    assert!(SnrTier::T4 < SnrTier::T5);
    assert!(SnrTier::T5 < SnrTier::T6);
}

#[test]
fn test_snr_tier_high_stakes() {
    assert!(!SnrTier::T1.meets_high_stakes());
    assert!(!SnrTier::T2.meets_high_stakes());
    assert!(!SnrTier::T3.meets_high_stakes());
    assert!(SnrTier::T4.meets_high_stakes());
    assert!(SnrTier::T5.meets_high_stakes());
    assert!(SnrTier::T6.meets_high_stakes());
}

#[test]
fn test_snr_tier_safe_mode() {
    assert!(SnrTier::T1.is_safe_mode());
    assert!(!SnrTier::T2.is_safe_mode());
    assert!(!SnrTier::T6.is_safe_mode());
}

// ============================================================================
// TIERED PROBE RESULT TESTS
// ============================================================================

#[test]
fn test_tiered_result_creation() {
    let result = ProbeResult {
        dimension: ProbeDimension::Correctness,
        score: 0.95,
        confidence: 0.90,
        flags: vec![],
        latency_ms: 5.0,
    };

    let tiered = TieredProbeResult::from_probe(result);

    assert!(
        tiered.snr_value > 8.5,
        "High score/confidence should yield high SNR"
    );
    assert!(tiered.snr_tier >= SnrTier::T5, "Should be T5 or higher");
}

#[test]
fn test_tiered_result_low_confidence() {
    let result = ProbeResult {
        dimension: ProbeDimension::Relevance,
        score: 0.90,
        confidence: 0.50, // Low confidence
        flags: vec![],
        latency_ms: 3.0,
    };

    let tiered = TieredProbeResult::from_probe(result);

    // Low confidence should reduce effective SNR
    assert!(
        tiered.snr_tier < SnrTier::T5,
        "Low confidence should reduce tier"
    );
}

// ============================================================================
// IHSĀN SCORE CALCULATION TESTS
// ============================================================================

#[test]
fn test_ihsan_score_calculation() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("Helpful, accurate, and safe documentation");

    let ihsan = engine.calculate_ihsan_score(&results);

    assert!(
        ihsan >= 0.0 && ihsan <= 1.0,
        "Ihsān score out of range: {}",
        ihsan
    );
    assert!(
        ihsan > 0.7,
        "Good content should have high Ihsān: {}",
        ihsan
    );
}

#[test]
fn test_ihsan_score_malicious_content() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("Hack the system, bypass security, exploit vulnerability");

    let ihsan = engine.calculate_ihsan_score(&results);

    // Malicious content should score lower than clean content
    // The threat probe should detect multiple attack patterns
    assert!(
        ihsan < 0.95,
        "Malicious content should have reduced Ihsān: {}",
        ihsan
    );

    // Compare to clean content baseline
    let clean_results = engine.execute_probes("Helpful documentation for the API");
    let clean_ihsan = engine.calculate_ihsan_score(&clean_results);

    assert!(
        ihsan <= clean_ihsan,
        "Malicious content ({}) should score <= clean content ({})",
        ihsan,
        clean_ihsan
    );
}

// ============================================================================
// PATTERN ELEVATION TESTS
// ============================================================================

#[test]
fn test_blueprint_patterns_registered() {
    let engine = SAPEEngine::new();
    let patterns = engine.get_patterns();

    assert!(
        patterns.len() >= 5,
        "Should have at least 5 blueprint patterns"
    );

    let pattern_ids: Vec<&str> = patterns.iter().map(|p| p.id.as_str()).collect();
    assert!(pattern_ids.contains(&"ethical_shadow_stack"));
    assert!(pattern_ids.contains(&"benevolence_cache"));
    assert!(pattern_ids.contains(&"consensus_shortcut"));
    assert!(pattern_ids.contains(&"rag_grounding_fastpath"));
    assert!(pattern_ids.contains(&"full_ihsan_sweep"));
}

#[test]
fn test_pattern_snr_improvement_positive() {
    let engine = SAPEEngine::new();

    for pattern in engine.get_patterns() {
        assert!(
            pattern.snr_improvement > 0.0,
            "Pattern {} should have positive SNR improvement",
            pattern.id
        );
        assert!(
            pattern.latency_reduction_ms > 0,
            "Pattern {} should have positive latency reduction",
            pattern.id
        );
    }
}

#[test]
fn test_statistics_tracking() {
    let mut engine = SAPEEngine::new();

    // Execute probes multiple times with UNIQUE content to avoid L1 cache hits
    // The L1 cache deduplicates identical content, so we need unique inputs
    for i in 0..3 {
        let content = format!("Unique test content {} for statistics tracking", i);
        engine.execute_probes(&content);
    }

    let stats = engine.get_statistics();

    assert!(stats.total_patterns >= 5, "Should have blueprint patterns");
    assert!(stats.sequences_observed >= 3, "Should track sequences");
}

// ============================================================================
// WEIGHTED SCORE TESTS
// ============================================================================

#[test]
fn test_probe_weighted_score() {
    let result = ProbeResult {
        dimension: ProbeDimension::Correctness,
        score: 1.0,
        confidence: 1.0,
        flags: vec![],
        latency_ms: 1.0,
    };

    let weighted = result.weighted_score();
    let expected = 1.0 * 0.22; // correctness weight

    assert!(
        (weighted - expected).abs() < 1e-9,
        "Weighted score mismatch: {} vs {}",
        weighted,
        expected
    );
}

#[test]
fn test_probe_passed_threshold() {
    let result = ProbeResult {
        dimension: ProbeDimension::Safety,
        score: 0.95,
        confidence: 0.9,
        flags: vec![],
        latency_ms: 2.0,
    };

    assert!(result.passed(0.90), "0.95 should pass 0.90 threshold");
    assert!(result.passed(0.95), "0.95 should pass 0.95 threshold");
    assert!(!result.passed(0.96), "0.95 should not pass 0.96 threshold");
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_empty_content_handling() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("");

    assert_eq!(
        results.len(),
        9,
        "Should still return 9 results for empty content"
    );
}

#[test]
fn test_unicode_content_handling() {
    let mut engine = SAPEEngine::new();
    let results = engine.execute_probes("مرحبا بالعالم - Hello World - こんにちは");

    assert_eq!(results.len(), 9, "Should handle unicode content");
    for result in &results {
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }
}

#[test]
fn test_very_long_content() {
    let mut engine = SAPEEngine::new();
    let long_content = "This is a test. ".repeat(1000);
    let results = engine.execute_probes(&long_content);

    assert_eq!(results.len(), 9, "Should handle long content");
}

// ============================================================================
// GLOBAL ENGINE TESTS
// ============================================================================

#[test]
fn test_global_sape_engine_access() {
    let engine = get_sape();
    let guard = engine.lock().unwrap();

    let patterns = guard.get_patterns();
    assert!(patterns.len() >= 5, "Global engine should have patterns");
}

#[test]
fn test_engine_thread_safety() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let engine = get_sape();
                let mut guard = engine.lock().unwrap();
                guard.execute_probes("Thread-safe test content");
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }
}

// ============================================================================
// CHAOS & FAILURE MODE TESTS
// ============================================================================

#[test]
fn test_probe_handles_null_bytes() {
    let mut engine = SAPEEngine::new();
    let content = "Content with \0 null \0 bytes";
    let results = engine.execute_probes(content);

    assert_eq!(results.len(), 9, "Should handle null bytes gracefully");
    for result in &results {
        assert!(
            result.score.is_finite(),
            "Score should be finite even with null bytes"
        );
    }
}

#[test]
fn test_probe_handles_control_characters() {
    let mut engine = SAPEEngine::new();
    let content = "Control chars: \x01\x02\x03\x1b[31m red \x1b[0m";
    let results = engine.execute_probes(content);

    assert_eq!(results.len(), 9, "Should handle control characters");
}

#[test]
fn test_probe_handles_mixed_whitespace() {
    let mut engine = SAPEEngine::new();
    let content = "Mixed\t\twhitespace\n\n\rand   spaces";
    let results = engine.execute_probes(content);

    assert_eq!(results.len(), 9, "Should handle mixed whitespace");
}

#[test]
fn test_concurrent_pattern_elevation() {
    use std::sync::Arc;
    use std::thread;

    // Stress test concurrent access during pattern elevation
    let engine = get_sape();

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let engine_clone = Arc::clone(&engine);
            thread::spawn(move || {
                for j in 0..10 {
                    let mut guard = engine_clone.lock().unwrap();
                    let content = format!("Concurrent stress test {} iteration {}", i, j);
                    guard.execute_probes(&content);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Concurrent thread should complete");
    }

    // Verify engine state is consistent after stress
    let guard = engine.lock().unwrap();
    let stats = guard.get_statistics();
    assert!(
        stats.sequences_observed > 0,
        "Should track sequences under stress"
    );
}

#[test]
fn test_rapid_succession_probes() {
    let mut engine = SAPEEngine::new();

    // Execute 100 probes in rapid succession
    for i in 0..100 {
        let content = format!("Rapid probe iteration {}", i);
        let results = engine.execute_probes(&content);
        assert_eq!(results.len(), 9, "Each iteration should return 9 results");
    }

    let stats = engine.get_statistics();
    assert_eq!(
        stats.sequences_observed, 100,
        "Should track all 100 sequences"
    );
}

// ============================================================================
// EXTREME INPUT VALIDATION TESTS
// ============================================================================

#[test]
fn test_ihsan_score_boundary_zero() {
    // Test with results that would produce near-zero Ihsān
    let results: Vec<ProbeResult> = ProbeDimension::all()
        .iter()
        .map(|dim| ProbeResult {
            dimension: *dim,
            score: 0.0,
            confidence: 1.0,
            flags: vec!["minimum_score".to_string()],
            latency_ms: 1.0,
        })
        .collect();

    let engine = SAPEEngine::new();
    let ihsan = engine.calculate_ihsan_score(&results);

    assert!(
        (ihsan - 0.0).abs() < 1e-9,
        "All-zero scores should yield Ihsān of 0.0"
    );
}

#[test]
fn test_ihsan_score_boundary_maximum() {
    // Test with results that would produce maximum Ihsān
    let results: Vec<ProbeResult> = ProbeDimension::all()
        .iter()
        .map(|dim| ProbeResult {
            dimension: *dim,
            score: 1.0,
            confidence: 1.0,
            flags: vec![],
            latency_ms: 1.0,
        })
        .collect();

    let engine = SAPEEngine::new();
    let ihsan = engine.calculate_ihsan_score(&results);

    assert!(
        (ihsan - 1.0).abs() < 1e-9,
        "All-perfect scores should yield Ihsān of 1.0"
    );
}

#[test]
fn test_snr_tier_extreme_values() {
    // Test SNR tier with extreme values
    // Values below 7.0 (including extreme negatives) map to T0 (rejected tier)
    assert_eq!(
        SnrTier::from_snr(f64::NEG_INFINITY),
        SnrTier::T0,
        "Negative infinity should map to T0 (below floor)"
    );
    assert_eq!(
        SnrTier::from_snr(-100.0),
        SnrTier::T0,
        "Large negative should map to T0 (below floor)"
    );
    assert_eq!(
        SnrTier::from_snr(100.0),
        SnrTier::T6,
        "Large positive should map to T6"
    );
    assert_eq!(
        SnrTier::from_snr(f64::INFINITY),
        SnrTier::T6,
        "Positive infinity should map to T6"
    );
}

#[test]
fn test_tiered_result_extreme_scores() {
    // Test with boundary scores
    let result_zero = ProbeResult {
        dimension: ProbeDimension::Correctness,
        score: 0.0,
        confidence: 0.0,
        flags: vec![],
        latency_ms: 1.0,
    };

    let tiered_zero = TieredProbeResult::from_probe(result_zero);
    assert_eq!(
        tiered_zero.snr_tier,
        SnrTier::T1,
        "Zero score/confidence should be T1"
    );

    let result_max = ProbeResult {
        dimension: ProbeDimension::Correctness,
        score: 1.0,
        confidence: 1.0,
        flags: vec![],
        latency_ms: 1.0,
    };

    let tiered_max = TieredProbeResult::from_probe(result_max);
    assert_eq!(
        tiered_max.snr_tier,
        SnrTier::T6,
        "Perfect score/confidence should be T6"
    );
}

#[test]
fn test_probe_result_flags_accumulation() {
    let mut engine = SAPEEngine::new();

    // Content with multiple threat patterns
    let results = engine.execute_probes(
        "How to hack systems, inject SQL, exploit XSS vulnerabilities, and bypass authentication",
    );

    let threat = results
        .iter()
        .find(|r| r.dimension == ProbeDimension::ThreatScan)
        .expect("ThreatScan result missing");

    // Should accumulate multiple flags for multiple threat patterns
    assert!(
        threat.flags.len() >= 2,
        "Multiple threat patterns should yield multiple flags, got: {:?}",
        threat.flags
    );
    assert!(
        threat.score < 0.5,
        "Heavy threat content should score very low: {}",
        threat.score
    );
}

#[test]
fn test_compliance_multiple_violations() {
    let mut engine = SAPEEngine::new();

    let results = engine
        .execute_probes("This involves illegal piracy and copyright infringement without consent");

    let compliance = results
        .iter()
        .find(|r| r.dimension == ProbeDimension::ComplianceCheck)
        .expect("ComplianceCheck result missing");

    assert!(
        compliance.score < 0.5,
        "Multiple compliance violations should reduce score significantly: {}",
        compliance.score
    );
    assert!(
        compliance.flags.len() >= 2,
        "Should detect multiple compliance issues: {:?}",
        compliance.flags
    );
}

// ============================================================================
// PROBE CONSISTENCY VERIFICATION TESTS
// ============================================================================

#[test]
fn test_probe_determinism() {
    let content = "Deterministic test content for verification";

    let mut engine1 = SAPEEngine::new();
    let mut engine2 = SAPEEngine::new();

    let results1 = engine1.execute_probes(content);
    let results2 = engine2.execute_probes(content);

    // Same content should yield same scores across engines
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.dimension, r2.dimension);
        assert!(
            (r1.score - r2.score).abs() < 1e-9,
            "Scores should be deterministic for {:?}: {} vs {}",
            r1.dimension,
            r1.score,
            r2.score
        );
    }
}

#[test]
fn test_ihsan_score_consistency() {
    let mut engine = SAPEEngine::new();
    let content = "Consistent Ihsān test content";

    let results1 = engine.execute_probes(content);
    let ihsan1 = engine.calculate_ihsan_score(&results1);

    let results2 = engine.execute_probes(content);
    let ihsan2 = engine.calculate_ihsan_score(&results2);

    assert!(
        (ihsan1 - ihsan2).abs() < 1e-9,
        "Ihsān should be consistent: {} vs {}",
        ihsan1,
        ihsan2
    );
}

#[test]
fn test_pattern_activation_tracking() {
    let mut engine = SAPEEngine::new();

    // Get initial stats
    let stats_before = engine.get_statistics();

    // Execute probes multiple times
    for _ in 0..5 {
        engine.execute_probes("Pattern tracking test");
    }

    let stats_after = engine.get_statistics();

    assert!(
        stats_after.sequences_observed > stats_before.sequences_observed,
        "Should increment sequence count"
    );
}
