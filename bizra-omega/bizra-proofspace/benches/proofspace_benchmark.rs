//! BIZRA Proof Space Validator — Performance Benchmarks
//!
//! Measures: JCS canonicalization, block ID computation, full validation pipeline
//!
//! Standing on Giants: Criterion • RFC 8785 • Ed25519 • SHA-256

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use bizra_proofspace::{
    jcs_canonicalize, compute_block_id,
    ProofSpaceValidator, BlockBuilder, BlockType, BlockStatus,
    BizraBlock, BlockCore, BlockBody, UnsignedBlock, BlockCoreUnsigned,
    Dependencies, Inputs, ExternalRef, ExternalRefType,
    Assumption, Criticality, Constraint, ConstraintType,
    ProofPack, ReproductionStep, ValidationMethod, ValidationMethodType,
    ExpectedOutcome, FailureMode, ConfidenceBounds,
    ImpactClaim, Beneficiary, BeneficiaryType, Measurement, Uncertainty, UncertaintyType,
    TimeHorizon,
    EthicalEnvelope, HarmAnalysis, PotentialHarm, HarmType, Severity,
    MisuseRisk, MisuseDifficulty, ContextLimits, Reversibility,
    HumanOverrideConditions,
    Signatures, VerifierSignature, VerificationType,
};

// =============================================================================
// TEST FIXTURES
// =============================================================================

/// Create a minimal block body (fewest allocations)
fn minimal_body() -> BlockBody {
    BlockBody {
        dependencies: Dependencies {
            inputs: Inputs {
                block_refs: vec![],
                external_refs: vec![],
            },
            assumptions: vec![],
            constraints: vec![],
        },
        proof_pack: ProofPack {
            reproduction_steps: vec![ReproductionStep {
                step_number: 1,
                instruction: "Execute test".to_string(),
                expected_hash: None,
                timeout_ms: 1000,
            }],
            validation_method: ValidationMethod {
                r#type: ValidationMethodType::DeterministicReplay,
                config_jcs: "{}".to_string(),
                oracle: None,
            },
            expected_outcome: ExpectedOutcome {
                result_hash: "a".repeat(64),
                success_criteria: "All tests pass".to_string(),
                tolerance: None,
            },
            failure_modes: vec![],
            confidence_bounds: ConfidenceBounds {
                confidence_level: 0.95,
                statistical_power: None,
                sample_size: None,
                effect_size: None,
            },
        },
        impact_claim: ImpactClaim {
            what_changed: "Test improvement".to_string(),
            who_benefits: vec![Beneficiary {
                beneficiary_type: BeneficiaryType::Community,
                description: "Developers".to_string(),
                count_estimate: Some(100),
            }],
            how_measured: vec![],
            impact_score: 0.8,
            uncertainty: vec![],
            time_horizon: TimeHorizon::Medium,
        },
        ethical_envelope: EthicalEnvelope {
            harm_analysis: HarmAnalysis {
                potential_harms: vec![],
                net_harm_score: 0.1,
            },
            misuse_risk: MisuseRisk {
                attack_vectors: vec![],
                difficulty: MisuseDifficulty::Hard,
                impact_if_successful: "Minimal".to_string(),
                preventive_measures: vec![],
            },
            context_limits: ContextLimits {
                valid_domains: vec!["testing".to_string()],
                invalid_domains: vec![],
                required_context: vec![],
            },
            reversibility: Reversibility {
                is_reversible: true,
                reversal_procedure: Some("Rollback".to_string()),
                window_hours: Some(24),
            },
            human_override_conditions: HumanOverrideConditions {
                can_override: true,
                required_authority_level: Some(5),
                override_procedures: vec!["Contact admin".to_string()],
                audit_trail_required: true,
            },
            formal_assertions: vec![],
        },
    }
}

/// Create a full-featured block body (many allocations, realistic)
fn full_body() -> BlockBody {
    BlockBody {
        dependencies: Dependencies {
            inputs: Inputs {
                block_refs: vec!["b".repeat(64), "c".repeat(64), "d".repeat(64)],
                external_refs: vec![
                    ExternalRef {
                        uri: "https://example.com/dataset-v1.tar.gz".to_string(),
                        hash: Some("e".repeat(64)),
                        r#type: ExternalRefType::Https,
                    },
                    ExternalRef {
                        uri: "ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
                        hash: Some("f".repeat(64)),
                        r#type: ExternalRefType::Ipfs,
                    },
                ],
            },
            assumptions: vec![
                Assumption {
                    statement: "Input data is UTF-8 encoded".to_string(),
                    test_method: "Validate encoding on ingest".to_string(),
                    criticality: Criticality::High,
                },
                Assumption {
                    statement: "Network latency < 100ms".to_string(),
                    test_method: "Ping test".to_string(),
                    criticality: Criticality::Medium,
                },
            ],
            constraints: vec![
                Constraint {
                    r#type: ConstraintType::Ethical,
                    description: "No PII in output".to_string(),
                    enforceable: true,
                },
                Constraint {
                    r#type: ConstraintType::Compute,
                    description: "Max 4GB RAM".to_string(),
                    enforceable: true,
                },
            ],
        },
        proof_pack: ProofPack {
            reproduction_steps: vec![
                ReproductionStep {
                    step_number: 1,
                    instruction: "Clone repository at commit abc123".to_string(),
                    expected_hash: Some("1".repeat(64)),
                    timeout_ms: 30_000,
                },
                ReproductionStep {
                    step_number: 2,
                    instruction: "Install dependencies via cargo build".to_string(),
                    expected_hash: None,
                    timeout_ms: 120_000,
                },
                ReproductionStep {
                    step_number: 3,
                    instruction: "Execute benchmark suite".to_string(),
                    expected_hash: Some("2".repeat(64)),
                    timeout_ms: 600_000,
                },
            ],
            validation_method: ValidationMethod {
                r#type: ValidationMethodType::DeterministicReplay,
                config_jcs: r#"{"seed":42,"rounds":100}"#.to_string(),
                oracle: Some("https://oracle.bizra.dev/validate".to_string()),
            },
            expected_outcome: ExpectedOutcome {
                result_hash: "3".repeat(64),
                success_criteria: "P99 latency < 250ns, throughput > 1M msg/s".to_string(),
                tolerance: Some(0.05),
            },
            failure_modes: vec![
                FailureMode {
                    condition: "Memory exhaustion".to_string(),
                    probability: 0.01,
                    mitigation: "OOM kill and restart".to_string(),
                },
                FailureMode {
                    condition: "Network partition".to_string(),
                    probability: 0.05,
                    mitigation: "Graceful degradation to local-only mode".to_string(),
                },
            ],
            confidence_bounds: ConfidenceBounds {
                confidence_level: 0.99,
                statistical_power: Some(0.95),
                sample_size: Some(10_000),
                effect_size: Some(0.3),
            },
        },
        impact_claim: ImpactClaim {
            what_changed: "Reduced IPC latency from 10ms to 250ns".to_string(),
            who_benefits: vec![
                Beneficiary {
                    beneficiary_type: BeneficiaryType::Individual,
                    description: "End users experience faster responses".to_string(),
                    count_estimate: Some(10_000),
                },
                Beneficiary {
                    beneficiary_type: BeneficiaryType::Community,
                    description: "Open-source LLM community".to_string(),
                    count_estimate: Some(500),
                },
                Beneficiary {
                    beneficiary_type: BeneficiaryType::Civilization,
                    description: "Sovereign AI infrastructure".to_string(),
                    count_estimate: None,
                },
            ],
            how_measured: vec![Measurement {
                metric: "P99 IPC latency".to_string(),
                baseline: 10_000_000.0,
                observed: 250.0,
                unit: "nanoseconds".to_string(),
                methodology: "Criterion benchmark with 10K iterations".to_string(),
            }],
            impact_score: 0.85,
            uncertainty: vec![Uncertainty {
                r#type: UncertaintyType::Aleatoric,
                magnitude: 0.1,
                description: "Measurement noise from OS scheduling".to_string(),
            }],
            time_horizon: TimeHorizon::Long,
        },
        ethical_envelope: EthicalEnvelope {
            harm_analysis: HarmAnalysis {
                potential_harms: vec![PotentialHarm {
                    r#type: HarmType::Indirect,
                    severity: Severity::Low,
                    likelihood: 0.05,
                    affected_parties: vec!["resource-constrained users".to_string()],
                    mitigation: "Provide fallback TCP transport".to_string(),
                }],
                net_harm_score: 0.05,
            },
            misuse_risk: MisuseRisk {
                attack_vectors: vec!["Shared memory tampering".to_string()],
                difficulty: MisuseDifficulty::Hard,
                impact_if_successful: "Message corruption".to_string(),
                preventive_measures: vec![
                    "HMAC message authentication".to_string(),
                    "Memory-mapped segment permissions".to_string(),
                ],
            },
            context_limits: ContextLimits {
                valid_domains: vec!["AI inference".to_string(), "IPC transport".to_string()],
                invalid_domains: vec!["Safety-critical real-time systems".to_string()],
                required_context: vec!["Linux 5.x+ or Windows 10+".to_string()],
            },
            reversibility: Reversibility {
                is_reversible: true,
                reversal_procedure: Some("Revert to TCP transport path".to_string()),
                window_hours: Some(168),
            },
            human_override_conditions: HumanOverrideConditions {
                can_override: true,
                required_authority_level: Some(3),
                override_procedures: vec![
                    "Set IPC_TRANSPORT=tcp in config".to_string(),
                    "Restart bridge service".to_string(),
                ],
                audit_trail_required: true,
            },
            formal_assertions: vec![
                "(assert (< latency_ns 250))".to_string(),
                "(assert (>= throughput_msg_per_sec 1000000))".to_string(),
            ],
        },
    }
}

/// Create a minimal unsigned block for canonicalization benchmarks
fn minimal_unsigned() -> UnsignedBlock {
    UnsignedBlock {
        core: BlockCoreUnsigned {
            block_type: BlockType::KnowledgeBlock,
            creator_node: "a".repeat(64),
            created_at: 1234567890000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: None,
        },
        body: minimal_body(),
    }
}

/// Create a full-featured unsigned block
fn full_unsigned() -> UnsignedBlock {
    UnsignedBlock {
        core: BlockCoreUnsigned {
            block_type: BlockType::ProofBlock,
            creator_node: "a".repeat(64),
            created_at: 1717171717000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Submitted,
            parent_block: Some("9".repeat(64)),
        },
        body: full_body(),
    }
}

/// Create a full BizraBlock for validation benchmarks
fn valid_block() -> BizraBlock {
    let body = minimal_body();
    let unsigned = UnsignedBlock {
        core: BlockCoreUnsigned {
            block_type: BlockType::KnowledgeBlock,
            creator_node: "a".repeat(64),
            created_at: 1234567890000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: None,
        },
        body: body.clone(),
    };
    let block_id = compute_block_id(&unsigned).unwrap();

    BizraBlock {
        core: BlockCore {
            block_id,
            block_type: BlockType::KnowledgeBlock,
            creator_node: "a".repeat(64),
            created_at: 1234567890000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: None,
        },
        body,
        signatures: Signatures {
            creator_signature: "c".repeat(128),
            verifier_signatures: vec![],
            resource_pool_signature: None,
            deprecation_signature: None,
        },
    }
}

/// Create a full-featured BizraBlock
fn full_valid_block() -> BizraBlock {
    let body = full_body();
    let unsigned = UnsignedBlock {
        core: BlockCoreUnsigned {
            block_type: BlockType::ProofBlock,
            creator_node: "a".repeat(64),
            created_at: 1717171717000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: Some("9".repeat(64)),
        },
        body: body.clone(),
    };
    let block_id = compute_block_id(&unsigned).unwrap();

    BizraBlock {
        core: BlockCore {
            block_id,
            block_type: BlockType::ProofBlock,
            creator_node: "a".repeat(64),
            created_at: 1717171717000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: Some("9".repeat(64)),
        },
        body,
        signatures: Signatures {
            creator_signature: "c".repeat(128),
            verifier_signatures: vec![
                VerifierSignature {
                    verifier_node: "d".repeat(64),
                    signature: "e".repeat(128),
                    verification_type: VerificationType::Reproduction,
                    timestamp: 1717171718000,
                    confidence: 0.95,
                },
                VerifierSignature {
                    verifier_node: "f".repeat(64),
                    signature: "0".repeat(128),
                    verification_type: VerificationType::EthicalReview,
                    timestamp: 1717171719000,
                    confidence: 0.90,
                },
            ],
            resource_pool_signature: None,
            deprecation_signature: None,
        },
    }
}

// =============================================================================
// JCS CANONICALIZATION BENCHMARKS
// =============================================================================

fn bench_jcs_canonicalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("jcs_canonicalize");
    group.throughput(Throughput::Elements(1));

    let minimal = minimal_unsigned();
    group.bench_function("minimal_block", |b| {
        b.iter(|| black_box(jcs_canonicalize(black_box(&minimal)).unwrap()));
    });

    let full = full_unsigned();
    group.bench_function("full_block", |b| {
        b.iter(|| black_box(jcs_canonicalize(black_box(&full)).unwrap()));
    });

    group.finish();
}

// =============================================================================
// BLOCK ID COMPUTATION BENCHMARKS (SHA-256 over JCS)
// =============================================================================

fn bench_compute_block_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_block_id");
    group.throughput(Throughput::Elements(1));

    let minimal = minimal_unsigned();
    group.bench_function("minimal_block", |b| {
        b.iter(|| black_box(compute_block_id(black_box(&minimal)).unwrap()));
    });

    let full = full_unsigned();
    group.bench_function("full_block", |b| {
        b.iter(|| black_box(compute_block_id(black_box(&full)).unwrap()));
    });

    group.finish();
}

// =============================================================================
// FULL VALIDATION BENCHMARKS
// =============================================================================

fn bench_validate(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate");
    group.throughput(Throughput::Elements(1));

    // Non-strict mode (skips FATE gate enforcement)
    let validator_lenient = ProofSpaceValidator::new(false);
    let block = valid_block();

    group.bench_function("minimal_lenient", |b| {
        b.iter(|| black_box(validator_lenient.validate(black_box(&block))));
    });

    // Strict mode (full FATE gate enforcement)
    let validator_strict = ProofSpaceValidator::new(true);

    group.bench_function("minimal_strict", |b| {
        b.iter(|| black_box(validator_strict.validate(black_box(&block))));
    });

    // Full-featured block validation
    let full_block = full_valid_block();

    group.bench_function("full_lenient", |b| {
        b.iter(|| black_box(validator_lenient.validate(black_box(&full_block))));
    });

    group.bench_function("full_strict", |b| {
        b.iter(|| black_box(validator_strict.validate(black_box(&full_block))));
    });

    group.finish();
}

// =============================================================================
// BLOCK BUILDER BENCHMARKS
// =============================================================================

fn bench_block_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_builder");
    group.throughput(Throughput::Elements(1));

    group.bench_function("build_minimal", |b| {
        b.iter(|| {
            let body = minimal_body();
            let result = BlockBuilder::new(
                BlockType::KnowledgeBlock,
                "a".repeat(64),
            )
            .status(BlockStatus::Draft)
            .body(body)
            .build_unsigned();
            black_box(result.unwrap());
        });
    });

    group.bench_function("build_full", |b| {
        b.iter(|| {
            let body = full_body();
            let result = BlockBuilder::new(
                BlockType::ProofBlock,
                "a".repeat(64),
            )
            .status(BlockStatus::Submitted)
            .parent_block("9".repeat(64))
            .body(body)
            .build_unsigned();
            black_box(result.unwrap());
        });
    });

    group.finish();
}

// =============================================================================
// THROUGHPUT BENCHMARKS
// =============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    let batch_size = 100u64;
    group.throughput(Throughput::Elements(batch_size));

    let validator = ProofSpaceValidator::new(false);
    let block = valid_block();

    group.bench_function("batch_validate_100", |b| {
        b.iter(|| {
            for _ in 0..batch_size {
                let result = validator.validate(black_box(&block));
                black_box(result);
            }
        });
    });

    let unsigned = minimal_unsigned();

    group.bench_function("batch_block_id_100", |b| {
        b.iter(|| {
            for _ in 0..batch_size {
                let id = compute_block_id(black_box(&unsigned)).unwrap();
                black_box(id);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_jcs_canonicalize,
    bench_compute_block_id,
    bench_validate,
    bench_block_builder,
    bench_throughput,
);
criterion_main!(benches);
