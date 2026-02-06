//! BIZRA PROOF SPACE VALIDATOR - Demo CLI
//!
//! Demonstrates life/death judgments on blocks.

use bizra_proofspace::*;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("      BIZRA PROOF SPACE VALIDATOR v1.0.0 - LIFE/DEATH DEMO");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Create a sample block body
    let body = create_sample_body();

    // Build an unsigned block
    println!("▸ Building Block...");
    let creator_node = "a".repeat(64); // Placeholder public key
    let (unsigned, block_id) = BlockBuilder::new(BlockType::KnowledgeBlock, creator_node.clone())
        .status(BlockStatus::Submitted)
        .body(body.clone())
        .build_unsigned()
        .expect("Failed to build unsigned block");

    println!("  Block Type: {:?}", unsigned.core.block_type);
    println!("  Created At: {} ms", unsigned.core.created_at);
    println!("  Version: {}", unsigned.core.version);
    println!("  Computed block_id: {}", &block_id[..16]);
    println!();

    // Create a full block (with placeholder signature)
    let block = BizraBlock {
        core: BlockCore {
            block_id: block_id.clone(),
            block_type: unsigned.core.block_type,
            creator_node: unsigned.core.creator_node.clone(),
            created_at: unsigned.core.created_at,
            version: unsigned.core.version.clone(),
            status: unsigned.core.status,
            parent_block: None,
        },
        body,
        signatures: Signatures {
            creator_signature: "b".repeat(128), // Placeholder (invalid)
            verifier_signatures: vec![],
            resource_pool_signature: None,
            deprecation_signature: None,
        },
    };

    // Validate with non-strict mode first (skip signature verification issue)
    println!("▸ Validation (non-strict mode)...");
    let validator = ProofSpaceValidator::new(false);
    let result = validator.validate(&block);

    println!("  Verdict: {:?}", result.verdict);
    println!("  Block ID Match: {}", result.block_id == result.computed_block_id);
    println!();

    println!("▸ FATE Scores:");
    println!("  Ihsān Score:      {:.3} (threshold: {:.3})",
             result.fate_scores.ihsan_score, IHSAN_THRESHOLD);
    println!("  Adl Score:        {:.3}", result.fate_scores.adl_score);
    println!("  Harm Score:       {:.3} (max: {:.3})",
             result.fate_scores.harm_score, MAX_HARM_SCORE);
    println!("  Confidence Score: {:.3} (min: {:.3})",
             result.fate_scores.confidence_score, MIN_CONFIDENCE);
    println!();

    if !result.errors.is_empty() {
        println!("▸ Validation Errors ({}):", result.errors.len());
        for (i, error) in result.errors.iter().enumerate() {
            println!("  {}. {}", i + 1, error);
        }
        println!();
    }

    if !result.warnings.is_empty() {
        println!("▸ Warnings ({}):", result.warnings.len());
        for warning in &result.warnings {
            println!("  - {}", warning);
        }
        println!();
    }

    // Demo: Test various validation scenarios
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    VALIDATION SCENARIOS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Scenario 1: Block with circular dependency
    println!("▸ Scenario 1: Circular Dependency");
    let mut circular_body = create_sample_body();
    circular_body.dependencies.inputs.block_refs = vec![block_id.clone()];

    let circular_block = BizraBlock {
        core: BlockCore {
            block_id: block_id.clone(),
            block_type: BlockType::KnowledgeBlock,
            creator_node: creator_node.clone(),
            created_at: 1234567890000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: None,
        },
        body: circular_body,
        signatures: Signatures {
            creator_signature: "b".repeat(128),
            verifier_signatures: vec![],
            resource_pool_signature: None,
            deprecation_signature: None,
        },
    };

    let circular_result = validator.validate(&circular_block);
    println!("  Verdict: {:?}", circular_result.verdict);
    println!("  Circular dependency detected: {}",
             circular_result.errors.iter().any(|e|
                 matches!(e, ValidationError::CircularDependency { .. })));
    println!();

    // Scenario 2: High harm score
    println!("▸ Scenario 2: High Harm Score (strict mode)");
    let mut harmful_body = create_sample_body();
    harmful_body.ethical_envelope.harm_analysis.net_harm_score = 0.8;

    let harmful_block = BizraBlock {
        core: BlockCore {
            block_id: "c".repeat(64),
            block_type: BlockType::WorkflowBlock,
            creator_node: creator_node.clone(),
            created_at: 1234567890000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: None,
        },
        body: harmful_body,
        signatures: Signatures {
            creator_signature: "d".repeat(128),
            verifier_signatures: vec![],
            resource_pool_signature: None,
            deprecation_signature: None,
        },
    };

    let strict_validator = ProofSpaceValidator::new(true);
    let harmful_result = strict_validator.validate(&harmful_block);
    println!("  Verdict: {:?}", harmful_result.verdict);
    println!("  Harm gate triggered: {}",
             harmful_result.errors.iter().any(|e|
                 matches!(e, ValidationError::HarmScoreTooHigh { .. })));
    println!();

    // Scenario 3: Invalid SMT-LIB2 assertion
    println!("▸ Scenario 3: Invalid Formal Assertion");
    let mut bad_smt_body = create_sample_body();
    bad_smt_body.ethical_envelope.formal_assertions = vec![
        "(assert (= x 1))".to_string(),       // Valid
        "assert (= y 2)".to_string(),          // Invalid: missing open paren
    ];

    let bad_smt_block = BizraBlock {
        core: BlockCore {
            block_id: "e".repeat(64),
            block_type: BlockType::ProofBlock,
            creator_node: creator_node.clone(),
            created_at: 1234567890000,
            version: "1.0.0".to_string(),
            status: BlockStatus::Draft,
            parent_block: None,
        },
        body: bad_smt_body,
        signatures: Signatures {
            creator_signature: "f".repeat(128),
            verifier_signatures: vec![],
            resource_pool_signature: None,
            deprecation_signature: None,
        },
    };

    let smt_result = validator.validate(&bad_smt_block);
    println!("  Verdict: {:?}", smt_result.verdict);
    println!("  Invalid assertion detected: {}",
             smt_result.errors.iter().any(|e|
                 matches!(e, ValidationError::InvalidFormalAssertion { .. })));
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Schema Version:       {}", SCHEMA_VERSION);
    println!("  Ihsān Threshold:      {}", IHSAN_THRESHOLD);
    println!("  Max Harm Score:       {}", MAX_HARM_SCORE);
    println!("  Min Confidence:       {}", MIN_CONFIDENCE);
    println!("  Min Verifiers:        {}", MIN_VERIFIER_SIGNATURES);
    println!();
    println!("  Canonicalization:     RFC 8785 (JCS)");
    println!("  Hash Algorithm:       SHA-256");
    println!("  Signature Scheme:     Ed25519");
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  STANDING ON GIANTS:");
    println!("  RFC 8785 (2019) • NIST FIPS 180-4 • Bernstein (2012)");
    println!("  SMT-LIB2 (2010) • Z3 (2008) • Al-Ghazali (1095)");
    println!("═══════════════════════════════════════════════════════════════════");
}

fn create_sample_body() -> BlockBody {
    BlockBody {
        dependencies: Dependencies {
            inputs: Inputs {
                block_refs: vec![],
                external_refs: vec![ExternalRef {
                    uri: "https://github.com/bizra/core".to_string(),
                    hash: Some("d".repeat(64)),
                    r#type: ExternalRefType::Git,
                }],
            },
            assumptions: vec![Assumption {
                statement: "System has at least 4GB RAM".to_string(),
                test_method: "Check /proc/meminfo".to_string(),
                criticality: Criticality::Medium,
            }],
            constraints: vec![Constraint {
                r#type: ConstraintType::Ethical,
                description: "Must not process personal data without consent".to_string(),
                enforceable: true,
            }],
        },
        proof_pack: ProofPack {
            reproduction_steps: vec![
                ReproductionStep {
                    step_number: 1,
                    instruction: "Clone repository".to_string(),
                    expected_hash: None,
                    timeout_ms: 60000,
                },
                ReproductionStep {
                    step_number: 2,
                    instruction: "Run cargo test".to_string(),
                    expected_hash: Some("e".repeat(64)),
                    timeout_ms: 300000,
                },
            ],
            validation_method: ValidationMethod {
                r#type: ValidationMethodType::DeterministicReplay,
                config_jcs: r#"{"seed":42,"iterations":100}"#.to_string(),
                oracle: None,
            },
            expected_outcome: ExpectedOutcome {
                result_hash: "f".repeat(64),
                success_criteria: "All 11 tests pass with 0 failures".to_string(),
                tolerance: None,
            },
            failure_modes: vec![FailureMode {
                condition: "Network timeout during clone".to_string(),
                probability: 0.05,
                mitigation: "Retry with exponential backoff".to_string(),
            }],
            confidence_bounds: ConfidenceBounds {
                confidence_level: 0.95,
                statistical_power: Some(0.8),
                sample_size: Some(100),
                effect_size: Some(0.5),
            },
        },
        impact_claim: ImpactClaim {
            what_changed: "Mobile agent framework with sovereign ethics".to_string(),
            who_benefits: vec![
                Beneficiary {
                    beneficiary_type: BeneficiaryType::Community,
                    description: "Open source developers".to_string(),
                    count_estimate: Some(10000),
                },
                Beneficiary {
                    beneficiary_type: BeneficiaryType::Civilization,
                    description: "Future generations via ethical AI foundations".to_string(),
                    count_estimate: None,
                },
            ],
            how_measured: vec![Measurement {
                metric: "Test coverage".to_string(),
                baseline: 0.0,
                observed: 100.0,
                unit: "percent".to_string(),
                methodology: "cargo tarpaulin".to_string(),
            }],
            impact_score: 0.85,
            uncertainty: vec![Uncertainty {
                r#type: UncertaintyType::Epistemic,
                magnitude: 0.15,
                description: "Adoption rate uncertainty".to_string(),
            }],
            time_horizon: TimeHorizon::Long,
        },
        ethical_envelope: EthicalEnvelope {
            harm_analysis: HarmAnalysis {
                potential_harms: vec![PotentialHarm {
                    r#type: HarmType::Indirect,
                    severity: Severity::Low,
                    likelihood: 0.1,
                    affected_parties: vec!["Legacy system operators".to_string()],
                    mitigation: "Provide migration guide".to_string(),
                }],
                net_harm_score: 0.05,
            },
            misuse_risk: MisuseRisk {
                attack_vectors: vec!["Malicious agent injection".to_string()],
                difficulty: MisuseDifficulty::Hard,
                impact_if_successful: "Unauthorized resource access".to_string(),
                preventive_measures: vec![
                    "FATE gate enforcement".to_string(),
                    "Permit-based capability system".to_string(),
                ],
            },
            context_limits: ContextLimits {
                valid_domains: vec!["distributed systems".to_string(), "agent orchestration".to_string()],
                invalid_domains: vec!["weapons".to_string(), "surveillance".to_string()],
                required_context: vec!["Understanding of Byzantine fault tolerance".to_string()],
            },
            reversibility: Reversibility {
                is_reversible: true,
                reversal_procedure: Some("Terminate agents and revoke permits".to_string()),
                window_hours: Some(168), // 1 week
            },
            human_override_conditions: HumanOverrideConditions {
                can_override: true,
                required_authority_level: Some(7),
                override_procedures: vec![
                    "Contact Node0 administrator".to_string(),
                    "Provide justification in writing".to_string(),
                ],
                audit_trail_required: true,
            },
            formal_assertions: vec![
                "(assert (>= ihsan_score 0.95))".to_string(),
                "(assert (<= gini_coefficient 0.35))".to_string(),
                "(assert (=> is_reversible (not (= reversal_procedure nil))))".to_string(),
            ],
        },
    }
}
