//! BIZRA PROOF SPACE VALIDATOR v1.0.0
//!
//! Life/death judgments for civilization-grade blocks.
//!
//! # Block Schema Specification
//!
//! - Canonicalization: RFC 8785 (JCS)
//! - Hash: `block_id = SHA-256(JCS(UnsignedBlock))`
//! - UnsignedBlock = { core(without block_id), body }
//! - Signatures excluded from hash input
//!
//! # Standing on Giants
//!
//! - RFC 8785 (2019): JSON Canonicalization Scheme
//! - SHA-256: NIST FIPS 180-4
//! - Ed25519: Bernstein et al. (2012)
//! - SMT-LIB2: Barrett et al. (2010)
//! - Z3: de Moura & Bjørner (2008)

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use thiserror::Error;

// =============================================================================
// CONSTANTS - Single Source of Truth (locked)
// =============================================================================

/// Current schema version
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Ihsān threshold: 0.95 excellence constraint
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// Maximum Adl Gini coefficient
pub const ADL_GINI_MAX: f64 = 0.35;

/// Maximum harm score allowed for acceptance
pub const MAX_HARM_SCORE: f64 = 0.3;

/// Minimum confidence for acceptance
pub const MIN_CONFIDENCE: f64 = 0.8;

/// Minimum verifier signatures for acceptance
pub const MIN_VERIFIER_SIGNATURES: usize = 2;

// String length limits
pub const MAX_URI_LENGTH: usize = 2048;
pub const MAX_SHORT_STRING: usize = 64;
pub const MAX_MEDIUM_STRING: usize = 128;
pub const MAX_LONG_STRING: usize = 256;
pub const MAX_DESCRIPTION_STRING: usize = 512;
pub const MAX_INSTRUCTION_STRING: usize = 1024;

// =============================================================================
// VALIDATION PATTERNS
// =============================================================================

static HEX64_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[a-f0-9]{64}$").unwrap());
static HEX128_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[a-f0-9]{128}$").unwrap());

// =============================================================================
// ERROR TYPES
// =============================================================================

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    #[error("Schema version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },

    #[error("Invalid hash format: {field} must be 64 lowercase hex chars")]
    InvalidHashFormat { field: String },

    #[error("Invalid signature format: {field} must be 128 lowercase hex chars")]
    InvalidSignatureFormat { field: String },

    #[error("Invalid public key format: {field} must be 64 lowercase hex chars")]
    InvalidPublicKeyFormat { field: String },

    #[error("String too long: {field} ({actual} > {max})")]
    StringTooLong {
        field: String,
        actual: usize,
        max: usize,
    },

    #[error("Value out of range: {field} ({actual} not in {min}..{max})")]
    ValueOutOfRange {
        field: String,
        actual: f64,
        min: f64,
        max: f64,
    },

    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid timestamp: {field} must be positive")]
    InvalidTimestamp { field: String },

    #[error("Block ID mismatch: computed {computed}, declared {declared}")]
    BlockIdMismatch { computed: String, declared: String },

    #[error("Invalid creator signature")]
    InvalidCreatorSignature,

    #[error("Invalid verifier signature from {verifier}")]
    InvalidVerifierSignature { verifier: String },

    #[error("Circular dependency: block {block_id} references itself")]
    CircularDependency { block_id: String },

    #[error("Duplicate dependency: {block_id} appears multiple times")]
    DuplicateDependency { block_id: String },

    #[error("Step numbers not strictly increasing: {details}")]
    InvalidStepOrder { details: String },

    #[error("FATE gate violation: {gate} - {reason}")]
    FateViolation { gate: String, reason: String },

    #[error("Harm score too high: {score} > {max}")]
    HarmScoreTooHigh { score: f64, max: f64 },

    #[error("Insufficient verifier signatures: {count} < {required}")]
    InsufficientVerifiers { count: usize, required: usize },

    #[error("Invalid formal assertion at index {index}: {reason}")]
    InvalidFormalAssertion { index: usize, reason: String },

    #[error("JCS canonicalization failed: {reason}")]
    CanonicalizationFailed { reason: String },

    #[error("JSON parsing error: {reason}")]
    JsonError { reason: String },

    #[error("Cryptographic error: {reason}")]
    CryptoError { reason: String },
}

pub type Result<T> = std::result::Result<T, ValidationError>;

// =============================================================================
// BLOCK TYPES (matching TypeScript spec exactly)
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockType {
    KnowledgeBlock,
    WorkflowBlock,
    ToolBlock,
    ServiceBlock,
    ProofBlock,
    MissionBlock,
    VerdictBlock,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockStatus {
    Draft,
    Submitted,
    Accepted,
    Deprecated,
}

// =============================================================================
// CORE STRUCTURES
// =============================================================================

/// BlockCore - identity + lifecycle (block_id EXCLUDED from hash)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockCore {
    pub block_id: String, // 64 hex - EXCLUDED from hash input
    pub block_type: BlockType,
    pub creator_node: String, // 64 hex
    pub created_at: u64,      // Unix ms UTC
    pub version: String,      // "1.0.0"
    pub status: BlockStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_block: Option<String>, // 64 hex
}

/// BlockCoreUnsigned - for hash computation (no block_id)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockCoreUnsigned {
    pub block_type: BlockType,
    pub creator_node: String,
    pub created_at: u64,
    pub version: String,
    pub status: BlockStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_block: Option<String>,
}

impl From<&BlockCore> for BlockCoreUnsigned {
    fn from(core: &BlockCore) -> Self {
        BlockCoreUnsigned {
            block_type: core.block_type,
            creator_node: core.creator_node.clone(),
            created_at: core.created_at,
            version: core.version.clone(),
            status: core.status,
            parent_block: core.parent_block.clone(),
        }
    }
}

// =============================================================================
// DEPENDENCIES
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExternalRefType {
    Git,
    Ipfs,
    Https,
    Doi,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalRef {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    pub r#type: ExternalRefType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Criticality {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assumption {
    pub statement: String,
    pub test_method: String,
    pub criticality: Criticality,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConstraintType {
    Ethical,
    Legal,
    Physical,
    Compute,
    Time,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub r#type: ConstraintType,
    pub description: String,
    pub enforceable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inputs {
    pub block_refs: Vec<String>,
    pub external_refs: Vec<ExternalRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependencies {
    pub inputs: Inputs,
    pub assumptions: Vec<Assumption>,
    pub constraints: Vec<Constraint>,
}

// =============================================================================
// PROOF PACK
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproductionStep {
    pub step_number: u32,
    pub instruction: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_hash: Option<String>,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidationMethodType {
    DeterministicReplay,
    StatisticalTest,
    FormalProof,
    SatSolver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMethod {
    pub r#type: ValidationMethodType,
    pub config_jcs: String, // JCS-canonical JSON string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    pub result_hash: String,
    pub success_criteria: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMode {
    pub condition: String,
    pub probability: f64,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBounds {
    pub confidence_level: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_power: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect_size: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofPack {
    pub reproduction_steps: Vec<ReproductionStep>,
    pub validation_method: ValidationMethod,
    pub expected_outcome: ExpectedOutcome,
    pub failure_modes: Vec<FailureMode>,
    pub confidence_bounds: ConfidenceBounds,
}

// =============================================================================
// IMPACT CLAIM
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BeneficiaryType {
    Individual,
    Community,
    Civilization,
    Ecosystem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beneficiary {
    pub beneficiary_type: BeneficiaryType,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_estimate: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub metric: String,
    pub baseline: f64,
    pub observed: f64,
    pub unit: String,
    pub methodology: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UncertaintyType {
    Aleatoric,
    Epistemic,
    Systemic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uncertainty {
    pub r#type: UncertaintyType,
    pub magnitude: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TimeHorizon {
    Immediate,
    Short,
    Medium,
    Long,
    Perpetual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactClaim {
    pub what_changed: String,
    pub who_benefits: Vec<Beneficiary>,
    pub how_measured: Vec<Measurement>,
    pub impact_score: f64,
    pub uncertainty: Vec<Uncertainty>,
    pub time_horizon: TimeHorizon,
}

// =============================================================================
// ETHICAL ENVELOPE
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmType {
    Direct,
    Indirect,
    Cascading,
    Existential,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Severity {
    Negligible,
    Low,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialHarm {
    pub r#type: HarmType,
    pub severity: Severity,
    pub likelihood: f64,
    pub affected_parties: Vec<String>,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmAnalysis {
    pub potential_harms: Vec<PotentialHarm>,
    pub net_harm_score: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MisuseDifficulty {
    Trivial,
    Easy,
    Moderate,
    Hard,
    Impossible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisuseRisk {
    pub attack_vectors: Vec<String>,
    pub difficulty: MisuseDifficulty,
    pub impact_if_successful: String,
    pub preventive_measures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLimits {
    pub valid_domains: Vec<String>,
    pub invalid_domains: Vec<String>,
    pub required_context: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reversibility {
    pub is_reversible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reversal_procedure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_hours: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanOverrideConditions {
    pub can_override: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_authority_level: Option<u8>,
    pub override_procedures: Vec<String>,
    pub audit_trail_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalEnvelope {
    pub harm_analysis: HarmAnalysis,
    pub misuse_risk: MisuseRisk,
    pub context_limits: ContextLimits,
    pub reversibility: Reversibility,
    pub human_override_conditions: HumanOverrideConditions,
    pub formal_assertions: Vec<String>, // SMT-LIB2 fragments
}

// =============================================================================
// SIGNATURES
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerificationType {
    Reproduction,
    Audit,
    FormalProof,
    EthicalReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSignature {
    pub verifier_node: String, // 64 hex
    pub signature: String,     // 128 hex
    pub verification_type: VerificationType,
    pub timestamp: u64, // Unix ms
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signatures {
    pub creator_signature: String, // 128 hex
    pub verifier_signatures: Vec<VerifierSignature>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_pool_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecation_signature: Option<String>,
}

// =============================================================================
// BLOCK BODY & FULL BLOCK
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBody {
    pub dependencies: Dependencies,
    pub proof_pack: ProofPack,
    pub impact_claim: ImpactClaim,
    pub ethical_envelope: EthicalEnvelope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BizraBlock {
    pub core: BlockCore,
    pub body: BlockBody,
    pub signatures: Signatures,
}

/// UnsignedBlock - the hash input (no block_id, no signatures)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsignedBlock {
    pub core: BlockCoreUnsigned,
    pub body: BlockBody,
}

impl BizraBlock {
    /// Extract the unsigned portion for hashing
    pub fn to_unsigned(&self) -> UnsignedBlock {
        UnsignedBlock {
            core: BlockCoreUnsigned::from(&self.core),
            body: self.body.clone(),
        }
    }
}

// =============================================================================
// VERDICT TYPES
// =============================================================================

/// The life/death judgment
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Verdict {
    /// Block is valid and accepted into the proof space
    Live,
    /// Block failed validation - cannot be accepted
    Dead,
    /// Block structure valid but awaiting verification
    Pending,
}

/// Detailed validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub verdict: Verdict,
    pub block_id: String,
    pub computed_block_id: String,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
    pub fate_scores: FateScores,
    pub validation_timestamp: u64,
}

/// FATE gate scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateScores {
    pub ihsan_score: f64,      // Excellence score (0..1)
    pub adl_score: f64,        // Justice/fairness score (0..1)
    pub harm_score: f64,       // Net harm (0..1, lower is better)
    pub confidence_score: f64, // Verification confidence (0..1)
}

// =============================================================================
// CANONICALIZATION (RFC 8785 JCS)
// =============================================================================

/// Canonicalize to JCS (RFC 8785) bytes
pub fn jcs_canonicalize<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    // First serialize to serde_json::Value
    let json_value = serde_json::to_value(value).map_err(|e| ValidationError::JsonError {
        reason: e.to_string(),
    })?;

    // Use json-canon for RFC 8785 canonicalization
    let canonical = json_canon::to_string(&json_value).map_err(|e| {
        ValidationError::CanonicalizationFailed {
            reason: e.to_string(),
        }
    })?;

    Ok(canonical.into_bytes())
}

/// Compute block_id = SHA-256(JCS(UnsignedBlock))
pub fn compute_block_id(unsigned: &UnsignedBlock) -> Result<String> {
    let canonical_bytes = jcs_canonicalize(unsigned)?;
    let mut hasher = Sha256::new();
    hasher.update(&canonical_bytes);
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

// =============================================================================
// VALIDATOR
// =============================================================================

/// The Proof Space Validator
pub struct ProofSpaceValidator {
    /// Strict mode enforces all FATE gates
    strict_mode: bool,
}

impl ProofSpaceValidator {
    pub fn new(strict_mode: bool) -> Self {
        ProofSpaceValidator { strict_mode }
    }

    /// Full validation - returns verdict
    pub fn validate(&self, block: &BizraBlock) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // 1. Validate structure
        if let Err(e) = self.validate_structure(block) {
            errors.push(e);
        }

        // 2. Validate field constraints
        errors.extend(self.validate_field_constraints(block));

        // 3. Compute and verify block_id
        let computed_block_id = match compute_block_id(&block.to_unsigned()) {
            Ok(id) => id,
            Err(e) => {
                errors.push(e);
                String::new()
            }
        };

        if !computed_block_id.is_empty() && computed_block_id != block.core.block_id {
            errors.push(ValidationError::BlockIdMismatch {
                computed: computed_block_id.clone(),
                declared: block.core.block_id.clone(),
            });
        }

        // 4. Verify signatures
        if let Err(e) = self.verify_creator_signature(block, &computed_block_id) {
            errors.push(e);
        }

        for (i, vs) in block.signatures.verifier_signatures.iter().enumerate() {
            if let Err(e) = self.verify_verifier_signature(vs, &computed_block_id) {
                errors.push(e);
                warnings.push(format!("Verifier signature {} invalid", i));
            }
        }

        // 5. Check dependencies
        errors.extend(self.validate_dependencies(block));

        // 6. Validate proof pack
        errors.extend(self.validate_proof_pack(block));

        // 7. Validate ethical envelope
        errors.extend(self.validate_ethical_envelope(block));

        // 8. Compute FATE scores
        let fate_scores = self.compute_fate_scores(block);

        // 9. FATE gate enforcement
        if self.strict_mode {
            errors.extend(self.enforce_fate_gates(&fate_scores));
        }

        // 10. Check verifier count for acceptance
        if block.core.status == BlockStatus::Accepted {
            let valid_verifiers = block
                .signatures
                .verifier_signatures
                .iter()
                .filter(|v| v.confidence >= MIN_CONFIDENCE)
                .count();
            if valid_verifiers < MIN_VERIFIER_SIGNATURES {
                errors.push(ValidationError::InsufficientVerifiers {
                    count: valid_verifiers,
                    required: MIN_VERIFIER_SIGNATURES,
                });
            }
        }

        // Determine verdict
        let verdict = if errors.is_empty() {
            if block.core.status == BlockStatus::Accepted {
                Verdict::Live
            } else {
                Verdict::Pending
            }
        } else {
            Verdict::Dead
        };

        ValidationResult {
            verdict,
            block_id: block.core.block_id.clone(),
            computed_block_id,
            errors,
            warnings,
            fate_scores,
            validation_timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    /// Validate basic structure
    fn validate_structure(&self, block: &BizraBlock) -> Result<()> {
        // Version check
        if block.core.version != SCHEMA_VERSION {
            return Err(ValidationError::VersionMismatch {
                expected: SCHEMA_VERSION.to_string(),
                actual: block.core.version.clone(),
            });
        }

        // Hash format checks
        if !HEX64_PATTERN.is_match(&block.core.block_id) {
            return Err(ValidationError::InvalidHashFormat {
                field: "core.block_id".to_string(),
            });
        }

        if !HEX64_PATTERN.is_match(&block.core.creator_node) {
            return Err(ValidationError::InvalidPublicKeyFormat {
                field: "core.creator_node".to_string(),
            });
        }

        if !HEX128_PATTERN.is_match(&block.signatures.creator_signature) {
            return Err(ValidationError::InvalidSignatureFormat {
                field: "signatures.creator_signature".to_string(),
            });
        }

        // Timestamp check
        if block.core.created_at == 0 {
            return Err(ValidationError::InvalidTimestamp {
                field: "core.created_at".to_string(),
            });
        }

        Ok(())
    }

    /// Validate all field length/range constraints
    fn validate_field_constraints(&self, block: &BizraBlock) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Dependencies
        for ext_ref in &block.body.dependencies.inputs.external_refs {
            if ext_ref.uri.len() > MAX_URI_LENGTH {
                errors.push(ValidationError::StringTooLong {
                    field: "external_ref.uri".to_string(),
                    actual: ext_ref.uri.len(),
                    max: MAX_URI_LENGTH,
                });
            }
            if let Some(ref hash) = ext_ref.hash {
                if !HEX64_PATTERN.is_match(hash) {
                    errors.push(ValidationError::InvalidHashFormat {
                        field: "external_ref.hash".to_string(),
                    });
                }
            }
        }

        for assumption in &block.body.dependencies.assumptions {
            if assumption.statement.len() > MAX_DESCRIPTION_STRING {
                errors.push(ValidationError::StringTooLong {
                    field: "assumption.statement".to_string(),
                    actual: assumption.statement.len(),
                    max: MAX_DESCRIPTION_STRING,
                });
            }
            if assumption.test_method.len() > MAX_DESCRIPTION_STRING {
                errors.push(ValidationError::StringTooLong {
                    field: "assumption.test_method".to_string(),
                    actual: assumption.test_method.len(),
                    max: MAX_DESCRIPTION_STRING,
                });
            }
        }

        // Proof pack
        for step in &block.body.proof_pack.reproduction_steps {
            if step.instruction.len() > MAX_INSTRUCTION_STRING {
                errors.push(ValidationError::StringTooLong {
                    field: "reproduction_step.instruction".to_string(),
                    actual: step.instruction.len(),
                    max: MAX_INSTRUCTION_STRING,
                });
            }
            if step.timeout_ms == 0 {
                errors.push(ValidationError::ValueOutOfRange {
                    field: "reproduction_step.timeout_ms".to_string(),
                    actual: step.timeout_ms as f64,
                    min: 1.0,
                    max: f64::MAX,
                });
            }
        }

        // Impact claim
        if block.body.impact_claim.what_changed.len() > MAX_LONG_STRING {
            errors.push(ValidationError::StringTooLong {
                field: "impact_claim.what_changed".to_string(),
                actual: block.body.impact_claim.what_changed.len(),
                max: MAX_LONG_STRING,
            });
        }

        if block.body.impact_claim.impact_score < 0.0 || block.body.impact_claim.impact_score > 1.0
        {
            errors.push(ValidationError::ValueOutOfRange {
                field: "impact_claim.impact_score".to_string(),
                actual: block.body.impact_claim.impact_score,
                min: 0.0,
                max: 1.0,
            });
        }

        // Ethical envelope
        if block.body.ethical_envelope.harm_analysis.net_harm_score < 0.0
            || block.body.ethical_envelope.harm_analysis.net_harm_score > 1.0
        {
            errors.push(ValidationError::ValueOutOfRange {
                field: "harm_analysis.net_harm_score".to_string(),
                actual: block.body.ethical_envelope.harm_analysis.net_harm_score,
                min: 0.0,
                max: 1.0,
            });
        }

        // Confidence bounds
        let conf = &block.body.proof_pack.confidence_bounds;
        if conf.confidence_level < 0.0 || conf.confidence_level > 1.0 {
            errors.push(ValidationError::ValueOutOfRange {
                field: "confidence_bounds.confidence_level".to_string(),
                actual: conf.confidence_level,
                min: 0.0,
                max: 1.0,
            });
        }

        errors
    }

    /// Validate dependencies for circular refs and duplicates
    fn validate_dependencies(&self, block: &BizraBlock) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let mut seen_refs = HashSet::new();

        for block_ref in &block.body.dependencies.inputs.block_refs {
            // Check format
            if !HEX64_PATTERN.is_match(block_ref) {
                errors.push(ValidationError::InvalidHashFormat {
                    field: format!("block_refs[{}]", block_ref),
                });
                continue;
            }

            // Check circular dependency
            if block_ref == &block.core.block_id {
                errors.push(ValidationError::CircularDependency {
                    block_id: block_ref.clone(),
                });
            }

            // Check duplicates
            if !seen_refs.insert(block_ref.clone()) {
                errors.push(ValidationError::DuplicateDependency {
                    block_id: block_ref.clone(),
                });
            }
        }

        errors
    }

    /// Validate proof pack structure
    fn validate_proof_pack(&self, block: &BizraBlock) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let steps = &block.body.proof_pack.reproduction_steps;

        // Check step numbers are strictly increasing
        if !steps.is_empty() {
            let mut prev = 0u32;
            for step in steps {
                if step.step_number <= prev {
                    errors.push(ValidationError::InvalidStepOrder {
                        details: format!(
                            "Step {} not greater than previous {}",
                            step.step_number, prev
                        ),
                    });
                }
                prev = step.step_number;
            }
        }

        // Validate expected outcome hash
        if !HEX64_PATTERN.is_match(&block.body.proof_pack.expected_outcome.result_hash) {
            errors.push(ValidationError::InvalidHashFormat {
                field: "expected_outcome.result_hash".to_string(),
            });
        }

        // Validate failure mode probabilities
        for fm in &block.body.proof_pack.failure_modes {
            if fm.probability < 0.0 || fm.probability > 1.0 {
                errors.push(ValidationError::ValueOutOfRange {
                    field: "failure_mode.probability".to_string(),
                    actual: fm.probability,
                    min: 0.0,
                    max: 1.0,
                });
            }
        }

        errors
    }

    /// Validate ethical envelope
    fn validate_ethical_envelope(&self, block: &BizraBlock) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let env = &block.body.ethical_envelope;

        // Validate harm likelihoods
        for harm in &env.harm_analysis.potential_harms {
            if harm.likelihood < 0.0 || harm.likelihood > 1.0 {
                errors.push(ValidationError::ValueOutOfRange {
                    field: "potential_harm.likelihood".to_string(),
                    actual: harm.likelihood,
                    min: 0.0,
                    max: 1.0,
                });
            }
        }

        // Validate reversibility consistency
        if env.reversibility.is_reversible && env.reversibility.reversal_procedure.is_none() {
            errors.push(ValidationError::MissingField {
                field: "reversibility.reversal_procedure (required when is_reversible=true)"
                    .to_string(),
            });
        }

        // Validate human override consistency
        if env.human_override_conditions.can_override {
            if let Some(level) = env.human_override_conditions.required_authority_level {
                if !(1..=10).contains(&level) {
                    errors.push(ValidationError::ValueOutOfRange {
                        field: "human_override_conditions.required_authority_level".to_string(),
                        actual: level as f64,
                        min: 1.0,
                        max: 10.0,
                    });
                }
            }
        }

        // Validate formal assertions (basic SMT-LIB2 check)
        for (i, assertion) in env.formal_assertions.iter().enumerate() {
            if !self.is_valid_smtlib2_fragment(assertion) {
                errors.push(ValidationError::InvalidFormalAssertion {
                    index: i,
                    reason: "Invalid SMT-LIB2 syntax".to_string(),
                });
            }
        }

        errors
    }

    /// Basic SMT-LIB2 syntax check
    fn is_valid_smtlib2_fragment(&self, assertion: &str) -> bool {
        // Normalize newlines
        let normalized = assertion.replace("\r\n", "\n").replace("\r", "\n");

        // Basic checks: must have balanced parentheses and start with '('
        let trimmed = normalized.trim();
        if trimmed.is_empty() {
            return true; // Empty is allowed
        }

        if !trimmed.starts_with('(') {
            return false;
        }

        let mut depth = 0i32;
        for c in trimmed.chars() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return false;
            }
        }

        depth == 0
    }

    /// Verify creator signature
    fn verify_creator_signature(&self, block: &BizraBlock, block_id: &str) -> Result<()> {
        let pk_bytes = hex::decode(&block.core.creator_node).map_err(|_| {
            ValidationError::InvalidPublicKeyFormat {
                field: "creator_node".to_string(),
            }
        })?;

        let sig_bytes = hex::decode(&block.signatures.creator_signature).map_err(|_| {
            ValidationError::InvalidSignatureFormat {
                field: "creator_signature".to_string(),
            }
        })?;

        let verifying_key = VerifyingKey::from_bytes(&pk_bytes.try_into().map_err(|_| {
            ValidationError::CryptoError {
                reason: "Invalid public key length".to_string(),
            }
        })?)
        .map_err(|e| ValidationError::CryptoError {
            reason: format!("Invalid public key: {}", e),
        })?;

        let signature = Signature::from_bytes(&sig_bytes.try_into().map_err(|_| {
            ValidationError::CryptoError {
                reason: "Invalid signature length".to_string(),
            }
        })?);

        // Sign the hash of the block_id (which is already SHA-256 of UnsignedBlock)
        let hash_bytes = hex::decode(block_id).map_err(|_| ValidationError::CryptoError {
            reason: "Invalid block_id hex".to_string(),
        })?;

        verifying_key
            .verify(&hash_bytes, &signature)
            .map_err(|_| ValidationError::InvalidCreatorSignature)
    }

    /// Verify a verifier signature
    fn verify_verifier_signature(&self, vs: &VerifierSignature, block_id: &str) -> Result<()> {
        let pk_bytes = hex::decode(&vs.verifier_node).map_err(|_| {
            ValidationError::InvalidPublicKeyFormat {
                field: "verifier_node".to_string(),
            }
        })?;

        let sig_bytes =
            hex::decode(&vs.signature).map_err(|_| ValidationError::InvalidSignatureFormat {
                field: "verifier_signature".to_string(),
            })?;

        let verifying_key = VerifyingKey::from_bytes(&pk_bytes.try_into().map_err(|_| {
            ValidationError::CryptoError {
                reason: "Invalid public key length".to_string(),
            }
        })?)
        .map_err(|e| ValidationError::CryptoError {
            reason: format!("Invalid verifier public key: {}", e),
        })?;

        let signature = Signature::from_bytes(&sig_bytes.try_into().map_err(|_| {
            ValidationError::CryptoError {
                reason: "Invalid signature length".to_string(),
            }
        })?);

        let hash_bytes = hex::decode(block_id).map_err(|_| ValidationError::CryptoError {
            reason: "Invalid block_id hex".to_string(),
        })?;

        verifying_key.verify(&hash_bytes, &signature).map_err(|_| {
            ValidationError::InvalidVerifierSignature {
                verifier: vs.verifier_node.clone(),
            }
        })
    }

    /// Compute FATE scores from block content
    fn compute_fate_scores(&self, block: &BizraBlock) -> FateScores {
        // Ihsan score: based on impact, confidence, and completeness
        let impact = block.body.impact_claim.impact_score;
        let confidence = block.body.proof_pack.confidence_bounds.confidence_level;
        let completeness = self.compute_completeness_score(block);
        let ihsan_score = (impact * 0.4 + confidence * 0.4 + completeness * 0.2).min(1.0);

        // Adl score: fairness of benefit distribution
        let adl_score = self.compute_fairness_score(block);

        // Harm score: direct from ethical envelope
        let harm_score = block.body.ethical_envelope.harm_analysis.net_harm_score;

        // Confidence score: average verifier confidence
        let confidence_score = if block.signatures.verifier_signatures.is_empty() {
            0.0
        } else {
            block
                .signatures
                .verifier_signatures
                .iter()
                .map(|v| v.confidence)
                .sum::<f64>()
                / block.signatures.verifier_signatures.len() as f64
        };

        FateScores {
            ihsan_score,
            adl_score,
            harm_score,
            confidence_score,
        }
    }

    /// Compute completeness score
    fn compute_completeness_score(&self, block: &BizraBlock) -> f64 {
        let mut score = 0.0;
        let mut total = 0.0;

        // Has dependencies
        total += 1.0;
        if !block.body.dependencies.inputs.block_refs.is_empty()
            || !block.body.dependencies.inputs.external_refs.is_empty()
        {
            score += 1.0;
        }

        // Has assumptions
        total += 1.0;
        if !block.body.dependencies.assumptions.is_empty() {
            score += 1.0;
        }

        // Has reproduction steps
        total += 1.0;
        if !block.body.proof_pack.reproduction_steps.is_empty() {
            score += 1.0;
        }

        // Has failure modes
        total += 1.0;
        if !block.body.proof_pack.failure_modes.is_empty() {
            score += 1.0;
        }

        // Has beneficiaries
        total += 1.0;
        if !block.body.impact_claim.who_benefits.is_empty() {
            score += 1.0;
        }

        // Has measurements
        total += 1.0;
        if !block.body.impact_claim.how_measured.is_empty() {
            score += 1.0;
        }

        // Has formal assertions
        total += 1.0;
        if !block.body.ethical_envelope.formal_assertions.is_empty() {
            score += 1.0;
        }

        if total > 0.0 {
            score / total
        } else {
            0.0
        }
    }

    /// Compute fairness score based on beneficiary distribution
    fn compute_fairness_score(&self, block: &BizraBlock) -> f64 {
        let beneficiaries = &block.body.impact_claim.who_benefits;
        if beneficiaries.is_empty() {
            return 0.5; // Neutral if no beneficiaries specified
        }

        // Higher score if benefits multiple beneficiary types
        let types: HashSet<_> = beneficiaries.iter().map(|b| b.beneficiary_type).collect();
        let type_diversity = types.len() as f64 / 4.0; // 4 possible types

        // Higher score for civilization/ecosystem level impact
        let has_broad_impact = types.contains(&BeneficiaryType::Civilization)
            || types.contains(&BeneficiaryType::Ecosystem);

        let broad_bonus = if has_broad_impact { 0.2 } else { 0.0 };

        (type_diversity * 0.8 + broad_bonus).min(1.0)
    }

    /// Enforce FATE gates
    fn enforce_fate_gates(&self, scores: &FateScores) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Ihsan gate
        if scores.ihsan_score < IHSAN_THRESHOLD {
            errors.push(ValidationError::FateViolation {
                gate: "IHSAN".to_string(),
                reason: format!(
                    "Score {:.3} < threshold {:.3}",
                    scores.ihsan_score, IHSAN_THRESHOLD
                ),
            });
        }

        // Harm gate
        if scores.harm_score > MAX_HARM_SCORE {
            errors.push(ValidationError::HarmScoreTooHigh {
                score: scores.harm_score,
                max: MAX_HARM_SCORE,
            });
        }

        errors
    }
}

impl Default for ProofSpaceValidator {
    fn default() -> Self {
        Self::new(true)
    }
}

// =============================================================================
// BLOCK BUILDER (for creating valid blocks)
// =============================================================================

/// Builder for creating properly formatted blocks
pub struct BlockBuilder {
    block_type: BlockType,
    creator_node: String,
    status: BlockStatus,
    parent_block: Option<String>,
    body: Option<BlockBody>,
}

impl BlockBuilder {
    pub fn new(block_type: BlockType, creator_node: String) -> Self {
        BlockBuilder {
            block_type,
            creator_node,
            status: BlockStatus::Draft,
            parent_block: None,
            body: None,
        }
    }

    pub fn status(mut self, status: BlockStatus) -> Self {
        self.status = status;
        self
    }

    pub fn parent_block(mut self, parent: String) -> Self {
        self.parent_block = Some(parent);
        self
    }

    pub fn body(mut self, body: BlockBody) -> Self {
        self.body = Some(body);
        self
    }

    /// Build the unsigned block and compute block_id
    pub fn build_unsigned(self) -> Result<(UnsignedBlock, String)> {
        let body = self.body.ok_or(ValidationError::MissingField {
            field: "body".to_string(),
        })?;

        let unsigned = UnsignedBlock {
            core: BlockCoreUnsigned {
                block_type: self.block_type,
                creator_node: self.creator_node,
                created_at: chrono::Utc::now().timestamp_millis() as u64,
                version: SCHEMA_VERSION.to_string(),
                status: self.status,
                parent_block: self.parent_block,
            },
            body,
        };

        let block_id = compute_block_id(&unsigned)?;
        Ok((unsigned, block_id))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_minimal_body() -> BlockBody {
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
                impact_score: 0.5,
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

    #[test]
    fn test_jcs_canonicalization() {
        let body = create_minimal_body();
        let unsigned = UnsignedBlock {
            core: BlockCoreUnsigned {
                block_type: BlockType::KnowledgeBlock,
                creator_node: "a".repeat(64),
                created_at: 1234567890000,
                version: "1.0.0".to_string(),
                status: BlockStatus::Draft,
                parent_block: None,
            },
            body,
        };

        let result = jcs_canonicalize(&unsigned);
        assert!(result.is_ok());

        // Same input should produce same output
        let bytes1 = result.unwrap();
        let bytes2 = jcs_canonicalize(&unsigned).unwrap();
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_compute_block_id_deterministic() {
        let body = create_minimal_body();
        let unsigned = UnsignedBlock {
            core: BlockCoreUnsigned {
                block_type: BlockType::KnowledgeBlock,
                creator_node: "a".repeat(64),
                created_at: 1234567890000,
                version: "1.0.0".to_string(),
                status: BlockStatus::Draft,
                parent_block: None,
            },
            body,
        };

        let id1 = compute_block_id(&unsigned).unwrap();
        let id2 = compute_block_id(&unsigned).unwrap();

        assert_eq!(id1, id2);
        assert!(HEX64_PATTERN.is_match(&id1));
    }

    #[test]
    fn test_hex_patterns() {
        assert!(HEX64_PATTERN.is_match(&"a".repeat(64)));
        assert!(!HEX64_PATTERN.is_match(&"A".repeat(64))); // Must be lowercase
        assert!(!HEX64_PATTERN.is_match(&"a".repeat(63)));
        assert!(!HEX64_PATTERN.is_match(&"a".repeat(65)));

        assert!(HEX128_PATTERN.is_match(&"b".repeat(128)));
        assert!(!HEX128_PATTERN.is_match(&"b".repeat(127)));
    }

    #[test]
    fn test_smtlib2_validation() {
        let validator = ProofSpaceValidator::new(true);

        // Valid assertions
        assert!(validator.is_valid_smtlib2_fragment("(assert (= x 1))"));
        assert!(validator.is_valid_smtlib2_fragment("(assert (and (> x 0) (< x 10)))"));
        assert!(validator.is_valid_smtlib2_fragment("")); // Empty allowed

        // Invalid assertions
        assert!(!validator.is_valid_smtlib2_fragment("assert (= x 1)")); // Missing open paren
        assert!(!validator.is_valid_smtlib2_fragment("(assert (= x 1)")); // Unbalanced
        assert!(!validator.is_valid_smtlib2_fragment("((assert)")); // Unbalanced
    }

    #[test]
    fn test_fate_scores_computation() {
        let body = create_minimal_body();
        let block = BizraBlock {
            core: BlockCore {
                block_id: "a".repeat(64),
                block_type: BlockType::KnowledgeBlock,
                creator_node: "b".repeat(64),
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
        };

        let validator = ProofSpaceValidator::new(false);
        let scores = validator.compute_fate_scores(&block);

        assert!(scores.ihsan_score >= 0.0 && scores.ihsan_score <= 1.0);
        assert!(scores.adl_score >= 0.0 && scores.adl_score <= 1.0);
        assert!(scores.harm_score >= 0.0 && scores.harm_score <= 1.0);
        assert_eq!(scores.confidence_score, 0.0); // No verifiers
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut body = create_minimal_body();
        let block_id = "a".repeat(64);
        body.dependencies.inputs.block_refs = vec![block_id.clone()];

        let block = BizraBlock {
            core: BlockCore {
                block_id: block_id.clone(),
                block_type: BlockType::KnowledgeBlock,
                creator_node: "b".repeat(64),
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
        };

        let validator = ProofSpaceValidator::new(false);
        let errors = validator.validate_dependencies(&block);

        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::CircularDependency { .. })));
    }

    #[test]
    fn test_step_order_validation() {
        let mut body = create_minimal_body();
        body.proof_pack.reproduction_steps = vec![
            ReproductionStep {
                step_number: 1,
                instruction: "First".to_string(),
                expected_hash: None,
                timeout_ms: 1000,
            },
            ReproductionStep {
                step_number: 1, // Invalid: not strictly increasing
                instruction: "Second".to_string(),
                expected_hash: None,
                timeout_ms: 1000,
            },
        ];

        let block = BizraBlock {
            core: BlockCore {
                block_id: "a".repeat(64),
                block_type: BlockType::KnowledgeBlock,
                creator_node: "b".repeat(64),
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
        };

        let validator = ProofSpaceValidator::new(false);
        let errors = validator.validate_proof_pack(&block);

        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidStepOrder { .. })));
    }

    #[test]
    fn test_block_builder() {
        let body = create_minimal_body();
        let result = BlockBuilder::new(BlockType::KnowledgeBlock, "a".repeat(64))
            .status(BlockStatus::Draft)
            .body(body)
            .build_unsigned();

        assert!(result.is_ok());
        let (unsigned, block_id) = result.unwrap();
        assert!(HEX64_PATTERN.is_match(&block_id));
        assert_eq!(unsigned.core.version, SCHEMA_VERSION);
    }
}
