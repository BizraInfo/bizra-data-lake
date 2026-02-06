//! Genesis Receipt — Cryptographic Bootstrap Attestation
//!
//! The GenesisReceipt captures the complete configuration state at chain
//! initialization, providing cryptographic proof of the initial conditions.
//!
//! # Security Properties
//!
//! - Dual signature scheme: Ed25519 (classical) + Dilithium-5 (post-quantum)
//! - Blake3 hashing for all configuration manifests
//! - Merkle root anchoring for decentralized storage (0G)
//! - BFT quorum validation (4/6 SAT validators)
//!
//! # Standing on Giants
//!
//! - **Lamport**: Distributed timestamp ordering
//! - **Bernstein**: Ed25519 signature scheme
//! - **NIST**: Dilithium-5 post-quantum cryptography

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::identity::{hex_decode, hex_encode};
use crate::IHSAN_THRESHOLD;

// ═══════════════════════════════════════════════════════════════════════════════
// Error Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors that can occur during genesis receipt operations
#[derive(Error, Debug, Clone)]
pub enum GenesisError {
    /// Invalid chain ID format
    #[error("Invalid chain ID: {0}")]
    InvalidChainId(String),

    /// Invalid node ID (not valid hex-encoded Ed25519 public key)
    #[error("Invalid node ID: expected 64-character hex string")]
    InvalidNodeId,

    /// Invalid timestamp format
    #[error("Invalid timestamp: expected RFC 3339 format, got {0}")]
    InvalidTimestamp(String),

    /// Ihsan threshold out of valid range
    #[error("Ihsan threshold {0} out of valid range [0.0, 1.0]")]
    InvalidIhsanThreshold(f32),

    /// Ihsan threshold below production minimum
    #[error("Ihsan threshold {actual} below production minimum {minimum}")]
    IhsanBelowMinimum { actual: f32, minimum: f32 },

    /// Invalid policy dimensions
    #[error("Invalid policy dimensions: expected {expected}, got {actual}")]
    InvalidPolicyDimensions { expected: u8, actual: u8 },

    /// SAT quorum exceeds validator count
    #[error("SAT quorum {quorum} exceeds validator count {validators}")]
    QuorumExceedsValidators { quorum: u8, validators: usize },

    /// Insufficient validators for BFT
    #[error("Insufficient validators: need at least {minimum} for BFT, got {actual}")]
    InsufficientValidators { minimum: usize, actual: usize },

    /// Quorum too low for BFT (need > 2/3)
    #[error("Quorum {quorum} too low for BFT: need at least {minimum} of {validators} validators")]
    QuorumTooLowForBFT {
        quorum: u8,
        minimum: usize,
        validators: usize,
    },

    /// Invalid classical signature
    #[error("Invalid classical signature: expected 64 bytes")]
    InvalidClassicalSignature,

    /// Invalid PQC signature
    #[error("Invalid PQC signature: expected Dilithium-5 signature")]
    InvalidPQCSignature,

    /// Hash verification failed
    #[error("Hash verification failed for {field}: expected {expected}, got {actual}")]
    HashMismatch {
        field: String,
        expected: String,
        actual: String,
    },

    /// Cryptographic suite mismatch
    #[error("Cryptographic suite mismatch: expected {expected}, got {actual}")]
    CryptoSuiteMismatch { expected: String, actual: String },

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Signature verification failed
    #[error("Signature verification failed: {0}")]
    SignatureInvalid(String),
}

/// Result type for genesis operations
pub type GenesisResult<T> = Result<T, GenesisError>;

// ═══════════════════════════════════════════════════════════════════════════════
// Execution Context
// ═══════════════════════════════════════════════════════════════════════════════

/// Execution environment for the genesis receipt
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionContext {
    /// Development mode: relaxed thresholds, verbose logging
    Development,
    /// Staging mode: production-like with additional monitoring
    Staging,
    /// Production mode: full constraints enforced
    Production,
    /// Critical mode: enhanced security for high-value operations
    Critical,
}

impl ExecutionContext {
    /// Returns the minimum Ihsan threshold for this context
    pub fn min_ihsan_threshold(&self) -> f32 {
        match self {
            ExecutionContext::Development => 0.80,
            ExecutionContext::Staging => 0.90,
            ExecutionContext::Production => 0.95,
            ExecutionContext::Critical => 0.98,
        }
    }

    /// Returns whether relaxed validation is allowed
    pub fn allows_relaxed_validation(&self) -> bool {
        matches!(self, ExecutionContext::Development)
    }

    /// Returns the minimum number of SAT validators required
    pub fn min_sat_validators(&self) -> usize {
        match self {
            ExecutionContext::Development => 1,
            ExecutionContext::Staging => 3,
            ExecutionContext::Production => 6,
            ExecutionContext::Critical => 9,
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        ExecutionContext::Production
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cryptographic Manifest
// ═══════════════════════════════════════════════════════════════════════════════

/// Cryptographic algorithms and parameters used in the genesis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoManifest {
    /// Classical signature algorithm (e.g., "ed25519")
    pub classical_sig: String,
    /// Post-quantum signature algorithm (e.g., "dilithium5")
    pub pqc_sig: String,
    /// Hash algorithm (e.g., "blake3")
    pub hash_algo: String,
    /// Key derivation function (e.g., "argon2id")
    pub kdf: String,
    /// Encryption algorithm for at-rest data (e.g., "xchacha20-poly1305")
    pub encryption: String,
    /// Minimum key size in bits
    pub min_key_bits: u16,
    /// NIST security level (1-5)
    pub nist_level: u8,
}

impl CryptoManifest {
    /// Create the default BIZRA cryptographic manifest
    pub fn bizra_default() -> Self {
        Self {
            classical_sig: "ed25519".to_string(),
            pqc_sig: "dilithium5".to_string(),
            hash_algo: "blake3".to_string(),
            kdf: "argon2id".to_string(),
            encryption: "xchacha20-poly1305".to_string(),
            min_key_bits: 256,
            nist_level: 5,
        }
    }

    /// Validate the cryptographic manifest
    pub fn validate(&self) -> GenesisResult<()> {
        // Validate classical signature
        if self.classical_sig != "ed25519" {
            return Err(GenesisError::CryptoSuiteMismatch {
                expected: "ed25519".to_string(),
                actual: self.classical_sig.clone(),
            });
        }

        // Validate PQC signature
        if self.pqc_sig != "dilithium5" {
            return Err(GenesisError::CryptoSuiteMismatch {
                expected: "dilithium5".to_string(),
                actual: self.pqc_sig.clone(),
            });
        }

        // Validate hash algorithm
        if self.hash_algo != "blake3" {
            return Err(GenesisError::CryptoSuiteMismatch {
                expected: "blake3".to_string(),
                actual: self.hash_algo.clone(),
            });
        }

        // Validate NIST level
        if self.nist_level < 3 {
            return Err(GenesisError::CryptoSuiteMismatch {
                expected: "NIST level >= 3".to_string(),
                actual: format!("NIST level {}", self.nist_level),
            });
        }

        Ok(())
    }

    /// Calculate Blake3 hash of the manifest
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(self.classical_sig.as_bytes());
        hasher.update(self.pqc_sig.as_bytes());
        hasher.update(self.hash_algo.as_bytes());
        hasher.update(self.kdf.as_bytes());
        hasher.update(self.encryption.as_bytes());
        hasher.update(&self.min_key_bits.to_le_bytes());
        hasher.update(&[self.nist_level]);
        *hasher.finalize().as_bytes()
    }
}

impl Default for CryptoManifest {
    fn default() -> Self {
        Self::bizra_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Genesis Receipt
// ═══════════════════════════════════════════════════════════════════════════════

/// The Genesis Receipt captures the complete configuration state at chain
/// initialization, providing cryptographic proof of the initial conditions.
///
/// # Fields
///
/// - **Identity**: Chain ID, node ID, and genesis timestamp
/// - **Configuration Hashes**: Blake3 hashes of workspace, policy, tools, and neural weights
/// - **Ihsan Configuration**: Excellence threshold and policy dimensions
/// - **Consensus**: SAT quorum and validator public keys
/// - **Cryptography**: Algorithm manifest
/// - **Anchoring**: Optional Merkle root for 0G storage
/// - **Seal**: Dual Ed25519 + Dilithium-5 signatures
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenesisReceipt {
    // ─────────────────────────────────────────────────────────────────────────
    // Identity
    // ─────────────────────────────────────────────────────────────────────────
    /// Chain identifier (e.g., "bizra-mainnet-v1")
    pub chain_id: String,
    /// Node ID as hex-encoded Ed25519 public key
    pub node_id: String,
    /// Genesis timestamp in RFC 3339 format
    pub genesis_utc: String,

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration Hashes (Blake3)
    // ─────────────────────────────────────────────────────────────────────────
    /// Blake3 hash of the workspace configuration
    #[serde(with = "hex_array_32")]
    pub workspace_hash: [u8; 32],
    /// Blake3 hash of the policy manifest
    #[serde(with = "hex_array_32")]
    pub policy_manifest_hash: [u8; 32],
    /// Blake3 hash of the tool allowlist
    #[serde(with = "hex_array_32")]
    pub tool_allowlist_hash: [u8; 32],
    /// Blake3 hash of the neural network weights
    #[serde(with = "hex_array_32")]
    pub neural_weights_hash: [u8; 32],

    // ─────────────────────────────────────────────────────────────────────────
    // Ihsan Configuration
    // ─────────────────────────────────────────────────────────────────────────
    /// Ihsan excellence threshold (0.95 for production)
    pub ihsan_threshold: f32,
    /// Number of policy dimensions (8 for full vector)
    pub policy_dimensions: u8,
    /// Execution context (Development, Staging, Production, Critical)
    pub execution_context: ExecutionContext,

    // ─────────────────────────────────────────────────────────────────────────
    // Consensus
    // ─────────────────────────────────────────────────────────────────────────
    /// SAT quorum threshold (e.g., 4 out of 6 for BFT)
    pub sat_quorum: u8,
    /// SAT validator public keys (Ed25519)
    #[serde(with = "hex_vec_32")]
    pub sat_validators: Vec<[u8; 32]>,

    // ─────────────────────────────────────────────────────────────────────────
    // Cryptography
    // ─────────────────────────────────────────────────────────────────────────
    /// Cryptographic algorithm manifest
    pub crypto: CryptoManifest,

    // ─────────────────────────────────────────────────────────────────────────
    // Anchoring
    // ─────────────────────────────────────────────────────────────────────────
    /// Optional Merkle root for 0G storage anchoring
    #[serde(with = "hex_option_32")]
    pub merkle_root_0g: Option<[u8; 32]>,

    // ─────────────────────────────────────────────────────────────────────────
    // Seal (Ed25519 + Dilithium-5 dual signature)
    // ─────────────────────────────────────────────────────────────────────────
    /// Classical Ed25519 signature (64 bytes)
    #[serde(with = "hex_array_64")]
    pub seal_classical: [u8; 64],
    /// Post-quantum Dilithium-5 signature
    #[serde(with = "hex_vec")]
    pub seal_pqc: Vec<u8>,
}

impl GenesisReceipt {
    /// Create a new unsigned genesis receipt builder
    pub fn builder() -> GenesisReceiptBuilder {
        GenesisReceiptBuilder::default()
    }

    /// Verify all invariants of the genesis receipt
    ///
    /// # Checks Performed
    ///
    /// 1. Chain ID format validation
    /// 2. Node ID format validation (64-character hex)
    /// 3. Timestamp format validation (RFC 3339)
    /// 4. Ihsan threshold range and context minimum
    /// 5. Policy dimensions (must be 8)
    /// 6. SAT quorum vs validator count
    /// 7. Minimum validators for BFT
    /// 8. Cryptographic manifest validation
    /// 9. Signature length validation
    ///
    /// Note: Cryptographic signature verification requires the signing keys
    /// and should be performed separately using `verify_signatures()`.
    pub fn verify(&self) -> GenesisResult<()> {
        // ─────────────────────────────────────────────────────────────────────
        // 1. Chain ID validation
        // ─────────────────────────────────────────────────────────────────────
        if self.chain_id.is_empty() {
            return Err(GenesisError::InvalidChainId(
                "chain_id cannot be empty".to_string(),
            ));
        }
        if !self.chain_id.starts_with("bizra-") {
            return Err(GenesisError::InvalidChainId(format!(
                "chain_id must start with 'bizra-', got '{}'",
                self.chain_id
            )));
        }

        // ─────────────────────────────────────────────────────────────────────
        // 2. Node ID validation (64-character hex = 32 bytes)
        // ─────────────────────────────────────────────────────────────────────
        if self.node_id.len() != 64 {
            return Err(GenesisError::InvalidNodeId);
        }
        if hex_decode(&self.node_id).is_err() {
            return Err(GenesisError::InvalidNodeId);
        }

        // ─────────────────────────────────────────────────────────────────────
        // 3. Timestamp validation (RFC 3339)
        // ─────────────────────────────────────────────────────────────────────
        if chrono::DateTime::parse_from_rfc3339(&self.genesis_utc).is_err() {
            return Err(GenesisError::InvalidTimestamp(self.genesis_utc.clone()));
        }

        // ─────────────────────────────────────────────────────────────────────
        // 4. Ihsan threshold validation
        // ─────────────────────────────────────────────────────────────────────
        if !(0.0..=1.0).contains(&self.ihsan_threshold) {
            return Err(GenesisError::InvalidIhsanThreshold(self.ihsan_threshold));
        }

        let min_threshold = self.execution_context.min_ihsan_threshold();
        if self.ihsan_threshold < min_threshold {
            return Err(GenesisError::IhsanBelowMinimum {
                actual: self.ihsan_threshold,
                minimum: min_threshold,
            });
        }

        // ─────────────────────────────────────────────────────────────────────
        // 5. Policy dimensions validation
        // ─────────────────────────────────────────────────────────────────────
        const EXPECTED_POLICY_DIMENSIONS: u8 = 8;
        if self.policy_dimensions != EXPECTED_POLICY_DIMENSIONS {
            return Err(GenesisError::InvalidPolicyDimensions {
                expected: EXPECTED_POLICY_DIMENSIONS,
                actual: self.policy_dimensions,
            });
        }

        // ─────────────────────────────────────────────────────────────────────
        // 6. SAT quorum validation
        // ─────────────────────────────────────────────────────────────────────
        if self.sat_quorum as usize > self.sat_validators.len() {
            return Err(GenesisError::QuorumExceedsValidators {
                quorum: self.sat_quorum,
                validators: self.sat_validators.len(),
            });
        }

        // ─────────────────────────────────────────────────────────────────────
        // 7. Minimum validators for BFT
        // ─────────────────────────────────────────────────────────────────────
        let min_validators = self.execution_context.min_sat_validators();
        if self.sat_validators.len() < min_validators {
            return Err(GenesisError::InsufficientValidators {
                minimum: min_validators,
                actual: self.sat_validators.len(),
            });
        }

        // Verify BFT quorum: need > 2/3 for Byzantine fault tolerance
        let bft_min_quorum = (self.sat_validators.len() * 2 / 3) + 1;
        if (self.sat_quorum as usize) < bft_min_quorum {
            return Err(GenesisError::QuorumTooLowForBFT {
                quorum: self.sat_quorum,
                minimum: bft_min_quorum,
                validators: self.sat_validators.len(),
            });
        }

        // ─────────────────────────────────────────────────────────────────────
        // 8. Cryptographic manifest validation
        // ─────────────────────────────────────────────────────────────────────
        self.crypto.validate()?;

        // ─────────────────────────────────────────────────────────────────────
        // 9. Signature length validation
        // ─────────────────────────────────────────────────────────────────────
        // Classical signature should be 64 bytes (Ed25519)
        // (array size enforces this at compile time)

        // PQC signature should be Dilithium-5 (~4627 bytes for signature)
        // Allow empty for unsigned receipts
        if !self.seal_pqc.is_empty() {
            const DILITHIUM5_SIG_MIN: usize = 4500;
            const DILITHIUM5_SIG_MAX: usize = 4700;
            if self.seal_pqc.len() < DILITHIUM5_SIG_MIN
                || self.seal_pqc.len() > DILITHIUM5_SIG_MAX
            {
                return Err(GenesisError::InvalidPQCSignature);
            }
        }

        Ok(())
    }

    /// Calculate the Blake3 hash of this genesis receipt (excluding signatures)
    ///
    /// This hash is used as the message for signature generation and verification.
    pub fn calculate_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();

        // Domain separation
        hasher.update(b"bizra-genesis-v1:");

        // Identity
        hasher.update(self.chain_id.as_bytes());
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.genesis_utc.as_bytes());

        // Configuration hashes
        hasher.update(&self.workspace_hash);
        hasher.update(&self.policy_manifest_hash);
        hasher.update(&self.tool_allowlist_hash);
        hasher.update(&self.neural_weights_hash);

        // Ihsan configuration
        hasher.update(&self.ihsan_threshold.to_le_bytes());
        hasher.update(&[self.policy_dimensions]);
        hasher.update(&[self.execution_context as u8]);

        // Consensus
        hasher.update(&[self.sat_quorum]);
        for validator in &self.sat_validators {
            hasher.update(validator);
        }

        // Crypto manifest hash
        hasher.update(&self.crypto.hash());

        // Merkle root (if present)
        if let Some(root) = &self.merkle_root_0g {
            hasher.update(&[1u8]); // Present flag
            hasher.update(root);
        } else {
            hasher.update(&[0u8]); // Absent flag
        }

        *hasher.finalize().as_bytes()
    }

    /// Get the hash as a hex string
    pub fn hash_hex(&self) -> String {
        hex_encode(&self.calculate_hash())
    }

    /// Check if the receipt has valid signatures (non-empty)
    pub fn is_signed(&self) -> bool {
        // Check if classical signature is non-zero
        let classical_non_zero = self.seal_classical.iter().any(|&b| b != 0);
        // Check if PQC signature is present
        let pqc_present = !self.seal_pqc.is_empty();

        classical_non_zero && pqc_present
    }

    /// Verify that the provided configuration hash matches
    pub fn verify_workspace_hash(&self, data: &[u8]) -> GenesisResult<()> {
        let computed = blake3::hash(data);
        if computed.as_bytes() != &self.workspace_hash {
            return Err(GenesisError::HashMismatch {
                field: "workspace_hash".to_string(),
                expected: hex_encode(&self.workspace_hash),
                actual: hex_encode(computed.as_bytes()),
            });
        }
        Ok(())
    }

    /// Verify that the provided policy manifest hash matches
    pub fn verify_policy_hash(&self, data: &[u8]) -> GenesisResult<()> {
        let computed = blake3::hash(data);
        if computed.as_bytes() != &self.policy_manifest_hash {
            return Err(GenesisError::HashMismatch {
                field: "policy_manifest_hash".to_string(),
                expected: hex_encode(&self.policy_manifest_hash),
                actual: hex_encode(computed.as_bytes()),
            });
        }
        Ok(())
    }

    /// Verify that the provided tool allowlist hash matches
    pub fn verify_tool_allowlist_hash(&self, data: &[u8]) -> GenesisResult<()> {
        let computed = blake3::hash(data);
        if computed.as_bytes() != &self.tool_allowlist_hash {
            return Err(GenesisError::HashMismatch {
                field: "tool_allowlist_hash".to_string(),
                expected: hex_encode(&self.tool_allowlist_hash),
                actual: hex_encode(computed.as_bytes()),
            });
        }
        Ok(())
    }

    /// Verify that the provided neural weights hash matches
    pub fn verify_neural_weights_hash(&self, data: &[u8]) -> GenesisResult<()> {
        let computed = blake3::hash(data);
        if computed.as_bytes() != &self.neural_weights_hash {
            return Err(GenesisError::HashMismatch {
                field: "neural_weights_hash".to_string(),
                expected: hex_encode(&self.neural_weights_hash),
                actual: hex_encode(computed.as_bytes()),
            });
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Builder Pattern
// ═══════════════════════════════════════════════════════════════════════════════

/// Builder for creating GenesisReceipt instances
#[derive(Default)]
pub struct GenesisReceiptBuilder {
    chain_id: Option<String>,
    node_id: Option<String>,
    genesis_utc: Option<String>,
    workspace_hash: Option<[u8; 32]>,
    policy_manifest_hash: Option<[u8; 32]>,
    tool_allowlist_hash: Option<[u8; 32]>,
    neural_weights_hash: Option<[u8; 32]>,
    ihsan_threshold: Option<f32>,
    policy_dimensions: Option<u8>,
    execution_context: Option<ExecutionContext>,
    sat_quorum: Option<u8>,
    sat_validators: Option<Vec<[u8; 32]>>,
    crypto: Option<CryptoManifest>,
    merkle_root_0g: Option<[u8; 32]>,
}

impl GenesisReceiptBuilder {
    /// Set the chain ID
    pub fn chain_id(mut self, id: impl Into<String>) -> Self {
        self.chain_id = Some(id.into());
        self
    }

    /// Set the node ID
    pub fn node_id(mut self, id: impl Into<String>) -> Self {
        self.node_id = Some(id.into());
        self
    }

    /// Set the genesis timestamp (RFC 3339)
    pub fn genesis_utc(mut self, ts: impl Into<String>) -> Self {
        self.genesis_utc = Some(ts.into());
        self
    }

    /// Set the genesis timestamp to now
    pub fn genesis_now(mut self) -> Self {
        self.genesis_utc = Some(chrono::Utc::now().to_rfc3339());
        self
    }

    /// Set the workspace hash
    pub fn workspace_hash(mut self, hash: [u8; 32]) -> Self {
        self.workspace_hash = Some(hash);
        self
    }

    /// Set the workspace hash from data
    pub fn workspace_data(mut self, data: &[u8]) -> Self {
        self.workspace_hash = Some(*blake3::hash(data).as_bytes());
        self
    }

    /// Set the policy manifest hash
    pub fn policy_manifest_hash(mut self, hash: [u8; 32]) -> Self {
        self.policy_manifest_hash = Some(hash);
        self
    }

    /// Set the policy manifest hash from data
    pub fn policy_manifest_data(mut self, data: &[u8]) -> Self {
        self.policy_manifest_hash = Some(*blake3::hash(data).as_bytes());
        self
    }

    /// Set the tool allowlist hash
    pub fn tool_allowlist_hash(mut self, hash: [u8; 32]) -> Self {
        self.tool_allowlist_hash = Some(hash);
        self
    }

    /// Set the tool allowlist hash from data
    pub fn tool_allowlist_data(mut self, data: &[u8]) -> Self {
        self.tool_allowlist_hash = Some(*blake3::hash(data).as_bytes());
        self
    }

    /// Set the neural weights hash
    pub fn neural_weights_hash(mut self, hash: [u8; 32]) -> Self {
        self.neural_weights_hash = Some(hash);
        self
    }

    /// Set the neural weights hash from data
    pub fn neural_weights_data(mut self, data: &[u8]) -> Self {
        self.neural_weights_hash = Some(*blake3::hash(data).as_bytes());
        self
    }

    /// Set the Ihsan threshold
    pub fn ihsan_threshold(mut self, threshold: f32) -> Self {
        self.ihsan_threshold = Some(threshold);
        self
    }

    /// Set the policy dimensions
    pub fn policy_dimensions(mut self, dims: u8) -> Self {
        self.policy_dimensions = Some(dims);
        self
    }

    /// Set the execution context
    pub fn execution_context(mut self, ctx: ExecutionContext) -> Self {
        self.execution_context = Some(ctx);
        self
    }

    /// Set the SAT quorum
    pub fn sat_quorum(mut self, quorum: u8) -> Self {
        self.sat_quorum = Some(quorum);
        self
    }

    /// Set the SAT validators
    pub fn sat_validators(mut self, validators: Vec<[u8; 32]>) -> Self {
        self.sat_validators = Some(validators);
        self
    }

    /// Add a SAT validator
    pub fn add_validator(mut self, validator: [u8; 32]) -> Self {
        self.sat_validators
            .get_or_insert_with(Vec::new)
            .push(validator);
        self
    }

    /// Set the cryptographic manifest
    pub fn crypto(mut self, manifest: CryptoManifest) -> Self {
        self.crypto = Some(manifest);
        self
    }

    /// Set the Merkle root for 0G anchoring
    pub fn merkle_root_0g(mut self, root: [u8; 32]) -> Self {
        self.merkle_root_0g = Some(root);
        self
    }

    /// Build an unsigned genesis receipt
    ///
    /// Returns a receipt with zero signatures that must be signed before use.
    pub fn build_unsigned(self) -> GenesisResult<GenesisReceipt> {
        let receipt = GenesisReceipt {
            chain_id: self
                .chain_id
                .ok_or_else(|| GenesisError::MissingField("chain_id".to_string()))?,
            node_id: self
                .node_id
                .ok_or_else(|| GenesisError::MissingField("node_id".to_string()))?,
            genesis_utc: self
                .genesis_utc
                .ok_or_else(|| GenesisError::MissingField("genesis_utc".to_string()))?,
            workspace_hash: self
                .workspace_hash
                .ok_or_else(|| GenesisError::MissingField("workspace_hash".to_string()))?,
            policy_manifest_hash: self
                .policy_manifest_hash
                .ok_or_else(|| GenesisError::MissingField("policy_manifest_hash".to_string()))?,
            tool_allowlist_hash: self
                .tool_allowlist_hash
                .ok_or_else(|| GenesisError::MissingField("tool_allowlist_hash".to_string()))?,
            neural_weights_hash: self
                .neural_weights_hash
                .ok_or_else(|| GenesisError::MissingField("neural_weights_hash".to_string()))?,
            ihsan_threshold: self.ihsan_threshold.unwrap_or(IHSAN_THRESHOLD as f32),
            policy_dimensions: self.policy_dimensions.unwrap_or(8),
            execution_context: self.execution_context.unwrap_or_default(),
            sat_quorum: self
                .sat_quorum
                .ok_or_else(|| GenesisError::MissingField("sat_quorum".to_string()))?,
            sat_validators: self
                .sat_validators
                .ok_or_else(|| GenesisError::MissingField("sat_validators".to_string()))?,
            crypto: self.crypto.unwrap_or_default(),
            merkle_root_0g: self.merkle_root_0g,
            seal_classical: [0u8; 64],
            seal_pqc: Vec::new(),
        };

        // Verify structure (but not signatures)
        receipt.verify()?;

        Ok(receipt)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Serde Helpers for Hex Encoding
// ═══════════════════════════════════════════════════════════════════════════════

mod hex_array_32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 64 {
            return Err(serde::de::Error::custom("expected 64 hex characters"));
        }
        let bytes: Result<Vec<u8>, _> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect();
        let bytes = bytes.map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid length"))
    }
}

mod hex_array_64 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 128 {
            return Err(serde::de::Error::custom("expected 128 hex characters"));
        }
        let bytes: Result<Vec<u8>, _> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect();
        let bytes = bytes.map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid length"))
    }
}

mod hex_vec_32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(items: &[[u8; 32]], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(items.len()))?;
        for bytes in items {
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            seq.serialize_element(&hex)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<[u8; 32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let strings: Vec<String> = Vec::deserialize(deserializer)?;
        strings
            .into_iter()
            .map(|s| {
                if s.len() != 64 {
                    return Err(serde::de::Error::custom("expected 64 hex characters"));
                }
                let bytes: Result<Vec<u8>, _> = (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
                    .collect();
                let bytes = bytes.map_err(serde::de::Error::custom)?;
                bytes
                    .try_into()
                    .map_err(|_| serde::de::Error::custom("invalid length"))
            })
            .collect()
    }
}

mod hex_option_32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(opt: &Option<[u8; 32]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match opt {
            Some(bytes) => {
                let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                serializer.serialize_some(&hex)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<[u8; 32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(s) => {
                if s.len() != 64 {
                    return Err(serde::de::Error::custom("expected 64 hex characters"));
                }
                let bytes: Result<Vec<u8>, _> = (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
                    .collect();
                let bytes = bytes.map_err(serde::de::Error::custom)?;
                let arr: [u8; 32] = bytes
                    .try_into()
                    .map_err(|_| serde::de::Error::custom("invalid length"))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }
}

mod hex_vec {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.is_empty() {
            return Ok(Vec::new());
        }
        if s.len() % 2 != 0 {
            return Err(serde::de::Error::custom("hex string must have even length"));
        }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(serde::de::Error::custom))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Calculate Blake3 hash of arbitrary data
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Calculate Blake3 hash with domain separation
pub fn blake3_domain_hash(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(b":");
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator() -> [u8; 32] {
        let mut v = [0u8; 32];
        v[0] = 1; // Non-zero to make it valid
        v
    }

    fn create_test_receipt() -> GenesisReceipt {
        let validators: Vec<[u8; 32]> = (0..6).map(|i| {
            let mut v = [0u8; 32];
            v[0] = (i + 1) as u8;
            v
        }).collect();

        // BFT requires > 2/3 quorum: for 6 validators, need at least 5 (6*2/3=4, need >4)
        GenesisReceipt::builder()
            .chain_id("bizra-testnet-v1")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_data(b"workspace config")
            .policy_manifest_data(b"policy manifest")
            .tool_allowlist_data(b"tool allowlist")
            .neural_weights_data(b"neural weights")
            .ihsan_threshold(0.95)
            .policy_dimensions(8)
            .execution_context(ExecutionContext::Production)
            .sat_quorum(5) // BFT: >2/3 of 6 = 5
            .sat_validators(validators)
            .build_unsigned()
            .expect("should build test receipt")
    }

    #[test]
    fn test_execution_context_thresholds() {
        assert_eq!(ExecutionContext::Development.min_ihsan_threshold(), 0.80);
        assert_eq!(ExecutionContext::Staging.min_ihsan_threshold(), 0.90);
        assert_eq!(ExecutionContext::Production.min_ihsan_threshold(), 0.95);
        assert_eq!(ExecutionContext::Critical.min_ihsan_threshold(), 0.98);
    }

    #[test]
    fn test_crypto_manifest_default() {
        let manifest = CryptoManifest::bizra_default();
        assert_eq!(manifest.classical_sig, "ed25519");
        assert_eq!(manifest.pqc_sig, "dilithium5");
        assert_eq!(manifest.hash_algo, "blake3");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_crypto_manifest_invalid() {
        let mut manifest = CryptoManifest::bizra_default();
        manifest.classical_sig = "rsa".to_string();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_genesis_receipt_builder() {
        let receipt = create_test_receipt();
        assert_eq!(receipt.chain_id, "bizra-testnet-v1");
        assert_eq!(receipt.ihsan_threshold, 0.95);
        assert_eq!(receipt.policy_dimensions, 8);
        assert_eq!(receipt.sat_quorum, 5); // BFT quorum
        assert_eq!(receipt.sat_validators.len(), 6);
    }

    #[test]
    fn test_genesis_receipt_verify() {
        let receipt = create_test_receipt();
        assert!(receipt.verify().is_ok());
    }

    #[test]
    fn test_genesis_receipt_invalid_chain_id() {
        let validators: Vec<[u8; 32]> = (0..6).map(|_| create_test_validator()).collect();

        let result = GenesisReceipt::builder()
            .chain_id("invalid-chain")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .sat_quorum(4)
            .sat_validators(validators)
            .build_unsigned();

        assert!(matches!(result, Err(GenesisError::InvalidChainId(_))));
    }

    #[test]
    fn test_genesis_receipt_invalid_node_id() {
        let validators: Vec<[u8; 32]> = (0..6).map(|_| create_test_validator()).collect();

        let result = GenesisReceipt::builder()
            .chain_id("bizra-testnet-v1")
            .node_id("short")
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .sat_quorum(4)
            .sat_validators(validators)
            .build_unsigned();

        assert!(matches!(result, Err(GenesisError::InvalidNodeId)));
    }

    #[test]
    fn test_genesis_receipt_ihsan_below_minimum() {
        let validators: Vec<[u8; 32]> = (0..6).map(|_| create_test_validator()).collect();

        let result = GenesisReceipt::builder()
            .chain_id("bizra-testnet-v1")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .ihsan_threshold(0.80) // Below production minimum
            .execution_context(ExecutionContext::Production)
            .sat_quorum(4)
            .sat_validators(validators)
            .build_unsigned();

        assert!(matches!(result, Err(GenesisError::IhsanBelowMinimum { .. })));
    }

    #[test]
    fn test_genesis_receipt_hash_deterministic() {
        let receipt = create_test_receipt();
        let hash1 = receipt.calculate_hash();
        let hash2 = receipt.calculate_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_genesis_receipt_hash_changes_with_data() {
        let receipt1 = create_test_receipt();
        let mut receipt2 = create_test_receipt();

        // Modify one receipt
        receipt2.ihsan_threshold = 0.96;

        assert_ne!(receipt1.calculate_hash(), receipt2.calculate_hash());
    }

    #[test]
    fn test_genesis_receipt_serialization() {
        let receipt = create_test_receipt();
        let json = serde_json::to_string_pretty(&receipt).expect("serialize");
        let parsed: GenesisReceipt = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(receipt.chain_id, parsed.chain_id);
        assert_eq!(receipt.node_id, parsed.node_id);
        assert_eq!(receipt.ihsan_threshold, parsed.ihsan_threshold);
        assert_eq!(receipt.workspace_hash, parsed.workspace_hash);
    }

    #[test]
    fn test_genesis_receipt_is_signed() {
        let receipt = create_test_receipt();
        assert!(!receipt.is_signed()); // Unsigned receipt
    }

    #[test]
    fn test_verify_workspace_hash() {
        let receipt = create_test_receipt();
        let data = b"workspace config";
        assert!(receipt.verify_workspace_hash(data).is_ok());
        assert!(receipt.verify_workspace_hash(b"wrong data").is_err());
    }

    #[test]
    fn test_blake3_domain_hash() {
        let hash1 = blake3_domain_hash("domain1", b"data");
        let hash2 = blake3_domain_hash("domain2", b"data");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_insufficient_validators() {
        let validators: Vec<[u8; 32]> = (0..3).map(|_| create_test_validator()).collect();

        let result = GenesisReceipt::builder()
            .chain_id("bizra-testnet-v1")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .execution_context(ExecutionContext::Production)
            .sat_quorum(2)
            .sat_validators(validators)
            .build_unsigned();

        assert!(matches!(result, Err(GenesisError::InsufficientValidators { .. })));
    }

    #[test]
    fn test_quorum_exceeds_validators() {
        let validators: Vec<[u8; 32]> = (0..6).map(|_| create_test_validator()).collect();

        let result = GenesisReceipt::builder()
            .chain_id("bizra-testnet-v1")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .sat_quorum(7) // More than validators
            .sat_validators(validators)
            .build_unsigned();

        assert!(matches!(result, Err(GenesisError::QuorumExceedsValidators { .. })));
    }

    #[test]
    fn test_development_context_relaxed() {
        let validators: Vec<[u8; 32]> = (0..1).map(|_| create_test_validator()).collect();

        let result = GenesisReceipt::builder()
            .chain_id("bizra-dev-v1")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .ihsan_threshold(0.80)
            .execution_context(ExecutionContext::Development)
            .sat_quorum(1)
            .sat_validators(validators)
            .build_unsigned();

        assert!(result.is_ok());
    }

    #[test]
    fn test_quorum_too_low_for_bft() {
        let validators: Vec<[u8; 32]> = (0..6).map(|_| create_test_validator()).collect();

        // 4/6 = 66.7%, but BFT requires >2/3, so need 5
        let result = GenesisReceipt::builder()
            .chain_id("bizra-testnet-v1")
            .node_id("a".repeat(64))
            .genesis_now()
            .workspace_hash([0u8; 32])
            .policy_manifest_hash([0u8; 32])
            .tool_allowlist_hash([0u8; 32])
            .neural_weights_hash([0u8; 32])
            .sat_quorum(4) // Too low for BFT with 6 validators
            .sat_validators(validators)
            .build_unsigned();

        assert!(matches!(result, Err(GenesisError::QuorumTooLowForBFT { .. })));
    }
}
