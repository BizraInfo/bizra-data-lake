// src/sovereignty/data.rs - Data Sovereignty (Pillar 2: Custody)
//
// Principle: Local-first storage by default. Encryption at rest,
// explicit export/import, explicit sharing. No "silent telemetry".

use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;

/// Data residency classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataResidency {
    /// Data stored locally on this node
    Local,
    /// Data replicated to federation nodes
    Federated,
    /// Data exported to external system (with consent)
    Exported,
    /// Data marked for deletion
    MarkedForDeletion,
}

/// Data category for access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataCategory {
    /// System configuration
    Config,
    /// User profile and preferences
    UserProfile,
    /// Receipts and audit logs
    Receipts,
    /// Patterns and learned behaviors
    Patterns,
    /// Evidence artifacts
    Evidence,
    /// Model weights and artifacts
    ModelArtifacts,
    /// Temporary working data
    Transient,
}

/// Data sovereignty rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSovereigntyRules {
    /// Allow federation sharing for this category
    pub allow_federation: bool,
    /// Require encryption at rest
    pub require_encryption: bool,
    /// Allow export to external systems
    pub allow_export: bool,
    /// Retention period in days (None = indefinite)
    pub retention_days: Option<u32>,
    /// Require explicit consent for access
    pub require_consent: bool,
}

impl DataSovereigntyRules {
    /// Default rules for each category
    pub fn for_category(category: DataCategory) -> Self {
        match category {
            DataCategory::Config => Self {
                allow_federation: false,
                require_encryption: true,
                allow_export: true,
                retention_days: None,
                require_consent: false,
            },
            DataCategory::UserProfile => Self {
                allow_federation: false,
                require_encryption: true,
                allow_export: true,
                retention_days: None,
                require_consent: true,
            },
            DataCategory::Receipts => Self {
                allow_federation: true,
                require_encryption: true,
                allow_export: true,
                retention_days: Some(365 * 7), // 7 years
                require_consent: false,
            },
            DataCategory::Patterns => Self {
                allow_federation: true, // Core of PFP
                require_encryption: false,
                allow_export: true,
                retention_days: None,
                require_consent: false,
            },
            DataCategory::Evidence => Self {
                allow_federation: true,
                require_encryption: true,
                allow_export: true,
                retention_days: Some(365 * 7), // 7 years
                require_consent: false,
            },
            DataCategory::ModelArtifacts => Self {
                allow_federation: false, // Local only
                require_encryption: false,
                allow_export: false,
                retention_days: None,
                require_consent: false,
            },
            DataCategory::Transient => Self {
                allow_federation: false,
                require_encryption: false,
                allow_export: false,
                retention_days: Some(1), // 24 hours
                require_consent: false,
            },
        }
    }
}

/// Data export request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExportRequest {
    /// Requester identity
    pub requester_id: String,
    /// Categories to export
    pub categories: Vec<DataCategory>,
    /// Export format
    pub format: ExportFormat,
    /// Reason for export
    pub reason: String,
    /// Consent token (if required)
    pub consent_token: Option<String>,
}

/// Export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON-LD (linked data)
    JsonLd,
    /// Plain JSON
    Json,
    /// Protocol Buffers
    Protobuf,
    /// Encrypted archive
    EncryptedArchive,
}

/// Data export result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExportResult {
    /// Export ID
    pub export_id: String,
    /// Categories exported
    pub categories_exported: Vec<DataCategory>,
    /// Total size in bytes
    pub total_bytes: u64,
    /// Export path (if file)
    pub export_path: Option<PathBuf>,
    /// Export timestamp
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// Integrity hash
    pub integrity_hash: String,
}

/// Data deletion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDeletionRequest {
    /// Requester identity
    pub requester_id: String,
    /// Categories to delete
    pub categories: Vec<DataCategory>,
    /// Reason for deletion
    pub reason: String,
    /// Consent token (required)
    pub consent_token: String,
}

/// Data deletion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDeletionResult {
    /// Deletion ID
    pub deletion_id: String,
    /// Categories deleted
    pub categories_deleted: Vec<DataCategory>,
    /// Items deleted count
    pub items_deleted: u64,
    /// Deletion timestamp
    pub deleted_at: chrono::DateTime<chrono::Utc>,
    /// Verification hash (proof of deletion)
    pub verification_hash: String,
}

/// Telemetry policy (anti-telemetry by default)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPolicy {
    /// Allow any telemetry
    pub enabled: bool,
    /// Allowed endpoints (empty = none)
    pub allowed_endpoints: Vec<String>,
    /// Data categories that can be telemetered
    pub allowed_categories: Vec<DataCategory>,
    /// Require explicit opt-in
    pub require_opt_in: bool,
}

impl Default for TelemetryPolicy {
    fn default() -> Self {
        // ANTI-TELEMETRY: Disabled by default
        Self {
            enabled: false,
            allowed_endpoints: Vec::new(),
            allowed_categories: Vec::new(),
            require_opt_in: true,
        }
    }
}

impl TelemetryPolicy {
    /// Check if a telemetry request is allowed
    pub fn is_allowed(&self, endpoint: &str, category: DataCategory) -> bool {
        if !self.enabled {
            return false;
        }

        if !self.allowed_endpoints.iter().any(|e| e == endpoint) {
            return false;
        }

        if !self.allowed_categories.contains(&category) {
            return false;
        }

        true
    }
}

/// Data custody manager
pub struct DataCustodian {
    /// Rules per category
    rules: HashMap<DataCategory, DataSovereigntyRules>,
    /// Telemetry policy
    telemetry_policy: TelemetryPolicy,
    /// Local storage root
    storage_root: PathBuf,
}

impl DataCustodian {
    /// Create with default rules
    pub fn new(storage_root: PathBuf) -> Self {
        let mut rules = HashMap::new();
        for category in [
            DataCategory::Config,
            DataCategory::UserProfile,
            DataCategory::Receipts,
            DataCategory::Patterns,
            DataCategory::Evidence,
            DataCategory::ModelArtifacts,
            DataCategory::Transient,
        ] {
            rules.insert(category, DataSovereigntyRules::for_category(category));
        }

        Self {
            rules,
            telemetry_policy: TelemetryPolicy::default(),
            storage_root,
        }
    }

    /// Check if data can be federated
    pub fn can_federate(&self, category: DataCategory) -> bool {
        self.rules
            .get(&category)
            .map(|r| r.allow_federation)
            .unwrap_or(false)
    }

    /// Check if data can be exported
    pub fn can_export(&self, category: DataCategory) -> bool {
        self.rules
            .get(&category)
            .map(|r| r.allow_export)
            .unwrap_or(false)
    }

    /// Check if telemetry is allowed
    pub fn is_telemetry_allowed(&self, endpoint: &str, category: DataCategory) -> bool {
        self.telemetry_policy.is_allowed(endpoint, category)
    }

    /// Get storage path for category
    pub fn storage_path(&self, category: DataCategory) -> PathBuf {
        let subdir = match category {
            DataCategory::Config => "config",
            DataCategory::UserProfile => "profiles",
            DataCategory::Receipts => "receipts",
            DataCategory::Patterns => "patterns",
            DataCategory::Evidence => "evidence",
            DataCategory::ModelArtifacts => "models",
            DataCategory::Transient => "temp",
        };

        self.storage_root.join(subdir)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENCRYPTION AT REST (P0 Gap Fix)
// ═══════════════════════════════════════════════════════════════════════════════

/// Encryption key derivation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivation {
    /// Direct key (32 bytes)
    Direct,
    /// Derived from passphrase via SHA256
    Passphrase,
    /// Derived from node identity key
    IdentityDerived,
}

/// Encrypted data envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedEnvelope {
    /// Format version
    pub version: u8,
    /// Key derivation method used
    pub key_derivation: KeyDerivation,
    /// Nonce (12 bytes, hex encoded)
    pub nonce: String,
    /// Ciphertext (hex encoded)
    pub ciphertext: String,
    /// Data category (for key selection)
    pub category: DataCategory,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Integrity tag (first 8 bytes of SHA256 of plaintext)
    pub integrity_tag: String,
}

/// Encryption at rest manager
pub struct EncryptionManager {
    /// Master key (32 bytes)
    master_key: [u8; 32],
    /// Category-specific derived keys
    derived_keys: HashMap<DataCategory, [u8; 32]>,
}

impl EncryptionManager {
    /// Create from master key bytes
    pub fn from_key(master_key: [u8; 32]) -> Self {
        let mut derived_keys = HashMap::new();

        // Derive category-specific keys using HKDF-like construction
        for category in [
            DataCategory::Config,
            DataCategory::UserProfile,
            DataCategory::Receipts,
            DataCategory::Patterns,
            DataCategory::Evidence,
            DataCategory::ModelArtifacts,
            DataCategory::Transient,
        ] {
            let category_name = format!("{:?}", category);
            let derived = Self::derive_key(&master_key, category_name.as_bytes());
            derived_keys.insert(category, derived);
        }

        Self {
            master_key,
            derived_keys,
        }
    }

    /// Create from passphrase
    pub fn from_passphrase(passphrase: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"BIZRA-SOVEREIGNTY-KEY-V1:");
        hasher.update(passphrase.as_bytes());
        let key: [u8; 32] = hasher.finalize().into();
        Self::from_key(key)
    }

    /// Derive a key for specific purpose
    fn derive_key(master: &[u8; 32], context: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(master);
        hasher.update(b":");
        hasher.update(context);
        hasher.finalize().into()
    }

    /// Get key for category
    fn key_for(&self, category: DataCategory) -> &[u8; 32] {
        self.derived_keys.get(&category).unwrap_or(&self.master_key)
    }

    /// Encrypt data for a category
    pub fn encrypt(
        &self,
        plaintext: &[u8],
        category: DataCategory,
    ) -> Result<EncryptedEnvelope, EncryptionError> {
        let key = self.key_for(category);
        let cipher =
            ChaCha20Poly1305::new_from_slice(key).map_err(|_| EncryptionError::InvalidKey)?;

        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|_| EncryptionError::EncryptionFailed)?;

        // Compute integrity tag
        let mut hasher = Sha256::new();
        hasher.update(plaintext);
        let hash = hasher.finalize();
        let integrity_tag = hex::encode(&hash[..8]);

        Ok(EncryptedEnvelope {
            version: 1,
            key_derivation: KeyDerivation::IdentityDerived,
            nonce: hex::encode(nonce.as_slice()),
            ciphertext: hex::encode(&ciphertext),
            category,
            created_at: chrono::Utc::now(),
            integrity_tag,
        })
    }

    /// Decrypt an envelope
    pub fn decrypt(&self, envelope: &EncryptedEnvelope) -> Result<Vec<u8>, EncryptionError> {
        let key = self.key_for(envelope.category);
        let cipher =
            ChaCha20Poly1305::new_from_slice(key).map_err(|_| EncryptionError::InvalidKey)?;

        let nonce_bytes =
            hex::decode(&envelope.nonce).map_err(|_| EncryptionError::InvalidNonce)?;
        if nonce_bytes.len() != 12 {
            return Err(EncryptionError::InvalidNonce);
        }
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext =
            hex::decode(&envelope.ciphertext).map_err(|_| EncryptionError::InvalidCiphertext)?;

        let plaintext = cipher
            .decrypt(nonce, ciphertext.as_slice())
            .map_err(|_| EncryptionError::DecryptionFailed)?;

        // Verify integrity tag
        let mut hasher = Sha256::new();
        hasher.update(&plaintext);
        let hash = hasher.finalize();
        let computed_tag = hex::encode(&hash[..8]);

        if computed_tag != envelope.integrity_tag {
            return Err(EncryptionError::IntegrityCheckFailed);
        }

        Ok(plaintext)
    }

    /// Encrypt and serialize to JSON
    pub fn encrypt_json<T: Serialize>(
        &self,
        data: &T,
        category: DataCategory,
    ) -> Result<String, EncryptionError> {
        let plaintext =
            serde_json::to_vec(data).map_err(|_| EncryptionError::SerializationFailed)?;
        let envelope = self.encrypt(&plaintext, category)?;
        serde_json::to_string(&envelope).map_err(|_| EncryptionError::SerializationFailed)
    }

    /// Decrypt and deserialize from JSON
    pub fn decrypt_json<T: for<'de> Deserialize<'de>>(
        &self,
        json: &str,
    ) -> Result<T, EncryptionError> {
        let envelope: EncryptedEnvelope =
            serde_json::from_str(json).map_err(|_| EncryptionError::DeserializationFailed)?;
        let plaintext = self.decrypt(&envelope)?;
        serde_json::from_slice(&plaintext).map_err(|_| EncryptionError::DeserializationFailed)
    }
}

/// Encryption errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionError {
    /// Invalid encryption key
    InvalidKey,
    /// Invalid nonce
    InvalidNonce,
    /// Invalid ciphertext
    InvalidCiphertext,
    /// Encryption failed
    EncryptionFailed,
    /// Decryption failed (wrong key or corrupted)
    DecryptionFailed,
    /// Integrity check failed
    IntegrityCheckFailed,
    /// Serialization failed
    SerializationFailed,
    /// Deserialization failed
    DeserializationFailed,
}

impl std::fmt::Display for EncryptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKey => write!(f, "Invalid encryption key"),
            Self::InvalidNonce => write!(f, "Invalid nonce"),
            Self::InvalidCiphertext => write!(f, "Invalid ciphertext"),
            Self::EncryptionFailed => write!(f, "Encryption failed"),
            Self::DecryptionFailed => write!(f, "Decryption failed"),
            Self::IntegrityCheckFailed => write!(f, "Integrity check failed"),
            Self::SerializationFailed => write!(f, "Serialization failed"),
            Self::DeserializationFailed => write!(f, "Deserialization failed"),
        }
    }
}

impl std::error::Error for EncryptionError {}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA EXPORT API (P1: GDPR/Portability Compliance)
// ═══════════════════════════════════════════════════════════════════════════════

/// Comprehensive data export package for portability/GDPR compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExportPackage {
    /// Export metadata
    pub metadata: ExportMetadata,
    /// User profile data
    pub user_profile: Option<ExportedUserProfile>,
    /// Configuration data
    pub configuration: Option<serde_json::Value>,
    /// Receipts and audit trail
    pub receipts: Vec<ExportedReceipt>,
    /// Patterns and learned preferences
    pub patterns: Vec<ExportedPattern>,
    /// Evidence artifacts
    pub evidence: Vec<ExportedEvidence>,
    /// Data processing activities (GDPR Article 30)
    pub processing_activities: Vec<ProcessingActivity>,
    /// Integrity manifest
    pub integrity: ExportIntegrity,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Export ID (unique identifier)
    pub export_id: String,
    /// Export version
    pub version: String,
    /// Requester identity
    pub requester_id: String,
    /// Export timestamp
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// Categories included
    pub categories: Vec<DataCategory>,
    /// Export reason
    pub reason: String,
    /// Format
    pub format: ExportFormat,
    /// Node that performed export
    pub node_id: String,
    /// Node fingerprint (for verification)
    pub node_fingerprint: String,
}

/// Exported user profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedUserProfile {
    /// User ID
    pub user_id: String,
    /// Display name
    pub display_name: Option<String>,
    /// Preferences
    pub preferences: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Exported receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedReceipt {
    /// Receipt ID
    pub receipt_id: String,
    /// Receipt type
    pub receipt_type: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Summary
    pub summary: String,
    /// Integrity hash
    pub integrity_hash: String,
}

/// Exported pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedPattern {
    /// Pattern ID
    pub pattern_id: String,
    /// Pattern name
    pub name: String,
    /// Activation count
    pub activation_count: u64,
    /// Last activated
    pub last_activated: Option<chrono::DateTime<chrono::Utc>>,
    /// Is elevated
    pub elevated: bool,
}

/// Exported evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedEvidence {
    /// Evidence ID
    pub evidence_id: String,
    /// Evidence type
    pub evidence_type: String,
    /// Path
    pub path: String,
    /// Hash
    pub hash: String,
    /// Size bytes
    pub size_bytes: u64,
}

/// Processing activity record (GDPR Article 30)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivity {
    /// Activity ID
    pub activity_id: String,
    /// Purpose of processing
    pub purpose: String,
    /// Legal basis
    pub legal_basis: String,
    /// Data categories processed
    pub categories: Vec<DataCategory>,
    /// Recipients (if any)
    pub recipients: Vec<String>,
    /// Retention period description
    pub retention: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Export integrity manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportIntegrity {
    /// SHA256 hash of all data sections
    pub package_hash: String,
    /// Merkle root of individual items
    pub merkle_root: String,
    /// Signing node ID
    pub signed_by: String,
    /// Signature (hex)
    pub signature: String,
}

/// Data Export Service
pub struct DataExportService {
    /// Data custodian
    custodian: DataCustodian,
    /// Encryption manager (optional)
    encryption: Option<EncryptionManager>,
    /// Node ID
    node_id: String,
    /// Node fingerprint
    node_fingerprint: String,
}

impl DataExportService {
    /// Create new export service
    pub fn new(storage_root: PathBuf, node_id: String, node_fingerprint: String) -> Self {
        Self {
            custodian: DataCustodian::new(storage_root),
            encryption: None,
            node_id,
            node_fingerprint,
        }
    }

    /// Enable encryption for exports
    pub fn with_encryption(mut self, manager: EncryptionManager) -> Self {
        self.encryption = Some(manager);
        self
    }

    /// Generate a full data export package
    pub fn export_all(
        &self,
        request: &DataExportRequest,
    ) -> Result<DataExportPackage, DataExportError> {
        // Validate request
        for category in &request.categories {
            if !self.custodian.can_export(*category) {
                return Err(DataExportError::CategoryNotExportable(*category));
            }

            let rules = DataSovereigntyRules::for_category(*category);
            if rules.require_consent && request.consent_token.is_none() {
                return Err(DataExportError::ConsentRequired(*category));
            }
        }

        let export_id = format!("export-{}", uuid::Uuid::new_v4());
        let now = chrono::Utc::now();

        // Build export package
        let metadata = ExportMetadata {
            export_id: export_id.clone(),
            version: "1.0".to_string(),
            requester_id: request.requester_id.clone(),
            exported_at: now,
            categories: request.categories.clone(),
            reason: request.reason.clone(),
            format: request.format,
            node_id: self.node_id.clone(),
            node_fingerprint: self.node_fingerprint.clone(),
        };

        // Collect data for each category
        let user_profile = if request.categories.contains(&DataCategory::UserProfile) {
            Some(self.collect_user_profile(&request.requester_id)?)
        } else {
            None
        };

        let configuration = if request.categories.contains(&DataCategory::Config) {
            Some(self.collect_configuration()?)
        } else {
            None
        };

        let receipts = if request.categories.contains(&DataCategory::Receipts) {
            self.collect_receipts(&request.requester_id)?
        } else {
            Vec::new()
        };

        let patterns = if request.categories.contains(&DataCategory::Patterns) {
            self.collect_patterns(&request.requester_id)?
        } else {
            Vec::new()
        };

        let evidence = if request.categories.contains(&DataCategory::Evidence) {
            self.collect_evidence(&request.requester_id)?
        } else {
            Vec::new()
        };

        let processing_activities = self.get_processing_activities(&request.requester_id)?;

        // Calculate integrity
        let integrity = self.compute_integrity(&metadata, &receipts, &patterns)?;

        Ok(DataExportPackage {
            metadata,
            user_profile,
            configuration,
            receipts,
            patterns,
            evidence,
            processing_activities,
            integrity,
        })
    }

    /// Export to JSON string
    pub fn export_to_json(&self, request: &DataExportRequest) -> Result<String, DataExportError> {
        let package = self.export_all(request)?;
        serde_json::to_string_pretty(&package).map_err(|_| DataExportError::SerializationFailed)
    }

    /// Export to encrypted JSON (if encryption enabled)
    pub fn export_to_encrypted_json(
        &self,
        request: &DataExportRequest,
    ) -> Result<String, DataExportError> {
        let encryption = self
            .encryption
            .as_ref()
            .ok_or(DataExportError::EncryptionNotEnabled)?;

        let package = self.export_all(request)?;
        encryption
            .encrypt_json(&package, DataCategory::UserProfile)
            .map_err(|_| DataExportError::EncryptionFailed)
    }

    // Private collection methods (stubs - to be connected to actual storage)

    fn collect_user_profile(&self, user_id: &str) -> Result<ExportedUserProfile, DataExportError> {
        Ok(ExportedUserProfile {
            user_id: user_id.to_string(),
            display_name: None,
            preferences: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }

    fn collect_configuration(&self) -> Result<serde_json::Value, DataExportError> {
        Ok(serde_json::json!({
            "version": "1.0",
            "sovereignty_mode": "local_first",
            "federation_enabled": false,
        }))
    }

    fn collect_receipts(&self, _user_id: &str) -> Result<Vec<ExportedReceipt>, DataExportError> {
        // TODO: Connect to actual receipt storage
        Ok(Vec::new())
    }

    fn collect_patterns(&self, _user_id: &str) -> Result<Vec<ExportedPattern>, DataExportError> {
        // TODO: Connect to actual pattern storage
        Ok(Vec::new())
    }

    fn collect_evidence(&self, _user_id: &str) -> Result<Vec<ExportedEvidence>, DataExportError> {
        // TODO: Connect to actual evidence storage
        Ok(Vec::new())
    }

    fn get_processing_activities(
        &self,
        _user_id: &str,
    ) -> Result<Vec<ProcessingActivity>, DataExportError> {
        // Standard BIZRA processing activities
        Ok(vec![
            ProcessingActivity {
                activity_id: "pa-001".to_string(),
                purpose: "PAT-SAT dual-agentic task execution".to_string(),
                legal_basis: "Legitimate interest / User consent".to_string(),
                categories: vec![DataCategory::Receipts, DataCategory::Patterns],
                recipients: Vec::new(),
                retention: "7 years for receipts, indefinite for patterns".to_string(),
                timestamp: chrono::Utc::now(),
            },
            ProcessingActivity {
                activity_id: "pa-002".to_string(),
                purpose: "Pattern federation for distributed learning".to_string(),
                legal_basis: "User consent".to_string(),
                categories: vec![DataCategory::Patterns],
                recipients: vec!["Federation nodes (if enabled)".to_string()],
                retention: "Indefinite".to_string(),
                timestamp: chrono::Utc::now(),
            },
        ])
    }

    fn compute_integrity(
        &self,
        metadata: &ExportMetadata,
        receipts: &[ExportedReceipt],
        patterns: &[ExportedPattern],
    ) -> Result<ExportIntegrity, DataExportError> {
        let mut hasher = Sha256::new();

        // Hash metadata
        hasher.update(
            serde_json::to_string(metadata)
                .unwrap_or_default()
                .as_bytes(),
        );

        // Hash receipts
        for receipt in receipts {
            hasher.update(receipt.receipt_id.as_bytes());
            hasher.update(receipt.integrity_hash.as_bytes());
        }

        // Hash patterns
        for pattern in patterns {
            hasher.update(pattern.pattern_id.as_bytes());
        }

        let hash = hasher.finalize();
        let package_hash = hex::encode(&hash);

        // Simplified merkle root (just the package hash for now)
        let merkle_root = package_hash.clone();

        Ok(ExportIntegrity {
            package_hash,
            merkle_root,
            signed_by: self.node_id.clone(),
            signature: "unsigned".to_string(), // TODO: Sign with node key
        })
    }
}

/// Data export errors
#[derive(Debug, Clone)]
pub enum DataExportError {
    /// Category not exportable
    CategoryNotExportable(DataCategory),
    /// Consent required for category
    ConsentRequired(DataCategory),
    /// Serialization failed
    SerializationFailed,
    /// Encryption not enabled
    EncryptionNotEnabled,
    /// Encryption failed
    EncryptionFailed,
    /// Storage error
    StorageError(String),
}

impl std::fmt::Display for DataExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CategoryNotExportable(c) => write!(f, "Category {:?} is not exportable", c),
            Self::ConsentRequired(c) => write!(f, "Consent required for category {:?}", c),
            Self::SerializationFailed => write!(f, "Serialization failed"),
            Self::EncryptionNotEnabled => write!(f, "Encryption not enabled"),
            Self::EncryptionFailed => write!(f, "Encryption failed"),
            Self::StorageError(s) => write!(f, "Storage error: {}", s),
        }
    }
}

impl std::error::Error for DataExportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_telemetry_disabled() {
        let policy = TelemetryPolicy::default();
        assert!(!policy.enabled);
        assert!(!policy.is_allowed("https://example.com", DataCategory::Config));
    }

    #[test]
    fn test_data_rules_patterns_allow_federation() {
        let rules = DataSovereigntyRules::for_category(DataCategory::Patterns);
        assert!(rules.allow_federation);
    }

    #[test]
    fn test_data_rules_model_no_federation() {
        let rules = DataSovereigntyRules::for_category(DataCategory::ModelArtifacts);
        assert!(!rules.allow_federation);
        assert!(!rules.allow_export);
    }

    #[test]
    fn test_custodian_federation_check() {
        let custodian = DataCustodian::new(PathBuf::from("/tmp/bizra"));

        assert!(custodian.can_federate(DataCategory::Patterns));
        assert!(!custodian.can_federate(DataCategory::ModelArtifacts));
    }

    #[test]
    fn test_encryption_roundtrip() {
        let manager = EncryptionManager::from_passphrase("test-passphrase");
        let plaintext = b"Sovereign data at rest";

        let envelope = manager.encrypt(plaintext, DataCategory::Config).unwrap();
        let decrypted = manager.decrypt(&envelope).unwrap();

        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_encryption_json_roundtrip() {
        let manager = EncryptionManager::from_passphrase("test-passphrase");

        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct TestData {
            message: String,
            value: i32,
        }

        let data = TestData {
            message: "Encrypted at rest".to_string(),
            value: 42,
        };

        let encrypted = manager
            .encrypt_json(&data, DataCategory::UserProfile)
            .unwrap();
        let decrypted: TestData = manager.decrypt_json(&encrypted).unwrap();

        assert_eq!(data, decrypted);
    }

    #[test]
    fn test_wrong_key_fails() {
        let manager1 = EncryptionManager::from_passphrase("passphrase-1");
        let manager2 = EncryptionManager::from_passphrase("passphrase-2");

        let envelope = manager1.encrypt(b"secret", DataCategory::Config).unwrap();
        let result = manager2.decrypt(&envelope);

        assert!(result.is_err());
    }

    #[test]
    fn test_category_key_isolation() {
        let manager = EncryptionManager::from_passphrase("test");

        // Encrypt with one category
        let envelope = manager.encrypt(b"data", DataCategory::Config).unwrap();

        // Modify envelope to claim different category
        let mut tampered = envelope.clone();
        tampered.category = DataCategory::UserProfile;

        // Should fail because derived key is different
        let result = manager.decrypt(&tampered);
        assert!(result.is_err());
    }
}
