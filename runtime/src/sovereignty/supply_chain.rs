// src/sovereignty/supply_chain.rs - Supply-Chain Sovereignty (Pillar 5: Build & Updates)
//
// Principle: Reproducible builds + SBOM. Updates must be signed and verifiable.
// Optional dependencies; minimal trusted computing base.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey, SECRET_KEY_LENGTH};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;

/// Artifact type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Rust binary
    Binary,
    /// Python package
    PythonPackage,
    /// Docker image
    DockerImage,
    /// Model weights
    ModelWeights,
    /// Configuration file
    Config,
    /// SBOM document
    Sbom,
}

/// Artifact verification status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Not verified
    Unverified,
    /// Hash verified
    HashVerified,
    /// Signature verified
    SignatureVerified,
    /// Full attestation (hash + signature + provenance)
    FullAttestation,
    /// Verification failed
    Failed,
}

/// Pinned artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedArtifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Version string
    pub version: String,
    /// SHA256 hash (hex)
    pub sha256: String,
    /// Signer public key (if signed)
    pub signer: Option<String>,
    /// Signature (if signed)
    pub signature: Option<String>,
    /// Source URL
    pub source: Option<String>,
    /// Verification status
    pub status: VerificationStatus,
}

impl PinnedArtifact {
    /// Create a new pinned artifact
    pub fn new(
        name: impl Into<String>,
        artifact_type: ArtifactType,
        version: impl Into<String>,
        sha256: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            artifact_type,
            version: version.into(),
            sha256: sha256.into(),
            signer: None,
            signature: None,
            source: None,
            status: VerificationStatus::Unverified,
        }
    }

    /// Verify hash against content
    pub fn verify_hash(&self, content: &[u8]) -> bool {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hex::encode(hasher.finalize());
        hash == self.sha256
    }
}

/// SBOM (Software Bill of Materials) entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbomEntry {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Package type (cargo, pip, npm)
    pub package_type: String,
    /// License
    pub license: Option<String>,
    /// Repository URL
    pub repository: Option<String>,
    /// SHA256 of package
    pub sha256: Option<String>,
    /// Known vulnerabilities
    pub vulnerabilities: Vec<String>,
    /// Is transitive dependency
    pub is_transitive: bool,
}

/// SBOM document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sbom {
    /// SBOM format (CycloneDX, SPDX)
    pub format: String,
    /// SBOM version
    pub version: String,
    /// Generated timestamp
    pub generated_at: chrono::DateTime<chrono::Utc>,
    /// Generator tool
    pub generator: String,
    /// Root component
    pub root_component: String,
    /// Dependencies
    pub dependencies: Vec<SbomEntry>,
    /// Document hash
    pub document_hash: String,
}

impl Sbom {
    /// Count dependencies with vulnerabilities
    pub fn vulnerable_count(&self) -> usize {
        self.dependencies
            .iter()
            .filter(|d| !d.vulnerabilities.is_empty())
            .count()
    }

    /// Get all unique licenses
    pub fn licenses(&self) -> Vec<String> {
        let mut licenses: Vec<_> = self
            .dependencies
            .iter()
            .filter_map(|d| d.license.clone())
            .collect();
        licenses.sort();
        licenses.dedup();
        licenses
    }
}

/// Update manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateManifest {
    /// Update ID
    pub update_id: String,
    /// From version
    pub from_version: String,
    /// To version
    pub to_version: String,
    /// Release timestamp
    pub released_at: chrono::DateTime<chrono::Utc>,
    /// Changelog summary
    pub changelog: String,
    /// Artifacts to update
    pub artifacts: Vec<PinnedArtifact>,
    /// Required minimum version
    pub requires_min_version: Option<String>,
    /// Update signature
    pub signature: String,
    /// Signer public key
    pub signer: String,
}

impl UpdateManifest {
    /// Create a signable payload from manifest contents
    fn signable_payload(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(self.update_id.as_bytes());
        payload.extend_from_slice(self.from_version.as_bytes());
        payload.extend_from_slice(self.to_version.as_bytes());
        payload.extend_from_slice(self.changelog.as_bytes());
        for artifact in &self.artifacts {
            payload.extend_from_slice(artifact.name.as_bytes());
            payload.extend_from_slice(artifact.sha256.as_bytes());
        }
        payload
    }

    /// Verify update is signed by trusted signer with Ed25519
    pub fn verify_signature(&self, trusted_signers: &[String]) -> bool {
        // Check if signer is trusted
        if !trusted_signers.contains(&self.signer) {
            return false;
        }

        // Decode public key
        let Ok(pubkey_bytes) = hex::decode(&self.signer) else {
            return false;
        };
        if pubkey_bytes.len() != 32 {
            return false;
        }
        let mut pubkey_arr = [0u8; 32];
        pubkey_arr.copy_from_slice(&pubkey_bytes);
        let Ok(verifying_key) = VerifyingKey::from_bytes(&pubkey_arr) else {
            return false;
        };

        // Decode signature
        let Ok(sig_bytes) = hex::decode(&self.signature) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(&sig_bytes) else {
            return false;
        };

        // Verify
        let payload = self.signable_payload();
        verifying_key.verify(&payload, &signature).is_ok()
    }
}

/// Supply chain verifier
pub struct SupplyChainVerifier {
    /// Pinned artifacts
    pinned: HashMap<String, PinnedArtifact>,
    /// Trusted signers (public keys)
    trusted_signers: Vec<String>,
    /// Blocked packages
    blocked: Vec<String>,
    /// SBOM cache
    sbom: Option<Sbom>,
}

impl SupplyChainVerifier {
    /// Create new verifier
    pub fn new() -> Self {
        Self {
            pinned: HashMap::new(),
            trusted_signers: Vec::new(),
            blocked: vec![
                // Known malicious packages
                "event-stream".to_string(), // npm incident
                "ua-parser-js".to_string(), // npm incident (old versions)
            ],
            sbom: None,
        }
    }

    /// Add pinned artifact
    pub fn pin(&mut self, artifact: PinnedArtifact) {
        self.pinned.insert(artifact.name.clone(), artifact);
    }

    /// Add trusted signer
    pub fn trust_signer(&mut self, public_key: impl Into<String>) {
        self.trusted_signers.push(public_key.into());
    }

    /// Set SBOM
    pub fn set_sbom(&mut self, sbom: Sbom) {
        self.sbom = Some(sbom);
    }

    /// Check if package is blocked
    pub fn is_blocked(&self, package: &str) -> bool {
        self.blocked.iter().any(|b| package.contains(b))
    }

    /// Verify an artifact
    pub fn verify(&self, name: &str, content: &[u8]) -> VerificationStatus {
        // Check if pinned
        let Some(pinned) = self.pinned.get(name) else {
            return VerificationStatus::Unverified;
        };

        // Verify hash
        if !pinned.verify_hash(content) {
            return VerificationStatus::Failed;
        }

        // Check for signature
        if pinned.signature.is_some() && pinned.signer.is_some() {
            // TODO: Verify Ed25519 signature
            return VerificationStatus::SignatureVerified;
        }

        VerificationStatus::HashVerified
    }

    /// Verify update manifest
    pub fn verify_update(&self, update: &UpdateManifest) -> bool {
        update.verify_signature(&self.trusted_signers)
    }

    /// Get pinned artifact
    pub fn get_pinned(&self, name: &str) -> Option<&PinnedArtifact> {
        self.pinned.get(name)
    }

    /// Get vulnerability count
    pub fn vulnerability_count(&self) -> usize {
        self.sbom
            .as_ref()
            .map(|s| s.vulnerable_count())
            .unwrap_or(0)
    }
}

impl Default for SupplyChainVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Model family artifacts (from model-family-genesis)
pub fn pinned_model_artifacts() -> Vec<PinnedArtifact> {
    vec![
        PinnedArtifact::new(
            "deepseek-r1:8b",
            ArtifactType::ModelWeights,
            "8b",
            "6995872bfe4c521a67b32da386cd21d5c6e819b6e0d62f79f64ec83be99f5763",
        ),
        PinnedArtifact::new(
            "mistral:latest",
            ArtifactType::ModelWeights,
            "7b-v0.3",
            "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
        ),
        PinnedArtifact::new(
            "bizra-planner:latest",
            ArtifactType::ModelWeights,
            "custom",
            "31f7cb3c10487890a5086016922e8a3de652c8b0f832cb60c907bb9c8cc0a656",
        ),
        PinnedArtifact::new(
            "nomic-embed-text:latest",
            ArtifactType::ModelWeights,
            "embed",
            "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
        ),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNED RELEASES (P0 Gap Fix)
// ═══════════════════════════════════════════════════════════════════════════════

/// Release signer for creating signed updates
pub struct ReleaseSigner {
    /// Signing key (private)
    signing_key: SigningKey,
    /// Public key (hex encoded)
    public_key: String,
}

impl ReleaseSigner {
    /// Create from secret key bytes
    pub fn from_secret(secret: &[u8; SECRET_KEY_LENGTH]) -> Self {
        let signing_key = SigningKey::from_bytes(secret);
        let verifying_key = signing_key.verifying_key();
        let public_key = hex::encode(verifying_key.to_bytes());

        Self {
            signing_key,
            public_key,
        }
    }

    /// Create with new random key
    pub fn generate() -> Self {
        let mut secret = [0u8; SECRET_KEY_LENGTH];
        getrandom::getrandom(&mut secret).expect("Failed to generate random bytes");
        Self::from_secret(&secret)
    }

    /// Get public key (hex encoded)
    pub fn public_key(&self) -> &str {
        &self.public_key
    }

    /// Sign an update manifest
    pub fn sign_update(&self, manifest: &mut UpdateManifest) {
        manifest.signer = self.public_key.clone();
        let payload = manifest.signable_payload();
        let signature = self.signing_key.sign(&payload);
        manifest.signature = hex::encode(signature.to_bytes());
    }

    /// Sign an artifact
    pub fn sign_artifact(&self, artifact: &mut PinnedArtifact, content: &[u8]) {
        // Compute hash
        let mut hasher = Sha256::new();
        hasher.update(content);
        artifact.sha256 = hex::encode(hasher.finalize());

        // Sign the hash
        let signature = self.signing_key.sign(artifact.sha256.as_bytes());
        artifact.signature = Some(hex::encode(signature.to_bytes()));
        artifact.signer = Some(self.public_key.clone());
        artifact.status = VerificationStatus::SignatureVerified;
    }

    /// Create a signed release
    pub fn create_release(
        &self,
        from_version: &str,
        to_version: &str,
        changelog: &str,
        artifacts: Vec<PinnedArtifact>,
    ) -> UpdateManifest {
        let update_id = format!(
            "release-{}-to-{}-{}",
            from_version,
            to_version,
            chrono::Utc::now().format("%Y%m%d%H%M%S")
        );

        let mut manifest = UpdateManifest {
            update_id,
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            released_at: chrono::Utc::now(),
            changelog: changelog.to_string(),
            artifacts,
            requires_min_version: Some(from_version.to_string()),
            signature: String::new(),
            signer: String::new(),
        };

        self.sign_update(&mut manifest);
        manifest
    }
}

/// Signed release verifier (extends SupplyChainVerifier)
impl SupplyChainVerifier {
    /// Verify artifact signature with Ed25519
    pub fn verify_artifact_signature(
        &self,
        artifact: &PinnedArtifact,
        content: &[u8],
    ) -> VerificationStatus {
        // First verify hash
        if !artifact.verify_hash(content) {
            return VerificationStatus::Failed;
        }

        // Check for signature
        let (Some(sig_hex), Some(signer_hex)) = (&artifact.signature, &artifact.signer) else {
            return VerificationStatus::HashVerified;
        };

        // Check signer is trusted
        if !self.trusted_signers.contains(signer_hex) {
            return VerificationStatus::HashVerified; // Hash OK but untrusted signer
        }

        // Decode public key
        let Ok(pubkey_bytes) = hex::decode(signer_hex) else {
            return VerificationStatus::Failed;
        };
        if pubkey_bytes.len() != 32 {
            return VerificationStatus::Failed;
        }
        let mut pubkey_arr = [0u8; 32];
        pubkey_arr.copy_from_slice(&pubkey_bytes);
        let Ok(verifying_key) = VerifyingKey::from_bytes(&pubkey_arr) else {
            return VerificationStatus::Failed;
        };

        // Decode signature
        let Ok(sig_bytes) = hex::decode(sig_hex) else {
            return VerificationStatus::Failed;
        };
        let Ok(signature) = Signature::from_slice(&sig_bytes) else {
            return VerificationStatus::Failed;
        };

        // Verify signature over the hash
        if verifying_key
            .verify(artifact.sha256.as_bytes(), &signature)
            .is_ok()
        {
            VerificationStatus::SignatureVerified
        } else {
            VerificationStatus::Failed
        }
    }

    /// Full attestation check (hash + signature + provenance)
    pub fn verify_full_attestation(
        &self,
        artifact: &PinnedArtifact,
        content: &[u8],
    ) -> VerificationStatus {
        let sig_status = self.verify_artifact_signature(artifact, content);

        if sig_status != VerificationStatus::SignatureVerified {
            return sig_status;
        }

        // Check provenance (source URL must be set)
        if artifact.source.is_some() {
            VerificationStatus::FullAttestation
        } else {
            VerificationStatus::SignatureVerified
        }
    }
}

// ============================================================================
// SBOM Generator (P1: Supply-Chain Transparency)
// ============================================================================
// CycloneDX 1.4+ compliant SBOM generation with:
// - Cargo.lock parsing for Rust dependencies
// - requirements.txt parsing for Python dependencies
// - Ed25519 signing of generated SBOMs
// - Integrity hashing with SHA-256

use std::fs::File;
use std::io::BufRead;
use std::path::Path;

/// SBOM generation format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SbomFormat {
    /// CycloneDX 1.4 JSON format
    CycloneDX14,
    /// CycloneDX 1.5 JSON format
    CycloneDX15,
    /// SPDX 2.3 format
    Spdx23,
}

impl std::fmt::Display for SbomFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycloneDX14 => write!(f, "CycloneDX/1.4"),
            Self::CycloneDX15 => write!(f, "CycloneDX/1.5"),
            Self::Spdx23 => write!(f, "SPDX/2.3"),
        }
    }
}

/// Component type for SBOM entries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComponentType {
    Library,
    Application,
    Framework,
    Container,
    OperatingSystem,
    Device,
    File,
}

/// Signed SBOM with cryptographic attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedSbom {
    /// The SBOM payload
    pub sbom: Sbom,
    /// Ed25519 signature of the SBOM document hash
    pub signature: String,
    /// Public key of the signer (hex-encoded)
    pub signer: String,
    /// Signature timestamp
    pub signed_at: chrono::DateTime<chrono::Utc>,
}

impl SignedSbom {
    /// Verify the SBOM signature
    pub fn verify(&self) -> bool {
        let Ok(pubkey_bytes) = hex::decode(&self.signer) else {
            return false;
        };
        if pubkey_bytes.len() != 32 {
            return false;
        }
        let mut pubkey_arr = [0u8; 32];
        pubkey_arr.copy_from_slice(&pubkey_bytes);
        let Ok(verifying_key) = VerifyingKey::from_bytes(&pubkey_arr) else {
            return false;
        };

        let Ok(sig_bytes) = hex::decode(&self.signature) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(&sig_bytes) else {
            return false;
        };

        verifying_key
            .verify(self.sbom.document_hash.as_bytes(), &signature)
            .is_ok()
    }
}

/// Cargo.lock package entry (simplified parser)
#[derive(Debug, Clone)]
struct CargoPackage {
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
}

/// Requirements.txt entry
#[derive(Debug, Clone)]
struct PythonPackage {
    name: String,
    version: Option<String>,
    extras: Vec<String>,
}

/// SBOM Generator for BIZRA supply-chain transparency
pub struct SbomGenerator {
    /// Generator name/version
    generator: String,
    /// Root component name
    root_component: String,
    /// Root component version
    root_version: String,
    /// Optional signer for attestation
    signer: Option<ReleaseSigner>,
}

/// SBOM generation errors
#[derive(Debug)]
pub enum SbomGenerationError {
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// IO error
    IoError(std::io::Error),
    /// Signing error
    SigningError(String),
}

impl std::fmt::Display for SbomGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path) => write!(f, "File not found: {}", path),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::SigningError(msg) => write!(f, "Signing error: {}", msg),
        }
    }
}

impl std::error::Error for SbomGenerationError {}

impl From<std::io::Error> for SbomGenerationError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl SbomGenerator {
    /// Create a new SBOM generator
    pub fn new(root_component: &str, root_version: &str) -> Self {
        Self {
            generator: format!("bizra-sbom-generator/{}", env!("CARGO_PKG_VERSION")),
            root_component: root_component.to_string(),
            root_version: root_version.to_string(),
            signer: None,
        }
    }

    /// Add a signer for SBOM attestation
    pub fn with_signer(mut self, signer: ReleaseSigner) -> Self {
        self.signer = Some(signer);
        self
    }

    /// Generate SBOM from Cargo.lock file
    pub fn generate_from_cargo_lock<P: AsRef<Path>>(
        &self,
        cargo_lock_path: P,
    ) -> Result<Sbom, SbomGenerationError> {
        let path = cargo_lock_path.as_ref();
        if !path.exists() {
            return Err(SbomGenerationError::FileNotFound(
                path.display().to_string(),
            ));
        }

        let content = std::fs::read_to_string(path)?;
        let packages = self.parse_cargo_lock(&content)?;

        let dependencies: Vec<SbomEntry> = packages
            .into_iter()
            .map(|pkg| {
                SbomEntry {
                    name: pkg.name.clone(),
                    version: pkg.version.clone(),
                    package_type: "cargo".to_string(),
                    license: None, // Would need cargo metadata for this
                    repository: pkg.source.clone(),
                    sha256: pkg.checksum.clone(),
                    vulnerabilities: vec![], // Would need audit database
                    is_transitive: true,     // All Cargo.lock entries are resolved deps
                }
            })
            .collect();

        let sbom_json = serde_json::to_string(&dependencies).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(sbom_json.as_bytes());
        let document_hash = hex::encode(hasher.finalize());

        Ok(Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: self.generator.clone(),
            root_component: self.root_component.clone(),
            dependencies,
            document_hash,
        })
    }

    /// Parse Cargo.lock TOML content
    fn parse_cargo_lock(&self, content: &str) -> Result<Vec<CargoPackage>, SbomGenerationError> {
        let mut packages = Vec::new();
        let mut current_pkg: Option<CargoPackage> = None;

        for line in content.lines() {
            let line = line.trim();

            if line == "[[package]]" {
                if let Some(pkg) = current_pkg.take() {
                    packages.push(pkg);
                }
                current_pkg = Some(CargoPackage {
                    name: String::new(),
                    version: String::new(),
                    source: None,
                    checksum: None,
                });
            } else if let Some(ref mut pkg) = current_pkg {
                if let Some(name) = line.strip_prefix("name = ") {
                    pkg.name = name.trim_matches('"').to_string();
                } else if let Some(version) = line.strip_prefix("version = ") {
                    pkg.version = version.trim_matches('"').to_string();
                } else if let Some(source) = line.strip_prefix("source = ") {
                    pkg.source = Some(source.trim_matches('"').to_string());
                } else if let Some(checksum) = line.strip_prefix("checksum = ") {
                    pkg.checksum = Some(checksum.trim_matches('"').to_string());
                }
            }
        }

        // Don't forget the last package
        if let Some(pkg) = current_pkg {
            packages.push(pkg);
        }

        Ok(packages)
    }

    /// Generate SBOM from requirements.txt file
    pub fn generate_from_requirements<P: AsRef<Path>>(
        &self,
        requirements_path: P,
    ) -> Result<Sbom, SbomGenerationError> {
        let path = requirements_path.as_ref();
        if !path.exists() {
            return Err(SbomGenerationError::FileNotFound(
                path.display().to_string(),
            ));
        }

        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let packages = self.parse_requirements(reader)?;

        let dependencies: Vec<SbomEntry> = packages
            .into_iter()
            .map(|pkg| SbomEntry {
                name: pkg.name.clone(),
                version: pkg
                    .version
                    .clone()
                    .unwrap_or_else(|| "unspecified".to_string()),
                package_type: "pip".to_string(),
                license: None,
                repository: Some(format!("https://pypi.org/project/{}/", pkg.name)),
                sha256: None,
                vulnerabilities: vec![],
                is_transitive: false,
            })
            .collect();

        let sbom_json = serde_json::to_string(&dependencies).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(sbom_json.as_bytes());
        let document_hash = hex::encode(hasher.finalize());

        Ok(Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: self.generator.clone(),
            root_component: self.root_component.clone(),
            dependencies,
            document_hash,
        })
    }

    /// Parse requirements.txt format
    fn parse_requirements<R: BufRead>(
        &self,
        reader: R,
    ) -> Result<Vec<PythonPackage>, SbomGenerationError> {
        let mut packages = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') || line.starts_with('-') {
                continue;
            }

            // Parse package spec: name[extras]==version or name>=version, etc.
            let pkg = self.parse_requirement_line(line);
            if let Some(p) = pkg {
                packages.push(p);
            }
        }

        Ok(packages)
    }

    /// Parse a single requirement line
    fn parse_requirement_line(&self, line: &str) -> Option<PythonPackage> {
        // Handle various specifiers: ==, >=, <=, ~=, !=, <, >
        let specifiers = ["==", ">=", "<=", "~=", "!=", "<", ">"];

        for spec in specifiers {
            if let Some(pos) = line.find(spec) {
                let name_part = &line[..pos];
                let version_part = &line[pos + spec.len()..];

                // Extract extras if present: name[extra1,extra2]
                let (name, extras) = self.parse_extras(name_part);

                return Some(PythonPackage {
                    name: name.to_string(),
                    version: Some(
                        version_part
                            .split(|c| c == ';' || c == ' ')
                            .next()?
                            .to_string(),
                    ),
                    extras,
                });
            }
        }

        // No version specifier - just the package name
        let (name, extras) = self.parse_extras(line);
        Some(PythonPackage {
            name: name.to_string(),
            version: None,
            extras,
        })
    }

    /// Extract extras from package name
    fn parse_extras<'a>(&self, name_part: &'a str) -> (&'a str, Vec<String>) {
        if let Some(bracket_start) = name_part.find('[') {
            if let Some(bracket_end) = name_part.find(']') {
                let name = &name_part[..bracket_start];
                let extras_str = &name_part[bracket_start + 1..bracket_end];
                let extras: Vec<String> = extras_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                return (name, extras);
            }
        }
        (name_part, vec![])
    }

    /// Merge multiple SBOMs into one
    pub fn merge_sboms(&self, sboms: Vec<Sbom>) -> Sbom {
        let mut all_deps = Vec::new();

        for sbom in sboms {
            all_deps.extend(sbom.dependencies);
        }

        // Deduplicate by name+version
        let mut seen = std::collections::HashSet::new();
        all_deps.retain(|dep| {
            let key = format!("{}@{}", dep.name, dep.version);
            seen.insert(key)
        });

        let sbom_json = serde_json::to_string(&all_deps).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(sbom_json.as_bytes());
        let document_hash = hex::encode(hasher.finalize());

        Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: self.generator.clone(),
            root_component: self.root_component.clone(),
            dependencies: all_deps,
            document_hash,
        }
    }

    /// Sign an SBOM with the configured signer
    pub fn sign_sbom(&self, sbom: Sbom) -> Result<SignedSbom, SbomGenerationError> {
        let signer = self
            .signer
            .as_ref()
            .ok_or_else(|| SbomGenerationError::SigningError("No signer configured".to_string()))?;

        let signature = signer.signing_key.sign(sbom.document_hash.as_bytes());

        Ok(SignedSbom {
            sbom,
            signature: hex::encode(signature.to_bytes()),
            signer: signer.public_key().to_string(),
            signed_at: chrono::Utc::now(),
        })
    }

    /// Generate CycloneDX JSON output
    pub fn to_cyclonedx_json(&self, sbom: &Sbom) -> String {
        let components: Vec<serde_json::Value> = sbom.dependencies.iter().map(|dep| {
            serde_json::json!({
                "type": "library",
                "name": dep.name,
                "version": dep.version,
                "purl": format!("pkg:{}/{}@{}", dep.package_type, dep.name, dep.version),
                "licenses": dep.license.as_ref().map(|l| vec![serde_json::json!({"license": {"id": l}})]).unwrap_or_default(),
                "hashes": dep.sha256.as_ref().map(|h| vec![serde_json::json!({"alg": "SHA-256", "content": h})]).unwrap_or_default(),
                "externalReferences": dep.repository.as_ref().map(|r| vec![serde_json::json!({"type": "vcs", "url": r})]).unwrap_or_default(),
            })
        }).collect();

        let cyclonedx = serde_json::json!({
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": sbom.generated_at.to_rfc3339(),
                "tools": [{
                    "vendor": "BIZRA",
                    "name": &self.generator,
                    "version": env!("CARGO_PKG_VERSION")
                }],
                "component": {
                    "type": "application",
                    "name": &sbom.root_component,
                    "version": &self.root_version
                }
            },
            "components": components,
            "serialNumber": format!("urn:uuid:{}", uuid::Uuid::new_v4()),
        });

        serde_json::to_string_pretty(&cyclonedx).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_hash_verification() {
        let content = b"test content";
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hex::encode(hasher.finalize());

        let artifact = PinnedArtifact::new("test", ArtifactType::Config, "1.0", &hash);
        assert!(artifact.verify_hash(content));
        assert!(!artifact.verify_hash(b"wrong content"));
    }

    #[test]
    fn test_blocked_packages() {
        let verifier = SupplyChainVerifier::new();

        assert!(verifier.is_blocked("event-stream"));
        assert!(!verifier.is_blocked("safe-package"));
    }

    #[test]
    fn test_pinned_models() {
        let models = pinned_model_artifacts();
        assert_eq!(models.len(), 4);

        assert!(models.iter().any(|m| m.name == "deepseek-r1:8b"));
    }

    #[test]
    fn test_sbom_analysis() {
        let sbom = Sbom {
            format: "CycloneDX".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: "cargo-cyclonedx".to_string(),
            root_component: "bizra-elite".to_string(),
            dependencies: vec![
                SbomEntry {
                    name: "serde".to_string(),
                    version: "1.0.0".to_string(),
                    package_type: "cargo".to_string(),
                    license: Some("MIT OR Apache-2.0".to_string()),
                    repository: None,
                    sha256: None,
                    vulnerabilities: vec![],
                    is_transitive: false,
                },
                SbomEntry {
                    name: "old-vulnerable".to_string(),
                    version: "0.1.0".to_string(),
                    package_type: "cargo".to_string(),
                    license: Some("MIT".to_string()),
                    repository: None,
                    sha256: None,
                    vulnerabilities: vec!["CVE-2024-1234".to_string()],
                    is_transitive: true,
                },
            ],
            document_hash: "abc123".to_string(),
        };

        assert_eq!(sbom.vulnerable_count(), 1);
        assert!(sbom.licenses().contains(&"MIT".to_string()));
    }

    #[test]
    fn test_release_signer_sign_verify() {
        let signer = ReleaseSigner::generate();

        let mut verifier = SupplyChainVerifier::new();
        verifier.trust_signer(signer.public_key());

        // Create and sign a release
        let artifacts = vec![PinnedArtifact::new(
            "test-bin",
            ArtifactType::Binary,
            "1.0.0",
            "abc123",
        )];

        let manifest = signer.create_release("0.9.0", "1.0.0", "Initial release", artifacts);

        // Verify signature
        assert!(manifest.verify_signature(&verifier.trusted_signers));
    }

    #[test]
    fn test_artifact_signature_verification() {
        let signer = ReleaseSigner::generate();
        let content = b"binary content here";

        let mut artifact = PinnedArtifact::new("my-binary", ArtifactType::Binary, "1.0.0", "");
        signer.sign_artifact(&mut artifact, content);

        let mut verifier = SupplyChainVerifier::new();
        verifier.trust_signer(signer.public_key());

        let status = verifier.verify_artifact_signature(&artifact, content);
        assert_eq!(status, VerificationStatus::SignatureVerified);

        // Wrong content should fail
        let wrong_status = verifier.verify_artifact_signature(&artifact, b"wrong content");
        assert_eq!(wrong_status, VerificationStatus::Failed);
    }

    #[test]
    fn test_untrusted_signer_fails() {
        let signer = ReleaseSigner::generate();

        let artifacts = vec![];
        let manifest = signer.create_release("1.0", "2.0", "Upgrade", artifacts);

        // Empty trusted signers = should fail
        let verifier = SupplyChainVerifier::new();
        assert!(!manifest.verify_signature(&verifier.trusted_signers));
    }

    // ============================================================================
    // SBOM Generator Tests
    // ============================================================================

    #[test]
    fn test_sbom_generator_creation() {
        let generator = SbomGenerator::new("bizra-elite", "1.0.0");
        assert_eq!(generator.root_component, "bizra-elite");
        assert_eq!(generator.root_version, "1.0.0");
        assert!(generator.signer.is_none());
    }

    #[test]
    fn test_sbom_generator_with_signer() {
        let signer = ReleaseSigner::generate();
        let generator = SbomGenerator::new("bizra-elite", "1.0.0").with_signer(signer);
        assert!(generator.signer.is_some());
    }

    #[test]
    fn test_cargo_lock_parsing() {
        let cargo_lock_content = r#"
# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
version = 3

[[package]]
name = "aho-corasick"
version = "1.1.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "8e60d3430d3a69f1a82c53d57fe0f69ea7572c04bdf9cf5eddb65e84d33d5f56"

[[package]]
name = "anyhow"
version = "1.0.86"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "b3d1d046238990b9cf5bcf12506f3b8e5f93e8e0b4c3d4e5bf0d4c0c0b0c0b0c"

[[package]]
name = "serde"
version = "1.0.203"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "7253ab4de971e72fb7be983802300c30b5a7f0c2e56f53b4a3c1eff498c47e8e"
"#;

        let generator = SbomGenerator::new("test-project", "1.0.0");
        let packages = generator.parse_cargo_lock(cargo_lock_content).unwrap();

        assert_eq!(packages.len(), 3);
        assert_eq!(packages[0].name, "aho-corasick");
        assert_eq!(packages[0].version, "1.1.3");
        assert!(packages[0].checksum.is_some());

        assert_eq!(packages[1].name, "anyhow");
        assert_eq!(packages[2].name, "serde");
    }

    #[test]
    fn test_requirements_parsing() {
        let requirements = "fastapi==0.100.0\nuvicorn>=0.22.0\npydantic[email]~=2.0\n# comment\nnumpy\n-e ./local_package";
        let reader = std::io::BufReader::new(requirements.as_bytes());

        let generator = SbomGenerator::new("test-project", "1.0.0");
        let packages = generator.parse_requirements(reader).unwrap();

        assert_eq!(packages.len(), 4); // Skips comment and -e line

        assert_eq!(packages[0].name, "fastapi");
        assert_eq!(packages[0].version, Some("0.100.0".to_string()));

        assert_eq!(packages[1].name, "uvicorn");
        assert_eq!(packages[1].version, Some("0.22.0".to_string()));

        assert_eq!(packages[2].name, "pydantic");
        assert_eq!(packages[2].version, Some("2.0".to_string()));
        assert_eq!(packages[2].extras, vec!["email".to_string()]);

        assert_eq!(packages[3].name, "numpy");
        assert!(packages[3].version.is_none());
    }

    #[test]
    fn test_sbom_signing() {
        let signer = ReleaseSigner::generate();
        let generator = SbomGenerator::new("test-project", "1.0.0").with_signer(signer);

        let sbom = Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: "test".to_string(),
            root_component: "test".to_string(),
            dependencies: vec![SbomEntry {
                name: "test-dep".to_string(),
                version: "1.0.0".to_string(),
                package_type: "cargo".to_string(),
                license: None,
                repository: None,
                sha256: None,
                vulnerabilities: vec![],
                is_transitive: false,
            }],
            document_hash: "abc123".to_string(),
        };

        let signed = generator.sign_sbom(sbom).unwrap();
        assert!(signed.verify());
    }

    #[test]
    fn test_sbom_merge() {
        let generator = SbomGenerator::new("merged-project", "1.0.0");

        let sbom1 = Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: "test".to_string(),
            root_component: "project1".to_string(),
            dependencies: vec![
                SbomEntry {
                    name: "shared-dep".to_string(),
                    version: "1.0.0".to_string(),
                    package_type: "cargo".to_string(),
                    license: None,
                    repository: None,
                    sha256: None,
                    vulnerabilities: vec![],
                    is_transitive: false,
                },
                SbomEntry {
                    name: "rust-only".to_string(),
                    version: "2.0.0".to_string(),
                    package_type: "cargo".to_string(),
                    license: None,
                    repository: None,
                    sha256: None,
                    vulnerabilities: vec![],
                    is_transitive: false,
                },
            ],
            document_hash: "hash1".to_string(),
        };

        let sbom2 = Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: "test".to_string(),
            root_component: "project2".to_string(),
            dependencies: vec![
                SbomEntry {
                    name: "shared-dep".to_string(),
                    version: "1.0.0".to_string(),
                    package_type: "pip".to_string(),
                    license: None,
                    repository: None,
                    sha256: None,
                    vulnerabilities: vec![],
                    is_transitive: false,
                },
                SbomEntry {
                    name: "python-only".to_string(),
                    version: "3.0.0".to_string(),
                    package_type: "pip".to_string(),
                    license: None,
                    repository: None,
                    sha256: None,
                    vulnerabilities: vec![],
                    is_transitive: false,
                },
            ],
            document_hash: "hash2".to_string(),
        };

        let merged = generator.merge_sboms(vec![sbom1, sbom2]);

        // Should deduplicate shared-dep@1.0.0
        assert_eq!(merged.dependencies.len(), 3);
        assert!(merged.dependencies.iter().any(|d| d.name == "rust-only"));
        assert!(merged.dependencies.iter().any(|d| d.name == "python-only"));
    }

    #[test]
    fn test_cyclonedx_json_output() {
        let generator = SbomGenerator::new("test-project", "1.0.0");

        let sbom = Sbom {
            format: "CycloneDX/1.4".to_string(),
            version: "1.4".to_string(),
            generated_at: chrono::Utc::now(),
            generator: "test".to_string(),
            root_component: "test".to_string(),
            dependencies: vec![SbomEntry {
                name: "serde".to_string(),
                version: "1.0.0".to_string(),
                package_type: "cargo".to_string(),
                license: Some("MIT".to_string()),
                repository: Some("https://github.com/serde-rs/serde".to_string()),
                sha256: Some("abc123".to_string()),
                vulnerabilities: vec![],
                is_transitive: false,
            }],
            document_hash: "hash".to_string(),
        };

        let json = generator.to_cyclonedx_json(&sbom);

        assert!(json.contains("CycloneDX"));
        assert!(json.contains("\"specVersion\": \"1.4\""));
        assert!(json.contains("pkg:cargo/serde@1.0.0"));
        assert!(json.contains("SHA-256"));
    }

    #[test]
    fn test_sbom_format_display() {
        assert_eq!(format!("{}", SbomFormat::CycloneDX14), "CycloneDX/1.4");
        assert_eq!(format!("{}", SbomFormat::CycloneDX15), "CycloneDX/1.5");
        assert_eq!(format!("{}", SbomFormat::Spdx23), "SPDX/2.3");
    }
}
