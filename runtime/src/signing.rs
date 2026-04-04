// src/signing.rs - Shared Ed25519 signing module for BIZRA receipts
// Extracted from PCI envelope for reuse across all receipt types
//
// Key Storage: ~/.bizra/node0/keys/signing.key (persistent Ed25519 keypair)
// Domain Separation: "bizra-receipt-v1:" prefix prevents cross-protocol attacks
// Canonical JSON: RFC 8785 JCS (sorted keys, no whitespace)

use ed25519_dalek::{Signature as Ed25519Sig, Signer, SigningKey, Verifier, VerifyingKey};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;
use tracing::info;

/// Domain separator for receipt signatures (prevents cross-protocol attacks)
const RECEIPT_DOMAIN: &str = "bizra-receipt-v1:";

/// Key file permissions (Unix only: 0o600 = owner read/write only)
#[cfg(unix)]
const KEY_FILE_MODE: u32 = 0o600;

/// Signing errors
#[derive(Debug, Error)]
pub enum SigningError {
    #[error("Signing key not found at path: {path}")]
    KeyNotFound { path: PathBuf },

    #[error("Failed to load signing key: {source}")]
    KeyLoadFailed {
        #[source]
        source: std::io::Error,
    },

    #[error("Invalid key format: {reason}")]
    InvalidKey { reason: String },

    #[error("Signing operation failed: {reason}")]
    SigningFailed { reason: String },

    #[error("Signature verification failed: {reason}")]
    VerificationFailed { reason: String },
}

/// Receipt signature with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptSignature {
    /// Hex-encoded Ed25519 signature (64 bytes = 128 hex chars)
    pub signature_hex: String,
    /// Hex-encoded public key of signer (32 bytes = 64 hex chars)
    pub signer_public_key: String,
    /// Domain separator used for this signature
    pub domain: String,
}

/// Ed25519 signer for BIZRA receipts
pub struct ReceiptSigner {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    public_key_hex: String,
}

impl ReceiptSigner {
    /// Load or generate persistent signing key from keystore
    ///
    /// Path: ~/.bizra/node0/keys/signing.key
    /// Format: Hex-encoded Ed25519 private key (64 hex chars)
    pub fn from_keystore() -> Result<Self, SigningError> {
        let key_path = Self::get_key_path()?;

        if key_path.exists() {
            Self::load_from_file(&key_path)
        } else {
            Self::generate_and_save(&key_path)
        }
    }

    /// Construct signer from existing signing key
    pub fn from_key(signing_key: SigningKey) -> Self {
        let verifying_key = signing_key.verifying_key();
        let public_key_hex = hex::encode(verifying_key.as_bytes());

        Self {
            signing_key,
            verifying_key,
            public_key_hex,
        }
    }

    /// Sign receipt content with domain separation
    ///
    /// Process:
    /// 1. Prepend domain separator ("bizra-receipt-v1:")
    /// 2. SHA-256 hash the combined bytes
    /// 3. Ed25519 sign the hash
    pub fn sign_receipt(&self, canonical_json: &[u8]) -> ReceiptSignature {
        // Domain-separated input
        let mut data_to_sign = Vec::with_capacity(RECEIPT_DOMAIN.len() + canonical_json.len());
        data_to_sign.extend_from_slice(RECEIPT_DOMAIN.as_bytes());
        data_to_sign.extend_from_slice(canonical_json);

        // SHA-256 hash (prevents length extension attacks)
        let hash = Sha256::digest(&data_to_sign);

        // Ed25519 signature
        let signature = self.signing_key.sign(&hash);

        ReceiptSignature {
            signature_hex: hex::encode(signature.to_bytes()),
            signer_public_key: self.public_key_hex.clone(),
            domain: RECEIPT_DOMAIN.to_string(),
        }
    }

    /// Verify receipt signature (static method)
    ///
    /// Returns Ok(true) if signature is valid, Ok(false) if invalid
    /// Returns Err only on malformed inputs (hex decode failures, etc)
    pub fn verify_receipt(
        public_key_hex: &str,
        canonical_json: &[u8],
        signature_hex: &str,
    ) -> Result<bool, SigningError> {
        // Parse public key
        let pk_bytes = hex::decode(public_key_hex).map_err(|_| SigningError::InvalidKey {
            reason: "Invalid hex encoding for public key".to_string(),
        })?;

        if pk_bytes.len() != 32 {
            return Err(SigningError::InvalidKey {
                reason: format!("Public key must be 32 bytes, got {}", pk_bytes.len()),
            });
        }

        let pk_array: [u8; 32] = pk_bytes.try_into().expect("length already checked");
        let verifying_key = VerifyingKey::from_bytes(&pk_array).map_err(|e| {
            SigningError::InvalidKey {
                reason: format!("Invalid Ed25519 public key: {}", e),
            }
        })?;

        // Parse signature
        let sig_bytes = hex::decode(signature_hex).map_err(|_| SigningError::InvalidKey {
            reason: "Invalid hex encoding for signature".to_string(),
        })?;

        if sig_bytes.len() != 64 {
            return Err(SigningError::InvalidKey {
                reason: format!("Signature must be 64 bytes, got {}", sig_bytes.len()),
            });
        }

        let sig_array: [u8; 64] = sig_bytes.try_into().expect("length already checked");
        let signature = Ed25519Sig::from_bytes(&sig_array);

        // Reconstruct domain-separated input
        let mut data_to_verify = Vec::with_capacity(RECEIPT_DOMAIN.len() + canonical_json.len());
        data_to_verify.extend_from_slice(RECEIPT_DOMAIN.as_bytes());
        data_to_verify.extend_from_slice(canonical_json);

        // SHA-256 hash
        let hash = Sha256::digest(&data_to_verify);

        // Verify
        Ok(verifying_key.verify(&hash, &signature).is_ok())
    }

    /// Get hex-encoded public key
    pub fn public_key_hex(&self) -> &str {
        &self.public_key_hex
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PRIVATE KEY MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════

    fn get_key_path() -> Result<PathBuf, SigningError> {
        let home_dir = home::home_dir().ok_or_else(|| SigningError::KeyLoadFailed {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ),
        })?;

        Ok(home_dir
            .join(".bizra")
            .join("node0")
            .join("keys")
            .join("signing.key"))
    }

    fn load_from_file(path: &PathBuf) -> Result<Self, SigningError> {
        let hex_key =
            fs::read_to_string(path).map_err(|e| SigningError::KeyLoadFailed { source: e })?;

        let key_bytes = hex::decode(hex_key.trim()).map_err(|_| SigningError::InvalidKey {
            reason: "Key file contains invalid hex".to_string(),
        })?;

        if key_bytes.len() != 32 {
            return Err(SigningError::InvalidKey {
                reason: format!("Expected 32 bytes, got {}", key_bytes.len()),
            });
        }

        let key_array: [u8; 32] = key_bytes.try_into().expect("length already checked");
        let signing_key = SigningKey::from_bytes(&key_array);

        info!(path = ?path, "Loaded Ed25519 signing key from keystore");

        Ok(Self::from_key(signing_key))
    }

    fn generate_and_save(path: &PathBuf) -> Result<Self, SigningError> {
        // Generate new keypair (32 random bytes for Ed25519)
        let mut key_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| SigningError::KeyLoadFailed { source: e })?;
        }

        // Write private key (hex-encoded)
        let hex_key = hex::encode(signing_key.to_bytes());
        fs::write(path, &hex_key).map_err(|e| SigningError::KeyLoadFailed { source: e })?;

        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = fs::Permissions::from_mode(KEY_FILE_MODE);
            fs::set_permissions(path, permissions)
                .map_err(|e| SigningError::KeyLoadFailed { source: e })?;
        }

        // Write public key (for convenience)
        let pub_path = path.with_file_name("signing.pub");
        let hex_pub = hex::encode(verifying_key.as_bytes());
        fs::write(&pub_path, &hex_pub).map_err(|e| SigningError::KeyLoadFailed { source: e })?;

        info!(
            path = ?path,
            public_key = %hex_pub,
            "Generated new Ed25519 signing keypair"
        );

        Ok(Self::from_key(signing_key))
    }
}

/// Serialize value to canonical JSON bytes (RFC 8785 JCS)
///
/// Properties:
/// - Keys sorted lexicographically
/// - No whitespace
/// - Deterministic encoding
///
/// This ensures the same struct always produces the same bytes,
/// which is critical for signature verification.
pub fn canonical_json_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    // Convert to serde_json::Value first
    let json_value =
        serde_json::to_value(value).expect("Failed to serialize to serde_json::Value");

    // Use the canonical serializer from PCI envelope
    // (Copied here to avoid circular dependency)
    fn serialize_value(value: &serde_json::Value, out: &mut Vec<u8>) {
        match value {
            serde_json::Value::Null => out.extend_from_slice(b"null"),
            serde_json::Value::Bool(b) => {
                out.extend_from_slice(if *b { b"true" } else { b"false" })
            }
            serde_json::Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
            serde_json::Value::String(s) => {
                out.push(b'"');
                for c in s.chars() {
                    match c {
                        '"' => out.extend_from_slice(b"\\\""),
                        '\\' => out.extend_from_slice(b"\\\\"),
                        '\n' => out.extend_from_slice(b"\\n"),
                        '\r' => out.extend_from_slice(b"\\r"),
                        '\t' => out.extend_from_slice(b"\\t"),
                        c if c.is_control() => {
                            out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes())
                        }
                        c => out.extend_from_slice(c.to_string().as_bytes()),
                    }
                }
                out.push(b'"');
            }
            serde_json::Value::Array(arr) => {
                out.push(b'[');
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    serialize_value(v, out);
                }
                out.push(b']');
            }
            serde_json::Value::Object(obj) => {
                out.push(b'{');
                // Sort keys lexicographically using BTreeMap
                let sorted: std::collections::BTreeMap<_, _> = obj.iter().collect();
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.push(b'"');
                    out.extend_from_slice(k.as_bytes());
                    out.push(b'"');
                    out.push(b':');
                    serialize_value(v, out);
                }
                out.push(b'}');
            }
        }
    }

    let mut out = Vec::new();
    serialize_value(&json_value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_key_generation_and_roundtrip() {
        // Generate keypair
        let mut key_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let signer = ReceiptSigner::from_key(signing_key);

        // Sign test data
        let test_data = b"test receipt content";
        let signature = signer.sign_receipt(test_data);

        // Verify signature
        let is_valid = ReceiptSigner::verify_receipt(
            &signature.signer_public_key,
            test_data,
            &signature.signature_hex,
        )
        .expect("Verification should not error");

        assert!(is_valid, "Signature should be valid");

        // Check signature format
        assert_eq!(signature.signature_hex.len(), 128); // 64 bytes = 128 hex
        assert_eq!(signature.signer_public_key.len(), 64); // 32 bytes = 64 hex
        assert_eq!(signature.domain, RECEIPT_DOMAIN);
    }

    #[test]
    fn test_tampered_content_fails() {
        let mut key_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let signer = ReceiptSigner::from_key(signing_key);

        // Sign original data
        let original_data = b"original content";
        let signature = signer.sign_receipt(original_data);

        // Try to verify with tampered data
        let tampered_data = b"tampered content";
        let is_valid = ReceiptSigner::verify_receipt(
            &signature.signer_public_key,
            tampered_data,
            &signature.signature_hex,
        )
        .expect("Verification should not error");

        assert!(!is_valid, "Tampered content should fail verification");
    }

    #[test]
    fn test_domain_separation() {
        let mut key_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let signer = ReceiptSigner::from_key(signing_key);

        let data = b"content";
        let signature = signer.sign_receipt(data);

        // Construct input with wrong domain
        let wrong_domain = b"wrong-domain:";
        let mut wrong_input = Vec::new();
        wrong_input.extend_from_slice(wrong_domain);
        wrong_input.extend_from_slice(data);

        // Try to verify by manually constructing with wrong domain
        let pk_bytes = hex::decode(&signature.signer_public_key).unwrap();
        let pk_array: [u8; 32] = pk_bytes.try_into().unwrap();
        let verifying_key = VerifyingKey::from_bytes(&pk_array).unwrap();

        let sig_bytes = hex::decode(&signature.signature_hex).unwrap();
        let sig_array: [u8; 64] = sig_bytes.try_into().unwrap();
        let sig = Ed25519Sig::from_bytes(&sig_array);

        let hash = Sha256::digest(&wrong_input);
        let result = verifying_key.verify(&hash, &sig);

        assert!(
            result.is_err(),
            "Wrong domain should fail verification (domain separation)"
        );
    }

    #[test]
    fn test_canonical_json_determinism() {
        // Same struct should produce same bytes
        #[derive(Serialize)]
        struct TestReceipt {
            zebra: bool,
            apple: i32,
            mango: String,
        }

        let receipt1 = TestReceipt {
            zebra: true,
            apple: 42,
            mango: "test".to_string(),
        };

        let receipt2 = TestReceipt {
            zebra: true,
            apple: 42,
            mango: "test".to_string(),
        };

        let bytes1 = canonical_json_bytes(&receipt1);
        let bytes2 = canonical_json_bytes(&receipt2);

        assert_eq!(bytes1, bytes2, "Same struct should produce same bytes");

        // Check that keys are sorted (apple < mango < zebra)
        let json_str = String::from_utf8_lossy(&bytes1);
        assert_eq!(json_str, r#"{"apple":42,"mango":"test","zebra":true}"#);
    }

    #[test]
    fn test_canonical_json_same_signature() {
        let mut key_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let signer = ReceiptSigner::from_key(signing_key);

        // Same content in different order
        let obj1 = json!({"z": 1, "a": 2});
        let obj2 = json!({"a": 2, "z": 1});

        let bytes1 = canonical_json_bytes(&obj1);
        let bytes2 = canonical_json_bytes(&obj2);

        // Should produce identical bytes (keys sorted)
        assert_eq!(bytes1, bytes2);

        // Should produce identical signatures
        let sig1 = signer.sign_receipt(&bytes1);
        let sig2 = signer.sign_receipt(&bytes2);

        assert_eq!(sig1.signature_hex, sig2.signature_hex);
    }

    #[test]
    fn test_key_persistence_to_temp() {
        use std::env;

        // Create temp directory
        let temp_dir = env::temp_dir().join(format!("bizra_test_{}", rand::random::<u32>()));
        let key_path = temp_dir.join("signing.key");
        let pub_path = temp_dir.join("signing.pub");

        // Generate and save
        let signer1 = ReceiptSigner::generate_and_save(&key_path).expect("Failed to generate key");
        let pub_key1 = signer1.public_key_hex().to_string();

        // Load from file
        let signer2 = ReceiptSigner::load_from_file(&key_path).expect("Failed to load key");
        let pub_key2 = signer2.public_key_hex().to_string();

        // Should have same public key
        assert_eq!(pub_key1, pub_key2);

        // Sign with both, should verify
        let data = b"test persistence";
        let sig1 = signer1.sign_receipt(data);
        let sig2 = signer2.sign_receipt(data);

        assert_eq!(
            sig1.signature_hex, sig2.signature_hex,
            "Same key should produce same signature"
        );

        // Cleanup
        let _ = fs::remove_file(&key_path);
        let _ = fs::remove_file(&pub_path);
        let _ = fs::remove_dir(&temp_dir);
    }

    #[test]
    fn test_invalid_inputs() {
        // Invalid hex
        let result = ReceiptSigner::verify_receipt("not-hex", b"data", "also-not-hex");
        assert!(result.is_err(), "Invalid hex should error");

        // Wrong length public key
        let result = ReceiptSigner::verify_receipt(
            "aabbcc", // Too short
            b"data",
            &"00".repeat(128),
        );
        assert!(result.is_err(), "Wrong length public key should error");

        // Wrong length signature
        let result = ReceiptSigner::verify_receipt(
            &"00".repeat(64),
            b"data",
            "aabbcc", // Too short
        );
        assert!(result.is_err(), "Wrong length signature should error");
    }
}
