//! ML-DSA-87 Post-Quantum Signatures for BIZRA
//!
//! Provides post-quantum cryptographic signatures for CapabilityCards
//! and other sovereignty-critical operations. ML-DSA-87 is NIST's
//! standardized post-quantum signature algorithm (successor to Dilithium-5).

use napi::bindgen_prelude::*;
use napi_derive::napi;
use pqcrypto_mldsa::mldsa87;
use pqcrypto_traits::sign::{PublicKey, SecretKey, DetachedSignature};
use serde::{Deserialize, Serialize};

/// Dilithium-5 keypair for post-quantum signatures
#[napi]
#[derive(Clone)]
pub struct DilithiumKeypair {
    public_key: Vec<u8>,
    secret_key: Vec<u8>,
}

#[napi]
impl DilithiumKeypair {
    /// Generate a new ML-DSA-87 keypair
    #[napi(factory)]
    pub fn generate() -> Result<Self> {
        let (pk, sk) = mldsa87::keypair();

        Ok(Self {
            public_key: pk.as_bytes().to_vec(),
            secret_key: sk.as_bytes().to_vec(),
        })
    }

    /// Get the public key bytes
    #[napi]
    pub fn public_key(&self) -> Vec<u8> {
        self.public_key.clone()
    }

    /// Get the public key as hex string
    #[napi]
    pub fn public_key_hex(&self) -> String {
        hex::encode(&self.public_key)
    }

    /// Sign a message using ML-DSA-87
    #[napi]
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        let sk = mldsa87::SecretKey::from_bytes(&self.secret_key)
            .map_err(|_| Error::from_reason("Invalid secret key"))?;

        let signature = mldsa87::detached_sign(message, &sk);

        Ok(signature.as_bytes().to_vec())
    }

    /// Verify a signature
    #[napi]
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool> {
        let pk = mldsa87::PublicKey::from_bytes(&self.public_key)
            .map_err(|_| Error::from_reason("Invalid public key"))?;

        let sig = mldsa87::DetachedSignature::from_bytes(signature)
            .map_err(|_| Error::from_reason("Invalid signature format"))?;

        Ok(mldsa87::verify_detached_signature(&sig, message, &pk).is_ok())
    }

    /// Export keypair to JSON
    #[napi]
    pub fn to_json(&self) -> Result<String> {
        let keypair_data = KeypairData {
            algorithm: "ML-DSA-87".to_string(),
            public_key: hex::encode(&self.public_key),
            secret_key: hex::encode(&self.secret_key),
        };

        serde_json::to_string(&keypair_data)
            .map_err(|e| Error::from_reason(format!("Serialization error: {}", e)))
    }

    /// Import keypair from JSON
    #[napi(factory)]
    pub fn from_json(json: String) -> Result<Self> {
        let data: KeypairData = serde_json::from_str(&json)
            .map_err(|e| Error::from_reason(format!("Parse error: {}", e)))?;

        if data.algorithm != "ML-DSA-87" && data.algorithm != "Dilithium-5" {
            return Err(Error::from_reason("Algorithm mismatch: expected ML-DSA-87 or Dilithium-5"));
        }

        Ok(Self {
            public_key: hex::decode(&data.public_key)
                .map_err(|_| Error::from_reason("Invalid public key hex"))?,
            secret_key: hex::decode(&data.secret_key)
                .map_err(|_| Error::from_reason("Invalid secret key hex"))?,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct KeypairData {
    algorithm: String,
    public_key: String,
    secret_key: String,
}

/// Standalone signature verification (no secret key needed)
#[napi]
pub fn verify_dilithium_signature(
    public_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<bool> {
    let pk = mldsa87::PublicKey::from_bytes(public_key)
        .map_err(|_| Error::from_reason("Invalid public key"))?;

    let sig = mldsa87::DetachedSignature::from_bytes(signature)
        .map_err(|_| Error::from_reason("Invalid signature format"))?;

    Ok(mldsa87::verify_detached_signature(&sig, message, &pk).is_ok())
}

/// Signed message container
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedData {
    pub message: String,
    pub signature: String,
    pub public_key: String,
    pub algorithm: String,
    pub timestamp: String,
}

impl SignedData {
    /// Create a new signed data container
    pub fn new(message: &str, keypair: &DilithiumKeypair) -> Result<Self> {
        let signature = keypair.sign(message.as_bytes())?;

        Ok(Self {
            message: message.to_string(),
            signature: hex::encode(&signature),
            public_key: keypair.public_key_hex(),
            algorithm: "ML-DSA-87".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Verify this signed data
    pub fn verify(&self) -> Result<bool> {
        let public_key = hex::decode(&self.public_key)
            .map_err(|_| Error::from_reason("Invalid public key hex"))?;
        let signature = hex::decode(&self.signature)
            .map_err(|_| Error::from_reason("Invalid signature hex"))?;

        verify_dilithium_signature(&public_key, self.message.as_bytes(), &signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = DilithiumKeypair::generate().unwrap();
        assert!(!keypair.public_key.is_empty());
        assert!(!keypair.secret_key.is_empty());
    }

    #[test]
    fn test_sign_and_verify() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let message = b"BIZRA Constitution acceptance";

        let signature = keypair.sign(message).unwrap();
        assert!(keypair.verify(message, &signature).unwrap());
    }

    #[test]
    fn test_invalid_signature() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let message = b"BIZRA Constitution acceptance";
        let wrong_message = b"Modified message";

        let signature = keypair.sign(message).unwrap();
        assert!(!keypair.verify(wrong_message, &signature).unwrap());
    }

    #[test]
    fn test_keypair_serialization() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let json = keypair.to_json().unwrap();
        let restored = DilithiumKeypair::from_json(json).unwrap();

        assert_eq!(keypair.public_key, restored.public_key);
    }

    #[test]
    fn test_signed_data() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let signed = SignedData::new("Test message", &keypair).unwrap();

        assert!(signed.verify().unwrap());
    }
}
