//! Node Identity — Ed25519 + BLAKE3 Cryptography

use blake3::Hasher;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::DOMAIN_PREFIX;

/// Unique node identifier
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn from_verifying_key(key: &VerifyingKey) -> Self {
        let bytes = key.as_bytes();
        Self(hex_encode(&bytes[..16]))
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node_{}", &self.0[..8])
    }
}

/// Complete node identity with signing capability
pub struct NodeIdentity {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    node_id: NodeId,
}

impl NodeIdentity {
    /// Generate new random identity
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self::from_signing_key(signing_key)
    }

    /// Restore from secret bytes
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(bytes);
        Self::from_signing_key(signing_key)
    }

    fn from_signing_key(signing_key: SigningKey) -> Self {
        let verifying_key = signing_key.verifying_key();
        let node_id = NodeId::from_verifying_key(&verifying_key);
        Self { signing_key, verifying_key, node_id }
    }

    pub fn node_id(&self) -> &NodeId { &self.node_id }
    pub fn verifying_key(&self) -> &VerifyingKey { &self.verifying_key }
    pub fn public_key_hex(&self) -> String { hex_encode(self.verifying_key.as_bytes()) }
    pub fn public_key_bytes(&self) -> [u8; 32] { *self.verifying_key.as_bytes() }
    pub fn secret_bytes(&self) -> [u8; 32] { self.signing_key.to_bytes() }
    pub fn signing_key(&self) -> &SigningKey { &self.signing_key }

    /// Sign with domain separation
    pub fn sign(&self, message: &[u8]) -> String {
        let digest = domain_separated_digest(message);
        let signature = self.signing_key.sign(digest.as_bytes());
        hex_encode(&signature.to_bytes())
    }

    /// Verify signature
    pub fn verify(message: &[u8], signature_hex: &str, verifying_key: &VerifyingKey) -> bool {
        let digest = domain_separated_digest(message);
        let sig_bytes = match hex_decode(signature_hex) {
            Ok(b) if b.len() == 64 => b,
            _ => return false,
        };
        let sig_array: [u8; 64] = match sig_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let signature = Signature::from_bytes(&sig_array);
        verifying_key.verify(digest.as_bytes(), &signature).is_ok()
    }

    /// Verify with public key hex
    pub fn verify_with_hex(message: &[u8], signature_hex: &str, public_key_hex: &str) -> bool {
        let pk_bytes = match hex_decode(public_key_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => return false,
        };
        let pk_array: [u8; 32] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let verifying_key = match VerifyingKey::from_bytes(&pk_array) {
            Ok(k) => k,
            Err(_) => return false,
        };
        Self::verify(message, signature_hex, &verifying_key)
    }
}

/// Domain-separated BLAKE3 digest
pub fn domain_separated_digest(message: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(message);
    hasher.finalize().to_hex().to_string()
}

pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn hex_decode(s: &str) -> Result<Vec<u8>, ()> {
    if s.len() % 2 != 0 { return Err(()); }
    (0..s.len()).step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify() {
        let identity = NodeIdentity::generate();
        let msg = b"test message";
        let sig = identity.sign(msg);
        assert!(NodeIdentity::verify(msg, &sig, identity.verifying_key()));
    }

    #[test]
    fn test_domain_separation() {
        let d1 = domain_separated_digest(b"hello");
        let d2 = blake3::hash(b"hello").to_hex().to_string();
        assert_ne!(d1, d2);
    }
}
