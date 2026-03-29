//! # MissionEnvelope — Cross-Layer Contract #1
//!
//! Canonical cross-layer mission object with constitutional context
//! and BLAKE3 canonical hash for cross-language verification.
//!
//! Serialization follows the golden-vector protocol (proven identical
//! across Rust and Python in golden_vector.rs / golden_vector.py).

use blake3::Hasher;
use serde::{Deserialize, Serialize};

/// Domain prefix for envelope hashing.
pub const DOMAIN_ENVELOPE: &str = "bizra-envelope-v1";

const FIXED_POINT_P: f64 = 1_000_000.0;

/// Constitutional context at time of mission submission.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConstitutionalContext {
    pub ihsan_threshold: f64,
    pub snr_threshold: f64,
    pub gini_threshold: f64,
    pub policy_version: String,
}

impl Default for ConstitutionalContext {
    fn default() -> Self {
        Self {
            ihsan_threshold: 0.95,
            snr_threshold: 0.85,
            gini_threshold: 0.35,
            policy_version: "0.90.0".to_string(),
        }
    }
}

/// The MissionEnvelope — canonical cross-layer contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionEnvelope {
    pub mission_id: String,
    pub initiator_id: String,
    pub payload_hash: [u8; 32],
    pub constitutional_context: ConstitutionalContext,
    pub created_at: u64,
    pub expires_at: u64,
    pub canonical_hash: [u8; 32],
}

impl MissionEnvelope {
    /// Create a new envelope.
    pub fn new(
        mission_id: String,
        initiator_id: String,
        payload: &[u8],
        context: ConstitutionalContext,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Self {
        let payload_hash: [u8; 32] = blake3::hash(payload).into();
        let mut envelope = Self {
            mission_id,
            initiator_id,
            payload_hash,
            constitutional_context: context,
            created_at: now_ms,
            expires_at: now_ms + ttl_ms,
            canonical_hash: [0; 32],
        };
        envelope.canonical_hash = envelope.compute_hash();
        envelope
    }

    /// Domain-separated BLAKE3 hash using golden-vector protocol.
    pub fn compute_hash(&self) -> [u8; 32] {
        let serialized = self.serialize_canonical();
        let mut hasher = Hasher::new();
        hasher.update(DOMAIN_ENVELOPE.as_bytes());
        hasher.update(b":");
        hasher.update(&serialized);
        hasher.finalize().into()
    }

    fn serialize_canonical(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        let mid = self.mission_id.as_bytes();
        buf.extend_from_slice(&(mid.len() as u32).to_le_bytes());
        buf.extend_from_slice(mid);
        let iid = self.initiator_id.as_bytes();
        buf.extend_from_slice(&(iid.len() as u32).to_le_bytes());
        buf.extend_from_slice(iid);
        buf.extend_from_slice(&self.payload_hash);
        buf.extend_from_slice(
            &((self.constitutional_context.ihsan_threshold * FIXED_POINT_P).round() as u64)
                .to_le_bytes(),
        );
        buf.extend_from_slice(
            &((self.constitutional_context.snr_threshold * FIXED_POINT_P).round() as u64)
                .to_le_bytes(),
        );
        buf.extend_from_slice(
            &((self.constitutional_context.gini_threshold * FIXED_POINT_P).round() as u64)
                .to_le_bytes(),
        );
        let pv = self.constitutional_context.policy_version.as_bytes();
        buf.extend_from_slice(&(pv.len() as u32).to_le_bytes());
        buf.extend_from_slice(pv);
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.expires_at.to_le_bytes());
        buf
    }

    /// Verify envelope integrity and expiration.
    pub fn verify(&self, now_ms: u64) -> Result<(), EnvelopeError> {
        if self.expires_at < now_ms {
            return Err(EnvelopeError::Expired);
        }
        if self.compute_hash() != self.canonical_hash {
            return Err(EnvelopeError::IntegrityFailure);
        }
        Ok(())
    }
}

/// Envelope verification errors.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvelopeError {
    Expired,
    IntegrityFailure,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expired => write!(f, "Envelope expired"),
            Self::IntegrityFailure => write!(f, "Envelope integrity failure"),
        }
    }
}

impl std::error::Error for EnvelopeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_envelope() -> MissionEnvelope {
        MissionEnvelope::new(
            "test-mission-001".to_string(),
            "node0-genesis".to_string(),
            b"Sort my inbox by priority",
            ConstitutionalContext::default(),
            1711584000000,
            120_000,
        )
    }

    #[test]
    fn test_envelope_hash_valid() {
        let env = test_envelope();
        assert_ne!(env.canonical_hash, [0; 32]);
        assert_eq!(env.compute_hash(), env.canonical_hash);
    }

    #[test]
    fn test_envelope_deterministic() {
        assert_eq!(
            test_envelope().canonical_hash,
            test_envelope().canonical_hash
        );
    }

    #[test]
    fn test_envelope_verify_passes() {
        assert!(test_envelope().verify(1711584000000).is_ok());
    }

    #[test]
    fn test_envelope_verify_fails_expired() {
        assert_eq!(
            test_envelope().verify(1711584200000),
            Err(EnvelopeError::Expired)
        );
    }

    #[test]
    fn test_envelope_verify_fails_tamper() {
        let mut env = test_envelope();
        env.initiator_id = "tampered".to_string();
        assert_eq!(
            env.verify(1711584000000),
            Err(EnvelopeError::IntegrityFailure)
        );
    }

    #[test]
    fn test_different_payloads_different_hashes() {
        let e1 = MissionEnvelope::new(
            "m1".into(),
            "n".into(),
            b"A",
            ConstitutionalContext::default(),
            1000,
            120_000,
        );
        let e2 = MissionEnvelope::new(
            "m1".into(),
            "n".into(),
            b"B",
            ConstitutionalContext::default(),
            1000,
            120_000,
        );
        assert_ne!(e1.canonical_hash, e2.canonical_hash);
    }
}
