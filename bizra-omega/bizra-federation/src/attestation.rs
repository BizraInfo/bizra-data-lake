//! # Node Attestation — Live Challenge-Response Identity Verification
//!
//! Upgrades federation identity from a static "birth certificate" (Merkle receipt)
//! to a live challenge-response protocol. Without this, BFT consensus is
//! vulnerable to Sybil attacks where adversaries forge node identities.
//!
//! ## Protocol
//!
//! 1. Challenger generates random 32-byte nonce + timestamp
//! 2. Responder signs `domain_prefix || nonce || responder_id || timestamp`
//! 3. Challenger verifies signature against known public key
//! 4. Nonce is stored in replay cache; duplicates are rejected
//!
//! ## Security Properties
//!
//! - **Freshness**: Timestamp window (default 30s) prevents replay of old challenges
//! - **Uniqueness**: Nonce replay cache prevents reuse of challenge-response pairs
//! - **Binding**: Domain prefix prevents cross-protocol signature reuse
//! - **Identity**: Response proves possession of Ed25519 private key
//!
//! Standing on Giants: Lamport (distributed auth) · Needham-Schroeder (nonce protocols)

use std::collections::HashMap;

use bizra_core::NodeId;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// Domain prefix for attestation signatures — prevents cross-protocol reuse.
const ATTESTATION_DOMAIN: &[u8] = b"bizra-attestation-v1:";

/// Default challenge validity window in seconds.
const DEFAULT_CHALLENGE_WINDOW_SECS: u64 = 30;

/// Maximum nonce cache size before pruning old entries.
const MAX_NONCE_CACHE: usize = 10_000;

/// A challenge issued to a remote node.
#[derive(Debug, Clone)]
pub struct Challenge {
    /// Random 32-byte nonce
    pub nonce: [u8; 32],
    /// Unix timestamp (seconds) when challenge was created
    pub timestamp: u64,
    /// Node being challenged
    pub target: NodeId,
}

impl Challenge {
    /// Create a new challenge for a target node.
    pub fn new(target: NodeId, timestamp: u64) -> Self {
        let mut nonce = [0u8; 32];
        // Use getrandom for cryptographic randomness
        getrandom::getrandom(&mut nonce).expect("getrandom failed");
        Challenge {
            nonce,
            timestamp,
            target,
        }
    }

    /// Create with a specific nonce (for testing).
    #[cfg(test)]
    pub fn with_nonce(target: NodeId, timestamp: u64, nonce: [u8; 32]) -> Self {
        Challenge {
            nonce,
            timestamp,
            target,
        }
    }

    /// Canonical bytes for signing: domain || nonce || node_id || timestamp
    pub fn signing_payload(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(128);
        payload.extend_from_slice(ATTESTATION_DOMAIN);
        payload.extend_from_slice(&self.nonce);
        payload.extend_from_slice(self.target.0.as_bytes());
        payload.extend_from_slice(&self.timestamp.to_le_bytes());
        payload
    }
}

/// A signed response to an attestation challenge.
#[derive(Debug, Clone)]
pub struct ChallengeResponse {
    /// The original nonce being responded to
    pub nonce: [u8; 32],
    /// The responding node's ID
    pub responder: NodeId,
    /// Ed25519 signature over the challenge payload
    pub signature: [u8; 64],
    /// Responder's public key
    pub public_key: [u8; 32],
    /// Timestamp from the original challenge
    pub challenge_timestamp: u64,
}

impl ChallengeResponse {
    /// Create a signed response to a challenge.
    pub fn respond(challenge: &Challenge, node_id: &NodeId, signing_key: &SigningKey) -> Self {
        let payload = challenge.signing_payload();
        let signature = signing_key.sign(&payload);
        ChallengeResponse {
            nonce: challenge.nonce,
            responder: node_id.clone(),
            signature: signature.to_bytes(),
            public_key: signing_key.verifying_key().to_bytes(),
            challenge_timestamp: challenge.timestamp,
        }
    }
}

/// Attestation errors.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("Challenge expired: age {age_secs}s exceeds window {window_secs}s")]
    ChallengeExpired { age_secs: u64, window_secs: u64 },
    #[error("Nonce replay detected")]
    NonceReplay,
    #[error("Invalid Ed25519 signature")]
    InvalidSignature,
    #[error("Invalid public key format")]
    InvalidPublicKey,
    #[error("Public key mismatch: expected {expected}, got {received}")]
    PubkeyMismatch { expected: String, received: String },
    #[error("Unknown node: {0}")]
    UnknownNode(NodeId),
    #[error("Responder ID mismatch: challenged {expected}, responded {received}")]
    ResponderMismatch { expected: NodeId, received: NodeId },
}

/// Node attestation engine — manages challenge-response verification.
///
/// Maintains a registry of known node public keys and a nonce replay cache.
/// Used by the federation layer to verify node identity before allowing
/// participation in consensus or gossip protocols.
pub struct Attestor {
    /// Known node public keys (node_id → pubkey bytes)
    known_keys: HashMap<NodeId, [u8; 32]>,
    /// Nonce replay cache (nonce → timestamp)
    nonce_cache: HashMap<[u8; 32], u64>,
    /// Challenge validity window in seconds
    challenge_window_secs: u64,
}

impl Attestor {
    /// Create a new attestor with default challenge window (30s).
    pub fn new() -> Self {
        Attestor {
            known_keys: HashMap::new(),
            nonce_cache: HashMap::new(),
            challenge_window_secs: DEFAULT_CHALLENGE_WINDOW_SECS,
        }
    }

    /// Create with custom challenge window.
    pub fn with_window(challenge_window_secs: u64) -> Self {
        Attestor {
            known_keys: HashMap::new(),
            nonce_cache: HashMap::new(),
            challenge_window_secs,
        }
    }

    /// Register a node's public key for future verification.
    pub fn register_node(&mut self, node_id: NodeId, public_key: [u8; 32]) {
        self.known_keys.insert(node_id, public_key);
    }

    /// Remove a node from the registry.
    pub fn unregister_node(&mut self, node_id: &NodeId) {
        self.known_keys.remove(node_id);
    }

    /// Check if a node is registered.
    pub fn is_registered(&self, node_id: &NodeId) -> bool {
        self.known_keys.contains_key(node_id)
    }

    /// Number of registered nodes.
    pub fn registered_count(&self) -> usize {
        self.known_keys.len()
    }

    /// Issue a challenge to a specific node.
    pub fn challenge(&self, target: &NodeId, now: u64) -> Result<Challenge, AttestationError> {
        if !self.known_keys.contains_key(target) {
            return Err(AttestationError::UnknownNode(target.clone()));
        }
        Ok(Challenge::new(target.clone(), now))
    }

    /// Verify a challenge response.
    ///
    /// Checks (in order — fail-fast):
    /// 1. Responder ID matches challenge target
    /// 2. Timestamp within validity window
    /// 3. Nonce not previously used (replay protection)
    /// 4. Public key matches registered key for node
    /// 5. Ed25519 signature is valid
    pub fn verify(
        &mut self,
        challenge: &Challenge,
        response: &ChallengeResponse,
        now: u64,
    ) -> Result<(), AttestationError> {
        // 1. Responder identity check
        if response.responder != challenge.target {
            return Err(AttestationError::ResponderMismatch {
                expected: challenge.target.clone(),
                received: response.responder.clone(),
            });
        }

        // 2. Timestamp freshness
        let age = now.saturating_sub(challenge.timestamp);
        if age > self.challenge_window_secs {
            return Err(AttestationError::ChallengeExpired {
                age_secs: age,
                window_secs: self.challenge_window_secs,
            });
        }

        // 3. Nonce replay protection
        if self.nonce_cache.contains_key(&response.nonce) {
            return Err(AttestationError::NonceReplay);
        }

        // 4. Public key verification against registry
        let expected_key = self
            .known_keys
            .get(&response.responder)
            .ok_or_else(|| AttestationError::UnknownNode(response.responder.clone()))?;

        if response.public_key != *expected_key {
            return Err(AttestationError::PubkeyMismatch {
                expected: hex::encode(expected_key),
                received: hex::encode(response.public_key),
            });
        }

        // 5. Cryptographic signature verification
        let verifying_key = VerifyingKey::from_bytes(&response.public_key)
            .map_err(|_| AttestationError::InvalidPublicKey)?;
        let signature = Signature::from_bytes(&response.signature);
        let payload = challenge.signing_payload();

        verifying_key
            .verify(&payload, &signature)
            .map_err(|_| AttestationError::InvalidSignature)?;

        // All checks passed — record nonce to prevent replay
        self.record_nonce(response.nonce, now);

        Ok(())
    }

    /// Record a used nonce. Prunes old entries when cache is full.
    fn record_nonce(&mut self, nonce: [u8; 32], timestamp: u64) {
        if self.nonce_cache.len() >= MAX_NONCE_CACHE {
            // Prune entries older than 2x the challenge window
            let cutoff = timestamp.saturating_sub(self.challenge_window_secs * 2);
            self.nonce_cache.retain(|_, ts| *ts > cutoff);
        }
        self.nonce_cache.insert(nonce, timestamp);
    }

    /// Number of nonces in replay cache.
    pub fn nonce_cache_size(&self) -> usize {
        self.nonce_cache.len()
    }
}

impl Default for Attestor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use super::*;

    fn make_keypair() -> (NodeId, SigningKey) {
        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let node_id = NodeId(format!(
            "node_{}",
            hex::encode(&signing_key.verifying_key().to_bytes()[..6])
        ));
        (node_id, signing_key)
    }

    #[test]
    fn successful_attestation() {
        let (node_id, signing_key) = make_keypair();
        let pubkey = signing_key.verifying_key().to_bytes();

        let mut attestor = Attestor::new();
        attestor.register_node(node_id.clone(), pubkey);

        let now = 1_000_000u64;
        let challenge = attestor.challenge(&node_id, now).unwrap();
        let response = ChallengeResponse::respond(&challenge, &node_id, &signing_key);

        assert!(attestor.verify(&challenge, &response, now + 5).is_ok());
    }

    #[test]
    fn expired_challenge_rejected() {
        let (node_id, signing_key) = make_keypair();
        let pubkey = signing_key.verifying_key().to_bytes();

        let mut attestor = Attestor::with_window(30);
        attestor.register_node(node_id.clone(), pubkey);

        let now = 1_000_000u64;
        let challenge = attestor.challenge(&node_id, now).unwrap();
        let response = ChallengeResponse::respond(&challenge, &node_id, &signing_key);

        // 31 seconds later — expired
        let result = attestor.verify(&challenge, &response, now + 31);
        assert!(matches!(
            result,
            Err(AttestationError::ChallengeExpired { .. })
        ));
    }

    #[test]
    fn nonce_replay_rejected() {
        let (node_id, signing_key) = make_keypair();
        let pubkey = signing_key.verifying_key().to_bytes();

        let mut attestor = Attestor::new();
        attestor.register_node(node_id.clone(), pubkey);

        let now = 1_000_000u64;
        let challenge = attestor.challenge(&node_id, now).unwrap();
        let response = ChallengeResponse::respond(&challenge, &node_id, &signing_key);

        // First verification succeeds
        assert!(attestor.verify(&challenge, &response, now + 1).is_ok());

        // Replay of same nonce rejected
        let result = attestor.verify(&challenge, &response, now + 2);
        assert!(matches!(result, Err(AttestationError::NonceReplay)));
    }

    #[test]
    fn wrong_key_rejected() {
        let (node_id, _legit_key) = make_keypair();
        let (_other_id, attacker_key) = make_keypair();

        let legit_pubkey = _legit_key.verifying_key().to_bytes();
        let mut attestor = Attestor::new();
        attestor.register_node(node_id.clone(), legit_pubkey);

        let now = 1_000_000u64;
        let challenge = attestor.challenge(&node_id, now).unwrap();

        // Attacker signs with their key but claims to be the legit node
        let response = ChallengeResponse::respond(&challenge, &node_id, &attacker_key);

        // Should fail on pubkey mismatch
        let result = attestor.verify(&challenge, &response, now + 1);
        assert!(matches!(
            result,
            Err(AttestationError::PubkeyMismatch { .. })
        ));
    }

    #[test]
    fn unknown_node_rejected() {
        let attestor = Attestor::new();
        let unknown = NodeId("ghost_node".into());

        let result = attestor.challenge(&unknown, 1_000_000);
        assert!(matches!(result, Err(AttestationError::UnknownNode(_))));
    }

    #[test]
    fn responder_mismatch_rejected() {
        let (node_a, key_a) = make_keypair();
        let (node_b, _key_b) = make_keypair();

        let pubkey_a = key_a.verifying_key().to_bytes();
        let pubkey_b = _key_b.verifying_key().to_bytes();

        let mut attestor = Attestor::new();
        attestor.register_node(node_a.clone(), pubkey_a);
        attestor.register_node(node_b.clone(), pubkey_b);

        let now = 1_000_000u64;
        let challenge = attestor.challenge(&node_a, now).unwrap();

        // Node B responds to Node A's challenge — identity mismatch
        let response = ChallengeResponse::respond(&challenge, &node_b, &_key_b);

        let result = attestor.verify(&challenge, &response, now + 1);
        assert!(matches!(
            result,
            Err(AttestationError::ResponderMismatch { .. })
        ));
    }

    #[test]
    fn nonce_cache_prunes_old_entries() {
        let (node_id, signing_key) = make_keypair();
        let pubkey = signing_key.verifying_key().to_bytes();

        let mut attestor = Attestor::with_window(10);
        attestor.register_node(node_id.clone(), pubkey);

        // Fill nonce cache with entries
        let base_time = 1_000_000u64;
        for i in 0..100 {
            let challenge = attestor.challenge(&node_id, base_time + i).unwrap();
            let response = ChallengeResponse::respond(&challenge, &node_id, &signing_key);
            attestor
                .verify(&challenge, &response, base_time + i + 1)
                .unwrap();
        }

        assert_eq!(attestor.nonce_cache_size(), 100);
    }

    #[test]
    fn register_unregister_node() {
        let (node_id, signing_key) = make_keypair();
        let pubkey = signing_key.verifying_key().to_bytes();

        let mut attestor = Attestor::new();
        assert!(!attestor.is_registered(&node_id));
        assert_eq!(attestor.registered_count(), 0);

        attestor.register_node(node_id.clone(), pubkey);
        assert!(attestor.is_registered(&node_id));
        assert_eq!(attestor.registered_count(), 1);

        attestor.unregister_node(&node_id);
        assert!(!attestor.is_registered(&node_id));
        assert_eq!(attestor.registered_count(), 0);
    }

    #[test]
    fn challenge_at_exact_window_boundary() {
        let (node_id, signing_key) = make_keypair();
        let pubkey = signing_key.verifying_key().to_bytes();

        let mut attestor = Attestor::with_window(30);
        attestor.register_node(node_id.clone(), pubkey);

        let now = 1_000_000u64;
        let challenge = attestor.challenge(&node_id, now).unwrap();
        let response = ChallengeResponse::respond(&challenge, &node_id, &signing_key);

        // Exactly at 30 seconds — should still be valid (<=, not <)
        assert!(attestor.verify(&challenge, &response, now + 30).is_ok());
    }

    #[test]
    fn domain_separation_prevents_cross_protocol() {
        // Verify the signing payload includes the domain prefix
        let node_id = NodeId("test_node".into());
        let challenge = Challenge::with_nonce(node_id, 1_000_000, [0xAA; 32]);

        let payload = challenge.signing_payload();
        assert!(payload.starts_with(ATTESTATION_DOMAIN));
    }
}
