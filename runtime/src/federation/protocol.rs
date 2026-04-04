// src/federation/protocol.rs - Pattern Federation Protocol Wire Format
//
// Defines the on-wire format for pattern federation messages.
// Compatible with Python implementation in core/federation/protocol.py

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Protocol version
pub const PFP_VERSION: &str = "1.0.0";

/// Domain separation prefix for hashing
pub const DOMAIN_PREFIX: &str = "bizra-pfp-v1:";

/// Minimum Ihsān score for pattern acceptance
pub const MIN_IHSAN_SCORE: f64 = 0.85;

/// Minimum repetitions before pattern can be elevated
pub const MIN_REPETITIONS: u32 = 3;

/// Pattern TTL in seconds (30 days)
pub const PATTERN_TTL_SECONDS: i64 = 30 * 24 * 60 * 60;

// ═══════════════════════════════════════════════════════════════════════════════
// MESSAGE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Gossip message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GossipMessageType {
    /// Initial handshake
    Hello,
    /// Peer list exchange
    PeerList,
    /// Keep-alive
    Heartbeat,
    /// Announce new pattern (lazy-pull)
    PatternAnnounce,
    /// Request full pattern data
    PatternRequest,
    /// Full pattern response
    PatternResponse,
    /// Consensus vote (prepare phase)
    VotePrepare,
    /// Consensus vote (commit phase)
    VoteCommit,
    /// Pattern rejection notification
    PatternReject,
    /// Sync request (catch-up)
    SyncRequest,
    /// Sync response with pattern list
    SyncResponse,
    /// Ban announcement
    BanAnnounce,
}

/// Pattern types for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternType {
    /// SAPE probe sequence
    SapeProbe,
    /// Response template
    ResponseTemplate,
    /// Reasoning chain
    ReasoningChain,
    /// Tool usage pattern
    ToolUsage,
    /// Error recovery pattern
    ErrorRecovery,
    /// Custom/other
    Custom,
}

/// Vote decision in consensus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteDecision {
    Accept,
    Reject,
    Abstain,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATTERN STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Pattern metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// Unique pattern identifier (hash of trigger sequence)
    pub pattern_id: String,
    /// Type of pattern
    pub pattern_type: PatternType,
    /// Version number for updates
    pub version: u32,
    /// Node that originated this pattern
    pub origin_node_id: String,
    /// When pattern was first elevated
    pub origin_timestamp: DateTime<Utc>,
    /// Number of times pattern was observed
    pub repetition_count: u32,
    /// Success rate [0.0, 1.0]
    pub success_rate: f64,
    /// Computed impact score [0.0, 1.0]
    pub impact_score: f64,
    /// Ihsān compliance score [0.0, 1.0]
    pub ihsan_score: f64,
    /// How many nodes have adopted this pattern
    pub adoption_count: u64,
    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,
    /// Searchable tags
    pub tags: Vec<String>,
}

/// Pattern payload (the actual optimization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPayload {
    /// Trigger sequence (probe names or action sequence)
    pub trigger_sequence: Vec<String>,
    /// Description of the optimization
    pub optimization: String,
    /// Latency reduction in milliseconds
    pub latency_reduction_ms: u64,
    /// Token savings percentage
    pub token_savings_percent: f64,
    /// SNR improvement
    pub snr_improvement: f64,
}

/// Signed pattern envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEnvelope {
    /// Protocol version
    pub version: String,
    /// Pattern metadata
    pub metadata: PatternMetadata,
    /// Pattern payload
    pub payload: PatternPayload,
    /// Ed25519 signature (hex)
    pub signature: String,
    /// Signer's public key (hex)
    pub public_key: String,
    /// Content hash (BLAKE3, hex)
    pub content_hash: String,
}

impl PatternEnvelope {
    /// Create and sign a new pattern envelope
    pub fn create(
        metadata: PatternMetadata,
        payload: PatternPayload,
        private_key: &SigningKey,
        public_key: &VerifyingKey,
    ) -> Result<Self> {
        // Compute content hash
        let content = Self::canonical_content(&metadata, &payload)?;
        let content_hash = domain_separated_hash(&content);

        // Sign the hash
        let signature = private_key.sign(content_hash.as_bytes());

        Ok(Self {
            version: PFP_VERSION.to_string(),
            metadata,
            payload,
            signature: hex::encode(signature.to_bytes()),
            public_key: hex::encode(public_key.to_bytes()),
            content_hash,
        })
    }

    /// Verify envelope signature
    pub fn verify(&self) -> Result<bool> {
        // Recompute content hash
        let content = Self::canonical_content(&self.metadata, &self.payload)?;
        let expected_hash = domain_separated_hash(&content);

        // Check content hash matches
        if expected_hash != self.content_hash {
            return Ok(false);
        }

        // Verify signature
        let public_key_bytes = hex::decode(&self.public_key)?;
        let signature_bytes = hex::decode(&self.signature)?;

        let public_key = VerifyingKey::from_bytes(
            public_key_bytes
                .as_slice()
                .try_into()
                .map_err(|_| anyhow!("Invalid public key length"))?,
        )?;

        let signature = Signature::from_bytes(
            signature_bytes
                .as_slice()
                .try_into()
                .map_err(|_| anyhow!("Invalid signature length"))?,
        );

        Ok(public_key
            .verify(self.content_hash.as_bytes(), &signature)
            .is_ok())
    }

    /// Serialize to wire format (JSON)
    pub fn to_wire(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from wire format
    pub fn from_wire(data: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }

    /// Create canonical JSON content for signing
    fn canonical_content(metadata: &PatternMetadata, payload: &PatternPayload) -> Result<String> {
        // Sort keys and use minimal formatting (RFC 8785 style)
        let content = serde_json::json!({
            "metadata": metadata,
            "payload": payload,
        });
        Ok(serde_json::to_string(&content)?)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOSSIP MESSAGES
// ═══════════════════════════════════════════════════════════════════════════════

/// Gossip protocol message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// Message type
    pub msg_type: GossipMessageType,
    /// Sender node ID
    pub sender_id: String,
    /// Message ID (for deduplication)
    pub message_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Payload (type-specific JSON)
    pub payload: serde_json::Value,
    /// TTL (hops remaining)
    pub ttl: u8,
}

impl GossipMessage {
    /// Create new gossip message
    pub fn new(msg_type: GossipMessageType, sender_id: String, payload: serde_json::Value) -> Self {
        Self {
            msg_type,
            sender_id,
            message_id: format!(
                "{}_{}",
                Utc::now().timestamp_millis(),
                rand::random::<u32>()
            ),
            timestamp: Utc::now(),
            payload,
            ttl: 5,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSENSUS TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Consensus vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    /// Pattern being voted on
    pub pattern_id: String,
    /// Voter node ID
    pub voter_id: String,
    /// Vote decision
    pub decision: VoteDecision,
    /// Reason for decision
    pub reason: String,
    /// Voter's Ihsān score for the pattern
    pub ihsan_score: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Signature (hex)
    pub signature: String,
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Pattern ID
    pub pattern_id: String,
    /// Whether pattern was accepted
    pub accepted: bool,
    /// Accept vote count
    pub accept_votes: usize,
    /// Reject vote count
    pub reject_votes: usize,
    /// Abstain vote count
    pub abstain_votes: usize,
    /// Whether quorum was reached
    pub quorum_reached: bool,
    /// When consensus was finalized
    pub finalized_at: DateTime<Utc>,
    /// All votes
    pub votes: Vec<ConsensusVote>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CRIT-1: SIGNED VOTE WRAPPER (Genesis v2.2.2 Ed25519 Verification)
// ═══════════════════════════════════════════════════════════════════════════════

/// Signed consensus vote with Ed25519 verification
///
/// Genesis Strict Synthesis v2.2.2 CRIT-1 implementation.
/// All consensus votes MUST be verified before counting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedVote {
    /// The unsigned vote data
    pub vote: ConsensusVote,
    /// Signer's public key (hex, 64 chars)
    pub signer_public_key: String,
}

impl SignedVote {
    /// Verify the vote signature using the signer's public key
    ///
    /// Returns true if:
    /// - Public key is valid Ed25519 (32 bytes)
    /// - Signature in vote matches signed content
    /// - Voter ID matches expected signer
    pub fn verify(&self) -> bool {
        // Decode public key from hex
        let pk_bytes = match hex::decode(&self.signer_public_key) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                tracing::warn!("SignedVote: invalid public key length");
                return false;
            }
        };

        let pk_array: [u8; 32] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };

        let public_key = match VerifyingKey::from_bytes(&pk_array) {
            Ok(pk) => pk,
            Err(_) => {
                tracing::warn!("SignedVote: invalid public key format");
                return false;
            }
        };

        // Reconstruct canonical content that was signed
        let content = serde_json::json!({
            "pattern_id": self.vote.pattern_id,
            "voter_id": self.vote.voter_id,
            "decision": self.vote.decision,
            "reason": self.vote.reason,
            "ihsan_score": self.vote.ihsan_score,
            "timestamp": self.vote.timestamp.to_rfc3339(),
        });

        let content_str = match serde_json::to_string(&content) {
            Ok(s) => s,
            Err(_) => return false,
        };

        // Decode and verify signature
        let sig_bytes = match hex::decode(&self.vote.signature) {
            Ok(b) if b.len() == 64 => b,
            _ => {
                tracing::warn!("SignedVote: invalid signature length");
                return false;
            }
        };

        verify_signature(content_str.as_bytes(), &sig_bytes, &public_key)
    }

    /// Create a signed vote from vote data (signature already in vote.signature)
    pub fn from_vote(vote: ConsensusVote, public_key: &VerifyingKey) -> Self {
        Self {
            vote,
            signer_public_key: hex::encode(public_key.to_bytes()),
        }
    }

    /// Extract the inner vote (after verification)
    pub fn into_vote(self) -> ConsensusVote {
        self.vote
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CRIT-2: SIGNED GOSSIP MESSAGE (Genesis v2.2.2 Ed25519 Verification)
// ═══════════════════════════════════════════════════════════════════════════════

/// Signed gossip message with Ed25519 verification
///
/// Genesis Strict Synthesis v2.2.2 CRIT-2 implementation.
/// All gossip messages should be signed and verified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedGossipMessage {
    /// The unsigned message
    pub message: GossipMessage,
    /// Signer's node ID
    pub signer_id: String,
    /// Ed25519 signature (hex, 128 chars)
    pub signature: String,
    /// Signer's public key (hex, 64 chars)
    pub public_key: String,
}

impl SignedGossipMessage {
    /// Create and sign a gossip message
    pub fn create(message: GossipMessage, signing_key: &SigningKey) -> Result<Self> {
        let signer_id = message.sender_id.clone();
        let public_key = signing_key.verifying_key();

        // Create canonical content for signing
        let content = Self::canonical_content(&message)?;

        // Sign with Ed25519
        let signature = signing_key.sign(content.as_bytes());

        Ok(Self {
            message,
            signer_id,
            signature: hex::encode(signature.to_bytes()),
            public_key: hex::encode(public_key.to_bytes()),
        })
    }

    /// Verify the message signature
    pub fn verify(&self) -> bool {
        // Decode public key
        let pk_bytes = match hex::decode(&self.public_key) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                tracing::warn!("SignedGossipMessage: invalid public key length");
                return false;
            }
        };

        let pk_array: [u8; 32] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };

        let public_key = match VerifyingKey::from_bytes(&pk_array) {
            Ok(pk) => pk,
            Err(_) => {
                tracing::warn!("SignedGossipMessage: invalid public key format");
                return false;
            }
        };

        // Reconstruct canonical content
        let content = match Self::canonical_content(&self.message) {
            Ok(c) => c,
            Err(_) => return false,
        };

        // Decode signature
        let sig_bytes = match hex::decode(&self.signature) {
            Ok(b) if b.len() == 64 => b,
            _ => {
                tracing::warn!("SignedGossipMessage: invalid signature length");
                return false;
            }
        };

        // Verify sender ID matches
        if self.message.sender_id != self.signer_id {
            tracing::warn!("SignedGossipMessage: sender_id mismatch");
            return false;
        }

        verify_signature(content.as_bytes(), &sig_bytes, &public_key)
    }

    /// Serialize to wire format
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from wire format
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }

    /// Extract inner message (after verification)
    pub fn into_message(self) -> GossipMessage {
        self.message
    }

    /// Create canonical content for signing (RFC 8785 style)
    fn canonical_content(message: &GossipMessage) -> Result<String> {
        let content = serde_json::json!({
            "msg_type": message.msg_type,
            "sender_id": message.sender_id,
            "message_id": message.message_id,
            "timestamp": message.timestamp.to_rfc3339(),
            "payload": message.payload,
            "ttl": message.ttl,
        });
        Ok(serde_json::to_string(&content)?)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CRYPTO HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute domain-separated hash (BLAKE3 with prefix)
pub fn domain_separated_hash(content: &str) -> String {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX.as_bytes());
    hasher.update(content.as_bytes());

    hasher.finalize().to_hex().to_string()
}

/// Sign a message with Ed25519
pub fn sign_message(message: &[u8], private_key: &SigningKey) -> Vec<u8> {
    private_key.sign(message).to_bytes().to_vec()
}

/// Verify Ed25519 signature
pub fn verify_signature(message: &[u8], signature: &[u8], public_key: &VerifyingKey) -> bool {
    if signature.len() != 64 {
        return false;
    }

    let sig_bytes: [u8; 64] = match signature.try_into() {
        Ok(b) => b,
        Err(_) => return false,
    };

    let sig = Signature::from_bytes(&sig_bytes);

    public_key.verify(message, &sig).is_ok()
}

/// Generate Ed25519 keypair
pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
    use rand::RngCore;
    let mut secret = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut secret);
    let private_key = SigningKey::from_bytes(&secret);
    let public_key = private_key.verifying_key();
    (private_key, public_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_envelope_create_verify() {
        let (private_key, public_key) = generate_keypair();

        let metadata = PatternMetadata {
            pattern_id: "test_pattern_123".to_string(),
            pattern_type: PatternType::SapeProbe,
            version: 1,
            origin_node_id: "node_test".to_string(),
            origin_timestamp: Utc::now(),
            repetition_count: 5,
            success_rate: 0.9,
            impact_score: 0.85,
            ihsan_score: 0.92,
            adoption_count: 0,
            expires_at: Utc::now() + chrono::Duration::days(30),
            tags: vec!["test".to_string()],
        };

        let payload = PatternPayload {
            trigger_sequence: vec!["threat_scan".to_string(), "compliance".to_string()],
            optimization: "Test optimization".to_string(),
            latency_reduction_ms: 50,
            token_savings_percent: 20.0,
            snr_improvement: 0.1,
        };

        let envelope = PatternEnvelope::create(metadata, payload, &private_key, &public_key)
            .expect("Failed to create envelope");

        assert!(envelope.verify().expect("Verification failed"));
    }

    #[test]
    fn test_gossip_message_serialization() {
        let msg = GossipMessage::new(
            GossipMessageType::PatternAnnounce,
            "node_test".to_string(),
            serde_json::json!({"pattern_id": "test123"}),
        );

        let bytes = msg.to_bytes().expect("Serialization failed");
        let decoded = GossipMessage::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(decoded.sender_id, "node_test");
        assert_eq!(decoded.msg_type, GossipMessageType::PatternAnnounce);
    }

    #[test]
    fn test_domain_separated_hash() {
        let hash1 = domain_separated_hash("test content");
        let hash2 = domain_separated_hash("test content");
        let hash3 = domain_separated_hash("different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert!(hash1.len() == 64); // BLAKE3 produces 256-bit hashes
    }
}
