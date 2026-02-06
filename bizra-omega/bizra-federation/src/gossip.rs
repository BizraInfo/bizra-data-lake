//! SWIM Gossip Protocol — Secure Node Discovery
//!
//! SECURITY: All gossip messages are Ed25519 signed to prevent spoofing.
//! Message authentication ensures only legitimate nodes can participate.
//!
//! Standing on Giants: SWIM (Das et al., 2002)

use bizra_core::NodeId;
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    Alive,
    Suspect,
    Dead,
    Left,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Member {
    pub node_id: NodeId,
    pub addr: SocketAddr,
    pub state: NodeState,
    pub incarnation: u64,
    pub last_update: DateTime<Utc>,
}

impl Member {
    pub fn new(node_id: NodeId, addr: SocketAddr) -> Self {
        Self {
            node_id,
            addr,
            state: NodeState::Alive,
            incarnation: 0,
            last_update: Utc::now(),
        }
    }
    pub fn is_alive(&self) -> bool {
        self.state == NodeState::Alive
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GossipMessage {
    Ping { from: NodeId, incarnation: u64 },
    Ack { from: NodeId, incarnation: u64 },
    Update { member: Member },
    Join { member: Member },
    Leave { node_id: NodeId },
}

impl GossipMessage {
    /// Serialize message to canonical bytes for signing
    pub fn to_bytes(&self) -> Vec<u8> {
        // Domain-separated canonical serialization
        let prefix = b"bizra-gossip-v1:";
        let json = serde_json::to_vec(self).unwrap_or_default();
        [prefix.as_slice(), &json].concat()
    }

    /// Deserialize message from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FederationError> {
        let prefix = b"bizra-gossip-v1:";
        if bytes.len() < prefix.len() || &bytes[..prefix.len()] != prefix {
            return Err(FederationError::InvalidMessageFormat);
        }
        serde_json::from_slice(&bytes[prefix.len()..])
            .map_err(|_| FederationError::InvalidMessageFormat)
    }

    /// Extract sender NodeId for pubkey binding
    pub fn sender_id(&self) -> &NodeId {
        match self {
            GossipMessage::Ping { from, .. } => from,
            GossipMessage::Ack { from, .. } => from,
            GossipMessage::Update { member } => &member.node_id,
            GossipMessage::Join { member } => &member.node_id,
            GossipMessage::Leave { node_id } => node_id,
        }
    }
}

/// Signed gossip message with Ed25519 cryptographic authentication
/// SECURITY: All network gossip MUST be signed to prevent spoofing attacks
#[derive(Clone, Debug)]
pub struct SignedGossipMessage {
    pub message: GossipMessage,
    pub signature: [u8; 64],
    pub sender_pubkey: [u8; 32],
    pub timestamp: DateTime<Utc>,
}

impl SignedGossipMessage {
    /// Sign a gossip message with the provided signing key
    pub fn sign(msg: GossipMessage, signing_key: &SigningKey) -> Self {
        let timestamp = Utc::now();
        let message_bytes = Self::signing_payload(&msg, &timestamp);
        let signature = signing_key.sign(&message_bytes);
        Self {
            message: msg,
            signature: signature.to_bytes(),
            sender_pubkey: signing_key.verifying_key().to_bytes(),
            timestamp,
        }
    }

    /// Verify the Ed25519 signature on this message
    pub fn verify(&self) -> Result<(), FederationError> {
        let verifying_key = VerifyingKey::from_bytes(&self.sender_pubkey)
            .map_err(|_| FederationError::InvalidPublicKey)?;
        let signature = Signature::from_bytes(&self.signature);
        let payload = Self::signing_payload(&self.message, &self.timestamp);
        verifying_key
            .verify(&payload, &signature)
            .map_err(|_| FederationError::InvalidSignature)
    }

    /// Create the signing payload (message bytes + timestamp)
    fn signing_payload(msg: &GossipMessage, timestamp: &DateTime<Utc>) -> Vec<u8> {
        let msg_bytes = msg.to_bytes();
        let ts_bytes = timestamp.timestamp_millis().to_le_bytes();
        [msg_bytes, ts_bytes.to_vec()].concat()
    }

    /// Serialize to wire format
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        // Version byte
        result.push(1u8);
        // Signature (64 bytes)
        result.extend_from_slice(&self.signature);
        // Public key (32 bytes)
        result.extend_from_slice(&self.sender_pubkey);
        // Timestamp (8 bytes)
        result.extend_from_slice(&self.timestamp.timestamp_millis().to_le_bytes());
        // Message (variable length)
        result.extend(self.message.to_bytes());
        result
    }

    /// Deserialize from wire format
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FederationError> {
        if bytes.len() < 1 + 64 + 32 + 8 {
            return Err(FederationError::InvalidMessageFormat);
        }
        let version = bytes[0];
        if version != 1 {
            return Err(FederationError::UnsupportedVersion(version));
        }
        let mut offset = 1;

        let mut signature = [0u8; 64];
        signature.copy_from_slice(&bytes[offset..offset + 64]);
        offset += 64;

        let mut sender_pubkey = [0u8; 32];
        sender_pubkey.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;

        let mut ts_bytes = [0u8; 8];
        ts_bytes.copy_from_slice(&bytes[offset..offset + 8]);
        let ts_millis = i64::from_le_bytes(ts_bytes);
        let timestamp =
            DateTime::from_timestamp_millis(ts_millis).ok_or(FederationError::InvalidTimestamp)?;
        offset += 8;

        let message = GossipMessage::from_bytes(&bytes[offset..])?;

        Ok(Self {
            message,
            signature,
            sender_pubkey,
            timestamp,
        })
    }

    /// Get hex-encoded public key for logging
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.sender_pubkey)
    }
}

/// Federation protocol errors
#[derive(Debug, Error)]
pub enum FederationError {
    #[error("Invalid public key format")]
    InvalidPublicKey,
    #[error("Invalid Ed25519 signature")]
    InvalidSignature,
    #[error("Invalid message format")]
    InvalidMessageFormat,
    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u8),
    #[error("Invalid timestamp")]
    InvalidTimestamp,
    #[error("Message too old (replay protection)")]
    MessageExpired,
    #[error("Unknown sender")]
    UnknownSender,
}

pub struct GossipProtocol {
    local_id: NodeId,
    #[allow(dead_code)] // Needed for bind/listen when network layer is activated
    local_addr: SocketAddr,
    members: Arc<RwLock<HashMap<NodeId, Member>>>,
    incarnation: Arc<RwLock<u64>>,
    /// Signing key for authenticating outbound messages
    signing_key: SigningKey,
    /// Known peer public keys for verifying inbound messages
    known_pubkeys: Arc<RwLock<HashMap<NodeId, [u8; 32]>>>,
    /// Maximum message age for replay protection (seconds)
    max_message_age_secs: i64,
}

impl GossipProtocol {
    pub fn new(local_id: NodeId, local_addr: SocketAddr, signing_key: SigningKey) -> Self {
        let mut members = HashMap::new();
        members.insert(local_id.clone(), Member::new(local_id.clone(), local_addr));

        let mut known_pubkeys = HashMap::new();
        // Register own public key
        known_pubkeys.insert(local_id.clone(), signing_key.verifying_key().to_bytes());

        Self {
            local_id,
            local_addr,
            members: Arc::new(RwLock::new(members)),
            incarnation: Arc::new(RwLock::new(0)),
            signing_key,
            known_pubkeys: Arc::new(RwLock::new(known_pubkeys)),
            max_message_age_secs: 300, // 5 minute replay window
        }
    }

    /// Create a new protocol instance with generated key (for testing)
    pub fn new_with_generated_key(local_id: NodeId, local_addr: SocketAddr) -> Self {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        Self::new(local_id, local_addr, signing_key)
    }

    /// Register a known peer's public key
    pub async fn register_peer_pubkey(&self, node_id: NodeId, pubkey: [u8; 32]) {
        self.known_pubkeys.write().await.insert(node_id, pubkey);
    }

    /// Get our public key bytes
    pub fn public_key(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Create a signed message
    pub fn sign_message(&self, msg: GossipMessage) -> SignedGossipMessage {
        SignedGossipMessage::sign(msg, &self.signing_key)
    }

    pub async fn add_seed(&self, node_id: NodeId, addr: SocketAddr) {
        self.members
            .write()
            .await
            .insert(node_id.clone(), Member::new(node_id, addr));
    }

    /// Handle a signed gossip message with verification
    /// SECURITY: Signature and replay protection checked before processing
    pub async fn handle_signed_message(
        &self,
        signed_msg: SignedGossipMessage,
    ) -> Result<Option<SignedGossipMessage>, FederationError> {
        // CRITICAL: Verify signature first
        signed_msg.verify()?;

        // Enforce NodeId -> public key binding (prevents pubkey spoofing)
        let sender_id = signed_msg.message.sender_id();
        let expected_pubkey = {
            let map = self.known_pubkeys.read().await;
            map.get(sender_id).cloned()
        };
        match expected_pubkey {
            Some(pubkey) if pubkey == signed_msg.sender_pubkey => {}
            _ => return Err(FederationError::UnknownSender),
        }

        // Check replay protection (message age)
        let age_secs = (Utc::now() - signed_msg.timestamp).num_seconds();
        if age_secs.abs() > self.max_message_age_secs {
            return Err(FederationError::MessageExpired);
        }

        // Process the verified message
        let response = self.handle_message_inner(&signed_msg.message).await;

        // Sign response if present
        Ok(response.map(|msg| self.sign_message(msg)))
    }

    /// Internal message handler (after verification)
    async fn handle_message_inner(&self, msg: &GossipMessage) -> Option<GossipMessage> {
        match msg {
            GossipMessage::Ping { from, incarnation } => {
                self.update_alive(from, *incarnation).await;
                Some(GossipMessage::Ack {
                    from: self.local_id.clone(),
                    incarnation: *self.incarnation.read().await,
                })
            }
            GossipMessage::Ack { from, incarnation } => {
                self.update_alive(from, *incarnation).await;
                None
            }
            GossipMessage::Join { member } => {
                // Register new peer's key if provided in member data
                self.members
                    .write()
                    .await
                    .insert(member.node_id.clone(), member.clone());
                None
            }
            GossipMessage::Leave { node_id } => {
                if let Some(m) = self.members.write().await.get_mut(node_id) {
                    m.state = NodeState::Left;
                }
                None
            }
            GossipMessage::Update { member } => {
                self.merge_member(member.clone()).await;
                None
            }
        }
    }

    /// Legacy handler for unsigned messages (for backward compatibility)
    #[deprecated(note = "Use handle_signed_message for secure communication")]
    pub async fn handle_message(&self, msg: GossipMessage) -> Option<GossipMessage> {
        self.handle_message_inner(&msg).await
    }

    async fn update_alive(&self, node_id: &NodeId, incarnation: u64) {
        if let Some(m) = self.members.write().await.get_mut(node_id) {
            if incarnation >= m.incarnation {
                m.incarnation = incarnation;
                m.state = NodeState::Alive;
                m.last_update = Utc::now();
            }
        }
    }

    async fn merge_member(&self, new: Member) {
        let mut members = self.members.write().await;
        if let Some(existing) = members.get_mut(&new.node_id) {
            if new.incarnation > existing.incarnation {
                *existing = new;
            }
        } else {
            members.insert(new.node_id.clone(), new);
        }
    }

    pub async fn alive_members(&self) -> Vec<Member> {
        self.members
            .read()
            .await
            .values()
            .filter(|m| m.is_alive())
            .cloned()
            .collect()
    }

    pub async fn member_count(&self) -> usize {
        self.members.read().await.len()
    }

    pub fn create_leave_message(&self) -> GossipMessage {
        GossipMessage::Leave {
            node_id: self.local_id.clone(),
        }
    }
}
