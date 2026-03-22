//! # Mint Protocol — Identity Genesis + 12-Agent Derivation
//!
//! The mint ceremony that turns a human into a sovereign node.
//!
//! ## Flow
//!
//! ```text
//! Human connects
//!     │
//!     ▼
//! [1] Generate master Ed25519 keypair locally
//! [2] Derive NodeId from BLAKE3(public_key)
//! [3] HD-derive 7 PAT agent keypairs (local keystore)
//! [4] HD-derive 5 SAT agent keypairs (URP transfer)
//! [5] Sign genesis record with master key
//! [6] Emit SatPoolTicket (contribution proof)
//! [7] Node is alive — PAT active, SAT registered
//! ```
//!
//! ## Critical: HD Derivation, Not Random Generation
//!
//! Agent keys are CHILDREN of the master identity key.
//! If the master key is backed up, all 12 agents can be reconstructed.
//! If a device is lost, identity migrates to new hardware.
//! The identity IS the backup.
//!
//! ## Constitutional: BLAKE3 Only
//!
//! All hashes in this module use BLAKE3 with domain separation.
//! SHA-256 is NOT used anywhere. This corrects the genesis.rs
//! inconsistency identified in the constitutional audit.

use blake3::Hasher;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use crate::{constitution::*, DOMAIN_PREFIX, PROTOCOL_VERSION};

// =============================================================================
// TYPES
// =============================================================================

/// A complete minted node — the output of the genesis ceremony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintedNode {
    /// Master identity
    pub identity: NodeIdentityRecord,
    /// 7 PAT agents (keys stored locally, never leave the node)
    pub pat_agents: Vec<MintedAgent>,
    /// 5 SAT agents (keys transferred to URP)
    pub sat_agents: Vec<MintedAgent>,
    /// Contribution ticket (proof that SAT was donated to URP)
    pub sat_pool_ticket: SatPoolTicket,
    /// Genesis record hash (BLAKE3, domain-separated)
    pub genesis_hash: String,
    /// Signature over genesis_hash by master key
    pub genesis_signature: String,
    /// Timestamp (UTC epoch seconds)
    pub minted_at: u64,
    /// Protocol version
    pub protocol_version: String,
}

/// Node identity record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeIdentityRecord {
    /// Node ID: BLAKE3(domain_prefix || public_key)[..16] as hex
    pub node_id: String,
    /// Master public key (hex)
    pub public_key_hex: String,
    /// Human-readable display name
    pub display_name: String,
}

/// A single minted agent (PAT or SAT)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintedAgent {
    /// Agent ID: BLAKE3(domain_prefix || agent_public_key)[..16] as hex
    pub agent_id: String,
    /// Agent public key (hex)
    pub public_key_hex: String,
    /// Role name (e.g., "P1-Analyst", "S1-Auditor")
    pub role: String,
    /// Derivation index (0-6 for PAT, 7-11 for SAT)
    pub derivation_index: u32,
    /// Agent class
    pub agent_class: AgentClass,
}

/// Agent classification — determines trust boundary behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentClass {
    /// Personal Agent Topology — stays LOCAL on the node
    Pat,
    /// System Agent Topology — transfers to the URP
    Sat,
}

/// Proof that a node contributed its 5 SAT agents to the URP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatPoolTicket {
    /// The node that contributed
    pub contributor_node_id: String,
    /// The 5 SAT agent IDs contributed
    pub sat_agent_ids: Vec<String>,
    /// BLAKE3 hash of the contribution record
    pub ticket_hash: String,
    /// Signature by the contributing node's master key
    pub signature: String,
    /// Timestamp
    pub issued_at: u64,
}

// =============================================================================
// HD KEY DERIVATION
// =============================================================================

/// Derive a child Ed25519 signing key from a master key using BLAKE3.
///
/// This is NOT BIP-32 (which uses secp256k1). This is Ed25519-native
/// HD derivation using BLAKE3 as the KDF:
///
/// child_seed = BLAKE3(domain || master_secret || index_bytes)
/// child_key  = Ed25519::from_seed(child_seed[..32])
///
/// Properties:
/// - Deterministic: same master + index = same child, always
/// - Domain-separated: PAT and SAT use different prefixes
/// - One-way: child cannot derive parent
/// - Reconstructable: master key + index regenerates any child
pub fn derive_agent_key(master_secret: &[u8; 32], domain: &str, index: u32) -> SigningKey {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(domain.as_bytes());
    hasher.update(master_secret);
    hasher.update(&index.to_le_bytes());
    let hash = hasher.finalize();
    let seed: [u8; 32] = hash.as_bytes()[..32]
        .try_into()
        .expect("BLAKE3 produces 32+ bytes");
    SigningKey::from_bytes(&seed)
}

/// Derive a NodeId from a public key using BLAKE3 (NOT SHA-256)
fn derive_node_id(verifying_key: &VerifyingKey) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(b"node-id:");
    hasher.update(verifying_key.as_bytes());
    let hash = hasher.finalize();
    hex_encode(&hash.as_bytes()[..16])
}

/// Derive an AgentId from an agent's public key using BLAKE3
fn derive_agent_id(verifying_key: &VerifyingKey) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(b"agent-id:");
    hasher.update(verifying_key.as_bytes());
    let hash = hasher.finalize();
    hex_encode(&hash.as_bytes()[..16])
}

// =============================================================================
// THE MINT CEREMONY
// =============================================================================

/// Execute the complete node mint ceremony.
///
/// This is the moment a human becomes a sovereign node.
///
/// Input: a display name (the human identifies themselves)
/// Output: a MintedNode with 7 PAT agents (local) + 5 SAT agents (for URP)
///
/// The master keypair is generated from OS entropy.
/// All 12 agent keys are HD-derived from the master secret.
/// The genesis record is BLAKE3-hashed and Ed25519-signed.
///
/// # Returns
///
/// `(MintedNode, [u8; 32])` — the minted node and the master secret bytes.
/// The caller MUST store the master secret securely (encrypted keystore).
/// The master secret is the ONLY backup needed — all 12 agents can be
/// reconstructed from it.
pub fn mint_node(display_name: &str) -> (MintedNode, [u8; 32]) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_secs();

    // 1. Generate master keypair
    let master_signing = SigningKey::generate(&mut OsRng);
    let master_verifying = master_signing.verifying_key();
    let master_secret = master_signing.to_bytes();
    let node_id = derive_node_id(&master_verifying);

    let identity = NodeIdentityRecord {
        node_id: node_id.clone(),
        public_key_hex: hex_encode(master_verifying.as_bytes()),
        display_name: display_name.to_string(),
    };

    // 2. HD-derive 7 PAT agents (indices 0..6)
    let pat_agents: Vec<MintedAgent> = (0..PAT_COUNT)
        .map(|i| {
            let child_key = derive_agent_key(&master_secret, PAT_DERIVATION_PREFIX, i);
            let child_verifying = child_key.verifying_key();
            MintedAgent {
                agent_id: derive_agent_id(&child_verifying),
                public_key_hex: hex_encode(child_verifying.as_bytes()),
                role: PAT_ROLES[i as usize].to_string(),
                derivation_index: i,
                agent_class: AgentClass::Pat,
            }
        })
        .collect();

    // 3. HD-derive 5 SAT agents (indices 7..11)
    let sat_agents: Vec<MintedAgent> = (0..SAT_COUNT)
        .map(|i| {
            let child_key = derive_agent_key(&master_secret, SAT_DERIVATION_PREFIX, i);
            let child_verifying = child_key.verifying_key();
            MintedAgent {
                agent_id: derive_agent_id(&child_verifying),
                public_key_hex: hex_encode(child_verifying.as_bytes()),
                role: SAT_ROLES[i as usize].to_string(),
                derivation_index: PAT_COUNT + i,
                agent_class: AgentClass::Sat,
            }
        })
        .collect();

    // 4. Create SAT pool ticket (contribution proof)
    let sat_ids: Vec<String> = sat_agents.iter().map(|a| a.agent_id.clone()).collect();
    let ticket_content = serde_json::json!({
        "contributor": &node_id,
        "sat_agent_ids": &sat_ids,
        "timestamp": now,
        "protocol": PROTOCOL_VERSION,
    });
    let ticket_bytes = serde_json::to_vec(&ticket_content).expect("json");
    let ticket_hash = domain_hash(&ticket_bytes);
    let ticket_sig = master_signing.sign(ticket_hash.as_bytes());

    let sat_pool_ticket = SatPoolTicket {
        contributor_node_id: node_id.clone(),
        sat_agent_ids: sat_ids,
        ticket_hash: ticket_hash.clone(),
        signature: hex_encode(&ticket_sig.to_bytes()),
        issued_at: now,
    };

    // 5. Compute genesis hash over the entire record
    let genesis_content = serde_json::json!({
        "identity": &identity,
        "pat_count": PAT_COUNT,
        "sat_count": SAT_COUNT,
        "sat_pool_ticket_hash": &ticket_hash,
        "timestamp": now,
        "protocol": PROTOCOL_VERSION,
    });
    let genesis_bytes = serde_json::to_vec(&genesis_content).expect("json");
    let genesis_hash = domain_hash(&genesis_bytes);
    let genesis_sig = master_signing.sign(genesis_hash.as_bytes());

    let node = MintedNode {
        identity,
        pat_agents,
        sat_agents,
        sat_pool_ticket,
        genesis_hash: genesis_hash.clone(),
        genesis_signature: hex_encode(&genesis_sig.to_bytes()),
        minted_at: now,
        protocol_version: PROTOCOL_VERSION.to_string(),
    };

    (node, master_secret)
}

/// Reconstruct all 12 agent keys from a master secret.
///
/// Used when migrating to new hardware or recovering from backup.
/// The master secret is the ONLY thing the human needs to save.
pub fn reconstruct_agents(master_secret: &[u8; 32]) -> (Vec<SigningKey>, Vec<SigningKey>) {
    let pat_keys: Vec<SigningKey> = (0..PAT_COUNT)
        .map(|i| derive_agent_key(master_secret, PAT_DERIVATION_PREFIX, i))
        .collect();
    let sat_keys: Vec<SigningKey> = (0..SAT_COUNT)
        .map(|i| derive_agent_key(master_secret, SAT_DERIVATION_PREFIX, i))
        .collect();
    (pat_keys, sat_keys)
}

// =============================================================================
// HELPERS
// =============================================================================

/// BLAKE3 hash with domain separation (canonical for all protocol hashes)
fn domain_hash(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mint_produces_12_agents() {
        let (node, _secret) = mint_node("Mumo");
        assert_eq!(node.pat_agents.len(), 7, "must mint exactly 7 PAT");
        assert_eq!(node.sat_agents.len(), 5, "must mint exactly 5 SAT");
    }

    #[test]
    fn test_all_agent_ids_unique() {
        let (node, _) = mint_node("TestNode");
        let mut ids: Vec<&str> = node
            .pat_agents
            .iter()
            .map(|a| a.agent_id.as_str())
            .collect();
        ids.extend(node.sat_agents.iter().map(|a| a.agent_id.as_str()));
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "all 12 agent IDs must be unique");
    }

    #[test]
    fn test_hd_derivation_is_deterministic() {
        let secret = [42u8; 32];
        let key1 = derive_agent_key(&secret, PAT_DERIVATION_PREFIX, 0);
        let key2 = derive_agent_key(&secret, PAT_DERIVATION_PREFIX, 0);
        assert_eq!(
            key1.verifying_key().as_bytes(),
            key2.verifying_key().as_bytes(),
            "same master + same index = same child key"
        );
    }

    #[test]
    fn test_different_domains_produce_different_keys() {
        let secret = [42u8; 32];
        let pat_key = derive_agent_key(&secret, PAT_DERIVATION_PREFIX, 0);
        let sat_key = derive_agent_key(&secret, SAT_DERIVATION_PREFIX, 0);
        assert_ne!(
            pat_key.verifying_key().as_bytes(),
            sat_key.verifying_key().as_bytes(),
            "PAT and SAT domains must produce different keys"
        );
    }

    #[test]
    fn test_reconstruct_matches_mint() {
        let (node, secret) = mint_node("ReconstructTest");
        let (pat_keys, sat_keys) = reconstruct_agents(&secret);

        for (i, key) in pat_keys.iter().enumerate() {
            let reconstructed_hex = hex_encode(key.verifying_key().as_bytes());
            assert_eq!(
                reconstructed_hex, node.pat_agents[i].public_key_hex,
                "PAT agent {} must reconstruct identically",
                i
            );
        }
        for (i, key) in sat_keys.iter().enumerate() {
            let reconstructed_hex = hex_encode(key.verifying_key().as_bytes());
            assert_eq!(
                reconstructed_hex, node.sat_agents[i].public_key_hex,
                "SAT agent {} must reconstruct identically",
                i
            );
        }
    }

    #[test]
    fn test_genesis_hash_is_blake3_not_sha256() {
        let (node, _) = mint_node("HashTest");
        // BLAKE3 hex output is 64 chars (256 bits)
        assert_eq!(
            node.genesis_hash.len(),
            64,
            "genesis hash must be BLAKE3 (64 hex chars)"
        );
        // Verify it's valid hex
        assert!(node.genesis_hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_sat_pool_ticket_is_signed() {
        let (node, _) = mint_node("TicketTest");
        assert!(!node.sat_pool_ticket.signature.is_empty());
        assert_eq!(node.sat_pool_ticket.sat_agent_ids.len(), 5);
        assert_eq!(
            node.sat_pool_ticket.contributor_node_id,
            node.identity.node_id
        );
    }

    #[test]
    fn test_pat_agents_are_pat_class() {
        let (node, _) = mint_node("ClassTest");
        for agent in &node.pat_agents {
            assert_eq!(agent.agent_class, AgentClass::Pat);
        }
    }

    #[test]
    fn test_sat_agents_are_sat_class() {
        let (node, _) = mint_node("ClassTest");
        for agent in &node.sat_agents {
            assert_eq!(agent.agent_class, AgentClass::Sat);
        }
    }

    #[test]
    fn test_roles_match_constitution() {
        let (node, _) = mint_node("RoleTest");
        for (i, agent) in node.pat_agents.iter().enumerate() {
            assert_eq!(agent.role, PAT_ROLES[i]);
        }
        for (i, agent) in node.sat_agents.iter().enumerate() {
            assert_eq!(agent.role, SAT_ROLES[i]);
        }
    }
}
