// bizra-node/src/identity_registry.rs
// ============================================================
// Identity Registry — Bind agents to cryptographic identities
// ============================================================
//
// Genesis review P0: "receipts need identity binding. Any node
// can claim to have signed a receipt. The identity registry
// binds agent IDs to expected public keys."
//
// Every agent minted on Block 0 gets an Ed25519 keypair.
// The registry maps agent_id → verifying_key.
// Receipt verification checks: is the signer in the registry?
//
// Standing on:
// - Bernstein (2006): Ed25519 — fast, constant-time, deterministic
// - PKI design: allowlist of known good identities
// - Amanah: no claim without identity-bound proof
// ============================================================

use std::collections::HashMap;

use ed25519_dalek::{SigningKey, VerifyingKey};

/// An agent's cryptographic identity.
#[derive(Debug, Clone)]
pub struct AgentIdentity {
    /// Agent ID (e.g., "P1-Navigator", "S3-Mediator")
    pub agent_id: String,
    /// Ed25519 public key for verifying this agent's signatures
    pub verifying_key: VerifyingKey,
    /// Whether this identity is active (can sign new receipts)
    pub active: bool,
    /// Block number where this identity was registered
    pub registered_at_block: u64,
}

/// The Identity Registry — maps agent IDs to verified public keys.
///
/// Constitutional rule: a receipt is only valid if its signer
/// is in the registry AND active. Unknown signers are rejected.
#[derive(Debug)]
pub struct IdentityRegistry {
    /// Agent ID → Identity mapping
    agents: HashMap<String, AgentIdentity>,
    /// Node's own signing key (for minting new identities)
    node_signing_key: Option<SigningKey>,
    /// Node's verifying key (published to federation)
    node_verifying_key: Option<VerifyingKey>,
}

impl IdentityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            node_signing_key: None,
            node_verifying_key: None,
        }
    }

    /// Initialize with a node keypair. Generates Ed25519 if none provided.
    pub fn with_node_key(signing_key: SigningKey) -> Self {
        let verifying_key = signing_key.verifying_key();
        Self {
            agents: HashMap::new(),
            node_signing_key: Some(signing_key),
            node_verifying_key: Some(verifying_key),
        }
    }

    /// Register an agent identity. Returns the verifying key.
    /// Called during genesis mint or agent creation.
    pub fn register_agent(&mut self, agent_id: &str, block_number: u64) -> VerifyingKey {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        let identity = AgentIdentity {
            agent_id: agent_id.to_string(),
            verifying_key,
            active: true,
            registered_at_block: block_number,
        };

        self.agents.insert(agent_id.to_string(), identity);
        verifying_key
    }

    /// Verify that a signer is registered and active.
    /// Amanah gate: unknown signers are rejected.
    pub fn verify_signer(&self, verifying_key: &VerifyingKey) -> Option<&AgentIdentity> {
        self.agents
            .values()
            .find(|id| id.active && id.verifying_key == *verifying_key)
    }

    /// Deactivate an agent identity (revocation).
    pub fn deactivate(&mut self, agent_id: &str) -> bool {
        if let Some(identity) = self.agents.get_mut(agent_id) {
            identity.active = false;
            true
        } else {
            false
        }
    }

    /// Get the node's verifying key (for federation announcement).
    pub fn node_verifying_key(&self) -> Option<&VerifyingKey> {
        self.node_verifying_key.as_ref()
    }

    /// Get the node's signing key (for receipt signing).
    pub fn node_signing_key(&self) -> Option<&SigningKey> {
        self.node_signing_key.as_ref()
    }

    /// Number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Number of active agents.
    pub fn active_count(&self) -> usize {
        self.agents.values().filter(|id| id.active).count()
    }

    /// Mint the genesis agent roster (Block 0 ceremony).
    /// Creates Ed25519 keypairs for all 12 founding agents.
    pub fn mint_genesis_agents(&mut self) -> Vec<(String, VerifyingKey)> {
        let pat_agents = [
            "P1-Navigator",
            "P2-Scholar",
            "P3-Artisan",
            "P4-Guardian",
            "P5-Mentor",
            "P6-Diplomat",
            "P7-Oracle",
        ];
        let sat_agents = [
            "S1-Validator",
            "S2-Oracle",
            "S3-Mediator",
            "S4-Archivist",
            "S5-Sentinel",
        ];

        let mut minted = Vec::new();
        for agent_id in pat_agents.iter().chain(sat_agents.iter()) {
            let vk = self.register_agent(agent_id, 0);
            minted.push((agent_id.to_string(), vk));
        }
        minted
    }
}

impl Default for IdentityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_mints_12_agents() {
        let mut registry = IdentityRegistry::new();
        let minted = registry.mint_genesis_agents();
        assert_eq!(minted.len(), 12);
        assert_eq!(registry.agent_count(), 12);
        assert_eq!(registry.active_count(), 12);
    }

    #[test]
    fn verify_registered_signer() {
        let mut registry = IdentityRegistry::new();
        let vk = registry.register_agent("P1-Navigator", 0);
        let found = registry.verify_signer(&vk);
        assert!(found.is_some());
        assert_eq!(found.unwrap().agent_id, "P1-Navigator");
    }

    #[test]
    fn reject_unknown_signer() {
        let registry = IdentityRegistry::new();
        let random_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let found = registry.verify_signer(&random_key.verifying_key());
        assert!(found.is_none());
    }

    #[test]
    fn deactivated_agent_rejected() {
        let mut registry = IdentityRegistry::new();
        let vk = registry.register_agent("P4-Guardian", 0);
        assert!(registry.verify_signer(&vk).is_some());

        registry.deactivate("P4-Guardian");
        assert!(registry.verify_signer(&vk).is_none());
        assert_eq!(registry.active_count(), 0);
    }

    #[test]
    fn node_keypair_initialization() {
        let key = SigningKey::generate(&mut rand::rngs::OsRng);
        let registry = IdentityRegistry::with_node_key(key);
        assert!(registry.node_verifying_key().is_some());
        assert!(registry.node_signing_key().is_some());
    }
}
