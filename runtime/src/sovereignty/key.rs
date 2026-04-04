// src/sovereignty/key.rs - Key Sovereignty (Pillar 1: Identity)
//
// Principle: All identities (node, agent, user) are rooted in keys you control.
// Signed actions, signed updates, signed artifacts.

use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey, PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;
use tracing::{debug, info, warn};

/// Identity types in the BIZRA system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IdentityType {
    /// Genesis Node (Node0)
    GenesisNode,
    /// Regular participant node
    Node,
    /// PAT agent identity
    PatAgent,
    /// SAT agent identity
    SatAgent,
    /// Human user identity
    User,
    /// Service identity (MCP, federation, etc.)
    Service,
}

/// A sovereign identity with its keypair
#[derive(Clone)]
pub struct SovereignIdentity {
    /// Identity type
    pub identity_type: IdentityType,
    /// Unique identifier
    pub id: String,
    /// Ed25519 signing key (private)
    signing_key: SigningKey,
    /// Ed25519 verifying key (public)
    pub verifying_key: VerifyingKey,
    /// Optional display name
    pub display_name: Option<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Parent identity ID (for derived identities)
    pub parent_id: Option<String>,
}

impl SovereignIdentity {
    /// Generate a new identity with fresh keypair
    pub fn generate(identity_type: IdentityType, id: String) -> Self {
        let mut secret = [0u8; SECRET_KEY_LENGTH];
        rand::thread_rng().fill_bytes(&mut secret);
        let signing_key = SigningKey::from_bytes(&secret);
        let verifying_key = signing_key.verifying_key();

        Self {
            identity_type,
            id,
            signing_key,
            verifying_key,
            display_name: None,
            created_at: chrono::Utc::now(),
            parent_id: None,
        }
    }

    /// Create identity from existing secret key bytes
    pub fn from_secret(
        identity_type: IdentityType,
        id: String,
        secret: &[u8; SECRET_KEY_LENGTH],
    ) -> Self {
        let signing_key = SigningKey::from_bytes(secret);
        let verifying_key = signing_key.verifying_key();

        Self {
            identity_type,
            id,
            signing_key,
            verifying_key,
            display_name: None,
            created_at: chrono::Utc::now(),
            parent_id: None,
        }
    }

    /// Derive a child identity (e.g., agent from node)
    pub fn derive_child(&self, child_type: IdentityType, child_id: String) -> Self {
        // Derive child key material from parent + child ID
        let mut hasher = Sha256::new();
        hasher.update(self.signing_key.to_bytes());
        hasher.update(child_id.as_bytes());
        let derived: [u8; 32] = hasher.finalize().into();

        let signing_key = SigningKey::from_bytes(&derived);
        let verifying_key = signing_key.verifying_key();

        Self {
            identity_type: child_type,
            id: child_id,
            signing_key,
            verifying_key,
            display_name: None,
            created_at: chrono::Utc::now(),
            parent_id: Some(self.id.clone()),
        }
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        self.verifying_key.verify(message, signature).is_ok()
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> [u8; PUBLIC_KEY_LENGTH] {
        self.verifying_key.to_bytes()
    }

    /// Get fingerprint (first 8 bytes of SHA256 of public key)
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.verifying_key.to_bytes());
        let hash = hasher.finalize();
        hex::encode(&hash[..8])
    }

    /// Export public identity info (safe to share)
    pub fn public_info(&self) -> PublicIdentity {
        PublicIdentity {
            identity_type: self.identity_type,
            id: self.id.clone(),
            public_key: hex::encode(self.verifying_key.to_bytes()),
            fingerprint: self.fingerprint(),
            display_name: self.display_name.clone(),
            created_at: self.created_at,
            parent_id: self.parent_id.clone(),
        }
    }
}

/// Public identity information (safe to share)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicIdentity {
    pub identity_type: IdentityType,
    pub id: String,
    pub public_key: String,
    pub fingerprint: String,
    pub display_name: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub parent_id: Option<String>,
}

impl PublicIdentity {
    /// Get verifying key from public identity
    pub fn verifying_key(&self) -> Option<VerifyingKey> {
        let bytes = hex::decode(&self.public_key).ok()?;
        if bytes.len() != PUBLIC_KEY_LENGTH {
            return None;
        }
        let mut arr = [0u8; PUBLIC_KEY_LENGTH];
        arr.copy_from_slice(&bytes);
        VerifyingKey::from_bytes(&arr).ok()
    }

    /// Verify a signature using this public identity
    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        self.verifying_key()
            .map(|vk| vk.verify(message, signature).is_ok())
            .unwrap_or(false)
    }
}

/// Signed artifact wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedArtifact<T: Serialize> {
    /// The payload
    pub payload: T,
    /// Signer's public identity
    pub signer: PublicIdentity,
    /// Ed25519 signature (hex encoded)
    pub signature: String,
    /// Timestamp of signing
    pub signed_at: chrono::DateTime<chrono::Utc>,
}

impl<T: Serialize + for<'de> Deserialize<'de>> SignedArtifact<T> {
    /// Create and sign an artifact
    pub fn sign(payload: T, identity: &SovereignIdentity) -> anyhow::Result<Self> {
        let payload_bytes = serde_json::to_vec(&payload)?;
        let signature = identity.sign(&payload_bytes);

        Ok(Self {
            payload,
            signer: identity.public_info(),
            signature: hex::encode(signature.to_bytes()),
            signed_at: chrono::Utc::now(),
        })
    }

    /// Verify the artifact signature
    pub fn verify(&self) -> bool {
        let Ok(payload_bytes) = serde_json::to_vec(&self.payload) else {
            return false;
        };

        let Ok(sig_bytes) = hex::decode(&self.signature) else {
            return false;
        };

        if sig_bytes.len() != 64 {
            return false;
        }

        let Ok(signature) = Signature::from_slice(&sig_bytes) else {
            return false;
        };

        self.signer.verify(&payload_bytes, &signature)
    }
}

/// Identity registry for managing multiple identities
pub struct IdentityRegistry {
    /// Node's primary identity
    node_identity: Option<SovereignIdentity>,
    /// Agent identities (derived from node)
    agent_identities: RwLock<HashMap<String, SovereignIdentity>>,
    /// Known external identities (public only)
    known_identities: RwLock<HashMap<String, PublicIdentity>>,
}

impl IdentityRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            node_identity: None,
            agent_identities: RwLock::new(HashMap::new()),
            known_identities: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize with node identity
    pub fn with_node_identity(identity: SovereignIdentity) -> Self {
        Self {
            node_identity: Some(identity),
            agent_identities: RwLock::new(HashMap::new()),
            known_identities: RwLock::new(HashMap::new()),
        }
    }

    /// Get node identity
    pub fn node_identity(&self) -> Option<&SovereignIdentity> {
        self.node_identity.as_ref()
    }

    /// Create agent identity (derived from node)
    pub fn create_agent_identity(
        &self,
        agent_type: IdentityType,
        agent_id: String,
    ) -> Option<PublicIdentity> {
        let node = self.node_identity.as_ref()?;
        let agent = node.derive_child(agent_type, agent_id.clone());
        let public = agent.public_info();

        if let Ok(mut agents) = self.agent_identities.write() {
            agents.insert(agent_id, agent);
        }

        Some(public)
    }

    /// Get agent identity
    pub fn get_agent(&self, agent_id: &str) -> Option<PublicIdentity> {
        self.agent_identities
            .read()
            .ok()?
            .get(agent_id)
            .map(|a| a.public_info())
    }

    /// Sign with agent identity
    pub fn sign_as_agent<T: Serialize + for<'de> Deserialize<'de>>(
        &self,
        agent_id: &str,
        payload: T,
    ) -> Option<SignedArtifact<T>> {
        let agents = self.agent_identities.read().ok()?;
        let agent = agents.get(agent_id)?;
        SignedArtifact::sign(payload, agent).ok()
    }

    /// Register external identity
    pub fn register_external(&self, identity: PublicIdentity) {
        if let Ok(mut known) = self.known_identities.write() {
            known.insert(identity.id.clone(), identity);
        }
    }

    /// Verify signature from known identity
    pub fn verify_from_known(
        &self,
        signer_id: &str,
        message: &[u8],
        signature: &Signature,
    ) -> bool {
        self.known_identities
            .read()
            .ok()
            .and_then(|k| k.get(signer_id).cloned())
            .map(|id| id.verify(message, signature))
            .unwrap_or(false)
    }
}

impl Default for IdentityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AGENT KEYPAIR REGISTRY (P1: Per-Agent Signing Keys)
// ============================================================================

/// Default PAT agent IDs (from AGENTS.md / MAG-7 squad)
pub const PAT_AGENTS: &[&str] = &[
    "PRIME",    // Strategist & Graph Orchestrator
    "GNOSTIC",  // Memory Custodian & Grounding
    "TEKNE",    // Implementation & Code Ops
    "AESTHETE", // UX & Design Polish
    "LOGOS",    // Critic, Safety & SNR Verification
    "AXON",     // Synthesis & Conflict Resolution
    "KAIROS",   // Executor & Delivery Control
];

/// Default SAT agent IDs (from architecture)
pub const SAT_AGENTS: &[&str] = &[
    "security_guardian",   // Security validation
    "ethics_validator",    // Ethics/Ihsān enforcement
    "performance_monitor", // Performance budget checks
    "consistency_checker", // Consistency validation
    "resource_optimizer",  // Resource constraint checks
];

/// Agent keypair manager for PAT/SAT agents
pub struct AgentKeypairRegistry {
    /// Parent node identity for key derivation
    node_identity: SovereignIdentity,
    /// PAT agent keypairs
    pat_agents: HashMap<String, SovereignIdentity>,
    /// SAT agent keypairs
    sat_agents: HashMap<String, SovereignIdentity>,
    /// Key rotation history (agent_id -> Vec of (timestamp, old_fingerprint))
    rotation_history: RwLock<HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, String)>>>,
}

impl AgentKeypairRegistry {
    /// Create with node identity, deriving all agent keypairs
    pub fn new(node_identity: SovereignIdentity) -> Self {
        let mut pat_agents = HashMap::new();
        let mut sat_agents = HashMap::new();

        // Derive PAT agent keys
        for agent_id in PAT_AGENTS {
            let agent = node_identity.derive_child(IdentityType::PatAgent, agent_id.to_string());
            info!(
                agent_id = %agent_id,
                fingerprint = %agent.fingerprint(),
                "🔑 PAT agent keypair derived"
            );
            pat_agents.insert(agent_id.to_string(), agent);
        }

        // Derive SAT agent keys
        for agent_id in SAT_AGENTS {
            let agent = node_identity.derive_child(IdentityType::SatAgent, agent_id.to_string());
            info!(
                agent_id = %agent_id,
                fingerprint = %agent.fingerprint(),
                "🔑 SAT agent keypair derived"
            );
            sat_agents.insert(agent_id.to_string(), agent);
        }

        Self {
            node_identity,
            pat_agents,
            sat_agents,
            rotation_history: RwLock::new(HashMap::new()),
        }
    }

    /// Generate a new node and all agent keys (fresh start)
    pub fn generate_fresh(node_id: String) -> Self {
        let node = SovereignIdentity::generate(IdentityType::GenesisNode, node_id);
        Self::new(node)
    }

    /// Get PAT agent identity
    pub fn get_pat_agent(&self, agent_id: &str) -> Option<&SovereignIdentity> {
        self.pat_agents.get(agent_id)
    }

    /// Get SAT agent identity
    pub fn get_sat_agent(&self, agent_id: &str) -> Option<&SovereignIdentity> {
        self.sat_agents.get(agent_id)
    }

    /// Get any agent by ID (PAT or SAT)
    pub fn get_agent(&self, agent_id: &str) -> Option<&SovereignIdentity> {
        self.pat_agents
            .get(agent_id)
            .or_else(|| self.sat_agents.get(agent_id))
    }

    /// Sign message as specific agent
    pub fn sign_as_agent(
        &self,
        agent_id: &str,
        message: &[u8],
    ) -> Option<(Signature, PublicIdentity)> {
        self.get_agent(agent_id)
            .map(|agent| (agent.sign(message), agent.public_info()))
    }

    /// Sign artifact as specific agent
    pub fn sign_artifact_as_agent<T: Serialize + for<'de> Deserialize<'de>>(
        &self,
        agent_id: &str,
        payload: T,
    ) -> Option<SignedArtifact<T>> {
        self.get_agent(agent_id)
            .and_then(|agent| SignedArtifact::sign(payload, agent).ok())
    }

    /// Verify signature from known agent
    pub fn verify_agent_signature(
        &self,
        agent_id: &str,
        message: &[u8],
        signature: &Signature,
    ) -> bool {
        self.get_agent(agent_id)
            .map(|agent| agent.verify(message, signature))
            .unwrap_or(false)
    }

    /// Rotate an agent's keypair (re-derive with new salt)
    pub fn rotate_agent_key(&mut self, agent_id: &str) -> Option<PublicIdentity> {
        // Determine if PAT or SAT
        let is_pat = self.pat_agents.contains_key(agent_id);
        let is_sat = self.sat_agents.contains_key(agent_id);

        if !is_pat && !is_sat {
            return None;
        }

        // Record old fingerprint in history
        if let Some(old_agent) = self.get_agent(agent_id) {
            let old_fp = old_agent.fingerprint();
            if let Ok(mut history) = self.rotation_history.write() {
                history
                    .entry(agent_id.to_string())
                    .or_default()
                    .push((chrono::Utc::now(), old_fp));
            }
        }

        // Derive new key with rotation counter
        let rotation_count = self
            .rotation_history
            .read()
            .ok()
            .and_then(|h| h.get(agent_id).map(|v| v.len()))
            .unwrap_or(0);

        // Create rotated ID for derivation
        let rotated_id = format!("{}:rotation:{}", agent_id, rotation_count);
        let identity_type = if is_pat {
            IdentityType::PatAgent
        } else {
            IdentityType::SatAgent
        };
        let new_agent = self.node_identity.derive_child(identity_type, rotated_id);

        // Re-insert with original ID but new key
        let mut final_agent = new_agent;
        final_agent.id = agent_id.to_string(); // Keep original ID

        let public = final_agent.public_info();

        if is_pat {
            self.pat_agents.insert(agent_id.to_string(), final_agent);
        } else {
            self.sat_agents.insert(agent_id.to_string(), final_agent);
        }

        info!(
            agent_id = %agent_id,
            new_fingerprint = %public.fingerprint,
            rotation_count = rotation_count + 1,
            "🔄 Agent keypair rotated"
        );

        Some(public)
    }

    /// Get all PAT agent public identities
    pub fn all_pat_public(&self) -> Vec<PublicIdentity> {
        self.pat_agents.values().map(|a| a.public_info()).collect()
    }

    /// Get all SAT agent public identities
    pub fn all_sat_public(&self) -> Vec<PublicIdentity> {
        self.sat_agents.values().map(|a| a.public_info()).collect()
    }

    /// Get node public identity
    pub fn node_public(&self) -> PublicIdentity {
        self.node_identity.public_info()
    }

    /// Get rotation history for an agent
    pub fn rotation_history(&self, agent_id: &str) -> Vec<(chrono::DateTime<chrono::Utc>, String)> {
        self.rotation_history
            .read()
            .ok()
            .and_then(|h| h.get(agent_id).cloned())
            .unwrap_or_default()
    }

    /// Total agent count
    pub fn agent_count(&self) -> usize {
        self.pat_agents.len() + self.sat_agents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_generation() {
        let identity = SovereignIdentity::generate(IdentityType::Node, "node-001".to_string());
        assert_eq!(identity.id, "node-001");
        assert_eq!(identity.identity_type, IdentityType::Node);
        assert!(!identity.fingerprint().is_empty());
    }

    #[test]
    fn test_sign_and_verify() {
        let identity = SovereignIdentity::generate(IdentityType::Node, "test".to_string());
        let message = b"Hello, sovereignty!";

        let signature = identity.sign(message);
        assert!(identity.verify(message, &signature));

        // Wrong message should fail
        assert!(!identity.verify(b"Wrong message", &signature));
    }

    #[test]
    fn test_derive_child() {
        let parent = SovereignIdentity::generate(IdentityType::Node, "node-001".to_string());
        let child = parent.derive_child(IdentityType::PatAgent, "agent-prime".to_string());

        assert_eq!(child.identity_type, IdentityType::PatAgent);
        assert_eq!(child.parent_id, Some("node-001".to_string()));

        // Child should have different key
        assert_ne!(parent.public_key_bytes(), child.public_key_bytes());
    }

    #[test]
    fn test_signed_artifact() {
        let identity = SovereignIdentity::generate(IdentityType::Node, "test".to_string());

        #[derive(Serialize, Deserialize)]
        struct TestPayload {
            message: String,
        }

        let payload = TestPayload {
            message: "Sovereignty matters".to_string(),
        };

        let signed = SignedArtifact::sign(payload, &identity).unwrap();
        assert!(signed.verify());
    }

    #[test]
    fn test_public_identity_verify() {
        let identity = SovereignIdentity::generate(IdentityType::Node, "test".to_string());
        let public = identity.public_info();

        let message = b"Test message";
        let signature = identity.sign(message);

        assert!(public.verify(message, &signature));
    }

    // =========================================================================
    // Agent Keypair Registry Tests (P1)
    // =========================================================================

    #[test]
    fn test_agent_keypair_registry_creation() {
        let registry = AgentKeypairRegistry::generate_fresh("node-genesis".to_string());

        // Should have all PAT agents
        assert_eq!(registry.pat_agents.len(), PAT_AGENTS.len());
        for agent_id in PAT_AGENTS {
            assert!(registry.get_pat_agent(agent_id).is_some());
        }

        // Should have all SAT agents
        assert_eq!(registry.sat_agents.len(), SAT_AGENTS.len());
        for agent_id in SAT_AGENTS {
            assert!(registry.get_sat_agent(agent_id).is_some());
        }

        // Total count
        assert_eq!(registry.agent_count(), PAT_AGENTS.len() + SAT_AGENTS.len());
    }

    #[test]
    fn test_agent_keypair_signing() {
        let registry = AgentKeypairRegistry::generate_fresh("node-genesis".to_string());
        let message = b"Test message for PRIME agent";

        // Sign as PRIME
        let (signature, public) = registry.sign_as_agent("PRIME", message).unwrap();
        assert_eq!(public.id, "PRIME");

        // Verify signature
        assert!(registry.verify_agent_signature("PRIME", message, &signature));

        // Wrong agent should fail verification
        assert!(!registry.verify_agent_signature("GNOSTIC", message, &signature));
    }

    #[test]
    fn test_agent_keypair_derivation_deterministic() {
        // Same node identity should derive same agent keys
        let secret: [u8; 32] = [0x42; 32];
        let node1 =
            SovereignIdentity::from_secret(IdentityType::GenesisNode, "node".into(), &secret);
        let node2 =
            SovereignIdentity::from_secret(IdentityType::GenesisNode, "node".into(), &secret);

        let reg1 = AgentKeypairRegistry::new(node1);
        let reg2 = AgentKeypairRegistry::new(node2);

        // PRIME should have same public key in both registries
        let prime1 = reg1.get_pat_agent("PRIME").unwrap();
        let prime2 = reg2.get_pat_agent("PRIME").unwrap();
        assert_eq!(prime1.public_key_bytes(), prime2.public_key_bytes());
    }

    #[test]
    fn test_agent_keypair_rotation() {
        let mut registry = AgentKeypairRegistry::generate_fresh("node-genesis".to_string());

        let original = registry.get_pat_agent("LOGOS").unwrap().public_info();
        let original_fp = original.fingerprint.clone();

        // Rotate LOGOS key
        let rotated = registry.rotate_agent_key("LOGOS").unwrap();

        // Fingerprint should be different
        assert_ne!(original_fp, rotated.fingerprint);

        // ID should remain the same
        assert_eq!(rotated.id, "LOGOS");

        // Rotation history should be recorded
        let history = registry.rotation_history("LOGOS");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].1, original_fp);
    }

    #[test]
    fn test_agent_artifact_signing() {
        let registry = AgentKeypairRegistry::generate_fresh("node-genesis".to_string());

        #[derive(Serialize, Deserialize)]
        struct TestPayload {
            action: String,
            score: f64,
        }

        let payload = TestPayload {
            action: "validate".to_string(),
            score: 0.97,
        };

        // Sign as security_guardian (SAT)
        let signed = registry
            .sign_artifact_as_agent("security_guardian", payload)
            .unwrap();

        assert!(signed.verify());
        assert_eq!(signed.signer.id, "security_guardian");
        assert_eq!(signed.signer.identity_type, IdentityType::SatAgent);
    }

    #[test]
    fn test_all_agents_have_unique_keys() {
        let registry = AgentKeypairRegistry::generate_fresh("node-genesis".to_string());

        let mut all_fingerprints: Vec<String> = Vec::new();

        for agent_id in PAT_AGENTS {
            all_fingerprints.push(registry.get_pat_agent(agent_id).unwrap().fingerprint());
        }
        for agent_id in SAT_AGENTS {
            all_fingerprints.push(registry.get_sat_agent(agent_id).unwrap().fingerprint());
        }

        // Check all fingerprints are unique
        let unique_count = all_fingerprints
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(unique_count, all_fingerprints.len());
    }
}
