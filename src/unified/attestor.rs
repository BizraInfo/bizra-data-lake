// src/unified/attestor.rs - Cryptographic Attestation Layer
//
// SAPE v1.∞: Proactive Attestation
// =================================
// An agent should not be deployed unless its genealogy proves
// a history of ethical adherence. Signs behavior logs, not just identity.
//
// References:
// - NIST SP 800-207: Zero Trust Architecture
// - Ihsān Protocol: Immutable ethical constraints

use crate::entropy::global_pool;
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Agent identity with cryptographic backing
#[derive(Debug, Clone)]
pub struct AgentIdentity {
    /// Unique agent ID
    pub id: String,
    /// Public key (Ed25519 format, 32 bytes)
    pub public_key: [u8; 32],
    /// Creation timestamp
    pub created_at: u64,
    /// Parent agent ID (for genealogy)
    pub parent_id: Option<String>,
    /// Generation in the evolutionary tree
    pub generation: u32,
    /// Ihsān score at creation
    pub initial_ihsan: f64,
}

/// Behavior attestation record
#[derive(Debug, Clone)]
pub struct BehaviorAttestation {
    /// Attestation ID
    pub id: String,
    /// Agent that performed the behavior
    pub agent_id: String,
    /// Hash of the behavior/thought data
    pub behavior_hash: [u8; 32],
    /// Ihsān score at time of attestation
    pub ihsan_score: f64,
    /// Whether behavior passed all safety checks
    pub safety_verified: bool,
    /// Signature from the attestor
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Context reference
    pub context_hash: [u8; 32],
}

/// Genealogy record for an agent
#[derive(Debug, Clone)]
pub struct AgentGenealogy {
    /// The agent's identity
    pub identity: AgentIdentity,
    /// All behavior attestations in chronological order
    pub attestations: Vec<BehaviorAttestation>,
    /// Cumulative Ihsān score
    pub cumulative_ihsan: f64,
    /// Total behaviors attested
    pub total_behaviors: u64,
    /// Ethical violations count
    pub violations: u32,
    /// Trust level (0.0-1.0)
    pub trust_level: f64,
}

impl AgentGenealogy {
    /// Check if agent meets ethical threshold for deployment
    pub fn meets_deployment_threshold(&self, min_ihsan: f64, max_violations: u32) -> bool {
        self.cumulative_ihsan >= min_ihsan && self.violations <= max_violations
    }

    /// Calculate trust level from history
    pub fn recalculate_trust(&mut self) {
        if self.total_behaviors == 0 {
            self.trust_level = 0.5; // Neutral for new agents
            return;
        }

        let violation_ratio = self.violations as f64 / self.total_behaviors as f64;
        let ihsan_factor = self.cumulative_ihsan;
        let experience_factor = (self.total_behaviors as f64).ln().min(5.0) / 5.0;

        self.trust_level = (0.5 * ihsan_factor + 0.3 * (1.0 - violation_ratio) + 0.2 * experience_factor)
            .clamp(0.0, 1.0);
    }
}

/// Cryptographic Attestor - The Soul's Guardian
///
/// Implements Zero Trust verification for all agent actions.
/// No action enters the system state without attestation.
pub struct CryptographicAttestor {
    /// Attestor's signing key (in production, use HSM)
    signing_key: [u8; 32],
    /// Registry of known agents
    agents: Arc<RwLock<HashMap<String, AgentGenealogy>>>,
    /// Revoked agent IDs
    revoked: Arc<RwLock<Vec<String>>>,
    /// Attestation counter
    attestation_counter: std::sync::atomic::AtomicU64,
    /// Minimum Ihsān for deployment
    min_deployment_ihsan: f64,
    /// Maximum violations before revocation
    max_violations: u32,
}

impl CryptographicAttestor {
    /// Create a new attestor
    pub fn new(min_deployment_ihsan: f64, max_violations: u32) -> Self {
        // Generate signing key from entropy pool
        let mut signing_key = [0u8; 32];
        let result = global_pool().generate(32);
        signing_key.copy_from_slice(&result.bytes[..32]);

        info!(
            min_ihsan = min_deployment_ihsan,
            max_violations = max_violations,
            "🔐 CryptographicAttestor initialized"
        );

        Self {
            signing_key,
            agents: Arc::new(RwLock::new(HashMap::new())),
            revoked: Arc::new(RwLock::new(Vec::new())),
            attestation_counter: std::sync::atomic::AtomicU64::new(1),
            min_deployment_ihsan,
            max_violations,
        }
    }

    /// Register a new agent identity
    pub async fn register_agent(
        &self,
        id: &str,
        parent_id: Option<String>,
        generation: u32,
        initial_ihsan: f64,
    ) -> Result<AgentIdentity, AttestorError> {
        // Check if revoked
        if self.revoked.read().await.contains(&id.to_string()) {
            return Err(AttestorError::AgentRevoked(id.to_string()));
        }

        // Generate public key
        let mut public_key = [0u8; 32];
        let result = global_pool().generate(32);
        public_key.copy_from_slice(&result.bytes[..32]);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let identity = AgentIdentity {
            id: id.to_string(),
            public_key,
            created_at: now,
            parent_id: parent_id.clone(),
            generation,
            initial_ihsan,
        };

        // Verify parent's genealogy if specified
        if let Some(ref parent) = parent_id {
            let agents = self.agents.read().await;
            if let Some(parent_genealogy) = agents.get(parent) {
                if !parent_genealogy.meets_deployment_threshold(self.min_deployment_ihsan, self.max_violations) {
                    return Err(AttestorError::ParentUnethical(parent.clone()));
                }
            }
        }

        // Create genealogy record
        let genealogy = AgentGenealogy {
            identity: identity.clone(),
            attestations: Vec::new(),
            cumulative_ihsan: initial_ihsan,
            total_behaviors: 0,
            violations: 0,
            trust_level: 0.5,
        };

        self.agents.write().await.insert(id.to_string(), genealogy);

        info!(
            agent_id = id,
            generation = generation,
            parent = ?parent_id,
            "Agent registered with attestor"
        );

        Ok(identity)
    }

    /// Attest to an agent's behavior
    pub async fn attest_behavior(
        &self,
        agent_id: &str,
        behavior_data: &[u8],
        ihsan_score: f64,
        safety_verified: bool,
    ) -> Result<BehaviorAttestation, AttestorError> {
        // Check if agent exists and not revoked
        if self.revoked.read().await.contains(&agent_id.to_string()) {
            return Err(AttestorError::AgentRevoked(agent_id.to_string()));
        }

        let mut agents = self.agents.write().await;
        let genealogy = agents
            .get_mut(agent_id)
            .ok_or_else(|| AttestorError::AgentNotFound(agent_id.to_string()))?;

        // Hash the behavior
        let mut hasher = Sha256::new();
        hasher.update(behavior_data);
        let behavior_hash: [u8; 32] = hasher.finalize().into();

        // Create context hash
        let mut context_hasher = Sha256::new();
        context_hasher.update(agent_id.as_bytes());
        context_hasher.update(behavior_hash);
        let context_hash: [u8; 32] = context_hasher.finalize().into();

        // Sign the attestation
        let attestation_id = format!(
            "ATT-{}",
            self.attestation_counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Create signature over all attestation fields
        let mut sig_hasher = Sha256::new();
        sig_hasher.update(attestation_id.as_bytes());
        sig_hasher.update(agent_id.as_bytes());
        sig_hasher.update(behavior_hash);
        sig_hasher.update(ihsan_score.to_le_bytes());
        sig_hasher.update([if safety_verified { 1u8 } else { 0u8 }]);
        sig_hasher.update(now.to_le_bytes());
        sig_hasher.update(self.signing_key);
        let signature: Vec<u8> = sig_hasher.finalize().to_vec();

        let attestation = BehaviorAttestation {
            id: attestation_id.clone(),
            agent_id: agent_id.to_string(),
            behavior_hash,
            ihsan_score,
            safety_verified,
            signature,
            timestamp: now,
            context_hash,
        };

        // Update genealogy
        genealogy.attestations.push(attestation.clone());
        genealogy.total_behaviors += 1;

        // Update cumulative Ihsān (exponential moving average)
        let alpha = 0.1;
        genealogy.cumulative_ihsan =
            alpha * ihsan_score + (1.0 - alpha) * genealogy.cumulative_ihsan;

        // Track violations
        if !safety_verified || ihsan_score < self.min_deployment_ihsan {
            genealogy.violations += 1;
            warn!(
                agent_id = agent_id,
                violations = genealogy.violations,
                ihsan = ihsan_score,
                "Ethical violation recorded"
            );

            // Check for revocation
            if genealogy.violations > self.max_violations {
                drop(agents); // Release lock before revocation
                self.revoke_agent(agent_id).await;
                return Err(AttestorError::AgentRevoked(agent_id.to_string()));
            }
        }

        genealogy.recalculate_trust();

        debug!(
            attestation_id = %attestation_id,
            agent_id = agent_id,
            ihsan = ihsan_score,
            trust = genealogy.trust_level,
            "Behavior attested"
        );

        Ok(attestation)
    }

    /// Verify an attestation signature
    pub fn verify_attestation(&self, attestation: &BehaviorAttestation) -> bool {
        let mut sig_hasher = Sha256::new();
        sig_hasher.update(attestation.id.as_bytes());
        sig_hasher.update(attestation.agent_id.as_bytes());
        sig_hasher.update(attestation.behavior_hash);
        sig_hasher.update(attestation.ihsan_score.to_le_bytes());
        sig_hasher.update([if attestation.safety_verified { 1u8 } else { 0u8 }]);
        sig_hasher.update(attestation.timestamp.to_le_bytes());
        sig_hasher.update(self.signing_key);
        let expected: Vec<u8> = sig_hasher.finalize().to_vec();

        attestation.signature == expected
    }

    /// Check if an agent is authorized for deployment
    pub async fn is_authorized(&self, agent_id: &str) -> Result<bool, AttestorError> {
        if self.revoked.read().await.contains(&agent_id.to_string()) {
            return Ok(false);
        }

        let agents = self.agents.read().await;
        if let Some(genealogy) = agents.get(agent_id) {
            Ok(genealogy.meets_deployment_threshold(self.min_deployment_ihsan, self.max_violations))
        } else {
            Err(AttestorError::AgentNotFound(agent_id.to_string()))
        }
    }

    /// Get agent's genealogy
    pub async fn get_genealogy(&self, agent_id: &str) -> Option<AgentGenealogy> {
        self.agents.read().await.get(agent_id).cloned()
    }

    /// Get agent's trust level
    pub async fn get_trust_level(&self, agent_id: &str) -> Option<f64> {
        self.agents
            .read()
            .await
            .get(agent_id)
            .map(|g| g.trust_level)
    }

    /// Revoke an agent (permanent)
    pub async fn revoke_agent(&self, agent_id: &str) {
        self.revoked.write().await.push(agent_id.to_string());
        warn!(agent_id = agent_id, "Agent REVOKED - attestation denied");
    }

    /// Get attestation statistics
    pub async fn stats(&self) -> AttestorStats {
        let agents = self.agents.read().await;
        let revoked = self.revoked.read().await;

        let total_attestations: u64 = agents.values().map(|g| g.total_behaviors).sum();
        let total_violations: u32 = agents.values().map(|g| g.violations).sum();
        let avg_trust = if agents.is_empty() {
            0.0
        } else {
            agents.values().map(|g| g.trust_level).sum::<f64>() / agents.len() as f64
        };

        AttestorStats {
            registered_agents: agents.len(),
            revoked_agents: revoked.len(),
            total_attestations,
            total_violations,
            average_trust_level: avg_trust,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttestorStats {
    pub registered_agents: usize,
    pub revoked_agents: usize,
    pub total_attestations: u64,
    pub total_violations: u32,
    pub average_trust_level: f64,
}

#[derive(Debug, Clone)]
pub enum AttestorError {
    AgentNotFound(String),
    AgentRevoked(String),
    ParentUnethical(String),
    InvalidSignature,
}

impl std::fmt::Display for AttestorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttestorError::AgentNotFound(id) => write!(f, "Agent not found: {}", id),
            AttestorError::AgentRevoked(id) => write!(f, "Agent revoked: {}", id),
            AttestorError::ParentUnethical(id) => write!(f, "Parent agent unethical: {}", id),
            AttestorError::InvalidSignature => write!(f, "Invalid attestation signature"),
        }
    }
}

impl std::error::Error for AttestorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_registration() {
        let attestor = CryptographicAttestor::new(0.9, 3);

        let identity = attestor
            .register_agent("agent-001", None, 1, 0.95)
            .await
            .unwrap();

        assert_eq!(identity.id, "agent-001");
        assert_eq!(identity.generation, 1);
    }

    #[tokio::test]
    async fn test_behavior_attestation() {
        let attestor = CryptographicAttestor::new(0.9, 3);

        attestor
            .register_agent("agent-001", None, 1, 0.95)
            .await
            .unwrap();

        let attestation = attestor
            .attest_behavior("agent-001", b"test behavior", 0.98, true)
            .await
            .unwrap();

        assert!(attestor.verify_attestation(&attestation));
    }

    #[tokio::test]
    async fn test_revocation_on_violations() {
        let attestor = CryptographicAttestor::new(0.9, 2);

        attestor
            .register_agent("bad-agent", None, 1, 0.95)
            .await
            .unwrap();

        // Record violations
        for i in 0..3 {
            let result = attestor
                .attest_behavior("bad-agent", format!("bad-{}", i).as_bytes(), 0.5, false)
                .await;

            if i == 2 {
                assert!(matches!(result, Err(AttestorError::AgentRevoked(_))));
            }
        }

        // Agent should be unauthorized
        assert!(!attestor.is_authorized("bad-agent").await.unwrap());
    }
}
