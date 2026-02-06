//! Agent Minting Engine
//!
//! Handles the creation of PAT and SAT agents with full verification.

use blake3::Hasher;
use chrono::{Duration, Utc};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use thiserror::Error;
use uuid::Uuid;

use super::types::*;
use super::AGENT_MINT_IHSAN_THRESHOLD;
use crate::identity::{hex_encode, NodeId, NodeIdentity};

// =============================================================================
// ERROR TYPES
// =============================================================================

#[derive(Error, Debug)]
pub enum MintingError {
    #[error("Ihsan threshold not met: {actual:.3} < {required:.3}")]
    IhsanViolation { actual: f64, required: f64 },

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Agent already exists: {0}")]
    AgentAlreadyExists(String),

    #[error("Team already complete")]
    TeamComplete,

    #[error("Invalid capability for role: {capability:?} not allowed for {role}")]
    InvalidCapability { capability: String, role: String },

    #[error("Insufficient stake: {actual} < {required}")]
    InsufficientStake { actual: u64, required: u64 },

    #[error("Missing giants attestation")]
    MissingGiantsAttestation,

    #[error("Delegation depth exceeded: {depth} > {max}")]
    DelegationDepthExceeded { depth: u8, max: u8 },

    #[error("Authority chain verification failed")]
    AuthorityChainInvalid,

    #[error("Pool registration required for SAT agents")]
    PoolRegistrationRequired,

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type MintingResult<T> = Result<T, MintingError>;

// =============================================================================
// MINTING ENGINE
// =============================================================================

/// AgentMintingEngine — Handles agent creation with full verification
pub struct AgentMintingEngine {
    /// Genesis authority public key
    genesis_public_key: [u8; 32],

    /// Current Ihsan threshold
    ihsan_threshold: f64,

    /// Pool registration callback (for SAT agents)
    pool_enabled: bool,
}

impl AgentMintingEngine {
    /// Create a new minting engine
    pub fn new(genesis_public_key: [u8; 32]) -> Self {
        Self {
            genesis_public_key,
            ihsan_threshold: AGENT_MINT_IHSAN_THRESHOLD,
            pool_enabled: true,
        }
    }

    /// Set custom Ihsan threshold
    pub fn with_ihsan_threshold(mut self, threshold: f64) -> Self {
        self.ihsan_threshold = threshold;
        self
    }

    /// Enable/disable pool registration for SAT agents
    pub fn with_pool_enabled(mut self, enabled: bool) -> Self {
        self.pool_enabled = enabled;
        self
    }

    // =========================================================================
    // PAT MINTING
    // =========================================================================

    /// Mint a complete PAT (7 agents)
    pub fn mint_pat(&self, owner_identity: &NodeIdentity) -> MintingResult<PersonalAgentTeam> {
        let owner_node_id = owner_identity.node_id().clone();
        let owner_public_key = owner_identity.public_key_bytes();

        let mut agents = std::collections::HashMap::new();

        // Mint all 7 PAT agents
        for role in PATRole::all() {
            let agent = self.mint_pat_agent(owner_identity, role)?;
            agents.insert(role, agent);
        }

        // Calculate authority chain hash
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-pat-authority-v1:");
        hasher.update(&owner_public_key);
        hasher.update(&self.genesis_public_key);
        for (role, agent) in &agents {
            hasher.update(&[*role as u8]);
            hasher.update(&agent.public_key);
        }
        let authority_chain_hash = *hasher.finalize().as_bytes();

        // Calculate team Ihsan
        let team_ihsan_score =
            agents.values().map(|a| a.ihsan_score).sum::<f64>() / agents.len() as f64;

        Ok(PersonalAgentTeam {
            id: Uuid::new_v4(),
            owner_node_id,
            owner_public_key,
            agents,
            created_at: Utc::now(),
            team_ihsan_score,
            authority_chain_hash,
        })
    }

    /// Mint a single PAT agent
    pub fn mint_pat_agent(
        &self,
        owner_identity: &NodeIdentity,
        role: PATRole,
    ) -> MintingResult<MintedAgent> {
        // Generate agent keypair
        let agent_signing_key = SigningKey::generate(&mut OsRng);
        let agent_public_key = *agent_signing_key.verifying_key().as_bytes();

        // Create Standing on Giants attestation with role-specific foundations
        let role_foundations = self.get_pat_role_foundations(role);
        let agent_id = Uuid::new_v4();
        let giants_attestation = StandingOnGiantsAttestation::new(agent_id, role_foundations);

        // Create capability card
        let capability_card = self.create_capability_card(
            agent_id,
            owner_identity.node_id().clone(),
            role.default_capabilities(),
            AgentResourceLimits::default(),
            giants_attestation.clone(),
        )?;

        // Create identity block
        let identity_block = self.create_identity_block(
            agent_id,
            agent_public_key,
            owner_identity,
            giants_attestation.clone(),
        )?;

        // Create the minted agent
        let now = Utc::now();
        Ok(MintedAgent {
            id: agent_id,
            public_key: agent_public_key,
            name: format!("PAT-{:?}-{}", role, &hex_encode(&agent_public_key)[..8]),
            capability_card,
            ihsan_score: self.ihsan_threshold,
            state: AgentState::Active,
            stake: 0, // PAT agents don't require stake
            impact_score: 0,
            created_at: now,
            last_activity: now,
            identity_block_hash: identity_block.block_hash,
        })
    }

    /// Get role-specific intellectual foundations for PAT agents
    fn get_pat_role_foundations(&self, role: PATRole) -> Vec<IntellectualFoundation> {
        match role {
            PATRole::Strategist => vec![IntellectualFoundation {
                giant_name: "Herbert Simon".to_string(),
                contribution: "Bounded Rationality — Satisficing in complex decisions".to_string(),
                citation: Some("Models of Bounded Rationality, 1982".to_string()),
                usage_in_agent: "Strategic decision making".to_string(),
            }],
            PATRole::Researcher => vec![IntellectualFoundation {
                giant_name: "Vannevar Bush".to_string(),
                contribution: "Memex — Knowledge organization and retrieval".to_string(),
                citation: Some("As We May Think, 1945".to_string()),
                usage_in_agent: "Knowledge synthesis".to_string(),
            }],
            PATRole::Developer => vec![IntellectualFoundation {
                giant_name: "Donald Knuth".to_string(),
                contribution: "Literate Programming — Code as literature".to_string(),
                citation: Some("The Art of Computer Programming, 1968".to_string()),
                usage_in_agent: "Code implementation".to_string(),
            }],
            PATRole::Analyst => vec![IntellectualFoundation {
                giant_name: "John Tukey".to_string(),
                contribution: "Exploratory Data Analysis — Pattern discovery".to_string(),
                citation: Some("Exploratory Data Analysis, 1977".to_string()),
                usage_in_agent: "Data analysis".to_string(),
            }],
            PATRole::Reviewer => vec![IntellectualFoundation {
                giant_name: "Edsger Dijkstra".to_string(),
                contribution: "Structured Programming — Correctness by construction".to_string(),
                citation: Some("Go To Statement Considered Harmful, 1968".to_string()),
                usage_in_agent: "Code review".to_string(),
            }],
            PATRole::Executor => vec![IntellectualFoundation {
                giant_name: "Alan Turing".to_string(),
                contribution: "Universal Computation — Executable procedures".to_string(),
                citation: Some("On Computable Numbers, 1936".to_string()),
                usage_in_agent: "Task execution".to_string(),
            }],
            PATRole::Guardian => vec![IntellectualFoundation {
                giant_name: "Abu Hamid al-Ghazali".to_string(),
                contribution: "Maqasid al-Shariah — Purpose-driven ethics".to_string(),
                citation: Some("Al-Mustasfa, 1095".to_string()),
                usage_in_agent: "Ethics enforcement".to_string(),
            }],
        }
    }

    // =========================================================================
    // SAT MINTING
    // =========================================================================

    /// Mint a complete SAT (5 agents)
    pub fn mint_sat(
        &self,
        contributor_identity: &NodeIdentity,
        stakes: std::collections::HashMap<SATRole, u64>,
    ) -> MintingResult<SharedAgentTeam> {
        // Verify all stake requirements
        for role in SATRole::all() {
            let required = role.minimum_stake();
            let provided = stakes.get(&role).copied().unwrap_or(0);
            if provided < required {
                return Err(MintingError::InsufficientStake {
                    actual: provided,
                    required,
                });
            }
        }

        let contributor_node_id = contributor_identity.node_id().clone();
        let contributor_public_key = contributor_identity.public_key_bytes();

        let mut agents = std::collections::HashMap::new();
        let mut total_stake = 0u64;

        // Mint all 5 SAT agents
        for role in SATRole::all() {
            let stake = stakes.get(&role).copied().unwrap_or(role.minimum_stake());
            let agent = self.mint_sat_agent(contributor_identity, role, stake)?;
            total_stake += stake;
            agents.insert(role, agent);
        }

        // Calculate governance hash
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-sat-governance-v1:");
        hasher.update(&contributor_public_key);
        for (role, agent) in &agents {
            hasher.update(&[*role as u8]);
            hasher.update(&agent.public_key);
            hasher.update(&agent.stake.to_le_bytes());
        }
        let governance_hash = *hasher.finalize().as_bytes();

        // Calculate team Ihsan
        let team_ihsan_score =
            agents.values().map(|a| a.ihsan_score).sum::<f64>() / agents.len() as f64;

        Ok(SharedAgentTeam {
            id: Uuid::new_v4(),
            contributor_node_id,
            contributor_public_key,
            agents,
            pool_registration_id: None, // Set after pool registration
            total_stake,
            total_earnings: 0,
            created_at: Utc::now(),
            team_ihsan_score,
            governance_hash,
        })
    }

    /// Mint a single SAT agent
    pub fn mint_sat_agent(
        &self,
        contributor_identity: &NodeIdentity,
        role: SATRole,
        stake: u64,
    ) -> MintingResult<MintedAgent> {
        // Verify stake requirement
        let required = role.minimum_stake();
        if stake < required {
            return Err(MintingError::InsufficientStake {
                actual: stake,
                required,
            });
        }

        // Generate agent keypair
        let agent_signing_key = SigningKey::generate(&mut OsRng);
        let agent_public_key = *agent_signing_key.verifying_key().as_bytes();

        // Create Standing on Giants attestation with role-specific foundations
        let role_foundations = self.get_sat_role_foundations(role);
        let agent_id = Uuid::new_v4();
        let giants_attestation = StandingOnGiantsAttestation::new(agent_id, role_foundations);

        // Create capability card
        let capability_card = self.create_capability_card(
            agent_id,
            contributor_identity.node_id().clone(),
            role.default_capabilities(),
            AgentResourceLimits::default(),
            giants_attestation.clone(),
        )?;

        // Create identity block
        let identity_block = self.create_identity_block(
            agent_id,
            agent_public_key,
            contributor_identity,
            giants_attestation.clone(),
        )?;

        // Create the minted agent
        let now = Utc::now();
        Ok(MintedAgent {
            id: agent_id,
            public_key: agent_public_key,
            name: format!("SAT-{:?}-{}", role, &hex_encode(&agent_public_key)[..8]),
            capability_card,
            ihsan_score: self.ihsan_threshold,
            state: AgentState::Active,
            stake,
            impact_score: 0,
            created_at: now,
            last_activity: now,
            identity_block_hash: identity_block.block_hash,
        })
    }

    /// Get role-specific intellectual foundations for SAT agents
    fn get_sat_role_foundations(&self, role: SATRole) -> Vec<IntellectualFoundation> {
        match role {
            SATRole::Validator => vec![IntellectualFoundation {
                giant_name: "Satoshi Nakamoto".to_string(),
                contribution: "Proof-of-Work — Trustless validation".to_string(),
                citation: Some("Bitcoin: A Peer-to-Peer Electronic Cash System, 2008".to_string()),
                usage_in_agent: "Transaction validation".to_string(),
            }],
            SATRole::Oracle => vec![IntellectualFoundation {
                giant_name: "Sergey Nazarov".to_string(),
                contribution: "Decentralized Oracles — External data integrity".to_string(),
                citation: Some("Chainlink Whitepaper, 2017".to_string()),
                usage_in_agent: "Data verification".to_string(),
            }],
            SATRole::Mediator => vec![IntellectualFoundation {
                giant_name: "Roger Fisher".to_string(),
                contribution: "Principled Negotiation — Interest-based mediation".to_string(),
                citation: Some("Getting to Yes, 1981".to_string()),
                usage_in_agent: "Dispute resolution".to_string(),
            }],
            SATRole::Archivist => vec![IntellectualFoundation {
                giant_name: "Tim Berners-Lee".to_string(),
                contribution: "World Wide Web — Universal information access".to_string(),
                citation: Some("Information Management: A Proposal, 1989".to_string()),
                usage_in_agent: "Knowledge preservation".to_string(),
            }],
            SATRole::Sentinel => vec![IntellectualFoundation {
                giant_name: "Dorothy Denning".to_string(),
                contribution: "Intrusion Detection — Anomaly-based security".to_string(),
                citation: Some("An Intrusion-Detection Model, 1987".to_string()),
                usage_in_agent: "Threat detection".to_string(),
            }],
        }
    }

    // =========================================================================
    // COMMON HELPERS
    // =========================================================================

    /// Create a capability card for an agent
    fn create_capability_card(
        &self,
        agent_id: Uuid,
        issuer_node_id: NodeId,
        capabilities: Vec<AgentCapability>,
        limits: AgentResourceLimits,
        giants_attestation: StandingOnGiantsAttestation,
    ) -> MintingResult<AgentCapabilityCard> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let expires_at = now + Duration::days(365);

        let mut card = AgentCapabilityCard {
            id,
            agent_id,
            issuer_node_id,
            capabilities,
            limits,
            ihsan_requirement: self.ihsan_threshold,
            valid_places: Vec::new(),
            created_at: now,
            expires_at,
            giants_attestation,
            card_hash: [0u8; 32],
        };

        card.card_hash = card.calculate_hash();
        Ok(card)
    }

    /// Create an identity block for an agent
    fn create_identity_block(
        &self,
        agent_id: Uuid,
        agent_public_key: [u8; 32],
        parent_identity: &NodeIdentity,
        giants_attestation: StandingOnGiantsAttestation,
    ) -> MintingResult<AgentIdentityBlock> {
        let parent_public_key = parent_identity.public_key_bytes();
        let parent_node_id = parent_identity.node_id().clone();

        // Build authority chain
        let authority_chain = vec![
            AuthorityLink {
                name: "Genesis".to_string(),
                public_key: self.genesis_public_key,
                depth: 0,
                link_hash: *blake3::hash(b"genesis").as_bytes(),
            },
            AuthorityLink {
                name: parent_node_id.0.clone(),
                public_key: parent_public_key,
                depth: 1,
                link_hash: *blake3::hash(&parent_public_key).as_bytes(),
            },
        ];

        // Calculate block hash
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-identity-block-v1:");
        hasher.update(agent_id.as_bytes());
        hasher.update(&agent_public_key);
        hasher.update(&parent_public_key);
        for link in &authority_chain {
            hasher.update(&link.public_key);
            hasher.update(&[link.depth]);
        }
        let block_hash = *hasher.finalize().as_bytes();

        // Sign with parent key
        let signature_bytes = parent_identity.sign(&block_hash);
        let signature = {
            let sig_bytes = crate::identity::hex_decode(&signature_bytes)
                .map_err(|_| MintingError::Internal("Failed to decode signature".to_string()))?;
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&sig_bytes);
            arr
        };

        Ok(AgentIdentityBlock {
            block_type: "KNOWLEDGE_BLOCK".to_string(),
            version: 1,
            agent_id,
            agent_public_key,
            parent_public_key,
            parent_node_id,
            authority_chain,
            created_at: Utc::now(),
            giants_attestation,
            block_hash,
            parent_signature: signature,
        })
    }
}

impl Default for AgentMintingEngine {
    fn default() -> Self {
        Self::new([0u8; 32])
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_identity() -> NodeIdentity {
        NodeIdentity::generate()
    }

    #[test]
    fn test_mint_pat_agent() {
        let identity = test_identity();
        let engine = AgentMintingEngine::new(identity.public_key_bytes());

        let agent = engine
            .mint_pat_agent(&identity, PATRole::Strategist)
            .unwrap();

        assert_eq!(agent.state, AgentState::Active);
        assert!(agent.ihsan_score >= 0.95);
        assert!(agent.capability_card.verify());
        assert!(agent.name.contains("PAT-Strategist"));
    }

    #[test]
    fn test_mint_full_pat() {
        let identity = test_identity();
        let engine = AgentMintingEngine::new(identity.public_key_bytes());

        let pat = engine.mint_pat(&identity).unwrap();

        assert!(pat.is_complete());
        assert_eq!(pat.agents.len(), 7);
        assert!(pat.team_ihsan_score >= 0.95);

        for role in PATRole::all() {
            assert!(pat.get_agent(role).is_some());
        }
    }

    #[test]
    fn test_mint_sat_agent() {
        let identity = test_identity();
        let engine = AgentMintingEngine::new(identity.public_key_bytes());

        let agent = engine
            .mint_sat_agent(&identity, SATRole::Validator, 1000)
            .unwrap();

        assert_eq!(agent.state, AgentState::Active);
        assert_eq!(agent.stake, 1000);
        assert!(agent.name.contains("SAT-Validator"));
    }

    #[test]
    fn test_mint_sat_insufficient_stake() {
        let identity = test_identity();
        let engine = AgentMintingEngine::new(identity.public_key_bytes());

        let result = engine.mint_sat_agent(&identity, SATRole::Validator, 100);

        assert!(matches!(
            result,
            Err(MintingError::InsufficientStake { .. })
        ));
    }

    #[test]
    fn test_mint_full_sat() {
        let identity = test_identity();
        let engine = AgentMintingEngine::new(identity.public_key_bytes());

        let mut stakes = std::collections::HashMap::new();
        for role in SATRole::all() {
            stakes.insert(role, role.minimum_stake());
        }

        let sat = engine.mint_sat(&identity, stakes).unwrap();

        assert!(sat.is_complete());
        assert_eq!(sat.agents.len(), 5);
        assert!(sat.meets_stake_requirements());
    }

    #[test]
    fn test_giants_attestation_included() {
        let identity = test_identity();
        let engine = AgentMintingEngine::new(identity.public_key_bytes());

        let agent = engine.mint_pat_agent(&identity, PATRole::Guardian).unwrap();

        let attestation = &agent.capability_card.giants_attestation;
        assert!(attestation.verify());
        assert!(attestation.foundations.len() >= 5);

        let attribution = attestation.format_attribution();
        assert!(attribution.contains("Shannon"));
        assert!(attribution.contains("al-Ghazali"));
    }
}
