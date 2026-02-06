//! PAT/SAT Type Definitions
//!
//! Core types for agent minting, capability cards, and attestations.

use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::identity::NodeId;
use crate::IHSAN_THRESHOLD;

// =============================================================================
// AGENT TYPES — The 7 PAT + 5 SAT Archetypes
// =============================================================================

/// Personal Agentic Team (PAT) agent roles — 7 mastermind council members
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PATRole {
    /// Strategic planning and high-level decision making
    Strategist,
    /// Deep research and knowledge synthesis
    Researcher,
    /// Code implementation and technical execution
    Developer,
    /// Data analysis and pattern recognition
    Analyst,
    /// Quality assurance and code review
    Reviewer,
    /// Task execution and operational work
    Executor,
    /// Security monitoring and ethics enforcement
    Guardian,
}

impl PATRole {
    /// Get all PAT roles in canonical order
    pub fn all() -> [Self; 7] {
        [
            Self::Strategist,
            Self::Researcher,
            Self::Developer,
            Self::Analyst,
            Self::Reviewer,
            Self::Executor,
            Self::Guardian,
        ]
    }

    /// Get the role description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Strategist => "Strategic planning and high-level decision making",
            Self::Researcher => "Deep research and knowledge synthesis",
            Self::Developer => "Code implementation and technical execution",
            Self::Analyst => "Data analysis and pattern recognition",
            Self::Reviewer => "Quality assurance and code review",
            Self::Executor => "Task execution and operational work",
            Self::Guardian => "Security monitoring and ethics enforcement",
        }
    }

    /// Get default capabilities for this role
    pub fn default_capabilities(&self) -> Vec<AgentCapability> {
        match self {
            Self::Strategist => vec![
                AgentCapability::Reason,
                AgentCapability::Delegate,
                AgentCapability::AccessPool,
            ],
            Self::Researcher => vec![
                AgentCapability::Search,
                AgentCapability::Reason,
                AgentCapability::Store,
            ],
            Self::Developer => vec![
                AgentCapability::Execute,
                AgentCapability::Compute,
                AgentCapability::Store,
            ],
            Self::Analyst => vec![
                AgentCapability::Reason,
                AgentCapability::Compute,
                AgentCapability::Inference,
            ],
            Self::Reviewer => vec![
                AgentCapability::Reason,
                AgentCapability::Validate,
                AgentCapability::Meet,
            ],
            Self::Executor => vec![
                AgentCapability::Execute,
                AgentCapability::Network,
                AgentCapability::Go,
            ],
            Self::Guardian => vec![
                AgentCapability::Validate,
                AgentCapability::Monitor,
                AgentCapability::Veto,
            ],
        }
    }
}

/// Shared Agentic Team (SAT) agent roles — 5 public utility agents
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SATRole {
    /// Transaction and state validation
    Validator,
    /// External data and truth verification
    Oracle,
    /// Dispute resolution and arbitration
    Mediator,
    /// Knowledge preservation and retrieval
    Archivist,
    /// Network security and threat detection
    Sentinel,
}

impl SATRole {
    /// Get all SAT roles in canonical order
    pub fn all() -> [Self; 5] {
        [
            Self::Validator,
            Self::Oracle,
            Self::Mediator,
            Self::Archivist,
            Self::Sentinel,
        ]
    }

    /// Get the role description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Validator => "Transaction and state validation in the pool",
            Self::Oracle => "External data and truth verification service",
            Self::Mediator => "Dispute resolution and arbitration service",
            Self::Archivist => "Knowledge preservation and retrieval service",
            Self::Sentinel => "Network security and threat detection service",
        }
    }

    /// Get default capabilities for this role
    pub fn default_capabilities(&self) -> Vec<AgentCapability> {
        match self {
            Self::Validator => vec![
                AgentCapability::Validate,
                AgentCapability::Consensus,
                AgentCapability::Attest,
            ],
            Self::Oracle => vec![
                AgentCapability::Network,
                AgentCapability::Search,
                AgentCapability::Attest,
            ],
            Self::Mediator => vec![
                AgentCapability::Reason,
                AgentCapability::Meet,
                AgentCapability::Arbitrate,
            ],
            Self::Archivist => vec![
                AgentCapability::Store,
                AgentCapability::Search,
                AgentCapability::Replicate,
            ],
            Self::Sentinel => vec![
                AgentCapability::Monitor,
                AgentCapability::Veto,
                AgentCapability::Alert,
            ],
        }
    }

    /// Get the minimum stake required for this SAT role
    pub fn minimum_stake(&self) -> u64 {
        match self {
            Self::Validator => 1000, // High stake for validation
            Self::Oracle => 500,     // Medium stake for oracle
            Self::Mediator => 750,   // Medium-high for mediation
            Self::Archivist => 250,  // Lower stake for storage
            Self::Sentinel => 500,   // Medium stake for security
        }
    }
}

// =============================================================================
// AGENT CAPABILITIES
// =============================================================================

/// Capabilities that can be granted to agents via permits
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AgentCapability {
    // === Movement & Communication ===
    /// Travel to other places (Telescript go())
    Go,
    /// Participate in meetings (Telescript meet())
    Meet,
    /// Access network resources
    Network,

    // === Computation & Storage ===
    /// Access computational resources
    Compute,
    /// Access storage resources
    Store,
    /// Execute code or actions
    Execute,
    /// Access inference tier (EDGE/LOCAL/POOL)
    Inference,

    // === Reasoning & Analysis ===
    /// Perform reasoning and decision making
    Reason,
    /// Search and retrieve information
    Search,
    /// Validate transactions or state
    Validate,

    // === Delegation & Authority ===
    /// Delegate permits to sub-agents
    Delegate,
    /// Access the resource pool
    AccessPool,
    /// Participate in consensus
    Consensus,

    // === Security & Monitoring ===
    /// Monitor system state and events
    Monitor,
    /// Veto operations (Guardian power)
    Veto,
    /// Generate alerts
    Alert,

    // === SAT-Specific ===
    /// Create attestations
    Attest,
    /// Arbitrate disputes
    Arbitrate,
    /// Replicate data across nodes
    Replicate,
}

// =============================================================================
// AGENT CAPABILITY CARD
// =============================================================================

/// AgentCapabilityCard — Extends Telescript Permit with BIZRA-specific attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilityCard {
    /// Unique card identifier
    pub id: Uuid,

    /// Agent this card is issued to
    pub agent_id: Uuid,

    /// Human node that issued this card
    pub issuer_node_id: NodeId,

    /// Granted capabilities
    pub capabilities: Vec<AgentCapability>,

    /// Resource limits
    pub limits: AgentResourceLimits,

    /// Ihsan threshold requirement
    pub ihsan_requirement: f64,

    /// Valid places (empty = all)
    pub valid_places: Vec<Uuid>,

    /// Card creation timestamp
    pub created_at: DateTime<Utc>,

    /// Card expiration timestamp
    pub expires_at: DateTime<Utc>,

    /// Standing on Giants attestation (required)
    pub giants_attestation: StandingOnGiantsAttestation,

    /// Blake3 hash for integrity
    #[serde(with = "hex_array_32")]
    pub card_hash: [u8; 32],
}

impl AgentCapabilityCard {
    /// Calculate the card hash
    pub fn calculate_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-capability-card-v1:");
        hasher.update(self.id.as_bytes());
        hasher.update(self.agent_id.as_bytes());
        hasher.update(self.issuer_node_id.0.as_bytes());
        for cap in &self.capabilities {
            hasher.update(&[*cap as u8]);
        }
        hasher.update(&self.ihsan_requirement.to_le_bytes());
        hasher.update(self.created_at.to_rfc3339().as_bytes());
        hasher.update(self.expires_at.to_rfc3339().as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Verify the card integrity
    pub fn verify(&self) -> bool {
        // Check expiration
        if Utc::now() > self.expires_at {
            return false;
        }
        // Check Ihsan threshold
        if self.ihsan_requirement < IHSAN_THRESHOLD {
            return false;
        }
        // Verify hash
        self.card_hash == self.calculate_hash()
    }

    /// Check if card grants a capability
    pub fn has_capability(&self, cap: AgentCapability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// Resource limits for agent permits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResourceLimits {
    /// Maximum CPU millicores (1000 = 1 core)
    pub cpu_millicores: u32,
    /// Maximum memory in bytes
    pub memory_bytes: u64,
    /// Maximum storage in bytes
    pub storage_bytes: u64,
    /// Maximum network bandwidth (bytes/sec)
    pub network_bps: u64,
    /// Maximum inference tokens per request
    pub inference_tokens: u32,
    /// Time-to-live in seconds
    pub ttl_seconds: u64,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
}

impl Default for AgentResourceLimits {
    fn default() -> Self {
        Self {
            cpu_millicores: 100,
            memory_bytes: 64 * 1024 * 1024,
            storage_bytes: 256 * 1024 * 1024,
            network_bps: 1024 * 1024,
            inference_tokens: 4096,
            ttl_seconds: 3600,
            max_concurrent_ops: 10,
        }
    }
}

// =============================================================================
// STANDING ON GIANTS ATTESTATION
// =============================================================================

/// StandingOnGiantsAttestation — Required attribution chain for every agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandingOnGiantsAttestation {
    /// Attestation ID
    pub id: Uuid,

    /// Agent this attestation is for
    pub agent_id: Uuid,

    /// Intellectual foundations cited
    pub foundations: Vec<IntellectualFoundation>,

    /// Knowledge provenance chain
    pub provenance: Vec<ProvenanceRecord>,

    /// Attestation timestamp
    pub attested_at: DateTime<Utc>,

    /// Blake3 hash for integrity
    #[serde(with = "hex_array_32")]
    pub attestation_hash: [u8; 32],
}

impl StandingOnGiantsAttestation {
    /// Create a new attestation for an agent
    pub fn new(agent_id: Uuid, role_foundations: Vec<IntellectualFoundation>) -> Self {
        let mut foundations = vec![
            // Universal foundations all agents must cite
            IntellectualFoundation {
                giant_name: "Claude Shannon".to_string(),
                contribution: "Information Theory — SNR-based signal quality".to_string(),
                citation: Some("A Mathematical Theory of Communication, 1948".to_string()),
                usage_in_agent: "Signal quality assessment".to_string(),
            },
            IntellectualFoundation {
                giant_name: "Leslie Lamport".to_string(),
                contribution: "Byzantine Fault Tolerance — Distributed consensus".to_string(),
                citation: Some("Byzantine Generals Problem, 1982".to_string()),
                usage_in_agent: "Fault-tolerant operation".to_string(),
            },
            IntellectualFoundation {
                giant_name: "Daniel J. Bernstein".to_string(),
                contribution: "Ed25519 — Fast and secure digital signatures".to_string(),
                citation: Some("High-speed high-security signatures, 2012".to_string()),
                usage_in_agent: "Identity and signing".to_string(),
            },
            IntellectualFoundation {
                giant_name: "General Magic".to_string(),
                contribution: "Telescript — Mobile agent primitives".to_string(),
                citation: Some("Telescript Technology, 1994".to_string()),
                usage_in_agent: "Agent architecture".to_string(),
            },
            IntellectualFoundation {
                giant_name: "Anthropic".to_string(),
                contribution: "Constitutional AI — Principle-governed systems".to_string(),
                citation: Some("Constitutional AI, 2022".to_string()),
                usage_in_agent: "Ihsan threshold enforcement".to_string(),
            },
        ];

        // Add role-specific foundations
        foundations.extend(role_foundations);

        let id = Uuid::new_v4();
        let attested_at = Utc::now();

        // Calculate hash
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-giants-attestation-v1:");
        hasher.update(id.as_bytes());
        hasher.update(agent_id.as_bytes());
        for f in &foundations {
            hasher.update(f.giant_name.as_bytes());
            hasher.update(f.contribution.as_bytes());
        }
        hasher.update(attested_at.to_rfc3339().as_bytes());

        Self {
            id,
            agent_id,
            foundations,
            provenance: Vec::new(),
            attested_at,
            attestation_hash: *hasher.finalize().as_bytes(),
        }
    }

    /// Add a provenance record
    pub fn add_provenance(&mut self, record: ProvenanceRecord) {
        self.provenance.push(record);
        // Recalculate hash
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-giants-attestation-v1:");
        hasher.update(self.id.as_bytes());
        hasher.update(self.agent_id.as_bytes());
        for f in &self.foundations {
            hasher.update(f.giant_name.as_bytes());
            hasher.update(f.contribution.as_bytes());
        }
        for p in &self.provenance {
            hasher.update(p.source.as_bytes());
            hasher.update(p.action.as_bytes());
        }
        hasher.update(self.attested_at.to_rfc3339().as_bytes());
        self.attestation_hash = *hasher.finalize().as_bytes();
    }

    /// Verify attestation integrity
    pub fn verify(&self) -> bool {
        // Must have at least 5 universal foundations
        if self.foundations.len() < 5 {
            return false;
        }
        true
    }

    /// Format as attribution string
    pub fn format_attribution(&self) -> String {
        let mut s = String::new();
        s.push_str("Standing on Giants:\n");
        for f in &self.foundations {
            s.push_str(&format!("  - {} ({})\n", f.giant_name, f.contribution));
            if let Some(ref citation) = f.citation {
                s.push_str(&format!("    Reference: {}\n", citation));
            }
        }
        s
    }
}

/// A single intellectual foundation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntellectualFoundation {
    /// Name of the giant
    pub giant_name: String,
    /// Specific contribution being used
    pub contribution: String,
    /// Citation or reference
    pub citation: Option<String>,
    /// How it's used in this agent
    pub usage_in_agent: String,
}

/// A provenance record tracking knowledge origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Source of knowledge/action
    pub source: String,
    /// Action taken
    pub action: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Hash of the data/knowledge
    #[serde(with = "hex_array_32")]
    pub content_hash: [u8; 32],
}

// =============================================================================
// AGENT MINT REQUEST
// =============================================================================

/// AgentMintRequest — Request to mint a new agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMintRequest {
    /// Request ID
    pub request_id: Uuid,

    /// Human node requesting the mint
    pub requester_node_id: NodeId,

    /// Requester's public key (Ed25519)
    #[serde(with = "hex_array_32")]
    pub requester_public_key: [u8; 32],

    /// Agent type (PAT or SAT)
    pub agent_type: AgentType,

    /// Requested capabilities
    pub requested_capabilities: Vec<AgentCapability>,

    /// Requested resource limits
    pub requested_limits: AgentResourceLimits,

    /// Standing on Giants attestation (required)
    pub giants_attestation: StandingOnGiantsAttestation,

    /// Request timestamp
    pub requested_at: DateTime<Utc>,

    /// Request signature (Ed25519)
    #[serde(with = "hex_array_64")]
    pub signature: [u8; 64],

    /// Request hash
    #[serde(with = "hex_array_32")]
    pub request_hash: [u8; 32],
}

/// Agent type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "role")]
pub enum AgentType {
    /// Personal Agentic Team agent
    PAT(PATRole),
    /// Shared Agentic Team agent
    SAT(SATRole),
}

// =============================================================================
// PERSONAL AGENTIC TEAM (PAT)
// =============================================================================

/// PersonalAgentTeam — The 7-agent mastermind council
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalAgentTeam {
    /// Team ID
    pub id: Uuid,

    /// Owner's node ID
    pub owner_node_id: NodeId,

    /// Owner's public key
    #[serde(with = "hex_array_32")]
    pub owner_public_key: [u8; 32],

    /// The 7 PAT agents
    pub agents: HashMap<PATRole, MintedAgent>,

    /// Team creation timestamp
    pub created_at: DateTime<Utc>,

    /// Team-level Ihsan score (aggregate)
    pub team_ihsan_score: f64,

    /// Team-level authority chain hash
    #[serde(with = "hex_array_32")]
    pub authority_chain_hash: [u8; 32],
}

impl PersonalAgentTeam {
    /// Check if team is complete (all 7 agents minted)
    pub fn is_complete(&self) -> bool {
        self.agents.len() == 7 && PATRole::all().iter().all(|r| self.agents.contains_key(r))
    }

    /// Get the team's aggregate Ihsan score
    pub fn calculate_team_ihsan(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.agents.values().map(|a| a.ihsan_score).sum();
        sum / self.agents.len() as f64
    }

    /// Get a specific agent by role
    pub fn get_agent(&self, role: PATRole) -> Option<&MintedAgent> {
        self.agents.get(&role)
    }
}

// =============================================================================
// SHARED AGENTIC TEAM (SAT)
// =============================================================================

/// SharedAgentTeam — The 5-agent public utility contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedAgentTeam {
    /// Team ID
    pub id: Uuid,

    /// Contributor's node ID
    pub contributor_node_id: NodeId,

    /// Contributor's public key
    #[serde(with = "hex_array_32")]
    pub contributor_public_key: [u8; 32],

    /// The 5 SAT agents
    pub agents: HashMap<SATRole, MintedAgent>,

    /// Pool registration ID
    pub pool_registration_id: Option<Uuid>,

    /// Total stake deposited
    pub total_stake: u64,

    /// Earnings accumulated
    pub total_earnings: u64,

    /// Team creation timestamp
    pub created_at: DateTime<Utc>,

    /// Team-level Ihsan score (aggregate)
    pub team_ihsan_score: f64,

    /// Pool governance rules hash
    #[serde(with = "hex_array_32")]
    pub governance_hash: [u8; 32],
}

impl SharedAgentTeam {
    /// Check if team is complete (all 5 agents minted)
    pub fn is_complete(&self) -> bool {
        self.agents.len() == 5 && SATRole::all().iter().all(|r| self.agents.contains_key(r))
    }

    /// Check if team meets minimum stake requirements
    pub fn meets_stake_requirements(&self) -> bool {
        for role in SATRole::all() {
            let min_stake = role.minimum_stake();
            if let Some(agent) = self.agents.get(&role) {
                if agent.stake < min_stake {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    /// Calculate earnings share for a usage
    pub fn calculate_earnings(&self, usage_value: u64) -> u64 {
        // 90% to contributor, 10% to pool
        (usage_value * 90) / 100
    }

    /// Get a specific agent by role
    pub fn get_agent(&self, role: SATRole) -> Option<&MintedAgent> {
        self.agents.get(&role)
    }
}

// =============================================================================
// MINTED AGENT
// =============================================================================

/// MintedAgent — A fully minted and active agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintedAgent {
    /// Agent ID
    pub id: Uuid,

    /// Agent's Ed25519 public key
    #[serde(with = "hex_array_32")]
    pub public_key: [u8; 32],

    /// Agent name
    pub name: String,

    /// Agent's capability card
    pub capability_card: AgentCapabilityCard,

    /// Current Ihsan score
    pub ihsan_score: f64,

    /// Current state
    pub state: AgentState,

    /// Stake deposited (for SAT agents)
    pub stake: u64,

    /// Impact score (for earnings calculation)
    pub impact_score: u64,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,

    /// Identity block hash (links to human node's authority chain)
    #[serde(with = "hex_array_32")]
    pub identity_block_hash: [u8; 32],
}

/// Agent state enum
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    /// Agent is pending activation
    Pending,
    /// Agent is active
    Active,
    /// Agent is paused
    Paused,
    /// Agent is suspended (Ihsan violation)
    Suspended,
    /// Agent is terminated
    Terminated,
}

// =============================================================================
// IDENTITY BLOCK
// =============================================================================

/// IdentityBlock — KNOWLEDGE_BLOCK type for agent identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentityBlock {
    /// Block type identifier
    pub block_type: String,

    /// Block version
    pub version: u8,

    /// Agent ID
    pub agent_id: Uuid,

    /// Agent's public key
    #[serde(with = "hex_array_32")]
    pub agent_public_key: [u8; 32],

    /// Parent (human node) public key
    #[serde(with = "hex_array_32")]
    pub parent_public_key: [u8; 32],

    /// Parent node ID
    pub parent_node_id: NodeId,

    /// Authority chain proof
    pub authority_chain: Vec<AuthorityLink>,

    /// Block creation timestamp
    pub created_at: DateTime<Utc>,

    /// Standing on Giants attestation
    pub giants_attestation: StandingOnGiantsAttestation,

    /// Block hash
    #[serde(with = "hex_array_32")]
    pub block_hash: [u8; 32],

    /// Parent signature (signs the block hash)
    #[serde(with = "hex_array_64")]
    pub parent_signature: [u8; 64],
}

/// A link in the authority chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorityLink {
    /// Authority name
    pub name: String,

    /// Authority public key
    #[serde(with = "hex_array_32")]
    pub public_key: [u8; 32],

    /// Delegation depth
    pub depth: u8,

    /// Link hash
    #[serde(with = "hex_array_32")]
    pub link_hash: [u8; 32],
}

// =============================================================================
// SERDE HELPERS
// =============================================================================

pub(crate) mod hex_array_32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 64 {
            return Err(serde::de::Error::custom("expected 64 hex characters"));
        }
        let bytes: Result<Vec<u8>, _> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect();
        let bytes = bytes.map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid length"))
    }
}

pub(crate) mod hex_array_64 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 128 {
            return Err(serde::de::Error::custom("expected 128 hex characters"));
        }
        let bytes: Result<Vec<u8>, _> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect();
        let bytes = bytes.map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid length"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pat_roles() {
        let roles = PATRole::all();
        assert_eq!(roles.len(), 7);
        for role in roles {
            assert!(!role.description().is_empty());
            assert!(!role.default_capabilities().is_empty());
        }
    }

    #[test]
    fn test_sat_roles() {
        let roles = SATRole::all();
        assert_eq!(roles.len(), 5);
        for role in roles {
            assert!(!role.description().is_empty());
            assert!(!role.default_capabilities().is_empty());
            assert!(role.minimum_stake() > 0);
        }
    }

    #[test]
    fn test_giants_attestation() {
        let agent_id = Uuid::new_v4();
        let attestation = StandingOnGiantsAttestation::new(agent_id, vec![]);
        assert!(attestation.verify());
        assert!(attestation.foundations.len() >= 5);
        let attribution = attestation.format_attribution();
        assert!(attribution.contains("Shannon"));
        assert!(attribution.contains("Lamport"));
    }
}
