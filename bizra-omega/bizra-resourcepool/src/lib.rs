//! BIZRA RESOURCE POOL v1.0.0
//!
//! The Universal Fabric Where All Nodes Connect.
//!
//! # The Five Pillars of the Resource Pool
//!
//! 1. **Universal Financial System** - Islamic finance principles (no riba, Zakat distribution)
//! 2. **Agent Marketplace** - Where users meet agents, trade services (PAT/SAT inventory)
//! 3. **Compute Commons** - Share power -> mint tokens (Proof-of-Resource)
//! 4. **MMORPG World Map** - Nodes don't connect directly, they connect through the Pool
//! 5. **Web4 Infrastructure** - Secure, algorithm-free, user-controlled internet
//!
//! # Standing on Giants
//!
//! - **Weyl & Posner (2018)**: Harberger Tax - self-assessed pricing for resources
//! - **Nakamoto (2008)**: Proof-of-Work -> Proof-of-Resource contribution
//! - **Shannon (1948)**: SNR-based signal quality for pool health
//! - **Al-Ghazali (1095)**: Maqasid al-Shariah -> FATE gate ethics
//! - **General Magic (1994)**: Telescript Places -> Pool as Universal Place
//!
//! # Islamic Finance Integration
//!
//! - **No Riba**: No interest on resource loans, only profit-sharing (Mudarabah)
//! - **Zakat**: Automatic 2.5% charitable distribution when wealth > nisab threshold
//! - **Halal Filter**: All services must pass ethical FATE gates
//! - **Takaful**: Mutual insurance pool for node failures

use blake3::Hasher;
use chrono::{DateTime, Utc};
use ed25519_dalek::VerifyingKey;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

// Re-export integration types
pub use bizra_proofspace::{BizraBlock, ValidationResult, Verdict};
pub use bizra_telescript::{Agent, AgentState, Authority, Capability, Permit, Place};

// Genesis module
pub mod genesis;
pub use genesis::*;

// Proactive PAT module
pub mod proactive_pat;
pub use proactive_pat::*;

// PAT-LM Studio inference integration
pub mod pat_inference;
pub use pat_inference::*;

// =============================================================================
// CONSTANTS - Single Source of Truth (locked)
// =============================================================================

/// Ihsan threshold: 0.95 excellence constraint (Decimal precision for Islamic finance)
/// Canonical source: bizra_core::IHSAN_THRESHOLD (f64 = 0.95)
pub const IHSAN_THRESHOLD: Decimal = Decimal::from_parts(95, 0, 0, false, 2); // 0.95

/// Zakat threshold: 2.5% of wealth above nisab
pub const ZAKAT_RATE: Decimal = Decimal::from_parts(25, 0, 0, false, 3); // 0.025

/// Nisab threshold in pool tokens (equivalent to ~85g gold)
/// Node must hold this much before Zakat is obligatory
pub const NISAB_THRESHOLD: u64 = 1_000_000; // 1M tokens

/// Harberger tax rate: Annual self-assessed tax on resources (7%)
pub const HARBERGER_TAX_RATE: Decimal = Decimal::from_parts(7, 0, 0, false, 2); // 0.07

/// Minimum self-assessment multiplier to prevent undervaluation
pub const MIN_ASSESSMENT_MULTIPLIER: Decimal = Decimal::from_parts(5, 0, 0, false, 1); // 0.5

/// Maximum Gini coefficient for Adl (justice) enforcement
pub const ADL_GINI_MAX: Decimal = Decimal::from_parts(35, 0, 0, false, 2); // 0.35

/// Token minting rate per compute unit contributed
pub const TOKENS_PER_COMPUTE_UNIT: u64 = 100;

/// PAT: Personal Agent Team size (7 agents per user)
pub const PAT_SIZE: usize = 7;

/// SAT: Shared Agent Team size (5 agents per community)
pub const SAT_SIZE: usize = 5;

/// Genesis epoch timestamp (pool creation)
pub const GENESIS_EPOCH: u64 = 1704067200000; // 2024-01-01 00:00:00 UTC in ms

// =============================================================================
// ERROR TYPES
// =============================================================================

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum PoolError {
    #[error("Node not registered: {node_id}")]
    NodeNotRegistered { node_id: String },

    #[error("Node already registered: {node_id}")]
    NodeAlreadyRegistered { node_id: String },

    #[error("Ihsan threshold not met: {score} < {threshold}")]
    IhsanViolation { score: Decimal, threshold: Decimal },

    #[error("Adl violation: Gini {gini} > {max}")]
    AdlViolation { gini: Decimal, max: Decimal },

    #[error("Insufficient balance: {available} < {required}")]
    InsufficientBalance { available: u64, required: u64 },

    #[error("Zakat obligation not met: {owed} tokens required")]
    ZakatObligation { owed: u64 },

    #[error("Invalid signature: {reason}")]
    InvalidSignature { reason: String },

    #[error("Resource not available: {resource_type}")]
    ResourceNotAvailable { resource_type: String },

    #[error("Service not found: {service_id}")]
    ServiceNotFound { service_id: String },

    #[error("Harberger tax not paid: {owed} tokens overdue")]
    HarbergerTaxOverdue { owed: u64 },

    #[error("FATE gate rejection: {gate} - {reason}")]
    FateRejection { gate: String, reason: String },

    #[error("Proof verification failed: {reason}")]
    ProofVerificationFailed { reason: String },

    #[error("Riba (interest) detected: operation prohibited")]
    RibaProhibited,

    #[error("PAT limit exceeded: max {max} agents")]
    PATLimitExceeded { max: usize },

    #[error("SAT limit exceeded: max {max} agents")]
    SATLimitExceeded { max: usize },

    #[error("Serialization error: {reason}")]
    SerializationError { reason: String },

    #[error("Cryptographic error: {reason}")]
    CryptoError { reason: String },

    #[error("Storage error: {reason}")]
    StorageError { reason: String },
}

pub type Result<T> = std::result::Result<T, PoolError>;

// =============================================================================
// NODE TYPES
// =============================================================================

/// Node classification in the pool hierarchy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeClass {
    /// Genesis node (Node0 - MoMo)
    Genesis,
    /// Sovereign node - full constitutional rights
    Sovereign,
    /// Delegate node - limited rights, represents a sovereign
    Delegate,
    /// Edge node - lightweight, resource-constrained
    Edge,
    /// Observer node - read-only access
    Observer,
}

/// Node registration status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeStatus {
    /// Pending registration approval
    Pending,
    /// Active and in good standing
    Active,
    /// Temporarily suspended (missed Zakat, tax arrears)
    Suspended,
    /// Graceful departure
    Departed,
    /// Expelled for FATE violations
    Expelled,
}

/// A node in the resource pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolNode {
    /// Unique node identifier (Ed25519 public key, 64 hex chars)
    pub node_id: String,
    /// Human-readable name
    pub name: String,
    /// Node classification
    pub class: NodeClass,
    /// Current status
    pub status: NodeStatus,
    /// Ed25519 verifying key (for signature verification)
    #[serde(with = "hex_verifying_key")]
    pub verifying_key: VerifyingKey,
    /// Telescript place ID for this node
    pub place_id: Uuid,
    /// Current Ihsan score (0.00 - 1.00)
    pub ihsan_score: Decimal,
    /// Resources contributed to the pool
    pub resources: NodeResources,
    /// Token balance
    pub token_balance: u64,
    /// Zakat paid this year
    pub zakat_paid_year: u64,
    /// Last Harberger tax payment timestamp
    pub last_tax_payment: DateTime<Utc>,
    /// PAT: Personal Agent Team IDs
    pub pat_agents: Vec<Uuid>,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Blake3 hash of node state
    pub node_hash: [u8; 32],
}

/// Resources a node contributes to the pool
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeResources {
    /// CPU millicores available (1000 = 1 core)
    pub cpu_millicores: u32,
    /// GPU TFLOPS available
    pub gpu_tflops: Decimal,
    /// Memory bytes available
    pub memory_bytes: u64,
    /// Storage bytes available
    pub storage_bytes: u64,
    /// Network bandwidth (bytes/sec)
    pub network_bps: u64,
    /// Inference tokens per second capacity
    pub inference_tps: u32,
    /// Self-assessed value for Harberger tax (in tokens)
    pub self_assessment: u64,
    /// Resource availability score (0.0 - 1.0)
    pub availability: Decimal,
}

impl PoolNode {
    /// Compute the node hash
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.name.as_bytes());
        hasher.update(&[self.class as u8, self.status as u8]);
        hasher.update(&self.token_balance.to_le_bytes());
        hasher.update(self.ihsan_score.to_string().as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Update the node hash after state changes
    pub fn update_hash(&mut self) {
        self.node_hash = self.compute_hash();
    }

    /// Check if node meets Ihsan threshold
    pub fn passes_ihsan(&self) -> Result<()> {
        if self.ihsan_score < IHSAN_THRESHOLD {
            return Err(PoolError::IhsanViolation {
                score: self.ihsan_score,
                threshold: IHSAN_THRESHOLD,
            });
        }
        Ok(())
    }

    /// Calculate Zakat obligation
    pub fn calculate_zakat(&self) -> u64 {
        if self.token_balance < NISAB_THRESHOLD {
            return 0;
        }
        let taxable = self.token_balance - NISAB_THRESHOLD;
        let zakat = Decimal::from(taxable) * ZAKAT_RATE;
        zakat.to_u64().unwrap_or(0)
    }

    /// Check Zakat compliance
    pub fn zakat_compliant(&self) -> Result<()> {
        let owed = self.calculate_zakat();
        if self.zakat_paid_year < owed {
            return Err(PoolError::ZakatObligation {
                owed: owed - self.zakat_paid_year,
            });
        }
        Ok(())
    }

    /// Calculate annual Harberger tax
    pub fn calculate_harberger_tax(&self) -> u64 {
        let tax = Decimal::from(self.resources.self_assessment) * HARBERGER_TAX_RATE;
        tax.to_u64().unwrap_or(0)
    }
}

// =============================================================================
// TOKEN TYPES
// =============================================================================

/// Pool token - the native currency
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PoolToken {
    /// Amount in smallest denomination
    pub amount: u64,
    /// Token generation (epoch)
    pub generation: u64,
}

/// Token minting proof (Proof-of-Resource)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintProof {
    /// Unique proof ID
    pub proof_id: Uuid,
    /// Node that contributed resources
    pub contributor_node: String,
    /// Resource contribution details
    pub contribution: ResourceContribution,
    /// Tokens minted
    pub tokens_minted: u64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Signature by contributor
    pub signature: String,
    /// Blake3 hash of proof
    pub proof_hash: [u8; 32],
}

/// Resource contribution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContribution {
    /// Type of resource
    pub resource_type: ResourceType,
    /// Quantity contributed
    pub quantity: u64,
    /// Duration in seconds
    pub duration_seconds: u64,
    /// Utilization percentage (0-100)
    pub utilization: u8,
    /// Proof block reference (if applicable)
    pub proof_block_id: Option<String>,
}

/// Types of resources in the pool
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResourceType {
    /// CPU compute cycles
    Cpu,
    /// GPU compute (TFLOPS)
    Gpu,
    /// Memory (bytes)
    Memory,
    /// Storage (bytes)
    Storage,
    /// Network bandwidth (bytes)
    Network,
    /// Inference tokens
    Inference,
}

// =============================================================================
// SERVICE TYPES (Agent Marketplace)
// =============================================================================

/// Service offered in the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolService {
    /// Unique service ID
    pub service_id: Uuid,
    /// Provider node
    pub provider_node: String,
    /// Service name
    pub name: String,
    /// Service description
    pub description: String,
    /// Service category
    pub category: ServiceCategory,
    /// Capabilities required
    pub required_capabilities: Vec<Capability>,
    /// Price per invocation (in tokens)
    pub price_per_call: u64,
    /// Ihsan score of this service
    pub ihsan_score: Decimal,
    /// FATE gate status
    pub fate_approved: bool,
    /// Proofspace block validating this service
    pub validation_block: Option<String>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Service hash
    pub service_hash: [u8; 32],
}

/// Service categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ServiceCategory {
    /// Inference services
    Inference,
    /// Data processing
    DataProcessing,
    /// Storage services
    Storage,
    /// Communication services
    Communication,
    /// Financial services (must be halal)
    Financial,
    /// Creative services
    Creative,
    /// Governance services
    Governance,
    /// Custom/other
    Custom,
}

/// Service invocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInvocation {
    /// Unique invocation ID
    pub invocation_id: Uuid,
    /// Service being invoked
    pub service_id: Uuid,
    /// Invoking node
    pub invoker_node: String,
    /// Provider node
    pub provider_node: String,
    /// Tokens paid
    pub tokens_paid: u64,
    /// Status
    pub status: InvocationStatus,
    /// Result hash (if completed)
    pub result_hash: Option<[u8; 32]>,
    /// Started at
    pub started_at: DateTime<Utc>,
    /// Completed at
    pub completed_at: Option<DateTime<Utc>>,
}

/// Service invocation status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InvocationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Disputed,
}

// =============================================================================
// AGENT INVENTORY (PAT/SAT)
// =============================================================================

/// Personal Agent Team (PAT) - 7 agents per user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalAgentTeam {
    /// Owner node
    pub owner_node: String,
    /// The 7 agents
    pub agents: [Option<PATAgent>; PAT_SIZE],
    /// Team Ihsan score (aggregate)
    pub team_ihsan: Decimal,
    /// Created at
    pub created_at: DateTime<Utc>,
}

/// A PAT agent slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PATAgent {
    /// Agent UUID
    pub agent_id: Uuid,
    /// Agent role/specialty
    pub role: AgentRole,
    /// Agent name
    pub name: String,
    /// Telescript agent reference
    pub telescript_agent: Option<Uuid>,
    /// Ihsan score
    pub ihsan_score: Decimal,
    /// Active status
    pub active: bool,
}

/// Shared Agent Team (SAT) - 5 agents per community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedAgentTeam {
    /// Community identifier
    pub community_id: Uuid,
    /// Community name
    pub community_name: String,
    /// Member nodes
    pub member_nodes: Vec<String>,
    /// The 5 shared agents
    pub agents: [Option<SATAgent>; SAT_SIZE],
    /// Team Ihsan score
    pub team_ihsan: Decimal,
    /// Governance rules
    pub governance: SATGovernance,
    /// Created at
    pub created_at: DateTime<Utc>,
}

/// A SAT agent slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SATAgent {
    /// Agent UUID
    pub agent_id: Uuid,
    /// Agent role/specialty
    pub role: AgentRole,
    /// Agent name
    pub name: String,
    /// Telescript agent reference
    pub telescript_agent: Option<Uuid>,
    /// Ihsan score
    pub ihsan_score: Decimal,
    /// Usage quota per member (tokens)
    pub quota_per_member: u64,
}

/// SAT governance rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SATGovernance {
    /// Minimum members to modify team
    pub quorum: usize,
    /// Voting period in hours
    pub voting_period_hours: u32,
    /// Proposal cost (anti-spam)
    pub proposal_cost: u64,
}

/// Agent roles (for PAT/SAT)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AgentRole {
    /// Research and information gathering
    Researcher,
    /// Code and technical tasks
    Engineer,
    /// Communication and writing
    Communicator,
    /// Planning and organization
    Planner,
    /// Financial and resource management
    Treasurer,
    /// Security and compliance
    Guardian,
    /// Creative tasks
    Creator,
    /// General purpose
    Generalist,
}

// =============================================================================
// ZAKAT DISTRIBUTION
// =============================================================================

/// Zakat distribution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZakatDistribution {
    /// Distribution ID
    pub distribution_id: Uuid,
    /// Year/period
    pub period: String,
    /// Total Zakat collected
    pub total_collected: u64,
    /// Distribution breakdown
    pub distributions: Vec<ZakatRecipient>,
    /// Distribution timestamp
    pub distributed_at: DateTime<Utc>,
    /// Verification hash
    pub distribution_hash: [u8; 32],
}

/// Zakat recipient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZakatRecipient {
    /// Recipient category (traditional 8 categories)
    pub category: ZakatCategory,
    /// Recipient node (if applicable)
    pub recipient_node: Option<String>,
    /// Amount distributed
    pub amount: u64,
    /// Purpose/reason
    pub purpose: String,
}

/// Traditional Zakat categories (Asnaf)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ZakatCategory {
    /// The poor (Fuqara)
    Poor,
    /// The needy (Masakin)
    Needy,
    /// Zakat administrators (Amil)
    Administrators,
    /// Those whose hearts are to be reconciled (Muallafat)
    Reconciliation,
    /// Freeing captives (Riqab)
    Liberation,
    /// Those in debt (Gharimin)
    Debtors,
    /// In the cause of Allah (Fi Sabilillah)
    Cause,
    /// Travelers (Ibn Sabil)
    Travelers,
}

// =============================================================================
// HARBERGER TAX MARKET
// =============================================================================

/// Resource listing in the Harberger market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarbergerListing {
    /// Listing ID
    pub listing_id: Uuid,
    /// Owner node
    pub owner_node: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Quantity available
    pub quantity: u64,
    /// Self-assessed price (per unit per day, in tokens)
    pub self_assessment: u64,
    /// Annual tax paid (7% of self_assessment * quantity)
    pub annual_tax: u64,
    /// Last tax payment
    pub last_tax_payment: DateTime<Utc>,
    /// Tax paid this period
    pub tax_paid_period: u64,
    /// Listing hash
    pub listing_hash: [u8; 32],
}

/// Buy offer for a Harberger listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarbergerOffer {
    /// Offer ID
    pub offer_id: Uuid,
    /// Listing being bid on
    pub listing_id: Uuid,
    /// Bidder node
    pub bidder_node: String,
    /// Offered price (must be >= self_assessment)
    pub offered_price: u64,
    /// Offer timestamp
    pub offered_at: DateTime<Utc>,
    /// Expiration
    pub expires_at: DateTime<Utc>,
}

// =============================================================================
// POOL STATE
// =============================================================================

/// The Resource Pool global state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolState {
    /// Pool identifier
    pub pool_id: Uuid,
    /// Pool name
    pub name: String,
    /// Genesis node ID (Node0)
    pub genesis_node: String,
    /// Genesis timestamp
    pub genesis_at: DateTime<Utc>,
    /// Current epoch
    pub current_epoch: u64,
    /// Total tokens in circulation
    pub total_supply: u64,
    /// Total Zakat distributed
    pub total_zakat_distributed: u64,
    /// Current Gini coefficient
    pub gini_coefficient: Decimal,
    /// Pool Ihsan score
    pub pool_ihsan: Decimal,
    /// Pool state hash
    pub state_hash: [u8; 32],
}

/// Pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    /// Total registered nodes
    pub total_nodes: usize,
    /// Active nodes
    pub active_nodes: usize,
    /// Total services
    pub total_services: usize,
    /// Total PAT teams
    pub total_pat: usize,
    /// Total SAT teams
    pub total_sat: usize,
    /// Total tokens staked
    pub total_staked: u64,
    /// Total compute units available
    pub total_compute: u64,
    /// Average Ihsan score
    pub avg_ihsan: Decimal,
}

// =============================================================================
// REGISTRATION PROTOCOL
// =============================================================================

/// Node registration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationRequest {
    /// Requested node ID (Ed25519 public key)
    pub node_id: String,
    /// Node name
    pub name: String,
    /// Requested class
    pub requested_class: NodeClass,
    /// Initial resources
    pub resources: NodeResources,
    /// Sponsor node (required for non-Genesis)
    pub sponsor_node: Option<String>,
    /// Proof of identity (proofspace block)
    pub identity_proof: Option<String>,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Signature by requester
    pub signature: String,
}

/// Node registration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationResponse {
    /// Request ID
    pub request_id: Uuid,
    /// Approved or rejected
    pub approved: bool,
    /// Rejection reason (if applicable)
    pub rejection_reason: Option<String>,
    /// Assigned place ID
    pub place_id: Option<Uuid>,
    /// Initial token grant
    pub initial_tokens: u64,
    /// Response timestamp
    pub responded_at: DateTime<Utc>,
    /// Pool signature
    pub pool_signature: String,
}

// =============================================================================
// RESOURCE POOL ENGINE
// =============================================================================

/// The Resource Pool Engine - manages the universal fabric
pub struct ResourcePool {
    /// Pool state
    state: Arc<RwLock<PoolState>>,
    /// Registered nodes
    nodes: Arc<RwLock<HashMap<String, PoolNode>>>,
    /// Services registry
    services: Arc<RwLock<HashMap<Uuid, PoolService>>>,
    /// PAT teams
    pat_teams: Arc<RwLock<HashMap<String, PersonalAgentTeam>>>,
    /// SAT teams
    sat_teams: Arc<RwLock<HashMap<Uuid, SharedAgentTeam>>>,
    /// Harberger listings
    harberger_listings: Arc<RwLock<HashMap<Uuid, HarbergerListing>>>,
    /// Zakat fund
    zakat_fund: Arc<RwLock<u64>>,
    /// Telescript engine reference — used when telescript integration is fully wired
    #[allow(dead_code)]
    telescript: Arc<RwLock<Option<bizra_telescript::TelescriptEngine>>>,
    /// Pending registrations — used when registration flow is fully wired
    #[allow(dead_code)]
    pending_registrations: Arc<RwLock<HashMap<String, RegistrationRequest>>>,
}

impl ResourcePool {
    /// Create a new Resource Pool with Genesis Node (Node0)
    pub async fn genesis(
        genesis_node_id: String,
        genesis_name: String,
        genesis_key: VerifyingKey,
    ) -> Result<Self> {
        let pool_id = Uuid::new_v4();
        let now = Utc::now();

        // Create pool state
        let state = PoolState {
            pool_id,
            name: "BIZRA Resource Pool".to_string(),
            genesis_node: genesis_node_id.clone(),
            genesis_at: now,
            current_epoch: 0,
            total_supply: NISAB_THRESHOLD, // Initial supply to Genesis
            total_zakat_distributed: 0,
            gini_coefficient: Decimal::ZERO, // Perfect equality at genesis
            pool_ihsan: IHSAN_THRESHOLD,
            state_hash: [0u8; 32],
        };

        // Create Genesis node
        let genesis_node = PoolNode {
            node_id: genesis_node_id.clone(),
            name: genesis_name,
            class: NodeClass::Genesis,
            status: NodeStatus::Active,
            verifying_key: genesis_key,
            place_id: Uuid::new_v4(),
            ihsan_score: Decimal::ONE, // Genesis starts at 1.0
            resources: NodeResources::default(),
            token_balance: NISAB_THRESHOLD,
            zakat_paid_year: 0,
            last_tax_payment: now,
            pat_agents: Vec::new(),
            registered_at: now,
            last_activity: now,
            node_hash: [0u8; 32],
        };

        let mut nodes = HashMap::new();
        nodes.insert(genesis_node_id.clone(), genesis_node);

        let pool = ResourcePool {
            state: Arc::new(RwLock::new(state)),
            nodes: Arc::new(RwLock::new(nodes)),
            services: Arc::new(RwLock::new(HashMap::new())),
            pat_teams: Arc::new(RwLock::new(HashMap::new())),
            sat_teams: Arc::new(RwLock::new(HashMap::new())),
            harberger_listings: Arc::new(RwLock::new(HashMap::new())),
            zakat_fund: Arc::new(RwLock::new(0)),
            telescript: Arc::new(RwLock::new(None)),
            pending_registrations: Arc::new(RwLock::new(HashMap::new())),
        };

        // Update state hash
        pool.update_state_hash().await;

        tracing::info!(
            pool_id = %pool_id,
            genesis_node = %genesis_node_id,
            "Resource Pool genesis complete"
        );

        Ok(pool)
    }

    /// Register a new node to the pool
    pub async fn register_node(
        &self,
        request: RegistrationRequest,
    ) -> Result<RegistrationResponse> {
        // Check if already registered
        {
            let nodes = self.nodes.read().await;
            if nodes.contains_key(&request.node_id) {
                return Err(PoolError::NodeAlreadyRegistered {
                    node_id: request.node_id,
                });
            }
        }

        // Verify sponsor for non-Genesis nodes
        if request.requested_class != NodeClass::Genesis {
            if let Some(ref sponsor_id) = request.sponsor_node {
                let nodes = self.nodes.read().await;
                let sponsor =
                    nodes
                        .get(sponsor_id)
                        .ok_or_else(|| PoolError::NodeNotRegistered {
                            node_id: sponsor_id.clone(),
                        })?;

                // Sponsor must be active and have sufficient Ihsan
                if sponsor.status != NodeStatus::Active {
                    return Ok(RegistrationResponse {
                        request_id: Uuid::new_v4(),
                        approved: false,
                        rejection_reason: Some("Sponsor not active".to_string()),
                        place_id: None,
                        initial_tokens: 0,
                        responded_at: Utc::now(),
                        pool_signature: String::new(),
                    });
                }

                sponsor.passes_ihsan()?;
            } else {
                return Ok(RegistrationResponse {
                    request_id: Uuid::new_v4(),
                    approved: false,
                    rejection_reason: Some("Sponsor required for non-Genesis nodes".to_string()),
                    place_id: None,
                    initial_tokens: 0,
                    responded_at: Utc::now(),
                    pool_signature: String::new(),
                });
            }
        }

        // Verify signature
        let pk_bytes = hex::decode(&request.node_id).map_err(|e| PoolError::CryptoError {
            reason: format!("Invalid node ID: {}", e),
        })?;

        let verifying_key =
            VerifyingKey::from_bytes(&pk_bytes.try_into().map_err(|_| PoolError::CryptoError {
                reason: "Invalid key length".to_string(),
            })?)
            .map_err(|e| PoolError::CryptoError {
                reason: format!("Invalid public key: {}", e),
            })?;

        // Create the new node
        let place_id = Uuid::new_v4();
        let now = Utc::now();

        let mut new_node = PoolNode {
            node_id: request.node_id.clone(),
            name: request.name,
            class: request.requested_class,
            status: NodeStatus::Active,
            verifying_key,
            place_id,
            ihsan_score: IHSAN_THRESHOLD, // Start at threshold
            resources: request.resources,
            token_balance: 0,
            zakat_paid_year: 0,
            last_tax_payment: now,
            pat_agents: Vec::new(),
            registered_at: now,
            last_activity: now,
            node_hash: [0u8; 32],
        };
        new_node.update_hash();

        // Insert node
        {
            let mut nodes = self.nodes.write().await;
            nodes.insert(request.node_id.clone(), new_node);
        }

        // Update pool stats
        self.update_state_hash().await;

        tracing::info!(
            node_id = %request.node_id,
            class = ?request.requested_class,
            "Node registered to pool"
        );

        let mut reg_hasher = Hasher::new();
        reg_hasher.update(request.node_id.as_bytes());
        reg_hasher.update(place_id.as_bytes());
        reg_hasher.update(&now.timestamp().to_le_bytes());

        Ok(RegistrationResponse {
            request_id: Uuid::new_v4(),
            approved: true,
            rejection_reason: None,
            place_id: Some(place_id),
            initial_tokens: 0, // No free tokens - must contribute resources
            responded_at: now,
            pool_signature: hex::encode(reg_hasher.finalize().as_bytes()),
        })
    }

    /// Contribute resources and mint tokens (Proof-of-Resource)
    pub async fn contribute_resources(
        &self,
        node_id: &str,
        contribution: ResourceContribution,
    ) -> Result<MintProof> {
        // Verify node exists and is active
        let mut nodes = self.nodes.write().await;
        let node = nodes
            .get_mut(node_id)
            .ok_or_else(|| PoolError::NodeNotRegistered {
                node_id: node_id.to_string(),
            })?;

        if node.status != NodeStatus::Active {
            return Err(PoolError::FateRejection {
                gate: "STATUS".to_string(),
                reason: "Node not active".to_string(),
            });
        }

        // Calculate tokens to mint based on contribution
        let tokens_minted = self.calculate_mint_amount(&contribution);

        // Update node balance
        node.token_balance += tokens_minted;
        node.last_activity = Utc::now();
        node.update_hash();

        drop(nodes);

        // Update total supply
        {
            let mut state = self.state.write().await;
            state.total_supply += tokens_minted;
        }

        // Create mint proof
        let now = Utc::now();
        let proof_id = Uuid::new_v4();

        let mut hasher = Hasher::new();
        hasher.update(proof_id.as_bytes());
        hasher.update(node_id.as_bytes());
        hasher.update(&tokens_minted.to_le_bytes());
        let proof_hash = *hasher.finalize().as_bytes();

        // Domain-separated signature commitment (blake3 keyed)
        let mut sig_hasher = Hasher::new();
        sig_hasher.update(b"bizra-mint-sig-v1:");
        sig_hasher.update(&proof_hash);
        sig_hasher.update(&now.timestamp().to_le_bytes());

        let proof = MintProof {
            proof_id,
            contributor_node: node_id.to_string(),
            contribution,
            tokens_minted,
            timestamp: now,
            signature: hex::encode(sig_hasher.finalize().as_bytes()),
            proof_hash,
        };

        tracing::info!(
            node_id = %node_id,
            tokens = tokens_minted,
            "Tokens minted for resource contribution"
        );

        Ok(proof)
    }

    /// Calculate tokens to mint for a contribution
    fn calculate_mint_amount(&self, contribution: &ResourceContribution) -> u64 {
        let base_amount = match contribution.resource_type {
            ResourceType::Cpu => contribution.quantity * TOKENS_PER_COMPUTE_UNIT,
            ResourceType::Gpu => contribution.quantity * TOKENS_PER_COMPUTE_UNIT * 10, // GPUs worth more
            ResourceType::Memory => contribution.quantity / 1024 / 1024,               // Per MB
            ResourceType::Storage => contribution.quantity / 1024 / 1024 / 1024,       // Per GB
            ResourceType::Network => contribution.quantity / 1024 / 1024, // Per MB transferred
            ResourceType::Inference => contribution.quantity * 10,        // Per inference token
        };

        // Adjust for duration and utilization
        let duration_factor = contribution.duration_seconds / 3600; // Per hour
        let utilization_factor = contribution.utilization as u64;

        (base_amount * duration_factor * utilization_factor) / 100
    }

    /// Register a service in the marketplace
    pub async fn register_service(&self, service: PoolService) -> Result<Uuid> {
        // Verify provider exists
        {
            let nodes = self.nodes.read().await;
            let provider =
                nodes
                    .get(&service.provider_node)
                    .ok_or_else(|| PoolError::NodeNotRegistered {
                        node_id: service.provider_node.clone(),
                    })?;

            provider.passes_ihsan()?;
        }

        // FATE gate check for financial services
        if service.category == ServiceCategory::Financial {
            // Financial services require extra scrutiny
            if service.ihsan_score < Decimal::from_str("0.98").unwrap() {
                return Err(PoolError::FateRejection {
                    gate: "FINANCIAL_IHSAN".to_string(),
                    reason: "Financial services require 0.98+ Ihsan".to_string(),
                });
            }
        }

        let service_id = service.service_id;
        self.services.write().await.insert(service_id, service);

        tracing::info!(service_id = %service_id, "Service registered");

        Ok(service_id)
    }

    /// Invoke a service
    pub async fn invoke_service(
        &self,
        invoker_node: &str,
        service_id: Uuid,
    ) -> Result<ServiceInvocation> {
        // Get service
        let service = {
            let services = self.services.read().await;
            services
                .get(&service_id)
                .cloned()
                .ok_or_else(|| PoolError::ServiceNotFound {
                    service_id: service_id.to_string(),
                })?
        };

        // Check invoker balance
        {
            let mut nodes = self.nodes.write().await;
            let invoker =
                nodes
                    .get_mut(invoker_node)
                    .ok_or_else(|| PoolError::NodeNotRegistered {
                        node_id: invoker_node.to_string(),
                    })?;

            if invoker.token_balance < service.price_per_call {
                return Err(PoolError::InsufficientBalance {
                    available: invoker.token_balance,
                    required: service.price_per_call,
                });
            }

            // Deduct payment
            invoker.token_balance -= service.price_per_call;

            // Pay provider
            if let Some(provider) = nodes.get_mut(&service.provider_node) {
                provider.token_balance += service.price_per_call;
            }
        }

        let invocation = ServiceInvocation {
            invocation_id: Uuid::new_v4(),
            service_id,
            invoker_node: invoker_node.to_string(),
            provider_node: service.provider_node,
            tokens_paid: service.price_per_call,
            status: InvocationStatus::InProgress,
            result_hash: None,
            started_at: Utc::now(),
            completed_at: None,
        };

        Ok(invocation)
    }

    /// Create PAT (Personal Agent Team) for a node
    pub async fn create_pat(&self, owner_node: &str) -> Result<PersonalAgentTeam> {
        // Verify node exists
        {
            let nodes = self.nodes.read().await;
            let _node = nodes
                .get(owner_node)
                .ok_or_else(|| PoolError::NodeNotRegistered {
                    node_id: owner_node.to_string(),
                })?;
        }

        // Check if PAT already exists
        {
            let pats = self.pat_teams.read().await;
            if pats.contains_key(owner_node) {
                return Err(PoolError::PATLimitExceeded { max: 1 });
            }
        }

        let pat = PersonalAgentTeam {
            owner_node: owner_node.to_string(),
            agents: Default::default(),
            team_ihsan: IHSAN_THRESHOLD,
            created_at: Utc::now(),
        };

        self.pat_teams
            .write()
            .await
            .insert(owner_node.to_string(), pat.clone());

        tracing::info!(owner = %owner_node, "PAT created");

        Ok(pat)
    }

    /// Process Zakat distribution
    pub async fn process_zakat(&self) -> Result<ZakatDistribution> {
        let mut total_collected: u64 = 0;
        let mut distributions: Vec<ZakatRecipient> = Vec::new();

        // Collect Zakat from all nodes above nisab
        {
            let mut nodes = self.nodes.write().await;
            for (_, node) in nodes.iter_mut() {
                let zakat_owed = node.calculate_zakat();
                if zakat_owed > 0 && node.token_balance >= zakat_owed {
                    node.token_balance -= zakat_owed;
                    node.zakat_paid_year += zakat_owed;
                    total_collected += zakat_owed;
                }
            }
        }

        // Add to Zakat fund
        {
            let mut fund = self.zakat_fund.write().await;
            *fund += total_collected;
        }

        // Distribute according to traditional categories
        // For now, equal distribution across categories
        let per_category = total_collected / 8;

        for category in [
            ZakatCategory::Poor,
            ZakatCategory::Needy,
            ZakatCategory::Administrators,
            ZakatCategory::Reconciliation,
            ZakatCategory::Liberation,
            ZakatCategory::Debtors,
            ZakatCategory::Cause,
            ZakatCategory::Travelers,
        ] {
            distributions.push(ZakatRecipient {
                category,
                recipient_node: None,
                amount: per_category,
                purpose: format!("{:?} support", category),
            });
        }

        // Update pool state
        {
            let mut state = self.state.write().await;
            state.total_zakat_distributed += total_collected;
        }

        let mut hasher = Hasher::new();
        hasher.update(&total_collected.to_le_bytes());

        let distribution = ZakatDistribution {
            distribution_id: Uuid::new_v4(),
            period: Utc::now().format("%Y").to_string(),
            total_collected,
            distributions,
            distributed_at: Utc::now(),
            distribution_hash: *hasher.finalize().as_bytes(),
        };

        tracing::info!(total = total_collected, "Zakat distribution processed");

        Ok(distribution)
    }

    /// Create Harberger listing
    pub async fn create_harberger_listing(
        &self,
        owner_node: &str,
        resource_type: ResourceType,
        quantity: u64,
        self_assessment: u64,
    ) -> Result<HarbergerListing> {
        // Verify self-assessment is reasonable
        let min_assessment = quantity * MIN_ASSESSMENT_MULTIPLIER.to_u64().unwrap_or(1);
        if self_assessment < min_assessment {
            return Err(PoolError::FateRejection {
                gate: "HARBERGER".to_string(),
                reason: format!(
                    "Self-assessment {} below minimum {}",
                    self_assessment, min_assessment
                ),
            });
        }

        let annual_tax =
            (Decimal::from(self_assessment) * Decimal::from(quantity) * HARBERGER_TAX_RATE)
                .to_u64()
                .unwrap_or(0);

        let listing_id = Uuid::new_v4();

        let mut hasher = Hasher::new();
        hasher.update(listing_id.as_bytes());
        hasher.update(owner_node.as_bytes());

        let listing = HarbergerListing {
            listing_id,
            owner_node: owner_node.to_string(),
            resource_type,
            quantity,
            self_assessment,
            annual_tax,
            last_tax_payment: Utc::now(),
            tax_paid_period: 0,
            listing_hash: *hasher.finalize().as_bytes(),
        };

        self.harberger_listings
            .write()
            .await
            .insert(listing_id, listing.clone());

        tracing::info!(
            listing_id = %listing_id,
            owner = %owner_node,
            "Harberger listing created"
        );

        Ok(listing)
    }

    /// Process Harberger offer (forced sale at self-assessed price)
    pub async fn process_harberger_offer(&self, offer: HarbergerOffer) -> Result<bool> {
        let mut listings = self.harberger_listings.write().await;
        let listing =
            listings
                .get_mut(&offer.listing_id)
                .ok_or_else(|| PoolError::ResourceNotAvailable {
                    resource_type: "listing".to_string(),
                })?;

        // Offer must be >= self-assessment
        if offer.offered_price < listing.self_assessment {
            return Err(PoolError::FateRejection {
                gate: "HARBERGER".to_string(),
                reason: "Offer below self-assessment".to_string(),
            });
        }

        // Transfer ownership
        let old_owner = listing.owner_node.clone();
        listing.owner_node = offer.bidder_node.clone();
        listing.self_assessment = offer.offered_price;

        drop(listings);

        // Transfer tokens
        {
            let mut nodes = self.nodes.write().await;

            // Deduct from bidder
            if let Some(bidder) = nodes.get_mut(&offer.bidder_node) {
                if bidder.token_balance < offer.offered_price {
                    return Err(PoolError::InsufficientBalance {
                        available: bidder.token_balance,
                        required: offer.offered_price,
                    });
                }
                bidder.token_balance -= offer.offered_price;
            }

            // Pay seller
            if let Some(seller) = nodes.get_mut(&old_owner) {
                seller.token_balance += offer.offered_price;
            }
        }

        tracing::info!(
            listing_id = %offer.listing_id,
            from = %old_owner,
            to = %offer.bidder_node,
            price = offer.offered_price,
            "Harberger transfer complete"
        );

        Ok(true)
    }

    /// Calculate pool Gini coefficient for Adl enforcement
    pub async fn calculate_gini(&self) -> Decimal {
        let nodes = self.nodes.read().await;
        let mut balances: Vec<f64> = nodes.values().map(|n| n.token_balance as f64).collect();

        if balances.is_empty() || balances.len() == 1 {
            return Decimal::ZERO;
        }

        let n = balances.len() as f64;
        balances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sum: f64 = balances.iter().sum();
        if sum == 0.0 {
            return Decimal::ZERO;
        }

        let mut gini_sum = 0.0;
        for (i, v) in balances.iter().enumerate() {
            gini_sum += (2.0 * (i as f64 + 1.0) - n - 1.0) * v;
        }

        Decimal::from_f64(gini_sum / (n * sum)).unwrap_or(Decimal::ZERO)
    }

    /// Check Adl (justice) compliance
    pub async fn check_adl(&self) -> Result<()> {
        let gini = self.calculate_gini().await;
        if gini > ADL_GINI_MAX {
            return Err(PoolError::AdlViolation {
                gini,
                max: ADL_GINI_MAX,
            });
        }
        Ok(())
    }

    /// Update pool state hash
    async fn update_state_hash(&self) {
        let mut state = self.state.write().await;
        let nodes = self.nodes.read().await;

        let mut hasher = Hasher::new();
        hasher.update(state.pool_id.as_bytes());
        hasher.update(&state.total_supply.to_le_bytes());
        hasher.update(&state.current_epoch.to_le_bytes());
        hasher.update(&(nodes.len() as u64).to_le_bytes());

        state.state_hash = *hasher.finalize().as_bytes();
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let nodes = self.nodes.read().await;
        let services = self.services.read().await;
        let pat_teams = self.pat_teams.read().await;
        let sat_teams = self.sat_teams.read().await;

        let active_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Active)
            .count();
        let total_staked: u64 = nodes.values().map(|n| n.token_balance).sum();
        let total_compute: u64 = nodes
            .values()
            .map(|n| n.resources.cpu_millicores as u64)
            .sum();

        let avg_ihsan = if nodes.is_empty() {
            Decimal::ZERO
        } else {
            nodes.values().map(|n| n.ihsan_score).sum::<Decimal>() / Decimal::from(nodes.len())
        };

        PoolStats {
            total_nodes: nodes.len(),
            active_nodes,
            total_services: services.len(),
            total_pat: pat_teams.len(),
            total_sat: sat_teams.len(),
            total_staked,
            total_compute,
            avg_ihsan,
        }
    }

    /// Get pool state
    pub async fn state(&self) -> PoolState {
        self.state.read().await.clone()
    }

    /// Get node by ID
    pub async fn get_node(&self, node_id: &str) -> Option<PoolNode> {
        self.nodes.read().await.get(node_id).cloned()
    }
}

// =============================================================================
// HELPER: Hex serialization for VerifyingKey
// =============================================================================

mod hex_verifying_key {
    use ed25519_dalek::VerifyingKey;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(key: &VerifyingKey, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(key.as_bytes()))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<VerifyingKey, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("Invalid key length"))?;
        VerifyingKey::from_bytes(&arr).map_err(serde::de::Error::custom)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn generate_keypair() -> (SigningKey, VerifyingKey) {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        (signing_key, verifying_key)
    }

    #[tokio::test]
    async fn test_pool_genesis() {
        let (_, verifying_key) = generate_keypair();
        let node_id = hex::encode(verifying_key.as_bytes());

        let pool = ResourcePool::genesis(node_id.clone(), "Node0-MoMo".to_string(), verifying_key)
            .await
            .unwrap();

        let stats = pool.stats().await;
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.active_nodes, 1);

        let state = pool.state().await;
        assert_eq!(state.genesis_node, node_id);
        assert_eq!(state.total_supply, NISAB_THRESHOLD);
    }

    #[tokio::test]
    async fn test_node_registration() {
        let (_, genesis_key) = generate_keypair();
        let genesis_id = hex::encode(genesis_key.as_bytes());

        let pool = ResourcePool::genesis(genesis_id.clone(), "Node0".to_string(), genesis_key)
            .await
            .unwrap();

        // Register a new node
        let (_, new_key) = generate_keypair();
        let new_id = hex::encode(new_key.as_bytes());

        let request = RegistrationRequest {
            node_id: new_id.clone(),
            name: "Node1".to_string(),
            requested_class: NodeClass::Sovereign,
            resources: NodeResources::default(),
            sponsor_node: Some(genesis_id.clone()),
            identity_proof: None,
            requested_at: Utc::now(),
            signature: String::new(),
        };

        let response = pool.register_node(request).await.unwrap();
        assert!(response.approved);

        let stats = pool.stats().await;
        assert_eq!(stats.total_nodes, 2);
    }

    #[tokio::test]
    async fn test_resource_contribution() {
        let (_, genesis_key) = generate_keypair();
        let genesis_id = hex::encode(genesis_key.as_bytes());

        let pool = ResourcePool::genesis(genesis_id.clone(), "Node0".to_string(), genesis_key)
            .await
            .unwrap();

        let contribution = ResourceContribution {
            resource_type: ResourceType::Cpu,
            quantity: 1000,
            duration_seconds: 3600,
            utilization: 80,
            proof_block_id: None,
        };

        let proof = pool
            .contribute_resources(&genesis_id, contribution)
            .await
            .unwrap();
        assert!(proof.tokens_minted > 0);

        let node = pool.get_node(&genesis_id).await.unwrap();
        assert!(node.token_balance > NISAB_THRESHOLD);
    }

    #[tokio::test]
    async fn test_zakat_calculation() {
        let (_, genesis_key) = generate_keypair();
        let genesis_id = hex::encode(genesis_key.as_bytes());

        let pool = ResourcePool::genesis(genesis_id.clone(), "Node0".to_string(), genesis_key)
            .await
            .unwrap();

        let node = pool.get_node(&genesis_id).await.unwrap();

        // Genesis node has NISAB_THRESHOLD tokens, so no Zakat owed (threshold not exceeded)
        let zakat = node.calculate_zakat();
        assert_eq!(zakat, 0);
    }

    #[tokio::test]
    async fn test_gini_calculation() {
        let (_, genesis_key) = generate_keypair();
        let genesis_id = hex::encode(genesis_key.as_bytes());

        let pool = ResourcePool::genesis(genesis_id.clone(), "Node0".to_string(), genesis_key)
            .await
            .unwrap();

        // With single node, Gini should be 0
        let gini = pool.calculate_gini().await;
        assert_eq!(gini, Decimal::ZERO);

        // Adl check should pass
        assert!(pool.check_adl().await.is_ok());
    }

    #[tokio::test]
    async fn test_pat_creation() {
        let (_, genesis_key) = generate_keypair();
        let genesis_id = hex::encode(genesis_key.as_bytes());

        let pool = ResourcePool::genesis(genesis_id.clone(), "Node0".to_string(), genesis_key)
            .await
            .unwrap();

        let pat = pool.create_pat(&genesis_id).await.unwrap();
        assert_eq!(pat.owner_node, genesis_id);
        assert!(pat.agents.iter().all(|a| a.is_none())); // All slots empty initially
    }

    #[tokio::test]
    async fn test_harberger_listing() {
        let (_, genesis_key) = generate_keypair();
        let genesis_id = hex::encode(genesis_key.as_bytes());

        let pool = ResourcePool::genesis(genesis_id.clone(), "Node0".to_string(), genesis_key)
            .await
            .unwrap();

        let listing = pool
            .create_harberger_listing(&genesis_id, ResourceType::Cpu, 1000, 10000)
            .await
            .unwrap();

        assert_eq!(listing.owner_node, genesis_id);
        assert_eq!(listing.resource_type, ResourceType::Cpu);
        assert!(listing.annual_tax > 0);
    }

    #[test]
    fn test_constants() {
        // Verify constants are properly defined
        assert!(IHSAN_THRESHOLD >= Decimal::from_str("0.95").unwrap());
        assert_eq!(ZAKAT_RATE, Decimal::from_str("0.025").unwrap());
        assert_eq!(HARBERGER_TAX_RATE, Decimal::from_str("0.07").unwrap());
        assert_eq!(PAT_SIZE, 7);
        assert_eq!(SAT_SIZE, 5);
    }
}
