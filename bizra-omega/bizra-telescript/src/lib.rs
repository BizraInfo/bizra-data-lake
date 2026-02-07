//! TELESCRIPT-BIZRA BRIDGE v0.1.0
//!
//! A fusion of General Magic's Telescript (1990s mobile agent technology)
//! with BIZRA's sovereign ethics framework.
//!
//! # Standing on Giants
//!
//! - **General Magic (1990-1994)**: Telescript primitives - Places, Agents, Permits
//! - **Shannon (1948)**: SNR-based signal quality in agent communication
//! - **Lamport (1982)**: Byzantine fault tolerance in agent consensus
//! - **Al-Ghazali (1095)**: Maqasid al-Shariah → FATE gate ethics
//! - **Anthropic (2023)**: Constitutional AI → Ihsān threshold (≥0.95)
//!
//! # The 9 Primitive Types
//!
//! 1. `Authority` - Who granted the permit (chain of delegation)
//! 2. `Permit` - What capabilities are allowed
//! 3. `Place` - Where agents can go (hosts, services)
//! 4. `Agent` - The mobile code unit itself
//! 5. `AgentState` - Lifecycle: Created, Traveling, Meeting, Frozen, Terminated
//! 6. `Ticket` - Proof of transit rights
//! 7. `Value` - Economic units for resource accounting
//! 8. `Meeting` - Synchronous rendezvous between agents
//! 9. `Connection` - Async channel for agent communication
//!
//! # FATE Gate Enforcement
//!
//! Every operation passes through FATE (Fairness, Accountability, Transparency, Ethics):
//! - Ihsān threshold: 950/1000 minimum
//! - Adl (justice): Gini coefficient ≤ 0.35
//! - No assumptions: لا نفترض

use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

// =============================================================================
// CONSTANTS - Single Source of Truth
// =============================================================================

/// Ihsān threshold: 950/1000 = 0.95 excellence constraint
pub const IHSAN_THRESHOLD: u32 = 950;

/// Maximum Gini coefficient for Adl (justice) enforcement
pub const ADL_GINI_MAX: f64 = 0.35;

/// SNR minimum for signal quality
pub const SNR_MINIMUM: f64 = 0.85;

/// Maximum permit delegation depth
pub const MAX_DELEGATION_DEPTH: u8 = 7;

// =============================================================================
// ERROR TYPES
// =============================================================================

#[derive(Error, Debug)]
pub enum TelescriptError {
    #[error("FATE gate rejected: {0}")]
    FateRejection(String),

    #[error("Ihsān threshold not met: {score}/1000 < {threshold}/1000")]
    IhsanViolation { score: u32, threshold: u32 },

    #[error("Adl violation: Gini {gini:.3} > {max:.3}")]
    AdlViolation { gini: f64, max: f64 },

    #[error("Permit denied: {0}")]
    PermitDenied(String),

    #[error("Place not found: {0}")]
    PlaceNotFound(String),

    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition { from: AgentState, to: AgentState },

    #[error("Delegation depth exceeded: {depth} > {max}")]
    DelegationDepthExceeded { depth: u8, max: u8 },

    #[error("Meeting timeout: {0}")]
    MeetingTimeout(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),
}

pub type Result<T> = std::result::Result<T, TelescriptError>;

// =============================================================================
// PRIMITIVE TYPE 1: AUTHORITY
// =============================================================================

/// Authority represents the chain of delegation for permits.
/// Every permit traces back to the Genesis Authority (Node0).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Authority {
    /// Unique identifier for this authority
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// The authority that delegated to this one (None = Genesis)
    pub delegated_from: Option<Box<Authority>>,
    /// Depth in the delegation chain (0 = Genesis)
    pub delegation_depth: u8,
    /// Blake3 hash of the authority chain
    pub chain_hash: [u8; 32],
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl Authority {
    /// Create the Genesis Authority (Node0 - root of all trust)
    pub fn genesis() -> Self {
        let id = Uuid::new_v4();
        let mut hasher = Hasher::new();
        hasher.update(b"BIZRA_GENESIS_AUTHORITY_NODE0");
        hasher.update(id.as_bytes());

        Authority {
            id,
            name: "Node0-Genesis".to_string(),
            delegated_from: None,
            delegation_depth: 0,
            chain_hash: *hasher.finalize().as_bytes(),
            created_at: Utc::now(),
        }
    }

    /// Delegate authority to a new entity
    pub fn delegate(&self, name: &str) -> Result<Authority> {
        let new_depth = self.delegation_depth + 1;
        if new_depth > MAX_DELEGATION_DEPTH {
            return Err(TelescriptError::DelegationDepthExceeded {
                depth: new_depth,
                max: MAX_DELEGATION_DEPTH,
            });
        }

        let id = Uuid::new_v4();
        let mut hasher = Hasher::new();
        hasher.update(&self.chain_hash);
        hasher.update(id.as_bytes());
        hasher.update(name.as_bytes());

        Ok(Authority {
            id,
            name: name.to_string(),
            delegated_from: Some(Box::new(self.clone())),
            delegation_depth: new_depth,
            chain_hash: *hasher.finalize().as_bytes(),
            created_at: Utc::now(),
        })
    }

    /// Verify the chain of trust back to Genesis
    pub fn verify_chain(&self) -> bool {
        let mut hasher = Hasher::new();

        match &self.delegated_from {
            None => {
                // Genesis authority - verify it's properly formed
                hasher.update(b"BIZRA_GENESIS_AUTHORITY_NODE0");
                hasher.update(self.id.as_bytes());
                self.chain_hash == *hasher.finalize().as_bytes()
            }
            Some(parent) => {
                // Delegated authority - verify parent and chain
                if !parent.verify_chain() {
                    return false;
                }
                hasher.update(&parent.chain_hash);
                hasher.update(self.id.as_bytes());
                hasher.update(self.name.as_bytes());
                self.chain_hash == *hasher.finalize().as_bytes()
            }
        }
    }
}

// =============================================================================
// PRIMITIVE TYPE 2: PERMIT
// =============================================================================

/// Capabilities that can be granted to agents
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Travel to other places
    Go,
    /// Create new agents
    Create,
    /// Participate in meetings
    Meet,
    /// Access computational resources
    Compute,
    /// Access storage resources
    Store,
    /// Access network resources
    Network,
    /// Access inference tier (EDGE/LOCAL/POOL)
    Inference,
    /// Modify own state
    SelfModify,
    /// Delegate permits to sub-agents
    Delegate,
}

/// Resource limits for permits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU units (1000 = 1 core)
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
}

impl Default for ResourceLimits {
    fn default() -> Self {
        ResourceLimits {
            cpu_millicores: 100,              // 0.1 cores
            memory_bytes: 64 * 1024 * 1024,   // 64 MB
            storage_bytes: 256 * 1024 * 1024, // 256 MB
            network_bps: 1024 * 1024,         // 1 MB/s
            inference_tokens: 4096,           // Standard context
            ttl_seconds: 3600,                // 1 hour
        }
    }
}

/// Permit defines what an agent is allowed to do
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permit {
    /// Unique identifier
    pub id: Uuid,
    /// The authority that issued this permit
    pub issuer: Authority,
    /// Granted capabilities
    pub capabilities: Vec<Capability>,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Places this permit is valid for (empty = all)
    pub valid_places: Vec<Uuid>,
    /// Ihsān score requirement for this permit
    pub ihsan_requirement: u32,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,
    /// Blake3 hash of permit contents
    pub permit_hash: [u8; 32],
}

impl Permit {
    /// Create a new permit from an authority
    pub fn new(
        issuer: Authority,
        capabilities: Vec<Capability>,
        limits: ResourceLimits,
        ttl_seconds: u64,
    ) -> Self {
        let id = Uuid::new_v4();
        let created_at = Utc::now();
        let expires_at = created_at + chrono::Duration::seconds(ttl_seconds as i64);

        let mut hasher = Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(&issuer.chain_hash);
        for cap in &capabilities {
            hasher.update(&[*cap as u8]);
        }
        hasher.update(&limits.ttl_seconds.to_le_bytes());

        Permit {
            id,
            issuer,
            capabilities,
            limits,
            valid_places: Vec::new(),
            ihsan_requirement: IHSAN_THRESHOLD,
            created_at,
            expires_at,
            permit_hash: *hasher.finalize().as_bytes(),
        }
    }

    /// Check if permit grants a capability
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Check if permit is valid for a place
    pub fn valid_for_place(&self, place_id: &Uuid) -> bool {
        self.valid_places.is_empty() || self.valid_places.contains(place_id)
    }

    /// Check if permit has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Verify permit integrity
    pub fn verify(&self) -> bool {
        if !self.issuer.verify_chain() {
            return false;
        }
        if self.is_expired() {
            return false;
        }

        let mut hasher = Hasher::new();
        hasher.update(self.id.as_bytes());
        hasher.update(&self.issuer.chain_hash);
        for cap in &self.capabilities {
            hasher.update(&[*cap as u8]);
        }
        hasher.update(&self.limits.ttl_seconds.to_le_bytes());

        self.permit_hash == *hasher.finalize().as_bytes()
    }
}

// =============================================================================
// PRIMITIVE TYPE 3: PLACE
// =============================================================================

/// Place represents a location where agents can exist and operate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Place {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name (Telename)
    pub telename: String,
    /// Network address (if remote)
    pub address: Option<String>,
    /// Capabilities available at this place
    pub available_capabilities: Vec<Capability>,
    /// Current Ihsān score of this place
    pub ihsan_score: u32,
    /// Gini coefficient (resource distribution fairness)
    pub gini_coefficient: f64,
    /// Maximum agents allowed
    pub max_agents: usize,
    /// Current agent count
    pub current_agents: usize,
    /// Blake3 hash for integrity
    pub place_hash: [u8; 32],
}

impl Place {
    /// Create a new place
    pub fn new(telename: &str, address: Option<String>) -> Self {
        let id = Uuid::new_v4();

        let mut hasher = Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(telename.as_bytes());
        if let Some(ref addr) = address {
            hasher.update(addr.as_bytes());
        }

        Place {
            id,
            telename: telename.to_string(),
            address,
            available_capabilities: vec![
                Capability::Go,
                Capability::Meet,
                Capability::Compute,
                Capability::Store,
            ],
            ihsan_score: IHSAN_THRESHOLD,
            gini_coefficient: 0.25, // Default fair distribution
            max_agents: 1000,
            current_agents: 0,
            place_hash: *hasher.finalize().as_bytes(),
        }
    }

    /// Check if place passes FATE gates
    pub fn passes_fate(&self) -> Result<()> {
        if self.ihsan_score < IHSAN_THRESHOLD {
            return Err(TelescriptError::IhsanViolation {
                score: self.ihsan_score,
                threshold: IHSAN_THRESHOLD,
            });
        }
        if self.gini_coefficient > ADL_GINI_MAX {
            return Err(TelescriptError::AdlViolation {
                gini: self.gini_coefficient,
                max: ADL_GINI_MAX,
            });
        }
        Ok(())
    }

    /// Check if place can accept an agent
    pub fn can_accept(&self, permit: &Permit) -> Result<()> {
        self.passes_fate()?;

        if self.current_agents >= self.max_agents {
            return Err(TelescriptError::FateRejection(
                "Place at capacity".to_string(),
            ));
        }

        if !permit.valid_for_place(&self.id) {
            return Err(TelescriptError::PermitDenied(format!(
                "Permit not valid for place {}",
                self.telename
            )));
        }

        // Check if place provides required capabilities
        for cap in &permit.capabilities {
            if !self.available_capabilities.contains(cap) {
                return Err(TelescriptError::PermitDenied(format!(
                    "Place {} does not provide capability {:?}",
                    self.telename, cap
                )));
            }
        }

        Ok(())
    }
}

// =============================================================================
// PRIMITIVE TYPE 4 & 5: AGENT & AGENT STATE
// =============================================================================

/// Agent lifecycle states
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentState {
    /// Just created, not yet active
    Created,
    /// Active and running at a place
    Active,
    /// Traveling between places (code in transit)
    Traveling,
    /// In a meeting with another agent
    Meeting,
    /// Frozen (paused execution)
    Frozen,
    /// Terminated (cannot be reactivated)
    Terminated,
}

/// The mobile agent itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Current state
    pub state: AgentState,
    /// The permit governing this agent
    pub permit: Permit,
    /// Current place (None if traveling)
    pub current_place: Option<Uuid>,
    /// Agent code/logic (serialized)
    pub code: Vec<u8>,
    /// Agent data/context (serialized)
    pub data: HashMap<String, Vec<u8>>,
    /// Ihsān score accumulated by this agent
    pub ihsan_score: u32,
    /// Resource usage tracking
    pub resource_usage: ResourceUsage,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Blake3 hash of agent state
    pub agent_hash: [u8; 32],
}

/// Track agent resource consumption
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_used: u64,
    pub memory_used: u64,
    pub storage_used: u64,
    pub network_used: u64,
    pub inference_tokens_used: u64,
}

impl Agent {
    /// Create a new agent
    pub fn new(name: &str, permit: Permit, code: Vec<u8>) -> Self {
        let id = Uuid::new_v4();
        let now = Utc::now();

        let mut hasher = Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&permit.permit_hash);
        hasher.update(&code);

        Agent {
            id,
            name: name.to_string(),
            state: AgentState::Created,
            permit,
            current_place: None,
            code,
            data: HashMap::new(),
            ihsan_score: IHSAN_THRESHOLD,
            resource_usage: ResourceUsage::default(),
            created_at: now,
            last_activity: now,
            agent_hash: *hasher.finalize().as_bytes(),
        }
    }

    /// Transition agent state
    pub fn transition(&mut self, new_state: AgentState) -> Result<()> {
        let valid = matches!(
            (self.state, new_state),
            (AgentState::Created, AgentState::Active)
                | (AgentState::Active, AgentState::Traveling)
                | (AgentState::Active, AgentState::Meeting)
                | (AgentState::Active, AgentState::Frozen)
                | (AgentState::Active, AgentState::Terminated)
                | (AgentState::Traveling, AgentState::Active)
                | (AgentState::Traveling, AgentState::Terminated)
                | (AgentState::Meeting, AgentState::Active)
                | (AgentState::Meeting, AgentState::Terminated)
                | (AgentState::Frozen, AgentState::Active)
                | (AgentState::Frozen, AgentState::Terminated)
        );

        if !valid {
            return Err(TelescriptError::InvalidStateTransition {
                from: self.state,
                to: new_state,
            });
        }

        self.state = new_state;
        self.last_activity = Utc::now();
        self.update_hash();
        Ok(())
    }

    /// Update the agent hash after state changes
    fn update_hash(&mut self) {
        let mut hasher = Hasher::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.name.as_bytes());
        hasher.update(&self.permit.permit_hash);
        hasher.update(&self.code);
        hasher.update(&[self.state as u8]);
        if let Some(place_id) = &self.current_place {
            hasher.update(place_id.as_bytes());
        }
        self.agent_hash = *hasher.finalize().as_bytes();
    }

    /// Check if agent passes FATE gates
    pub fn passes_fate(&self) -> Result<()> {
        if self.ihsan_score < IHSAN_THRESHOLD {
            return Err(TelescriptError::IhsanViolation {
                score: self.ihsan_score,
                threshold: IHSAN_THRESHOLD,
            });
        }
        if !self.permit.verify() {
            return Err(TelescriptError::PermitDenied("Invalid permit".to_string()));
        }
        Ok(())
    }
}

// =============================================================================
// PRIMITIVE TYPE 6: TICKET
// =============================================================================

/// Ticket authorizes travel between places
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticket {
    /// Unique identifier
    pub id: Uuid,
    /// Agent this ticket is for
    pub agent_id: Uuid,
    /// Origin place
    pub from_place: Uuid,
    /// Destination place
    pub to_place: Uuid,
    /// Departure time
    pub departure_at: DateTime<Utc>,
    /// Expiration time
    pub expires_at: DateTime<Utc>,
    /// Blake3 signature of ticket
    pub ticket_hash: [u8; 32],
}

impl Ticket {
    /// Issue a new ticket
    pub fn issue(
        agent: &Agent,
        from_place: Uuid,
        to_place: Uuid,
        valid_seconds: u64,
    ) -> Result<Self> {
        // Check if agent has Go capability
        if !agent.permit.has_capability(Capability::Go) {
            return Err(TelescriptError::PermitDenied(
                "Agent lacks Go capability".to_string(),
            ));
        }

        let id = Uuid::new_v4();
        let now = Utc::now();
        let expires_at = now + chrono::Duration::seconds(valid_seconds as i64);

        let mut hasher = Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(agent.id.as_bytes());
        hasher.update(from_place.as_bytes());
        hasher.update(to_place.as_bytes());

        Ok(Ticket {
            id,
            agent_id: agent.id,
            from_place,
            to_place,
            departure_at: now,
            expires_at,
            ticket_hash: *hasher.finalize().as_bytes(),
        })
    }

    /// Verify ticket authenticity
    pub fn verify(&self) -> bool {
        if Utc::now() > self.expires_at {
            return false;
        }

        let mut hasher = Hasher::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.agent_id.as_bytes());
        hasher.update(self.from_place.as_bytes());
        hasher.update(self.to_place.as_bytes());

        self.ticket_hash == *hasher.finalize().as_bytes()
    }
}

// =============================================================================
// PRIMITIVE TYPE 7: VALUE
// =============================================================================

/// Value represents economic units for resource accounting
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Value {
    /// Amount in smallest denomination (like satoshis)
    pub amount: u64,
    /// Value type
    pub value_type: ValueType,
}

/// Types of value in the BIZRA economy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValueType {
    /// Compute credits
    Compute,
    /// Storage credits
    Storage,
    /// Network bandwidth credits
    Network,
    /// Inference tokens
    Inference,
    /// Reputation/trust score
    Reputation,
}

impl Value {
    pub fn new(amount: u64, value_type: ValueType) -> Self {
        Value { amount, value_type }
    }

    /// Transfer value (returns new Value for recipient)
    pub fn transfer(&mut self, amount: u64) -> Result<Value> {
        if amount > self.amount {
            return Err(TelescriptError::FateRejection(
                "Insufficient balance".to_string(),
            ));
        }
        self.amount -= amount;
        Ok(Value::new(amount, self.value_type))
    }
}

// =============================================================================
// PRIMITIVE TYPE 8: MEETING
// =============================================================================

/// Meeting represents synchronous rendezvous between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meeting {
    /// Unique identifier
    pub id: Uuid,
    /// Place where meeting occurs
    pub place_id: Uuid,
    /// Initiating agent
    pub initiator_id: Uuid,
    /// Responding agent
    pub responder_id: Uuid,
    /// Meeting state
    pub state: MeetingState,
    /// Data exchanged during meeting
    pub exchange: Vec<MeetingMessage>,
    /// Started at
    pub started_at: DateTime<Utc>,
    /// Ended at (if completed)
    pub ended_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeetingState {
    Pending,
    Active,
    Completed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingMessage {
    pub from_agent: Uuid,
    pub to_agent: Uuid,
    pub payload: Vec<u8>,
    pub timestamp: DateTime<Utc>,
}

impl Meeting {
    /// Start a new meeting request
    pub fn request(place: &Place, initiator: &Agent, responder_id: Uuid) -> Result<Self> {
        // Both agents need Meet capability
        if !initiator.permit.has_capability(Capability::Meet) {
            return Err(TelescriptError::PermitDenied(
                "Initiator lacks Meet capability".to_string(),
            ));
        }

        // Place must pass FATE
        place.passes_fate()?;

        Ok(Meeting {
            id: Uuid::new_v4(),
            place_id: place.id,
            initiator_id: initiator.id,
            responder_id,
            state: MeetingState::Pending,
            exchange: Vec::new(),
            started_at: Utc::now(),
            ended_at: None,
        })
    }

    /// Accept meeting (responder side)
    pub fn accept(&mut self, responder: &Agent) -> Result<()> {
        if responder.id != self.responder_id {
            return Err(TelescriptError::FateRejection(
                "Responder mismatch".to_string(),
            ));
        }
        if !responder.permit.has_capability(Capability::Meet) {
            return Err(TelescriptError::PermitDenied(
                "Responder lacks Meet capability".to_string(),
            ));
        }
        self.state = MeetingState::Active;
        Ok(())
    }

    /// Send message during meeting
    pub fn send(&mut self, from_agent: &Agent, payload: Vec<u8>) -> Result<()> {
        if self.state != MeetingState::Active {
            return Err(TelescriptError::FateRejection(
                "Meeting not active".to_string(),
            ));
        }

        let to_agent = if from_agent.id == self.initiator_id {
            self.responder_id
        } else if from_agent.id == self.responder_id {
            self.initiator_id
        } else {
            return Err(TelescriptError::FateRejection(
                "Agent not in meeting".to_string(),
            ));
        };

        self.exchange.push(MeetingMessage {
            from_agent: from_agent.id,
            to_agent,
            payload,
            timestamp: Utc::now(),
        });

        Ok(())
    }

    /// Complete the meeting
    pub fn complete(&mut self) {
        self.state = MeetingState::Completed;
        self.ended_at = Some(Utc::now());
    }
}

// =============================================================================
// PRIMITIVE TYPE 9: CONNECTION
// =============================================================================

/// Connection represents async channel for agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Unique identifier
    pub id: Uuid,
    /// Local agent
    pub local_agent: Uuid,
    /// Remote agent
    pub remote_agent: Uuid,
    /// Connection state
    pub state: ConnectionState,
    /// Quality metrics (SNR-based)
    pub quality: ConnectionQuality,
    /// Established at
    pub established_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    Degraded,
    Disconnected,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConnectionQuality {
    /// Signal-to-Noise Ratio (Shannon, 1948)
    pub snr: f64,
    /// Latency in milliseconds
    pub latency_ms: u32,
    /// Packets lost (0.0 - 1.0)
    pub packet_loss: f64,
    /// Last measured
    pub measured_at: DateTime<Utc>,
}

impl Connection {
    /// Establish a new connection
    pub fn establish(local_agent: &Agent, remote_agent_id: Uuid) -> Result<Self> {
        if !local_agent.permit.has_capability(Capability::Network) {
            return Err(TelescriptError::PermitDenied(
                "Agent lacks Network capability".to_string(),
            ));
        }

        Ok(Connection {
            id: Uuid::new_v4(),
            local_agent: local_agent.id,
            remote_agent: remote_agent_id,
            state: ConnectionState::Connecting,
            quality: ConnectionQuality {
                snr: 1.0,
                latency_ms: 0,
                packet_loss: 0.0,
                measured_at: Utc::now(),
            },
            established_at: Utc::now(),
        })
    }

    /// Check if connection quality passes SNR threshold
    pub fn passes_snr(&self) -> Result<()> {
        if self.quality.snr < SNR_MINIMUM {
            return Err(TelescriptError::ConnectionError(format!(
                "SNR {:.3} below minimum {:.3}",
                self.quality.snr, SNR_MINIMUM
            )));
        }
        Ok(())
    }
}

// =============================================================================
// TELESCRIPT ENGINE - The Runtime
// =============================================================================

/// The Telescript Engine manages the runtime environment for mobile agents
pub struct TelescriptEngine {
    /// Genesis authority (root of trust)
    genesis: Authority,
    /// Registered places
    places: Arc<RwLock<HashMap<Uuid, Place>>>,
    /// Active agents
    agents: Arc<RwLock<HashMap<Uuid, Agent>>>,
    /// Active meetings
    meetings: Arc<RwLock<HashMap<Uuid, Meeting>>>,
    /// Active connections
    connections: Arc<RwLock<HashMap<Uuid, Connection>>>,
    /// Proof of impact log
    impact_log: Arc<RwLock<Vec<ImpactAttestation>>>,
}

/// Immutable record of agent activity for Proof-of-Impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAttestation {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub action: String,
    pub impact_score: u32,
    pub timestamp: DateTime<Utc>,
    pub attestation_hash: [u8; 32],
}

impl TelescriptEngine {
    /// Create a new Telescript Engine with Genesis Authority
    pub fn new() -> Self {
        let genesis = Authority::genesis();

        // Create the Genesis Place (Node0)
        let genesis_place = Place::new("node0://genesis", None);

        let mut places = HashMap::new();
        places.insert(genesis_place.id, genesis_place);

        TelescriptEngine {
            genesis,
            places: Arc::new(RwLock::new(places)),
            agents: Arc::new(RwLock::new(HashMap::new())),
            meetings: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            impact_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get the Genesis Authority
    pub fn genesis_authority(&self) -> &Authority {
        &self.genesis
    }

    /// Create a new place
    pub async fn create_place(&self, telename: &str, address: Option<String>) -> Result<Uuid> {
        let place = Place::new(telename, address);
        place.passes_fate()?;

        let id = place.id;
        self.places.write().await.insert(id, place);

        Ok(id)
    }

    /// Create a new agent with FATE gate check
    pub async fn create_agent(
        &self,
        name: &str,
        permit: Permit,
        code: Vec<u8>,
        place_id: Uuid,
    ) -> Result<Uuid> {
        // Verify permit
        if !permit.verify() {
            return Err(TelescriptError::PermitDenied("Invalid permit".to_string()));
        }

        // Get place and check FATE
        let places = self.places.read().await;
        let place = places
            .get(&place_id)
            .ok_or_else(|| TelescriptError::PlaceNotFound(place_id.to_string()))?;

        place.can_accept(&permit)?;
        drop(places);

        // Create agent
        let mut agent = Agent::new(name, permit, code);
        agent.current_place = Some(place_id);
        agent.transition(AgentState::Active)?;

        let agent_id = agent.id;

        // Update place count
        {
            let mut places = self.places.write().await;
            if let Some(p) = places.get_mut(&place_id) {
                p.current_agents += 1;
            }
        }

        // Store agent
        self.agents.write().await.insert(agent_id, agent.clone());

        // Record impact
        self.record_impact(agent_id, "agent_created", 10).await;

        Ok(agent_id)
    }

    /// Move agent to another place (Telescript go())
    pub async fn go(&self, agent_id: Uuid, to_place_id: Uuid) -> Result<Ticket> {
        let mut agents = self.agents.write().await;
        let agent = agents
            .get_mut(&agent_id)
            .ok_or_else(|| TelescriptError::AgentNotFound(agent_id.to_string()))?;

        let from_place_id = agent.current_place.ok_or_else(|| {
            TelescriptError::FateRejection("Agent has no current place".to_string())
        })?;

        // Check permits and FATE
        agent.passes_fate()?;

        let places = self.places.read().await;
        let to_place = places
            .get(&to_place_id)
            .ok_or_else(|| TelescriptError::PlaceNotFound(to_place_id.to_string()))?;

        to_place.can_accept(&agent.permit)?;
        drop(places);

        // Issue ticket
        let ticket = Ticket::issue(agent, from_place_id, to_place_id, 300)?;

        // Transition to traveling
        agent.transition(AgentState::Traveling)?;
        agent.current_place = None;

        // Update place counts
        drop(agents);
        {
            let mut places = self.places.write().await;
            if let Some(p) = places.get_mut(&from_place_id) {
                p.current_agents = p.current_agents.saturating_sub(1);
            }
        }

        // Record impact
        self.record_impact(agent_id, "agent_traveled", 5).await;

        Ok(ticket)
    }

    /// Complete travel (arrive at destination)
    pub async fn arrive(&self, ticket: Ticket) -> Result<()> {
        if !ticket.verify() {
            return Err(TelescriptError::PermitDenied("Invalid ticket".to_string()));
        }

        let mut agents = self.agents.write().await;
        let agent = agents
            .get_mut(&ticket.agent_id)
            .ok_or_else(|| TelescriptError::AgentNotFound(ticket.agent_id.to_string()))?;

        if agent.state != AgentState::Traveling {
            return Err(TelescriptError::InvalidStateTransition {
                from: agent.state,
                to: AgentState::Active,
            });
        }

        agent.current_place = Some(ticket.to_place);
        agent.transition(AgentState::Active)?;

        drop(agents);

        // Update place count
        {
            let mut places = self.places.write().await;
            if let Some(p) = places.get_mut(&ticket.to_place) {
                p.current_agents += 1;
            }
        }

        Ok(())
    }

    /// Start a meeting between agents (Telescript meet())
    pub async fn meet(&self, initiator_id: Uuid, responder_id: Uuid) -> Result<Uuid> {
        let agents = self.agents.read().await;

        let initiator = agents
            .get(&initiator_id)
            .ok_or_else(|| TelescriptError::AgentNotFound(initiator_id.to_string()))?;

        let responder = agents
            .get(&responder_id)
            .ok_or_else(|| TelescriptError::AgentNotFound(responder_id.to_string()))?;

        // Both must be at same place
        let place_id = initiator
            .current_place
            .ok_or_else(|| TelescriptError::FateRejection("Initiator has no place".to_string()))?;

        if responder.current_place != Some(place_id) {
            return Err(TelescriptError::FateRejection(
                "Agents not at same place".to_string(),
            ));
        }

        let places = self.places.read().await;
        let place = places
            .get(&place_id)
            .ok_or_else(|| TelescriptError::PlaceNotFound(place_id.to_string()))?;

        let meeting = Meeting::request(place, initiator, responder_id)?;
        let meeting_id = meeting.id;

        drop(places);
        drop(agents);

        self.meetings.write().await.insert(meeting_id, meeting);

        // Record impact
        self.record_impact(initiator_id, "meeting_initiated", 8)
            .await;

        Ok(meeting_id)
    }

    /// Record an impact attestation
    async fn record_impact(&self, agent_id: Uuid, action: &str, score: u32) {
        let mut hasher = Hasher::new();
        hasher.update(agent_id.as_bytes());
        hasher.update(action.as_bytes());
        hasher.update(&score.to_le_bytes());

        let attestation = ImpactAttestation {
            id: Uuid::new_v4(),
            agent_id,
            action: action.to_string(),
            impact_score: score,
            timestamp: Utc::now(),
            attestation_hash: *hasher.finalize().as_bytes(),
        };

        self.impact_log.write().await.push(attestation);
    }

    /// Calculate Gini coefficient for a set of values (Adl enforcement)
    pub fn calculate_gini(values: &[f64]) -> f64 {
        if values.is_empty() || values.len() == 1 {
            return 0.0;
        }

        let n = values.len() as f64;
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sum: f64 = sorted.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }

        let mut gini_sum = 0.0;
        for (i, v) in sorted.iter().enumerate() {
            gini_sum += (2.0 * (i as f64 + 1.0) - n - 1.0) * v;
        }

        gini_sum / (n * sum)
    }

    /// Get engine statistics
    pub async fn stats(&self) -> EngineStats {
        EngineStats {
            places_count: self.places.read().await.len(),
            agents_count: self.agents.read().await.len(),
            meetings_count: self.meetings.read().await.len(),
            connections_count: self.connections.read().await.len(),
            impact_log_count: self.impact_log.read().await.len(),
        }
    }
}

impl Default for TelescriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub places_count: usize,
    pub agents_count: usize,
    pub meetings_count: usize,
    pub connections_count: usize,
    pub impact_log_count: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_authority() {
        let genesis = Authority::genesis();
        assert_eq!(genesis.delegation_depth, 0);
        assert!(genesis.delegated_from.is_none());
        assert!(genesis.verify_chain());
    }

    #[test]
    fn test_authority_delegation() {
        let genesis = Authority::genesis();
        let child = genesis.delegate("node1").unwrap();

        assert_eq!(child.delegation_depth, 1);
        assert!(child.verify_chain());
    }

    #[test]
    fn test_max_delegation_depth() {
        let mut auth = Authority::genesis();
        for i in 0..MAX_DELEGATION_DEPTH {
            auth = auth.delegate(&format!("node{}", i + 1)).unwrap();
        }

        let result = auth.delegate("too_deep");
        assert!(matches!(
            result,
            Err(TelescriptError::DelegationDepthExceeded { .. })
        ));
    }

    #[test]
    fn test_permit_creation() {
        let genesis = Authority::genesis();
        let permit = Permit::new(
            genesis,
            vec![Capability::Go, Capability::Meet],
            ResourceLimits::default(),
            3600,
        );

        assert!(permit.verify());
        assert!(permit.has_capability(Capability::Go));
        assert!(!permit.has_capability(Capability::Create));
    }

    #[test]
    fn test_place_fate_check() {
        let mut place = Place::new("test://place", None);
        assert!(place.passes_fate().is_ok());

        // Lower Ihsān below threshold
        place.ihsan_score = 900;
        assert!(matches!(
            place.passes_fate(),
            Err(TelescriptError::IhsanViolation { .. })
        ));

        // Reset Ihsān but raise Gini
        place.ihsan_score = IHSAN_THRESHOLD;
        place.gini_coefficient = 0.40;
        assert!(matches!(
            place.passes_fate(),
            Err(TelescriptError::AdlViolation { .. })
        ));
    }

    #[test]
    fn test_agent_state_transitions() {
        let genesis = Authority::genesis();
        let permit = Permit::new(genesis, vec![], ResourceLimits::default(), 3600);
        let mut agent = Agent::new("test_agent", permit, vec![]);

        assert_eq!(agent.state, AgentState::Created);

        assert!(agent.transition(AgentState::Active).is_ok());
        assert_eq!(agent.state, AgentState::Active);

        assert!(agent.transition(AgentState::Traveling).is_ok());
        assert_eq!(agent.state, AgentState::Traveling);

        // Invalid transition
        let result = agent.transition(AgentState::Meeting);
        assert!(matches!(
            result,
            Err(TelescriptError::InvalidStateTransition { .. })
        ));
    }

    #[test]
    fn test_gini_calculation() {
        // Perfect equality
        let equal = vec![10.0, 10.0, 10.0, 10.0];
        assert!((TelescriptEngine::calculate_gini(&equal) - 0.0).abs() < 0.001);

        // Some inequality
        let unequal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let gini = TelescriptEngine::calculate_gini(&unequal);
        assert!(gini > 0.0 && gini < 0.5);
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = TelescriptEngine::new();
        let stats = engine.stats().await;

        assert_eq!(stats.places_count, 1); // Genesis place
        assert_eq!(stats.agents_count, 0);
    }

    #[tokio::test]
    async fn test_create_place_and_agent() {
        let engine = TelescriptEngine::new();

        // Create a place
        let place_id = engine.create_place("test://local", None).await.unwrap();

        // Create a permit
        let permit = Permit::new(
            engine.genesis_authority().clone(),
            vec![Capability::Go, Capability::Meet, Capability::Compute],
            ResourceLimits::default(),
            3600,
        );

        // Create an agent
        let agent_id = engine
            .create_agent("test_agent", permit, vec![1, 2, 3], place_id)
            .await
            .unwrap();

        let stats = engine.stats().await;
        assert_eq!(stats.places_count, 2);
        assert_eq!(stats.agents_count, 1);
        assert_eq!(stats.impact_log_count, 1);

        // Verify agent exists
        let agents = engine.agents.read().await;
        let agent = agents.get(&agent_id).unwrap();
        assert_eq!(agent.current_place, Some(place_id));
        assert_eq!(agent.state, AgentState::Active);
    }

    #[tokio::test]
    async fn test_agent_travel() {
        let engine = TelescriptEngine::new();

        // Create two places
        let place1 = engine.create_place("test://place1", None).await.unwrap();
        let place2 = engine.create_place("test://place2", None).await.unwrap();

        // Create agent with Go capability
        let permit = Permit::new(
            engine.genesis_authority().clone(),
            vec![Capability::Go],
            ResourceLimits::default(),
            3600,
        );

        let agent_id = engine
            .create_agent("traveler", permit, vec![], place1)
            .await
            .unwrap();

        // Travel to place2
        let ticket = engine.go(agent_id, place2).await.unwrap();
        assert!(ticket.verify());

        // Verify agent is traveling
        {
            let agents = engine.agents.read().await;
            let agent = agents.get(&agent_id).unwrap();
            assert_eq!(agent.state, AgentState::Traveling);
            assert!(agent.current_place.is_none());
        }

        // Arrive at destination
        engine.arrive(ticket).await.unwrap();

        // Verify agent is at new place
        {
            let agents = engine.agents.read().await;
            let agent = agents.get(&agent_id).unwrap();
            assert_eq!(agent.state, AgentState::Active);
            assert_eq!(agent.current_place, Some(place2));
        }
    }

    #[test]
    fn test_value_transfer() {
        let mut source = Value::new(100, ValueType::Compute);
        let transferred = source.transfer(30).unwrap();

        assert_eq!(source.amount, 70);
        assert_eq!(transferred.amount, 30);

        // Cannot overdraw
        let result = source.transfer(100);
        assert!(result.is_err());
    }
}
