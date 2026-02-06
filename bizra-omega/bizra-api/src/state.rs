//! Application State — Shared across all handlers

use std::sync::Arc;
use tokio::sync::RwLock;

use bizra_core::{Constitution, NodeIdentity};
use bizra_federation::{ConsensusEngine, GossipProtocol};
use bizra_inference::gateway::InferenceGateway;

/// Global application state
pub struct AppState {
    /// Node identity (Ed25519 keypair)
    pub identity: Arc<RwLock<Option<NodeIdentity>>>,

    /// Constitution for validation
    pub constitution: Constitution,

    /// Inference gateway
    pub inference: Arc<RwLock<Option<InferenceGateway>>>,

    /// Gossip protocol for federation
    pub gossip: Arc<RwLock<Option<GossipProtocol>>>,

    /// Consensus engine for pattern elevation
    pub consensus: Arc<RwLock<Option<ConsensusEngine>>>,

    /// Request counter for metrics
    pub request_count: Arc<std::sync::atomic::AtomicU64>,

    /// Start time for uptime calculation
    pub start_time: std::time::Instant,
}

impl AppState {
    /// Create new application state with constitution
    pub fn new(constitution: Constitution) -> Self {
        Self {
            identity: Arc::new(RwLock::new(None)),
            constitution,
            inference: Arc::new(RwLock::new(None)),
            gossip: Arc::new(RwLock::new(None)),
            consensus: Arc::new(RwLock::new(None)),
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            start_time: std::time::Instant::now(),
        }
    }

    /// Initialize with identity
    pub async fn with_identity(self, identity: NodeIdentity) -> Self {
        *self.identity.write().await = Some(identity);
        self
    }

    /// Initialize with inference gateway
    pub async fn with_gateway(self, gateway: InferenceGateway) -> Self {
        *self.inference.write().await = Some(gateway);
        self
    }

    /// Initialize with gossip protocol
    pub async fn with_gossip(self, gossip: GossipProtocol) -> Self {
        *self.gossip.write().await = Some(gossip);
        self
    }

    /// Initialize with consensus engine
    pub async fn with_consensus(self, consensus: ConsensusEngine) -> Self {
        *self.consensus.write().await = Some(consensus);
        self
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Increment request counter
    pub fn increment_requests(&self) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get request count
    pub fn get_request_count(&self) -> u64 {
        self.request_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(Constitution::default())
    }
}
