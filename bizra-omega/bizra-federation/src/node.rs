//! Federation Node

use crate::consensus::ConsensusEngine;
use crate::gossip::GossipProtocol;
use bizra_core::{Constitution, NodeId, NodeIdentity};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfig {
    pub name: String,
    pub gossip_addr: SocketAddr,
    pub seeds: Vec<SocketAddr>,
    pub data_dir: String,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            name: "bizra-node".into(),
            gossip_addr: ([0, 0, 0, 0], crate::DEFAULT_GOSSIP_PORT).into(),
            seeds: Vec::new(),
            data_dir: "./data".into(),
        }
    }
}

pub struct FederationNode {
    identity: Arc<NodeIdentity>,
    config: NodeConfig,
    gossip: Arc<RwLock<GossipProtocol>>,
    consensus: Arc<RwLock<ConsensusEngine>>,
    running: Arc<RwLock<bool>>,
}

impl FederationNode {
    pub fn new(config: NodeConfig, identity: NodeIdentity, _constitution: Constitution) -> Self {
        let gossip = GossipProtocol::new(
            identity.node_id().clone(),
            config.gossip_addr,
            identity.signing_key().clone(),
        );
        let consensus =
            ConsensusEngine::new(NodeIdentity::from_secret_bytes(&identity.secret_bytes()));
        Self {
            identity: Arc::new(identity),
            config,
            gossip: Arc::new(RwLock::new(gossip)),
            consensus: Arc::new(RwLock::new(consensus)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    pub fn node_id(&self) -> &NodeId {
        self.identity.node_id()
    }
    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub async fn start(&self) -> Result<(), NodeError> {
        if *self.running.read().await {
            return Err(NodeError::AlreadyRunning);
        }
        for seed in &self.config.seeds {
            let seed_id = NodeId(format!("seed_{}", seed.port()));
            self.gossip.write().await.add_seed(seed_id, *seed).await;
        }
        *self.running.write().await = true;
        tracing::info!(node_id = %self.identity.node_id(), "Federation node started");
        Ok(())
    }

    pub async fn stop(&self) -> Result<(), NodeError> {
        if !*self.running.read().await {
            return Err(NodeError::NotRunning);
        }
        *self.running.write().await = false;
        Ok(())
    }

    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
    pub async fn peer_count(&self) -> usize {
        self.gossip
            .read()
            .await
            .member_count()
            .await
            .saturating_sub(1)
    }

    pub async fn propose_pattern(
        &self,
        pattern: serde_json::Value,
        ihsan: f64,
    ) -> Result<String, NodeError> {
        let proposal = self
            .consensus
            .write()
            .await
            .propose(pattern, ihsan)
            .map_err(NodeError::Consensus)?;
        Ok(proposal.id)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("Already running")]
    AlreadyRunning,
    #[error("Not running")]
    NotRunning,
    #[error("Consensus: {0}")]
    Consensus(#[from] crate::consensus::ConsensusError),
}
