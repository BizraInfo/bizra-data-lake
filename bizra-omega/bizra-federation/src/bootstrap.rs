//! Federation Bootstrap — Node discovery and network initialization
//!
//! This module handles:
//! - Seed node discovery
//! - Network bootstrapping
//! - Peer exchange protocols
//! - DNS-based discovery
//!
//! Giants: Hashicorp (Serf/Consul), Bitcoin (peer discovery)

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::time::timeout;

use crate::gossip::GossipProtocol;
use bizra_core::NodeId;

/// Bootstrap configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Seed nodes to connect to
    pub seed_nodes: Vec<String>,
    /// Local bind address
    pub bind_addr: String,
    /// Discovery timeout
    pub discovery_timeout_secs: u64,
    /// Maximum peers to discover
    pub max_peers: usize,
    /// Enable mDNS discovery
    pub enable_mdns: bool,
    /// Enable DNS-SD discovery
    pub enable_dns_sd: bool,
    /// Bootstrap retry interval
    pub retry_interval_secs: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            seed_nodes: vec![],
            bind_addr: "0.0.0.0:7946".into(),
            discovery_timeout_secs: 30,
            max_peers: 100,
            enable_mdns: true,
            enable_dns_sd: false,
            retry_interval_secs: 5,
        }
    }
}

/// Bootstrap result
#[derive(Debug)]
pub struct BootstrapResult {
    pub local_addr: SocketAddr,
    pub discovered_peers: Vec<PeerInfo>,
    pub connected_count: usize,
    pub failed_seeds: Vec<String>,
}

/// Discovered peer information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub address: SocketAddr,
    pub version: String,
    pub capabilities: Vec<String>,
}

/// Federation bootstrapper
pub struct Bootstrapper {
    config: BootstrapConfig,
    local_id: NodeId,
}

impl Bootstrapper {
    /// Create new bootstrapper
    pub fn new(config: BootstrapConfig, local_id: NodeId) -> Self {
        Self { config, local_id }
    }

    /// Bootstrap the federation network
    pub async fn bootstrap(&self) -> Result<BootstrapResult, BootstrapError> {
        tracing::info!(
            seeds = ?self.config.seed_nodes,
            bind = %self.config.bind_addr,
            "Starting federation bootstrap..."
        );

        let local_addr: SocketAddr = self
            .config
            .bind_addr
            .parse()
            .map_err(|_| BootstrapError::InvalidAddress(self.config.bind_addr.clone()))?;

        let mut discovered_peers = Vec::new();
        let mut failed_seeds = Vec::new();

        // Try each seed node
        for seed in &self.config.seed_nodes {
            match self.probe_seed(seed).await {
                Ok(peers) => {
                    tracing::info!(seed = %seed, peers = peers.len(), "Seed responded");
                    for peer in peers {
                        if !discovered_peers
                            .iter()
                            .any(|p: &PeerInfo| p.node_id == peer.node_id)
                        {
                            discovered_peers.push(peer);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(seed = %seed, error = %e, "Seed probe failed");
                    failed_seeds.push(seed.clone());
                }
            }

            if discovered_peers.len() >= self.config.max_peers {
                break;
            }
        }

        // Try mDNS discovery if enabled
        if self.config.enable_mdns && discovered_peers.len() < self.config.max_peers {
            if let Ok(mdns_peers) = self.discover_mdns().await {
                for peer in mdns_peers {
                    if !discovered_peers.iter().any(|p| p.node_id == peer.node_id) {
                        discovered_peers.push(peer);
                    }
                }
            }
        }

        let connected_count = discovered_peers.len();

        tracing::info!(
            discovered = connected_count,
            failed = failed_seeds.len(),
            "Bootstrap complete"
        );

        Ok(BootstrapResult {
            local_addr,
            discovered_peers,
            connected_count,
            failed_seeds,
        })
    }

    /// Probe a seed node for peers
    async fn probe_seed(&self, seed: &str) -> Result<Vec<PeerInfo>, BootstrapError> {
        let addr: SocketAddr = seed
            .parse()
            .map_err(|_| BootstrapError::InvalidAddress(seed.into()))?;

        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| BootstrapError::Network(e.to_string()))?;

        // Send discovery request
        let request = DiscoveryRequest {
            from: self.local_id.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        };
        let request_bytes = serde_json::to_vec(&request)
            .map_err(|e| BootstrapError::Serialization(e.to_string()))?;

        socket
            .send_to(&request_bytes, addr)
            .await
            .map_err(|e| BootstrapError::Network(e.to_string()))?;

        // Wait for response
        let mut buf = vec![0u8; 65535];
        let timeout_duration = Duration::from_secs(self.config.discovery_timeout_secs);

        match timeout(timeout_duration, socket.recv_from(&mut buf)).await {
            Ok(Ok((len, _))) => {
                let response: DiscoveryResponse = serde_json::from_slice(&buf[..len])
                    .map_err(|e| BootstrapError::Serialization(e.to_string()))?;
                Ok(response.peers)
            }
            Ok(Err(e)) => Err(BootstrapError::Network(e.to_string())),
            Err(_) => Err(BootstrapError::Timeout),
        }
    }

    /// Discover peers via mDNS
    async fn discover_mdns(&self) -> Result<Vec<PeerInfo>, BootstrapError> {
        tracing::debug!("mDNS discovery not yet implemented");
        Ok(vec![])
    }

    /// Create a bootstrap server to respond to discovery requests
    pub async fn serve_discovery(
        gossip: &GossipProtocol,
        bind_addr: SocketAddr,
    ) -> Result<(), BootstrapError> {
        let socket = UdpSocket::bind(bind_addr)
            .await
            .map_err(|e| BootstrapError::Network(e.to_string()))?;

        tracing::info!(addr = %bind_addr, "Discovery server started");

        let mut buf = vec![0u8; 65535];

        loop {
            match socket.recv_from(&mut buf).await {
                Ok((len, src)) => {
                    if let Ok(request) = serde_json::from_slice::<DiscoveryRequest>(&buf[..len]) {
                        tracing::debug!(from = %request.from.0, src = %src, "Discovery request");

                        // Build peer list from gossip members
                        let members = gossip.alive_members().await;
                        let peers: Vec<PeerInfo> = members
                            .iter()
                            .map(|m| PeerInfo {
                                node_id: m.node_id.clone(),
                                address: m.addr,
                                version: env!("CARGO_PKG_VERSION").into(),
                                capabilities: vec!["inference".into(), "consensus".into()],
                            })
                            .collect();

                        let response = DiscoveryResponse { peers };
                        if let Ok(response_bytes) = serde_json::to_vec(&response) {
                            let _ = socket.send_to(&response_bytes, src).await;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Discovery recv error");
                }
            }
        }
    }
}

/// Discovery request message
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiscoveryRequest {
    from: NodeId,
    version: String,
}

/// Discovery response message
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiscoveryResponse {
    peers: Vec<PeerInfo>,
}

/// Bootstrap errors
#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("Invalid address: {0}")]
    InvalidAddress(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Discovery timeout")]
    Timeout,
    #[error("No seeds available")]
    NoSeeds,
}

/// Well-known BIZRA seed nodes (for production)
pub fn default_seed_nodes() -> Vec<String> {
    vec![
        // These would be real seed nodes in production
        // "seed1.bizra.network:7946".into(),
        // "seed2.bizra.network:7946".into(),
    ]
}

/// Create bootstrap config from environment
pub fn config_from_env() -> BootstrapConfig {
    let mut config = BootstrapConfig::default();

    if let Ok(seeds) = std::env::var("BIZRA_SEEDS") {
        config.seed_nodes = seeds.split(',').map(|s| s.trim().to_string()).collect();
    }

    if let Ok(bind) = std::env::var("BIZRA_BIND") {
        config.bind_addr = bind;
    }

    if let Ok(port) = std::env::var("BIZRA_PORT") {
        if let Ok(p) = port.parse::<u16>() {
            config.bind_addr = format!("0.0.0.0:{}", p);
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BootstrapConfig::default();
        assert!(config.seed_nodes.is_empty());
        assert_eq!(config.bind_addr, "0.0.0.0:7946");
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("BIZRA_PORT", "8000");
        let config = config_from_env();
        assert_eq!(config.bind_addr, "0.0.0.0:8000");
        std::env::remove_var("BIZRA_PORT");
    }
}
