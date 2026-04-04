// src/federation/mod.rs - Pattern Federation Protocol (PFP) v1.0
//
// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  PATTERN FEDERATION PROTOCOL - Network Effect Activation Layer            ║
// ╠════════════════════════════════════════════════════════════════════════════╣
// ║                                                                            ║
// ║  Transforms isolated PAT nodes into collectively intelligent network.     ║
// ║  Each node's SAPE learning becomes every node's learning.                 ║
// ║                                                                            ║
// ║  Network Architecture:                                                     ║
// ║  ─────────────────────                                                     ║
// ║                                                                            ║
// ║    Node₁ ←──────→ Node₂                                                   ║
// ║      ↕      ↘   ↙      ↕                                                  ║
// ║      ↕        ✕        ↕          Gossip Protocol                         ║
// ║      ↕      ↙   ↘      ↕          (Epidemic broadcast)                    ║
// ║    Node₃ ←──────→ Node₄                                                   ║
// ║                                                                            ║
// ║  Value ∝ n² (Metcalfe's Law)                                              ║
// ║  إحسان Standard: Excellence through collective intelligence               ║
// ╚════════════════════════════════════════════════════════════════════════════╝

pub mod consensus;
pub mod gossip;
pub mod protocol;

use crate::sape::{get_sape, ElevatedPattern, ProbeDimension};
use anyhow::Result;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub use consensus::{ConsensusPhase, ConsensusState, PatternConsensus, CONSENSUS_QUORUM};
pub use gossip::{GossipProtocol, PeerInfo, PeerState};
pub use protocol::{
    ConsensusResult,
    ConsensusVote,
    GossipMessage,
    GossipMessageType,
    PatternEnvelope,
    PatternMetadata,
    PatternPayload,
    PatternType,
    SignedGossipMessage,
    // CRIT-1 & CRIT-2: Genesis v2.2.2 Ed25519 signed types
    SignedVote,
    VoteDecision,
    MIN_IHSAN_SCORE,
    PFP_VERSION,
};

/// Federation configuration
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Node identifier
    pub node_id: String,
    /// Listen address
    pub listen_addr: SocketAddr,
    /// Bootstrap peers
    pub bootstrap_peers: Vec<SocketAddr>,
    /// Auto-propagate elevated patterns
    pub auto_propagate: bool,
    /// Auto-adopt consensus-accepted patterns
    pub auto_adopt: bool,
    /// Max patterns to store
    pub max_patterns: usize,
    /// Max peers to connect
    pub max_peers: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            node_id: format!("node_{}", hex::encode(&rand::random::<[u8; 8]>())),
            listen_addr: "0.0.0.0:9999".parse().unwrap(),
            bootstrap_peers: Vec::new(),
            auto_propagate: true,
            auto_adopt: true,
            max_patterns: 10000,
            max_peers: 50,
        }
    }
}

impl FederationConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        let node_id = std::env::var("BIZRA_NODE_ID")
            .unwrap_or_else(|_| format!("node_{}", hex::encode(&rand::random::<[u8; 8]>())));

        let port: u16 = std::env::var("BIZRA_FED_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(9999);

        let listen_addr = format!("0.0.0.0:{}", port).parse().unwrap();

        Self {
            node_id,
            listen_addr,
            ..Default::default()
        }
    }
}

/// Federation statistics
#[derive(Debug, Default, Clone)]
pub struct FederationStats {
    pub local_patterns: usize,
    pub federated_patterns: usize,
    pub patterns_sent: u64,
    pub patterns_received: u64,
    pub consensus_proposed: u64,
    pub consensus_accepted: u64,
    pub connected_peers: usize,
    pub network_multiplier: f64,
}

/// Pattern Federation Coordinator
///
/// Manages the lifecycle of federated patterns:
/// 1. Elevation: SAPE elevates local pattern
/// 2. Broadcast: Gossip announces pattern to network
/// 3. Consensus: Validators vote on pattern
/// 4. Adoption: Accepted patterns integrated into local SAPE
pub struct PatternFederation {
    config: FederationConfig,

    /// Local patterns (elevated by this node)
    local_patterns: Arc<RwLock<HashMap<String, PatternEnvelope>>>,

    /// Federated patterns (received from network)
    federated_patterns: Arc<RwLock<HashMap<String, PatternEnvelope>>>,

    /// Gossip protocol handler
    gossip: Arc<RwLock<GossipProtocol>>,

    /// Consensus engine
    consensus: Arc<RwLock<PatternConsensus>>,

    /// Ed25519 keypair
    private_key: ed25519_dalek::SigningKey,
    public_key: ed25519_dalek::VerifyingKey,

    /// Statistics
    stats: Arc<RwLock<FederationStats>>,

    /// Running state
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl PatternFederation {
    /// Create new federation coordinator
    pub fn new(config: FederationConfig) -> Result<Self> {
        // Generate or load keypair
        use rand::RngCore;
        let mut secret = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret);
        let private_key = ed25519_dalek::SigningKey::from_bytes(&secret);
        let public_key = private_key.verifying_key();

        let node_id = config.node_id.clone();

        let gossip = GossipProtocol::new(node_id.clone(), config.listen_addr);

        let consensus = PatternConsensus::new(
            node_id,
            private_key.to_bytes().to_vec(),
            public_key.to_bytes().to_vec(),
        );

        Ok(Self {
            config,
            local_patterns: Arc::new(RwLock::new(HashMap::new())),
            federated_patterns: Arc::new(RwLock::new(HashMap::new())),
            gossip: Arc::new(RwLock::new(gossip)),
            consensus: Arc::new(RwLock::new(consensus)),
            private_key,
            public_key,
            stats: Arc::new(RwLock::new(FederationStats::default())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start federation services
    pub async fn start(&self) -> Result<()> {
        if self.running.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Start gossip protocol
        {
            let mut gossip = self.gossip.write().await;
            gossip.start().await?;
        }

        // Connect to bootstrap peers
        for peer in &self.config.bootstrap_peers {
            let gossip = self.gossip.clone();
            let addr = *peer;
            tokio::spawn(async move {
                let mut g = gossip.write().await;
                if let Err(e) = g.connect_to_peer(addr).await {
                    warn!("Failed to connect to bootstrap peer {}: {}", addr, e);
                }
            });
        }

        info!(
            "🌐 Pattern Federation started on {}",
            self.config.listen_addr
        );
        info!("   Node ID: {}", self.config.node_id);

        Ok(())
    }

    /// Stop federation services
    pub async fn stop(&self) -> Result<()> {
        if !self.running.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);

        {
            let mut gossip = self.gossip.write().await;
            gossip.stop().await?;
        }

        info!("🔌 Pattern Federation stopped");
        Ok(())
    }

    /// Elevate a locally discovered pattern and broadcast to network
    ///
    /// This is the main integration point with SAPE.
    pub async fn elevate_pattern(&self, pattern: &ElevatedPattern) -> Result<PatternEnvelope> {
        let metadata = PatternMetadata {
            pattern_id: pattern.id.clone(),
            pattern_type: PatternType::SapeProbe,
            version: 1,
            origin_node_id: self.config.node_id.clone(),
            origin_timestamp: chrono::Utc::now(),
            repetition_count: pattern.activation_count as u32,
            success_rate: 0.85, // Default from SAPE
            impact_score: self.compute_impact_score(pattern),
            ihsan_score: 0.92, // From SAPE probes
            adoption_count: 0,
            expires_at: chrono::Utc::now() + chrono::Duration::days(30),
            tags: vec![],
        };

        let payload = PatternPayload {
            trigger_sequence: pattern.trigger_sequence.clone(),
            optimization: pattern.optimization.clone(),
            latency_reduction_ms: pattern.latency_reduction_ms,
            token_savings_percent: pattern.token_savings_percent,
            snr_improvement: pattern.snr_improvement,
        };

        // Create signed envelope
        let envelope =
            PatternEnvelope::create(metadata, payload, &self.private_key, &self.public_key)?;

        // Store locally
        {
            let mut local = self.local_patterns.write().await;
            local.insert(envelope.metadata.pattern_id.clone(), envelope.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.local_patterns += 1;
        }

        info!(
            "📤 Elevated pattern {} (impact={:.2})",
            &envelope.metadata.pattern_id[..16],
            envelope.metadata.impact_score
        );

        // Broadcast to network
        if self.config.auto_propagate {
            let gossip = self.gossip.read().await;
            gossip.broadcast_pattern(&envelope).await?;

            let mut stats = self.stats.write().await;
            stats.patterns_sent += 1;
        }

        Ok(envelope)
    }

    /// Compute pattern impact score
    fn compute_impact_score(&self, pattern: &ElevatedPattern) -> f64 {
        let rep_score = ((pattern.activation_count as f64) / 10.0).sqrt().min(1.0);
        let latency_score = (pattern.latency_reduction_ms as f64 / 200.0).min(1.0);

        0.30 * rep_score + 0.30 * 0.85 + 0.25 * 0.92 + 0.15 * latency_score
    }

    /// Handle pattern received from network
    pub async fn handle_received_pattern(&self, envelope: PatternEnvelope) -> Result<()> {
        let pattern_id = envelope.metadata.pattern_id.clone();

        // Already have it?
        {
            let local = self.local_patterns.read().await;
            if local.contains_key(&pattern_id) {
                return Ok(());
            }

            let federated = self.federated_patterns.read().await;
            if federated.contains_key(&pattern_id) {
                return Ok(());
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.patterns_received += 1;
        }

        info!(
            "📥 Received pattern {} from {}",
            &pattern_id[..16],
            &envelope.metadata.origin_node_id[..16]
        );

        // Verify envelope
        if !envelope.verify()? {
            warn!(
                "Invalid pattern {} - verification failed",
                &pattern_id[..16]
            );
            return Ok(());
        }

        // Propose for consensus
        {
            let mut consensus = self.consensus.write().await;
            let state = consensus.propose_pattern(&envelope).await?;

            if let Some(ref result) = state.result {
                if result.accepted {
                    self.adopt_pattern(envelope).await?;
                }
            }
        }

        Ok(())
    }

    /// Adopt a consensus-accepted pattern into local SAPE
    async fn adopt_pattern(&self, envelope: PatternEnvelope) -> Result<()> {
        if !self.config.auto_adopt {
            return Ok(());
        }

        let pattern_id = envelope.metadata.pattern_id.clone();

        // Store in federated patterns
        {
            let mut federated = self.federated_patterns.write().await;
            federated.insert(pattern_id.clone(), envelope.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.federated_patterns += 1;
            stats.consensus_accepted += 1;
        }

        info!("✅ Adopted federated pattern {}", &pattern_id[..16]);

        // Register with SAPE
        self.register_with_sape(&envelope)?;

        // Update network multiplier
        self.update_network_multiplier().await;

        Ok(())
    }

    /// Register federated pattern with local SAPE engine
    fn register_with_sape(&self, envelope: &PatternEnvelope) -> Result<()> {
        let sape = get_sape();
        let mut engine = sape
            .lock()
            .map_err(|e| anyhow::anyhow!("SAPE lock failed: {}", e))?;

        let pattern = ElevatedPattern {
            id: format!("fed_{}", envelope.metadata.pattern_id),
            name: format!(
                "Network: {}...",
                envelope
                    .payload
                    .trigger_sequence
                    .first()
                    .unwrap_or(&"unknown".to_string())
            ),
            trigger_sequence: envelope.payload.trigger_sequence.clone(),
            optimization: envelope.payload.optimization.clone(),
            snr_improvement: envelope.payload.snr_improvement,
            latency_reduction_ms: envelope.payload.latency_reduction_ms,
            token_savings_percent: envelope.payload.token_savings_percent,
            activation_count: envelope.metadata.adoption_count,
            created_at: envelope.metadata.origin_timestamp,
        };

        engine.register_pattern(pattern);

        Ok(())
    }

    /// Update network multiplier based on federation state
    async fn update_network_multiplier(&self) {
        let local_count = self.local_patterns.read().await.len();
        let fed_count = self.federated_patterns.read().await.len();
        let peer_count = self.gossip.read().await.connected_peer_count();

        let total_patterns = local_count + fed_count;

        if total_patterns > 0 && peer_count > 0 {
            let pattern_factor = (total_patterns as f64 + 1.0).log10() / 10.0;
            let peer_factor = (peer_count as f64 / 10.0).min(1.0);

            let mut stats = self.stats.write().await;
            stats.network_multiplier = 1.0 + pattern_factor * peer_factor;
        }
    }

    /// Get all patterns (local + federated)
    pub async fn get_all_patterns(&self) -> Vec<PatternEnvelope> {
        let local = self.local_patterns.read().await;
        let federated = self.federated_patterns.read().await;

        local.values().chain(federated.values()).cloned().collect()
    }

    /// Get federation statistics
    pub async fn get_stats(&self) -> FederationStats {
        let mut stats = self.stats.read().await.clone();
        stats.local_patterns = self.local_patterns.read().await.len();
        stats.federated_patterns = self.federated_patterns.read().await.len();
        stats.connected_peers = self.gossip.read().await.connected_peer_count();
        stats
    }

    /// Connect to a peer
    pub async fn connect_to_peer(&self, addr: SocketAddr) -> Result<()> {
        let mut gossip = self.gossip.write().await;
        gossip.connect_to_peer(addr).await
    }
}

/// Global federation instance
static FEDERATION: std::sync::OnceLock<Arc<tokio::sync::RwLock<PatternFederation>>> =
    std::sync::OnceLock::new();

/// Get or initialize global federation
pub fn get_federation() -> Arc<tokio::sync::RwLock<PatternFederation>> {
    FEDERATION
        .get_or_init(|| {
            let config = FederationConfig::from_env();
            let federation =
                PatternFederation::new(config).expect("Failed to create PatternFederation");
            Arc::new(tokio::sync::RwLock::new(federation))
        })
        .clone()
}

/// Start global federation
pub async fn start_federation() -> Result<()> {
    let federation = get_federation();
    let fed = federation.read().await;
    fed.start().await
}

/// Elevate pattern to network (convenience function)
pub async fn elevate_pattern(pattern: &ElevatedPattern) -> Result<PatternEnvelope> {
    let federation = get_federation();
    let fed = federation.read().await;
    fed.elevate_pattern(pattern).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert!(config.node_id.starts_with("node_"));
        assert_eq!(config.listen_addr.port(), 9999);
        assert!(config.auto_propagate);
        assert!(config.auto_adopt);
    }

    #[tokio::test]
    async fn test_federation_creation() {
        let config = FederationConfig::default();
        let fed = PatternFederation::new(config);
        assert!(fed.is_ok());
    }
}
