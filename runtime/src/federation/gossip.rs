// src/federation/gossip.rs - Epidemic Gossip Protocol for Pattern Propagation
//
// Implements lazy-pull gossip for efficient pattern distribution.
// Compatible with Python implementation in core/federation/gossip.py

use crate::federation::protocol::{GossipMessage, GossipMessageType, PatternEnvelope};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Gossip fanout (number of peers to forward to)
pub const GOSSIP_FANOUT: usize = 3;

/// Gossip interval in seconds
pub const GOSSIP_INTERVAL_SEC: u64 = 5;

/// Heartbeat interval in seconds
pub const HEARTBEAT_INTERVAL_SEC: u64 = 30;

/// Peer timeout in seconds
pub const PEER_TIMEOUT_SEC: u64 = 90;

/// Maximum peers to track
pub const MAX_PEERS: usize = 50;

/// Maximum message size (1MB)
pub const MAX_MESSAGE_SIZE: usize = 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════════
// PEER STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Peer connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerState {
    Unknown,
    Connecting,
    Connected,
    Disconnected,
    Banned,
}

/// Information about a peer
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub node_id: String,
    pub addr: SocketAddr,
    pub state: PeerState,
    pub last_seen: Instant,
    pub last_heartbeat: Instant,
    pub patterns_received: u64,
    pub patterns_sent: u64,
    pub reputation: f64,
    pub rate_limit_tokens: u32,
    pub rate_limit_last_refill: Instant,
}

impl PeerInfo {
    pub fn new(node_id: String, addr: SocketAddr) -> Self {
        let now = Instant::now();
        Self {
            node_id,
            addr,
            state: PeerState::Unknown,
            last_seen: now,
            last_heartbeat: now,
            patterns_received: 0,
            patterns_sent: 0,
            reputation: 1.0,
            rate_limit_tokens: 100,
            rate_limit_last_refill: now,
        }
    }

    pub fn is_alive(&self) -> bool {
        self.state == PeerState::Connected
            && self.last_seen.elapsed() < Duration::from_secs(PEER_TIMEOUT_SEC)
    }

    /// Check and consume rate limit token
    pub fn check_rate_limit(&mut self) -> bool {
        // Refill tokens (1 per second, max 100)
        let elapsed = self.rate_limit_last_refill.elapsed().as_secs() as u32;
        if elapsed > 0 {
            self.rate_limit_tokens = (self.rate_limit_tokens + elapsed).min(100);
            self.rate_limit_last_refill = Instant::now();
        }

        if self.rate_limit_tokens > 0 {
            self.rate_limit_tokens -= 1;
            true
        } else {
            false
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOSSIP PROTOCOL
// ═══════════════════════════════════════════════════════════════════════════════

/// Gossip protocol statistics
#[derive(Debug, Default, Clone)]
pub struct GossipStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub patterns_announced: u64,
    pub patterns_requested: u64,
    pub patterns_served: u64,
}

/// Gossip protocol implementation
pub struct GossipProtocol {
    /// This node's ID
    node_id: String,
    /// Listen address
    listen_addr: SocketAddr,
    /// Connected peers
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    /// Known pattern IDs (for deduplication)
    known_patterns: Arc<RwLock<HashSet<String>>>,
    /// Pattern cache for serving requests
    pattern_cache: Arc<RwLock<HashMap<String, PatternEnvelope>>>,
    /// Message deduplication
    seen_messages: Arc<RwLock<HashSet<String>>>,
    /// Statistics
    stats: Arc<RwLock<GossipStats>>,
    /// Running flag
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Callback for received patterns
    on_pattern_received: Arc<RwLock<Option<Box<dyn Fn(PatternEnvelope) + Send + Sync>>>>,
}

impl GossipProtocol {
    /// Create new gossip protocol handler
    pub fn new(node_id: String, listen_addr: SocketAddr) -> Self {
        Self {
            node_id,
            listen_addr,
            peers: Arc::new(RwLock::new(HashMap::new())),
            known_patterns: Arc::new(RwLock::new(HashSet::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            seen_messages: Arc::new(RwLock::new(HashSet::new())),
            stats: Arc::new(RwLock::new(GossipStats::default())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            on_pattern_received: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the gossip protocol
    pub async fn start(&mut self) -> Result<()> {
        if self.running.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Start TCP listener
        let listener = TcpListener::bind(self.listen_addr).await?;
        info!("👂 Gossip listening on {}", self.listen_addr);

        // Spawn listener task
        let peers = self.peers.clone();
        let pattern_cache = self.pattern_cache.clone();
        let known_patterns = self.known_patterns.clone();
        let seen_messages = self.seen_messages.clone();
        let stats = self.stats.clone();
        let running = self.running.clone();
        let node_id = self.node_id.clone();
        let on_pattern = self.on_pattern_received.clone();

        tokio::spawn(async move {
            Self::accept_loop(
                listener,
                peers,
                pattern_cache,
                known_patterns,
                seen_messages,
                stats,
                running,
                node_id,
                on_pattern,
            )
            .await;
        });

        // Spawn heartbeat task
        let peers_hb = self.peers.clone();
        let running_hb = self.running.clone();
        let node_id_hb = self.node_id.clone();

        tokio::spawn(async move {
            Self::heartbeat_loop(peers_hb, running_hb, node_id_hb).await;
        });

        Ok(())
    }

    /// Stop the gossip protocol
    pub async fn stop(&mut self) -> Result<()> {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Accept loop for incoming connections
    async fn accept_loop(
        listener: TcpListener,
        peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
        pattern_cache: Arc<RwLock<HashMap<String, PatternEnvelope>>>,
        known_patterns: Arc<RwLock<HashSet<String>>>,
        seen_messages: Arc<RwLock<HashSet<String>>>,
        stats: Arc<RwLock<GossipStats>>,
        running: Arc<std::sync::atomic::AtomicBool>,
        node_id: String,
        on_pattern: Arc<RwLock<Option<Box<dyn Fn(PatternEnvelope) + Send + Sync>>>>,
    ) {
        while running.load(std::sync::atomic::Ordering::SeqCst) {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    let peers = peers.clone();
                    let pattern_cache = pattern_cache.clone();
                    let known_patterns = known_patterns.clone();
                    let seen_messages = seen_messages.clone();
                    let stats = stats.clone();
                    let node_id = node_id.clone();
                    let on_pattern = on_pattern.clone();

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(
                            stream,
                            addr,
                            peers,
                            pattern_cache,
                            known_patterns,
                            seen_messages,
                            stats,
                            node_id,
                            on_pattern,
                        )
                        .await
                        {
                            debug!("Connection error from {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }
    }

    /// Handle an incoming connection
    async fn handle_connection(
        mut stream: TcpStream,
        addr: SocketAddr,
        peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
        pattern_cache: Arc<RwLock<HashMap<String, PatternEnvelope>>>,
        known_patterns: Arc<RwLock<HashSet<String>>>,
        seen_messages: Arc<RwLock<HashSet<String>>>,
        stats: Arc<RwLock<GossipStats>>,
        node_id: String,
        on_pattern: Arc<RwLock<Option<Box<dyn Fn(PatternEnvelope) + Send + Sync>>>>,
    ) -> Result<()> {
        // Read message length
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > MAX_MESSAGE_SIZE {
            return Err(anyhow::anyhow!("Message too large: {}", len));
        }

        // Read message
        let mut buf = vec![0u8; len];
        stream.read_exact(&mut buf).await?;

        let msg = GossipMessage::from_bytes(&buf)?;

        // Check for duplicate
        {
            let mut seen = seen_messages.write().await;
            if seen.contains(&msg.message_id) {
                return Ok(());
            }
            seen.insert(msg.message_id.clone());

            // Limit seen messages cache
            if seen.len() > 10000 {
                seen.clear();
            }
        }

        // Update stats
        {
            let mut s = stats.write().await;
            s.messages_received += 1;
        }

        // Handle message based on type
        match msg.msg_type {
            GossipMessageType::Hello => {
                Self::handle_hello(msg, addr, &peers).await?;
            }
            GossipMessageType::Heartbeat => {
                Self::handle_heartbeat(&msg.sender_id, &peers).await?;
            }
            GossipMessageType::PatternAnnounce => {
                Self::handle_pattern_announce(
                    msg,
                    addr,
                    &known_patterns,
                    &pattern_cache,
                    &stats,
                    &on_pattern,
                )
                .await?;
            }
            GossipMessageType::PatternRequest => {
                Self::handle_pattern_request(msg, &mut stream, &pattern_cache, &stats).await?;
            }
            GossipMessageType::PatternResponse => {
                Self::handle_pattern_response(msg, &known_patterns, &on_pattern).await?;
            }
            _ => {
                debug!("Unhandled message type: {:?}", msg.msg_type);
            }
        }

        Ok(())
    }

    /// Handle HELLO message
    async fn handle_hello(
        msg: GossipMessage,
        addr: SocketAddr,
        peers: &Arc<RwLock<HashMap<String, PeerInfo>>>,
    ) -> Result<()> {
        let mut peers_guard = peers.write().await;

        if peers_guard.len() >= MAX_PEERS {
            return Ok(());
        }

        let mut peer = PeerInfo::new(msg.sender_id.clone(), addr);
        peer.state = PeerState::Connected;
        peers_guard.insert(msg.sender_id.clone(), peer);

        info!(
            "🤝 Peer connected: {} ({})",
            &msg.sender_id[..16.min(msg.sender_id.len())],
            addr
        );

        Ok(())
    }

    /// Handle HEARTBEAT message
    async fn handle_heartbeat(
        sender_id: &str,
        peers: &Arc<RwLock<HashMap<String, PeerInfo>>>,
    ) -> Result<()> {
        let mut peers_guard = peers.write().await;

        if let Some(peer) = peers_guard.get_mut(sender_id) {
            peer.last_seen = Instant::now();
            peer.last_heartbeat = Instant::now();
        }

        Ok(())
    }

    /// Handle PATTERN_ANNOUNCE message (lazy-pull)
    async fn handle_pattern_announce(
        msg: GossipMessage,
        addr: SocketAddr,
        known_patterns: &Arc<RwLock<HashSet<String>>>,
        pattern_cache: &Arc<RwLock<HashMap<String, PatternEnvelope>>>,
        stats: &Arc<RwLock<GossipStats>>,
        on_pattern: &Arc<RwLock<Option<Box<dyn Fn(PatternEnvelope) + Send + Sync>>>>,
    ) -> Result<()> {
        let pattern_id = msg
            .payload
            .get("pattern_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing pattern_id"))?;

        // Check if we already have it
        {
            let known = known_patterns.read().await;
            if known.contains(pattern_id) {
                return Ok(());
            }
        }

        // Request full pattern
        let request = GossipMessage::new(
            GossipMessageType::PatternRequest,
            "self".to_string(), // Will be replaced
            serde_json::json!({"pattern_id": pattern_id}),
        );

        // Connect and request
        if let Ok(mut stream) = TcpStream::connect(addr).await {
            let data = request.to_bytes()?;
            let len = (data.len() as u32).to_be_bytes();
            stream.write_all(&len).await?;
            stream.write_all(&data).await?;

            // Read response
            let mut len_buf = [0u8; 4];
            if stream.read_exact(&mut len_buf).await.is_ok() {
                let len = u32::from_be_bytes(len_buf) as usize;
                if len <= MAX_MESSAGE_SIZE {
                    let mut buf = vec![0u8; len];
                    if stream.read_exact(&mut buf).await.is_ok() {
                        let response = GossipMessage::from_bytes(&buf)?;
                        Self::handle_pattern_response(response, known_patterns, on_pattern).await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle PATTERN_REQUEST message
    async fn handle_pattern_request(
        msg: GossipMessage,
        stream: &mut TcpStream,
        pattern_cache: &Arc<RwLock<HashMap<String, PatternEnvelope>>>,
        stats: &Arc<RwLock<GossipStats>>,
    ) -> Result<()> {
        let pattern_id = msg
            .payload
            .get("pattern_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing pattern_id"))?;

        let cache = pattern_cache.read().await;

        if let Some(envelope) = cache.get(pattern_id) {
            let response = GossipMessage::new(
                GossipMessageType::PatternResponse,
                "self".to_string(),
                serde_json::to_value(envelope)?,
            );

            let data = response.to_bytes()?;
            let len = (data.len() as u32).to_be_bytes();
            stream.write_all(&len).await?;
            stream.write_all(&data).await?;

            let mut s = stats.write().await;
            s.patterns_served += 1;
        }

        Ok(())
    }

    /// Handle PATTERN_RESPONSE message
    async fn handle_pattern_response(
        msg: GossipMessage,
        known_patterns: &Arc<RwLock<HashSet<String>>>,
        on_pattern: &Arc<RwLock<Option<Box<dyn Fn(PatternEnvelope) + Send + Sync>>>>,
    ) -> Result<()> {
        let envelope: PatternEnvelope = serde_json::from_value(msg.payload)?;

        // Verify envelope
        if !envelope.verify()? {
            warn!("Invalid pattern received - verification failed");
            return Ok(());
        }

        // Mark as known
        {
            let mut known = known_patterns.write().await;
            known.insert(envelope.metadata.pattern_id.clone());
        }

        // Notify callback
        {
            let callback = on_pattern.read().await;
            if let Some(ref cb) = *callback {
                cb(envelope);
            }
        }

        Ok(())
    }

    /// Heartbeat loop
    async fn heartbeat_loop(
        peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
        running: Arc<std::sync::atomic::AtomicBool>,
        node_id: String,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(HEARTBEAT_INTERVAL_SEC));

        while running.load(std::sync::atomic::Ordering::SeqCst) {
            interval.tick().await;

            let peers_snapshot: Vec<(String, SocketAddr)>;
            {
                let peers_guard = peers.read().await;
                peers_snapshot = peers_guard
                    .iter()
                    .filter(|(_, p)| p.is_alive())
                    .map(|(id, p)| (id.clone(), p.addr))
                    .collect();
            }

            for (_, addr) in peers_snapshot {
                let msg = GossipMessage::new(
                    GossipMessageType::Heartbeat,
                    node_id.clone(),
                    serde_json::json!({}),
                );

                if let Ok(mut stream) = TcpStream::connect(addr).await {
                    if let Ok(data) = msg.to_bytes() {
                        let len = (data.len() as u32).to_be_bytes();
                        let _ = stream.write_all(&len).await;
                        let _ = stream.write_all(&data).await;
                    }
                }
            }

            // Cleanup dead peers
            {
                let mut peers_guard = peers.write().await;
                peers_guard.retain(|_, p| {
                    p.last_seen.elapsed() < Duration::from_secs(PEER_TIMEOUT_SEC * 2)
                });
            }
        }
    }

    /// Connect to a peer
    pub async fn connect_to_peer(&mut self, addr: SocketAddr) -> Result<()> {
        let msg = GossipMessage::new(
            GossipMessageType::Hello,
            self.node_id.clone(),
            serde_json::json!({"version": crate::federation::protocol::PFP_VERSION}),
        );

        let mut stream = TcpStream::connect(addr).await?;
        let data = msg.to_bytes()?;
        let len = (data.len() as u32).to_be_bytes();
        stream.write_all(&len).await?;
        stream.write_all(&data).await?;

        // Add to peers
        let peer = PeerInfo::new(format!("peer_{}", addr), addr);
        self.peers.write().await.insert(peer.node_id.clone(), peer);

        info!("🔗 Connected to peer {}", addr);

        Ok(())
    }

    /// Broadcast a pattern to the network
    pub async fn broadcast_pattern(&self, envelope: &PatternEnvelope) -> Result<()> {
        // Add to cache
        {
            let mut cache = self.pattern_cache.write().await;
            cache.insert(envelope.metadata.pattern_id.clone(), envelope.clone());
        }

        // Mark as known
        {
            let mut known = self.known_patterns.write().await;
            known.insert(envelope.metadata.pattern_id.clone());
        }

        // Create announce message
        let msg = GossipMessage::new(
            GossipMessageType::PatternAnnounce,
            self.node_id.clone(),
            serde_json::json!({
                "pattern_id": envelope.metadata.pattern_id,
                "origin_node_id": envelope.metadata.origin_node_id,
                "impact_score": envelope.metadata.impact_score,
            }),
        );

        // Get alive peers
        let peers_snapshot: Vec<SocketAddr> = {
            let peers_guard = self.peers.read().await;
            peers_guard
                .values()
                .filter(|p| p.is_alive())
                .take(GOSSIP_FANOUT)
                .map(|p| p.addr)
                .collect()
        };

        let peer_count = peers_snapshot.len();

        // Send to peers
        for addr in peers_snapshot {
            let msg_clone = msg.clone();
            tokio::spawn(async move {
                if let Ok(mut stream) = TcpStream::connect(addr).await {
                    if let Ok(data) = msg_clone.to_bytes() {
                        let len = (data.len() as u32).to_be_bytes();
                        let _ = stream.write_all(&len).await;
                        let _ = stream.write_all(&data).await;
                    }
                }
            });
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.patterns_announced += 1;
            stats.messages_sent += peer_count as u64;
        }

        Ok(())
    }

    /// Set callback for received patterns
    pub fn on_pattern_received<F>(&self, callback: F)
    where
        F: Fn(PatternEnvelope) + Send + Sync + 'static,
    {
        let mut cb = self.on_pattern_received.try_write().unwrap();
        *cb = Some(Box::new(callback));
    }

    /// Get connected peer count
    pub fn connected_peer_count(&self) -> usize {
        // Synchronous version for simple queries
        0 // Would need async in real implementation
    }

    /// Get statistics
    pub async fn get_stats(&self) -> GossipStats {
        self.stats.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_info() {
        let mut peer = PeerInfo::new("test_node".to_string(), "127.0.0.1:9999".parse().unwrap());

        assert_eq!(peer.state, PeerState::Unknown);
        assert!(peer.check_rate_limit()); // Should have tokens

        // Exhaust tokens
        for _ in 0..100 {
            peer.check_rate_limit();
        }
        assert!(!peer.check_rate_limit()); // Should be rate limited
    }

    #[tokio::test]
    async fn test_gossip_protocol_creation() {
        let gossip = GossipProtocol::new("test_node".to_string(), "127.0.0.1:0".parse().unwrap());

        assert_eq!(gossip.node_id, "test_node");
    }
}
