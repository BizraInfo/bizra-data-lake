//! Application State — Shared across all handlers

use std::{sync::Arc, time::Instant};

use bizra_core::{Constitution, NodeIdentity};
use bizra_federation::{ConsensusEngine, GossipProtocol};
use bizra_inference::gateway::InferenceGateway;
use dashmap::DashMap;
use tokio::sync::RwLock;

/// Per-client token bucket for request throttling.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    capacity: f64,
    refill_rate_per_second: f64,
}

impl TokenBucket {
    /// Create a token bucket with `capacity` and full refill over `window_secs`.
    pub fn new(capacity: f64, window_secs: u64) -> Self {
        let refill_rate_per_second = if window_secs == 0 {
            capacity
        } else {
            capacity / window_secs as f64
        };

        Self {
            tokens: capacity,
            last_refill: Instant::now(),
            capacity,
            refill_rate_per_second,
        }
    }

    /// Attempt to consume a token at current time.
    pub fn try_consume(&mut self) -> bool {
        self.try_consume_at(Instant::now())
    }

    /// Attempt to consume a token at a specific instant (for deterministic tests).
    pub fn try_consume_at(&mut self, now: Instant) -> bool {
        let elapsed = now
            .checked_duration_since(self.last_refill)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        self.tokens = (self.tokens + elapsed * self.refill_rate_per_second).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

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

    /// Per-client token buckets for rate limiting.
    pub rate_limits: Arc<DashMap<String, TokenBucket>>,

    /// Optional API bearer token for privileged routes.
    pub api_token: Option<String>,

    /// Allowed CORS origins in production mode.
    pub cors_origins: Arc<Vec<String>>,

    /// Start time for uptime calculation
    pub start_time: Instant,
}

impl AppState {
    /// Create new application state with constitution
    pub fn new(constitution: Constitution) -> Self {
        let api_token = std::env::var("BIZRA_API_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        let cors_origins = std::env::var("BIZRA_CORS_ALLOWED_ORIGINS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|origin| !origin.is_empty())
                    .map(|origin| origin.to_string())
                    .collect::<Vec<String>>()
            })
            .filter(|origins| !origins.is_empty())
            .unwrap_or_else(|| {
                vec![
                    "http://localhost:5173".to_string(),
                    "http://127.0.0.1:5173".to_string(),
                ]
            });

        Self {
            identity: Arc::new(RwLock::new(None)),
            constitution,
            inference: Arc::new(RwLock::new(None)),
            gossip: Arc::new(RwLock::new(None)),
            consensus: Arc::new(RwLock::new(None)),
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            rate_limits: Arc::new(DashMap::new()),
            api_token,
            cors_origins: Arc::new(cors_origins),
            start_time: Instant::now(),
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

    /// Increment request counter, returning the new count.
    pub fn increment_requests(&self) -> u64 {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    /// Get request count
    pub fn get_request_count(&self) -> u64 {
        self.request_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Access configured API bearer token, if set.
    pub fn api_token(&self) -> Option<&str> {
        self.api_token.as_deref()
    }

    /// Access CORS origins configured for production mode.
    pub fn cors_origins(&self) -> &[String] {
        self.cors_origins.as_ref().as_slice()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(Constitution::default())
    }
}
