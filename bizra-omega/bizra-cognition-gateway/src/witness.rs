//! BIZRA witness-grade chain-head observation — Cycle-8 Day 4
//!
//! بسم الله الرحمن الرحيم
//!
//! Closes the 4th (economic) modality of the Four-Modality Golden
//! Standard in witness-grade form: nodes ping allowlisted peers with
//! their sealed chain-head; peers store the observation and serve it on
//! GET. Any skeptical stranger can query a witness to detect tampering
//! — if Node A claims chain_head X at observed_at_ns T, but a witness
//! reported chain_head Y for the same T, the divergence is publicly
//! detectable in bounded time.
//!
//! --- DOCTRINAL CONSTRAINT (Cycle-8, 2026-04-19) ---
//!
//! T=0 economic finality is WITNESS-GRADE DETECTABILITY ONLY. The
//! following primitives are explicitly Horizon / Layer B, NOT Day 4
//! scope:
//!   - bonded stakes
//!   - slashing mechanisms
//!   - DAO governance
//!   - challenge-period economics
//!   - token system
//!
//! Witness-grade closure means: divergence is detectable, transferable
//! (anyone can produce the proof of mismatch), and bounded in cost to
//! verify. That is the T=0 fourth modality. Nothing more.
//!
//! --- NON-GOALS TODAY (deferred to named later days) ---
//!
//!   - Ed25519 signatures on observations (Day 5 proof-of-priority).
//!   - Persistent on-disk storage of observations (Day 5+; today's
//!     in-memory Mutex<HashMap> is intentional minimum).
//!   - Witness peer auto-discovery (Horizon).
//!   - Byzantine-tolerant consensus among witnesses (Horizon).
//!   - Challenge/dispute protocol (Horizon).
//!
//! --- CONSTITUTIONAL ALIGNMENT ---
//!
//! - CLAIM_MUST_BIND: every observation carries a chain_head hash; the
//!   receiving witness cannot forge an observation without knowing the
//!   node's claimed head. Day 5 adds Ed25519 binding.
//! - NO_SHADOW_STATE: the store echoes what was received verbatim; no
//!   derived state, no simulated observation. If no ping arrived, GET
//!   returns 404 — NOT a fabricated "unknown but plausible" head.
//! - ZANN_ZERO: witness refuses to answer about a node it has never
//!   heard from (404). No assumption. No guess.
//!
//! --- END DOCSTRING ---

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// ════════════════════════════════════════════════════════════════════
// WitnessObservation — the wire type
// ════════════════════════════════════════════════════════════════════

/// A node's observed chain-head, sent to witness peers after every
/// seal. Day 4 ships unsigned; Day 5 adds Ed25519 binding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WitnessObservation {
    /// Identifier of the node making the observation (e.g., "node0").
    pub node_id: String,
    /// The current chain head as a 64-char lowercase hex string.
    pub chain_head_hex: String,
    /// Chain length (number of receipts) at observation time.
    pub chain_length: u64,
    /// Monotonic timestamp (nanoseconds since UNIX epoch) at observation.
    pub observed_at_ns: u64,
}

// ════════════════════════════════════════════════════════════════════
// WitnessStore — in-memory storage (Day 4 minimum)
// ════════════════════════════════════════════════════════════════════

/// In-memory store of the latest observation per node_id.
///
/// Day 4 deliberately uses `Mutex<HashMap>` with no persistence. Day 5+
/// will add a disk-backed store that survives witness-daemon restart;
/// until then, witnesses are ephemeral — a pinger must re-ping after
/// any witness restart.
#[derive(Clone, Default)]
pub struct WitnessStore {
    inner: Arc<RwLock<HashMap<String, WitnessObservation>>>,
}

impl WitnessStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the latest observation for a node. Overwrites any prior
    /// observation for the same node_id (last-write-wins; Day 4 scope).
    pub async fn record(&self, obs: WitnessObservation) {
        let mut map = self.inner.write().await;
        map.insert(obs.node_id.clone(), obs);
    }

    /// Retrieve the latest observation for a node, if any.
    pub async fn get(&self, node_id: &str) -> Option<WitnessObservation> {
        let map = self.inner.read().await;
        map.get(node_id).cloned()
    }

    /// Number of nodes currently observed (for operator visibility).
    pub async fn len(&self) -> usize {
        let map = self.inner.read().await;
        map.len()
    }

    /// Whether any observations are stored.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

// ════════════════════════════════════════════════════════════════════
// Axum handlers
// ════════════════════════════════════════════════════════════════════

/// POST /witness/head — receive a chain-head observation from a peer.
///
/// Day 4 scope: accept + store unsigned observations. No authentication,
/// no rate-limiting. Any client on the gateway's address can submit.
/// Day 5 adds Ed25519 signature verification before storage.
pub async fn post_head(
    State(store): State<WitnessStore>,
    Json(obs): Json<WitnessObservation>,
) -> (StatusCode, Json<serde_json::Value>) {
    let node_id = obs.node_id.clone();
    let chain_length = obs.chain_length;
    store.record(obs).await;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "stored": true,
            "node_id": node_id,
            "chain_length": chain_length,
        })),
    )
}

/// GET /witness/head/:node_id — retrieve the latest observation for a node.
///
/// Returns 404 if we have never received an observation for `node_id`
/// (NO_SHADOW_STATE: no fabricated response).
pub async fn get_head(
    State(store): State<WitnessStore>,
    Path(node_id): Path<String>,
) -> Result<Json<WitnessObservation>, StatusCode> {
    match store.get(&node_id).await {
        Some(obs) => Ok(Json(obs)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

// ════════════════════════════════════════════════════════════════════
// Client — ping a peer's witness endpoint
// ════════════════════════════════════════════════════════════════════

/// POST a chain-head observation to a witness peer URL.
///
/// `peer_url` should be the base URL of the peer's gateway
/// (e.g., `http://witness1.example.com:7421`). This function appends
/// `/witness/head` and POSTs the observation as JSON.
///
/// Returns `Ok(())` on 2xx; `Err(reason)` otherwise.
///
/// Day 4 scope: fire-and-confirm to a single URL. Day 5+ will add
/// parallel fan-out to multiple peers and retry/backoff policy.
pub async fn ping_witness(peer_url: &str, obs: &WitnessObservation) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("client build failed: {e}"))?;

    let url = format!("{}/witness/head", peer_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .json(obs)
        .send()
        .await
        .map_err(|e| format!("ping request failed: {e}"))?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("witness at {url} responded {}", resp.status()))
    }
}

// ════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_obs(node_id: &str, chain_length: u64, head_byte: u8) -> WitnessObservation {
        WitnessObservation {
            node_id: node_id.to_string(),
            chain_head_hex: (0..64).map(|_| head_byte as char).collect::<String>(),
            chain_length,
            observed_at_ns: 1_700_000_000_000_000_000,
        }
    }

    #[tokio::test]
    async fn store_records_and_retrieves_observation() {
        let store = WitnessStore::new();
        let obs = sample_obs("node-test-1", 42, b'a');
        store.record(obs.clone()).await;
        let retrieved = store.get("node-test-1").await.unwrap();
        assert_eq!(retrieved, obs);
    }

    #[tokio::test]
    async fn store_returns_none_for_unknown_node() {
        let store = WitnessStore::new();
        let retrieved = store.get("no-such-node").await;
        assert!(retrieved.is_none(), "NO_SHADOW_STATE: unknown nodes yield None, not a fabricated observation");
    }

    #[tokio::test]
    async fn store_overwrites_with_latest_observation() {
        let store = WitnessStore::new();
        let obs1 = sample_obs("node-x", 10, b'1');
        let obs2 = sample_obs("node-x", 20, b'2');
        store.record(obs1).await;
        store.record(obs2.clone()).await;
        let retrieved = store.get("node-x").await.unwrap();
        assert_eq!(retrieved.chain_length, 20);
        assert_eq!(retrieved.chain_head_hex, obs2.chain_head_hex);
    }

    #[tokio::test]
    async fn store_is_empty_on_construction() {
        let store = WitnessStore::new();
        assert!(store.is_empty().await);
        assert_eq!(store.len().await, 0);
    }

    #[tokio::test]
    async fn store_len_counts_distinct_nodes() {
        let store = WitnessStore::new();
        store.record(sample_obs("node-a", 1, b'a')).await;
        store.record(sample_obs("node-b", 1, b'b')).await;
        store.record(sample_obs("node-a", 2, b'a')).await; // overwrite
        assert_eq!(store.len().await, 2);
    }

    #[test]
    fn observation_json_round_trip_preserves_fields() {
        let obs = WitnessObservation {
            node_id: "n0".to_string(),
            chain_head_hex: "deadbeef".repeat(8),
            chain_length: 123,
            observed_at_ns: 42,
        };
        let json = serde_json::to_string(&obs).unwrap();
        let back: WitnessObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, obs);
    }

    #[test]
    fn observation_json_has_expected_field_names() {
        let obs = WitnessObservation {
            node_id: "n".to_string(),
            chain_head_hex: "00".repeat(32),
            chain_length: 1,
            observed_at_ns: 0,
        };
        let json = serde_json::to_value(&obs).unwrap();
        assert!(json.get("node_id").is_some());
        assert!(json.get("chain_head_hex").is_some());
        assert!(json.get("chain_length").is_some());
        assert!(json.get("observed_at_ns").is_some());
    }
}
