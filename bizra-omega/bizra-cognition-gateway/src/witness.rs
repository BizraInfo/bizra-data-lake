//! BIZRA witness-grade chain-head observation — Cycle-8 Days 4+5
//!
//! بسم الله الرحمن الرحيم
//!
//! Closes the 4th (economic) modality of the Four-Modality Golden
//! Standard in witness-grade form: nodes ping allowlisted peers with
//! their sealed chain-head (Ed25519-signed per Day 5); peers verify
//! the signature, store the observation, and serve it on GET. Any
//! skeptical stranger can query a witness to detect tampering — if
//! Node A claims chain_head X, but a witness reports chain_head Y
//! for the same timestamp, the divergence is publicly detectable in
//! bounded time, and the signed observation is transferable evidence.
//!
//! --- DOCTRINAL CONSTRAINT (Cycle-8, 2026-04-19) ---
//!
//! T=0 economic finality is WITNESS-GRADE DETECTABILITY ONLY. These
//! primitives are Horizon / Layer B, NOT Day 4/5 scope:
//!   - bonded stakes
//!   - slashing mechanisms
//!   - DAO governance
//!   - challenge-period economics
//!   - token system
//!
//! Witness-grade closure = divergence is detectable, transferable
//! (anyone can produce the proof of mismatch), and bounded in cost to
//! verify. That is the T=0 fourth modality. Nothing more.
//!
//! --- DAY 5 UPGRADE: Ed25519 signatures ---
//!
//! Day 4 stored unsigned observations. Day 5 requires every accepted
//! observation to carry a valid Ed25519 signature over the canonical
//! bytes of its payload. Witnesses reject observations whose signature
//! fails verification (CLAIM_MUST_BIND at the wire boundary).
//!
//! Canonical signing bytes layout (deterministic; used for both sign
//! and verify):
//!   node_id_len (4 LE) | node_id_utf8
//!   chain_head_hex_len (4 LE) | chain_head_hex_utf8
//!   chain_length (8 LE)
//!   observed_at_ns (8 LE)
//!
//! --- NON-GOALS TODAY (deferred to named later days) ---
//!
//!   - Disk-persistent witness store (Day 6+; in-memory Mutex<HashMap>).
//!   - Witness peer auto-discovery (Horizon).
//!   - Byzantine-tolerant consensus among witnesses (Horizon).
//!   - Challenge/dispute protocol (Horizon).
//!   - Key rotation, revocation, TTL (Horizon; Day 5 keys are per-run).
//!
//! --- CONSTITUTIONAL ALIGNMENT ---
//!
//! - CLAIM_MUST_BIND: Day 5 signature cryptographically binds the
//!   observation to the signing key. Invalid or missing signature →
//!   store refuses. The store cannot be populated by an unbound claim.
//! - NO_SHADOW_STATE: the store echoes what was received verbatim; no
//!   derived state. GET for a node we have never heard from returns
//!   404 — NEVER a fabricated observation.
//! - ZANN_ZERO: witness refuses to answer about a node it has never
//!   received a valid signed observation for. No assumption.
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
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// ════════════════════════════════════════════════════════════════════
// WitnessObservation — the unsigned wire payload
// ════════════════════════════════════════════════════════════════════

/// A node's observed chain-head. The unsigned payload shape.
/// On the wire, this is always wrapped in `SignedWitnessObservation`.
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

impl WitnessObservation {
    /// Produce the canonical byte sequence signed over. Deterministic:
    /// identical observation produces identical bytes on every machine.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            4 + self.node_id.len() + 4 + self.chain_head_hex.len() + 8 + 8,
        );
        buf.extend_from_slice(&(self.node_id.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.node_id.as_bytes());
        buf.extend_from_slice(&(self.chain_head_hex.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.chain_head_hex.as_bytes());
        buf.extend_from_slice(&self.chain_length.to_le_bytes());
        buf.extend_from_slice(&self.observed_at_ns.to_le_bytes());
        buf
    }
}

// ════════════════════════════════════════════════════════════════════
// SignedWitnessObservation — what actually flows on the wire
// ════════════════════════════════════════════════════════════════════

/// A witness observation bound to its signing key by Ed25519.
///
/// CLAIM_MUST_BIND: the signature cryptographically binds the payload
/// to the declared public key; any tampering with the observation
/// invalidates the signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedWitnessObservation {
    pub observation: WitnessObservation,
    /// Ed25519 public key of the signer — 64-char lowercase hex (32 bytes).
    pub pubkey_hex: String,
    /// Ed25519 signature over `observation.canonical_bytes()` —
    /// 128-char lowercase hex (64 bytes).
    pub signature_hex: String,
}

impl SignedWitnessObservation {
    /// Verify the Ed25519 signature against the declared public key
    /// over the observation's canonical bytes. Returns `Ok(())` on
    /// valid signature; `Err(reason)` otherwise.
    pub fn verify(&self) -> Result<(), String> {
        let pk_bytes = hex::decode(&self.pubkey_hex)
            .map_err(|e| format!("pubkey_hex decode: {e}"))?;
        if pk_bytes.len() != 32 {
            return Err(format!("pubkey must be 32 bytes, got {}", pk_bytes.len()));
        }
        let pk_arr: [u8; 32] = pk_bytes
            .as_slice()
            .try_into()
            .map_err(|_| "pubkey slice conversion failed".to_string())?;
        let verifying_key = VerifyingKey::from_bytes(&pk_arr)
            .map_err(|e| format!("pubkey parse: {e}"))?;

        let sig_bytes = hex::decode(&self.signature_hex)
            .map_err(|e| format!("signature_hex decode: {e}"))?;
        if sig_bytes.len() != 64 {
            return Err(format!("signature must be 64 bytes, got {}", sig_bytes.len()));
        }
        let sig_arr: [u8; 64] = sig_bytes
            .as_slice()
            .try_into()
            .map_err(|_| "signature slice conversion failed".to_string())?;
        let signature = Signature::from_bytes(&sig_arr);

        verifying_key
            .verify(&self.observation.canonical_bytes(), &signature)
            .map_err(|e| format!("signature verify failed: {e}"))
    }
}

/// Sign an observation with the given Ed25519 SigningKey.
/// Deterministic over (observation, key).
pub fn sign_observation(
    observation: WitnessObservation,
    key: &SigningKey,
) -> SignedWitnessObservation {
    let canonical = observation.canonical_bytes();
    let signature: Signature = key.sign(&canonical);
    let pubkey: VerifyingKey = key.verifying_key();
    SignedWitnessObservation {
        observation,
        pubkey_hex: hex::encode(pubkey.to_bytes()),
        signature_hex: hex::encode(signature.to_bytes()),
    }
}

// ════════════════════════════════════════════════════════════════════
// WitnessStore — in-memory storage (Days 4+5 minimum)
// ════════════════════════════════════════════════════════════════════

/// In-memory store of the latest signed observation per node_id.
///
/// Day 5 stores `SignedWitnessObservation` (the signed wire form) so
/// GET returns the FULL transferable evidence — anyone can re-verify
/// the signature independently of this store.
#[derive(Clone, Default)]
pub struct WitnessStore {
    inner: Arc<RwLock<HashMap<String, SignedWitnessObservation>>>,
}

impl WitnessStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the latest signed observation for a node. Caller MUST
    /// verify the signature before calling this — the store does NOT
    /// re-verify. (Verification is the route handler's responsibility
    /// so the error can become an HTTP 401.)
    pub async fn record(&self, signed: SignedWitnessObservation) {
        let mut map = self.inner.write().await;
        map.insert(signed.observation.node_id.clone(), signed);
    }

    /// Retrieve the latest signed observation for a node, if any.
    pub async fn get(&self, node_id: &str) -> Option<SignedWitnessObservation> {
        let map = self.inner.read().await;
        map.get(node_id).cloned()
    }

    /// Number of nodes currently observed.
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

/// POST /witness/head — receive a signed observation from a peer.
///
/// Day 5 flow: verify Ed25519 signature first, reject with 401 on
/// failure (CLAIM_MUST_BIND enforced at the wire), store only if
/// signature is valid.
pub async fn post_head(
    State(store): State<WitnessStore>,
    Json(signed): Json<SignedWitnessObservation>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(reason) = signed.verify() {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "stored": false,
                "error": {
                    "code": "SIGNATURE_INVALID",
                    "message": reason,
                },
            })),
        );
    }

    let node_id = signed.observation.node_id.clone();
    let chain_length = signed.observation.chain_length;
    store.record(signed).await;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "stored": true,
            "node_id": node_id,
            "chain_length": chain_length,
            "verified": true,
        })),
    )
}

/// GET /witness/head/:node_id — retrieve the latest signed observation.
///
/// Returns 404 if no observation has been received for `node_id`
/// (NO_SHADOW_STATE: no fabricated response).
pub async fn get_head(
    State(store): State<WitnessStore>,
    Path(node_id): Path<String>,
) -> Result<Json<SignedWitnessObservation>, StatusCode> {
    match store.get(&node_id).await {
        Some(signed) => Ok(Json(signed)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

// ════════════════════════════════════════════════════════════════════
// Client — ping a peer's witness endpoint
// ════════════════════════════════════════════════════════════════════

/// Sign + POST an observation to a witness peer URL.
///
/// Day 5 contract: the client signs the observation with the provided
/// key before sending. Receiving witnesses will reject with 401 if the
/// signature does not verify.
pub async fn ping_witness(
    peer_url: &str,
    observation: WitnessObservation,
    key: &SigningKey,
) -> Result<(), String> {
    let signed = sign_observation(observation, key);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("client build failed: {e}"))?;

    let url = format!("{}/witness/head", peer_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .json(&signed)
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
    use rand::rngs::OsRng;

    fn sample_obs(node_id: &str, chain_length: u64, head_byte: u8) -> WitnessObservation {
        WitnessObservation {
            node_id: node_id.to_string(),
            chain_head_hex: (0..64).map(|_| head_byte as char).collect::<String>(),
            chain_length,
            observed_at_ns: 1_700_000_000_000_000_000,
        }
    }

    fn test_signing_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    #[tokio::test]
    async fn store_records_and_retrieves_signed_observation() {
        let store = WitnessStore::new();
        let key = test_signing_key();
        let signed = sign_observation(sample_obs("node-test-1", 42, b'a'), &key);
        store.record(signed.clone()).await;
        let retrieved = store.get("node-test-1").await.unwrap();
        assert_eq!(retrieved, signed);
    }

    #[tokio::test]
    async fn store_returns_none_for_unknown_node() {
        let store = WitnessStore::new();
        let retrieved = store.get("no-such-node").await;
        assert!(
            retrieved.is_none(),
            "NO_SHADOW_STATE: unknown nodes yield None"
        );
    }

    #[tokio::test]
    async fn store_overwrites_with_latest_observation() {
        let store = WitnessStore::new();
        let key = test_signing_key();
        let s1 = sign_observation(sample_obs("node-x", 10, b'1'), &key);
        let s2 = sign_observation(sample_obs("node-x", 20, b'2'), &key);
        store.record(s1).await;
        store.record(s2.clone()).await;
        let retrieved = store.get("node-x").await.unwrap();
        assert_eq!(retrieved.observation.chain_length, 20);
        assert_eq!(retrieved, s2);
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
        let key = test_signing_key();
        store
            .record(sign_observation(sample_obs("node-a", 1, b'a'), &key))
            .await;
        store
            .record(sign_observation(sample_obs("node-b", 1, b'b'), &key))
            .await;
        store
            .record(sign_observation(sample_obs("node-a", 2, b'a'), &key))
            .await;
        assert_eq!(store.len().await, 2);
    }

    #[test]
    fn canonical_bytes_is_deterministic() {
        let o1 = sample_obs("n", 7, b'x');
        let o2 = sample_obs("n", 7, b'x');
        assert_eq!(o1.canonical_bytes(), o2.canonical_bytes());
    }

    #[test]
    fn canonical_bytes_changes_on_field_change() {
        let base = sample_obs("n", 7, b'x');
        let mut tamper_node = base.clone();
        tamper_node.node_id = "m".to_string();
        assert_ne!(base.canonical_bytes(), tamper_node.canonical_bytes());

        let mut tamper_head = base.clone();
        tamper_head.chain_head_hex = (0..64).map(|_| 'y').collect::<String>();
        assert_ne!(base.canonical_bytes(), tamper_head.canonical_bytes());

        let mut tamper_len = base.clone();
        tamper_len.chain_length = 8;
        assert_ne!(base.canonical_bytes(), tamper_len.canonical_bytes());

        let mut tamper_ts = base.clone();
        tamper_ts.observed_at_ns = 42;
        assert_ne!(base.canonical_bytes(), tamper_ts.canonical_bytes());
    }

    #[test]
    fn sign_and_verify_round_trip() {
        let key = test_signing_key();
        let signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        signed
            .verify()
            .expect("a freshly signed observation must verify");
    }

    #[test]
    fn verify_rejects_tampered_chain_head() {
        let key = test_signing_key();
        let mut signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        // Tamper AFTER signing — signature should no longer verify.
        signed.observation.chain_head_hex = (0..64).map(|_| 'z').collect::<String>();
        assert!(
            signed.verify().is_err(),
            "CLAIM_MUST_BIND: tampered payload must fail verification"
        );
    }

    #[test]
    fn verify_rejects_tampered_chain_length() {
        let key = test_signing_key();
        let mut signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        signed.observation.chain_length = 99_999;
        assert!(signed.verify().is_err());
    }

    #[test]
    fn verify_rejects_tampered_signature_hex() {
        let key = test_signing_key();
        let mut signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        // Flip one hex character in the signature.
        let mut bytes: Vec<u8> = signed.signature_hex.as_bytes().to_vec();
        bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
        signed.signature_hex = String::from_utf8(bytes).unwrap();
        assert!(signed.verify().is_err());
    }

    #[test]
    fn verify_rejects_pubkey_swap() {
        let key_a = test_signing_key();
        let key_b = test_signing_key();
        let mut signed = sign_observation(sample_obs("node0", 1, b'a'), &key_a);
        // Replace pubkey with a different identity — signature no longer binds.
        signed.pubkey_hex = hex::encode(key_b.verifying_key().to_bytes());
        assert!(signed.verify().is_err());
    }

    #[test]
    fn verify_rejects_malformed_pubkey() {
        let key = test_signing_key();
        let mut signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        signed.pubkey_hex = "not-hex".to_string();
        assert!(signed.verify().is_err());
    }

    #[test]
    fn verify_rejects_wrong_length_pubkey() {
        let key = test_signing_key();
        let mut signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        signed.pubkey_hex = "aa".repeat(16); // 16 bytes, not 32
        assert!(signed.verify().is_err());
    }

    #[test]
    fn signed_observation_json_round_trip() {
        let key = test_signing_key();
        let signed = sign_observation(sample_obs("node0", 1, b'a'), &key);
        let json = serde_json::to_string(&signed).unwrap();
        let back: SignedWitnessObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, signed);
        // And the round-tripped copy still verifies.
        back.verify().unwrap();
    }

    #[test]
    fn same_key_produces_consistent_pubkey() {
        let key = test_signing_key();
        let s1 = sign_observation(sample_obs("node-a", 1, b'a'), &key);
        let s2 = sign_observation(sample_obs("node-b", 2, b'b'), &key);
        assert_eq!(s1.pubkey_hex, s2.pubkey_hex);
    }
}
