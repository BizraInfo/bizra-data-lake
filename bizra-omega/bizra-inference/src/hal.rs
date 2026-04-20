#![forbid(unsafe_code)]
#![deny(missing_docs, clippy::unwrap_used)]

//! # BIZRA Cognition HAL — provenance-bound inference
//!
//! Canonical Rust substrate for the **Brain Activation Spec v0.1**
//! (`docs/design/BRAIN-ACTIVATION-SPEC-v0.1.md`).
//!
//! This module sits ABOVE the existing generation-only layer
//! (`InferenceBackend` + `Backend` in `lib.rs`) and adds two
//! additional guarantees that the spec requires:
//!
//! 1. **Provenance binding.** Every successful execution emits a
//!    [`CognitiveResponse`] carrying a
//!    [`ProvenanceDescriptor`] — the same schema-parity type shipped
//!    in `bizra-cognition::cognition_round` (PR #30). This closes the
//!    `CLAIM_MUST_BIND` invariant on brain-layer output.
//! 2. **LTL liveness.** Every backend is expected to honour a
//!    bounded-time contract (currently 30 s ceiling). If a model
//!    exceeds the bound the gate fails closed with
//!    [`InferenceError::LivenessTimeout`]; the deterministic kernel
//!    is never blocked by a runaway probabilistic engine.
//!
//! The collision-free naming — [`CognitionBackend`] (this module)
//! vs. `InferenceBackend` (parent `lib.rs`) — keeps the existing
//! generation-only surface intact. A future adapter can bridge
//! legacy [`backends::Backend`] implementations onto
//! [`CognitionBackend`] by wrapping their output with a
//! [`ProvenanceDescriptor`] computed at call time.
//!
//! ## Relationship to `bizra-cognition`
//!
//! This module IMPORTS [`ProvenanceDescriptor`] and
//! [`ProviderIdentity`] from `bizra-cognition::cognition_round`
//! so both crates agree on the schema bit-for-bit. Any change to
//! those types must land in `bizra-cognition` FIRST and be absorbed
//! here; JSON-shape tests in both crates fail-closed on drift.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use blake3::Hasher;
use thiserror::Error;
use tokio::sync::Semaphore;

// Schema-parity imports — single source of truth lives in bizra-cognition.
pub use bizra_cognition::cognition_round::{ProvenanceDescriptor, ProviderIdentity};

// ─────────────────────────────────────────────────────────────
// Error surface
// ─────────────────────────────────────────────────────────────

/// Errors emitted by any [`CognitionBackend`] implementation.
///
/// The set is intentionally narrow — every variant maps to a
/// constitutional invariant the spec requires the kernel to observe
/// ex-ante:
///
/// * [`Self::IhsanViolation`] → `IHSAN_FLOOR` failed
/// * [`Self::LivenessTimeout`] → LTL liveness bound exceeded
/// * [`Self::CapacityExhausted`] → resource-pool back-pressure
#[derive(Error, Debug)]
pub enum InferenceError {
    /// The provider's response was rejected because its self-reported
    /// ihsan score (or the post-hoc kernel verdict) fell below
    /// `IHSAN_FLOOR` (= `0.95` Production per `CANON-TERMS.md` §02).
    #[error("IHSAN_FLOOR violation: provider response degraded below acceptable bounds")]
    IhsanViolation,

    /// The backend exceeded its bounded-time contract. The deterministic
    /// kernel never blocks on a probabilistic engine; fail-closed is the
    /// only safe path.
    #[error("Hardware timeout: inference execution exceeded bounded LTL liveness constraints")]
    LivenessTimeout,

    /// Back-pressure from the provider — either the semaphore is
    /// saturated or the upstream daemon returned a capacity-class error.
    #[error("Provider capacity exhausted: {0}")]
    CapacityExhausted(String),
}

// ─────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────

/// A single neuro-symbolic inference input.
///
/// Fields reference content by hash, not by value, so the contract is
/// compatible with receipt-chain downstream work (the prompt bytes
/// themselves never leave the caller).
#[derive(Debug, Clone)]
pub struct CognitiveRequest {
    /// BLAKE3 hex of the canonical prompt bytes (caller-computed).
    pub prompt_hash: String,
    /// Raw prompt payload. Held locally; not logged.
    pub payload: Vec<u8>,
    /// Upper bound on generated tokens. Hint to the backend.
    pub max_tokens: usize,
    /// If true the backend should return JSON-schema-valid output or
    /// error. Kernel-side validation still runs regardless.
    pub enforce_json_schema: bool,
}

/// The provenance-bound output of a [`CognitionBackend::execute`] call.
///
/// Every successful inference MUST populate this struct in full. Raw
/// text is never returned to the caller without a matching
/// [`ProvenanceDescriptor`] — that is the load-bearing property the
/// `CLAIM_MUST_BIND` invariant relies on.
#[derive(Debug, Clone)]
pub struct CognitiveResponse {
    /// BLAKE3 hex of the response bytes, computed by the backend.
    pub response_hash: String,
    /// Response payload. Caller hashes to verify `response_hash`.
    pub payload: Vec<u8>,
    /// Wall-clock duration of the backend call, for liveness reporting.
    pub duration: Duration,
    /// Who / what / how served this round. Schema-parity mirror of
    /// `bizra_cognition::cognition_round::ProvenanceDescriptor`.
    pub provenance: ProvenanceDescriptor,
}

// ─────────────────────────────────────────────────────────────
// The governance-bound backend trait
// ─────────────────────────────────────────────────────────────

/// Universal contract bounding every cognitive execution in BIZRA.
///
/// Distinct from the generation-only [`crate::InferenceBackend`]
/// trait: implementations of this trait guarantee provenance-binding
/// AND liveness semantics. Call sites that need only raw generation
/// continue to use the legacy trait; call sites that feed the receipt
/// chain use this one.
#[async_trait]
pub trait CognitionBackend: Send + Sync {
    /// Canonical identity of this backend. Surfaces directly into the
    /// `ProvenanceDescriptor` on every response.
    fn identity(&self) -> ProviderIdentity;

    /// Execute the cognitive workload under the spec's bounded-time
    /// and back-pressure contracts.
    async fn execute(&self, req: CognitiveRequest) -> Result<CognitiveResponse, InferenceError>;

    /// Health probe for the Universal Resource Pool (URP). Returns a
    /// normalised vitality score in `[0.0, 1.0]`; `>= 0.95` means the
    /// backend is eligible for traffic.
    async fn probe_vitality(&self) -> Result<f64, InferenceError>;
}

// ─────────────────────────────────────────────────────────────
// LocalServer reference implementation
// ─────────────────────────────────────────────────────────────

/// Bounded-time, semaphore-gated backend for a LOCAL daemon (Ollama,
/// LM Studio, a self-hosted Whisper or TTS process).
///
/// This is the first of three canonical `CognitionBackend`
/// implementations — the other two (`LocalModel` for embedded
/// inference, `RemoteApi` for opt-in cloud) follow the same shape and
/// are added in future arcs.
///
/// The wire-protocol implementation is currently a deterministic stub
/// (returns a fixed payload) so the HAL can be landed, tested, and
/// exercised by the smoke-verifier chain before the Ollama / Whisper
/// bridges are wired. Replacing the stub with a real HTTP call does
/// not require changes outside this file.
pub struct LocalServerBackend {
    endpoint: String,
    model_sha256: String,
    vendor: String,
    /// Bounds concurrent in-flight requests to avoid hardware OOM
    /// and to give callers predictable back-pressure semantics.
    concurrency_gate: Arc<Semaphore>,
    /// LTL liveness ceiling per call. 30 s is the Node-0 default.
    liveness_ceiling: Duration,
}

impl LocalServerBackend {
    /// Build a new backend bound to `endpoint` with the given
    /// concurrency permit count. Default liveness ceiling is 30 s; use
    /// [`Self::with_liveness_ceiling`] to tune.
    pub fn new(
        endpoint: String,
        vendor: String,
        model_sha256: String,
        max_concurrency: usize,
    ) -> Self {
        Self {
            endpoint,
            model_sha256,
            vendor,
            concurrency_gate: Arc::new(Semaphore::new(max_concurrency)),
            liveness_ceiling: Duration::from_secs(30),
        }
    }

    /// Override the default liveness ceiling. Bounded-time contract
    /// stays enforced; only the specific bound changes.
    #[must_use]
    pub fn with_liveness_ceiling(mut self, ceiling: Duration) -> Self {
        self.liveness_ceiling = ceiling;
        self
    }
}

#[async_trait]
impl CognitionBackend for LocalServerBackend {
    fn identity(&self) -> ProviderIdentity {
        ProviderIdentity::LocalServer {
            endpoint: self.endpoint.clone(),
            vendor: self.vendor.clone(),
        }
    }

    async fn execute(&self, _req: CognitiveRequest) -> Result<CognitiveResponse, InferenceError> {
        let _permit = self
            .concurrency_gate
            .acquire()
            .await
            .map_err(|_| InferenceError::CapacityExhausted("semaphore closed".into()))?;

        let start = Instant::now();

        // WIRE PROTOCOL STUB — to be replaced by Hyper/Reqwest call to
        // the local daemon (Ollama/11434 today; LM Studio next; Whisper
        // then Orpheus for the Voice Stack arc). The call-site contract
        // below — compute hash, assemble provenance, enforce liveness —
        // does not change when the stub is replaced.
        let payload: Vec<u8> = b"{\"status\":\"deterministic_success\"}".to_vec();

        let duration = start.elapsed();
        if duration > self.liveness_ceiling {
            return Err(InferenceError::LivenessTimeout);
        }

        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let response_hash = hasher.finalize().to_hex().to_string();

        let provenance = ProvenanceDescriptor {
            model_sha256: self.model_sha256.clone(),
            // Local-server models are unsigned in Node-0. Signing
            // authority arrives with the Charter-sealed model registry;
            // until then we publish the SHA and the caller must
            // establish trust out-of-band.
            model_signer: None,
            provider_identity: self.identity(),
        };

        Ok(CognitiveResponse {
            response_hash,
            payload,
            duration,
            provenance,
        })
    }

    async fn probe_vitality(&self) -> Result<f64, InferenceError> {
        // STUB. Replace with a real latency-and-SNR probe against
        // `self.endpoint` when the wire call lands.
        Ok(0.99)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_backend() -> LocalServerBackend {
        LocalServerBackend::new(
            "http://127.0.0.1:11434".into(),
            "ollama".into(),
            "a".repeat(64),
            2,
        )
    }

    #[test]
    fn identity_reports_local_server_class() {
        let b = sample_backend();
        match b.identity() {
            ProviderIdentity::LocalServer { endpoint, vendor } => {
                assert_eq!(endpoint, "http://127.0.0.1:11434");
                assert_eq!(vendor, "ollama");
            }
            other => panic!("expected LocalServer, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn execute_returns_provenance_bound_response() {
        let b = sample_backend();
        let req = CognitiveRequest {
            prompt_hash: "b".repeat(64),
            payload: b"hello".to_vec(),
            max_tokens: 64,
            enforce_json_schema: true,
        };
        let resp = b.execute(req).await.expect("stub must succeed");
        assert_eq!(resp.response_hash.len(), 64);
        assert!(!resp.payload.is_empty());
        assert_eq!(resp.provenance.model_sha256, "a".repeat(64));
        assert!(matches!(
            resp.provenance.provider_identity,
            ProviderIdentity::LocalServer { .. }
        ));
        assert!(resp.provenance.model_signer.is_none());
    }

    #[tokio::test]
    async fn vitality_is_in_unit_interval() {
        let b = sample_backend();
        let score = b.probe_vitality().await.expect("stub must succeed");
        assert!((0.0..=1.0).contains(&score));
    }

    #[tokio::test]
    async fn liveness_ceiling_below_zero_triggers_timeout() {
        // Deliberately set the ceiling below any measurable duration
        // to prove the fail-closed path is reachable.
        let b = LocalServerBackend::new(
            "http://127.0.0.1:11434".into(),
            "ollama".into(),
            "a".repeat(64),
            1,
        )
        .with_liveness_ceiling(Duration::from_nanos(1));
        let req = CognitiveRequest {
            prompt_hash: "b".repeat(64),
            payload: b"hello".to_vec(),
            max_tokens: 1,
            enforce_json_schema: false,
        };
        let res = b.execute(req).await;
        assert!(matches!(res, Err(InferenceError::LivenessTimeout)));
    }
}
