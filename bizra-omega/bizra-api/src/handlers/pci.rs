//! PCI Protocol Handlers — Envelope creation and verification

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use bizra_core::{PCIEnvelope, pci::gates::{default_gate_chain, GateChain, GateContext}};
use crate::{state::AppState, error::ApiError};

#[derive(Deserialize)]
pub struct CreateEnvelopeRequest {
    pub payload: Value,
    #[serde(default = "default_ttl")]
    pub ttl: u64,
    #[serde(default)]
    pub provenance: Vec<String>,
}

fn default_ttl() -> u64 { 3600 }

#[derive(Serialize)]
pub struct EnvelopeResponse {
    pub id: String,
    pub public_key: String,
    pub signature: String,
    pub ttl: u64,
    pub timestamp: String,
    pub content_hash: String,
}

/// Create a new PCI envelope
pub async fn create_envelope(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateEnvelopeRequest>,
) -> Result<Json<EnvelopeResponse>, ApiError> {
    let identity = state.identity.read().await;
    let identity = identity.as_ref().ok_or(ApiError::IdentityNotInitialized)?;

    let envelope = PCIEnvelope::create(identity, req.payload, req.ttl, req.provenance)
        .map_err(|e| ApiError::PCIVerificationFailed(format!("{:?}", e)))?;

    Ok(Json(EnvelopeResponse {
        id: envelope.id,
        public_key: envelope.public_key,
        signature: envelope.signature,
        ttl: envelope.ttl,
        timestamp: envelope.timestamp.to_rfc3339(),
        content_hash: envelope.content_hash,
    }))
}

#[derive(Deserialize)]
pub struct VerifyEnvelopeRequest {
    pub envelope: PCIEnvelope<Value>,
}

#[derive(Serialize)]
pub struct VerifyEnvelopeResponse {
    pub valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Verify a PCI envelope
pub async fn verify_envelope(
    Json(req): Json<VerifyEnvelopeRequest>,
) -> Result<Json<VerifyEnvelopeResponse>, ApiError> {
    match req.envelope.verify() {
        Ok(()) => Ok(Json(VerifyEnvelopeResponse {
            valid: true,
            error: None,
        })),
        Err(e) => Ok(Json(VerifyEnvelopeResponse {
            valid: false,
            error: Some(format!("{:?}", e)),
        })),
    }
}

#[derive(Deserialize)]
pub struct CheckGatesRequest {
    pub sender_id: String,
    pub envelope_id: String,
    pub content: String,
    #[serde(default)]
    pub snr_score: Option<f64>,
    #[serde(default)]
    pub ihsan_score: Option<f64>,
}

#[derive(Serialize)]
pub struct GateResultResponse {
    pub gate: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub duration_us: u64,
}

#[derive(Serialize)]
pub struct CheckGatesResponse {
    pub all_passed: bool,
    pub results: Vec<GateResultResponse>,
}

/// Check content against gate chain
pub async fn check_gates(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CheckGatesRequest>,
) -> Result<Json<CheckGatesResponse>, ApiError> {
    let chain = default_gate_chain();

    let ctx = GateContext {
        sender_id: req.sender_id,
        envelope_id: req.envelope_id,
        content: req.content.into_bytes(),
        constitution: state.constitution.clone(),
        snr_score: req.snr_score,
        ihsan_score: req.ihsan_score,
    };

    let results = chain.verify(&ctx);
    let all_passed = GateChain::all_passed(&results);

    let gate_results: Vec<GateResultResponse> = results
        .into_iter()
        .map(|r| GateResultResponse {
            gate: r.gate,
            passed: r.passed,
            code: if r.passed { None } else { Some(format!("{:?}", r.code)) },
            duration_us: r.duration.as_micros() as u64,
        })
        .collect();

    Ok(Json(CheckGatesResponse {
        all_passed,
        results: gate_results,
    }))
}
