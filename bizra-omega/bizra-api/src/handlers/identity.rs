//! Identity Handlers — Ed25519 key management

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{error::ApiError, state::AppState};
use bizra_core::NodeIdentity;

#[derive(Serialize)]
pub struct GenerateResponse {
    pub node_id: String,
    pub public_key: String,
}

/// Generate new identity
pub async fn generate(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GenerateResponse>, ApiError> {
    let identity = NodeIdentity::generate();
    let response = GenerateResponse {
        node_id: identity.node_id().0.clone(),
        public_key: identity.public_key_hex(),
    };

    *state.identity.write().await = Some(identity);

    tracing::info!(node_id = %response.node_id, "Identity generated");
    Ok(Json(response))
}

#[derive(Deserialize)]
pub struct SignRequest {
    pub message: String,
}

#[derive(Serialize)]
pub struct SignResponse {
    pub signature: String,
    pub public_key: String,
}

/// Sign a message
pub async fn sign_message(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SignRequest>,
) -> Result<Json<SignResponse>, ApiError> {
    let identity = state.identity.read().await;
    let identity = identity.as_ref().ok_or(ApiError::IdentityNotInitialized)?;

    let signature = identity.sign(req.message.as_bytes());

    Ok(Json(SignResponse {
        signature,
        public_key: identity.public_key_hex(),
    }))
}

#[derive(Deserialize)]
pub struct VerifyRequest {
    pub message: String,
    pub signature: String,
    pub public_key: String,
}

#[derive(Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
}

/// Verify a signature
pub async fn verify_signature(
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, ApiError> {
    // Use verify_with_hex which takes hex strings directly
    let valid =
        NodeIdentity::verify_with_hex(req.message.as_bytes(), &req.signature, &req.public_key);

    Ok(Json(VerifyResponse { valid }))
}
