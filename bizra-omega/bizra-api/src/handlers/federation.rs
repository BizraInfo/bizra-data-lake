//! Federation Handlers — Gossip and consensus status

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{error::ApiError, state::AppState};

#[derive(Serialize)]
pub struct FederationStatus {
    pub connected: bool,
    pub node_id: Option<String>,
    pub peer_count: usize,
    pub gossip_round: u64,
    pub consensus_height: u64,
    pub pattern_count: usize,
}

/// Get federation status
pub async fn status(State(state): State<Arc<AppState>>) -> Json<FederationStatus> {
    let federation = state.gossip.read().await;
    let identity = state.identity.read().await;

    if let Some(_fed) = federation.as_ref() {
        // In real implementation, query the federation node
        Json(FederationStatus {
            connected: true,
            node_id: identity.as_ref().map(|i| i.node_id().0.clone()),
            peer_count: 0,
            gossip_round: 0,
            consensus_height: 0,
            pattern_count: 0,
        })
    } else {
        Json(FederationStatus {
            connected: false,
            node_id: identity.as_ref().map(|i| i.node_id().0.clone()),
            peer_count: 0,
            gossip_round: 0,
            consensus_height: 0,
            pattern_count: 0,
        })
    }
}

#[derive(Serialize)]
pub struct PeerInfo {
    pub node_id: String,
    pub address: String,
    pub state: String,
    pub last_seen: String,
    pub incarnation: u64,
}

#[derive(Serialize)]
pub struct ListPeersResponse {
    pub peers: Vec<PeerInfo>,
}

/// List federation peers
pub async fn list_peers(State(state): State<Arc<AppState>>) -> Json<ListPeersResponse> {
    let federation = state.gossip.read().await;

    if let Some(_fed) = federation.as_ref() {
        // In real implementation, get members from gossip protocol
        Json(ListPeersResponse { peers: vec![] })
    } else {
        Json(ListPeersResponse { peers: vec![] })
    }
}

#[derive(Deserialize)]
pub struct ProposeRequest {
    pub pattern_id: String,
    pub pattern_embedding: Vec<f32>,
    pub source_node: String,
    pub confidence: f64,
}

#[derive(Serialize)]
pub struct ProposeResponse {
    pub proposal_id: String,
    pub status: String,
    pub quorum_needed: usize,
    pub votes_received: usize,
}

/// Propose a pattern for elevation
pub async fn propose(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProposeRequest>,
) -> Result<Json<ProposeResponse>, ApiError> {
    let federation = state.gossip.read().await;

    if federation.is_none() {
        return Err(ApiError::FederationError("Federation not connected".into()));
    }

    // Validate confidence meets Ihsan threshold
    if req.confidence < bizra_core::IHSAN_THRESHOLD {
        return Err(ApiError::ConstitutionViolation(format!(
            "Confidence {} below Ihsan threshold {}",
            req.confidence,
            bizra_core::IHSAN_THRESHOLD
        )));
    }

    // In real implementation, submit to consensus engine
    let proposal_id = format!(
        "prop_{}",
        uuid::Uuid::new_v4().to_string().split('-').next().unwrap()
    );

    Ok(Json(ProposeResponse {
        proposal_id,
        status: "submitted".into(),
        quorum_needed: 3,  // Example: 2f+1 with f=1
        votes_received: 1, // Self-vote
    }))
}
