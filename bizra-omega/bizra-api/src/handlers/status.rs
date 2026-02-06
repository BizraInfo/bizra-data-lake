//! System Status Handler

use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::state::AppState;

#[derive(Serialize)]
pub struct SystemStatus {
    pub node_id: Option<String>,
    pub uptime_secs: u64,
    pub total_requests: u64,
    pub identity_initialized: bool,
    pub inference_ready: bool,
    pub federation_connected: bool,
    pub constitution_version: String,
}

pub async fn system_status(
    State(state): State<Arc<AppState>>,
) -> Json<SystemStatus> {
    let identity = state.identity.read().await;
    let inference = state.inference.read().await;
    let federation = state.gossip.read().await;

    Json(SystemStatus {
        node_id: identity.as_ref().map(|i| i.node_id().0.clone()),
        uptime_secs: state.uptime_secs(),
        total_requests: state.get_request_count(),
        identity_initialized: identity.is_some(),
        inference_ready: inference.is_some(),
        federation_connected: federation.is_some(),
        constitution_version: state.constitution.version.clone(),
    })
}
