//! Comprehensive tests for bizra-api — state, config, error, router construction
//!
//! Phase 13: Test Sprint

use axum::http::StatusCode;
use axum::response::IntoResponse;
use bizra_api::error::ApiError;
use bizra_api::{AppState, ServerConfig, API_VERSION};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// ServerConfig
// ---------------------------------------------------------------------------

#[test]
fn server_config_default_values() {
    let cfg = ServerConfig::default();
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.port, 3001);
    assert!(cfg.enable_metrics);
    assert_eq!(cfg.max_connections, 10000);
    assert_eq!(cfg.request_timeout_ms, 30000);
}

#[test]
fn server_config_clone() {
    let cfg = ServerConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cloned.port, 3001);
}

#[test]
fn api_version_is_v1() {
    assert_eq!(API_VERSION, "v1");
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

#[test]
fn app_state_default_no_identity() {
    let state = AppState::default();
    assert_eq!(state.get_request_count(), 0);
    assert!(state.uptime_secs() < 2); // Just created
}

#[test]
fn app_state_increment_requests() {
    let state = AppState::default();
    state.increment_requests();
    state.increment_requests();
    state.increment_requests();
    assert_eq!(state.get_request_count(), 3);
}

#[test]
fn app_state_uptime_increases() {
    let state = AppState::default();
    // Uptime should be 0 or very close
    assert!(state.uptime_secs() < 2);
}

#[tokio::test]
async fn app_state_with_identity() {
    use bizra_core::NodeIdentity;
    let state = AppState::default();
    let identity = NodeIdentity::generate();
    let node_id = identity.node_id().0.clone();
    let state = state.with_identity(identity).await;
    let locked = state.identity.read().await;
    assert!(locked.is_some());
    assert_eq!(locked.as_ref().unwrap().node_id().0, node_id);
}

// ---------------------------------------------------------------------------
// ApiError → StatusCode mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn api_error_identity_not_initialized_503() {
    let err = ApiError::IdentityNotInitialized;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn api_error_invalid_signature_401() {
    let err = ApiError::InvalidSignature;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn api_error_pci_verification_400() {
    let err = ApiError::PCIVerificationFailed("test".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn api_error_gate_check_403() {
    let err = ApiError::GateCheckFailed("gate".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn api_error_inference_500() {
    let err = ApiError::InferenceError("timeout".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn api_error_federation_500() {
    let err = ApiError::FederationError("disconnect".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn api_error_constitution_violation_403() {
    let err = ApiError::ConstitutionViolation("ihsan".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn api_error_rate_limit_429() {
    let err = ApiError::RateLimitExceeded;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn api_error_bad_request_400() {
    let err = ApiError::BadRequest("missing field".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn api_error_internal_500() {
    let err = ApiError::Internal("panic".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

// ---------------------------------------------------------------------------
// Router construction — doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn build_router_does_not_panic() {
    let state = Arc::new(AppState::default());
    let _router = bizra_api::build_router(state);
}

// ---------------------------------------------------------------------------
// Health handler
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_check_returns_healthy() {
    use bizra_api::handlers::health::health_check;
    let response = health_check().await;
    assert_eq!(response.0.status, "healthy");
}

// ---------------------------------------------------------------------------
// Status handler
// ---------------------------------------------------------------------------

#[tokio::test]
async fn system_status_no_identity() {
    use axum::extract::State;
    use bizra_api::handlers::status::system_status;

    let state = Arc::new(AppState::default());
    let resp = system_status(State(state)).await;
    assert!(resp.0.node_id.is_none());
    assert!(!resp.0.identity_initialized);
    assert!(!resp.0.inference_ready);
    assert!(!resp.0.federation_connected);
}

#[tokio::test]
async fn system_status_with_identity() {
    use axum::extract::State;
    use bizra_api::handlers::status::system_status;
    use bizra_core::NodeIdentity;

    let state = AppState::default();
    let identity = NodeIdentity::generate();
    let state = Arc::new(state.with_identity(identity).await);
    let resp = system_status(State(state)).await;
    assert!(resp.0.node_id.is_some());
    assert!(resp.0.identity_initialized);
}
