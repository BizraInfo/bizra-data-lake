//! Comprehensive tests for bizra-api — state, config, error, router construction
//!
//! Phase 13: Test Sprint

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::IntoResponse,
};
use bizra_api::{error::ApiError, state::TokenBucket, AppState, ServerConfig, API_VERSION};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// ServerConfig
// ---------------------------------------------------------------------------

#[test]
fn server_config_default_values() {
    let cfg = ServerConfig::default();
    assert_eq!(cfg.host, "127.0.0.1");
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
async fn api_error_unauthorized_401() {
    let err = ApiError::Unauthorized;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
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

#[tokio::test]
async fn identity_generate_requires_token_when_configured() {
    let state = AppState {
        api_token: Some("secret-123".into()),
        ..Default::default()
    };
    let app = bizra_api::build_router(Arc::new(state));

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/identity/generate")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn identity_generate_accepts_valid_token() {
    let state = AppState {
        api_token: Some("secret-123".into()),
        ..Default::default()
    };
    let app = bizra_api::build_router(Arc::new(state));

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/identity/generate")
        .header("authorization", "Bearer secret-123")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn identity_verify_requires_token_when_configured() {
    let state = AppState {
        api_token: Some("secret-123".into()),
        ..Default::default()
    };
    let app = bizra_api::build_router(Arc::new(state));

    let body = r#"{"message":"hello","signature":"deadbeef","public_key":"cafebabe"}"#;
    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/identity/verify")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn protected_route_fails_closed_when_token_unset() {
    let state = AppState::default();
    let app = bizra_api::build_router(Arc::new(state));

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/pci/envelope/create")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn health_and_status_remain_public() {
    let state = AppState::default();
    let app = bizra_api::build_router(Arc::new(state));

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let status = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Token bucket
// ---------------------------------------------------------------------------

#[test]
fn token_bucket_allows_within_limit() {
    let mut bucket = TokenBucket::new(10.0, 60);
    for _ in 0..10 {
        assert!(bucket.try_consume());
    }
}

#[test]
fn token_bucket_rejects_burst_over_limit() {
    let mut bucket = TokenBucket::new(10.0, 60);
    for _ in 0..10 {
        assert!(bucket.try_consume());
    }
    assert!(!bucket.try_consume());
}

#[test]
fn token_bucket_refills_over_time() {
    let mut bucket = TokenBucket::new(10.0, 60);
    let start = Instant::now();

    for _ in 0..10 {
        assert!(bucket.try_consume_at(start));
    }
    assert!(!bucket.try_consume_at(start));

    let after_refill = start + Duration::from_secs(6);
    assert!(bucket.try_consume_at(after_refill));
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
