//! BIZRA Sovereign API Gateway
//!
//! Unified REST/WebSocket interface for all BIZRA services:
//! - Identity management (Ed25519 keypairs)
//! - PCI Protocol (envelope creation/verification)
//! - Inference gateway (tiered model access)
//! - Federation status (gossip + consensus)
//!
//! Giants: Fielding (REST), Kleppmann (DDIA), Axum team

pub mod routes;
pub mod state;
pub mod handlers;
pub mod middleware;
pub mod error;
pub mod websocket;

pub use error::ApiError;
pub use state::AppState;

use axum::{
    Router,
    routing::{get, post},
    middleware as axum_middleware,
};
use tower_http::{
    cors::{CorsLayer, Any},
    trace::TraceLayer,
    compression::CompressionLayer,
};
use std::sync::Arc;

/// API version
pub const API_VERSION: &str = "v1";

/// Build the complete API router
pub fn build_router(state: Arc<AppState>) -> Router {
    let api_routes = Router::new()
        // Health & Status
        .route("/health", get(handlers::health::health_check))
        .route("/status", get(handlers::status::system_status))
        .route("/metrics", get(handlers::metrics::prometheus_metrics))

        // Identity
        .route("/identity/generate", post(handlers::identity::generate))
        .route("/identity/sign", post(handlers::identity::sign_message))
        .route("/identity/verify", post(handlers::identity::verify_signature))

        // PCI Protocol
        .route("/pci/envelope/create", post(handlers::pci::create_envelope))
        .route("/pci/envelope/verify", post(handlers::pci::verify_envelope))
        .route("/pci/gates/check", post(handlers::pci::check_gates))

        // Inference
        .route("/inference/generate", post(handlers::inference::generate))
        .route("/inference/models", get(handlers::inference::list_models))
        .route("/inference/tier", post(handlers::inference::select_tier))

        // Federation
        .route("/federation/status", get(handlers::federation::status))
        .route("/federation/peers", get(handlers::federation::list_peers))
        .route("/federation/propose", post(handlers::federation::propose))

        // Constitution
        .route("/constitution", get(handlers::constitution::get_constitution))
        .route("/constitution/check", post(handlers::constitution::check_compliance))

        // WebSocket for real-time updates
        .route("/ws", get(websocket::ws_handler));

    Router::new()
        .nest(&format!("/api/{}", API_VERSION), api_routes)
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        )
        .layer(axum_middleware::from_fn_with_state(
            state.clone(),
            middleware::rate_limit::rate_limiter
        ))
        .with_state(state)
}

/// Server configuration
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_metrics: bool,
    pub max_connections: usize,
    pub request_timeout_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 3001,
            enable_metrics: true,
            max_connections: 10000,
            request_timeout_ms: 30000,
        }
    }
}

/// Start the API server
pub async fn serve(config: ServerConfig, state: Arc<AppState>) -> Result<(), ApiError> {
    let router = build_router(state);
    let addr = format!("{}:{}", config.host, config.port);

    tracing::info!(
        addr = %addr,
        version = API_VERSION,
        "Starting BIZRA Sovereign API"
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
