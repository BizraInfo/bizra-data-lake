// src/http.rs - HTTP API Server

use crate::{
    ihsan,
    pat_enhanced::EnhancedPATOrchestrator,
    types::{AdapterModes, DualAgenticRequest, DualAgenticResponse, EnhancedDualAgenticRequest},
    MetaAlphaDualAgentic,
};
use axum::{
    extract::State,
    http::{header, HeaderMap, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::{collections::HashSet, sync::Arc};
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};
use uuid::Uuid;

pub async fn create_http_server(
    system: Arc<MetaAlphaDualAgentic>,
    port: u16,
) -> anyhow::Result<()> {
    let enhanced_pat = Arc::new(EnhancedPATOrchestrator::new().await?);

    let (api_token, api_token_generated) = api_token_from_env_or_generate();
    if api_token_generated {
        warn!(
            "BIZRA_API_TOKEN not set; generated ephemeral token for this run: {}",
            api_token
        );
    }

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/dual/execute", post(execute_dual))
        .route("/enhanced/execute", post(execute_enhanced))
        .route("/stats", get(stats))
        .layer(cors_layer())
        .layer(TraceLayer::new_for_http())
        .with_state((system, enhanced_pat, api_token));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

    info!("🌐 HTTP Server listening on http://127.0.0.1:{}", port);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> impl IntoResponse {
    let constitution = ihsan::constitution();
    let ihsan_env = ihsan::current_env();
    let ihsan_artifact_class = "docs";
    let ihsan_threshold_applied = constitution.threshold_for(&ihsan_env, ihsan_artifact_class);
    Json(serde_json::json!({
        "name": "BIZRA META ALPHA ELITE - Complete Unified System",
        "version": "2.0.0",
        "architecture": "PAT(7) + SAT(5) + Full Arsenal",
        "capabilities": [
            "MCP Integration",
            "A2A Protocol",
            "Multi-Reasoning (CoT, ToT, GoT, ReAct, Reflexion)",
            "Sub-Agent Spawning",
            "Swarm Intelligence",
            "Hook System",
            "Slash Commands",
        ],
        "status": "EXPERIMENTAL",
        "adapter_modes": AdapterModes::current(),
        "ihsan": {
            "constitution_id": constitution.id(),
            "threshold_baseline": constitution.threshold(),
            "env": ihsan_env,
            "artifact_class": ihsan_artifact_class,
            "threshold_applied": ihsan_threshold_applied,
        },
        "truth": {
            "capabilities": "SIMULATED_BY_DEFAULT",
        },
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn stats(
    State((_system, _, _)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
) -> impl IntoResponse {
    let constitution = ihsan::constitution();
    let ihsan_env = ihsan::current_env();
    let ihsan_artifact_class = "docs";
    let ihsan_threshold_applied = constitution.threshold_for(&ihsan_env, ihsan_artifact_class);
    Json(serde_json::json!({
        "pat_agents": 7,
        "sat_agents": 5,
        "total_agents": 12,
        "reasoning_methods": 5,
        "mcp_tools": 4,
        "uptime": "operational",
        "adapter_modes": AdapterModes::current(),
        "ihsan_constitution_id": constitution.id(),
        "ihsan_threshold_baseline": constitution.threshold(),
        "ihsan_env": ihsan_env,
        "ihsan_artifact_class": ihsan_artifact_class,
        "ihsan_threshold_applied": ihsan_threshold_applied,
    }))
}

async fn execute_dual(
    State((system, _, api_token)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
    headers: HeaderMap,
    Json(request): Json<DualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, String)> {
    if !is_authorized(&headers, api_token.as_ref()) {
        return Err((
            StatusCode::UNAUTHORIZED,
            "Unauthorized: missing or invalid API token".to_string(),
        ));
    }

    match system.execute(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Execution failed: {}", e),
        )),
    }
}

async fn execute_enhanced(
    State((_, enhanced_pat, api_token)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
    headers: HeaderMap,
    Json(request): Json<EnhancedDualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, String)> {
    if !is_authorized(&headers, api_token.as_ref()) {
        return Err((
            StatusCode::UNAUTHORIZED,
            "Unauthorized: missing or invalid API token".to_string(),
        ));
    }

    match enhanced_pat.execute_enhanced(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Enhanced execution failed: {}", e),
        )),
    }
}

fn api_token_from_env_or_generate() -> (Arc<str>, bool) {
    match std::env::var("BIZRA_API_TOKEN") {
        Ok(v) if !v.trim().is_empty() => (Arc::<str>::from(v.trim().to_string()), false),
        _ => (Arc::<str>::from(Uuid::new_v4().simple().to_string()), true),
    }
}

fn parse_extra_cors_origins() -> HashSet<String> {
    let mut set = HashSet::new();
    let Some(raw) = std::env::var("BIZRA_CORS_ORIGINS").ok() else {
        return set;
    };

    for item in raw.split(',') {
        let origin = item.trim();
        if origin.is_empty() {
            continue;
        }
        set.insert(origin.to_string());
    }

    set
}

fn cors_layer() -> CorsLayer {
    let extra = Arc::new(parse_extra_cors_origins());

    CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::HeaderName::from_static("x-bizra-token"),
        ])
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            is_loopback_origin(origin) || origin.to_str().ok().is_some_and(|s| extra.contains(s))
        }))
}

fn is_loopback_origin(origin: &header::HeaderValue) -> bool {
    let Ok(origin_str) = origin.to_str() else {
        return false;
    };

    let lower = origin_str.to_ascii_lowercase();
    let without_scheme = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"))
        .unwrap_or(lower.as_str());

    let host_port = without_scheme.split('/').next().unwrap_or_default();
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split(']').next().unwrap_or_default()
    } else {
        host_port.split(':').next().unwrap_or_default()
    };

    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

fn extract_presented_token(headers: &HeaderMap) -> Option<String> {
    if let Some(authz) = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
    {
        if let Some(token) = authz
            .strip_prefix("Bearer ")
            .or_else(|| authz.strip_prefix("bearer "))
        {
            let trimmed = token.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    if let Some(tok) = headers
        .get("x-bizra-token")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.trim())
    {
        if !tok.is_empty() {
            return Some(tok.to_string());
        }
    }

    None
}

fn is_authorized(headers: &HeaderMap, expected: &str) -> bool {
    let Some(presented) = extract_presented_token(headers) else {
        return false;
    };
    presented == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_origin_predicate_is_reasonable() {
        for origin in [
            "http://localhost:5173",
            "https://localhost",
            "http://127.0.0.1:8080",
            "http://[::1]:3000",
        ] {
            let hv = header::HeaderValue::from_str(origin).unwrap();
            assert!(is_loopback_origin(&hv), "expected loopback: {origin}");
        }

        for origin in ["https://example.com", "http://10.0.0.1:3000"] {
            let hv = header::HeaderValue::from_str(origin).unwrap();
            assert!(!is_loopback_origin(&hv), "expected non-loopback: {origin}");
        }
    }

    #[test]
    fn extract_presented_token_prefers_bearer_then_fallback_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_static("Bearer abc123"),
        );
        headers.insert(
            "x-bizra-token",
            header::HeaderValue::from_static("should_not_be_used"),
        );
        assert_eq!(extract_presented_token(&headers).as_deref(), Some("abc123"));

        let mut headers2 = HeaderMap::new();
        headers2.insert("x-bizra-token", header::HeaderValue::from_static("xyz"));
        assert_eq!(extract_presented_token(&headers2).as_deref(), Some("xyz"));
    }

    #[test]
    fn is_authorized_matches_expected_token() {
        let expected = "secret";

        let headers_missing = HeaderMap::new();
        assert!(!is_authorized(&headers_missing, expected));

        let mut headers_bearer = HeaderMap::new();
        headers_bearer.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_static("Bearer secret"),
        );
        assert!(is_authorized(&headers_bearer, expected));

        let mut headers_alt = HeaderMap::new();
        headers_alt.insert("x-bizra-token", header::HeaderValue::from_static("secret"));
        assert!(is_authorized(&headers_alt, expected));
    }
}
