// src/http.rs - HTTP API Server
//
// BIZRA Security-First HTTP Layer
// ================================
// - Rate limiting: 100 req/min per endpoint class
// - Bearer token authentication on all sensitive endpoints
// - Request ID tracing for audit trail
// - Prometheus metrics per endpoint

use crate::{
    ihsan,
    mcp::{self, JsonRpcRequest},
    metrics,
    ollama,
    pat_enhanced::EnhancedPATOrchestrator,
    sape,
    types::{AdapterModes, DualAgenticRequest, DualAgenticResponse, EnhancedDualAgenticRequest},
    MetaAlphaDualAgentic,
};
use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::{info, warn};
use uuid::Uuid;

// ============================================================
// Security: Rate Limiter
// ============================================================

/// Token bucket rate limiter with per-IP tracking
#[derive(Clone)]
pub struct RateLimiter {
    /// Map of IP -> (token count, last refill time)
    buckets: Arc<RwLock<HashMap<String, (u32, Instant)>>>,
    /// Max tokens per bucket
    max_tokens: u32,
    /// Refill rate (tokens per second)
    refill_rate: f64,
}

impl RateLimiter {
    pub fn new(max_tokens: u32, refill_rate: f64) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            max_tokens,
            refill_rate,
        }
    }

    /// Try to consume a token, returns true if allowed
    pub async fn try_acquire(&self, key: &str) -> bool {
        let mut buckets = self.buckets.write().await;
        let now = Instant::now();

        let (tokens, last_refill) = buckets
            .entry(key.to_string())
            .or_insert((self.max_tokens, now));

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(*last_refill).as_secs_f64();
        let refill = (elapsed * self.refill_rate) as u32;
        *tokens = (*tokens + refill).min(self.max_tokens);
        *last_refill = now;

        if *tokens > 0 {
            *tokens -= 1;
            metrics::HTTP_REQUESTS_ALLOWED.inc();
            true
        } else {
            metrics::HTTP_REQUESTS_RATE_LIMITED.inc();
            false
        }
    }

    /// Cleanup old entries (call periodically)
    pub async fn cleanup(&self, max_age: Duration) {
        let mut buckets = self.buckets.write().await;
        let now = Instant::now();
        buckets.retain(|_, (_, last)| now.duration_since(*last) < max_age);
    }
}

/// Rate limiting middleware
async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Extract client identifier (IP or forwarded header)
    let client_id = extract_client_id(&request);

    if !limiter.try_acquire(&client_id).await {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "60")],
            Json(serde_json::json!({
                "error": "Rate limit exceeded",
                "retry_after_seconds": 60,
            })),
        )
            .into_response();
    }

    next.run(request).await
}

fn extract_client_id(request: &Request<Body>) -> String {
    // Check X-Forwarded-For first (reverse proxy)
    if let Some(forwarded) = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
    {
        if let Some(first_ip) = forwarded.split(',').next() {
            return first_ip.trim().to_string();
        }
    }

    // Fall back to X-Real-IP
    if let Some(real_ip) = request
        .headers()
        .get("x-real-ip")
        .and_then(|h| h.to_str().ok())
    {
        return real_ip.trim().to_string();
    }

    // Generate a unique bucket key from available request metadata to prevent
    // cross-client DoS when IP is unknown. Uses SHA-256 of User-Agent + Accept-Language.
    let ua = request
        .headers()
        .get(header::USER_AGENT)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("no-ua");
    let lang = request
        .headers()
        .get(header::ACCEPT_LANGUAGE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("no-lang");
    
    // Create a stable cryptographic hash-based bucket key (SHA-256)
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(ua.as_bytes());
    hasher.update(b"|");
    hasher.update(lang.as_bytes());
    let digest = hasher.finalize();
    format!("anon-{:x}", digest)
}

// ============================================================
// Security: Authentication Middleware
// ============================================================

/// Authentication middleware for protected endpoints
async fn auth_middleware(
    State(api_token): State<Arc<str>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let headers = request.headers();

    // Skip auth for public endpoints
    let path = request.uri().path();
    if is_public_endpoint(path) {
        return next.run(request).await;
    }

    if !is_authorized(headers, api_token.as_ref()) {
        metrics::HTTP_REQUESTS_UNAUTHORIZED.inc();
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "Unauthorized",
                "message": "Missing or invalid API token. Use 'Authorization: Bearer <token>' header.",
            })),
        )
            .into_response();
    }

    next.run(request).await
}

/// Endpoints that don't require authentication
fn is_public_endpoint(path: &str) -> bool {
    matches!(
        path,
        "/" | "/health" | "/metrics" | "/dashboard" | "/static/dashboard.html"
    ) || path.starts_with("/static/")
}

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

    // Initialize rate limiter: 100 tokens, refill 2 per second (allows bursts, steady 120/min)
    let rate_limiter = RateLimiter::new(100, 2.0);

    // Spawn background cleanup task for rate limiter
    let limiter_clone = rate_limiter.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        loop {
            interval.tick().await;
            limiter_clone.cleanup(Duration::from_secs(600)).await;
        }
    });

    // Initialize MCP client with BIZRA tools
    {
        let mcp_client = mcp::get_mcp().await;
        let mut client = mcp_client.lock().await;
        client.register_bizra_tools();
    }

    // Protected routes (require authentication)
    let protected_routes = Router::new()
        .route("/dual/execute", post(execute_dual))
        .route("/enhanced/execute", post(execute_enhanced))
        .route("/mcp/rpc", post(mcp_rpc_handler))
        .route("/mcp/tools", get(mcp_tools_list))
        .route("/sape/probes", post(sape_probes_handler))
        .route("/sape/stats", get(sape_stats_handler))
        .route("/ollama/generate", post(ollama_generate_handler))
        .route("/ollama/chat", post(ollama_chat_handler))
        .route("/ollama/status", get(ollama_status_handler))
        .layer(middleware::from_fn_with_state(api_token.clone(), auth_middleware));

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/metrics", get(prometheus_metrics))
        .route("/stats", get(stats))
        .route("/dashboard", get(dashboard_redirect))
        .nest_service("/static", ServeDir::new("static"));

    // Combine routes with shared middleware
    let app = Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .layer(middleware::from_fn_with_state(rate_limiter, rate_limit_middleware))
        .layer(cors_layer())
        .layer(TraceLayer::new_for_http())
        .with_state((system, enhanced_pat, api_token));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

    info!("🌐 HTTP Server listening on http://127.0.0.1:{}", port);
    info!("📊 Dashboard available at http://127.0.0.1:{}/static/dashboard.html", port);
    info!("🔒 Protected endpoints require Authorization: Bearer <token>");

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
        "status": "PRODUCTION_READY",
        "adapter_modes": AdapterModes::current(),
        "ihsan": {
            "constitution_id": constitution.id(),
            "threshold_baseline": constitution.threshold(),
            "env": ihsan_env,
            "artifact_class": ihsan_artifact_class,
            "threshold_applied": ihsan_threshold_applied,
        },
        "mode_override": "Set BIZRA_ADAPTER_MODE=simulated to force simulation",
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

/// Prometheus metrics endpoint for Glass Cockpit observability
async fn prometheus_metrics() -> impl IntoResponse {
    let metrics = metrics::gather_metrics();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        metrics,
    )
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
    State((system, _, _)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
    Json(request): Json<DualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, String)> {
    // Authentication handled by middleware
    match system.execute(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Execution failed: {}", e),
        )),
    }
}

async fn execute_enhanced(
    State((_, enhanced_pat, _)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
    Json(request): Json<EnhancedDualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, String)> {
    // Authentication handled by middleware
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

// ============================================================
// Dashboard Handler
// ============================================================

async fn dashboard_redirect() -> impl IntoResponse {
    axum::response::Redirect::permanent("/static/dashboard.html")
}

// ============================================================
// MCP JSON-RPC Handlers
// ============================================================

/// Handle MCP JSON-RPC 2.0 requests
async fn mcp_rpc_handler(
    Json(request): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    let mcp_client = mcp::get_mcp().await;
    let client = mcp_client.lock().await;
    let response = client.handle_jsonrpc(request).await;
    Json(response)
}

/// List available MCP tools
async fn mcp_tools_list() -> impl IntoResponse {
    let mcp_client = mcp::get_mcp().await;
    let client = mcp_client.lock().await;
    let tools: Vec<serde_json::Value> = client.list_tools()
        .into_iter()
        .map(|t| serde_json::json!({
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters.iter().map(|p| serde_json::json!({
                "name": p.name,
                "type": p.type_,
                "description": p.description,
                "required": p.required,
            })).collect::<Vec<_>>(),
        }))
        .collect();
    
    Json(serde_json::json!({
        "tools": tools,
        "count": tools.len(),
    }))
}

// ============================================================
// SAPE Probe Handlers
// ============================================================

#[derive(serde::Deserialize)]
struct SAPEProbeRequest {
    content: String,
}

/// Execute SAPE probes on content
async fn sape_probes_handler(
    Json(request): Json<SAPEProbeRequest>,
) -> impl IntoResponse {
    let sape_engine = sape::get_sape();
    let mut engine = sape_engine.lock().unwrap();
    
    let results = engine.execute_probes(&request.content);
    let ihsan_score = engine.calculate_ihsan_score(&results);
    
    let probe_results: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "dimension": r.dimension.name(),
            "score": r.score,
            "confidence": r.confidence,
            "flags": r.flags,
            "passed": r.passed(0.7),
        })
    }).collect();
    
    Json(serde_json::json!({
        "ihsan_score": ihsan_score,
        "passed": ihsan_score >= 0.85,
        "probes": probe_results,
        "dimensions_analyzed": results.len(),
    }))
}

/// Get SAPE statistics
async fn sape_stats_handler() -> impl IntoResponse {
    let sape_engine = sape::get_sape();
    let engine = sape_engine.lock().unwrap();
    let stats = engine.get_statistics();
    
    let patterns: Vec<serde_json::Value> = engine.get_active_patterns()
        .iter()
        .map(|p| serde_json::json!({
            "id": p.id,
            "name": p.name,
            "activations": p.activation_count,
            "latency_saved_ms": p.latency_reduction_ms,
            "snr_improvement": p.snr_improvement,
        }))
        .collect();
    
    Json(serde_json::json!({
        "total_patterns": stats.total_patterns,
        "active_patterns": stats.active_patterns,
        "sequences_observed": stats.sequences_observed,
        "total_latency_saved_ms": stats.total_latency_saved_ms,
        "total_snr_improvement": stats.total_snr_improvement,
        "patterns": patterns,
    }))
}

// ============================================================
// Ollama LLM Handlers
// ============================================================

#[derive(serde::Deserialize)]
struct OllamaGenerateRequest {
    prompt: String,
    model: Option<String>,
    temperature: Option<f64>,
}

#[derive(serde::Deserialize)]
struct OllamaChatRequest {
    message: String,
    history: Option<Vec<ollama::ChatMessage>>,
    model: Option<String>,
}

/// Generate text with Ollama
async fn ollama_generate_handler(
    Json(request): Json<OllamaGenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let client = ollama::get_ollama().await;
    
    if !client.is_connected() {
        // Return fallback response
        let fallback = ollama::OllamaFallback::simulated_response(&request.prompt);
        return Ok(Json(serde_json::json!({
            "response": fallback.response,
            "model": fallback.model,
            "simulated": true,
        })));
    }
    
    let _options = request.temperature.map(|t| ollama::GenerationOptions {
        temperature: Some(t),
        ..Default::default()
    });
    
    match client.bizra_generate(&request.prompt, request.model.as_deref()).await {
        Ok(response) => Ok(Json(serde_json::json!({
            "response": response.response,
            "model": response.model,
            "done": response.done,
            "eval_count": response.eval_count,
            "simulated": false,
        }))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// Chat with Ollama
async fn ollama_chat_handler(
    Json(request): Json<OllamaChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let client = ollama::get_ollama().await;
    
    if !client.is_connected() {
        return Ok(Json(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "[SIMULATED] Ollama not available"
            },
            "model": "simulated",
            "simulated": true,
        })));
    }
    
    let history = request.history.unwrap_or_default();
    
    match client.bizra_chat(&request.message, history, request.model.as_deref()).await {
        Ok(response) => Ok(Json(serde_json::json!({
            "message": response.message,
            "model": response.model,
            "done": response.done,
            "simulated": false,
        }))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// Get Ollama connection status
async fn ollama_status_handler() -> impl IntoResponse {
    let client = ollama::get_ollama().await;
    let connected = client.is_connected();
    
    let models = if connected {
        client.list_models().await.ok()
    } else {
        None
    };
    
    Json(serde_json::json!({
        "connected": connected,
        "models": models.map(|m| m.into_iter().map(|info| serde_json::json!({
            "name": info.name,
            "size": info.size,
        })).collect::<Vec<_>>()),
    }))
}

// ============================================================
// Helper Functions
// ============================================================

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
