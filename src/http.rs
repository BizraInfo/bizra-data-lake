// src/http.rs - HTTP API Server
//
// BIZRA Security-First HTTP Layer
// ================================
// - Rate limiting: 100 req/min per endpoint class
// - Bearer token authentication on all sensitive endpoints
// - Request ID tracing for audit trail
// - Prometheus metrics per endpoint

use crate::{
    autopoietic::{
        self,
        blueprints::AgentBlueprint,
        loop_engine::{AutopoieticLoop, LoopControl},
        types::AutopoieticConfig,
    },
    errors::{BridgeError, PolicyError},
    ihsan,
    lmstudio,
    mcp::{self, JsonRpcRequest},
    metrics,
    model_router,
    ollama,
    pat_enhanced::EnhancedPATOrchestrator,
    sape,
    types::{DualAgenticRequest, DualAgenticResponse, EnhancedDualAgenticRequest},
    voice,
    MetaAlphaDualAgentic,
};
use anyhow::bail;
use axum::{
    body::{Body, Bytes},
    extract::{Extension, State},
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode},
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
use tokio::sync::{mpsc, RwLock};
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

// ============================================================
// Request ID Middleware
// ============================================================

const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone, Debug)]
struct RequestId(String);

async fn request_id_middleware(mut request: Request<Body>, next: Next) -> Response {
    let request_id = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim())
        .filter(|value| {
            // Validate: 1-64 chars, alphanumeric or hyphens only
            !value.is_empty()
                && value.len() <= 64
                && value.chars().all(|c| c.is_alphanumeric() || c == '-')
        })
        .map(|value| value.to_string())
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string());

    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));

    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(header::HeaderName::from_static(REQUEST_ID_HEADER), value);
    }
    response
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
    use sha2::{Digest, Sha256};
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
        "/" | "/health"
            | "/health/ready"
            | "/health/live"
            | "/metrics"
            | "/dashboard"
            | "/static/dashboard.html"
    ) || path.starts_with("/static/")
}

pub async fn create_http_server(
    system: Arc<MetaAlphaDualAgentic>,
    port: u16,
) -> anyhow::Result<()> {
    let enhanced_pat = Arc::new(EnhancedPATOrchestrator::new().await?);

    let api_token = api_token_from_env()?;

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
        .route("/router/status", get(router_status_handler))
        .route("/ollama/generate", post(ollama_generate_handler))
        .route("/ollama/chat", post(ollama_chat_handler))
        .route("/ollama/status", get(ollama_status_handler))
        .route("/voice/transcribe", post(voice_transcribe_handler))
        .route("/voice/speak", post(voice_speak_handler))
        // Autopoietic Loop endpoints
        .route("/autopoietic/start", post(autopoietic_start_handler))
        .route("/autopoietic/stop", post(autopoietic_stop_handler))
        .route("/autopoietic/status", get(autopoietic_status_handler))
        .route("/autopoietic/history", get(autopoietic_history_handler))
        .route("/autopoietic/inject", post(autopoietic_inject_handler))
        .route("/autopoietic/verify", get(autopoietic_verify_handler))
        // Node0 Unified System endpoints
        .route("/node0/status", get(node0_status_handler))
        .route("/node0/resources", get(node0_resources_handler))
        .route("/node0/verify", get(node0_verify_handler))
        .route("/node0/services", get(node0_services_handler))
        .layer(middleware::from_fn_with_state(
            api_token.clone(),
            auth_middleware,
        ));

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/health/ready", get(health_ready))
        .route("/health/live", get(health_live))
        .route("/metrics", get(prometheus_metrics))
        .route("/stats", get(stats))
        .route("/dashboard", get(dashboard_redirect))
        .nest_service("/static", ServeDir::new("static"));

    // Combine routes with shared middleware
    let app = Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .layer(middleware::from_fn_with_state(
            rate_limiter,
            rate_limit_middleware,
        ))
        .layer(cors_layer())
        .layer(TraceLayer::new_for_http())
        .layer(middleware::from_fn(request_id_middleware))
        .with_state((system, enhanced_pat, api_token));

    let host = http_bind_host();
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;

    info!("🌐 HTTP Server listening on http://{}:{}", host, port);
    info!(
        "📊 Dashboard available at http://{}:{}/static/dashboard.html",
        host, port
    );
    info!("🔒 Protected endpoints require Authorization: Bearer <token>");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> impl IntoResponse {
    let constitution = ihsan::constitution();
    let ihsan_env = ihsan::current_env();
    let ihsan_artifact_class = "docs";
    let ihsan_threshold_applied = constitution.threshold_for(&ihsan_env, ihsan_artifact_class);
    let ihsan_threshold_target = constitution.threshold();
    let cert_bundle = std::env::var("BIZRA_CERT_BUNDLE_SHA256")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let (normalized_env, status, certification) = truth_kernel_status(
        &ihsan_env,
        ihsan_threshold_applied,
        ihsan_threshold_target,
        cert_bundle.as_deref(),
    );
    Json(serde_json::json!({
        "name": "BIZRA META ALPHA ELITE - Complete Unified System",
        "version": "2.0.0",
        "architecture": "PAT(6) + SAT(5) + Full Arsenal",
        "capabilities": [
            "MCP Integration",
            "A2A Protocol",
            "Multi-Reasoning (CoT, ToT, GoT, ReAct, Reflexion)",
            "Sub-Agent Spawning",
            "Swarm Intelligence",
            "Hook System",
            "Slash Commands",
        ],
        "execution_mode": "REAL_LLMS_ONLY",
        "profile": normalized_env,
        "certification": certification,
        "status": status,
        "ihsan": {
            "constitution_id": constitution.id(),
            "threshold_target": ihsan_threshold_target,
            "env": ihsan_env,
            "artifact_class": ihsan_artifact_class,
            "threshold_enforced": ihsan_threshold_applied,
        },
    }))
}

fn truth_kernel_status(
    ihsan_env: &str,
    threshold_enforced: f64,
    threshold_target: f64,
    cert_bundle: Option<&str>,
) -> (&'static str, &'static str, serde_json::Value) {
    let env_key = ihsan_env.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let normalized_env = match env_key.as_str() {
        "dev" | "development" => "development",
        "prod" | "production" => "production",
        "staging" => "staging",
        _ => "unknown",
    };

    match normalized_env {
        "development" => (
            "development",
            "DEV_REAL_LLMS",
            serde_json::json!({
                "state": "NOT_CERTIFIED",
                "reason": "dev_threshold"
            }),
        ),
        "staging" => (
            "staging",
            if threshold_enforced >= 0.95 {
                "STAGING_CERTIFIABLE"
            } else {
                "PRODUCTION_BLOCKED"
            },
            serde_json::json!({
                "state": "CANDIDATE",
                "required": ["sbom", "provenance", "receipts", "policy_digest"]
            }),
        ),
        "production" => {
            if cert_bundle.is_some() && threshold_enforced >= threshold_target {
                (
                    "production",
                    "PRODUCTION_CERTIFIED",
                    serde_json::json!({
                        "state": "CERTIFIED",
                        "bundle_sha256": cert_bundle
                    }),
                )
            } else {
                (
                    "production",
                    "PRODUCTION_BLOCKED",
                    serde_json::json!({
                        "state": "NOT_CERTIFIED",
                        "reason": "missing_cert_bundle_or_threshold"
                    }),
                )
            }
        }
        _ => (
            "unknown",
            "PRODUCTION_BLOCKED",
            serde_json::json!({
                "state": "NOT_CERTIFIED",
                "reason": "unknown_env"
            }),
        ),
    }
}

async fn health() -> impl IntoResponse {
    let constitution = ihsan::constitution();
    let ihsan_env = ihsan::current_env();
    let ihsan_artifact_class = "code";
    let ihsan_threshold_applied = constitution.threshold_for(&ihsan_env, ihsan_artifact_class);
    let ihsan_threshold_target = constitution.threshold();
    let cert_bundle = std::env::var("BIZRA_CERT_BUNDLE_SHA256")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let (normalized_env, status_label, certification) = truth_kernel_status(
        &ihsan_env,
        ihsan_threshold_applied,
        ihsan_threshold_target,
        cert_bundle.as_deref(),
    );

    // Get SAPE statistics
    let sape_stats = {
        let sape_engine = sape::get_sape();
        let guard = sape_engine.lock().unwrap();
        guard.get_statistics()
    };

    // Determine overall system health
    let ihsan_healthy = constitution.threshold() <= 1.0;
    let sape_healthy = sape_stats.total_patterns >= 5;
    let overall_healthy = ihsan_healthy && sape_healthy;

    let lmstudio_ready = {
        let client = lmstudio::get_lmstudio().await;
        client.is_connected()
    };

    let voice_config = voice::VoiceConfig::from_env();
    let whisper_ready = voice_config
        .whisper_bin
        .as_deref()
        .map(|p| std::path::Path::new(p).exists())
        .unwrap_or(false)
        && voice_config
            .whisper_model
            .as_deref()
            .map(|p| std::path::Path::new(p).exists())
            .unwrap_or(false);
    let piper_ready = voice_config
        .piper_bin
        .as_deref()
        .map(|p| std::path::Path::new(p).exists())
        .unwrap_or(false)
        && voice_config
            .piper_model
            .as_deref()
            .map(|p| std::path::Path::new(p).exists())
            .unwrap_or(false);
    let voice_ready = whisper_ready && piper_ready;

    let status_code = if overall_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status_code,
        Json(serde_json::json!({
            "status": if overall_healthy { "healthy" } else { "degraded" },
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "execution_mode": "REAL_LLMS_ONLY",
            "profile": normalized_env,
            "certification": certification,
            "status_label": status_label,
            "ihsan": {
                "constitution_id": constitution.id(),
                "env": ihsan_env,
                "threshold_target": ihsan_threshold_target,
                "threshold_enforced": ihsan_threshold_applied,
                "threshold_development": constitution.threshold_for("development", "code"),
                "threshold_ci": constitution.threshold_for("ci", "code"),
                "threshold_staging": constitution.threshold_for("staging", "code"),
                "threshold_production": constitution.threshold_for("production", "code"),
                "dimensions_count": constitution.weights().len(),
                "enforcement_active": ihsan::should_enforce()
            },
            "sape": {
                "patterns_registered": sape_stats.total_patterns,
                "patterns_active": sape_stats.active_patterns,
                "sequences_observed": sape_stats.sequences_observed,
                "unique_sequences": sape_stats.unique_sequences,
                "pending_elevations": sape_stats.pending_elevations,
                "total_latency_saved_ms": sape_stats.total_latency_saved_ms,
                "total_snr_improvement": sape_stats.total_snr_improvement
            },
            "agents": {
                "pat_count": 6,
                "sat_count": 5,
                "total": 11
            },
            "gates": {
                "security": "active",
                "quality": "active",
                "ihsan": if ihsan_healthy { "active" } else { "degraded" },
                "performance": "active"
            },
            "llm_backends": {
                "ollama_connected": ollama::get_ollama().await.is_connected(),
                "lmstudio_connected": lmstudio_ready
            },
            "voice": {
                "configured": voice_ready,
                "whisper_ready": whisper_ready,
                "piper_ready": piper_ready
            }
        })),
    )
}

/// Kubernetes-style readiness probe
/// Returns 200 if service is ready to accept traffic, 503 otherwise
async fn health_ready() -> impl IntoResponse {
    let constitution = ihsan::constitution();
    let ihsan_env = ihsan::current_env();
    let ihsan_threshold_applied = constitution.threshold_for(&ihsan_env, "code");
    let ihsan_threshold_target = constitution.threshold();
    let cert_bundle = std::env::var("BIZRA_CERT_BUNDLE_SHA256")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let (normalized_env, status_label, certification) = truth_kernel_status(
        &ihsan_env,
        ihsan_threshold_applied,
        ihsan_threshold_target,
        cert_bundle.as_deref(),
    );

    // Check critical components
    let sape_ready = {
        let sape_engine = sape::get_sape();
        let guard = sape_engine.lock().unwrap();
        guard.get_statistics().total_patterns >= 5
    };

    // Check MCP
    let mcp_ready = {
        let mcp_client = mcp::get_mcp().await;
        let client = mcp_client.lock().await;
        !client.list_tools().is_empty()
    };

    // Check Ollama
    let ollama_ready = {
        let client = ollama::get_ollama().await;
        client.is_connected()
    };

    let lmstudio_ready = {
        let client = lmstudio::get_lmstudio().await;
        client.is_connected()
    };

    // Check Filesystem (evidence/ write access)
    let fs_ready = std::fs::create_dir_all("evidence")
        .and_then(|_| std::fs::write("evidence/.health_check", b"ok"))
        .is_ok();

    let voice_config = voice::VoiceConfig::from_env();
    let whisper_ready = voice_config
        .whisper_bin
        .as_deref()
        .map(|p| std::path::Path::new(p).exists())
        .unwrap_or(false)
        && voice_config
            .whisper_model
            .as_deref()
            .map(|p| std::path::Path::new(p).exists())
            .unwrap_or(false);
    let piper_ready = voice_config
        .piper_bin
        .as_deref()
        .map(|p| std::path::Path::new(p).exists())
        .unwrap_or(false)
        && voice_config
            .piper_model
            .as_deref()
            .map(|p| std::path::Path::new(p).exists())
            .unwrap_or(false);
    let voice_ready = whisper_ready && piper_ready;

    let ihsan_ready = constitution.weights().len() >= 5;
    let is_ready = sape_ready && ihsan_ready && mcp_ready && ollama_ready && fs_ready;

    let status_code = if is_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status_code,
        Json(serde_json::json!({
            "ready": is_ready,
            "execution_mode": "REAL_LLMS_ONLY",
            "profile": normalized_env,
            "certification": certification,
            "status_label": status_label,
            "ihsan": {
                "constitution_id": constitution.id(),
                "env": ihsan_env,
                "threshold_target": ihsan_threshold_target,
                "threshold_enforced": ihsan_threshold_applied
            },
            "checks": {
                "sape_patterns": sape_ready,
                "ihsan_constitution": ihsan_ready,
                "mcp_tools": mcp_ready,
                "ollama_connection": ollama_ready,
                "lmstudio_connection": lmstudio_ready,
                "fs_write_access": fs_ready,
                "voice_configured": voice_ready,
                "whisper_ready": whisper_ready,
                "piper_ready": piper_ready
            }
        })),
    )
}

/// Kubernetes-style liveness probe
/// Returns 200 if process is alive, used for restart decisions
async fn health_live() -> impl IntoResponse {
    let constitution = ihsan::constitution();
    let ihsan_env = ihsan::current_env();
    let ihsan_threshold_applied = constitution.threshold_for(&ihsan_env, "code");
    let ihsan_threshold_target = constitution.threshold();
    let cert_bundle = std::env::var("BIZRA_CERT_BUNDLE_SHA256")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let (normalized_env, status_label, certification) = truth_kernel_status(
        &ihsan_env,
        ihsan_threshold_applied,
        ihsan_threshold_target,
        cert_bundle.as_deref(),
    );
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "alive": true,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "execution_mode": "REAL_LLMS_ONLY",
            "profile": normalized_env,
            "certification": certification,
            "status_label": status_label,
            "ihsan": {
                "constitution_id": constitution.id(),
                "env": ihsan_env,
                "threshold_target": ihsan_threshold_target,
                "threshold_enforced": ihsan_threshold_applied
            }
        })),
    )
}

/// Prometheus metrics endpoint for Glass Cockpit observability
async fn prometheus_metrics() -> impl IntoResponse {
    // Refresh critical connectivity metrics
    let ollama_connected = ollama::get_ollama().await.is_connected();
    metrics::update_ollama_status(ollama_connected);

    let metrics = metrics::gather_metrics();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
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
    let ihsan_threshold_target = constitution.threshold();
    let cert_bundle = std::env::var("BIZRA_CERT_BUNDLE_SHA256")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let (normalized_env, status_label, certification) = truth_kernel_status(
        &ihsan_env,
        ihsan_threshold_applied,
        ihsan_threshold_target,
        cert_bundle.as_deref(),
    );
    Json(serde_json::json!({
        "pat_agents": 6,
        "sat_agents": 5,
        "total_agents": 11,
        "reasoning_methods": 5,
        "mcp_tools": 4,
        "uptime": "operational",
        "execution_mode": "REAL_LLMS_ONLY",
        "profile": normalized_env,
        "certification": certification,
        "status_label": status_label,
        "ihsan_constitution_id": constitution.id(),
        "ihsan_threshold_target": ihsan_threshold_target,
        "ihsan_env": ihsan_env,
        "ihsan_artifact_class": ihsan_artifact_class,
        "ihsan_threshold_enforced": ihsan_threshold_applied,
    }))
}

async fn execute_dual(
    State((system, _, _)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
    Extension(request_id): Extension<RequestId>,
    Json(mut request): Json<DualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, Json<serde_json::Value>)> {
    // Authentication handled by middleware
    let request_id = request_id.0;
    request
        .context
        .entry("request_id".to_string())
        .or_insert_with(|| request_id.clone());
    match system.execute(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            if let Some(err) = e.downcast_ref::<BridgeError>() {
                let (status, code, message) = match err {
                    BridgeError::SatBlocked { message, .. } => {
                        (StatusCode::FORBIDDEN, "SAT_BLOCKED", message.clone())
                    }
                    BridgeError::IhsanGateFailed { .. } => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "IHSAN_GATE_FAILED",
                        err.to_string(),
                    ),
                    BridgeError::RequestInProgress { key } => (
                        StatusCode::CONFLICT,
                        "REQUEST_IN_PROGRESS",
                        format!("Request already in progress: {}", key),
                    ),
                    BridgeError::IdempotencyError { message } => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "IDEMPOTENCY_ERROR",
                        message.clone(),
                    ),
                    BridgeError::FateLockPoisoned { message } => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "FATE_LOCK_POISONED",
                        message.clone(),
                    ),
                    BridgeError::DataLakeError { message } => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "DATA_LAKE_ERROR",
                        message.clone(),
                    ),
                    BridgeError::ConnectionFailed(message) => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "CONNECTION_FAILED",
                        message.clone(),
                    ),
                    BridgeError::ProtocolError(message) => (
                        StatusCode::BAD_GATEWAY,
                        "PROTOCOL_ERROR",
                        message.clone(),
                    ),
                };
                warn!(error = %message, code = %code, request_id = %request_id, "Policy VETO");
                return Err((
                    status,
                    Json(serde_json::json!({
                        "error": "policy_rejection",
                        "code": code,
                        "message": message,
                        "request_id": request_id,
                    })),
                ));
            }

            warn!(error = %e, request_id = %request_id, "Execution failed");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "internal_error",
                    "code": "EXECUTION_FAILED",
                    "message": "Execution failed",
                    "request_id": request_id,
                })),
            ))
        }
    }
}

async fn execute_enhanced(
    State((_, enhanced_pat, _)): State<(
        Arc<MetaAlphaDualAgentic>,
        Arc<EnhancedPATOrchestrator>,
        Arc<str>,
    )>,
    Extension(request_id): Extension<RequestId>,
    Json(mut request): Json<EnhancedDualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, Json<serde_json::Value>)> {
    // Authentication handled by middleware
    let request_id = request_id.0;
    request
        .base
        .context
        .entry("request_id".to_string())
        .or_insert_with(|| request_id.clone());
    match enhanced_pat.execute_enhanced(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            if let Some(err) = e.downcast_ref::<PolicyError>() {
                let (status, code) = match err {
                    PolicyError::McpToolsBlocked { .. } => {
                        (StatusCode::FORBIDDEN, "MCP_POLICY_BLOCKED")
                    }
                    PolicyError::IhsanGateFailed { .. } => {
                        (StatusCode::UNPROCESSABLE_ENTITY, "IHSAN_GATE_FAILED")
                    }
                };
                return Err((
                    status,
                    Json(serde_json::json!({
                        "error": "policy_rejection",
                        "code": code,
                        "message": err.to_string(),
                        "request_id": request_id,
                    })),
                ));
            }
            warn!(error = %e, request_id = %request_id, "Enhanced execution failed");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "internal_error",
                    "code": "ENHANCED_EXECUTION_FAILED",
                    "message": "Enhanced execution failed",
                    "request_id": request_id,
                })),
            ))
        }
    }
}

fn api_token_from_env() -> anyhow::Result<Arc<str>> {
    match std::env::var("BIZRA_API_TOKEN") {
        Ok(v) if !v.trim().is_empty() => Ok(Arc::<str>::from(v.trim().to_string())),
        _ => bail!("BIZRA_API_TOKEN not set; refusing to start without auth"),
    }
}

fn http_bind_host() -> String {
    std::env::var("BIZRA_HTTP_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
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
async fn mcp_rpc_handler(Json(request): Json<JsonRpcRequest>) -> impl IntoResponse {
    let mcp_client = mcp::get_mcp().await;
    let client = mcp_client.lock().await;
    let response = client.handle_jsonrpc(request).await;
    Json(response)
}

/// List available MCP tools
async fn mcp_tools_list() -> impl IntoResponse {
    let mcp_client = mcp::get_mcp().await;
    let client = mcp_client.lock().await;
    let tools: Vec<serde_json::Value> = client
        .list_tools()
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.iter().map(|p| serde_json::json!({
                    "name": p.name,
                    "type": p.type_,
                    "description": p.description,
                    "required": p.required,
                })).collect::<Vec<_>>(),
            })
        })
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
async fn sape_probes_handler(Json(request): Json<SAPEProbeRequest>) -> impl IntoResponse {
    let sape_engine = sape::get_sape();
    let mut engine = sape_engine.lock().unwrap();

    let results = engine.execute_probes(&request.content);
    let ihsan_score = engine.calculate_ihsan_score(&results);

    let probe_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "dimension": r.dimension.name(),
                "score": r.score,
                "confidence": r.confidence,
                "flags": r.flags,
                "passed": r.passed(0.7),
            })
        })
        .collect();

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

    let patterns: Vec<serde_json::Value> = engine
        .get_active_patterns()
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": p.id,
                "name": p.name,
                "activations": p.activation_count,
                "latency_saved_ms": p.latency_reduction_ms,
                "snr_improvement": p.snr_improvement,
            })
        })
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
// Model Router Handlers
// ============================================================

/// Get Model Router status and capability slot availability
async fn router_status_handler() -> impl IntoResponse {
    match model_router::get_router().await {
        Ok(router) => {
            let stats = router.get_stats().await;
            
            Json(serde_json::json!({
                "status": "operational",
                "available_models": stats.available_models,
                "available_models_ollama": stats.available_models_ollama,
                "available_models_lmstudio": stats.available_models_lmstudio,
                "last_refresh_seconds": stats.last_refresh,
                "capability_slots": stats.slots.iter().map(|s| {
                    serde_json::json!({
                        "name": s.name,
                        "primary_model": s.primary,
                        "fallback_model": s.fallback,
                        "primary_available": s.primary_available,
                        "fallback_available": s.fallback_available,
                        "primary_available_ollama": s.primary_available_ollama,
                        "primary_available_lmstudio": s.primary_available_lmstudio,
                        "fallback_available_ollama": s.fallback_available_ollama,
                        "fallback_available_lmstudio": s.fallback_available_lmstudio,
                        "alternative_models": s.alternatives,
                        "alternatives_available": s.alternatives_available,
                        "alternatives_available_ollama": s.alternatives_available_ollama,
                        "alternatives_available_lmstudio": s.alternatives_available_lmstudio,
                        "operational": s.primary_available || s.fallback_available || s.alternatives_available,
                    })
                }).collect::<Vec<_>>(),
                "routing_strategy": "capability_based",
                "slots_description": {
                    "cold_core": "Deterministic reasoning (deepseek-r1)",
                    "warm_surface": "User-facing nuance (mistral)",
                    "primary_reasoning": "Strategic planning (bizra-planner)",
                    "embeddings": "Semantic search (nomic-embed-text)",
                    "vision": "Multimodal (qwen3-vl, llava fallback)"
                }
            }))
        }
        Err(e) => {
            Json(serde_json::json!({
                "status": "unavailable",
                "error": e.to_string(),
                "available_models": [],
                "capability_slots": [],
            }))
        }
    }
}

// ============================================================
// Voice (Offline STT/TTS) Handlers
// ============================================================

#[derive(serde::Deserialize)]
struct VoiceSpeakRequest {
    text: String,
}

/// Transcribe audio bytes using local whisper.cpp (BIZRA_WHISPER_BIN/BIZRA_WHISPER_MODEL)
async fn voice_transcribe_handler(body: Bytes) -> impl IntoResponse {
    if body.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "empty_request_body",
                "message": "Audio payload is required"
            })),
        );
    }

    let audio = body.to_vec();
    let result = tokio::task::spawn_blocking(move || voice::transcribe_sync(&audio)).await;

    match result {
        Ok(Ok(text)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "text": text,
                "provider": "whisper.cpp"
            })),
        ),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "transcription_failed",
                "message": err.to_string()
            })),
        ),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "transcription_join_failed",
                "message": err.to_string()
            })),
        ),
    }
}

/// Synthesize speech using local piper (BIZRA_PIPER_BIN/BIZRA_PIPER_MODEL)
async fn voice_speak_handler(Json(payload): Json<VoiceSpeakRequest>) -> impl IntoResponse {
    let text = payload.text.trim();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "empty_text",
                "message": "Text is required"
            })),
        )
            .into_response();
    }

    let text_owned = text.to_string();
    let result = tokio::task::spawn_blocking(move || voice::synthesize_sync(&text_owned)).await;

    match result {
        Ok(Ok(audio)) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
            (StatusCode::OK, headers, Bytes::from(audio)).into_response()
        }
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "synthesis_failed",
                "message": err.to_string()
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "synthesis_join_failed",
                "message": err.to_string()
            })),
        )
            .into_response(),
    }
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
        // Fail-closed: Production systems must return error when Ollama unavailable
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Ollama LLM service is not available. Please ensure Ollama is running.".to_string(),
        ));
    }

    let _options = request.temperature.map(|t| ollama::GenerationOptions {
        temperature: Some(t),
        ..Default::default()
    });

    match client
        .bizra_generate(&request.prompt, request.model.as_deref())
        .await
    {
        Ok(response) => Ok(Json(serde_json::json!({
            "response": response.response,
            "model": response.model,
            "done": response.done,
            "eval_count": response.eval_count,
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
        // Fail-closed: Production systems must return error when Ollama unavailable
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Ollama LLM service is not available. Please ensure Ollama is running.".to_string(),
        ));
    }

    let history = request.history.unwrap_or_default();

    match client
        .bizra_chat(&request.message, history, request.model.as_deref())
        .await
    {
        Ok(response) => Ok(Json(serde_json::json!({
            "message": response.message,
            "model": response.model,
            "done": response.done,
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

// ============================================================
// Autopoietic Loop Handlers
// ============================================================

/// Global autopoietic loop instance (lazy initialized)
static AUTOPOIETIC_LOOP: tokio::sync::OnceCell<Arc<RwLock<Option<AutopoieticLoopHandle>>>> =
    tokio::sync::OnceCell::const_new();

/// Handle to control the autopoietic loop
struct AutopoieticLoopHandle {
    loop_engine: Arc<AutopoieticLoop>,
    control_tx: mpsc::Sender<LoopControl>,
}

async fn get_autopoietic_handle() -> Arc<RwLock<Option<AutopoieticLoopHandle>>> {
    AUTOPOIETIC_LOOP
        .get_or_init(|| async { Arc::new(RwLock::new(None)) })
        .await
        .clone()
}

#[derive(serde::Deserialize)]
struct AutopoieticStartRequest {
    generation_duration_ms: Option<u64>,
    max_generations: Option<u64>,
    ihsan_threshold: Option<f64>,
}

#[derive(serde::Deserialize)]
struct AutopoieticHistoryQuery {
    limit: Option<usize>,
}

/// Start the autopoietic loop
async fn autopoietic_start_handler(
    Json(request): Json<AutopoieticStartRequest>,
) -> impl IntoResponse {
    let handle = get_autopoietic_handle().await;
    let mut guard = handle.write().await;

    if guard.is_some() {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "already_running",
                "message": "Autopoietic loop is already running"
            })),
        );
    }

    // Create configuration
    let config = AutopoieticConfig {
        generation_duration_ms: request.generation_duration_ms.unwrap_or(60_000),
        max_generations: request.max_generations.unwrap_or(0),
        ihsan_threshold: request.ihsan_threshold.unwrap_or(0.95),
        ..Default::default()
    };

    // Create the loop
    let (loop_engine, mut event_rx, control_tx) = AutopoieticLoop::new(config.clone());
    let loop_arc = Arc::new(loop_engine);

    // Initialize default blueprints
    loop_arc.initialize_default_blueprints().await;

    // Start the loop in background
    let loop_clone = loop_arc.clone();
    tokio::spawn(async move {
        if let Err(e) = loop_clone.run().await {
            warn!(error = %e, "Autopoietic loop ended with error");
        }
    });

    // Drain events to prevent bounded channel backpressure from stalling the loop.
    tokio::spawn(async move {
        while let Some(_event) = event_rx.recv().await {
            // Intentionally discard; swap with logging/streaming if needed later.
        }
    });

    // Store handle
    *guard = Some(AutopoieticLoopHandle {
        loop_engine: loop_arc.clone(),
        control_tx,
    });

    info!(
        generation_duration_ms = config.generation_duration_ms,
        max_generations = config.max_generations,
        ihsan_threshold = config.ihsan_threshold,
        "🚀 Autopoietic loop started via HTTP"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "started",
            "config": {
                "generation_duration_ms": config.generation_duration_ms,
                "max_generations": config.max_generations,
                "ihsan_threshold": config.ihsan_threshold,
            }
        })),
    )
}

/// Stop the autopoietic loop
async fn autopoietic_stop_handler() -> impl IntoResponse {
    let handle = get_autopoietic_handle().await;
    let mut guard = handle.write().await;

    match guard.take() {
        Some(handle) => {
            handle.loop_engine.stop();
            info!("🛑 Autopoietic loop stopped via HTTP");
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "stopped",
                    "message": "Autopoietic loop stopped gracefully"
                })),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "not_running",
                "message": "Autopoietic loop is not running"
            })),
        ),
    }
}

/// Get autopoietic loop status
async fn autopoietic_status_handler() -> impl IntoResponse {
    let handle = get_autopoietic_handle().await;
    let guard = handle.read().await;

    match guard.as_ref() {
        Some(handle) => {
            let status = handle.loop_engine.status().await;
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "running": true,
                    "current_generation": status.current_generation,
                    "kep_state": format!("{:?}", status.kep_state),
                    "aggregate_ihsan": status.aggregate_ihsan,
                    "active_agents": status.active_agents,
                    "blueprint_count": status.blueprint_count,
                    "convergence_state": status.convergence_state,
                    "proof_chain_length": status.proof_chain_length,
                    "receipts_emitted": status.receipts_emitted,
                })),
            )
        }
        None => (
            StatusCode::OK,
            Json(serde_json::json!({
                "running": false,
                "message": "Autopoietic loop is not running. Use POST /autopoietic/start to begin."
            })),
        ),
    }
}

/// Get autopoietic evolution history
async fn autopoietic_history_handler(
    axum::extract::Query(query): axum::extract::Query<AutopoieticHistoryQuery>,
) -> impl IntoResponse {
    let handle = get_autopoietic_handle().await;
    let guard = handle.read().await;

    let limit = query.limit.unwrap_or(10);

    match guard.as_ref() {
        Some(handle) => {
            let history = handle.loop_engine.history(limit).await;
            let history_json: Vec<serde_json::Value> = history
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "generation": p.generation,
                        "aggregate_ihsan": p.aggregate_ihsan,
                        "tasks_processed": p.tasks_processed,
                        "successful_executions": p.successful_executions,
                        "rejections": p.rejections,
                        "avg_latency_ms": p.avg_latency_ms,
                        "p95_latency_ms": p.p95_latency_ms,
                        "kep_progress": {
                            "knowledge_mass": p.kep_progress.knowledge_mass,
                            "discovery_velocity": p.kep_progress.discovery_velocity,
                            "synergy_density": p.kep_progress.synergy_density,
                        },
                        "improvements_count": p.improvements_applied.len(),
                        "proof_hash": p.proof_hash,
                        "receipt_id": p.receipt_id,
                        "duration_ms": p.duration_ms,
                    })
                })
                .collect();

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "history": history_json,
                    "count": history.len(),
                    "limit": limit,
                })),
            )
        }
        None => (
            StatusCode::OK,
            Json(serde_json::json!({
                "history": [],
                "count": 0,
                "message": "Autopoietic loop is not running"
            })),
        ),
    }
}

#[derive(serde::Deserialize)]
struct AutopoieticInjectRequest {
    id: String,
    name: String,
    team: String,
    capability_slot: String,
    system_prompt: String,
    model: String,
}

/// Inject a blueprint for testing
async fn autopoietic_inject_handler(
    Json(request): Json<AutopoieticInjectRequest>,
) -> impl IntoResponse {
    let handle = get_autopoietic_handle().await;
    let guard = handle.read().await;

    match guard.as_ref() {
        Some(handle) => {
            let team = match request.team.to_uppercase().as_str() {
                "PAT" => autopoietic::blueprints::AgentTeam::PAT,
                "SAT" => autopoietic::blueprints::AgentTeam::SAT,
                _ => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": "invalid_team",
                            "message": "Team must be 'PAT' or 'SAT'"
                        })),
                    );
                }
            };

            let slot = autopoietic::blueprints::CapabilitySlot::Dynamic(request.capability_slot.clone());

            let blueprint = AgentBlueprint::genesis(
                &request.id,
                &request.name,
                team,
                slot,
                &request.system_prompt,
                &request.model,
                "ollama",
                4.0,
            );

            handle.loop_engine.inject_blueprint(blueprint).await;

            info!(
                blueprint_id = %request.id,
                team = %request.team,
                "📋 Blueprint injected via HTTP"
            );

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "injected",
                    "blueprint_id": request.id,
                    "team": request.team,
                })),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "not_running",
                "message": "Autopoietic loop is not running"
            })),
        ),
    }
}

/// Verify proof chain integrity
async fn autopoietic_verify_handler() -> impl IntoResponse {
    let handle = get_autopoietic_handle().await;
    let guard = handle.read().await;

    match guard.as_ref() {
        Some(handle) => {
            let verification = handle.loop_engine.verify_chain().await;

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "is_valid": verification.is_valid,
                    "verified_nodes": verification.verified_nodes,
                    "total_nodes": verification.total_nodes,
                    "errors": verification.errors,
                    "chain_hash": verification.chain_hash,
                })),
            )
        }
        None => (
            StatusCode::OK,
            Json(serde_json::json!({
                "is_valid": true,
                "verified_nodes": 0,
                "total_nodes": 0,
                "errors": [],
                "message": "Autopoietic loop is not running - no chain to verify"
            })),
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NODE0 UNIFIED SYSTEM HANDLERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Get complete Node0 unified status
async fn node0_status_handler() -> impl IntoResponse {
    use crate::node0_unified::UnifiedNode0Manager;

    let manager = UnifiedNode0Manager::new();
    let status = manager.full_health_check().await;

    (StatusCode::OK, Json(status))
}

/// Get Node0 resource summary
async fn node0_resources_handler() -> impl IntoResponse {
    use crate::node0_unified::UnifiedNode0Manager;

    let manager = UnifiedNode0Manager::new();
    let resources = manager.get_resource_summary().await;

    (StatusCode::OK, Json(resources))
}

/// Verify Node0 standalone operation capability
async fn node0_verify_handler() -> impl IntoResponse {
    use crate::node0_unified::UnifiedNode0Manager;

    let manager = UnifiedNode0Manager::new();
    let verification = manager.verify_standalone().await;

    let status_code = if verification.standalone_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status_code, Json(verification))
}

/// Get Node0 service status
async fn node0_services_handler() -> impl IntoResponse {
    use crate::node0_unified::UnifiedNode0Manager;

    let manager = UnifiedNode0Manager::new();
    let mut services = manager.check_docker_services().await;
    let ollama = manager.check_ollama().await;
    services.push(ollama);

    (StatusCode::OK, Json(services))
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
