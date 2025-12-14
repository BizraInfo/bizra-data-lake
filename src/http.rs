// src/http.rs - HTTP API Server

use crate::{
    pat_enhanced::EnhancedPATOrchestrator,
    types::{EnhancedDualAgenticRequest, DualAgenticResponse, DualAgenticRequest},
    MetaAlphaDualAgentic,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing::info;

pub async fn create_http_server(
    system: Arc<MetaAlphaDualAgentic>,
    port: u16,
) -> anyhow::Result<()> {
    let enhanced_pat = Arc::new(EnhancedPATOrchestrator::new().await?);
    
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/dual/execute", post(execute_dual))
        .route("/enhanced/execute", post(execute_enhanced))
        .route("/stats", get(stats))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state((system, enhanced_pat));
    
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await?;
    
    info!("🌐 HTTP Server listening on http://0.0.0.0:{}", port);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn root() -> impl IntoResponse {
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
        "status": "PRODUCTION",
        "ihsan": "إحسان",
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn stats(
    State((_system, _)): State<(Arc<MetaAlphaDualAgentic>, Arc<EnhancedPATOrchestrator>)>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "pat_agents": 7,
        "sat_agents": 5,
        "total_agents": 12,
        "reasoning_methods": 5,
        "mcp_tools": 4,
        "uptime": "operational",
    }))
}

async fn execute_dual(
    State((system, _)): State<(Arc<MetaAlphaDualAgentic>, Arc<EnhancedPATOrchestrator>)>,
    Json(request): Json<DualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, String)> {
    match system.execute(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Execution failed: {}", e),
        )),
    }
}

async fn execute_enhanced(
    State((_, enhanced_pat)): State<(Arc<MetaAlphaDualAgentic>, Arc<EnhancedPATOrchestrator>)>,
    Json(request): Json<EnhancedDualAgenticRequest>,
) -> Result<Json<DualAgenticResponse>, (StatusCode, String)> {
    match enhanced_pat.execute_enhanced(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Enhanced execution failed: {}", e),
        )),
    }
}
