//! Cockpit HTTP Client
//!
//! Fetches live data from the Python cockpit server at localhost:8420.
//! Non-blocking: if the cockpit is unreachable, returns None gracefully.

#![allow(dead_code)]

use std::time::Duration;

use serde::Deserialize;

/// Cockpit base URL (configurable via BIZRA_COCKPIT_URL env var)
fn base_url() -> String {
    std::env::var("BIZRA_COCKPIT_URL").unwrap_or_else(|_| "http://localhost:8420".to_string())
}

/// Blocking HTTP client with fast timeout (cockpit is local)
fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(800))
        .connect_timeout(Duration::from_millis(400))
        .build()
        .unwrap_or_default()
}

// ── Response types ──────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct CockpitAgent {
    pub agent_id: String,
    pub role: String,
    pub team: String,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentsResponse {
    pub agents: Vec<CockpitAgent>,
    #[serde(default)]
    pub pat_count: usize,
    #[serde(default)]
    pub sat_count: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReceiptsResponse {
    pub receipts: Vec<serde_json::Value>,
    #[serde(default)]
    pub count: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ActivationResponse {
    #[serde(default)]
    pub activated: bool,
    pub chain_file: Option<String>,
    #[serde(default)]
    pub receipt_count: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeResponse {
    pub pat_runtime: Option<String>,
    pub sat_runtime: Option<String>,
    pub dema_router: Option<String>,
    pub fate_boundary: Option<String>,
    pub proactive_scheduler: Option<String>,
    pub urp_service: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FateResponse {
    #[serde(default)]
    pub total_checks: usize,
    #[serde(default)]
    pub verdicts: std::collections::HashMap<String, usize>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub status: Option<String>,
    pub uptime: Option<String>,
    pub node_id: Option<String>,
    pub runtime_version: Option<String>,
}

/// Aggregate of all cockpit data in a single snapshot
#[derive(Debug, Clone, Default)]
pub struct CockpitSnapshot {
    pub agents: Option<AgentsResponse>,
    pub receipts: Option<ReceiptsResponse>,
    pub activation: Option<ActivationResponse>,
    pub runtime: Option<RuntimeResponse>,
    pub fate: Option<FateResponse>,
    pub health: Option<HealthResponse>,
    pub reachable: bool,
}

// ── Fetch functions ─────────────────────────────────────────

fn fetch_json<T: serde::de::DeserializeOwned>(path: &str) -> Option<T> {
    let url = format!("{}{}", base_url(), path);
    client().get(&url).send().ok()?.json().ok()
}

/// Fetch all cockpit endpoints in a single pass.
/// If the cockpit is down, returns CockpitSnapshot { reachable: false, .. }.
pub fn fetch_snapshot() -> CockpitSnapshot {
    // Probe health first — if unreachable, skip the rest
    let health: Option<HealthResponse> = fetch_json("/api/health");
    if health.is_none() {
        return CockpitSnapshot::default();
    }

    CockpitSnapshot {
        agents: fetch_json("/api/agents"),
        receipts: fetch_json("/api/receipts"),
        activation: fetch_json("/api/activation"),
        runtime: fetch_json("/api/runtime"),
        fate: fetch_json("/api/fate"),
        health,
        reachable: true,
    }
}

/// Quick reachability check — just hits /api/health
pub fn is_reachable() -> bool {
    fetch_json::<HealthResponse>("/api/health").is_some()
}
