// src/apex/orchestrator.rs - Apex Orchestrator
//
// Unified entry point for all agent operations. Integrates:
// - bridge.rs PAT-SAT flow
// - Thompson Sampling routing
// - Context optimization
// - SONA self-learning
// - Circuit breaker fault tolerance
//
// HTTP Endpoints:
// - POST /apex/execute - Execute a task with intelligent routing
// - POST /apex/route - Get routing decision without execution
// - GET /apex/metrics - Get orchestrator metrics

use crate::bridge::BridgeCoordinator;
use crate::fate::{Escalation, EscalationLevel, FATECoordinator};
use crate::ihsan;
use crate::metrics;
use crate::model_router::CapabilitySlot;
use crate::receipts::{ReceiptEmitter, ReceiptType};
use crate::types::{AgentResult, DualAgenticRequest, DualAgenticResponse};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, instrument, warn};

use super::circuit_breaker::{CircuitBreakerManager, CircuitState, CircuitStats};
use super::context_optimizer::{ContextOptimizer, ContextPriority, ContextSegment, ContextSource};
use super::learning::{LearningLoop, LearningStats, PerformanceRecord};
use super::router::{AgentCapability, RouterStats, RoutingDecision, ThompsonSamplingRouter};
use super::{ApexError, ApexResult};

/// Apex execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApexExecuteRequest {
    /// User identifier
    pub user_id: String,
    /// Task to execute
    pub task: String,
    /// Requirements for execution
    #[serde(default)]
    pub requirements: Vec<String>,
    /// Target output
    pub target: String,
    /// Additional context
    #[serde(default)]
    pub context: HashMap<String, String>,
    /// Force specific agent (bypasses routing)
    pub force_agent: Option<String>,
    /// Enable context optimization
    #[serde(default = "default_true")]
    pub optimize_context: bool,
    /// Enable learning feedback
    #[serde(default = "default_true")]
    pub enable_learning: bool,
}

fn default_true() -> bool {
    true
}

impl From<ApexExecuteRequest> for DualAgenticRequest {
    fn from(req: ApexExecuteRequest) -> Self {
        DualAgenticRequest {
            user_id: req.user_id,
            task: req.task,
            requirements: req.requirements,
            target: req.target,
            context: req.context,
            ..Default::default()
        }
    }
}

/// Apex execution response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApexExecuteResponse {
    /// Whether execution succeeded
    pub success: bool,
    /// PAT contributions
    pub pat_contributions: Vec<String>,
    /// SAT contributions
    pub sat_contributions: Vec<String>,
    /// Synergy score
    pub synergy_score: f64,
    /// Ihsan score
    pub ihsan_score: f64,
    /// Total latency
    pub latency_ms: u64,
    /// Routing decision used
    pub routing: Option<RoutingDecision>,
    /// Receipt ID
    pub receipt_id: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional metadata
    pub meta: serde_json::Value,
}

/// Apex routing request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApexRouteRequest {
    /// Task content for routing decision
    pub task: String,
    /// Include capability matrix in response
    #[serde(default)]
    pub include_capabilities: bool,
}

/// Apex routing response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApexRouteResponse {
    /// Routing decision
    pub decision: RoutingDecision,
    /// Alternative agents
    pub alternatives: Vec<String>,
    /// Whether agent circuit is open
    pub circuit_open: bool,
    /// Capability slot selected
    pub slot: String,
}

/// Apex metrics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApexMetricsResponse {
    /// Router statistics
    pub router: RouterStats,
    /// Learning statistics
    pub learning: LearningStats,
    /// Circuit breaker statistics
    pub circuits: Vec<CircuitStats>,
    /// Context optimizer statistics
    pub optimizer: super::context_optimizer::OptimizerStats,
    /// System health
    pub health: ApexHealth,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApexHealth {
    /// Whether system is healthy
    pub healthy: bool,
    /// Number of open circuits
    pub open_circuits: usize,
    /// Average quality score
    pub avg_quality: f64,
    /// Total requests processed
    pub total_requests: u64,
    /// Requests in last hour
    pub requests_last_hour: u64,
    /// Ihsan gate pass rate
    pub ihsan_pass_rate: f64,
}

/// Shared orchestrator state
pub struct ApexState {
    /// Bridge coordinator for PAT-SAT flow
    pub bridge: Option<BridgeCoordinator>,
    /// Thompson Sampling router
    pub router: Arc<ThompsonSamplingRouter>,
    /// Context optimizer
    pub optimizer: ContextOptimizer,
    /// Learning loop
    pub learning: Arc<LearningLoop>,
    /// Circuit breaker manager
    pub circuits: CircuitBreakerManager,
    /// Receipt emitter
    pub receipts: ReceiptEmitter,
    /// Request counter
    pub request_count: std::sync::atomic::AtomicU64,
    /// Ihsan pass counter
    pub ihsan_passes: std::sync::atomic::AtomicU64,
}

/// Apex Orchestrator - Main entry point
pub struct ApexOrchestrator {
    /// Shared state (wrapped for thread safety)
    state: Arc<RwLock<ApexState>>,
}

impl ApexOrchestrator {
    /// Create new Apex Orchestrator
    pub async fn new() -> ApexResult<Self> {
        info!("🚀 Initializing Apex Orchestrator");

        // Initialize components
        let router = Arc::new(ThompsonSamplingRouter::new());
        let learning = Arc::new(LearningLoop::with_router(router.clone()));

        // Register default agents
        Self::register_default_agents(&router)?;

        let state = ApexState {
            bridge: None, // Will be initialized lazily
            router,
            optimizer: ContextOptimizer::new(8192),
            learning,
            circuits: CircuitBreakerManager::new(),
            receipts: ReceiptEmitter::default(),
            request_count: std::sync::atomic::AtomicU64::new(0),
            ihsan_passes: std::sync::atomic::AtomicU64::new(0),
        };

        info!("✅ Apex Orchestrator initialized");

        Ok(Self {
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Create with bridge coordinator
    pub async fn with_bridge(bridge: BridgeCoordinator) -> ApexResult<Self> {
        let orchestrator = Self::new().await?;

        {
            let mut state = orchestrator.state.write().await;
            state.bridge = Some(bridge);
        }

        info!("✅ Apex Orchestrator initialized with Bridge Coordinator");
        Ok(orchestrator)
    }

    /// Register default agents for routing
    fn register_default_agents(router: &ThompsonSamplingRouter) -> ApexResult<()> {
        // PAT Agents
        let pat_agents = vec![
            AgentCapability {
                agent_id: "MasterReasoner".to_string(),
                name: "Master Reasoner".to_string(),
                slots: vec![CapabilitySlot::ColdCore, CapabilitySlot::PrimaryReasoning],
                specializations: vec!["reasoning".to_string(), "planning".to_string()],
                max_concurrency: 5,
                current_load: 0.0,
                available: true,
            },
            AgentCapability {
                agent_id: "MemoryArchitect".to_string(),
                name: "Memory Architect".to_string(),
                slots: vec![CapabilitySlot::Embeddings],
                specializations: vec!["memory".to_string(), "retrieval".to_string()],
                max_concurrency: 10,
                current_load: 0.0,
                available: true,
            },
            AgentCapability {
                agent_id: "CreativeSynthesizer".to_string(),
                name: "Creative Synthesizer".to_string(),
                slots: vec![CapabilitySlot::WarmSurface],
                specializations: vec!["creative".to_string(), "generation".to_string()],
                max_concurrency: 5,
                current_load: 0.0,
                available: true,
            },
            AgentCapability {
                agent_id: "EthicsGuardian".to_string(),
                name: "Ethics Guardian".to_string(),
                slots: vec![CapabilitySlot::ColdCore],
                specializations: vec!["ethics".to_string(), "validation".to_string()],
                max_concurrency: 10,
                current_load: 0.0,
                available: true,
            },
            AgentCapability {
                agent_id: "Communicator".to_string(),
                name: "Communicator".to_string(),
                slots: vec![CapabilitySlot::WarmSurface],
                specializations: vec!["communication".to_string(), "formatting".to_string()],
                max_concurrency: 10,
                current_load: 0.0,
                available: true,
            },
        ];

        for agent in pat_agents {
            router.register_agent(agent)?;
        }

        info!("📋 Default agents registered");
        Ok(())
    }

    /// Execute a task through the orchestrator
    #[instrument(skip(self, request))]
    pub async fn execute(&self, request: ApexExecuteRequest) -> ApexResult<ApexExecuteResponse> {
        let start = Instant::now();
        let state = self.state.read().await;

        // Increment request counter
        state
            .request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Step 1: Route to best agent
        let routing = if let Some(ref agent) = request.force_agent {
            // Forced agent selection
            RoutingDecision {
                agent_id: agent.clone(),
                ts_score: 1.0,
                expected_reward: 0.5,
                uncertainty: 0.5,
                is_exploration: false,
                slot: CapabilitySlot::WarmSurface,
                alternatives: vec![],
                timestamp: current_timestamp_millis(),
            }
        } else {
            state.router.select_agent(&request.task)?
        };

        // Step 2: Check circuit breaker
        if !state.circuits.allow_request(&routing.agent_id)? {
            warn!(
                agent = %routing.agent_id,
                "Circuit breaker OPEN - failing fast"
            );

            return Ok(ApexExecuteResponse {
                success: false,
                pat_contributions: vec![],
                sat_contributions: vec![],
                synergy_score: 0.0,
                ihsan_score: 0.0,
                latency_ms: start.elapsed().as_millis() as u64,
                routing: Some(routing.clone()),
                receipt_id: None,
                error: Some(format!(
                    "Circuit breaker open for agent: {}",
                    routing.agent_id
                )),
                meta: serde_json::json!({
                    "circuit_state": "OPEN",
                    "agent_id": routing.agent_id,
                }),
            });
        }

        // Step 3: Optimize context if enabled
        let optimized_task = if request.optimize_context {
            match state.optimizer.compress_context(&request.task) {
                Ok(result) => {
                    if result.ratio < 0.9 {
                        info!(
                            original = result.original_tokens,
                            compressed = result.compressed_tokens,
                            ratio = result.ratio,
                            "Context optimized"
                        );
                    }
                    result.content
                }
                Err(e) => {
                    warn!(error = %e, "Context optimization failed, using original");
                    request.task.clone()
                }
            }
        } else {
            request.task.clone()
        };

        // Step 4: Execute through bridge (if available)
        let (response, execution_error) = if let Some(ref bridge) = state.bridge {
            let mut bridge_request: DualAgenticRequest = request.clone().into();
            bridge_request.task = optimized_task.clone();

            match bridge.execute(bridge_request).await {
                Ok(resp) => (Some(resp), None),
                Err(e) => {
                    state.circuits.record_failure(&routing.agent_id)?;
                    (None, Some(e.to_string()))
                }
            }
        } else {
            // No bridge - simulate execution for testing
            let simulated = DualAgenticResponse {
                pat_contributions: vec![format!("Simulated response from {}", routing.agent_id)],
                sat_contributions: vec!["Simulated SAT validation".to_string()],
                synergy_score: 0.85,
                ihsan_score: 0.96,
                latency: start.elapsed(),
                meta: serde_json::json!({"mode": "simulated"}),
            };
            (Some(simulated), None)
        };

        // Step 5: Process result
        let (success, final_response) = match (response, execution_error) {
            (Some(resp), None) => {
                // Record success
                state.circuits.record_success(&routing.agent_id)?;

                // Update Ihsan pass counter
                if resp.ihsan_score >= ihsan::constitution().threshold() {
                    state
                        .ihsan_passes
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                // Step 6: Learning feedback
                if request.enable_learning {
                    let output = resp.pat_contributions.join("\n");
                    let task_id = generate_task_id(&request.task);

                    if let Err(e) = state.learning.learn(
                        &routing.agent_id,
                        &task_id,
                        &request.task,
                        &output,
                        start.elapsed(),
                        true,
                        None,
                    ) {
                        warn!(error = %e, "Learning feedback failed");
                    }
                }

                (
                    true,
                    ApexExecuteResponse {
                        success: true,
                        pat_contributions: resp.pat_contributions,
                        sat_contributions: resp.sat_contributions,
                        synergy_score: resp.synergy_score,
                        ihsan_score: resp.ihsan_score,
                        latency_ms: start.elapsed().as_millis() as u64,
                        routing: Some(routing),
                        receipt_id: None, // Will be set by receipts
                        error: None,
                        meta: resp.meta,
                    },
                )
            }
            (_, Some(err)) => {
                // Learning from failure
                if request.enable_learning {
                    let task_id = generate_task_id(&request.task);
                    let _ = state.learning.learn(
                        &routing.agent_id,
                        &task_id,
                        &request.task,
                        "",
                        start.elapsed(),
                        false,
                        Some(err.clone()),
                    );
                }

                (
                    false,
                    ApexExecuteResponse {
                        success: false,
                        pat_contributions: vec![],
                        sat_contributions: vec![],
                        synergy_score: 0.0,
                        ihsan_score: 0.0,
                        latency_ms: start.elapsed().as_millis() as u64,
                        routing: Some(routing),
                        receipt_id: None,
                        error: Some(err),
                        meta: serde_json::json!({"failure": true}),
                    },
                )
            }
            (None, None) => {
                // Shouldn't happen but handle gracefully
                (
                    false,
                    ApexExecuteResponse {
                        success: false,
                        pat_contributions: vec![],
                        sat_contributions: vec![],
                        synergy_score: 0.0,
                        ihsan_score: 0.0,
                        latency_ms: start.elapsed().as_millis() as u64,
                        routing: Some(routing),
                        receipt_id: None,
                        error: Some("Unknown execution error".to_string()),
                        meta: serde_json::json!({}),
                    },
                )
            }
        };

        // Record metrics
        metrics::record_request_completion(
            if success { "success" } else { "failure" },
            start.elapsed().as_secs_f64(),
            final_response.synergy_score,
        );

        Ok(final_response)
    }

    /// Get routing decision without execution
    pub async fn route(&self, request: ApexRouteRequest) -> ApexResult<ApexRouteResponse> {
        let state = self.state.read().await;

        let decision = state.router.select_agent(&request.task)?;
        let circuit_open = state.circuits.get_state(&decision.agent_id) == CircuitState::Open;

        Ok(ApexRouteResponse {
            slot: decision.slot.name().to_string(),
            alternatives: decision.alternatives.clone(),
            circuit_open,
            decision,
        })
    }

    /// Get orchestrator metrics
    pub async fn get_metrics(&self) -> ApexResult<ApexMetricsResponse> {
        let state = self.state.read().await;

        let router_stats = state.router.get_stats();
        let learning_stats = state.learning.get_stats()?;
        let circuit_stats = state.circuits.get_all_stats();
        let optimizer_stats = state.optimizer.get_stats();

        let total_requests = state
            .request_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let ihsan_passes = state
            .ihsan_passes
            .load(std::sync::atomic::Ordering::Relaxed);
        let open_circuits = state.circuits.get_open_circuits().len();

        let health = ApexHealth {
            // System is healthy if:
            // 1. Fewer than 3 open circuits AND
            // 2. Either no records yet (fresh start) OR avg_quality >= 0.7
            healthy: open_circuits < 3
                && (learning_stats.total_records == 0 || learning_stats.avg_quality >= 0.7),
            open_circuits,
            avg_quality: learning_stats.avg_quality,
            total_requests,
            requests_last_hour: learning_stats.total_records as u64, // Approximation
            ihsan_pass_rate: if total_requests > 0 {
                ihsan_passes as f64 / total_requests as f64
            } else {
                1.0
            },
        };

        Ok(ApexMetricsResponse {
            router: router_stats,
            learning: learning_stats,
            circuits: circuit_stats,
            optimizer: optimizer_stats,
            health,
        })
    }

    /// Create Axum router for HTTP endpoints
    pub fn create_router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/apex/execute", post(handle_execute))
            .route("/apex/route", post(handle_route))
            .route("/apex/metrics", get(handle_metrics))
            .with_state(self)
    }
}

// HTTP Handlers

async fn handle_execute(
    State(orchestrator): State<Arc<ApexOrchestrator>>,
    Json(request): Json<ApexExecuteRequest>,
) -> impl IntoResponse {
    match orchestrator.execute(request).await {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(e) => {
            error!(error = %e, "Apex execute failed");
            let error_response = ApexExecuteResponse {
                success: false,
                pat_contributions: vec![],
                sat_contributions: vec![],
                synergy_score: 0.0,
                ihsan_score: 0.0,
                latency_ms: 0,
                routing: None,
                receipt_id: None,
                error: Some(e.to_string()),
                meta: serde_json::json!({}),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
        }
    }
}

async fn handle_route(
    State(orchestrator): State<Arc<ApexOrchestrator>>,
    Json(request): Json<ApexRouteRequest>,
) -> impl IntoResponse {
    match orchestrator.route(request).await {
        Ok(response) => (StatusCode::OK, Json(serde_json::json!(response))),
        Err(e) => {
            error!(error = %e, "Apex route failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        }
    }
}

async fn handle_metrics(State(orchestrator): State<Arc<ApexOrchestrator>>) -> impl IntoResponse {
    match orchestrator.get_metrics().await {
        Ok(metrics) => (StatusCode::OK, Json(serde_json::json!(metrics))),
        Err(e) => {
            error!(error = %e, "Apex metrics failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        }
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Generate task ID from content
fn generate_task_id(task: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(task.as_bytes());
    format!(
        "task_{:x}",
        &hash[..8].iter().fold(0u64, |acc, &b| acc << 8 | b as u64)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = ApexOrchestrator::new().await.unwrap();
        let metrics = orchestrator.get_metrics().await.unwrap();

        assert!(metrics.router.total_agents > 0);
        assert!(metrics.health.healthy);
    }

    #[tokio::test]
    async fn test_routing() {
        let orchestrator = ApexOrchestrator::new().await.unwrap();

        let request = ApexRouteRequest {
            task: "Verify the security of this deployment".to_string(),
            include_capabilities: false,
        };

        let response = orchestrator.route(request).await.unwrap();

        assert!(!response.decision.agent_id.is_empty());
        assert!(!response.circuit_open);
    }

    #[tokio::test]
    async fn test_execution_without_bridge() {
        let orchestrator = ApexOrchestrator::new().await.unwrap();

        let request = ApexExecuteRequest {
            user_id: "test_user".to_string(),
            task: "Generate a simple greeting".to_string(),
            requirements: vec![],
            target: "greeting".to_string(),
            context: HashMap::new(),
            force_agent: None,
            optimize_context: true,
            enable_learning: true,
        };

        let response = orchestrator.execute(request).await.unwrap();

        // Without bridge, execution is simulated
        assert!(response.success);
        assert!(response.routing.is_some());
    }

    #[tokio::test]
    async fn test_circuit_breaker_integration() {
        let orchestrator = ApexOrchestrator::new().await.unwrap();

        // Get the agent ID that would be selected
        let route_request = ApexRouteRequest {
            task: "Test task".to_string(),
            include_capabilities: false,
        };
        let route_response = orchestrator.route(route_request).await.unwrap();
        let agent_id = route_response.decision.agent_id;

        // Manually trip the circuit breaker
        {
            let state = orchestrator.state.read().await;
            for _ in 0..5 {
                state.circuits.record_failure(&agent_id).unwrap();
            }
        }

        // Execute should fail fast
        let request = ApexExecuteRequest {
            user_id: "test_user".to_string(),
            task: "Test task".to_string(),
            requirements: vec![],
            target: "test".to_string(),
            context: HashMap::new(),
            force_agent: Some(agent_id.clone()),
            optimize_context: false,
            enable_learning: false,
        };

        let response = orchestrator.execute(request).await.unwrap();

        assert!(!response.success);
        assert!(response.error.unwrap().contains("Circuit breaker"));
    }
}
