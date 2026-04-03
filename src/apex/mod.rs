// src/apex/mod.rs - Apex Orchestrator Module
//
// Unified entry point for all agent operations with:
// - Thompson Sampling routing for intelligent agent selection
// - Context optimization for token efficiency
// - SONA self-learning feedback loops
// - Circuit breaker fault tolerance
//
// Integration with existing BIZRA infrastructure:
// - bridge.rs PAT-SAT flow
// - model_router.rs capability slots
// - sape.rs probe system
// - idempotency.rs patterns

pub mod circuit_breaker;
pub mod context_optimizer;
pub mod learning;
pub mod orchestrator;
pub mod router;

// Re-exports for convenience
pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use context_optimizer::ContextOptimizer;
pub use learning::{LearningLoop, PerformanceRecord};
pub use orchestrator::ApexOrchestrator;
pub use router::{CapabilityMatrix, ThompsonSamplingRouter};

use thiserror::Error;

/// Apex orchestrator errors
#[derive(Error, Debug)]
pub enum ApexError {
    #[error("Routing failed: {message}")]
    RoutingError { message: String },

    #[error("Context optimization failed: {message}")]
    ContextError { message: String },

    #[error("Circuit breaker open for agent: {agent_name}")]
    CircuitOpen { agent_name: String },

    #[error("Learning loop failed: {message}")]
    LearningError { message: String },

    #[error("Agent execution failed: {agent_name} - {message}")]
    AgentExecutionError { agent_name: String, message: String },

    #[error("FATE escalation required: {escalation_id}")]
    FateEscalation { escalation_id: String },

    #[error("Ihsan threshold not met: score={score:.4} < threshold={threshold:.4}")]
    IhsanGateFailed { score: f64, threshold: f64 },

    #[error("SAT validation failed: {message}")]
    SatValidationFailed { message: String },

    #[error("Bridge error: {0}")]
    BridgeError(#[from] crate::errors::BridgeError),

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Result type for Apex operations
pub type ApexResult<T> = Result<T, ApexError>;
