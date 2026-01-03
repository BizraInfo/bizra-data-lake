// src/errors.rs - Error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PATError {
    #[error("Agent execution failed: {0}")]
    AgentExecutionError(String),

    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),
}

#[derive(Error, Debug)]
pub enum SATError {
    #[error("Validation failed: {0}")]
    ValidationError(String),

    #[error("Consensus not reached: {0}")]
    ConsensusError(String),

    #[error("Security violation: {0}")]
    SecurityError(String),
}

#[derive(Error, Debug)]
pub enum SystemError {
    #[error("Bridge coordination failed: {0}")]
    BridgeError(String),

    #[error("Resource exhaustion: {0}")]
    ResourceError(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("SAT BLOCKED: {message} (escalation={escalation_id}, receipt={receipt_id})")]
    SatBlocked {
        message: String,
        escalation_id: String,
        receipt_id: String,
    },
    #[error(
        "IHSAN GATE FAILED: env={env} score={score:.4} < threshold={threshold:.4} (escalation={escalation_id})"
    )]
    IhsanGateFailed {
        env: String,
        score: f64,
        threshold: f64,
        escalation_id: String,
    },
}

#[derive(Error, Debug)]
pub enum PolicyError {
    #[error("MCP TOOLS BLOCKED: {message}")]
    McpToolsBlocked { message: String },
    #[error("IHSAN GATE FAILED: env={env} score={score:.4} < threshold={threshold:.4}")]
    IhsanGateFailed {
        env: String,
        score: f64,
        threshold: f64,
    },
}

#[derive(serde::Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OpErrorCode {
    SatBlocked,
    IhsanGateFailed,
    McpPolicyBlocked,
    ExecutionFailed,
    InternalError,
}
