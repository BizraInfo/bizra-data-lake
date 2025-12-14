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
