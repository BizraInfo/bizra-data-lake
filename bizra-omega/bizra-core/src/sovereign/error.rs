//! Sovereign Error Types — Comprehensive Error Handling
//!
//! Implements proper Result-based error propagation following
//! Rust best practices and the Standing on Giants protocol.
//!
//! # Error Categories
//!
//! - **SNR Errors**: Signal quality violations
//! - **Ihsan Errors**: Excellence threshold failures
//! - **Validation Errors**: Input/format issues
//! - **Operation Errors**: Runtime failures
//!
//! # Design Principles (Standing on Giants)
//!
//! - **Bernstein**: Fail-secure by default
//! - **Torvalds**: Explicit error handling, no silent failures
//! - **Lamport**: Errors carry enough context for distributed debugging

use std::fmt;
use thiserror::Error;

/// Sovereign operation result type
pub type SovereignResult<T> = Result<T, SovereignError>;

/// Comprehensive error type for sovereign operations
#[derive(Error, Debug, Clone)]
pub enum SovereignError {
    // ═══════════════════════════════════════════════════════════════════════
    // SNR Errors (Shannon-inspired)
    // ═══════════════════════════════════════════════════════════════════════
    /// SNR score below minimum threshold
    #[error("SNR violation: score {actual:.4} below threshold {threshold:.4}")]
    SNRBelowThreshold { actual: f64, threshold: f64 },

    /// Noise level exceeds acceptable bounds
    #[error("Excessive noise: {noise_level:.4} exceeds maximum {max_noise:.4}")]
    ExcessiveNoise { noise_level: f64, max_noise: f64 },

    /// Signal strength insufficient
    #[error("Weak signal: strength {strength:.4} below minimum {minimum:.4}")]
    WeakSignal { strength: f64, minimum: f64 },

    /// Diversity score too low (repetitive content)
    #[error("Low diversity: {diversity:.4} indicates repetitive content")]
    LowDiversity { diversity: f64 },

    // ═══════════════════════════════════════════════════════════════════════
    // Ihsan Errors (Excellence constraint)
    // ═══════════════════════════════════════════════════════════════════════
    /// Ihsan score below excellence threshold
    #[error("Ihsan violation: score {actual:.4} below excellence threshold {threshold:.4}")]
    IhsanViolation { actual: f64, threshold: f64 },

    /// Quality gate rejection
    #[error("Quality gate '{gate}' rejected: {reason}")]
    QualityGateRejection { gate: String, reason: String },

    // ═══════════════════════════════════════════════════════════════════════
    // Validation Errors
    // ═══════════════════════════════════════════════════════════════════════
    /// Input exceeds maximum allowed length
    #[error("Input too large: {size} bytes exceeds maximum {max_size} bytes")]
    InputTooLarge { size: usize, max_size: usize },

    /// Input below minimum required length
    #[error("Input too small: {size} bytes below minimum {min_size} bytes")]
    InputTooSmall { size: usize, min_size: usize },

    /// Empty input not allowed
    #[error("Empty input: content cannot be empty")]
    EmptyInput,

    /// Invalid JSON structure
    #[error("Invalid JSON: {message}")]
    InvalidJson { message: String },

    /// Schema validation failed
    #[error("Schema validation failed: {message}")]
    SchemaValidation { message: String },

    // ═══════════════════════════════════════════════════════════════════════
    // Operation Errors
    // ═══════════════════════════════════════════════════════════════════════
    /// Operation timed out
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// Circuit breaker is open
    #[error("Circuit breaker open: {service} unavailable, retry after {retry_after_ms}ms")]
    CircuitBreakerOpen {
        service: String,
        retry_after_ms: u64,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {limit} requests per {window_seconds}s")]
    RateLimitExceeded { limit: u32, window_seconds: u32 },

    /// Internal operation failed
    #[error("Operation failed: {operation} - {reason}")]
    OperationFailed { operation: String, reason: String },

    // ═══════════════════════════════════════════════════════════════════════
    // Graph-of-Thoughts Errors
    // ═══════════════════════════════════════════════════════════════════════
    /// No consensus reached in reasoning paths
    #[error("No consensus: {successful}/{total} paths succeeded, threshold: {threshold}")]
    NoConsensus {
        successful: usize,
        total: usize,
        threshold: usize,
    },

    /// Reasoning path failed
    #[error("Reasoning path '{path_id}' failed at thought '{thought_id}': {reason}")]
    ReasoningPathFailed {
        path_id: String,
        thought_id: String,
        reason: String,
    },

    /// Maximum reasoning depth exceeded
    #[error("Maximum reasoning depth {max_depth} exceeded")]
    MaxDepthExceeded { max_depth: usize },

    // ═══════════════════════════════════════════════════════════════════════
    // Identity Errors
    // ═══════════════════════════════════════════════════════════════════════
    /// Identity not initialized
    #[error("Identity not initialized: call with_identity() first")]
    IdentityNotInitialized,

    /// Signature verification failed
    #[error("Signature verification failed: {reason}")]
    SignatureInvalid { reason: String },
}

impl SovereignError {
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            SovereignError::Timeout { .. }
                | SovereignError::CircuitBreakerOpen { .. }
                | SovereignError::RateLimitExceeded { .. }
        )
    }

    /// Check if this error is a quality violation
    pub fn is_quality_violation(&self) -> bool {
        matches!(
            self,
            SovereignError::SNRBelowThreshold { .. }
                | SovereignError::IhsanViolation { .. }
                | SovereignError::QualityGateRejection { .. }
                | SovereignError::LowDiversity { .. }
                | SovereignError::WeakSignal { .. }
                | SovereignError::ExcessiveNoise { .. }
        )
    }

    /// Get error severity (0.0 = info, 1.0 = critical)
    pub fn severity(&self) -> f64 {
        match self {
            // Critical errors
            SovereignError::SignatureInvalid { .. } => 1.0,
            SovereignError::IdentityNotInitialized => 0.9,

            // High severity
            SovereignError::IhsanViolation { .. } => 0.8,
            SovereignError::SNRBelowThreshold { .. } => 0.7,

            // Medium severity
            SovereignError::NoConsensus { .. } => 0.6,
            SovereignError::QualityGateRejection { .. } => 0.5,
            SovereignError::OperationFailed { .. } => 0.5,

            // Low severity (recoverable)
            SovereignError::Timeout { .. } => 0.3,
            SovereignError::CircuitBreakerOpen { .. } => 0.3,
            SovereignError::RateLimitExceeded { .. } => 0.2,

            // Validation (user error)
            SovereignError::InputTooLarge { .. } => 0.2,
            SovereignError::InputTooSmall { .. } => 0.2,
            SovereignError::EmptyInput => 0.1,
            SovereignError::InvalidJson { .. } => 0.2,
            SovereignError::SchemaValidation { .. } => 0.2,

            // Other
            _ => 0.5,
        }
    }

    /// Create an SNR threshold error
    pub fn snr_below(actual: f64, threshold: f64) -> Self {
        SovereignError::SNRBelowThreshold { actual, threshold }
    }

    /// Create an Ihsan violation error
    pub fn ihsan_violation(actual: f64, threshold: f64) -> Self {
        SovereignError::IhsanViolation { actual, threshold }
    }

    /// Create an input too large error
    pub fn input_too_large(size: usize, max_size: usize) -> Self {
        SovereignError::InputTooLarge { size, max_size }
    }

    /// Create an operation failed error
    pub fn operation_failed(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        SovereignError::OperationFailed {
            operation: operation.into(),
            reason: reason.into(),
        }
    }
}

/// Error context for distributed debugging (Lamport-inspired)
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Unique error ID
    pub error_id: String,
    /// Timestamp (microseconds since epoch)
    pub timestamp_us: u64,
    /// Node ID where error occurred
    pub node_id: Option<String>,
    /// Operation trace
    pub trace: Vec<String>,
    /// The actual error
    pub error: SovereignError,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(error: SovereignError) -> Self {
        Self {
            error_id: uuid::Uuid::new_v4().to_string(),
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            node_id: None,
            trace: Vec::new(),
            error,
        }
    }

    /// Add operation to trace
    pub fn with_trace(mut self, operation: impl Into<String>) -> Self {
        self.trace.push(operation.into());
        self
    }

    /// Set node ID
    pub fn with_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} (node: {}, trace: {:?})",
            self.error_id,
            self.error,
            self.node_id.as_deref().unwrap_or("unknown"),
            self.trace
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_error() {
        let err = SovereignError::snr_below(0.80, 0.85);
        assert!(err.is_quality_violation());
        assert!(!err.is_recoverable());
        assert!(err.severity() > 0.5);
    }

    #[test]
    fn test_ihsan_error() {
        let err = SovereignError::ihsan_violation(0.92, 0.95);
        assert!(err.is_quality_violation());
        assert_eq!(err.severity(), 0.8);
    }

    #[test]
    fn test_recoverable_errors() {
        let timeout = SovereignError::Timeout { duration_ms: 5000 };
        assert!(timeout.is_recoverable());

        let circuit = SovereignError::CircuitBreakerOpen {
            service: "inference".into(),
            retry_after_ms: 10000,
        };
        assert!(circuit.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let err = SovereignError::snr_below(0.80, 0.85);
        let ctx = ErrorContext::new(err)
            .with_node("node-001")
            .with_trace("validate_content")
            .with_trace("check_snr");

        assert!(ctx.node_id.is_some());
        assert_eq!(ctx.trace.len(), 2);
    }

    #[test]
    fn test_error_display() {
        let err = SovereignError::SNRBelowThreshold {
            actual: 0.8234,
            threshold: 0.85,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0.8234"));
        assert!(msg.contains("0.85"));
    }
}
