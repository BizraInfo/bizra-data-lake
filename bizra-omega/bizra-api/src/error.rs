//! API Error Types

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Identity not initialized")]
    IdentityNotInitialized,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("PCI envelope verification failed: {0}")]
    PCIVerificationFailed(String),

    #[error("Gate check failed: {0}")]
    GateCheckFailed(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Federation error: {0}")]
    FederationError(String),

    #[error("Constitution violation: {0}")]
    ConstitutionViolation(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message, details) = match &self {
            ApiError::IdentityNotInitialized => (
                StatusCode::SERVICE_UNAVAILABLE,
                "IDENTITY_NOT_INITIALIZED",
                self.to_string(),
                None,
            ),
            ApiError::InvalidSignature => (
                StatusCode::UNAUTHORIZED,
                "INVALID_SIGNATURE",
                self.to_string(),
                None,
            ),
            ApiError::PCIVerificationFailed(d) => (
                StatusCode::BAD_REQUEST,
                "PCI_VERIFICATION_FAILED",
                "PCI envelope verification failed".into(),
                Some(d.clone()),
            ),
            ApiError::GateCheckFailed(d) => (
                StatusCode::FORBIDDEN,
                "GATE_CHECK_FAILED",
                "Gate validation failed".into(),
                Some(d.clone()),
            ),
            ApiError::InferenceError(d) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INFERENCE_ERROR",
                "Inference failed".into(),
                Some(d.clone()),
            ),
            ApiError::FederationError(d) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "FEDERATION_ERROR",
                "Federation error".into(),
                Some(d.clone()),
            ),
            ApiError::ConstitutionViolation(d) => (
                StatusCode::FORBIDDEN,
                "CONSTITUTION_VIOLATION",
                "Constitution violation".into(),
                Some(d.clone()),
            ),
            ApiError::RateLimitExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "RATE_LIMIT_EXCEEDED",
                self.to_string(),
                None,
            ),
            ApiError::BadRequest(d) => (
                StatusCode::BAD_REQUEST,
                "BAD_REQUEST",
                "Invalid request".into(),
                Some(d.clone()),
            ),
            ApiError::Internal(d) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "Internal server error".into(),
                Some(d.clone()),
            ),
            ApiError::Io(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "IO_ERROR",
                "IO error".into(),
                Some(e.to_string()),
            ),
        };

        let body = Json(ErrorResponse {
            error: message,
            code: code.into(),
            details,
        });

        (status, body).into_response()
    }
}
