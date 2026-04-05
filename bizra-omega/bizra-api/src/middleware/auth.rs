//! Authentication middleware for privileged API routes.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::{error::ApiError, state::AppState};

const BEARER_PREFIX: &str = "Bearer ";

/// Constant-time byte comparison to prevent timing side-channel attacks.
/// Always compares every byte regardless of early mismatches.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Require `Authorization: Bearer <token>` for protected endpoints.
///
/// Fail-closed by default: if `BIZRA_API_TOKEN` is unset, requests are denied.
pub async fn require_api_token(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(expected) = state.api_token() else {
        tracing::error!("BIZRA_API_TOKEN is not set; denying protected request");
        return ApiError::Unauthorized.into_response();
    };

    let provided = request
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .map(|value| {
            if let Some(token) = value.strip_prefix(BEARER_PREFIX) {
                token.trim()
            } else {
                value
            }
        });

    match provided {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
            next.run(request).await
        }
        _ => ApiError::Unauthorized.into_response(),
    }
}
