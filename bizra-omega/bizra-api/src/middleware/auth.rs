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
        Some(token) if token == expected => next.run(request).await,
        _ => ApiError::Unauthorized.into_response(),
    }
}
