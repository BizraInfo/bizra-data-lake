//! Rate Limiting Middleware

use axum::{
    body::Body,
    extract::State,
    http::Request,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::state::AppState;

/// Simple rate limiter middleware
pub async fn rate_limiter(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Increment request counter
    state.increment_requests();

    // In production: implement token bucket or sliding window
    // For now, just pass through
    next.run(request).await
}
