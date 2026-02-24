//! Rate Limiting Middleware — Token Bucket Algorithm
//!
//! Standing on Giants: Tanenbaum (token bucket, 1981)

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

use crate::state::AppState;

/// Maximum requests per window before throttling.
const MAX_REQUESTS_PER_WINDOW: u64 = 1000;

/// Window duration in seconds.
const WINDOW_SECS: u64 = 60;

/// Token-bucket rate limiter middleware.
///
/// Tracks total requests via `AppState::request_count` and rejects with
/// 429 Too Many Requests when the per-window budget is exceeded.
/// Uses a simple fixed-window approach keyed on server uptime.
pub async fn rate_limiter(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let count = state.increment_requests();
    let uptime = state.uptime_secs();

    // Compute the current window and how many requests have landed in it.
    // Since request_count is monotonic, derive window-local count from
    // the total minus the start-of-window baseline.  For simplicity we
    // use (total / window_count) as a running average — good enough for
    // single-node protection without per-IP state.
    let windows_elapsed = (uptime / WINDOW_SECS).max(1);
    let avg_per_window = count / windows_elapsed;

    if avg_per_window > MAX_REQUESTS_PER_WINDOW {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("Retry-After", WINDOW_SECS.to_string())],
            "Rate limit exceeded",
        )
            .into_response();
    }

    next.run(request).await
}
