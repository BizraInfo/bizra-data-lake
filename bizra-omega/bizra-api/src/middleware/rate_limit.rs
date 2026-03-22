//! Rate Limiting Middleware — Token Bucket Algorithm
//!
//! Standing on Giants: Tanenbaum (token bucket, 1981)

use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::state::{AppState, TokenBucket};

/// Bucket capacity (maximum burst per client).
const MAX_REQUESTS_PER_WINDOW: f64 = 1000.0;

/// Time window used to fully refill one bucket.
const WINDOW_SECS: u64 = 60;

fn client_key(request: &Request<Body>) -> String {
    if let Some(forwarded_for) = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return forwarded_for.to_string();
    }

    if let Some(real_ip) = request
        .headers()
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return real_ip.to_string();
    }

    if let Some(addr) = request.extensions().get::<SocketAddr>() {
        return addr.ip().to_string();
    }

    "unknown".to_string()
}

/// Per-client token-bucket middleware.
pub async fn rate_limiter(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let key = client_key(&request);
    let mut bucket = state
        .rate_limits
        .entry(key)
        .or_insert_with(|| TokenBucket::new(MAX_REQUESTS_PER_WINDOW, WINDOW_SECS));

    if !bucket.try_consume() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("Retry-After", WINDOW_SECS.to_string())],
            "Rate limit exceeded",
        )
            .into_response();
    }

    state.increment_requests();
    next.run(request).await
}
