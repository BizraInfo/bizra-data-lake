//! Prometheus Metrics Handler

use axum::{extract::State, response::IntoResponse};
use std::sync::Arc;

use crate::state::AppState;

pub async fn prometheus_metrics(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let uptime = state.uptime_secs();
    let requests = state.get_request_count();

    // Prometheus text format
    format!(
        r#"# HELP bizra_uptime_seconds Time since server start
# TYPE bizra_uptime_seconds gauge
bizra_uptime_seconds {}

# HELP bizra_requests_total Total API requests
# TYPE bizra_requests_total counter
bizra_requests_total {}

# HELP bizra_ihsan_threshold Ihsan excellence threshold
# TYPE bizra_ihsan_threshold gauge
bizra_ihsan_threshold {}

# HELP bizra_snr_threshold Signal-to-noise ratio threshold
# TYPE bizra_snr_threshold gauge
bizra_snr_threshold {}
"#,
        uptime,
        requests,
        bizra_core::IHSAN_THRESHOLD,
        bizra_core::SNR_THRESHOLD,
    )
}
