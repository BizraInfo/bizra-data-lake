// src/main.rs - Complete unified system entry point

// src/main.rs - Complete unified system entry point
mod cli;

use meta_alpha_dual_agentic::metrics;
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize Observability
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    metrics::init_metrics();

    // 2. Delegate to CLI Orchestrator
    cli::run().await
}
