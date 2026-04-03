// src/lib.rs - Library entry point

pub mod a2a;
pub mod apex;
pub mod autopoietic;
pub mod bizra_integration;
pub mod blockchain;
pub mod bridge;
pub mod crypto_proofs;
pub mod embeddings;
pub mod entropy;
pub mod errors;
pub mod fate;
pub mod federation;
pub mod http;
pub mod idempotency;
pub mod ifc;
pub mod ihsan;
pub mod kernel;
pub mod lmstudio;
pub mod mcp;
pub mod merkle;
pub mod mission;
pub mod metrics;
pub mod model_router;
pub mod node0_unified;
pub mod ollama;
pub mod pat;
pub mod pat_enhanced;
pub mod pci;
pub mod reasoning;
pub mod receipts;
pub mod sape;
pub mod signing;
pub mod sape_parallel;
pub mod sat;
pub mod sovereign;
pub mod sovereign_runtime_omega;
pub mod sovereignty;
pub mod synapse;
pub mod types;
pub mod unified;
pub mod utils;
pub mod vectors;
pub mod voice;
pub mod wisdom;

use bridge::BridgeCoordinator;
use tracing::info;
use types::{DualAgenticRequest, DualAgenticResponse};

/// Complete Meta Alpha Dual Agentic System
pub struct MetaAlphaDualAgentic {
    bridge: BridgeCoordinator,
}

impl MetaAlphaDualAgentic {
    /// Initialize the complete system
    pub async fn initialize() -> anyhow::Result<Self> {
        info!("🚀 Initializing BIZRA META ALPHA ELITE - Complete Unified System");

        let bridge = BridgeCoordinator::new().await?;

        info!("✅ Core system initialized successfully");

        Ok(Self { bridge })
    }

    /// Execute dual-agentic workflow
    pub async fn execute(
        &self,
        request: DualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        self.bridge.execute(request).await
    }
}

// Re-export for convenience
pub use http::create_http_server;
