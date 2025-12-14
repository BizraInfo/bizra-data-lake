// src/lib.rs - Library entry point

pub mod a2a;
pub mod bizra_integration;
pub mod bridge;
pub mod errors;
pub mod http;
pub mod mcp;
pub mod pat;
pub mod pat_enhanced;
pub mod reasoning;
pub mod sat;
pub mod types;

use bridge::BridgeCoordinator;
use types::{DualAgenticRequest, DualAgenticResponse};
use tracing::info;

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
