//! tests/integration_intelligence.rs
//! Runs against local endpoints. Skips gracefully if offline.
use std::time::Duration;

use bizra_node0::llm_gateway::{LlmGateway, LlmRequest, SlotsConfig};

#[tokio::test]
async fn intelligence_connectivity_smoke() {
    // Only run when explicitly enabled to avoid CI failures.
    if std::env::var("BIZRA_INT_TESTS").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set BIZRA_INT_TESTS=1 to run integration tests.");
        return;
    }

    let cfg_text = std::fs::read_to_string("docs/runtime/slots.yaml")
        .expect("docs/runtime/slots.yaml missing");
    let cfg: SlotsConfig = serde_yaml::from_str(&cfg_text).expect("slots.yaml parse failed");
    let gw = LlmGateway::new(cfg);

    // Quick call to cold_core
    let r = gw.complete(LlmRequest {
        slot: "cold_core".to_string(),
        prompt: "What is 2+2? Return ONLY the number.".to_string(),
    }).await;

    match r {
        Ok(out) => assert!(out.text.trim().starts_with('4'), "unexpected: {}", out.text),
        Err(e) => panic!("Connectivity failed: {e:?}"),
    }
}
