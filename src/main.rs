// src/main.rs - Complete unified system entry point

use meta_alpha_dual_agentic::types::{
    AdapterModes, DualAgenticRequest, EnhancedDualAgenticRequest, ReasoningMethod, SlashCommand,
};
use meta_alpha_dual_agentic::{create_http_server, ihsan, metrics, pat_enhanced, MetaAlphaDualAgentic};
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Initialize Prometheus metrics
    metrics::init_metrics();

    let constitution = ihsan::constitution();
    let adapter_modes = AdapterModes::current();
    println!(
        "BIZRA Meta Alpha (experimental)\n- version: {}\n- ihsan_constitution: {} (threshold {})\n- adapter_modes: {:?}\n",
        env!("CARGO_PKG_VERSION"),
        constitution.id(),
        constitution.threshold(),
        adapter_modes,
    );

    let system = Arc::new(MetaAlphaDualAgentic::initialize().await?);
    println!("Core system initialized");

    let enhanced_pat = Arc::new(pat_enhanced::EnhancedPATOrchestrator::new().await?);
    println!("Enhanced PAT initialized");

    demo_complete_system(&system, &enhanced_pat).await?;

    println!("Starting HTTP server on http://127.0.0.1:8080");
    create_http_server(system, 8080).await?;

    Ok(())
}

async fn demo_complete_system(
    base_system: &MetaAlphaDualAgentic,
    enhanced_pat: &pat_enhanced::EnhancedPATOrchestrator,
) -> anyhow::Result<()> {
    println!("\nDemo: Complete System Execution");

    // Test 1: Slash command - List tools
    println!("\n[Test 1: Slash Command - /tools]");
    let request1 = EnhancedDualAgenticRequest {
        base: DualAgenticRequest {
            user_id: "demo_user".to_string(),
            task: "List available tools".to_string(),
            requirements: vec![],
            target: "tools".to_string(),
            ..Default::default()
        },
        slash_command: Some(SlashCommand::Tools {
            filter: "search".to_string(),
        }),
        ..Default::default()
    };

    let response1 = enhanced_pat.execute_enhanced(request1).await?;
    println!("Result: {} tools found", response1.pat_contributions.len());

    // Test 2: Multi-reasoning with Graph of Thought
    println!("\n[Test 2: Multi-Reasoning - Graph of Thought]");
    let request2 = EnhancedDualAgenticRequest {
        base: DualAgenticRequest {
            user_id: "demo_user".to_string(),
            task: "Design strategic roadmap for BIZRA".to_string(),
            requirements: vec!["innovation".to_string(), "sustainability".to_string()],
            target: "strategic_plan".to_string(),
            ..Default::default()
        },
        reasoning_preference: Some(ReasoningMethod::GraphOfThought),
        ..Default::default()
    };

    let response2 = enhanced_pat.execute_enhanced(request2).await?;
    println!("Ihsan score: {:.3}", response2.ihsan_score);
    println!("Synergy score: {:.3}", response2.synergy_score);
    println!("Latency: {:?}", response2.latency);

    // Test 3: Sub-agent spawning
    println!("\n[Test 3: Sub-Agent Spawning]");
    let request3 = EnhancedDualAgenticRequest {
        base: DualAgenticRequest {
            user_id: "demo_user".to_string(),
            task: "Complex multi-phase project".to_string(),
            requirements: vec!["research".to_string(), "development".to_string()],
            target: "full_project".to_string(),
            ..Default::default()
        },
        enable_sub_agents: true,
        slash_command: Some(SlashCommand::Spawn {
            role: "Research Specialist".to_string(),
            task: "Market analysis for BIZRA ecosystem".to_string(),
        }),
        ..Default::default()
    };

    let response3 = enhanced_pat.execute_enhanced(request3).await?;
    let total_sub_agents = response3
        .meta
        .get("total_sub_agents")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    println!("Sub-agents spawned: {}", total_sub_agents);

    // Test 4: Base dual-agentic execution
    println!("\n[Test 4: Base Dual-Agentic Execution]");
    let request4 = DualAgenticRequest {
        user_id: "demo_user".to_string(),
        task: "Complete production deployment strategy".to_string(),
        requirements: vec!["scalability".to_string(), "reliability".to_string()],
        target: "production".to_string(),
        ..Default::default()
    };

    let response4 = base_system.execute(request4).await?;
    println!("PAT contributions: {}", response4.pat_contributions.len());
    println!("SAT evaluations: {}", response4.sat_contributions.len());
    println!("Synergy score: {:.3}", response4.synergy_score);
    println!("Ihsan score: {:.3}", response4.ihsan_score);

    println!("\nDemo complete; adapters are simulated by default.\n");
    Ok(())
}
