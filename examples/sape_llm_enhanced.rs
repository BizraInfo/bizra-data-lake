// examples/sape_llm_enhanced.rs
//
// Demonstrates the LLM-enhanced SAPE probe mode for critical dimensions:
// - Correctness
// - Groundedness
// - Relevance
//
// Usage:
//   cargo run --example sape_llm_enhanced

use meta_alpha_dual_agentic::sape::{get_sape, ProbeDimension};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info,meta_alpha_dual_agentic=debug")
        .init();

    println!("🧠 SAPE LLM-Enhanced Probe Demonstration\n");
    println!("This example shows the difference between heuristic and LLM-enhanced probes.\n");

    // Test content that's challenging for heuristics but good for LLM
    let test_content = r#"
According to the 2023 Climate Report by the IPCC, global temperatures have risen
by 1.1°C since pre-industrial times. This finding is supported by satellite data
from NASA's GISS Surface Temperature Analysis. The report concludes that immediate
action is required to limit warming to 1.5°C, as outlined in the Paris Agreement.
"#;

    println!("📝 Test Content:");
    println!("{}", test_content);
    println!("\n{}\n", "=".repeat(60));

    // Get SAPE engine
    let sape = get_sape();

    // Test 1: Heuristic-only mode
    println!("🔍 Test 1: HEURISTIC-ONLY MODE");
    {
        let mut engine = sape.lock().unwrap();
        engine.set_llm_enhanced(false);

        let results = engine.execute_probes(test_content);

        for result in &results {
            if matches!(
                result.dimension,
                ProbeDimension::Correctness
                    | ProbeDimension::Groundedness
                    | ProbeDimension::Relevance
            ) {
                println!(
                    "  {:15} → Score: {:.3}, Confidence: {:.3}, Flags: {:?}",
                    result.dimension.name(),
                    result.score,
                    result.confidence,
                    result.flags
                );
            }
        }
    }

    println!("\n{}\n", "=".repeat(60));

    // Test 2: LLM-enhanced mode (async required)
    println!("🧠 Test 2: LLM-ENHANCED MODE");
    {
        let mut engine = sape.lock().unwrap();
        engine.set_llm_enhanced(true);
        drop(engine); // Release lock before async operation

        let mut engine = sape.lock().unwrap();
        let results = engine.execute_probes_enhanced(test_content).await;

        for result in &results {
            if matches!(
                result.dimension,
                ProbeDimension::Correctness
                    | ProbeDimension::Groundedness
                    | ProbeDimension::Relevance
            ) {
                println!(
                    "  {:15} → Score: {:.3}, Confidence: {:.3}, Flags: {:?}",
                    result.dimension.name(),
                    result.score,
                    result.confidence,
                    result.flags
                );
            }
        }
    }

    println!("\n{}\n", "=".repeat(60));
    println!("✅ Demonstration complete!");
    println!("\nKey Differences:");
    println!("• Heuristic mode: Fast, keyword-based, lower confidence");
    println!("• LLM-enhanced mode: Slower, semantic understanding, higher accuracy");
    println!("• LLM probes gracefully fall back to heuristics on failure");

    Ok(())
}
