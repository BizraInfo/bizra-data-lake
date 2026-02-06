//! TELESCRIPT-BIZRA BRIDGE Demo
//!
//! Demonstrates the mobile agent framework with sovereign ethics.

use bizra_telescript::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("          TELESCRIPT-BIZRA BRIDGE v0.1.0 - DEMO");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Create the engine with Genesis Authority
    let engine = TelescriptEngine::new();
    println!("✓ Engine initialized with Genesis Authority");
    println!("  Authority: {}", engine.genesis_authority().name);
    println!();

    // Create places
    println!("▸ Creating Places...");
    let place_dubai = engine.create_place("node0://dubai.bizra", None).await?;
    let place_cairo = engine.create_place("node1://cairo.bizra", None).await?;
    println!(
        "  ✓ Place created: node0://dubai.bizra (ID: {})",
        place_dubai
    );
    println!(
        "  ✓ Place created: node1://cairo.bizra (ID: {})",
        place_cairo
    );
    println!();

    // Create authority delegation chain
    println!("▸ Authority Delegation Chain...");
    let genesis = engine.genesis_authority().clone();
    let region_auth = genesis.delegate("MENA-Region")?;
    let node_auth = region_auth.delegate("Dubai-Node")?;
    println!("  Genesis → MENA-Region → Dubai-Node");
    println!("  Delegation depth: {}", node_auth.delegation_depth);
    println!("  Chain verified: {}", node_auth.verify_chain());
    println!();

    // Create a permit (with capabilities available at Places)
    println!("▸ Creating Permit...");
    let permit = Permit::new(
        node_auth.clone(),
        vec![
            Capability::Go,
            Capability::Meet,
            Capability::Compute,
            Capability::Store,
        ],
        ResourceLimits {
            cpu_millicores: 500,
            memory_bytes: 256 * 1024 * 1024,
            storage_bytes: 1024 * 1024 * 1024,
            network_bps: 10 * 1024 * 1024,
            inference_tokens: 8192,
            ttl_seconds: 86400,
        },
        86400,
    );
    println!("  ✓ Permit issued with capabilities: Go, Meet, Compute, Store");
    println!("  Ihsān requirement: {}/1000", permit.ihsan_requirement);
    println!("  Valid: {}", permit.verify());
    println!();

    // Create an agent
    println!("▸ Creating Agent...");
    let agent_code = b"async fn execute() { seek_knowledge().await }".to_vec();
    let agent_id = engine
        .create_agent("Seeker-001", permit, agent_code, place_dubai)
        .await?;
    println!("  ✓ Agent created: Seeker-001 (ID: {})", agent_id);
    println!();

    // Travel to another place
    println!("▸ Agent Travel (Telescript go())...");
    println!("  Traveling from Dubai to Cairo...");
    let ticket = engine.go(agent_id, place_cairo).await?;
    println!("  ✓ Ticket issued (ID: {})", ticket.id);
    println!("  Ticket valid: {}", ticket.verify());

    // Arrive at destination
    engine.arrive(ticket).await?;
    println!("  ✓ Agent arrived at Cairo");
    println!();

    // Display Gini calculation
    println!("▸ Adl (Justice) Enforcement...");
    let fair_distribution = vec![10.0, 11.0, 12.0, 10.0, 11.0];
    let unfair_distribution = vec![1.0, 2.0, 3.0, 100.0];

    let gini_fair = TelescriptEngine::calculate_gini(&fair_distribution);
    let gini_unfair = TelescriptEngine::calculate_gini(&unfair_distribution);

    println!(
        "  Fair distribution [10,11,12,10,11]: Gini = {:.4}",
        gini_fair
    );
    println!(
        "  Unfair distribution [1,2,3,100]: Gini = {:.4}",
        gini_unfair
    );
    println!("  ADL_GINI_MAX threshold: {:.2}", ADL_GINI_MAX);
    println!(
        "  Fair passes ADL: {}",
        if gini_fair <= ADL_GINI_MAX {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  Unfair passes ADL: {}",
        if gini_unfair <= ADL_GINI_MAX {
            "✓"
        } else {
            "✗"
        }
    );
    println!();

    // Display engine statistics
    let stats = engine.stats().await;
    println!("▸ Engine Statistics...");
    println!("  Places: {}", stats.places_count);
    println!("  Agents: {}", stats.agents_count);
    println!("  Meetings: {}", stats.meetings_count);
    println!("  Connections: {}", stats.connections_count);
    println!("  Impact Log Entries: {}", stats.impact_log_count);
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  STANDING ON GIANTS:");
    println!("  Shannon (1948) • Lamport (1982) • Al-Ghazali (1095)");
    println!("  General Magic (1990) • Anthropic (2023)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  بذرة واحدة تصنع غابة");
    println!("  One seed makes a forest.");
    println!();

    Ok(())
}
