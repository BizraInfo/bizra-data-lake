//! NODE0 GENESIS CEREMONY - THE LIFE KISS
//!
//! This binary executes the Genesis ceremony for MoMo (Node0).
//! It mints the first identity, shares hardware and knowledge,
//! creates the 7 PAT and 5 SAT, and brings the Resource Pool to life.
//!
//! بذرة واحدة تصنع غابة
//! One seed makes a forest.

use bizra_resourcepool::genesis::*;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // MoMo's hardware contribution
    let hardware = HardwareContribution {
        cpu: "Intel i9-14900HX".to_string(),
        cpu_cores: 24,
        gpu: "NVIDIA RTX 4090 (MSI Titan GT77 HX)".to_string(),
        gpu_vram: 16 * 1024 * 1024 * 1024,      // 16 GB
        ram: 128 * 1024 * 1024 * 1024,          // 128 GB
        storage: 4 * 1024 * 1024 * 1024 * 1024, // 4 TB
        network_bps: 1000 * 1024 * 1024,        // 1 Gbps
        compute_units_per_day: 100_000,
    };

    // MoMo's knowledge contribution (BIZRA Data Lake)
    let knowledge = KnowledgeContribution {
        conversations: 1241,
        messages: 24746,
        data_size: 5_800_000_000, // 5.4 GB + indexing
        date_range: ("2023-07-22".to_string(), "2025-12-16".to_string()),
        concepts: vec![
            "BIZRA".to_string(),
            "Ihsān".to_string(),
            "Federation".to_string(),
            "Sovereign AI".to_string(),
            "Proof of Impact".to_string(),
            "Telescript".to_string(),
            "Resource Pool".to_string(),
            "Web4".to_string(),
        ],
        knowledge_hash: compute_knowledge_hash(),
    };

    // Execute the Genesis ceremony
    let mut engine = GenesisEngine::new();
    let genesis = engine.execute_genesis("MoMo (محمد)", "Dubai, UAE (GMT+4)", hardware, knowledge);

    // Save the genesis record
    let genesis_json = serde_json::to_string_pretty(&genesis)?;

    // Create sovereign state directory if needed
    let state_dir = Path::new("/mnt/c/BIZRA-DATA-LAKE/sovereign_state");
    if !state_dir.exists() {
        fs::create_dir_all(state_dir)?;
    }

    // Write genesis record
    let genesis_path = state_dir.join("node0_genesis.json");
    fs::write(&genesis_path, &genesis_json)?;
    println!("✓ Genesis record saved to: {}", genesis_path.display());

    // Write genesis hash for quick verification
    let hash_path = state_dir.join("genesis_hash.txt");
    fs::write(&hash_path, hex::encode(genesis.genesis_hash))?;
    println!("✓ Genesis hash saved to: {}", hash_path.display());

    // Write PAT team roster
    let pat_roster = genesis
        .pat_team
        .agents
        .iter()
        .map(|a| format!("{}: {} ({})", a.role, a.agent_id, &a.public_key[..16]))
        .collect::<Vec<_>>()
        .join("\n");
    let pat_path = state_dir.join("pat_roster.txt");
    fs::write(&pat_path, &pat_roster)?;
    println!("✓ PAT roster saved to: {}", pat_path.display());

    // Write SAT team roster
    let sat_roster = genesis
        .sat_team
        .agents
        .iter()
        .map(|a| format!("{}: {} ({})", a.role, a.agent_id, &a.public_key[..16]))
        .collect::<Vec<_>>()
        .join("\n");
    let sat_path = state_dir.join("sat_roster.txt");
    fs::write(&sat_path, &sat_roster)?;
    println!("✓ SAT roster saved to: {}", sat_path.display());

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    NODE0 IS NOW LIVE");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Genesis Hash: {}", hex::encode(genesis.genesis_hash));
    println!("  Node ID: {}", genesis.identity.node_id);
    println!("  Timestamp: {}", genesis.timestamp);
    println!();
    println!("  The Resource Pool has received the life kiss from Node0.");
    println!("  The first seed has been planted.");
    println!("  The forest begins.");
    println!();

    Ok(())
}

/// Compute hash of the BIZRA Data Lake knowledge
fn compute_knowledge_hash() -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"BIZRA_DATA_LAKE_V1:");
    hasher.update(b"conversations:1241:");
    hasher.update(b"messages:24746:");
    hasher.update(b"date_range:2023-07-22:2025-12-16:");
    hasher.update(b"sources:ChatGPT:DeepSeek:Claude:Kimi:GoogleAI:");
    hasher.update(b"concepts:BIZRA:Ihsan:Federation:Sovereign:Proof:Telescript:Pool:Web4:");
    hasher.update(b"standing_on_giants:Shannon:Lamport:Al-Ghazali:Harberger:Rawls:Anthropic:");
    hasher.finalize().into()
}
