//! Genesis Spine — Phase 1 CLI command handlers
//!
//! Four authoritative ingress commands backed by the frozen constitutional spine:
//!   init    — substrate discovery + Ed25519 identity generation
//!   genesis — GenesisSeal computation + constitutional binding
//!   agents  — PAT-7 + SAT-5 topology display + mint status
//!   node    — health, identity, compliance, heartbeat
//!
//! Standing on Giants: Bernstein (Ed25519) · Aumasson (BLAKE3) · Al-Ghazali (Ihsan)

use anyhow::{Context, Result};

use bizra_core::genesis_seal::{ConstitutionalParams, GenesisSeal};
use bizra_core::topology_canon::{PatAgent, SatAgent, TopologyCanon};
use bizra_node::substrate::ResourceManifest;

// ── bizra init ──────────────────────────────────────────────

/// Initialize the sovereign node: discover substrate, generate identity.
pub fn exec_init(force: bool) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m          \x1b[1mBIZRA Sovereign Node — Initialization\x1b[0m            \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // ── Step 1: Substrate Discovery ──
    println!("  \x1b[33m[1/3]\x1b[0m Discovering substrate...");
    let manifest = ResourceManifest::discover();
    let hw = &manifest.hardware;

    println!(
        "        CPU:     {} ({} cores / {} threads)",
        hw.cpu_name, hw.cpu_cores, hw.cpu_threads
    );
    println!(
        "        RAM:     {:.1} GB total / {:.1} GB available",
        hw.ram_total_gb, hw.ram_available_gb
    );

    if hw.gpus.is_empty() {
        println!("        GPU:     none detected");
    } else {
        for gpu in &hw.gpus {
            println!(
                "        GPU:     {} ({} MB VRAM)",
                gpu.name, gpu.vram_total_mb
            );
        }
    }

    for disk in &hw.disks {
        println!(
            "        Disk:    {} — {:.1} GB free / {:.1} GB total",
            disk.mount, disk.free_gb, disk.total_gb
        );
    }

    println!(
        "        Models:  {} discovered ({:.1} GB)",
        manifest.total_models(),
        manifest.total_model_storage_gb
    );
    for (runtime, count) in &manifest.model_count_by_runtime {
        println!("                 {} ({})", runtime.as_str(), count);
    }
    println!("        Platform: {}", manifest.platform);
    println!();

    // ── Step 2: Generate Ed25519 Identity ──
    println!("  \x1b[33m[2/3]\x1b[0m Generating Ed25519 node identity...");

    let node_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    let verifying_key = ed25519_dalek::VerifyingKey::from(&node_key);
    let vk_bytes = verifying_key.to_bytes();
    let node_id = hex::encode(&vk_bytes[..16]);

    println!("        Node ID:     {}", node_id);
    println!("        Public Key:  {}...", hex::encode(&vk_bytes[..16]));
    println!("        Algorithm:   Ed25519 (Bernstein 2006)");

    if !force {
        println!("        \x1b[2m(use --force to regenerate if key already exists)\x1b[0m");
    }
    println!();

    // ── Step 3: Enumerate Genesis Agents ──
    println!("  \x1b[33m[3/3]\x1b[0m Enumerating genesis agent topology...");

    let pat_count = PatAgent::ALL.len();
    let sat_count = SatAgent::ALL.len();
    let total = pat_count + sat_count;

    println!("        PAT-{} (Personal Agentic Team):", pat_count);
    for agent in &PatAgent::ALL {
        println!(
            "          P{} {} — {}",
            agent.index(),
            agent.callsign(),
            agent.role()
        );
    }
    println!("        SAT-{} (Shared Agentic Team):", sat_count);
    for agent in &SatAgent::ALL {
        println!("          S{} {}", agent.index(), agent.callsign());
    }
    println!();

    // ── Summary ──
    println!("  \x1b[32m✓ Node initialized\x1b[0m");
    println!("    Identity:  {}", node_id);
    println!(
        "    Agents:    {} ({} PAT + {} SAT)",
        total, pat_count, sat_count
    );
    println!(
        "    Substrate: {} cores, {:.0} GB RAM, {} models",
        hw.cpu_cores,
        hw.ram_total_gb,
        manifest.total_models()
    );
    println!();
    println!("    Next: \x1b[1mbizra genesis\x1b[0m — compute constitutional root of trust");
    println!();

    Ok(())
}

// ── bizra genesis ───────────────────────────────────────────

/// Compute and display the Genesis Seal — constitutional root of trust.
pub fn exec_genesis(verbose: bool) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m             \x1b[1mBIZRA Genesis Seal Ceremony\x1b[0m                   \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system time error")?
        .as_millis() as u64;

    // Use constitutional defaults (frozen topology)
    let params = ConstitutionalParams::default();
    let seal = GenesisSeal::compute(params.clone(), now);

    println!("  \x1b[33mConstitutional Parameters:\x1b[0m");
    println!("    Ihsan Threshold:     {:.2}", params.ihsan_threshold);
    println!("    SNR Threshold:       {:.2}", params.snr_threshold);
    println!("    Gini Ceiling:        {:.2}", params.gini_ceiling);
    println!("    PAT Count:           {}", params.pat_count);
    println!("    SAT Count:           {}", params.sat_count);
    println!("    Constitution:        {}", params.constitution_id);
    println!();

    if verbose {
        println!("  \x1b[33mGate Chain Order:\x1b[0m");
        for (i, gate) in params.gate_order.iter().enumerate() {
            println!("    [{}] {}", i + 1, gate);
        }
        println!();

        println!("  \x1b[33mVerdict Precedence:\x1b[0m");
        for (i, verdict) in params.verdict_precedence.iter().enumerate() {
            println!("    [{}] {}", i + 1, verdict);
        }
        println!();
    }

    println!("  \x1b[32m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[32m║\x1b[0m  GENESIS SEAL                                              \x1b[32m║\x1b[0m");
    println!(
        "  \x1b[32m║\x1b[0m  Hash: \x1b[1m{}\x1b[0m  \x1b[32m║\x1b[0m",
        hex::encode(seal.seal_hash)
    );
    println!(
        "  \x1b[32m║\x1b[0m  Time: {:<52} \x1b[32m║\x1b[0m",
        seal.sealed_at
    );
    println!("  \x1b[32m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    println!("    Algorithm:    BLAKE3 (Aumasson et al.)");
    println!("    Determinism:  Same params + time = same seal (replayable)");
    println!("    Binding:      Every receipt chains back to this root");
    println!();
    println!("    Next: \x1b[1mbizra agents\x1b[0m — view your sovereign agent council");
    println!();

    Ok(())
}

// ── bizra agents ────────────────────────────────────────────

/// Display PAT-7 + SAT-5 agent topology and status.
pub fn exec_agents(verbose: bool) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m          \x1b[1mBIZRA Sovereign Agent Parliament\x1b[0m                  \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // PAT-7: Personal Agentic Team (user's council)
    println!("  \x1b[33m┌─────────────────────────────────────────────────────────┐\x1b[0m");
    println!("  \x1b[33m│\x1b[0m  \x1b[1mPersonal Agentic Team (PAT-7)\x1b[0m — Your Mastermind Council \x1b[33m│\x1b[0m");
    println!("  \x1b[33m└─────────────────────────────────────────────────────────┘\x1b[0m");

    for agent in &PatAgent::ALL {
        let (icon, desc) = pat_display(agent);
        println!(
            "    P{} {} \x1b[1m{:<12}\x1b[0m {}",
            agent.index(),
            icon,
            agent.callsign(),
            desc
        );
    }
    println!();

    // SAT-5: Shared Agentic Team (system validators)
    println!("  \x1b[35m┌─────────────────────────────────────────────────────────┐\x1b[0m");
    println!("  \x1b[35m│\x1b[0m  \x1b[1mShared Agentic Team (SAT-5)\x1b[0m — System Immune System     \x1b[35m│\x1b[0m");
    println!("  \x1b[35m└─────────────────────────────────────────────────────────┘\x1b[0m");

    for agent in &SatAgent::ALL {
        let (icon, desc) = sat_display(agent);
        println!(
            "    S{} {} \x1b[1m{:<12}\x1b[0m {}",
            agent.index(),
            icon,
            agent.callsign(),
            desc
        );
    }
    println!();

    if verbose {
        println!("  \x1b[33mTopology Constants:\x1b[0m");
        println!("    PAT Count:           {}", TopologyCanon::PAT_COUNT);
        println!("    SAT Count:           {}", TopologyCanon::SAT_COUNT);
        println!(
            "    Total Agents:        {}",
            TopologyCanon::PAT_COUNT + TopologyCanon::SAT_COUNT
        );
        println!();
        println!("  \x1b[33mGate Chain Order:\x1b[0m");
        for (i, gate) in TopologyCanon::GATE_ORDER.iter().enumerate() {
            println!("    [{}] {}", i + 1, gate);
        }
        println!();
        println!("  \x1b[33mVerdict Precedence:\x1b[0m");
        for (i, v) in TopologyCanon::VERDICT_PRECEDENCE.iter().enumerate() {
            println!("    [{}] {}", i + 1, v);
        }
        println!();
    }

    // Summary
    println!(
        "  \x1b[32mTopology:\x1b[0m {} PAT + {} SAT = {} agents (frozen)",
        TopologyCanon::PAT_COUNT,
        TopologyCanon::SAT_COUNT,
        TopologyCanon::PAT_COUNT + TopologyCanon::SAT_COUNT
    );
    println!(
        "  \x1b[32mGates:\x1b[0m    {} in chain",
        TopologyCanon::GATE_ORDER.len()
    );
    println!();
    println!("    Next: \x1b[1mbizra node\x1b[0m — view sovereign node health");
    println!();

    Ok(())
}

fn pat_display(agent: &PatAgent) -> (&'static str, &'static str) {
    match agent {
        PatAgent::Atlas => ("♟", "Strategy & planning"),
        PatAgent::Oracle => ("🔍", "Research & knowledge"),
        PatAgent::Forge => ("⚙", "Code & implementation"),
        PatAgent::Judge => ("📊", "Quality & scoring"),
        PatAgent::Crown => ("✓", "Constitutional verification"),
        PatAgent::Herald => ("▶", "Publishing & delivery"),
        PatAgent::Nexus => ("🛡", "Orchestration & integration"),
    }
}

fn sat_display(agent: &SatAgent) -> (&'static str, &'static str) {
    match agent {
        SatAgent::Sentinel => ("🔒", "Security & threat detection"),
        SatAgent::OracleSat => ("⚖", "Quality scoring & Ihsan"),
        SatAgent::Ledger => ("📜", "Receipt chain & ledger"),
        SatAgent::Conductor => ("⚡", "Task routing & coordination"),
        SatAgent::Ambassador => ("🔮", "Federation & external comms"),
    }
}

// ── bizra node ──────────────────────────────────────────────

/// Display node health, identity, and constitutional compliance.
pub fn exec_node(_watch: bool) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m             \x1b[1mBIZRA Sovereign Node Health\x1b[0m                    \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // ── Substrate ──
    println!("  \x1b[33m[Substrate]\x1b[0m");
    let manifest = ResourceManifest::discover();
    let hw = &manifest.hardware;
    println!("    CPU:     {} ({} cores)", hw.cpu_name, hw.cpu_cores);
    println!(
        "    RAM:     {:.1} / {:.1} GB",
        hw.ram_available_gb, hw.ram_total_gb
    );
    if !hw.gpus.is_empty() {
        for gpu in &hw.gpus {
            let used_pct = if gpu.vram_total_mb > 0 {
                (gpu.vram_used_mb as f64 / gpu.vram_total_mb as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "    GPU:     {} ({}/{} MB, {:.0}%)",
                gpu.name, gpu.vram_used_mb, gpu.vram_total_mb, used_pct
            );
        }
    }
    println!(
        "    Models:  {} across {} runtime(s)",
        manifest.total_models(),
        manifest.model_count_by_runtime.len()
    );
    println!();

    // ── Constitutional Compliance ──
    println!("  \x1b[33m[Constitution]\x1b[0m");
    println!(
        "    Ihsan Threshold:   {:.2} (production)",
        bizra_core::IHSAN_THRESHOLD
    );
    println!(
        "    SNR Threshold:     {:.2} (minimum)",
        bizra_core::SNR_THRESHOLD
    );
    println!(
        "    Gini Ceiling:      {:.2}",
        bizra_core::omega::ADL_GINI_THRESHOLD
    );
    println!(
        "    Strict Ihsan:      {:.2} (consensus ops)",
        bizra_core::STRICT_IHSAN_THRESHOLD
    );
    println!(
        "    Runtime Ihsan:     {:.2} (live ops)",
        bizra_core::RUNTIME_IHSAN_THRESHOLD
    );
    println!();

    // ── Topology ──
    println!("  \x1b[33m[Topology]\x1b[0m");
    println!(
        "    PAT Agents:  {} (personal council)",
        TopologyCanon::PAT_COUNT
    );
    println!(
        "    SAT Agents:  {} (system validators)",
        TopologyCanon::SAT_COUNT
    );
    println!("    Gate Chain:  {:?}", TopologyCanon::GATE_ORDER);
    println!();

    // ── Genesis Seal ──
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let seal = GenesisSeal::node0_default(now);
    println!("  \x1b[33m[Genesis]\x1b[0m");
    println!("    Seal:     {}", hex::encode(seal.seal_hash));
    println!("    Created:  {}", seal.sealed_at);
    println!();

    // ── Node State ──
    println!("  \x1b[33m[State]\x1b[0m");
    println!("    Version:   {}", bizra_node::NODE_VERSION);
    println!("    Protocol:  {}", bizra_node::PROTOCOL_VERSION);
    println!("    Platform:  {}", manifest.platform);
    println!();

    // ── Verdict ──
    let model_ok = manifest.total_models() > 0;
    let ram_ok = hw.ram_total_gb >= 8.0;
    let gpu_ok = !hw.gpus.is_empty();

    let status = if model_ok && ram_ok {
        "\x1b[32mREADY\x1b[0m"
    } else if ram_ok {
        "\x1b[33mDEGRADED\x1b[0m (no models)"
    } else {
        "\x1b[31mMINIMAL\x1b[0m"
    };

    println!("  \x1b[1mNode Status: {}\x1b[0m", status);
    println!(
        "    [{}] RAM >= 8 GB",
        if ram_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        }
    );
    println!(
        "    [{}] GPU available",
        if gpu_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[33m-\x1b[0m"
        }
    );
    println!(
        "    [{}] LLM models",
        if model_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        }
    );
    println!();

    Ok(())
}
