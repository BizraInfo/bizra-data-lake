//! Genesis Spine — CLI command handlers (Phases 1-3)
//!
//! Authoritative ingress commands backed by the frozen constitutional spine:
//!   init     — substrate discovery + Ed25519 identity generation
//!   genesis  — GenesisSeal computation + constitutional binding
//!   agents   — PAT-7 + SAT-5 topology display + mint status
//!   node     — health, identity, compliance, heartbeat
//!   mission  — governed execution through gate chain → signed receipt
//!   receipt  — receipt display + BLAKE3/Ed25519 verification (disk-persisted)
//!   trust    — constitutional compliance surface
//!   manifest — daily receipt manifest (proof-of-life artifact)
//!
//! Standing on Giants: Bernstein (Ed25519) · Aumasson (BLAKE3) · Al-Ghazali (Ihsan)

use std::fs::{self, OpenOptions};
use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use bizra_core::genesis_seal::{ConstitutionalParams, GenesisSeal};
use bizra_core::topology_canon::{PatAgent, SatAgent, TopologyCanon};
use bizra_node::mission_bridge;
use bizra_node::substrate::ResourceManifest;
use bizra_node::AgentRuntime;

// ── Receipt Ledger (disk-persisted) ────────────────────────

/// A single entry in the receipt ledger (JSONL on disk).
#[derive(Serialize, Deserialize)]
struct LedgerEntry {
    objective: String,
    verifying_key_hex: String,
    receipt: bizra_mission::receipt::MissionReceipt,
}

/// Canonical ledger path: ~/.bizra/receipts.jsonl
fn ledger_path() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".bizra").join("receipts.jsonl")
}

/// Append a receipt entry to the ledger.
fn append_to_ledger(entry: &LedgerEntry) -> Result<()> {
    let path = ledger_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .context("failed to open receipt ledger")?;
    let json = serde_json::to_string(entry).context("failed to serialize receipt")?;
    writeln!(file, "{json}").context("failed to write receipt")?;
    Ok(())
}

/// Load all ledger entries from disk.
fn load_ledger() -> Result<Vec<LedgerEntry>> {
    let path = ledger_path();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(&path).context("failed to open receipt ledger")?;
    let reader = std::io::BufReader::new(file);
    let mut entries = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<LedgerEntry>(trimmed) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

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

// ── bizra mission ───────────────────────────────────────────

/// Execute a governed mission through the constitutional pipeline.
pub fn exec_mission(objective: &str) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m           \x1b[1mBIZRA Governed Mission Execution\x1b[0m                 \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    println!("  \x1b[33mObjective:\x1b[0m {}", objective);
    println!();

    // ── Discover available models ──
    println!("  \x1b[33m[1/4]\x1b[0m Discovering substrate models...");
    let manifest = ResourceManifest::discover();
    let model_names = mission_bridge::extract_model_names(&manifest);
    println!("        {} models available", model_names.len());
    if model_names.is_empty() {
        println!("  \x1b[31m✗ No models available — mission cannot proceed\x1b[0m");
        println!("    Install models: ollama pull qwen2.5:3b");
        println!();
        return Ok(());
    }

    // ── Generate ephemeral signing key ──
    println!("  \x1b[33m[2/4]\x1b[0m Generating Ed25519 signing key...");
    let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    let verifying_key = ed25519_dalek::VerifyingKey::from(&signing_key);

    // ── Set Ihsan and timestamp ──
    let ihsan = bizra_node::IhsanScore::from_f64(0.96);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // ── Execute governed mission ──
    println!("  \x1b[33m[3/4]\x1b[0m Executing through constitutional pipeline...");
    println!("        Gate chain: {:?}", TopologyCanon::GATE_ORDER);

    let mut runtime = AgentRuntime::new();
    let result = mission_bridge::execute_governed_mission(
        &mut runtime,
        &ihsan,
        objective,
        now,
        &model_names,
        None, // genesis receipt (no chain predecessor)
        Some(&signing_key),
    );

    // ── Display receipt ──
    println!("  \x1b[33m[4/4]\x1b[0m Receipt emitted.");
    println!();

    let receipt = &result.receipt;
    let state_label = format!("{:?}", receipt.final_state);
    let state_color = if receipt.is_success() {
        "\x1b[32m"
    } else if receipt.is_degraded() {
        "\x1b[33m"
    } else {
        "\x1b[31m"
    };

    println!("  \x1b[32m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[32m║\x1b[0m  MISSION RECEIPT                                           \x1b[32m║\x1b[0m");
    println!("  \x1b[32m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!("    Receipt ID:  {}", receipt.id_hex());
    println!("    State:       {}{}\x1b[0m", state_color, state_label);
    println!(
        "    Ihsan:       {}",
        receipt
            .ihsan_score
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "—".into())
    );
    println!(
        "    Guardian:    {}",
        match receipt.guardian_approved {
            Some(true) => "\x1b[32mAPPROVED\x1b[0m",
            Some(false) => "\x1b[31mVETOED\x1b[0m",
            None => "—",
        }
    );
    println!(
        "    Model:       {}",
        receipt.chosen_model.as_deref().unwrap_or("none")
    );
    println!(
        "    Signed:      {}",
        if receipt.is_signed() {
            "\x1b[32mYes (Ed25519)\x1b[0m"
        } else {
            "\x1b[31mNo\x1b[0m"
        }
    );
    println!(
        "    Hash Valid:  {}",
        if receipt.verify_hash() {
            "\x1b[32mYes (BLAKE3)\x1b[0m"
        } else {
            "\x1b[31mNo\x1b[0m"
        }
    );
    println!(
        "    Sig Valid:   {}",
        if receipt.verify_signature(&verifying_key) {
            "\x1b[32mYes\x1b[0m"
        } else {
            "\x1b[31mNo\x1b[0m"
        }
    );
    println!("    Tier:        {}", receipt.degradation_tier);
    println!(
        "    States:      {}",
        receipt
            .states_traversed
            .iter()
            .map(|s| format!("{:?}", s))
            .collect::<Vec<_>>()
            .join(" → ")
    );

    if let Some(ref resp) = result.runtime_response {
        println!();
        println!("  \x1b[33mResponse:\x1b[0m");
        let content = resp.response.content.as_str();
        for line in content.lines().take(20) {
            println!("    {}", line);
        }
    }

    // Persist to disk ledger (cross-process reliable)
    let entry = LedgerEntry {
        objective: objective.to_string(),
        verifying_key_hex: hex::encode(verifying_key.to_bytes()),
        receipt: result.receipt.clone(),
    };
    match append_to_ledger(&entry) {
        Ok(()) => println!(
            "    \x1b[32mLedger:\x1b[0m  Persisted to {}",
            ledger_path().display()
        ),
        Err(e) => println!("    \x1b[31mLedger:\x1b[0m  Failed to persist: {e}"),
    }

    println!();
    println!("    Next: \x1b[1mbizra receipt --verify\x1b[0m — verify receipt integrity");
    println!();

    Ok(())
}

// ── bizra receipt ───────────────────────────────────────────

/// Display and optionally verify the last mission receipt (from disk ledger).
pub fn exec_receipt(verify: bool) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m             \x1b[1mBIZRA Receipt Verification\x1b[0m                     \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    let entries = load_ledger().context("failed to load receipt ledger")?;

    if entries.is_empty() {
        println!("  No receipts on disk. Run \x1b[1mbizra mission \"<objective>\"\x1b[0m first.");
        println!("  Ledger: {}", ledger_path().display());
        println!();
        return Ok(());
    }

    let last = entries.last().unwrap();
    let receipt = &last.receipt;

    println!(
        "  \x1b[33mReceipt\x1b[0m ({} of {} in ledger):",
        entries.len(),
        entries.len()
    );
    println!("    ID:          {}", receipt.id_hex());
    println!("    Objective:   {}", last.objective);
    println!("    State:       {:?}", receipt.final_state);
    println!(
        "    Ihsan:       {}",
        receipt
            .ihsan_score
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "—".into())
    );
    println!(
        "    Model:       {}",
        receipt.chosen_model.as_deref().unwrap_or("none")
    );
    println!("    Tier:        {}", receipt.degradation_tier);
    println!("    Signed:      {}", receipt.is_signed());
    println!(
        "    Chain Link:  {}",
        receipt
            .previous_receipt_hash
            .map(hex::encode)
            .unwrap_or_else(|| "genesis (no predecessor)".into())
    );
    println!("    Ledger:      {}", ledger_path().display());
    println!();

    if verify {
        println!("  \x1b[33mVerification:\x1b[0m");

        // Reconstruct verifying key from hex
        let vk_bytes = hex::decode(&last.verifying_key_hex)
            .ok()
            .and_then(|b| <[u8; 32]>::try_from(b).ok());

        let hash_ok = receipt.verify_hash();
        println!(
            "    [{}] BLAKE3 hash integrity",
            if hash_ok {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            }
        );

        if let Some(vk_bytes) = vk_bytes {
            if let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(&vk_bytes) {
                let sig_ok = receipt.verify_signature(&vk);
                println!(
                    "    [{}] Ed25519 signature",
                    if sig_ok {
                        "\x1b[32m✓\x1b[0m"
                    } else {
                        "\x1b[31m✗\x1b[0m"
                    }
                );

                // Chain verification: check previous receipt link
                let prev = if entries.len() >= 2 {
                    Some(&entries[entries.len() - 2].receipt)
                } else {
                    None
                };
                let full_ok = receipt.verify_full(&vk, prev);
                println!(
                    "    [{}] Full integrity (hash + sig + chain)",
                    if full_ok {
                        "\x1b[32m✓\x1b[0m"
                    } else {
                        "\x1b[31m✗\x1b[0m"
                    }
                );
                println!();

                if hash_ok && sig_ok && full_ok {
                    println!("  \x1b[32m✓ Receipt is cryptographically valid\x1b[0m");
                } else {
                    println!("  \x1b[31m✗ Receipt verification FAILED\x1b[0m");
                }
            } else {
                println!("    [\x1b[31m✗\x1b[0m] Ed25519 key reconstruction failed");
            }
        } else {
            println!("    [\x1b[33m-\x1b[0m] No verifying key in ledger entry");
        }

        // Chain walk: verify all entries
        println!();
        println!(
            "  \x1b[33mChain Integrity ({} receipts):\x1b[0m",
            entries.len()
        );
        let mut chain_ok = true;
        for (i, entry) in entries.iter().enumerate() {
            let h = entry.receipt.verify_hash();
            if !h {
                chain_ok = false;
            }
            let mark = if h {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            };
            let id_short = &entry.receipt.id_hex()[..16];
            println!(
                "    [{}] #{}: {}… {:?}",
                mark,
                i + 1,
                id_short,
                entry.receipt.final_state
            );
        }
        println!();
        if chain_ok {
            println!(
                "  \x1b[32m✓ All {} receipts have valid BLAKE3 hashes\x1b[0m",
                entries.len()
            );
        } else {
            println!("  \x1b[31m✗ Chain contains tampered receipts\x1b[0m");
        }
        println!();
    }

    Ok(())
}

// ── bizra trust ────────────────────────────────────────────

/// Constitutional compliance surface — the operator's trust panel.
pub fn exec_trust() -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m           \x1b[1mBIZRA Constitutional Trust Surface\x1b[0m               \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // ── Constitutional Thresholds ──
    println!("  \x1b[33m[Constitutional Law]\x1b[0m");
    let checks = [
        ("Ihsan (production)", bizra_core::IHSAN_THRESHOLD, 0.95),
        ("SNR (minimum)", bizra_core::SNR_THRESHOLD, 0.85),
        (
            "Gini (ceiling)",
            bizra_core::omega::ADL_GINI_THRESHOLD,
            0.35,
        ),
        ("Strict Ihsan", bizra_core::STRICT_IHSAN_THRESHOLD, 0.99),
        ("Runtime Ihsan", bizra_core::RUNTIME_IHSAN_THRESHOLD, 1.0),
    ];

    let mut law_ok = true;
    for (name, actual, expected) in &checks {
        let ok = (actual - expected).abs() < f64::EPSILON;
        if !ok {
            law_ok = false;
        }
        println!(
            "    [{}] {:<20} {:.2} (expected {:.2})",
            if ok {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            },
            name,
            actual,
            expected
        );
    }
    println!();

    // ── Topology Frozen ──
    println!("  \x1b[33m[Topology]\x1b[0m");
    let pat_ok = TopologyCanon::PAT_COUNT == 7;
    let sat_ok = TopologyCanon::SAT_COUNT == 5;
    let gate_ok = TopologyCanon::GATE_ORDER.len() == 3;
    println!(
        "    [{}] PAT-{} agents (expected 7)",
        if pat_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        },
        TopologyCanon::PAT_COUNT
    );
    println!(
        "    [{}] SAT-{} agents (expected 5)",
        if sat_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        },
        TopologyCanon::SAT_COUNT
    );
    println!(
        "    [{}] {}-gate chain {:?}",
        if gate_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        },
        TopologyCanon::GATE_ORDER.len(),
        TopologyCanon::GATE_ORDER
    );
    println!();

    // ── Genesis Seal ──
    println!("  \x1b[33m[Genesis Seal]\x1b[0m");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let params = ConstitutionalParams::default();
    let seal = GenesisSeal::compute(params, now);
    let seal_ok = seal.seal_hash != [0u8; 32];
    println!(
        "    [{}] Seal computable (BLAKE3 deterministic)",
        if seal_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        }
    );
    println!("    Hash: {}…", &hex::encode(seal.seal_hash)[..32]);
    println!();

    // ── Receipt Chain ──
    println!("  \x1b[33m[Receipt Ledger]\x1b[0m");
    let path = ledger_path();
    if path.exists() {
        let entries = load_ledger().unwrap_or_default();
        let total = entries.len();
        let valid = entries.iter().filter(|e| e.receipt.verify_hash()).count();
        let signed = entries.iter().filter(|e| e.receipt.is_signed()).count();
        let complete = entries.iter().filter(|e| e.receipt.is_success()).count();

        let chain_ok = valid == total;
        println!(
            "    [{}] {} receipts on disk ({} valid hashes)",
            if chain_ok {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            },
            total,
            valid
        );
        println!(
            "    [{}] {} signed (Ed25519)",
            if signed == total {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[33m-\x1b[0m"
            },
            signed
        );
        println!(
            "    Complete: {}  Degraded/Failed: {}",
            complete,
            total - complete
        );
    } else {
        println!("    [\x1b[33m-\x1b[0m] No receipt ledger yet");
        println!("    Run \x1b[1mbizra mission\x1b[0m to emit the first receipt");
    }
    println!();

    // ── Substrate ──
    println!("  \x1b[33m[Substrate]\x1b[0m");
    let manifest = ResourceManifest::discover();
    let model_ok = manifest.total_models() > 0;
    let ram_ok = manifest.hardware.ram_total_gb >= 8.0;
    println!(
        "    [{}] {} LLM models available",
        if model_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        },
        manifest.total_models()
    );
    println!(
        "    [{}] {:.1} GB RAM",
        if ram_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        },
        manifest.hardware.ram_total_gb
    );
    println!();

    // ── Verdict ──
    let all_ok = law_ok && pat_ok && sat_ok && gate_ok && seal_ok && model_ok && ram_ok;
    if all_ok {
        println!("  \x1b[32m╔════════════════════════════════════════════════════════════╗\x1b[0m");
        println!("  \x1b[32m║  TRUST VERDICT: SOVEREIGN — all constitutional checks PASS ║\x1b[0m");
        println!("  \x1b[32m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    } else {
        println!("  \x1b[33m╔════════════════════════════════════════════════════════════╗\x1b[0m");
        println!("  \x1b[33m║  TRUST VERDICT: DEGRADED — review failing checks above     ║\x1b[0m");
        println!("  \x1b[33m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    }
    println!();

    Ok(())
}

// ��─ bizra manifest ─────────────────────────────────────────

/// Daily receipt manifest — proof-of-life artifact.
pub fn exec_manifest() -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m              \x1b[1mBIZRA Daily Manifest\x1b[0m                          \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    let entries = load_ledger().context("failed to load receipt ledger")?;

    if entries.is_empty() {
        println!("  No receipts. Run \x1b[1mbizra mission\x1b[0m to generate proof.");
        println!();
        return Ok(());
    }

    // Filter to today's receipts
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let day_start = now - (now % 86400); // UTC midnight

    let today: Vec<&LedgerEntry> = entries
        .iter()
        .filter(|e| e.receipt.completed_at >= day_start)
        .collect();

    let total = today.len();
    let complete = today.iter().filter(|e| e.receipt.is_success()).count();
    let degraded = today.iter().filter(|e| e.receipt.is_degraded()).count();
    let failed = total - complete - degraded;
    let signed = today.iter().filter(|e| e.receipt.is_signed()).count();
    let hash_ok = today.iter().filter(|e| e.receipt.verify_hash()).count();

    // Unique models used
    let mut models: Vec<String> = today
        .iter()
        .filter_map(|e| e.receipt.chosen_model.clone())
        .collect();
    models.sort();
    models.dedup();

    // Total states traversed
    let total_states: usize = today.iter().map(|e| e.receipt.states_traversed.len()).sum();

    // Manifest hash: BLAKE3 of all receipt IDs concatenated
    let mut manifest_hasher = blake3::Hasher::new();
    for entry in &today {
        manifest_hasher.update(&entry.receipt.receipt_id);
    }
    let manifest_hash = manifest_hasher.finalize();

    println!(
        "  \x1b[33mDate:\x1b[0m {} UTC",
        chrono::Utc::now().format("%Y-%m-%d")
    );
    println!();

    println!("  \x1b[33m[Summary]\x1b[0m");
    println!("    Missions:     {}", total);
    println!(
        "    Complete:     \x1b[32m{}\x1b[0m  Degraded: \x1b[33m{}\x1b[0m  Failed: \x1b[31m{}\x1b[0m",
        complete, degraded, failed
    );
    println!("    Signed:       {}/{}", signed, total);
    println!("    Hash Valid:   {}/{}", hash_ok, total);
    println!("    States:       {} total transitions", total_states);
    println!(
        "    Models:       {}",
        if models.is_empty() {
            "none".to_string()
        } else {
            models.join(", ")
        }
    );
    println!();

    // Receipt list
    println!("  \x1b[33m[Receipts]\x1b[0m");
    for (i, entry) in today.iter().enumerate() {
        let r = &entry.receipt;
        let state_color = if r.is_success() {
            "\x1b[32m"
        } else if r.is_degraded() {
            "\x1b[33m"
        } else {
            "\x1b[31m"
        };
        println!(
            "    #{}: {}…  {}{:?}\x1b[0m  {}",
            i + 1,
            &r.id_hex()[..16],
            state_color,
            r.final_state,
            entry.objective
        );
    }
    println!();

    // All-time stats
    let all_total = entries.len();
    let all_complete = entries.iter().filter(|e| e.receipt.is_success()).count();
    println!("  \x1b[33m[All-Time]\x1b[0m");
    println!(
        "    Total receipts: {}  ({} complete, {:.0}% success rate)",
        all_total,
        all_complete,
        if all_total > 0 {
            (all_complete as f64 / all_total as f64) * 100.0
        } else {
            0.0
        }
    );
    println!();

    // Manifest seal
    println!("  \x1b[32m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[32m║\x1b[0m  MANIFEST SEAL                                             \x1b[32m║\x1b[0m");
    println!(
        "  \x1b[32m║\x1b[0m  Hash:    \x1b[1m{}\x1b[0m  \x1b[32m║\x1b[0m",
        hex::encode(manifest_hash.as_bytes())
    );
    println!("  \x1b[32m║\x1b[0m  Count:   {:<55}\x1b[32m║\x1b[0m", total);
    println!("  \x1b[32m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    Ok(())
}

// ── bizra replay ────────────────────────────────────────────

/// Replay a mission from its receipt ID — re-execute and chain.
pub fn exec_replay(id_prefix: &str) -> Result<()> {
    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m             \x1b[1mBIZRA Mission Replay\x1b[0m                            \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    if id_prefix.len() < 8 {
        println!("  \x1b[31mReceipt ID prefix must be at least 8 hex characters.\x1b[0m");
        println!();
        return Ok(());
    }

    // ── Find receipt by prefix ──
    let entries = load_ledger().context("failed to load receipt ledger")?;
    let matches: Vec<&LedgerEntry> = entries
        .iter()
        .filter(|e| e.receipt.id_hex().starts_with(id_prefix))
        .collect();

    if matches.is_empty() {
        println!(
            "  \x1b[31mNo receipt matching prefix '{}' in ledger.\x1b[0m",
            id_prefix
        );
        println!(
            "  Ledger: {} ({} entries)",
            ledger_path().display(),
            entries.len()
        );
        println!();
        return Ok(());
    }
    if matches.len() > 1 {
        println!(
            "  \x1b[33mAmbiguous prefix '{}' matches {} receipts. Be more specific.\x1b[0m",
            id_prefix,
            matches.len()
        );
        for m in &matches {
            println!("    {}…  {}", &m.receipt.id_hex()[..16], m.objective);
        }
        println!();
        return Ok(());
    }

    let original = matches[0];
    let original_id = original.receipt.id_hex();

    println!("  \x1b[33mOriginal Receipt:\x1b[0m {}…", &original_id[..16]);
    println!("  \x1b[33mObjective:\x1b[0m       {}", original.objective);
    println!(
        "  \x1b[33mOriginal State:\x1b[0m  {:?}",
        original.receipt.final_state
    );
    println!();

    // ── Re-execute through governed pipeline ──
    println!("  \x1b[33m[1/3]\x1b[0m Discovering substrate models...");
    let manifest = ResourceManifest::discover();
    let model_names = mission_bridge::extract_model_names(&manifest);
    if model_names.is_empty() {
        println!("  \x1b[31m✗ No models available — replay cannot proceed\x1b[0m");
        println!();
        return Ok(());
    }

    println!("  \x1b[33m[2/3]\x1b[0m Re-executing with chain link to original...");
    let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    let verifying_key = ed25519_dalek::VerifyingKey::from(&signing_key);
    let ihsan = bizra_node::IhsanScore::from_f64(0.96);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut runtime = AgentRuntime::new();
    let result = mission_bridge::execute_governed_mission(
        &mut runtime,
        &ihsan,
        &original.objective,
        now,
        &model_names,
        Some(original.receipt.receipt_id), // chain to original
        Some(&signing_key),
    );

    // ── Display replay receipt ──
    println!("  \x1b[33m[3/3]\x1b[0m Replay receipt emitted.");
    println!();

    let receipt = &result.receipt;
    let state_color = if receipt.is_success() {
        "\x1b[32m"
    } else if receipt.is_degraded() {
        "\x1b[33m"
    } else {
        "\x1b[31m"
    };

    println!("  \x1b[32m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[32m║\x1b[0m  REPLAY RECEIPT                                            \x1b[32m║\x1b[0m");
    println!("  \x1b[32m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!("    New ID:       {}", receipt.id_hex());
    println!("    Original:     {}…", &original_id[..16]);
    println!(
        "    State:        {}{:?}\x1b[0m",
        state_color, receipt.final_state
    );
    println!(
        "    Ihsan:        {}",
        receipt
            .ihsan_score
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "—".into())
    );
    println!(
        "    Model:        {}",
        receipt.chosen_model.as_deref().unwrap_or("none")
    );
    println!(
        "    Chained To:   {}",
        receipt
            .previous_receipt_hash
            .map(hex::encode)
            .unwrap_or_else(|| "none".into())
    );
    println!(
        "    Signed:       {}",
        if receipt.is_signed() {
            "\x1b[32mYes (Ed25519)\x1b[0m"
        } else {
            "\x1b[31mNo\x1b[0m"
        }
    );
    println!(
        "    Hash Valid:   {}",
        if receipt.verify_hash() {
            "\x1b[32mYes (BLAKE3)\x1b[0m"
        } else {
            "\x1b[31mNo\x1b[0m"
        }
    );

    // ── Persist replay receipt ──
    let entry = LedgerEntry {
        objective: format!("[REPLAY] {}", original.objective),
        verifying_key_hex: hex::encode(verifying_key.to_bytes()),
        receipt: result.receipt.clone(),
    };
    match append_to_ledger(&entry) {
        Ok(()) => println!(
            "    \x1b[32mLedger:\x1b[0m   Persisted (chain length: {})",
            entries.len() + 1
        ),
        Err(e) => println!("    \x1b[31mLedger:\x1b[0m   Failed: {e}"),
    }
    println!();

    // ── Determinism check ──
    let same_state = receipt.final_state == original.receipt.final_state;
    println!(
        "  \x1b[33mDeterminism:\x1b[0m {}",
        if same_state {
            "\x1b[32mSame final state as original — replay consistent\x1b[0m"
        } else {
            "\x1b[33mDifferent final state — non-deterministic (expected for LLM inference)\x1b[0m"
        }
    );
    println!();

    Ok(())
}

// ── bizra brief ─────────────────────────────────────────────

/// Sovereign morning briefing — proactive system intelligence.
/// The Ghost layer's first visible surface: aggregates health,
/// receipts, trust, models, and recommendations into one view.
pub fn exec_brief() -> Result<()> {
    use chrono::Timelike;

    let now = chrono::Local::now();
    let greeting = match now.hour() {
        5..=11 => "Good morning",
        12..=16 => "Good afternoon",
        17..=21 => "Good evening",
        _ => "Sovereign briefing",
    };

    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!(
        "  \x1b[36m║\x1b[0m  \x1b[1m{}, MoMo.\x1b[0m{}\x1b[36m║\x1b[0m",
        greeting,
        " ".repeat(43 - greeting.len())
    );
    println!("  \x1b[36m║\x1b[0m  BIZRA Sovereign Node — Daily Brief                        \x1b[36m║\x1b[0m");
    println!(
        "  \x1b[36m║\x1b[0m  {:<56}\x1b[36m║\x1b[0m",
        now.format("%A, %B %e, %Y • %H:%M")
    );
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // ── Substrate ──
    let manifest = ResourceManifest::discover();
    let hw = &manifest.hardware;
    let ram_pct = if hw.ram_total_gb > 0.0 {
        ((hw.ram_total_gb - hw.ram_available_gb) / hw.ram_total_gb) * 100.0
    } else {
        0.0
    };

    println!("  \x1b[33m[Substrate]\x1b[0m");
    println!(
        "    {} • {} cores • {:.0} GB RAM ({:.0}% used)",
        hw.cpu_name, hw.cpu_cores, hw.ram_total_gb, ram_pct
    );
    if let Some(gpu) = hw.gpus.first() {
        let gpu_pct = if gpu.vram_total_mb > 0 {
            (gpu.vram_used_mb as f64 / gpu.vram_total_mb as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "    {} • {}/{} MB VRAM ({:.0}%)",
            gpu.name, gpu.vram_used_mb, gpu.vram_total_mb, gpu_pct
        );
    }
    println!(
        "    {} models across {} runtime(s)",
        manifest.total_models(),
        manifest.model_count_by_runtime.len()
    );
    println!();

    // ── Runtime Health ──
    let runtime = AgentRuntime::new();
    let health = runtime.health();
    println!("  \x1b[33m[Runtime]\x1b[0m");
    println!(
        "    State: \x1b[{}m{:?}\x1b[0m  •  Reflex: {}  •  Rules: {}",
        match health.state {
            bizra_node::RuntimeState::Ready => "32",
            bizra_node::RuntimeState::Degraded => "33",
            _ => "31",
        },
        health.state,
        health.reflex_mode.as_str(),
        health.reflex_rules
    );
    println!(
        "    Agents: {}/{}  •  Vetoes: {}  •  Knows-Me: {:.2}",
        health.agents_active, health.agents_registered, health.total_vetoes, health.knows_me_score
    );
    println!();

    // ── Receipt Chain ──
    println!("  \x1b[33m[Receipts]\x1b[0m");
    let entries = load_ledger().unwrap_or_default();
    if entries.is_empty() {
        println!("    No receipts yet. Your first mission awaits.");
    } else {
        let total = entries.len();
        let complete = entries.iter().filter(|e| e.receipt.is_success()).count();
        let degraded = entries.iter().filter(|e| e.receipt.is_degraded()).count();
        let failed = total - complete - degraded;
        let all_valid = entries.iter().all(|e| e.receipt.verify_hash());

        println!(
            "    {} total  •  \x1b[32m{} complete\x1b[0m  •  \x1b[33m{} degraded\x1b[0m  •  \x1b[31m{} failed\x1b[0m",
            total, complete, degraded, failed
        );
        println!(
            "    Chain: {}",
            if all_valid {
                "\x1b[32m✓ All hashes valid\x1b[0m"
            } else {
                "\x1b[31m✗ Chain integrity broken\x1b[0m"
            }
        );

        // Last mission
        let last = entries.last().unwrap();
        let state_color = if last.receipt.is_success() {
            "\x1b[32m"
        } else if last.receipt.is_degraded() {
            "\x1b[33m"
        } else {
            "\x1b[31m"
        };
        println!(
            "    Last:  {}…  {}{:?}\x1b[0m  {}",
            &last.receipt.id_hex()[..12],
            state_color,
            last.receipt.final_state,
            last.objective
        );
    }
    println!();

    // ── Constitutional Trust ──
    println!("  \x1b[33m[Constitution]\x1b[0m");
    let ihsan_ok = (bizra_core::IHSAN_THRESHOLD - 0.95).abs() < f64::EPSILON;
    let snr_ok = (bizra_core::SNR_THRESHOLD - 0.85).abs() < f64::EPSILON;
    let gini_ok = (bizra_core::omega::ADL_GINI_THRESHOLD - 0.35).abs() < f64::EPSILON;
    let topo_ok = TopologyCanon::PAT_COUNT == 7 && TopologyCanon::SAT_COUNT == 5;
    let gates_ok = TopologyCanon::GATE_ORDER.len() == 3;
    let all_trust = ihsan_ok && snr_ok && gini_ok && topo_ok && gates_ok;

    println!(
        "    Ihsan: {:.2}  SNR: {:.2}  Gini: {:.2}  PAT/SAT: {}/{}  Gates: {}",
        bizra_core::IHSAN_THRESHOLD,
        bizra_core::SNR_THRESHOLD,
        bizra_core::omega::ADL_GINI_THRESHOLD,
        TopologyCanon::PAT_COUNT,
        TopologyCanon::SAT_COUNT,
        TopologyCanon::GATE_ORDER.len()
    );
    println!(
        "    Trust: {}",
        if all_trust {
            "\x1b[32mSOVEREIGN\x1b[0m"
        } else {
            "\x1b[33mDEGRADED — run bizra trust\x1b[0m"
        }
    );
    println!();

    // ── Models ──
    println!("  \x1b[33m[Models]\x1b[0m");
    let model_names = mission_bridge::extract_model_names(&manifest);
    let is_vision = |m: &str| m.contains("moondream") || m.contains("VL") || m.contains("vision");
    let text_models: Vec<&String> = model_names.iter().filter(|m| !is_vision(m)).collect();
    let vision_models: Vec<&String> = model_names.iter().filter(|m| is_vision(m)).collect();

    if !text_models.is_empty() {
        println!(
            "    Text:   {} ({})",
            text_models.len(),
            text_models
                .iter()
                .take(3)
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if !vision_models.is_empty() {
        println!(
            "    Vision: {} ({})",
            vision_models.len(),
            vision_models
                .iter()
                .take(2)
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if model_names.is_empty() {
        println!("    \x1b[31mNo models available\x1b[0m");
    }
    println!();

    // ── Recommendations ──
    println!("  \x1b[33m[Recommendations]\x1b[0m");
    let mut recs: Vec<&str> = Vec::new();

    if entries.is_empty() {
        recs.push("Run your first mission: bizra mission \"<objective>\"");
    }
    if manifest.total_models() == 0 {
        recs.push("Install a model: ollama pull qwen2.5:3b");
    }
    if text_models.is_empty() && !model_names.is_empty() {
        recs.push("Install a text model — only vision models detected");
    }
    if !entries.is_empty() && entries.iter().any(|e| !e.receipt.verify_hash()) {
        recs.push("Receipt chain has invalid hashes — run: bizra receipt --verify");
    }
    if recs.is_empty() {
        recs.push("System healthy. Ready for sovereign missions.");
    }

    for rec in &recs {
        println!("    → {rec}");
    }
    println!();

    println!("  \x1b[2m\"Every human is a node. Every node is a seed.\" — بذرة\x1b[0m");
    println!();

    Ok(())
}
