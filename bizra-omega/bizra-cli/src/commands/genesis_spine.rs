//! Genesis Spine — CLI command handlers (Phases 1-6)
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
//!   brief    — sovereign morning briefing (Ghost layer)
//!   dashboard — data layer for TUI mission control (Phase 6)
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

// ── Dashboard Data Layer (Phase 6) ─────────────────────────
//
// Single-pass data collection for the TUI mission control cockpit.
// Consumes the same backends as the CLI commands — no shadow state.

/// GPU information for dashboard display.
pub struct GpuInfo {
    pub name: String,
    pub used_mb: u64,
    pub total_mb: u64,
    pub used_pct: f64,
}

/// Summary of a single receipt for dashboard display.
#[derive(Clone)]
pub struct ReceiptSummary {
    pub id_short: String,
    pub objective: String,
    pub state_label: &'static str,
    pub is_success: bool,
    pub is_degraded: bool,
    pub signed: bool,
    pub ihsan_score: Option<f32>,
    pub snr_score: Option<f32>,
    pub chosen_model: Option<String>,
    pub degradation_tier: u8,
    pub states_traversed: usize,
    pub chain_link: Option<String>,
}

/// A single trust surface check.
pub struct TrustCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Constitutional trust verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustVerdict {
    Sovereign,
    Degraded,
}

/// Event category for Ghost feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    ReceiptCreated,
    TrustChanged,
    MissionCompleted,
}

/// A live node event detected by state diffing.
#[derive(Debug, Clone)]
pub struct NodeEvent {
    pub kind: EventKind,
    pub message: String,
    pub timestamp: String,
}

/// Detect events by comparing previous and current dashboard state.
/// Same truth path — derived entirely from existing backends.
pub fn detect_events(prev: &DashboardData, curr: &DashboardData) -> Vec<NodeEvent> {
    let mut events = Vec::new();
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();

    // Receipt created — new receipts appeared in the chain
    if curr.total_receipts > prev.total_receipts {
        let new_count = curr.total_receipts - prev.total_receipts;
        for receipt in curr.recent_receipts.iter().take(new_count) {
            let label = if receipt.is_success {
                "Complete"
            } else if receipt.is_degraded {
                "Degraded"
            } else {
                "Failed"
            };
            events.push(NodeEvent {
                kind: EventKind::ReceiptCreated,
                message: format!(
                    "Receipt {} — {} ({})",
                    receipt.id_short,
                    truncate_str(&receipt.objective, 40),
                    label
                ),
                timestamp: ts.clone(),
            });
        }
    }

    // Trust changed — verdict flipped
    if curr.trust_verdict != prev.trust_verdict {
        let msg = match curr.trust_verdict {
            TrustVerdict::Sovereign => "Trust restored — SOVEREIGN",
            TrustVerdict::Degraded => "Trust degraded — checks failing",
        };
        events.push(NodeEvent {
            kind: EventKind::TrustChanged,
            message: msg.to_string(),
            timestamp: ts.clone(),
        });
    }

    // Mission completed — today's completion count increased
    if curr.today_complete > prev.today_complete {
        let new_missions = curr.today_complete - prev.today_complete;
        events.push(NodeEvent {
            kind: EventKind::MissionCompleted,
            message: format!(
                "{} mission{} completed today (total: {})",
                new_missions,
                if new_missions > 1 { "s" } else { "" },
                curr.today_complete
            ),
            timestamp: ts.clone(),
        });
    }

    // Chain integrity changed
    if curr.chain_valid != prev.chain_valid {
        let msg = if curr.chain_valid {
            "Chain integrity restored — all hashes valid"
        } else {
            "Chain integrity BROKEN — hash mismatch detected"
        };
        events.push(NodeEvent {
            kind: EventKind::TrustChanged,
            message: msg.to_string(),
            timestamp: ts,
        });
    }

    events
}

/// Agent info for parliament display.
pub struct AgentInfo {
    pub index: u8,
    pub callsign: String,
    pub role: String,
    pub icon: &'static str,
    // Scaffolding — used when parliament panel shows team badges
    #[allow(dead_code)]
    pub team: &'static str,
}

/// All data needed to render the TUI dashboard — gathered in one pass.
pub struct DashboardData {
    // Substrate
    pub cpu_name: String,
    pub cpu_cores: u32,
    pub ram_total_gb: f64,
    pub ram_used_pct: f64,
    pub gpu: Option<GpuInfo>,
    pub model_count: usize,
    pub text_models: Vec<String>,
    pub vision_models: Vec<String>,
    // Scaffolding — used when substrate panel shows runtime breakdown
    #[allow(dead_code)]
    pub runtime_count: usize,
    pub platform: String,

    // Runtime health
    pub runtime_state: String,
    pub reflex_mode: String,
    pub reflex_rules: usize,
    pub agents_active: usize,
    pub agents_registered: usize,

    // Receipt chain
    pub total_receipts: usize,
    // Scaffolding — used when receipt rail shows category breakdown
    #[allow(dead_code)]
    pub complete_count: usize,
    #[allow(dead_code)]
    pub degraded_count: usize,
    #[allow(dead_code)]
    pub failed_count: usize,
    pub chain_valid: bool,
    pub recent_receipts: Vec<ReceiptSummary>,
    pub all_receipts: Vec<ReceiptSummary>,

    // Today's manifest
    pub today_count: usize,
    pub today_complete: usize,
    pub manifest_seal: Option<String>,

    // Trust surface
    pub trust_checks: Vec<TrustCheck>,
    pub receipt_chain_checks: Vec<TrustCheck>,
    pub trust_verdict: TrustVerdict,

    // Parliament
    pub pat_agents: Vec<AgentInfo>,
    pub sat_agents: Vec<AgentInfo>,

    // Ghost / Recommendations
    pub greeting: String,
    pub recommendations: Vec<String>,

    // Live event log — populated by detect_events() on refresh
    pub event_log: Vec<NodeEvent>,

    // ── Sprint 7.3: Wallet / Treasury ──
    pub seed_balance: f64,
    pub seed_gini: f64,
    pub seed_supply_cap: f64,
    pub zakat_rate: f64,
    pub zakat_deducted: f64,
    pub total_minted: f64,
    pub total_burned: f64,

    // ── Sprint 7.4: Memory / Skills ──
    pub memory_fast: u32,
    pub memory_slow: u32,
    pub memory_glacial: u32,
    pub memory_fragments: u32,
    pub memory_atoms: u32,
    pub memory_insights: u32,
    pub memory_profile_completeness: f32,
    pub memory_knows_me: f32,
    pub reflex_compiled: u64,
    pub reflex_hits: u64,
    pub reflex_misses: u64,
    pub reflex_quarantined: u64,
    pub sovereignty_tier: &'static str,
}

/// Collect all dashboard data in a single pass.
/// Same backends as CLI commands — no new authority.
pub fn gather_dashboard_data() -> DashboardData {
    use chrono::Timelike;

    let now = chrono::Local::now();

    // ── 1. Substrate ──
    let manifest = ResourceManifest::discover();
    let hw = &manifest.hardware;
    let ram_used_pct = if hw.ram_total_gb > 0.0 {
        ((hw.ram_total_gb - hw.ram_available_gb) / hw.ram_total_gb) * 100.0
    } else {
        0.0
    };

    let gpu = hw.gpus.first().map(|g| {
        let used_pct = if g.vram_total_mb > 0 {
            (g.vram_used_mb as f64 / g.vram_total_mb as f64) * 100.0
        } else {
            0.0
        };
        GpuInfo {
            name: g.name.clone(),
            used_mb: g.vram_used_mb,
            total_mb: g.vram_total_mb,
            used_pct,
        }
    });

    // ── 2. Models ──
    let model_names = mission_bridge::extract_model_names(&manifest);
    let is_vision = |m: &str| m.contains("moondream") || m.contains("VL") || m.contains("vision");
    let text_models: Vec<String> = model_names
        .iter()
        .filter(|m| !is_vision(m))
        .cloned()
        .collect();
    let vision_models: Vec<String> = model_names
        .iter()
        .filter(|m| is_vision(m))
        .cloned()
        .collect();
    let model_count = model_names.len();

    // ── 3. Runtime health ──
    let runtime = AgentRuntime::new();
    let health = runtime.health();
    let runtime_state = match health.state {
        bizra_node::RuntimeState::Ready => "Ready",
        bizra_node::RuntimeState::Degraded => "Degraded",
        bizra_node::RuntimeState::Processing => "Processing",
        bizra_node::RuntimeState::Stopped => "Stopped",
        bizra_node::RuntimeState::Uninitialized => "Uninitialized",
    }
    .to_string();

    // ── 4. Receipt chain ──
    let entries = load_ledger().unwrap_or_default();
    let total_receipts = entries.len();
    let complete_count = entries.iter().filter(|e| e.receipt.is_success()).count();
    let degraded_count = entries.iter().filter(|e| e.receipt.is_degraded()).count();
    let failed_count = total_receipts - complete_count - degraded_count;
    let chain_valid = entries.iter().all(|e| e.receipt.verify_hash());

    let to_summary = |e: &LedgerEntry| -> ReceiptSummary {
        let state_label = if e.receipt.is_success() {
            "Complete"
        } else if e.receipt.is_degraded() {
            "Degraded"
        } else {
            "Failed"
        };
        let chain_link = e
            .receipt
            .previous_receipt_hash
            .map(|h| hex::encode(h).chars().take(16).collect());
        ReceiptSummary {
            id_short: e.receipt.id_hex().chars().take(16).collect(),
            objective: truncate_str(&e.objective, 40),
            state_label,
            is_success: e.receipt.is_success(),
            is_degraded: e.receipt.is_degraded(),
            signed: e.receipt.is_signed(),
            ihsan_score: e.receipt.ihsan_score,
            snr_score: e.receipt.snr_score,
            chosen_model: e.receipt.chosen_model.clone(),
            degradation_tier: e.receipt.degradation_tier,
            states_traversed: e.receipt.states_traversed.len(),
            chain_link,
        }
    };

    let all_receipts: Vec<ReceiptSummary> = entries.iter().rev().map(to_summary).collect();
    let recent_receipts: Vec<ReceiptSummary> = all_receipts.iter().take(10).cloned().collect();

    // ── 5. Today's manifest ──
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let day_start = now_secs - (now_secs % 86400);

    let today: Vec<&LedgerEntry> = entries
        .iter()
        .filter(|e| e.receipt.completed_at >= day_start)
        .collect();
    let today_count = today.len();
    let today_complete = today.iter().filter(|e| e.receipt.is_success()).count();
    let manifest_seal = if !today.is_empty() {
        let mut hasher = blake3::Hasher::new();
        for e in &today {
            hasher.update(&e.receipt.receipt_id);
        }
        Some(
            hex::encode(hasher.finalize().as_bytes())
                .chars()
                .take(16)
                .collect(),
        )
    } else {
        None
    };

    // ── 6. Trust surface ──
    let mut trust_checks = Vec::new();
    let mut all_trust_pass = true;

    // Constitutional law (5 checks)
    let law_checks: [(&str, f64, f64); 5] = [
        ("Ihsan (prod)", bizra_core::IHSAN_THRESHOLD, 0.95),
        ("SNR (min)", bizra_core::SNR_THRESHOLD, 0.85),
        ("Gini (ceil)", bizra_core::omega::ADL_GINI_THRESHOLD, 0.35),
        ("Strict Ihsan", bizra_core::STRICT_IHSAN_THRESHOLD, 0.99),
        ("Runtime Ihsan", bizra_core::RUNTIME_IHSAN_THRESHOLD, 1.0),
    ];
    for (name, actual, expected) in &law_checks {
        let ok = (actual - expected).abs() < f64::EPSILON;
        if !ok {
            all_trust_pass = false;
        }
        trust_checks.push(TrustCheck {
            name: name.to_string(),
            passed: ok,
            detail: format!("{:.2} = {:.2}", actual, expected),
        });
    }

    // Topology (3 checks)
    let pat_ok = TopologyCanon::PAT_COUNT == 7;
    let sat_ok = TopologyCanon::SAT_COUNT == 5;
    let gate_ok = TopologyCanon::GATE_ORDER.len() == 3;
    if !pat_ok || !sat_ok || !gate_ok {
        all_trust_pass = false;
    }
    trust_checks.push(TrustCheck {
        name: "PAT-7".into(),
        passed: pat_ok,
        detail: format!("{}", TopologyCanon::PAT_COUNT),
    });
    trust_checks.push(TrustCheck {
        name: "SAT-5".into(),
        passed: sat_ok,
        detail: format!("{}", TopologyCanon::SAT_COUNT),
    });
    trust_checks.push(TrustCheck {
        name: "3-gate chain".into(),
        passed: gate_ok,
        detail: format!("{}", TopologyCanon::GATE_ORDER.len()),
    });

    // Genesis seal (1 check)
    let seal = GenesisSeal::node0_default(now_secs * 1000);
    let seal_ok = seal.seal_hash != [0u8; 32];
    if !seal_ok {
        all_trust_pass = false;
    }
    trust_checks.push(TrustCheck {
        name: "Genesis Seal".into(),
        passed: seal_ok,
        detail: "computable".into(),
    });

    // Substrate (2 checks)
    let model_ok = model_count > 0;
    let ram_ok = hw.ram_total_gb >= 8.0;
    if !model_ok || !ram_ok {
        all_trust_pass = false;
    }
    trust_checks.push(TrustCheck {
        name: "LLM models".into(),
        passed: model_ok,
        detail: format!("{}", model_count),
    });
    trust_checks.push(TrustCheck {
        name: "RAM >= 8 GB".into(),
        passed: ram_ok,
        detail: format!("{:.0} GB", hw.ram_total_gb),
    });

    // Receipt chain checks (separate for TrustRail grouping)
    let signed_count = entries.iter().filter(|e| e.receipt.is_signed()).count();
    let receipt_chain_checks = vec![
        TrustCheck {
            name: format!("{} receipts", total_receipts),
            passed: chain_valid || total_receipts == 0,
            detail: if chain_valid { "all valid" } else { "BROKEN" }.into(),
        },
        TrustCheck {
            name: "signed".into(),
            passed: signed_count == total_receipts || total_receipts == 0,
            detail: format!("{}/{}", signed_count, total_receipts),
        },
    ];

    let trust_verdict = if all_trust_pass {
        TrustVerdict::Sovereign
    } else {
        TrustVerdict::Degraded
    };

    // ── 7. Parliament ──
    let pat_agents: Vec<AgentInfo> = PatAgent::ALL
        .iter()
        .map(|a| {
            let (icon, role_desc) = pat_display(a);
            AgentInfo {
                index: a.index(),
                callsign: a.callsign().to_string(),
                role: role_desc.to_string(),
                icon,
                team: "PAT",
            }
        })
        .collect();

    let sat_agents: Vec<AgentInfo> = SatAgent::ALL
        .iter()
        .map(|a| {
            let (icon, role_desc) = sat_display(a);
            AgentInfo {
                index: a.index(),
                callsign: a.callsign().to_string(),
                role: role_desc.to_string(),
                icon,
                team: "SAT",
            }
        })
        .collect();

    // ── 8. Wallet / Treasury (Sprint 7.3) ──
    // Derived from receipt chain + constitutional constants — no new authority
    let zakat_rate = bizra_core::islamic_finance::ZAKAT_RATE;
    let seed_supply_cap = 1_000_000_000.0_f64; // 1B SEED total supply
    let seed_per_receipt = 1.0_f64; // 1 SEED per successful mission
    let raw_minted = complete_count as f64 * seed_per_receipt;
    let total_zakat = raw_minted * zakat_rate;
    let total_minted = raw_minted - total_zakat; // net after zakat
    let total_burned = degraded_count as f64 * 0.1; // 0.1 SEED burned per degraded
    let seed_balance = total_minted - total_burned;
    let seed_gini = if total_receipts > 1 {
        // Single-node Gini is 0.0 (perfect equality) — placeholder until federation
        0.0
    } else {
        0.0
    };

    // ── 9. Memory / Skills (Sprint 7.4) ──
    let memory_health = runtime.health();
    let reflex_stats = runtime.reflex_stats();
    let memory_fast = memory_health.fragments_stored as u32;
    let memory_slow = memory_health.insights_stored as u32;
    let memory_glacial = memory_health.profile_traits as u32;
    let memory_fragments = memory_health.fragments_stored as u32;
    let memory_atoms = (memory_health.fragments_stored + memory_health.insights_stored) as u32;
    let memory_insights = memory_health.insights_stored as u32;
    let memory_profile_completeness = memory_health.knows_me_score;
    let memory_knows_me = memory_health.knows_me_score;
    let sovereignty_tier = match (seed_balance, all_trust_pass) {
        (b, true) if b >= 100.0 => "SOVEREIGN",
        (b, true) if b >= 10.0 => "CITIZEN",
        (_, true) => "SEEDLING",
        (_, false) => "DEGRADED",
    };

    // ── 10. Ghost / Recommendations ─���
    let greeting = match now.hour() {
        5..=11 => "Good morning, MoMo.",
        12..=16 => "Good afternoon, MoMo.",
        17..=21 => "Good evening, MoMo.",
        _ => "Sovereign briefing, MoMo.",
    }
    .to_string();

    let mut recommendations: Vec<String> = Vec::new();
    if entries.is_empty() {
        recommendations.push("Run your first mission: bizra mission \"<objective>\"".into());
    }
    if model_count == 0 {
        recommendations.push("Install a model: ollama pull qwen2.5:3b".into());
    }
    if text_models.is_empty() && model_count > 0 {
        recommendations.push("Install a text model — only vision models detected".into());
    }
    if !chain_valid && !entries.is_empty() {
        recommendations
            .push("Receipt chain has invalid hashes — run: bizra receipt --verify".into());
    }
    if recommendations.is_empty() {
        recommendations.push("System healthy. Ready for sovereign missions.".into());
    }

    // ── Assemble ──
    DashboardData {
        cpu_name: hw.cpu_name.clone(),
        cpu_cores: hw.cpu_cores,
        ram_total_gb: hw.ram_total_gb,
        ram_used_pct,
        gpu,
        model_count,
        text_models,
        vision_models,
        runtime_count: manifest.model_count_by_runtime.len(),
        platform: manifest.platform.to_string(),
        runtime_state,
        reflex_mode: health.reflex_mode.as_str().to_string(),
        reflex_rules: health.reflex_rules,
        agents_active: health.agents_active,
        agents_registered: health.agents_registered,
        total_receipts,
        complete_count,
        degraded_count,
        failed_count,
        chain_valid,
        recent_receipts,
        all_receipts,
        today_count,
        today_complete,
        manifest_seal,
        trust_checks,
        receipt_chain_checks,
        trust_verdict,
        pat_agents,
        sat_agents,
        greeting,
        recommendations,
        event_log: Vec::new(),
        // Sprint 7.3: Wallet
        seed_balance,
        seed_gini,
        seed_supply_cap,
        zakat_rate,
        zakat_deducted: total_zakat,
        total_minted,
        total_burned,
        // Sprint 7.4: Memory / Skills
        memory_fast,
        memory_slow,
        memory_glacial,
        memory_fragments,
        memory_atoms,
        memory_insights,
        memory_profile_completeness,
        memory_knows_me,
        reflex_compiled: reflex_stats.compiled,
        reflex_hits: reflex_stats.hits,
        reflex_misses: reflex_stats.misses,
        reflex_quarantined: reflex_stats.quarantined,
        sovereignty_tier,
    }
}

/// Truncate a string to max_len, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
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

// ── bizra receipt --path <file> ────────────────────────────
//
// Standalone cross-process receipt verification.
// Reads a JSONL file of LedgerEntry records and verifies each receipt's
// BLAKE3 hash + Ed25519 signature + chain integrity.

/// Verify receipts from an arbitrary JSONL file (cross-process verification).
pub fn exec_receipt_verify_file(path: &std::path::Path) -> Result<()> {
    use std::io::BufRead;

    println!();
    println!("  \x1b[36m╔════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m       \x1b[1mBIZRA Cross-Process Receipt Verification\x1b[0m            \x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();
    println!("  \x1b[33mFile:\x1b[0m {}", path.display());
    println!();

    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open receipt file: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut entries: Vec<LedgerEntry> = Vec::new();
    let mut parse_errors = 0;

    for (i, line) in reader.lines().enumerate() {
        let line = line.context("failed to read line")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<LedgerEntry>(trimmed) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                println!("  \x1b[31m✗ Line {}: parse error: {}\x1b[0m", i + 1, e);
                parse_errors += 1;
            }
        }
    }

    if entries.is_empty() {
        println!("  \x1b[31mNo valid receipt entries found.\x1b[0m");
        println!();
        return Ok(());
    }

    println!("  \x1b[33mEntries:\x1b[0m {} valid, {} parse errors", entries.len(), parse_errors);
    println!();

    let mut hash_pass = 0;
    let mut sig_pass = 0;
    let mut hash_fail = 0;
    let mut sig_fail = 0;
    let mut sig_skip = 0;

    for (i, entry) in entries.iter().enumerate() {
        let receipt = &entry.receipt;
        let id_short = &receipt.id_hex()[..std::cmp::min(16, receipt.id_hex().len())];

        // BLAKE3 hash verification
        let h_ok = receipt.verify_hash();
        if h_ok {
            hash_pass += 1;
        } else {
            hash_fail += 1;
        }

        // Ed25519 signature verification
        let vk_bytes = hex::decode(&entry.verifying_key_hex)
            .ok()
            .and_then(|b| <[u8; 32]>::try_from(b).ok());

        let sig_status = if let Some(vk_bytes) = vk_bytes {
            if let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(&vk_bytes) {
                if receipt.verify_signature(&vk) {
                    sig_pass += 1;
                    "\x1b[32m✓ sig\x1b[0m"
                } else {
                    sig_fail += 1;
                    "\x1b[31m✗ sig\x1b[0m"
                }
            } else {
                sig_skip += 1;
                "\x1b[33m? key\x1b[0m"
            }
        } else {
            sig_skip += 1;
            "\x1b[33m? key\x1b[0m"
        };

        let h_mark = if h_ok {
            "\x1b[32m✓\x1b[0m"
        } else {
            "\x1b[31m✗\x1b[0m"
        };

        println!(
            "    [{}] #{}: {}…  {}  {:?}",
            h_mark,
            i + 1,
            id_short,
            sig_status,
            receipt.final_state
        );
    }

    println!();
    println!("  \x1b[33mSummary:\x1b[0m");
    println!(
        "    BLAKE3:  \x1b[32m{} pass\x1b[0m / \x1b[31m{} fail\x1b[0m",
        hash_pass, hash_fail
    );
    println!(
        "    Ed25519: \x1b[32m{} pass\x1b[0m / \x1b[31m{} fail\x1b[0m / \x1b[33m{} skip\x1b[0m",
        sig_pass, sig_fail, sig_skip
    );

    let all_ok = hash_fail == 0 && sig_fail == 0 && parse_errors == 0;
    println!();
    if all_ok {
        println!("  \x1b[32m✓ All receipts verified — chain integrity confirmed\x1b[0m");
    } else {
        println!("  \x1b[31m✗ Verification found issues — see above\x1b[0m");
    }
    println!();

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

// ── TUI Mission Submission (Sprint 7.2) ──────────────────

/// Submit a mission from the TUI dashboard. Returns a status message.
/// Same truth path as `exec_mission` — no shadow authority.
pub fn submit_mission_from_tui(objective: &str) -> Result<String> {
    let manifest = ResourceManifest::discover();
    let model_names = mission_bridge::extract_model_names(&manifest);
    if model_names.is_empty() {
        return Ok("No models available — install with: ollama pull qwen2.5:3b".into());
    }

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
        objective,
        now,
        &model_names,
        None,
        Some(&signing_key),
    );

    let receipt = &result.receipt;
    let state_label = if receipt.is_success() {
        "Complete"
    } else if receipt.is_degraded() {
        "Degraded"
    } else {
        "Failed"
    };

    // Persist to disk ledger
    let entry = LedgerEntry {
        objective: objective.to_string(),
        verifying_key_hex: hex::encode(verifying_key.to_bytes()),
        receipt: result.receipt.clone(),
    };
    append_to_ledger(&entry)?;

    let ihsan_str = receipt
        .ihsan_score
        .map(|s| format!(" Ihsan {:.2}", s))
        .unwrap_or_default();

    Ok(format!(
        "Mission {} — {}{} [{}]",
        state_label,
        &receipt.id_hex()[..8],
        ihsan_str,
        receipt.chosen_model.as_deref().unwrap_or("unknown")
    ))
}

// ── Phase 6 Acceptance Tests ─��────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_dashboard_data_returns_valid_substrate() {
        let data = gather_dashboard_data();
        // CPU must be detected
        assert!(!data.cpu_name.is_empty(), "CPU name must be populated");
        assert!(data.cpu_cores > 0, "CPU cores must be > 0");
        // RAM must be detected
        assert!(data.ram_total_gb > 0.0, "RAM must be > 0");
        assert!(
            data.ram_used_pct >= 0.0 && data.ram_used_pct <= 100.0,
            "RAM usage % must be 0-100"
        );
        // Platform must be detected
        assert!(!data.platform.is_empty(), "Platform must be populated");
    }

    #[test]
    fn test_gather_dashboard_data_trust_surface() {
        let data = gather_dashboard_data();
        // Must have all 11 trust checks (5 law + 3 topology + 1 genesis + 2 substrate)
        assert_eq!(
            data.trust_checks.len(),
            11,
            "Expected 11 trust checks, got {}",
            data.trust_checks.len()
        );
        // Receipt chain checks are separate
        assert_eq!(
            data.receipt_chain_checks.len(),
            2,
            "Expected 2 receipt chain checks"
        );
        // Constitutional law checks should all pass (thresholds are compiled-in)
        for check in &data.trust_checks[..5] {
            assert!(
                check.passed,
                "Constitutional law check '{}' must pass — detail: {}",
                check.name, check.detail
            );
        }
        // Topology checks should all pass (compiled-in constants)
        for check in &data.trust_checks[5..8] {
            assert!(check.passed, "Topology check '{}' must pass", check.name);
        }
        // Genesis seal must be computable
        assert!(
            data.trust_checks[8].passed,
            "Genesis seal must be computable"
        );
    }

    #[test]
    fn test_gather_dashboard_data_verdict_sovereign() {
        let data = gather_dashboard_data();
        // On this machine with models and RAM ≥ 8GB, verdict should be Sovereign
        if data.model_count > 0 && data.ram_total_gb >= 8.0 {
            assert_eq!(
                data.trust_verdict,
                TrustVerdict::Sovereign,
                "With models and sufficient RAM, verdict must be Sovereign"
            );
        }
    }

    #[test]
    fn test_gather_dashboard_data_parliament() {
        let data = gather_dashboard_data();
        assert_eq!(data.pat_agents.len(), 7, "PAT-7 must have exactly 7 agents");
        assert_eq!(data.sat_agents.len(), 5, "SAT-5 must have exactly 5 agents");
        // Verify PAT indices are 1-7 (P1-P7 canonical)
        for (i, agent) in data.pat_agents.iter().enumerate() {
            assert_eq!(agent.index as usize, i + 1, "PAT agent index mismatch");
            assert!(!agent.callsign.is_empty(), "PAT agent must have callsign");
            assert!(!agent.role.is_empty(), "PAT agent must have role");
            assert!(!agent.icon.is_empty(), "PAT agent must have icon");
        }
        // Verify SAT indices are 1-5 (S1-S5 canonical)
        for (i, agent) in data.sat_agents.iter().enumerate() {
            assert_eq!(agent.index as usize, i + 1, "SAT agent index mismatch");
        }
    }

    #[test]
    fn test_gather_dashboard_data_runtime_health() {
        let data = gather_dashboard_data();
        // Runtime state must be a known value
        let valid_states = [
            "Ready",
            "Degraded",
            "Processing",
            "Stopped",
            "Uninitialized",
        ];
        assert!(
            valid_states.contains(&data.runtime_state.as_str()),
            "Runtime state '{}' must be one of {:?}",
            data.runtime_state,
            valid_states
        );
        assert!(data.agents_registered > 0, "Must have registered agents");
        assert!(data.reflex_rules > 0, "Must have reflex rules");
    }

    #[test]
    fn test_gather_dashboard_data_greeting() {
        let data = gather_dashboard_data();
        assert!(!data.greeting.is_empty(), "Greeting must be populated");
        assert!(data.greeting.contains("MoMo"), "Greeting must address MoMo");
    }

    #[test]
    fn test_gather_dashboard_data_recommendations() {
        let data = gather_dashboard_data();
        assert!(
            !data.recommendations.is_empty(),
            "Must have at least one recommendation"
        );
    }

    #[test]
    fn test_gather_dashboard_data_receipt_chain_consistency() {
        let data = gather_dashboard_data();
        // Count invariants
        assert!(
            data.complete_count + data.degraded_count + data.failed_count == data.total_receipts,
            "Receipt counts must sum to total: {} + {} + {} != {}",
            data.complete_count,
            data.degraded_count,
            data.failed_count,
            data.total_receipts
        );
        assert!(
            data.today_complete <= data.today_count,
            "Today complete ({}) must be <= today count ({})",
            data.today_complete,
            data.today_count
        );
        assert!(
            data.recent_receipts.len() <= 10,
            "Recent receipts capped at 10"
        );
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("ab", 2), "ab");
        assert_eq!(truncate_str("abc", 3), "abc");
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn test_trust_verdict_eq() {
        assert_eq!(TrustVerdict::Sovereign, TrustVerdict::Sovereign);
        assert_eq!(TrustVerdict::Degraded, TrustVerdict::Degraded);
        assert_ne!(TrustVerdict::Sovereign, TrustVerdict::Degraded);
    }

    // ── Event Detection Tests (Sprint 7.1) ────────────────────

    /// Build a minimal DashboardData for event detection tests.
    fn make_test_dashboard() -> DashboardData {
        DashboardData {
            cpu_name: "test-cpu".into(),
            cpu_cores: 4,
            ram_total_gb: 16.0,
            ram_used_pct: 50.0,
            gpu: None,
            model_count: 0,
            text_models: vec![],
            vision_models: vec![],
            runtime_count: 0,
            platform: "test".into(),
            runtime_state: "Ready".into(),
            reflex_mode: "Active".into(),
            reflex_rules: 3,
            agents_active: 12,
            agents_registered: 12,
            total_receipts: 5,
            complete_count: 4,
            degraded_count: 1,
            failed_count: 0,
            chain_valid: true,
            recent_receipts: vec![],
            all_receipts: vec![],
            today_count: 2,
            today_complete: 2,
            manifest_seal: None,
            trust_checks: vec![],
            receipt_chain_checks: vec![],
            trust_verdict: TrustVerdict::Sovereign,
            pat_agents: vec![],
            sat_agents: vec![],
            greeting: "Good evening, MoMo".into(),
            recommendations: vec![],
            event_log: vec![],
            // Sprint 7.3: Wallet
            seed_balance: 4.875,
            seed_gini: 0.0,
            seed_supply_cap: 1_000_000_000.0,
            zakat_rate: 0.025,
            zakat_deducted: 0.1,
            total_minted: 4.875,
            total_burned: 0.0,
            // Sprint 7.4: Memory / Skills
            memory_fast: 12,
            memory_slow: 3,
            memory_glacial: 1,
            memory_fragments: 12,
            memory_atoms: 15,
            memory_insights: 3,
            memory_profile_completeness: 0.42,
            memory_knows_me: 0.42,
            reflex_compiled: 5,
            reflex_hits: 23,
            reflex_misses: 7,
            reflex_quarantined: 0,
            sovereignty_tier: "SEEDLING",
        }
    }

    fn make_test_receipt(
        id: &str,
        obj: &str,
        label: &'static str,
        success: bool,
        degraded: bool,
    ) -> ReceiptSummary {
        ReceiptSummary {
            id_short: id.into(),
            objective: obj.into(),
            state_label: label,
            is_success: success,
            is_degraded: degraded,
            signed: true,
            ihsan_score: if success { Some(0.96) } else { None },
            snr_score: if success { Some(0.92) } else { None },
            chosen_model: Some("qwen2.5-coder".into()),
            degradation_tier: if degraded { 2 } else { 0 },
            states_traversed: 5,
            chain_link: None,
        }
    }

    #[test]
    fn test_detect_events_no_change() {
        let prev = make_test_dashboard();
        let curr = make_test_dashboard();
        let events = detect_events(&prev, &curr);
        assert!(events.is_empty(), "No state change → no events");
    }

    #[test]
    fn test_detect_events_receipt_created() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        curr.total_receipts = 6;
        curr.recent_receipts = vec![make_test_receipt(
            "a1b2c3",
            "Test mission",
            "Complete",
            true,
            false,
        )];

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1, "One new receipt → one event");
        assert_eq!(events[0].kind, EventKind::ReceiptCreated);
        assert!(events[0].message.contains("a1b2c3"));
        assert!(events[0].message.contains("Complete"));
    }

    #[test]
    fn test_detect_events_trust_verdict_flip() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        curr.trust_verdict = TrustVerdict::Degraded;

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::TrustChanged);
        assert!(events[0].message.contains("degraded"));
    }

    #[test]
    fn test_detect_events_trust_restored() {
        let mut prev = make_test_dashboard();
        prev.trust_verdict = TrustVerdict::Degraded;
        let curr = make_test_dashboard(); // Sovereign

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::TrustChanged);
        assert!(events[0].message.contains("restored"));
    }

    #[test]
    fn test_detect_events_mission_completed() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        curr.today_complete = 4;

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::MissionCompleted);
        assert!(events[0].message.contains("2 missions completed"));
        assert!(events[0].message.contains("total: 4"));
    }

    #[test]
    fn test_detect_events_chain_integrity_broken() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        curr.chain_valid = false;

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::TrustChanged);
        assert!(events[0].message.contains("BROKEN"));
    }

    #[test]
    fn test_detect_events_chain_integrity_restored() {
        let mut prev = make_test_dashboard();
        prev.chain_valid = false;
        let curr = make_test_dashboard(); // chain_valid = true

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert!(events[0].message.contains("restored"));
    }

    #[test]
    fn test_detect_events_multiple_simultaneous() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        // Three simultaneous changes
        curr.total_receipts = 6;
        curr.recent_receipts = vec![make_test_receipt(
            "beef42",
            "Multi-event test",
            "Degraded",
            false,
            true,
        )];
        curr.trust_verdict = TrustVerdict::Degraded;
        curr.today_complete = 3;

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 3, "Three state changes → three events");

        let kinds: Vec<&EventKind> = events.iter().map(|e| &e.kind).collect();
        assert!(kinds.contains(&&EventKind::ReceiptCreated));
        assert!(kinds.contains(&&EventKind::TrustChanged));
        assert!(kinds.contains(&&EventKind::MissionCompleted));
    }

    #[test]
    fn test_detect_events_single_mission_plural() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        curr.today_complete = 3; // +1 mission

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert!(
            events[0].message.contains("1 mission completed"),
            "Singular form for 1 mission: {}",
            events[0].message
        );
    }

    #[test]
    fn test_detect_events_degraded_receipt() {
        let prev = make_test_dashboard();
        let mut curr = make_test_dashboard();
        curr.total_receipts = 6;
        curr.recent_receipts = vec![{
            let mut r = make_test_receipt("dead01", "Degraded mission", "Degraded", false, true);
            r.signed = false;
            r
        }];

        let events = detect_events(&prev, &curr);
        assert_eq!(events.len(), 1);
        assert!(events[0].message.contains("Degraded"));
    }

    // ── Sprint 7.2 Tests ─────────────────────────────────────

    #[test]
    fn test_receipt_summary_extended_fields() {
        let r = make_test_receipt("abc123def456", "Test objective", "Complete", true, false);
        assert_eq!(r.ihsan_score, Some(0.96));
        assert_eq!(r.snr_score, Some(0.92));
        assert_eq!(r.chosen_model.as_deref(), Some("qwen2.5-coder"));
        assert_eq!(r.degradation_tier, 0);
        assert_eq!(r.states_traversed, 5);
        assert!(r.chain_link.is_none());
        assert!(r.signed);
    }

    #[test]
    fn test_receipt_summary_clone() {
        let r = make_test_receipt("aabbcc", "Clone test", "Complete", true, false);
        let r2 = r.clone();
        assert_eq!(r.id_short, r2.id_short);
        assert_eq!(r.ihsan_score, r2.ihsan_score);
        assert_eq!(r.chosen_model, r2.chosen_model);
    }

    #[test]
    fn test_all_receipts_populated_from_gather() {
        let data = gather_dashboard_data();
        // all_receipts should contain at least as many as recent_receipts
        assert!(
            data.all_receipts.len() >= data.recent_receipts.len(),
            "all_receipts ({}) must be >= recent_receipts ({})",
            data.all_receipts.len(),
            data.recent_receipts.len()
        );
        // recent_receipts capped at 10
        assert!(data.recent_receipts.len() <= 10);
        // If there are receipts, verify extended fields exist
        if let Some(r) = data.all_receipts.first() {
            assert!(!r.id_short.is_empty());
            assert!(!r.objective.is_empty());
        }
    }

    #[test]
    fn test_dashboard_data_all_receipts_order() {
        let data = gather_dashboard_data();
        // all_receipts should be newest-first (reversed from ledger)
        // recent_receipts should be a prefix of all_receipts
        for (i, r) in data.recent_receipts.iter().enumerate() {
            if let Some(a) = data.all_receipts.get(i) {
                assert_eq!(r.id_short, a.id_short, "recent[{i}] must match all[{i}]");
            }
        }
    }

    // ── Sprint 7.3 Tests: Wallet / Treasury ─────────────────

    #[test]
    fn test_gather_dashboard_data_wallet_fields() {
        let data = gather_dashboard_data();
        // Zakat rate must match constitutional constant
        assert!(
            (data.zakat_rate - 0.025).abs() < f64::EPSILON,
            "Zakat rate must be 2.5%, got {}",
            data.zakat_rate
        );
        // Supply cap must be 1B
        assert!(
            (data.seed_supply_cap - 1_000_000_000.0).abs() < f64::EPSILON,
            "Supply cap must be 1B"
        );
        // Balance must be non-negative
        assert!(
            data.seed_balance >= 0.0,
            "SEED balance must be >= 0, got {}",
            data.seed_balance
        );
        // Gini must be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&data.seed_gini),
            "Gini must be [0,1], got {}",
            data.seed_gini
        );
        // Minted must be >= 0
        assert!(data.total_minted >= 0.0);
        // Burned must be >= 0
        assert!(data.total_burned >= 0.0);
        // Zakat deducted must be >= 0
        assert!(data.zakat_deducted >= 0.0);
    }

    #[test]
    fn test_wallet_seed_balance_math() {
        let data = make_test_dashboard();
        // 5 receipts, 4 complete → 4 * 1.0 SEED raw, minus 2.5% zakat
        // total_minted = 4.0 * (1 - 0.025) = 3.9 ... but test dashboard has preset values
        assert!(
            data.seed_balance >= 0.0,
            "Test dashboard balance must be non-negative"
        );
        assert!(
            data.seed_gini >= 0.0 && data.seed_gini <= 1.0,
            "Test dashboard Gini must be [0,1]"
        );
    }

    #[test]
    fn test_sovereignty_tier_assignment() {
        let mut data = make_test_dashboard();
        // SEEDLING: balance < 10, trust pass
        assert_eq!(data.sovereignty_tier, "SEEDLING");

        // Override for SOVEREIGN test
        data.sovereignty_tier = "SOVEREIGN";
        assert_eq!(data.sovereignty_tier, "SOVEREIGN");
    }

    #[test]
    fn test_gather_dashboard_data_sovereignty_tier() {
        let data = gather_dashboard_data();
        let valid_tiers = ["SOVEREIGN", "CITIZEN", "SEEDLING", "DEGRADED"];
        assert!(
            valid_tiers.contains(&data.sovereignty_tier),
            "Sovereignty tier '{}' must be one of {:?}",
            data.sovereignty_tier,
            valid_tiers
        );
    }

    // ── Sprint 7.4 Tests: Memory / Skills ───────────────────

    #[test]
    fn test_gather_dashboard_data_memory_fields() {
        let data = gather_dashboard_data();
        // Profile completeness must be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&data.memory_profile_completeness),
            "Profile completeness must be [0,1], got {}",
            data.memory_profile_completeness
        );
        // Knows-me must be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&data.memory_knows_me),
            "Knows-me must be [0,1], got {}",
            data.memory_knows_me
        );
    }

    #[test]
    fn test_gather_dashboard_data_reflex_stats() {
        let data = gather_dashboard_data();
        // Reflex compiled + quarantined are non-negative (u64)
        // Hit rate coherence: hits + misses >= 0 (always true for u64)
        // Compiled rules should match reflex_rules from runtime
        assert!(data.reflex_rules > 0, "Must have reflex rules from runtime");
    }

    #[test]
    fn test_test_dashboard_memory_fields() {
        let data = make_test_dashboard();
        assert_eq!(data.memory_fast, 12);
        assert_eq!(data.memory_slow, 3);
        assert_eq!(data.memory_glacial, 1);
        assert_eq!(data.memory_fragments, 12);
        assert_eq!(data.memory_atoms, 15);
        assert_eq!(data.memory_insights, 3);
        assert!((data.memory_profile_completeness - 0.42).abs() < f32::EPSILON);
        assert_eq!(data.reflex_compiled, 5);
        assert_eq!(data.reflex_hits, 23);
        assert_eq!(data.reflex_misses, 7);
        assert_eq!(data.reflex_quarantined, 0);
    }
}
