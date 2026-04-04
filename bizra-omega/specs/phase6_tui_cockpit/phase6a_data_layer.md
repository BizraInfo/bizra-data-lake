# Phase 6A — Data Layer Extraction

## Purpose

Extract data-gathering logic from genesis_spine.rs print-heavy functions into reusable structs that the TUI can consume.

## Structs

```pseudocode
// ── In genesis_spine.rs (or a new dashboard.rs sibling) ──

pub struct ReceiptSummary {
    id_short: String,        // first 16 hex chars of receipt_id
    objective: String,       // truncated to ~40 chars
    state_label: String,     // "Complete" | "Degraded" | "Failed"
    is_success: bool,
    is_degraded: bool,
    signed: bool,
    model: Option<String>,
}

pub struct TrustCheck {
    name: String,            // e.g. "Ihsan (production)"
    passed: bool,
    detail: String,          // e.g. "0.95 = 0.95"
}

pub enum TrustVerdict { Sovereign, Degraded }

pub struct DashboardData {
    // Substrate
    cpu_name: String,
    cpu_cores: u32,
    ram_total_gb: f64,
    ram_used_pct: f64,
    gpu: Option<GpuInfo>,        // (name, used_mb, total_mb, pct)
    model_count: usize,
    text_models: Vec<String>,
    vision_models: Vec<String>,
    runtime_count: usize,
    platform: String,

    // Runtime health
    runtime_state: String,       // "Ready" | "Degraded" | "Offline"
    reflex_mode: String,
    reflex_rules: u32,
    agents_active: u32,
    agents_registered: u32,

    // Receipt chain
    total_receipts: usize,
    complete_count: usize,
    degraded_count: usize,
    failed_count: usize,
    chain_valid: bool,
    recent_receipts: Vec<ReceiptSummary>,  // last 10

    // Today's manifest
    today_count: usize,
    today_complete: usize,
    manifest_seal: Option<String>,

    // Trust
    trust_checks: Vec<TrustCheck>,
    trust_verdict: TrustVerdict,
    receipt_chain_checks: Vec<TrustCheck>,

    // Parliament
    pat_agents: Vec<AgentInfo>,
    sat_agents: Vec<AgentInfo>,

    // Ghost / Recommendations
    greeting: String,
    recommendations: Vec<String>,

    // Timestamp
    gathered_at: chrono::DateTime<chrono::Local>,
}

pub struct GpuInfo {
    name: String,
    used_mb: u64,
    total_mb: u64,
    used_pct: f64,
}

pub struct AgentInfo {
    index: u8,
    callsign: String,
    role: String,
    icon: String,
    team: String,  // "PAT" | "SAT"
}
```

## gather_dashboard_data() Pseudocode

```pseudocode
pub fn gather_dashboard_data() -> DashboardData {
    let now = chrono::Local::now()

    // ── 1. Substrate ──
    let manifest = ResourceManifest::discover()
    let hw = manifest.hardware
    let ram_used_pct = if hw.ram_total_gb > 0 {
        ((hw.ram_total_gb - hw.ram_available_gb) / hw.ram_total_gb) * 100.0
    } else { 0.0 }

    let gpu = hw.gpus.first().map(|g| GpuInfo {
        name: g.name.clone(),
        used_mb: g.vram_used_mb,
        total_mb: g.vram_total_mb,
        used_pct: if g.vram_total_mb > 0 {
            (g.vram_used_mb as f64 / g.vram_total_mb as f64) * 100.0
        } else { 0.0 },
    })

    // ── 2. Models ──
    let model_names = mission_bridge::extract_model_names(&manifest)
    let is_vision = |m: &str| m.contains("moondream") || m.contains("VL") || m.contains("vision")
    let text_models = model_names.iter().filter(|m| !is_vision(m)).cloned().collect()
    let vision_models = model_names.iter().filter(|m| is_vision(m)).cloned().collect()

    // ── 3. Runtime health ──
    let runtime = AgentRuntime::new()
    let health = runtime.health()

    // ── 4. Receipt chain ──
    let entries = load_ledger().unwrap_or_default()
    let total_receipts = entries.len()
    let complete_count = entries.iter().filter(|e| e.receipt.is_success()).count()
    let degraded_count = entries.iter().filter(|e| e.receipt.is_degraded()).count()
    let failed_count = total_receipts - complete_count - degraded_count
    let chain_valid = entries.iter().all(|e| e.receipt.verify_hash())

    // Recent receipts (last 10, reversed for display)
    let recent_receipts = entries.iter().rev().take(10).map(|e| {
        ReceiptSummary {
            id_short: e.receipt.id_hex()[..16].to_string(),
            objective: truncate_str(&e.objective, 40),
            state_label: if e.receipt.is_success() { "Complete" }
                         else if e.receipt.is_degraded() { "Degraded" }
                         else { "Failed" }.to_string(),
            is_success: e.receipt.is_success(),
            is_degraded: e.receipt.is_degraded(),
            signed: e.receipt.is_signed(),
            model: e.receipt.chosen_model.clone(),
        }
    }).collect()

    // ── 5. Today's manifest ──
    let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).as_secs()
    let day_start = now_secs - (now_secs % 86400)
    let today: Vec<_> = entries.iter()
        .filter(|e| e.receipt.completed_at >= day_start)
        .collect()
    let today_count = today.len()
    let today_complete = today.iter().filter(|e| e.receipt.is_success()).count()
    let manifest_seal = if !today.is_empty() {
        let mut hasher = blake3::Hasher::new()
        for e in &today { hasher.update(&e.receipt.receipt_id) }
        Some(hex::encode(hasher.finalize().as_bytes())[..16].to_string())
    } else { None }

    // ── 6. Trust checks ──
    // Same 13 checks as exec_trust()
    let mut trust_checks = Vec::new()
    let mut all_trust_pass = true

    // Constitutional law (5 checks)
    for (name, actual, expected) in [
        ("Ihsan (production)", IHSAN_THRESHOLD, 0.95),
        ("SNR (minimum)", SNR_THRESHOLD, 0.85),
        ("Gini (ceiling)", ADL_GINI_THRESHOLD, 0.35),
        ("Strict Ihsan", STRICT_IHSAN_THRESHOLD, 0.99),
        ("Runtime Ihsan", RUNTIME_IHSAN_THRESHOLD, 1.0),
    ] {
        let ok = (actual - expected).abs() < f64::EPSILON
        if !ok { all_trust_pass = false }
        trust_checks.push(TrustCheck {
            name: name.to_string(),
            passed: ok,
            detail: format!("{:.2} = {:.2}", actual, expected),
        })
    }

    // Topology (3 checks)
    let pat_ok = TopologyCanon::PAT_COUNT == 7
    let sat_ok = TopologyCanon::SAT_COUNT == 5
    let gate_ok = TopologyCanon::GATE_ORDER.len() == 3
    if !pat_ok || !sat_ok || !gate_ok { all_trust_pass = false }
    trust_checks.push(TrustCheck { name: "PAT-7".into(), passed: pat_ok, detail: format!("{}", TopologyCanon::PAT_COUNT) })
    trust_checks.push(TrustCheck { name: "SAT-5".into(), passed: sat_ok, detail: format!("{}", TopologyCanon::SAT_COUNT) })
    trust_checks.push(TrustCheck { name: "3-gate chain".into(), passed: gate_ok, detail: format!("{}", TopologyCanon::GATE_ORDER.len()) })

    // Genesis seal (1 check)
    let seal = GenesisSeal::node0_default(now_secs * 1000)  // millis
    let seal_ok = seal.seal_hash != [0u8; 32]
    if !seal_ok { all_trust_pass = false }
    trust_checks.push(TrustCheck { name: "Genesis Seal".into(), passed: seal_ok, detail: "computable".into() })

    // Receipt chain checks (2 checks)
    let receipt_chain_checks = vec![
        TrustCheck {
            name: format!("{} receipts", total_receipts),
            passed: chain_valid,
            detail: if chain_valid { "all valid" } else { "BROKEN" }.into(),
        },
        TrustCheck {
            name: "signed".into(),
            passed: entries.iter().all(|e| e.receipt.is_signed()),
            detail: format!("{}/{}", entries.iter().filter(|e| e.receipt.is_signed()).count(), total_receipts),
        },
    ]

    // Substrate checks (2 checks)
    let model_ok = manifest.total_models() > 0
    let ram_ok = hw.ram_total_gb >= 8.0
    if !model_ok || !ram_ok { all_trust_pass = false }
    trust_checks.push(TrustCheck { name: "LLM models".into(), passed: model_ok, detail: format!("{}", manifest.total_models()) })
    trust_checks.push(TrustCheck { name: "RAM >= 8 GB".into(), passed: ram_ok, detail: format!("{:.0} GB", hw.ram_total_gb) })

    let trust_verdict = if all_trust_pass { TrustVerdict::Sovereign } else { TrustVerdict::Degraded }

    // ── 7. Parliament ──
    let pat_agents = PatAgent::ALL.iter().map(|a| {
        let (icon, role_desc) = pat_display(a)
        AgentInfo { index: a.index(), callsign: a.callsign().to_string(), role: role_desc.to_string(), icon: icon.to_string(), team: "PAT".to_string() }
    }).collect()

    let sat_agents = SatAgent::ALL.iter().map(|a| {
        let (icon, role_desc) = sat_display(a)
        AgentInfo { index: a.index(), callsign: a.callsign().to_string(), role: role_desc.to_string(), icon: icon.to_string(), team: "SAT".to_string() }
    }).collect()

    // ── 8. Ghost / Recommendations ──
    let greeting = match now.hour() {
        5..=11 => "Good morning, MoMo.",
        12..=16 => "Good afternoon, MoMo.",
        17..=21 => "Good evening, MoMo.",
        _ => "Sovereign briefing, MoMo.",
    }.to_string()

    let mut recommendations = Vec::new()
    if entries.is_empty() { recommendations.push("Run your first mission: bizra mission \"<objective>\"") }
    if model_count == 0 { recommendations.push("Install a model: ollama pull qwen2.5:3b") }
    if text_models.is_empty() && model_count > 0 { recommendations.push("Install a text model — only vision models detected") }
    if !chain_valid { recommendations.push("Receipt chain has invalid hashes — run: bizra receipt --verify") }
    if recommendations.is_empty() { recommendations.push("System healthy. Ready for sovereign missions.") }

    // ── Assemble ──
    DashboardData { ... all fields ... }
}
```

## Integration with App

```pseudocode
// In app.rs — add to App struct:
pub dashboard_data: Option<DashboardData>,
pub last_refresh: Option<Instant>,

// In App::new():
dashboard_data: None,
last_refresh: None,

// In run_app() main loop — after event poll:
// Periodic refresh (every 5 seconds)
let should_refresh = app.last_refresh
    .map(|t| t.elapsed() > Duration::from_secs(5))
    .unwrap_or(true);

if should_refresh {
    app.dashboard_data = Some(gather_dashboard_data());
    app.last_refresh = Some(Instant::now());
}
```

## TDD Anchors

```pseudocode
#[test] fn test_gather_dashboard_data_no_panic() {
    // Calling gather_dashboard_data() should never panic,
    // even with no models, no receipts, no GPU.
    let data = gather_dashboard_data();
    assert!(!data.greeting.is_empty());
    assert!(data.trust_checks.len() >= 10);
    assert!(data.pat_agents.len() == 7);
    assert!(data.sat_agents.len() == 5);
}

#[test] fn test_trust_verdict_sovereign() {
    // With default constitutional constants, verdict should be Sovereign
    // (assuming models exist and RAM >= 8 GB on dev machine)
    let data = gather_dashboard_data();
    // Note: model/RAM checks may fail in CI — test the logic, not the env
    for check in &data.trust_checks[..5] {
        assert!(check.passed, "constitutional check '{}' should pass", check.name);
    }
}

#[test] fn test_receipt_summary_truncation() {
    let summary = ReceiptSummary {
        id_short: "a7f68f1f74f2c089".to_string(),
        objective: "a".repeat(100),
        ..
    };
    assert!(summary.objective.len() <= 43); // 40 + "..."
}
```
