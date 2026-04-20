//! dema — BIZRA Principal CLI
//!
//! بسم الله الرحمن الرحيم
//!
//! The operator's terminal face for Dema. Talks to bizra-cognition-gateway
//! over localhost HTTP. Designed for Mumo's eye: terse, honest, no theatrics.
//!
//! Usage:
//!   dema                       → status (chain head, length, last activity)
//!   dema health                → gateway health + domain tag
//!   dema chain                 → chain head + length + latest timestamp
//!   dema receipt <h>           → show one receipt by hex id
//!   dema activate              → submit the principal activation intent (generic)
//!   dema activate-principal    → lawful principal activation via Cycle-7 G2 path
//!   dema submit "..."          → submit a custom intent
//!
//! Env:
//!   BIZRA_COGNITION_GATEWAY_URL (default http://127.0.0.1:7421)
//!   BIZRA_IDENTITY_ANCHOR       (default sovereign_state/identity/credentials.json)
//!
//! ci-hygiene waiver (2026-04-18): DTOs carry fields consumed via serde only.
#![allow(dead_code)]

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::process::ExitCode;

const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:7421";
const DEFAULT_ACTIVATION_INTENT: &str = "activate my dual agentic system as Node0 principal";
const DEFAULT_QUALITY_SCORE: f64 = 0.98;

#[derive(Parser)]
#[command(name = "dema")]
#[command(author = "Mumo (محمد) <sovereign@bizra.node0>")]
#[command(version)]
#[command(about = "BIZRA Principal CLI — the operator's terminal face for Dema")]
#[command(long_about = r#"
Dema from the terminal. Same gateway the /dema console uses.

QUICK START:
    dema              Status at a glance (chain head + length)
    dema activate     Submit principal activation intent
    dema submit "..." Custom intent
    dema chain        Full chain head view
    dema receipt <h>  Inspect one receipt by hex id

The gateway must be running locally (or set BIZRA_COGNITION_GATEWAY_URL).
"#)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Output JSON instead of human-readable format
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Gateway liveness check
    Health,
    /// Show chain head, length, latest timestamp
    Chain,
    /// Inspect one receipt by its 64-char hex id
    Receipt {
        /// Hex receipt id
        hash: String,
    },
    /// Submit the principal activation intent (Mumo's canonical mission)
    Activate {
        /// Override quality score (default 0.98, must be ≥ 0.95 for IHSAN_FLOOR)
        #[arg(long, default_value_t = DEFAULT_QUALITY_SCORE)]
        quality: f64,
    },
    /// Cycle-7 G2 — lawful principal activation via /principal/activate
    ///
    /// Loads the Python-authored node identity anchor, builds an
    /// activation-specific envelope, and seals both the NodeLifecycle
    /// mission receipt AND the PrincipalActivationReceipt to the chain.
    /// Persists the principal profile to sovereign_state/dema_cache/.
    ActivatePrincipal {
        /// Principal name (e.g. your given name)
        #[arg(long, default_value = "Mumo")]
        name: String,
        /// Declared role (default: node0_principal)
        #[arg(long, default_value = "node0_principal")]
        role: String,
        /// Quality score (default 0.98, must be ≥ 0.95 for IHSAN_FLOOR)
        #[arg(long, default_value_t = DEFAULT_QUALITY_SCORE)]
        quality: f64,
        /// Path to the node identity anchor JSON
        #[arg(long)]
        anchor: Option<String>,
    },
    /// Submit a custom mission intent
    Submit {
        /// Mission intent (free text)
        intent: String,
        /// Override quality score (default 0.98)
        #[arg(long, default_value_t = DEFAULT_QUALITY_SCORE)]
        quality: f64,
    },
    /// Cycle-7 G4 — register a local resource in the dema_cache registry
    ///
    /// Kind accepts the canonical variants (filesystem, network, process,
    /// credential) or any custom string. Unknown strings round-trip as
    /// Custom(...) so older/newer builds do not reject each other's data.
    RegisterResource {
        /// Resource kind (filesystem|network|process|credential|<custom>)
        #[arg(long)]
        kind: String,
        /// Resource id (path, host:port, pid, credential handle, ...)
        #[arg(long)]
        id: String,
        /// Human-readable summary
        #[arg(long, default_value = "")]
        summary: String,
        /// Mark this resource as allowlisted (G5 `dema organize` prerequisite)
        #[arg(long)]
        allowlisted: bool,
    },
    /// Cycle-7 G4 — list every resource registered in dema_cache
    ListResources,
    /// Cycle-7 G4 — show the Universal Resource Pattern projection
    Urp,
    /// Cycle-7 G5 — first real operator mission: read-only organize of
    /// an allowlisted filesystem path. Registers a listing digest,
    /// seals a MissionExecuted receipt to the chain.
    Organize {
        /// Absolute path to an allowlisted directory
        path: String,
        /// Quality score (default 0.98, must be ≥ 0.95 for IHSAN_FLOOR)
        #[arg(long, default_value_t = DEFAULT_QUALITY_SCORE)]
        quality: f64,
    },
    /// Cycle-7 G6 — Proof-of-Impact ledger. By default prints the
    /// operator-visible summary (totals + per-kind buckets). Use
    /// --full to print every entry.
    Poi {
        /// Show every ledger entry instead of the summary.
        #[arg(long)]
        full: bool,
    },
}

#[derive(Deserialize)]
struct Health {
    status: String,
    domain: String,
}

#[derive(Deserialize)]
struct ChainHead {
    head: String,
    length: usize,
    #[serde(rename = "latestTimestamp")]
    latest_timestamp: Option<u64>,
}

#[derive(Deserialize, Serialize)]
struct Receipt {
    id: String,
    kind: String,
    timestamp: Option<u64>,
    #[serde(rename = "prevChain")]
    prev_chain: String,
    #[serde(rename = "payloadHash")]
    payload_hash: String,
}

#[derive(Deserialize)]
struct GateVerdict {
    #[serde(rename = "scorerId")]
    scorer_id: String,
    invariant: Option<String>,
    verdict: String,
    reason: String,
    score: Option<f64>,
}

#[derive(Deserialize)]
struct RejectedClaim {
    invariant: String,
    reason: String,
    #[serde(rename = "remediationPath")]
    remediation_path: String,
    #[serde(rename = "escalationAllowed")]
    escalation_allowed: bool,
}

#[derive(Deserialize)]
struct Admissibility {
    verdict: String,
    #[serde(rename = "gateVerdicts")]
    gate_verdicts: Vec<GateVerdict>,
    rejected: Option<RejectedClaim>,
}

#[derive(Deserialize)]
struct SubmitOk {
    #[serde(rename = "missionId")]
    mission_id: String,
    admissibility: Admissibility,
    #[serde(rename = "receiptId")]
    receipt_id: String,
    #[serde(rename = "finalStage")]
    final_stage: String,
    #[serde(rename = "chainHead")]
    chain_head: String,
}

#[derive(Deserialize)]
struct SubmitError {
    error: ErrorBody,
}

#[derive(Deserialize)]
struct ErrorBody {
    code: String,
    message: String,
    #[serde(default)]
    admissibility: Option<Admissibility>,
}

// ─── Cycle-7 G2 principal activation DTOs ──────────────────────────────────

#[derive(Serialize)]
struct ActivatePrincipalRequest<'a> {
    #[serde(rename = "principalName")]
    principal_name: &'a str,
    #[serde(rename = "declaredRole")]
    declared_role: &'a str,
    #[serde(rename = "qualityScore")]
    quality_score: f64,
    #[serde(rename = "identityAnchorPath")]
    identity_anchor_path: &'a str,
}

#[derive(Deserialize)]
struct ActivatePrincipalOk {
    #[serde(rename = "missionId")]
    mission_id: String,
    #[serde(rename = "missionReceiptId")]
    mission_receipt_id: String,
    #[serde(rename = "principalActivationReceiptId")]
    principal_activation_receipt_id: String,
    #[serde(rename = "principalId")]
    principal_id: String,
    #[serde(rename = "profileHash")]
    profile_hash: String,
    #[serde(rename = "chainHead")]
    chain_head: String,
    #[serde(rename = "finalStage")]
    final_stage: String,
    admissibility: Admissibility,
    #[serde(rename = "cacheWarning")]
    cache_warning: Option<String>,
}

// ─── Cycle-7 G4 — /resources DTOs ──────────────────────────────────────────

#[derive(Serialize)]
struct RegisterResourceReq<'a> {
    kind: &'a str,
    id: &'a str,
    summary: &'a str,
    allowlisted: bool,
}

#[derive(Deserialize)]
struct ResourceOk {
    kind: String,
    id: String,
    summary: String,
    allowlisted: bool,
}

#[derive(Deserialize)]
struct RegisterResourceOk {
    outcome: String,
    resource: ResourceOk,
}

#[derive(Deserialize)]
struct ListResourcesOk {
    resources: Vec<ResourceOk>,
}

#[derive(Deserialize)]
struct UrpBucketOk {
    kind: String,
    resources: Vec<ResourceOk>,
}

#[derive(Deserialize)]
struct UrpViewOk {
    #[serde(rename = "totalCount")]
    total_count: usize,
    #[serde(rename = "allowlistedCount")]
    allowlisted_count: usize,
    buckets: Vec<UrpBucketOk>,
}

// ─── Cycle-7 G5 — organize DTOs ────────────────────────────────────────────

#[derive(Serialize)]
struct OrganizeReq<'a> {
    path: &'a str,
    #[serde(rename = "qualityScore")]
    quality_score: f64,
}

#[derive(Deserialize)]
struct OrganizeEntryOk {
    name: String,
    kind: String,
}

// ─── Cycle-7 G6 — poi DTOs ─────────────────────────────────────────────

#[derive(Deserialize)]
struct PoiEntryOk {
    #[serde(rename = "receiptId")]
    receipt_id: String,
    #[serde(rename = "receiptKindName")]
    receipt_kind_name: String,
    #[serde(rename = "qualityScore")]
    quality_score: f64,
    #[serde(rename = "gateMinScore")]
    gate_min_score: f64,
    #[serde(rename = "entryCount")]
    entry_count: u32,
    #[serde(rename = "impactScore")]
    impact_score: f64,
    #[serde(rename = "timestampNs")]
    timestamp_ns: u64,
    #[serde(rename = "principalId", default)]
    principal_id: Option<String>,
}

#[derive(Deserialize)]
struct PoiLedgerOk {
    #[serde(rename = "chainHead")]
    chain_head: String,
    entries: Vec<PoiEntryOk>,
}

#[derive(Deserialize)]
struct PoiPerKindOk {
    kind: String,
    count: usize,
    #[serde(rename = "totalImpact")]
    total_impact: f64,
    #[serde(rename = "avgImpact")]
    avg_impact: f64,
}

#[derive(Deserialize)]
struct PoiSummaryOk {
    #[serde(rename = "chainHead")]
    chain_head: String,
    #[serde(rename = "totalEntries")]
    total_entries: usize,
    #[serde(rename = "totalImpact")]
    total_impact: f64,
    #[serde(rename = "avgImpact")]
    avg_impact: f64,
    #[serde(rename = "maxImpact")]
    max_impact: f64,
    #[serde(rename = "byKind")]
    by_kind: Vec<PoiPerKindOk>,
}

#[derive(Deserialize)]
struct OrganizeOk {
    #[serde(rename = "missionId")]
    mission_id: String,
    #[serde(rename = "missionReceiptId")]
    mission_receipt_id: String,
    #[serde(rename = "organizeReceiptId")]
    organize_receipt_id: String,
    #[serde(rename = "chainHead")]
    chain_head: String,
    path: String,
    #[serde(rename = "listingDigest")]
    listing_digest: String,
    #[serde(rename = "fileCount")]
    file_count: u32,
    #[serde(rename = "dirCount")]
    dir_count: u32,
    #[serde(rename = "entryCount")]
    entry_count: u32,
    entries: Vec<OrganizeEntryOk>,
    admissibility: Admissibility,
}

#[derive(Serialize)]
struct SubmitRequest<'a> {
    intent: &'a str,
    #[serde(rename = "currentState")]
    current_state: StateSnapshot<'a>,
    #[serde(rename = "idealState")]
    ideal_state: StateSnapshot<'a>,
    #[serde(rename = "qualityScore")]
    quality_score: f64,
    originator: &'a str,
}

#[derive(Serialize)]
struct StateSnapshot<'a> {
    summary: &'a str,
    metric: f64,
}

fn gateway_url() -> String {
    std::env::var("BIZRA_COGNITION_GATEWAY_URL").unwrap_or_else(|_| DEFAULT_GATEWAY_URL.to_string())
}

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to build HTTP client")
}

// ─── Formatters ─────────────────────────────────────────────────────────────

fn print_health(h: &Health) {
    println!("  gateway: {} ({})", h.status, h.domain);
}

fn print_chain(c: &ChainHead) {
    let head_short = if c.head.chars().all(|ch| ch == '0') {
        format!("{} (genesis)", &c.head[..16])
    } else {
        format!("{}...", &c.head[..24])
    };
    let ts = c
        .latest_timestamp
        .map(|t| t.to_string())
        .unwrap_or_else(|| "—".into());
    println!("  head:     {}", head_short);
    println!("  length:   {}", c.length);
    println!("  latest:   {}", ts);
    if c.length == 0 {
        println!();
        println!(
            "  (chain empty — submit an intent with `dema activate` or `dema submit \"...\"`)"
        );
    }
}

fn print_receipt(r: &Receipt) {
    println!("  id:          {}", r.id);
    println!("  kind:        {}", r.kind);
    println!(
        "  timestamp:   {}",
        r.timestamp
            .map(|t| t.to_string())
            .unwrap_or_else(|| "—".into())
    );
    println!("  prev_chain:  {}...", &r.prev_chain[..24]);
    println!("  payload:     {}...", &r.payload_hash[..24]);
}

fn print_admissibility(a: &Admissibility, indent: &str) {
    for gv in &a.gate_verdicts {
        let mark = if gv.verdict == "Permit" { "✓" } else { "✗" };
        let score = gv
            .score
            .map(|s| format!("  score={:.4}", s))
            .unwrap_or_default();
        println!(
            "{}  {} {:<18} {:<8}{}",
            indent, mark, gv.scorer_id, gv.verdict, score
        );
    }
    println!("{}  verdict: {}", indent, a.verdict);
    if let Some(rej) = &a.rejected {
        println!("{}  rejected-by: {}", indent, rej.invariant);
        println!("{}  reason:      {}", indent, rej.reason);
        println!("{}  remediation: {}", indent, rej.remediation_path);
        println!(
            "{}  escalation:  {}",
            indent,
            if rej.escalation_allowed {
                "allowed"
            } else {
                "denied"
            }
        );
    }
}

fn print_submit_ok(s: &SubmitOk) {
    println!("  mission:     {}", s.mission_id);
    println!("  admissibility:");
    print_admissibility(&s.admissibility, "  ");
    println!("  receipt:     {}", s.receipt_id);
    println!("  stage:       {}", s.final_stage);
    println!("  chain_head:  {}", s.chain_head);
    if s.chain_head == s.receipt_id {
        println!("  ✓ chain head equals receipt id — sealed");
    }
}

fn print_submit_rejected(e: &ErrorBody) {
    println!("  REJECTED  (code: {})", e.code);
    println!("  message:  {}", e.message);
    if let Some(a) = &e.admissibility {
        println!();
        print_admissibility(a, "  ");
    }
}

// ─── Handlers ───────────────────────────────────────────────────────────────

fn cmd_health(json: bool) -> Result<()> {
    let url = format!("{}/health", gateway_url());
    let resp = client()?
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        return Err(anyhow!("gateway returned HTTP {}", resp.status()));
    }
    let body: Health = resp.json().context("decode /health")?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": body.status,
                "domain": body.domain,
            }))?
        );
    } else {
        print_health(&body);
    }
    Ok(())
}

fn cmd_chain(json: bool) -> Result<ChainHead> {
    let url = format!("{}/chain", gateway_url());
    let resp = client()?
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        return Err(anyhow!("gateway returned HTTP {}", resp.status()));
    }
    let body: ChainHead = resp.json().context("decode /chain")?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "head": body.head,
                "length": body.length,
                "latestTimestamp": body.latest_timestamp,
            }))?
        );
    } else {
        print_chain(&body);
    }
    Ok(body)
}

fn cmd_receipt(hash: &str, json: bool) -> Result<()> {
    let url = format!("{}/chain/{}", gateway_url(), hash);
    let resp = client()?
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    if resp.status().as_u16() == 404 {
        return Err(anyhow!("no receipt with hash {} in chain", hash));
    }
    if !resp.status().is_success() {
        return Err(anyhow!("gateway returned HTTP {}", resp.status()));
    }
    let body: Receipt = resp.json().context("decode receipt")?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    } else {
        print_receipt(&body);
    }
    Ok(())
}

fn cmd_submit(intent: &str, quality: f64, json: bool) -> Result<bool> {
    let url = format!("{}/mission", gateway_url());
    let req = SubmitRequest {
        intent,
        current_state: StateSnapshot {
            summary: "Principal state pre-mission",
            metric: 0.0,
        },
        ideal_state: StateSnapshot {
            summary: "Mission canonical, receipted",
            metric: 1.0,
        },
        quality_score: quality,
        originator: "Operator",
    };
    let resp = client()?
        .post(&url)
        .json(&req)
        .send()
        .with_context(|| format!("POST {}", url))?;

    let status = resp.status();
    let text = resp.text().context("read response body")?;

    if status.is_success() {
        let body: SubmitOk = serde_json::from_str(&text).context("decode submit OK")?;
        if json {
            println!("{}", text);
        } else {
            print_submit_ok(&body);
        }
        Ok(true)
    } else if status.as_u16() == 422 {
        let body: SubmitError = serde_json::from_str(&text).context("decode reject 422")?;
        if json {
            println!("{}", text);
        } else {
            print_submit_rejected(&body.error);
        }
        Ok(false)
    } else {
        Err(anyhow!(
            "gateway returned HTTP {} — body: {}",
            status,
            &text[..text.len().min(500)]
        ))
    }
}

fn default_anchor_path() -> String {
    std::env::var("BIZRA_IDENTITY_ANCHOR")
        .unwrap_or_else(|_| "sovereign_state/identity/credentials.json".to_string())
}

fn print_activate_principal_ok(a: &ActivatePrincipalOk) {
    println!("  principal:         {}", a.principal_id);
    println!("  mission:           {}", a.mission_id);
    println!("  mission_receipt:   {}", a.mission_receipt_id);
    println!("  activation_receipt:{}", a.principal_activation_receipt_id);
    println!("  profile_hash:      {}", a.profile_hash);
    println!("  stage:             {}", a.final_stage);
    println!("  chain_head:        {}", a.chain_head);
    println!("  admissibility:");
    print_admissibility(&a.admissibility, "  ");
    if a.chain_head == a.principal_activation_receipt_id {
        println!("  ✓ chain head equals principal activation receipt — sealed");
    }
    if let Some(w) = &a.cache_warning {
        println!("  ⚠ cache warning: {}", w);
    } else {
        let root = std::env::var("BIZRA_DEMA_CACHE_ROOT")
            .unwrap_or_else(|_| "sovereign_state".to_string());
        println!("  ✓ profile persisted to {}/dema_cache/", root);
    }
}

fn cmd_activate_principal(
    name: &str,
    role: &str,
    quality: f64,
    anchor: Option<&str>,
    json: bool,
) -> Result<bool> {
    let anchor_path_owned = anchor
        .map(|s| s.to_string())
        .unwrap_or_else(default_anchor_path);
    let req = ActivatePrincipalRequest {
        principal_name: name,
        declared_role: role,
        quality_score: quality,
        identity_anchor_path: &anchor_path_owned,
    };
    let url = format!("{}/principal/activate", gateway_url());
    let resp = client()?
        .post(&url)
        .json(&req)
        .send()
        .with_context(|| format!("POST {}", url))?;

    let status = resp.status();
    let text = resp.text().context("read response body")?;

    if status.is_success() {
        let body: ActivatePrincipalOk =
            serde_json::from_str(&text).context("decode principal activate OK")?;
        if json {
            println!("{}", text);
        } else {
            print_activate_principal_ok(&body);
        }
        Ok(true)
    } else if status.as_u16() == 422 {
        let body: SubmitError = serde_json::from_str(&text).context("decode reject 422")?;
        if json {
            println!("{}", text);
        } else {
            print_submit_rejected(&body.error);
        }
        Ok(false)
    } else {
        Err(anyhow!(
            "gateway returned HTTP {} — body: {}",
            status,
            &text[..text.len().min(500)]
        ))
    }
}

// ─── Cycle-7 G4 — resource registry CLI ────────────────────────────────

fn cmd_register_resource(
    kind: &str,
    id: &str,
    summary: &str,
    allowlisted: bool,
    json: bool,
) -> Result<()> {
    let req = RegisterResourceReq {
        kind,
        id,
        summary,
        allowlisted,
    };
    let url = format!("{}/resources/register", gateway_url());
    let resp = client()?
        .post(&url)
        .json(&req)
        .send()
        .with_context(|| format!("POST {}", url))?;
    let status = resp.status();
    let text = resp.text().context("read response body")?;
    if !status.is_success() {
        return Err(anyhow!(
            "gateway returned HTTP {} — body: {}",
            status,
            &text[..text.len().min(500)]
        ));
    }
    let body: RegisterResourceOk =
        serde_json::from_str(&text).context("decode register response")?;
    if json {
        println!("{}", text);
    } else {
        println!("register-resource — {}", body.outcome);
        println!("  kind:        {}", body.resource.kind);
        println!("  id:          {}", body.resource.id);
        println!("  summary:     {}", body.resource.summary);
        println!("  allowlisted: {}", body.resource.allowlisted);
    }
    Ok(())
}

fn cmd_list_resources(json: bool) -> Result<()> {
    let url = format!("{}/resources/list", gateway_url());
    let resp = client()?
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    let status = resp.status();
    let text = resp.text().context("read response body")?;
    if !status.is_success() {
        return Err(anyhow!(
            "gateway returned HTTP {} — body: {}",
            status,
            &text[..text.len().min(500)]
        ));
    }
    if json {
        println!("{}", text);
        return Ok(());
    }
    let body: ListResourcesOk = serde_json::from_str(&text).context("decode list response")?;
    if body.resources.is_empty() {
        println!("list-resources — (empty)");
        return Ok(());
    }
    println!("list-resources — {} registered", body.resources.len());
    for r in &body.resources {
        let marker = if r.allowlisted { "✓" } else { " " };
        println!("  {} {:<12} {}", marker, r.kind, r.id);
        if !r.summary.is_empty() {
            println!("              {}", r.summary);
        }
    }
    Ok(())
}

fn cmd_urp(json: bool) -> Result<()> {
    let url = format!("{}/resources/urp", gateway_url());
    let resp = client()?
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    let status = resp.status();
    let text = resp.text().context("read response body")?;
    if !status.is_success() {
        return Err(anyhow!(
            "gateway returned HTTP {} — body: {}",
            status,
            &text[..text.len().min(500)]
        ));
    }
    if json {
        println!("{}", text);
        return Ok(());
    }
    let body: UrpViewOk = serde_json::from_str(&text).context("decode urp response")?;
    println!(
        "URP — {} resources ({} allowlisted)",
        body.total_count, body.allowlisted_count
    );
    if body.buckets.is_empty() {
        println!("  (registry empty)");
        return Ok(());
    }
    for b in &body.buckets {
        println!();
        println!("  [{}] ({} entries)", b.kind, b.resources.len());
        for r in &b.resources {
            let marker = if r.allowlisted { "✓" } else { " " };
            println!("    {} {}", marker, r.id);
        }
    }
    Ok(())
}

// ─── Cycle-7 G5 — organize CLI ─────────────────────────────────────────

fn print_organize_ok(o: &OrganizeOk) {
    println!("organize — permitted, receipted, sealed");
    println!("  path:                {}", o.path);
    println!("  mission_id:          {}", o.mission_id);
    println!("  mission_receipt:     {}", o.mission_receipt_id);
    println!("  organize_receipt:    {}", o.organize_receipt_id);
    println!("  chain_head:          {}", o.chain_head);
    println!("  listing_digest:      {}", o.listing_digest);
    println!(
        "  summary:             {} entries ({} files, {} dirs)",
        o.entry_count, o.file_count, o.dir_count
    );
    if !o.entries.is_empty() {
        println!("  entries:");
        for e in &o.entries {
            let marker = match e.kind.as_str() {
                "directory" => "d",
                "file" => "f",
                "symlink" => "l",
                _ => "?",
            };
            println!("    {} {}", marker, e.name);
        }
    }
    println!("  admissibility:       verdict={}", o.admissibility.verdict);
    if o.chain_head == o.organize_receipt_id {
        println!("  ✓ chain head == organize receipt — sealed");
    }
}

fn cmd_organize(path: &str, quality: f64, json: bool) -> Result<bool> {
    let req = OrganizeReq {
        path,
        quality_score: quality,
    };
    let url = format!("{}/missions/organize", gateway_url());
    let resp = client()?
        .post(&url)
        .json(&req)
        .send()
        .with_context(|| format!("POST {}", url))?;
    let status = resp.status();
    let text = resp.text().context("read response body")?;
    if status.is_success() {
        let body: OrganizeOk = serde_json::from_str(&text).context("decode organize OK")?;
        if json {
            println!("{}", text);
        } else {
            print_organize_ok(&body);
        }
        Ok(true)
    } else if status.as_u16() == 422 {
        // Admissibility rejection — honest outcome, not a CLI error.
        let body: SubmitError = serde_json::from_str(&text).context("decode reject 422")?;
        if json {
            println!("{}", text);
        } else {
            print_submit_rejected(&body.error);
        }
        Ok(false)
    } else if status.as_u16() == 403 {
        // Pre-gate refusal — not allowlisted.
        if json {
            println!("{}", text);
        } else {
            let v: serde_json::Value =
                serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
            let msg = v["error"]["message"].as_str().unwrap_or("not allowlisted");
            eprintln!("dema organize REFUSED: {}", msg);
        }
        Ok(false)
    } else {
        Err(anyhow!(
            "gateway returned HTTP {} — body: {}",
            status,
            &text[..text.len().min(500)]
        ))
    }
}

// ─── Cycle-7 G6 — poi CLI ──────────────────────────────────────────────

fn print_poi_summary(s: &PoiSummaryOk) {
    println!("POI ledger — {} entries", s.total_entries);
    if s.total_entries == 0 {
        println!("  (ledger empty)");
        return;
    }
    println!("  total impact: {:.4}", s.total_impact);
    println!("  avg impact:   {:.4}", s.avg_impact);
    println!("  max impact:   {:.4}", s.max_impact);
    println!("  chain head:   {}", s.chain_head);
    println!();
    println!("  by kind:");
    for b in &s.by_kind {
        println!(
            "    [{}] count={} total={:.4} avg={:.4}",
            b.kind, b.count, b.total_impact, b.avg_impact
        );
    }
}

fn print_poi_ledger(l: &PoiLedgerOk) {
    println!("POI ledger — {} entries", l.entries.len());
    if l.entries.is_empty() {
        println!("  (ledger empty)");
        return;
    }
    println!("  chain head: {}", l.chain_head);
    for e in &l.entries {
        println!();
        println!("  [{}] impact={:.4}", e.receipt_kind_name, e.impact_score);
        println!("    receipt:       {}", e.receipt_id);
        println!(
            "    quality/gate:  {:.4} / {:.4}",
            e.quality_score, e.gate_min_score
        );
        println!("    entry_count:   {}", e.entry_count);
        println!("    timestamp_ns:  {}", e.timestamp_ns);
        if let Some(pid) = &e.principal_id {
            println!("    principal:     {}", pid);
        }
    }
}

fn cmd_poi(full: bool, json: bool) -> Result<()> {
    if full {
        let url = format!("{}/poi/ledger", gateway_url());
        let resp = client()?
            .get(&url)
            .send()
            .with_context(|| format!("GET {}", url))?;
        let status = resp.status();
        let text = resp.text().context("read response body")?;
        if !status.is_success() {
            return Err(anyhow!(
                "gateway returned HTTP {} — body: {}",
                status,
                &text[..text.len().min(500)]
            ));
        }
        if json {
            println!("{}", text);
            return Ok(());
        }
        let body: PoiLedgerOk = serde_json::from_str(&text).context("decode ledger")?;
        print_poi_ledger(&body);
    } else {
        let url = format!("{}/poi/summary", gateway_url());
        let resp = client()?
            .get(&url)
            .send()
            .with_context(|| format!("GET {}", url))?;
        let status = resp.status();
        let text = resp.text().context("read response body")?;
        if !status.is_success() {
            return Err(anyhow!(
                "gateway returned HTTP {} — body: {}",
                status,
                &text[..text.len().min(500)]
            ));
        }
        if json {
            println!("{}", text);
            return Ok(());
        }
        let body: PoiSummaryOk = serde_json::from_str(&text).context("decode summary")?;
        print_poi_summary(&body);
    }
    Ok(())
}

fn cmd_status(json: bool) -> Result<()> {
    // Default no-arg behavior: health + chain summary in one shot
    let h_url = format!("{}/health", gateway_url());
    let h_resp = client()?.get(&h_url).send().ok();
    match h_resp {
        Some(r) if r.status().is_success() => {
            let h: Health = r.json().context("decode /health")?;
            if !json {
                println!("DEMA — {}  ({})", h.status, h.domain);
                println!();
            }
        }
        _ => {
            if !json {
                println!("DEMA — gateway UNREACHABLE at {}", gateway_url());
                println!("  (start it: `cargo run -p bizra-cognition-gateway` or run the release binary)");
            }
            return Err(anyhow!("gateway not reachable"));
        }
    }
    let _ = cmd_chain(json)?;
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        None => cmd_status(cli.json),
        Some(Command::Health) => cmd_health(cli.json),
        Some(Command::Chain) => cmd_chain(cli.json).map(|_| ()),
        Some(Command::Receipt { hash }) => cmd_receipt(&hash, cli.json),
        Some(Command::Activate { quality }) => {
            match cmd_submit(DEFAULT_ACTIVATION_INTENT, quality, cli.json) {
                Ok(true) => Ok(()),
                Ok(false) => {
                    // Rejection is not an error for the CLI — it's a truthful outcome
                    return ExitCode::from(2);
                }
                Err(e) => Err(e),
            }
        }
        Some(Command::ActivatePrincipal {
            name,
            role,
            quality,
            anchor,
        }) => match cmd_activate_principal(&name, &role, quality, anchor.as_deref(), cli.json) {
            Ok(true) => Ok(()),
            Ok(false) => return ExitCode::from(2),
            Err(e) => Err(e),
        },
        Some(Command::Submit { intent, quality }) => match cmd_submit(&intent, quality, cli.json) {
            Ok(true) => Ok(()),
            Ok(false) => return ExitCode::from(2),
            Err(e) => Err(e),
        },
        Some(Command::RegisterResource {
            kind,
            id,
            summary,
            allowlisted,
        }) => cmd_register_resource(&kind, &id, &summary, allowlisted, cli.json),
        Some(Command::ListResources) => cmd_list_resources(cli.json),
        Some(Command::Urp) => cmd_urp(cli.json),
        Some(Command::Organize { path, quality }) => match cmd_organize(&path, quality, cli.json) {
            Ok(true) => Ok(()),
            Ok(false) => return ExitCode::from(2),
            Err(e) => Err(e),
        },
        Some(Command::Poi { full }) => cmd_poi(full, cli.json),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dema: {}", e);
            ExitCode::FAILURE
        }
    }
}
