//! dema — BIZRA Principal CLI
//!
//! بسم الله الرحمن الرحيم
//!
//! The operator's terminal face for Dema. Talks to bizra-cognition-gateway
//! over localhost HTTP. Designed for Mumo's eye: terse, honest, no theatrics.
//!
//! Usage:
//!   dema              → status (chain head, length, last activity)
//!   dema health       → gateway health + domain tag
//!   dema chain        → chain head + length + latest timestamp
//!   dema receipt <h>  → show one receipt by hex id
//!   dema activate     → submit the principal activation intent
//!   dema submit "..." → submit a custom intent
//!
//! Env:
//!   BIZRA_COGNITION_GATEWAY_URL (default http://127.0.0.1:7421)

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
    /// Submit a custom mission intent
    Submit {
        /// Mission intent (free text)
        intent: String,
        /// Override quality score (default 0.98)
        #[arg(long, default_value_t = DEFAULT_QUALITY_SCORE)]
        quality: f64,
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
        Some(Command::Submit { intent, quality }) => match cmd_submit(&intent, quality, cli.json) {
            Ok(true) => Ok(()),
            Ok(false) => return ExitCode::from(2),
            Err(e) => Err(e),
        },
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dema: {}", e);
            ExitCode::FAILURE
        }
    }
}
