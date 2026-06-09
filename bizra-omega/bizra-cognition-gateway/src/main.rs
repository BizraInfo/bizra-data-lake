// bizra-cognition-gateway
//
// HTTP projection of bizra-cognition runtime state for the Dema Console UI.
// NO_SHADOW_STATE: this gateway never owns truth. It holds a CognitionRuntime
// and returns its state verbatim. All mission writes go through
// CognitionRuntime::submit_mission which honors the 5-gate admissibility chain.
//
// v0.2 scope (Cycle-5 G3): v0.1 read surface + POST /mission for principal
// activation. Chain starts empty; first successful mission emits the founding
// activation receipt lineage.
//
// Spearpoint A: HTTP boundary contracts live in `mod contracts` and emit
// to `bindings/*.ts` via ts-rs on `cargo test`. CI enforces drift detection.
//
// ── ci-hygiene waivers (2026-04-18) ─────────────────────────────────
//   - dead_code: axum handler DTOs carry fields consumed by serde only.
//   - result_large_err: axum error tuples are inline for tower::Layer.
#![allow(dead_code, clippy::result_large_err)]

mod contracts;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use bizra_cognition::admissibility_freeze_v1::{
    AdmissibilityClaim, AdmissibilityResult, EconomicPattern, StateMutation, Verdict,
};
use bizra_cognition::mission_freeze_v1::{
    MissionEnvelope, MissionStage, Originator, StateSnapshot,
};
use bizra_cognition::poi_ledger::PoiEntry;
use bizra_cognition::principal_activation::{NodeIdentityAnchor, PrincipalActivationEnvelope};
use bizra_cognition::receipts::{Blake3Hash, InMemoryPayloadStore, ReceiptChain, ReceiptKind};
use bizra_cognition::resource_registry::{RegisterOutcome, ResourceKind, TypedResource, UrpView};
use bizra_cognition::runtime::OrganizeOutcome;
use bizra_cognition::runtime::{
    CognitionRuntime, MissionReplayResult, MissionRuntimeError, MissionRuntimeRecord,
};
use bizra_cognition::thought_graph::{AgentCtx, ThoughtGraph};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

const DOMAIN: &str = "bizra-cognition-gateway-v1";

#[derive(Clone)]
struct AppState {
    runtime: Arc<RwLock<CognitionRuntime>>,
}

// ─── DTOs ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    domain: &'static str,
}

#[derive(Serialize)]
struct ReceiptChainHeadDto {
    head: String,
    length: usize,
    #[serde(rename = "latestTimestamp")]
    latest_timestamp: Option<u64>,
    /// Cycle-6 G1 Phase 2 — count of verified envelopes in the attached
    /// sovereign_state snapshot (0 if no snapshot or in-memory mode).
    #[serde(rename = "sovereignEnvelopes", skip_serializing_if = "is_zero_usize")]
    sovereign_envelopes: usize,
    /// Total entries across all sovereign_state envelopes.
    #[serde(rename = "sovereignEntries", skip_serializing_if = "is_zero_usize")]
    sovereign_entries: usize,
}

fn is_zero_usize(n: &usize) -> bool {
    *n == 0
}

#[derive(Serialize)]
struct ReceiptDto {
    id: String,
    kind: String,
    timestamp: Option<u64>,
    #[serde(rename = "prevChain")]
    prev_chain: String,
    #[serde(rename = "payloadHash")]
    payload_hash: String,
    /// Cycle-6 G1 Phase 2 — true when this receipt came from the
    /// Python-authoritative sovereign_state/ durable projection
    /// rather than the in-memory ReceiptChain populated this session.
    #[serde(rename = "durable", skip_serializing_if = "std::ops::Not::not")]
    durable: bool,
}

#[derive(Deserialize)]
struct StateSnapshotDto {
    hash: String,
    summary: String,
    metric: f64,
}

#[derive(Deserialize)]
struct SubmitMissionRequest {
    intent: String,
    #[serde(rename = "operatorSessionId")]
    operator_session_id: String,
    #[serde(rename = "currentState")]
    current_state: StateSnapshotDto,
    #[serde(rename = "idealState")]
    ideal_state: StateSnapshotDto,
    #[serde(rename = "evidenceHash")]
    evidence_hash: String,
    #[serde(rename = "qualityScore")]
    quality_score: f64,
    #[serde(rename = "derivesFromCanonical")]
    derives_from_canonical: bool,
    #[serde(rename = "faceOnly")]
    face_only: bool,
    #[serde(rename = "economicPattern", default)]
    economic_pattern: Option<String>,
    #[serde(rename = "timestampNs", default)]
    timestamp_ns: Option<u64>,
}

#[derive(Serialize)]
struct GateVerdictDto {
    /// Scorer id — one of "IHSAN_FLOOR", "ZANN_ZERO", "RIBA_ZERO",
    /// "CLAIM_MUST_BIND", "NO_SHADOW_STATE", or "CHAIN" for aggregate.
    #[serde(rename = "scorerId")]
    scorer_id: String,
    /// Invariant name if the scorer was an invariant-gate; None for aggregate.
    invariant: Option<&'static str>,
    verdict: &'static str,
    reason: String,
    score: Option<f64>,
}

#[derive(Serialize)]
struct AdmissibilityResultDto {
    verdict: &'static str,
    #[serde(rename = "gateVerdicts")]
    gate_verdicts: Vec<GateVerdictDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rejected: Option<RejectedClaimDto>,
}

#[derive(Serialize)]
struct RejectedClaimDto {
    invariant: &'static str,
    reason: String,
    #[serde(rename = "remediationPath")]
    remediation_path: String,
    #[serde(rename = "escalationAllowed")]
    escalation_allowed: bool,
}

#[derive(Serialize)]
struct SubmitMissionResponse {
    #[serde(rename = "missionId")]
    mission_id: String,
    admissibility: AdmissibilityResultDto,
    #[serde(rename = "receiptId")]
    receipt_id: String,
    #[serde(rename = "finalStage")]
    final_stage: &'static str,
    #[serde(rename = "chainHead")]
    chain_head: String,
}

#[derive(Serialize)]
struct GetMissionResponse {
    #[serde(rename = "missionId")]
    mission_id: String,
    intent: String,
    stage: &'static str,
    rejected: bool,
    #[serde(rename = "timestampNs")]
    timestamp_ns: u64,
    admissibility: AdmissibilityResultDto,
    #[serde(rename = "receiptId", skip_serializing_if = "Option::is_none")]
    receipt_id: Option<String>,
    #[serde(rename = "chainHead")]
    chain_head: String,
}

// ─── Cycle-7 G2 live-walk — principal activation DTOs ──────────────────────

#[derive(Deserialize)]
struct ActivatePrincipalRequest {
    #[serde(rename = "principalName")]
    principal_name: String,
    #[serde(rename = "declaredRole", default = "default_declared_role")]
    declared_role: String,
    #[serde(rename = "qualityScore", default = "default_principal_quality")]
    quality_score: f64,
    /// Absolute or CWD-relative path to the node identity anchor JSON
    /// (Python-authored sovereign_state/identity/credentials.json).
    /// Defaults to BIZRA_IDENTITY_ANCHOR env var or the canonical path.
    #[serde(
        rename = "identityAnchorPath",
        default = "default_identity_anchor_path"
    )]
    identity_anchor_path: String,
}

fn default_declared_role() -> String {
    "node0_principal".to_string()
}

fn default_principal_quality() -> f64 {
    0.98
}

fn default_identity_anchor_path() -> String {
    std::env::var("BIZRA_IDENTITY_ANCHOR")
        .unwrap_or_else(|_| "sovereign_state/identity/credentials.json".to_string())
}

#[derive(Serialize)]
struct ActivatePrincipalResponse {
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
    final_stage: &'static str,
    admissibility: AdmissibilityResultDto,
    #[serde(rename = "cacheWarning", skip_serializing_if = "Option::is_none")]
    cache_warning: Option<String>,
    /// Path to the `dema_cache/` directory the gateway attached for this
    /// activation, or omitted if no cache was attached. Absolute if
    /// BIZRA_DEMA_CACHE_ROOT was set to an absolute path; relative
    /// otherwise — no canonicalization is performed so the reported path
    /// matches the one the runtime actually writes to, even under
    /// relative-cwd deployments. Derived server-side from the runtime's
    /// attached PrincipalProfileCache so clients can echo the
    /// authoritative persist location without reading their own env
    /// (CLI env may diverge from gateway env when talking to a remote
    /// gateway via BIZRA_COGNITION_GATEWAY_URL).
    #[serde(rename = "effectiveCacheDir", skip_serializing_if = "Option::is_none")]
    effective_cache_dir: Option<String>,
}

// ─── Cycle-7 G4 — /resources DTOs ──────────────────────────────────────────

#[derive(Deserialize)]
struct RegisterResourceRequest {
    kind: String,
    id: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    allowlisted: bool,
}

#[derive(Serialize)]
struct ResourceDto {
    kind: String,
    id: String,
    summary: String,
    allowlisted: bool,
}

impl From<&TypedResource> for ResourceDto {
    fn from(r: &TypedResource) -> Self {
        ResourceDto {
            kind: r.kind.as_str().to_string(),
            id: r.id.clone(),
            summary: r.summary.clone(),
            allowlisted: r.allowlisted,
        }
    }
}

#[derive(Serialize)]
struct RegisterResourceResponse {
    outcome: &'static str,
    resource: ResourceDto,
}

#[derive(Serialize)]
struct ListResourcesResponse {
    resources: Vec<ResourceDto>,
}

#[derive(Serialize)]
struct UrpBucketDto {
    kind: String,
    resources: Vec<ResourceDto>,
}

#[derive(Serialize)]
struct UrpViewDto {
    #[serde(rename = "totalCount")]
    total_count: usize,
    #[serde(rename = "allowlistedCount")]
    allowlisted_count: usize,
    buckets: Vec<UrpBucketDto>,
}

impl From<&UrpView> for UrpViewDto {
    fn from(v: &UrpView) -> Self {
        UrpViewDto {
            total_count: v.total_count,
            allowlisted_count: v.allowlisted_count,
            buckets: v
                .buckets
                .iter()
                .map(|b| UrpBucketDto {
                    kind: b.kind.as_str().to_string(),
                    resources: b.resources.iter().map(ResourceDto::from).collect(),
                })
                .collect(),
        }
    }
}

fn register_outcome_name(o: RegisterOutcome) -> &'static str {
    match o {
        RegisterOutcome::Created => "created",
        RegisterOutcome::Updated => "updated",
        RegisterOutcome::Idempotent => "idempotent",
    }
}

// ─── Cycle-7 G5 — /missions/organize DTOs ─────────────────────────────────

#[derive(Deserialize)]
struct OrganizeRequest {
    path: String,
    #[serde(rename = "qualityScore", default = "default_organize_quality")]
    quality_score: f64,
}

fn default_organize_quality() -> f64 {
    0.98
}

#[derive(Serialize)]
struct OrganizeEntryDto {
    name: String,
    kind: &'static str,
}

// ─── Cycle-7 G6 — /poi DTOs ───────────────────────────────────────────────

#[derive(Serialize)]
struct PoiEntryDto {
    #[serde(rename = "receiptId")]
    receipt_id: String,
    #[serde(rename = "receiptKindByte")]
    receipt_kind_byte: u8,
    #[serde(rename = "receiptKindName")]
    receipt_kind_name: &'static str,
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
    #[serde(rename = "principalId", skip_serializing_if = "Option::is_none")]
    principal_id: Option<String>,
}

impl From<&PoiEntry> for PoiEntryDto {
    fn from(e: &PoiEntry) -> Self {
        let receipt_kind_name = match e.receipt_kind_byte {
            0x61 => "PrincipalActivation",
            0x70 => "MissionExecuted",
            _ => "Unknown",
        };
        PoiEntryDto {
            receipt_id: hex32(&e.receipt_id),
            receipt_kind_byte: e.receipt_kind_byte,
            receipt_kind_name,
            quality_score: e.quality_score,
            gate_min_score: e.gate_min_score,
            entry_count: e.entry_count,
            impact_score: e.impact_score,
            timestamp_ns: e.timestamp_ns,
            principal_id: e.principal_id.as_ref().map(hex32),
        }
    }
}

#[derive(Serialize)]
struct PoiLedgerResponse {
    #[serde(rename = "chainHead")]
    chain_head: String,
    entries: Vec<PoiEntryDto>,
}

#[derive(Serialize)]
struct PoiPerKindDto {
    kind: &'static str,
    count: usize,
    #[serde(rename = "totalImpact")]
    total_impact: f64,
    #[serde(rename = "avgImpact")]
    avg_impact: f64,
}

#[derive(Serialize)]
struct PoiSummaryResponse {
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
    by_kind: Vec<PoiPerKindDto>,
}

#[derive(Serialize)]
struct OrganizeResponse {
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
    entries: Vec<OrganizeEntryDto>,
    #[serde(rename = "timestampNs")]
    timestamp_ns: u64,
    admissibility: AdmissibilityResultDto,
}

#[derive(Serialize)]
struct ReplayMissionResponse {
    #[serde(rename = "missionId")]
    mission_id: String,
    #[serde(rename = "replayResult")]
    replay_result: &'static str,
    #[serde(rename = "replayScope")]
    replay_scope: &'static str,
    #[serde(rename = "matchesPrevious")]
    matches_previous: bool,
    #[serde(rename = "chainHead")]
    chain_head: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
    domain: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    admissibility: Option<AdmissibilityResultDto>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn kind_name(k: ReceiptKind) -> &'static str {
    match k {
        ReceiptKind::Genesis => "Genesis",
        ReceiptKind::CognitionBoot => "CognitionBoot",
        ReceiptKind::Myelination => "Myelination",
        ReceiptKind::Demyelination => "Demyelination",
        ReceiptKind::ReasoningSession => "ReasoningSession",
        ReceiptKind::GovernanceDecision => "GovernanceDecision",
        ReceiptKind::NodeLifecycle => "NodeLifecycle",
        ReceiptKind::Manifest => "Manifest",
        ReceiptKind::PrincipalActivation => "PrincipalActivation",
        ReceiptKind::MissionExecuted => "MissionExecuted",
        ReceiptKind::DegradedPath => "DegradedPath",
    }
}

fn verdict_name(v: Verdict) -> &'static str {
    match v {
        Verdict::Permit => "Permit",
        Verdict::Reject => "Reject",
        Verdict::Review => "Review",
        Verdict::ScoreOnly => "ScoreOnly",
    }
}

fn stage_name(s: MissionStage) -> &'static str {
    match s {
        MissionStage::Intent => "Intent",
        MissionStage::Mission => "Mission",
        MissionStage::Claim => "Claim",
        MissionStage::Admissibility => "Admissibility",
        MissionStage::Execution => "Execution",
        MissionStage::Receipt => "Receipt",
        MissionStage::Canonicalization => "Canonicalization",
        MissionStage::Replayability => "Replayability",
        MissionStage::Reflex => "Reflex",
    }
}

fn hex32(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(s).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Some(out)
}

fn state_snapshot_from_dto(
    dto: StateSnapshotDto,
    field: &'static str,
) -> Result<StateSnapshot, (StatusCode, Json<ErrorResponse>)> {
    let hash = parse_hex32(&dto.hash).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_STATE_HASH",
                    message: format!("{} '{}' is not a 64-char hex string", field, dto.hash),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;
    Ok(StateSnapshot {
        hash,
        summary: dto.summary,
        metric: dto.metric,
    })
}

fn now_ns() -> Result<u64, (StatusCode, Json<ErrorResponse>)> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorBody {
                        code: "CLOCK_FAILURE",
                        message: format!("monotonic clock read failed: {}", e),
                        domain: DOMAIN,
                        admissibility: None,
                    },
                }),
            )
        })
}

fn parse_economic_pattern(
    value: Option<&str>,
) -> Result<Option<EconomicPattern>, (StatusCode, Json<ErrorResponse>)> {
    match value {
        None => Ok(None),
        Some("none") => Ok(Some(EconomicPattern::None)),
        Some("peer_exchange") => Ok(Some(EconomicPattern::PeerExchange)),
        Some("profit_sharing") => Ok(Some(EconomicPattern::ProfitSharing)),
        Some("fixed_return_lending") => Ok(Some(EconomicPattern::FixedReturnLending)),
        Some("hidden_fee_extraction") => Ok(Some(EconomicPattern::HiddenFeeExtraction)),
        Some("asymmetric_exploitation") => Ok(Some(EconomicPattern::AsymmetricExploitation)),
        Some(other) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_ECONOMIC_PATTERN",
                    message: format!("unsupported economicPattern '{}'", other),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
    }
}

fn replay_result_name(result: MissionReplayResult) -> &'static str {
    match result {
        MissionReplayResult::Match => "MATCH",
        MissionReplayResult::Divergent => "DIVERGENT",
    }
}

fn admissibility_to_dto(result: &AdmissibilityResult) -> AdmissibilityResultDto {
    let gate_verdicts = result
        .gate_verdicts
        .iter()
        .map(|gv| GateVerdictDto {
            scorer_id: gv.scorer_id.clone(),
            invariant: gv.invariant.map(|i| i.name()),
            verdict: verdict_name(gv.verdict),
            reason: gv.reason.clone(),
            score: gv.score,
        })
        .collect();

    AdmissibilityResultDto {
        verdict: verdict_name(result.verdict),
        gate_verdicts,
        rejected: result.rejected.as_ref().map(|r| RejectedClaimDto {
            invariant: r.invariant.name(),
            reason: r.reject_reason.clone(),
            remediation_path: r.remediation_path.clone(),
            escalation_allowed: r.escalation_allowed,
        }),
    }
}

fn mission_record_to_dto(record: &MissionRuntimeRecord) -> GetMissionResponse {
    GetMissionResponse {
        mission_id: hex32(&record.envelope.mission_id),
        intent: record.envelope.intent_text.clone(),
        stage: stage_name(record.stage),
        rejected: record.rejected,
        timestamp_ns: record.timestamp_ns,
        admissibility: admissibility_to_dto(&record.admissibility),
        receipt_id: record.receipt_id.map(|id| hex32(&id)),
        chain_head: hex32(&record.chain_head),
    }
}

/// Fresh in-memory runtime with no env dependencies. Used by the default
/// in-memory path of `bootstrap_runtime` AND by tests that need a runtime
/// decoupled from process env (so exported BIZRA_* vars in the test
/// environment don't silently alter the runtime under test).
fn fresh_in_memory_runtime(genesis: Blake3Hash) -> CognitionRuntime {
    // Empty-graph bootstrap. submit_mission only touches self.chain + self.missions
    // + admissibility evaluation — no graph traversal. This is the minimum viable
    // runtime for the G3 activation surface. Future arcs will attach PAT-7/SAT-5
    // factories via configure_cognition::default_pat7_sat5_config.
    let graph = ThoughtGraph::from_parts(HashMap::new(), Vec::new(), HashMap::new(), genesis);
    let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
    let ctx = AgentCtx {
        receipt_chain: genesis,
    };
    CognitionRuntime::new(graph, chain, ctx)
}

fn bootstrap_runtime(genesis: Blake3Hash) -> CognitionRuntime {
    // Cycle-6 G1 Phase 1 — try sovereign_state bootstrap if BIZRA_SOVEREIGN_STATE_PATH is set.
    //   - env unset       → in-memory bootstrap (dev mode, preserves Cycle-5 behavior)
    //   - env set, path missing → warn, fall back to in-memory
    //   - env set, path present, load OK → attach snapshot, serve durable-read
    //   - env set, path present, load FAILED → fail-closed startup (exit 1)
    //
    // Both branches flow into the shared dema_cache attachment block below
    // so BIZRA_DEMA_CACHE_ROOT is honored regardless of which bootstrap
    // path produced `rt` (fixes the case where setting both env vars
    // previously skipped cache attachment because sovereign-state loaded
    // first and returned early).
    let mut rt: CognitionRuntime = if let Ok(path_str) = std::env::var("BIZRA_SOVEREIGN_STATE_PATH")
    {
        let p = std::path::Path::new(&path_str);
        if p.exists() {
            match CognitionRuntime::from_sovereign_state(p) {
                Ok(rt) => {
                    if let Some(snap) = rt.sovereign_snapshot() {
                        tracing::info!(
                            target: DOMAIN,
                            envelopes = snap.envelopes_count(),
                            entries = snap.total_entries(),
                            block_zero = snap.block_zero_present,
                            path = %p.display(),
                            "bootstrap from sovereign_state OK (durable-read enabled)"
                        );
                    }
                    rt
                }
                Err(e) => {
                    tracing::error!(
                        target: DOMAIN,
                        error = %e,
                        path = %p.display(),
                        "BIZRA_SOVEREIGN_STATE_PATH load FAILED — aborting startup (fail-closed per Cycle-6 G1 canon)"
                    );
                    std::process::exit(1);
                }
            }
        } else {
            tracing::warn!(
                target: DOMAIN,
                path = %p.display(),
                "BIZRA_SOVEREIGN_STATE_PATH set but path missing; falling back to in-memory bootstrap"
            );
            fresh_in_memory_runtime(genesis)
        }
    } else {
        fresh_in_memory_runtime(genesis)
    };

    // Cycle-7 G2 Commit-3 — optional dema_cache attachment. When
    // BIZRA_DEMA_CACHE_ROOT is set to a non-empty value, permitted principal
    // activations will persist the profile to <root>/dema_cache/principal.json.
    // An empty string is treated as unset — Rust's std::env::var returns
    // Ok("") for an exported-but-empty var, which would otherwise cause
    // dema_cache/ to be attached at cwd. Fail-closed against that drift.
    // Restart: rehydrate on next boot via rehydrate_principal_from_cache.
    if let Ok(root_str) = std::env::var("BIZRA_DEMA_CACHE_ROOT").map(|s| s.trim().to_string()) {
        if !root_str.is_empty() {
            let root = std::path::PathBuf::from(root_str);
            rt.attach_dema_cache(&root);
            match rt.rehydrate_principal_from_cache() {
                Ok(true) => tracing::info!(
                    target: DOMAIN,
                    root = %root.display(),
                    "dema_cache attached and principal profile rehydrated from disk"
                ),
                Ok(false) => tracing::info!(
                    target: DOMAIN,
                    root = %root.display(),
                    "dema_cache attached; no principal profile present yet"
                ),
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "dema_cache attached but rehydrate failed — will rebuild from chain if needed"
                ),
            }
            // Cycle-7 G3 Commit-1 — log whether a prior receipt-history
            // snapshot is present. attach_dema_cache has already initialized
            // the ReceiptHistoryCache alongside PrincipalProfileCache.
            match rt.rehydrate_receipt_history_from_cache() {
                Ok(Some(snap)) => tracing::info!(
                    target: DOMAIN,
                    records = snap.records.len(),
                    "receipt_history cache present from prior session"
                ),
                Ok(None) => tracing::info!(
                    target: DOMAIN,
                    "receipt_history cache empty; will initialize on first chain advance"
                ),
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "receipt_history cache malformed — will rebuild from chain on next advance"
                ),
            }
            // Cycle-7 G3 Commit-2 — manifest_history presence.
            match rt.rehydrate_manifest_history_from_cache() {
                Ok(Some(snap)) => tracing::info!(
                    target: DOMAIN,
                    manifests = snap.manifests.len(),
                    "manifest_history cache present from prior session"
                ),
                Ok(None) => tracing::info!(
                    target: DOMAIN,
                    "manifest_history cache empty; will initialize on first permitted mission"
                ),
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "manifest_history cache malformed — will rebuild from missions on next permit"
                ),
            }
            // Cycle-7 G3 Commit-3 — mission_log presence.
            match rt.rehydrate_mission_log_from_cache() {
                Ok(Some(snap)) => tracing::info!(
                    target: DOMAIN,
                    entries = snap.entries.len(),
                    "mission_log cache present from prior session"
                ),
                Ok(None) => tracing::info!(
                    target: DOMAIN,
                    "mission_log cache empty; will initialize on first mission attempt"
                ),
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "mission_log cache malformed — will rebuild from missions on next attempt"
                ),
            }
            // Cycle-7 G3 Commit-4 — state_snapshots presence.
            match rt.rehydrate_state_snapshots_from_cache() {
                Ok(Some(snap)) => tracing::info!(
                    target: DOMAIN,
                    entries = snap.entries.len(),
                    "state_snapshots cache present from prior session"
                ),
                Ok(None) => tracing::info!(
                    target: DOMAIN,
                    "state_snapshots cache empty; will initialize on first mission attempt"
                ),
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "state_snapshots cache malformed — will rebuild from missions on next attempt"
                ),
            }
            // Cycle-7 G6 — restore PoI ledger in-memory state from the
            // disk cache so `/poi/ledger` and `/poi/summary` are correct
            // across gateway restarts. Chain stays truth; a future
            // rebuild-from-chain call can verify+repair if needed.
            match rt.load_poi_entries_from_cache() {
                Ok(true) => tracing::info!(
                    target: DOMAIN,
                    entries = rt.poi_entries().len(),
                    "poi_ledger restored from prior session"
                ),
                Ok(false) => tracing::info!(
                    target: DOMAIN,
                    "poi_ledger cache empty; ledger begins fresh this session"
                ),
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "poi_ledger cache malformed — ledger begins fresh; rebuild from chain will repair"
                ),
            }

            // Cycle-7 G3 Commit-5 — resource_registry seed. Ensure the
            // empty file exists at boot so G4 (URP + allowlist) can assume
            // the schema is already locked.
            match rt.seed_resource_registry_if_missing() {
                Ok(Some(true)) => tracing::info!(
                    target: DOMAIN,
                    "resource_registry seeded empty (G3 scope; G4 fills)"
                ),
                Ok(Some(false)) => match rt.rehydrate_resource_registry_from_cache() {
                    Ok(Some(snap)) => tracing::info!(
                        target: DOMAIN,
                        resources = snap.resources.len(),
                        "resource_registry cache present from prior session"
                    ),
                    Ok(None) => tracing::warn!(
                        target: DOMAIN,
                        "resource_registry seed reported existing but read returned None"
                    ),
                    Err(e) => tracing::warn!(
                        target: DOMAIN,
                        error = %e,
                        "resource_registry cache malformed — delete to re-seed"
                    ),
                },
                Ok(None) => {}
                Err(e) => tracing::warn!(
                    target: DOMAIN,
                    error = %e,
                    "resource_registry seed failed"
                ),
            }
        } else {
            tracing::info!(
                target: DOMAIN,
                "BIZRA_DEMA_CACHE_ROOT set but empty — treated as unset (no dema_cache attached)"
            );
        }
    }

    // Cycle-6 Arc 3 — authoritative receipt chain persistence. When
    // BIZRA_RECEIPT_STORE_PATH is set, replace the in-memory payload store
    // with sled + chain_snapshot.json. Fail-closed on corrupt store load.
    // Distinct from BIZRA_DEMA_CACHE_ROOT (derived cache only).
    match rt.bootstrap_authoritative_receipt_store_from_env(genesis) {
        Ok(true) => {
            if let Some(store) = rt.receipt_chain_store() {
                tracing::info!(
                    target: DOMAIN,
                    root = %store.root().display(),
                    chain_len = rt.chain.len(),
                    head = %hex32(&rt.chain.head()),
                    "authoritative receipt store bootstrapped (Cycle-6 Arc 3)"
                );
            }
        }
        Ok(false) => {}
        Err(e) => {
            tracing::error!(
                target: DOMAIN,
                error = %e,
                "BIZRA_RECEIPT_STORE_PATH bootstrap FAILED — aborting startup (fail-closed)"
            );
            std::process::exit(1);
        }
    }

    rt
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        domain: DOMAIN,
    })
}

async fn get_chain(State(state): State<AppState>) -> Json<ReceiptChainHeadDto> {
    let rt = state.runtime.read().await;
    // Cycle-6 G1 Phase 2 — if a sovereign_state snapshot is attached, report
    // the durable entry count alongside the in-memory length. The head stays
    // the in-memory head (most recent activity this session); durable entries
    // are pre-restart history accessible via /chain/{hash} fall-through.
    let (sovereign_envelopes, sovereign_entries) = match rt.sovereign_snapshot() {
        Some(snap) => (snap.envelopes_count(), snap.total_entries()),
        None => (0, 0),
    };
    Json(ReceiptChainHeadDto {
        head: hex32(&rt.chain.head()),
        length: rt.chain.len(),
        latest_timestamp: rt.chain.latest_timestamp(),
        sovereign_envelopes,
        sovereign_entries,
    })
}

async fn get_chain_receipt(
    State(state): State<AppState>,
    Path(hash_hex): Path<String>,
) -> Result<Json<ReceiptDto>, (StatusCode, Json<ErrorResponse>)> {
    let target = parse_hex32(&hash_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_HASH",
                    message: format!("hash '{}' is not a 64-char hex string", hash_hex),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    let rt = state.runtime.read().await;
    // Check in-memory chain first (newest activity this session).
    for record in rt.chain.records() {
        if record.hash == target {
            return Ok(Json(ReceiptDto {
                id: hex32(&record.hash),
                kind: kind_name(record.kind).to_string(),
                timestamp: None,
                prev_chain: hex32(&record.prev),
                payload_hash: hex32(&record.hash),
                durable: false,
            }));
        }
    }

    // Cycle-6 G1 Phase 2 — fall through to sovereign_state snapshot if attached.
    // This closes the niyyah §G1 verification: seal receipt X -> restart gateway
    // -> /chain/X still returns the receipt. After restart, the in-memory chain
    // is empty but the snapshot holds the Python-authored durable history.
    if let Some(snap) = rt.sovereign_snapshot() {
        if let Some(entry) = snap.find_entry_by_hash(&hash_hex) {
            return Ok(Json(ReceiptDto {
                id: entry.hash.clone(),
                kind: entry.event.clone(),
                timestamp: None,
                prev_chain: entry.prev_hash.clone(),
                payload_hash: entry.hash.clone(),
                durable: true,
            }));
        }
    }

    Err((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: ErrorBody {
                code: "RECEIPT_NOT_FOUND",
                message: format!("no receipt with hash {} in chain", hash_hex),
                domain: DOMAIN,
                admissibility: None,
            },
        }),
    ))
}

async fn post_mission(
    State(state): State<AppState>,
    Json(req): Json<SubmitMissionRequest>,
) -> Result<Json<SubmitMissionResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.intent.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "EMPTY_INTENT",
                    message: "mission intent must be non-empty".into(),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        ));
    }

    let operator_session_id = parse_hex32(&req.operator_session_id).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_OPERATOR_SESSION_ID",
                    message: format!(
                        "operatorSessionId '{}' is not a 64-char hex string",
                        req.operator_session_id
                    ),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;
    let evidence_hash = parse_hex32(&req.evidence_hash).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_EVIDENCE_HASH",
                    message: format!(
                        "evidenceHash '{}' is not a 64-char hex string",
                        req.evidence_hash
                    ),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;
    let originator = Originator::Operator {
        session_id: operator_session_id,
    };
    if req.timestamp_ns.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "TIMESTAMP_RUNTIME_OWNED",
                    message: "timestampNs is runtime-owned and cannot be supplied by the caller"
                        .into(),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        ));
    }
    let ts_ns = now_ns()?;

    let current = state_snapshot_from_dto(req.current_state, "currentState.hash")?;
    let ideal = state_snapshot_from_dto(req.ideal_state, "idealState.hash")?;

    let envelope =
        MissionEnvelope::from_intent(req.intent.clone(), current, ideal, originator, ts_ns);
    let claim_id = envelope.extract_claim_id();

    let claim = AdmissibilityClaim {
        claim_id,
        has_evidence: true,
        evidence_hash: Some(evidence_hash),
        economic_pattern: parse_economic_pattern(req.economic_pattern.as_deref())?,
        state_mutation: Some(StateMutation {
            derives_from_canonical: req.derives_from_canonical,
            face_only: req.face_only,
        }),
        quality_score: req.quality_score,
        timestamp_ns: ts_ns,
    };

    // G2-hardening contract: submit_mission returns a MissionRuntimeRecord
    // on BOTH permit and reject paths. Rejection is not an error — it is
    // structured state. Branch on record.rejected to surface to the operator.
    let mut rt = state.runtime.write().await;
    match rt.submit_mission(envelope, claim) {
        Ok(record) if !record.rejected => {
            let receipt_id = record.receipt_id.ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorBody {
                            code: "PERMIT_WITHOUT_RECEIPT",
                            message:
                                "permit record missing receipt_id — runtime invariant violated"
                                    .into(),
                            domain: DOMAIN,
                            admissibility: None,
                        },
                    }),
                )
            })?;

            Ok(Json(SubmitMissionResponse {
                mission_id: hex32(&record.envelope.mission_id),
                admissibility: admissibility_to_dto(&record.admissibility),
                receipt_id: hex32(&receipt_id),
                final_stage: stage_name(record.stage),
                chain_head: hex32(&record.chain_head),
            }))
        }
        Ok(record) => {
            // record.rejected == true: structured rejection. Chain NOT advanced.
            // Rejection preserved in missions registry; caller receives HTTP 422
            // with full admissibility detail + RejectedClaim remediation path.
            Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: ErrorBody {
                        code: "ADMISSIBILITY_REJECTED",
                        message: "mission rejected by admissibility chain".into(),
                        domain: DOMAIN,
                        admissibility: Some(admissibility_to_dto(&record.admissibility)),
                    },
                }),
            ))
        }
        Err(MissionRuntimeError::ClaimMismatch { expected, got }) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "CLAIM_MISMATCH",
                    message: format!(
                        "claim_id mismatch (expected {} got {})",
                        hex32(&expected),
                        hex32(&got)
                    ),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::DuplicateMission(id)) => Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "DUPLICATE_MISSION",
                    message: format!("mission {} already exists", hex32(&id)),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::Chain(e)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "CHAIN_ERROR",
                    message: format!("{:?}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::Clock(msg)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "CLOCK_FAILURE",
                    message: msg,
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::MissionNotFound(_)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "UNEXPECTED",
                    message: "mission-not-found on submit path".into(),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
    }
}

// ─── Cycle-7 G2 live-walk — POST /principal/activate ───────────────────────
//
// Wraps CognitionRuntime::submit_principal_activation. Loads the Python-
// authored node identity anchor, builds a PrincipalActivationEnvelope,
// and threads it through the lawful mission loop + PrincipalActivationReceipt
// append. On reject returns HTTP 422 with structured remediation text
// per niyyah Frozen Law #5 (fail-closed honestly).

async fn post_principal_activate(
    State(state): State<AppState>,
    Json(req): Json<ActivatePrincipalRequest>,
) -> Result<Json<ActivatePrincipalResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.principal_name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "EMPTY_PRINCIPAL_NAME",
                    message: "principalName must be non-empty".into(),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        ));
    }

    let anchor_path = std::path::PathBuf::from(&req.identity_anchor_path);
    let anchor = NodeIdentityAnchor::load(&anchor_path).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "IDENTITY_ANCHOR_LOAD",
                    message: format!(
                        "failed to load node identity anchor at {}: {}",
                        anchor_path.display(),
                        e
                    ),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    let ts_ns = now_ns()?;
    let envelope = PrincipalActivationEnvelope::from_anchor(
        req.principal_name.clone(),
        req.declared_role.clone(),
        &anchor,
        ts_ns,
    )
    .map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "ENVELOPE_BUILD_FAILED",
                    message: format!("failed to build activation envelope: {}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    let mut rt = state.runtime.write().await;
    let record = rt
        .submit_principal_activation(envelope, req.quality_score)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorBody {
                        code: "RUNTIME_ERROR",
                        message: format!("submit_principal_activation failed: {:?}", e),
                        domain: DOMAIN,
                        admissibility: None,
                    },
                }),
            )
        })?;

    if record.rejected {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "ADMISSIBILITY_REJECTED",
                    message: record
                        .remediation
                        .unwrap_or_else(|| "activation rejected".into()),
                    domain: DOMAIN,
                    admissibility: Some(admissibility_to_dto(&record.mission_record.admissibility)),
                },
            }),
        ));
    }

    let profile = record
        .profile
        .as_ref()
        .expect("permit invariant: profile must be Some");
    let pa_receipt = record
        .activation_receipt
        .as_ref()
        .expect("permit invariant: activation_receipt must be Some");
    let mission_receipt_id = record
        .mission_record
        .receipt_id
        .expect("permit invariant: mission receipt_id must be Some");

    Ok(Json(ActivatePrincipalResponse {
        mission_id: hex32(&record.mission_record.envelope.mission_id),
        mission_receipt_id: hex32(&mission_receipt_id),
        principal_activation_receipt_id: hex32(&pa_receipt.receipt_id),
        principal_id: hex32(&profile.principal_id),
        profile_hash: hex32(&pa_receipt.principal_profile_hash),
        chain_head: hex32(&rt.chain.head()),
        final_stage: stage_name(record.mission_record.stage),
        admissibility: admissibility_to_dto(&record.mission_record.admissibility),
        cache_warning: record.cache_warning,
        effective_cache_dir: record.effective_cache_dir.map(|p| p.display().to_string()),
    }))
}

// ─── Cycle-7 G4 — /resources handlers ──────────────────────────────────────

async fn post_resource_register(
    State(state): State<AppState>,
    Json(req): Json<RegisterResourceRequest>,
) -> Result<Json<RegisterResourceResponse>, (StatusCode, Json<ErrorResponse>)> {
    let kind = ResourceKind::from_str(&req.kind);
    let resource = TypedResource::new(kind, req.id, req.summary, req.allowlisted).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_RESOURCE",
                    message: format!("{}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    let rt = state.runtime.read().await;
    let outcome = rt.register_resource(resource.clone()).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "REGISTER_FAILED",
                    message: format!("{}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    Ok(Json(RegisterResourceResponse {
        outcome: register_outcome_name(outcome),
        resource: ResourceDto::from(&resource),
    }))
}

async fn get_resources_list(
    State(state): State<AppState>,
) -> Result<Json<ListResourcesResponse>, (StatusCode, Json<ErrorResponse>)> {
    let rt = state.runtime.read().await;
    let resources = rt.list_resources().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "LIST_FAILED",
                    message: format!("{}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;
    Ok(Json(ListResourcesResponse {
        resources: resources.iter().map(ResourceDto::from).collect(),
    }))
}

// ─── Cycle-7 G6 — /poi handlers ──────────────────────────────────────────

async fn get_poi_ledger(State(state): State<AppState>) -> Json<PoiLedgerResponse> {
    let rt = state.runtime.read().await;
    let snap = rt.poi_ledger_snapshot();
    Json(PoiLedgerResponse {
        chain_head: hex32(&snap.chain_head),
        entries: snap.entries.iter().map(PoiEntryDto::from).collect(),
    })
}

async fn get_poi_summary(State(state): State<AppState>) -> Json<PoiSummaryResponse> {
    let rt = state.runtime.read().await;
    let snap = rt.poi_ledger_snapshot();
    let total = snap.entries.len();
    let total_impact: f64 = snap.entries.iter().map(|e| e.impact_score).sum();
    let max_impact = snap
        .entries
        .iter()
        .map(|e| e.impact_score)
        .fold(0.0_f64, f64::max);
    let avg_impact = if total > 0 {
        total_impact / (total as f64)
    } else {
        0.0
    };

    let kinds: [(u8, &'static str); 2] = [(0x61, "PrincipalActivation"), (0x70, "MissionExecuted")];
    let mut by_kind: Vec<PoiPerKindDto> = Vec::with_capacity(kinds.len());
    for (byte, name) in kinds {
        let scoped: Vec<&PoiEntry> = snap
            .entries
            .iter()
            .filter(|e| e.receipt_kind_byte == byte)
            .collect();
        let count = scoped.len();
        if count == 0 {
            continue;
        }
        let total: f64 = scoped.iter().map(|e| e.impact_score).sum();
        by_kind.push(PoiPerKindDto {
            kind: name,
            count,
            total_impact: total,
            avg_impact: total / (count as f64),
        });
    }

    Json(PoiSummaryResponse {
        chain_head: hex32(&snap.chain_head),
        total_entries: total,
        total_impact,
        avg_impact,
        max_impact,
        by_kind,
    })
}

async fn post_organize(
    State(state): State<AppState>,
    Json(req): Json<OrganizeRequest>,
) -> Result<Json<OrganizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.path.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "EMPTY_PATH",
                    message: "path must be non-empty".into(),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        ));
    }
    let path = std::path::PathBuf::from(&req.path);
    let mut rt = state.runtime.write().await;
    let outcome = rt
        .submit_organize_mission(&path, req.quality_score)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorBody {
                        code: "RUNTIME_ERROR",
                        message: format!("submit_organize_mission failed: {:?}", e),
                        domain: DOMAIN,
                        admissibility: None,
                    },
                }),
            )
        })?;

    match outcome {
        OrganizeOutcome::NotAllowlisted { path, remediation } => Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "PATH_NOT_ALLOWLISTED",
                    message: format!("{} — {}", path, remediation),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        OrganizeOutcome::IoError { path, error } => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "ORGANIZE_IO_ERROR",
                    message: format!("failed to read {}: {}", path, error),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        OrganizeOutcome::Rejected {
            mission_record,
            remediation,
        } => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "ADMISSIBILITY_REJECTED",
                    message: remediation,
                    domain: DOMAIN,
                    admissibility: Some(admissibility_to_dto(&mission_record.admissibility)),
                },
            }),
        )),
        OrganizeOutcome::Executed {
            mission_record,
            organize_receipt,
            listing,
        } => {
            let mission_receipt_id = mission_record
                .receipt_id
                .expect("permit invariant: receipt_id must be Some");
            let entries: Vec<OrganizeEntryDto> = listing
                .entries
                .iter()
                .map(|e| OrganizeEntryDto {
                    name: e.name.clone(),
                    kind: e.kind_str(),
                })
                .collect();
            Ok(Json(OrganizeResponse {
                mission_id: hex32(&mission_record.envelope.mission_id),
                mission_receipt_id: hex32(&mission_receipt_id),
                organize_receipt_id: hex32(&organize_receipt.receipt_id),
                chain_head: hex32(&rt.chain.head()),
                path: listing.path.clone(),
                listing_digest: hex32(&organize_receipt.listing_digest),
                file_count: organize_receipt.file_count,
                dir_count: organize_receipt.dir_count,
                entry_count: organize_receipt.entry_count,
                entries,
                timestamp_ns: organize_receipt.timestamp_ns,
                admissibility: admissibility_to_dto(&mission_record.admissibility),
            }))
        }
    }
}

async fn get_resources_urp(
    State(state): State<AppState>,
) -> Result<Json<UrpViewDto>, (StatusCode, Json<ErrorResponse>)> {
    let rt = state.runtime.read().await;
    let view = rt.urp_view().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "URP_FAILED",
                    message: format!("{}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;
    Ok(Json(UrpViewDto::from(&view)))
}

async fn get_mission(
    State(state): State<AppState>,
    Path(hash_hex): Path<String>,
) -> Result<Json<GetMissionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mission_id = parse_hex32(&hash_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_MISSION_ID",
                    message: format!("mission id '{}' is not a 64-char hex string", hash_hex),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    let rt = state.runtime.read().await;
    let record = rt.mission_by_id(&mission_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "MISSION_NOT_FOUND",
                    message: format!("mission {} not found", hash_hex),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;
    Ok(Json(mission_record_to_dto(record)))
}

async fn replay_mission(
    State(state): State<AppState>,
    Path(hash_hex): Path<String>,
) -> Result<Json<ReplayMissionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mission_id = parse_hex32(&hash_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "INVALID_MISSION_ID",
                    message: format!("mission id '{}' is not a 64-char hex string", hash_hex),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )
    })?;

    let rt = state.runtime.read().await;
    match rt.rehydrate_mission(&mission_id) {
        Ok(report) => Ok(Json(ReplayMissionResponse {
            mission_id: hex32(&report.mission_id),
            replay_result: replay_result_name(report.replay_result),
            replay_scope: "CLAIM_ADMISSIBILITY_ONLY",
            matches_previous: report.matches_previous,
            chain_head: hex32(&report.chain_head),
        })),
        Err(MissionRuntimeError::MissionNotFound(_)) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "MISSION_NOT_FOUND",
                    message: format!("mission {} not found", hash_hex),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::Clock(msg)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "CLOCK_FAILURE",
                    message: msg,
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::Chain(e)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "CHAIN_ERROR",
                    message: format!("{:?}", e),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::ClaimMismatch { expected, got }) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "CLAIM_MISMATCH",
                    message: format!(
                        "claim_id mismatch (expected {} got {})",
                        hex32(&expected),
                        hex32(&got)
                    ),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
        Err(MissionRuntimeError::DuplicateMission(id)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "DUPLICATE_MISSION",
                    message: format!("mission {} already exists", hex32(&id)),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        )),
    }
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/chain", get(get_chain))
        .route("/chain/:hash", get(get_chain_receipt))
        .route("/mission", post(post_mission))
        .route("/missions", post(post_mission))
        .route("/missions/:hash", get(get_mission))
        .route("/missions/:hash/replay", post(replay_mission))
        .route("/principal/activate", post(post_principal_activate))
        .route("/resources/register", post(post_resource_register))
        .route("/resources/list", get(get_resources_list))
        .route("/resources/urp", get(get_resources_urp))
        .route("/missions/organize", post(post_organize))
        .route("/poi/ledger", get(get_poi_ledger))
        .route("/poi/summary", get(get_poi_summary))
        .with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let genesis = [0u8; 32];
    let runtime = bootstrap_runtime(genesis);
    let state = AppState {
        runtime: Arc::new(RwLock::new(runtime)),
    };

    let port: u16 = std::env::var("BIZRA_COGNITION_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7421);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    tracing::info!(%addr, domain = DOMAIN, "bizra-cognition-gateway v0.2 listening");

    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, router(state).into_make_service())
        .await
        .expect("serve");
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — verify JSON contract + principal activation end-to-end.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn new_state() -> AppState {
        AppState {
            runtime: Arc::new(RwLock::new(bootstrap_runtime([0u8; 32]))),
        }
    }

    /// Env-free AppState: constructs the runtime directly via
    /// `fresh_in_memory_runtime` rather than going through `bootstrap_runtime`.
    /// This decouples tests from whatever BIZRA_* env vars the test runner
    /// happens to export — guaranteeing the "no cache attached" state
    /// regardless of the host environment.
    fn new_state_env_free() -> AppState {
        AppState {
            runtime: Arc::new(RwLock::new(fresh_in_memory_runtime([0u8; 32]))),
        }
    }

    /// Build an AppState whose CognitionRuntime was bootstrapped from a
    /// live sovereign_state/ tempdir. Used to exercise Phase 2 durable-read
    /// fall-through without touching process env vars.
    fn new_state_with_sovereign(root: &std::path::Path) -> AppState {
        let rt = CognitionRuntime::from_sovereign_state(root)
            .expect("valid sovereign_state fixture should bootstrap");
        AppState {
            runtime: Arc::new(RwLock::new(rt)),
        }
    }

    fn write_two_entry_fixture(root: &std::path::Path) -> (String, String) {
        use bizra_cognition::sovereign_state::{chain_entry_hash, hex_digest, GENESIS_PREV_HEX};
        use std::fs;
        let receipts = root.join("receipts");
        fs::create_dir_all(&receipts).unwrap();
        let r_a = serde_json::json!({"event": "step_a", "n": 1});
        let r_b = serde_json::json!({"event": "step_b", "n": 2});
        fs::write(
            receipts.join("step_a_2026.json"),
            serde_json::to_vec(&r_a).unwrap(),
        )
        .unwrap();
        fs::write(
            receipts.join("step_b_2026.json"),
            serde_json::to_vec(&r_b).unwrap(),
        )
        .unwrap();
        let h_a = hex_digest(&chain_entry_hash(GENESIS_PREV_HEX, &r_a).unwrap());
        let h_b = hex_digest(&chain_entry_hash(&h_a, &r_b).unwrap());
        let env = serde_json::json!({
            "chain_type": "gateway_phase2_test",
            "node_id": "GW-TEST",
            "timestamp": "2026-01-01T00:00:00Z",
            "receipts": 2,
            "chain": [
                {"file": "step_a_2026.json", "event": "step_a", "hash": h_a, "prev_hash": GENESIS_PREV_HEX},
                {"file": "step_b_2026.json", "event": "step_b", "hash": h_b, "prev_hash": h_a}
            ],
            "head_hash": h_b
        });
        fs::write(
            receipts.join("activation_chain_2026-01-01T00:00:00Z.json"),
            serde_json::to_vec(&env).unwrap(),
        )
        .unwrap();
        (h_a, h_b)
    }

    fn activation_request() -> serde_json::Value {
        serde_json::json!({
            "intent": "activate my dual agentic system",
            "operatorSessionId": "11".repeat(32),
            "currentState": {
                "hash": "22".repeat(32),
                "summary": "Principal not yet activated",
                "metric": 0.0
            },
            "idealState": {
                "hash": "33".repeat(32),
                "summary": "Principal activated, PAT-7 and SAT-5 reachable through Dema",
                "metric": 1.0
            },
            "evidenceHash": "44".repeat(32),
            "qualityScore": 0.98,
            "derivesFromCanonical": true,
            "faceOnly": false
        })
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn empty_chain_returns_zero_head_null_timestamp() {
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .uri("/chain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["head"], "0".repeat(64));
        assert_eq!(v["length"], 0);
        assert!(v["latestTimestamp"].is_null());
        // Phase 2: no snapshot attached → sovereignEntries omitted via skip_serializing_if
        assert!(v.get("sovereignEntries").is_none());
    }

    // ========================================================================
    // Cycle-6 G1 Phase 2 — durable-read fall-through tests
    // niyyah §G1: seal receipt X -> restart gateway -> /chain/X still returns.
    // ========================================================================

    #[tokio::test]
    async fn phase2_chain_summary_exposes_sovereign_counts_when_attached() {
        let td = tempfile::TempDir::new().unwrap();
        write_two_entry_fixture(td.path());
        let state = new_state_with_sovereign(td.path());

        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/chain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // in-memory chain is still empty (no missions submitted this session)
        assert_eq!(v["length"], 0);
        // but the snapshot is exposed
        assert_eq!(v["sovereignEnvelopes"], 1);
        assert_eq!(v["sovereignEntries"], 2);
    }

    #[tokio::test]
    async fn phase2_chain_receipt_fallthrough_returns_durable_entry() {
        let td = tempfile::TempDir::new().unwrap();
        let (hash_a, hash_b) = write_two_entry_fixture(td.path());
        let state = new_state_with_sovereign(td.path());

        // Query by hash_b — the head of the snapshot chain
        let uri = format!("/chain/{}", hash_b);
        let res = router(state.clone())
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::OK,
            "durable receipt should be served from snapshot"
        );
        let body = to_bytes(res.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["id"], hash_b);
        assert_eq!(v["kind"], "step_b");
        assert_eq!(v["prevChain"], hash_a);
        assert_eq!(v["durable"], true);
    }

    #[tokio::test]
    async fn phase2_unknown_hash_with_sovereign_still_returns_404() {
        let td = tempfile::TempDir::new().unwrap();
        write_two_entry_fixture(td.path());
        let state = new_state_with_sovereign(td.path());

        let uri = format!("/chain/{}", "f".repeat(64));
        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn post_mission_activation_permits_and_seals_receipt() {
        // The principal activation end-to-end: intent in, receipt out.
        let state = new_state();
        let body = serde_json::to_vec(&activation_request()).unwrap();

        let res = router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/missions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK, "activation should PERMIT");
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(v["admissibility"]["verdict"], "Permit");
        assert_eq!(v["finalStage"], "Replayability");
        assert_eq!(v["missionId"].as_str().unwrap().len(), 64);
        assert_eq!(v["receiptId"].as_str().unwrap().len(), 64);
        // Cycle-7 G1 (add18501): chainHead is the Manifest receipt appended
        // after the NodeLifecycle receipt, so it is DISTINCT from receiptId.
        assert_ne!(
            v["chainHead"], v["receiptId"],
            "post-G1: Manifest is appended after NodeLifecycle, advancing head"
        );

        // Chain length: 1 mission + 5 gate verdicts + 1 NodeLifecycle + 1 Manifest = 8.
        let chain_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/chain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let chain_body = to_bytes(chain_res.into_body(), 1024).await.unwrap();
        let chain: serde_json::Value = serde_json::from_slice(&chain_body).unwrap();
        assert_eq!(chain["length"], 8);

        let mission_id = v["missionId"].as_str().unwrap();

        let mission_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .uri(format!("/missions/{}", mission_id))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mission_res.status(), StatusCode::OK);
        let mission_body = to_bytes(mission_res.into_body(), 4096).await.unwrap();
        let mission: serde_json::Value = serde_json::from_slice(&mission_body).unwrap();
        assert_eq!(mission["missionId"], mission_id);
        assert_eq!(mission["stage"], "Replayability");
        assert_eq!(mission["rejected"], false);
        assert_eq!(mission["receiptId"], v["receiptId"]);
        // G1: stored chain_head mirrors the submit response's chain_head
        // (the Manifest head), not the NodeLifecycle receiptId.
        assert_eq!(mission["chainHead"], v["chainHead"]);

        let replay_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/missions/{}/replay", mission_id))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay_res.status(), StatusCode::OK);
        let replay_body = to_bytes(replay_res.into_body(), 4096).await.unwrap();
        let replay: serde_json::Value = serde_json::from_slice(&replay_body).unwrap();
        assert_eq!(replay["missionId"], mission_id);
        assert_eq!(replay["replayResult"], "MATCH");
        assert_eq!(replay["replayScope"], "CLAIM_ADMISSIBILITY_ONLY");
        assert_eq!(replay["matchesPrevious"], true);
    }

    #[tokio::test]
    async fn get_mission_preserves_original_chain_head_after_later_submissions() {
        let state = new_state();

        let first_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/missions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&activation_request()).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first_res.status(), StatusCode::OK);
        let first_body = to_bytes(first_res.into_body(), 4096).await.unwrap();
        let first: serde_json::Value = serde_json::from_slice(&first_body).unwrap();

        let mut second_request = activation_request();
        second_request["intent"] = serde_json::json!("activate a second lawful mission");
        let second_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/missions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&second_request).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second_res.status(), StatusCode::OK);

        let mission_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/missions/{}",
                        first["missionId"].as_str().unwrap()
                    ))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mission_res.status(), StatusCode::OK);
        let mission_body = to_bytes(mission_res.into_body(), 4096).await.unwrap();
        let mission: serde_json::Value = serde_json::from_slice(&mission_body).unwrap();
        // G1: the first mission's stored chain_head is immutable — it is the
        // Manifest head sealed at the time of its submission, not the later
        // chain head after a second submission.
        assert_eq!(mission["chainHead"], first["chainHead"]);
    }

    #[tokio::test]
    async fn post_mission_rejects_caller_supplied_timestamp() {
        let mut req_body = activation_request();
        req_body["timestampNs"] = serde_json::json!(123u64);

        let body = serde_json::to_vec(&req_body).unwrap();
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/missions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "TIMESTAMP_RUNTIME_OWNED");
    }

    #[tokio::test]
    async fn post_mission_below_ihsan_floor_rejects_with_422() {
        // A mission with quality_score below IHSAN_FLOOR (0.95) must be rejected
        // structurally — HTTP 422, error.admissibility populated.
        let mut req_body = activation_request();
        req_body["qualityScore"] = serde_json::json!(0.50);

        let body = serde_json::to_vec(&req_body).unwrap();
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/missions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(v["error"]["code"], "ADMISSIBILITY_REJECTED");
        assert_eq!(v["error"]["admissibility"]["verdict"], "Reject");
        assert!(v["error"]["admissibility"]["rejected"].is_object());
    }

    #[tokio::test]
    async fn post_mission_empty_intent_returns_400() {
        let mut req_body = activation_request();
        req_body["intent"] = serde_json::json!("   ");

        let body = serde_json::to_vec(&req_body).unwrap();
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/missions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unknown_receipt_hash_returns_404_structured_error() {
        let nonexistent = "a".repeat(64);
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .uri(format!("/chain/{}", nonexistent))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn malformed_hash_returns_400() {
        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .uri("/chain/not-hex")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    // ─── Cycle-7 G2 live walk — POST /principal/activate ─────────────────────

    fn write_test_identity_anchor(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("credentials.json");
        std::fs::write(
            &path,
            br#"{"node_id":"NODE0","public_key":"0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a","created_at":"2026-04-13T23:54:59Z"}"#,
        )
        .unwrap();
        path
    }

    fn principal_request(anchor_path: &std::path::Path, quality: f64) -> serde_json::Value {
        serde_json::json!({
            "principalName": "Mumo",
            "declaredRole": "node0_principal",
            "qualityScore": quality,
            "identityAnchorPath": anchor_path.to_str().unwrap(),
        })
    }

    #[tokio::test]
    async fn post_principal_activate_permits_and_seals_receipt_and_profile() {
        let td = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td.path());
        let body = serde_json::to_vec(&principal_request(&anchor, 0.98)).unwrap();

        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK, "activation should PERMIT");
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(v["admissibility"]["verdict"], "Permit");
        assert_eq!(v["finalStage"], "Replayability");
        assert_eq!(v["missionId"].as_str().unwrap().len(), 64);
        assert_eq!(v["missionReceiptId"].as_str().unwrap().len(), 64);
        assert_eq!(
            v["principalActivationReceiptId"].as_str().unwrap().len(),
            64
        );
        assert_eq!(v["principalId"].as_str().unwrap().len(), 64);
        assert_eq!(v["profileHash"].as_str().unwrap().len(), 64);
        // Chain head after permit activation is the PrincipalActivationReceipt id.
        assert_eq!(v["chainHead"], v["principalActivationReceiptId"]);
        assert_ne!(
            v["principalActivationReceiptId"], v["missionReceiptId"],
            "PA receipt must be distinct from the NodeLifecycle mission receipt"
        );
    }

    // ─── effective_cache_dir wire-contract tests ─────────────────────────
    //
    // Proves the server's authoritative report of the attached dema_cache
    // dir. The CLI and web face echo this verbatim rather than reading
    // their own env (ZANN_ZERO + CLAIM_MUST_BIND under remote/divergent
    // env scenarios).

    #[tokio::test]
    async fn post_principal_activate_omits_effective_cache_dir_when_no_cache_attached() {
        let td = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td.path());
        let body = serde_json::to_vec(&principal_request(&anchor, 0.98)).unwrap();

        // Use the env-free helper rather than new_state() → guarantees no
        // cache regardless of whether BIZRA_DEMA_CACHE_ROOT is exported in
        // the test runner environment.
        let res = router(new_state_env_free())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(v["admissibility"]["verdict"], "Permit");
        assert!(
            v.get("effectiveCacheDir").is_none(),
            "no cache attached → effectiveCacheDir must be omitted, got {:?}",
            v.get("effectiveCacheDir")
        );
        assert!(
            v.get("cacheWarning").is_none(),
            "no cache attached → no warning either; warning is for write-failure only"
        );
    }

    #[tokio::test]
    async fn post_principal_activate_reports_server_authoritative_cache_dir_when_attached() {
        let td_anchor = tempfile::TempDir::new().unwrap();
        let td_cache = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td_anchor.path());
        // Build the runtime env-free, then attach the tmp cache manually. Mirrors the
        // no-cache test's `new_state_env_free` pattern so BIZRA_DEMA_CACHE_ROOT /
        // BIZRA_SOVEREIGN_STATE_PATH in the test runner environment cannot leak in.
        let state = {
            let mut rt = fresh_in_memory_runtime([0u8; 32]);
            rt.attach_dema_cache(td_cache.path());
            AppState {
                runtime: Arc::new(RwLock::new(rt)),
            }
        };
        let body = serde_json::to_vec(&principal_request(&anchor, 0.98)).unwrap();

        let res = router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(v["admissibility"]["verdict"], "Permit");
        let expected = td_cache.path().join("dema_cache");
        assert_eq!(
            v["effectiveCacheDir"].as_str().unwrap(),
            expected.display().to_string(),
            "effectiveCacheDir must echo the exact PathBuf::join path the runtime attached"
        );
        assert!(
            v.get("cacheWarning").is_none(),
            "write to a fresh tmp dir should succeed → no warning"
        );
        // Evidence: the file actually landed at the reported path.
        assert!(
            expected.join("principal.json").exists(),
            "principal.json must exist at the server-reported effectiveCacheDir"
        );
    }

    #[tokio::test]
    async fn post_principal_activate_grows_chain_by_nine() {
        let td = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td.path());
        let state = new_state();
        let body = serde_json::to_vec(&principal_request(&anchor, 0.98)).unwrap();

        let res = router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // Chain length: 1 envelope + 5 gates + NodeLifecycle + Manifest + PA = 9.
        let chain_res = router(state.clone())
            .oneshot(
                Request::builder()
                    .uri("/chain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let chain_body = to_bytes(chain_res.into_body(), 1024).await.unwrap();
        let chain: serde_json::Value = serde_json::from_slice(&chain_body).unwrap();
        assert_eq!(chain["length"], 9);
    }

    #[tokio::test]
    async fn post_principal_activate_below_ihsan_floor_rejects_with_422() {
        let td = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td.path());
        let body = serde_json::to_vec(&principal_request(&anchor, 0.40)).unwrap();

        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            res.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "quality 0.40 below IHSAN_FLOOR must 422"
        );
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "ADMISSIBILITY_REJECTED");
        assert!(
            v["error"]["message"].as_str().unwrap().contains("REJECTED"),
            "reject message must be honest"
        );
    }

    #[tokio::test]
    async fn post_principal_activate_missing_anchor_returns_400() {
        let body = serde_json::to_vec(&serde_json::json!({
            "principalName": "Mumo",
            "identityAnchorPath": "/tmp/__definitely_missing_node_anchor__",
        }))
        .unwrap();

        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "IDENTITY_ANCHOR_LOAD");
    }

    #[tokio::test]
    async fn post_principal_activate_empty_name_returns_400() {
        let td = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td.path());
        let body = serde_json::to_vec(&serde_json::json!({
            "principalName": "",
            "identityAnchorPath": anchor.to_str().unwrap(),
        }))
        .unwrap();

        let res = router(new_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "EMPTY_PRINCIPAL_NAME");
    }

    // ─── Cycle-7 G4 — /resources endpoint tests ─────────────────────────

    fn new_state_with_dema_cache(root: &std::path::Path) -> AppState {
        let mut rt = bootstrap_runtime([0u8; 32]);
        rt.attach_dema_cache(root);
        AppState {
            runtime: Arc::new(RwLock::new(rt)),
        }
    }

    async fn register_resource(
        app: axum::Router,
        kind: &str,
        id: &str,
        allowlisted: bool,
    ) -> axum::response::Response {
        let body = serde_json::json!({
            "kind": kind,
            "id": id,
            "summary": format!("test {}", id),
            "allowlisted": allowlisted,
        })
        .to_string();
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/resources/register")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn post_resource_register_creates_new_entry() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let app = router(state);
        let res = register_resource(app, "filesystem", "/home/mumo/docs", true).await;
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["outcome"], "created");
        assert_eq!(v["resource"]["kind"], "filesystem");
        assert_eq!(v["resource"]["id"], "/home/mumo/docs");
        assert_eq!(v["resource"]["allowlisted"], true);
    }

    #[tokio::test]
    async fn post_resource_register_same_twice_is_idempotent() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(router(state.clone()), "filesystem", "/a", true).await;
        let res = register_resource(router(state), "filesystem", "/a", true).await;
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["outcome"], "idempotent");
    }

    #[tokio::test]
    async fn post_resource_register_empty_id_returns_400() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let res = register_resource(router(state), "filesystem", "", true).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "INVALID_RESOURCE");
    }

    #[tokio::test]
    async fn get_resources_list_reflects_registered_resources() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(router(state.clone()), "filesystem", "/a", true).await;
        let _ = register_resource(router(state.clone()), "network", "host:80", false).await;

        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/resources/list")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["resources"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn get_resources_urp_groups_by_kind_and_reports_counts() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(router(state.clone()), "filesystem", "/a", true).await;
        let _ = register_resource(router(state.clone()), "filesystem", "/b", false).await;
        let _ = register_resource(router(state.clone()), "network", "host:80", true).await;

        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/resources/urp")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["totalCount"], 3);
        assert_eq!(v["allowlistedCount"], 2);
        let buckets = v["buckets"].as_array().unwrap();
        assert_eq!(buckets.len(), 2);
        // alphabetical: filesystem, network
        assert_eq!(buckets[0]["kind"], "filesystem");
        assert_eq!(buckets[1]["kind"], "network");
    }

    // ─── Cycle-7 G5 — /missions/organize tests ─────────────────────────

    async fn post_organize(
        app: axum::Router,
        path: &str,
        quality: f64,
    ) -> axum::response::Response {
        let body = serde_json::json!({
            "path": path,
            "qualityScore": quality,
        })
        .to_string();
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/missions/organize")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    fn write_g5_fixture(root: &std::path::Path) {
        use std::fs;
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("alpha.txt"), b"hello").unwrap();
        fs::write(root.join("beta.txt"), b"world").unwrap();
        fs::create_dir_all(root.join("subdir")).unwrap();
    }

    #[tokio::test]
    async fn post_organize_allowlisted_path_returns_200_with_sealed_receipt() {
        let td = tempfile::TempDir::new().unwrap();
        let target = td.path().join("target");
        write_g5_fixture(&target);
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(
            router(state.clone()),
            "filesystem",
            &target.to_string_lossy(),
            true,
        )
        .await;

        let res = post_organize(router(state), &target.to_string_lossy(), 0.98).await;
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 16384).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["fileCount"], 2);
        assert_eq!(v["dirCount"], 1);
        assert_eq!(v["entryCount"], 3);
        assert_eq!(v["chainHead"], v["organizeReceiptId"]);
        assert_eq!(v["admissibility"]["verdict"], "Permit");
    }

    #[tokio::test]
    async fn post_organize_non_allowlisted_returns_403() {
        let td = tempfile::TempDir::new().unwrap();
        let target = td.path().join("target");
        write_g5_fixture(&target);
        let state = new_state_with_dema_cache(td.path());
        // NOT registered.
        let res = post_organize(router(state), &target.to_string_lossy(), 0.98).await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "PATH_NOT_ALLOWLISTED");
    }

    #[tokio::test]
    async fn post_organize_missing_path_returns_400_io_error() {
        let td = tempfile::TempDir::new().unwrap();
        let ghost = td.path().join("ghost-dir");
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(
            router(state.clone()),
            "filesystem",
            &ghost.to_string_lossy(),
            true,
        )
        .await;
        let res = post_organize(router(state), &ghost.to_string_lossy(), 0.98).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "ORGANIZE_IO_ERROR");
    }

    #[tokio::test]
    async fn post_organize_below_ihsan_floor_returns_422() {
        let td = tempfile::TempDir::new().unwrap();
        let target = td.path().join("target");
        write_g5_fixture(&target);
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(
            router(state.clone()),
            "filesystem",
            &target.to_string_lossy(),
            true,
        )
        .await;
        let res = post_organize(router(state), &target.to_string_lossy(), 0.40).await;
        assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "ADMISSIBILITY_REJECTED");
    }

    #[tokio::test]
    async fn post_organize_empty_path_returns_400() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let res = post_organize(router(state), "", 0.98).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "EMPTY_PATH");
    }

    // ─── Cycle-7 G6 — /poi tests ──────────────────────────────────────

    #[tokio::test]
    async fn get_poi_ledger_empty_on_fresh_runtime() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/poi/ledger")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["entries"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn get_poi_summary_reports_zero_when_empty() {
        let td = tempfile::TempDir::new().unwrap();
        let state = new_state_with_dema_cache(td.path());
        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/poi/summary")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["totalEntries"], 0);
        assert_eq!(v["totalImpact"], 0.0);
        assert_eq!(v["byKind"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn get_poi_ledger_contains_entry_after_organize() {
        let td = tempfile::TempDir::new().unwrap();
        let target = td.path().join("target");
        write_g5_fixture(&target);
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(
            router(state.clone()),
            "filesystem",
            &target.to_string_lossy(),
            true,
        )
        .await;
        let _ = post_organize(router(state.clone()), &target.to_string_lossy(), 0.98).await;

        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/poi/ledger")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 16384).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let entries = v["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["receiptKindName"], "MissionExecuted");
        assert_eq!(entries[0]["entryCount"], 3);
    }

    #[tokio::test]
    async fn get_poi_summary_aggregates_per_kind() {
        let td = tempfile::TempDir::new().unwrap();
        let target = td.path().join("target");
        write_g5_fixture(&target);
        let state = new_state_with_dema_cache(td.path());
        let _ = register_resource(
            router(state.clone()),
            "filesystem",
            &target.to_string_lossy(),
            true,
        )
        .await;
        let _ = post_organize(router(state.clone()), &target.to_string_lossy(), 0.98).await;
        // 2nd organize to get a 2nd MissionExecuted entry
        let _ = post_organize(router(state.clone()), &target.to_string_lossy(), 0.98).await;

        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/poi/summary")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["totalEntries"], 2);
        let by_kind = v["byKind"].as_array().unwrap();
        assert_eq!(by_kind.len(), 1);
        assert_eq!(by_kind[0]["kind"], "MissionExecuted");
        assert_eq!(by_kind[0]["count"], 2);
    }

    #[tokio::test]
    async fn get_poi_summary_splits_activation_and_execution_buckets() {
        let td = tempfile::TempDir::new().unwrap();
        let target = td.path().join("target");
        write_g5_fixture(&target);
        // Seed identity anchor so activation can run.
        std::fs::create_dir_all(td.path().join("identity")).unwrap();
        std::fs::write(
            td.path().join("identity/credentials.json"),
            br#"{"node_id":"NODE0","public_key":"0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a","created_at":"2026-04-18T08:00:00Z"}"#,
        )
        .unwrap();
        let state = new_state_with_dema_cache(td.path());

        // 1) activate
        let anchor_path = td.path().join("identity/credentials.json");
        let activate_body = serde_json::json!({
            "principalName": "Mumo",
            "declaredRole": "node0_principal",
            "qualityScore": 0.98,
            "identityAnchorPath": anchor_path.to_string_lossy(),
        })
        .to_string();
        let _ = router(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(activate_body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // 2) register + organize
        let _ = register_resource(
            router(state.clone()),
            "filesystem",
            &target.to_string_lossy(),
            true,
        )
        .await;
        let _ = post_organize(router(state.clone()), &target.to_string_lossy(), 0.98).await;

        // 3) summary
        let res = router(state)
            .oneshot(
                Request::builder()
                    .uri("/poi/summary")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(res.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["totalEntries"], 2);
        let by_kind = v["byKind"].as_array().unwrap();
        assert_eq!(by_kind.len(), 2);
        let names: Vec<&str> = by_kind
            .iter()
            .map(|b| b["kind"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"PrincipalActivation"));
        assert!(names.contains(&"MissionExecuted"));
    }

    #[tokio::test]
    async fn authoritative_receipt_store_survives_gateway_rebootstrap() {
        let td_store = tempfile::TempDir::new().unwrap();
        let td_anchor = tempfile::TempDir::new().unwrap();
        let anchor = write_test_identity_anchor(td_anchor.path());
        let genesis = [0u8; 32];

        let mut rt1 = fresh_in_memory_runtime(genesis);
        rt1.bootstrap_authoritative_receipt_store_at(td_store.path(), genesis)
            .expect("first bootstrap");
        let state1 = AppState {
            runtime: Arc::new(RwLock::new(rt1)),
        };

        let body = serde_json::to_vec(&principal_request(&anchor, 0.98)).unwrap();
        let activate = router(state1.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/principal/activate")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(activate.status(), StatusCode::OK);

        let chain_res = router(state1.clone())
            .oneshot(
                Request::builder()
                    .uri("/chain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let chain_body = to_bytes(chain_res.into_body(), 1024).await.unwrap();
        let chain_after: serde_json::Value = serde_json::from_slice(&chain_body).unwrap();
        let len_after = chain_after["length"].as_u64().unwrap();
        let head_hex = chain_after["head"].as_str().unwrap().to_string();
        assert_eq!(len_after, 9);

        let receipt_res = router(state1.clone())
            .oneshot(
                Request::builder()
                    .uri(format!("/chain/{head_hex}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(receipt_res.status(), StatusCode::OK);

        drop(state1);

        assert!(
            td_store
                .path()
                .join("chain_snapshot.json")
                .exists(),
            "authoritative chain snapshot must be written after mission"
        );

        let mut rt2 = fresh_in_memory_runtime(genesis);
        rt2.bootstrap_authoritative_receipt_store_at(td_store.path(), genesis)
            .expect("second bootstrap");
        assert_eq!(rt2.chain.len(), len_after as usize);
        assert_eq!(hex32(&rt2.chain.head()), head_hex);
        assert!(
            rt2.chain
                .fetch_payload_bytes(&rt2.chain.head())
                .expect("fetch head payload")
                .is_some()
        );

        let state2 = AppState {
            runtime: Arc::new(RwLock::new(rt2)),
        };
        let chain_restart = router(state2)
            .oneshot(
                Request::builder()
                    .uri("/chain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let restart_body = to_bytes(chain_restart.into_body(), 1024).await.unwrap();
        let chain_restart: serde_json::Value = serde_json::from_slice(&restart_body).unwrap();
        assert_eq!(chain_restart["length"], len_after);
        assert_eq!(chain_restart["head"].as_str().unwrap(), head_hex);
    }
}
