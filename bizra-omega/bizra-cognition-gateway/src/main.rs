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
use bizra_cognition::receipts::{
    Blake3Hash, InMemoryPayloadStore, ReceiptChain, ReceiptKind,
};
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
}

#[derive(Serialize)]
struct ReceiptDto {
    id: String,
    kind: &'static str,
    timestamp: Option<u64>,
    #[serde(rename = "prevChain")]
    prev_chain: String,
    #[serde(rename = "payloadHash")]
    payload_hash: String,
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
    Ok(StateSnapshot { hash, summary: dto.summary, metric: dto.metric })
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

fn bootstrap_runtime(genesis: Blake3Hash) -> CognitionRuntime {
    // Cycle-6 G1 Phase 1 — try sovereign_state bootstrap if BIZRA_SOVEREIGN_STATE_PATH is set.
    //   - env unset       → in-memory bootstrap (dev mode, preserves Cycle-5 behavior)
    //   - env set, path missing → warn, fall back to in-memory
    //   - env set, path present, load OK → attach snapshot, serve durable-read
    //   - env set, path present, load FAILED → fail-closed startup (exit 1)
    if let Ok(path_str) = std::env::var("BIZRA_SOVEREIGN_STATE_PATH") {
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
                    return rt;
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
        }
    }

    // Default in-memory bootstrap (dev / no env var / missing path).
    // Empty-graph bootstrap. submit_mission only touches self.chain + self.missions
    // + admissibility evaluation — no graph traversal. This is the minimum viable
    // runtime for the G3 activation surface. Future arcs will attach PAT-7/SAT-5
    // factories via configure_cognition::default_pat7_sat5_config.
    let graph = ThoughtGraph::from_parts(
        HashMap::new(),
        Vec::new(),
        HashMap::new(),
        genesis,
    );
    let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
    let ctx = AgentCtx { receipt_chain: genesis };
    CognitionRuntime::new(graph, chain, ctx)
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok", domain: DOMAIN })
}

async fn get_chain(State(state): State<AppState>) -> Json<ReceiptChainHeadDto> {
    let rt = state.runtime.read().await;
    Json(ReceiptChainHeadDto {
        head: hex32(&rt.chain.head()),
        length: rt.chain.len(),
        latest_timestamp: rt.chain.latest_timestamp(),
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
    for record in rt.chain.records() {
        if record.hash == target {
            return Ok(Json(ReceiptDto {
                id: hex32(&record.hash),
                kind: kind_name(record.kind),
                timestamp: None,
                prev_chain: hex32(&record.prev),
                payload_hash: hex32(&record.hash),
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
                    message: "timestampNs is runtime-owned and cannot be supplied by the caller".into(),
                    domain: DOMAIN,
                    admissibility: None,
                },
            }),
        ));
    }
    let ts_ns = now_ns()?;

    let current = state_snapshot_from_dto(req.current_state, "currentState.hash")?;
    let ideal = state_snapshot_from_dto(req.ideal_state, "idealState.hash")?;

    let envelope = MissionEnvelope::from_intent(
        req.intent.clone(),
        current,
        ideal,
        originator,
        ts_ns,
    );
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
                            message: "permit record missing receipt_id — runtime invariant violated".into(),
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
    let state = AppState { runtime: Arc::new(RwLock::new(runtime)) };

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
            .oneshot(Request::builder().uri("/health").body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn empty_chain_returns_zero_head_null_timestamp() {
        let res = router(new_state())
            .oneshot(Request::builder().uri("/chain").body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["head"], "0".repeat(64));
        assert_eq!(v["length"], 0);
        assert!(v["latestTimestamp"].is_null());
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
        assert_eq!(v["chainHead"], v["receiptId"]);

        // Chain should now show 7: 1 mission + 5 gate verdicts + 1 final receipt.
        let chain_res = router(state.clone())
            .oneshot(Request::builder().uri("/chain").body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        let chain_body = to_bytes(chain_res.into_body(), 1024).await.unwrap();
        let chain: serde_json::Value = serde_json::from_slice(&chain_body).unwrap();
        assert_eq!(chain["length"], 7);

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
        assert_eq!(mission["chainHead"], v["receiptId"]);

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
                    .uri(format!("/missions/{}", first["missionId"].as_str().unwrap()))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mission_res.status(), StatusCode::OK);
        let mission_body = to_bytes(mission_res.into_body(), 4096).await.unwrap();
        let mission: serde_json::Value = serde_json::from_slice(&mission_body).unwrap();
        assert_eq!(mission["chainHead"], first["receiptId"]);
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
}
