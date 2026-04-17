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
    AdmissibilityClaim, AdmissibilityResult, Verdict,
};
use bizra_cognition::mission_freeze_v1::{
    MissionEnvelope, Originator, StateSnapshot,
};
use bizra_cognition::receipts::{
    Blake3Hash, InMemoryPayloadStore, ReceiptChain, ReceiptKind,
};
use bizra_cognition::runtime::{CognitionRuntime, MissionRuntimeError};
use bizra_cognition::thought_graph::{AgentCtx, ThoughtGraph};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

const DOMAIN: &str = "bizra-cognition-gateway-v1";
const DEFAULT_QUALITY_SCORE: f64 = 0.98;

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
    #[serde(default)]
    hash: Option<String>,
    summary: String,
    metric: f64,
}

#[derive(Deserialize)]
struct SubmitMissionRequest {
    intent: String,
    #[serde(rename = "currentState")]
    current_state: StateSnapshotDto,
    #[serde(rename = "idealState")]
    ideal_state: StateSnapshotDto,
    #[serde(default)]
    originator: Option<String>,
    #[serde(rename = "qualityScore", default)]
    quality_score: Option<f64>,
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

fn stage_name(s: bizra_cognition::mission_freeze_v1::MissionStage) -> &'static str {
    use bizra_cognition::mission_freeze_v1::MissionStage;
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

fn state_snapshot_from_dto(dto: StateSnapshotDto, default_hash: u8) -> StateSnapshot {
    let hash = dto
        .hash
        .as_deref()
        .and_then(parse_hex32)
        .unwrap_or([default_hash; 32]);
    StateSnapshot {
        hash,
        summary: dto.summary,
        metric: dto.metric,
    }
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

fn bootstrap_runtime(genesis: Blake3Hash) -> CognitionRuntime {
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

    // v0.2: gateway does not yet carry operator session identity end-to-end.
    // All missions are attributed to Originator::System until auth-session
    // propagation is added (future arc). The request's "originator" hint is
    // retained but not authoritative at this layer.
    let _originator_hint = req.originator.as_deref();
    let originator = Originator::System;
    let quality_score = req.quality_score.unwrap_or(DEFAULT_QUALITY_SCORE);
    let ts_ns = now_ns()?;

    let current = state_snapshot_from_dto(req.current_state, 0x11);
    let ideal = state_snapshot_from_dto(req.ideal_state, 0x22);

    let mut envelope = MissionEnvelope::from_intent(
        req.intent.clone(),
        current,
        ideal,
        originator,
        ts_ns,
    );
    let claim_id = envelope.extract_claim_id();
    envelope.advance_stage(); // S2 -> S3 (matches submit_mission's expectation)

    let claim = AdmissibilityClaim {
        claim_id,
        has_evidence: true,
        evidence_hash: Some(envelope.mission_id),
        economic_pattern: None,
        state_mutation: None,
        quality_score,
        timestamp_ns: ts_ns,
    };

    // Reset envelope stage so submit_mission can advance it correctly.
    // from_intent() returns S2 Mission; submit_mission advances to S3 via its own
    // advance_stage call. We must pass the envelope at stage=Mission.
    envelope.stage = bizra_cognition::mission_freeze_v1::MissionStage::Mission;

    let mut rt = state.runtime.write().await;
    match rt.submit_mission(envelope, claim) {
        Ok(mission_id) => {
            let record = rt.mission_by_id(&mission_id).ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorBody {
                            code: "POST_SUBMIT_LOOKUP_FAILED",
                            message: "mission submitted but not retrievable".into(),
                            domain: DOMAIN,
                            admissibility: None,
                        },
                    }),
                )
            })?;

            Ok(Json(SubmitMissionResponse {
                mission_id: hex32(&mission_id),
                admissibility: admissibility_to_dto(&record.admissibility),
                receipt_id: hex32(&record.final_receipt.receipt_id),
                final_stage: stage_name(record.envelope.stage),
                chain_head: hex32(&rt.chain.head()),
            }))
        }
        Err(MissionRuntimeError::Rejected(result)) => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: ErrorBody {
                    code: "ADMISSIBILITY_REJECTED",
                    message: "mission rejected by admissibility chain".into(),
                    domain: DOMAIN,
                    admissibility: Some(admissibility_to_dto(&result)),
                },
            }),
        )),
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

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/chain", get(get_chain))
        .route("/chain/:hash", get(get_chain_receipt))
        .route("/mission", post(post_mission))
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
            "currentState": {
                "summary": "Principal not yet activated",
                "metric": 0.0
            },
            "idealState": {
                "summary": "Principal activated, PAT-7 and SAT-5 reachable through Dema",
                "metric": 1.0
            },
            "originator": "Operator",
            "qualityScore": 0.98
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
                    .uri("/mission")
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
                    .uri("/mission")
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
                    .uri("/mission")
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
