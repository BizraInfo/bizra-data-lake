// bizra-cognition-gateway
//
// Read-only HTTP projection of bizra-cognition state for the Dema Console UI.
// NO_SHADOW_STATE: this gateway never owns truth. It holds a ReceiptChain and
// returns its state verbatim. Writes must enter through bizra-cognition proper.
//
// v0.1 scope: health + GET /chain + GET /chain/:hash. Chain starts empty with
// a zero genesis hash until a real runtime is attached.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use bizra_cognition::receipts::{
    InMemoryPayloadStore, ReceiptChain, ReceiptKind,
};
use serde::Serialize;
use tokio::sync::RwLock;

const DOMAIN: &str = "bizra-cognition-gateway-v1";

#[derive(Clone)]
struct AppState {
    chain: Arc<RwLock<ReceiptChain>>,
}

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
    // Nullable: header-only projection cannot recover payload timestamps for
    // most kinds without decoding. See NO_SHADOW_STATE.
    timestamp: Option<u64>,
    #[serde(rename = "prevChain")]
    prev_chain: String,
    #[serde(rename = "payloadHash")]
    payload_hash: String,
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
}

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

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok", domain: DOMAIN })
}

async fn get_chain(State(state): State<AppState>) -> Json<ReceiptChainHeadDto> {
    let chain = state.chain.read().await;
    Json(ReceiptChainHeadDto {
        head: hex32(&chain.head()),
        length: chain.len(),
        latest_timestamp: chain.latest_timestamp(),
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
                },
            }),
        )
    })?;

    let chain = state.chain.read().await;
    for record in chain.records() {
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
            },
        }),
    ))
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/chain", get(get_chain))
        .route("/chain/:hash", get(get_chain_receipt))
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
    let store = Box::new(InMemoryPayloadStore::new());
    let chain = ReceiptChain::new(genesis, store);
    let state = AppState { chain: Arc::new(RwLock::new(chain)) };

    let port: u16 = std::env::var("BIZRA_COGNITION_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7421);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    tracing::info!(%addr, domain = DOMAIN, "bizra-cognition-gateway listening");

    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, router(state).into_make_service())
        .await
        .expect("serve");
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — verify JSON contract against lib/dema/types.ts.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn new_state() -> AppState {
        let store = Box::new(InMemoryPayloadStore::new());
        let chain = ReceiptChain::new([0u8; 32], store);
        AppState { chain: Arc::new(RwLock::new(chain)) }
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = router(new_state());
        let res = app
            .oneshot(Request::builder().uri("/health").body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = to_bytes(res.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["domain"], DOMAIN);
    }

    #[tokio::test]
    async fn empty_chain_returns_zero_head_null_timestamp() {
        let app = router(new_state());
        let res = app
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
    async fn unknown_receipt_hash_returns_404_structured_error() {
        let app = router(new_state());
        let nonexistent = "a".repeat(64);
        let res = app
            .oneshot(
                Request::builder()
                    .uri(format!("/chain/{}", nonexistent))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(res.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "RECEIPT_NOT_FOUND");
        assert_eq!(v["error"]["domain"], DOMAIN);
    }

    #[tokio::test]
    async fn malformed_hash_returns_400() {
        let app = router(new_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/chain/not-hex")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(res.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "INVALID_HASH");
    }
}
