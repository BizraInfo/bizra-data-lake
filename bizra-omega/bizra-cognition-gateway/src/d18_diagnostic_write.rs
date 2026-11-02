//! D18 gateway-native one-shot diagnostic write actuator.
//!
//! Design envelope sha256:
//! 07a430fc41b2c54dde97be900bc50681fe9c1a626edf1965bdb08752d4fcffe6
//!
//! Default: DISABLED. Arming requires BOTH:
//! - `BIZRA_D18_DIAGNOSTIC_WRITE_ARMED=1`
//! - `BIZRA_D18_DIAGNOSTIC_LEASE_PATH` → pinned lease JSON
//!
//! transport_class = loopback_TCP
//! target_class = A_QUARANTINE_UNDER_EXISTING_RW (under BIZRA_DEMA_CACHE_ROOT)

use axum::extract::ConnectInfo;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

const DOMAIN: &str = "bizra-cognition-gateway-v1";
const TRANSPORT_CLASS: &str = "loopback_TCP";
const TARGET_CLASS: &str = "A_QUARANTINE_UNDER_EXISTING_RW";
const HTTP_PATH: &str = "/diagnostic/write-correlation";
const MAX_BODY_DEFAULT: usize = 4096;

static WRITE_SPENT: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Serialize)]
pub struct StatusBody {
    schema: &'static str,
    status: &'static str,
    armed: bool,
    spent: bool,
    transport_class: &'static str,
    target_class: &'static str,
    http_path: &'static str,
    unit: &'static str,
    writer_pid: u32,
    domain: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}

#[derive(Debug, Deserialize)]
struct Lease {
    schema: String,
    experiment_id: String,
    nonce_sha256: String,
    sandbox_root: String,
    target_abspath: String,
    #[serde(default = "default_max_body")]
    max_body_bytes: usize,
    seal_id: String,
}

fn default_max_body() -> usize {
    MAX_BODY_DEFAULT
}

#[derive(Debug, Deserialize)]
struct WriteRequest {
    experiment_id: String,
    nonce_sha256: String,
    target_abspath: String,
}

#[derive(Debug, Serialize)]
struct WriteOk {
    schema: &'static str,
    ok: bool,
    status: &'static str,
    experiment_id: String,
    nonce_sha256: String,
    target_abspath: String,
    artifact_sha256: String,
    bytes: usize,
    writer_pid: u32,
    request_id: String,
    seal_id: String,
    transport_class: &'static str,
    target_class: &'static str,
    domain: &'static str,
}

#[derive(Debug, Serialize)]
struct RefuseBody {
    schema: &'static str,
    ok: bool,
    status: &'static str,
    code: &'static str,
    reason: String,
    transport_class: &'static str,
    target_class: &'static str,
    writer_pid: u32,
    domain: &'static str,
}

fn env_armed() -> bool {
    matches!(
        std::env::var("BIZRA_D18_DIAGNOSTIC_WRITE_ARMED")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn lease_path_env() -> Option<PathBuf> {
    std::env::var_os("BIZRA_D18_DIAGNOSTIC_LEASE_PATH").map(PathBuf::from)
}

fn cache_root_env() -> Option<PathBuf> {
    std::env::var_os("BIZRA_DEMA_CACHE_ROOT").map(PathBuf::from)
}

fn status_now(status: &'static str, reason: Option<&'static str>) -> StatusBody {
    StatusBody {
        schema: "bizra.dema.d18-gateway-diagnostic-write.status.v1",
        status,
        armed: env_armed() && lease_path_env().is_some(),
        spent: WRITE_SPENT.load(Ordering::SeqCst),
        transport_class: TRANSPORT_CLASS,
        target_class: TARGET_CLASS,
        http_path: HTTP_PATH,
        unit: "bizra-node0-gateway.service",
        writer_pid: std::process::id(),
        domain: DOMAIN,
        reason,
    }
}

fn refuse(code: &'static str, reason: impl Into<String>, http: StatusCode) -> Response {
    (
        http,
        Json(RefuseBody {
            schema: "bizra.dema.d18-gateway-diagnostic-write.refuse.v1",
            ok: false,
            status: "REFUSED",
            code,
            reason: reason.into(),
            transport_class: TRANSPORT_CLASS,
            target_class: TARGET_CLASS,
            writer_pid: std::process::id(),
            domain: DOMAIN,
        }),
    )
        .into_response()
}

fn is_loopback(addr: SocketAddr) -> bool {
    match addr.ip() {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn validate_target_class_a(lease: &Lease) -> Result<PathBuf, &'static str> {
    let cache_root = cache_root_env().ok_or("BIZRA_DEMA_CACHE_ROOT_UNSET")?;
    let cache_canon = fs::canonicalize(&cache_root).map_err(|_| "CACHE_ROOT_NOT_CANONICAL")?;

    if lease.sandbox_root.contains("..") || lease.target_abspath.contains("..") {
        return Err("PATH_TRAVERSAL_REFUSED");
    }

    let sandbox_canon =
        fs::canonicalize(&lease.sandbox_root).map_err(|_| "SANDBOX_ROOT_MISSING")?;
    if !sandbox_canon.starts_with(&cache_canon) {
        return Err("SANDBOX_OUTSIDE_CACHE_ROOT");
    }
    let sandbox_name = sandbox_canon
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("SANDBOX_NAME_INVALID")?;
    if !sandbox_name.starts_with("d18-diag-sandbox-") {
        return Err("SANDBOX_NAME_PREFIX_REFUSED");
    }

    let target = Path::new(&lease.target_abspath);
    let file_name = target.file_name().ok_or("TARGET_NAME_INVALID")?;
    let parent = target.parent().ok_or("TARGET_PARENT_INVALID")?;
    let parent_canon = fs::canonicalize(parent).map_err(|_| "TARGET_PARENT_MISSING")?;
    if parent_canon != sandbox_canon {
        return Err("TARGET_NOT_EXACT_SANDBOX_CHILD");
    }
    let expected = sandbox_canon.join(file_name);
    if expected != PathBuf::from(&lease.target_abspath) {
        return Err("TARGET_PATH_MISMATCH_SANDBOX");
    }
    if expected.exists() {
        return Err("TARGET_ALREADY_EXISTS");
    }

    let target_str = expected.to_string_lossy();
    for needle in [
        "/.dema/",
        "/campaigns/",
        "campaign-contract",
        "identity-root",
        "authoritative_receipt_store",
        "/keys/",
        "economics",
    ] {
        if target_str.contains(needle) {
            return Err("FORBIDDEN_TARGET_SURFACE");
        }
    }

    Ok(expected)
}

fn load_lease(path: &Path) -> Result<Lease, String> {
    let bytes = fs::read(path).map_err(|e| format!("LEASE_READ_FAILED:{e}"))?;
    if bytes.len() > 64 * 1024 {
        return Err("LEASE_TOO_LARGE".into());
    }
    let lease: Lease =
        serde_json::from_slice(&bytes).map_err(|e| format!("LEASE_JSON_INVALID:{e}"))?;
    if lease.schema != "bizra.dema.d18-gateway-diagnostic-lease.v1" {
        return Err("LEASE_SCHEMA_REFUSED".into());
    }
    if lease.nonce_sha256.len() != 64 || !lease.nonce_sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("LEASE_NONCE_INVALID".into());
    }
    if lease.experiment_id.is_empty() || lease.seal_id.is_empty() {
        return Err("LEASE_FIELDS_EMPTY".into());
    }
    Ok(lease)
}

/// GET /diagnostic/write-correlation — status only; never writes.
pub async fn diagnostic_write_status() -> impl IntoResponse {
    if WRITE_SPENT.load(Ordering::SeqCst) {
        return (StatusCode::GONE, Json(status_now("SPENT", Some("lease_spent")))).into_response();
    }
    if !env_armed() {
        return (
            StatusCode::FORBIDDEN,
            Json(status_now("DISABLED", Some("armed_env_not_set"))),
        )
            .into_response();
    }
    if lease_path_env().is_none() {
        return (
            StatusCode::FORBIDDEN,
            Json(status_now("DISABLED", Some("lease_path_unset"))),
        )
            .into_response();
    }
    (
        StatusCode::OK,
        Json(status_now("ARMED", Some("awaiting_one_shot_post"))),
    )
        .into_response()
}

/// POST /diagnostic/write-correlation — one-shot write when armed; else refuse.
pub async fn diagnostic_write_post(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    body: axum::body::Bytes,
) -> Response {
    if !is_loopback(addr) {
        return refuse(
            "NON_LOOPBACK_PEER",
            format!("peer {addr} is not loopback"),
            StatusCode::FORBIDDEN,
        );
    }
    if !env_armed() {
        return refuse(
            "DISABLED",
            "BIZRA_D18_DIAGNOSTIC_WRITE_ARMED not set; actuator disabled by default",
            StatusCode::FORBIDDEN,
        );
    }
    let Some(lease_path) = lease_path_env() else {
        return refuse(
            "DISABLED",
            "BIZRA_D18_DIAGNOSTIC_LEASE_PATH unset",
            StatusCode::FORBIDDEN,
        );
    };
    if WRITE_SPENT.load(Ordering::SeqCst) {
        return refuse("SPENT", "one-shot lease already spent", StatusCode::GONE);
    }
    if body.len() > MAX_BODY_DEFAULT {
        return refuse(
            "BODY_TOO_LARGE",
            format!("body {} exceeds {MAX_BODY_DEFAULT}", body.len()),
            StatusCode::PAYLOAD_TOO_LARGE,
        );
    }

    let req: WriteRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return refuse(
                "REQUEST_JSON_INVALID",
                e.to_string(),
                StatusCode::BAD_REQUEST,
            )
        }
    };

    let lease = match load_lease(&lease_path) {
        Ok(l) => l,
        Err(e) => return refuse("LEASE_LOAD_FAILED", e, StatusCode::FORBIDDEN),
    };

    if req.experiment_id != lease.experiment_id {
        return refuse(
            "EXPERIMENT_ID_MISMATCH",
            "request experiment_id != lease",
            StatusCode::FORBIDDEN,
        );
    }
    if req.nonce_sha256 != lease.nonce_sha256 {
        return refuse(
            "NONCE_MISMATCH",
            "request nonce_sha256 != lease",
            StatusCode::FORBIDDEN,
        );
    }
    if req.target_abspath != lease.target_abspath {
        return refuse(
            "TARGET_MISMATCH",
            "request target_abspath != lease; arbitrary paths refused",
            StatusCode::FORBIDDEN,
        );
    }
    if body.len() > lease.max_body_bytes {
        return refuse(
            "BODY_EXCEEDS_LEASE",
            format!(
                "body {} exceeds lease max {}",
                body.len(),
                lease.max_body_bytes
            ),
            StatusCode::PAYLOAD_TOO_LARGE,
        );
    }

    let target_path = match validate_target_class_a(&lease) {
        Ok(v) => v,
        Err(code) => return refuse(code, code, StatusCode::FORBIDDEN),
    };

    if WRITE_SPENT
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return refuse("SPENT", "one-shot race: already spent", StatusCode::GONE);
    }

    let request_id = format!(
        "{:016x}{:08x}",
        std::process::id() as u64,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    );

    let written_at_utc = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .to_string();

    let artifact = serde_json::json!({
        "schema": "bizra.dema.d18-gateway-diagnostic-write.artifact.v1",
        "experiment_id": lease.experiment_id,
        "nonce_sha256": lease.nonce_sha256,
        "seal_id": lease.seal_id,
        "target_abspath": lease.target_abspath,
        "writer_pid": std::process::id(),
        "unit": "bizra-node0-gateway.service",
        "transport_class": TRANSPORT_CLASS,
        "target_class": TARGET_CLASS,
        "request_id": request_id,
        "written_at_unix": written_at_utc,
        "domain": DOMAIN,
    });
    let bytes = match serde_json::to_vec_pretty(&artifact) {
        Ok(mut v) => {
            v.push(b'\n');
            v
        }
        Err(e) => {
            return refuse(
                "ARTIFACT_SERIALIZE_FAILED",
                e.to_string(),
                StatusCode::INTERNAL_SERVER_ERROR,
            )
        }
    };
    if bytes.len() > lease.max_body_bytes {
        return refuse(
            "ARTIFACT_TOO_LARGE",
            format!("artifact {} exceeds lease", bytes.len()),
            StatusCode::PAYLOAD_TOO_LARGE,
        );
    }

    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target_path)
    {
        Ok(f) => f,
        Err(e) => {
            return refuse("OPEN_EXCL_FAILED", format!("{e}"), StatusCode::CONFLICT);
        }
    };
    if let Err(e) = file.write_all(&bytes) {
        let _ = fs::remove_file(&target_path);
        return refuse(
            "WRITE_FAILED",
            format!("{e}"),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }
    let _ = file.sync_all();

    let artifact_sha256 = sha256_hex(&bytes);
    let spent_marker = PathBuf::from(format!("{}.spent", lease_path.display()));
    let _ = File::create(&spent_marker).and_then(|mut f| {
        writeln!(
            f,
            "{{\"spent\":true,\"artifact_sha256\":\"{artifact_sha256}\",\"request_id\":\"{request_id}\"}}"
        )
    });

    (
        StatusCode::CREATED,
        Json(WriteOk {
            schema: "bizra.dema.d18-gateway-diagnostic-write.result.v1",
            ok: true,
            status: "GATEWAY_ONE_SHOT_SANDBOX_WRITE_CORRELATED",
            experiment_id: lease.experiment_id,
            nonce_sha256: lease.nonce_sha256,
            target_abspath: lease.target_abspath,
            artifact_sha256,
            bytes: bytes.len(),
            writer_pid: std::process::id(),
            request_id,
            seal_id: lease.seal_id,
            transport_class: TRANSPORT_CLASS,
            target_class: TARGET_CLASS,
            domain: DOMAIN,
        }),
    )
        .into_response()
}
