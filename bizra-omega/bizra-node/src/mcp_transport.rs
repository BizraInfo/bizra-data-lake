// bizra-node/src/mcp_transport.rs — MCP JSON-RPC 2.0 Transport Layer
//
// Wraps the existing tab-delimited protocol in JSON-RPC 2.0 framing over TCP.
// This is an *additional* transport — the stdio protocol remains unchanged.
//
// Wire format:
//   Client sends: {"jsonrpc":"2.0","method":"ping","id":1}\n
//   Server sends: {"jsonrpc":"2.0","result":{"status":"pong"},"id":1}\n
//
// Standing on: Lamport (1978), Cerf & Kahn (1974), Anthropic MCP (2024)

use crate::protocol::{Command, ErrorCode, Response};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;

/// Maximum JSON-RPC request line length (64 KB).
/// Prevents OOM from unbounded read_line (CVE-SEC-1).
pub const MAX_LINE_LENGTH: usize = 65_536;

/// Maximum number of requests in a single JSON-RPC batch.
pub const MAX_BATCH_SIZE: usize = 100;

// -- JSON-RPC 2.0 Types --

/// JSON-RPC 2.0 request (deserialized from wire).
#[derive(Debug, Clone)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Value,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

/// TCP transport configuration.
#[derive(Debug, Clone)]
pub struct McpTransportConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: u16,
    pub read_timeout_ms: u64,
}

impl Default for McpTransportConfig {
    fn default() -> Self {
        McpTransportConfig {
            host: "127.0.0.1".to_string(),
            port: 9741,
            max_connections: 16,
            read_timeout_ms: 30_000,
        }
    }
}

// -- Error Code Mapping --

/// Map protocol ErrorCode to JSON-RPC 2.0 error code.
fn error_code_to_rpc(code: &ErrorCode) -> i32 {
    match code {
        ErrorCode::BadCommand => -32601,    // Method not found
        ErrorCode::MissingArg => -32602,    // Invalid params
        ErrorCode::ParseError => -32700,    // Parse error
        ErrorCode::InvalidArg => -32602,    // Invalid params
        ErrorCode::InternalError => -32603, // Internal error
    }
}

// -- Param Extraction Helpers --

/// Extract a required string parameter.
pub fn require_str(params: &Value, key: &str) -> Result<String, String> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Missing required param: {}", key))
}

/// Extract an optional string parameter with a default.
pub fn optional_str(params: &Value, key: &str, default: &str) -> String {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

/// Extract an optional u64 parameter with a default.
pub fn optional_u64(params: &Value, key: &str, default: u64) -> u64 {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

/// Extract an optional u32 parameter with a default.
pub fn optional_u32(params: &Value, key: &str, default: u32) -> u32 {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(default)
}

/// Extract a required u16 parameter with bounds checking.
pub fn require_u16(params: &Value, key: &str) -> Result<u16, String> {
    let v = params
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("Missing required param: {}", key))?;
    u16::try_from(v).map_err(|_| format!("Param '{}' value {} exceeds u16 range", key, v))
}

/// Extract a required JSON value parameter (for nested objects).
fn require_value(params: &Value, key: &str) -> Result<Value, String> {
    params
        .get(key)
        .cloned()
        .ok_or_else(|| format!("Missing required param: {}", key))
}

/// Current time in seconds since UNIX epoch.
pub fn current_time_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// -- Parse: JSON-RPC line -> (Command, id) --

/// Parse a JSON-RPC 2.0 request line into a Command and request id.
///
/// Returns `Err(json_string)` with a pre-formatted JSON-RPC error response
/// if the input is invalid.
pub fn parse_jsonrpc(line: &str) -> Result<(Command, Value), String> {
    let value: Value = serde_json::from_str(line)
        .map_err(|_| make_error_response(-32700, "Parse error", None, Value::Null))?;
    let id = value.get("id").cloned().unwrap_or(Value::Null);

    match value.get("jsonrpc").and_then(|v| v.as_str()) {
        Some("2.0") => {}
        _ => {
            return Err(make_error_response(
                -32600,
                "Invalid Request: missing jsonrpc 2.0",
                None,
                id,
            ))
        }
    }
    let method = match value.get("method").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => {
            return Err(make_error_response(
                -32600,
                "Invalid Request: missing method",
                None,
                id,
            ))
        }
    };
    let params = value.get("params").cloned().unwrap_or_else(|| json!({}));

    match method_to_command(method, &params) {
        Ok(cmd) => Ok((cmd, id)),
        Err((code, msg)) => Err(make_error_response(code, &msg, None, id)),
    }
}

/// Map a JSON-RPC method string and params to a Command variant.
fn method_to_command(method: &str, params: &Value) -> Result<Command, (i32, String)> {
    match method {
        "receive" => {
            let content = require_str(params, "content").map_err(|m| (-32602, m))?;
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::Receive { content, timestamp })
        }
        "teach" => {
            let kind = require_str(params, "kind").map_err(|m| (-32602, m))?;
            let content = require_str(params, "content").map_err(|m| (-32602, m))?;
            let confidence = optional_u32(params, "confidence", 800) as u16;
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::Teach {
                kind,
                content,
                confidence,
                timestamp,
            })
        }
        "synthesize" => {
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::Synthesize { timestamp })
        }
        "query" => {
            let key = require_str(params, "key").map_err(|m| (-32602, m))?;
            Ok(Command::Query { key })
        }
        "profile" => Ok(Command::Profile),
        "knows_me" => Ok(Command::KnowsMe),
        "health" => Ok(Command::Health),
        "explain" => {
            let action_hash = require_str(params, "action_hash").map_err(|m| (-32602, m))?;
            Ok(Command::Explain { action_hash })
        }
        "reflex_stats" => Ok(Command::ReflexStats),
        "reflex_invalidate" => {
            let trigger_hash = require_str(params, "trigger_hash").map_err(|m| (-32602, m))?;
            Ok(Command::ReflexInvalidate { trigger_hash })
        }
        "plan_action" => {
            let payload = require_value(params, "payload").map_err(|m| (-32602, m))?;
            Ok(Command::PlanAction {
                payload_json: payload.to_string(),
            })
        }
        "run_action" => {
            let plan_id = optional_str(params, "plan_id", "");
            let payload = require_value(params, "payload").map_err(|m| (-32602, m))?;
            Ok(Command::RunAction {
                plan_id,
                payload_json: payload.to_string(),
            })
        }
        "action_status" => {
            let action_id = require_str(params, "action_id").map_err(|m| (-32602, m))?;
            Ok(Command::ActionStatus { action_id })
        }
        "action_history" => {
            let limit = optional_u32(params, "limit", 20);
            let cursor = optional_str(params, "cursor", "");
            Ok(Command::ActionHistory { limit, cursor })
        }
        "start_session" => {
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::StartSession { timestamp })
        }
        "end_session" => {
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::EndSession { timestamp })
        }
        "ihsan" => {
            let score = require_u16(params, "score").map_err(|m| (-32602, m))?;
            Ok(Command::Ihsan { score })
        }
        "ping" => Ok(Command::Ping),
        "version" => Ok(Command::Version),
        "shutdown" => Ok(Command::Shutdown),
        "intent_classify" => {
            let content = require_str(params, "content").map_err(|m| (-32602, m))?;
            Ok(Command::IntentClassify { content })
        }
        "guardian_check" => {
            let content = require_str(params, "content").map_err(|m| (-32602, m))?;
            Ok(Command::GuardianCheck { content })
        }
        "action_dispatch" => {
            let channel = require_str(params, "channel").map_err(|m| (-32602, m))?;
            let payload = require_value(params, "payload").map_err(|m| (-32602, m))?;
            Ok(Command::ActionDispatch {
                channel,
                payload_json: payload.to_string(),
            })
        }

        // -- SAP v0 Protocol --
        "sap_meet_open" => {
            let profile = optional_str(params, "profile", "sap-ads-retail-v0");
            let initiator_role = optional_str(params, "initiator_role", "visitor");
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::SapMeetOpen {
                profile,
                initiator_role,
                timestamp,
            })
        }
        "sap_message" => {
            let session_id = require_str(params, "session_id").map_err(|m| (-32602, m))?;
            let content = require_str(params, "content").map_err(|m| (-32602, m))?;
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::SapMessage {
                session_id,
                content,
                timestamp,
            })
        }
        "sap_disclosure" => {
            let session_id = require_str(params, "session_id").map_err(|m| (-32602, m))?;
            Ok(Command::SapDisclosure { session_id })
        }
        "sap_consent_request" => {
            let session_id = require_str(params, "session_id").map_err(|m| (-32602, m))?;
            let scopes = require_value(params, "scopes").map_err(|m| (-32602, m))?;
            Ok(Command::SapConsentRequest {
                session_id,
                scopes_json: scopes.to_string(),
            })
        }
        "sap_consent_revoke" => {
            let session_id = require_str(params, "session_id").map_err(|m| (-32602, m))?;
            let receipt_id = require_str(params, "receipt_id").map_err(|m| (-32602, m))?;
            Ok(Command::SapConsentRevoke {
                session_id,
                receipt_id,
            })
        }
        "sap_session_close" => {
            let session_id = require_str(params, "session_id").map_err(|m| (-32602, m))?;
            let timestamp = optional_u64(params, "timestamp", current_time_secs());
            Ok(Command::SapSessionClose {
                session_id,
                timestamp,
            })
        }

        _ => Err((-32601, format!("Method not found: {}", method))),
    }
}

// -- Response Conversion: Response -> JSON-RPC string --

/// Convert a protocol Response to a JSON-RPC 2.0 response string.
pub fn response_to_jsonrpc(response: &Response, id: &Value) -> String {
    match response {
        Response::Ok(fields) => {
            let mut result_obj = serde_json::Map::new();
            for (key, value) in fields {
                result_obj.insert(key.clone(), Value::String(value.clone()));
            }
            json!({ "jsonrpc": "2.0", "result": Value::Object(result_obj), "id": id }).to_string()
        }
        Response::Err(code, message) => {
            let rpc_code = error_code_to_rpc(code);
            json!({
                "jsonrpc": "2.0",
                "error": { "code": rpc_code, "message": message, "data": { "code": code.as_str() } },
                "id": id,
            }).to_string()
        }
    }
}

// -- Batch Processing --

/// Process a batch of JSON-RPC requests. Returns a JSON array of responses.
pub fn handle_batch<F>(json_str: &str, mut handler: F) -> String
where
    F: FnMut(Command) -> Response,
{
    let items: Vec<Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return make_error_response(-32700, "Parse error", None, Value::Null),
    };
    if items.is_empty() {
        return json!([json!({
            "jsonrpc": "2.0",
            "error": { "code": -32600, "message": "Empty batch" },
            "id": Value::Null,
        })])
        .to_string();
    }
    if items.len() > MAX_BATCH_SIZE {
        return make_error_response(
            -32600,
            &format!(
                "Batch too large: {} items (max {})",
                items.len(),
                MAX_BATCH_SIZE
            ),
            None,
            Value::Null,
        );
    }
    let mut responses = Vec::with_capacity(items.len());
    for item in &items {
        let line = item.to_string();
        match parse_jsonrpc(&line) {
            Ok((cmd, id)) => {
                let resp = handler(cmd);
                responses.push(response_to_jsonrpc(&resp, &id));
            }
            Err(error_json) => responses.push(error_json),
        }
    }
    format!("[{}]", responses.join(","))
}

// -- TCP Listener --

/// Start a TCP listener that accepts JSON-RPC 2.0 requests.
///
/// `handler_fn` is called for each parsed Command. Blocks the calling thread.
/// Uses std::net (synchronous), one OS thread per connection.
pub fn start_tcp_listener<F>(config: McpTransportConfig, handler_fn: Arc<F>)
where
    F: Fn(Command) -> Response + Send + Sync + 'static,
{
    let addr = format!("{}:{}", config.host, config.port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("MCP transport: failed to bind {}: {}", addr, e);
            return;
        }
    };
    eprintln!("MCP transport listening on {}", addr);
    let active_connections = Arc::new(AtomicU16::new(0));

    for stream_result in listener.incoming() {
        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("MCP transport: accept error: {}", e);
                continue;
            }
        };
        let current = active_connections.load(Ordering::Acquire);
        if current >= config.max_connections {
            eprintln!("MCP transport: connection limit reached ({})", current);
            drop(stream);
            continue;
        }
        active_connections.fetch_add(1, Ordering::AcqRel);
        let handler = Arc::clone(&handler_fn);
        let conns = Arc::clone(&active_connections);
        let timeout_ms = config.read_timeout_ms;
        std::thread::spawn(move || {
            handle_tcp_connection(stream, &*handler, timeout_ms);
            conns.fetch_sub(1, Ordering::AcqRel);
        });
    }
}

/// Read a single newline-delimited line with bounded length.
///
/// Returns `Ok(Some(line))` on success, `Ok(None)` on EOF,
/// or `Err` with `InvalidData` if the line exceeds `max_len` bytes.
/// This prevents OOM from an attacker sending a multi-GB line.
fn read_bounded_line(reader: &mut impl BufRead, max_len: usize) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => {
                return Ok(if buf.is_empty() {
                    None
                } else {
                    Some(String::from_utf8_lossy(&buf).into_owned())
                });
            }
            Ok(_) => {
                if byte[0] == b'\n' {
                    return Ok(Some(String::from_utf8_lossy(&buf).into_owned()));
                }
                if buf.len() >= max_len {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "line too long",
                    ));
                }
                buf.push(byte[0]);
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return Ok(if buf.is_empty() {
                    None
                } else {
                    Some(String::from_utf8_lossy(&buf).into_owned())
                });
            }
            Err(e) => return Err(e),
        }
    }
}

/// Handle a single TCP connection: read lines, parse JSON-RPC, dispatch, respond.
fn handle_tcp_connection<F>(stream: std::net::TcpStream, handler: &F, timeout_ms: u64)
where
    F: Fn(Command) -> Response,
{
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms)));
    let read_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(read_stream);
    let mut writer = BufWriter::new(&stream);

    loop {
        let line = match read_bounded_line(&mut reader, MAX_LINE_LENGTH) {
            Ok(Some(l)) => l,
            Ok(None) => break, // EOF
            Err(e) => {
                if e.kind() == std::io::ErrorKind::InvalidData {
                    // Line too long — send error and disconnect
                    let err = make_error_response(-32600, "Request too large", None, Value::Null);
                    let _ = writeln!(writer, "{}", err);
                    let _ = writer.flush();
                }
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Batch request (JSON array)
        if trimmed.starts_with('[') {
            let batch_response = handle_batch(trimmed, handler);
            if writeln!(writer, "{}", batch_response).is_err() {
                break;
            }
            let _ = writer.flush();
            continue;
        }
        // Single request
        match parse_jsonrpc(trimmed) {
            Ok((cmd, id)) => {
                let is_shutdown = matches!(cmd, Command::Shutdown);
                let response = handler(cmd);
                let wire = response_to_jsonrpc(&response, &id);
                if writeln!(writer, "{}", wire).is_err() {
                    break;
                }
                let _ = writer.flush();
                if is_shutdown {
                    return;
                }
            }
            Err(error_json) => {
                if writeln!(writer, "{}", error_json).is_err() {
                    break;
                }
                let _ = writer.flush();
            }
        }
    }
}

/// Build a JSON-RPC error response string.
fn make_error_response(code: i32, message: &str, data: Option<Value>, id: Value) -> String {
    let error_obj = if let Some(d) = data {
        json!({ "code": code, "message": message, "data": d })
    } else {
        json!({ "code": code, "message": message })
    };
    json!({ "jsonrpc": "2.0", "error": error_obj, "id": id }).to_string()
}

// -- Unit Tests (smoke; full coverage in tests/mcp_transport_tests.rs) --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ping_smoke() {
        let (cmd, id) = parse_jsonrpc(r#"{"jsonrpc":"2.0","method":"ping","id":1}"#).unwrap();
        assert_eq!(cmd, Command::Ping);
        assert_eq!(id, json!(1));
    }

    #[test]
    fn parse_invalid_json_returns_32700() {
        let v: Value = serde_json::from_str(&parse_jsonrpc("not json").unwrap_err()).unwrap();
        assert_eq!(v["error"]["code"], -32700);
    }

    #[test]
    fn response_ok_roundtrip() {
        let wire = response_to_jsonrpc(&Response::ok_single("pong", "true"), &json!(1));
        let v: Value = serde_json::from_str(&wire).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["result"]["pong"], "true");
    }

    #[test]
    fn response_err_roundtrip() {
        let resp = Response::Err(ErrorCode::MissingArg, "content required".to_string());
        let v: Value = serde_json::from_str(&response_to_jsonrpc(&resp, &json!(1))).unwrap();
        assert_eq!(v["error"]["code"], -32602);
        assert_eq!(v["error"]["data"]["code"], "MISSING_ARG");
    }

    // -- SAP v0 JSON-RPC Tests --

    #[test]
    fn parse_sap_meet_open_rpc() {
        let (cmd, id) = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_meet_open","params":{"profile":"sap-ads-retail-v0","initiator_role":"visitor","timestamp":1000},"id":10}"#,
        )
        .unwrap();
        assert_eq!(id, json!(10));
        match cmd {
            Command::SapMeetOpen {
                profile,
                initiator_role,
                timestamp,
            } => {
                assert_eq!(profile, "sap-ads-retail-v0");
                assert_eq!(initiator_role, "visitor");
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("expected SapMeetOpen"),
        }
    }

    #[test]
    fn parse_sap_meet_open_defaults_rpc() {
        let (cmd, _) =
            parse_jsonrpc(r#"{"jsonrpc":"2.0","method":"sap_meet_open","params":{},"id":11}"#)
                .unwrap();
        match cmd {
            Command::SapMeetOpen {
                profile,
                initiator_role,
                ..
            } => {
                assert_eq!(profile, "sap-ads-retail-v0");
                assert_eq!(initiator_role, "visitor");
            }
            _ => panic!("expected SapMeetOpen"),
        }
    }

    #[test]
    fn parse_sap_message_rpc() {
        let (cmd, _) = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_message","params":{"session_id":"ses_1","content":"hello","timestamp":2000},"id":12}"#,
        )
        .unwrap();
        match cmd {
            Command::SapMessage {
                session_id,
                content,
                timestamp,
            } => {
                assert_eq!(session_id, "ses_1");
                assert_eq!(content, "hello");
                assert_eq!(timestamp, 2000);
            }
            _ => panic!("expected SapMessage"),
        }
    }

    #[test]
    fn parse_sap_message_missing_session_rpc() {
        let err = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_message","params":{"content":"hello"},"id":13}"#,
        )
        .unwrap_err();
        let v: Value = serde_json::from_str(&err).unwrap();
        assert_eq!(v["error"]["code"], -32602);
    }

    #[test]
    fn parse_sap_disclosure_rpc() {
        let (cmd, _) = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_disclosure","params":{"session_id":"ses_1"},"id":14}"#,
        )
        .unwrap();
        match cmd {
            Command::SapDisclosure { session_id } => {
                assert_eq!(session_id, "ses_1");
            }
            _ => panic!("expected SapDisclosure"),
        }
    }

    #[test]
    fn parse_sap_consent_request_rpc() {
        let (cmd, _) = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_consent_request","params":{"session_id":"ses_1","scopes":["name","email"]},"id":15}"#,
        )
        .unwrap();
        match cmd {
            Command::SapConsentRequest {
                session_id,
                scopes_json,
            } => {
                assert_eq!(session_id, "ses_1");
                assert!(scopes_json.contains("name"));
            }
            _ => panic!("expected SapConsentRequest"),
        }
    }

    #[test]
    fn parse_sap_consent_revoke_rpc() {
        let (cmd, _) = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_consent_revoke","params":{"session_id":"ses_1","receipt_id":"rcpt_1"},"id":16}"#,
        )
        .unwrap();
        match cmd {
            Command::SapConsentRevoke {
                session_id,
                receipt_id,
            } => {
                assert_eq!(session_id, "ses_1");
                assert_eq!(receipt_id, "rcpt_1");
            }
            _ => panic!("expected SapConsentRevoke"),
        }
    }

    #[test]
    fn parse_sap_session_close_rpc() {
        let (cmd, _) = parse_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"sap_session_close","params":{"session_id":"ses_1","timestamp":3000},"id":17}"#,
        )
        .unwrap();
        match cmd {
            Command::SapSessionClose {
                session_id,
                timestamp,
            } => {
                assert_eq!(session_id, "ses_1");
                assert_eq!(timestamp, 3000);
            }
            _ => panic!("expected SapSessionClose"),
        }
    }
}
