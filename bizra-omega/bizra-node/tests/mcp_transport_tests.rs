// bizra-node/tests/mcp_transport_tests.rs
// ============================================================
// Integration tests for the MCP JSON-RPC 2.0 Transport Layer
// ============================================================
//
// Tests cover:
//   - JSON-RPC parsing for every command category
//   - Missing/invalid parameter handling
//   - Response conversion (Ok and Err variants)
//   - Batch request processing
//   - Round-trip: parse request -> create response -> format JSON-RPC
//   - TCP listener integration (connect, send, receive)
// ============================================================

use bizra_node::mcp_transport::{
    current_time_secs, handle_batch, parse_jsonrpc, response_to_jsonrpc, start_tcp_listener,
    McpTransportConfig,
};
use bizra_node::protocol::{Command, ErrorCode, Response};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::Arc;

// ============================================================
// PARSE TESTS — method to Command mapping
// ============================================================

#[test]
fn parse_receive_request() {
    let input = r#"{"jsonrpc":"2.0","method":"receive","params":{"content":"hello"},"id":1}"#;
    let (cmd, id) = parse_jsonrpc(input).unwrap();
    match cmd {
        Command::Receive { content, timestamp } => {
            assert_eq!(content, "hello");
            assert!(timestamp > 0);
        }
        _ => panic!("expected Receive, got {:?}", cmd),
    }
    assert_eq!(id, json!(1));
}

#[test]
fn parse_receive_with_explicit_timestamp() {
    let input = r#"{"jsonrpc":"2.0","method":"receive","params":{"content":"hello","timestamp":100},"id":2}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::Receive {
            content: "hello".to_string(),
            timestamp: 100,
        }
    );
}

#[test]
fn parse_teach_all_fields() {
    let input = r#"{"jsonrpc":"2.0","method":"teach","params":{"kind":"fact","content":"Earth is round","confidence":950},"id":3}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::Teach {
            kind: "fact".to_string(),
            content: "Earth is round".to_string(),
            confidence: 950,
            timestamp: current_time_secs(), // approximate
        }
    );
    // Verify timestamp is close to now (within 2 seconds)
    if let Command::Teach { timestamp, .. } = cmd {
        let now = current_time_secs();
        assert!(now.abs_diff(timestamp) <= 2, "timestamp should be near now");
    }
}

#[test]
fn parse_teach_defaults() {
    let input =
        r#"{"jsonrpc":"2.0","method":"teach","params":{"kind":"fact","content":"test"},"id":4}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    if let Command::Teach { confidence, .. } = cmd {
        assert_eq!(confidence, 800); // default
    } else {
        panic!("expected Teach");
    }
}

#[test]
fn parse_no_params_commands() {
    let cases = vec![
        (r#"{"jsonrpc":"2.0","method":"ping","id":1}"#, Command::Ping),
        (
            r#"{"jsonrpc":"2.0","method":"version","id":2}"#,
            Command::Version,
        ),
        (
            r#"{"jsonrpc":"2.0","method":"shutdown","id":3}"#,
            Command::Shutdown,
        ),
        (
            r#"{"jsonrpc":"2.0","method":"profile","id":4}"#,
            Command::Profile,
        ),
        (
            r#"{"jsonrpc":"2.0","method":"knows_me","id":5}"#,
            Command::KnowsMe,
        ),
        (
            r#"{"jsonrpc":"2.0","method":"health","id":6}"#,
            Command::Health,
        ),
        (
            r#"{"jsonrpc":"2.0","method":"reflex_stats","id":7}"#,
            Command::ReflexStats,
        ),
    ];

    for (input, expected) in cases {
        let (cmd, _) = parse_jsonrpc(input).unwrap();
        assert_eq!(cmd, expected, "failed for input: {}", input);
    }
}

#[test]
fn parse_plan_action_nested_payload() {
    let input = r#"{"jsonrpc":"2.0","method":"plan_action","params":{"payload":{"steps":[{"channel":"DesktopRpc","kind":"Click","payload":"click button"}]}},"id":5}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    if let Command::PlanAction { payload_json } = cmd {
        let parsed: Value = serde_json::from_str(&payload_json).unwrap();
        assert!(parsed.get("steps").is_some());
    } else {
        panic!("expected PlanAction");
    }
}

#[test]
fn parse_run_action() {
    let input = r#"{"jsonrpc":"2.0","method":"run_action","params":{"plan_id":"pln_1","payload":{"x":1}},"id":6}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    if let Command::RunAction {
        plan_id,
        payload_json,
    } = cmd
    {
        assert_eq!(plan_id, "pln_1");
        let parsed: Value = serde_json::from_str(&payload_json).unwrap();
        assert_eq!(parsed["x"], 1);
    } else {
        panic!("expected RunAction");
    }
}

#[test]
fn parse_intent_classify() {
    let input = r#"{"jsonrpc":"2.0","method":"intent_classify","params":{"content":"help me code"},"id":7}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::IntentClassify {
            content: "help me code".to_string(),
        }
    );
}

#[test]
fn parse_action_dispatch() {
    let input = r#"{"jsonrpc":"2.0","method":"action_dispatch","params":{"channel":"llm","payload":{"prompt":"hello"}},"id":8}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    if let Command::ActionDispatch {
        channel,
        payload_json,
    } = cmd
    {
        assert_eq!(channel, "llm");
        let parsed: Value = serde_json::from_str(&payload_json).unwrap();
        assert_eq!(parsed["prompt"], "hello");
    } else {
        panic!("expected ActionDispatch");
    }
}

#[test]
fn parse_guardian_check() {
    let input = r#"{"jsonrpc":"2.0","method":"guardian_check","params":{"content":"plan next sprint"},"id":9}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::GuardianCheck {
            content: "plan next sprint".to_string(),
        }
    );
}

#[test]
fn parse_explain() {
    let input =
        r#"{"jsonrpc":"2.0","method":"explain","params":{"action_hash":"abcdef123456"},"id":10}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::Explain {
            action_hash: "abcdef123456".to_string(),
        }
    );
}

#[test]
fn parse_synthesize() {
    let input = r#"{"jsonrpc":"2.0","method":"synthesize","params":{"timestamp":5000},"id":10}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(cmd, Command::Synthesize { timestamp: 5000 });
}

#[test]
fn parse_action_history_with_defaults() {
    let input = r#"{"jsonrpc":"2.0","method":"action_history","id":11}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::ActionHistory {
            limit: 20,
            cursor: String::new(),
        }
    );
}

#[test]
fn parse_ihsan() {
    let input = r#"{"jsonrpc":"2.0","method":"ihsan","params":{"score":9800},"id":12}"#;
    let (cmd, _) = parse_jsonrpc(input).unwrap();
    assert_eq!(cmd, Command::Ihsan { score: 9800 });
}

// ============================================================
// ERROR HANDLING TESTS
// ============================================================

#[test]
fn parse_unknown_method_error() {
    let input = r#"{"jsonrpc":"2.0","method":"unknown","id":6}"#;
    let err = parse_jsonrpc(input).unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32601);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Method not found"));
    assert_eq!(v["id"], 6);
}

#[test]
fn parse_missing_required_param_error() {
    let input = r#"{"jsonrpc":"2.0","method":"receive","params":{},"id":7}"#;
    let err = parse_jsonrpc(input).unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32602);
    assert!(v["error"]["message"].as_str().unwrap().contains("content"));
    assert_eq!(v["id"], 7);
}

#[test]
fn parse_invalid_json_error() {
    let err = parse_jsonrpc("not json").unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32700);
}

#[test]
fn parse_missing_jsonrpc_version() {
    let input = r#"{"method":"ping","id":1}"#;
    let err = parse_jsonrpc(input).unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32600);
}

#[test]
fn parse_wrong_jsonrpc_version() {
    let input = r#"{"jsonrpc":"1.0","method":"ping","id":1}"#;
    let err = parse_jsonrpc(input).unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32600);
}

#[test]
fn parse_missing_method() {
    let input = r#"{"jsonrpc":"2.0","id":1}"#;
    let err = parse_jsonrpc(input).unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32600);
}

#[test]
fn parse_missing_ihsan_score() {
    let input = r#"{"jsonrpc":"2.0","method":"ihsan","params":{},"id":1}"#;
    let err = parse_jsonrpc(input).unwrap_err();
    let v: Value = serde_json::from_str(&err).unwrap();
    assert_eq!(v["error"]["code"], -32602);
}

// ============================================================
// RESPONSE CONVERSION TESTS
// ============================================================

#[test]
fn response_ok_to_jsonrpc_format() {
    let resp = Response::Ok(vec![
        ("intent".to_string(), "Code".to_string()),
        ("confidence".to_string(), "0.85".to_string()),
    ]);
    let wire = response_to_jsonrpc(&resp, &json!(1));
    let v: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(v["jsonrpc"], "2.0");
    assert_eq!(v["result"]["intent"], "Code");
    assert_eq!(v["result"]["confidence"], "0.85");
    assert_eq!(v["id"], 1);
    // Must not have error field
    assert!(v.get("error").is_none() || v["error"].is_null());
}

#[test]
fn response_err_to_jsonrpc_format() {
    let resp = Response::Err(ErrorCode::MissingArg, "content required".to_string());
    let wire = response_to_jsonrpc(&resp, &json!(1));
    let v: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(v["jsonrpc"], "2.0");
    assert_eq!(v["error"]["code"], -32602);
    assert_eq!(v["error"]["message"], "content required");
    assert_eq!(v["error"]["data"]["code"], "MISSING_ARG");
    assert_eq!(v["id"], 1);
}

#[test]
fn response_err_all_error_codes() {
    let cases = vec![
        (ErrorCode::BadCommand, -32601, "BAD_COMMAND"),
        (ErrorCode::MissingArg, -32602, "MISSING_ARG"),
        (ErrorCode::ParseError, -32700, "PARSE_ERROR"),
        (ErrorCode::InvalidArg, -32602, "INVALID_ARG"),
        (ErrorCode::InternalError, -32603, "INTERNAL_ERROR"),
    ];

    for (code, expected_rpc_code, expected_data_code) in cases {
        let resp = Response::Err(code, "test".to_string());
        let wire = response_to_jsonrpc(&resp, &json!(1));
        let v: Value = serde_json::from_str(&wire).unwrap();
        assert_eq!(
            v["error"]["code"], expected_rpc_code,
            "wrong code for {:?}",
            code
        );
        assert_eq!(
            v["error"]["data"]["code"], expected_data_code,
            "wrong data.code for {:?}",
            code
        );
    }
}

// ============================================================
// BATCH PROCESSING TESTS
// ============================================================

#[test]
fn batch_two_requests() {
    let input =
        r#"[{"jsonrpc":"2.0","method":"ping","id":1},{"jsonrpc":"2.0","method":"version","id":2}]"#;
    let result = handle_batch(input, |cmd| match cmd {
        Command::Ping => Response::ok_single("pong", "true"),
        Command::Version => Response::ok_single("version", "0.1.0"),
        _ => Response::err(ErrorCode::BadCommand, "unexpected"),
    });
    let v: Value = serde_json::from_str(&result).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 2);

    // Each element is a JSON object (parsed from the joined string)
    assert_eq!(arr[0]["result"]["pong"], "true");
    assert_eq!(arr[0]["id"], 1);

    assert_eq!(arr[1]["result"]["version"], "0.1.0");
    assert_eq!(arr[1]["id"], 2);
}

#[test]
fn batch_empty_array() {
    let result = handle_batch("[]", |_| Response::ok_single("x", "y"));
    let v: Value = serde_json::from_str(&result).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["error"]["code"], -32600);
}

#[test]
fn batch_invalid_json() {
    let result = handle_batch("not json", |_| Response::ok_single("x", "y"));
    let v: Value = serde_json::from_str(&result).unwrap();
    assert_eq!(v["error"]["code"], -32700);
}

// ============================================================
// ROUND-TRIP TESTS
// ============================================================

#[test]
fn roundtrip_ping() {
    let input = r#"{"jsonrpc":"2.0","method":"ping","id":42}"#;
    let (cmd, id) = parse_jsonrpc(input).unwrap();
    assert_eq!(cmd, Command::Ping);

    let resp = Response::ok_single("pong", "true");
    let wire = response_to_jsonrpc(&resp, &id);
    let v: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(v["jsonrpc"], "2.0");
    assert_eq!(v["result"]["pong"], "true");
    assert_eq!(v["id"], 42);
}

#[test]
fn roundtrip_receive_then_error() {
    // Parse a valid receive
    let input = r#"{"jsonrpc":"2.0","method":"receive","params":{"content":"test","timestamp":1000},"id":"req-1"}"#;
    let (cmd, id) = parse_jsonrpc(input).unwrap();
    assert_eq!(
        cmd,
        Command::Receive {
            content: "test".to_string(),
            timestamp: 1000,
        }
    );
    assert_eq!(id, json!("req-1"));

    // Simulate an error response
    let resp = Response::err(ErrorCode::InternalError, "processing failed");
    let wire = response_to_jsonrpc(&resp, &id);
    let v: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(v["error"]["code"], -32603);
    assert_eq!(v["id"], "req-1");
}

#[test]
fn roundtrip_string_id() {
    let input = r#"{"jsonrpc":"2.0","method":"health","id":"uuid-123"}"#;
    let (cmd, id) = parse_jsonrpc(input).unwrap();
    assert_eq!(cmd, Command::Health);

    let resp = Response::ok_single("state", "Ready");
    let wire = response_to_jsonrpc(&resp, &id);
    let v: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(v["id"], "uuid-123");
}

#[test]
fn roundtrip_null_id() {
    let input = r#"{"jsonrpc":"2.0","method":"ping","id":null}"#;
    let (_, id) = parse_jsonrpc(input).unwrap();
    assert!(id.is_null());

    let resp = Response::ok_single("pong", "true");
    let wire = response_to_jsonrpc(&resp, &id);
    let v: Value = serde_json::from_str(&wire).unwrap();
    assert!(v["id"].is_null());
}

// ============================================================
// TCP INTEGRATION TESTS
// ============================================================

#[test]
fn tcp_roundtrip_ping() {
    // Start listener on a random port
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener); // Release so the real listener can bind

    let handler = Arc::new(|cmd: Command| -> Response {
        match cmd {
            Command::Ping => Response::ok_single("pong", "true"),
            Command::Version => Response::ok_single("version", "0.1.0"),
            Command::Shutdown => Response::ok_single("shutdown", "true"),
            _ => Response::err(ErrorCode::BadCommand, "not handled"),
        }
    });

    let config = McpTransportConfig {
        host: "127.0.0.1".to_string(),
        port,
        max_connections: 4,
        read_timeout_ms: 5000,
    };

    let config_clone = config.clone();
    let handler_clone = Arc::clone(&handler);
    std::thread::spawn(move || {
        start_tcp_listener(config_clone, handler_clone);
    });

    // Give the listener a moment to start
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Connect and send a ping
    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();

    let request = r#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
    writeln!(stream, "{}", request).unwrap();
    stream.flush().unwrap();

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line).unwrap();

    let v: Value = serde_json::from_str(response_line.trim()).unwrap();
    assert_eq!(v["jsonrpc"], "2.0");
    assert_eq!(v["result"]["pong"], "true");
    assert_eq!(v["id"], 1);
}

#[test]
fn tcp_persistent_connection_multiple_requests() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let handler = Arc::new(|cmd: Command| -> Response {
        match cmd {
            Command::Ping => Response::ok_single("pong", "true"),
            Command::Version => Response::ok_single("version", "0.1.0"),
            _ => Response::err(ErrorCode::BadCommand, "not handled"),
        }
    });

    let config = McpTransportConfig {
        host: "127.0.0.1".to_string(),
        port,
        max_connections: 4,
        read_timeout_ms: 5000,
    };

    let config_clone = config.clone();
    let handler_clone = Arc::clone(&handler);
    std::thread::spawn(move || {
        start_tcp_listener(config_clone, handler_clone);
    });

    std::thread::sleep(std::time::Duration::from_millis(100));

    let stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();

    let mut writer = std::io::BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);

    // Request 1: ping
    writeln!(writer, r#"{{"jsonrpc":"2.0","method":"ping","id":1}}"#).unwrap();
    writer.flush().unwrap();

    let mut line1 = String::new();
    reader.read_line(&mut line1).unwrap();
    let v1: Value = serde_json::from_str(line1.trim()).unwrap();
    assert_eq!(v1["result"]["pong"], "true");

    // Request 2: version (same connection)
    writeln!(writer, r#"{{"jsonrpc":"2.0","method":"version","id":2}}"#).unwrap();
    writer.flush().unwrap();

    let mut line2 = String::new();
    reader.read_line(&mut line2).unwrap();
    let v2: Value = serde_json::from_str(line2.trim()).unwrap();
    assert_eq!(v2["result"]["version"], "0.1.0");
    assert_eq!(v2["id"], 2);
}

#[test]
fn tcp_error_for_unknown_method() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let handler = Arc::new(|_cmd: Command| -> Response { Response::ok_single("ok", "true") });

    let config = McpTransportConfig {
        host: "127.0.0.1".to_string(),
        port,
        max_connections: 4,
        read_timeout_ms: 5000,
    };

    std::thread::spawn(move || {
        start_tcp_listener(config, handler);
    });

    std::thread::sleep(std::time::Duration::from_millis(100));

    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();

    writeln!(
        stream,
        r#"{{"jsonrpc":"2.0","method":"nonexistent","id":99}}"#
    )
    .unwrap();
    stream.flush().unwrap();

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line).unwrap();

    let v: Value = serde_json::from_str(response_line.trim()).unwrap();
    assert_eq!(v["error"]["code"], -32601);
    assert_eq!(v["id"], 99);
}
