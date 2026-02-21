# Phase 2: MCP JSON-RPC Transport Layer

## Sprint 3 — Alpha-100 Action Infrastructure (Protocol Upgrade)

> Standing on Giants: Lamport (state machine replication, 1978) · Cerf & Kahn (TCP/IP layering, 1974) · Fielding (REST layered system, 2000) · Anthropic (MCP specification, 2024)
> artifact: `bizra-node/src/protocol.rs`, `bizra-node/src/main.rs`, `bizra-node/src/handler.rs`

---

## 1. Context

The current `bizra-node` speaks a custom tab-delimited protocol over stdin/stdout:
```
RECEIVE\t<content>\t<timestamp>\n
→ OK\tresponse=...\n
```

22 commands defined in `protocol.rs::Command` enum. Wire format: `parse_command(line) → Command`, `Response::to_wire() → String`.

**What exists:** Full protocol with 22 commands, handler for all commands, tab-delimited wire format.
**What's missing:** JSON-RPC 2.0 framing, TCP listener, concurrent connection handling.

The MCP specification requires JSON-RPC 2.0 over stdio or SSE. Sprint 3 adds a TCP JSON-RPC 2.0 transport as an **additional** listener — the stdio transport stays for backward compatibility and local development.

---

## 2. Functional Requirements

### FR-1: JSON-RPC 2.0 Framing
- Wrap existing `Command` enum in JSON-RPC 2.0 request format
- Wrap existing `Response` enum in JSON-RPC 2.0 response format
- Support `method` strings matching command names (lowercase snake_case)
- Support `params` as JSON object with named parameters
- Support batch requests (JSON array of requests)

### FR-2: TCP Transport
- Listen on configurable port (default: `9741`, distinct from AHK bridge `9742`)
- Accept concurrent connections (max 16, configurable via `NodeConfig`)
- Newline-delimited JSON (one JSON object per line, like the AHK bridge)
- Optional TLS (deferred to Sprint 4 — local-only for now)

### FR-3: Dual-Mode Operation
- Node starts BOTH transports simultaneously:
  - Stdio: existing `BufRead` + `println!` loop (unchanged)
  - TCP/JSON-RPC: new listener thread/task
- `--stdio-only` flag disables TCP listener
- `--tcp-only` flag disables stdio listener
- Default: both active

### FR-4: Method Mapping
Map each `Command` variant to a JSON-RPC method name:

| Command Variant | JSON-RPC Method | Params |
|----------------|-----------------|--------|
| `Receive` | `receive` | `{ content: string, timestamp?: u64 }` |
| `Teach` | `teach` | `{ kind: string, content: string, confidence?: u16, timestamp?: u64 }` |
| `Synthesize` | `synthesize` | `{ timestamp?: u64 }` |
| `Query` | `query` | `{ key: string }` |
| `Profile` | `profile` | `{}` |
| `KnowsMe` | `knows_me` | `{}` |
| `Health` | `health` | `{}` |
| `Explain` | `explain` | `{ action_hash: string }` |
| `ReflexStats` | `reflex_stats` | `{}` |
| `ReflexInvalidate` | `reflex_invalidate` | `{ trigger_hash: string }` |
| `PlanAction` | `plan_action` | `{ payload: object }` |
| `RunAction` | `run_action` | `{ plan_id?: string, payload: object }` |
| `ActionStatus` | `action_status` | `{ action_id: string }` |
| `ActionHistory` | `action_history` | `{ limit?: u32, cursor?: string }` |
| `StartSession` | `start_session` | `{ timestamp?: u64 }` |
| `EndSession` | `end_session` | `{ timestamp?: u64 }` |
| `Ihsan` | `ihsan` | `{ score: u16 }` |
| `Ping` | `ping` | `{}` |
| `Version` | `version` | `{}` |
| `Shutdown` | `shutdown` | `{}` |
| `IntentClassify` | `intent_classify` | `{ content: string }` |
| `ActionDispatch` | `action_dispatch` | `{ channel: string, payload: object }` |

### FR-5: Response Mapping
```json
// Success (Response::Ok)
{
  "jsonrpc": "2.0",
  "result": { "key1": "value1", "key2": "value2" },
  "id": 1
}

// Error (Response::Err)
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "error message",
    "data": { "code": "BAD_COMMAND" }
  },
  "id": 1
}
```

Error code mapping:
| `ErrorCode` | JSON-RPC code |
|-------------|---------------|
| `BadCommand` | `-32601` (method not found) |
| `MissingArg` | `-32602` (invalid params) |
| `ParseError` | `-32700` (parse error) |
| `InvalidArg` | `-32602` (invalid params) |
| `InternalError` | `-32603` (internal error) |

---

## 3. Pseudocode

### 3.1 Module Structure (`bizra-node/src/mcp_transport.rs`)

```pseudocode
MODULE mcp_transport

IMPORT protocol::{Command, Response, ErrorCode}
IMPORT serde_json::{Value, json}

-- JSON-RPC 2.0 request
STRUCT JsonRpcRequest:
    jsonrpc: String        -- must be "2.0"
    method: String
    params: Option<Value>  -- named params (object) or positional (array)
    id: Value              -- number, string, or null

-- JSON-RPC 2.0 response
STRUCT JsonRpcResponse:
    jsonrpc: String        -- always "2.0"
    result: Option<Value>  -- present on success
    error: Option<JsonRpcError>  -- present on failure
    id: Value              -- matches request id

STRUCT JsonRpcError:
    code: i32
    message: String
    data: Option<Value>

-- Parse JSON-RPC request → Command
FUNCTION parse_jsonrpc(line: &str) -> Result<(Command, Value), JsonRpcResponse>:
    value = serde_json::from_str(line)?
    IF value["jsonrpc"] != "2.0":
        RETURN Err(jsonrpc_error(-32600, "Invalid Request", value["id"]))

    method = value["method"].as_str()?
    params = value["params"].clone().unwrap_or(json!({}))
    id = value["id"].clone()

    command = method_to_command(method, params)?
    RETURN Ok((command, id))

-- Map method string → Command variant
FUNCTION method_to_command(method: &str, params: Value) -> Result<Command, (i32, String)>:
    MATCH method:
        "receive":
            content = require_str(params, "content")?
            timestamp = optional_u64(params, "timestamp", current_time_secs())
            RETURN Command::Receive { content, timestamp }

        "teach":
            kind = require_str(params, "kind")?
            content = require_str(params, "content")?
            confidence = optional_u16(params, "confidence", 800)
            timestamp = optional_u64(params, "timestamp", current_time_secs())
            RETURN Command::Teach { kind, content, confidence, timestamp }

        "synthesize":
            timestamp = optional_u64(params, "timestamp", current_time_secs())
            RETURN Command::Synthesize { timestamp }

        "query":
            key = require_str(params, "key")?
            RETURN Command::Query { key }

        "profile":
            RETURN Command::Profile

        "knows_me":
            RETURN Command::KnowsMe

        "health":
            RETURN Command::Health

        "explain":
            action_hash = require_str(params, "action_hash")?
            RETURN Command::Explain { action_hash }

        "reflex_stats":
            RETURN Command::ReflexStats

        "reflex_invalidate":
            trigger_hash = require_str(params, "trigger_hash")?
            RETURN Command::ReflexInvalidate { trigger_hash }

        "plan_action":
            payload = require_value(params, "payload")?
            RETURN Command::PlanAction { payload_json: payload.to_string() }

        "run_action":
            plan_id = optional_str(params, "plan_id", "")
            payload = require_value(params, "payload")?
            RETURN Command::RunAction { plan_id, payload_json: payload.to_string() }

        "action_status":
            action_id = require_str(params, "action_id")?
            RETURN Command::ActionStatus { action_id }

        "action_history":
            limit = optional_u32(params, "limit", 20)
            cursor = optional_str(params, "cursor", "")
            RETURN Command::ActionHistory { limit, cursor }

        "start_session":
            timestamp = optional_u64(params, "timestamp", current_time_secs())
            RETURN Command::StartSession { timestamp }

        "end_session":
            timestamp = optional_u64(params, "timestamp", current_time_secs())
            RETURN Command::EndSession { timestamp }

        "ihsan":
            score = require_u16(params, "score")?
            RETURN Command::Ihsan { score }

        "ping":
            RETURN Command::Ping

        "version":
            RETURN Command::Version

        "shutdown":
            RETURN Command::Shutdown

        "intent_classify":
            content = require_str(params, "content")?
            RETURN Command::IntentClassify { content }

        "action_dispatch":
            channel = require_str(params, "channel")?
            payload = require_value(params, "payload")?
            RETURN Command::ActionDispatch { channel, payload_json: payload.to_string() }

        _:
            RETURN Err((-32601, "Method not found: " + method))

-- Convert Response → JSON-RPC response
FUNCTION response_to_jsonrpc(response: Response, id: Value) -> String:
    MATCH response:
        Response::Ok(fields):
            result_obj = {}
            FOR (key, value) IN fields:
                result_obj[key] = value
            RETURN json!({
                "jsonrpc": "2.0",
                "result": result_obj,
                "id": id
            }).to_string()

        Response::Err(code, message):
            rpc_code = error_code_to_rpc(code)
            RETURN json!({
                "jsonrpc": "2.0",
                "error": {
                    "code": rpc_code,
                    "message": message,
                    "data": { "code": code.as_str() }
                },
                "id": id
            }).to_string()
```

### 3.2 TCP Listener

```pseudocode
STRUCT McpTransportConfig:
    host: String          -- default: "127.0.0.1"
    port: u16             -- default: 9741
    max_connections: u16  -- default: 16
    read_timeout_ms: u64  -- default: 30000

FUNCTION start_tcp_listener(config: McpTransportConfig, node: SharedNode):
    listener = TcpListener::bind(config.host + ":" + config.port)?
    LOG_INFO("MCP transport listening on " + config.host + ":" + config.port)

    active_connections = AtomicU16(0)

    FOR stream IN listener.incoming():
        IF active_connections.load() >= config.max_connections:
            stream.close()
            LOG_WARN("MCP transport: connection limit reached")
            CONTINUE

        active_connections.fetch_add(1)
        conn = stream
        node_ref = node.clone()
        config_ref = config.clone()

        SPAWN_THREAD:
            handle_tcp_connection(conn, node_ref, config_ref)
            active_connections.fetch_sub(1)

FUNCTION handle_tcp_connection(stream, node, config):
    stream.set_read_timeout(config.read_timeout_ms)
    reader = BufReader::new(stream.clone())
    writer = BufWriter::new(stream)

    -- Support persistent connections (multiple requests per connection)
    FOR line IN reader.lines():
        IF line.is_empty():
            CONTINUE

        -- Check for batch request (JSON array)
        trimmed = line.trim()
        IF trimmed.starts_with('['):
            responses = handle_batch(trimmed, node)
            writer.write_line(json!(responses).to_string())
            CONTINUE

        -- Single request
        MATCH parse_jsonrpc(trimmed):
            Ok((command, id)):
                -- Reuse existing handler (FR-3 dual-mode)
                response = node.handle_command(command)
                wire = response_to_jsonrpc(response, id)
                writer.write_line(wire)

                IF command IS Command::Shutdown:
                    RETURN  -- Close connection after shutdown

            Err(error_response):
                writer.write_line(error_response.to_string())

        writer.flush()

FUNCTION handle_batch(json_str, node) -> Vec<JsonRpcResponse>:
    requests = serde_json::from_str::<Vec<Value>>(json_str)?
    IF requests.is_empty():
        RETURN [jsonrpc_error(-32600, "Empty batch", null)]

    responses = []
    FOR request IN requests:
        line = request.to_string()
        MATCH parse_jsonrpc(line):
            Ok((command, id)):
                response = node.handle_command(command)
                responses.push(response_to_jsonrpc(response, id))
            Err(error_response):
                responses.push(error_response)
    RETURN responses
```

### 3.3 Dual-Mode Main Entry Modification

```pseudocode
-- Modification to bizra-node/src/main.rs
-- Add to NodeConfig:
STRUCT NodeConfig:
    ... (existing fields)
    tcp_enabled: bool      -- default: true
    tcp_host: String       -- default: "127.0.0.1"
    tcp_port: u16          -- default: 9741
    stdio_enabled: bool    -- default: true

-- Add CLI args:
--   --stdio-only    (sets tcp_enabled=false)
--   --tcp-only      (sets stdio_enabled=false)
--   --tcp-port N    (sets tcp_port=N)

-- In main():
FUNCTION main():
    config = parse_args()
    node = Node::new(config)

    IF config.tcp_enabled:
        tcp_config = McpTransportConfig {
            host: config.tcp_host,
            port: config.tcp_port,
            max_connections: 16,
            read_timeout_ms: 30000,
        }
        SPAWN_THREAD start_tcp_listener(tcp_config, node.shared())

    IF config.stdio_enabled:
        -- Existing stdio loop (unchanged)
        run_stdio_loop(node)
    ELSE:
        -- Block on TCP thread (park main thread)
        park_until_shutdown()
```

---

## 4. File Inventory

| File | Action | ~Lines | Purpose |
|------|--------|--------|---------|
| `bizra-omega/bizra-node/src/mcp_transport.rs` | CREATE | ~280 | JSON-RPC ↔ Command translation + TCP listener |
| `bizra-omega/bizra-node/src/lib.rs` | MODIFY | +1 | Add `pub mod mcp_transport;` |
| `bizra-omega/bizra-node/src/main.rs` | MODIFY | +30 | CLI args + spawn TCP listener thread |
| `bizra-omega/bizra-node/src/node.rs` | MODIFY | +10 | Add `shared()` method for Arc wrapping |
| `bizra-omega/bizra-node/tests/mcp_transport_tests.rs` | CREATE | ~200 | Integration tests for JSON-RPC framing |

---

## 5. TDD Anchors

```
TEST parse_valid_receive_request
  → input: {"jsonrpc":"2.0","method":"receive","params":{"content":"hello"},"id":1}
  → Expect: Command::Receive { content: "hello", timestamp: <now> }

TEST parse_receive_with_explicit_timestamp
  → input: {"jsonrpc":"2.0","method":"receive","params":{"content":"hello","timestamp":100},"id":2}
  → Expect: Command::Receive { content: "hello", timestamp: 100 }

TEST parse_teach_all_fields
  → input: {"jsonrpc":"2.0","method":"teach","params":{"kind":"fact","content":"Earth is round","confidence":950},"id":3}
  → Expect: Command::Teach { kind: "fact", content: "Earth is round", confidence: 950, timestamp: <now> }

TEST parse_no_params_commands
  → input: {"jsonrpc":"2.0","method":"ping","id":4}
  → Expect: Command::Ping

TEST parse_plan_action_nested_payload
  → input: {"jsonrpc":"2.0","method":"plan_action","params":{"payload":{"steps":[{"channel":"DesktopRpc","kind":"Click","payload":"click button"}]}},"id":5}
  → Expect: Command::PlanAction { payload_json: '{"steps":[...]}' }

TEST parse_unknown_method_error
  → input: {"jsonrpc":"2.0","method":"unknown","id":6}
  → Expect: JsonRpcError { code: -32601, message: "Method not found: unknown" }

TEST parse_missing_required_param
  → input: {"jsonrpc":"2.0","method":"receive","params":{},"id":7}
  → Expect: JsonRpcError { code: -32602, message: "Missing required param: content" }

TEST parse_invalid_json
  → input: "not json"
  → Expect: JsonRpcError { code: -32700, message: "Parse error" }

TEST response_ok_to_jsonrpc
  → input: Response::Ok(vec![("intent", "Code"), ("confidence", "0.85")])
  → Expect: {"jsonrpc":"2.0","result":{"intent":"Code","confidence":"0.85"},"id":1}

TEST response_err_to_jsonrpc
  → input: Response::Err(ErrorCode::MissingArg, "content required")
  → Expect: {"jsonrpc":"2.0","error":{"code":-32602,"message":"content required","data":{"code":"MISSING_ARG"}},"id":1}

TEST batch_request_processing
  → input: [{"jsonrpc":"2.0","method":"ping","id":1},{"jsonrpc":"2.0","method":"version","id":2}]
  → Expect: array with 2 responses

TEST tcp_roundtrip_integration
  → Start TCP listener on random port
  → Connect, send ping request, read response
  → Expect: {"jsonrpc":"2.0","result":{"status":"pong"},"id":1}

TEST persistent_connection_multiple_requests
  → Connect once, send ping then version on same connection
  → Expect: 2 valid responses on same stream

TEST connection_limit_enforcement
  → Open 16 connections, try 17th
  → Expect: 17th rejected or queued
```

---

## 6. Integration Points

| From | To | Contract |
|------|----|----------|
| External MCP client | `mcp_transport.rs::handle_tcp_connection` | JSON-RPC 2.0 over TCP:9741 |
| `mcp_transport.rs::method_to_command` | `protocol.rs::Command` | 1:1 variant mapping |
| `mcp_transport.rs::response_to_jsonrpc` | `protocol.rs::Response` | Structured field → JSON |
| `main.rs` | `mcp_transport.rs::start_tcp_listener` | Spawned in separate thread |
| `handler.rs::handle_command` | Unchanged | Same handler serves both transports |

**Key architectural decision:** The handler layer is transport-agnostic. Both stdio and TCP transports produce `Command` and consume `Response`. No handler code changes needed.

---

## 7. Edge Cases

- **Invalid JSON-RPC version:** Reject with `-32600` (Invalid Request)
- **Missing `id` field (notification):** Process but do not respond (per JSON-RPC 2.0 spec)
- **`id: null` (valid per spec):** Respond with `"id": null`
- **Concurrent `Shutdown` from two transports:** First wins, second is no-op
- **Binary/non-UTF8 content:** `serde_json` rejects, returns parse error
- **Empty batch `[]`:** Return single error response

---

## 8. Non-Goals (Deferred)

- **TLS encryption** — Local-only in Sprint 3; TLS is Sprint 4
- **SSE transport** — MCP spec supports SSE; TCP-first for simplicity
- **Authentication on MCP port** — Sprint 3 is localhost-only; auth is Sprint 4
- **HTTP transport** — Raw TCP with newline framing; HTTP wrapping is Sprint 4
- **JSON-RPC notifications (no id)** — Parse but discard; bidirectional push is Sprint 4
