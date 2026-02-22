# Phase 1: AHK Bridge Server

## Sprint 3 — Alpha-100 Action Infrastructure (Desktop Actuator)

> Standing on Giants: General Magic (Telescript mobile agents, 1994) · Boyd (OODA Act phase, 1976) · Lamport (distributed state machines, 1978)
> artifact: `bizra-node/src/action_executor.rs:351-410`, `bizra-agent/src/action_types.rs`

---

## 1. Context

Sprint 2 built the Rust-side bridge client (`action_executor.rs:call_bridge()`):
- JSON-RPC 2.0 over newline-delimited TCP to `127.0.0.1:9742`
- Two methods: `invoke_skill`, `actuator_execute`
- Auth: `X-BIZRA-TOKEN` header, `X-BIZRA-TS`, `X-BIZRA-NONCE`
- Timeout: 3000ms default (`ActionExecutorConfig.timeout_ms`)

**What exists:** The outbound caller. **What's missing:** The inbound listener on port 9742.

---

## 2. Functional Requirements

### FR-1: TCP Server
- Listen on `127.0.0.1:9742` (configurable via `BIZRA_BRIDGE_PORT` env var)
- Accept multiple concurrent connections (connection pool, max 8)
- Read one JSON line per request, write one JSON line response
- Graceful shutdown on SIGTERM / script exit

### FR-2: Authentication
- Validate `X-BIZRA-TOKEN` against `BIZRA_BRIDGE_TOKEN` env var
- Validate `X-BIZRA-TS` within 30-second clock skew window
- Validate `X-BIZRA-NONCE` uniqueness within sliding 60-second window (replay protection)
- Reject with `{"jsonrpc":"2.0","error":{"code":-32600,"message":"Unauthorized"},"id":null}` on failure

### FR-3: Method Dispatch
```
invoke_skill(params: { skill: string, inputs: object }) -> { result: string }
actuator_execute(params: { code: string, intent: string, target_app: any }) -> { result: string }
```

### FR-4: Desktop Actuators (via AHK)
- `actuator_execute` with `intent: "click"` -> `Click, %code%`
- `actuator_execute` with `intent: "type"` -> `Send, %code%`
- `actuator_execute` with `intent: "execute"` -> `Run, %code%` (guarded)
- `actuator_execute` with `intent: "read"` -> window control text extraction
- `invoke_skill` -> dispatch to registered skill scripts in `skills/` directory

### FR-5: Safety Guards
- `code` field max length: 4096 chars
- Blocked patterns: `rm -rf`, `format c:`, `del /f /s`, `shutdown`, `taskkill` (case-insensitive)
- `target_app` allowlist: configured via `bridge_config.ini` (default: empty = any)
- Failed guard -> `{"jsonrpc":"2.0","error":{"code":-32001,"message":"Guardian veto: <reason>"},"id":id}`

---

## 3. Pseudocode

### 3.1 Main Entry (`ahk_bridge.ahk`)

```pseudocode
FUNCTION main():
    config = load_config("bridge_config.ini")
    token = ENV["BIZRA_BRIDGE_TOKEN"]
    IF token IS EMPTY:
        LOG_ERROR("BIZRA_BRIDGE_TOKEN not set, exiting")
        EXIT(1)

    port = ENV["BIZRA_BRIDGE_PORT"] OR config.port OR 9742
    nonce_cache = SlidingWindowSet(window_seconds=60, max_entries=1024)

    server = TCPServer(host="127.0.0.1", port=port, max_connections=8)
    LOG_INFO("AHK Bridge listening on 127.0.0.1:" + port)

    WHILE server.is_running():
        connection = server.accept()
        SPAWN handle_connection(connection, token, nonce_cache, config)
```

### 3.2 Connection Handler

```pseudocode
FUNCTION handle_connection(conn, token, nonce_cache, config):
    line = conn.read_line(timeout=5000)
    IF line IS EMPTY:
        conn.close()
        RETURN

    TRY:
        request = JSON.parse(line)
    CATCH:
        conn.write_line(jsonrpc_error(-32700, "Parse error", NULL))
        conn.close()
        RETURN

    id = request["id"]

    -- Authentication gate (FR-2)
    auth_result = authenticate(request["headers"], token, nonce_cache)
    IF auth_result.failed:
        conn.write_line(jsonrpc_error(-32600, auth_result.reason, id))
        conn.close()
        RETURN

    -- Method dispatch (FR-3)
    method = request["method"]
    params = request["params"] OR {}

    MATCH method:
        "invoke_skill":
            result = handle_invoke_skill(params, config)
        "actuator_execute":
            result = handle_actuator_execute(params, config)
        _:
            result = ERROR(-32601, "Method not found: " + method)

    IF result.is_error:
        conn.write_line(jsonrpc_error(result.code, result.message, id))
    ELSE:
        conn.write_line(jsonrpc_result(result.value, id))

    conn.close()
```

### 3.3 Authentication

```pseudocode
FUNCTION authenticate(headers, expected_token, nonce_cache):
    token = headers["X-BIZRA-TOKEN"]
    ts = headers["X-BIZRA-TS"]
    nonce = headers["X-BIZRA-NONCE"]

    IF token != expected_token:
        RETURN AuthFailed("Invalid token")

    now_ms = current_time_ms()
    IF ABS(now_ms - ts) > 30000:
        RETURN AuthFailed("Timestamp outside 30s window")

    IF nonce_cache.contains(nonce):
        RETURN AuthFailed("Nonce replay detected")

    nonce_cache.insert(nonce)
    RETURN AuthOk
```

### 3.4 Actuator Execute

```pseudocode
FUNCTION handle_actuator_execute(params, config):
    code = params["code"]
    intent = params["intent"] OR "execute"
    target_app = params["target_app"]

    -- FR-5 safety guards
    IF LEN(code) > 4096:
        RETURN ERROR(-32001, "Guardian veto: payload too large")

    blocked = ["rm -rf", "format c:", "del /f /s", "shutdown", "taskkill"]
    FOR pattern IN blocked:
        IF LOWER(code) CONTAINS pattern:
            RETURN ERROR(-32001, "Guardian veto: blocked pattern '" + pattern + "'")

    IF config.app_allowlist IS NOT EMPTY AND target_app NOT IN config.app_allowlist:
        RETURN ERROR(-32001, "Guardian veto: target_app not in allowlist")

    -- Intent dispatch (FR-4)
    MATCH LOWER(intent):
        "click":
            -- AHK: Click at coordinates or control
            AHK_EXECUTE("Click, " + code)
            RETURN OK("click executed")

        "type":
            -- AHK: Send keystrokes
            -- Escape special AHK characters in code
            safe_code = ahk_escape(code)
            AHK_EXECUTE("Send, " + safe_code)
            RETURN OK("type executed")

        "execute":
            -- AHK: Run command (most dangerous — extra validation)
            IF NOT config.allow_run:
                RETURN ERROR(-32001, "Guardian veto: Run disabled in config")
            AHK_EXECUTE("Run, " + code)
            RETURN OK("run executed")

        "read":
            -- AHK: ControlGetText or WinGetText
            text = AHK_READ_CONTROL(code, target_app)
            RETURN OK(text)

        _:
            RETURN ERROR(-32602, "Unknown intent: " + intent)
```

### 3.5 Invoke Skill

```pseudocode
FUNCTION handle_invoke_skill(params, config):
    skill = params["skill"]
    inputs = params["inputs"] OR {}

    IF skill IS EMPTY:
        RETURN ERROR(-32602, "params.skill is required")

    -- Skill scripts live in skills/ directory, one .ahk per skill
    skill_path = config.skills_dir + "/" + sanitize_filename(skill) + ".ahk"

    IF NOT FILE_EXISTS(skill_path):
        RETURN ERROR(-32602, "Skill not found: " + skill)

    -- Pass inputs as JSON via temp file (AHK string limits)
    temp_file = TEMP_DIR + "/bizra_skill_" + NONCE() + ".json"
    WRITE_FILE(temp_file, JSON.stringify(inputs))

    TRY:
        output = AHK_RUN_SCRIPT(skill_path, temp_file, timeout=10000)
        DELETE_FILE(temp_file)
        RETURN OK(output)
    CATCH timeout:
        DELETE_FILE(temp_file)
        RETURN ERROR(-32000, "Skill timeout after 10s")
    CATCH error:
        DELETE_FILE(temp_file)
        RETURN ERROR(-32000, "Skill error: " + error.message)
```

---

## 4. File Inventory

| File | Action | ~Lines | Purpose |
|------|--------|--------|---------|
| `filedfs/ahk_bridge.ahk` | CREATE | ~300 | TCP server + method dispatch |
| `filedfs/bridge_config.ini` | CREATE | ~20 | Port, allowlist, skills_dir, allow_run |
| `filedfs/skills/` | CREATE (dir) | — | Skill script directory (empty initially) |
| `filedfs/skills/hello_world.ahk` | CREATE | ~10 | Example skill for smoke testing |

---

## 5. TDD Anchors

```
TEST auth_valid_token_passes
  → Send request with correct token, ts within 30s, fresh nonce
  → Expect: method dispatch reached

TEST auth_bad_token_rejects
  → Send request with wrong token
  → Expect: {"error":{"code":-32600,"message":"Invalid token"}}

TEST auth_stale_timestamp_rejects
  → Send request with ts = now - 60000
  → Expect: rejected with "Timestamp outside 30s window"

TEST auth_replay_nonce_rejects
  → Send same nonce twice within 60s
  → Expect: second request rejected

TEST actuator_click_executes
  → params: { code: "100, 200", intent: "click" }
  → Expect: OK result

TEST actuator_blocked_pattern_vetoes
  → params: { code: "rm -rf /", intent: "execute" }
  → Expect: Guardian veto error

TEST actuator_payload_too_large_vetoes
  → params: { code: "A" * 5000, intent: "type" }
  → Expect: Guardian veto error

TEST skill_invoke_existing_skill
  → params: { skill: "hello_world", inputs: {} }
  → Expect: OK with skill output

TEST skill_invoke_missing_skill
  → params: { skill: "nonexistent", inputs: {} }
  → Expect: error "Skill not found"

TEST unknown_method_rejected
  → method: "drop_database"
  → Expect: {"error":{"code":-32601,"message":"Method not found"}}
```

---

## 6. Integration Points

| From | To | Contract |
|------|----|----------|
| `action_executor.rs:call_bridge("actuator_execute", ...)` | `ahk_bridge.ahk::handle_actuator_execute` | JSON-RPC 2.0 over TCP:9742 |
| `action_executor.rs:call_bridge("invoke_skill", ...)` | `ahk_bridge.ahk::handle_invoke_skill` | JSON-RPC 2.0 over TCP:9742 |
| `BIZRA_BRIDGE_TOKEN` env var | Shared secret | Set by installer (Sprint 1) |
| `bridge_config.ini` | `ahk_bridge.ahk` | INI file in same directory |

---

## 7. Edge Cases

- **AHK not installed:** Bridge script fails to start. Node's `call_bridge()` returns `BRIDGE_UNREACHABLE`. Graceful degradation — DesktopRpc actions fail, other channels unaffected.
- **Port already in use:** Exit with clear error message, suggest checking for orphaned bridge process.
- **Concurrent requests:** Connection pool max 8. Connection #9 gets TCP RST.
- **UTF-8 in payloads:** AHK v2 handles UTF-8 natively. AHK v1 requires BOM or explicit encoding — prefer v2.
- **Long-running skills:** 10s timeout per skill invocation. Timeout kills child process and returns error.

---

## 8. Non-Goals (Deferred to Sprint 4+)

- Multi-monitor awareness (screen coordinate mapping)
- OCR-based element detection
- AHK-to-EventBus reverse event flow (desktop events → Node)
- Browser automation via AHK (Sprint 3 Phase 2 handles browser via MCP)
