# Phase 57.08: Corrected First Heartbeat Spec + Pseudocode

Date: 2026-03-02
Status: Implementation-ready (minimal-delta path)
Supersedes: assumptions in earlier Phase 57 docs that referenced non-existent RPCs.

## Goal

Deliver one deterministic end-to-end demo task:

`AHK trigger -> DesktopBridge intake -> Browser research -> Desktop action via AHK HDA -> completion receipt`

## Non-Goals (for heartbeat v1)

- No dependency on MCP gateway `query/ingest` placeholder path.
- No full production orchestrator migration.
- No `ghost_overlay.ahk` integration in v1.

## Constraints and Truths

- DesktopBridge has real JSON-RPC handlers, including HDA proxy methods, in [desktop_bridge.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/desktop_bridge.py#L392).
- AHK bridge is a separate TCP server with overlapping default port; must split ports.
- Browser client has reliable `mock` mode for deterministic demo behavior in [browser_mcp_client.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/browser_mcp_client.py#L136).

## Architecture (v1)

1. Control Plane
- `bin/bizra_bridge.ahk` sends JSON-RPC to DesktopBridge on `127.0.0.1:9742`.

2. Browser Leg
- Python uses `BrowserMCPClient(mode="mock")` and runs `research(query)`.

3. Desktop Leg
- DesktopBridge proxies HDA calls (`open_app`, `type_text`, etc.) to AHK bridge via `_rpc_to_ahk`.
- AHK bridge runs on `127.0.0.1:9743`.

4. Result
- Heartbeat RPC returns combined result object + receipt reference.

## Required Configuration

1. Shared auth token
- `BIZRA_BRIDGE_TOKEN=<strong-random-token>` for both Python bridge and AHK server/client.

2. Port split
- DesktopBridge ingress remains `9742` (fixed server bind).
- AHK bridge server uses `9743`:
- `BIZRA_BRIDGE_PORT=9743` when launching `filedfs/ahk_bridge.ahk`.
- `BIZRA_AHK_BRIDGE_PORT=9743` for DesktopBridge process.

3. Browser mode
- Use `mock` for heartbeat v1 deterministic output.

## Minimal Code Delta

### Delta A: Add heartbeat RPC method in DesktopBridge handler map

- Add method name: `heartbeat_demo`
- Purpose: orchestrate both browser and desktop legs in one call.

### Delta B: Implement `_handle_heartbeat_demo(params)`

Inputs:
- `query: str` required
- `target_app: str` default `"Notepad"`

Outputs:
- `success: bool`
- `browser: {...}`
- `desktop: {...}`
- `summary: str`
- `receipt` (existing bridge receipt chain)

### Delta C (optional alternative): client adapter for ChannelDispatcher

If preferring dispatcher route, add a tiny client adapter exposing:
- `async send_command(method: str, payload: dict) -> dict`

Then feed adapter + `BrowserMCPClient(mode="mock")` into `ChannelDispatcher`.

## Pseudocode (Recommended: Direct Heartbeat Handler)

```python
# in core/bridges/desktop_bridge.py

async def _handle_heartbeat_demo(self, params: dict[str, Any]) -> dict[str, Any]:
    query = str(params.get("query", "")).strip()
    target_app = str(params.get("target_app", "Notepad"))
    if not query:
        return {"success": False, "error": "query is required"}

    # 1) Browser leg (deterministic for demo)
    browser = BrowserMCPClient(mode="mock")
    browser_result = await browser.research(query)
    summary = browser_result.get("summary", "No summary")

    # 2) Desktop leg via existing HDA proxy -> AHK bridge
    open_res = await self._rpc_to_ahk("open_app", {"app": target_app})
    if not open_res or open_res.get("error"):
        return {
            "success": False,
            "stage": "desktop_open",
            "browser": browser_result,
            "desktop": {"open_app": open_res},
            "error": "failed to open target app",
        }

    type_res = await self._rpc_to_ahk("type_text", {"text": summary})
    if not type_res or type_res.get("error"):
        return {
            "success": False,
            "stage": "desktop_type",
            "browser": browser_result,
            "desktop": {"open_app": open_res, "type_text": type_res},
            "error": "failed to type summary",
        }

    return {
        "success": True,
        "query": query,
        "browser": browser_result,
        "desktop": {"open_app": open_res, "type_text": type_res},
        "summary": summary,
    }
```

## JSON-RPC Request/Response Contract (Heartbeat)

Request:
```json
{
  "jsonrpc": "2.0",
  "method": "heartbeat_demo",
  "params": {
    "query": "Find top AI agent frameworks and summarize",
    "target_app": "Notepad"
  },
  "id": "hb-001",
  "headers": {
    "X-BIZRA-TOKEN": "${BIZRA_BRIDGE_TOKEN}",
    "X-BIZRA-TS": "${epoch_ms}",
    "X-BIZRA-NONCE": "${nonce_hex}"
  }
}
```

Success response:
```json
{
  "jsonrpc": "2.0",
  "id": "hb-001",
  "result": {
    "success": true,
    "query": "...",
    "browser": {"mode": "mock", "results": [], "summary": "..."},
    "desktop": {"open_app": {"ok": true}, "type_text": {"ok": true}},
    "summary": "...",
    "receipt": {"receipt_id": "...", "status": "accepted"}
  }
}
```

## Acceptance Criteria

1. Functional
- AHK hotkey triggers one RPC that completes both browser + desktop legs.
- Desktop shows typed heartbeat summary in target app.

2. Determinism
- Browser leg works offline in `mock` mode.

3. Safety
- Missing/invalid auth token returns auth failure.
- Missing AHK server returns graceful stage-specific error.

4. Evidence
- Bridge emits receipt with accepted/rejected status.

## TDD Anchors

1. `tests/integration/test_phase57_heartbeat_demo.py::test_heartbeat_demo_success_mock_browser`
2. `tests/integration/test_phase57_heartbeat_demo.py::test_heartbeat_demo_fails_without_query`
3. `tests/integration/test_phase57_heartbeat_demo.py::test_heartbeat_demo_handles_ahk_unreachable`
4. `tests/core/bridges/test_desktop_bridge_auth.py::test_heartbeat_requires_auth_headers`
5. `tests/core/bridges/test_desktop_bridge_port_split.py::test_ahk_port_override_uses_bizra_ahk_bridge_port`

## Runbook (v1)

1. Start AHK server on 9743 (Windows shell):
```powershell
$env:BIZRA_BRIDGE_TOKEN="<token>"
$env:BIZRA_BRIDGE_PORT="9743"
AutoHotkey64.exe C:\BIZRA-DATA-LAKE\filedfs\ahk_bridge.ahk
```

2. Start Python DesktopBridge with AHK port override:
```bash
export BIZRA_BRIDGE_TOKEN="<token>"
export BIZRA_AHK_BRIDGE_PORT="9743"
python -m core.bridges.desktop_bridge
```

3. Start AHK client (`bin/bizra_bridge.ahk`) for manual trigger, or send direct JSON-RPC from test harness.

## Risks and Mitigations

1. Port conflict regression
- Mitigation: explicit env override + startup log assertion for AHK port.

2. MCP drift
- Mitigation: lock v1 heartbeat to `BrowserMCPClient(mode="mock")`.

3. Interface drift (`dispatch_action` mismatch)
- Mitigation: do not route v1 through `ghost_overlay.ahk`; use DesktopBridge native handler map.
