# Phase 57.07: Wirable Inventory Reality Check (Corrected)

Date: 2026-03-02
Status: Ground-truth inventory from current workspace

## Scope

This file is the corrected inventory for the First Heartbeat demo surfaces:
- Desktop bridge
- Browser MCP client
- Channel dispatcher
- Sovereign orchestrator + event bus
- AHK bridge/client
- MCP gateway
- A2A/PAT/Living Memory support layers

## Summary

What is real and directly usable now:
- `core/bridges/desktop_bridge.py` is functional JSON-RPC server with auth and HDA proxy handlers.
- `core/bridges/browser_mcp_client.py` is functional and supports `mock`, `direct`, and `mcp` modes.
- `core/bridges/channel_dispatcher.py` is functional planner/dispatcher with lazy browser mock fallback.
- `core/sovereign/orchestrator.py` and `core/sovereign/event_bus.py` are functional.
- `filedfs/ahk_bridge.ahk` (server) and `bin/bizra_bridge.ahk` (client hotkeys) are functional.
- `core/a2a/*`, `core/pat/*`, `core/living_memory/*` are largely functional and usable as support layers.

What is not fully real for this demo path:
- `tools/mcp/mcp_gateway.py` query/ingest are placeholders (`TODO` blocks).
- `scripts/ghost_overlay.ahk` is prototype-only and not aligned with DesktopBridge handler map.

## Corrected Status Matrix

| Surface | Status | Evidence |
|---|---|---|
| `core/bridges/desktop_bridge.py` | FUNCTIONAL (server + HDA proxy) | Handler map includes `ping/status/sovereign_query/open_app/type_text/...` in [desktop_bridge.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/desktop_bridge.py#L392); AHK RPC proxy in [desktop_bridge.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/desktop_bridge.py#L1182) |
| `core/bridges/browser_mcp_client.py` | FUNCTIONAL | `search/fetch_page/research` in [browser_mcp_client.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/browser_mcp_client.py#L142) |
| `core/bridges/channel_dispatcher.py` | FUNCTIONAL (planner + dependency-aware dispatch) | `decompose/dispatch_all` in [channel_dispatcher.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/channel_dispatcher.py#L91) and [channel_dispatcher.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/channel_dispatcher.py#L180) |
| `core/sovereign/orchestrator.py` | FUNCTIONAL | submit/execute/run in [orchestrator.py](/mnt/c/BIZRA-DATA-LAKE/core/sovereign/orchestrator.py#L746), [orchestrator.py](/mnt/c/BIZRA-DATA-LAKE/core/sovereign/orchestrator.py#L761), [orchestrator.py](/mnt/c/BIZRA-DATA-LAKE/core/sovereign/orchestrator.py#L938) |
| `core/sovereign/event_bus.py` | FUNCTIONAL | `EventBus.publish/emit/start` in [event_bus.py](/mnt/c/BIZRA-DATA-LAKE/core/sovereign/event_bus.py#L87), [event_bus.py](/mnt/c/BIZRA-DATA-LAKE/core/sovereign/event_bus.py#L94), [event_bus.py](/mnt/c/BIZRA-DATA-LAKE/core/sovereign/event_bus.py#L143) |
| `filedfs/ahk_bridge.ahk` | FUNCTIONAL (server) | JSON-RPC server startup in [ahk_bridge.ahk](/mnt/c/BIZRA-DATA-LAKE/filedfs/ahk_bridge.ahk#L54) |
| `bin/bizra_bridge.ahk` | FUNCTIONAL (client hotkeys) | `ConnectBridge/SendCommand` and hotkeys in [bizra_bridge.ahk](/mnt/c/BIZRA-DATA-LAKE/bin/bizra_bridge.ahk#L212), [bizra_bridge.ahk](/mnt/c/BIZRA-DATA-LAKE/bin/bizra_bridge.ahk#L312) |
| `tools/mcp/mcp_gateway.py` | PARTIAL/STUB | placeholder `TODO` query/ingest in [mcp_gateway.py](/mnt/c/BIZRA-DATA-LAKE/tools/mcp/mcp_gateway.py#L329) and [mcp_gateway.py](/mnt/c/BIZRA-DATA-LAKE/tools/mcp/mcp_gateway.py#L357) |
| `scripts/ghost_overlay.ahk` | PROTOTYPE | not the recommended demo control path |
| `core/telescript/*` | ABSENT in this workspace | no `core/telescript` directory |

## Key Wiring Risks (Real)

1. Port collision by default
- DesktopBridge server binds fixed `9742` in [desktop_bridge.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/desktop_bridge.py#L62)
- AHK server defaults to `9742` via env/config in [ahk_bridge.ahk](/mnt/c/BIZRA-DATA-LAKE/filedfs/ahk_bridge.ahk#L46) and [bridge_config.ini](/mnt/c/BIZRA-DATA-LAKE/filedfs/bridge_config.ini#L12)
- Client `bin/bizra_bridge.ahk` connects to `9742` in [bizra_bridge.ahk](/mnt/c/BIZRA-DATA-LAKE/bin/bizra_bridge.ahk#L28)

2. Auth is fail-closed on bridge startup
- DesktopBridge requires `BIZRA_BRIDGE_TOKEN` in [desktop_bridge.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/desktop_bridge.py#L1647)
- Missing token prevents startup.

3. Dispatcher desktop interface mismatch risk
- Dispatcher expects desktop object exposing `send_command` or `dispatch` in [channel_dispatcher.py](/mnt/c/BIZRA-DATA-LAKE/core/bridges/channel_dispatcher.py#L247)
- `DesktopBridge` class is a server class and does not expose that client method surface.

## Corrected Practical Conclusion

A true first heartbeat demo is feasible now with minimal glue code if we:
- split ports (keep DesktopBridge at 9742, move AHK server to 9743),
- run BrowserMCPClient in `mock` mode for deterministic browser leg,
- add a tiny DesktopBridge client adapter for `ChannelDispatcher` (or add a direct heartbeat RPC handler in DesktopBridge),
- avoid dependency on MCP gateway query/ingest placeholders.
