# Module 08 — Desktop Automation

> **Domain:** HDA, AHK bridge, Telescript, Ghost Overlay, Tauri host
> **Source Specs:** Phase 57 (first heartbeat), Alpha-100, Phase 43 (node0 identity)
> **Key Paths:** `core/sovereign/`, `core/bridges/`, `bizra-omega/bizra-telescript/`

## 8.1 HDA Client (Human Desktop Automation)

**Status:** [x] BUILT
**Path:** `core/sovereign/` (HDAClient class)

Async TCP JSON-RPC client for AutoHotkey HDA server.
Port: 9743 (was collision on 9742 with bridge, fixed).

**Integration:** Used by MissionOrchestrator in EXECUTE phase.
**Env:** `BIZRA_HDA_PORT`

---

## 8.2 AHK Bridge

**Status:** [x] BUILT
**Path:** `core/bridges/` (AHK bridge module)

Connects Python sovereign runtime to AutoHotkey desktop automation.
Demo hotkey: `Win+Shift+B` -> InputBox -> mission execution -> toast + briefing file.

**Port:** 9743 (`BIZRA_HDA_PORT`)
**Spec:** Alpha-100 Sprint 3

---

## 8.3 MCP Transport Bridge

**Status:** [x] BUILT
**Path:** `core/bridges/` (MCP client, BrowserMCPClient)

Model Context Protocol client for browser and tool integration.
Used by MissionOrchestrator for web interactions.

---

## 8.4 Telescript Mobile Agents (Rust)

**Status:** [x] BUILT
**Path:** `bizra-omega/bizra-telescript/`

Mobile agent scripts that can migrate between nodes.
Rust implementation with serializable agent state.

Standing on Giants: General Magic — Telescript

---

## 8.5 Mission Bridge Server

**Status:** [~] PARTIAL
**Path:** `scripts/start_mission_bridge.sh`, model warmup, systemd service
**Built:** Bridge process, port management, basic startup
**Gap:** No auto-reconnect, no health monitoring of bridge process

### TDD Anchor
```
def test_mission_bridge_reconnect():
    bridge = MissionBridge(port=9742)
    bridge.start()
    bridge.simulate_disconnect()
    assert bridge.reconnect(timeout_seconds=5)
    assert bridge.is_healthy()
```

---

## 8.6 Ghost Overlay (Transparent UI)

**Status:** [~] PARTIAL
**Path:** Some overlay concept exists in frontend specs
**Gap:** No transparent always-on-top overlay for desktop. Spec describes a
semi-transparent widget showing agent status, notifications, mission progress.

### Pseudocode
```
# Tauri or Electron-based ghost overlay
class GhostOverlay:
    """Always-on-top transparent overlay for agent status"""

    def __init__(self):
        self.window = create_window(
            transparent=True,
            always_on_top=True,
            click_through=True,  # Mouse passes through
            position="bottom-right"
        )

    def show_notification(self, msg: str, duration_ms: int = 3000):
        self.window.animate_in(msg)
        schedule(self.window.animate_out, delay_ms=duration_ms)

    def show_agent_status(self, agents: List[AgentStatus]):
        self.window.update_status_bar(agents)
```

---

## 8.7 Tauri Desktop Host App

**Status:** [ ] NOT BUILT
**Path:** `filedfs/tauri.conf.json` (config file only)
**Spec:** Phase 43 — 4-layer runtime capsule with Tauri as desktop host
**Gap:** Config exists but no Tauri application code. No installer, tray, wallet.

### Pseudocode
```
// src-tauri/src/main.rs
fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize sovereign runtime
            let runtime = SovereignRuntime::new()?;
            app.manage(runtime);

            // System tray with sovereignty status
            let tray = SystemTray::new()
                .with_menu(tray_menu())
                .with_tooltip("BIZRA Node0 — SEED tier");
            app.tray_handle().set_menu(tray);

            // Start proactive kernel in background
            std::thread::spawn(move || {
                runtime.start_proactive_loop();
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_sovereignty_status,
            execute_mission,
            get_agent_status,
            manage_wallet,
        ])
        .run(tauri::generate_context!())
}
```

### TDD Anchors
```
#[test]
fn test_tauri_app_starts() {
    let app = create_test_app();
    assert!(app.state::<SovereignRuntime>().is_ok());
}

#[test]
fn test_tray_shows_tier() {
    let app = create_test_app();
    let tray = app.tray_handle();
    assert!(tray.tooltip().contains("SEED") || tray.tooltip().contains("SPROUT"));
}
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 8.1 HDA Client | BUILT | Async TCP |
| 8.2 AHK Bridge | BUILT | Hotkey |
| 8.3 MCP Transport | BUILT | Browser |
| 8.4 Telescript | BUILT | Rust |
| 8.5 Mission Bridge | PARTIAL | No reconnect |
| 8.6 Ghost Overlay | PARTIAL | No UI |
| 8.7 Tauri Host | NOT BUILT | Config only |
| **TOTAL** | **4/7 + 2P + 1N** | **71%** |
