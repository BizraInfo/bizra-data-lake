# Phase 57.04: Component Wiring — Pseudocode

## Boot Sequence

The MissionOrchestrator must be wired into the existing DesktopBridge
at startup. This is the initialization code that connects everything.

```python
# core/sovereign/mission_boot.py
"""
Boot sequence for the Mission system.
Called from node0_activate.py during startup.

Order of operations:
1. Initialize LivingMemory (SQLite)
2. Create MissionOrchestrator
3. Connect to AHK HDA (optional — demo works without it)
4. Wire into DesktopBridge as a new RPC method
5. Emit system_ready event
"""

async def boot_mission_system(bridge: DesktopBridge, config: dict):
    """
    Wire the MissionOrchestrator into an existing DesktopBridge.

    Parameters:
        bridge: Running DesktopBridge instance (already on port 9742)
        config: {
            "memory_path": Path to LivingMemory storage,
            "evidence_path": Path to evidence ledger JSONL,
            "hda_port": int (default 9743),
            "gateway": Optional InferenceGateway,
        }
    """
    from core.sovereign.mission import MissionOrchestrator

    orchestrator = MissionOrchestrator(config)
    await orchestrator.initialize()

    # If inference gateway exists, inject it for LLM synthesis
    if config.get("gateway"):
        orchestrator.gateway = config["gateway"]

    # Register the execute_mission method on the bridge
    bridge.register_method("execute_mission", orchestrator.handle_rpc)

    return orchestrator
```

## DesktopBridge Integration

```python
# Additions to core/bridges/desktop_bridge.py
# In the _dispatch_method() handler:

class DesktopBridge:
    # ... existing code ...

    def __init__(self, ...):
        # ... existing init ...
        self._extra_methods: dict[str, Callable] = {}

    def register_method(self, name: str, handler: Callable):
        """Register an additional RPC method dynamically."""
        self._extra_methods[name] = handler

    async def _dispatch_method(self, method: str, params: dict) -> dict:
        """Route JSON-RPC method to handler."""
        # Existing methods
        if method == "ping":
            return await self._handle_ping(params)
        elif method == "status":
            return await self._handle_status(params)
        elif method == "sovereign_query":
            return await self._handle_sovereign_query(params)
        elif method == "invoke_skill":
            return await self._handle_invoke_skill(params)
        elif method == "list_skills":
            return await self._handle_list_skills(params)
        elif method == "get_receipt":
            return await self._handle_get_receipt(params)
        # NEW: Dynamic method routing
        elif method in self._extra_methods:
            return await self._extra_methods[method](params)
        else:
            raise ValueError(f"Unknown method: {method}")
```

## MissionOrchestrator RPC Handler

```python
# In core/sovereign/mission.py

class MissionOrchestrator:
    # ... core pipeline from phase_57_03 ...

    async def handle_rpc(self, params: dict) -> dict:
        """
        JSON-RPC entry point. Called by DesktopBridge when
        AHK client sends execute_mission.

        params = {
            "description": "Research AI agent frameworks...",
            "context": {  # Optional, captured by AHK client
                "active_window": "Visual Studio Code",
                "clipboard": "some text",
            }
        }
        """
        description = params.get("description", "")
        if not description:
            return {"error": "Missing 'description' parameter"}

        # Build MissionRequest
        mission_id = secrets.token_hex(16)  # 32-char hex

        context = DesktopContext(
            active_window_title=params.get("context", {}).get(
                "active_window", "unknown"
            ),
            clipboard_text=params.get("context", {}).get(
                "clipboard", ""
            )[:4096],
            screen_geometry=params.get("context", {}).get(
                "screen", {}
            ),
        )

        request = MissionRequest(
            mission_id=mission_id,
            description=description,
            context=context,
            timestamp=time.time(),
            source="ahk_hotkey",
        )

        # Execute the full pipeline
        result = await self.execute(request)

        # Return JSON-RPC compatible response
        return {
            "mission_id": result.mission_id,
            "status": result.status,
            "synthesis": result.synthesis[:2000],  # Truncate for RPC
            "briefing_path": result.briefing_path,
            "evidence_receipt_id": result.evidence_receipt_id,
            "ihsan_score": result.ihsan_score,
            "snr_score": result.snr_score,
            "duration_ms": result.duration_ms,
            "channels": [
                {
                    "channel": cr.channel,
                    "success": cr.success,
                    "duration_ms": cr.duration_ms,
                }
                for cr in result.channels_executed
            ],
        }
```

## AHK Client Addition

```autohotkey
; Addition to bin/bizra_bridge.ahk
; New hotkey: Win+Shift+B → Execute Mission

#!b::  ; Win+Shift+B (# = Win, ! = Shift, b = B)
{
    ; Step 1: Capture desktop context BEFORE showing input box
    active_title := WinGetTitle("A")
    clipboard_text := A_Clipboard

    ; Step 2: Get mission description from user
    mission_desc := ""
    ib := InputBox("What should BIZRA do?", "BIZRA Mission", "w400 h120")
    if ib.Result = "Cancel"
        return
    mission_desc := ib.Value

    if (mission_desc = "")
        return

    ; Step 3: Build execute_mission request with context
    params := '{'
    params .= '"description": "' . EscapeJson(mission_desc) . '",'
    params .= '"context": {'
    params .= '  "active_window": "' . EscapeJson(active_title) . '",'
    params .= '  "clipboard": "' . EscapeJson(SubStr(clipboard_text, 1, 4096)) . '"'
    params .= '}}'

    ; Step 4: Show "working" tooltip
    ToolTip("BIZRA: Executing mission...")

    ; Step 5: Send to Python bridge
    result := SendBizraCommand("execute_mission", params)

    ; Step 6: Display result
    ToolTip("")  ; Clear working tooltip

    if (result.Has("error")) {
        MsgBox("Mission failed: " . result["error"], "BIZRA Error")
        return
    }

    ; Show toast-style notification
    status := result.Has("status") ? result["status"] : "UNKNOWN"
    ihsan := result.Has("ihsan_score") ? Round(result["ihsan_score"], 3) : "N/A"
    duration := result.Has("duration_ms") ? Round(result["duration_ms"]) : "N/A"
    briefing := result.Has("briefing_path") ? result["briefing_path"] : ""
    receipt := result.Has("evidence_receipt_id") ? result["evidence_receipt_id"] : ""

    msg := "Mission: " . status . "`n"
    msg .= "Ihsan: " . ihsan . "`n"
    msg .= "Duration: " . duration . "ms`n"
    if (briefing != "")
        msg .= "Briefing: " . briefing . "`n"
    if (receipt != "")
        msg .= "Receipt: " . SubStr(receipt, 1, 16) . "..."

    ToolTip(msg)
    SetTimer(() => ToolTip(""), -8000)  ; Clear after 8 seconds

    ; Step 7: Open briefing file if created
    if (briefing != "") {
        ; Convert WSL path to Windows path for opening
        win_path := StrReplace(briefing, "/mnt/c/", "C:\")
        win_path := StrReplace(win_path, "/", "\")
        Run(win_path)
    }
}
```

## Event Bus Wiring

```python
# Mission lifecycle events emitted by MissionOrchestrator
# These integrate with the existing 12 Rust subscribers

MISSION_TOPICS = {
    "mission.system_ready",    # Boot complete
    "mission.started",         # User initiated a mission
    "mission.decomposed",      # Channels identified
    "mission.channel_started", # Individual channel executing
    "mission.channel_done",    # Individual channel complete
    "mission.synthesized",     # Results merged
    "mission.gate_passed",     # Ihsan/SNR gate passed
    "mission.gate_failed",     # Gate rejected — recovery attempted
    "mission.evidence_emitted",# Receipt written to ledger
    "mission.completed",       # Full pipeline done
    "mission.failed",          # Unrecoverable failure
}

# Optional: Wire to Rust event bus for constitutional monitoring
# The Rust bus already has ihsan.breach and action.intent subscribers
# Mission events flow through Python bus → Rust bridge → 12 subscribers
```

## Node0 Integration Point

```python
# In scripts/node0_activate.py — the existing entrypoint
# Add mission boot to the startup sequence

class Node0ProactiveKernel:
    async def _boot(self):
        # ... existing boot sequence ...

        # After DesktopBridge is created and listening:
        if self.bridge:
            from core.sovereign.mission_boot import boot_mission_system
            self.mission_orchestrator = await boot_mission_system(
                bridge=self.bridge,
                config={
                    "memory_path": self.config.memory_path,
                    "evidence_path": self.config.evidence_path,
                    "hda_port": 9743,
                    "gateway": self.inference_gateway,
                },
            )
            self.logger.info("Mission system wired to DesktopBridge")
```

## Graceful Degradation Matrix

The system must work at every level of capability:

```
Level 0 (Minimum viable): Python only, no AHK, no LLM
  → Template synthesis, direct file I/O, mock browser
  → WORKS: Creates briefing file with mock research data
  → PROOF: Evidence receipt still generated

Level 1 (+ Browser): Python + httpx installed
  → Real DuckDuckGo search via BrowserMCPClient direct mode
  → WORKS: Real research results in briefing
  → PROOF: Evidence + real web sources

Level 2 (+ HDA): Python + AHK HDA running on Windows
  → Desktop context capture (active window, clipboard)
  → Desktop file opening after briefing created
  → WORKS: Context-aware research + auto-open result

Level 3 (+ LLM): Python + LM Studio/Ollama connected
  → LLM-powered synthesis instead of template
  → Quality scoring with real content analysis
  → WORKS: Professional-grade briefing output

Level 4 (Full Stack): All of the above + Rust bus + PCI gates
  → Constitutional enforcement end-to-end
  → 12 Rust subscriber notifications
  → Hash-chained evidence with PCI envelope
  → WORKS: The full BIZRA sovereignty experience
```
