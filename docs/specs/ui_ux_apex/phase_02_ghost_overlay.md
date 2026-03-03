# UI/UX APEX — Phase 02: Ghost Overlay (PROTOTYPE FIRST)

> AHK/HDA-powered proactive HUD projected over any active application.
> Sprint priority: 1 — highest immediate Phantom Limb impact.

> Standing on Giants: Fitts (target acquisition, 1954) · Norman (affordance + feedback, 1988) ·
> Gibson (affordances in the wild, 1979) · Engelbart (augmented intellect, 1962)
> Repo anchors: `bizra-omega/bizra-action/src/channels/mod.rs` (AHK channel),
>               `core/orchestration/`, `scripts/node0_activate.py`

---

## 1. Functional Requirements

| ID | Requirement |
|----|-------------|
| GO-01 | Overlay renders as a frosted-glass HUD on top of any foreground window |
| GO-02 | BIZRA proactively suggests actions based on HHMM predicted intent |
| GO-03 | Overlay does NOT capture keystrokes from the target application |
| GO-04 | Sovereign Gesture (hotkey or mouse flick) dismisses OR solidifies overlay |
| GO-05 | Solidified action passes through ConstitutionalGate (Ihsān ≥ 0.95) before execution |
| GO-06 | Action is dispatched to AHK channel via Action Bus (JSON-RPC over stdio) |
| GO-07 | Overlay auto-dismisses after `GHOST_IDLE_TIMEOUT_MS` with no gesture |
| GO-08 | Multiple suggestions shown as ranked cards (max 3, by HHMM confidence) |
| GO-09 | Each card shows: action label, predicted intent, confidence %, Ihsān pre-check |
| GO-10 | Overlay position is context-sensitive (near cursor or target element region) |
| GO-11 | "Veto Receipt" appears in Iḥsān Gauge when action is blocked |

---

## 2. Edge Cases & Constraints

```
EDGE CASE: Foreground app is full-screen exclusive (game, video) →
           Overlay renders in windowed overlay mode (WS_EX_LAYERED + WS_EX_TOPMOST),
           only if target app is NOT full-screen exclusive (detected via DXGI).
           If full-screen exclusive: suppress overlay silently.

EDGE CASE: HHMM confidence < UNIFIED_SNR_THRESHOLD (0.85) →
           Overlay not shown; prediction logged but suppressed.

EDGE CASE: ConstitutionalGate REJECTED →
           Overlay shows "BLOCKED" state with veto reason.
           No action dispatched. Emits Iḥsān Gauge event.

EDGE CASE: AHK channel unavailable (daemon not running) →
           Overlay shows warning badge; suggestions are preview-only (no dispatch).

EDGE CASE: Multiple rapid HHMM predictions within 500ms →
           Debounce: take highest-confidence prediction only.

EDGE CASE: User triggers Sovereign Gesture on BLOCKED card →
           Show ConstitutionalGate explanation; offer appeal (manual override workflow).

CONSTRAINT: Overlay window class must NOT be the same as target app class
            (prevents recursive overlay-on-overlay).
CONSTRAINT: Overlay reads context from screen region via accessibility APIs
            (UI Automation / AXUIElement), NOT keystroke capture.
CONSTRAINT: All config values (hotkey, timeout, threshold) from config/proactive_config.yaml,
            never hardcoded.
CONSTRAINT: Sovereign Gesture for solidify: Win+Shift+G (default, user-configurable).
CONSTRAINT: Sovereign Gesture for dismiss: Esc (always active while overlay visible).
```

---

## 3. Data Model

```typescript
// Ghost Overlay — client types

type SovereignGesture = "solidify" | "dismiss" | "scroll_next" | "scroll_prev";

interface OverlaySuggestion {
  id: string;                  // UUID
  action_label: string;        // e.g. "Batch Rename selected cells"
  intent_summary: string;      // e.g. "HHMM: batch_rename (87% confident)"
  hhmm_confidence: number;     // 0-1
  ihsan_precheck: "pass" | "pending" | "blocked";
  ihsan_score: number;         // 0-1
  block_reason?: string;       // present when ihsan_precheck == "blocked"
  ahk_action_id: string;       // BizraAction id to dispatch on solidify
  target_region?: DOMRect;     // optional highlight region in target app
}

interface GhostOverlayState {
  visible: boolean;
  suggestions: OverlaySuggestion[];  // max 3, sorted by hhmm_confidence desc
  active_index: number;              // which card is focused
  auto_dismiss_at: number;           // unix ms
  position: { x: number; y: number };
}
```

---

## 4. Pseudocode

### 4.1 GhostOverlayDaemon (Python — Node0 integration point)

```
MODULE GhostOverlayDaemon:
  // Runs as a coroutine inside the Node0 proactive loop.
  // Observes active window + HHMM predictions → emits overlay events.

  CONSTANTS (from config, never hardcoded):
    GHOST_IDLE_TIMEOUT_MS    // e.g. 5000
    GHOST_DEBOUNCE_MS        // e.g. 500
    MAX_SUGGESTIONS          // 3
    HHMM_MIN_CONFIDENCE      // = UNIFIED_SNR_THRESHOLD (from constants.py)
    IHSAN_GATE_THRESHOLD     // = UNIFIED_IHSAN_THRESHOLD (from constants.py)

  STATE:
    pending_prediction: Optional[Prediction]
    debounce_timer: Timer
    overlay_visible: bool = False

  ASYNC FUNCTION run():
    SUBSCRIBE to hhmm_prediction_stream():
      ON prediction(intent, confidence, context):
        IF confidence < HHMM_MIN_CONFIDENCE:
          LOG debug("Suppressed low-confidence prediction", confidence)
          RETURN
        // Debounce
        cancel(debounce_timer)
        pending_prediction = {intent, confidence, context}
        debounce_timer = schedule(GHOST_DEBOUNCE_MS, emit_overlay)

  ASYNC FUNCTION emit_overlay():
    IF overlay_visible:
      RETURN  // don't stack overlays

    suggestions = generate_suggestions(pending_prediction)
    gated = []
    FOR s in suggestions[:MAX_SUGGESTIONS]:
      result = await constitutional_gate.check(s.ahk_action_id, s.intent_summary)
      s.ihsan_precheck = "pass" IF result.approved ELSE "blocked"
      s.ihsan_score    = result.ihsan_score
      s.block_reason   = result.reason IF NOT result.approved
      gated.append(s)

    cursor_pos = get_cursor_screen_position()
    overlay_event = {
      type: "show_overlay",
      suggestions: gated,
      position: cursor_pos,
      auto_dismiss_at: now() + GHOST_IDLE_TIMEOUT_MS
    }
    emit_to_ui("ghost_overlay", overlay_event)
    overlay_visible = True

  FUNCTION generate_suggestions(prediction) -> List[OverlaySuggestion]:
    // Map HHMM intent to candidate BizraActions
    action_templates = lookup_intent_templates(prediction.intent)
    // Fill in context from active window (via UIAutomation/AX)
    context = read_active_window_context()
    suggestions = []
    FOR template in action_templates:
      action = materialize_action(template, context)
      suggestions.append(OverlaySuggestion(
        id           = uuid4(),
        action_label = action.label,
        intent_summary = f"HHMM: {prediction.intent} ({int(prediction.confidence*100)}% confident)",
        hhmm_confidence = prediction.confidence,
        ihsan_precheck  = "pending",
        ahk_action_id   = action.id,
        target_region   = context.highlighted_region,
      ))
    RETURN sorted(suggestions, key=lambda s: s.hhmm_confidence, reverse=True)
```

### 4.2 GhostOverlayUI (Electron/AHK overlay window)

```
MODULE GhostOverlayUI:
  // Rendered as WS_EX_LAYERED + WS_EX_TOPMOST window.
  // Receives events from GhostOverlayDaemon via IPC/WebSocket.

  STATE:
    overlay: GhostOverlayState
    dismiss_timer: Timer

  FUNCTION on_show_overlay(event):
    IF target_app_is_fullscreen_exclusive():
      RETURN  // suppress silently

    overlay.suggestions = event.suggestions
    overlay.position    = snap_to_safe_region(event.position)
    overlay.active_index = 0
    overlay.visible = True
    overlay.auto_dismiss_at = event.auto_dismiss_at
    render_overlay()
    start_dismiss_timer(event.auto_dismiss_at - now())

  FUNCTION render_overlay():
    // Frosted glass container
    draw_glass_panel(
      position  = overlay.position,
      width     = 360,
      height    = 80 + 72 * len(overlay.suggestions),
      style     = { background: "rgba(8,14,27,0.88)", blur: 12, border: "#C9A962 at 18% opacity" }
    )
    // Header
    draw_text("Sovereign Suggestion", font="Playfair Display 13px", color="#C9A962")

    FOR i, suggestion in enumerate(overlay.suggestions):
      is_active = (i == overlay.active_index)
      draw_suggestion_card(suggestion, is_active)

    // Gesture hint footer
    draw_text("Win+Shift+G to act  ·  Esc to dismiss", font="Inter 11px", color="#6e6a60")

  FUNCTION draw_suggestion_card(s, is_active):
    bg_color = "rgba(201,169,98,0.07)" IF is_active ELSE "rgba(255,255,255,0.02)"
    border   = "#C9A962 at 40%" IF is_active ELSE "#C9A962 at 10%"
    draw_glass_card(bg=bg_color, border=border)
    draw_text(s.action_label,     font="Inter 13px 500", color="#F8F4EC")
    draw_text(s.intent_summary,   font="JetBrains Mono 11px", color="#6e6a60")

    IF s.ihsan_precheck == "pass":
      draw_badge("✓ Iḥsān " + format(s.ihsan_score, ".2f"), color="#2eb86a")
    ELIF s.ihsan_precheck == "blocked":
      draw_badge("✗ BLOCKED", color="#c93a4a")
      draw_text(s.block_reason, font="Inter 11px", color="#c93a4a")
    ELSE:  // pending
      draw_spinner(color="#C9A962")

  FUNCTION on_sovereign_gesture(gesture):
    IF gesture == "dismiss":
      hide_overlay()
    ELIF gesture == "solidify":
      active = overlay.suggestions[overlay.active_index]
      IF active.ihsan_precheck == "pass":
        dispatch_action(active.ahk_action_id)
        show_dispatch_flash()
        hide_overlay()
      ELIF active.ihsan_precheck == "blocked":
        show_veto_explanation(active)
    ELIF gesture == "scroll_next":
      overlay.active_index = (overlay.active_index + 1) % len(overlay.suggestions)
      render_overlay()
    ELIF gesture == "scroll_prev":
      overlay.active_index = (overlay.active_index - 1) % len(overlay.suggestions)
      render_overlay()

  FUNCTION dispatch_action(action_id):
    // Send to Action Bus AHK channel
    payload = {
      channel: "Ahk",
      action_id: action_id,
      permit_scope: "ghost_overlay"
    }
    ipc_send("action_bus", payload)
    emit_event_emitter("action_completed", action_id)

  FUNCTION show_dispatch_flash():
    // 600ms gold flash on overlay border, then fade
    animate(border_color, from="#C9A962", to="transparent", duration=600)

  FUNCTION hide_overlay():
    overlay.visible = False
    cancel(dismiss_timer)
    clear_glass_panel()
```

### 4.3 AHK Channel Integration (Rust — production stub reference)

```
// In bizra-omega/bizra-action/src/channels/mod.rs (production implementation):
//
// AhkChannel implements ChannelHandler:
//   - Spawns AutoHotkey v2 runtime as a subprocess
//   - Sends BizraAction as JSON over stdio pipe
//   - Reads result JSON from stdout
//   - Returns ActionPayload::Text(result) or ChannelError

// The Ghost Overlay uses Channel::Ahk exclusively.
// BizraAction variants dispatched by Ghost Overlay:
//   BatchRename  { files: Vec<String>, pattern: String }
//   MergeRegion  { app: String, region: DOMRect }
//   AutoFill     { app: String, field: String, value_template: String }
// (Defined in bizra-action/src/types.rs — to be extended in next sprint)
```

---

## 5. TDD Anchors

```python
# tests/ui_ux_apex/test_ghost_overlay.py

class TestGhostOverlayDaemon:
    def test_low_confidence_suppressed(self, daemon, mock_hhmm):
        """Predictions below HHMM_MIN_CONFIDENCE do not trigger overlay."""
        from core.integration.constants import UNIFIED_SNR_THRESHOLD
        mock_hhmm.emit(intent="batch_rename", confidence=UNIFIED_SNR_THRESHOLD - 0.01)
        daemon.run_tick()
        assert not daemon.overlay_visible

    def test_debounce_drops_rapid_predictions(self, daemon, mock_hhmm):
        """Two predictions within GHOST_DEBOUNCE_MS → only last one used."""
        mock_hhmm.emit(intent="sort", confidence=0.88)
        mock_hhmm.emit(intent="batch_rename", confidence=0.91)
        daemon.advance_time(daemon.GHOST_DEBOUNCE_MS + 1)
        assert daemon.last_emitted_intent == "batch_rename"

    def test_constitutional_gate_called_per_suggestion(self, daemon, mock_gate, mock_hhmm):
        """Each suggestion passes through ConstitutionalGate."""
        mock_hhmm.emit(intent="batch_rename", confidence=0.92)
        daemon.run_tick()
        assert mock_gate.check.call_count >= 1

    def test_blocked_suggestion_no_dispatch(self, daemon, mock_gate, mock_ahk_channel):
        """Solidifying a BLOCKED suggestion must not dispatch to AHK channel."""
        mock_gate.return_approved = False
        daemon.solidify_active()
        assert mock_ahk_channel.execute.not_called()

class TestGhostOverlayUI:
    def test_fullscreen_exclusive_suppressed(self, overlay_ui, mock_dxgi):
        """Overlay invisible when target app is fullscreen exclusive."""
        mock_dxgi.is_fullscreen_exclusive = True
        overlay_ui.on_show_overlay(sample_event())
        assert not overlay_ui.overlay.visible

    def test_escape_dismisses(self, overlay_ui):
        overlay_ui.on_show_overlay(sample_event())
        assert overlay_ui.overlay.visible
        overlay_ui.on_sovereign_gesture("dismiss")
        assert not overlay_ui.overlay.visible

    def test_max_3_suggestions_rendered(self, overlay_ui):
        event = sample_event(n_suggestions=5)
        overlay_ui.on_show_overlay(event)
        assert len(overlay_ui.overlay.suggestions) <= 3

    def test_no_hardcoded_hotkey(self, overlay_ui_source):
        """Win+Shift+G loaded from config, not hardcoded in source."""
        assert "Win+Shift+G" not in overlay_ui_source
```
