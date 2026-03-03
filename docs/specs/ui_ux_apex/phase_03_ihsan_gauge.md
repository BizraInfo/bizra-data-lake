# UI/UX APEX — Phase 03: Iḥsān Gauge — The Ethical Mirror

> Real-time excellence telemetry with Veto Receipt system.
> Sprint priority: 2 (immediately after Ghost Overlay — closes the ethical feedback loop).

> Standing on Giants: Al-Ghazali (Iḥsān, 1095) · Kahneman (System 1/2 feedback, 2011) ·
> Norman (feedback loop design, 1988) · Anthropic (constitutional AI, 2023)
> Repo anchors: `core/governance/constitutional_gate.py`,
>               `core/integration/constants.py` (UNIFIED_IHSAN_THRESHOLD, etc.)

---

## 1. Functional Requirements

| ID | Requirement |
|----|-------------|
| IG-01 | Persistent gauge widget visible in BIZRA chrome at all times |
| IG-02 | Gauge shows session rolling-average Iḥsān score (0–1), updated per action |
| IG-03 | Gauge animates smoothly between score updates (200ms ease-out) |
| IG-04 | Color band: 0–0.85 → red, 0.85–0.95 → amber, 0.95–0.99 → green, 0.99–1.0 → gold |
| IG-05 | "Veto Receipt" panel appears when ConstitutionalGate REJECTED an action |
| IG-06 | Veto Receipt shows: timestamp, blocked action, Iḥsān score, gate reason, SNR score |
| IG-07 | Veto Receipt uses JetBrains Mono, styled as a printed receipt / audit log |
| IG-08 | Receipt message format: "I protected your sovereignty because [reason]" |
| IG-09 | Historical Veto Receipts accessible in a scrollable "Guardian Log" |
| IG-10 | Gauge icon is the Guardian seal (SVG lock + Seed of Life overlay) |
| IG-11 | Score explainer tooltip: shows dimension breakdown on hover |

---

## 2. Edge Cases & Constraints

```
EDGE CASE: No actions in session → gauge shows dashes (—) not 0.00
EDGE CASE: Iḥsān score temporarily below 0.85 (degraded run) →
           Gauge border pulses red; "Guardian Alert" toast notification shown
EDGE CASE: Rapid-fire actions (>10/sec) → throttle UI updates to max 5/sec
EDGE CASE: Long gate reason string → truncate at 200 chars in Veto Receipt,
           "Show full reason" expander available
EDGE CASE: No network (offline mode) → gauge shows cached last score with "OFFLINE" badge
CONSTRAINT: Iḥsān thresholds are NOT hardcoded in UI;
            fetched from /api/v1/config/thresholds on startup (sourced from constants.py)
CONSTRAINT: Veto Receipts stored in local IndexedDB (encrypted at rest); max 500 entries
CONSTRAINT: Guardian Log is read-only; no delete action in UI
CONSTRAINT: Gauge widget must not exceed 120×120px in compact mode
```

---

## 3. Data Model

```typescript
// Iḥsān Gauge — client types

interface IhsanThresholds {
  minimum:     number;   // 0.85 UNIFIED_SNR_THRESHOLD
  production:  number;   // 0.95 UNIFIED_IHSAN_THRESHOLD
  strict:      number;   // 0.99 STRICT_IHSAN_THRESHOLD
}

interface ActionScore {
  action_id:       string;
  action_label:    string;
  ihsan_score:     number;
  snr_score:       number;
  gate_result:     "APPROVED" | "NEEDS_REVIEW" | "REJECTED";
  timestamp:       number;   // unix ms
}

interface VetoReceipt {
  id:              string;   // UUID
  timestamp:       number;
  blocked_action:  string;
  ihsan_score:     number;
  snr_score:       number;
  gate_reason:     string;
  full_reason:     string;   // untruncated
  dimension_scores: IhsanDimensions;
}

interface IhsanDimensions {
  signal_density:        number;   // weight 0.35
  evidence_grounding:    number;   // weight 0.25
  contradiction_res:     number;   // weight 0.20
  actionability:         number;   // weight 0.20
}

interface GaugeState {
  session_average:  number;         // rolling mean
  last_score:       ActionScore | null;
  thresholds:       IhsanThresholds;
  action_history:   ActionScore[];  // last 100
  veto_receipts:    VetoReceipt[];  // last 500, local
}
```

---

## 4. Pseudocode

### 4.1 IhsanGaugeController

```
MODULE IhsanGaugeController:

  STATE:
    state: GaugeState
    thresholds: IhsanThresholds   // fetched from API on init
    rolling_window: Deque[float]  // max 100 entries

  FUNCTION init():
    thresholds = await fetch_thresholds("/api/v1/config/thresholds")
    history    = await load_local_history(IndexedDB, key="ihsan_action_history")
    receipts   = await load_local_history(IndexedDB, key="ihsan_veto_receipts")
    state = GaugeState(
      session_average = compute_rolling_average(history),
      last_score      = history[-1] IF history ELSE null,
      thresholds      = thresholds,
      action_history  = history,
      veto_receipts   = receipts,
    )
    render_gauge()

    // Subscribe to live action scores from Node0
    ws = open_kernel_ws("/api/v1/ihsan/stream")
    ws.on("action_scored", on_action_scored)
    ws.on("veto_receipt",  on_veto_receipt)

  FUNCTION on_action_scored(event: ActionScore):
    rolling_window.append(event.ihsan_score)
    IF len(rolling_window) > 100:
      rolling_window.popleft()
    state.session_average = mean(rolling_window)
    state.last_score      = event
    state.action_history.append(event)
    save_local(IndexedDB, "ihsan_action_history", state.action_history)
    render_gauge(animate=True)

    IF state.session_average < thresholds.minimum:
      show_guardian_alert()

  FUNCTION on_veto_receipt(receipt: VetoReceipt):
    state.veto_receipts.prepend(receipt)
    IF len(state.veto_receipts) > 500:
      state.veto_receipts = state.veto_receipts[:500]
    save_local(IndexedDB, "ihsan_veto_receipts", state.veto_receipts)
    show_veto_receipt_panel(receipt)
    render_gauge()

  FUNCTION render_gauge():
    score = state.session_average
    color = gauge_color(score, thresholds)
    gauge_widget.update(score=score, color=color, animate=True)

  FUNCTION gauge_color(score, thresholds) -> string:
    IF score >= thresholds.strict:      RETURN #C9A962  // Genesis Gold (masterpiece)
    IF score >= thresholds.production:  RETURN #2eb86a  // Operational Green
    IF score >= thresholds.minimum:     RETURN #c97e2e  // Amber
    RETURN #c93a4a                                       // Alert Red

  FUNCTION show_guardian_alert():
    toast_notification(
      title   = "Guardian Alert",
      message = f"Session Iḥsān ({format(state.session_average, '.2f')}) "
                f"below minimum threshold {thresholds.minimum}",
      style   = { border: #c93a4a, icon: guardian_seal_red_svg },
      duration = 6000
    )
    pulse_gauge_border(color=#c93a4a, cycles=3)
```

### 4.2 VetoReceiptPanel

```
MODULE VetoReceiptPanel:

  FUNCTION show_veto_receipt_panel(receipt: VetoReceipt):
    panel = create_panel(
      title   = "Guardian Veto Receipt",
      style   = { background: #0a1220, border: #c93a4a at 40%, font: "JetBrains Mono" },
      width   = 420,
      height  = "auto",
    )
    render_receipt_content(panel, receipt)
    animate_slide_in(panel, from="bottom-right")

  FUNCTION render_receipt_content(panel, receipt):
    // Receipt header — timestamp and guardian seal icon
    draw_guardian_seal_icon(color=#c93a4a, size=32)
    draw_text(
      format_receipt_header(receipt.timestamp),
      font="JetBrains Mono 11px", color="#6e6a60"
    )

    // Sovereignty statement (the emotional hook)
    draw_text(
      "I protected your sovereignty because",
      font="Playfair Display italic 14px", color="#F8F4EC"
    )
    draw_text(
      truncate(receipt.gate_reason, 200),
      font="JetBrains Mono 12px", color="#c97e2e"
    )
    IF len(receipt.full_reason) > 200:
      draw_expander("Show full reason", FUNCTION():
        panel.expand_text(receipt.full_reason)
      )

    // Score block
    draw_separator()
    draw_score_row("Iḥsān Score",  receipt.ihsan_score, threshold=thresholds.production)
    draw_score_row("SNR Score",    receipt.snr_score,   threshold=thresholds.minimum)

    // Dimension breakdown
    draw_dimension_bars(receipt.dimension_scores)

    // Blocked action label
    draw_separator()
    draw_text("BLOCKED ACTION:", font="JetBrains Mono 11px 600", color="#6e6a60")
    draw_text(receipt.blocked_action, font="JetBrains Mono 12px", color="#F8F4EC")

    // Footer
    draw_text(
      "This receipt is sealed in your Guardian Log.",
      font="Inter 11px", color="#3a3730"
    )
    draw_close_button()

  FUNCTION format_receipt_header(timestamp_ms) -> string:
    dt = format_iso8601(timestamp_ms)
    RETURN f"RECEIPT · {dt} · Node0 Constitutional Gate"

  FUNCTION draw_score_row(label, score, threshold):
    color = #2eb86a IF score >= threshold ELSE #c93a4a
    draw_text(f"{label}:  {format(score, '.4f')}", color=color)
```

### 4.3 GuardianLog (scrollable historical receipts)

```
MODULE GuardianLog:

  FUNCTION open():
    panel = create_fullscreen_panel(
      title = "Guardian Log — Veto History",
      style = { background: #050B14 }
    )
    receipts = load_local(IndexedDB, "ihsan_veto_receipts")
    IF len(receipts) == 0:
      draw_empty_state(
        icon    = guardian_seal_svg,
        message = "No vetoes recorded. Your sovereignty is intact."
      )
      RETURN

    FOR receipt in receipts:
      draw_receipt_row(receipt)

  FUNCTION draw_receipt_row(receipt):
    row = draw_row(
      left  = format_timestamp_relative(receipt.timestamp),
      mid   = truncate(receipt.blocked_action, 60),
      right = format(receipt.ihsan_score, ".3f"),
      color = #c93a4a IF receipt.ihsan_score < thresholds.production ELSE #c97e2e,
    )
    row.on_click(FUNCTION(): show_veto_receipt_panel(receipt))
```

### 4.4 Iḥsān Stream Endpoint (Python — Node0)

```
MODULE IhsanStreamEndpoint:
  // WebSocket: /api/v1/ihsan/stream
  // Emits action_scored and veto_receipt events in real-time.

  ON connect(ws):
    SUBSCRIBE to constitutional_gate_events():
      ON gate_result(event):
        IF event.result == "APPROVED" OR event.result == "NEEDS_REVIEW":
          score = ActionScore(
            action_id    = event.action_id,
            action_label = event.action_label,
            ihsan_score  = event.ihsan_score,
            snr_score    = event.snr_score,
            gate_result  = event.result,
            timestamp    = now_ms(),
          )
          ws.send("action_scored", score)

        ELIF event.result == "REJECTED":
          receipt = VetoReceipt(
            id             = uuid4(),
            timestamp      = now_ms(),
            blocked_action = event.action_label,
            ihsan_score    = event.ihsan_score,
            snr_score      = event.snr_score,
            gate_reason    = event.reason[:200],
            full_reason    = event.reason,
            dimension_scores = event.dimensions,
          )
          ws.send("veto_receipt", receipt)
```

---

## 5. TDD Anchors

```python
# tests/ui_ux_apex/test_ihsan_gauge.py

class TestIhsanGaugeController:
    def test_thresholds_not_hardcoded(self, gauge):
        """Thresholds fetched from API, not stored as literals in gauge source."""
        from core.integration.constants import (
            UNIFIED_IHSAN_THRESHOLD, STRICT_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD)
        assert gauge.thresholds.production == UNIFIED_IHSAN_THRESHOLD
        assert gauge.thresholds.strict     == STRICT_IHSAN_THRESHOLD
        assert gauge.thresholds.minimum    == UNIFIED_SNR_THRESHOLD

    def test_color_gold_at_strict_threshold(self, gauge):
        """Score >= STRICT_IHSAN_THRESHOLD → Genesis Gold #C9A962."""
        from core.integration.constants import STRICT_IHSAN_THRESHOLD
        color = gauge.gauge_color(STRICT_IHSAN_THRESHOLD, gauge.thresholds)
        assert color == "#C9A962"

    def test_color_red_below_minimum(self, gauge):
        """Score < minimum → Alert Red."""
        color = gauge.gauge_color(0.70, gauge.thresholds)
        assert color == "#c93a4a"

    def test_guardian_alert_fires_below_minimum(self, gauge, mock_toast):
        """Rolling average below minimum triggers Guardian Alert toast."""
        from core.integration.constants import UNIFIED_SNR_THRESHOLD
        gauge.on_action_scored(ActionScore(ihsan_score=UNIFIED_SNR_THRESHOLD - 0.01, ...))
        assert mock_toast.called_with_title("Guardian Alert")

    def test_rolling_window_max_100(self, gauge):
        """Rolling window never exceeds 100 entries."""
        for i in range(150):
            gauge.on_action_scored(ActionScore(ihsan_score=0.97, ...))
        assert len(gauge.rolling_window) == 100

class TestVetoReceiptPanel:
    def test_sovereignty_statement_present(self, panel, sample_receipt):
        """Rendered receipt contains 'I protected your sovereignty'."""
        html = panel.render_receipt_content(sample_receipt)
        assert "I protected your sovereignty" in html

    def test_long_reason_truncated(self, panel):
        """Gate reason >200 chars is truncated with expander."""
        receipt = VetoReceipt(gate_reason="x" * 300, full_reason="x" * 300)
        html = panel.render_receipt_content(receipt)
        assert "Show full reason" in html

    def test_guardian_log_readonly(self, guardian_log):
        """GuardianLog renders no delete/edit controls."""
        html = guardian_log.render()
        assert "delete" not in html.lower()
        assert "remove" not in html.lower()
```
