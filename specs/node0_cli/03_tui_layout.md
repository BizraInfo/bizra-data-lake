# Phase 3: TUI Layout
## Living Parliament View — The Primary Screen
### References: 00_cli_master_spec.md §5

---

## 1. Design Principle

The TUI should feel like a **living mission control center**, not a dead command shell.
Every panel updates in real time during mission execution.

Framework: **ratatui** (Rust) — already in `bizra-cli` dependencies.

## 2. Layout Grid (5 zones)

```
┌─────────────────┬──────────────────────┬─────────────────┐
│                  │                      │                  │
│   ZONE A         │   ZONE B             │   ZONE C         │
│   PAT-7 Panel   │   Mission Panel      │   SAT-5 Panel   │
│   (left, 20%)   │   (center, 50%)      │   (right, 20%)  │
│                  │                      │                  │
├──────────────────┴──────────────────────┴─────────────────┤
│                                                            │
│   ZONE D: Ghost Feed (bottom, 15%)                        │
│                                                            │
├────────────────────────────────┬───────────────────────────┤
│   ZONE E: Receipt Rail (30%)  │  ZONE F: Trust Rail (30%) │
├────────────────────────────────┴───────────────────────────┤
│   ZONE G: Command Input (bottom bar)                       │
│   بذرة ›                                                   │
└────────────────────────────────────────────────────────────┘
```

## 3. Pseudocode: Zone A — PAT-7 Panel

```
WIDGET PatPanel:
    PROPS: agents: [AgentState; 7]

    RENDER:
        title("PAT-7 — Your Council")
        FOR agent IN agents:
            row(
                icon=agent.icon,
                name=agent.name,
                status=status_dot(agent),
                    # ● green = active/working
                    # ◐ yellow = assigned
                    # ○ gray = idle
                    # ✗ red = error
                detail=IF agent.status == WORKING:
                    f"[{agent.current_task[:20]}...]"
                ELIF agent.last_receipt:
                    f"✉ {agent.last_receipt[:8]}"
                ELSE:
                    "—"
            )

    STATUS_DOT(agent):
        MATCH agent.status:
            WORKING  → "●" (green)
            ASSIGNED → "◐" (yellow)
            IDLE     → "○" (gray)
            ERROR    → "✗" (red)
            DONE     → "✓" (teal)
```

## 4. Pseudocode: Zone B — Mission Panel

```
WIDGET MissionPanel:
    PROPS: mission: Option<MissionState>

    RENDER:
        IF mission IS NONE:
            centered_text("No active mission")
            hint("Type: mission \"your objective\"")
            RETURN

        title(f"Mission: {mission.id}")
        objective_text(mission.objective, wrap=40)

        # Progress bar (10 phases)
        progress_bar(
            current=mission.phase_index,
            total=10,
            phases=["ENV","PRE","RET","DEC","EXE","SYN","GATE","REC","MEM","DONE"],
            active_color=BIZRA_TEAL,
        )

        # Scoring gauge (appears during/after GATE phase)
        IF mission.phase >= GATING:
            gauge_row("SNR",   mission.snr_score,   threshold=0.85)
            gauge_row("Ihsan", mission.ihsan_score,  threshold=0.95)
            verdict_badge(mission.verdict)
                # PERMIT = green badge
                # REVIEW = yellow badge
                # QUARANTINED = red badge

        # Agent contribution summary (appears during EXECUTE)
        IF mission.phase >= EXECUTING:
            FOR result IN mission.agent_results:
                agent_result_row(
                    name=result.agent_name,
                    tokens=result.tokens_generated,
                    status=result.status,
                )

    GAUGE_ROW(label, value, threshold):
        bar_width = 30
        filled = int(value * bar_width)
        threshold_pos = int(threshold * bar_width)
        color = GREEN IF value >= threshold ELSE YELLOW IF value >= 0.5 ELSE RED
        render(f"{label}: {value:.2f} [{bar(filled, bar_width, threshold_pos)}]")
```

## 5. Pseudocode: Zone C — SAT-5 Panel

```
WIDGET SatPanel:
    PROPS: validators: [SatState; 5]

    RENDER:
        title("SAT-5 — Validators")
        FOR sat IN validators:
            row(
                icon=sat.icon,
                name=sat.name,
                status=gate_status(sat),
                    # ● PASS (green)
                    # ⚠ FLAG (yellow)
                    # ✗ BLOCK (red)
                    # ○ PENDING (gray)
            )

        # Overall integrity
        all_pass = all(s.status == PASS FOR s IN validators)
        IF all_pass:
            footer("Constitutional Integrity: ✓ INTACT")
        ELSE:
            flags = [s.name FOR s IN validators IF s.status == FLAG]
            footer(f"Flags: {', '.join(flags)}")
```

## 6. Pseudocode: Zone D — Ghost Feed

```
WIDGET GhostFeed:
    PROPS: entries: Vec<GhostEntry>
    STATE: scroll_offset: usize = 0

    RENDER:
        title("Ghost Feed")  # Proactive agent insights
        visible = entries[scroll_offset..scroll_offset+3]
        FOR entry IN visible:
            row(
                icon="🔮",
                agent=entry.agent_name,
                message=entry.message,
                timestamp=relative_time(entry.timestamp),
                    # "2m ago", "just now", "yesterday"
            )

    # Ghost entries are generated:
    # - At mission completion (Guardian insight)
    # - At TUI launch (morning brief summary)
    # - When similar prior mission found (Strategist hint)
    # - When anomaly detected (Analyst warning)

    GHOST_SOURCES:
        ON mission_complete(verdict):
            IF verdict.decision == "REVIEW":
                add("Guardian", "SNR below threshold — consider adding citations")
            IF verdict.snr_score > prev_mission.snr_score:
                add("Analyst", f"SNR improved +{delta:.2f} from last run")

        ON tui_launch():
            brief = generate_morning_brief()
            add("Strategist", brief.top_insight)
            IF brief.risks:
                add("Guardian", f"Risk: {brief.risks[0]}")

        ON similar_mission_found(prior):
            add("Strategist", f"Similar mission PERMIT'd {days_ago}d ago — warm path?")
```

## 7. Pseudocode: Zone E/F — Receipt + Trust Rails

```
WIDGET ReceiptRail:
    PROPS: receipts: Vec<ReceiptSummary>

    RENDER:
        title("Receipts")
        FOR receipt IN receipts[..5]:  # Last 5
            compact_row(
                icon="✉",
                hash=receipt.hash[:8],
                decision=badge(receipt.decision),
                time=relative_time(receipt.timestamp),
            )

WIDGET TrustRail:
    PROPS: chain: ChainStatus, wallet: WalletBalance

    RENDER:
        title("Trust")
        row("Chain:", f"{chain.length} entries",
            badge("INTACT" IF chain.valid ELSE "BROKEN"))
        row("Parity:", f"{chain.cross_lang_pass}/5",
            badge("✓" IF chain.cross_lang_pass == 5 ELSE "!"))
        row("SEED:", f"{wallet.seed_balance:,.0f}")
        row("IMPT:", f"{wallet.impt_balance:,.1f}")
```

## 8. Pseudocode: Zone G — Command Input

```
WIDGET CommandInput:
    PROPS: prompt: str = "بذرة ›"
    STATE: buffer: String = ""
    STATE: history: Vec<String> = []
    STATE: history_index: usize = 0

    ON_KEY(key):
        MATCH key:
            Enter   → dispatch_command(buffer); buffer = ""
            Up      → history_prev()
            Down    → history_next()
            Tab     → autocomplete(buffer)
            Char(c) → buffer.push(c)
            Backsp  → buffer.pop()

    AUTOCOMPLETE(partial):
        commands = ["mission", "agents", "trust", "receipt", "memory", "node"]
        matches = [c FOR c IN commands IF c.starts_with(partial)]
        IF len(matches) == 1:
            buffer = matches[0] + " "
```

## 9. Existing Widget Mapping

| Existing Widget | v0 Zone | Action |
|-----------------|---------|--------|
| `header.rs` | Top bar | Keep, add node ID + timezone |
| `agent_card.rs` | Zone A | Extend with live status |
| `fate_gauge.rs` | Zone B | Extend with SNR/Ihsan gauges |
| `status_bar.rs` | Zone G | Extend with command input |
| — | Zone C (SAT) | NEW |
| — | Zone D (Ghost) | NEW |
| — | Zone E (Receipt) | NEW |
| — | Zone F (Trust) | NEW |

## 10. TDD Anchors

```rust
#[test]
fn test_tui_renders_all_zones() {
    let app = App::new(test_state());
    let frame = render_frame(&app);
    assert!(frame.contains_widget("pat_panel"));
    assert!(frame.contains_widget("mission_panel"));
    assert!(frame.contains_widget("sat_panel"));
    assert!(frame.contains_widget("ghost_feed"));
    assert!(frame.contains_widget("receipt_rail"));
    assert!(frame.contains_widget("trust_rail"));
    assert!(frame.contains_widget("command_input"));
}

#[test]
fn test_mission_panel_updates_during_execution() {
    let mut app = App::new(test_state());
    app.start_mission("test");
    assert_eq!(app.mission_panel.status, "ENVELOPED");
    app.advance_phase();
    assert_eq!(app.mission_panel.status, "PREFLIGHT ✓");
}

#[test]
fn test_ghost_feed_adds_entry_on_completion() {
    let mut app = App::new(test_state());
    app.complete_mission(GateVerdict::review(0.61));
    assert!(!app.ghost_feed.entries.is_empty());
    assert!(app.ghost_feed.entries[0].message.contains("threshold"));
}

#[test]
fn test_agent_status_dots_correct() {
    assert_eq!(status_dot(AgentStatus::Working), "●");
    assert_eq!(status_dot(AgentStatus::Idle), "○");
    assert_eq!(status_dot(AgentStatus::Error), "✗");
}
```

## 11. Validation Gate

```
[ ] TUI renders all 7 zones without overlap
[ ] PAT-7 panel shows 7 agents with live status
[ ] SAT-5 panel shows 5 validators with gate status
[ ] Mission panel shows real-time 10-phase progress
[ ] Ghost feed shows proactive insights (not just logs)
[ ] Receipt rail shows last 5 receipts
[ ] Trust rail shows chain integrity + wallet
[ ] Command input supports history + autocomplete
[ ] Responsive on 80×24 minimum terminal size
```

---

*The TUI is not a dashboard. It is the face of the parliament.*
