# Phase 6C — Dashboard Wiring

## Purpose

Replace the existing `render_dashboard()` in main.rs with the 7-zone sovereign cockpit layout, wire data flow, add periodic refresh.

---

## Layout Pseudocode

```pseudocode
fn render_dashboard(f: &mut Frame, app: &App, area: Rect) {
    // Guard: need data to render
    let Some(data) = &app.dashboard_data else {
        // Show loading message on first frame
        let block = Block::default()
            .title(" Dashboard ")
            .borders(Borders::ALL)
            .style(Theme::panel())
        let msg = Paragraph::new("Gathering sovereign intelligence...")
            .block(block)
        f.render_widget(msg, area)
        return
    }

    // ── 3-Column Layout ──
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30),  // Left: Parliament + Substrate
            Constraint::Percentage(35),  // Center: Ghost + Receipt
            Constraint::Percentage(35),  // Right: Trust
        ])
        .split(area)

    // ── Left Column: Parliament (top 60%) + Substrate (bottom 40%) ──
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60),  // Parliament
            Constraint::Percentage(40),  // Substrate
        ])
        .split(columns[0])

    // Zone 2: Parliament
    let parliament = ParliamentPanel::new(&data.pat_agents, &data.sat_agents)
    f.render_widget(parliament, left[0])

    // Zone 5: Substrate
    let substrate = SubstratePanel::from_data(data)
    f.render_widget(substrate, left[1])

    // ── Center Column: Ghost (top 50%) + Receipt (bottom 50%) ──
    let center = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),  // Ghost
            Constraint::Percentage(50),  // Receipt
        ])
        .split(columns[1])

    // Zone 3: Ghost Feed
    let ghost = GhostFeed::from_data(data)
    f.render_widget(ghost, center[0])

    // Zone 6: Receipt Rail
    let receipt = ReceiptRail::from_data(data)
    f.render_widget(receipt, center[1])

    // ── Right Column: Trust Rail (full height) ──
    // Zone 4: Trust Rail
    let trust = TrustRail::from_data(data)
    f.render_widget(trust, columns[2])
}
```

---

## Header Augmentation

Modify existing `Header` widget to accept trust/model info:

```pseudocode
pub struct Header<'a> {
    node_name: &'a str,
    active_view: ActiveView,
    lmstudio_connected: bool,
    voice_active: bool,
    // NEW fields:
    trust_sovereign: bool,       // true = SOVEREIGN, false = DEGRADED
    model_count: usize,
}

impl Header {
    // NEW builder methods:
    pub fn trust(mut self, sovereign: bool) -> Self {
        self.trust_sovereign = sovereign; self
    }
    pub fn models(mut self, count: usize) -> Self {
        self.model_count = count; self
    }
}

// In render():
// After the right-side status indicators, add:
let trust_indicator = if self.trust_sovereign {
    Span::styled(" ● SOVEREIGN ", Theme::ihsan())
} else {
    Span::styled(" ⚠ DEGRADED ", Theme::warning())
}

let model_indicator = Span::styled(
    format!(" {} models ", self.model_count),
    if self.model_count > 0 { Theme::text() } else { Theme::error() }
)
```

Call site in `ui()`:

```pseudocode
let (trust_ok, model_ct) = app.dashboard_data.as_ref()
    .map(|d| (matches!(d.trust_verdict, TrustVerdict::Sovereign), d.model_count))
    .unwrap_or((false, 0));

let header = Header::new(&app.node_name, app.active_view)
    .lmstudio(app.lmstudio_connected)
    .voice(app.voice_active)
    .trust(trust_ok)
    .models(model_ct)
```

---

## StatusBar Augmentation

Add manifest summary:

```pseudocode
pub struct StatusBar<'a> {
    // existing fields...
    // NEW:
    manifest_summary: Option<&'a str>,  // e.g. "3/3✓ today"
}

impl StatusBar {
    pub fn manifest(mut self, summary: Option<&'a str>) -> Self {
        self.manifest_summary = summary; self
    }
}

// In render(), after hints and before status message:
if let Some(manifest) = self.manifest_summary {
    spans.push(Span::styled(" │ ", Theme::muted()))
    spans.push(Span::styled(format!("Manifest: {manifest}"), Theme::ihsan()))
}
```

Call site:

```pseudocode
let manifest_text = app.dashboard_data.as_ref().map(|d| {
    format!("{}/{}✓ today", d.today_count, d.today_complete)
});

let status = StatusBar::new(app.input_mode)
    .agent(app.selected_agent.map(|a| a.name()))
    .message(app.status_message.as_ref().map(|(m, _)| m.as_str()))
    .manifest(manifest_text.as_deref())
```

---

## Periodic Refresh in run_app()

```pseudocode
fn run_app(terminal, app) -> Result<()> {
    // Initial data gather (before first render)
    app.dashboard_data = Some(gather_dashboard_data())
    app.last_refresh = Some(Instant::now())

    loop {
        terminal.draw(|f| ui(f, app))?

        // Event poll (existing 100ms timeout)
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match app.input_mode {
                    InputMode::Normal => match key.code {
                        // ... existing key handlers ...

                        // NEW: manual refresh
                        KeyCode::Char('r') => {
                            app.dashboard_data = Some(gather_dashboard_data())
                            app.last_refresh = Some(Instant::now())
                            app.set_status("Dashboard refreshed")
                        }

                        _ => {}
                    }
                    // ... existing insert/command handlers ...
                }
            }
        }

        // Periodic refresh (every 5 seconds)
        if app.last_refresh
            .map(|t| t.elapsed() > Duration::from_secs(5))
            .unwrap_or(true)
        {
            app.dashboard_data = Some(gather_dashboard_data())
            app.last_refresh = Some(Instant::now())
        }

        app.clear_expired_status()
        if app.should_quit { return Ok(()) }
    }
}
```

---

## App Struct Changes

```pseudocode
// In app.rs, add to App struct:
use std::time::Instant;

pub struct App {
    // ... existing fields ...

    /// Dashboard live data (refreshed periodically)
    pub dashboard_data: Option<DashboardData>,

    /// Last data refresh timestamp
    pub last_refresh: Option<Instant>,
}

// In App::new():
Self {
    // ... existing defaults ...
    dashboard_data: None,
    last_refresh: None,
}
```

---

## Import Changes in main.rs

```pseudocode
// Add to the imports used in render_dashboard:
use crate::widgets::{
    AgentCard, FateGauge, Header, StatusBar,
    // NEW:
    GhostFeed, ParliamentPanel, ReceiptRail, SubstratePanel, TrustRail,
};

// Import gather function:
use commands::genesis_spine::gather_dashboard_data;
```

This requires making `gather_dashboard_data`, `DashboardData`, and all sub-structs `pub`.

---

## Other Views (unchanged)

The existing views remain intact:
- `render_agents()` — full agent card grid (unchanged)
- `render_chat()` — chat messages (unchanged)
- `render_tasks()`, `render_treasury()`, `render_settings()` — stubs (unchanged)

Only the Dashboard view (default, view 1) gets the 7-zone treatment.

---

## TDD Anchors

```pseudocode
#[test] fn test_dashboard_renders_without_data() {
    // When dashboard_data is None, show loading message (no panic)
    let app = App::new()  // dashboard_data = None
    let backend = TestBackend::new(120, 40)
    let mut terminal = Terminal::new(backend).unwrap()
    terminal.draw(|f| render_dashboard(f, &app, f.area())).unwrap()
    // Should contain "Gathering" message
}

#[test] fn test_dashboard_renders_with_data() {
    // With real data, no panic
    let mut app = App::new()
    app.dashboard_data = Some(gather_dashboard_data())
    let backend = TestBackend::new(120, 40)
    let mut terminal = Terminal::new(backend).unwrap()
    terminal.draw(|f| render_dashboard(f, &app, f.area())).unwrap()
    // Should render without panic
}

#[test] fn test_periodic_refresh_timing() {
    // After 5 seconds, dashboard_data should be refreshed
    // (unit test the timing logic, not the actual refresh)
    let last = Instant::now() - Duration::from_secs(6)
    assert!(last.elapsed() > Duration::from_secs(5))
}
```
