# Phase 6B — New Widgets

## Purpose

Five new ratatui widgets that consume `DashboardData` and render into the 7-zone layout.

All widgets follow the existing pattern:
- Builder struct with borrowed data
- `impl Widget for WidgetName` with `render(self, area, buf)`
- Use `Theme::*` styles from theme.rs
- Use `borders::ARABIC` (rounded corners)
- Use `symbols::*` for status indicators

---

## Widget 1: ParliamentPanel

**File:** `widgets/parliament_panel.rs`

```pseudocode
pub struct ParliamentPanel<'a> {
    pat_agents: &'a [AgentInfo],
    sat_agents: &'a [AgentInfo],
}

impl<'a> ParliamentPanel<'a> {
    pub fn new(pat: &'a [AgentInfo], sat: &'a [AgentInfo]) -> Self
}

impl Widget for ParliamentPanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Block with title " Parliament (12 agents) "
        let block = Block::default()
            .title(Span::styled(" Parliament (N agents) ", Theme::title()))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel())

        let inner = block.inner(area)
        block.render(area, buf)

        let mut lines = Vec::new()

        // PAT header
        lines.push(Line::from(Span::styled("PAT-7 (Your Council)", Theme::ihsan())))

        // PAT agents — one line each
        for agent in self.pat_agents {
            lines.push(Line::from(vec![
                Span::styled(format!(" P{} ", agent.index), Theme::muted()),
                Span::styled(&agent.icon, Theme::pat_agent(&agent.callsign)),
                Span::styled(format!(" {:<12}", agent.callsign), Theme::highlight()),
                Span::styled(&agent.role, Theme::text()),
            ]))
        }

        // Blank line
        lines.push(Line::from(""))

        // SAT header
        lines.push(Line::from(Span::styled("SAT-5 (System Immune)", Theme::subtitle())))

        // SAT agents
        for agent in self.sat_agents {
            lines.push(Line::from(vec![
                Span::styled(format!(" S{} ", agent.index), Theme::muted()),
                Span::styled(&agent.icon, Theme::text()),
                Span::styled(format!(" {:<12}", agent.callsign), Theme::highlight()),
                Span::styled(&agent.role, Theme::text()),
            ]))
        }

        Paragraph::new(lines).render(inner, buf)
    }
}
```

---

## Widget 2: GhostFeed

**File:** `widgets/ghost_feed.rs`

```pseudocode
pub struct GhostFeed<'a> {
    greeting: &'a str,
    runtime_state: &'a str,
    agents_active: u32,
    agents_registered: u32,
    reflex_mode: &'a str,
    reflex_rules: u32,
    recommendations: &'a [String],
}

impl<'a> GhostFeed<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self {
            greeting: &data.greeting,
            runtime_state: &data.runtime_state,
            agents_active: data.agents_active,
            agents_registered: data.agents_registered,
            reflex_mode: &data.reflex_mode,
            reflex_rules: data.reflex_rules,
            recommendations: &data.recommendations,
        }
    }
}

impl Widget for GhostFeed<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(Span::styled(
                format!(" {} Ghost ", symbols::STAR),
                Theme::ihsan()
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel())

        let inner = block.inner(area)
        block.render(area, buf)

        let mut lines = Vec::new()

        // Greeting
        lines.push(Line::from(Span::styled(self.greeting, Theme::highlight())))
        lines.push(Line::from(""))

        // Runtime state
        let state_style = match self.runtime_state {
            "Ready" => Theme::success(),
            "Degraded" => Theme::warning(),
            _ => Theme::error(),
        }
        lines.push(Line::from(vec![
            Span::styled("Runtime: ", Theme::muted()),
            Span::styled(self.runtime_state, state_style),
        ]))

        // Agents
        lines.push(Line::from(vec![
            Span::styled("Agents: ", Theme::muted()),
            Span::styled(
                format!("{}/{} active", self.agents_active, self.agents_registered),
                Theme::text()
            ),
        ]))

        // Reflex
        lines.push(Line::from(vec![
            Span::styled("Reflex: ", Theme::muted()),
            Span::styled(
                format!("{} ({} rules)", self.reflex_mode, self.reflex_rules),
                Theme::text()
            ),
        ]))

        lines.push(Line::from(""))

        // Recommendations
        for rec in self.recommendations {
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", symbols::ARROW_RIGHT), Theme::ihsan()),
                Span::styled(
                    truncate(rec, inner.width as usize - 4),
                    Theme::text()
                ),
            ]))
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf)
    }
}
```

---

## Widget 3: TrustRail

**File:** `widgets/trust_rail.rs`

```pseudocode
pub struct TrustRail<'a> {
    checks: &'a [TrustCheck],
    receipt_checks: &'a [TrustCheck],
    verdict: &'a TrustVerdict,
}

impl<'a> TrustRail<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self {
            checks: &data.trust_checks,
            receipt_checks: &data.receipt_chain_checks,
            verdict: &data.trust_verdict,
        }
    }
}

impl Widget for TrustRail<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let is_sovereign = matches!(self.verdict, TrustVerdict::Sovereign)

        let block = Block::default()
            .title(Span::styled(" Trust Surface ", Theme::title()))
            .borders(Borders::ALL)
            .border_set(if is_sovereign { borders::IMPORTANT } else { borders::ARABIC })
            .border_style(if is_sovereign { Theme::panel_border_focused() } else { Theme::panel_border() })
            .style(Theme::panel())

        let inner = block.inner(area)
        block.render(area, buf)

        let mut lines = Vec::new()

        // Verdict banner
        let (verdict_text, verdict_style) = if is_sovereign {
            ("VERDICT: ✓ SOVEREIGN", Theme::ihsan())
        } else {
            ("VERDICT: ⚠ DEGRADED", Theme::warning())
        }
        lines.push(Line::from(Span::styled(verdict_text, verdict_style)))
        lines.push(Line::from(""))

        // Group checks by category
        // First 5: Constitutional Law
        lines.push(Line::from(Span::styled("[Constitutional Law]", Theme::subtitle())))
        for check in self.checks.iter().take(5) {
            let mark = if check.passed { symbols::SUCCESS } else { symbols::ERROR }
            let style = if check.passed { Theme::success() } else { Theme::error() }
            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", mark), style),
                Span::styled(format!("{:<16}", check.name), Theme::text()),
                Span::styled(&check.detail, Theme::muted()),
            ]))
        }
        lines.push(Line::from(""))

        // Next 3: Topology
        lines.push(Line::from(Span::styled("[Topology]", Theme::subtitle())))
        for check in self.checks.iter().skip(5).take(3) {
            let mark = if check.passed { symbols::SUCCESS } else { symbols::ERROR }
            let style = if check.passed { Theme::success() } else { Theme::error() }
            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", mark), style),
                Span::styled(&check.name, Theme::text()),
            ]))
        }
        lines.push(Line::from(""))

        // Genesis (1 check)
        lines.push(Line::from(Span::styled("[Genesis]", Theme::subtitle())))
        if let Some(check) = self.checks.get(8) {
            let mark = if check.passed { symbols::SUCCESS } else { symbols::ERROR }
            let style = if check.passed { Theme::success() } else { Theme::error() }
            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", mark), style),
                Span::styled(&check.name, Theme::text()),
            ]))
        }
        lines.push(Line::from(""))

        // Receipt chain (2 checks)
        lines.push(Line::from(Span::styled("[Ledger]", Theme::subtitle())))
        for check in self.receipt_checks {
            let mark = if check.passed { symbols::SUCCESS } else { symbols::ERROR }
            let style = if check.passed { Theme::success() } else { Theme::error() }
            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", mark), style),
                Span::styled(&check.name, Theme::text()),
                Span::styled(format!(" {}", check.detail), Theme::muted()),
            ]))
        }
        lines.push(Line::from(""))

        // Substrate (last 2 checks)
        lines.push(Line::from(Span::styled("[Substrate]", Theme::subtitle())))
        for check in self.checks.iter().skip(9) {
            let mark = if check.passed { symbols::SUCCESS } else { symbols::ERROR }
            let style = if check.passed { Theme::success() } else { Theme::error() }
            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", mark), style),
                Span::styled(&check.name, Theme::text()),
                Span::styled(format!(" {}", check.detail), Theme::muted()),
            ]))
        }

        let scroll = if lines.len() > inner.height as usize {
            (lines.len() - inner.height as usize) as u16
        } else { 0 }

        Paragraph::new(lines)
            .scroll((scroll, 0))
            .render(inner, buf)
    }
}
```

---

## Widget 4: SubstratePanel

**File:** `widgets/substrate_panel.rs`

```pseudocode
pub struct SubstratePanel<'a> {
    cpu_name: &'a str,
    cpu_cores: u32,
    ram_total_gb: f64,
    ram_used_pct: f64,
    gpu: Option<&'a GpuInfo>,
    model_count: usize,
    text_count: usize,
    vision_count: usize,
    platform: &'a str,
}

impl<'a> SubstratePanel<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self {
            cpu_name: &data.cpu_name,
            cpu_cores: data.cpu_cores,
            ram_total_gb: data.ram_total_gb,
            ram_used_pct: data.ram_used_pct,
            gpu: data.gpu.as_ref(),
            model_count: data.model_count,
            text_count: data.text_models.len(),
            vision_count: data.vision_models.len(),
            platform: &data.platform,
        }
    }
}

impl Widget for SubstratePanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(Span::styled(" Substrate ", Theme::title()))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel())

        let inner = block.inner(area)
        block.render(area, buf)

        let mut lines = Vec::new()

        // CPU
        lines.push(Line::from(vec![
            Span::styled(
                truncate(self.cpu_name, inner.width as usize - 12),
                Theme::text()
            ),
            Span::styled(format!(" • {} cores", self.cpu_cores), Theme::muted()),
        ]))

        // RAM with usage indicator
        let ram_style = if self.ram_used_pct > 90.0 { Theme::error() }
                        else if self.ram_used_pct > 75.0 { Theme::warning() }
                        else { Theme::success() }
        lines.push(Line::from(vec![
            Span::styled("RAM: ", Theme::muted()),
            Span::styled(format!("{:.0} GB", self.ram_total_gb), Theme::text()),
            Span::styled(format!(" ({:.0}% used)", self.ram_used_pct), ram_style),
        ]))

        // GPU
        if let Some(gpu) = self.gpu {
            let gpu_style = if gpu.used_pct > 90.0 { Theme::error() }
                            else if gpu.used_pct > 75.0 { Theme::warning() }
                            else { Theme::success() }
            lines.push(Line::from(vec![
                Span::styled("GPU: ", Theme::muted()),
                Span::styled(
                    truncate(&gpu.name, inner.width as usize - 20),
                    Theme::text()
                ),
            ]))
            lines.push(Line::from(vec![
                Span::styled("     ", Theme::muted()),
                Span::styled(
                    format!("{}/{} MB", gpu.used_mb, gpu.total_mb),
                    gpu_style
                ),
                Span::styled(format!(" ({:.0}%)", gpu.used_pct), gpu_style),
            ]))
        }

        // Models
        lines.push(Line::from(vec![
            Span::styled("Models: ", Theme::muted()),
            Span::styled(
                format!("{} ({} text, {} vision)", self.model_count, self.text_count, self.vision_count),
                if self.model_count > 0 { Theme::text() } else { Theme::error() }
            ),
        ]))

        // Platform
        lines.push(Line::from(vec![
            Span::styled("Platform: ", Theme::muted()),
            Span::styled(self.platform, Theme::text()),
        ]))

        Paragraph::new(lines).render(inner, buf)
    }
}
```

---

## Widget 5: ReceiptRail

**File:** `widgets/receipt_rail.rs`

```pseudocode
pub struct ReceiptRail<'a> {
    total: usize,
    chain_valid: bool,
    today_count: usize,
    today_complete: usize,
    manifest_seal: Option<&'a str>,
    recent: &'a [ReceiptSummary],
}

impl<'a> ReceiptRail<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self {
            total: data.total_receipts,
            chain_valid: data.chain_valid,
            today_count: data.today_count,
            today_complete: data.today_complete,
            manifest_seal: data.manifest_seal.as_deref(),
            recent: &data.recent_receipts,
        }
    }
}

impl Widget for ReceiptRail<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(Span::styled(
                format!(" Receipt Chain ({}) ", self.total),
                Theme::title()
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel())

        let inner = block.inner(area)
        block.render(area, buf)

        let mut lines = Vec::new()

        // Chain status
        let (chain_mark, chain_style) = if self.chain_valid {
            (symbols::SUCCESS, Theme::success())
        } else {
            (symbols::ERROR, Theme::error())
        }
        lines.push(Line::from(vec![
            Span::styled(format!("Chain: {} ", chain_mark), chain_style),
            Span::styled(
                if self.chain_valid { "All hashes valid" } else { "CHAIN BROKEN" },
                chain_style
            ),
        ]))

        // Today summary
        let today_style = if self.today_complete == self.today_count && self.today_count > 0 {
            Theme::success()
        } else if self.today_count > 0 {
            Theme::warning()
        } else {
            Theme::muted()
        }
        lines.push(Line::from(vec![
            Span::styled("Today: ", Theme::muted()),
            Span::styled(
                format!("{} missions ({}✓)", self.today_count, self.today_complete),
                today_style
            ),
        ]))
        lines.push(Line::from(""))

        // Recent receipts (as many as fit)
        let max_receipts = (inner.height as usize).saturating_sub(5)  // 3 header + 2 footer
        for (i, r) in self.recent.iter().take(max_receipts).enumerate() {
            let state_style = if r.is_success { Theme::success() }
                              else if r.is_degraded { Theme::warning() }
                              else { Theme::error() }
            let num = self.total - i  // descending order
            lines.push(Line::from(vec![
                Span::styled(format!("#{} ", num), Theme::muted()),
                Span::styled(format!("{}… ", r.id_short.get(..8).unwrap_or(&r.id_short)), Theme::text()),
                Span::styled(format!("{:<9}", r.state_label), state_style),
                Span::styled(
                    truncate(&r.objective, inner.width as usize - 24),
                    Theme::muted()
                ),
            ]))
        }

        // Manifest seal (footer)
        if !lines.is_empty() { lines.push(Line::from("")) }
        if let Some(seal) = self.manifest_seal {
            lines.push(Line::from(vec![
                Span::styled("Manifest: ", Theme::muted()),
                Span::styled(format!("{}…", seal), Theme::ihsan()),
                Span::styled(format!(" ({} today)", self.today_count), Theme::muted()),
            ]))
        } else if self.total == 0 {
            lines.push(Line::from(Span::styled(
                "No receipts yet. Run bizra mission.",
                Theme::muted()
            )))
        }

        Paragraph::new(lines).render(inner, buf)
    }
}
```

---

## Registration

**File:** `widgets/mod.rs` — add:

```rust
mod ghost_feed;
mod parliament_panel;
mod receipt_rail;
mod substrate_panel;
mod trust_rail;

pub use ghost_feed::GhostFeed;
pub use parliament_panel::ParliamentPanel;
pub use receipt_rail::ReceiptRail;
pub use substrate_panel::SubstratePanel;
pub use trust_rail::TrustRail;
```

---

## TDD Anchors

```pseudocode
// Widget rendering tests use ratatui::backend::TestBackend

#[test] fn test_parliament_panel_renders() {
    let pat = vec![AgentInfo { index: 0, callsign: "Atlas".into(), .. }; 7]
    let sat = vec![AgentInfo { index: 0, callsign: "Sentinel".into(), .. }; 5]
    let backend = TestBackend::new(40, 20)
    let mut terminal = Terminal::new(backend).unwrap()
    terminal.draw(|f| {
        f.render_widget(ParliamentPanel::new(&pat, &sat), f.area())
    }).unwrap()
    let buf = terminal.backend().buffer()
    assert!(buf_contains(buf, "PAT-7"))
    assert!(buf_contains(buf, "SAT-5"))
}

#[test] fn test_trust_rail_sovereign() {
    // All checks passing → SOVEREIGN verdict shown
    let checks = (0..11).map(|i| TrustCheck { name: format!("check{i}"), passed: true, detail: "ok".into() }).collect()
    let receipt_checks = vec![TrustCheck { name: "receipts".into(), passed: true, detail: "5".into() }]
    // render and verify "SOVEREIGN" appears in buffer
}

#[test] fn test_receipt_rail_empty() {
    // No receipts → "No receipts yet" message
    let rail = ReceiptRail { total: 0, chain_valid: true, today_count: 0, today_complete: 0, manifest_seal: None, recent: &[] }
    // render and verify fallback message
}
```
