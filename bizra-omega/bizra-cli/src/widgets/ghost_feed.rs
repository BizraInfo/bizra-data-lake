//! Ghost Feed — Proactive briefing and recommendations
//!
//! The Ghost layer's TUI surface: greeting, runtime state,
//! and context-aware recommendations.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
};

use crate::{
    commands::genesis_spine::DashboardData,
    theme::{borders, symbols, Theme},
};

pub struct GhostFeed<'a> {
    greeting: &'a str,
    runtime_state: &'a str,
    agents_active: usize,
    agents_registered: usize,
    reflex_mode: &'a str,
    reflex_rules: usize,
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
                Theme::ihsan(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // Greeting
        lines.push(Line::from(Span::styled(self.greeting, Theme::highlight())));
        lines.push(Line::from(""));

        // Runtime state
        let state_style = match self.runtime_state {
            "Ready" => Theme::success(),
            "Degraded" => Theme::warning(),
            _ => Theme::error(),
        };
        lines.push(Line::from(vec![
            Span::styled("Runtime: ", Theme::muted()),
            Span::styled(self.runtime_state, state_style),
        ]));

        // Agents
        lines.push(Line::from(vec![
            Span::styled("Agents:  ", Theme::muted()),
            Span::styled(
                format!("{}/{} active", self.agents_active, self.agents_registered),
                Theme::text(),
            ),
        ]));

        // Reflex
        lines.push(Line::from(vec![
            Span::styled("Reflex:  ", Theme::muted()),
            Span::styled(
                format!("{} ({} rules)", self.reflex_mode, self.reflex_rules),
                Theme::text(),
            ),
        ]));

        lines.push(Line::from(""));

        // Recommendations
        for rec in self.recommendations {
            let max_w = (inner.width as usize).saturating_sub(4);
            let display = if rec.len() > max_w && max_w > 3 {
                format!("{}...", &rec[..max_w - 3])
            } else {
                rec.clone()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", symbols::ARROW_RIGHT), Theme::ihsan()),
                Span::styled(display, Theme::text()),
            ]));
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
