//! Memory Panel Widget — HHMM 3-layer cognitive memory
//!
//! Displays the three memory layers (fast/slow/glacial),
//! knowledge summary, and profile completeness.
//! Data sourced from AgentRuntime.health() — same truth path.

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

pub struct MemoryPanel<'a> {
    data: &'a DashboardData,
}

impl<'a> MemoryPanel<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self { data }
    }
}

impl Widget for MemoryPanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let d = self.data;

        let block = Block::default()
            .title(Span::styled(
                format!(" {} Memory (HHMM) ", symbols::CRESCENT),
                Theme::title(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // 3-Layer HHMM
        lines.push(Line::from(Span::styled(
            "── Cognitive Layers ──",
            Theme::muted(),
        )));

        let bar_w = inner.width.saturating_sub(22) as usize;
        let total = (d.memory_fast + d.memory_slow + d.memory_glacial).max(1);

        // Fast layer (reflexes, cache)
        let fast_pct = d.memory_fast as f32 / total as f32;
        let fast_bar = layer_bar(fast_pct, bar_w);
        lines.push(Line::from(vec![
            Span::styled("Fast:     ", Theme::muted()),
            Span::styled(format!("{:>4}", d.memory_fast), Theme::success()),
            Span::styled(format!(" {fast_bar}"), Theme::success()),
        ]));

        // Slow layer (insights, synthesis)
        let slow_pct = d.memory_slow as f32 / total as f32;
        let slow_bar = layer_bar(slow_pct, bar_w);
        lines.push(Line::from(vec![
            Span::styled("Slow:     ", Theme::muted()),
            Span::styled(format!("{:>4}", d.memory_slow), Theme::highlight()),
            Span::styled(format!(" {slow_bar}"), Theme::highlight()),
        ]));

        // Glacial layer (profile, identity)
        let glacial_pct = d.memory_glacial as f32 / total as f32;
        let glacial_bar = layer_bar(glacial_pct, bar_w);
        lines.push(Line::from(vec![
            Span::styled("Glacial:  ", Theme::muted()),
            Span::styled(format!("{:>4}", d.memory_glacial), Theme::ihsan()),
            Span::styled(format!(" {glacial_bar}"), Theme::ihsan()),
        ]));

        lines.push(Line::from(""));

        // Knowledge summary
        lines.push(Line::from(Span::styled("── Knowledge ──", Theme::muted())));
        lines.push(Line::from(vec![
            Span::styled("Fragments:  ", Theme::muted()),
            Span::styled(format!("{}", d.memory_fragments), Theme::text()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Atoms:      ", Theme::muted()),
            Span::styled(format!("{}", d.memory_atoms), Theme::text()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Insights:   ", Theme::muted()),
            Span::styled(format!("{}", d.memory_insights), Theme::text()),
        ]));

        lines.push(Line::from(""));

        // Profile completeness
        let profile_style = if d.memory_profile_completeness >= 0.8 {
            Theme::ihsan()
        } else if d.memory_profile_completeness >= 0.5 {
            Theme::success()
        } else if d.memory_profile_completeness >= 0.2 {
            Theme::warning()
        } else {
            Theme::muted()
        };
        let profile_bar = score_bar(d.memory_profile_completeness, bar_w);
        lines.push(Line::from(vec![
            Span::styled("Profile:  ", Theme::muted()),
            Span::styled(
                format!("{:.0}%", d.memory_profile_completeness * 100.0),
                profile_style,
            ),
            Span::styled(format!(" {profile_bar}"), profile_style),
        ]));

        // Knows-me score
        let knows_style = if d.memory_knows_me >= 0.7 {
            Theme::ihsan()
        } else if d.memory_knows_me >= 0.4 {
            Theme::warning()
        } else {
            Theme::muted()
        };
        lines.push(Line::from(vec![
            Span::styled("Knows Me: ", Theme::muted()),
            Span::styled(format!("{:.2}", d.memory_knows_me), knows_style),
        ]));

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

/// Proportional layer bar
fn layer_bar(pct: f32, width: usize) -> String {
    if width < 4 {
        return String::new();
    }
    let bar_w = width.min(12);
    let filled = ((pct * bar_w as f32).round() as usize).min(bar_w);
    let empty = bar_w - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

/// Score bar [0.0 .. 1.0]
fn score_bar(score: f32, width: usize) -> String {
    if width < 4 {
        return String::new();
    }
    let bar_w = width.min(12);
    let filled = ((score * bar_w as f32).round() as usize).min(bar_w);
    let empty = bar_w - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
