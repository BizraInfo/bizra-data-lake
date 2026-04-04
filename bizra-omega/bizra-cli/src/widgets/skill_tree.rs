//! Skill Tree Widget — Reflex stats, compiled rules, sovereignty tier
//!
//! Displays the reflex compilation pipeline (S1 reflex cache),
//! hit/miss rates, and sovereignty tier progression.
//! Data sourced from AgentRuntime.reflex_stats() — same truth path.

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

pub struct SkillTree<'a> {
    data: &'a DashboardData,
}

impl<'a> SkillTree<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self { data }
    }
}

impl Widget for SkillTree<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let d = self.data;

        let block = Block::default()
            .title(Span::styled(
                format!(" {} Skills & Reflexes ", symbols::AGENT),
                Theme::title(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // Reflex pipeline stats
        lines.push(Line::from(Span::styled(
            "── Reflex Pipeline ──",
            Theme::muted(),
        )));

        lines.push(Line::from(vec![
            Span::styled("Compiled:     ", Theme::muted()),
            Span::styled(format!("{}", d.reflex_compiled), Theme::success()),
            Span::styled(" rules", Theme::muted()),
        ]));

        // Hit rate
        let total_lookups = d.reflex_hits + d.reflex_misses;
        let hit_rate = if total_lookups > 0 {
            d.reflex_hits as f32 / total_lookups as f32
        } else {
            0.0
        };
        let hit_style = if hit_rate >= 0.8 {
            Theme::ihsan()
        } else if hit_rate >= 0.5 {
            Theme::success()
        } else {
            Theme::warning()
        };
        let bar_w = inner.width.saturating_sub(22) as usize;
        let hit_bar = rate_bar(hit_rate, bar_w);
        lines.push(Line::from(vec![
            Span::styled("Hit Rate:     ", Theme::muted()),
            Span::styled(format!("{:.0}%", hit_rate * 100.0), hit_style),
            Span::styled(format!(" {hit_bar}"), hit_style),
        ]));

        lines.push(Line::from(vec![
            Span::styled("Hits:         ", Theme::muted()),
            Span::styled(format!("{}", d.reflex_hits), Theme::success()),
            Span::styled("  Misses: ", Theme::muted()),
            Span::styled(format!("{}", d.reflex_misses), Theme::warning()),
        ]));

        // Quarantined
        let q_style = if d.reflex_quarantined == 0 {
            Theme::success()
        } else {
            Theme::error()
        };
        lines.push(Line::from(vec![
            Span::styled("Quarantined:  ", Theme::muted()),
            Span::styled(format!("{}", d.reflex_quarantined), q_style),
        ]));

        lines.push(Line::from(""));

        // Reflex mode
        lines.push(Line::from(Span::styled("── Runtime ──", Theme::muted())));
        lines.push(Line::from(vec![
            Span::styled("Mode:         ", Theme::muted()),
            Span::styled(&d.reflex_mode, Theme::text()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Active Rules: ", Theme::muted()),
            Span::styled(format!("{}", d.reflex_rules), Theme::text()),
        ]));

        lines.push(Line::from(""));

        // Sovereignty tier progression
        lines.push(Line::from(Span::styled(
            "── Sovereignty ──",
            Theme::muted(),
        )));

        let tiers = ["DEGRADED", "SEEDLING", "CITIZEN", "SOVEREIGN"];
        let current_idx = tiers
            .iter()
            .position(|t| *t == d.sovereignty_tier)
            .unwrap_or(0);

        let mut tier_spans = Vec::new();
        for (i, tier) in tiers.iter().enumerate() {
            let style = if i == current_idx {
                match *tier {
                    "SOVEREIGN" => Theme::ihsan(),
                    "CITIZEN" => Theme::success(),
                    "SEEDLING" => Theme::warning(),
                    _ => Theme::error(),
                }
            } else {
                Theme::muted()
            };

            if i == current_idx {
                tier_spans.push(Span::styled(format!("[{tier}]"), style));
            } else {
                tier_spans.push(Span::styled(format!(" {tier} "), style));
            }

            if i < tiers.len() - 1 {
                let arrow_style = if i < current_idx {
                    Theme::success()
                } else {
                    Theme::muted()
                };
                tier_spans.push(Span::styled(symbols::ARROW_RIGHT.to_string(), arrow_style));
            }
        }
        lines.push(Line::from(tier_spans));

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

/// Rate bar [0.0 .. 1.0]
fn rate_bar(rate: f32, width: usize) -> String {
    if width < 4 {
        return String::new();
    }
    let bar_w = width.min(12);
    let filled = ((rate * bar_w as f32).round() as usize).min(bar_w);
    let empty = bar_w - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
