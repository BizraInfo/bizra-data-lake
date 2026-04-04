//! Wallet Widget — SEED balance, Gini, zakat, supply cap
//!
//! Displays sovereign treasury state derived from constitutional constants
//! and the receipt chain. Same truth path — no shadow ledger.

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

pub struct Wallet<'a> {
    data: &'a DashboardData,
}

impl<'a> Wallet<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self { data }
    }
}

impl Widget for Wallet<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let d = self.data;

        let block = Block::default()
            .title(Span::styled(
                format!(" {} Treasury ", symbols::STAR),
                Theme::title(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // SEED Balance
        let balance_style = if d.seed_balance >= 100.0 {
            Theme::ihsan()
        } else if d.seed_balance >= 10.0 {
            Theme::success()
        } else if d.seed_balance > 0.0 {
            Theme::warning()
        } else {
            Theme::muted()
        };
        lines.push(Line::from(vec![
            Span::styled("SEED Balance: ", Theme::muted()),
            Span::styled(format!("{:.3}", d.seed_balance), balance_style),
            Span::styled(" SEED", Theme::muted()),
        ]));

        lines.push(Line::from(""));

        // Gini coefficient
        let gini_style = if d.seed_gini <= 0.35 {
            Theme::success()
        } else if d.seed_gini <= 0.60 {
            Theme::warning()
        } else {
            Theme::error()
        };
        let gini_bar = gauge_bar(
            d.seed_gini as f32,
            0.35,
            inner.width.saturating_sub(24) as usize,
        );
        lines.push(Line::from(vec![
            Span::styled("Gini (Adl):   ", Theme::muted()),
            Span::styled(format!("{:.3}", d.seed_gini), gini_style),
            Span::styled(format!(" {gini_bar}"), gini_style),
        ]));

        // Zakat
        lines.push(Line::from(vec![
            Span::styled("Zakat Rate:   ", Theme::muted()),
            Span::styled(format!("{:.1}%", d.zakat_rate * 100.0), Theme::ihsan()),
            Span::styled(
                format!("  ({:.3} deducted)", d.zakat_deducted),
                Theme::muted(),
            ),
        ]));

        lines.push(Line::from(""));

        // Supply
        lines.push(Line::from(Span::styled("── Supply ──", Theme::muted())));
        lines.push(Line::from(vec![
            Span::styled("Minted:       ", Theme::muted()),
            Span::styled(format!("{:.3}", d.total_minted), Theme::success()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Burned:       ", Theme::muted()),
            Span::styled(format!("{:.3}", d.total_burned), Theme::error()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Supply Cap:   ", Theme::muted()),
            Span::styled(format!("{:.0}", d.seed_supply_cap), Theme::text()),
        ]));

        lines.push(Line::from(""));

        // Sovereignty tier
        let tier_style = match d.sovereignty_tier {
            "SOVEREIGN" => Theme::ihsan(),
            "CITIZEN" => Theme::success(),
            "SEEDLING" => Theme::warning(),
            _ => Theme::error(),
        };
        lines.push(Line::from(vec![
            Span::styled("Tier:         ", Theme::muted()),
            Span::styled(d.sovereignty_tier, tier_style),
        ]));

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

/// Render a threshold gauge: filled up to value, threshold marker
fn gauge_bar(value: f32, threshold: f32, width: usize) -> String {
    if width < 4 {
        return String::new();
    }
    let bar_w = width.min(16);
    let filled = ((value / threshold.max(0.01) * bar_w as f32).round() as usize).min(bar_w);
    let empty = bar_w - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
