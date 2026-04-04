//! Receipt Rail — Receipt chain summary and manifest seal
//!
//! Shows chain integrity, today's missions, recent receipts,
//! and the daily BLAKE3 manifest seal.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::{
    commands::genesis_spine::DashboardData,
    theme::{borders, symbols, Theme},
};

pub struct ReceiptRail<'a> {
    total: usize,
    chain_valid: bool,
    today_count: usize,
    today_complete: usize,
    manifest_seal: Option<&'a str>,
    recent: &'a [crate::commands::genesis_spine::ReceiptSummary],
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
                format!(" Receipts ({}) ", self.total),
                Theme::title(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        if self.total == 0 {
            lines.push(Line::from(Span::styled("No receipts yet.", Theme::muted())));
            lines.push(Line::from(Span::styled(
                "Run: bizra mission \"<objective>\"",
                Theme::text(),
            )));
            Paragraph::new(lines).render(inner, buf);
            return;
        }

        // Chain status
        let (chain_mark, chain_style) = if self.chain_valid {
            (symbols::SUCCESS, Theme::success())
        } else {
            (symbols::ERROR, Theme::error())
        };
        lines.push(Line::from(vec![
            Span::styled(format!("Chain: {chain_mark} "), chain_style),
            Span::styled(
                if self.chain_valid {
                    "All hashes valid"
                } else {
                    "CHAIN BROKEN"
                },
                chain_style,
            ),
        ]));

        // Today summary
        let today_style = if self.today_complete == self.today_count && self.today_count > 0 {
            Theme::success()
        } else if self.today_count > 0 {
            Theme::warning()
        } else {
            Theme::muted()
        };
        lines.push(Line::from(vec![
            Span::styled("Today: ", Theme::muted()),
            Span::styled(
                format!(
                    "{} missions ({}{})",
                    self.today_count,
                    self.today_complete,
                    symbols::SUCCESS
                ),
                today_style,
            ),
        ]));
        lines.push(Line::from(""));

        // Recent receipts — as many as fit
        let max_receipts = (inner.height as usize).saturating_sub(5);
        for (i, r) in self.recent.iter().take(max_receipts).enumerate() {
            let state_style = if r.is_success {
                Theme::success()
            } else if r.is_degraded {
                Theme::warning()
            } else {
                Theme::error()
            };
            let num = self.total - i;
            let max_obj = (inner.width as usize).saturating_sub(26);
            let obj_display = if r.objective.len() > max_obj && max_obj > 3 {
                format!("{}...", &r.objective[..max_obj - 3])
            } else {
                r.objective.clone()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("#{num} "), Theme::muted()),
                Span::styled(
                    format!("{}… ", r.id_short.get(..8).unwrap_or(&r.id_short)),
                    Theme::text(),
                ),
                Span::styled(format!("{:<9}", r.state_label), state_style),
                Span::styled(obj_display, Theme::muted()),
            ]));
        }

        // Manifest seal footer
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        if let Some(seal) = self.manifest_seal {
            lines.push(Line::from(vec![
                Span::styled("Manifest: ", Theme::muted()),
                Span::styled(format!("{seal}…"), Theme::ihsan()),
            ]));
        }

        Paragraph::new(lines).render(inner, buf);
    }
}
