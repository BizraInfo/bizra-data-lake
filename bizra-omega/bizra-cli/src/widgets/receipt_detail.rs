//! Receipt Detail — Full receipt view with Ihsan, SNR, signature, chain
//!
//! Shows the complete details of a selected receipt including
//! quality scores, model used, signature status, and chain link.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
};

use crate::{
    commands::genesis_spine::ReceiptSummary,
    theme::{borders, symbols, Theme},
};

pub struct ReceiptDetail<'a> {
    receipt: &'a ReceiptSummary,
    index: usize,
    total: usize,
}

impl<'a> ReceiptDetail<'a> {
    pub fn new(receipt: &'a ReceiptSummary, index: usize, total: usize) -> Self {
        Self {
            receipt,
            index,
            total,
        }
    }
}

impl Widget for ReceiptDetail<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let r = self.receipt;

        let block = Block::default()
            .title(Span::styled(
                format!(
                    " Receipt #{} of {} — {} ",
                    self.total - self.index,
                    self.total,
                    &r.id_short[..8.min(r.id_short.len())]
                ),
                Theme::title(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border_focused())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // Receipt ID
        lines.push(Line::from(vec![
            Span::styled("ID:       ", Theme::muted()),
            Span::styled(&r.id_short, Theme::text()),
        ]));

        // Objective
        lines.push(Line::from(vec![
            Span::styled("Mission:  ", Theme::muted()),
            Span::styled(&r.objective, Theme::highlight()),
        ]));

        // State
        let state_style = if r.is_success {
            Theme::success()
        } else if r.is_degraded {
            Theme::warning()
        } else {
            Theme::error()
        };
        lines.push(Line::from(vec![
            Span::styled("State:    ", Theme::muted()),
            Span::styled(r.state_label, state_style),
            Span::styled(format!("  (tier {})", r.degradation_tier), Theme::muted()),
        ]));

        lines.push(Line::from(""));

        // ── Quality Scores ──
        lines.push(Line::from(Span::styled("── Quality ──", Theme::muted())));

        // Ihsan score
        if let Some(ihsan) = r.ihsan_score {
            let ihsan_style = if ihsan >= 0.95 {
                Theme::ihsan()
            } else if ihsan >= 0.85 {
                Theme::warning()
            } else {
                Theme::error()
            };
            let bar = score_bar(ihsan, inner.width.saturating_sub(20) as usize);
            lines.push(Line::from(vec![
                Span::styled("Ihsan:    ", Theme::muted()),
                Span::styled(format!("{:.2}", ihsan), ihsan_style),
                Span::styled(format!(" {bar}"), ihsan_style),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("Ihsan:    ", Theme::muted()),
                Span::styled("—", Theme::muted()),
            ]));
        }

        // SNR score
        if let Some(snr) = r.snr_score {
            let snr_style = if snr >= 0.85 {
                Theme::success()
            } else if snr >= 0.70 {
                Theme::warning()
            } else {
                Theme::error()
            };
            let bar = score_bar(snr, inner.width.saturating_sub(20) as usize);
            lines.push(Line::from(vec![
                Span::styled("SNR:      ", Theme::muted()),
                Span::styled(format!("{:.2}", snr), snr_style),
                Span::styled(format!(" {bar}"), snr_style),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("SNR:      ", Theme::muted()),
                Span::styled("—", Theme::muted()),
            ]));
        }

        lines.push(Line::from(""));

        // ── Provenance ──
        lines.push(Line::from(Span::styled("── Provenance ──", Theme::muted())));

        // Model
        let model_display = r.chosen_model.as_deref().unwrap_or("unknown");
        lines.push(Line::from(vec![
            Span::styled("Model:    ", Theme::muted()),
            Span::styled(model_display, Theme::text()),
        ]));

        // Signature
        let (sig_icon, sig_text, sig_style) = if r.signed {
            (symbols::SUCCESS, "Ed25519 verified", Theme::success())
        } else {
            (symbols::ERROR, "Unsigned", Theme::error())
        };
        lines.push(Line::from(vec![
            Span::styled("Signed:   ", Theme::muted()),
            Span::styled(format!("{sig_icon} {sig_text}"), sig_style),
        ]));

        // States traversed
        lines.push(Line::from(vec![
            Span::styled("States:   ", Theme::muted()),
            Span::styled(format!("{} transitions", r.states_traversed), Theme::text()),
        ]));

        // Chain link
        if let Some(ref link) = r.chain_link {
            lines.push(Line::from(vec![
                Span::styled("Chain:    ", Theme::muted()),
                Span::styled(format!("{} {link}…", symbols::ARROW_LEFT), Theme::text()),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("Chain:    ", Theme::muted()),
                Span::styled(format!("{} Genesis receipt", symbols::STAR), Theme::ihsan()),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "j/k navigate  Esc close",
            Theme::muted(),
        )));

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

/// Render a simple ASCII score bar: [████░░░░░░] style
fn score_bar(score: f32, width: usize) -> String {
    if width < 4 {
        return String::new();
    }
    let bar_w = width.min(20);
    let filled = ((score * bar_w as f32).round() as usize).min(bar_w);
    let empty = bar_w - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
