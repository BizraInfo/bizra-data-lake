//! Trust Rail — Constitutional compliance surface
//!
//! Renders the 13-check trust surface with pass/fail indicators
//! and SOVEREIGN/DEGRADED verdict banner.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::{
    commands::genesis_spine::{DashboardData, TrustVerdict},
    theme::{borders, symbols, Theme},
};

pub struct TrustRail<'a> {
    checks: &'a [crate::commands::genesis_spine::TrustCheck],
    receipt_checks: &'a [crate::commands::genesis_spine::TrustCheck],
    verdict: TrustVerdict,
}

impl<'a> TrustRail<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        Self {
            checks: &data.trust_checks,
            receipt_checks: &data.receipt_chain_checks,
            verdict: data.trust_verdict,
        }
    }
}

impl Widget for TrustRail<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let is_sovereign = self.verdict == TrustVerdict::Sovereign;

        let block = Block::default()
            .title(Span::styled(" Trust Surface ", Theme::title()))
            .borders(Borders::ALL)
            .border_set(if is_sovereign {
                borders::IMPORTANT
            } else {
                borders::ARABIC
            })
            .border_style(if is_sovereign {
                Theme::panel_border_focused()
            } else {
                Theme::panel_border()
            })
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // Verdict banner
        if is_sovereign {
            lines.push(Line::from(Span::styled(
                format!("{} SOVEREIGN", symbols::SUCCESS),
                Theme::ihsan(),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                format!("{} DEGRADED", symbols::WARNING),
                Theme::warning(),
            )));
        }
        lines.push(Line::from(""));

        // Constitutional Law (first 5 checks)
        lines.push(Line::from(Span::styled(
            "[Constitutional Law]",
            Theme::subtitle(),
        )));
        for check in self.checks.iter().take(5) {
            let (mark, style) = check_style(check.passed);
            lines.push(Line::from(vec![
                Span::styled(format!(" {mark} "), style),
                Span::styled(format!("{:<15}", check.name), Theme::text()),
                Span::styled(&*check.detail, Theme::muted()),
            ]));
        }
        lines.push(Line::from(""));

        // Topology (checks 5-7)
        lines.push(Line::from(Span::styled("[Topology]", Theme::subtitle())));
        for check in self.checks.iter().skip(5).take(3) {
            let (mark, style) = check_style(check.passed);
            lines.push(Line::from(vec![
                Span::styled(format!(" {mark} "), style),
                Span::styled(&*check.name, Theme::text()),
            ]));
        }
        lines.push(Line::from(""));

        // Genesis (check 8)
        lines.push(Line::from(Span::styled("[Genesis]", Theme::subtitle())));
        if let Some(check) = self.checks.get(8) {
            let (mark, style) = check_style(check.passed);
            lines.push(Line::from(vec![
                Span::styled(format!(" {mark} "), style),
                Span::styled(&*check.name, Theme::text()),
            ]));
        }
        lines.push(Line::from(""));

        // Receipt chain
        lines.push(Line::from(Span::styled("[Ledger]", Theme::subtitle())));
        for check in self.receipt_checks {
            let (mark, style) = check_style(check.passed);
            lines.push(Line::from(vec![
                Span::styled(format!(" {mark} "), style),
                Span::styled(&*check.name, Theme::text()),
                Span::styled(format!(" {}", check.detail), Theme::muted()),
            ]));
        }
        lines.push(Line::from(""));

        // Substrate (checks 9-10)
        lines.push(Line::from(Span::styled("[Substrate]", Theme::subtitle())));
        for check in self.checks.iter().skip(9) {
            let (mark, style) = check_style(check.passed);
            lines.push(Line::from(vec![
                Span::styled(format!(" {mark} "), style),
                Span::styled(&*check.name, Theme::text()),
                Span::styled(format!(" {}", check.detail), Theme::muted()),
            ]));
        }

        // Scroll if content exceeds area
        let scroll = if lines.len() > inner.height as usize {
            (lines.len() - inner.height as usize) as u16
        } else {
            0
        };

        Paragraph::new(lines).scroll((scroll, 0)).render(inner, buf);
    }
}

fn check_style(passed: bool) -> (&'static str, ratatui::style::Style) {
    if passed {
        (symbols::SUCCESS, Theme::success())
    } else {
        (symbols::ERROR, Theme::error())
    }
}
