//! FATE Gate Gauge Widget
//!
//! Displays FATE gate metrics with visual gauges.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Widget},
};

use crate::app::FATEGates;
use crate::theme::{colors, ihsan_style, metric_style, symbols, Theme};

pub struct FateGauge<'a> {
    gates: &'a FATEGates,
    show_labels: bool,
}

impl<'a> FateGauge<'a> {
    pub fn new(gates: &'a FATEGates) -> Self {
        Self {
            gates,
            show_labels: true,
        }
    }

    pub fn show_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }
}

impl Widget for FateGauge<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let all_pass = self.gates.all_pass();

        let block = Block::default()
            .title(Span::styled(
                format!(" {} FATE Gates ", if all_pass { symbols::GATE_PASS } else { symbols::GATE_PENDING }),
                if all_pass { Theme::ihsan() } else { Theme::warning() },
            ))
            .borders(Borders::ALL)
            .border_style(if all_pass { Theme::panel_border_focused() } else { Theme::panel_border() })
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        // Split into 4 gauge areas
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Length(2),
            ])
            .split(inner);

        // Ihsān (Excellence) - must be >= 0.95
        render_gate(buf, chunks[0], "Ihsān", self.gates.ihsan, 0.95, false);

        // Adl (Justice) - Gini must be <= 0.35
        render_gate(buf, chunks[1], "Adl", self.gates.adl_gini, 0.35, true);

        // Harm - must be <= 0.30
        render_gate(buf, chunks[2], "Harm", self.gates.harm, 0.30, true);

        // Confidence - must be >= 0.80
        render_gate(buf, chunks[3], "Conf", self.gates.confidence, 0.80, false);
    }
}

fn render_gate(buf: &mut Buffer, area: Rect, name: &str, value: f64, threshold: f64, inverse: bool) {
    if area.height < 1 {
        return;
    }

    let passes = if inverse { value <= threshold } else { value >= threshold };
    let symbol = if passes { symbols::GATE_PASS } else { symbols::GATE_FAIL };
    let style = metric_style(value, threshold, inverse);

    // Label
    let label = format!("{} {}: {:.2}", symbol, name, value);
    let label_span = Span::styled(label, style);

    if area.height >= 2 && area.width > 15 {
        // Show gauge
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(12), Constraint::Min(5)])
            .split(area);

        Paragraph::new(Line::from(label_span)).render(chunks[0], buf);

        // Gauge
        let ratio = if inverse {
            1.0 - (value / threshold).min(1.0)
        } else {
            (value / threshold).min(1.0)
        };

        let gauge = Gauge::default()
            .ratio(ratio.max(0.0).min(1.0))
            .gauge_style(style)
            .use_unicode(true);
        gauge.render(chunks[1], buf);
    } else {
        // Just show label
        Paragraph::new(Line::from(label_span)).render(area, buf);
    }
}
