//! BIZRA Header Widget
//!
//! The main header bar with ASCII art logo and navigation.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::app::ActiveView;
use crate::theme::{colors, Theme};

/// ASCII art logo for BIZRA (compact, single line with style)
const LOGO_COMPACT: &str = "◈ BIZRA";

pub struct Header<'a> {
    node_name: &'a str,
    active_view: ActiveView,
    lmstudio_connected: bool,
    voice_active: bool,
}

impl<'a> Header<'a> {
    pub fn new(node_name: &'a str, active_view: ActiveView) -> Self {
        Self {
            node_name,
            active_view,
            lmstudio_connected: false,
            voice_active: false,
        }
    }

    pub fn lmstudio(mut self, connected: bool) -> Self {
        self.lmstudio_connected = connected;
        self
    }

    pub fn voice(mut self, active: bool) -> Self {
        self.voice_active = active;
        self
    }
}

impl Widget for Header<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Background with gradient-like effect
        let block = Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(colors::GOLD))
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        // Left side: Stylized logo
        let logo = Span::styled(
            format!(" {} ", LOGO_COMPACT),
            Style::default()
                .fg(colors::GOLD)
                .add_modifier(Modifier::BOLD),
        );

        // Version badge
        let version = Span::styled(" v1.0 ", Style::default().fg(colors::MUTED));

        // Node name with diamond separator
        let node = Span::styled(format!(" {} ", self.node_name), Theme::subtitle());

        // Navigation tabs with improved styling
        let mut tabs = Vec::new();
        tabs.push(Span::styled(" ║ ", Style::default().fg(colors::MUTED)));

        for view in ActiveView::all() {
            let (style, prefix) = if *view == self.active_view {
                (
                    Style::default()
                        .fg(colors::GOLD)
                        .bg(colors::TWILIGHT)
                        .add_modifier(Modifier::BOLD),
                    "►",
                )
            } else {
                (Theme::unselected(), " ")
            };
            tabs.push(Span::styled(
                format!("{}[{}]{} ", prefix, view.key(), view.name()),
                style,
            ));
        }

        // Right side: Status indicators with improved visuals
        let lm_indicator = if self.lmstudio_connected {
            Span::styled(
                " ● LM Studio ",
                Style::default()
                    .fg(colors::EMERALD)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(" ○ LM Studio ", Theme::muted())
        };

        let voice_indicator = if self.voice_active {
            Span::styled(
                "◉ Voice ",
                Style::default()
                    .fg(colors::VOICE_ACTIVE)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled("○ Voice ", Theme::muted())
        };

        // Build the line
        let mut spans = vec![logo, version, node];
        spans.extend(tabs);

        // Calculate remaining space for right-aligned items
        let left_len: usize = spans.iter().map(|s| s.content.len()).sum();
        let right_len = 26; // Approximate length of status indicators

        if area.width as usize > left_len + right_len + 2 {
            let padding = area.width as usize - left_len - right_len - 2;
            spans.push(Span::raw(" ".repeat(padding)));
        }

        spans.push(Span::styled("│", Style::default().fg(colors::MUTED)));
        spans.push(lm_indicator);
        spans.push(voice_indicator);

        let line = Line::from(spans);
        let para = Paragraph::new(line);
        para.render(inner, buf);
    }
}
