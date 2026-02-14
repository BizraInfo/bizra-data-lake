//! Status Bar Widget
//!
//! Bottom status bar with mode indicator, help hints, and messages.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::app::InputMode;
use crate::theme::{colors, symbols, Theme};

pub struct StatusBar<'a> {
    mode: InputMode,
    message: Option<&'a str>,
    selected_agent: Option<&'a str>,
}

impl<'a> StatusBar<'a> {
    pub fn new(mode: InputMode) -> Self {
        Self {
            mode,
            message: None,
            selected_agent: None,
        }
    }

    pub fn message(mut self, msg: Option<&'a str>) -> Self {
        self.message = msg;
        self
    }

    pub fn agent(mut self, agent: Option<&'a str>) -> Self {
        self.selected_agent = agent;
        self
    }
}

impl Widget for StatusBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(colors::MUTED))
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        // Mode indicator
        let (mode_text, mode_style) = match self.mode {
            InputMode::Normal => ("NORMAL", Theme::muted()),
            InputMode::Editing => ("INSERT", Theme::success()),
            InputMode::Command => ("COMMAND", Theme::highlight()),
        };

        let mut spans = vec![
            Span::styled(format!(" {} ", mode_text), mode_style),
            Span::raw(" │ "),
        ];

        // Selected agent
        if let Some(agent) = self.selected_agent {
            spans.push(Span::styled(
                format!("{} {} ", symbols::AGENT, agent),
                Theme::pat_agent(agent),
            ));
            spans.push(Span::raw("│ "));
        }

        // Help hints based on mode
        let hints = match self.mode {
            InputMode::Normal => "q:Quit  Tab:View  j/k:Nav  i:Insert  /:Command",
            InputMode::Editing => "Esc:Normal  Enter:Send  ↑↓:History",
            InputMode::Command => "Esc:Cancel  Enter:Execute  Tab:Complete",
        };
        spans.push(Span::styled(hints, Theme::muted()));

        // Status message (right-aligned)
        if let Some(msg) = self.message {
            let left_len: usize = spans.iter().map(|s| s.content.len()).sum();
            let msg_len = msg.len() + 2;

            if area.width as usize > left_len + msg_len {
                let padding = area.width as usize - left_len - msg_len;
                spans.push(Span::raw(" ".repeat(padding)));
                spans.push(Span::styled(msg, Theme::warning()));
            }
        }

        let line = Line::from(spans);
        Paragraph::new(line).render(inner, buf);
    }
}
