//! PAT Agent Card Widget
//!
//! Displays a PAT agent with status, role, and activity.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::app::{AgentState, AgentStatus};
use crate::theme::{borders, symbols, Theme};

pub struct AgentCard<'a> {
    state: &'a AgentState,
    selected: bool,
    compact: bool,
}

impl<'a> AgentCard<'a> {
    pub fn new(state: &'a AgentState) -> Self {
        Self {
            state,
            selected: false,
            compact: false,
        }
    }

    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    pub fn compact(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }
}

impl Widget for AgentCard<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let role = self.state.role;

        // Border style based on selection
        let (border_style, border_set) = if self.selected {
            (Theme::panel_border_focused(), borders::FOCUSED)
        } else {
            (Theme::panel_border(), borders::STANDARD)
        };

        // Status indicator
        let status_symbol = match self.state.status {
            AgentStatus::Idle => symbols::INACTIVE,
            AgentStatus::Thinking => symbols::PENDING,
            AgentStatus::Speaking => symbols::SPEAKING,
            AgentStatus::Listening => symbols::LISTENING,
            AgentStatus::Error => symbols::ERROR,
        };

        let _status_style = match self.state.status {
            AgentStatus::Idle => Theme::muted(),
            AgentStatus::Thinking => Theme::status_pending(),
            AgentStatus::Speaking => Theme::voice_active(),
            AgentStatus::Listening => Theme::voice_listening(),
            AgentStatus::Error => Theme::error(),
        };

        // Title with icon and status
        let title = format!(" {} {} {} ", role.icon(), role.name(), status_symbol);

        let block = Block::default()
            .title(Span::styled(title, Theme::pat_agent(role.name())))
            .borders(Borders::ALL)
            .border_style(border_style)
            .border_set(border_set)
            .style(if self.selected {
                Theme::panel_focused()
            } else {
                Theme::panel()
            });

        let inner = block.inner(area);
        block.render(area, buf);

        if self.compact || inner.height < 3 {
            // Compact mode: just show current task
            if let Some(task) = &self.state.current_task {
                let line = Line::from(Span::styled(
                    truncate(task, inner.width as usize),
                    Theme::text(),
                ));
                Paragraph::new(line).render(inner, buf);
            }
        } else {
            // Full mode: show giants and more info
            let mut lines = Vec::new();

            // Giants line
            let giants: String = role.giants().join(" • ");
            lines.push(Line::from(vec![
                Span::styled("Giants: ", Theme::muted()),
                Span::styled(
                    truncate(&giants, inner.width as usize - 8),
                    Theme::highlight(),
                ),
            ]));

            // Current task
            if let Some(task) = &self.state.current_task {
                lines.push(Line::from(vec![
                    Span::styled("Task: ", Theme::muted()),
                    Span::styled(truncate(task, inner.width as usize - 6), Theme::text()),
                ]));
            } else {
                lines.push(Line::from(Span::styled("Ready", Theme::success())));
            }

            // Message count
            lines.push(Line::from(vec![
                Span::styled("Messages: ", Theme::muted()),
                Span::styled(self.state.messages_count.to_string(), Theme::text()),
            ]));

            Paragraph::new(lines).render(inner, buf);
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}
