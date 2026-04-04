//! Parliament Panel — PAT-7 + SAT-5 agent roster
//!
//! Displays the full sovereign agent parliament in compact list format.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::{
    commands::genesis_spine::AgentInfo,
    theme::{borders, Theme},
};

pub struct ParliamentPanel<'a> {
    pat_agents: &'a [AgentInfo],
    sat_agents: &'a [AgentInfo],
}

impl<'a> ParliamentPanel<'a> {
    pub fn new(pat: &'a [AgentInfo], sat: &'a [AgentInfo]) -> Self {
        Self {
            pat_agents: pat,
            sat_agents: sat,
        }
    }
}

impl Widget for ParliamentPanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let total = self.pat_agents.len() + self.sat_agents.len();
        let block = Block::default()
            .title(Span::styled(
                format!(" Parliament ({total}) "),
                Theme::title(),
            ))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // PAT header
        lines.push(Line::from(Span::styled(
            "PAT-7 (Your Council)",
            Theme::ihsan(),
        )));

        // PAT agents
        for agent in self.pat_agents {
            lines.push(Line::from(vec![
                Span::styled(format!(" P{} ", agent.index), Theme::muted()),
                Span::styled(format!("{} ", agent.icon), Theme::text()),
                Span::styled(format!("{:<12}", agent.callsign), Theme::highlight()),
                Span::styled(&*agent.role, Theme::text()),
            ]));
        }

        // Blank separator
        lines.push(Line::from(""));

        // SAT header
        lines.push(Line::from(Span::styled(
            "SAT-5 (System Immune)",
            Theme::subtitle(),
        )));

        // SAT agents
        for agent in self.sat_agents {
            lines.push(Line::from(vec![
                Span::styled(format!(" S{} ", agent.index), Theme::muted()),
                Span::styled(format!("{} ", agent.icon), Theme::text()),
                Span::styled(format!("{:<12}", agent.callsign), Theme::highlight()),
                Span::styled(&*agent.role, Theme::text()),
            ]));
        }

        Paragraph::new(lines).render(inner, buf);
    }
}
