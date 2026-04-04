//! Substrate Panel — Hardware and model inventory
//!
//! CPU, RAM, GPU, and LLM model summary.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use crate::{
    commands::genesis_spine::DashboardData,
    theme::{borders, Theme},
};

pub struct SubstratePanel<'a> {
    cpu_name: &'a str,
    cpu_cores: u32,
    ram_total_gb: f64,
    ram_used_pct: f64,
    gpu_name: Option<&'a str>,
    gpu_used_mb: u64,
    gpu_total_mb: u64,
    gpu_used_pct: f64,
    model_count: usize,
    text_count: usize,
    vision_count: usize,
    platform: &'a str,
}

impl<'a> SubstratePanel<'a> {
    pub fn from_data(data: &'a DashboardData) -> Self {
        let (gpu_name, gpu_used_mb, gpu_total_mb, gpu_used_pct) = data
            .gpu
            .as_ref()
            .map(|g| (Some(g.name.as_str()), g.used_mb, g.total_mb, g.used_pct))
            .unwrap_or((None, 0, 0, 0.0));

        Self {
            cpu_name: &data.cpu_name,
            cpu_cores: data.cpu_cores,
            ram_total_gb: data.ram_total_gb,
            ram_used_pct: data.ram_used_pct,
            gpu_name,
            gpu_used_mb,
            gpu_total_mb,
            gpu_used_pct,
            model_count: data.model_count,
            text_count: data.text_models.len(),
            vision_count: data.vision_models.len(),
            platform: &data.platform,
        }
    }
}

impl Widget for SubstratePanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(Span::styled(" Substrate ", Theme::title()))
            .borders(Borders::ALL)
            .border_set(borders::ARABIC)
            .border_style(Theme::panel_border())
            .style(Theme::panel());

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // CPU — truncate name to fit
        let max_cpu = (inner.width as usize).saturating_sub(12);
        let cpu_display = if self.cpu_name.len() > max_cpu && max_cpu > 3 {
            format!("{}...", &self.cpu_name[..max_cpu - 3])
        } else {
            self.cpu_name.to_string()
        };
        lines.push(Line::from(vec![
            Span::styled(cpu_display, Theme::text()),
            Span::styled(format!(" {} cores", self.cpu_cores), Theme::muted()),
        ]));

        // RAM with usage color
        let ram_style = if self.ram_used_pct > 90.0 {
            Theme::error()
        } else if self.ram_used_pct > 75.0 {
            Theme::warning()
        } else {
            Theme::success()
        };
        lines.push(Line::from(vec![
            Span::styled("RAM:  ", Theme::muted()),
            Span::styled(format!("{:.0} GB", self.ram_total_gb), Theme::text()),
            Span::styled(format!(" ({:.0}% used)", self.ram_used_pct), ram_style),
        ]));

        // GPU
        if let Some(name) = self.gpu_name {
            let gpu_style = if self.gpu_used_pct > 90.0 {
                Theme::error()
            } else if self.gpu_used_pct > 75.0 {
                Theme::warning()
            } else {
                Theme::success()
            };
            let max_gpu = (inner.width as usize).saturating_sub(6);
            let gpu_display = if name.len() > max_gpu && max_gpu > 3 {
                format!("{}...", &name[..max_gpu - 3])
            } else {
                name.to_string()
            };
            lines.push(Line::from(vec![
                Span::styled("GPU:  ", Theme::muted()),
                Span::styled(gpu_display, Theme::text()),
            ]));
            lines.push(Line::from(vec![
                Span::styled("      ", Theme::muted()),
                Span::styled(
                    format!(
                        "{}/{} MB ({:.0}%)",
                        self.gpu_used_mb, self.gpu_total_mb, self.gpu_used_pct
                    ),
                    gpu_style,
                ),
            ]));
        }

        // Models
        let model_style = if self.model_count > 0 {
            Theme::text()
        } else {
            Theme::error()
        };
        lines.push(Line::from(vec![
            Span::styled("LLMs: ", Theme::muted()),
            Span::styled(
                format!(
                    "{} ({} text, {} vision)",
                    self.model_count, self.text_count, self.vision_count
                ),
                model_style,
            ),
        ]));

        // Platform
        lines.push(Line::from(vec![
            Span::styled("Sys:  ", Theme::muted()),
            Span::styled(self.platform, Theme::text()),
        ]));

        Paragraph::new(lines).render(inner, buf);
    }
}
