//! BIZRA Visual Theme — Arabic-meets-Tech Aesthetic
//!
//! Design Philosophy:
//! - Clean, minimal, purposeful
//! - Arabic calligraphic influence in borders
//! - Dubai night sky color palette
//! - Information hierarchy through color
//! - Ihsān (إحسان) - Excellence in every pixel

use ratatui::style::{Color, Modifier, Style};

/// BIZRA Color Palette — Dubai Night Sky
pub mod colors {
    use super::Color;

    // Primary Colors
    pub const GOLD: Color = Color::Rgb(212, 175, 55);        // إحسان - Excellence
    pub const EMERALD: Color = Color::Rgb(80, 200, 120);     // Success, Active
    pub const AZURE: Color = Color::Rgb(0, 127, 255);        // Information
    pub const PEARL: Color = Color::Rgb(234, 234, 234);      // Text, Borders

    // Background Colors
    pub const DEEP_SPACE: Color = Color::Rgb(10, 10, 20);    // Main background
    pub const MIDNIGHT: Color = Color::Rgb(20, 20, 35);      // Panel background
    pub const TWILIGHT: Color = Color::Rgb(30, 30, 50);      // Highlighted background

    // Semantic Colors
    pub const IHSAN: Color = GOLD;                           // Excellence threshold met
    pub const ACTIVE: Color = EMERALD;                       // Active, success
    pub const WARNING: Color = Color::Rgb(255, 191, 0);      // Amber warning
    pub const DANGER: Color = Color::Rgb(220, 53, 69);       // Error, violation
    pub const MUTED: Color = Color::Rgb(108, 117, 125);      // Inactive, disabled

    // PAT Agent Colors (each agent has a signature color)
    pub const PAT_STRATEGIST: Color = Color::Rgb(147, 112, 219);  // Purple - Strategy
    pub const PAT_RESEARCHER: Color = Color::Rgb(70, 130, 180);   // Steel Blue - Knowledge
    pub const PAT_DEVELOPER: Color = Color::Rgb(34, 139, 34);     // Forest Green - Code
    pub const PAT_ANALYST: Color = Color::Rgb(255, 140, 0);       // Dark Orange - Data
    pub const PAT_REVIEWER: Color = Color::Rgb(178, 34, 34);      // Firebrick - Quality
    pub const PAT_EXECUTOR: Color = Color::Rgb(70, 70, 70);       // Dark Gray - Action
    pub const PAT_GUARDIAN: Color = GOLD;                         // Gold - Protection

    // Voice indicator
    pub const VOICE_ACTIVE: Color = Color::Rgb(138, 43, 226);    // BlueViolet - Speaking
    pub const VOICE_LISTENING: Color = Color::Rgb(0, 191, 255);  // DeepSkyBlue - Listening
}

/// BIZRA Border Styles — Inspired by Arabic Geometry
pub mod borders {
    /// Standard border set
    pub const STANDARD: ratatui::symbols::border::Set = ratatui::symbols::border::ROUNDED;

    /// Double border for important panels
    pub const IMPORTANT: ratatui::symbols::border::Set = ratatui::symbols::border::DOUBLE;

    /// Thick border for active/focused
    pub const FOCUSED: ratatui::symbols::border::Set = ratatui::symbols::border::THICK;

    /// Custom Arabic-inspired corner set (using Unicode box drawing)
    pub const ARABIC: ratatui::symbols::border::Set = ratatui::symbols::border::Set {
        top_left: "╭",
        top_right: "╮",
        bottom_left: "╰",
        bottom_right: "╯",
        vertical_left: "│",
        vertical_right: "│",
        horizontal_top: "─",
        horizontal_bottom: "─",
    };
}

/// Style presets for consistent UI
pub struct Theme;

impl Theme {
    // === Text Styles ===

    pub fn title() -> Style {
        Style::default()
            .fg(colors::GOLD)
            .add_modifier(Modifier::BOLD)
    }

    pub fn subtitle() -> Style {
        Style::default()
            .fg(colors::PEARL)
            .add_modifier(Modifier::ITALIC)
    }

    pub fn text() -> Style {
        Style::default().fg(colors::PEARL)
    }

    pub fn muted() -> Style {
        Style::default().fg(colors::MUTED)
    }

    pub fn highlight() -> Style {
        Style::default()
            .fg(colors::AZURE)
            .add_modifier(Modifier::BOLD)
    }

    pub fn success() -> Style {
        Style::default().fg(colors::ACTIVE)
    }

    pub fn warning() -> Style {
        Style::default().fg(colors::WARNING)
    }

    pub fn error() -> Style {
        Style::default().fg(colors::DANGER)
    }

    pub fn ihsan() -> Style {
        Style::default()
            .fg(colors::IHSAN)
            .add_modifier(Modifier::BOLD)
    }

    // === Panel Styles ===

    pub fn panel() -> Style {
        Style::default()
            .bg(colors::MIDNIGHT)
            .fg(colors::PEARL)
    }

    pub fn panel_focused() -> Style {
        Style::default()
            .bg(colors::TWILIGHT)
            .fg(colors::PEARL)
    }

    pub fn panel_border() -> Style {
        Style::default().fg(colors::MUTED)
    }

    pub fn panel_border_focused() -> Style {
        Style::default().fg(colors::GOLD)
    }

    // === PAT Agent Styles ===

    pub fn pat_agent(role: &str) -> Style {
        let color = match role.to_lowercase().as_str() {
            "strategist" => colors::PAT_STRATEGIST,
            "researcher" => colors::PAT_RESEARCHER,
            "developer" => colors::PAT_DEVELOPER,
            "analyst" => colors::PAT_ANALYST,
            "reviewer" => colors::PAT_REVIEWER,
            "executor" => colors::PAT_EXECUTOR,
            "guardian" => colors::PAT_GUARDIAN,
            _ => colors::PEARL,
        };
        Style::default().fg(color)
    }

    pub fn pat_agent_active(role: &str) -> Style {
        Self::pat_agent(role).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    }

    // === Status Indicators ===

    pub fn status_active() -> Style {
        Style::default()
            .fg(colors::ACTIVE)
            .add_modifier(Modifier::BOLD)
    }

    pub fn status_pending() -> Style {
        Style::default().fg(colors::WARNING)
    }

    pub fn status_error() -> Style {
        Style::default()
            .fg(colors::DANGER)
            .add_modifier(Modifier::BOLD)
    }

    // === Voice Styles ===

    pub fn voice_active() -> Style {
        Style::default()
            .fg(colors::VOICE_ACTIVE)
            .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK)
    }

    pub fn voice_listening() -> Style {
        Style::default()
            .fg(colors::VOICE_LISTENING)
            .add_modifier(Modifier::BOLD)
    }

    // === Progress Bar Styles ===

    pub fn gauge_filled() -> Style {
        Style::default().fg(colors::GOLD).bg(colors::MIDNIGHT)
    }

    pub fn gauge_unfilled() -> Style {
        Style::default().fg(colors::MUTED).bg(colors::DEEP_SPACE)
    }

    // === Selection Styles ===

    pub fn selected() -> Style {
        Style::default()
            .bg(colors::TWILIGHT)
            .fg(colors::GOLD)
            .add_modifier(Modifier::BOLD)
    }

    pub fn unselected() -> Style {
        Style::default().fg(colors::PEARL)
    }
}

/// Unicode symbols for BIZRA UI
pub mod symbols {
    // Status indicators
    pub const ACTIVE: &str = "●";
    pub const INACTIVE: &str = "○";
    pub const PENDING: &str = "◐";
    pub const ERROR: &str = "✗";
    pub const SUCCESS: &str = "✓";
    pub const WARNING: &str = "⚠";

    // Voice indicators
    pub const VOICE_ON: &str = "🎤";
    pub const VOICE_OFF: &str = "🔇";
    pub const LISTENING: &str = "👂";
    pub const SPEAKING: &str = "🔊";

    // PAT indicators
    pub const AGENT: &str = "◆";
    pub const AGENT_ACTIVE: &str = "◇";

    // FATE Gates
    pub const GATE_PASS: &str = "✓";
    pub const GATE_FAIL: &str = "✗";
    pub const GATE_PENDING: &str = "○";

    // Navigation
    pub const ARROW_RIGHT: &str = "→";
    pub const ARROW_LEFT: &str = "←";
    pub const ARROW_UP: &str = "↑";
    pub const ARROW_DOWN: &str = "↓";

    // Separators
    pub const SEPARATOR: &str = "│";
    pub const DOT: &str = "·";
    pub const BULLET: &str = "•";

    // Arabic-inspired
    pub const BISMILLAH: &str = "﷽";
    pub const STAR: &str = "✦";
    pub const CRESCENT: &str = "☾";
}

/// Format a metric value with color based on threshold
pub fn metric_style(value: f64, threshold: f64, inverse: bool) -> Style {
    let passes = if inverse { value <= threshold } else { value >= threshold };
    if passes {
        Theme::ihsan()
    } else if (value - threshold).abs() < 0.05 {
        Theme::warning()
    } else {
        Theme::error()
    }
}

/// Format Ihsān score with appropriate styling
pub fn ihsan_style(score: f64) -> Style {
    if score >= 0.95 {
        Theme::ihsan()
    } else if score >= 0.85 {
        Theme::warning()
    } else {
        Theme::error()
    }
}
