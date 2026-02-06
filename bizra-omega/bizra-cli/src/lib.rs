//! BIZRA CLI/TUI Library
//!
//! بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
//!
//! Your Personal Command Center for the Sovereign Node.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           BIZRA CLI/TUI                                 │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  Config Layer   │ Profile, MCP, A2A, Skills, Hooks, Prompts            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  App Layer      │ State management, PAT agents, FATE gates             │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  Command Layer  │ Slash commands, proactive suggestions                 │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  Widget Layer   │ TUI components, theme, rendering                      │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod app;
pub mod commands;
pub mod config;
pub mod inference;
pub mod theme;
pub mod widgets;

pub use app::App;
pub use config::Config;
pub use inference::LMStudio;
pub use theme::Theme;
