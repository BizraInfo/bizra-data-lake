//! BIZRA Configuration Module
//!
//! Loads and manages all configuration files for the CLI/TUI.

use anyhow::Result;
use chrono::Timelike;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration paths
pub struct ConfigPaths {
    pub base: PathBuf,
    pub profile: PathBuf,
    pub mcp: PathBuf,
    pub a2a: PathBuf,
    pub commands: PathBuf,
    pub hooks: PathBuf,
    pub prompts: PathBuf,
    pub skills: PathBuf,
    pub proactive: PathBuf,
}

impl ConfigPaths {
    pub fn new(base: PathBuf) -> Self {
        Self {
            profile: base.join("sovereign_profile.yaml"),
            mcp: base.join("mcp_servers.yaml"),
            a2a: base.join("a2a_protocol.yaml"),
            commands: base.join("slash_commands.yaml"),
            hooks: base.join("hooks.yaml"),
            prompts: base.join("prompt_library.yaml"),
            skills: base.join("skills.yaml"),
            proactive: base.join("proactive.yaml"),
            base,
        }
    }

    pub fn default() -> Self {
        let base = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("config");
        Self::new(base)
    }
}

/// User identity from profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    pub title: String,
    pub location: String,
    pub timezone: String,
    pub node_id: String,
    pub genesis_hash: String,
}

impl Default for Identity {
    fn default() -> Self {
        Self {
            name: "MoMo (محمد)".to_string(),
            title: "Sovereign Node0 Architect".to_string(),
            location: "Dubai, UAE".to_string(),
            timezone: "GMT+4".to_string(),
            node_id: "node0_ce5af35c848ce889".to_string(),
            genesis_hash: "a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b"
                .to_string(),
        }
    }
}

/// FATE Gate thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FATEThresholds {
    pub ihsan: f64,
    pub adl_gini: f64,
    pub harm: f64,
    pub confidence: f64,
}

impl Default for FATEThresholds {
    fn default() -> Self {
        Self {
            ihsan: 0.95,
            adl_gini: 0.35,
            harm: 0.30,
            confidence: 0.80,
        }
    }
}

/// Working patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingPatterns {
    pub deep_work_hours: Vec<String>,
    pub review_time: String,
    pub planning_time: String,
    pub communication_style: String,
    pub decision_style: String,
    pub autonomy_level: String,
}

impl Default for WorkingPatterns {
    fn default() -> Self {
        Self {
            deep_work_hours: vec!["09:00-12:00".to_string(), "15:00-18:00".to_string()],
            review_time: "20:00".to_string(),
            planning_time: "08:00".to_string(),
            communication_style: "concise".to_string(),
            decision_style: "calculated".to_string(),
            autonomy_level: "high".to_string(),
        }
    }
}

/// PAT Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PATAgentConfig {
    pub voice: String,
    pub personality: String,
    pub specialties: Vec<String>,
    pub auto_engage_on: Vec<String>,
}

/// Main configuration structure
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub identity: Identity,
    pub fate_thresholds: FATEThresholds,
    pub patterns: WorkingPatterns,
    pub pat_agents: HashMap<String, PATAgentConfig>,
    pub slash_commands: HashMap<String, SlashCommand>,
    pub proactive_mode: String,
}

/// Slash command definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashCommand {
    pub name: String,
    pub description: String,
    pub agent: Option<String>,
    pub shortcut: Option<String>,
}

impl Config {
    /// Load configuration from files
    pub fn load() -> Result<Self> {
        let paths = ConfigPaths::default();
        Self::load_from(paths)
    }

    /// Load configuration from specific paths
    pub fn load_from(paths: ConfigPaths) -> Result<Self> {
        // For now, return defaults
        // In production, parse YAML files
        Ok(Self::default())
    }

    /// Get PAT agent config
    pub fn get_agent(&self, name: &str) -> Option<&PATAgentConfig> {
        self.pat_agents.get(name)
    }

    /// Check if current time is in deep work hours
    pub fn is_deep_work_time(&self) -> bool {
        use chrono::Local;
        let now = Local::now();
        let hour = now.hour();

        // Simplified check - in production, parse the time ranges
        (9..=12).contains(&hour) || (15..=18).contains(&hour)
    }

    /// Get greeting based on time of day
    pub fn get_greeting(&self) -> &'static str {
        use chrono::Local;
        let hour = Local::now().hour();

        match hour {
            6..=11 => "صباح الخير (Good morning)",
            12..=16 => "مساء الخير (Good afternoon)",
            17..=21 => "مساء النور (Good evening)",
            _ => "أهلاً (Hello)",
        }
    }
}

/// Proactive suggestion
#[derive(Debug, Clone)]
pub struct ProactiveSuggestion {
    pub message: String,
    pub action: Option<String>,
    pub priority: SuggestionPriority,
    pub category: SuggestionCategory,
}

#[derive(Debug, Clone, Copy)]
pub enum SuggestionPriority {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy)]
pub enum SuggestionCategory {
    QuickWin,
    NextLogical,
    GoalAligned,
    LearningOpportunity,
}

impl ProactiveSuggestion {
    pub fn quick_win(message: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            action: Some(action.into()),
            priority: SuggestionPriority::Medium,
            category: SuggestionCategory::QuickWin,
        }
    }

    pub fn next_step(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            action: None,
            priority: SuggestionPriority::Low,
            category: SuggestionCategory::NextLogical,
        }
    }
}

/// Proactive engine for generating suggestions
pub struct ProactiveEngine {
    config: Config,
    last_suggestion_time: Option<chrono::DateTime<chrono::Local>>,
}

impl ProactiveEngine {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            last_suggestion_time: None,
        }
    }

    /// Generate contextual suggestion
    pub fn suggest(&mut self) -> Option<ProactiveSuggestion> {
        use chrono::Local;
        let now = Local::now();

        // Don't suggest too frequently
        if let Some(last) = self.last_suggestion_time {
            if (now - last).num_seconds() < 60 {
                return None;
            }
        }

        self.last_suggestion_time = Some(now);

        // Context-aware suggestions
        let hour = now.hour();

        match hour {
            8 => Some(ProactiveSuggestion::next_step(
                "Morning! Run /morning for your daily brief",
            )),
            12 => Some(ProactiveSuggestion::quick_win(
                "Midday check: any quick tasks to clear?",
                "/task list quick",
            )),
            17 => Some(ProactiveSuggestion::next_step(
                "End of day: capture today's learnings with /learn",
            )),
            20 => Some(ProactiveSuggestion::next_step(
                "Evening review time. Run /daily-review?",
            )),
            _ => None,
        }
    }

    /// Check if we should show morning brief
    pub fn should_show_morning_brief(&self) -> bool {
        use chrono::Local;
        let now = Local::now();
        now.hour() >= 6 && now.hour() <= 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.identity.name, "MoMo (محمد)");
        assert_eq!(config.fate_thresholds.ihsan, 0.95);
    }

    #[test]
    fn test_greeting() {
        let config = Config::default();
        let greeting = config.get_greeting();
        assert!(!greeting.is_empty());
    }
}
