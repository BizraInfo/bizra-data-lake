//! BIZRA Application State
//!
//! Central state management for the TUI application.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PAT Agent roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PATRole {
    Strategist,
    Researcher,
    Developer,
    Analyst,
    Reviewer,
    Executor,
    Guardian,
}

impl PATRole {
    pub fn all() -> &'static [PATRole] {
        &[
            PATRole::Strategist,
            PATRole::Researcher,
            PATRole::Developer,
            PATRole::Analyst,
            PATRole::Reviewer,
            PATRole::Executor,
            PATRole::Guardian,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            PATRole::Strategist => "Strategist",
            PATRole::Researcher => "Researcher",
            PATRole::Developer => "Developer",
            PATRole::Analyst => "Analyst",
            PATRole::Reviewer => "Reviewer",
            PATRole::Executor => "Executor",
            PATRole::Guardian => "Guardian",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            PATRole::Strategist => "♟",
            PATRole::Researcher => "🔍",
            PATRole::Developer => "⚙",
            PATRole::Analyst => "📊",
            PATRole::Reviewer => "✓",
            PATRole::Executor => "▶",
            PATRole::Guardian => "🛡",
        }
    }

    pub fn giants(&self) -> &'static [&'static str] {
        match self {
            PATRole::Strategist => &["Sun Tzu", "Clausewitz", "Porter"],
            PATRole::Researcher => &["Shannon", "Turing", "Dijkstra"],
            PATRole::Developer => &["Knuth", "Ritchie", "Torvalds"],
            PATRole::Analyst => &["Tukey", "Tufte", "Cleveland"],
            PATRole::Reviewer => &["Fagan", "Parnas", "Brooks"],
            PATRole::Executor => &["Toyota", "Deming", "Ohno"],
            PATRole::Guardian => &["Al-Ghazali", "Rawls", "Anthropic"],
        }
    }
}

/// FATE Gate status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FATEGates {
    pub ihsan: f64,      // Excellence (≥0.95)
    pub adl_gini: f64,   // Justice/Fairness (≤0.35)
    pub harm: f64,       // Harm score (≤0.30)
    pub confidence: f64, // Confidence (≥0.80)
}

impl Default for FATEGates {
    fn default() -> Self {
        Self {
            ihsan: 0.0,
            adl_gini: 1.0,
            harm: 1.0,
            confidence: 0.0,
        }
    }
}

impl FATEGates {
    pub fn all_pass(&self) -> bool {
        self.ihsan >= 0.95 && self.adl_gini <= 0.35 && self.harm <= 0.30 && self.confidence >= 0.80
    }
}

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Thinking,
    Speaking,
    Listening,
    Error,
}

/// PAT Agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub role: PATRole,
    pub status: AgentStatus,
    pub current_task: Option<String>,
    pub last_response: Option<String>,
    pub messages_count: u32,
}

impl AgentState {
    pub fn new(role: PATRole) -> Self {
        Self {
            role,
            status: AgentStatus::Idle,
            current_task: None,
            last_response: None,
            messages_count: 0,
        }
    }
}

/// System metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub gpu_memory: f64,
    pub inference_latency_ms: u64,
    pub tokens_per_second: f64,
}

/// Active view/tab
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveView {
    #[default]
    Dashboard,
    Agents,
    Chat,
    Tasks,
    Treasury,
    Settings,
}

impl ActiveView {
    pub fn all() -> &'static [ActiveView] {
        &[
            ActiveView::Dashboard,
            ActiveView::Agents,
            ActiveView::Chat,
            ActiveView::Tasks,
            ActiveView::Treasury,
            ActiveView::Settings,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ActiveView::Dashboard => "Dashboard",
            ActiveView::Agents => "PAT Agents",
            ActiveView::Chat => "Chat",
            ActiveView::Tasks => "Tasks",
            ActiveView::Treasury => "Treasury",
            ActiveView::Settings => "Settings",
        }
    }

    pub fn key(&self) -> char {
        match self {
            ActiveView::Dashboard => '1',
            ActiveView::Agents => '2',
            ActiveView::Chat => '3',
            ActiveView::Tasks => '4',
            ActiveView::Treasury => '5',
            ActiveView::Settings => '6',
        }
    }
}

/// Input mode for chat
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    #[default]
    Normal,
    Editing,
    Command,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "user", "assistant", or agent name
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub agent: Option<PATRole>,
}

/// Task item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskItem {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub assigned_to: Option<PATRole>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Review,
    Completed,
    Blocked,
}

/// Main application state
pub struct App {
    /// Current active view
    pub active_view: ActiveView,

    /// Input mode
    pub input_mode: InputMode,

    /// Current input buffer
    pub input: String,

    /// Command history
    pub command_history: Vec<String>,
    pub history_index: Option<usize>,

    /// PAT agent states
    pub agents: HashMap<PATRole, AgentState>,

    /// Currently selected agent
    pub selected_agent: Option<PATRole>,

    /// FATE gate metrics
    pub fate_gates: FATEGates,

    /// System metrics
    pub system_metrics: SystemMetrics,

    /// Chat messages
    pub chat_messages: Vec<ChatMessage>,
    pub chat_scroll: usize,

    /// Tasks
    pub tasks: Vec<TaskItem>,
    pub selected_task: Option<usize>,

    /// Node info
    pub node_id: String,
    pub node_name: String,
    pub genesis_hash: String,

    /// LM Studio status
    pub lmstudio_connected: bool,
    pub lmstudio_model: Option<String>,

    /// Voice status
    pub voice_active: bool,
    pub voice_listening: bool,

    /// Should quit
    pub should_quit: bool,

    /// Status message
    pub status_message: Option<(String, chrono::DateTime<Utc>)>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        let mut agents = HashMap::new();
        for role in PATRole::all() {
            agents.insert(*role, AgentState::new(*role));
        }

        Self {
            active_view: ActiveView::Dashboard,
            input_mode: InputMode::Normal,
            input: String::new(),
            command_history: Vec::new(),
            history_index: None,
            agents,
            selected_agent: Some(PATRole::Guardian),
            fate_gates: FATEGates::default(),
            system_metrics: SystemMetrics::default(),
            chat_messages: Vec::new(),
            chat_scroll: 0,
            tasks: Vec::new(),
            selected_task: None,
            node_id: "node0_ce5af35c848ce889".to_string(),
            node_name: "MoMo (محمد)".to_string(),
            genesis_hash: "a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b"
                .to_string(),
            lmstudio_connected: false,
            lmstudio_model: None,
            voice_active: false,
            voice_listening: false,
            should_quit: false,
            status_message: None,
        }
    }

    /// Set status message (auto-clears after 5 seconds)
    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), Utc::now()));
    }

    /// Clear expired status message
    pub fn clear_expired_status(&mut self) {
        if let Some((_, time)) = &self.status_message {
            if Utc::now().signed_duration_since(*time).num_seconds() > 5 {
                self.status_message = None;
            }
        }
    }

    /// Switch to next view
    pub fn next_view(&mut self) {
        let views = ActiveView::all();
        let current_idx = views
            .iter()
            .position(|v| *v == self.active_view)
            .unwrap_or(0);
        self.active_view = views[(current_idx + 1) % views.len()];
    }

    /// Switch to previous view
    pub fn prev_view(&mut self) {
        let views = ActiveView::all();
        let current_idx = views
            .iter()
            .position(|v| *v == self.active_view)
            .unwrap_or(0);
        self.active_view = views[(current_idx + views.len() - 1) % views.len()];
    }

    /// Select next agent
    pub fn next_agent(&mut self) {
        let roles = PATRole::all();
        if let Some(current) = self.selected_agent {
            let idx = roles.iter().position(|r| *r == current).unwrap_or(0);
            self.selected_agent = Some(roles[(idx + 1) % roles.len()]);
        } else {
            self.selected_agent = Some(roles[0]);
        }
    }

    /// Select previous agent
    pub fn prev_agent(&mut self) {
        let roles = PATRole::all();
        if let Some(current) = self.selected_agent {
            let idx = roles.iter().position(|r| *r == current).unwrap_or(0);
            self.selected_agent = Some(roles[(idx + roles.len() - 1) % roles.len()]);
        } else {
            self.selected_agent = Some(roles[roles.len() - 1]);
        }
    }

    /// Add chat message
    pub fn add_message(
        &mut self,
        role: impl Into<String>,
        content: impl Into<String>,
        agent: Option<PATRole>,
    ) {
        self.chat_messages.push(ChatMessage {
            role: role.into(),
            content: content.into(),
            timestamp: Utc::now(),
            agent,
        });
    }

    /// Process user input
    pub fn process_input(&mut self) {
        let input = self.input.trim().to_string();
        if input.is_empty() {
            return;
        }

        // Add to history
        self.command_history.push(input.clone());
        self.history_index = None;

        // Check for commands
        if input.starts_with('/') {
            self.process_command(&input);
        } else {
            // Regular chat message
            self.add_message("user", &input, None);

            // Send to selected agent for response
            if let Some(agent) = self.selected_agent {
                self.set_status(format!("Sending to {}...", agent.name()));

                // Call Python bridge for LLM response
                let agent_name = agent.name().to_lowercase();
                let response = self.call_llm(&agent_name, &input);

                self.add_message(agent.name(), &response, Some(agent));
                self.set_status("Ready");
            }
        }

        self.input.clear();
    }

    /// Process slash command
    fn process_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts.first().copied() {
            Some("/quit") | Some("/q") => {
                self.should_quit = true;
            }
            Some("/agent") | Some("/a") => {
                if let Some(name) = parts.get(1) {
                    match name.to_lowercase().as_str() {
                        "strategist" => self.selected_agent = Some(PATRole::Strategist),
                        "researcher" => self.selected_agent = Some(PATRole::Researcher),
                        "developer" => self.selected_agent = Some(PATRole::Developer),
                        "analyst" => self.selected_agent = Some(PATRole::Analyst),
                        "reviewer" => self.selected_agent = Some(PATRole::Reviewer),
                        "executor" => self.selected_agent = Some(PATRole::Executor),
                        "guardian" => self.selected_agent = Some(PATRole::Guardian),
                        _ => self.set_status(format!("Unknown agent: {}", name)),
                    }
                }
            }
            Some("/voice") | Some("/v") => {
                self.voice_active = !self.voice_active;
                self.set_status(if self.voice_active {
                    "Voice enabled"
                } else {
                    "Voice disabled"
                });
            }
            Some("/clear") => {
                self.chat_messages.clear();
                self.set_status("Chat cleared");
            }
            Some("/help") | Some("/h") => {
                self.add_message("system", r#"
Available commands:
  /agent <name>  - Switch to agent (strategist, researcher, developer, analyst, reviewer, executor, guardian)
  /voice         - Toggle voice mode
  /clear         - Clear chat history
  /quit          - Exit application
  /help          - Show this help

Keyboard shortcuts:
  Tab            - Switch views
  1-6            - Jump to view
  j/k            - Navigate agents
  i              - Enter edit mode
  Esc            - Exit edit mode
  q              - Quit (in normal mode)
"#, None);
            }
            _ => {
                self.set_status(format!("Unknown command: {}", cmd));
            }
        }
    }

    /// Call LLM via Python bridge
    fn call_llm(&self, agent: &str, message: &str) -> String {
        use std::process::Command;

        let bridge_path = "/mnt/c/BIZRA-DATA-LAKE/bizra_cli_bridge.py";
        let python_path = "/mnt/c/BIZRA-DATA-LAKE/.venv/bin/python";

        let mut cmd = Command::new(python_path);
        cmd.args([bridge_path, "agent", agent, message]);

        // Pass API key from environment
        if let Ok(key) = std::env::var("LM_STUDIO_API_KEY") {
            cmd.env("LM_STUDIO_API_KEY", key);
        }

        match cmd.output() {
            Ok(output) => {
                if output.status.success() {
                    // Parse JSON response
                    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                        if let Some(content) = json.get("content").and_then(|c| c.as_str()) {
                            return content.to_string();
                        }
                        if let Some(error) = json.get("error").and_then(|e| e.as_str()) {
                            return format!("Error: {}", error);
                        }
                    }
                    String::from_utf8_lossy(&output.stdout).to_string()
                } else {
                    // Try to parse error from stdout (bridge returns JSON errors)
                    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                        if let Some(error) = json.get("error").and_then(|e| e.as_str()) {
                            return format!("LM Studio: {}", error);
                        }
                    }
                    format!("Error: {}", String::from_utf8_lossy(&output.stderr))
                }
            }
            Err(e) => format!("Failed to call LLM: {}", e),
        }
    }
}
