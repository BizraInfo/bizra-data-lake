// bizra-agent/src/types.rs
// ============================================================
// Agent Type System
// ============================================================
// Defines the vocabulary of sovereign agent operation:
// - Who agents are (AgentId, AgentRole)
// - What they process (Message, Task)
// - How they think (AgentContext)
// - What they produce (Response)
//
// The PAT (Personal Agent Team) has 7 specialized roles.
// Each user gets their own team. The team learns together.
// ============================================================

use bizra_hooks::IhsanScore;
use bizra_memory::{AtomKind, Confidence};

// ============================================================
// AGENT IDENTITY
// ============================================================

/// Unique identifier for an agent instance
/// Combines role + user hash for per-user agent identity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u64);

impl AgentId {
    pub fn new(role: AgentRole, user_hash: u32) -> Self {
        Self(((role as u64) << 32) | user_hash as u64)
    }

    pub fn role(&self) -> AgentRole {
        AgentRole::from_u8((self.0 >> 32) as u8)
    }

    pub fn user_hash(&self) -> u32 {
        self.0 as u32
    }
}

// ============================================================
// AGENT ROLES — The 7 PAT Specialists
// ============================================================
// Each role has a distinct responsibility, expertise domain,
// and decision authority. Together they form a complete
// personal intelligence team.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AgentRole {
    /// Intent classification, task routing, workflow orchestration.
    /// The conductor of the PAT orchestra.
    Navigator = 0,

    /// Knowledge retrieval, research, fact verification.
    /// Owns the memory pipeline query interface.
    Scholar = 1,

    /// Content creation, code generation, artifact production.
    /// Transforms understanding into deliverables.
    Artisan = 2,

    /// Safety, privacy, إحسان enforcement, constitutional limits.
    /// Has VETO power — can halt any action.
    Guardian = 3,

    /// Learning, adaptation, personalization, user modeling.
    /// Feeds the memory synthesis pipeline.
    Mentor = 4,

    /// Communication style, tone matching, cultural awareness.
    /// Shapes HOW things are said, not WHAT is said.
    Diplomat = 5,

    /// Prediction, planning, proactive suggestions.
    /// Looks ahead — "you might also need..."
    Oracle = 6,
}

impl AgentRole {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Navigator,
            1 => Self::Scholar,
            2 => Self::Artisan,
            3 => Self::Guardian,
            4 => Self::Mentor,
            5 => Self::Diplomat,
            6 => Self::Oracle,
            _ => Self::Navigator, // Default fallback
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Navigator => "Navigator",
            Self::Scholar => "Scholar",
            Self::Artisan => "Artisan",
            Self::Guardian => "Guardian",
            Self::Mentor => "Mentor",
            Self::Diplomat => "Diplomat",
            Self::Oracle => "Oracle",
        }
    }

    /// What kind of memory atoms does this agent produce?
    pub fn primary_atom_kind(&self) -> AtomKind {
        match self {
            Self::Navigator => AtomKind::Pattern,
            Self::Scholar => AtomKind::Expertise,
            Self::Artisan => AtomKind::Expertise,
            Self::Guardian => AtomKind::Temporal, // Tracks safety patterns over time
            Self::Mentor => AtomKind::Goal,
            Self::Diplomat => AtomKind::Context,
            Self::Oracle => AtomKind::Pattern,
        }
    }

    /// Priority weight in consensus decisions (0.0-1.0)
    /// Guardian has highest weight — safety first
    pub fn consensus_weight(&self) -> f32 {
        match self {
            Self::Guardian => 1.0,  // Veto power
            Self::Navigator => 0.9, // Orchestrator authority
            Self::Scholar => 0.8,   // Knowledge authority
            Self::Mentor => 0.75,   // Personalization insight
            Self::Artisan => 0.7,   // Execution expertise
            Self::Diplomat => 0.65, // Style guidance
            Self::Oracle => 0.6,    // Predictive, lower certainty
        }
    }

    /// Does this role have veto authority?
    pub fn has_veto(&self) -> bool {
        matches!(self, Self::Guardian)
    }

    /// All 7 roles
    pub fn all() -> [AgentRole; 7] {
        [
            Self::Navigator,
            Self::Scholar,
            Self::Artisan,
            Self::Guardian,
            Self::Mentor,
            Self::Diplomat,
            Self::Oracle,
        ]
    }
}

// ============================================================
// MESSAGE — what enters the agent runtime
// ============================================================

pub const MESSAGE_CONTENT_SIZE: usize = 4096;

/// Fixed-size message content buffer
#[derive(Clone)]
pub struct MessageContent {
    data: [u8; MESSAGE_CONTENT_SIZE],
    len: u16,
}

impl MessageContent {
    pub fn new(text: &str) -> Self {
        let mut data = [0u8; MESSAGE_CONTENT_SIZE];
        let bytes = text.as_bytes();
        let len = bytes.len().min(MESSAGE_CONTENT_SIZE);
        data[..len].copy_from_slice(&bytes[..len]);
        Self {
            data,
            len: len as u16,
        }
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("[corrupted]")
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for MessageContent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let preview = if self.len() > 80 {
            &self.as_str()[..80]
        } else {
            self.as_str()
        };
        write!(f, "MessageContent({preview:?}...)")
    }
}

/// Unique message identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageId(pub u64);

impl MessageId {
    pub fn new(session_id: u32, sequence: u32) -> Self {
        Self(((session_id as u64) << 32) | sequence as u64)
    }

    pub fn session_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn sequence(&self) -> u32 {
        self.0 as u32
    }
}

/// Message direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageDirection {
    /// User → Agent
    Inbound = 0,
    /// Agent → User
    Outbound = 1,
    /// Agent → Agent (internal routing)
    Internal = 2,
}

/// A message flowing through the agent runtime
#[derive(Clone, Debug)]
pub struct Message {
    pub id: MessageId,
    pub direction: MessageDirection,
    pub content: MessageContent,
    pub timestamp: u64,
    pub source_agent: Option<AgentId>,
    pub target_agent: Option<AgentId>,
    pub ihsan_at_receipt: IhsanScore,
}

impl Message {
    /// Create a new inbound user message
    pub fn inbound(id: MessageId, content: &str, timestamp: u64, ihsan: IhsanScore) -> Self {
        Self {
            id,
            direction: MessageDirection::Inbound,
            content: MessageContent::new(content),
            timestamp,
            source_agent: None,
            target_agent: None,
            ihsan_at_receipt: ihsan,
        }
    }

    /// Create an internal agent-to-agent message
    pub fn internal(
        id: MessageId,
        content: &str,
        timestamp: u64,
        from: AgentId,
        to: AgentId,
        ihsan: IhsanScore,
    ) -> Self {
        Self {
            id,
            direction: MessageDirection::Internal,
            content: MessageContent::new(content),
            timestamp,
            source_agent: Some(from),
            target_agent: Some(to),
            ihsan_at_receipt: ihsan,
        }
    }
}

// ============================================================
// TASK — unit of work assigned to agents
// ============================================================

/// What kind of task is this?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskKind {
    /// Classify intent from user message
    ClassifyIntent = 0,
    /// Retrieve relevant context from memory
    RetrieveContext = 1,
    /// Generate response content
    GenerateResponse = 2,
    /// Check safety / إحسان compliance
    SafetyCheck = 3,
    /// Extract memory fragments from conversation
    ExtractMemory = 4,
    /// Adapt response style to user preferences
    AdaptStyle = 5,
    /// Generate proactive suggestions
    ProactiveSuggest = 6,
    /// Synthesize multi-agent results
    SynthesizeResults = 7,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TaskPriority {
    Background = 0,
    Normal = 1,
    High = 2,
    Critical = 3, // Safety checks, إحسان enforcement
}

/// Task state in the execution pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    Pending = 0,
    Assigned = 1,
    Running = 2,
    Completed = 3,
    Failed = 4,
    Vetoed = 5, // Halted by Guardian
}

/// Unique task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

impl TaskId {
    pub fn new(message_id: u32, sequence: u32) -> Self {
        Self(((message_id as u64) << 32) | sequence as u64)
    }
}

pub const TASK_OUTPUT_SIZE: usize = 2048;

/// Fixed-size task output buffer
#[derive(Clone)]
pub struct TaskOutput {
    data: [u8; TASK_OUTPUT_SIZE],
    len: u16,
}

impl TaskOutput {
    pub fn new(text: &str) -> Self {
        let mut data = [0u8; TASK_OUTPUT_SIZE];
        let bytes = text.as_bytes();
        let len = bytes.len().min(TASK_OUTPUT_SIZE);
        data[..len].copy_from_slice(&bytes[..len]);
        Self {
            data,
            len: len as u16,
        }
    }

    pub fn empty() -> Self {
        Self {
            data: [0u8; TASK_OUTPUT_SIZE],
            len: 0,
        }
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len as usize]).unwrap_or("[corrupted]")
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for TaskOutput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TaskOutput({}B)", self.len)
    }
}

/// A task assigned to an agent
#[derive(Clone, Debug)]
pub struct Task {
    pub id: TaskId,
    pub kind: TaskKind,
    pub priority: TaskPriority,
    pub state: TaskState,
    pub assigned_to: AgentRole,
    pub source_message: MessageId,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub output: TaskOutput,
    pub confidence: Confidence,
}

impl Task {
    pub fn new(
        id: TaskId,
        kind: TaskKind,
        priority: TaskPriority,
        assigned_to: AgentRole,
        source_message: MessageId,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            kind,
            priority,
            state: TaskState::Pending,
            assigned_to,
            source_message,
            created_at: timestamp,
            completed_at: None,
            output: TaskOutput::empty(),
            confidence: Confidence::new(0.0, 0),
        }
    }

    pub fn complete(&mut self, output: &str, confidence: Confidence, timestamp: u64) {
        self.state = TaskState::Completed;
        self.output = TaskOutput::new(output);
        self.confidence = confidence;
        self.completed_at = Some(timestamp);
    }

    pub fn fail(&mut self, reason: &str, timestamp: u64) {
        self.state = TaskState::Failed;
        self.output = TaskOutput::new(reason);
        self.completed_at = Some(timestamp);
    }

    pub fn veto(&mut self, reason: &str, timestamp: u64) {
        self.state = TaskState::Vetoed;
        self.output = TaskOutput::new(reason);
        self.completed_at = Some(timestamp);
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TaskState::Completed | TaskState::Failed | TaskState::Vetoed
        )
    }

    pub fn duration_us(&self) -> Option<u64> {
        self.completed_at
            .map(|end| end.saturating_sub(self.created_at))
    }
}

// ============================================================
// AGENT CONTEXT — assembled knowledge for response generation
// ============================================================

pub const MAX_CONTEXT_TRAITS: usize = 16;
pub const MAX_CONTEXT_INSIGHTS: usize = 8;

/// Key-value trait in context
#[derive(Clone, Debug)]
pub struct ContextTrait {
    pub key: [u8; 64],
    pub key_len: u8,
    pub value: [u8; 256],
    pub value_len: u16,
    pub confidence: Confidence,
}

impl ContextTrait {
    pub fn new(key: &str, value: &str, confidence: Confidence) -> Self {
        let mut k = [0u8; 64];
        let kb = key.as_bytes();
        let kl = kb.len().min(64);
        k[..kl].copy_from_slice(&kb[..kl]);

        let mut v = [0u8; 256];
        let vb = value.as_bytes();
        let vl = vb.len().min(256);
        v[..vl].copy_from_slice(&vb[..vl]);

        Self {
            key: k,
            key_len: kl as u8,
            value: v,
            value_len: vl as u16,
            confidence,
        }
    }

    pub fn key(&self) -> &str {
        core::str::from_utf8(&self.key[..self.key_len as usize]).unwrap_or("")
    }

    pub fn value(&self) -> &str {
        core::str::from_utf8(&self.value[..self.value_len as usize]).unwrap_or("")
    }
}

/// Assembled context for response generation
/// This is what gets passed to the LLM / response generator
pub struct AgentContext {
    pub message: MessageId,
    pub traits: Vec<ContextTrait>,
    pub insight_summaries: Vec<String>,
    pub knows_me_score: f32,
    pub session_history_depth: u16,
    pub current_ihsan: IhsanScore,
    pub assembled_at: u64,
    pub assembler: AgentRole,
}

impl AgentContext {
    pub fn empty(message: MessageId, timestamp: u64) -> Self {
        Self {
            message,
            traits: Vec::new(),
            insight_summaries: Vec::new(),
            knows_me_score: 0.0,
            session_history_depth: 0,
            current_ihsan: IhsanScore::from_raw(9900),
            assembled_at: timestamp,
            assembler: AgentRole::Scholar,
        }
    }

    pub fn add_trait(&mut self, key: &str, value: &str, confidence: Confidence) {
        if self.traits.len() < MAX_CONTEXT_TRAITS {
            self.traits.push(ContextTrait::new(key, value, confidence));
        }
    }

    pub fn add_insight(&mut self, summary: &str) {
        if self.insight_summaries.len() < MAX_CONTEXT_INSIGHTS {
            self.insight_summaries.push(summary.to_string());
        }
    }

    /// How rich is this context? (0.0 = empty, 1.0 = fully loaded)
    pub fn richness(&self) -> f32 {
        let trait_score = (self.traits.len() as f32 / MAX_CONTEXT_TRAITS as f32).min(1.0);
        let insight_score =
            (self.insight_summaries.len() as f32 / MAX_CONTEXT_INSIGHTS as f32).min(1.0);
        let knows_me = self.knows_me_score;
        trait_score * 0.4 + insight_score * 0.3 + knows_me * 0.3
    }
}

// ============================================================
// RESPONSE — what the agent runtime produces
// ============================================================

pub const RESPONSE_CONTENT_SIZE: usize = 8192;

/// Fixed-size response content
#[derive(Clone)]
pub struct ResponseContent {
    data: Vec<u8>,
    len: usize,
}

impl ResponseContent {
    pub fn new(text: &str) -> Self {
        let bytes = text.as_bytes();
        let len = bytes.len().min(RESPONSE_CONTENT_SIZE);
        Self {
            data: bytes[..len].to_vec(),
            len,
        }
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len]).unwrap_or("[corrupted]")
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::fmt::Debug for ResponseContent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ResponseContent({}B)", self.len)
    }
}

/// Agent response to a user message
#[derive(Clone, Debug)]
pub struct Response {
    pub message_id: MessageId,
    pub content: ResponseContent,
    pub confidence: Confidence,
    pub context_richness: f32,
    pub agents_consulted: u8,
    pub ihsan_at_generation: IhsanScore,
    pub generated_at: u64,
    pub generation_duration_us: u64,
    pub vetoed: bool,
    pub veto_reason: Option<String>,
}

impl Response {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        message_id: MessageId,
        content: &str,
        confidence: Confidence,
        context_richness: f32,
        agents: u8,
        ihsan: IhsanScore,
        timestamp: u64,
        duration_us: u64,
    ) -> Self {
        Self {
            message_id,
            content: ResponseContent::new(content),
            confidence,
            context_richness,
            agents_consulted: agents,
            ihsan_at_generation: ihsan,
            generated_at: timestamp,
            generation_duration_us: duration_us,
            vetoed: false,
            veto_reason: None,
        }
    }

    pub fn vetoed(message_id: MessageId, reason: &str, timestamp: u64) -> Self {
        Self {
            message_id,
            content: ResponseContent::new(""),
            confidence: Confidence::new(0.0, 0),
            context_richness: 0.0,
            agents_consulted: 1,
            ihsan_at_generation: IhsanScore::from_raw(9900),
            generated_at: timestamp,
            generation_duration_us: 0,
            vetoed: true,
            veto_reason: Some(reason.to_string()),
        }
    }
}

// ============================================================
// RUNTIME METRICS
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct RuntimeMetrics {
    pub messages_processed: u64,
    pub tasks_created: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_vetoed: u64,
    pub avg_response_time_us: u64,
    pub avg_context_richness: f32,
    pub avg_confidence: Confidence,
    pub current_ihsan: IhsanScore,
}

impl RuntimeMetrics {
    pub fn new() -> Self {
        Self {
            messages_processed: 0,
            tasks_created: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            tasks_vetoed: 0,
            avg_response_time_us: 0,
            avg_context_richness: 0.0,
            avg_confidence: Confidence::new(0.0, 0),
            current_ihsan: IhsanScore::from_raw(9900),
        }
    }

    pub fn task_success_rate(&self) -> f32 {
        if self.tasks_created == 0 {
            return 1.0;
        }
        self.tasks_completed as f32 / self.tasks_created as f32
    }

    pub fn veto_rate(&self) -> f32 {
        if self.tasks_created == 0 {
            return 0.0;
        }
        self.tasks_vetoed as f32 / self.tasks_created as f32
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_id_roundtrip() {
        let id = AgentId::new(AgentRole::Scholar, 0xBEEF);
        assert_eq!(id.role(), AgentRole::Scholar);
        assert_eq!(id.user_hash(), 0xBEEF);
    }

    #[test]
    fn all_roles_are_seven() {
        assert_eq!(AgentRole::all().len(), 7);
    }

    #[test]
    fn guardian_has_veto() {
        assert!(AgentRole::Guardian.has_veto());
        assert!(!AgentRole::Scholar.has_veto());
        assert!(!AgentRole::Artisan.has_veto());
    }

    #[test]
    fn guardian_has_highest_consensus_weight() {
        let max_weight = AgentRole::all()
            .iter()
            .map(|r| r.consensus_weight())
            .fold(0.0f32, f32::max);
        assert_eq!(AgentRole::Guardian.consensus_weight(), max_weight);
    }

    #[test]
    fn message_content_stores_and_retrieves() {
        let content = MessageContent::new("Hello, how can I help you today?");
        assert_eq!(content.as_str(), "Hello, how can I help you today?");
        assert!(!content.is_empty());
    }

    #[test]
    fn message_id_roundtrip() {
        let id = MessageId::new(42, 7);
        assert_eq!(id.session_id(), 42);
        assert_eq!(id.sequence(), 7);
    }

    #[test]
    fn inbound_message_creation() {
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Build me a Rust crate",
            1000,
            IhsanScore::from_raw(9900),
        );
        assert_eq!(msg.direction, MessageDirection::Inbound);
        assert!(msg.source_agent.is_none());
    }

    #[test]
    fn task_lifecycle() {
        let mut task = Task::new(
            TaskId::new(1, 1),
            TaskKind::ClassifyIntent,
            TaskPriority::High,
            AgentRole::Navigator,
            MessageId::new(1, 1),
            1000,
        );

        assert_eq!(task.state, TaskState::Pending);
        assert!(!task.is_terminal());

        task.complete("Intent: code_generation", Confidence::stated(0), 1050);
        assert_eq!(task.state, TaskState::Completed);
        assert!(task.is_terminal());
        assert_eq!(task.output.as_str(), "Intent: code_generation");
        assert_eq!(task.duration_us(), Some(50));
    }

    #[test]
    fn task_veto() {
        let mut task = Task::new(
            TaskId::new(1, 2),
            TaskKind::GenerateResponse,
            TaskPriority::Normal,
            AgentRole::Artisan,
            MessageId::new(1, 1),
            1000,
        );

        task.veto("Violates safety policy", 1010);
        assert_eq!(task.state, TaskState::Vetoed);
        assert!(task.is_terminal());
    }

    #[test]
    fn agent_context_richness() {
        let mut ctx = AgentContext::empty(MessageId::new(1, 1), 1000);
        assert_eq!(ctx.richness(), 0.0);

        ctx.add_trait("language", "Rust", Confidence::stated(0));
        ctx.add_trait("role", "architect", Confidence::inferred(0));
        ctx.add_insight("Expert Rust developer building distributed systems");
        ctx.knows_me_score = 0.7;

        assert!(ctx.richness() > 0.0);
    }

    #[test]
    fn response_creation() {
        let resp = Response::new(
            MessageId::new(1, 1),
            "Here's your Rust crate...",
            Confidence::stated(0),
            0.85,
            5,
            IhsanScore::from_raw(9900),
            2000,
            500,
        );

        assert!(!resp.vetoed);
        assert_eq!(resp.agents_consulted, 5);
        assert_eq!(resp.content.as_str(), "Here's your Rust crate...");
    }

    #[test]
    fn response_vetoed() {
        let resp = Response::vetoed(MessageId::new(1, 1), "Harmful content detected", 2000);

        assert!(resp.vetoed);
        assert!(resp.content.is_empty());
        assert_eq!(
            resp.veto_reason.as_deref(),
            Some("Harmful content detected")
        );
    }

    #[test]
    fn runtime_metrics_rates() {
        let mut metrics = RuntimeMetrics::new();
        metrics.tasks_created = 100;
        metrics.tasks_completed = 85;
        metrics.tasks_failed = 10;
        metrics.tasks_vetoed = 5;

        assert!((metrics.task_success_rate() - 0.85).abs() < 0.001);
        assert!((metrics.veto_rate() - 0.05).abs() < 0.001);
    }

    #[test]
    fn each_role_has_unique_name() {
        let names: Vec<&str> = AgentRole::all().iter().map(|r| r.name()).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Roles must have unique names");
                }
            }
        }
    }
}
