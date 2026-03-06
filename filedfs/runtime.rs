// bizra-agent/src/runtime.rs
// ============================================================
// Agent Runtime — the unified being
// ============================================================
// Owns all organs:
//   - MemoryPipeline (brain — knowledge, synthesis, recall)
//   - AgentRoster (team — 7 PAT specialists per user)
//   - TaskOrchestrator (coordinator — routes, plans, executes)
//   - ContextAssembler (bridge — memory → agent context)
//
// This is the single entry point. External code calls:
//   runtime.receive(message) → Response
//
// That one call triggers:
//   1. Session tracking
//   2. Memory context retrieval
//   3. Intent classification
//   4. Guardian safety check
//   5. Multi-agent task execution
//   6. Memory fragment extraction
//   7. Auto-synthesis when threshold met
//   8. Response assembly
//
// "My AI knows me" lives here.
// ============================================================

use bizra_hooks::IhsanScore;
use bizra_memory::MemoryPipeline;
use bizra_memory::pipeline::PipelineConfig;
use bizra_memory::types::{MemoryFragment, FragmentId, FragmentKind, Confidence};

use crate::types::*;
use crate::roster::AgentRoster;
use crate::orchestrator::{TaskOrchestrator, OrchestratorConfig};

// ============================================================
// RUNTIME CONFIG
// ============================================================

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Pipeline configuration for memory subsystem
    pub pipeline_config: PipelineConfig,
    /// Orchestrator configuration
    pub orchestrator_config: OrchestratorConfig,
    /// User hash for agent identity generation
    pub user_hash: u32,
    /// إحسان floor — system won't operate below this
    pub ihsan_floor: IhsanScore,
    /// Maximum conversations before forced synthesis
    pub max_conversations_before_synthesis: u32,
    /// Auto-extract memory fragments from messages
    pub auto_extract_memory: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            pipeline_config: PipelineConfig::default(),
            orchestrator_config: OrchestratorConfig::default(),
            user_hash: 0,
            ihsan_floor: IhsanScore::new(9500),
            max_conversations_before_synthesis: 5,
            auto_extract_memory: true,
        }
    }
}

impl RuntimeConfig {
    pub fn for_user(user_hash: u32) -> Self {
        Self {
            user_hash,
            ..Default::default()
        }
    }
}

// ============================================================
// RUNTIME STATE
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Not yet initialized
    Uninitialized,
    /// Ready to process messages
    Ready,
    /// Currently processing a message
    Processing,
    /// System degraded due to low إحسان
    Degraded,
    /// Shut down
    Stopped,
}

// ============================================================
// CONVERSATION SESSION
// ============================================================

#[derive(Debug, Clone)]
pub struct ConversationSession {
    pub session_id: u64,
    pub started_at: u64,
    pub message_count: u32,
    pub fragments_extracted: u32,
    pub active: bool,
}

impl ConversationSession {
    fn new(session_id: u64, timestamp: u64) -> Self {
        Self {
            session_id,
            started_at: timestamp,
            message_count: 0,
            fragments_extracted: 0,
            active: true,
        }
    }
}

// ============================================================
// RUNTIME HEALTH — full system snapshot
// ============================================================

#[derive(Debug, Clone)]
pub struct RuntimeHealth {
    pub state: RuntimeState,
    pub current_ihsan: IhsanScore,
    // Memory subsystem
    pub fragments_stored: usize,
    pub insights_stored: usize,
    pub profile_traits: usize,
    pub knows_me_score: f32,
    pub synthesis_rounds: u32,
    // Agent subsystem
    pub agents_registered: usize,
    pub agents_active: usize,
    pub total_vetoes: u64,
    // Orchestration
    pub messages_processed: u64,
    pub total_tasks: u64,
    // Session
    pub active_session: bool,
    pub total_conversations: u32,
    pub conversation_messages: u32,
}

// ============================================================
// AGENT RUNTIME — the unified being
// ============================================================

pub struct AgentRuntime {
    /// Memory: what we know about the user
    pipeline: MemoryPipeline,
    /// Team: the 7 PAT specialists
    roster: AgentRoster,
    /// Coordinator: routes tasks, enforces safety
    orchestrator: TaskOrchestrator,
    /// Configuration
    config: RuntimeConfig,
    /// Current state
    state: RuntimeState,
    /// Current إحسان score
    current_ihsan: IhsanScore,
    /// Active conversation session
    current_session: Option<ConversationSession>,
    /// Total conversations processed
    total_conversations: u32,
    /// Fragment sequence counter (monotonic within runtime)
    fragment_seq: u32,
    /// Metrics
    metrics: RuntimeMetrics,
}

impl AgentRuntime {
    // ================================================================
    // CONSTRUCTION
    // ================================================================

    pub fn new() -> Self {
        Self::with_config(RuntimeConfig::default())
    }

    pub fn for_user(user_hash: u32) -> Self {
        Self::with_config(RuntimeConfig::for_user(user_hash))
    }

    pub fn with_config(config: RuntimeConfig) -> Self {
        let pipeline = MemoryPipeline::with_config(config.pipeline_config);
        let orchestrator = TaskOrchestrator::with_config(config.orchestrator_config);
        // AgentRoster auto-creates all 7 PAT agents at construction
        let roster = AgentRoster::new(config.user_hash, 0);

        Self {
            pipeline,
            roster,
            orchestrator,
            config,
            state: RuntimeState::Ready,
            current_ihsan: IhsanScore::new(9900),
            current_session: None,
            total_conversations: 0,
            fragment_seq: 0,
            metrics: RuntimeMetrics::default(),
        }
    }

    // ================================================================
    // CONVERSATION LIFECYCLE
    // ================================================================

    /// Start a new conversation
    /// Returns session_id
    pub fn start_conversation(&mut self, timestamp: u64) -> u64 {
        // End any existing session
        if let Some(ref session) = self.current_session {
            if session.active {
                let sid = session.session_id;
                self.end_conversation(timestamp);
                // If we just ended one, start fresh
                let _ = sid;
            }
        }

        self.total_conversations += 1;
        let session_id = ((self.config.user_hash as u64) << 32)
            | self.total_conversations as u64;

        // Register with memory pipeline
        self.pipeline.start_session(session_id, timestamp);

        self.current_session = Some(ConversationSession::new(session_id, timestamp));
        session_id
    }

    /// End current conversation
    /// Triggers synthesis if enough fragments accumulated
    pub fn end_conversation(&mut self, timestamp: u64) -> Option<usize> {
        let session = self.current_session.take()?;

        if session.active {
            self.pipeline.end_session(session.session_id, timestamp)
        } else {
            None
        }
    }

    /// Check if there's an active conversation
    pub fn has_active_conversation(&self) -> bool {
        self.current_session.as_ref().map_or(false, |s| s.active)
    }

    // ================================================================
    // MESSAGE PROCESSING — the main entry point
    // ================================================================

    /// Process a user message through the full agent pipeline
    /// This is THE entry point. One call, full intelligence.
    ///
    /// Flow:
    /// 1. Validate state + إحسان
    /// 2. Auto-start conversation if needed
    /// 3. Route through orchestrator (intent → plan → execute)
    /// 4. Extract memory fragments from message
    /// 5. Update metrics
    /// 6. Return response
    pub fn receive(
        &mut self,
        message: Message,
        timestamp: u64,
    ) -> RuntimeResponse {
        // metrics tracked below

        // State check
        if self.state == RuntimeState::Stopped {
            return RuntimeResponse::error(
                message.id,
                "Runtime is stopped",
                timestamp,
            );
        }

        // إحسان gate
        if self.current_ihsan.raw() < self.config.ihsan_floor.raw() {
            self.state = RuntimeState::Degraded;
            return RuntimeResponse::degraded(
                message.id,
                self.current_ihsan,
                timestamp,
            );
        }

        // Auto-start conversation if needed
        if !self.has_active_conversation() {
            self.start_conversation(timestamp);
        }

        self.state = RuntimeState::Processing;

        // Track message in session
        if let Some(ref mut session) = self.current_session {
            session.message_count += 1;
        }

        // === ORCHESTRATE ===
        let result = self.orchestrator.process_message(
            &message,
            &mut self.roster,
            &mut self.pipeline,
            self.current_ihsan,
        );

        // === EXTRACT MEMORY ===
        let fragments_extracted = if self.config.auto_extract_memory {
            self.extract_memory_from_message(&message, timestamp)
        } else {
            0
        };

        if let Some(ref mut session) = self.current_session {
            session.fragments_extracted += fragments_extracted as u32;
        }

        // Update metrics
        self.metrics.messages_processed += 1;
        self.metrics.tasks_created += result.plan.task_count() as u64;
        // fragments tracked in session
        if !result.guardian_approved {
            self.metrics.tasks_vetoed += 1;
        }

        self.state = RuntimeState::Ready;

        RuntimeResponse {
            response: result.response,
            agents_consulted: result.agents_consulted,
            fragments_extracted,
            guardian_approved: result.guardian_approved,
            knows_me_score: self.pipeline.knows_me_score(),
            session_messages: self.current_session.as_ref()
                .map_or(0, |s| s.message_count),
        }
    }

    // ================================================================
    // MEMORY EXTRACTION — learn from every message
    // ================================================================

    /// Extract memory fragments from a message
    /// MVP: keyword-based extraction heuristics
    /// Production: LLM-powered structured extraction
    fn extract_memory_from_message(
        &mut self,
        message: &Message,
        timestamp: u64,
    ) -> usize {
        let content = message.content.as_str();
        let mut extracted = 0;

        // Only extract from user messages (inbound)
        if message.direction != MessageDirection::Inbound {
            return 0;
        }

        // Minimum content threshold
        if content.len() < 10 {
            return 0;
        }

        // Preference detection heuristics
        let preference_markers = [
            "i prefer", "i like", "i want", "i love", "i hate",
            "i need", "i enjoy", "my favorite", "i always",
        ];

        for marker in &preference_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(
                    FragmentKind::Preference,
                    extract,
                    Confidence::new(7500), // Heuristic = moderate confidence
                    timestamp,
                ) {
                    extracted += 1;
                }
                break; // One preference per message MVP
            }
        }

        // Goal detection
        let goal_markers = [
            "i'm working on", "my goal", "i'm trying to",
            "i want to achieve", "my project", "i'm building",
        ];

        for marker in &goal_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(
                    FragmentKind::Goal,
                    extract,
                    Confidence::new(7000),
                    timestamp,
                ) {
                    extracted += 1;
                }
                break;
            }
        }

        // Expertise detection
        let expertise_markers = [
            "i'm an expert in", "i work with", "i specialize in",
            "my expertise is", "i've been doing", "i know",
        ];

        for marker in &expertise_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(
                    FragmentKind::Expertise,
                    extract,
                    Confidence::new(7000),
                    timestamp,
                ) {
                    extracted += 1;
                }
                break;
            }
        }

        // Style detection (based on message characteristics)
        if content.len() > 500 {
            self.ingest_fragment(
                FragmentKind::Style,
                "User writes detailed, verbose messages",
                Confidence::new(6500),
                timestamp,
            );
        } else if content.len() < 50 && !content.contains(' ') {
            self.ingest_fragment(
                FragmentKind::Style,
                "User sends terse, command-like messages",
                Confidence::new(6500),
                timestamp,
            );
        }

        // Fact detection (statements about self)
        let fact_markers = [
            "i am", "i'm a", "i live in", "my name is",
            "i work at", "i'm from",
        ];

        for marker in &fact_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(
                    FragmentKind::Fact,
                    extract,
                    Confidence::new(8000), // Self-statements = higher confidence
                    timestamp,
                ) {
                    extracted += 1;
                }
                break;
            }
        }

        extracted
    }

    /// Ingest a single extracted fragment into the memory pipeline
    fn ingest_fragment(
        &mut self,
        kind: FragmentKind,
        content: &str,
        confidence: Confidence,
        timestamp: u64,
    ) -> bool {
        self.fragment_seq += 1;

        let fragment = MemoryFragment::new(
            FragmentId::new(
                self.current_session.as_ref().map_or(0, |s| s.session_id as u32),
                self.fragment_seq,
            ),
            kind,
            content,
            confidence,
            bizra_hooks::ComponentId::new("bizra-agent", "0.1"),
            timestamp,
            self.current_ihsan,
        );

        self.pipeline.ingest(fragment, timestamp).is_ok()
    }

    // ================================================================
    // MANUAL MEMORY OPERATIONS
    // ================================================================

    /// Manually teach the runtime something about the user
    pub fn teach(
        &mut self,
        kind: FragmentKind,
        content: &str,
        confidence: Confidence,
        timestamp: u64,
    ) -> bool {
        self.ingest_fragment(kind, content, confidence, timestamp)
    }

    /// Force a synthesis round
    pub fn synthesize(&mut self, timestamp: u64) -> usize {
        self.pipeline.force_synthesis(timestamp)
    }

    /// Query what the runtime knows about a specific trait
    pub fn query_trait(&mut self, key: &str) -> Option<(&str, Confidence)> {
        self.pipeline.query_trait(key)
    }

    /// Get the full user profile as known traits
    pub fn query_profile(&mut self) -> Vec<(&str, &str, Confidence)> {
        self.pipeline.query_profile()
    }

    /// The quantified "my AI knows me" metric
    pub fn knows_me_score(&self) -> f32 {
        self.pipeline.knows_me_score()
    }

    // ================================================================
    // IHSAN MANAGEMENT
    // ================================================================

    /// Update system إحسان score
    pub fn update_ihsan(&mut self, score: IhsanScore) {
        self.current_ihsan = score;
        self.pipeline.update_ihsan(score);

        if score.raw() < self.config.ihsan_floor.raw() {
            self.state = RuntimeState::Degraded;
        } else if self.state == RuntimeState::Degraded {
            self.state = RuntimeState::Ready;
        }
    }

    pub fn current_ihsan(&self) -> IhsanScore {
        self.current_ihsan
    }

    // ================================================================
    // LIFECYCLE
    // ================================================================

    /// Gracefully shut down the runtime
    pub fn shutdown(&mut self, timestamp: u64) {
        // End any active conversation
        self.end_conversation(timestamp);

        // Suspend all agents
        for role in AgentRole::all() {
            self.roster.suspend(role);
        }

        self.state = RuntimeState::Stopped;
    }

    pub fn state(&self) -> RuntimeState {
        self.state
    }

    // ================================================================
    // HEALTH & OBSERVABILITY
    // ================================================================

    /// Complete system health snapshot
    pub fn health(&self) -> RuntimeHealth {
        let pipeline_health = self.pipeline.health();
        let roster_snapshot = self.roster.snapshot();

        RuntimeHealth {
            state: self.state,
            current_ihsan: self.current_ihsan,
            // Memory
            fragments_stored: pipeline_health.fragments_stored,
            insights_stored: pipeline_health.insights_stored,
            profile_traits: pipeline_health.profile_traits,
            knows_me_score: self.pipeline.knows_me_score(),
            synthesis_rounds: pipeline_health.synthesis_rounds,
            // Agents — PAT always has 7 registered
            agents_registered: 7,
            agents_active: roster_snapshot.agents_available as usize,
            total_vetoes: self.orchestrator.total_vetoes(),
            // Orchestration
            messages_processed: self.metrics.messages_processed,
            total_tasks: self.metrics.tasks_created,
            // Session
            active_session: self.has_active_conversation(),
            total_conversations: self.total_conversations,
            conversation_messages: self.current_session.as_ref()
                .map_or(0, |s| s.message_count),
        }
    }

    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }

    /// Access the memory pipeline directly (for advanced queries)
    pub fn memory(&mut self) -> &mut MemoryPipeline {
        &mut self.pipeline
    }

    /// Access the agent roster directly
    pub fn roster(&self) -> &AgentRoster {
        &self.roster
    }

    /// Access roster mutably
    pub fn roster_mut(&mut self) -> &mut AgentRoster {
        &mut self.roster
    }
}

impl Default for AgentRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// RUNTIME RESPONSE — enriched output from receive()
// ============================================================

#[derive(Debug)]
pub struct RuntimeResponse {
    /// The actual response content
    pub response: Response,
    /// How many agents were consulted
    pub agents_consulted: u8,
    /// How many memory fragments were extracted from the input
    pub fragments_extracted: usize,
    /// Whether Guardian approved the action
    pub guardian_approved: bool,
    /// Current "knows me" score
    pub knows_me_score: f32,
    /// Messages in current session
    pub session_messages: u32,
}

impl RuntimeResponse {
    fn error(msg_id: MessageId, reason: &str, timestamp: u64) -> Self {
        Self {
            response: Response::vetoed(msg_id, reason, timestamp),
            agents_consulted: 0,
            fragments_extracted: 0,
            guardian_approved: false,
            knows_me_score: 0.0,
            session_messages: 0,
        }
    }

    fn degraded(msg_id: MessageId, _ihsan: IhsanScore, timestamp: u64) -> Self {
        Self {
            response: Response::vetoed(msg_id, "System degraded: ihsan below threshold", timestamp),
            agents_consulted: 0,
            fragments_extracted: 0,
            guardian_approved: false,
            knows_me_score: 0.0,
            session_messages: 0,
        }
    }

    /// Was the response successful (not error/degraded/vetoed)?
    pub fn is_ok(&self) -> bool {
        self.guardian_approved
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user_message(content: &str, timestamp: u64) -> Message {
        Message::inbound(
            MessageId::new(timestamp as u32, 1),
            content,
            timestamp,
            IhsanScore::new(9900),
        )
    }

    #[test]
    fn runtime_creation_registers_7_agents() {
        let runtime = AgentRuntime::for_user(42);
        assert_eq!(runtime.state(), RuntimeState::Ready);
        assert_eq!(runtime.roster().available_count(), 7);
    }

    #[test]
    fn runtime_conversation_lifecycle() {
        let mut runtime = AgentRuntime::for_user(42);

        let session_id = runtime.start_conversation(1000);
        assert!(session_id > 0);
        assert!(runtime.has_active_conversation());

        let insights = runtime.end_conversation(2000);
        assert!(!runtime.has_active_conversation());
        // No synthesis yet (not enough fragments)
        assert!(insights.is_none() || insights == Some(0));
    }

    #[test]
    fn runtime_receive_processes_message() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.start_conversation(1000);

        let msg = make_user_message("Hello, I prefer dark mode interfaces", 1001);
        let response = runtime.receive(msg, 1001);

        assert!(response.agents_consulted >= 1);
        assert!(response.session_messages == 1);
        // Should have extracted a preference
        assert!(response.fragments_extracted >= 1);
    }

    #[test]
    fn runtime_auto_starts_conversation() {
        let mut runtime = AgentRuntime::for_user(42);
        assert!(!runtime.has_active_conversation());

        // Sending a message without starting conversation should auto-start
        let msg = make_user_message("Hi there", 1000);
        let _response = runtime.receive(msg, 1000);

        assert!(runtime.has_active_conversation());
        assert_eq!(runtime.total_conversations, 1);
    }

    #[test]
    fn runtime_degraded_when_low_ihsan() {
        let mut runtime = AgentRuntime::for_user(42);

        runtime.update_ihsan(IhsanScore::new(9000));
        assert_eq!(runtime.state(), RuntimeState::Degraded);

        let msg = make_user_message("This should be rejected", 1000);
        let response = runtime.receive(msg, 1000);
        assert!(!response.is_ok());

        // Recovery
        runtime.update_ihsan(IhsanScore::new(9900));
        assert_eq!(runtime.state(), RuntimeState::Ready);
    }

    #[test]
    fn runtime_teach_and_query() {
        let mut runtime = AgentRuntime::for_user(42);

        runtime.teach(
            FragmentKind::Fact,
            "User lives in Dubai",
            Confidence::new(9000),
            1000,
        );

        runtime.teach(
            FragmentKind::Preference,
            "User prefers Rust over Python",
            Confidence::new(8500),
            1001,
        );

        // Synthesize to produce profile traits
        runtime.synthesize(2000);

        // Knowledge depth should be positive
        let health = runtime.health();
        assert!(health.fragments_stored >= 2);
    }

    #[test]
    fn runtime_knows_me_score_increases() {
        let mut runtime = AgentRuntime::for_user(42);

        let score_empty = runtime.knows_me_score();
        assert_eq!(score_empty, 0.0);

        // Teach it things
        for i in 0..10 {
            runtime.teach(
                FragmentKind::Preference,
                &format!("Preference number {}", i),
                Confidence::new(8000),
                1000 + i as u64,
            );
        }

        runtime.synthesize(2000);
        let score_after = runtime.knows_me_score();
        assert!(score_after > score_empty);
    }

    #[test]
    fn runtime_shutdown_cleans_up() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.start_conversation(1000);
        runtime.receive(make_user_message("Hello", 1001), 1001);

        runtime.shutdown(2000);
        assert_eq!(runtime.state(), RuntimeState::Stopped);
        assert!(!runtime.has_active_conversation());
    }

    #[test]
    fn runtime_extracts_facts_from_messages() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.start_conversation(1000);

        let msg = make_user_message("I am a software architect working on distributed systems", 1001);
        let response = runtime.receive(msg, 1001);

        // Should extract "i am a software architect..."
        assert!(response.fragments_extracted >= 1);
        assert!(runtime.health().fragments_stored >= 1);
    }

    #[test]
    fn runtime_extracts_goals_from_messages() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.start_conversation(1000);

        let msg = make_user_message("I'm working on a distributed AI platform called BIZRA", 1001);
        let response = runtime.receive(msg, 1001);

        assert!(response.fragments_extracted >= 1);
    }

    #[test]
    fn runtime_health_snapshot() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.start_conversation(1000);
        runtime.receive(make_user_message("I prefer dark themes", 1001), 1001);

        let health = runtime.health();
        assert_eq!(health.state, RuntimeState::Ready);
        assert_eq!(health.agents_registered, 7);
        assert!(health.active_session);
        assert!(health.messages_processed >= 1);
        assert!(health.current_ihsan.raw() >= 9900);
    }

    #[test]
    fn runtime_multiple_conversations() {
        let mut runtime = AgentRuntime::for_user(42);

        // Conversation 1
        runtime.start_conversation(1000);
        runtime.receive(make_user_message("I like Rust", 1001), 1001);
        runtime.end_conversation(2000);

        // Conversation 2
        runtime.start_conversation(3000);
        runtime.receive(make_user_message("I prefer functional programming", 3001), 3001);
        runtime.end_conversation(4000);

        assert_eq!(runtime.total_conversations, 2);
        // Knowledge should accumulate across conversations
        assert!(runtime.health().fragments_stored >= 2);
    }

    #[test]
    fn runtime_guardian_veto_tracked() {
        let runtime = AgentRuntime::for_user(42);
        // Initial state — no vetoes
        assert_eq!(runtime.health().total_vetoes, 0);
    }
}
