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
use bizra_memory::pipeline::PipelineConfig;
use bizra_memory::types::{Confidence, FragmentKind};
use bizra_memory::AtomKind;
use bizra_memory::MemoryPipeline;

use crate::context::IntentClassifier;
use crate::decision_registry::{CognitiveMode, DecisionArtifact, DecisionRegistry};
use crate::hash_namespace::{
    compute_action_hash, compute_trigger_hash, parse_hex_32, ActionHash, TriggerHash,
};
use crate::orchestrator::{OrchestratorConfig, TaskOrchestrator};
use crate::persistence::ReflexStore;
use crate::reflex_cache::{QuarantineReason, ReflexCache, ReflexMode, ReflexRule, ReflexStats};
use crate::reflex_compiler::{
    snr_score, CompileReasonCode, CompileSample, CompilerConfig, ReflexCompiler,
};
use crate::roster::AgentRoster;
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionMode {
    Disabled,
    Shadow,
    Active,
}

impl ActionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Shadow => "shadow",
            Self::Active => "active",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardianCheckResult {
    pub allowed: bool,
    pub reason: String,
}

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
    /// Reflex routing mode: disabled -> shadow -> active
    pub reflex_mode: ReflexMode,
    /// Required trust anchor for active reflex routing (64 hex chars)
    pub policy_hash_hex: String,
    /// Compiler gate: minimum successful chains before compile
    pub min_success_chains: usize,
    /// Compiler gate: minimum ihsan at compile time
    pub min_compile_ihsan: f32,
    /// Compiler gate: minimum SNR at compile time
    pub min_compile_snr: f32,
    /// Compiler gate: maximum accepted path variance
    pub max_path_variance: f32,
    /// Revalidation timer in seconds (default 7 days)
    pub revalidate_after_seconds: u64,
    /// Revalidation trigger by rule uses
    pub revalidate_after_uses: u64,
    /// Immediately quarantine on guardian veto/revalidation failure
    pub immediate_quarantine: bool,
    /// Action execution mode for explicit action protocol
    pub action_mode: ActionMode,
    /// Path to the reflex persistence store directory (empty = no persistence)
    pub reflex_store_path: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            pipeline_config: PipelineConfig::default(),
            orchestrator_config: OrchestratorConfig::default(),
            user_hash: 0,
            ihsan_floor: IhsanScore::from_raw(9500),
            max_conversations_before_synthesis: 5,
            auto_extract_memory: true,
            reflex_mode: ReflexMode::Disabled,
            policy_hash_hex: String::new(),
            min_success_chains: 3,
            min_compile_ihsan: 0.95,
            min_compile_snr: 0.90,
            max_path_variance: 0.10,
            revalidate_after_seconds: 604_800,
            revalidate_after_uses: 200,
            immediate_quarantine: true,
            action_mode: ActionMode::Disabled,
            reflex_store_path: String::new(),
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
    // Reflex/compiler subsystem
    pub reflex_mode: ReflexMode,
    pub reflex_rules: usize,
    pub reflex_hits: u64,
    pub reflex_misses: u64,
    pub decision_artifacts: usize,
    // Action execution telemetry (Node action layer)
    pub actions_planned: u64,
    pub actions_executed: u64,
    pub actions_failed: u64,
    pub guardian_action_vetoes: u64,
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
    /// System-1 compiled reflex rules
    reflex_cache: ReflexCache,
    /// System-2 -> System-1 compiler candidate tracker
    reflex_compiler: ReflexCompiler,
    /// Glass-box append-only decision artifact registry
    decision_registry: DecisionRegistry,
    /// Sovereign file-based reflex persistence store
    reflex_store: Option<ReflexStore>,
    /// Parsed policy hash (required for active reflex mode)
    policy_hash: Option<[u8; 32]>,
    /// Action-layer counters (updated by node protocol handlers)
    actions_planned: u64,
    actions_executed: u64,
    actions_failed: u64,
    guardian_action_vetoes: u64,
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
        let policy_hash = if config.policy_hash_hex.is_empty() {
            None
        } else {
            parse_hex_32(config.policy_hash_hex.as_str())
        };

        // Capture ihsan_floor before config fields are moved
        let initial_ihsan = config.ihsan_floor;

        let pipeline = MemoryPipeline::with_config(config.pipeline_config);
        let orchestrator = TaskOrchestrator::with_config(config.orchestrator_config);
        // AgentRoster auto-creates all 7 PAT agents at construction
        let roster = AgentRoster::new(config.user_hash, 0);

        // ── Reflex persistence: restore from disk or bootstrap ──
        let mut reflex_cache = ReflexCache::default();
        let reflex_store = if !config.reflex_store_path.is_empty() {
            match ReflexStore::open(&config.reflex_store_path) {
                Ok(store) => {
                    match store.restore_all() {
                        Ok(rules) if !rules.is_empty() => {
                            reflex_cache.replace_rules(rules);
                        }
                        _ => {
                            // No persisted rules — load bootstrap cold-start reflexes
                            reflex_cache.load_bootstrap_rules();
                        }
                    }
                    Some(store)
                }
                Err(_) => {
                    // Store unavailable — degrade gracefully with bootstrap
                    reflex_cache.load_bootstrap_rules();
                    None
                }
            }
        } else {
            // No path configured — in-memory only with bootstrap
            reflex_cache.load_bootstrap_rules();
            None
        };

        Self {
            pipeline,
            roster,
            orchestrator,
            config,
            state: RuntimeState::Ready,
            current_ihsan: initial_ihsan,
            current_session: None,
            total_conversations: 0,
            fragment_seq: 0,
            metrics: RuntimeMetrics::default(),
            reflex_cache,
            reflex_compiler: ReflexCompiler::default(),
            decision_registry: DecisionRegistry::default(),
            reflex_store,
            policy_hash,
            actions_planned: 0,
            actions_executed: 0,
            actions_failed: 0,
            guardian_action_vetoes: 0,
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
        let session_id = ((self.config.user_hash as u64) << 32) | self.total_conversations as u64;

        // Note: omega MemoryPipeline has no start_session; session tracking
        // is handled at the agent runtime level.

        self.current_session = Some(ConversationSession::new(session_id, timestamp));
        session_id
    }

    /// End current conversation
    /// Triggers synthesis if enough fragments accumulated
    pub fn end_conversation(&mut self, timestamp: u64) -> Option<usize> {
        let session = self.current_session.take()?;

        if session.active {
            // Run extraction and forced synthesis for accumulated fragments.
            // We use force_synthesize (not maybe_synthesize) because the end
            // of a conversation is a natural boundary — we should process all
            // pending atoms regardless of batch threshold.
            self.pipeline.extract(timestamp);
            let result = self.pipeline.force_synthesize(timestamp);
            Some(result.insights_produced as usize)
        } else {
            None
        }
    }

    /// Check if there's an active conversation
    pub fn has_active_conversation(&self) -> bool {
        self.current_session.as_ref().is_some_and(|s| s.active)
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
    pub fn receive(&mut self, message: Message, timestamp: u64) -> RuntimeResponse {
        // metrics tracked below

        // State check
        if self.state == RuntimeState::Stopped {
            return RuntimeResponse::error(message.id, "Runtime is stopped", timestamp);
        }

        // إحسان gate
        if self.current_ihsan.raw() < self.config.ihsan_floor.raw() {
            self.state = RuntimeState::Degraded;
            return RuntimeResponse::degraded(message.id, self.current_ihsan, timestamp);
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

        let (intent, _) = IntentClassifier::classify(message.content.as_str());
        let trigger_traits = self.select_trigger_traits(timestamp);
        let policy_for_trigger = self.policy_hash.unwrap_or([0u8; 32]);
        let trigger_hash = compute_trigger_hash(
            format!("{intent:?}").as_str(),
            trigger_traits.as_slice(),
            &policy_for_trigger,
        );
        let effective_mode = self.effective_reflex_mode();

        let mut reflex_hit = false;
        let mut decision_mode = CognitiveMode::System2;
        let mut selected_rule: Option<ReflexRule> = None;

        if let Some(rule) =
            self.reflex_cache
                .get_active(effective_mode, &trigger_hash, self.policy_hash, timestamp)
        {
            reflex_hit = true;
            decision_mode = CognitiveMode::System1;
            selected_rule = Some(rule);
        }

        // Guardian check remains mandatory even on reflex hit.
        let mut forced_revalidation = false;
        if reflex_hit {
            let guardian_ok =
                self.orchestrator
                    .guardian_check(&message, &mut self.roster, self.current_ihsan);
            if !guardian_ok {
                if self.config.immediate_quarantine {
                    let _ = self
                        .reflex_cache
                        .quarantine(trigger_hash, QuarantineReason::GuardianVeto);
                }
                reflex_hit = false;
                decision_mode = CognitiveMode::System2;
            } else {
                forced_revalidation = self.reflex_cache.needs_revalidation(
                    &trigger_hash,
                    timestamp,
                    self.config.revalidate_after_seconds,
                    self.config.revalidate_after_uses,
                );
            }
        }

        // === ORCHESTRATE ===
        // System2 always runs today; in active mode we mark System1 hit when
        // a compiled rule is selected and not vetoed/quarantined.
        let result = self.orchestrator.process_message(
            &message,
            &mut self.roster,
            &mut self.pipeline,
            self.current_ihsan,
        );

        if let Some(rule) = selected_rule {
            if decision_mode == CognitiveMode::System1 && forced_revalidation {
                let matched = result.guardian_approved
                    && result.chosen_route == rule.action_template.route_signature;
                self.reflex_cache
                    .mark_revalidated(&trigger_hash, timestamp, matched);
                if !matched {
                    decision_mode = CognitiveMode::System2;
                    reflex_hit = false;
                    if self.config.immediate_quarantine {
                        let _ = self
                            .reflex_cache
                            .quarantine(trigger_hash, QuarantineReason::RevalidationFailed);
                    }
                }
            }
        }

        if !result.guardian_approved {
            decision_mode = CognitiveMode::System2;
            reflex_hit = false;
            if self.config.immediate_quarantine {
                let _ = self
                    .reflex_cache
                    .quarantine(trigger_hash, QuarantineReason::GuardianVeto);
            }
        }

        // Feed compiler only from System-2 successful chains.
        if decision_mode == CognitiveMode::System2 && result.guardian_approved {
            let compile_sample = CompileSample {
                route_signature: result.chosen_route.clone(),
                path_signature: result.micro_path.join(">"),
                response_confidence: result.response.confidence.base,
                context_richness: result.response.context_richness,
                guardian_approved: result.guardian_approved,
                ihsan_at_decision: self.current_ihsan.as_f64() as f32,
                timestamp,
            };
            self.reflex_compiler
                .record_success(trigger_hash, compile_sample);

            if effective_mode != ReflexMode::Disabled {
                let compiler_cfg = CompilerConfig {
                    min_success_chains: self.config.min_success_chains,
                    min_compile_ihsan: self.config.min_compile_ihsan,
                    min_compile_snr: self.config.min_compile_snr,
                    max_path_variance: self.config.max_path_variance,
                };
                match self
                    .reflex_compiler
                    .evaluate(trigger_hash, compiler_cfg, policy_for_trigger)
                {
                    Ok(rule) => {
                        // Persist to disk before inserting into cache
                        if let Some(ref store) = self.reflex_store {
                            let _ = store.save_rule(&rule);
                        }
                        self.reflex_cache.insert_compiled(effective_mode, rule);
                    }
                    Err(
                        CompileReasonCode::InsufficientSamples
                        | CompileReasonCode::LowIhsan
                        | CompileReasonCode::LowSnr
                        | CompileReasonCode::PathVarianceHigh,
                    ) => {}
                }
            }
        }

        let action_hash =
            compute_action_hash(&trigger_hash, result.chosen_route.as_str(), timestamp);
        let artifact = DecisionArtifact {
            action_hash,
            trigger_hash,
            decision_mode,
            mission_phase: result.mission_phase,
            micro_path: result.micro_path.clone(),
            chosen_route: result.chosen_route.clone(),
            rejected_alternatives: result.rejected_alternatives.clone(),
            guardian_verdict: result.guardian_approved,
            ihsan_at_decision: self.current_ihsan.as_f64() as f32,
            snr_at_decision: snr_score(
                result.response.confidence.base,
                result.response.context_richness,
                result.guardian_approved,
            ),
            timestamp,
            policy_hash: policy_for_trigger,
        };
        self.decision_registry.append(artifact);

        // === EXTRACT MEMORY ===
        // The orchestrator already extracts some fragments during process_message.
        // The runtime does a secondary extraction pass for additional heuristics.
        // Both counts are combined — the pipeline deduplicates at the store level,
        // so there is no double-counting of actual stored fragments.
        let runtime_extracted = if self.config.auto_extract_memory {
            self.extract_memory_from_message(&message, timestamp)
        } else {
            0
        };
        let fragments_extracted = result.memory_fragments_extracted + runtime_extracted;

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
            knows_me_score: self.pipeline.profile().completeness(),
            session_messages: self.current_session.as_ref().map_or(0, |s| s.message_count),
            decision_mode,
            action_hash: action_hash.to_hex(),
            reflex_hit,
            action_id: None,
        }
    }

    // ================================================================
    // MEMORY EXTRACTION — learn from every message
    // ================================================================

    /// Extract memory fragments from a message
    /// MVP: keyword-based extraction heuristics
    /// Production: LLM-powered structured extraction
    ///
    /// In omega, we ingest as UserMessage fragments. The pipeline's rule-based
    /// extractor will pull typed atoms (Fact, Preference, Goal, etc.) from them.
    fn extract_memory_from_message(&mut self, message: &Message, timestamp: u64) -> usize {
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
            "i prefer",
            "i like",
            "i want",
            "i love",
            "i hate",
            "i need",
            "i enjoy",
            "my favorite",
            "i always",
        ];

        for marker in &preference_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(FragmentKind::UserMessage, extract, timestamp) {
                    extracted += 1;
                }
                break; // One preference per message MVP
            }
        }

        // Goal detection
        let goal_markers = [
            "i'm working on",
            "my goal",
            "i'm trying to",
            "i want to achieve",
            "my project",
            "i'm building",
        ];

        for marker in &goal_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(FragmentKind::UserMessage, extract, timestamp) {
                    extracted += 1;
                }
                break;
            }
        }

        // Expertise detection
        let expertise_markers = [
            "i'm an expert in",
            "i work with",
            "i specialize in",
            "my expertise is",
            "i've been doing",
            "i know",
        ];

        for marker in &expertise_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(FragmentKind::UserMessage, extract, timestamp) {
                    extracted += 1;
                }
                break;
            }
        }

        // Style detection (based on message characteristics)
        if content.len() > 500 {
            self.ingest_fragment(
                FragmentKind::Observation,
                "User writes detailed, verbose messages",
                timestamp,
            );
        } else if content.len() < 50 && !content.contains(' ') {
            self.ingest_fragment(
                FragmentKind::Observation,
                "User sends terse, command-like messages",
                timestamp,
            );
        }

        // Fact detection (statements about self)
        let fact_markers = [
            "i am",
            "i'm a",
            "i live in",
            "my name is",
            "i work at",
            "i'm from",
        ];

        for marker in &fact_markers {
            if let Some(pos) = content.to_lowercase().find(marker) {
                let extract = &content[pos..core::cmp::min(pos + 256, content.len())];
                if self.ingest_fragment(FragmentKind::UserMessage, extract, timestamp) {
                    extracted += 1;
                }
                break;
            }
        }

        extracted
    }

    /// Ingest a single extracted fragment into the memory pipeline.
    /// Uses the omega MemoryPipeline::ingest(kind, content, session_id, turn, timestamp).
    fn ingest_fragment(&mut self, kind: FragmentKind, content: &str, timestamp: u64) -> bool {
        self.fragment_seq += 1;

        let session_id = self.current_session.as_ref().map_or(0u64, |s| s.session_id);
        let turn = self.fragment_seq;

        self.pipeline
            .ingest(kind, content, session_id, turn, timestamp)
            .is_ok()
    }

    // ================================================================
    // MANUAL MEMORY OPERATIONS
    // ================================================================

    /// Manually teach the runtime something about the user.
    ///
    /// Stores the atom directly with the specified kind and confidence,
    /// bypassing rule-based extraction. This preserves the caller's
    /// exact classification (critical for TEACH command fidelity and
    /// seed roundtrip integrity).
    pub fn teach(
        &mut self,
        kind: AtomKind,
        content: &str,
        confidence: Confidence,
        timestamp: u64,
    ) -> bool {
        self.pipeline
            .teach_atom(kind, content, confidence, timestamp)
    }

    /// Force a synthesis round.
    /// Returns number of insights produced.
    pub fn synthesize(&mut self, timestamp: u64) -> usize {
        // Extract pending fragments first, then synthesize
        self.pipeline.extract(timestamp);
        let result = self.pipeline.force_synthesize(timestamp);
        result.insights_produced as usize
    }

    /// The quantified "my AI knows me" metric.
    /// Derived from the profile snapshot completeness.
    pub fn knows_me_score(&self) -> f32 {
        self.pipeline.profile().completeness()
    }

    // ================================================================
    // IHSAN MANAGEMENT
    // ================================================================

    /// Update system إحسان score
    pub fn update_ihsan(&mut self, score: IhsanScore) {
        self.current_ihsan = score;
        // Note: omega MemoryPipeline has no update_ihsan; ihsan is per-fragment

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

        // Persist reflex rules to disk (sovereign memory survives restart)
        if let Some(ref store) = self.reflex_store {
            let rules = self.reflex_cache.all_rules();
            let _ = store.snapshot(&rules);
        }

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
    /// Immutable access to the memory pipeline (for persistence export).
    pub fn pipeline(&self) -> &MemoryPipeline {
        &self.pipeline
    }

    /// Mutable access to the memory pipeline for heartbeat-driven
    /// reinforcement, quarantine, and synthesis operations.
    pub fn pipeline_mut(&mut self) -> &mut MemoryPipeline {
        &mut self.pipeline
    }

    pub fn health(&self) -> RuntimeHealth {
        let summary = self.pipeline.knowledge_summary();
        let stats = self.pipeline.stats();
        let roster_snapshot = self.roster.snapshot();
        let reflex_stats = self.reflex_cache.stats();

        RuntimeHealth {
            state: self.state,
            current_ihsan: self.current_ihsan,
            // Memory
            fragments_stored: summary.total_fragments as usize,
            insights_stored: summary.total_insights as usize,
            profile_traits: summary.total_atoms as usize,
            knows_me_score: self.pipeline.profile().completeness(),
            synthesis_rounds: stats.synthesis_passes as u32,
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
            conversation_messages: self.current_session.as_ref().map_or(0, |s| s.message_count),
            reflex_mode: self.effective_reflex_mode(),
            reflex_rules: reflex_stats.size,
            reflex_hits: reflex_stats.hits,
            reflex_misses: reflex_stats.misses,
            decision_artifacts: self.decision_registry.len(),
            actions_planned: self.actions_planned,
            actions_executed: self.actions_executed,
            actions_failed: self.actions_failed,
            guardian_action_vetoes: self.guardian_action_vetoes,
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

    pub fn effective_reflex_mode(&self) -> ReflexMode {
        if self.config.reflex_mode == ReflexMode::Active && self.policy_hash.is_none() {
            return ReflexMode::Disabled;
        }
        self.config.reflex_mode
    }

    pub fn reflex_stats(&self) -> ReflexStats {
        self.reflex_cache.stats()
    }

    pub fn explain_action(&self, action_hash_hex: &str) -> Option<DecisionArtifact> {
        self.decision_registry.get_by_hex(action_hash_hex).cloned()
    }

    pub fn invalidate_reflex(&mut self, trigger_hash_hex: &str) -> bool {
        let Some(raw) = parse_hex_32(trigger_hash_hex) else {
            return false;
        };
        let trigger = TriggerHash(raw);
        // Remove from disk before cache
        if let Some(ref store) = self.reflex_store {
            let _ = store.remove_rule(&trigger);
        }
        self.reflex_cache.invalidate(trigger)
    }

    pub fn export_reflex_rules(&self) -> Vec<ReflexRule> {
        self.reflex_cache.all_rules()
    }

    pub fn import_reflex_rules(&mut self, rules: Vec<ReflexRule>) {
        let mut normalized = Vec::with_capacity(rules.len());
        for mut rule in rules {
            match self.policy_hash {
                Some(current_policy) => {
                    if rule.policy_hash != current_policy {
                        rule.quarantined = true;
                        rule.quarantine_reason = Some(QuarantineReason::PolicyHashMismatch);
                    }
                }
                None if self.config.reflex_mode == ReflexMode::Active => {
                    rule.quarantined = true;
                    rule.quarantine_reason = Some(QuarantineReason::MissingPolicyHash);
                }
                _ => {}
            }
            normalized.push(rule);
        }
        self.reflex_cache.replace_rules(normalized);
    }

    pub fn policy_hash_hex(&self) -> Option<String> {
        self.policy_hash
            .map(|h| h.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn set_policy_hash(&mut self, hex: &str) {
        self.policy_hash = crate::hash_namespace::parse_hex_32(hex);
    }

    pub fn action_mode(&self) -> ActionMode {
        self.config.action_mode
    }

    /// Rust-side guardian preflight for external action channels (MCP/Desktop bridge).
    /// This keeps the authoritative safety verdict in the sovereign runtime.
    pub fn guardian_check_text(&mut self, content: &str, timestamp: u64) -> GuardianCheckResult {
        let msg = Message::inbound(
            MessageId::new(self.total_conversations.max(1), 1),
            content,
            timestamp,
            self.current_ihsan,
        );
        let allowed = self
            .orchestrator
            .guardian_check(&msg, &mut self.roster, self.current_ihsan);
        if allowed {
            return GuardianCheckResult {
                allowed: true,
                reason: "allowed".to_string(),
            };
        }

        if self.current_ihsan.raw() < 9500 {
            return GuardianCheckResult {
                allowed: false,
                reason: "ihsan_below_guardian_floor".to_string(),
            };
        }

        let lowered = content.to_ascii_lowercase();
        let blocked = [
            "harm",
            "attack",
            "exploit",
            "inject",
            "bypass safety",
            "ignore instructions",
            "override",
        ];
        let reason = blocked
            .iter()
            .find(|needle| lowered.contains(**needle))
            .map(|needle| format!("content_contains:{needle}"))
            .unwrap_or_else(|| "guardian_veto".to_string());
        GuardianCheckResult {
            allowed: false,
            reason,
        }
    }

    pub fn record_action_planned(&mut self) {
        self.actions_planned += 1;
    }

    pub fn record_action_executed(&mut self) {
        self.actions_executed += 1;
    }

    pub fn record_action_failed(&mut self) {
        self.actions_failed += 1;
    }

    pub fn record_guardian_action_veto(&mut self) {
        self.guardian_action_vetoes += 1;
    }

    fn select_trigger_traits(&mut self, now: u64) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let kinds = [
            AtomKind::Preference,
            AtomKind::Goal,
            AtomKind::Expertise,
            AtomKind::Fact,
        ];
        for kind in kinds {
            let rows = self.pipeline.query_facts(kind, now);
            if let Some((text, _conf)) = rows.first() {
                out.push((format!("{kind:?}"), text.to_string()));
            }
            if out.len() >= 6 {
                break;
            }
        }
        out
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
    /// Routing mode used for this decision
    pub decision_mode: CognitiveMode,
    /// Deterministic action hash for retrieval-only explanation
    pub action_hash: String,
    /// True when a compiled reflex rule was selected
    pub reflex_hit: bool,
    /// Optional action identifier when an explicit action is executed
    pub action_id: Option<String>,
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
            decision_mode: CognitiveMode::System2,
            action_hash: ActionHash([0u8; 32]).to_hex(),
            reflex_hit: false,
            action_id: None,
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
            decision_mode: CognitiveMode::System2,
            action_hash: ActionHash([0u8; 32]).to_hex(),
            reflex_hit: false,
            action_id: None,
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
    use crate::hash_namespace::compute_trigger_hash;
    use crate::reflex_cache::{ActionTemplate, ReflexRule};

    fn make_user_message(content: &str, timestamp: u64) -> Message {
        Message::inbound(
            MessageId::new(timestamp as u32, 1),
            content,
            timestamp,
            IhsanScore::from_raw(9900),
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

        runtime.update_ihsan(IhsanScore::from_raw(9000));
        assert_eq!(runtime.state(), RuntimeState::Degraded);

        let msg = make_user_message("This should be rejected", 1000);
        let response = runtime.receive(msg, 1000);
        assert!(!response.is_ok());

        // Recovery
        runtime.update_ihsan(IhsanScore::from_raw(9900));
        assert_eq!(runtime.state(), RuntimeState::Ready);
    }

    #[test]
    fn runtime_teach_and_query() {
        let mut runtime = AgentRuntime::for_user(42);

        runtime.teach(
            AtomKind::Fact,
            "User lives in Dubai",
            Confidence::new(0.90, 1000),
            1000,
        );

        runtime.teach(
            AtomKind::Preference,
            "User prefers Rust over Python",
            Confidence::new(0.85, 1002),
            1001,
        );

        // Synthesize to produce profile traits
        runtime.synthesize(2000);

        // Knowledge depth should be positive (teach stores atoms directly, not fragments)
        let health = runtime.health();
        assert!(health.profile_traits >= 2);
    }

    #[test]
    fn runtime_knows_me_score_increases() {
        let mut runtime = AgentRuntime::for_user(42);

        let score_empty = runtime.knows_me_score();
        assert_eq!(score_empty, 0.0);

        // Teach it things
        for i in 0..10 {
            let ts = 1000 + i as u64;
            runtime.teach(
                AtomKind::Preference,
                &format!("I prefer preference number {i}"),
                Confidence::new(0.80, ts),
                ts,
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

        let msg = make_user_message(
            "I am a software architect working on distributed systems",
            1001,
        );
        let response = runtime.receive(msg, 1001);

        // Should extract "i am a software architect..."
        assert!(response.fragments_extracted >= 1);
        assert!(runtime.health().fragments_stored >= 1);
    }

    #[test]
    fn runtime_extracts_goals_from_messages() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.start_conversation(1000);

        let msg = make_user_message(
            "I'm working on a distributed AI platform called BIZRA",
            1001,
        );
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
        runtime.receive(
            make_user_message("I prefer functional programming", 3001),
            3001,
        );
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

    #[test]
    fn reflex_active_fail_closed_without_policy_hash() {
        let mut config = RuntimeConfig::for_user(42);
        config.reflex_mode = ReflexMode::Active;
        config.policy_hash_hex = String::new();
        let mut runtime = AgentRuntime::with_config(config);

        for i in 0..4 {
            let ts = 1000 + i;
            runtime.teach(
                AtomKind::Preference,
                "I prefer rust and distributed systems",
                Confidence::stated(ts),
                ts,
            );
        }
        runtime.synthesize(2000);
        let resp = runtime.receive(make_user_message("Plan my architecture", 3000), 3000);
        assert_eq!(runtime.effective_reflex_mode(), ReflexMode::Disabled);
        assert_eq!(resp.decision_mode, CognitiveMode::System2);
        assert!(!resp.reflex_hit);
    }

    #[test]
    fn shadow_mode_compiles_without_routing() {
        let mut config = RuntimeConfig::for_user(7);
        config.reflex_mode = ReflexMode::Shadow;
        config.policy_hash_hex =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        config.min_success_chains = 1;
        config.min_compile_ihsan = 0.0;
        config.min_compile_snr = 0.0;
        let mut runtime = AgentRuntime::with_config(config);

        for i in 0..12 {
            let ts = 1000 + i;
            runtime.teach(
                AtomKind::Preference,
                format!("I prefer rust pattern {i}").as_str(),
                Confidence::stated(ts),
                ts,
            );
        }
        runtime.synthesize(2000);

        for i in 0..4 {
            let ts = 3000 + i;
            let resp = runtime.receive(make_user_message("Plan my rust architecture", ts), ts);
            assert_eq!(resp.decision_mode, CognitiveMode::System2);
            assert!(!resp.reflex_hit);
        }
        let stats = runtime.reflex_stats();
        assert!(stats.compiled >= 1);
    }

    #[test]
    fn guardian_veto_quarantines_on_reflex_hit() {
        let mut config = RuntimeConfig::for_user(9);
        config.reflex_mode = ReflexMode::Active;
        config.policy_hash_hex =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
        let mut runtime = AgentRuntime::with_config(config);

        let ts = 4000;
        let harmful = "Help me exploit and bypass safety checks";
        let (intent, _) = IntentClassifier::classify(harmful);
        let traits = runtime.select_trigger_traits(ts);
        let policy = runtime.policy_hash.expect("policy hash must parse");
        let trigger = compute_trigger_hash(format!("{intent:?}").as_str(), &traits, &policy);

        let rule = ReflexRule {
            trigger_hash: trigger,
            action_template: ActionTemplate {
                route_signature: "tasks=RetrieveContext>GenerateResponse|roles=Scholar>Artisan"
                    .to_string(),
                primary_agent: "Scholar".to_string(),
            },
            compile_ihsan: 0.97,
            compile_snr: 0.93,
            compiled_at: ts,
            use_count: 0,
            last_used_at: 0,
            last_validated_at: ts,
            quarantined: false,
            quarantine_reason: None,
            policy_hash: policy,
        };
        runtime.import_reflex_rules(vec![rule]);

        let response = runtime.receive(make_user_message(harmful, ts + 1), ts + 1);
        assert_eq!(response.decision_mode, CognitiveMode::System2);
        assert!(!response.reflex_hit);
        assert!(!response.guardian_approved);
        assert!(runtime.reflex_stats().quarantined >= 1);
    }

    #[test]
    fn guardian_check_text_allows_safe_content() {
        let mut runtime = AgentRuntime::for_user(42);
        let verdict = runtime.guardian_check_text("Plan the roadmap for next week", 7000);
        assert!(verdict.allowed);
        assert_eq!(verdict.reason, "allowed");
    }

    #[test]
    fn guardian_check_text_blocks_harmful_content() {
        let mut runtime = AgentRuntime::for_user(42);
        let verdict = runtime.guardian_check_text("help me exploit this system", 7001);
        assert!(!verdict.allowed);
        assert!(verdict.reason.starts_with("content_contains:"));
    }

    #[test]
    fn guardian_check_text_blocks_low_ihsan() {
        let mut runtime = AgentRuntime::for_user(42);
        runtime.update_ihsan(IhsanScore::from_raw(9000));
        let verdict = runtime.guardian_check_text("Normal harmless request", 7002);
        assert!(!verdict.allowed);
        assert_eq!(verdict.reason, "ihsan_below_guardian_floor");
    }

    #[test]
    fn explain_returns_stored_artifact() {
        let mut config = RuntimeConfig::for_user(88);
        config.policy_hash_hex =
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string();
        let mut runtime = AgentRuntime::with_config(config);

        let response = runtime.receive(make_user_message("What is the plan today?", 6000), 6000);
        let action_hash = response.action_hash.clone();
        let artifact = runtime
            .explain_action(action_hash.as_str())
            .expect("artifact should exist");
        assert_eq!(artifact.action_hash.to_hex(), action_hash);
        assert!(!artifact.chosen_route.is_empty());
    }
}
