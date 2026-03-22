//! # bizra-memory — The Soul of "My AI Knows Me"
//!
//! Memory Synthesis Pipeline: transforms conversations into understanding.
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    BizraMemory                          │
//! │                                                         │
//! │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────┐ │
//! │  │  Ingest   │→│  Extract  │→│ Synthesize │→│ Query  │ │
//! │  │ fragments │  │  atoms   │  │  insights  │  │results│ │
//! │  └──────────┘  └──────────┘  └───────────┘  └───────┘ │
//! │       ↑              ↑              ↑            ↑      │
//! │       └──────────────┴──────────────┴────────────┘      │
//! │                    InMemoryStore                         │
//! │                                                         │
//! │  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │
//! │  │ RuleExtr │  │ Synthesis │  │    Bridge (FFI)       │ │
//! │  │ (boot)   │  │  Engine   │  │  → CognitiveResonance │ │
//! │  │          │  │          │  │  → VectorSearch        │ │
//! │  └──────────┘  └──────────┘  └───────────────────────┘ │
//! │                                                         │
//! │  Registers as component in BizraSystem (bizra-hooks)    │
//! │  Events flow through hook pipeline with إحسان gate     │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//! ```rust
//! use bizra_memory::BizraMemory;
//!
//! let mut memory = BizraMemory::new();
//!
//! // Process conversation turns
//! memory.process_user_turn("I am Mumo, CEO of BIZRA", 1, 1, 1000);
//! memory.process_user_turn("I prefer Rust for sovereignty", 1, 2, 2000);
//!
//! // Query what the system knows
//! let facts = memory.what_do_i_know(3000);
//! let profile = memory.who_is_the_user();
//! ```

pub mod bridge;
pub mod pipeline;
pub mod store;
pub mod synthesis;
pub mod types;

// Re-exports for convenience
// Re-export hooks types used in our API
pub use bizra_hooks::{ComponentId, IhsanScore};
pub use bridge::{
    export_atoms_as_turns, BridgeHealth, BridgeStatus, ConversationTurnWire, ExtractionBatch,
    ExtractionContent, ExtractionResult, Extractor, RuleExtractor, SearchBatch, SearchResult,
    Searcher,
};
pub use pipeline::{
    GenesisSeedResult, KnowledgeSummary, MemoryPipeline, PipelineConfig, PipelineStats,
};
pub use store::{InMemoryStore, StoreError};
pub use synthesis::{SynthesisConfig, SynthesisEngine};
pub use types::*;

/// The unified memory system facade.
///
/// This is the top-level API. Everything routes through here:
/// - Conversation processing
/// - Knowledge queries
/// - Profile access
/// - Health monitoring
/// - Bridge status
pub struct BizraMemory {
    /// The core pipeline
    pipeline: MemoryPipeline,
    /// Bridge health tracking
    bridge_health: BridgeHealth,
    /// Component identity in BizraSystem
    component_id: ComponentId,
    /// Whether the system is active
    active: bool,
    /// Total turns processed
    turns_processed: u64,
}

impl BizraMemory {
    /// Create a new memory system with default configuration.
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// Create with custom pipeline configuration.
    pub fn with_config(config: PipelineConfig) -> Self {
        BizraMemory {
            pipeline: MemoryPipeline::with_config(config),
            bridge_health: BridgeHealth::disconnected(),
            component_id: ComponentId::from_name("memory-engine", "0.1.0"),
            active: true,
            turns_processed: 0,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Conversation Processing — The main input path
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Process a user message through the full pipeline.
    pub fn process_user_turn(
        &mut self,
        content: &str,
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> TurnResult {
        if !self.active {
            return TurnResult::inactive();
        }

        self.turns_processed += 1;
        let stage_results = self
            .pipeline
            .process_turn(content, true, session_id, turn, timestamp);

        TurnResult {
            ingested: stage_results.ingested,
            atoms_extracted: stage_results.atoms_extracted,
            insights_produced: stage_results
                .synthesis_result
                .map(|s| s.insights_produced)
                .unwrap_or(0),
            synthesis_triggered: stage_results.synthesis_result.is_some(),
        }
    }

    /// Process an assistant message (lower priority, context enrichment).
    pub fn process_assistant_turn(
        &mut self,
        content: &str,
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> TurnResult {
        if !self.active {
            return TurnResult::inactive();
        }

        self.turns_processed += 1;
        let stage_results = self
            .pipeline
            .process_turn(content, false, session_id, turn, timestamp);

        TurnResult {
            ingested: stage_results.ingested,
            atoms_extracted: stage_results.atoms_extracted,
            insights_produced: stage_results
                .synthesis_result
                .map(|s| s.insights_produced)
                .unwrap_or(0),
            synthesis_triggered: stage_results.synthesis_result.is_some(),
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Knowledge Queries — The main output path
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// "What do I know about this user?" — all reliable facts.
    pub fn what_do_i_know(&mut self, now: u64) -> Vec<(&str, f32)> {
        self.pipeline.query_facts(AtomKind::Fact, now)
    }

    /// "What does the user prefer?" — preferences with confidence.
    pub fn user_preferences(&mut self, now: u64) -> Vec<(&str, f32)> {
        self.pipeline.query_facts(AtomKind::Preference, now)
    }

    /// "What is the user working on?" — active goals.
    pub fn user_goals(&mut self, now: u64) -> Vec<(&str, f32)> {
        self.pipeline.query_facts(AtomKind::Goal, now)
    }

    /// "What does the user NOT want?" — negations and boundaries.
    pub fn user_boundaries(&mut self, now: u64) -> Vec<(&str, f32)> {
        self.pipeline.query_facts(AtomKind::Negation, now)
    }

    /// "What patterns have I observed?" — behavioral patterns.
    pub fn user_patterns(&mut self, now: u64) -> Vec<(&str, f32)> {
        self.pipeline.query_facts(AtomKind::Pattern, now)
    }

    /// "What are the user's principles?" — values and principles.
    pub fn user_principles(&mut self, now: u64) -> Vec<(&str, f32)> {
        self.pipeline.query_facts(AtomKind::Principle, now)
    }

    /// "What insights have been synthesized?" — connected understanding.
    pub fn insights(&mut self) -> Vec<(&str, f32)> {
        self.pipeline.query_insights(None)
    }

    /// "Who is the user?" — profile snapshot.
    pub fn who_is_the_user(&self) -> &ProfileSnapshot {
        self.pipeline.profile()
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // System Control
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Activate the memory system.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate (pause processing, queries still work).
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Is the system active?
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Force a synthesis pass (regardless of batch threshold).
    pub fn force_synthesis(&mut self, now: u64) {
        self.pipeline.force_synthesize(now);
    }

    /// Load a genesis seed file (TEACH format) with HHMM-aware TTL.
    ///
    /// Each atom kind gets the appropriate half-life for its cognitive layer:
    /// - Glacial (facts, principles, negations): 182-day half-life
    /// - Slow (expertise, patterns, preferences): 45-90 day half-life
    /// - Fast (goals, temporals, context): 3.5-15 day half-life
    pub fn load_genesis_seed(&mut self, seed_text: &str, base_timestamp: u64) -> GenesisSeedResult {
        self.pipeline.load_genesis_seed(seed_text, base_timestamp)
    }

    /// Get the component ID for BizraSystem registration.
    pub fn component_id(&self) -> ComponentId {
        self.component_id
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Health & Telemetry
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Full memory system health snapshot.
    pub fn health(&self) -> MemoryHealth {
        let knowledge = self.pipeline.knowledge_summary();
        let stats = self.pipeline.stats();

        MemoryHealth {
            active: self.active,
            turns_processed: self.turns_processed,
            fragments: knowledge.total_fragments,
            atoms: knowledge.total_atoms,
            active_atoms: knowledge.active_atoms,
            insights: knowledge.total_insights,
            profile_completeness: knowledge.profile_completeness,
            profile_sections: knowledge.profile_sections,
            synthesis_passes: stats.synthesis_passes,
            queries_served: stats.queries_served,
            bridge: self.bridge_health,
        }
    }

    /// Get pipeline stats.
    pub fn stats(&self) -> &PipelineStats {
        self.pipeline.stats()
    }

    /// Get knowledge summary.
    pub fn knowledge_summary(&self) -> KnowledgeSummary {
        self.pipeline.knowledge_summary()
    }
}

impl Default for BizraMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a conversation turn.
#[derive(Debug, Clone, Copy)]
pub struct TurnResult {
    pub ingested: bool,
    pub atoms_extracted: u32,
    pub insights_produced: u32,
    pub synthesis_triggered: bool,
}

impl TurnResult {
    fn inactive() -> Self {
        TurnResult {
            ingested: false,
            atoms_extracted: 0,
            insights_produced: 0,
            synthesis_triggered: false,
        }
    }
}

/// Complete memory system health.
#[derive(Debug, Clone, Copy)]
pub struct MemoryHealth {
    pub active: bool,
    pub turns_processed: u64,
    pub fragments: u32,
    pub atoms: u32,
    pub active_atoms: u32,
    pub insights: u32,
    pub profile_completeness: f32,
    pub profile_sections: u32,
    pub synthesis_passes: u64,
    pub queries_served: u64,
    pub bridge: BridgeHealth,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Integration Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_lifecycle_test() {
        let mut memory = BizraMemory::new();

        // Simulate a conversation
        let r1 = memory.process_user_turn("I am Mumo, the founder and CEO of BIZRA", 1, 1, 1000);
        assert!(r1.ingested);
        assert!(r1.atoms_extracted >= 1);

        let r2 = memory.process_user_turn("I prefer Rust for sovereign architecture", 1, 2, 2000);
        assert!(r2.ingested);

        let r3 = memory.process_user_turn(
            "I am building the world's first distributed AI platform",
            1,
            3,
            3000,
        );
        assert!(r3.ingested);

        // Check knowledge
        let facts = memory.what_do_i_know(3000);
        assert!(!facts.is_empty());

        let prefs = memory.user_preferences(3000);
        assert!(!prefs.is_empty());

        let goals = memory.user_goals(3000);
        assert!(!goals.is_empty());
    }

    #[test]
    fn conversation_builds_profile() {
        let config = PipelineConfig {
            synthesis_batch_size: 3,
            ..Default::default()
        };
        let mut memory = BizraMemory::with_config(config);

        // Build knowledge through conversation
        memory.process_user_turn("I am Mumo", 1, 1, 1000);
        memory.process_user_turn("I prefer sovereign tech", 1, 2, 2000);
        memory.process_user_turn("I always work after Fajr", 1, 3, 3000);
        memory.process_user_turn("My core principle is إحسان", 1, 4, 4000);

        // Force synthesis to ensure profile updates
        memory.force_synthesis(5000);

        let profile = memory.who_is_the_user();
        assert!(profile.total_atoms > 0);
        assert!(profile.section_count() > 0);
    }

    #[test]
    fn health_snapshot() {
        let mut memory = BizraMemory::new();

        memory.process_user_turn("I am building BIZRA", 1, 1, 1000);

        let health = memory.health();
        assert!(health.active);
        assert_eq!(health.turns_processed, 1);
        assert!(health.fragments >= 1);
        assert!(health.atoms >= 1);
        assert_eq!(health.bridge.status, BridgeStatus::Disconnected); // no Python yet
    }

    #[test]
    fn deactivated_memory_rejects_input() {
        let mut memory = BizraMemory::new();
        memory.deactivate();

        let result = memory.process_user_turn("test", 1, 1, 1000);
        assert!(!result.ingested);
        assert_eq!(result.atoms_extracted, 0);
    }

    #[test]
    fn multi_session_accumulation() {
        let mut memory = BizraMemory::new();

        // Session 1
        memory.process_user_turn("I am working on BIZRA", 1, 1, 1000);
        memory.process_user_turn("I need a hook system", 1, 2, 2000);

        // Session 2 (different session_id)
        memory.process_user_turn("I prefer zero dependencies", 2, 1, 10000);
        memory.process_user_turn("I am preparing investor materials", 2, 2, 11000);

        let summary = memory.knowledge_summary();
        assert_eq!(summary.total_fragments, 4);
        assert!(summary.total_atoms >= 4);
    }

    #[test]
    fn assistant_turns_captured() {
        let mut memory = BizraMemory::new();

        memory.process_user_turn("Help me design the memory system", 1, 1, 1000);
        memory.process_assistant_turn(
            "I recommend starting with a type system for memory atoms",
            1,
            2,
            2000,
        );

        let summary = memory.knowledge_summary();
        assert_eq!(summary.total_fragments, 2);
    }

    #[test]
    fn the_four_word_test() {
        // "My AI knows me" — the value proposition test.
        // After processing real conversation content, the system should
        // be able to answer questions about the user.

        let config = PipelineConfig {
            synthesis_batch_size: 2,
            ..Default::default()
        };
        let mut memory = BizraMemory::with_config(config);

        // Simulate real conversation turns
        memory.process_user_turn("I am Mumo, founder and CEO of BIZRA in Dubai", 1, 1, 1000);
        memory.process_user_turn(
            "I prefer building sovereign systems with zero dependencies",
            1,
            2,
            2000,
        );
        memory.process_user_turn("I don't want centralized cloud dependencies", 1, 3, 3000);
        memory.process_user_turn(
            "I always start my deep work after Fajr prayer every morning",
            1,
            4,
            4000,
        );
        memory.process_user_turn(
            "My guiding principle is إحسان — excellence as worship",
            1,
            5,
            5000,
        );
        memory.process_user_turn(
            "I am preparing investor pitch materials for Series A",
            1,
            6,
            6000,
        );

        memory.force_synthesis(7000);

        // THE TEST: Does the system know me?

        // Identity
        let facts = memory.what_do_i_know(7000);
        assert!(!facts.is_empty(), "System should know facts about user");

        // Preferences
        let prefs = memory.user_preferences(7000);
        assert!(!prefs.is_empty(), "System should know user preferences");

        // Boundaries
        let boundaries = memory.user_boundaries(7000);
        assert!(!boundaries.is_empty(), "System should know user boundaries");

        // Patterns
        let patterns = memory.user_patterns(7000);
        assert!(!patterns.is_empty(), "System should know user patterns");

        // Principles
        let principles = memory.user_principles(7000);
        assert!(!principles.is_empty(), "System should know user principles");

        // Goals
        let goals = memory.user_goals(7000);
        assert!(!goals.is_empty(), "System should know user goals");

        // Profile completeness
        let profile = memory.who_is_the_user();
        assert!(
            profile.section_count() >= 3,
            "Profile should have 3+ sections"
        );
        assert!(
            profile.completeness() >= 0.3,
            "Profile should be 30%+ complete"
        );

        // Insights exist
        let insights = memory.insights();
        assert!(
            !insights.is_empty(),
            "System should have synthesized insights"
        );
    }
}
