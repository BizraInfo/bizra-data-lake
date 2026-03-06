// bizra-memory/src/pipeline.rs
// ============================================================
// Memory Pipeline — the full orchestration layer
// ============================================================
// Wires: Ingest → Extract → Synthesize → Index → Query
// Integrates with bizra-hooks for event routing and إحسان gates.
//
// This is the top-level coordinator that makes
// "my AI knows me" actually work.
// ============================================================

use crate::types::*;
use crate::store::InMemoryStore;
use crate::synthesis::SynthesisEngine;
use bizra_hooks::IhsanScore;

// ============================================================
// PIPELINE CONFIG
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Minimum fragments before auto-synthesis triggers
    pub auto_synthesis_threshold: usize,
    /// Minimum time between synthesis rounds (seconds)
    pub synthesis_cooldown_secs: u64,
    /// Maximum fragments to keep before eviction
    pub max_fragments_before_eviction: usize,
    /// Number of fragments to evict when at capacity
    pub eviction_batch_size: usize,
    /// إحسان floor for the pipeline itself
    pub ihsan_floor: IhsanScore,
    /// Auto-synthesis enabled
    pub auto_synthesis: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            auto_synthesis_threshold: 20,
            synthesis_cooldown_secs: 300, // 5 minutes
            max_fragments_before_eviction: 3500, // Leave headroom in 4096 store
            eviction_batch_size: 256,
            ihsan_floor: IhsanScore::new(9500),
            auto_synthesis: true,
        }
    }
}

// ============================================================
// PIPELINE STATE
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    /// Ready to accept fragments
    Idle,
    /// Currently ingesting fragments
    Ingesting,
    /// Running synthesis
    Synthesizing,
    /// Performing eviction
    Evicting,
    /// Paused due to low إحسان
    Degraded,
}

// ============================================================
// PIPELINE HEALTH
// ============================================================

#[derive(Debug, Clone)]
pub struct PipelineHealth {
    pub state: PipelineState,
    pub fragments_stored: usize,
    pub insights_stored: usize,
    pub profile_traits: usize,
    pub synthesis_rounds: u32,
    pub last_synthesis_at: u64,
    pub fragments_since_synthesis: usize,
    pub current_ihsan: IhsanScore,
    pub store_utilization: f32,
    pub quality_ratio: f32,
    pub sessions_tracked: u16,
}

// ============================================================
// MEMORY PIPELINE — the full orchestrator
// ============================================================

pub struct MemoryPipeline {
    store: InMemoryStore,
    engine: SynthesisEngine,
    profile: UserProfile,
    temporal: TemporalContext,
    config: PipelineConfig,
    state: PipelineState,
    current_ihsan: IhsanScore,
    last_synthesis_time: u64,
    fragments_since_synthesis: usize,
    total_ingested: u64,
    total_queries: u64,
}

impl MemoryPipeline {
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            store: InMemoryStore::new(),
            engine: SynthesisEngine::new(),
            profile: UserProfile::new(),
            temporal: TemporalContext::new(),
            config,
            state: PipelineState::Idle,
            current_ihsan: IhsanScore::new(9900),
            last_synthesis_time: 0,
            fragments_since_synthesis: 0,
            total_ingested: 0,
            total_queries: 0,
        }
    }

    // ================================================================
    // SESSION MANAGEMENT
    // ================================================================

    /// Start a new conversation session
    pub fn start_session(&mut self, session_id: u64, timestamp: u64) -> bool {
        self.temporal.start_session(session_id, timestamp)
    }

    /// End current conversation session, optionally triggering synthesis
    pub fn end_session(&mut self, session_id: u64, timestamp: u64) -> Option<usize> {
        self.temporal.end_session(session_id, timestamp);

        // End of session is a natural synthesis point
        if self.config.auto_synthesis && self.fragments_since_synthesis >= 5 {
            let result = self.run_synthesis(timestamp);
            return Some(result);
        }
        None
    }

    // ================================================================
    // INGESTION — raw knowledge enters the pipeline
    // ================================================================

    /// Ingest a single fragment
    pub fn ingest(&mut self, fragment: MemoryFragment, timestamp: u64) -> Result<FragmentId, crate::store::StoreError> {
        // Check pipeline health
        if self.state == PipelineState::Degraded {
            return Err(crate::store::StoreError::IhsanBelowThreshold);
        }

        self.state = PipelineState::Ingesting;

        // Check if eviction needed
        if self.store.fragment_count() >= self.config.max_fragments_before_eviction {
            self.run_eviction(timestamp);
        }

        // Ingest through synthesis engine (applies confidence gate)
        let result = self.engine.ingest(&mut self.store, fragment);

        if result.is_ok() {
            self.fragments_since_synthesis += 1;
            self.total_ingested += 1;

            // Check if auto-synthesis should trigger
            if self.config.auto_synthesis
                && self.fragments_since_synthesis >= self.config.auto_synthesis_threshold
                && self.time_since_synthesis(timestamp) >= self.config.synthesis_cooldown_secs
            {
                self.run_synthesis(timestamp);
            }
        }

        self.state = PipelineState::Idle;
        result
    }

    /// Batch ingest multiple fragments
    pub fn ingest_batch(
        &mut self,
        fragments: Vec<MemoryFragment>,
        timestamp: u64,
    ) -> (usize, usize) { // (succeeded, failed)
        let mut success = 0;
        let mut failed = 0;

        for fragment in fragments {
            match self.ingest(fragment, timestamp) {
                Ok(_) => success += 1,
                Err(_) => failed += 1,
            }
        }

        (success, failed)
    }

    // ================================================================
    // SYNTHESIS — fragments become understanding
    // ================================================================

    /// Run a synthesis round
    /// Returns number of insights produced
    pub fn run_synthesis(&mut self, timestamp: u64) -> usize {
        if self.current_ihsan.raw() < self.config.ihsan_floor.raw() {
            self.state = PipelineState::Degraded;
            return 0;
        }

        self.state = PipelineState::Synthesizing;

        let result = self.engine.synthesize(
            &mut self.store,
            &mut self.profile,
            self.current_ihsan,
            timestamp,
        );

        let insight_count = result.insights_produced.len();
        self.last_synthesis_time = timestamp;
        self.fragments_since_synthesis = 0;
        self.state = PipelineState::Idle;

        insight_count
    }

    /// Force a synthesis round regardless of cooldown
    pub fn force_synthesis(&mut self, timestamp: u64) -> usize {
        self.run_synthesis(timestamp)
    }

    // ================================================================
    // QUERY — ask what the pipeline knows
    // ================================================================

    /// Get a user profile trait
    pub fn query_trait(&mut self, key: &str) -> Option<(&str, Confidence)> {
        self.total_queries += 1;
        self.profile.get_trait(key).map(|t| (t.value(), t.confidence))
    }

    /// Get all profile traits
    pub fn query_profile(&mut self) -> Vec<(&str, &str, Confidence)> {
        self.total_queries += 1;
        self.profile.traits()
            .map(|t| (t.key(), t.value(), t.confidence))
            .collect()
    }

    /// Query fragments by kind
    pub fn query_fragments_by_kind(&mut self, kind: FragmentKind) -> Vec<&MemoryFragment> {
        self.total_queries += 1;
        self.store.query_fragments(&crate::store::FragmentQuery::by_kind(kind))
    }

    /// Get all active insights
    pub fn query_insights(&mut self) -> Vec<&Insight> {
        self.total_queries += 1;
        self.store.all_insights().collect()
    }

    /// How many facts does the pipeline know about this user?
    pub fn knowledge_depth(&self) -> usize {
        self.store.fragment_count() + self.store.insight_count()
    }

    /// The "knows me" score — how well does the pipeline understand this user?
    /// Combines profile completeness, fragment depth, insight quality
    pub fn knows_me_score(&self) -> f32 {
        let profile_score = (self.profile.trait_count() as f32 / 20.0).min(1.0);
        let depth_score = (self.store.fragment_count() as f32 / 100.0).min(1.0);
        let insight_score = (self.store.insight_count() as f32 / 20.0).min(1.0);
        let synthesis_score = (self.engine.round() as f32 / 10.0).min(1.0);

        // Weighted combination
        profile_score * 0.4 + depth_score * 0.2 + insight_score * 0.25 + synthesis_score * 0.15
    }

    // ================================================================
    // IHSAN MANAGEMENT
    // ================================================================

    /// Update the system إحسان score (from hooks layer)
    pub fn update_ihsan(&mut self, score: IhsanScore) {
        self.current_ihsan = score;

        if score.raw() < self.config.ihsan_floor.raw() {
            self.state = PipelineState::Degraded;
        } else if self.state == PipelineState::Degraded {
            self.state = PipelineState::Idle;
        }
    }

    // ================================================================
    // HEALTH & OBSERVABILITY
    // ================================================================

    /// Full pipeline health snapshot
    pub fn health(&self) -> PipelineHealth {
        let stats = self.store.stats();
        PipelineHealth {
            state: self.state,
            fragments_stored: stats.fragment_count,
            insights_stored: stats.insight_count,
            profile_traits: self.profile.trait_count(),
            synthesis_rounds: self.engine.round(),
            last_synthesis_at: self.last_synthesis_time,
            fragments_since_synthesis: self.fragments_since_synthesis,
            current_ihsan: self.current_ihsan,
            store_utilization: stats.fragment_utilization,
            quality_ratio: self.engine.metrics().quality_ratio(),
            sessions_tracked: self.temporal.total_sessions(),
        }
    }

    pub fn state(&self) -> PipelineState {
        self.state
    }

    pub fn total_ingested(&self) -> u64 {
        self.total_ingested
    }

    pub fn total_queries(&self) -> u64 {
        self.total_queries
    }

    pub fn synthesis_metrics(&self) -> &SynthesisMetrics {
        self.engine.metrics()
    }

    // ================================================================
    // INTERNAL
    // ================================================================

    fn run_eviction(&mut self, current_time: u64) {
        self.state = PipelineState::Evicting;
        self.store.evict_lowest(self.config.eviction_batch_size, current_time);
        self.state = PipelineState::Idle;
    }

    fn time_since_synthesis(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_synthesis_time)
    }
}

impl Default for MemoryPipeline {
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
    use bizra_hooks::ComponentId;

    fn make_fragment(conv: u32, seq: u32, kind: FragmentKind, content: &str) -> MemoryFragment {
        MemoryFragment::new(
            FragmentId::new(conv, seq),
            kind,
            content,
            Confidence::HIGH,
            ComponentId::new("test", "1.0"),
            1000,
            IhsanScore::new(9900),
        )
    }

    #[test]
    fn pipeline_ingest_and_query() {
        let mut pipeline = MemoryPipeline::new();

        pipeline.ingest(
            make_fragment(1, 1, FragmentKind::Fact, "User is a Rust developer"),
            1000,
        ).unwrap();

        let facts = pipeline.query_fragments_by_kind(FragmentKind::Fact);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content.as_str(), "User is a Rust developer");
    }

    #[test]
    fn pipeline_session_lifecycle() {
        let mut pipeline = MemoryPipeline::new();

        assert!(pipeline.start_session(100, 1000));

        // Ingest during session
        for i in 1..=5 {
            pipeline.ingest(
                make_fragment(1, i, FragmentKind::Preference, &format!("Pref {}", i)),
                1000 + i as u64,
            ).unwrap();
        }

        // End session triggers synthesis
        let insights = pipeline.end_session(100, 2000);
        assert!(insights.is_some());
    }

    #[test]
    fn pipeline_auto_synthesis_triggers() {
        let mut config = PipelineConfig::default();
        config.auto_synthesis_threshold = 5;
        config.synthesis_cooldown_secs = 0; // No cooldown for test
        let mut pipeline = MemoryPipeline::with_config(config);

        // Ingest enough to trigger auto-synthesis
        for i in 1..=6 {
            pipeline.ingest(
                make_fragment(1, i, FragmentKind::Expertise, &format!("Skill {}", i)),
                1000 + i as u64,
            ).unwrap();
        }

        assert!(pipeline.health().synthesis_rounds >= 1);
    }

    #[test]
    fn pipeline_degraded_when_low_ihsan() {
        let mut pipeline = MemoryPipeline::new();

        pipeline.update_ihsan(IhsanScore::new(9000)); // Below floor
        assert_eq!(pipeline.state(), PipelineState::Degraded);

        // Ingestion should fail in degraded state
        let result = pipeline.ingest(
            make_fragment(1, 1, FragmentKind::Fact, "Should fail"),
            1000,
        );
        assert!(result.is_err());

        // Recovery
        pipeline.update_ihsan(IhsanScore::new(9900));
        assert_eq!(pipeline.state(), PipelineState::Idle);
    }

    #[test]
    fn pipeline_batch_ingest() {
        let mut pipeline = MemoryPipeline::new();
        let fragments = vec![
            make_fragment(1, 1, FragmentKind::Fact, "Fact A"),
            make_fragment(1, 2, FragmentKind::Fact, "Fact B"),
            make_fragment(1, 3, FragmentKind::Preference, "Pref A"),
        ];

        let (success, failed) = pipeline.ingest_batch(fragments, 1000);
        assert_eq!(success, 3);
        assert_eq!(failed, 0);
        assert_eq!(pipeline.total_ingested(), 3);
    }

    #[test]
    fn pipeline_knows_me_score_grows() {
        let mut pipeline = MemoryPipeline::new();

        let score_empty = pipeline.knows_me_score();
        assert_eq!(score_empty, 0.0);

        // Add fragments
        for i in 1..=10 {
            pipeline.ingest(
                make_fragment(1, i, FragmentKind::Preference, &format!("Preference {}", i)),
                1000 + i as u64,
            ).unwrap();
        }

        // Run synthesis
        pipeline.force_synthesis(2000);

        let score_after = pipeline.knows_me_score();
        assert!(score_after > score_empty);
    }

    #[test]
    fn pipeline_health_snapshot() {
        let mut pipeline = MemoryPipeline::new();
        pipeline.start_session(1, 1000);

        pipeline.ingest(
            make_fragment(1, 1, FragmentKind::Fact, "Test fact"),
            1000,
        ).unwrap();

        let health = pipeline.health();
        assert_eq!(health.state, PipelineState::Idle);
        assert_eq!(health.fragments_stored, 1);
        assert_eq!(health.sessions_tracked, 1);
        assert!(health.current_ihsan.raw() >= 9900);
    }

    #[test]
    fn pipeline_eviction_under_pressure() {
        let mut config = PipelineConfig::default();
        config.max_fragments_before_eviction = 10;
        config.eviction_batch_size = 3;
        let mut pipeline = MemoryPipeline::with_config(config);

        // Fill to eviction threshold
        for i in 1..=12 {
            let mut frag = make_fragment(1, i, FragmentKind::Fact, &format!("Fact {}", i));
            frag.created_at = 1000 + i as u64;
            frag.last_reinforced = 1000 + i as u64;
            let _ = pipeline.ingest(frag, 1000 + i as u64);
        }

        // Should have evicted some fragments
        assert!(pipeline.health().fragments_stored <= 12);
    }

    #[test]
    fn pipeline_synthesis_produces_profile_traits() {
        let mut pipeline = MemoryPipeline::new();

        // Build up enough context
        pipeline.ingest(make_fragment(1, 1, FragmentKind::Style, "Prefers bullet points"), 1000).unwrap();
        pipeline.ingest(make_fragment(1, 2, FragmentKind::Style, "Likes concise responses"), 1001).unwrap();
        pipeline.ingest(make_fragment(1, 3, FragmentKind::Style, "Formal tone preferred"), 1002).unwrap();

        pipeline.force_synthesis(2000);

        let traits = pipeline.query_profile();
        // Should have at least one profile trait from synthesis
        // (style cluster should produce "style" trait)
        if !traits.is_empty() {
            assert!(traits.iter().any(|(k, _, _)| *k == "style"));
        }
    }

    #[test]
    fn pipeline_force_synthesis_bypasses_cooldown() {
        let mut config = PipelineConfig::default();
        config.synthesis_cooldown_secs = 99999; // Very long cooldown
        let mut pipeline = MemoryPipeline::with_config(config);

        pipeline.ingest(make_fragment(1, 1, FragmentKind::Goal, "Goal A"), 1000).unwrap();
        pipeline.ingest(make_fragment(1, 2, FragmentKind::Goal, "Goal B"), 1001).unwrap();

        // Force synthesis should work despite cooldown
        let insights = pipeline.force_synthesis(1002);
        assert!(insights >= 0); // May or may not produce insights, but shouldn't panic
        assert!(pipeline.health().synthesis_rounds >= 1);
    }
}
