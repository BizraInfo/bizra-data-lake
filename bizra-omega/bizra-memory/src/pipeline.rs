//! # Memory Pipeline — The Five-Stage Orchestrator
//!
//! ```text
//! Ingest → Extract → Synthesize → Index → Query
//!   ↑                                       ↓
//! (EventBus)                          (EventBus)
//! ```
//!
//! Each stage is triggered by events flowing through bizra-hooks.
//! The pipeline coordinates the store and synthesis engine.
//!
//! ## Integration with BizraSystem
//! - Registers as a component ("memory-engine")
//! - Subscribes to conversation events
//! - Emits knowledge events (new insights, profile updates)
//! - إحسان score tracks knowledge quality

use crate::store::InMemoryStore;
use crate::synthesis::{SynthesisConfig, SynthesisEngine, SynthesisPassResult};
use crate::types::*;

/// Pipeline configuration.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Auto-synthesize after this many new atoms
    pub synthesis_batch_size: usize,
    /// Maximum fragments to process per ingest cycle
    pub max_ingest_batch: usize,
    /// Synthesis engine config
    pub synthesis: SynthesisConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            synthesis_batch_size: 10,
            max_ingest_batch: 32,
            synthesis: SynthesisConfig::default(),
        }
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone, Copy)]
pub struct PipelineStats {
    pub fragments_ingested: u64,
    pub atoms_extracted: u64,
    pub insights_synthesized: u64,
    pub queries_served: u64,
    pub synthesis_passes: u64,
    pub extract_passes: u64,
    /// Atoms pending synthesis
    pub pending_synthesis: u32,
    /// Fragments pending extraction
    pub pending_extraction: u32,
}

/// The Memory Pipeline — orchestrates the full Ingest→Query flow.
pub struct MemoryPipeline {
    /// Knowledge store
    store: InMemoryStore,
    /// Synthesis engine
    synthesis: SynthesisEngine,
    /// Configuration
    config: PipelineConfig,
    /// Statistics
    stats: PipelineStats,
    /// Atoms accumulated since last synthesis
    atoms_since_synthesis: usize,
}

impl MemoryPipeline {
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    pub fn with_config(config: PipelineConfig) -> Self {
        MemoryPipeline {
            store: InMemoryStore::new(),
            synthesis: SynthesisEngine::with_config(config.synthesis),
            config,
            stats: PipelineStats {
                fragments_ingested: 0,
                atoms_extracted: 0,
                insights_synthesized: 0,
                queries_served: 0,
                synthesis_passes: 0,
                extract_passes: 0,
                pending_synthesis: 0,
                pending_extraction: 0,
            },
            atoms_since_synthesis: 0,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Stage 1: Ingest — Raw content enters the system
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Ingest a conversation turn (user or assistant message).
    pub fn ingest(
        &mut self,
        kind: FragmentKind,
        content: &str,
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> Result<FragmentId, crate::store::StoreError> {
        let id =
            self.store
                .ingest_fragment(kind, content.as_bytes(), session_id, turn, timestamp)?;
        self.stats.fragments_ingested += 1;
        Ok(id)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Stage 2: Extract — Pull atoms from fragments
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Extract atoms from pending fragments.
    ///
    /// In production, this calls CognitiveResonance via FFI.
    /// For Node0 v1, we use rule-based extraction as bootstrap.
    pub fn extract(&mut self, timestamp: u64) -> u32 {
        self.stats.extract_passes += 1;
        let mut extracted = 0u32;

        // Collect fragment data (avoid borrow conflict)
        let pending: Vec<(FragmentId, u64, u32, u64, u32)> = self
            .store
            .pending_extraction()
            .take(self.config.max_ingest_batch)
            .map(|f| {
                (
                    f.header.id,
                    f.header.content_offset,
                    f.header.content_len,
                    f.header.session_id,
                    f.header.turn,
                )
            })
            .collect();

        let extractor = bizra_hooks::ComponentId::from_name("memory-engine", "0.1.0");

        for (frag_id, offset, len, session_id, turn) in pending {
            if let Some(content) = self.store.get_content_str(offset, len) {
                let content_owned = content.to_string();
                let provenance = Provenance::new(session_id, turn, extractor, timestamp);

                // Rule-based extraction (bootstrap — FFI to LLM in production)
                let atoms = self.rule_extract(&content_owned, frag_id, provenance, timestamp);
                extracted += atoms;
            }
            self.store.mark_extracted(&frag_id);
        }

        self.stats.atoms_extracted += extracted as u64;
        self.atoms_since_synthesis += extracted as usize;

        extracted
    }

    /// Rule-based atom extraction — the bootstrap extractor.
    ///
    /// Identifies patterns in text and creates atoms. Simple but effective
    /// for building the initial knowledge base before LLM extraction is wired.
    fn rule_extract(
        &mut self,
        content: &str,
        frag_id: FragmentId,
        provenance: Provenance,
        timestamp: u64,
    ) -> u32 {
        let mut count = 0u32;
        let lower = content.to_lowercase();

        // Pattern: "I am X" / "I'm X" → Fact or Relationship
        if lower.contains("i am ") || lower.contains("i'm ") {
            let _ = self.store.store_atom(
                AtomKind::Fact,
                content,
                frag_id,
                Confidence::stated(timestamp),
                provenance,
            );
            count += 1;
        }

        // Pattern: "I prefer X" / "I like X" → Preference
        if lower.contains("i prefer")
            || lower.contains("i like")
            || lower.contains("i want")
            || lower.contains("i need")
        {
            let _ = self.store.store_atom(
                AtomKind::Preference,
                content,
                frag_id,
                Confidence::stated(timestamp),
                provenance,
            );
            count += 1;
        }

        // Pattern: "working on X" / "building X" → Goal
        if lower.contains("working on")
            || lower.contains("building")
            || lower.contains("preparing")
            || lower.contains("planning")
        {
            let _ = self.store.store_atom(
                AtomKind::Goal,
                content,
                frag_id,
                Confidence::inferred(timestamp),
                provenance,
            );
            count += 1;
        }

        // Pattern: "I don't" / "never" / "not" → Negation
        if lower.contains("i don't")
            || lower.contains("i never")
            || lower.contains("i do not")
            || lower.contains("not interested")
        {
            let _ = self.store.store_atom(
                AtomKind::Negation,
                content,
                frag_id,
                Confidence::stated(timestamp),
                provenance,
            );
            count += 1;
        }

        // Pattern: "every day" / "usually" / "always" → Pattern
        if lower.contains("every day")
            || lower.contains("usually")
            || lower.contains("always")
            || lower.contains("every morning")
        {
            let _ = self.store.store_atom(
                AtomKind::Pattern,
                content,
                frag_id,
                Confidence::inferred(timestamp),
                provenance,
            );
            count += 1;
        }

        // Pattern: "my principle" / "I believe" / "my value" → Principle
        if lower.contains("my principle")
            || lower.contains("my guiding")
            || lower.contains("i believe")
            || lower.contains("my value")
            || lower.contains("my standard")
        {
            let _ = self.store.store_atom(
                AtomKind::Principle,
                content,
                frag_id,
                Confidence::stated(timestamp),
                provenance,
            );
            count += 1;
        }

        // Fallback: if no rules matched, store as Context
        if count == 0 && content.len() > 20 {
            let _ = self.store.store_atom(
                AtomKind::Context,
                content,
                frag_id,
                Confidence::speculative(timestamp),
                provenance,
            );
            count += 1;
        }

        count
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Stage 3: Synthesize — Atoms → Insights
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Run synthesis if enough atoms have accumulated.
    pub fn maybe_synthesize(&mut self, now: u64) -> Option<SynthesisPassResult> {
        if self.atoms_since_synthesis >= self.config.synthesis_batch_size {
            Some(self.force_synthesize(now))
        } else {
            None
        }
    }

    /// Force a synthesis pass regardless of batch threshold.
    pub fn force_synthesize(&mut self, now: u64) -> SynthesisPassResult {
        let result = self.synthesis.synthesize(&mut self.store, now);
        self.stats.synthesis_passes += 1;
        self.stats.insights_synthesized += result.insights_produced as u64;
        self.atoms_since_synthesis = 0;
        result
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Stage 4+5: Query — Retrieve knowledge
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Query the knowledge base for facts about the user.
    pub fn query_facts(&mut self, kind: AtomKind, now: u64) -> Vec<(&str, f32)> {
        self.stats.queries_served += 1;
        self.store
            .atoms_by_kind(kind)
            .filter(|a| a.header.confidence.is_reliable(now))
            .filter_map(|a| {
                self.store
                    .atom_content(a)
                    .map(|c| (c, a.header.confidence.effective_at(now)))
            })
            .collect()
    }

    /// Query insights by synthesis method.
    pub fn query_insights(&mut self, method: Option<SynthesisMethod>) -> Vec<(&str, f32)> {
        self.stats.queries_served += 1;
        self.store
            .valid_insights()
            .filter(|i| method.is_none_or(|m| i.header.synthesis_method == m))
            .filter_map(|i| {
                self.store
                    .insight_content(i)
                    .map(|c| (c, i.header.confidence.base))
            })
            .collect()
    }

    /// Get the user profile snapshot.
    pub fn profile(&self) -> &ProfileSnapshot {
        self.store.profile()
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Full Pipeline Pass — Ingest → Extract → Synthesize
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Process a conversation turn through the full pipeline.
    /// This is the main entry point for new content.
    pub fn process_turn(
        &mut self,
        content: &str,
        is_user: bool,
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> PipelineStageResults {
        let kind = if is_user {
            FragmentKind::UserMessage
        } else {
            FragmentKind::AssistantMessage
        };

        // Stage 1: Ingest
        let ingest_ok = self
            .ingest(kind, content, session_id, turn, timestamp)
            .is_ok();

        // Stage 2: Extract
        let atoms_extracted = self.extract(timestamp);

        // Stage 3: Synthesize (if threshold met)
        let synthesis = self.maybe_synthesize(timestamp);

        PipelineStageResults {
            ingested: ingest_ok,
            atoms_extracted,
            synthesis_result: synthesis,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Telemetry & Diagnostics
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }
    pub fn store(&self) -> &InMemoryStore {
        &self.store
    }

    pub fn knowledge_summary(&self) -> KnowledgeSummary {
        let profile = self.store.profile();
        KnowledgeSummary {
            total_fragments: self.store.fragment_count() as u32,
            total_atoms: self.store.atom_count() as u32,
            total_insights: self.store.insight_count() as u32,
            active_atoms: profile.active_atoms,
            profile_completeness: profile.completeness(),
            profile_sections: profile.section_count(),
        }
    }
}

impl Default for MemoryPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from processing a single turn through all stages.
#[derive(Debug)]
pub struct PipelineStageResults {
    pub ingested: bool,
    pub atoms_extracted: u32,
    pub synthesis_result: Option<SynthesisPassResult>,
}

/// High-level knowledge summary.
#[derive(Debug, Clone, Copy)]
pub struct KnowledgeSummary {
    pub total_fragments: u32,
    pub total_atoms: u32,
    pub total_insights: u32,
    pub active_atoms: u32,
    pub profile_completeness: f32,
    pub profile_sections: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_pipeline_single_turn() {
        let mut pipeline = MemoryPipeline::new();

        let result = pipeline.process_turn(
            "I am Mumo, the CEO of BIZRA. I prefer Rust for core systems.",
            true,
            1,
            1,
            1000,
        );

        assert!(result.ingested);
        assert!(result.atoms_extracted >= 2); // "I am" → Fact, "I prefer" → Preference
    }

    #[test]
    fn pipeline_accumulates_knowledge() {
        let mut pipeline = MemoryPipeline::new();

        pipeline.process_turn("I am working on BIZRA", true, 1, 1, 1000);
        pipeline.process_turn("I prefer sovereign architecture", true, 1, 2, 2000);
        pipeline.process_turn("I always work after Fajr prayer", true, 1, 3, 3000);

        let summary = pipeline.knowledge_summary();
        assert_eq!(summary.total_fragments, 3);
        assert!(summary.total_atoms >= 3);
    }

    #[test]
    fn synthesis_triggers_at_threshold() {
        let config = PipelineConfig {
            synthesis_batch_size: 3,
            ..Default::default()
        };
        let mut pipeline = MemoryPipeline::with_config(config);

        // Process turns until synthesis triggers
        pipeline.process_turn("I am Mumo", true, 1, 1, 1000);
        pipeline.process_turn("I like distributed systems", true, 1, 2, 2000);

        // Before threshold: no synthesis
        assert_eq!(pipeline.stats().synthesis_passes, 0);

        // This should push past threshold
        pipeline.process_turn("I need sovereign AI", true, 1, 3, 3000);

        // After threshold: synthesis should have run
        assert!(pipeline.stats().synthesis_passes >= 1);
        assert!(pipeline.store().insight_count() > 0);
    }

    #[test]
    fn query_facts_returns_relevant() {
        let mut pipeline = MemoryPipeline::new();

        pipeline.process_turn("I am the founder of BIZRA", true, 1, 1, 1000);
        pipeline.process_turn("I prefer building in Rust", true, 1, 2, 2000);

        let facts = pipeline.query_facts(AtomKind::Fact, 2000);
        assert!(!facts.is_empty());

        let prefs = pipeline.query_facts(AtomKind::Preference, 2000);
        assert!(!prefs.is_empty());
    }

    #[test]
    fn profile_builds_incrementally() {
        let mut pipeline = MemoryPipeline::new();

        // No sections populated initially
        assert_eq!(pipeline.profile().section_count(), 0);

        pipeline.process_turn("I am Mumo", true, 1, 1, 1000);
        pipeline.process_turn("I always review code in morning", true, 1, 2, 2000);

        // Force synthesis to update profile
        pipeline.force_synthesize(3000);

        let profile = pipeline.profile();
        assert!(profile.section_count() > 0);
        assert!(profile.total_atoms > 0);
    }

    #[test]
    fn duplicate_content_deduped() {
        let mut pipeline = MemoryPipeline::new();

        pipeline.process_turn("I am working on BIZRA", true, 1, 1, 1000);
        pipeline.process_turn("I am working on BIZRA", true, 1, 2, 2000); // duplicate

        assert_eq!(pipeline.knowledge_summary().total_fragments, 1);
    }

    #[test]
    fn assistant_messages_processed() {
        let mut pipeline = MemoryPipeline::new();

        pipeline.process_turn(
            "Based on your architecture, I recommend focusing on the hook system first",
            false,
            1,
            1,
            1000,
        );

        assert_eq!(pipeline.knowledge_summary().total_fragments, 1);
    }
}
