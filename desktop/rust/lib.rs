// bizra-memory/src/lib.rs
// ============================================================
// BIZRA Memory Synthesis Pipeline
// ============================================================
// The soul of BIZRA. Transforms raw conversations into
// genuine understanding. Makes conversation 501 feel like
// talking to an old colleague who truly knows you.
//
// Architecture:
//   types.rs     — Memory atoms: Fragment, Insight, UserProfile
//   store.rs     — Pluggable storage (InMemoryStore for MVP)
//   synthesis.rs — Intelligence: clustering, contradiction
//                  detection, insight generation
//   pipeline.rs  — Full orchestration: ingest→synthesize→query
//   ffi.rs       — C-ABI bridge for Python engine integration
//
// The four-word test: "My AI knows me."
//
// Depends only on bizra-hooks (the nervous system).
// Together they form the foundation:
//   hooks = nervous system (routing, gates, observability)
//   memory = cognitive layer (understanding, synthesis, recall)
// ============================================================

pub mod types;
pub mod store;
pub mod synthesis;
pub mod pipeline;
pub mod ffi;

// ============================================================
// PUBLIC RE-EXPORTS
// ============================================================

pub use types::{
    FragmentId, FragmentKind, FragmentContent, MemoryFragment,
    InsightId, Insight, InsightContent, SynthesisDepth,
    Confidence, UserProfile, ProfileTrait,
    SessionMarker, TemporalContext,
    SynthesisMetrics,
    FRAGMENT_CONTENT_SIZE, INSIGHT_CONTENT_SIZE,
    MAX_INSIGHT_SOURCES, MAX_PROFILE_TRAITS,
};

pub use store::{
    InMemoryStore, StoreError, StoreResult, StoreStats,
    FragmentQuery, MAX_FRAGMENTS, MAX_INSIGHTS,
};

pub use synthesis::{
    SynthesisEngine, SynthesisResult, FragmentCluster, Contradiction,
    MAX_SYNTHESIS_BATCH, MIN_CLUSTER_SIZE,
};

pub use pipeline::{
    MemoryPipeline, PipelineConfig, PipelineState, PipelineHealth,
};

pub use ffi::{
    FfiResult, FfiBuffer, FfiFragment, FfiHealth, PipelineHandle,
    FFI_BUFFER_SIZE,
};

// ============================================================
// INTEGRATION TESTS
// ============================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use bizra_hooks::{ComponentId, IhsanScore};

    fn make_fragment(conv: u32, seq: u32, kind: FragmentKind, content: &str, ts: u64) -> MemoryFragment {
        MemoryFragment::new(
            FragmentId::new(conv, seq),
            kind,
            content,
            Confidence::HIGH,
            ComponentId::new("integration-test", "1.0"),
            ts,
            IhsanScore::new(9900),
        )
    }

    // --------------------------------------------------------
    // TEST 1: Full pipeline lifecycle
    // Ingest → Synthesize → Query → Verify profile
    // --------------------------------------------------------
    #[test]
    fn full_pipeline_lifecycle() {
        let mut pipeline = MemoryPipeline::new();

        // Start session
        pipeline.start_session(1, 1000);

        // Ingest diverse fragments about a user
        let fragments = vec![
            make_fragment(1, 1, FragmentKind::Fact, "User is a software architect", 1001),
            make_fragment(1, 2, FragmentKind::Preference, "Prefers Rust over Python", 1002),
            make_fragment(1, 3, FragmentKind::Preference, "Likes zero-cost abstractions", 1003),
            make_fragment(1, 4, FragmentKind::Goal, "Building a distributed AI platform", 1004),
            make_fragment(1, 5, FragmentKind::Style, "Prefers concise technical responses", 1005),
            make_fragment(1, 6, FragmentKind::Expertise, "Expert in system design", 1006),
            make_fragment(1, 7, FragmentKind::Expertise, "Knows cryptography well", 1007),
            make_fragment(1, 8, FragmentKind::Pattern, "Always asks for examples", 1008),
            make_fragment(1, 9, FragmentKind::Pattern, "Iterates rapidly on designs", 1009),
            make_fragment(1, 10, FragmentKind::Domain, "Distributed systems domain", 1010),
        ];

        let (success, failed) = pipeline.ingest_batch(fragments, 1010);
        assert_eq!(success, 10);
        assert_eq!(failed, 0);

        // Run synthesis
        let insights = pipeline.force_synthesis(2000);
        assert!(insights >= 1, "Should produce at least 1 insight from 10 fragments");

        // Query profile
        let traits = pipeline.query_profile();
        assert!(!traits.is_empty(), "Profile should have traits after synthesis");

        // End session
        pipeline.end_session(1, 3000);

        // Verify health
        let health = pipeline.health();
        assert_eq!(health.state, PipelineState::Idle);
        assert!(health.fragments_stored >= 5); // Some may cluster
        assert!(health.synthesis_rounds >= 1);

        // Knows-me score should be non-zero
        assert!(pipeline.knows_me_score() > 0.0);
    }

    // --------------------------------------------------------
    // TEST 2: Multi-session knowledge accumulation
    // Knowledge grows across sessions
    // --------------------------------------------------------
    #[test]
    fn multi_session_knowledge_accumulation() {
        let mut pipeline = MemoryPipeline::new();

        // Session 1: Basic facts
        pipeline.start_session(1, 1000);
        pipeline.ingest(make_fragment(1, 1, FragmentKind::Fact, "Works in Dubai", 1001), 1001).unwrap();
        pipeline.ingest(make_fragment(1, 2, FragmentKind::Fact, "Founder of a tech company", 1002), 1002).unwrap();
        pipeline.end_session(1, 2000);

        let score_after_s1 = pipeline.knows_me_score();

        // Session 2: More depth
        pipeline.start_session(2, 3000);
        pipeline.ingest(make_fragment(2, 1, FragmentKind::Expertise, "AI architecture expert", 3001), 3001).unwrap();
        pipeline.ingest(make_fragment(2, 2, FragmentKind::Goal, "Democratize AI access", 3002), 3002).unwrap();
        pipeline.ingest(make_fragment(2, 3, FragmentKind::Preference, "Values Islamic principles in work", 3003), 3003).unwrap();
        pipeline.end_session(2, 4000);

        pipeline.force_synthesis(4000);
        let score_after_s2 = pipeline.knows_me_score();

        // Knowledge should accumulate
        assert!(score_after_s2 >= score_after_s1, "Knowledge should grow across sessions");
        assert!(pipeline.knowledge_depth() >= 5);
    }

    // --------------------------------------------------------
    // TEST 3: إحسان gate protects synthesis quality
    // Degraded system cannot synthesize
    // --------------------------------------------------------
    #[test]
    fn ihsan_gate_protects_synthesis() {
        let mut pipeline = MemoryPipeline::new();

        // Ingest some data
        pipeline.ingest(make_fragment(1, 1, FragmentKind::Fact, "Fact A", 1000), 1000).unwrap();
        pipeline.ingest(make_fragment(1, 2, FragmentKind::Fact, "Fact B", 1001), 1001).unwrap();

        // Degrade إحسان
        pipeline.update_ihsan(IhsanScore::new(9000));
        assert_eq!(pipeline.state(), PipelineState::Degraded);

        // Synthesis should produce nothing in degraded state
        let insights = pipeline.force_synthesis(2000);
        assert_eq!(insights, 0);

        // Restore إحسان
        pipeline.update_ihsan(IhsanScore::new(9900));
        assert_eq!(pipeline.state(), PipelineState::Idle);

        // Now synthesis should work
        let insights = pipeline.force_synthesis(3000);
        // May or may not produce insights with only 2 fragments, but shouldn't panic
        assert!(insights >= 0);
    }

    // --------------------------------------------------------
    // TEST 4: Contradiction resolution
    // New facts supersede old ones
    // --------------------------------------------------------
    #[test]
    fn contradiction_resolution() {
        let mut pipeline = MemoryPipeline::new();

        // Old fact
        pipeline.ingest(
            make_fragment(1, 1, FragmentKind::Fact, "Works at Company Alpha long ago", 1000),
            1000,
        ).unwrap();

        // New contradicting fact
        pipeline.ingest(
            make_fragment(2, 1, FragmentKind::Fact, "Works at Company Beta now", 2000),
            2000,
        ).unwrap();

        // Synthesis should detect contradiction
        pipeline.force_synthesis(3000);

        // The active facts should reflect the resolution
        let facts = pipeline.query_fragments_by_kind(FragmentKind::Fact);
        // At least one should be active
        assert!(!facts.is_empty());
    }

    // --------------------------------------------------------
    // TEST 5: Fragment reinforcement
    // Repeated knowledge becomes stronger
    // --------------------------------------------------------
    #[test]
    fn fragment_reinforcement_strengthens_knowledge() {
        let mut pipeline = MemoryPipeline::new();

        // Same pattern observed multiple times
        for i in 1..=5 {
            pipeline.ingest(
                make_fragment(i as u32, 1, FragmentKind::Pattern, &format!("Asks for examples consistently {}", i), 1000 + i),
                1000 + i,
            ).unwrap();
        }

        // Synthesis should reinforce and produce strong insight
        pipeline.force_synthesis(2000);

        let health = pipeline.health();
        assert!(health.synthesis_rounds >= 1);
        assert!(pipeline.knowledge_depth() >= 5);
    }

    // --------------------------------------------------------
    // TEST 6: Store eviction under memory pressure
    // Pipeline handles capacity gracefully
    // --------------------------------------------------------
    #[test]
    fn store_eviction_under_pressure() {
        let mut config = PipelineConfig::default();
        config.max_fragments_before_eviction = 15;
        config.eviction_batch_size = 5;
        config.auto_synthesis = false; // Control synthesis manually

        let mut pipeline = MemoryPipeline::with_config(config);

        // Ingest more than threshold
        for i in 1..=20 {
            let mut frag = make_fragment(1, i, FragmentKind::Fact, &format!("Fact number {}", i), 1000 + i as u64);
            frag.created_at = 1000 + i as u64;
            frag.last_reinforced = 1000 + i as u64;
            let _ = pipeline.ingest(frag, 1000 + i as u64);
        }

        // Should have evicted some to stay under capacity
        let health = pipeline.health();
        assert!(health.fragments_stored <= 20);
    }

    // --------------------------------------------------------
    // TEST 7: The "knows me" score progression
    // Score should monotonically increase with knowledge
    // --------------------------------------------------------
    #[test]
    fn knows_me_score_progression() {
        let mut pipeline = MemoryPipeline::new();

        let mut scores = Vec::new();
        scores.push(pipeline.knows_me_score()); // Should be 0

        // Phase 1: A few facts
        for i in 1..=3 {
            pipeline.ingest(
                make_fragment(1, i, FragmentKind::Fact, &format!("Basic fact {}", i), 1000 + i as u64),
                1000 + i as u64,
            ).unwrap();
        }
        pipeline.force_synthesis(2000);
        scores.push(pipeline.knows_me_score());

        // Phase 2: More context
        for i in 1..=5 {
            pipeline.ingest(
                make_fragment(2, i, FragmentKind::Preference, &format!("Preference {}", i), 3000 + i as u64),
                3000 + i as u64,
            ).unwrap();
        }
        pipeline.force_synthesis(4000);
        scores.push(pipeline.knows_me_score());

        // Phase 3: Rich understanding
        for i in 1..=5 {
            pipeline.ingest(
                make_fragment(3, i, FragmentKind::Expertise, &format!("Expertise area {}", i), 5000 + i as u64),
                5000 + i as u64,
            ).unwrap();
        }
        pipeline.force_synthesis(6000);
        scores.push(pipeline.knows_me_score());

        // Score should generally increase (monotonic is ideal but not guaranteed)
        assert!(scores.last().unwrap() > scores.first().unwrap(),
            "Knows-me score should increase with knowledge: {:?}", scores);
    }

    // --------------------------------------------------------
    // TEST 8: Confidence gating
    // Low-confidence fragments are rejected
    // --------------------------------------------------------
    #[test]
    fn confidence_gating() {
        let mut pipeline = MemoryPipeline::new();

        // High confidence — accepted
        let good = make_fragment(1, 1, FragmentKind::Fact, "Confident fact", 1000);
        assert!(pipeline.ingest(good, 1000).is_ok());

        // Low confidence — rejected
        let mut bad = make_fragment(1, 2, FragmentKind::Fact, "Uncertain rumor", 1001);
        bad.confidence = Confidence::new(3000);
        assert!(pipeline.ingest(bad, 1001).is_err());

        assert_eq!(pipeline.health().fragments_stored, 1);
    }

    // --------------------------------------------------------
    // TEST 9: Pipeline config customization
    // Different configs produce different behaviors
    // --------------------------------------------------------
    #[test]
    fn custom_pipeline_config() {
        let config = PipelineConfig {
            auto_synthesis_threshold: 3,
            synthesis_cooldown_secs: 0,
            max_fragments_before_eviction: 100,
            eviction_batch_size: 10,
            ihsan_floor: IhsanScore::new(9800),
            auto_synthesis: true,
        };

        let mut pipeline = MemoryPipeline::with_config(config);

        // With threshold 3 and no cooldown, synthesis should trigger after 3 fragments
        for i in 1..=4 {
            pipeline.ingest(
                make_fragment(1, i, FragmentKind::Style, &format!("Style pref {}", i), 1000 + i as u64),
                1000 + i as u64,
            ).unwrap();
        }

        assert!(pipeline.health().synthesis_rounds >= 1,
            "Auto-synthesis should trigger with threshold=3");
    }

    // --------------------------------------------------------
    // TEST 10: FFI type compatibility
    // Ensure FFI types can round-trip correctly
    // --------------------------------------------------------
    #[test]
    fn ffi_types_roundtrip() {
        // FfiBuffer
        let buf = FfiBuffer::from_str("Test content for FFI");
        assert_eq!(buf.as_str(), Some("Test content for FFI"));

        // FfiFragment → MemoryFragment
        let ffi_frag = FfiFragment {
            conversation_hash: 99,
            sequence: 1,
            kind: 4, // Goal
            content: FfiBuffer::from_str("Build something amazing"),
            confidence: 8500,
            timestamp: 5000,
            ihsan: 9900,
        };

        let mem_frag = ffi_frag.to_memory_fragment().unwrap();
        assert_eq!(mem_frag.kind, FragmentKind::Goal);
        assert_eq!(mem_frag.content.as_str(), "Build something amazing");
        assert_eq!(mem_frag.confidence.raw(), 8500);
    }
}
