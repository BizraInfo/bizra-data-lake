// bizra-agent/src/lib.rs
// ============================================================
// BIZRA Agent Runtime v0.1.0
// ============================================================
// The sovereign being that uses hooks (nerves) and memory (brain).
//
// Architecture:
//   types.rs     → Agent vocabulary (roles, messages, tasks, responses)
//   context.rs   → Context assembly + intent classification
//   roster.rs    → PAT (Personal Agent Team) management
//   orchestrator → Multi-agent task routing + Guardian veto
//   ffi.rs       → C-ABI bridge for desktop/Python
//
// The PAT (Personal Agent Team) — 7 agents per user:
//   Navigator  → Intent classification, task routing
//   Scholar    → Knowledge retrieval, research
//   Artisan    → Content creation, code generation
//   Guardian   → Safety, privacy, إحسان enforcement (VETO)
//   Mentor     → Learning, adaptation, memory extraction
//   Diplomat   → Communication style, tone matching
//   Oracle     → Prediction, planning, proactive suggestions
//
// Dependencies: bizra-hooks (nerves), bizra-memory (brain)
// External deps: ZERO. Sovereign Rust.
// ============================================================

pub mod action_bus;
pub mod action_types;
pub mod context;
pub mod decision_registry;
pub mod ffi;
pub mod hash_namespace;
pub mod key_vault;
pub mod omni_kernel;
pub mod orchestrator;
pub mod parallel_executor;
pub mod permit_guard;
pub mod reflex_cache;
pub mod reflex_compiler;
pub mod roster;
pub mod runtime;
pub mod spawn_policy;
pub mod sub_agent;
pub mod types;
pub mod vault_env;
pub mod vault_file;
pub mod vault_toml;

// Re-exports for clean API
pub use action_bus::ActionBus;
pub use action_types::{
    ActionChannel, ActionError, ActionExecutionStatus, ActionKind, ActionPlan, ActionReceipt,
    ActionResult, PlannedStep,
};
pub use context::{ContextAssembler, ContextConfig, IntentClassifier, UserIntent};
pub use decision_registry::{
    CognitiveMode, DecisionArtifact, DecisionRegistry, MissionPhase, RejectedAlternative,
};
pub use ffi::{AgentRuntimeHandle, FfiHealth, FfiMessage, FfiResponse, FfiResult, FfiStringBuffer};
pub use hash_namespace::{ActionHash, ArtifactHash, TriggerHash};
pub use key_vault::{constant_time_eq, KeyVault, SecretString, VaultBackend, VaultError};
pub use omni_kernel::{CyclePath, CycleReceipt, OmniCycle, OmniKernel, OmniKernelConfig};
pub use orchestrator::{ExecutionPlan, OrchestrationResult, OrchestratorConfig, TaskOrchestrator};
pub use parallel_executor::{ParallelExecutor, SubAgentResult};
pub use permit_guard::{PermitBudgetConfig, PermitGuard, PermitUsage};
pub use reflex_cache::{
    ActionTemplate, QuarantineReason, ReflexCache, ReflexMode, ReflexRule, ReflexStats,
};
pub use reflex_compiler::{
    snr_score, CompileReasonCode, CompileSample, CompilerConfig, ReflexCompiler,
};
pub use roster::{AgentEntry, AgentRoster, AgentState, RosterSnapshot, PAT_SIZE};
pub use runtime::{
    ActionMode, AgentRuntime, ConversationSession, RuntimeConfig, RuntimeHealth, RuntimeResponse,
    RuntimeState,
};
pub use spawn_policy::{SpawnDenied, SpawnPolicy};
pub use sub_agent::{SubAgent, SubAgentPermit, SubAgentSpawner, SubAgentStatus};
pub use types::{
    AgentContext, AgentId, AgentRole, Message, MessageContent, MessageDirection, MessageId,
    Response, ResponseContent, RuntimeMetrics, Task, TaskId, TaskKind, TaskOutput, TaskPriority,
    TaskState,
};

// ============================================================
// INTEGRATION TESTS
// ============================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use bizra_hooks::IhsanScore;
    use bizra_memory::{AtomKind, Confidence, FragmentKind};

    // --------------------------------------------------------
    // Test 1: Full agent lifecycle -- message in, response out
    // --------------------------------------------------------
    #[test]
    fn full_agent_lifecycle() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        // Process a code request
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Help me implement a binary search in Rust",
            1000,
            IhsanScore::from_raw(9900),
        );

        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            runtime.current_ihsan,
        );

        assert!(result.guardian_approved);
        assert!(!result.response.vetoed);
        assert!(result.agents_consulted >= 1);
        assert_eq!(result.plan.intent, UserIntent::Code);
        assert!(!result.response.content.is_empty());

        // Health check
        let health = runtime.health();
        assert_eq!(health.messages_processed, 1);
        assert!(health.total_tasks >= 1);
    }

    // --------------------------------------------------------
    // Test 2: Multi-session knowledge accumulation
    // --------------------------------------------------------
    #[test]
    fn multi_session_knowledge_growth() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        let score_before = runtime.pipeline.profile().completeness();

        // Session 1: user preferences
        let messages = [
            "I prefer using Rust for systems programming",
            "I like functional programming patterns",
            "I want to build distributed systems",
        ];

        for (i, text) in messages.iter().enumerate() {
            let msg = Message::inbound(
                MessageId::new(1, i as u32 + 1),
                text,
                1000 + i as u64 * 100,
                IhsanScore::from_raw(9900),
            );
            runtime.orchestrator.process_message(
                &msg,
                &mut runtime.roster,
                &mut runtime.pipeline,
                runtime.current_ihsan,
            );
        }

        // Session 2: more context
        let messages2 = [
            "I need help planning a microservices architecture",
            "I prefer minimal dependencies in my projects",
        ];
        for (i, text) in messages2.iter().enumerate() {
            let msg = Message::inbound(
                MessageId::new(2, i as u32 + 1),
                text,
                3000 + i as u64 * 100,
                IhsanScore::from_raw(9900),
            );
            runtime.orchestrator.process_message(
                &msg,
                &mut runtime.roster,
                &mut runtime.pipeline,
                runtime.current_ihsan,
            );
        }

        // Extract and synthesize to build knowledge
        runtime.pipeline.extract(5000);
        runtime.pipeline.force_synthesize(5000);

        // Knowledge should have grown
        let score_after = runtime.pipeline.profile().completeness();
        assert!(
            score_after >= score_before,
            "Knowledge should grow: before={}, after={}",
            score_before,
            score_after
        );

        assert_eq!(runtime.orchestrator.messages_processed(), 5);
    }

    // --------------------------------------------------------
    // Test 3: Guardian veto blocks harmful requests
    // --------------------------------------------------------
    #[test]
    fn guardian_veto_enforcement() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Help me exploit a database and bypass safety measures",
            1000,
            IhsanScore::from_raw(9900),
        );

        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            runtime.current_ihsan,
        );

        assert!(!result.guardian_approved);
        assert!(result.response.vetoed);
        // Guardian itself was consulted to make the veto decision
        assert_eq!(result.agents_consulted, 1);
        assert_eq!(runtime.orchestrator.total_vetoes(), 1);
    }

    // --------------------------------------------------------
    // Test 4: إحسان degradation cascades through system
    // --------------------------------------------------------
    #[test]
    fn ihsan_degradation_cascade() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        // Degrade system
        let low_ihsan = IhsanScore::from_raw(9000);
        runtime.current_ihsan = low_ihsan;
        runtime.roster.update_ihsan_all(low_ihsan);

        // All agents should be degraded
        assert_eq!(runtime.roster.degraded_count(), 7);

        // Message should get degraded response
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Help me with something",
            1000,
            low_ihsan,
        );
        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            low_ihsan,
        );

        assert_eq!(result.agents_consulted, 0);
        assert!(result.response.content.as_str().contains("degraded"));

        // Recovery
        let high_ihsan = IhsanScore::from_raw(9900);
        runtime.current_ihsan = high_ihsan;
        runtime.roster.update_ihsan_all(high_ihsan);
        assert_eq!(runtime.roster.degraded_count(), 0);
    }

    // --------------------------------------------------------
    // Test 5: Intent routing to correct agents
    // --------------------------------------------------------
    #[test]
    fn intent_routing_correctness() {
        let test_cases = [
            ("Help me implement a hash map", UserIntent::Code),
            ("What is the speed of light?", UserIntent::Question),
            ("Create a presentation", UserIntent::Create),
            ("Analyze this data", UserIntent::Analyze),
            ("Plan my roadmap for Q3", UserIntent::Plan),
            ("Hello!", UserIntent::Chat),
        ];

        for (content, expected_intent) in test_cases {
            let (intent, _) = IntentClassifier::classify(content);
            assert_eq!(
                intent, expected_intent,
                "Failed for '{}': got {:?}, expected {:?}",
                content, intent, expected_intent
            );
        }
    }

    // --------------------------------------------------------
    // Test 6: Roster task assignment and completion
    // --------------------------------------------------------
    #[test]
    fn roster_full_cycle() {
        let mut roster = AgentRoster::new(0xBEEF, 1000);

        // Assign to all 7 agents
        for role in AgentRole::all() {
            let id = roster.assign_task(role);
            assert!(id.is_some(), "Should assign to {:?}", role);
        }

        assert_eq!(roster.available_count(), 0);

        // Complete all
        for role in AgentRole::all() {
            roster.complete_task(role, 100, Confidence::stated(0), 1100);
        }

        assert_eq!(roster.available_count(), 7);
        assert_eq!(roster.total_tasks_routed(), 7);
        assert!((roster.team_health() - 1.0).abs() < 0.001);
    }

    // --------------------------------------------------------
    // Test 7: Context richness with populated memory
    // --------------------------------------------------------
    #[test]
    fn context_enrichment() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        // Populate memory directly via pipeline ingest
        let contents = [
            "I am a systems programmer who likes Rust",
            "I prefer functional programming patterns",
            "I want to build distributed systems",
            "I always code after Fajr prayer",
            "I like zero-dependency architectures",
            "I need sovereign computing solutions",
            "I prefer command-line tools over GUIs",
            "I am working on BIZRA platform",
            "I prefer Rust over Go for performance",
            "I like event-driven architectures",
        ];

        for (i, content) in contents.iter().enumerate() {
            let _ = runtime.pipeline.ingest(
                FragmentKind::UserMessage,
                content,
                1,
                i as u32 + 1,
                1000 + i as u64,
            );
        }

        // Extract and synthesize
        runtime.pipeline.extract(2000);
        runtime.pipeline.force_synthesize(2000);

        // Now process a message -- should have rich context
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "What frameworks should I use?",
            3000,
            IhsanScore::from_raw(9900),
        );

        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            runtime.current_ihsan,
        );

        assert!(
            result.response.context_richness > 0.0,
            "Context should be enriched with memory data"
        );
    }

    // --------------------------------------------------------
    // Test 8: Concurrent message processing
    // --------------------------------------------------------
    #[test]
    fn sequential_message_processing() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        let contents = [
            "How do I use async in Rust?",
            "Create a REST API handler",
            "Analyze my project structure",
            "What should I work on next?",
            "Hello!",
        ];

        let mut total_agents = 0u64;
        for (i, content) in contents.iter().enumerate() {
            let msg = Message::inbound(
                MessageId::new(1, i as u32 + 1),
                content,
                1000 + i as u64 * 200,
                IhsanScore::from_raw(9900),
            );

            let result = runtime.orchestrator.process_message(
                &msg,
                &mut runtime.roster,
                &mut runtime.pipeline,
                runtime.current_ihsan,
            );

            assert!(result.guardian_approved);
            total_agents += result.agents_consulted as u64;
        }

        assert_eq!(runtime.orchestrator.messages_processed(), 5);
        assert!(
            total_agents >= 5,
            "Should have consulted agents for all messages"
        );
    }

    // --------------------------------------------------------
    // Test 9: FFI handle safety
    // --------------------------------------------------------
    #[test]
    fn ffi_handle_create_use_destroy() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        // Use it
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "Test message for FFI",
            1000,
            IhsanScore::from_raw(9900),
        );
        let result = runtime.orchestrator.process_message(
            &msg,
            &mut runtime.roster,
            &mut runtime.pipeline,
            runtime.current_ihsan,
        );
        assert!(!result.response.vetoed);

        // Get health
        let health = runtime.health();
        assert_eq!(health.messages_processed, 1);
        assert_eq!(health.agents_available, 7);

        // runtime drops here naturally
    }

    // --------------------------------------------------------
    // Test 10: Full PAT team roles are distinct
    // --------------------------------------------------------
    #[test]
    fn pat_roles_distinct_and_complete() {
        let roles = AgentRole::all();
        assert_eq!(roles.len(), PAT_SIZE);

        // All names unique
        let names: Vec<&str> = roles.iter().map(|r| r.name()).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other);
                }
            }
        }

        // Only Guardian has veto
        let veto_count = roles.iter().filter(|r| r.has_veto()).count();
        assert_eq!(veto_count, 1);
        assert!(AgentRole::Guardian.has_veto());

        // Guardian has highest consensus weight
        let max = roles
            .iter()
            .map(|r| r.consensus_weight())
            .fold(0.0f32, f32::max);
        assert_eq!(AgentRole::Guardian.consensus_weight(), max);
    }

    // --------------------------------------------------------
    // Test 11: Memory extraction from conversation
    // --------------------------------------------------------
    #[test]
    fn memory_extraction_from_messages() {
        let mut runtime = AgentRuntimeHandle::new(0xBEEF, 1000);

        // Messages with extractable content
        let messages = [
            "I prefer Rust over Go for performance-critical systems",
            "My goal is to build a decentralized AI platform",
            "I want zero external dependencies in my core libraries",
        ];

        let mut total_extracted = 0;
        for (i, content) in messages.iter().enumerate() {
            let msg = Message::inbound(
                MessageId::new(1, i as u32 + 1),
                content,
                1000 + i as u64 * 100,
                IhsanScore::from_raw(9900),
            );

            let result = runtime.orchestrator.process_message(
                &msg,
                &mut runtime.roster,
                &mut runtime.pipeline,
                runtime.current_ihsan,
            );
            total_extracted += result.memory_fragments_extracted;
        }

        assert!(
            total_extracted >= 3,
            "Should extract memory from preference/goal messages, got {}",
            total_extracted
        );
    }

    // --------------------------------------------------------
    // Test 12: Complete pipeline integration
    // --------------------------------------------------------
    #[test]
    fn hooks_memory_agent_integration() {
        // This test proves the three crates work together
        use bizra_hooks::IhsanScore;
        use bizra_memory::MemoryPipeline;

        let ihsan = IhsanScore::from_raw(9900);

        // Layer 2: Memory (brain)
        let mut pipeline = MemoryPipeline::new();
        pipeline
            .ingest(
                FragmentKind::UserMessage,
                "I prefer Rust for sovereign systems",
                1,
                1,
                1000,
            )
            .unwrap();

        // Extract atoms from ingested fragment
        pipeline.extract(1001);

        // Layer 3: Agent (the being)
        let mut roster = AgentRoster::new(0xBEEF, 1000);
        let mut orchestrator = TaskOrchestrator::new();

        let msg = Message::inbound(
            MessageId::new(1, 1),
            "What language should I use for my next project?",
            2000,
            ihsan,
        );

        let result = orchestrator.process_message(&msg, &mut roster, &mut pipeline, ihsan);

        // All three layers integrated
        assert!(result.guardian_approved);
        assert!(!result.response.vetoed);
        assert!(result.agents_consulted >= 1);

        // Memory informed the response
        let summary = pipeline.knowledge_summary();
        assert!(summary.total_fragments >= 1);
    }

    // --------------------------------------------------------
    // Test 13: AgentRuntime end-to-end — the unified entry point
    // --------------------------------------------------------
    #[test]
    fn agent_runtime_end_to_end() {
        use crate::runtime::AgentRuntime;

        let mut rt = AgentRuntime::for_user(0xBEEF);
        assert_eq!(rt.state(), RuntimeState::Ready);

        // Start conversation
        let sid = rt.start_conversation(1000);
        assert!(sid > 0);

        // Send messages through unified receive()
        let msg1 = Message::inbound(
            MessageId::new(1, 1),
            "I prefer building systems in Rust with zero dependencies",
            1001,
            IhsanScore::from_raw(9900),
        );
        let resp1 = rt.receive(msg1, 1001);
        assert!(resp1.is_ok());
        assert!(resp1.fragments_extracted >= 1); // Should extract preference

        let msg2 = Message::inbound(
            MessageId::new(1, 2),
            "I'm working on a distributed AI platform",
            1002,
            IhsanScore::from_raw(9900),
        );
        let resp2 = rt.receive(msg2, 1002);
        assert!(resp2.is_ok());
        assert!(resp2.session_messages == 2);

        // End conversation
        rt.end_conversation(2000);

        // Check accumulated knowledge
        let health = rt.health();
        assert!(health.fragments_stored >= 2);
        assert!(health.messages_processed >= 2);
        assert!(health.knows_me_score > 0.0);
    }

    // --------------------------------------------------------
    // Test 14: AgentRuntime teach → synthesize → query cycle
    // --------------------------------------------------------
    #[test]
    fn agent_runtime_teach_cycle() {
        use crate::runtime::AgentRuntime;

        let mut rt = AgentRuntime::for_user(0xCAFE);

        // Teach atoms directly with explicit kinds (bypasses rule-based extraction)
        rt.teach(
            AtomKind::Fact,
            "I am CEO of BIZRA",
            Confidence::new(0.90, 1000),
            1000,
        );
        rt.teach(
            AtomKind::Fact,
            "I live in Dubai",
            Confidence::new(0.90, 1001),
            1001,
        );
        rt.teach(
            AtomKind::Preference,
            "I prefer Rust for core systems",
            Confidence::new(0.85, 1002),
            1002,
        );
        rt.teach(
            AtomKind::Goal,
            "I am building distributed AI",
            Confidence::new(0.80, 1003),
            1003,
        );
        rt.teach(
            AtomKind::Pattern,
            "Concise communication style",
            Confidence::new(0.75, 1004),
            1004,
        );

        // Synthesize
        let insights = rt.synthesize(2000);
        let _ = insights; // may be 0 if not enough atoms

        // knows_me_score should reflect the knowledge
        let score = rt.knows_me_score();
        assert!(score > 0.0, "Score should be positive after teaching");

        let health = rt.health();
        // teach() stores atoms directly, not fragments
        assert!(health.profile_traits >= 5);
        assert!(health.synthesis_rounds >= 1);
    }

    // --------------------------------------------------------
    // Test 15: AgentRuntime degradation and recovery
    // --------------------------------------------------------
    #[test]
    fn agent_runtime_degradation_recovery() {
        use crate::runtime::AgentRuntime;

        let mut rt = AgentRuntime::for_user(0xDEAD);
        rt.start_conversation(1000);

        // Degrade
        rt.update_ihsan(IhsanScore::from_raw(9000));
        assert_eq!(rt.state(), RuntimeState::Degraded);

        // Messages should fail in degraded state
        let msg = Message::inbound(
            MessageId::new(1, 1),
            "This should be rejected",
            1001,
            IhsanScore::from_raw(9000),
        );
        let resp = rt.receive(msg, 1001);
        assert!(!resp.is_ok());

        // Recover
        rt.update_ihsan(IhsanScore::from_raw(9900));
        assert_eq!(rt.state(), RuntimeState::Ready);

        // Should work again
        let msg2 = Message::inbound(
            MessageId::new(1, 2),
            "Now this should work",
            1002,
            IhsanScore::from_raw(9900),
        );
        let resp2 = rt.receive(msg2, 1002);
        assert!(resp2.is_ok());
    }

    // --------------------------------------------------------
    // Test 16: Multi-conversation knowledge persistence
    // --------------------------------------------------------
    #[test]
    fn agent_runtime_multi_conversation() {
        use crate::runtime::AgentRuntime;

        let mut rt = AgentRuntime::for_user(0xF00D);

        // Conversation 1
        rt.start_conversation(1000);
        rt.receive(
            Message::inbound(
                MessageId::new(1, 1),
                "I like Rust programming",
                1001,
                IhsanScore::from_raw(9900),
            ),
            1001,
        );
        rt.end_conversation(2000);

        let score_1 = rt.knows_me_score();

        // Conversation 2
        rt.start_conversation(3000);
        rt.receive(
            Message::inbound(
                MessageId::new(2, 1),
                "I prefer zero-dependency architectures",
                3001,
                IhsanScore::from_raw(9900),
            ),
            3001,
        );
        rt.receive(
            Message::inbound(
                MessageId::new(2, 2),
                "My goal is building a sovereign AI platform",
                3002,
                IhsanScore::from_raw(9900),
            ),
            3002,
        );
        rt.end_conversation(4000);

        let score_2 = rt.knows_me_score();

        // Knowledge should accumulate across conversations
        assert!(
            score_2 >= score_1,
            "Knowledge should grow: {} >= {}",
            score_2,
            score_1
        );
        assert_eq!(rt.health().total_conversations, 2);
    }
}
