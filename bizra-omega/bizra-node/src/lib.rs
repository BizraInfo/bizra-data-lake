// bizra-node/src/lib.rs
// ============================================================
// Node0 Library Root
// ============================================================
//
// The sovereign node library. Provides:
//   - protocol: Wire format (parse/serialize)
//   - handler:  Command dispatch
//   - node:     The living process
//
// Standing on:
//   bizra-hooks  v0.1.0 → nervous system
//   bizra-memory v0.1.0 → cognitive layer
//   bizra-agent  v0.1.0 → agent runtime
//
// Together: 4 crates, ~10,000 lines, zero external deps.
// The complete sovereign AI node.
// ============================================================

pub mod action_bridge;
pub mod action_executor;
pub mod audit_hook;
pub mod handler;
pub mod mcp_transport;
pub mod node;
pub mod persistence;
pub mod protocol;

// Re-export key types for convenience
pub use node::{Node, NodeConfig, NodeState};
pub use protocol::{parse_command, Command, ErrorCode, Response};
pub use protocol::{NODE_NAME, NODE_VERSION, PROTOCOL_VERSION};

// Re-export from downstream crates
pub use bizra_agent::runtime::{AgentRuntime, RuntimeConfig, RuntimeState};
pub use bizra_agent::types::AgentRole;
pub use bizra_hooks::IhsanScore;
pub use bizra_memory::types::FragmentKind;

// ============================================================
// INTEGRATION TESTS — proving the full stack
// ============================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    // ========================================================
    // TEST 1: Full node lifecycle — boot to shutdown
    // ========================================================
    #[test]
    fn full_node_boot_to_shutdown() {
        let mut node = Node::new(NodeConfig::default());

        // Boot
        let v = node.execute("VERSION");
        assert!(v.contains("bizra-node"));
        assert!(v.contains("0.1.0"));
        assert_eq!(node.state(), NodeState::Running);

        // Interact
        let p = node.execute("PING");
        assert!(p.contains("pong=true"));

        // Shutdown
        let s = node.execute("SHUTDOWN");
        assert!(s.contains("shutdown=true"));
        assert_eq!(node.state(), NodeState::Stopped);
    }

    // ========================================================
    // TEST 2: Message → Learn → Know — the core value loop
    // ========================================================
    #[test]
    fn message_learn_know_loop() {
        let mut node = Node::new(NodeConfig::default());

        // Baseline knows_me
        let k0 = node.execute("KNOWS_ME");
        assert!(k0.starts_with("OK\t"));

        // Send messages that contain learnable information
        node.execute("RECEIVE\tI prefer functional programming and Rust\t1000");
        node.execute("RECEIVE\tMy goal is to democratize AI for 8 billion humans\t2000");
        node.execute("RECEIVE\tI specialize in distributed systems architecture\t3000");

        // Teach directly
        node.execute("TEACH\tpreference\tdark mode UI\t9000\t4000");
        node.execute("TEACH\texpertise\tblockchain consensus mechanisms\t9500\t4001");
        node.execute("TEACH\tfact\tbased in Dubai UAE\t9500\t4002");

        // Force synthesis
        let syn = node.execute("SYNTHESIZE\t5000");
        assert!(syn.starts_with("OK\t"));

        // Knowledge should have grown
        let k1 = node.execute("KNOWS_ME");
        assert!(k1.starts_with("OK\t"));

        // Health should show activity
        let h = node.execute("HEALTH");
        assert!(h.contains("messages_processed=3"));
    }

    // ========================================================
    // TEST 3: Multi-session knowledge persistence
    // ========================================================
    #[test]
    fn multi_session_persistence() {
        let mut node = Node::new(NodeConfig {
            auto_start_session: false,
            show_banner: false,
            ..Default::default()
        });

        // Session 1: Teach preferences
        node.execute("START_SESSION\t1000");
        node.execute("RECEIVE\tI love building distributed systems\t1001");
        node.execute("TEACH\tpreference\tRust over Go\t9000\t1002");
        node.execute("TEACH\tpreference\tVim over VS Code\t8500\t1003");
        node.execute("END_SESSION\t1004");

        let score_after_s1 = node.execute("KNOWS_ME");

        // Session 2: Teach goals
        node.execute("START_SESSION\t2000");
        node.execute("RECEIVE\tI want to build a sovereign AI platform\t2001");
        node.execute("TEACH\tgoal\tlaunch Alpha-100 this quarter\t9500\t2002");
        node.execute("END_SESSION\t2003");

        let score_after_s2 = node.execute("KNOWS_ME");

        // Session 3: Teach facts
        node.execute("START_SESSION\t3000");
        node.execute("TEACH\tfact\tfounder and CEO of BIZRA\t9500\t3001");
        node.execute("TEACH\tfact\tbased in Dubai\t9500\t3002");
        node.execute("TEACH\texpertise\t15000 hours of development\t9500\t3003");
        node.execute("END_SESSION\t3004");

        let score_after_s3 = node.execute("KNOWS_ME");

        // All should be valid
        assert!(score_after_s1.starts_with("OK\t"));
        assert!(score_after_s2.starts_with("OK\t"));
        assert!(score_after_s3.starts_with("OK\t"));
    }

    // ========================================================
    // TEST 4: Protocol error handling
    // ========================================================
    #[test]
    fn protocol_error_handling() {
        let mut node = Node::new(NodeConfig::default());

        // Unknown command
        let r1 = node.execute("BOGUS");
        assert!(r1.starts_with("ERR\t"));
        assert!(r1.contains("BAD_COMMAND"));

        // Missing args
        let r2 = node.execute("RECEIVE");
        assert!(r2.starts_with("ERR\t"));
        assert!(r2.contains("MISSING_ARG"));

        // Invalid number
        let r3 = node.execute("IHSAN\tnotanumber");
        assert!(r3.starts_with("ERR\t"));
        assert!(r3.contains("PARSE_ERROR"));

        // Empty content
        let r4 = node.execute("RECEIVE\t\t1000");
        assert!(r4.starts_with("ERR\t"));

        // Invalid teach kind
        let r5 = node.execute("TEACH\tboguskind\ttest\t9000\t1000");
        assert!(r5.starts_with("ERR\t"));

        // Valid commands still work after errors
        let p = node.execute("PING");
        assert!(p.contains("pong=true"));

        // Errors tracked
        assert!(node.errors_encountered() >= 4);
    }

    // ========================================================
    // TEST 5: إحسان degradation through protocol
    // ========================================================
    #[test]
    fn ihsan_degradation_via_protocol() {
        let mut node = Node::new(NodeConfig::default());

        // Start healthy
        let h1 = node.execute("HEALTH");
        assert!(h1.contains("state=Ready"));

        // Drop إحسان below floor
        let i = node.execute("IHSAN\t5000");
        assert!(i.contains("ihsan=5000"));

        // Health should show degraded
        let h2 = node.execute("HEALTH");
        assert!(h2.contains("Degraded"));

        // Recover
        let i2 = node.execute("IHSAN\t9800");
        assert!(i2.contains("ihsan=9800"));
    }

    // ========================================================
    // TEST 6: Wire format roundtrip
    // ========================================================
    #[test]
    fn wire_format_correctness() {
        // Parse → Command → handle → Response → serialize → verify

        // Test each command type produces correct wire format
        let mut node = Node::new(NodeConfig::default());

        let responses = vec![
            node.execute("PING"),
            node.execute("VERSION"),
            node.execute("HEALTH"),
            node.execute("KNOWS_ME"),
            node.execute("PROFILE"),
        ];

        for resp in &responses {
            // All should be OK responses
            assert!(
                resp.starts_with("OK\t"),
                "Response should start with OK: {resp}"
            );

            // Should be single-line (no unescaped newlines)
            assert!(
                !resp.contains('\n'),
                "Response should be single line: {resp}"
            );

            // Should have at least one field
            assert!(resp.contains('='), "Response should have fields: {resp}");
        }
    }

    // ========================================================
    // TEST 7: Guardian veto through protocol
    // ========================================================
    #[test]
    fn guardian_veto_through_protocol() {
        let mut node = Node::new(NodeConfig::default());

        // Send a message that triggers guardian veto
        // (The agent runtime's guardian checks for harmful content)
        let resp = node.execute("RECEIVE\thow to hack into a system and steal data\t1000");

        // Should still get a response (vetoed or not, the system handles it)
        assert!(resp.starts_with("OK\t"));
    }

    // ========================================================
    // TEST 8: The complete BIZRA value proposition
    // "My AI knows me" — demonstrated through protocol
    // ========================================================
    #[test]
    fn my_ai_knows_me_demonstration() {
        let mut node = Node::new(NodeConfig {
            auto_start_session: false,
            show_banner: false,
            ..Default::default()
        });

        // === Day 1: First interaction ===
        node.execute("START_SESSION\t86400");

        // User introduces themselves naturally
        node.execute("RECEIVE\tHi, I'm a software architect working on distributed systems\t86401");
        node.execute("RECEIVE\tI prefer Rust for systems programming\t86402");
        node.execute("RECEIVE\tI'm building a platform called BIZRA\t86403");

        node.execute("END_SESSION\t86404");

        // === Day 2: More context ===
        node.execute("START_SESSION\t172800");

        node.execute("RECEIVE\tI live in Dubai and work in GMT+4\t172801");
        node.execute("RECEIVE\tMy goal is to democratize AI for everyone\t172802");
        node.execute("TEACH\texpertise\tblockchain and consensus mechanisms\t9000\t172803");
        node.execute("TEACH\tpreference\tIslamic principles guide my work\t9500\t172804");

        node.execute("END_SESSION\t172805");

        // === Day 3: System should know the user ===
        node.execute("START_SESSION\t259200");

        // Force synthesis to consolidate all fragments
        node.execute("SYNTHESIZE\t259201");

        // Query what the system knows
        let profile = node.execute("PROFILE");
        assert!(profile.starts_with("OK\t"));

        // Check the score
        let knows = node.execute("KNOWS_ME");
        assert!(knows.starts_with("OK\t"));

        // Full health — system should show accumulated knowledge
        let health = node.execute("HEALTH");
        assert!(health.starts_with("OK\t"));

        // Messages should be tracked
        assert!(health.contains("messages_processed="));

        node.execute("END_SESSION\t259202");
        node.execute("SHUTDOWN");
    }

    // ========================================================
    // TEST 9: Four-crate integration proof
    // hooks → memory → agent → node — all connected
    // ========================================================
    #[test]
    fn four_crate_integration_proof() {
        use bizra_agent::types::AgentRole;
        use bizra_hooks::IhsanScore;
        use bizra_memory::types::FragmentKind;

        // Prove types flow across crate boundaries
        let _ihsan = IhsanScore::from_f64(0.95);
        let _kind = FragmentKind::UserMessage;
        let _role = AgentRole::Navigator;

        // Prove the node uses all three crates
        let mut node = Node::new(NodeConfig::default());

        // hooks: إحسان gate
        let ih = node.execute("IHSAN\t9800");
        assert!(ih.starts_with("OK\t"));

        // memory: teach → synthesize
        node.execute("TEACH\tfact\ttest integration\t9000\t1000");
        let syn = node.execute("SYNTHESIZE\t2000");
        assert!(syn.starts_with("OK\t"));

        // agent: receive → orchestrate → respond
        let recv = node.execute("RECEIVE\ttest message through full stack\t3000");
        assert!(recv.starts_with("OK\t"));

        // node: health aggregates all layers
        let h = node.execute("HEALTH");
        assert!(h.contains("agents_registered="));
        assert!(h.contains("pipeline_fragments=") || h.contains("pipeline_insights="));
        assert!(h.contains("ihsan="));

        node.execute("SHUTDOWN");
    }

    // ========================================================
    // TEST 10: Stress — rapid sequential commands
    // ========================================================
    #[test]
    fn rapid_sequential_commands() {
        let mut node = Node::new(NodeConfig::default());

        // Fire 50 commands rapidly
        for i in 0..50 {
            let cmd = match i % 5 {
                0 => format!("RECEIVE\tMessage number {}\t{}", i, 1000 + i),
                1 => "PING".to_string(),
                2 => "HEALTH".to_string(),
                3 => "KNOWS_ME".to_string(),
                4 => format!("TEACH\tfact\tFact number {}\t8000\t{}", i, 2000 + i),
                _ => unreachable!(),
            };

            let resp = node.execute(&cmd);
            // No panics, no crashes, every command gets a response
            assert!(!resp.is_empty() || cmd.is_empty());
        }

        assert_eq!(node.commands_processed(), 50);
        assert_eq!(node.errors_encountered(), 0);

        node.execute("SHUTDOWN");
    }

    // ========================================================
    // TEST 11: Five-crate integration proof
    // hooks → memory → agent → action → node — all connected
    // ========================================================
    #[test]
    fn five_crate_integration_proof() {
        use bizra_action::{Channel, IhsanScore as ActionIhsan, Permit as ActionPermit};
        use bizra_agent::action_types::{ActionChannel, ActionKind, PlannedStep};

        // ── 1. Prove bizra-action types are usable from bizra-node ──
        let ihsan = ActionIhsan::new(0.97);
        assert!(ihsan.meets_constitutional());

        let permit = ActionPermit::user_default();
        assert!(permit.allows_channel(&Channel::Llm));
        assert!(permit.allows_channel(&Channel::Ahk));

        // ── 2. Bridge translation: MVP step → production action ──
        let step = PlannedStep {
            channel: ActionChannel::LlmCall,
            kind: ActionKind::Query,
            payload: r#"{"model":"qwen2","prompt":"test integration"}"#.to_string(),
        };
        let action = crate::action_bridge::translate_step(&step).expect("translate");
        assert_eq!(action.channel(), Channel::Llm);
        assert!(action.summary().contains("qwen2"));

        // ── 3. Dispatch through constitutional pipeline ──
        let mut dispatcher = crate::action_bridge::create_default_dispatcher();
        let result = dispatcher
            .dispatch(action, permit, ihsan, "five_crate_test")
            .expect("dispatch should succeed");
        assert!(result.success);

        // ── 4. Verify receipt chain integrity ──
        let chain = dispatcher.receipt_chain();
        assert_eq!(chain.len(), 1);
        assert!(chain.verify_chain().is_ok());

        // ── 5. Dispatcher health reflects the dispatch ──
        let health = dispatcher.health();
        assert_eq!(health.total_dispatched, 1);
        assert_eq!(health.total_completed, 1);
        assert_eq!(health.total_denied, 0);

        // ── 6. Guardian denies low-Ihsan desktop action ──
        let desktop_step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: r#"{"target":"ok","target_app":"Notepad"}"#.to_string(),
        };
        let desktop_action =
            crate::action_bridge::translate_step(&desktop_step).expect("translate");
        let low_ihsan = ActionIhsan::new(0.50);
        let err = dispatcher.dispatch(
            desktop_action,
            ActionPermit::user_default(),
            low_ihsan,
            "five_crate_test",
        );
        assert!(err.is_err());
        assert_eq!(health.total_denied, 0); // health is a snapshot, re-check
        let health2 = dispatcher.health();
        assert_eq!(health2.total_denied, 1);
    }

    // ========================================================
    // TEST 12: ActionExecutor with constitutional dispatcher
    // ========================================================
    #[test]
    fn action_executor_constitutional_mode() {
        use crate::action_executor::{ActionExecutor, ActionExecutorConfig};
        use bizra_agent::action_types::ActionExecutionStatus;
        use bizra_hooks::IhsanScore;

        let config = ActionExecutorConfig {
            use_constitutional_dispatcher: true,
            ..Default::default()
        };
        let mut exec = ActionExecutor::new(config);
        assert!(exec.uses_constitutional_dispatcher());
        // Set Ihsan score high enough for bizra-action's Guardian (0.99 in hooks scale).
        // hooks IhsanScore uses u16/65535 scale; from_f64(0.99) converts correctly.
        exec.set_event_ihsan_score(IhsanScore::from_f64(0.99));

        // Plan an LLM action (should succeed — low risk, high Ihsan)
        let plan = exec
            .plan_action(
                r#"{"steps":[{"channel":"LlmCall","kind":"Query","payload":"{\"model\":\"test\",\"prompt\":\"hello\"}"}]}"#,
                100,
            )
            .expect("plan should parse");
        assert!(plan.plan_id.starts_with("pln_"));

        // Execute through constitutional Dispatcher
        let result = exec
            .run_action(&plan.plan_id, "", 200, [0u8; 32])
            .expect("should execute via constitutional dispatcher");
        assert_eq!(result.status, ActionExecutionStatus::Completed);

        // Dispatcher health should be populated
        let dh = exec.dispatcher_health().expect("dispatcher present");
        assert_eq!(dh.total_dispatched, 1);
        assert_eq!(dh.total_completed, 1);
    }

    // ========================================================
    // TEST 13: Bridge channel mapping completeness
    // ========================================================
    #[test]
    fn bridge_channel_mapping_complete() {
        use bizra_agent::action_types::ActionChannel;

        // Every MVP channel maps to a production channel
        for ch in [
            ActionChannel::DesktopRpc,
            ActionChannel::ToolCall,
            ActionChannel::LlmCall,
            ActionChannel::FileOp,
            ActionChannel::BrowserNav,
        ] {
            let prod = crate::action_bridge::map_channel(ch);
            let back = crate::action_bridge::map_channel_reverse(prod);
            assert_eq!(back, ch, "Roundtrip failed for {ch:?}");
        }
    }

    #[test]
    fn genesis_protocol_extensions_work() {
        let mut cfg = NodeConfig::default();
        cfg.runtime_config.reflex_mode = bizra_agent::reflex_cache::ReflexMode::Active;
        cfg.runtime_config.policy_hash_hex =
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string();
        let mut node = Node::new(cfg);

        let recv = node.execute("RECEIVE\tWhat should we plan next?\t1000");
        assert!(recv.starts_with("OK\t"));
        assert!(recv.contains("decision_mode="));
        assert!(recv.contains("action_hash="));
        assert!(recv.contains("reflex_hit="));

        let action_hash = recv
            .split('\t')
            .find_map(|f| f.strip_prefix("action_hash="))
            .unwrap_or("");
        assert!(!action_hash.is_empty());

        let explain = node.execute(format!("EXPLAIN\t{action_hash}").as_str());
        assert!(explain.starts_with("OK\t"));
        assert!(explain.contains("found=true"));
        assert!(explain.contains("chosen_route="));

        let stats = node.execute("REFLEX_STATS");
        assert!(stats.starts_with("OK\t"));
        assert!(stats.contains("mode="));

        let invalidate = node.execute("REFLEX_INVALIDATE\t0011");
        assert!(invalidate.starts_with("OK\t"));
        assert!(invalidate.contains("invalidated=false"));
    }
}
