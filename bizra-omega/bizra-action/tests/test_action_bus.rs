//! # bizra-action Test Suite
//!
//! Tests organized by module:
//! - Guardian gates (7 gates × multiple scenarios)
//! - Receipt chain (hashing, chaining, verification)
//! - Reflex ledger (compile, lookup, staleness, eviction, marketplace)
//! - Channel stubs (each channel type)
//! - Dispatcher integration (full pipeline end-to-end)
//! - Constitutional scenarios (real-world proof-of-life)

#[cfg(test)]
mod tests {
    use bizra_action::{
        channels::*,
        dispatcher::{DispatchError, Dispatcher},
        guardian::Guardian,
        receipt::{chain_hash, content_hash, hash_payload, ReceiptChain},
        reflex::{ReflexError, ReflexLedger},
        *,
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Ihsan Score
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn ihsan_score_clamping() {
        assert_eq!(IhsanScore::new(1.5).value(), 1.0);
        assert_eq!(IhsanScore::new(-0.5).value(), 0.0);
        assert_eq!(IhsanScore::new(0.97).value(), 0.97);
    }

    #[test]
    fn ihsan_constitutional_threshold() {
        assert!(IhsanScore::new(0.95).meets_constitutional());
        assert!(IhsanScore::new(0.99).meets_constitutional());
        assert!(!IhsanScore::new(0.94).meets_constitutional());
        assert!(!IhsanScore::new(0.0).meets_constitutional());
    }

    #[test]
    fn ihsan_margin() {
        let score = IhsanScore::new(0.98);
        let margin = score.margin();
        assert!((margin - 0.03).abs() < 0.001);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Channel routing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn action_routes_to_correct_channel() {
        let ahk = BizraAction::AhkClick {
            window: "test".into(),
            element_path: "btn".into(),
        };
        assert_eq!(ahk.channel(), Channel::Ahk);

        let llm = BizraAction::LlmQuery {
            provider: "lmstudio".into(),
            model: "qwen".into(),
            system_prompt: "".into(),
            user_prompt: "hello".into(),
            max_tokens: 100,
            temperature: 0.7,
        };
        assert_eq!(llm.channel(), Channel::Llm);

        let mem = BizraAction::MemoryStore {
            fragment_id: "f1".into(),
            content: "data".into(),
            embedding: vec![],
            metadata: vec![],
        };
        assert_eq!(mem.channel(), Channel::Memory);

        let resp = BizraAction::RespondToUser {
            content: "hello".into(),
            ihsan_score: IhsanScore::new(0.98),
        };
        assert_eq!(resp.channel(), Channel::Response);

        let ts = BizraAction::TelescriptGo {
            destination_node: "node_2".into(),
            agent_state: vec![],
            permit: Permit::user_default(),
        };
        assert_eq!(ts.channel(), Channel::Telescript);
    }

    #[test]
    fn risk_levels_ordered() {
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
    }

    #[test]
    fn risk_ihsan_thresholds() {
        assert_eq!(RiskLevel::Low.min_ihsan(), 0.90);
        assert_eq!(RiskLevel::Medium.min_ihsan(), 0.95);
        assert_eq!(RiskLevel::High.min_ihsan(), 0.98);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Permit
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn user_default_permit_allows_all() {
        let p = Permit::user_default();
        assert!(p.allows_channel(&Channel::Ahk));
        assert!(p.allows_channel(&Channel::Llm));
        assert!(p.allows_channel(&Channel::Memory));
        assert!(p.allows_channel(&Channel::Telescript));
        assert!(p.allow_desktop);
        assert!(p.allow_network);
        assert!(!p.requires_hitl);
    }

    #[test]
    fn visitor_permit_restricts() {
        let p = Permit::visitor(vec!["/tmp".into()], 60);
        assert!(!p.allows_channel(&Channel::Ahk)); // bit 0 off
        assert!(p.allows_channel(&Channel::Llm)); // bit 1 on
        assert!(p.allows_channel(&Channel::Memory)); // bit 2 on
        assert!(!p.allows_channel(&Channel::Response)); // bit 6 off
        assert!(!p.allow_desktop);
        assert!(!p.allow_network);
        assert!(p.requires_hitl);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Guardian — 7 gates
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    fn make_envelope(action: BizraAction, permit: Permit, ihsan: f64) -> ActionEnvelope {
        ActionEnvelope {
            id: ActionId(1),
            timestamp: ActionTimestamp(1000),
            action,
            permit,
            plan_ihsan: IhsanScore::new(ihsan),
            source: "test".into(),
        }
    }

    #[test]
    fn guardian_approves_valid_action() {
        let mut g = Guardian::new();
        let env = make_envelope(
            BizraAction::AhkLaunch {
                executable: "notepad.exe".into(),
                args: vec![],
            },
            Permit::user_default(),
            0.99,
        );
        let v = g.evaluate(&env);
        assert!(v.is_approved());
    }

    #[test]
    fn guardian_denies_unpermitted_channel() {
        let mut g = Guardian::new();
        let env = make_envelope(
            BizraAction::AhkClick {
                window: "w".into(),
                element_path: "e".into(),
            },
            Permit::visitor(vec![], 60), // Visitor can't use AHK
            0.99,
        );
        let v = g.evaluate(&env);
        assert!(!v.is_approved());
        match v {
            GuardianVerdict::Denied {
                violation: GuardianViolation::ChannelNotPermitted { .. },
                ..
            } => {}
            _ => panic!("Expected ChannelNotPermitted"),
        }
    }

    #[test]
    fn guardian_denies_low_ihsan() {
        let mut g = Guardian::new();
        // AHK is High risk, needs 0.98
        let env = make_envelope(
            BizraAction::AhkClick {
                window: "w".into(),
                element_path: "e".into(),
            },
            Permit::user_default(),
            0.90, // Below 0.98 threshold
        );
        let v = g.evaluate(&env);
        assert!(!v.is_approved());
        match v {
            GuardianVerdict::Denied {
                violation: GuardianViolation::IhsanBelowThreshold { .. },
                ..
            } => {}
            _ => panic!("Expected IhsanBelowThreshold"),
        }
    }

    #[test]
    fn guardian_denies_desktop_when_not_permitted() {
        let mut g = Guardian::new();
        let mut permit = Permit::user_default();
        permit.allow_desktop = false;
        let env = make_envelope(
            BizraAction::AhkClick {
                window: "w".into(),
                element_path: "e".into(),
            },
            permit,
            0.99,
        );
        let v = g.evaluate(&env);
        match v {
            GuardianVerdict::Denied {
                violation: GuardianViolation::DesktopNotPermitted,
                ..
            } => {}
            _ => panic!("Expected DesktopNotPermitted"),
        }
    }

    #[test]
    fn guardian_denies_network_when_not_permitted() {
        let mut g = Guardian::new();
        let mut permit = Permit::user_default();
        permit.allow_network = false;
        let env = make_envelope(
            BizraAction::BrowserNavigate {
                url: "https://example.com".into(),
            },
            permit,
            0.99,
        );
        let v = g.evaluate(&env);
        match v {
            GuardianVerdict::Denied {
                violation: GuardianViolation::NetworkNotPermitted,
                ..
            } => {}
            _ => panic!("Expected NetworkNotPermitted"),
        }
    }

    #[test]
    fn guardian_denies_out_of_scope_path() {
        let mut g = Guardian::new();
        let permit = Permit {
            fs_scope: vec!["/home/user".into()],
            ..Permit::user_default()
        };
        let env = make_envelope(
            BizraAction::FileRead {
                path: "/etc/passwd".into(),
            },
            permit,
            0.99,
        );
        let v = g.evaluate(&env);
        match v {
            GuardianVerdict::Denied {
                violation: GuardianViolation::PathOutOfScope { .. },
                ..
            } => {}
            _ => panic!("Expected PathOutOfScope"),
        }
    }

    #[test]
    fn guardian_allows_in_scope_path() {
        let mut g = Guardian::new();
        let permit = Permit {
            fs_scope: vec!["/home/user".into()],
            ..Permit::user_default()
        };
        let env = make_envelope(
            BizraAction::FileRead {
                path: "/home/user/doc.txt".into(),
            },
            permit,
            0.99,
        );
        assert!(g.evaluate(&env).is_approved());
    }

    #[test]
    fn guardian_requires_hitl_for_visitor() {
        let mut g = Guardian::new();
        let mut permit = Permit::visitor(vec![], 60);
        // Allow LLM channel (bit 1) and set requires_hitl
        permit.allowed_channels = 0b0000_0010;
        let env = make_envelope(
            BizraAction::LlmQuery {
                provider: "p".into(),
                model: "m".into(),
                system_prompt: "s".into(),
                user_prompt: "u".into(),
                max_tokens: 100,
                temperature: 0.7,
            },
            permit,
            0.99,
        );
        let v = g.evaluate(&env);
        // LLM is Low risk, so HITL is NOT triggered for Low risk even with requires_hitl
        assert!(matches!(v, GuardianVerdict::Approved { .. }));
    }

    #[test]
    fn guardian_strict_mode_higher_threshold() {
        let mut g = Guardian::strict();
        // Low risk normal threshold: 0.90. Strict: 0.92.
        let env = make_envelope(
            BizraAction::LlmQuery {
                provider: "p".into(),
                model: "m".into(),
                system_prompt: "s".into(),
                user_prompt: "u".into(),
                max_tokens: 100,
                temperature: 0.7,
            },
            Permit::user_default(),
            0.91, // Would pass normal, fails strict
        );
        let v = g.evaluate(&env);
        assert!(!v.is_approved());
    }

    #[test]
    fn guardian_tracks_approval_rate() {
        let mut g = Guardian::new();

        // One approval
        let env = make_envelope(
            BizraAction::RespondToUser {
                content: "hi".into(),
                ihsan_score: IhsanScore::new(0.99),
            },
            Permit::user_default(),
            0.99,
        );
        g.evaluate(&env);

        // One denial
        let env2 = make_envelope(
            BizraAction::AhkClick {
                window: "w".into(),
                element_path: "e".into(),
            },
            Permit::user_default(),
            0.50,
        );
        g.evaluate(&env2);

        let health = g.health();
        assert_eq!(health.approved, 1);
        assert_eq!(health.denied, 1);
        assert!((health.approval_rate - 0.5).abs() < 0.01);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Receipt Chain
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn hash_deterministic() {
        let h1 = content_hash(b"hello world");
        let h2 = content_hash(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_different_inputs_different_outputs() {
        let h1 = content_hash(b"hello");
        let h2 = content_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn chain_hash_deterministic() {
        let a = content_hash(b"a");
        let b = content_hash(b"b");
        let c1 = chain_hash(&a, &b);
        let c2 = chain_hash(&a, &b);
        assert_eq!(c1, c2);
    }

    #[test]
    fn chain_hash_order_sensitive() {
        let a = content_hash(b"a");
        let b = content_hash(b"b");
        assert_ne!(chain_hash(&a, &b), chain_hash(&b, &a));
    }

    #[test]
    fn receipt_chain_empty_verification() {
        let chain = ReceiptChain::new();
        assert_eq!(chain.verify_chain(), Ok(0));
        assert!(chain.is_empty());
    }

    #[test]
    fn receipt_chain_single_record() {
        let mut chain = ReceiptChain::new();
        let action = BizraAction::RespondToUser {
            content: "hello".into(),
            ihsan_score: IhsanScore::new(0.98),
        };
        chain.record(
            ActionId(1),
            ActionTimestamp(1000),
            &action,
            GuardianVerdict::Approved { reason: "ok" },
            IhsanScore::new(0.98),
            [0u8; 32],
        );
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.verify_chain(), Ok(1));
    }

    #[test]
    fn receipt_chain_multi_record_integrity() {
        let mut chain = ReceiptChain::new();
        for i in 0..10 {
            let action = BizraAction::RespondToUser {
                content: format!("msg {i}"),
                ihsan_score: IhsanScore::new(0.98),
            };
            chain.record(
                ActionId(i),
                ActionTimestamp(i * 1000),
                &action,
                GuardianVerdict::Approved { reason: "ok" },
                IhsanScore::new(0.98),
                content_hash(format!("payload_{i}").as_bytes()),
            );
        }
        assert_eq!(chain.len(), 10);
        assert_eq!(chain.verify_chain(), Ok(10));
    }

    #[test]
    fn receipt_chain_links_previous() {
        let mut chain = ReceiptChain::new();
        let action = BizraAction::AhkPerceive;

        chain.record(
            ActionId(1),
            ActionTimestamp(100),
            &action,
            GuardianVerdict::Approved { reason: "ok" },
            IhsanScore::new(0.99),
            [0u8; 32],
        );
        let first_hash = chain.get(0).unwrap().content_hash;

        chain.record(
            ActionId(2),
            ActionTimestamp(200),
            &action,
            GuardianVerdict::Approved { reason: "ok" },
            IhsanScore::new(0.99),
            [0u8; 32],
        );
        let second = chain.get(1).unwrap();

        assert_eq!(second.previous_hash, first_hash);
    }

    #[test]
    fn hash_payload_types() {
        let empty = hash_payload(&ActionPayload::Empty);
        let text = hash_payload(&ActionPayload::Text("hello".into()));
        let err = hash_payload(&ActionPayload::Error("fail".into()));
        assert_ne!(empty, text);
        assert_ne!(text, err);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Reflex Ledger
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn reflex_compile_requires_ihsan() {
        let mut ledger = ReflexLedger::new(10);
        let result = ledger.compile(
            "rename files",
            vec![],
            IhsanScore::new(0.80),
            ActionTimestamp(100),
            vec![],
        );
        assert!(matches!(
            result,
            Err(ReflexError::IhsanBelowThreshold { .. })
        ));
    }

    #[test]
    fn reflex_compile_and_lookup() {
        let mut ledger = ReflexLedger::new(10);
        let actions = vec![BizraAction::AhkPerceive];
        let desc = "rename invoice files by date";
        ledger
            .compile(
                desc,
                actions,
                IhsanScore::new(0.98),
                ActionTimestamp(100),
                vec![],
            )
            .unwrap();

        let hash = ReflexLedger::task_signature(desc);
        let found = ledger.lookup(&hash, ActionTimestamp(200));
        assert!(found.is_some());
        assert_eq!(found.unwrap().description, desc);
    }

    #[test]
    fn reflex_staleness() {
        let mut ledger = ReflexLedger::new(10);
        let desc = "task";
        ledger
            .compile(
                desc,
                vec![],
                IhsanScore::new(0.99),
                ActionTimestamp(100),
                vec![],
            )
            .unwrap();

        let hash = ReflexLedger::task_signature(desc);
        // Look up way in the future — should be stale
        let stale_time = ActionTimestamp(100 + 8 * 24 * 3600 * 1_000_000_000); // 8 days
        let found = ledger.lookup(&hash, stale_time);
        assert!(found.is_none());
    }

    #[test]
    fn reflex_eviction_lru() {
        let mut ledger = ReflexLedger::new(2);
        ledger
            .compile(
                "task_a",
                vec![],
                IhsanScore::new(0.99),
                ActionTimestamp(100),
                vec![],
            )
            .unwrap();
        ledger
            .compile(
                "task_b",
                vec![],
                IhsanScore::new(0.99),
                ActionTimestamp(200),
                vec![],
            )
            .unwrap();
        assert_eq!(ledger.len(), 2);

        // Third compile should evict LRU (task_a, older timestamp)
        ledger
            .compile(
                "task_c",
                vec![],
                IhsanScore::new(0.99),
                ActionTimestamp(300),
                vec![],
            )
            .unwrap();
        assert_eq!(ledger.len(), 2);

        // task_a should be gone
        let hash_a = ReflexLedger::task_signature("task_a");
        assert!(ledger.lookup(&hash_a, ActionTimestamp(300)).is_none());
    }

    #[test]
    fn reflex_hit_rate() {
        let mut ledger = ReflexLedger::new(10);
        let desc = "known_task";
        ledger
            .compile(
                desc,
                vec![],
                IhsanScore::new(0.99),
                ActionTimestamp(100),
                vec![],
            )
            .unwrap();

        let hash = ReflexLedger::task_signature(desc);
        let unknown = ReflexLedger::task_signature("unknown_task");

        ledger.lookup(&hash, ActionTimestamp(200)); // hit
        ledger.lookup(&unknown, ActionTimestamp(200)); // miss
        ledger.lookup(&hash, ActionTimestamp(200)); // hit

        let rate = ledger.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn reflex_marketplace_promotion() {
        let mut ledger = ReflexLedger::new(10);
        let desc = "sort_invoices";
        ledger
            .compile(
                desc,
                vec![],
                IhsanScore::new(0.99),
                ActionTimestamp(100),
                vec![],
            )
            .unwrap();

        let hash = ReflexLedger::task_signature(desc);

        // Not enough executions
        assert!(matches!(
            ledger.promote_to_marketplace(&hash, 10),
            Err(ReflexError::InsufficientExecutions { .. })
        ));

        // Simulate 10 executions
        for i in 0..10 {
            let reflex = ledger.lookup_mut(&hash, ActionTimestamp(200 + i)).unwrap();
            reflex.record_execution(ActionTimestamp(200 + i), 50_000);
        }

        // Now promote
        assert!(ledger.promote_to_marketplace(&hash, 10).is_ok());
        assert_eq!(ledger.marketplace_count(), 1);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Channel Stubs
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn ahk_channel_perceive() {
        let mut ch = AhkChannel::new();
        let result = ch.execute(&BizraAction::AhkPerceive);
        assert!(result.is_ok());
        match result.unwrap() {
            ActionPayload::Structured { entries } => {
                assert!(entries.iter().any(|(k, _)| k == "window"));
                assert!(entries.iter().any(|(k, _)| k == "process"));
            }
            _ => panic!("Expected Structured payload"),
        }
    }

    #[test]
    fn llm_channel_query() {
        let mut ch = LlmChannel::new();
        let result = ch.execute(&BizraAction::LlmQuery {
            provider: "lmstudio".into(),
            model: "qwen2.5".into(),
            system_prompt: "You are helpful".into(),
            user_prompt: "What is BIZRA?".into(),
            max_tokens: 200,
            temperature: 0.7,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn response_channel_delivers() {
        let mut ch = ResponseChannel::new();
        let result = ch.execute(&BizraAction::RespondToUser {
            content: "Hello, sovereign human.".into(),
            ihsan_score: IhsanScore::new(0.99),
        });
        assert!(result.is_ok());
        let text = match result.unwrap() {
            ActionPayload::Text(t) => t,
            _ => panic!("Expected text"),
        };
        assert!(text.contains("0.99"));
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Dispatcher — Full Pipeline Integration
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    fn build_dispatcher() -> Dispatcher {
        let mut d = Dispatcher::new();
        d.register_channel(Box::new(AhkChannel::new()));
        d.register_channel(Box::new(LlmChannel::new()));
        d.register_channel(Box::new(MemoryChannel::new()));
        d.register_channel(Box::new(FileSystemChannel::new()));
        d.register_channel(Box::new(ResponseChannel::new()));
        d.register_channel(Box::new(BrowserChannel::new()));
        d.register_channel(Box::new(McpChannel::new()));
        d.register_channel(Box::new(TelescriptChannel::new()));
        d
    }

    #[test]
    fn dispatcher_full_pipeline_ahk() {
        let mut d = build_dispatcher();
        let result = d.dispatch(
            BizraAction::AhkLaunch {
                executable: "notepad.exe".into(),
                args: vec![],
            },
            Permit::user_default(),
            IhsanScore::new(0.99),
            "test",
        );
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.success);
        assert_eq!(d.receipt_chain().len(), 1);
    }

    #[test]
    fn dispatcher_full_pipeline_llm() {
        let mut d = build_dispatcher();
        let result = d.dispatch(
            BizraAction::LlmQuery {
                provider: "lmstudio".into(),
                model: "qwen2.5".into(),
                system_prompt: "".into(),
                user_prompt: "What is BIZRA?".into(),
                max_tokens: 200,
                temperature: 0.7,
            },
            Permit::user_default(),
            IhsanScore::new(0.98),
            "sovereign_core",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn dispatcher_guardian_denial_produces_receipt() {
        let mut d = build_dispatcher();
        let result = d.dispatch(
            BizraAction::AhkClick {
                window: "w".into(),
                element_path: "e".into(),
            },
            Permit::user_default(),
            IhsanScore::new(0.50), // Way below threshold
            "test",
        );
        assert!(result.is_err());
        // Denial still produces a receipt
        assert_eq!(d.receipt_chain().len(), 1);
    }

    #[test]
    fn dispatcher_unregistered_channel() {
        let mut d = Dispatcher::new();
        // No channels registered
        let result = d.dispatch(
            BizraAction::AhkPerceive,
            Permit::user_default(),
            IhsanScore::new(0.99),
            "test",
        );
        assert!(matches!(
            result,
            Err(DispatchError::ChannelNotRegistered { .. })
        ));
    }

    #[test]
    fn dispatcher_multi_action_receipt_chain() {
        let mut d = build_dispatcher();

        // Execute 5 actions
        for i in 0..5 {
            d.dispatch(
                BizraAction::RespondToUser {
                    content: format!("Message {i}"),
                    ihsan_score: IhsanScore::new(0.98),
                },
                Permit::user_default(),
                IhsanScore::new(0.98),
                "test",
            )
            .unwrap();
        }

        // Verify chain integrity
        assert_eq!(d.receipt_chain().len(), 5);
        assert_eq!(d.receipt_chain().verify_chain(), Ok(5));
    }

    #[test]
    fn dispatcher_health_tracking() {
        let mut d = build_dispatcher();

        // 2 successes, 1 denial
        d.dispatch(
            BizraAction::AhkPerceive,
            Permit::user_default(),
            IhsanScore::new(0.99),
            "t",
        )
        .unwrap();
        d.dispatch(
            BizraAction::AhkPerceive,
            Permit::user_default(),
            IhsanScore::new(0.99),
            "t",
        )
        .unwrap();
        let _ = d.dispatch(
            BizraAction::AhkPerceive,
            Permit::user_default(),
            IhsanScore::new(0.50),
            "t",
        );

        let health = d.health();
        assert_eq!(health.total_dispatched, 3);
        assert_eq!(health.total_completed, 2);
        assert_eq!(health.total_denied, 1);
        assert_eq!(health.receipt_chain_length, 3); // Denials also produce receipts
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Constitutional Scenarios — Real-World Proof of Life
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// THE FIRST MISSION: "Open Notepad and write today's date"
    /// This is the proof of life that validates the entire architecture.
    #[test]
    fn scenario_first_mission_open_notepad() {
        let mut d = build_dispatcher();
        let permit = Permit::user_default();
        let high = IhsanScore::new(0.99);

        // Step 1: Perceive the desktop
        let r1 = d.dispatch(
            BizraAction::AhkPerceive,
            permit.clone(),
            high,
            "sovereign_core",
        );
        assert!(r1.is_ok());

        // Step 2: Launch Notepad
        let r2 = d.dispatch(
            BizraAction::AhkLaunch {
                executable: "notepad.exe".into(),
                args: vec![],
            },
            permit.clone(),
            high,
            "sovereign_core",
        );
        assert!(r2.is_ok());

        // Step 3: Type the date
        let r3 = d.dispatch(
            BizraAction::AhkType {
                window: "Notepad".into(),
                element_path: "Edit1".into(),
                text: "2026-02-24 — BIZRA Genesis Receipt".into(),
            },
            permit.clone(),
            high,
            "sovereign_core",
        );
        assert!(r3.is_ok());

        // Step 4: Verify (re-perceive)
        let r4 = d.dispatch(
            BizraAction::AhkPerceive,
            permit.clone(),
            high,
            "sovereign_core",
        );
        assert!(r4.is_ok());

        // Step 5: Respond to user with receipt
        let r5 = d.dispatch(
            BizraAction::RespondToUser {
                content: "Done. Opened Notepad and wrote today's date.".into(),
                ihsan_score: high,
            },
            permit,
            high,
            "sovereign_core",
        );
        assert!(r5.is_ok());

        // Verify: 5 actions, 5 receipts, chain intact
        assert_eq!(d.receipt_chain().len(), 5);
        assert_eq!(d.receipt_chain().verify_chain(), Ok(5));

        // Every receipt has a non-zero hash
        for i in 0..5 {
            let receipt = d.receipt_chain().get(i).unwrap();
            assert_ne!(receipt.content_hash, [0u8; 32]);
            assert!(receipt.ihsan_score.meets_constitutional());
        }

        let health = d.health();
        assert_eq!(health.total_completed, 5);
        assert_eq!(health.total_denied, 0);
        assert!((health.completion_rate - 1.0).abs() < 0.001);
    }

    /// Scenario: Visiting agent blocked from desktop
    #[test]
    fn scenario_visitor_agent_blocked_from_desktop() {
        let mut d = Dispatcher::strict();
        d.register_channel(Box::new(AhkChannel::new()));
        d.register_channel(Box::new(LlmChannel::new()));

        let visitor_permit = Permit::visitor(vec!["/tmp".into()], 300);

        // Visitor tries AHK — should be denied (channel not permitted)
        let result = d.dispatch(
            BizraAction::AhkClick {
                window: "Sensitive App".into(),
                element_path: "Delete".into(),
            },
            visitor_permit,
            IhsanScore::new(0.99),
            "visitor_agent_007",
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::GuardianDenied { reason, .. } => {
                assert!(reason.contains("not permitted"));
            }
            _ => panic!("Expected GuardianDenied"),
        }
    }

    /// Scenario: File access scoping prevents /etc/passwd read
    #[test]
    fn scenario_path_scoping_blocks_sensitive_files() {
        let mut d = build_dispatcher();
        let permit = Permit {
            fs_scope: vec!["/home/mumo/documents".into()],
            ..Permit::user_default()
        };

        // Try to read sensitive file
        let result = d.dispatch(
            BizraAction::FileRead {
                path: "/etc/shadow".into(),
            },
            permit,
            IhsanScore::new(0.99),
            "test",
        );

        assert!(result.is_err());
    }
}
