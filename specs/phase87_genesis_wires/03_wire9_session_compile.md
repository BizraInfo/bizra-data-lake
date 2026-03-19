# Wire 9 — Session Compile (Subscriber #7)

## Problem

`handle_session_compile()` is a stub. When a session ends, nothing happens
to crystallize learnings. The reflex compiler exists (`bizra-agent/src/reflex_compiler.rs`)
but is never triggered by the EventBus lifecycle.

## Current State (stub)

```rust
pub fn handle_session_compile(_event: &Event) -> HookResult {
    // Evaluate session learning delta
    // If sufficient new patterns: trigger mini-compile
    HookResult::Continue  // ← does nothing
}
```

## Solution: Same Pattern as Wire 8 — Atomic Flag

```rust
// bizra-hooks/src/subscribers.rs

/// #7: SessionEnd → Signal mini-compile needed
pub fn handle_session_compile(_event: &Event) -> HookResult {
    SESSION_COMPILE_PENDING.fetch_add(1, Ordering::Relaxed);
    HookResult::Continue
}
```

## Pseudocode: Mini-Compile in Node

```rust
// bizra-node/src/node.rs — in end_session handling

impl Node {
    fn handle_end_session(&mut self, timestamp: u64) -> Response {
        // End the conversation in runtime
        self.runtime.end_conversation(timestamp);

        // ── NEW: Mini-compile check ──
        self.mini_compile(timestamp);

        // Emit session.end event for subscriber #7
        // (subscriber sets the flag, but we also compile here directly
        //  since we have &mut self)

        Response::ok("session_ended")
    }

    /// Mini-genesis compilation at session boundary.
    ///
    /// Three operations, in order:
    /// 1. Extract: pull atoms from unprocessed fragments
    /// 2. Synthesize: merge related atoms into insights
    /// 3. Compile reflexes: promote hot patterns to reflex cache
    fn mini_compile(&mut self, timestamp: u64) {
        // Step 1: Extract any unprocessed fragments
        self.runtime.pipeline_mut().extract(timestamp);

        // Step 2: Synthesize — merge atoms into higher-order knowledge
        let insights = self.runtime.synthesize(timestamp);

        // Step 3: Compile reflexes from repeated patterns
        //
        // The reflex compiler looks at the last N actions and their
        // outcomes. Patterns that succeeded 3+ times with Ihsān >= 0.95
        // are compiled into reflex rules (System 2 → System 1).
        let compiled = self.compile_reflexes(timestamp);

        if self.config.show_banner && (insights > 0 || compiled > 0) {
            eprintln!(
                "  compile: {} insights synthesized, {} reflexes compiled",
                insights, compiled
            );
        }
    }

    /// Compile reflexes from action history.
    ///
    /// A reflex is: trigger_pattern → action_template
    /// Compiled when: same trigger led to same successful action 3+ times
    fn compile_reflexes(&mut self, _timestamp: u64) -> usize {
        let history = self.action_executor.recent_receipts(50);
        if history.len() < 3 {
            return 0;
        }

        let mut compiled = 0;

        // Group receipts by trigger pattern (content hash of input)
        let mut groups: HashMap<ActionHash, Vec<&ActionReceipt>> = HashMap::new();
        for receipt in &history {
            groups.entry(receipt.trigger_hash()).or_default().push(receipt);
        }

        for (trigger, receipts) in &groups {
            // Only compile if 3+ successes with high Ihsān
            let successes: Vec<_> = receipts.iter()
                .filter(|r| r.success && r.ihsan_score >= 9500)
                .collect();

            if successes.len() >= 3 {
                // Extract the common action template
                let template = ActionTemplate::from_receipts(&successes);

                // Add to reflex cache (if not already compiled)
                if self.runtime.reflex_cache_mut().try_compile(*trigger, template) {
                    compiled += 1;
                }
            }
        }

        compiled
    }
}
```

## Self-Compilation Export (Existing Phase 86-B)

The Node already has a self-compilation interval (`SELF_COMPILE_INTERVAL = 50`).
Every 50 commands, it exports memory atoms as ConversationTurnWire records.

Wire 9 extends this: at session boundaries (not just every 50 commands),
also run the reflex compiler. This means:

1. **Every 50 commands**: extract + export (existing)
2. **Every session end**: extract + synthesize + compile reflexes (Wire 9)
3. **Every heartbeat**: drain reinforcement flags (Wire 8)

These three together close the autopoietic loop.

## TDD Anchors

```rust
#[cfg(test)]
mod wire9_tests {
    use super::*;

    #[test]
    fn session_compile_flag_set_on_session_end() {
        SESSION_COMPILE_PENDING.store(0, Ordering::Relaxed);

        let event = Event {
            id: EventId::new(1000, 0),
            source: ComponentId::from_name("test", "1.0.0"),
            topic: Topic::new(TOPIC_SESSION_END),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::MAX,
        };

        handle_session_compile(&event);
        assert_eq!(SESSION_COMPILE_PENDING.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn mini_compile_with_empty_history_compiles_nothing() {
        let mut node = Node::new(NodeConfig::default());
        let compiled = node.compile_reflexes(1000);
        assert_eq!(compiled, 0);
    }

    #[test]
    fn reflex_compiled_after_three_successes() {
        let mut node = Node::new(NodeConfig::default());

        // Simulate 3 successful actions with same trigger
        let trigger = ActionHash::from_content("organize files");
        for i in 0..3 {
            node.action_executor.record_receipt(ActionReceipt {
                trigger_hash: trigger,
                success: true,
                ihsan_score: 9700,
                timestamp: 1000 + i * 100,
                ..Default::default()
            });
        }

        let compiled = node.compile_reflexes(2000);
        assert_eq!(compiled, 1);

        // Same trigger again should NOT re-compile
        let compiled2 = node.compile_reflexes(3000);
        assert_eq!(compiled2, 0);
    }
}
```

## Blast Radius

| File | Change | Risk |
|------|--------|------|
| `bizra-hooks/src/subscribers.rs` | Update handler to set flag | Trivial |
| `bizra-node/src/node.rs` | Add mini_compile(), compile_reflexes() | Medium |
| `bizra-agent/src/reflex_cache.rs` | Add try_compile() | Low — additive |
| Existing tests | No change | Zero |
