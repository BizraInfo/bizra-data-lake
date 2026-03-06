//! HookChain — before/after/error interceptors for any named operation.
//!
//! Hook chains wrap operations with typed interceptors. This is how:
//! - Memory context gets injected before LLM calls
//! - إحسان scoring happens after every response
//! - Canary gates decide whether to route through new code
//! - Errors trigger rollback handlers
//!
//! Standing on Giants: Hoare (pre/postconditions, 1969) · AspectJ (AOP, 2001)

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::types::*;

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK FUNCTIONS — The interceptor signatures.
// ═══════════════════════════════════════════════════════════════════════════════

/// A before-hook receives the event and decides: proceed, modify, or abort.
type BeforeFn = Box<dyn Fn(&Event) -> HookAction + Send + Sync>;

/// An after-hook receives the event and the outcome for observation/scoring.
type AfterFn = Box<dyn Fn(&Event, &Outcome) + Send + Sync>;

/// An error-hook receives the event and the error string for recovery/logging.
type ErrorFn = Box<dyn Fn(&Event, &str) + Send + Sync>;

struct BeforeHook {
    name: String,
    priority: HookPriority,
    f: BeforeFn,
}

struct AfterHook {
    name: String,
    priority: HookPriority,
    f: AfterFn,
}

struct ErrorHook {
    name: String,
    priority: HookPriority,
    f: ErrorFn,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK CHAIN — Named operation with before/after/error interceptors.
// ═══════════════════════════════════════════════════════════════════════════════

/// A named hook chain for a specific operation (e.g., "llm_call", "memory_retrieve").
struct ChainInner {
    before: Vec<BeforeHook>,
    after: Vec<AfterHook>,
    error: Vec<ErrorHook>,
}

/// Thread-safe hook chain manager. Maps operation names to their hook chains.
#[derive(Clone)]
pub struct HookChain {
    chains: Arc<RwLock<HashMap<String, ChainInner>>>,
}

impl HookChain {
    pub fn new() -> Self {
        Self {
            chains: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a before-hook for an operation.
    pub fn before<F>(
        &self,
        operation: impl Into<String>,
        name: impl Into<String>,
        priority: HookPriority,
        f: F,
    ) -> HookResult<()>
    where
        F: Fn(&Event) -> HookAction + Send + Sync + 'static,
    {
        let op = operation.into();
        let mut chains = self
            .chains
            .write()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        let chain = chains.entry(op).or_insert_with(|| ChainInner {
            before: Vec::new(),
            after: Vec::new(),
            error: Vec::new(),
        });

        chain.before.push(BeforeHook {
            name: name.into(),
            priority,
            f: Box::new(f),
        });
        chain.before.sort_by_key(|h| h.priority);
        Ok(())
    }

    /// Register an after-hook for an operation.
    pub fn after<F>(
        &self,
        operation: impl Into<String>,
        name: impl Into<String>,
        priority: HookPriority,
        f: F,
    ) -> HookResult<()>
    where
        F: Fn(&Event, &Outcome) + Send + Sync + 'static,
    {
        let op = operation.into();
        let mut chains = self
            .chains
            .write()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        let chain = chains.entry(op).or_insert_with(|| ChainInner {
            before: Vec::new(),
            after: Vec::new(),
            error: Vec::new(),
        });

        chain.after.push(AfterHook {
            name: name.into(),
            priority,
            f: Box::new(f),
        });
        chain.after.sort_by_key(|h| h.priority);
        Ok(())
    }

    /// Register an error-hook for an operation.
    pub fn on_error<F>(
        &self,
        operation: impl Into<String>,
        name: impl Into<String>,
        priority: HookPriority,
        f: F,
    ) -> HookResult<()>
    where
        F: Fn(&Event, &str) + Send + Sync + 'static,
    {
        let op = operation.into();
        let mut chains = self
            .chains
            .write()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        let chain = chains.entry(op).or_insert_with(|| ChainInner {
            before: Vec::new(),
            after: Vec::new(),
            error: Vec::new(),
        });

        chain.error.push(ErrorHook {
            name: name.into(),
            priority,
            f: Box::new(f),
        });
        chain.error.sort_by_key(|h| h.priority);
        Ok(())
    }

    /// Execute the full hook chain for an operation.
    ///
    /// 1. Run before-hooks in priority order. If any returns Abort, stop.
    /// 2. Execute the operation (provided closure).
    /// 3. Run after-hooks with the outcome.
    /// 4. On error, run error-hooks.
    ///
    /// Returns the final outcome.
    pub fn execute<F>(
        &self,
        operation: &str,
        mut event: Event,
        op_fn: F,
    ) -> HookResult<Outcome>
    where
        F: FnOnce(&Event) -> Result<Payload, String>,
    {
        let chains = self
            .chains
            .read()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        let chain = chains.get(operation);

        // ── Before hooks ──
        if let Some(c) = chain {
            for hook in &c.before {
                match (hook.f)(&event) {
                    HookAction::Proceed => {} // Continue with current event.
                    HookAction::Modify(new_event) => {
                        event = new_event; // Continue with modified event.
                    }
                    HookAction::Abort(reason) => {
                        return Err(HookError::HookAborted {
                            hook_name: hook.name.clone(),
                            reason,
                        });
                    }
                }
            }
        }

        // ── Execute operation ──
        let outcome = match op_fn(&event) {
            Ok(payload) => Outcome::Success(payload),
            Err(err_msg) => {
                // Run error hooks.
                if let Some(c) = chain {
                    for hook in &c.error {
                        (hook.f)(&event, &err_msg);
                    }
                }
                Outcome::Failure(err_msg)
            }
        };

        // ── After hooks ──
        if let Some(c) = chain {
            for hook in &c.after {
                (hook.f)(&event, &outcome);
            }
        }

        Ok(outcome)
    }

    /// How many operations have registered hooks?
    pub fn operation_count(&self) -> usize {
        self.chains.read().map(|c| c.len()).unwrap_or(0)
    }

    /// How many total hooks are registered across all operations?
    pub fn total_hooks(&self) -> usize {
        self.chains
            .read()
            .map(|c| {
                c.values()
                    .map(|chain| chain.before.len() + chain.after.len() + chain.error.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// List all operations that have hooks.
    pub fn operations(&self) -> Vec<String> {
        self.chains
            .read()
            .map(|c| c.keys().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for HookChain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HookChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookChain")
            .field("operations", &self.operation_count())
            .field("total_hooks", &self.total_hooks())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    fn test_source() -> ComponentId {
        ComponentId(8888)
    }

    #[test]
    fn before_after_hooks_run() {
        let hooks = HookChain::new();
        let before_ran = Arc::new(AtomicBool::new(false));
        let after_ran = Arc::new(AtomicBool::new(false));

        let br = before_ran.clone();
        hooks
            .before("llm_call", "check", HookPriority::APP, move |_| {
                br.store(true, Ordering::Relaxed);
                HookAction::Proceed
            })
            .unwrap();

        let ar = after_ran.clone();
        hooks
            .after("llm_call", "score", HookPriority::APP, move |_, _| {
                ar.store(true, Ordering::Relaxed);
            })
            .unwrap();

        let event = Event::new(EventKind::AgentResponse, test_source());
        let result = hooks.execute("llm_call", event, |_| Ok(Payload::text("response")));

        assert!(before_ran.load(Ordering::Relaxed));
        assert!(after_ran.load(Ordering::Relaxed));
        assert!(matches!(result, Ok(Outcome::Success(_))));
    }

    #[test]
    fn before_hook_can_abort() {
        let hooks = HookChain::new();

        hooks
            .before("dangerous_op", "gate", HookPriority::SYSTEM, |_| {
                HookAction::Abort("blocked by FATE".into())
            })
            .unwrap();

        let event = Event::new(EventKind::MutationProposed, test_source());
        let result = hooks.execute("dangerous_op", event, |_| Ok(Payload::Empty));

        assert!(result.is_err());
        if let Err(HookError::HookAborted { hook_name, reason }) = result {
            assert_eq!(hook_name, "gate");
            assert_eq!(reason, "blocked by FATE");
        }
    }

    #[test]
    fn error_hooks_fire_on_failure() {
        let hooks = HookChain::new();
        let error_seen = Arc::new(AtomicBool::new(false));
        let es = error_seen.clone();

        hooks
            .on_error("flaky_op", "logger", HookPriority::INFRA, move |_, _err| {
                es.store(true, Ordering::Relaxed);
            })
            .unwrap();

        let event = Event::new(EventKind::TaskFailed, test_source());
        let result = hooks.execute("flaky_op", event, |_| Err("connection timeout".into()));

        assert!(error_seen.load(Ordering::Relaxed));
        assert!(matches!(result, Ok(Outcome::Failure(_))));
    }

    #[test]
    fn before_hook_can_modify_event() {
        let hooks = HookChain::new();
        let seen_text = Arc::new(Mutex::new(String::new()));
        let st = seen_text.clone();

        // Before-hook injects context.
        hooks
            .before("llm_call", "inject_memory", HookPriority::APP, |event| {
                let mut modified = event.clone();
                modified.payload = Payload::text("enriched with memory context");
                HookAction::Modify(modified)
            })
            .unwrap();

        // After-hook observes the (modified) event.
        hooks
            .after("llm_call", "observe", HookPriority::APP, move |event, _| {
                if let Payload::Text(t) = &event.payload {
                    *st.lock().unwrap() = t.clone();
                }
            })
            .unwrap();

        let event = Event::new(EventKind::UserMessage, test_source());
        hooks
            .execute("llm_call", event, |e| {
                // The operation sees the modified event.
                if let Payload::Text(t) = &e.payload {
                    Ok(Payload::text(format!("processed: {}", t)))
                } else {
                    Ok(Payload::text("no context"))
                }
            })
            .unwrap();

        // After-hook saw the modified payload.
        assert_eq!(*seen_text.lock().unwrap(), "enriched with memory context");
    }

    #[test]
    fn priority_order_matters() {
        let hooks = HookChain::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let o1 = order.clone();
        hooks
            .before("op", "user_hook", HookPriority::USER, move |_| {
                o1.lock().unwrap().push(3);
                HookAction::Proceed
            })
            .unwrap();

        let o2 = order.clone();
        hooks
            .before("op", "system_hook", HookPriority::SYSTEM, move |_| {
                o2.lock().unwrap().push(1);
                HookAction::Proceed
            })
            .unwrap();

        let o3 = order.clone();
        hooks
            .before("op", "app_hook", HookPriority::APP, move |_| {
                o3.lock().unwrap().push(2);
                HookAction::Proceed
            })
            .unwrap();

        let event = Event::new(EventKind::TaskStart, test_source());
        hooks.execute("op", event, |_| Ok(Payload::Empty)).unwrap();

        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn no_hooks_still_works() {
        let hooks = HookChain::new();
        let event = Event::new(EventKind::UserMessage, test_source());
        let result = hooks.execute("unhooked_op", event, |_| Ok(Payload::text("works")));
        assert!(matches!(result, Ok(Outcome::Success(_))));
    }
}
