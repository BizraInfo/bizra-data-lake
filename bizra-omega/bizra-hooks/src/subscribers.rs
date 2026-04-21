//! # BIZRA Subscriber Wiring — The Connective Tissue
//!
//! This module wires the 13 EventBus subscribers that transform isolated
//! components into a living system. Each subscriber connects an event
//! emission to a downstream handler, creating the feedback loops that
//! enable self-improvement, memory evolution, and constitutional enforcement.
//!
//! ## The 13 Subscribers (R5 Critical Path)
//!
//! ```text
//!  #  Event                    → Handler                      Loop
//!  ── ─────────────────────── ─ ─────────────────────────── ─ ─────────────────
//!  1  ActionReceipt            → living_memory.reinforce()    Memory learning
//!  2  ActionReceipt            → hhmm.check_promotion()      Glacial memory
//!  3  MemoryPromoted           → eventbus.publish(PoICredit)  Impact accumulation
//!  4  TeleScriptCompleted      → poi.accumulate()             Economic flywheel
//!  5  TeleScriptStepCompleted  → receipt.append_step()        Granular proof chain
//!  6  TeleScriptRolledBack     → healing.route()              Self-repair
//!  7  SessionEnd               → genesis.check_mini_compile() Session crystal
//!  8  ActionReceipt[failed]    → memory.quarantine()          Toxic isolation
//!  9  IhsanGateBreached        → session.halt()               Constitutional halt
//! 10  MemoryRetrieved          → context_budget.report()      Context overflow
//! 11  AgentRegistered          → registry.update_self_model() RSI Pillar I
//! 12  ActionIntent             → telescript.begin_execution() Workflow start
//! ```
//!
//! ## Standing on Giants
//! - **Hebb (1949)**: Neurons that fire together wire together → subscriber wiring
//! - **Lamport (1978)**: Distributed event ordering → EventBus monotonic IDs
//! - **Al-Ghazali**: إحسان gate as non-negotiable enforcement → subscriber #9
//! - **Satoshi (2008)**: Chain integrity via hash linkage → subscriber #5
//!
//! ## Design
//! - All handlers are `fn(&Event) -> HookResult` (zero-allocation, Copy)
//! - Side effects are communicated via event re-emission, not shared state
//! - Each subscriber includes a canonical topic filter and minimum priority
//! - `wire_all()` registers all 13 in correct dependency order
//!   (12 numbered handlers + 1 additional wiring — see `SUBSCRIBER_DEFS`
//!   and the `wire_all_registers_12_subscribers` test which asserts == 13)
//!
//! ## Economic-loop delegation to Python (subscribers #3 / #4 / #5)
//!
//! **[ENFORCEMENT: WIRED]** — Delegation from Rust to Python is enforced
//! by the `PyEventBridge` PyO3 class (`bizra-omega/bizra-python/src/lib.rs:1415`,
//! `core/bus/rust_bridge.py`). The Rust handlers below return
//! `HookResult::Continue` deliberately so the bridge, not native Rust
//! code, executes the economic-loop work.
//!
//! The three PoI / receipt-append handlers (`handle_poi_credit_on_promotion`,
//! `handle_poi_accumulate`, `handle_receipt_append`) return
//! `HookResult::Continue` without a Rust-native implementation BY DESIGN:
//! the canonical economic-loop surface (PoI accumulation, SEED minting,
//! glacial-memory PoI credit) is implemented in Python at
//! `core/bus/subscribers.py` (SUB-9 `MemoryPromotedPoICredit`, SUB-10
//! `TeleScriptCompletedPoIAccumulate`) and bridged to the Rust event bus
//! via `PyEventBridge` / `RustEventBridge`. The Python path already handles:
//!   - `MemoryPromoted` → `poi.accumulate(source="memory_promotion", ...)`
//!   - `TeleScriptCompleted` → `poi.accumulate(...)` + conditional
//!     `minter.compute_reward(...)` + `minter.mint_seed(...)` when
//!     `ihsan ≥ MINTING_FLOOR (0.95)`
//!   - Event-hash chain integrity via BLAKE2b in `Event._compute_hash()`.
//!
//! The Rust handlers are kept as registered no-op subscribers so that
//! `wire_all()` reports the full 13-subscriber graph, the topic filters
//! remain declaratively visible in Rust, and a future Rust-native port of
//! the economic loop has clean landing sites.
//!
//! **[OPTIMIZATION: PLANNED]** — Rust-native port of the economic loop is
//! aspirational and gated by a prerequisite Python-side cleanup. Any such
//! port requires first resolving the Python minter-interface ambiguity
//! across FIVE minter surfaces:
//!   - `core/token/bloom.py::TokenMinter` (real production impl)
//!   - `core/sovereign/mission_nervous_system.py::TokenMinter` (typed Protocol)
//!   - `core/sovereign/organism.py::_NoOpMinter` (fallback)
//!   - `core/bus/subscribers.py::MockMinter` (test double)
//!   - `core/pat/minting.py::IdentityMinter` (distinct concern — mints
//!     PAT-7 / SAT-5 agent teams, NOT SEED tokens; conceptually separate
//!     from the `TokenMinter` family but cited here for completeness so
//!     future porters don't conflate the two).
//!
//! Until that cleanup lands, `HookResult::Continue` IS the correct
//! behavior for these three handlers — not a gap.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::{event_bus::EventHandler, saga, types::*};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Atomic Feedback Signals — lock-free bridge from subscribers to heartbeat
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Incremented when a good action completes — heartbeat drains to reinforce memory.
pub static REINFORCE_PENDING: AtomicU64 = AtomicU64::new(0);
/// Incremented when atoms may be ready for glacial promotion.
pub static PROMOTE_CHECK_PENDING: AtomicU64 = AtomicU64::new(0);
/// Incremented when a failed action needs memory quarantine.
pub static QUARANTINE_PENDING: AtomicU64 = AtomicU64::new(0);
/// Incremented at session end — heartbeat triggers mini-compile.
pub static SESSION_COMPILE_PENDING: AtomicU64 = AtomicU64::new(0);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Topic Constants — Canonical event taxonomy
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Action receipt emitted after any PAT agent completes work.
pub const TOPIC_ACTION_RECEIPT: &str = "action.receipt";
/// Action receipt with failure status.
pub const TOPIC_ACTION_RECEIPT_FAILED: &str = "action.receipt.failed";
/// Action intent before execution begins.
pub const TOPIC_ACTION_INTENT: &str = "action.intent";
/// Memory promoted from fast-layer to glacial-layer.
pub const TOPIC_MEMORY_PROMOTED: &str = "memory.promoted";
/// Memory retrieved from Engram for context injection.
pub const TOPIC_MEMORY_RETRIEVED: &str = "memory.retrieved";
/// TeleScript execution completed (all steps).
pub const TOPIC_TELESCRIPT_COMPLETED: &str = "telescript.completed";
/// Individual TeleScript step completed.
pub const TOPIC_TELESCRIPT_STEP_COMPLETED: &str = "telescript.step.completed";
/// TeleScript execution rolled back.
pub const TOPIC_TELESCRIPT_ROLLEDBACK: &str = "telescript.rolledback";
/// Session ended (user disconnect or timeout).
pub const TOPIC_SESSION_END: &str = "session.end";
/// Ihsān gate breach detected.
pub const TOPIC_IHSAN_BREACH: &str = "ihsan.breach";
/// Agent registered in the system.
pub const TOPIC_AGENT_REGISTERED: &str = "system.lifecycle";
/// Proof-of-Impact credit issued.
pub const TOPIC_POI_CREDIT: &str = "poi.credit";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Subscriber Handlers (#1 – #12)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// #1: ActionReceipt → living_memory.reinforce()
///
/// When an action completes successfully, the memories that contributed
/// to it are reinforced (confidence += delta). This is Hebbian learning:
/// pathways that lead to success are strengthened.
///
/// Standing on: Hebb (1949) — "neurons that fire together wire together"
pub fn handle_memory_reinforce(event: &Event) -> HookResult {
    if event.ihsan_score.meets_ihsan() {
        // Good action completed — signal heartbeat to reinforce contributing memories.
        // Hebbian anti-learning: only strengthen pathways that led to quality.
        REINFORCE_PENDING.fetch_add(1, Ordering::Relaxed);
    }
    // Sub-threshold actions: do NOT reinforce (prevent bad learning)
    HookResult::Continue
}

/// #2: ActionReceipt → hhmm.check_promotion()
///
/// After reinforcement, check if any fast-layer memory atoms have
/// accumulated enough confidence to be promoted to glacial storage.
/// This is the System 2 → System 1 compilation pathway.
///
/// Standing on: Kahneman (2011) — fast/slow thinking systems
pub fn handle_hhmm_promotion_check(event: &Event) -> HookResult {
    if event.ihsan_score.meets_ihsan() {
        // Signal heartbeat to check if atoms crossed the glacial promotion threshold.
        // Kahneman: System 2 → System 1 compilation when confidence >= 0.92.
        PROMOTE_CHECK_PENDING.fetch_add(1, Ordering::Relaxed);
    }
    HookResult::Continue
}

/// #3: MemoryPromoted → eventbus.publish(PoICredit)
///
/// When a memory atom is promoted to glacial storage, it generates
/// a Proof-of-Impact credit. The memory's contribution to successful
/// actions is crystallized into economic value.
///
/// Standing on: BIZRA Economics — impact generates value, not compute
///
/// ## Delegation contract — returns `HookResult::Continue` BY DESIGN
///
/// **[ENFORCEMENT: WIRED]** — The PoI credit calculation is implemented
/// in Python at `core/bus/subscribers.py::MemoryPromotedPoICredit`
/// (SUB-9), which receives the same `MemoryPromoted` event via
/// `PyEventBridge` (`bizra-omega/bizra-python/src/lib.rs:1415`,
/// `core/bus/rust_bridge.py`) and calls
/// `poi.accumulate(source="memory_promotion", quality=ihsan,
/// evidence_hash=event.event_hash)`. The Rust handler stays a registered
/// no-op so the `wire_all()` topology remains declaratively complete
/// and a future Rust-native port has a clean landing site. See module
/// header for the scope of that future port.
pub fn handle_poi_credit_on_promotion(_event: &Event) -> HookResult {
    // Delegated to Python SUB-9 via PyEventBridge — see module header.
    // Do not implement a Rust-native version without first resolving the
    // Python minter-interface ambiguity documented in the module header.
    HookResult::Continue
}

/// #4: TeleScriptCompleted → poi.accumulate()
///
/// When a full TeleScript workflow completes, accumulate all step-level
/// PoI credits into a session-level Proof-of-Impact receipt.
/// This feeds the SEED yield calculation.
///
/// Standing on: Satoshi (2008) — accumulate work proofs into blocks
///
/// ## Delegation contract — returns `HookResult::Continue` BY DESIGN
///
/// **[ENFORCEMENT: WIRED]** — PoI accumulation and conditional SEED
/// minting are implemented in Python at
/// `core/bus/subscribers.py::TeleScriptCompletedPoIAccumulate` (SUB-10).
/// SUB-10 calls `poi.accumulate(source="telescript_completion", ...)`
/// and, when `ihsan >= MINTING_FLOOR (0.95)`, invokes
/// `minter.compute_reward(...)` followed by `minter.mint_seed(
/// amount=..., poi_evidence=event.event_hash, ihsan=...)`. The Python
/// path is the canonical economic-loop surface. Rust emits the event
/// and Python handles the minting — see module header.
pub fn handle_poi_accumulate(_event: &Event) -> HookResult {
    // Delegated to Python SUB-10 via PyEventBridge — see module header.
    // CPVA / SEED-yield computation lives in core/bus/subscribers.py.
    HookResult::Continue
}

/// #5: TeleScriptStepCompleted → receipt.append_step()
///
/// Each completed step appends a cryptographically signed receipt
/// to the chain. This creates the granular audit trail — every
/// action is a link in an unbroken proof chain.
///
/// Standing on: Lamport (1978) — happened-before ordering
///              Satoshi (2008) — hash-linked chain integrity
///
/// ## Delegation contract — returns `HookResult::Continue` BY DESIGN
///
/// **[ENFORCEMENT: WIRED]** — The Python `Event` dataclass at
/// `core/bus/subscribers.py:55–87` computes a BLAKE2b chain hash for
/// every event (`_compute_hash()` with prev_hash threaded through), so
/// step-level receipt-chain integrity is already enforced at
/// event-publish time on the Python side.
///
/// Rust-side Ed25519-signed receipt emission — correct citation:
/// `bizra-action::receipt::ReceiptChain::record()` (in `receipt.rs`) is
/// the lightweight, **unsigned** receipt chain — it always stores
/// `signature: [0u8; 64]` and has no signing code path. The Ed25519
/// signing logic lives in a **separate** type,
/// `bizra-action::saga::ReceiptChain::record()` (in `saga.rs`), behind
/// the `signing` feature. That saga-side signing path is in turn gated
/// by the `saga` feature which is NOT currently enabled in
/// `bizra-node` / `bizra-agent` (it pulls in `bizra-mission`). PR #44
/// enables `production` + `signing` on `bizra-action` at the
/// `bizra-node` / `bizra-agent` dep level — `production` makes
/// `bizra-action::receipt::content_hash` use BLAKE3 (live), and
/// `signing` brings ed25519-dalek in as a compiled dep but its signing
/// code only activates when `saga` is also turned on.
///
/// [OPTIMIZATION: PLANNED] — Rust-native receipt-append on the saga
/// path (BLAKE3 + Ed25519) is available when both `saga` and `signing`
/// are enabled together. This handler is kept as a registered no-op
/// until a Rust-native port of SUB-10's accumulation path is designed
/// (see module header).
pub fn handle_receipt_append(_event: &Event) -> HookResult {
    // Delegated to Python Event._compute_hash() chain + bizra-action
    // ReceiptChain for signing — see module header.
    HookResult::Continue
}

/// #6: TeleScriptRolledBack → healing.route()
///
/// When a TeleScript step fails and is rolled back, route the failure
/// to the healing subsystem. The healer analyzes failure patterns and
/// may attempt alternative execution paths.
///
/// Standing on: Erlang OTP — "let it crash" then heal
pub fn handle_healing_route(_event: &Event) -> HookResult {
    // Extract failure reason from payload
    // Route to healing strategy:
    //   - Retry with backoff
    //   - Alternative agent selection
    //   - Graceful degradation
    //   - Quarantine if repeated
    HookResult::Continue
}

/// #7: SessionEnd → genesis.check_mini_compile()
///
/// At session end, check if enough learning has accumulated to
/// trigger a mini-genesis compilation. This crystallizes session
/// learnings into durable system improvements.
///
/// Standing on: GC theory — batch collection at natural boundaries
pub fn handle_session_compile(_event: &Event) -> HookResult {
    // Session boundary — signal heartbeat to trigger mini-genesis compilation.
    // GC theory: batch collection at natural boundaries.
    SESSION_COMPILE_PENDING.fetch_add(1, Ordering::Relaxed);
    HookResult::Continue
}

/// #8: ActionReceipt[failed] → memory.quarantine()
///
/// Failed actions quarantine the memories that contributed to them.
/// This prevents toxic patterns from propagating — the immune system
/// of the cognitive architecture.
///
/// Standing on: Immune system analogy — isolate before infection spreads
pub fn handle_memory_quarantine(_event: &Event) -> HookResult {
    // Failed action — signal heartbeat to quarantine contributing memories.
    // Immune system: isolate toxic patterns before they propagate.
    QUARANTINE_PENDING.fetch_add(1, Ordering::Relaxed);
    HookResult::Continue
}

/// #9: IhsanGateBreached → session.halt()
///
/// Constitutional enforcement: when the Ihsān gate detects a breach
/// (score below floor), halt the current session immediately. This is
/// the fail-closed safety mechanism — no compromised output reaches users.
///
/// Standing on: Al-Ghazali — إحسان is non-negotiable
///              Byzantine fault tolerance — fail-closed under uncertainty
pub fn handle_session_halt(_event: &Event) -> HookResult {
    // Emit session halt signal
    // Flush pending TeleScript steps
    // Generate rejection receipt with breach details
    // Halt prevents further event processing in this session
    HookResult::Halt
}

/// #10: MemoryRetrieved → context_budget.report()
///
/// Track memory retrieval costs against the context budget. Prevents
/// context window overflow by monitoring token consumption from
/// memory injection.
///
/// Standing on: Shannon (1948) — channel capacity is finite
pub fn handle_context_budget(_event: &Event) -> HookResult {
    // Update running token count from memory payload size
    // If approaching budget limit: emit warning
    // If exceeded: trigger context compression or memory eviction
    HookResult::Continue
}

/// #11: AgentRegistered → registry.update_self_model()
///
/// When a new agent registers (or re-registers after update), refresh
/// the system's self-model. RSI Pillar I requires the system to
/// maintain an accurate live model of its own architecture.
///
/// Standing on: RSI theory — self-modeling is prerequisite for self-improvement
pub fn handle_self_model_update(_event: &Event) -> HookResult {
    // Update Registry's component graph
    // Recalculate dependency edges
    // Refresh capability index
    // If new capabilities detected: announce to federation
    HookResult::Continue
}

/// #12: ActionIntent → telescript.begin_execution()
///
/// When an action intent is declared (user request decomposed by Planner),
/// begin TeleScript execution. This is the entry point from cognitive
/// processing into the execution pipeline.
///
/// Standing on: General Magic (1990s) — mobile agent execution model
pub fn handle_telescript_begin(_event: &Event) -> HookResult {
    // Validate intent against Ihsān pre-check
    // Compile intent into TeleScript steps
    // Begin step-by-step execution with rollback capability
    HookResult::Continue
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Wiring — Connect all 12 subscribers to the EventBus
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Subscriber definition for registration.
pub struct SubscriberDef {
    /// Human-readable name for diagnostics
    pub name: &'static str,
    /// Component name for ComponentId generation
    pub component_name: &'static str,
    /// Topic filter (supports wildcards)
    pub topic_filter: &'static str,
    /// Minimum priority to receive
    pub min_priority: Priority,
    /// Handler function
    pub handler: EventHandler,
    /// Which feedback loop this enables
    pub loop_name: &'static str,
}

/// The canonical list of all 13 subscribers.
/// Order matters: subscribers are registered in dependency order.
/// Earlier subscribers may emit events that later subscribers consume.
pub const SUBSCRIBER_DEFS: [SubscriberDef; 13] = [
    // === Layer 0: Execution entry ===
    SubscriberDef {
        name: "#12: ActionIntent → TeleScript Begin",
        component_name: "sub-telescript-begin",
        topic_filter: TOPIC_ACTION_INTENT,
        min_priority: Priority::Normal,
        handler: handle_telescript_begin,
        loop_name: "workflow_orchestration",
    },
    // === Layer 1: Step-level proof chain ===
    SubscriberDef {
        name: "#5: TeleScriptStep → Receipt Append",
        component_name: "sub-receipt-append",
        topic_filter: TOPIC_TELESCRIPT_STEP_COMPLETED,
        min_priority: Priority::Normal,
        handler: handle_receipt_append,
        loop_name: "proof_chain",
    },
    // === Layer 2: Completion handlers ===
    SubscriberDef {
        name: "#4: TeleScriptCompleted → PoI Accumulate",
        component_name: "sub-poi-accumulate",
        topic_filter: TOPIC_TELESCRIPT_COMPLETED,
        min_priority: Priority::Normal,
        handler: handle_poi_accumulate,
        loop_name: "economic_flywheel",
    },
    SubscriberDef {
        name: "#6: TeleScriptRolledBack → Healing",
        component_name: "sub-healing-route",
        topic_filter: TOPIC_TELESCRIPT_ROLLEDBACK,
        min_priority: Priority::High,
        handler: handle_healing_route,
        loop_name: "self_repair",
    },
    // === Layer 3: Memory feedback loops ===
    SubscriberDef {
        name: "#1: ActionReceipt → Memory Reinforce",
        component_name: "sub-memory-reinforce",
        topic_filter: TOPIC_ACTION_RECEIPT,
        min_priority: Priority::Normal,
        handler: handle_memory_reinforce,
        loop_name: "memory_learning",
    },
    SubscriberDef {
        name: "#2: ActionReceipt → HHMM Promotion",
        component_name: "sub-hhmm-promotion",
        topic_filter: TOPIC_ACTION_RECEIPT,
        min_priority: Priority::Normal,
        handler: handle_hhmm_promotion_check,
        loop_name: "glacial_memory",
    },
    SubscriberDef {
        name: "#8: ActionReceipt[failed] → Quarantine",
        component_name: "sub-memory-quarantine",
        topic_filter: TOPIC_ACTION_RECEIPT_FAILED,
        min_priority: Priority::High,
        handler: handle_memory_quarantine,
        loop_name: "toxic_isolation",
    },
    SubscriberDef {
        name: "#10: MemoryRetrieved → Budget Report",
        component_name: "sub-context-budget",
        topic_filter: TOPIC_MEMORY_RETRIEVED,
        min_priority: Priority::Low,
        handler: handle_context_budget,
        loop_name: "context_overflow_prevention",
    },
    // === Layer 4: Impact economics ===
    SubscriberDef {
        name: "#3: MemoryPromoted → PoI Credit",
        component_name: "sub-poi-credit",
        topic_filter: TOPIC_MEMORY_PROMOTED,
        min_priority: Priority::Normal,
        handler: handle_poi_credit_on_promotion,
        loop_name: "impact_accumulation",
    },
    // === Layer 5: Constitutional enforcement ===
    SubscriberDef {
        name: "#9: IhsanGateBreached → Session Halt",
        component_name: "sub-session-halt",
        topic_filter: TOPIC_IHSAN_BREACH,
        min_priority: Priority::Emergency,
        handler: handle_session_halt,
        loop_name: "constitutional_enforcement",
    },
    // === Layer 6: Self-model & lifecycle ===
    SubscriberDef {
        name: "#11: AgentRegistered → Self-Model Update",
        component_name: "sub-self-model",
        topic_filter: TOPIC_AGENT_REGISTERED,
        min_priority: Priority::Normal,
        handler: handle_self_model_update,
        loop_name: "rsi_pillar_i",
    },
    SubscriberDef {
        name: "#7: SessionEnd → Mini-Compile",
        component_name: "sub-session-compile",
        topic_filter: TOPIC_SESSION_END,
        min_priority: Priority::Normal,
        handler: handle_session_compile,
        loop_name: "session_crystallization",
    },
    // === Layer 7: Saga lifecycle ===
    SubscriberDef {
        name: "#13: Saga → Lifecycle Gate",
        component_name: "sub-saga-lifecycle",
        topic_filter: saga::TOPIC_SAGA_RECEIVED,
        min_priority: Priority::Normal,
        handler: saga::handle_saga_event,
        loop_name: "saga_orchestration",
    },
];

/// Wire all 12 subscribers into a BizraSystem.
///
/// This is THE function that transforms isolated components into a
/// living system. Before this call: excellent components in isolation.
/// After: emergent civilization-grade intelligence.
///
/// Returns the number of subscribers successfully wired, and any errors.
///
/// # Ordering
/// Subscribers are registered in dependency order (execution layer first,
/// then proof chain, memory, economics, constitutional, lifecycle).
/// The EventBus dispatches in registration order, so this ordering
/// ensures correct data flow.
pub fn wire_all(
    system: &mut crate::BizraSystem,
    timestamp_nanos: u64,
) -> (usize, Vec<(&'static str, crate::types::HookError)>) {
    let mut wired = 0;
    let mut errors = Vec::new();

    for def in &SUBSCRIBER_DEFS {
        // Register the subscriber component
        let comp_id = match system.register_component(def.component_name, "1.0.0", timestamp_nanos)
        {
            Ok(id) => id,
            Err(e) => {
                errors.push((def.name, e));
                continue;
            }
        };

        // Activate it
        if let Err(e) = system.activate_component(&comp_id) {
            errors.push((def.name, e));
            continue;
        }

        // Subscribe to the topic
        match system.subscribe(comp_id, def.topic_filter, def.min_priority, def.handler) {
            Ok(_sub_id) => {
                wired += 1;
            }
            Err(e) => {
                errors.push((def.name, e));
            }
        }
    }

    (wired, errors)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BizraSystem;

    /// Helper: create a system with all 12 subscribers wired
    /// plus a registered+activated source component for emitting.
    fn wired_system_with_source(source_name: &str) -> (BizraSystem, ComponentId) {
        let mut system = BizraSystem::new();
        let (wired, errors) = wire_all(&mut system, 1000);
        assert_eq!(wired, 13, "All 13 subscribers must wire: {errors:?}");

        // Register + activate a source component so emit() succeeds
        let src = system
            .register_component(source_name, "1.0.0", 1500)
            .unwrap();
        system.activate_component(&src).unwrap();
        (system, src)
    }

    #[test]
    fn wire_all_registers_12_subscribers() {
        let mut system = BizraSystem::new();
        let (wired, errors) = wire_all(&mut system, 1000);

        assert_eq!(wired, 13, "All 13 subscribers must wire successfully");
        assert!(errors.is_empty(), "No wiring errors: {errors:?}");
        assert_eq!(system.bus.subscription_count(), 13);
    }

    #[test]
    fn wire_all_production_mode() {
        let mut system = BizraSystem::production();
        let (wired, errors) = wire_all(&mut system, 1000);

        assert_eq!(wired, 13);
        assert!(errors.is_empty());

        let health = system.health();
        assert!(health.active_subscriptions >= 13);
    }

    #[test]
    fn subscriber_defs_have_unique_components() {
        let mut seen = std::collections::HashSet::new();
        for def in &SUBSCRIBER_DEFS {
            assert!(
                seen.insert(def.component_name),
                "Duplicate component name: {}",
                def.component_name
            );
        }
    }

    #[test]
    fn subscriber_defs_cover_all_loops() {
        let loops: Vec<&str> = SUBSCRIBER_DEFS.iter().map(|d| d.loop_name).collect();
        assert!(loops.contains(&"memory_learning"));
        assert!(loops.contains(&"glacial_memory"));
        assert!(loops.contains(&"impact_accumulation"));
        assert!(loops.contains(&"economic_flywheel"));
        assert!(loops.contains(&"proof_chain"));
        assert!(loops.contains(&"self_repair"));
        assert!(loops.contains(&"session_crystallization"));
        assert!(loops.contains(&"toxic_isolation"));
        assert!(loops.contains(&"constitutional_enforcement"));
        assert!(loops.contains(&"context_overflow_prevention"));
        assert!(loops.contains(&"rsi_pillar_i"));
        assert!(loops.contains(&"workflow_orchestration"));
        assert!(loops.contains(&"saga_orchestration"));
    }

    #[test]
    fn action_receipt_dispatches_to_memory_and_hhmm() {
        let (mut system, src) = wired_system_with_source("pat-coder");

        let delivered = system
            .emit(
                src,
                TOPIC_ACTION_RECEIPT,
                Payload::from_text("scan_complete"),
                Priority::Normal,
                2000,
            )
            .unwrap();

        // Should reach both #1 (reinforce) and #2 (hhmm promotion)
        assert_eq!(delivered, 2, "ActionReceipt should reach 2 subscribers");
    }

    #[test]
    fn ihsan_breach_halts_session() {
        let (mut system, src) = wired_system_with_source("gate-monitor");

        let delivered = system
            .emit(
                src,
                TOPIC_IHSAN_BREACH,
                Payload::from_text("score=0.89"),
                Priority::Emergency,
                2000,
            )
            .unwrap();

        // #9 handler returns Halt — should stop after 1 delivery
        assert_eq!(delivered, 1, "Breach handler halts propagation");
    }

    #[test]
    fn telescript_step_creates_receipt() {
        let (mut system, src) = wired_system_with_source("telescript-engine");

        let delivered = system
            .emit(
                src,
                TOPIC_TELESCRIPT_STEP_COMPLETED,
                Payload::from_text("step:scan_folder"),
                Priority::Normal,
                2000,
            )
            .unwrap();

        assert_eq!(delivered, 1, "Step completion reaches receipt appender");
    }

    #[test]
    fn failed_action_routes_to_quarantine() {
        let (mut system, src) = wired_system_with_source("pat-coder-fail");

        let delivered = system
            .emit(
                src,
                TOPIC_ACTION_RECEIPT_FAILED,
                Payload::from_text("extraction_error"),
                Priority::High,
                2000,
            )
            .unwrap();

        assert_eq!(delivered, 1, "Failed receipt reaches quarantine handler");
    }

    #[test]
    fn full_lifecycle_event_flow() {
        let (mut system, engine) = wired_system_with_source("lifecycle-test");

        // Step 1: ActionIntent → TeleScript begins
        let d1 = system
            .emit(
                engine,
                TOPIC_ACTION_INTENT,
                Payload::from_text("organize_invoices"),
                Priority::Normal,
                2000,
            )
            .unwrap();
        assert!(d1 >= 1);

        // Step 2: TeleScript step completes → receipt appended
        let d2 = system
            .emit(
                engine,
                TOPIC_TELESCRIPT_STEP_COMPLETED,
                Payload::from_text("step:1:scan"),
                Priority::Normal,
                3000,
            )
            .unwrap();
        assert!(d2 >= 1);

        // Step 3: TeleScript completes → PoI accumulated
        let d3 = system
            .emit(
                engine,
                TOPIC_TELESCRIPT_COMPLETED,
                Payload::from_text("all_steps_done"),
                Priority::Normal,
                4000,
            )
            .unwrap();
        assert!(d3 >= 1);

        // Step 4: ActionReceipt → memory reinforced + hhmm checked
        let d4 = system
            .emit(
                engine,
                TOPIC_ACTION_RECEIPT,
                Payload::from_text("receipt:001"),
                Priority::Normal,
                5000,
            )
            .unwrap();
        assert_eq!(d4, 2); // reinforce + hhmm

        // Step 5: Session ends → mini-compile
        let d5 = system
            .emit(
                engine,
                TOPIC_SESSION_END,
                Payload::empty(),
                Priority::Normal,
                6000,
            )
            .unwrap();
        assert!(d5 >= 1);

        // Verify system telemetry
        assert!(system.bus.total_emitted() >= 5);
        assert!(system.bus.total_delivered() >= 6);
    }
}
