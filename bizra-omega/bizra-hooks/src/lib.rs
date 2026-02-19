//! # BIZRA Hook System — The Sovereign Nervous System
//!
//! `bizra-hooks` is the foundational crate of the BIZRA distributed AI platform.
//! It provides the event-driven backbone through which all components communicate,
//! register, and are governed by constitutional quality constraints.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    BizraSystem (Facade)                     │
//! ├─────────────┬──────────────┬──────────────┬─────────────────┤
//! │  Registry   │   EventBus   │ HookPipeline │  IhsanGate     │
//! │  (Pillar I) │  (Routing)   │  (Chain)     │  (Pillar V)    │
//! │             │              │              │                 │
//! │ Components  │ Pub/Sub      │ PreEmit      │ Lyapunov cert   │
//! │ Dependencies│ Topics       │ Route        │ Floor: 0.990   │
//! │ Self-Model  │ Priority     │ PreDeliver   │ Delta bounds   │
//! │ Snapshots   │ Queue        │ PostDeliver  │ Stability      │
//! └─────────────┴──────────────┴──────────────┴─────────────────┘
//! ```
//!
//! ## RSI Pillar Coverage
//!
//! | RSI Pillar | Component | Role |
//! |---|---|---|
//! | I: Self-Modeling | Registry | Live queryable architecture graph |
//! | IV: Safe Deployment | HookPipeline | Gated event processing |
//! | V: Stable Iteration | IhsanGate | Lyapunov certificate enforcer |
//!
//! ## Zero Dependencies
//!
//! This crate has zero external dependencies. It is pure Rust, `no_std`-compatible
//! at its core, and suitable for WASM, embedded, and bare-metal targets.
//! The nervous system depends on nothing. Everything depends on it.
//!
//! ## Quick Start
//!
//! ```rust
//! use bizra_hooks::BizraSystem;
//! use bizra_hooks::types::*;
//!
//! let mut system = BizraSystem::new();
//!
//! // Register a component
//! let mem_id = system.register_component("memory-engine", "1.0.0", 0).unwrap();
//! system.activate_component(&mem_id).unwrap();
//!
//! // Subscribe to events
//! system.subscribe(mem_id, "agent.*", Priority::Normal, |event| {
//!     // Process event
//!     HookResult::Continue
//! }).unwrap();
//!
//! // Emit an event
//! system.emit(mem_id, "agent.query", Payload::from_str("find context"), Priority::Normal, 1000);
//!
//! // Check system health
//! let health = system.health();
//! assert!(health.system_ihsan.meets_ihsan());
//! ```

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Module declarations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub mod types;
pub mod registry;
pub mod event_bus;
pub mod pipeline;
pub mod ihsan_gate;

// Re-exports for ergonomic usage
pub use types::*;
pub use registry::{Registry, RegistrySnapshot, Dependency, DependencyKind};
pub use event_bus::{EventBus, Subscription, EventHandler};
pub use pipeline::{HookPipeline, HookFn};
pub use ihsan_gate::{IhsanGate, GateConfig, GatePolicy, GateVerdict, GateAction};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BizraSystem — The Unified Facade
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// System-wide health snapshot.
#[derive(Debug, Clone, Copy)]
pub struct SystemHealth {
    /// Registry snapshot (component counts, statuses)
    pub registry: RegistrySnapshot,
    /// System-wide إحسان score
    pub system_ihsan: IhsanScore,
    /// EventBus stats
    pub events_emitted: u64,
    pub events_delivered: u64,
    pub events_dropped: u64,
    pub delivery_ratio: f64,
    /// Pipeline stats
    pub pipeline_processed: u64,
    pub pipeline_halted: u64,
    pub pipeline_pass_rate: f64,
    /// إحسان Gate stats
    pub gate_evaluations: u64,
    pub gate_violations: u64,
    pub gate_stability: f64,
    pub consecutive_stable: u64,
    /// Active subscriptions
    pub active_subscriptions: usize,
    /// Total hooks registered
    pub total_hooks: usize,
}

/// The unified BIZRA system — wires all subsystems together.
///
/// This is the single entry point for all nervous system operations.
/// Components register here, events flow through here, quality is
/// enforced here. If it's not in the BizraSystem, it doesn't exist.
pub struct BizraSystem {
    /// Component Registry (RSI Pillar I: Self-Model)
    pub registry: Registry,
    /// Event Bus (nervous system routing)
    pub bus: EventBus,
    /// Hook Pipeline (processing chain)
    pub pipeline: HookPipeline,
    /// إحسان Gate (RSI Pillar V: Lyapunov enforcement)
    pub gate: IhsanGate,
    /// Whether the إحسان gate is wired as a PreEmit hook
    gate_active: bool,
}

impl BizraSystem {
    /// Create a new BizraSystem with default configuration.
    pub fn new() -> Self {
        BizraSystem {
            registry: Registry::new(),
            bus: EventBus::new(),
            pipeline: HookPipeline::new(),
            gate: IhsanGate::new(),
            gate_active: false,
        }
    }

    /// Create with custom إحسان gate configuration.
    pub fn with_gate_config(config: GateConfig) -> Self {
        BizraSystem {
            registry: Registry::new(),
            bus: EventBus::new(),
            pipeline: HookPipeline::new(),
            gate: IhsanGate::with_config(config),
            gate_active: false,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Component Lifecycle
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Register a new component. Returns its ComponentId.
    pub fn register_component(
        &mut self,
        name: &str,
        version: &str,
        timestamp_nanos: u64,
    ) -> Result<ComponentId, HookError> {
        let id = self.registry.register(name, version, timestamp_nanos)?;

        // Emit lifecycle event
        self.bus.emit_simple(
            id,
            "system.lifecycle",
            Payload::from_str("registered"),
            Priority::High,
            timestamp_nanos,
        );

        Ok(id)
    }

    /// Activate a component (ready to send/receive events).
    pub fn activate_component(&mut self, id: &ComponentId) -> Result<(), HookError> {
        self.registry.activate(id)
    }

    /// Suspend a component (stops receiving events).
    pub fn suspend_component(&mut self, id: &ComponentId) -> Result<(), HookError> {
        self.registry.suspend(id)?;
        self.bus.set_active_for_component(id, false);
        Ok(())
    }

    /// Unregister a component. Removes from registry and all subscriptions.
    pub fn unregister_component(
        &mut self,
        id: &ComponentId,
        timestamp_nanos: u64,
    ) -> Result<ComponentMeta, HookError> {
        // Emit lifecycle event before removal
        self.bus.emit_simple(
            *id,
            "system.lifecycle",
            Payload::from_str("unregistering"),
            Priority::High,
            timestamp_nanos,
        );

        // Remove subscriptions
        self.bus.unsubscribe_all(id);

        // Remove from registry (also cleans dependency edges)
        self.registry.unregister(id)
    }

    /// Declare a dependency between components.
    pub fn add_dependency(
        &mut self,
        from: ComponentId,
        to: ComponentId,
        kind: DependencyKind,
    ) -> Result<(), HookError> {
        self.registry.add_dependency(from, to, kind)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Event Operations (high-level)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Subscribe a component to events matching a topic.
    pub fn subscribe(
        &mut self,
        component: ComponentId,
        topic: &str,
        min_priority: Priority,
        handler: EventHandler,
    ) -> Result<SubscriptionId, HookError> {
        // Verify component exists
        if self.registry.get(&component).is_none() {
            return Err(HookError::ComponentNotFound(component));
        }
        self.bus.subscribe(component, topic, min_priority, handler)
    }

    /// Emit an event through the full pipeline.
    ///
    /// Flow: Construct → Pipeline(PreEmit/Route/PreDeliver) → إحسان Gate → Bus → Pipeline(PostDeliver)
    pub fn emit(
        &mut self,
        source: ComponentId,
        topic: &str,
        payload: Payload,
        priority: Priority,
        timestamp_nanos: u64,
    ) -> Result<usize, HookError> {
        // Verify source is registered and active
        let meta = self.registry.get(&source)
            .ok_or(HookError::ComponentNotFound(source))?;

        if meta.status != ComponentStatus::Active {
            return Err(HookError::ComponentInactive(source));
        }

        // Construct event
        let id = self.bus.next_event_id(timestamp_nanos);
        let ihsan = meta.ihsan;

        let event = Event {
            id,
            source,
            topic: Topic::new(topic),
            priority,
            payload,
            ihsan_score: ihsan,
        };

        // Run through pipeline (PreEmit → Route → PreDeliver)
        let processed = self.pipeline.process_pre_delivery(event)?;

        // إحسان Gate evaluation
        if self.gate_active {
            let verdict = self.gate.evaluate(&processed);
            match verdict.action {
                GateAction::Rejected => {
                    return Err(HookError::IhsanGateRejected(verdict.score));
                }
                GateAction::Throttled => {
                    return Err(HookError::IhsanGateRejected(verdict.score));
                }
                _ => {} // Allow and Flagged pass through
            }
        }

        // Emit through bus
        let delivered = self.bus.emit(processed);

        // Track in registry
        self.registry.record_emit(&source, timestamp_nanos);

        // PostDeliver hooks
        self.pipeline.process_post_delivery(&processed);

        Ok(delivered)
    }

    /// Emit a raw event (bypasses component check — for system-level events).
    pub fn emit_raw(&mut self, event: Event) -> usize {
        self.bus.emit(event)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // إحسان Gate Control
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Activate the إحسان gate as a pre-emit filter.
    pub fn activate_gate(&mut self) {
        self.gate_active = true;
    }

    /// Deactivate the إحسان gate (observation only).
    pub fn deactivate_gate(&mut self) {
        self.gate_active = false;
    }

    /// Set the gate enforcement policy.
    pub fn set_gate_policy(&mut self, policy: GatePolicy) {
        self.gate.set_policy(policy);
    }

    /// Update a component's إحسان score.
    pub fn update_ihsan(
        &mut self,
        id: &ComponentId,
        score: IhsanScore,
    ) -> Result<(), HookError> {
        self.registry.update_ihsan(id, score)?;
        self.gate.set_score(*id, score, 0);
        Ok(())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Hook Pipeline Control
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Register a hook into the processing pipeline.
    pub fn register_hook(
        &mut self,
        phase: HookPhase,
        name: &str,
        priority: u8,
        hook_fn: HookFn,
    ) -> Result<HookId, HookError> {
        self.pipeline.register(phase, name, priority, hook_fn)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Health & Observability
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get comprehensive system health snapshot.
    pub fn health(&self) -> SystemHealth {
        let reg_snap = self.registry.snapshot();

        SystemHealth {
            registry: reg_snap,
            system_ihsan: self.registry.system_ihsan(),
            events_emitted: self.bus.total_emitted(),
            events_delivered: self.bus.total_delivered(),
            events_dropped: self.bus.total_dropped(),
            delivery_ratio: self.bus.delivery_ratio(),
            pipeline_processed: self.pipeline.total_processed(),
            pipeline_halted: self.pipeline.total_halted(),
            pipeline_pass_rate: self.pipeline.pass_rate(),
            gate_evaluations: self.gate.total_evaluations(),
            gate_violations: self.gate.total_violations(),
            gate_stability: self.gate.stability_score(),
            consecutive_stable: self.gate.consecutive_stable(),
            active_subscriptions: self.bus.subscription_count(),
            total_hooks: self.pipeline.total_hooks(),
        }
    }

    /// Quick check: is the system healthy?
    pub fn is_healthy(&self) -> bool {
        let health = self.health();
        health.system_ihsan.meets_ihsan()
            && health.registry.failed == 0
            && health.gate_stability > 0.95
    }

    /// Get system-wide إحسان score.
    pub fn system_ihsan(&self) -> IhsanScore {
        self.registry.system_ihsan()
    }
}

impl Default for BizraSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Integration Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_event_handler(_event: &Event) -> HookResult {
        HookResult::Continue
    }

    #[test]
    fn full_system_lifecycle() {
        let mut sys = BizraSystem::new();

        // Register components
        let mem = sys.register_component("memory-engine", "1.0.0", 1000).unwrap();
        let agent = sys.register_component("agent-runtime", "1.0.0", 1001).unwrap();

        // Activate
        sys.activate_component(&mem).unwrap();
        sys.activate_component(&agent).unwrap();

        // Add dependency
        sys.add_dependency(agent, mem, DependencyKind::Required).unwrap();

        // Subscribe agent to memory events
        sys.subscribe(agent, "memory.*", Priority::Normal, noop_event_handler).unwrap();

        // Emit event from memory engine
        let delivered = sys.emit(
            mem,
            "memory.indexed",
            Payload::from_str("500 vectors"),
            Priority::Normal,
            2000,
        ).unwrap();

        assert_eq!(delivered, 1);

        // Check health
        let health = sys.health();
        assert_eq!(health.registry.total_components, 2);
        assert_eq!(health.registry.active, 2);
        assert!(health.system_ihsan.meets_ihsan());
        assert_eq!(health.events_emitted, 3); // 2 lifecycle + 1 user event
    }

    #[test]
    fn ihsan_gate_blocks_degraded() {
        let mut sys = BizraSystem::with_gate_config(GateConfig {
            policy: GatePolicy::Reject,
            ..Default::default()
        });
        sys.activate_gate();

        let comp = sys.register_component("degraded", "1.0.0", 1000).unwrap();
        sys.activate_component(&comp).unwrap();

        // Degrade the component
        sys.update_ihsan(&comp, IhsanScore::from_f64(0.50)).unwrap();

        // Try to emit — should be rejected by gate
        let result = sys.emit(
            comp,
            "test.event",
            Payload::empty(),
            Priority::Normal,
            2000,
        );

        assert!(matches!(result, Err(HookError::IhsanGateRejected(_))));
    }

    #[test]
    fn inactive_component_cannot_emit() {
        let mut sys = BizraSystem::new();
        let comp = sys.register_component("lazy", "1.0.0", 1000).unwrap();
        // Don't activate — component is in Registered state

        let result = sys.emit(comp, "test.event", Payload::empty(), Priority::Normal, 2000);
        assert!(matches!(result, Err(HookError::ComponentInactive(_))));
    }

    #[test]
    fn unregistered_component_cannot_subscribe() {
        let mut sys = BizraSystem::new();
        let fake = ComponentId::from_name("ghost", "1.0.0");

        let result = sys.subscribe(fake, "test.*", Priority::Normal, noop_event_handler);
        assert!(matches!(result, Err(HookError::ComponentNotFound(_))));
    }

    #[test]
    fn system_health_snapshot() {
        let mut sys = BizraSystem::new();

        // Fresh system should be perfectly healthy
        assert!(sys.is_healthy());

        let comp = sys.register_component("worker", "1.0.0", 1000).unwrap();
        sys.activate_component(&comp).unwrap();

        // Degrade it
        sys.update_ihsan(&comp, IhsanScore::from_f64(0.80)).unwrap();
        assert!(!sys.is_healthy()); // System إحسان drops below floor
    }

    #[test]
    fn hook_pipeline_integration() {
        let mut sys = BizraSystem::new();

        // Register a PreEmit hook that passes everything
        sys.register_hook(
            HookPhase::PreEmit,
            "validator",
            0,
            |_| (HookResult::Continue, None),
        ).unwrap();

        let comp = sys.register_component("hooked", "1.0.0", 1000).unwrap();
        sys.activate_component(&comp).unwrap();

        let result = sys.emit(comp, "test.event", Payload::empty(), Priority::Normal, 2000);
        assert!(result.is_ok());

        assert_eq!(sys.pipeline.total_processed(), 1);
        assert_eq!(sys.pipeline.total_passed(), 1);
    }

    #[test]
    fn concurrent_components_scale() {
        let mut sys = BizraSystem::new();

        // Register 50 components
        let mut ids = Vec::new();
        for i in 0..50 {
            let name = format!("component-{}", i);
            let id = sys.register_component(&name, "1.0.0", i as u64).unwrap();
            sys.activate_component(&id).unwrap();
            ids.push(id);
        }

        // Each subscribes to a wildcard
        for &id in &ids {
            sys.subscribe(id, "broadcast.*", Priority::Low, noop_event_handler).unwrap();
        }

        // Broadcast from first component
        let delivered = sys.emit(
            ids[0],
            "broadcast.ping",
            Payload::from_str("alive"),
            Priority::Normal,
            10000,
        ).unwrap();

        // All 50 should receive (including the sender)
        assert_eq!(delivered, 50);

        let health = sys.health();
        assert_eq!(health.registry.total_components, 50);
        assert_eq!(health.active_subscriptions, 50);
    }

    #[test]
    fn dependency_impact_analysis() {
        let mut sys = BizraSystem::new();

        let core = sys.register_component("core-engine", "1.0.0", 1000).unwrap();
        let a = sys.register_component("agent-a", "1.0.0", 1001).unwrap();
        let b = sys.register_component("agent-b", "1.0.0", 1002).unwrap();

        sys.add_dependency(a, core, DependencyKind::Required).unwrap();
        sys.add_dependency(b, core, DependencyKind::Required).unwrap();

        // Removing core would break 2 components
        assert_eq!(sys.registry.removal_impact(&core), 2);
    }
}
