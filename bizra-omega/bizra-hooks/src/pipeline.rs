//! # Hook Pipeline — The Processing Chain
//!
//! The HookChain is an ordered sequence of hook functions that process events
//! at each phase of their lifecycle. This is where all the gates, transforms,
//! and enrichments compose into a single pipeline.
//!
//! ## Architecture
//! ```text
//! Event → [PreEmit hooks] → [Route hooks] → [PreDeliver hooks] → Delivery → [PostDeliver hooks]
//!              ↑                                                       ↑
//!         إحسان Gate                                              Telemetry
//!         (Lyapunov check)                                       (scoring)
//! ```
//!
//! ## Design
//! - Fixed-capacity hook array per phase (no heap allocation)
//! - Hooks execute in priority order within each phase
//! - Any hook can Halt the chain (constitutional veto)
//! - Transform hooks can modify events in-flight

use crate::types::*;

/// Maximum hooks per phase.
const MAX_HOOKS_PER_PHASE: usize = 32;

/// Total number of phases.
const PHASE_COUNT: usize = 4;

/// A hook function: takes an event, returns a result and optionally a modified event.
/// If Transform is returned, the `Event` in the output is used going forward.
pub type HookFn = fn(&Event) -> (HookResult, Option<Event>);

/// A registered hook entry.
#[derive(Clone, Copy)]
struct HookEntry {
    id: HookId,
    #[allow(dead_code)]
    name: Name,
    priority: u8, // 0 = highest priority (runs first within phase)
    hook_fn: HookFn,
    enabled: bool,
    /// Number of times this hook has been invoked
    invocations: u64,
    /// Number of times this hook returned Halt
    halts: u64,
}

/// Phase-specific hook array.
struct PhaseChain {
    hooks: [Option<HookEntry>; MAX_HOOKS_PER_PHASE],
    count: usize,
}

impl PhaseChain {
    const fn new() -> Self {
        PhaseChain {
            hooks: [None; MAX_HOOKS_PER_PHASE],
            count: 0,
        }
    }

    /// Add a hook to this phase chain, maintaining priority order.
    fn add(&mut self, entry: HookEntry) -> Result<(), HookError> {
        if self.count >= MAX_HOOKS_PER_PHASE {
            return Err(HookError::HookChainFull);
        }

        // Find insertion point (sorted by priority ascending = highest priority first)
        let pos = self.hooks[..self.count]
            .iter()
            .position(|h| {
                h.as_ref()
                    .map(|existing| entry.priority < existing.priority)
                    .unwrap_or(true)
            })
            .unwrap_or(self.count);

        // Shift elements right
        if pos < self.count {
            for i in (pos..self.count).rev() {
                self.hooks[i + 1] = self.hooks[i];
            }
        }

        self.hooks[pos] = Some(entry);
        self.count += 1;
        Ok(())
    }

    /// Remove a hook by ID.
    fn remove(&mut self, id: HookId) -> bool {
        for i in 0..self.count {
            if let Some(h) = &self.hooks[i] {
                if h.id == id {
                    // Shift left to fill gap
                    for j in i..self.count - 1 {
                        self.hooks[j] = self.hooks[j + 1];
                    }
                    self.hooks[self.count - 1] = None;
                    self.count -= 1;
                    return true;
                }
            }
        }
        false
    }

    /// Execute all hooks in this phase against an event.
    /// Returns the final event (possibly transformed) and whether to continue.
    fn execute(&mut self, mut event: Event) -> (Event, HookResult) {
        for i in 0..self.count {
            if let Some(hook) = &mut self.hooks[i] {
                if !hook.enabled {
                    continue;
                }

                hook.invocations += 1;
                let (result, transformed) = (hook.hook_fn)(&event);

                match result {
                    HookResult::Continue => {
                        if let Some(new_event) = transformed {
                            event = new_event;
                        }
                    }
                    HookResult::Transform => {
                        if let Some(new_event) = transformed {
                            event = new_event;
                        }
                        // Continue with transformed event
                    }
                    HookResult::Skip => {
                        return (event, HookResult::Skip);
                    }
                    HookResult::Halt => {
                        hook.halts += 1;
                        return (event, HookResult::Halt);
                    }
                }
            }
        }

        (event, HookResult::Continue)
    }
}

/// The complete Hook Pipeline — all four phases.
pub struct HookPipeline {
    /// Phase chains indexed by HookPhase discriminant
    phases: [PhaseChain; PHASE_COUNT],
    /// Next hook ID
    next_id: u64,
    /// Total events processed through the pipeline
    total_processed: u64,
    /// Events halted by pipeline hooks
    total_halted: u64,
    /// Events that passed all phases
    total_passed: u64,
}

impl HookPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        HookPipeline {
            phases: [
                PhaseChain::new(),
                PhaseChain::new(),
                PhaseChain::new(),
                PhaseChain::new(),
            ],
            next_id: 1,
            total_processed: 0,
            total_halted: 0,
            total_passed: 0,
        }
    }

    /// Register a hook into a specific phase.
    ///
    /// - `phase`: When this hook executes
    /// - `name`: Human-readable hook name
    /// - `priority`: 0 = runs first, 255 = runs last
    /// - `hook_fn`: The hook function
    pub fn register(
        &mut self,
        phase: HookPhase,
        name: &str,
        priority: u8,
        hook_fn: HookFn,
    ) -> Result<HookId, HookError> {
        let id = HookId::new(self.next_id);
        self.next_id += 1;

        let entry = HookEntry {
            id,
            name: Name::new(name),
            priority,
            hook_fn,
            enabled: true,
            invocations: 0,
            halts: 0,
        };

        let phase_idx = phase as usize;
        self.phases[phase_idx].add(entry)?;

        Ok(id)
    }

    /// Unregister a hook by ID (searches all phases).
    pub fn unregister(&mut self, id: HookId) -> bool {
        for phase in &mut self.phases {
            if phase.remove(id) {
                return true;
            }
        }
        false
    }

    /// Enable or disable a hook by ID.
    pub fn set_enabled(&mut self, id: HookId, enabled: bool) -> bool {
        for phase in &mut self.phases {
            for i in 0..phase.count {
                if let Some(h) = &mut phase.hooks[i] {
                    if h.id == id {
                        h.enabled = enabled;
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Process an event through the full pipeline.
    ///
    /// Runs: PreEmit → Route → PreDeliver phases.
    /// PostDeliver is run separately after actual delivery.
    ///
    /// Returns `Ok(processed_event)` if event should be delivered,
    /// or `Err(HookError)` if halted.
    pub fn process_pre_delivery(&mut self, event: Event) -> Result<Event, HookError> {
        self.total_processed += 1;

        // Phase 1: PreEmit (validation, إحسان gate)
        let (event, result) = self.phases[HookPhase::PreEmit as usize].execute(event);
        if result == HookResult::Halt {
            self.total_halted += 1;
            return Err(HookError::HookHalted(HookId::new(0)));
        }

        // Phase 2: Route (topic rewriting, priority adjustment)
        let (event, result) = self.phases[HookPhase::Route as usize].execute(event);
        if result == HookResult::Halt {
            self.total_halted += 1;
            return Err(HookError::HookHalted(HookId::new(0)));
        }

        // Phase 3: PreDeliver (transformation, enrichment)
        let (event, result) = self.phases[HookPhase::PreDeliver as usize].execute(event);
        if result == HookResult::Halt {
            self.total_halted += 1;
            return Err(HookError::HookHalted(HookId::new(0)));
        }

        self.total_passed += 1;
        Ok(event)
    }

    /// Run PostDeliver hooks (after event has been delivered to subscribers).
    /// PostDeliver hooks cannot halt — they're for logging, telemetry, scoring.
    pub fn process_post_delivery(&mut self, event: &Event) {
        let owned = *event;
        let _ = self.phases[HookPhase::PostDeliver as usize].execute(owned);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Telemetry
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }

    pub fn total_halted(&self) -> u64 {
        self.total_halted
    }

    pub fn total_passed(&self) -> u64 {
        self.total_passed
    }

    /// Pass-through rate (higher = fewer hooks halting events).
    pub fn pass_rate(&self) -> f64 {
        if self.total_processed == 0 {
            1.0
        } else {
            self.total_passed as f64 / self.total_processed as f64
        }
    }

    /// Number of hooks registered per phase.
    pub fn hooks_per_phase(&self) -> [usize; PHASE_COUNT] {
        [
            self.phases[0].count,
            self.phases[1].count,
            self.phases[2].count,
            self.phases[3].count,
        ]
    }

    /// Total hooks across all phases.
    pub fn total_hooks(&self) -> usize {
        self.phases.iter().map(|p| p.count).sum()
    }
}

impl Default for HookPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass_hook(_event: &Event) -> (HookResult, Option<Event>) {
        (HookResult::Continue, None)
    }

    fn halt_hook(_event: &Event) -> (HookResult, Option<Event>) {
        (HookResult::Halt, None)
    }

    fn upgrade_priority(event: &Event) -> (HookResult, Option<Event>) {
        let mut modified = *event;
        modified.priority = Priority::Critical;
        (HookResult::Transform, Some(modified))
    }

    fn make_event() -> Event {
        Event {
            id: EventId::new(1000, 0),
            source: ComponentId::from_name("test", "1.0.0"),
            topic: Topic::new("test.event"),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::MAX,
        }
    }

    #[test]
    fn empty_pipeline_passes_all() {
        let mut pipeline = HookPipeline::new();
        let event = make_event();

        let result = pipeline.process_pre_delivery(event);
        assert!(result.is_ok());
        assert_eq!(pipeline.total_passed(), 1);
    }

    #[test]
    fn pre_emit_halt_blocks_delivery() {
        let mut pipeline = HookPipeline::new();
        pipeline
            .register(HookPhase::PreEmit, "blocker", 0, halt_hook)
            .unwrap();

        let result = pipeline.process_pre_delivery(make_event());
        assert!(result.is_err());
        assert_eq!(pipeline.total_halted(), 1);
        assert_eq!(pipeline.total_passed(), 0);
    }

    #[test]
    fn transform_modifies_event() {
        let mut pipeline = HookPipeline::new();
        pipeline
            .register(HookPhase::Route, "upgrader", 0, upgrade_priority)
            .unwrap();

        let event = make_event();
        assert_eq!(event.priority, Priority::Normal);

        let result = pipeline.process_pre_delivery(event).unwrap();
        assert_eq!(result.priority, Priority::Critical);
    }

    #[test]
    fn priority_ordering_within_phase() {
        let mut pipeline = HookPipeline::new();

        // Register in reverse priority order
        pipeline
            .register(HookPhase::PreEmit, "last", 10, pass_hook)
            .unwrap();
        pipeline
            .register(HookPhase::PreEmit, "first", 0, pass_hook)
            .unwrap();
        pipeline
            .register(HookPhase::PreEmit, "middle", 5, pass_hook)
            .unwrap();

        let counts = pipeline.hooks_per_phase();
        assert_eq!(counts[0], 3); // 3 hooks in PreEmit
    }

    #[test]
    fn disable_hook_skips_execution() {
        let mut pipeline = HookPipeline::new();
        let id = pipeline
            .register(HookPhase::PreEmit, "blocker", 0, halt_hook)
            .unwrap();

        // Should halt
        assert!(pipeline.process_pre_delivery(make_event()).is_err());

        // Disable the halt hook
        pipeline.set_enabled(id, false);

        // Should now pass
        assert!(pipeline.process_pre_delivery(make_event()).is_ok());
    }

    #[test]
    fn unregister_removes_hook() {
        let mut pipeline = HookPipeline::new();
        let id = pipeline
            .register(HookPhase::PreEmit, "temp", 0, halt_hook)
            .unwrap();

        assert_eq!(pipeline.total_hooks(), 1);
        pipeline.unregister(id);
        assert_eq!(pipeline.total_hooks(), 0);
    }

    #[test]
    fn multi_phase_execution() {
        let mut pipeline = HookPipeline::new();

        pipeline
            .register(HookPhase::PreEmit, "validate", 0, pass_hook)
            .unwrap();
        pipeline
            .register(HookPhase::Route, "route", 0, pass_hook)
            .unwrap();
        pipeline
            .register(HookPhase::PreDeliver, "enrich", 0, pass_hook)
            .unwrap();
        pipeline
            .register(HookPhase::PostDeliver, "log", 0, pass_hook)
            .unwrap();

        let event = make_event();
        let processed = pipeline.process_pre_delivery(event).unwrap();

        // PostDeliver runs separately
        pipeline.process_post_delivery(&processed);

        assert_eq!(pipeline.total_processed(), 1);
        assert_eq!(pipeline.total_passed(), 1);
    }

    #[test]
    fn pass_rate_calculation() {
        let mut pipeline = HookPipeline::new();

        // Process 2 events that pass
        pipeline.process_pre_delivery(make_event()).unwrap();
        pipeline.process_pre_delivery(make_event()).unwrap();

        assert!((pipeline.pass_rate() - 1.0).abs() < f64::EPSILON);

        // Add a halt hook and process 1 more
        pipeline
            .register(HookPhase::PreEmit, "halt", 0, halt_hook)
            .unwrap();
        let _ = pipeline.process_pre_delivery(make_event());

        // 2 passed out of 3 total
        assert!((pipeline.pass_rate() - 2.0 / 3.0).abs() < 0.01);
    }
}
