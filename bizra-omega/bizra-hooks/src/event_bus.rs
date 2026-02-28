//! # Event Bus — The Nervous System
//!
//! Every signal in BIZRA flows through the EventBus. Components subscribe
//! to topics, emit events, and the bus routes them with priority ordering.
//!
//! ## Design
//! - Fixed-capacity subscriber array (no heap allocation)
//! - Topic-based routing with wildcard support
//! - Priority queue ordering: Emergency > Critical > High > Normal > Low
//! - Integration with HookChain for pre/post processing
//! - Integration with Registry for emit/consume tracking

use crate::types::*;

/// Number of topic-namespace shards (power of 2 for fast modulo).
const NUM_SHARDS: usize = 8;

/// Capacity per shard. NUM_SHARDS * SHARD_CAPACITY = 512 total slots.
const SHARD_CAPACITY: usize = 64;

/// Maximum pending events in the dispatch queue.
const MAX_PENDING: usize = 256;

/// Map a topic to its shard index via FNV-1a hash of the namespace prefix.
/// The namespace is everything before the first `.` in the topic string.
fn shard_index(topic: &Topic) -> usize {
    let s = topic.as_str();
    let prefix_end = s
        .as_bytes()
        .iter()
        .position(|&b| b == b'.')
        .unwrap_or(s.len());
    // FNV-1a 32-bit hash of the prefix bytes
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in &s.as_bytes()[..prefix_end] {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    (hash as usize) & (NUM_SHARDS - 1) // Bitwise AND since NUM_SHARDS is power of 2
}

/// A subscription: component + topic filter + priority filter.
#[derive(Clone, Copy)]
pub struct Subscription {
    pub id: SubscriptionId,
    pub component: ComponentId,
    pub topic_filter: Topic,
    /// Minimum priority to receive (events below this are filtered)
    pub min_priority: Priority,
    /// Is this subscription active?
    pub active: bool,
}

impl Subscription {
    /// Check if this subscription matches an event.
    pub fn matches(&self, event: &Event) -> bool {
        self.active
            && event.priority >= self.min_priority
            && event.topic.matches(&self.topic_filter)
    }
}

/// Callback function type for event handling.
/// Returns HookResult to allow subscribers to signal halt/skip.
pub type EventHandler = fn(&Event) -> HookResult;

/// A registered handler: subscription + function pointer.
#[derive(Clone, Copy)]
struct HandlerEntry {
    sub: Subscription,
    handler: EventHandler,
}

/// A single shard of the EventBus. Owns handlers for one topic namespace.
#[derive(Clone)]
struct EventShard {
    handlers: [Option<HandlerEntry>; SHARD_CAPACITY],
    sub_count: usize,
}

impl EventShard {
    const fn new() -> Self {
        EventShard {
            handlers: [None; SHARD_CAPACITY],
            sub_count: 0,
        }
    }

    fn subscribe(
        &mut self,
        component: ComponentId,
        topic_filter: &str,
        min_priority: Priority,
        handler: EventHandler,
        sub_id: SubscriptionId,
    ) -> Result<SubscriptionId, HookError> {
        let slot = self
            .handlers
            .iter()
            .position(|h| h.is_none())
            .ok_or(HookError::SubscribersFull)?;

        self.handlers[slot] = Some(HandlerEntry {
            sub: Subscription {
                id: sub_id,
                component,
                topic_filter: Topic::new(topic_filter),
                min_priority,
                active: true,
            },
            handler,
        });
        self.sub_count += 1;
        Ok(sub_id)
    }

    fn unsubscribe(&mut self, sub_id: SubscriptionId) -> bool {
        for entry in self.handlers.iter_mut() {
            if let Some(h) = entry {
                if h.sub.id == sub_id {
                    *entry = None;
                    self.sub_count -= 1;
                    return true;
                }
            }
        }
        false
    }

    fn unsubscribe_all(&mut self, component: &ComponentId) -> usize {
        let mut removed = 0;
        for entry in self.handlers.iter_mut() {
            if let Some(h) = entry {
                if h.sub.component == *component {
                    *entry = None;
                    removed += 1;
                }
            }
        }
        self.sub_count -= removed;
        removed
    }

    fn set_active(&mut self, sub_id: SubscriptionId, active: bool) -> bool {
        for h in self.handlers.iter_mut().flatten() {
            if h.sub.id == sub_id {
                h.sub.active = active;
                return true;
            }
        }
        false
    }

    fn set_active_for_component(&mut self, id: &ComponentId, active: bool) {
        for h in self.handlers.iter_mut().flatten() {
            if h.sub.component == *id {
                h.sub.active = active;
            }
        }
    }

    /// Dispatch an event to all matching handlers in this shard.
    /// Returns (delivered_count, should_halt).
    fn dispatch(&self, event: &Event) -> (usize, bool) {
        let mut delivered = 0;
        for h in self.handlers.iter().flatten() {
            if h.sub.matches(event) {
                let result = (h.handler)(event);
                delivered += 1;
                match result {
                    HookResult::Continue | HookResult::Transform => continue,
                    HookResult::Skip => break,
                    HookResult::Halt => return (delivered, true),
                }
            }
        }
        (delivered, false)
    }

    fn sub_count(&self) -> usize {
        self.sub_count
    }
}

/// The Event Bus — routes events between components via namespace sharding.
///
/// Topics are sharded by their namespace prefix (text before the first `.`).
/// Each shard dispatches independently, eliminating cross-namespace contention.
/// Topics without a `.` route to the global shard for cross-namespace wildcards.
pub struct EventBus {
    /// Per-namespace shards
    shards: [EventShard; NUM_SHARDS],
    /// Global shard for subscriptions without a namespace prefix
    global: EventShard,
    /// Next subscription ID (monotonic counter across all shards)
    next_sub_id: u32,

    /// Pending event queue (priority-ordered deferred dispatch)
    pending: [Option<Event>; MAX_PENDING],
    /// Number of pending events
    pending_count: usize,

    /// Total events emitted through this bus
    total_emitted: u64,
    /// Total events delivered to subscribers
    total_delivered: u64,
    /// Total events dropped (no matching subscriber)
    total_dropped: u64,

    /// Event ID sequence counter (per-nanosecond)
    sequence: u16,
    /// Last timestamp seen (for sequence reset)
    last_timestamp: u64,
}

impl EventBus {
    /// Create a new empty EventBus.
    pub fn new() -> Self {
        const EMPTY_SHARD: EventShard = EventShard::new();
        EventBus {
            shards: [EMPTY_SHARD; NUM_SHARDS],
            global: EventShard::new(),
            next_sub_id: 1,
            pending: [None; MAX_PENDING],
            pending_count: 0,
            total_emitted: 0,
            total_delivered: 0,
            total_dropped: 0,
            sequence: 0,
            last_timestamp: 0,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Subscribe / Unsubscribe
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Subscribe a component to events matching a topic filter.
    pub fn subscribe(
        &mut self,
        component: ComponentId,
        topic_filter: &str,
        min_priority: Priority,
        handler: EventHandler,
    ) -> Result<SubscriptionId, HookError> {
        let sub_id = SubscriptionId(self.next_sub_id);
        self.next_sub_id += 1;

        let topic = Topic::new(topic_filter);
        let shard = self.shard_for_topic_mut(&topic);
        shard.subscribe(component, topic_filter, min_priority, handler, sub_id)
    }

    /// Unsubscribe by subscription ID.
    pub fn unsubscribe(&mut self, sub_id: SubscriptionId) -> bool {
        // Search all shards (subscription could be in any)
        for shard in self.shards.iter_mut() {
            if shard.unsubscribe(sub_id) {
                return true;
            }
        }
        self.global.unsubscribe(sub_id)
    }

    /// Unsubscribe all handlers for a component.
    pub fn unsubscribe_all(&mut self, component: &ComponentId) -> usize {
        let mut removed = 0;
        for shard in self.shards.iter_mut() {
            removed += shard.unsubscribe_all(component);
        }
        removed += self.global.unsubscribe_all(component);
        removed
    }

    /// Pause/resume a subscription.
    pub fn set_active(&mut self, sub_id: SubscriptionId, active: bool) -> bool {
        for shard in self.shards.iter_mut() {
            if shard.set_active(sub_id, active) {
                return true;
            }
        }
        self.global.set_active(sub_id, active)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Emit — Send events into the nervous system
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Generate the next EventId with monotonic ordering.
    pub fn next_event_id(&mut self, timestamp_nanos: u64) -> EventId {
        if timestamp_nanos != self.last_timestamp {
            self.sequence = 0;
            self.last_timestamp = timestamp_nanos;
        } else {
            self.sequence = self.sequence.wrapping_add(1);
        }
        EventId::new(timestamp_nanos, self.sequence)
    }

    /// Emit an event. Dispatches to the target shard + global shard.
    ///
    /// Returns the number of subscribers that received the event.
    pub fn emit(&mut self, event: Event) -> usize {
        self.total_emitted += 1;

        // Dispatch to the target namespace shard
        let shard_idx = shard_index(&event.topic);
        let (shard_delivered, shard_halted) = self.shards[shard_idx].dispatch(&event);

        // Also dispatch to global shard (cross-namespace wildcards)
        let global_delivered = if !shard_halted {
            let (gd, _) = self.global.dispatch(&event);
            gd
        } else {
            0
        };

        let delivered = shard_delivered + global_delivered;
        if delivered == 0 {
            self.total_dropped += 1;
        } else {
            self.total_delivered += delivered as u64;
        }
        delivered
    }

    /// Emit a simple event with topic, payload, and source.
    pub fn emit_simple(
        &mut self,
        source: ComponentId,
        topic: &str,
        payload: Payload,
        priority: Priority,
        timestamp_nanos: u64,
    ) -> usize {
        let id = self.next_event_id(timestamp_nanos);
        let event = Event {
            id,
            source,
            topic: Topic::new(topic),
            priority,
            payload,
            ihsan_score: IhsanScore::MAX,
        };
        self.emit(event)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Queue — Deferred event processing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Enqueue an event for deferred processing (e.g., during initialization).
    pub fn enqueue(&mut self, event: Event) -> bool {
        if self.pending_count >= MAX_PENDING {
            return false;
        }

        // Insert sorted by priority (highest first)
        let insert_pos = self.pending[..self.pending_count]
            .iter()
            .position(|e| {
                e.as_ref()
                    .map(|existing| event.priority > existing.priority)
                    .unwrap_or(true)
            })
            .unwrap_or(self.pending_count);

        // Shift elements to make room
        if insert_pos < self.pending_count {
            for i in (insert_pos..self.pending_count).rev() {
                self.pending[i + 1] = self.pending[i];
            }
        }

        self.pending[insert_pos] = Some(event);
        self.pending_count += 1;
        true
    }

    /// Drain and dispatch all pending events. Returns number dispatched.
    pub fn flush(&mut self) -> usize {
        let mut dispatched = 0;

        // Take all pending events
        let count = self.pending_count;
        self.pending_count = 0;

        for i in 0..count {
            if let Some(event) = self.pending[i].take() {
                self.emit(event);
                dispatched += 1;
            }
        }

        dispatched
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Telemetry
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn subscription_count(&self) -> usize {
        let mut total = self.global.sub_count();
        for shard in &self.shards {
            total += shard.sub_count();
        }
        total
    }

    pub fn total_emitted(&self) -> u64 {
        self.total_emitted
    }

    pub fn total_delivered(&self) -> u64 {
        self.total_delivered
    }

    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    pub fn pending_count(&self) -> usize {
        self.pending_count
    }

    /// Pause/resume all subscriptions for a component.
    pub fn set_active_for_component(&mut self, id: &ComponentId, active: bool) {
        for shard in self.shards.iter_mut() {
            shard.set_active_for_component(id, active);
        }
        self.global.set_active_for_component(id, active);
    }

    /// Delivery ratio: delivered / emitted (0.0 - 1.0).
    pub fn delivery_ratio(&self) -> f64 {
        if self.total_emitted == 0 {
            1.0
        } else {
            self.total_delivered as f64 / self.total_emitted as f64
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Internal routing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Route a topic to its shard. Topics with a `.` go to a namespace shard;
    /// topics without go to the global shard.
    fn shard_for_topic_mut(&mut self, topic: &Topic) -> &mut EventShard {
        let s = topic.as_str();
        if s.contains('.') {
            &mut self.shards[shard_index(topic)]
        } else {
            &mut self.global
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_handler(_event: &Event) -> HookResult {
        HookResult::Continue
    }

    fn halt_handler(_event: &Event) -> HookResult {
        HookResult::Halt
    }

    fn make_source() -> ComponentId {
        ComponentId::from_name("test-source", "1.0.0")
    }

    #[test]
    fn subscribe_and_emit() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("listener", "1.0.0");

        bus.subscribe(comp, "test.topic", Priority::Low, noop_handler)
            .unwrap();

        let delivered = bus.emit_simple(
            make_source(),
            "test.topic",
            Payload::empty(),
            Priority::Normal,
            1000,
        );

        assert_eq!(delivered, 1);
        assert_eq!(bus.total_emitted(), 1);
        assert_eq!(bus.total_delivered(), 1);
    }

    #[test]
    fn wildcard_subscription() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("monitor", "1.0.0");

        bus.subscribe(comp, "system.*", Priority::Low, noop_handler)
            .unwrap();

        let d1 = bus.emit_simple(
            make_source(),
            "system.health",
            Payload::empty(),
            Priority::Normal,
            1000,
        );
        let d2 = bus.emit_simple(
            make_source(),
            "system.crash",
            Payload::empty(),
            Priority::Normal,
            1001,
        );
        let d3 = bus.emit_simple(
            make_source(),
            "agent.health",
            Payload::empty(),
            Priority::Normal,
            1002,
        );

        assert_eq!(d1, 1);
        assert_eq!(d2, 1);
        assert_eq!(d3, 0); // agent.* doesn't match system.*
    }

    #[test]
    fn priority_filtering() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("critical-only", "1.0.0");

        // Only receive Critical+
        bus.subscribe(comp, "test.*", Priority::Critical, noop_handler)
            .unwrap();

        let d1 = bus.emit_simple(
            make_source(),
            "test.low",
            Payload::empty(),
            Priority::Normal,
            1000,
        );
        let d2 = bus.emit_simple(
            make_source(),
            "test.crit",
            Payload::empty(),
            Priority::Critical,
            1001,
        );

        assert_eq!(d1, 0); // Normal < Critical filter
        assert_eq!(d2, 1);
    }

    #[test]
    fn halt_stops_propagation() {
        let mut bus = EventBus::new();
        let halt_comp = ComponentId::from_name("gatekeeper", "1.0.0");
        let pass_comp = ComponentId::from_name("listener", "1.0.0");

        // Gatekeeper subscribes first, halts all events
        bus.subscribe(halt_comp, "test.*", Priority::Low, halt_handler)
            .unwrap();
        bus.subscribe(pass_comp, "test.*", Priority::Low, noop_handler)
            .unwrap();

        let delivered = bus.emit_simple(
            make_source(),
            "test.event",
            Payload::empty(),
            Priority::Normal,
            1000,
        );

        // Only gatekeeper received it before halting
        assert_eq!(delivered, 1);
    }

    #[test]
    fn unsubscribe_works() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("temp", "1.0.0");

        let sub_id = bus
            .subscribe(comp, "test.*", Priority::Low, noop_handler)
            .unwrap();
        assert_eq!(bus.subscription_count(), 1);

        bus.unsubscribe(sub_id);
        assert_eq!(bus.subscription_count(), 0);

        let delivered = bus.emit_simple(
            make_source(),
            "test.event",
            Payload::empty(),
            Priority::Normal,
            1000,
        );
        assert_eq!(delivered, 0);
    }

    #[test]
    fn event_id_monotonic() {
        let mut bus = EventBus::new();

        let e1 = bus.next_event_id(1000);
        let e2 = bus.next_event_id(1000);
        let e3 = bus.next_event_id(1001);

        assert!(e1 < e2); // Same timestamp, different sequence
        assert!(e2 < e3); // Different timestamp
    }

    #[test]
    fn enqueue_and_flush() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("deferred", "1.0.0");
        bus.subscribe(comp, "deferred.*", Priority::Low, noop_handler)
            .unwrap();

        // Enqueue events before subscribers are ready
        let event = Event {
            id: EventId::new(1000, 0),
            source: make_source(),
            topic: Topic::new("deferred.test"),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::MAX,
        };

        assert!(bus.enqueue(event));
        assert_eq!(bus.pending_count(), 1);

        let flushed = bus.flush();
        assert_eq!(flushed, 1);
        assert_eq!(bus.pending_count(), 0);
    }

    #[test]
    fn delivery_ratio() {
        let mut bus = EventBus::new();

        // No subscribers → events drop
        bus.emit_simple(
            make_source(),
            "orphan.event",
            Payload::empty(),
            Priority::Normal,
            1000,
        );
        assert_eq!(bus.total_dropped(), 1);
        assert!(bus.delivery_ratio() < 1.0);
    }

    #[test]
    fn sharded_bus_isolates_namespaces() {
        let mut bus = EventBus::new();
        let action_comp = ComponentId::from_name("action-handler", "1.0.0");
        let memory_comp = ComponentId::from_name("memory-handler", "1.0.0");

        // Subscribe to different namespaces
        bus.subscribe(action_comp, "action.receipt", Priority::Low, noop_handler)
            .unwrap();
        bus.subscribe(memory_comp, "memory.promoted", Priority::Low, noop_handler)
            .unwrap();

        // Emit to action namespace — should reach only action handler
        let delivered = bus.emit_simple(
            make_source(),
            "action.receipt",
            Payload::empty(),
            Priority::Normal,
            1000,
        );
        assert_eq!(delivered, 1);

        // Emit to memory namespace — should reach only memory handler
        let delivered = bus.emit_simple(
            make_source(),
            "memory.promoted",
            Payload::empty(),
            Priority::Normal,
            1001,
        );
        assert_eq!(delivered, 1);
    }

    #[test]
    fn event_shard_subscribe_and_dispatch() {
        let mut shard = EventShard::new();
        let comp = ComponentId::from_name("test", "1.0.0");
        shard
            .subscribe(
                comp,
                "action.receipt",
                Priority::Low,
                noop_handler,
                SubscriptionId(1),
            )
            .unwrap();
        assert_eq!(shard.sub_count(), 1);

        let event = Event {
            id: EventId::new(1000, 0),
            source: make_source(),
            topic: Topic::new("action.receipt"),
            priority: Priority::Normal,
            payload: Payload::empty(),
            ihsan_score: IhsanScore::MAX,
        };
        let (delivered, halted) = shard.dispatch(&event);
        assert_eq!(delivered, 1);
        assert!(!halted);
    }

    #[test]
    fn shard_index_groups_same_namespace() {
        let t1 = Topic::new("action.receipt");
        let t2 = Topic::new("action.intent");
        let t3 = Topic::new("memory.promoted");
        assert_eq!(shard_index(&t1), shard_index(&t2));
        assert_ne!(shard_index(&t1), shard_index(&t3));
    }

    #[test]
    fn shard_index_within_bounds() {
        for topic in &[
            "action.x",
            "memory.x",
            "telescript.x",
            "session.x",
            "system.x",
            "ihsan.x",
            "poi.x",
            "unknown.x",
        ] {
            let idx = shard_index(&Topic::new(topic));
            assert!(
                idx < NUM_SHARDS,
                "shard_index({topic}) = {idx} >= {NUM_SHARDS}"
            );
        }
    }
}
