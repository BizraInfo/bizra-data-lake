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

/// Maximum number of active subscriptions.
const MAX_SUBSCRIPTIONS: usize = 512;

/// Maximum pending events in the dispatch queue.
const MAX_PENDING: usize = 256;

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

/// The Event Bus — routes events between components.
pub struct EventBus {
    /// Subscription entries (fixed array)
    handlers: [Option<HandlerEntry>; MAX_SUBSCRIPTIONS],
    /// Number of active subscriptions
    sub_count: usize,
    /// Next subscription ID
    next_sub_id: u32,

    /// Pending event queue (priority-ordered dispatch)
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
        EventBus {
            handlers: [None; MAX_SUBSCRIPTIONS],
            sub_count: 0,
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
        let slot = self.handlers.iter()
            .position(|h| h.is_none())
            .ok_or(HookError::SubscribersFull)?;

        let sub_id = SubscriptionId(self.next_sub_id);
        self.next_sub_id += 1;

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

    /// Unsubscribe by subscription ID.
    pub fn unsubscribe(&mut self, sub_id: SubscriptionId) -> bool {
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

    /// Unsubscribe all handlers for a component.
    pub fn unsubscribe_all(&mut self, component: &ComponentId) -> usize {
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

    /// Pause/resume a subscription.
    pub fn set_active(&mut self, sub_id: SubscriptionId, active: bool) -> bool {
        for entry in self.handlers.iter_mut() {
            if let Some(h) = entry {
                if h.sub.id == sub_id {
                    h.sub.active = active;
                    return true;
                }
            }
        }
        false
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

    /// Emit an event. Immediately dispatches to all matching subscribers.
    ///
    /// Returns the number of subscribers that received the event.
    pub fn emit(&mut self, event: Event) -> usize {
        self.total_emitted += 1;

        let mut delivered = 0;

        // Collect matching handlers and dispatch
        // We iterate in subscription order (FIFO registration)
        // Priority filtering happens in Subscription::matches()
        for entry in self.handlers.iter() {
            if let Some(h) = entry {
                if h.sub.matches(&event) {
                    let result = (h.handler)(&event);
                    delivered += 1;

                    match result {
                        HookResult::Continue => continue,
                        HookResult::Skip => break,
                        HookResult::Halt => {
                            // Event was halted by a subscriber
                            self.total_delivered += delivered as u64;
                            return delivered;
                        }
                        HookResult::Transform => {
                            // In a full implementation, the handler would
                            // modify the event. For now, continue.
                            continue;
                        }
                    }
                }
            }
        }

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
        self.sub_count
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
        for entry in self.handlers.iter_mut() {
            if let Some(h) = entry {
                if h.sub.component == *id {
                    h.sub.active = active;
                }
            }
        }
    }

    /// Delivery ratio: delivered / emitted (0.0 - 1.0).
    pub fn delivery_ratio(&self) -> f64 {
        if self.total_emitted == 0 {
            1.0
        } else {
            self.total_delivered as f64 / self.total_emitted as f64
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

        bus.subscribe(comp, "test.topic", Priority::Low, noop_handler).unwrap();

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

        bus.subscribe(comp, "system.*", Priority::Low, noop_handler).unwrap();

        let d1 = bus.emit_simple(make_source(), "system.health", Payload::empty(), Priority::Normal, 1000);
        let d2 = bus.emit_simple(make_source(), "system.crash", Payload::empty(), Priority::Normal, 1001);
        let d3 = bus.emit_simple(make_source(), "agent.health", Payload::empty(), Priority::Normal, 1002);

        assert_eq!(d1, 1);
        assert_eq!(d2, 1);
        assert_eq!(d3, 0); // agent.* doesn't match system.*
    }

    #[test]
    fn priority_filtering() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("critical-only", "1.0.0");

        // Only receive Critical+
        bus.subscribe(comp, "test.*", Priority::Critical, noop_handler).unwrap();

        let d1 = bus.emit_simple(make_source(), "test.low", Payload::empty(), Priority::Normal, 1000);
        let d2 = bus.emit_simple(make_source(), "test.crit", Payload::empty(), Priority::Critical, 1001);

        assert_eq!(d1, 0); // Normal < Critical filter
        assert_eq!(d2, 1);
    }

    #[test]
    fn halt_stops_propagation() {
        let mut bus = EventBus::new();
        let halt_comp = ComponentId::from_name("gatekeeper", "1.0.0");
        let pass_comp = ComponentId::from_name("listener", "1.0.0");

        // Gatekeeper subscribes first, halts all events
        bus.subscribe(halt_comp, "test.*", Priority::Low, halt_handler).unwrap();
        bus.subscribe(pass_comp, "test.*", Priority::Low, noop_handler).unwrap();

        let delivered = bus.emit_simple(make_source(), "test.event", Payload::empty(), Priority::Normal, 1000);

        // Only gatekeeper received it before halting
        assert_eq!(delivered, 1);
    }

    #[test]
    fn unsubscribe_works() {
        let mut bus = EventBus::new();
        let comp = ComponentId::from_name("temp", "1.0.0");

        let sub_id = bus.subscribe(comp, "test.*", Priority::Low, noop_handler).unwrap();
        assert_eq!(bus.subscription_count(), 1);

        bus.unsubscribe(sub_id);
        assert_eq!(bus.subscription_count(), 0);

        let delivered = bus.emit_simple(make_source(), "test.event", Payload::empty(), Priority::Normal, 1000);
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
        bus.subscribe(comp, "deferred.*", Priority::Low, noop_handler).unwrap();

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
        bus.emit_simple(make_source(), "orphan.event", Payload::empty(), Priority::Normal, 1000);
        assert_eq!(bus.total_dropped(), 1);
        assert!(bus.delivery_ratio() < 1.0);
    }
}
