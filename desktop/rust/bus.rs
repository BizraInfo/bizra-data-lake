//! EventBus — typed publish/subscribe for Node0.
//!
//! The bus is the nervous system. Every signal flows through it.
//! Subscribers are called synchronously in priority order.
//! Events are optionally stored in the append-only EventStore.
//!
//! Standing on Giants: Gamma et al. (Observer pattern, 1994) · Hewitt (Actor model, 1973)

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::store::EventStore;
use crate::types::*;

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSCRIBER — A named, prioritized callback for a specific event kind.
// ═══════════════════════════════════════════════════════════════════════════════

type SubscriberFn = Box<dyn Fn(&Event) + Send + Sync>;

struct Subscriber {
    name: String,
    priority: HookPriority,
    callback: SubscriberFn,
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT BUS — The central nervous system.
// ═══════════════════════════════════════════════════════════════════════════════

/// The EventBus. Thread-safe. Clonable (shares state via Arc).
#[derive(Clone)]
pub struct EventBus {
    /// Subscribers indexed by EventKind.
    subscribers: Arc<RwLock<HashMap<EventKind, Vec<Subscriber>>>>,
    /// Wildcard subscribers that receive ALL events.
    wildcards: Arc<RwLock<Vec<Subscriber>>>,
    /// Optional event store for persistence.
    store: Arc<Mutex<Option<EventStore>>>,
    /// Total events published (monotonic counter).
    published_count: Arc<std::sync::atomic::AtomicU64>,
}

impl EventBus {
    /// Create a new EventBus without persistent storage.
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            wildcards: Arc::new(RwLock::new(Vec::new())),
            store: Arc::new(Mutex::new(None)),
            published_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Create a new EventBus with an attached EventStore for persistence.
    pub fn with_store(store: EventStore) -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            wildcards: Arc::new(RwLock::new(Vec::new())),
            store: Arc::new(Mutex::new(Some(store))),
            published_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Subscribe to a specific event kind.
    pub fn subscribe<F>(
        &self,
        kind: EventKind,
        name: impl Into<String>,
        priority: HookPriority,
        callback: F,
    ) -> HookResult<()>
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        let mut subs = self
            .subscribers
            .write()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        let list = subs.entry(kind).or_default();
        list.push(Subscriber {
            name: name.into(),
            priority,
            callback: Box::new(callback),
        });
        // Keep sorted by priority (lowest first).
        list.sort_by_key(|s| s.priority);
        Ok(())
    }

    /// Subscribe to ALL events (for logging, metrics, store).
    pub fn subscribe_all<F>(
        &self,
        name: impl Into<String>,
        priority: HookPriority,
        callback: F,
    ) -> HookResult<()>
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        let mut wc = self
            .wildcards
            .write()
            .map_err(|e| HookError::LockPoisoned(e.to_string()))?;

        wc.push(Subscriber {
            name: name.into(),
            priority,
            callback: Box::new(callback),
        });
        wc.sort_by_key(|s| s.priority);
        Ok(())
    }

    /// Publish an event. Calls all matching subscribers in priority order.
    /// Optionally persists to the EventStore.
    pub fn publish(&self, event: Event) -> HookResult<EventId> {
        let id = event.id;

        // Persist to store if attached.
        if let Ok(mut store_lock) = self.store.lock() {
            if let Some(store) = store_lock.as_mut() {
                store.append(event.clone())?;
            }
        }

        // Call kind-specific subscribers.
        if let Ok(subs) = self.subscribers.read() {
            if let Some(list) = subs.get(&event.kind) {
                for sub in list {
                    (sub.callback)(&event);
                }
            }
        }

        // Call wildcard subscribers.
        if let Ok(wc) = self.wildcards.read() {
            for sub in wc.iter() {
                (sub.callback)(&event);
            }
        }

        self.published_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(id)
    }

    /// Publish and return the event (for chaining).
    pub fn emit(&self, kind: EventKind, source: ComponentId) -> HookResult<Event> {
        let event = Event::new(kind, source);
        self.publish(event.clone())?;
        Ok(event)
    }

    /// Total events published since bus creation.
    pub fn total_published(&self) -> u64 {
        self.published_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of subscribers for a specific event kind.
    pub fn subscriber_count(&self, kind: EventKind) -> usize {
        self.subscribers
            .read()
            .map(|s| s.get(&kind).map_or(0, |l| l.len()))
            .unwrap_or(0)
    }

    /// Total subscriber count across all kinds + wildcards.
    pub fn total_subscribers(&self) -> usize {
        let kind_count = self
            .subscribers
            .read()
            .map(|s| s.values().map(|l| l.len()).sum::<usize>())
            .unwrap_or(0);
        let wc_count = self.wildcards.read().map(|w| w.len()).unwrap_or(0);
        kind_count + wc_count
    }

    /// List all event kinds that have at least one subscriber.
    pub fn subscribed_kinds(&self) -> Vec<EventKind> {
        self.subscribers
            .read()
            .map(|s| s.keys().copied().collect())
            .unwrap_or_default()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("total_published", &self.total_published())
            .field("total_subscribers", &self.total_subscribers())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn test_source() -> ComponentId {
        ComponentId(9999)
    }

    #[test]
    fn publish_reaches_subscriber() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicU64::new(0));
        let count_clone = count.clone();

        bus.subscribe(
            EventKind::UserMessage,
            "test",
            HookPriority::APP,
            move |_| {
                count_clone.fetch_add(1, Ordering::Relaxed);
            },
        )
        .unwrap();

        bus.publish(Event::new(EventKind::UserMessage, test_source()))
            .unwrap();

        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn wildcard_receives_all_events() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicU64::new(0));
        let count_clone = count.clone();

        bus.subscribe_all("audit_log", HookPriority::INFRA, move |_| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();

        bus.publish(Event::new(EventKind::UserMessage, test_source()))
            .unwrap();
        bus.publish(Event::new(EventKind::SearchExecuted, test_source()))
            .unwrap();
        bus.publish(Event::new(EventKind::TaskComplete, test_source()))
            .unwrap();

        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn priority_ordering() {
        let bus = EventBus::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let o1 = order.clone();
        bus.subscribe(
            EventKind::UserMessage,
            "low_priority",
            HookPriority::USER,
            move |_| o1.lock().unwrap().push("user"),
        )
        .unwrap();

        let o2 = order.clone();
        bus.subscribe(
            EventKind::UserMessage,
            "high_priority",
            HookPriority::SYSTEM,
            move |_| o2.lock().unwrap().push("system"),
        )
        .unwrap();

        let o3 = order.clone();
        bus.subscribe(
            EventKind::UserMessage,
            "mid_priority",
            HookPriority::APP,
            move |_| o3.lock().unwrap().push("app"),
        )
        .unwrap();

        bus.publish(Event::new(EventKind::UserMessage, test_source()))
            .unwrap();

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["system", "app", "user"]);
    }

    #[test]
    fn unmatched_event_no_panic() {
        let bus = EventBus::new();
        // No subscribers — should not panic.
        let result = bus.publish(Event::new(EventKind::DesktopAction, test_source()));
        assert!(result.is_ok());
    }

    #[test]
    fn published_count_increments() {
        let bus = EventBus::new();
        assert_eq!(bus.total_published(), 0);

        bus.publish(Event::new(EventKind::UserMessage, test_source()))
            .unwrap();
        bus.publish(Event::new(EventKind::AgentResponse, test_source()))
            .unwrap();

        assert_eq!(bus.total_published(), 2);
    }

    #[test]
    fn bus_is_clone_safe() {
        let bus = EventBus::new();
        let bus2 = bus.clone();
        let count = Arc::new(AtomicU64::new(0));
        let cc = count.clone();

        bus.subscribe(
            EventKind::TaskStart,
            "counter",
            HookPriority::APP,
            move |_| {
                cc.fetch_add(1, Ordering::Relaxed);
            },
        )
        .unwrap();

        // Publishing on clone reaches subscriber registered on original.
        bus2.publish(Event::new(EventKind::TaskStart, test_source()))
            .unwrap();

        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
