// bizra-node/src/heartbeat.rs
// ============================================================
// Heartbeat — The sovereign pulse that drives continuous operation
// ============================================================
//
// Phase 86-B: 4-Loop HHMM EventBus Wiring
//
// The heartbeat is the timer-driven loop that makes the system
// self-sustaining. Without it, the Node is purely reactive (only
// acts when a user sends a command). With it, the Node drives
// continuous learning, synthesis, reflex compilation, and
// self-improvement — the S0→S1 transition.
//
// ## The 4 Loops (HHMM Hierarchy)
//
// ```
// Loop A: Perception → Memory       (fragment extraction → engram)
// Loop B: Memory → Cognition        (synthesis → insight)
// Loop C: Cognition → Action        (route → execute → receipt)
// Loop D: Action → Evolution        (reflex compile → skill promote)
//         └── Feedback: Loop D → Loop A (compiled reflexes serve next query)
// ```
//
// ## Standing on Giants
// - Maturana (1980): Autopoiesis — system produces conditions for own operation
// - Friston (2006): Free Energy — minimize prediction error through action
// - Deming (1950): PDCA — continuous improvement through closed-loop feedback
// - Boyd (1976): OODA — observe/orient/decide/act at clock speed
// ============================================================

use std::collections::VecDeque;
use std::fmt;

// ============================================================
// HEARTBEAT REPORT
// ============================================================

/// Telemetry from a single heartbeat tick.
///
/// The caller (daemon, MCP transport, or protocol command) inspects
/// this to decide whether to log, alert, or adjust tick interval.
#[derive(Debug, Clone, Default)]
pub struct HeartbeatReport {
    /// Monotonic heartbeat counter.
    pub heartbeat_count: u64,
    /// Timestamp (ms) when this heartbeat executed.
    pub timestamp_ms: u64,
    /// Number of pending memory fragments processed (Loop A).
    pub fragments_processed: usize,
    /// Whether synthesis was triggered this tick (Loop B).
    pub synthesis_triggered: bool,
    /// Number of pending receipts processed (Loop C).
    pub receipts_processed: usize,
    /// Number of reflexes compiled this tick (Loop D).
    pub reflexes_compiled: usize,
    /// Current system Ihsan score (raw u16, 0-65535).
    pub ihsan_raw: u16,
    /// Total events emitted across all loops this tick.
    pub events_emitted: usize,
}

impl fmt::Display for HeartbeatReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hb={} ts={} frags={} synth={} rcpts={} reflex={} ihsan={} events={}",
            self.heartbeat_count,
            self.timestamp_ms,
            self.fragments_processed,
            self.synthesis_triggered,
            self.receipts_processed,
            self.reflexes_compiled,
            self.ihsan_raw,
            self.events_emitted,
        )
    }
}

// ============================================================
// EVENT BRIDGE — Cross-loop event queue
// ============================================================

/// Cross-loop event carrier.
///
/// When Loop A emits an event, Loop B picks it up on the next
/// heartbeat tick via this bridge. Each event is a (topic, payload)
/// pair — lightweight, no allocation beyond the strings.
///
/// Constitutional bound: if the queue overflows, the oldest event
/// is dropped (degradation, not panic). This is the Erlang philosophy:
/// shed load gracefully under pressure.
///
/// Standing on: Lamport (1978) — event ordering preserved within queue.
#[derive(Debug, Clone)]
pub struct CrossLoopEvent {
    /// Topic string (e.g., "memory.fragment.stored")
    pub topic: String,
    /// Payload (typically serialized JSON or a compact descriptor)
    pub payload: String,
}

/// Bounded FIFO queue for cross-loop event propagation.
///
/// Each heartbeat tick: current loop emits → bridge enqueues →
/// next loop drains on following tick.
#[derive(Debug)]
pub struct EventBridge {
    /// Events waiting for the next loop to consume.
    pending: VecDeque<CrossLoopEvent>,
    /// Maximum queue depth (constitutional bound).
    max_depth: usize,
    /// Total events ever enqueued (monotonic).
    total_enqueued: u64,
    /// Total events dropped due to overflow (monotonic).
    total_dropped: u64,
}

impl EventBridge {
    /// Create a new bridge with bounded capacity.
    ///
    /// `max_depth` should be sized for worst-case burst between two
    /// heartbeat ticks. Default: 256 events.
    pub fn new(max_depth: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(max_depth),
            max_depth,
            total_enqueued: 0,
            total_dropped: 0,
        }
    }

    /// Enqueue an event from the current loop for the next loop.
    ///
    /// Returns `true` if enqueued without dropping. Returns `false`
    /// if the oldest event was evicted to make room (degradation).
    ///
    /// TDD anchor: test_queue_overflow_degrades_not_panics
    pub fn enqueue(&mut self, event: CrossLoopEvent) -> bool {
        self.total_enqueued += 1;
        if self.pending.len() >= self.max_depth {
            // Constitutional degradation: drop oldest, never panic.
            self.pending.pop_front();
            self.total_dropped += 1;
            self.pending.push_back(event);
            false
        } else {
            self.pending.push_back(event);
            true
        }
    }

    /// Drain all pending events for the next loop iteration.
    ///
    /// The caller processes these in order (Lamport ordering preserved).
    pub fn drain(&mut self) -> Vec<CrossLoopEvent> {
        self.pending.drain(..).collect()
    }

    /// Number of events currently waiting.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Total events ever enqueued (monotonic counter).
    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued
    }

    /// Total events dropped due to overflow.
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    /// Is the bridge empty?
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Default for EventBridge {
    fn default() -> Self {
        Self::new(256)
    }
}

// ============================================================
// HEARTBEAT CONFIGURATION
// ============================================================

/// Configuration for the heartbeat timer.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Interval between heartbeats in milliseconds.
    /// Default: 1000ms (1 Hz). Constitutional minimum: 100ms.
    pub interval_ms: u64,
    /// Interval between synthesis triggers in milliseconds.
    /// Default: 30000ms (30 seconds). Must be >= interval_ms.
    pub synthesis_interval_ms: u64,
    /// Maximum cross-loop event bridge depth.
    /// Default: 256 events per bridge.
    pub bridge_max_depth: usize,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval_ms: 1000,
            synthesis_interval_ms: 30_000,
            bridge_max_depth: 256,
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── EventBridge tests ────────────────────────────────────

    #[test]
    fn bridge_enqueue_and_drain() {
        let mut bridge = EventBridge::new(10);
        assert!(bridge.is_empty());
        assert_eq!(bridge.pending_count(), 0);

        let ok = bridge.enqueue(CrossLoopEvent {
            topic: "memory.fragment.stored".into(),
            payload: "atom_123".into(),
        });
        assert!(ok);
        assert_eq!(bridge.pending_count(), 1);
        assert_eq!(bridge.total_enqueued(), 1);

        let events = bridge.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].topic, "memory.fragment.stored");
        assert!(bridge.is_empty());
    }

    #[test]
    fn bridge_preserves_ordering() {
        let mut bridge = EventBridge::new(100);
        for i in 0..10 {
            bridge.enqueue(CrossLoopEvent {
                topic: format!("event.{i}"),
                payload: String::new(),
            });
        }
        let events = bridge.drain();
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.topic, format!("event.{i}"));
        }
    }

    #[test]
    fn bridge_overflow_degrades_not_panics() {
        let mut bridge = EventBridge::new(3);

        // Fill to capacity
        for i in 0..3 {
            let ok = bridge.enqueue(CrossLoopEvent {
                topic: format!("ev.{i}"),
                payload: String::new(),
            });
            assert!(ok);
        }
        assert_eq!(bridge.total_dropped(), 0);

        // Overflow: oldest (ev.0) gets dropped
        let ok = bridge.enqueue(CrossLoopEvent {
            topic: "ev.3".into(),
            payload: String::new(),
        });
        assert!(!ok); // signals degradation
        assert_eq!(bridge.total_dropped(), 1);
        assert_eq!(bridge.pending_count(), 3);

        // Verify oldest was dropped
        let events = bridge.drain();
        assert_eq!(events[0].topic, "ev.1"); // ev.0 was evicted
        assert_eq!(events[1].topic, "ev.2");
        assert_eq!(events[2].topic, "ev.3");
    }

    #[test]
    fn bridge_default_capacity() {
        let bridge = EventBridge::default();
        assert_eq!(bridge.pending_count(), 0);
        assert_eq!(bridge.total_enqueued(), 0);
        assert_eq!(bridge.total_dropped(), 0);
    }

    // ── HeartbeatReport tests ────────────────────────────────

    #[test]
    fn heartbeat_report_display() {
        let report = HeartbeatReport {
            heartbeat_count: 42,
            timestamp_ms: 1000,
            fragments_processed: 2,
            synthesis_triggered: true,
            receipts_processed: 1,
            reflexes_compiled: 0,
            ihsan_raw: 9500,
            events_emitted: 3,
        };
        let s = format!("{report}");
        assert!(s.contains("hb=42"));
        assert!(s.contains("synth=true"));
        assert!(s.contains("ihsan=9500"));
    }

    #[test]
    fn heartbeat_report_default() {
        let report = HeartbeatReport::default();
        assert_eq!(report.heartbeat_count, 0);
        assert_eq!(report.events_emitted, 0);
    }

    // ── HeartbeatConfig tests ────────────────────────────────

    #[test]
    fn heartbeat_config_defaults() {
        let config = HeartbeatConfig::default();
        assert_eq!(config.interval_ms, 1000);
        assert_eq!(config.synthesis_interval_ms, 30_000);
        assert_eq!(config.bridge_max_depth, 256);
        // Constitutional minimum
        assert!(config.interval_ms >= 100);
    }
}
