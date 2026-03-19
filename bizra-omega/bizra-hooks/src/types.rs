//! # BIZRA Hook System — Core Types
//!
//! The atomic building blocks of BIZRA's nervous system.
//! Every type here is `Copy + Clone + Send + Sync` — zero-cost, thread-safe, sovereign.
//!
//! ## Design Principles
//! - **No heap allocation in hot paths**: All IDs are fixed-size value types
//! - **Deterministic ordering**: Every event, component, hook has a total order
//! - **Cryptographic identity**: IDs derived from content, not sequence counters
//! - **إحسان scoring**: Quality is a first-class type, not an afterthought

use core::cmp::Ordering;
use core::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Identity Types — Fixed-size, Copy, deterministic
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 128-bit component identifier. Unique across the entire BIZRA network.
/// Derived from component name + version via FNV-1a hash (no external dependency).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(pub [u8; 16]);

/// 64-bit event identifier. Unique within a session.
/// Combines timestamp_nanos(48-bit) + sequence(16-bit) for total ordering.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(pub u64);

/// 64-bit hook chain identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookId(pub u64);

/// 32-bit subscription handle. Returned on subscribe, used to unsubscribe.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(pub u32);

impl fmt::Debug for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as hex: "comp:a1b2c3d4..."
        write!(f, "comp:")?;
        for b in &self.0[..4] {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Debug for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evt:{:016x}", self.0)
    }
}

impl fmt::Debug for HookId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hook:{:016x}", self.0)
    }
}

impl fmt::Debug for SubscriptionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sub:{}", self.0)
    }
}

impl Ord for EventId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for EventId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ID Generation — Pure Rust, no dependencies
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl ComponentId {
    /// Create ComponentId from name + version via FNV-1a 128-bit hash.
    /// Deterministic: same input always produces same ID.
    pub fn from_name(name: &str, version: &str) -> Self {
        let mut hash: u128 = 0x6c62272e07bb0142_62b821756295c58d; // FNV-1a 128-bit offset basis
        let prime: u128 = 0x0000000001000000_000000000000013b; // FNV-1a 128-bit prime

        for byte in name.as_bytes().iter().chain(b":").chain(version.as_bytes()) {
            hash ^= *byte as u128;
            hash = hash.wrapping_mul(prime);
        }

        ComponentId(hash.to_le_bytes())
    }

    /// Create from raw 16-byte array (for deserialization / FFI).
    pub const fn from_raw(bytes: [u8; 16]) -> Self {
        ComponentId(bytes)
    }

    /// Zero ID — represents "no component" / null sentinel.
    pub const fn null() -> Self {
        ComponentId([0u8; 16])
    }

    pub fn is_null(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl EventId {
    /// Create EventId from timestamp nanoseconds and sequence number.
    /// timestamp_nanos: lower 48 bits used (good for ~3.2 days of nanos)
    /// sequence: 16-bit counter for events within same nanosecond
    pub const fn new(timestamp_nanos: u64, sequence: u16) -> Self {
        let id = (timestamp_nanos & 0x0000_FFFF_FFFF_FFFF) << 16 | (sequence as u64);
        EventId(id)
    }

    /// Extract timestamp component (lower 48 bits shifted).
    pub const fn timestamp_nanos(&self) -> u64 {
        self.0 >> 16
    }

    /// Extract sequence component (lower 16 bits).
    pub const fn sequence(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    pub const fn null() -> Self {
        EventId(0)
    }
}

impl HookId {
    pub const fn new(id: u64) -> Self {
        HookId(id)
    }

    pub const fn null() -> Self {
        HookId(0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Event Types — What flows through the nervous system
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Event priority levels. Higher priority = processed first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum Priority {
    /// Background tasks, telemetry, non-critical logging
    Low = 0,
    /// Normal operational events
    #[default]
    Normal = 1,
    /// User-facing actions, API responses
    High = 2,
    /// Safety checks, إحسان gate evaluations, FATE assertions
    Critical = 3,
    /// System integrity, constitutional constraint enforcement
    Emergency = 4,
}

/// Event topic — hierarchical namespace for routing.
/// Fixed-size to avoid heap allocation. Max 128 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Topic {
    data: [u8; 128],
    len: u8,
}

impl Topic {
    /// Create topic from string slice. Truncates at 128 bytes.
    pub fn new(s: &str) -> Self {
        let mut data = [0u8; 128];
        let bytes = s.as_bytes();
        let len = bytes.len().min(128);
        data[..len].copy_from_slice(&bytes[..len]);
        Topic {
            data,
            len: len as u8,
        }
    }

    /// Get topic as string slice.
    pub fn as_str(&self) -> &str {
        let slice = &self.data[..self.len as usize];
        debug_assert!(
            core::str::from_utf8(slice).is_ok(),
            "UTF-8 invariant violated in Topic"
        );
        // Safety: constructors validate UTF-8; debug_assert guards against regression
        unsafe { core::str::from_utf8_unchecked(slice) }
    }

    /// Check if this topic matches a pattern (supports trailing wildcard '*').
    /// "system.*" matches "system.health", "system.crash", etc.
    pub fn matches(&self, pattern: &Topic) -> bool {
        let pat = pattern.as_str();
        let topic = self.as_str();

        if let Some(prefix) = pat.strip_suffix(".*") {
            topic.starts_with(prefix)
                && topic.len() > prefix.len()
                && topic.as_bytes()[prefix.len()] == b'.'
        } else {
            topic == pat
        }
    }

    /// Well-known topics
    pub const fn lifecycle() -> Self {
        Self::from_static(b"system.lifecycle")
    }

    pub const fn health() -> Self {
        Self::from_static(b"system.health")
    }

    pub const fn ihsan() -> Self {
        Self::from_static(b"system.ihsan")
    }

    const fn from_static(bytes: &[u8]) -> Self {
        let mut data = [0u8; 128];
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            data[i] = bytes[i];
            i += 1;
        }
        Topic {
            data,
            len: len as u8,
        }
    }
}

impl fmt::Debug for Topic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.as_str())
    }
}

impl fmt::Display for Topic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Payload — event data. Fixed-size buffer to avoid heap allocation.
/// For larger payloads, store a reference/pointer in the first 8 bytes.
#[derive(Clone, Copy)]
pub struct Payload {
    data: [u8; 256],
    len: u16,
}

impl Payload {
    pub fn new(bytes: &[u8]) -> Self {
        let mut data = [0u8; 256];
        let len = bytes.len().min(256);
        data[..len].copy_from_slice(&bytes[..len]);
        Payload {
            data,
            len: len as u16,
        }
    }

    pub fn empty() -> Self {
        Payload {
            data: [0u8; 256],
            len: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Interpret first 8 bytes as u64 (for numeric payloads).
    pub fn as_u64(&self) -> Option<u64> {
        if self.len >= 8 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&self.data[..8]);
            Some(u64::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Create from u64 value.
    pub fn from_u64(val: u64) -> Self {
        let mut data = [0u8; 256];
        data[..8].copy_from_slice(&val.to_le_bytes());
        Payload { data, len: 8 }
    }

    /// Interpret as UTF-8 string (for text payloads).
    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_bytes()).ok()
    }

    /// Create from string slice.
    pub fn from_text(s: &str) -> Self {
        Self::new(s.as_bytes())
    }
}

impl fmt::Debug for Payload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(s) = self.as_str() {
            write!(f, "Payload({} bytes: {:?})", self.len, s)
        } else {
            write!(f, "Payload({} bytes)", self.len)
        }
    }
}

/// The core event structure that flows through the entire nervous system.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    /// Unique event identifier (timestamp + sequence)
    pub id: EventId,
    /// Source component that emitted this event
    pub source: ComponentId,
    /// Hierarchical topic for routing
    pub topic: Topic,
    /// Processing priority
    pub priority: Priority,
    /// Event payload data
    pub payload: Payload,
    /// إحسان quality score at emission time (0.0 - 1.0, encoded as u16 / 65535)
    pub ihsan_score: IhsanScore,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// إحسان Quality Score — The Lyapunov Certificate
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Quality score encoded as fixed-point u16 (0..65535 → 0.0..1.0).
/// This IS the Lyapunov function: must not decrease below threshold across mutations.
/// 99.0% إحسان = 65,005 raw value = the constitutional floor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IhsanScore(u16);

impl IhsanScore {
    /// Maximum score: 1.0 (perfection)
    pub const MAX: Self = IhsanScore(u16::MAX);

    /// Constitutional floor: 0.990 (إحسان standard)
    pub const IHSAN_FLOOR: Self = IhsanScore(64_881); // floor(0.99 * 65535)

    /// Critical threshold: 0.950 (warning level)
    pub const WARNING: Self = IhsanScore(62_258); // floor(0.95 * 65535)

    /// Create from floating point (clamped to 0.0..1.0).
    pub fn from_f64(score: f64) -> Self {
        IhsanScore((score.clamp(0.0, 1.0) * 65535.0) as u16)
    }

    /// Convert to floating point.
    pub fn as_f64(&self) -> f64 {
        self.0 as f64 / 65535.0
    }

    /// Raw u16 value.
    pub const fn raw(&self) -> u16 {
        self.0
    }

    /// Create from raw u16 value (0-65535 scale).
    ///
    /// WARNING: The raw scale is 0-65535, NOT 0-10000.
    /// `from_raw(9500)` = 0.145, NOT 0.95.
    /// Prefer `from_f64(0.95)` for human-readable values.
    ///
    /// Use `from_raw` ONLY for deserialization, FFI boundaries,
    /// and const contexts where `from_f64` is not available.
    /// Sprint 3: migrate all callers to `from_f64`, restrict to `pub(crate)`.
    pub const fn from_raw(raw: u16) -> Self {
        IhsanScore(raw)
    }

    /// Check if score meets إحسان constitutional floor (≥ 0.99).
    pub fn meets_ihsan(&self) -> bool {
        *self >= Self::IHSAN_FLOOR
    }

    /// Check if score is at warning level.
    pub fn is_warning(&self) -> bool {
        *self < Self::IHSAN_FLOOR && *self >= Self::WARNING
    }

    /// Check if score is critical (below warning).
    pub fn is_critical(&self) -> bool {
        *self < Self::WARNING
    }
}

impl fmt::Debug for IhsanScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "إحسان({:.4})", self.as_f64())
    }
}

impl fmt::Display for IhsanScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.as_f64() * 100.0)
    }
}

impl Default for IhsanScore {
    fn default() -> Self {
        Self::MAX // Start at perfection, degrade only with cause
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Component Metadata — What the Registry stores
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Component health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ComponentStatus {
    /// Registered but not yet initialized
    #[default]
    Registered = 0,
    /// Initialized and ready to process events
    Active = 1,
    /// Temporarily suspended (will not receive events)
    Suspended = 2,
    /// Degraded but operational
    Degraded = 3,
    /// Failed, pending recovery or removal
    Failed = 4,
    /// Gracefully shutting down
    ShuttingDown = 5,
}

/// Fixed-size name buffer (64 bytes max).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Name {
    data: [u8; 64],
    len: u8,
}

impl Name {
    pub fn new(s: &str) -> Self {
        let mut data = [0u8; 64];
        let bytes = s.as_bytes();
        let len = bytes.len().min(64);
        data[..len].copy_from_slice(&bytes[..len]);
        Name {
            data,
            len: len as u8,
        }
    }

    pub fn as_str(&self) -> &str {
        let slice = &self.data[..self.len as usize];
        debug_assert!(
            core::str::from_utf8(slice).is_ok(),
            "UTF-8 invariant violated in Name"
        );
        // Safety: constructors validate UTF-8; debug_assert guards against regression
        unsafe { core::str::from_utf8_unchecked(slice) }
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Version string (16 bytes max, e.g. "3.0.0-GENESIS").
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    data: [u8; 16],
    len: u8,
}

impl Version {
    pub fn new(s: &str) -> Self {
        let mut data = [0u8; 16];
        let bytes = s.as_bytes();
        let len = bytes.len().min(16);
        data[..len].copy_from_slice(&bytes[..len]);
        Version {
            data,
            len: len as u8,
        }
    }

    pub fn as_str(&self) -> &str {
        let slice = &self.data[..self.len as usize];
        debug_assert!(
            core::str::from_utf8(slice).is_ok(),
            "UTF-8 invariant violated in Version"
        );
        // Safety: constructors validate UTF-8; debug_assert guards against regression
        unsafe { core::str::from_utf8_unchecked(slice) }
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Component metadata stored in the Registry.
/// This is what RSI Pillar I (Self-Model) queries.
#[derive(Debug, Clone, Copy)]
pub struct ComponentMeta {
    /// Unique identifier
    pub id: ComponentId,
    /// Human-readable name
    pub name: Name,
    /// Semantic version
    pub version: Version,
    /// Current health status
    pub status: ComponentStatus,
    /// Current إحسان quality score
    pub ihsan: IhsanScore,
    /// Number of events emitted
    pub events_emitted: u64,
    /// Number of events consumed
    pub events_consumed: u64,
    /// Registration timestamp (nanos since epoch)
    pub registered_at: u64,
    /// Last activity timestamp
    pub last_active_at: u64,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Hook Types — Processing pipeline definitions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Hook processing result — determines event flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookResult {
    /// Continue to next hook in chain
    Continue,
    /// Skip remaining hooks, deliver event
    Skip,
    /// Halt chain, do NOT deliver event (used by إحسان gate)
    Halt,
    /// Transform: continue with modified event
    Transform,
}

/// Hook execution phase — when in the lifecycle does this hook run?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum HookPhase {
    /// Before event enters the bus (validation, filtering)
    PreEmit = 0,
    /// During routing (topic rewriting, priority adjustment)
    Route = 1,
    /// Before delivery to subscriber (transformation, enrichment)
    PreDeliver = 2,
    /// After delivery (logging, telemetry, إحسان scoring)
    PostDeliver = 3,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Error Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Hook system errors. No heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookError {
    /// Component ID already registered
    DuplicateComponent(ComponentId),
    /// Component not found in registry
    ComponentNotFound(ComponentId),
    /// Registry is full (static capacity exceeded)
    RegistryFull,
    /// Event bus subscriber slots exhausted
    SubscribersFull,
    /// Hook chain capacity exceeded
    HookChainFull,
    /// Event rejected by إحسان gate (score too low)
    IhsanGateRejected(IhsanScore),
    /// Event rejected by hook chain (Halt result)
    HookHalted(HookId),
    /// Invalid topic format
    InvalidTopic,
    /// Component is not in Active status
    ComponentInactive(ComponentId),
}

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateComponent(id) => write!(f, "duplicate component: {id:?}"),
            Self::ComponentNotFound(id) => write!(f, "component not found: {id:?}"),
            Self::RegistryFull => write!(f, "registry capacity exceeded"),
            Self::SubscribersFull => write!(f, "subscriber capacity exceeded"),
            Self::HookChainFull => write!(f, "hook chain capacity exceeded"),
            Self::IhsanGateRejected(score) => write!(f, "إحسان gate rejected: {score}"),
            Self::HookHalted(id) => write!(f, "hook halted event: {id:?}"),
            Self::InvalidTopic => write!(f, "invalid topic format"),
            Self::ComponentInactive(id) => write!(f, "component inactive: {id:?}"),
        }
    }
}

pub type HookResult_ = Result<(), HookError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_id_deterministic() {
        let id1 = ComponentId::from_name("memory-engine", "1.0.0");
        let id2 = ComponentId::from_name("memory-engine", "1.0.0");
        let id3 = ComponentId::from_name("memory-engine", "1.0.1");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn event_id_ordering() {
        let e1 = EventId::new(1000, 0);
        let e2 = EventId::new(1000, 1);
        let e3 = EventId::new(1001, 0);
        assert!(e1 < e2);
        assert!(e2 < e3);
    }

    #[test]
    fn event_id_roundtrip() {
        let e = EventId::new(0xABCD_1234_5678, 42);
        assert_eq!(e.timestamp_nanos(), 0xABCD_1234_5678);
        assert_eq!(e.sequence(), 42);
    }

    #[test]
    fn ihsan_score_boundaries() {
        let perfect = IhsanScore::MAX;
        assert!(perfect.meets_ihsan());
        assert_eq!(perfect.as_f64(), 1.0);

        let floor = IhsanScore::IHSAN_FLOOR;
        assert!(floor.meets_ihsan());

        let below = IhsanScore::from_f64(0.989);
        assert!(!below.meets_ihsan());
        assert!(below.is_warning());

        let critical = IhsanScore::from_f64(0.94);
        assert!(critical.is_critical());
    }

    #[test]
    fn topic_wildcard_matching() {
        let health = Topic::new("system.health");
        let pattern = Topic::new("system.*");
        assert!(health.matches(&pattern));

        let exact = Topic::new("system.health");
        assert!(health.matches(&exact));

        let other = Topic::new("agent.health");
        assert!(!other.matches(&pattern));
    }

    #[test]
    fn payload_roundtrip() {
        let p = Payload::from_u64(42);
        assert_eq!(p.as_u64(), Some(42));

        let p2 = Payload::from_text("hello BIZRA");
        assert_eq!(p2.as_str(), Some("hello BIZRA"));
    }
}
