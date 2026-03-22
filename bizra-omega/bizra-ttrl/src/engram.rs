//! # Engram — Tier 2 O(1) Factual Memory (Paper 2)
//!
//! The Engram module is a **model-level** memory primitive that sits between
//! the system-level Reflex Cache (Tier 1) and full GPU inference (Tier 3).
//!
//! Tier 1 (Reflex Cache): compiled action plan → O(1) system hash lookup
//! Tier 2 (Engram):       static factual knowledge → O(1) CPU RAM lookup
//! Tier 3 (Full Inference): novel reasoning → GPU, 2-5 seconds
//!
//! ## Why Engram matters (CPVA)
//! ~25% of model computation is retrieving static facts that never change.
//! Engram intercepts these lookups in CPU RAM, skipping the GPU entirely.
//!
//! ## Economics (from the four-paper analysis)
//! | Path          | Cost      | Latency  |
//! |---------------|-----------|----------|
//! | Tier 1 hit    | $0.005    | <50ms    |
//! | Tier 2 hit    | $0.06     | 1-3s     |
//! | Tier 3 miss   | $0.10     | 2-5s     |
//!
//! ## Implementation note
//! This module does NOT depend on any GPU / model runtime.  It is a pure
//! CPU hash-map keyed by BLAKE3(intent_canonical).  Population of the map
//! is the responsibility of the offline Engram compiler (future sprint).
//!
//! Standing on Giants:
//! - DeepSeek MoE architecture (2025): Engram = specialised non-activated expert
//! - Shannon (1948): O(1) lookup as zero-entropy retrieval

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Result of an Engram lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngramResult {
    /// Fact found — GPU inference not required for this lookup.
    Hit { value: String, confidence: f64 },
    /// No entry; caller must fall through to full inference.
    Miss,
}

impl EngramResult {
    pub fn is_hit(&self) -> bool {
        matches!(self, EngramResult::Hit { .. })
    }

    /// Consume the result and return the value, panicking on Miss.
    /// Prefer `if let Hit { value, .. }` at call sites.
    pub fn value(self) -> Option<String> {
        match self {
            EngramResult::Hit { value, .. } => Some(value),
            EngramResult::Miss => None,
        }
    }
}

/// An Engram entry: a static fact with its confidence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramEntry {
    /// Canonical string representation of the fact.
    pub value: String,
    /// Confidence that this entry is still accurate (decays over time).
    pub confidence: f64,
    /// UNIX ms timestamp when the entry was written.
    pub written_at_ms: u64,
    /// Number of times this entry has been served (for eviction policy).
    pub hit_count: u64,
}

/// In-process Engram cache backed by a `HashMap<[u8;32], EngramEntry>`.
///
/// Keys are BLAKE3(intent_canonical_bytes).
/// Thread-safety is the responsibility of the caller (wrap in `Mutex` if needed).
#[derive(Debug, Default)]
pub struct EngramCache {
    store: HashMap<[u8; 32], EngramEntry>,
    hits: u64,
    misses: u64,
}

impl EngramCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite an Engram entry.
    pub fn insert(
        &mut self,
        intent_canonical: &[u8],
        value: impl Into<String>,
        confidence: f64,
        now_ms: u64,
    ) {
        let key = Self::key(intent_canonical);
        self.store.insert(
            key,
            EngramEntry {
                value: value.into(),
                confidence,
                written_at_ms: now_ms,
                hit_count: 0,
            },
        );
    }

    /// Look up an Engram entry by intent canonical bytes.
    /// Returns `Hit` only when the entry exists AND confidence ≥ `min_confidence`.
    pub fn lookup(&mut self, intent_canonical: &[u8], min_confidence: f64) -> EngramResult {
        let key = Self::key(intent_canonical);
        match self.store.get_mut(&key) {
            Some(entry) if entry.confidence >= min_confidence => {
                entry.hit_count += 1;
                self.hits += 1;
                EngramResult::Hit {
                    value: entry.value.clone(),
                    confidence: entry.confidence,
                }
            }
            _ => {
                self.misses += 1;
                EngramResult::Miss
            }
        }
    }

    /// Read-only lookup. Does NOT update hit/miss counters or entry.hit_count.
    /// Use for concurrent read access in the OmniKernel fast path.
    pub fn lookup_readonly(&self, intent_canonical: &[u8], min_confidence: f64) -> EngramResult {
        let key = Self::key(intent_canonical);
        match self.store.get(&key) {
            Some(entry) if entry.confidence >= min_confidence => EngramResult::Hit {
                value: entry.value.clone(),
                confidence: entry.confidence,
            },
            _ => EngramResult::Miss,
        }
    }

    /// Record a cache hit in telemetry. Call from the write path after
    /// a successful `lookup_readonly` to keep counters accurate.
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Evict entries whose confidence has dropped below `floor`.
    pub fn evict_stale(&mut self, floor: f64) {
        self.store.retain(|_, e| e.confidence >= floor);
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Hit rate as a fraction (0–1).  Returns 0.0 if no queries yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    fn key(intent_canonical: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"engram/v1:");
        hasher.update(intent_canonical);
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_and_miss() {
        let mut cache = EngramCache::new();
        cache.insert(b"what is 2+2", "4", 0.99, 1000);

        let r = cache.lookup(b"what is 2+2", 0.95);
        assert!(r.is_hit());
        assert_eq!(r.value(), Some("4".to_string()));

        let r2 = cache.lookup(b"what is the capital of mars", 0.95);
        assert!(!r2.is_hit());
    }

    #[test]
    fn test_confidence_gate() {
        let mut cache = EngramCache::new();
        cache.insert(b"stale fact", "old answer", 0.60, 1000);
        // min_confidence=0.95 → miss (0.60 < 0.95)
        assert!(!cache.lookup(b"stale fact", 0.95).is_hit());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = EngramCache::new();
        cache.insert(b"a", "x", 0.99, 0);
        cache.lookup(b"a", 0.95); // hit
        cache.lookup(b"b", 0.95); // miss
        assert!((cache.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lookup_readonly_returns_hit_without_mutating() {
        let mut cache = EngramCache::new();
        cache.insert(b"intent_bytes", "Paris", 0.99, 1000);

        let hits_before = cache.hits;
        let result = cache.lookup_readonly(b"intent_bytes", 0.95);
        assert!(result.is_hit());
        assert_eq!(result.value(), Some("Paris".to_string()));
        // hits counter should NOT change (read-only)
        assert_eq!(cache.hits, hits_before);
    }

    #[test]
    fn test_lookup_readonly_misses_below_confidence() {
        let mut cache = EngramCache::new();
        cache.insert(b"stale", "old", 0.60, 1000);
        let result = cache.lookup_readonly(b"stale", 0.95);
        assert!(!result.is_hit());
        // misses counter should NOT change either
        assert_eq!(cache.misses, 0);
    }

    #[test]
    fn test_record_hit_increments() {
        let mut cache = EngramCache::new();
        let before = cache.hits;
        cache.record_hit();
        assert_eq!(cache.hits, before + 1);
    }

    #[test]
    fn test_evict_stale() {
        let mut cache = EngramCache::new();
        cache.insert(b"fresh", "yes", 0.99, 0);
        cache.insert(b"stale", "old", 0.40, 0);
        cache.evict_stale(0.95);
        assert_eq!(cache.len(), 1);
    }
}
