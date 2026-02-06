//! Invariant Deduplication Cache
//!
//! TRICK 2: O(1) uniqueness check using BLAKE3 hashing.
//!
//! Prevents redundant work by tracking:
//! - Contract address + bytecode prefix hash
//! - First seen timestamp
//! - Bounty estimate (for prioritization)

use blake3::Hasher;
use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Metadata for cached invariants
#[derive(Debug, Clone, Copy)]
pub struct InvariantMeta {
    /// First seen timestamp (nanoseconds)
    pub first_seen: u64,
    /// Estimated bounty (USD cents)
    pub bounty_estimate: u64,
    /// Whether it's been submitted
    pub submitted: bool,
}

/// Thread-safe invariant deduplication cache
pub struct InvariantCache {
    /// Hash → Metadata map (fixed capacity)
    cache: RwLock<FxHashMap<[u8; 32], InvariantMeta>>,
    /// Maximum capacity
    capacity: usize,
}

impl InvariantCache {
    /// Create new cache with fixed capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
            capacity,
        }
    }

    /// Compute invariant hash from address and bytecode
    /// Uses only first 1KB for speed
    #[inline]
    pub fn compute_hash(addr: &[u8; 20], bytecode: &[u8]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(addr);
        hasher.update(&bytecode[..bytecode.len().min(1024)]);
        *hasher.finalize().as_bytes()
    }

    /// Check if invariant exists, insert if not
    /// Returns true if NEW (should process), false if DUPLICATE
    pub fn check_and_insert(&self, addr: &[u8; 20], bytecode: &[u8]) -> bool {
        let hash = Self::compute_hash(addr, bytecode);

        // Fast path: read lock check
        {
            let cache = self.cache.read().unwrap();
            if cache.contains_key(&hash) {
                return false; // Duplicate
            }
        }

        // Slow path: write lock insert
        {
            let mut cache = self.cache.write().unwrap();

            // Double-check after acquiring write lock
            if cache.contains_key(&hash) {
                return false;
            }

            // Check capacity (evict oldest if needed)
            if cache.len() >= self.capacity {
                // Simple eviction: remove first entry (in practice, use LRU)
                if let Some(key) = cache.keys().next().copied() {
                    cache.remove(&key);
                }
            }

            // Insert new entry
            cache.insert(
                hash,
                InvariantMeta {
                    first_seen: crate::pipeline::now_nanos(),
                    bounty_estimate: 0,
                    submitted: false,
                },
            );
        }

        true // New invariant
    }

    /// Check if hash exists (read-only)
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        let cache = self.cache.read().unwrap();
        cache.contains_key(hash)
    }

    /// Get metadata for hash
    pub fn get(&self, hash: &[u8; 32]) -> Option<InvariantMeta> {
        let cache = self.cache.read().unwrap();
        cache.get(hash).copied()
    }

    /// Update metadata for hash
    pub fn update(&self, hash: &[u8; 32], f: impl FnOnce(&mut InvariantMeta)) {
        let mut cache = self.cache.write().unwrap();
        if let Some(meta) = cache.get_mut(hash) {
            f(meta);
        }
    }

    /// Mark as submitted
    pub fn mark_submitted(&self, addr: &[u8; 20], bytecode: &[u8]) {
        let hash = Self::compute_hash(addr, bytecode);
        self.update(&hash, |meta| {
            meta.submitted = true;
        });
    }

    /// Get current size
    pub fn len(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get all hashes (for export)
    pub fn get_all_hashes(&self) -> Vec<[u8; 32]> {
        let cache = self.cache.read().unwrap();
        cache.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariant_cache_new() {
        let cache = InvariantCache::new(100);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_check_and_insert_new() {
        let cache = InvariantCache::new(100);
        let addr = [1u8; 20];
        let bytecode = b"test bytecode";

        // First insert should return true
        assert!(cache.check_and_insert(&addr, bytecode));
        assert_eq!(cache.len(), 1);

        // Second insert should return false (duplicate)
        assert!(!cache.check_and_insert(&addr, bytecode));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_different_addresses() {
        let cache = InvariantCache::new(100);
        let addr1 = [1u8; 20];
        let addr2 = [2u8; 20];
        let bytecode = b"same bytecode";

        // Different addresses should both be inserted
        assert!(cache.check_and_insert(&addr1, bytecode));
        assert!(cache.check_and_insert(&addr2, bytecode));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_capacity_eviction() {
        let cache = InvariantCache::new(3);

        for i in 0..5 {
            let addr = [i as u8; 20];
            cache.check_and_insert(&addr, b"bytecode");
        }

        // Should have evicted 2 entries
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let addr = [1u8; 20];
        let bytecode = b"test bytecode";

        let hash1 = InvariantCache::compute_hash(&addr, bytecode);
        let hash2 = InvariantCache::compute_hash(&addr, bytecode);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_mark_submitted() {
        let cache = InvariantCache::new(100);
        let addr = [1u8; 20];
        let bytecode = b"test bytecode";

        cache.check_and_insert(&addr, bytecode);
        cache.mark_submitted(&addr, bytecode);

        let hash = InvariantCache::compute_hash(&addr, bytecode);
        let meta = cache.get(&hash).unwrap();
        assert!(meta.submitted);
    }
}
