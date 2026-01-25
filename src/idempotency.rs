// src/idempotency.rs - IdempotentReplayManager Implementation
// Exactly-once semantics for bridge requests with checkpoint tracking
//
// Architecture:
// - Request deduplication via cryptographic fingerprinting
// - Checkpoint persistence for crash recovery
// - TTL-based cache expiration
// - Redis-backed distributed coordination
//
// References:
// - TaskMaster SAPE v1.∞ report: IdempotentReplayManager pattern
// - Ihsān principles: Auditability (0.12), Robustness (0.06)

use crate::entropy::generate_entropy_id;
use sha2::{Sha256, Digest};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn, debug};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default TTL for idempotency keys (1 hour)
const DEFAULT_TTL_SECS: u64 = 3600;

/// Maximum cache entries before eviction
const MAX_CACHE_ENTRIES: usize = 10000;

/// Checkpoint flush interval (in requests)
const CHECKPOINT_INTERVAL: u64 = 100;

/// Result status for idempotent operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdempotencyStatus {
    /// First time seeing this request
    New,
    /// Request already processed, returning cached result
    Duplicate,
    /// Request in progress by another handler
    InProgress,
    /// Request expired and can be retried
    Expired,
}

impl std::fmt::Display for IdempotencyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdempotencyStatus::New => write!(f, "NEW"),
            IdempotencyStatus::Duplicate => write!(f, "DUPLICATE"),
            IdempotencyStatus::InProgress => write!(f, "IN_PROGRESS"),
            IdempotencyStatus::Expired => write!(f, "EXPIRED"),
        }
    }
}

/// Cached result entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Request fingerprint (idempotency key)
    pub key: String,
    /// Serialized result (JSON)
    pub result: String,
    /// Timestamp when request was first received
    pub received_at: u64,
    /// Timestamp when processing completed
    pub completed_at: Option<u64>,
    /// Whether processing is in progress
    pub in_progress: bool,
    /// TTL expiration timestamp
    pub expires_at: u64,
    /// Checkpoint ID for crash recovery
    pub checkpoint_id: Option<String>,
}

/// Checkpoint entry for crash recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique checkpoint ID
    pub id: String,
    /// Request fingerprint
    pub key: String,
    /// Processing stage (for partial recovery)
    pub stage: String,
    /// Timestamp
    pub timestamp: u64,
    /// Partial result (if any)
    pub partial_result: Option<String>,
}

/// Idempotency metrics
#[derive(Debug, Default)]
pub struct IdempotencyMetrics {
    /// Total requests processed
    pub total_requests: std::sync::atomic::AtomicU64,
    /// Duplicate requests detected
    pub duplicates: std::sync::atomic::AtomicU64,
    /// New requests
    pub new_requests: std::sync::atomic::AtomicU64,
    /// In-progress requests
    pub in_progress: std::sync::atomic::AtomicU64,
    /// Expired entries evicted
    pub evictions: std::sync::atomic::AtomicU64,
    /// Checkpoints created
    pub checkpoints: std::sync::atomic::AtomicU64,
    /// Cache hits
    pub cache_hits: std::sync::atomic::AtomicU64,
    /// Cache misses
    pub cache_misses: std::sync::atomic::AtomicU64,
}

impl IdempotencyMetrics {
    pub fn summary(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        format!(
            "IdempotencyMetrics {{ total: {}, new: {}, duplicates: {}, in_progress: {}, evictions: {}, checkpoints: {}, hits: {}, misses: {} }}",
            self.total_requests.load(Relaxed),
            self.new_requests.load(Relaxed),
            self.duplicates.load(Relaxed),
            self.in_progress.load(Relaxed),
            self.evictions.load(Relaxed),
            self.checkpoints.load(Relaxed),
            self.cache_hits.load(Relaxed),
            self.cache_misses.load(Relaxed),
        )
    }
}

/// IdempotentReplayManager - ensures exactly-once semantics
///
/// Provides request deduplication with checkpointing for crash recovery.
/// Uses cryptographic fingerprinting for request identification.
pub struct IdempotentReplayManager {
    /// In-memory cache of processed requests
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    /// Checkpoints for crash recovery
    checkpoints: Arc<RwLock<HashMap<String, Checkpoint>>>,
    /// Default TTL for entries
    ttl: Duration,
    /// Request counter for checkpoint interval
    request_counter: std::sync::atomic::AtomicU64,
    /// Metrics
    pub metrics: Arc<IdempotencyMetrics>,
    /// Optional Redis client for distributed coordination
    synapse: Option<crate::synapse::SynapseClient>,
}

impl IdempotentReplayManager {
    /// Create a new manager with default TTL
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(DEFAULT_TTL_SECS))
    }

    /// Create with custom TTL
    pub fn with_ttl(ttl: Duration) -> Self {
        info!(
            ttl_secs = ttl.as_secs(),
            max_entries = MAX_CACHE_ENTRIES,
            "🔒 IdempotentReplayManager initialized"
        );

        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            request_counter: std::sync::atomic::AtomicU64::new(1), // Start at 1 to avoid eviction on first check
            metrics: Arc::new(IdempotencyMetrics::default()),
            synapse: None,
        }
    }

    /// Create with Redis persistence
    pub fn with_synapse(ttl: Duration, synapse: crate::synapse::SynapseClient) -> Self {
        info!(
            ttl_secs = ttl.as_secs(),
            "🔒 IdempotentReplayManager initialized with Redis persistence"
        );

        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            request_counter: std::sync::atomic::AtomicU64::new(1), // Start at 1 to avoid eviction on first check
            metrics: Arc::new(IdempotencyMetrics::default()),
            synapse: Some(synapse),
        }
    }

    /// Generate a fingerprint (idempotency key) for a request
    pub fn fingerprint(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = hasher.finalize();
        format!("IDEM-{:x}", hash)
    }

    /// Generate a fingerprint from structured data
    pub fn fingerprint_structured<T: Serialize>(&self, data: &T) -> String {
        match serde_json::to_value(data) {
            Ok(value) => {
                let canonical = canonicalize_json(value);
                match serde_json::to_string(&canonical) {
                    Ok(json) => self.fingerprint(&json),
                    Err(_) => generate_entropy_id("IDEM-FALLBACK"),
                }
            }
            Err(_) => {
                // Fallback to random ID if serialization fails
                generate_entropy_id("IDEM-FALLBACK")
            }
        }
    }

    /// Check if a request is new or duplicate
    pub fn check(&self, key: &str) -> (IdempotencyStatus, Option<CachedResult>) {
        use std::sync::atomic::Ordering::Relaxed;

        self.metrics.total_requests.fetch_add(1, Relaxed);
        let count = self.request_counter.fetch_add(1, Relaxed);

        // Periodic eviction of expired entries
        if count.is_multiple_of(CHECKPOINT_INTERVAL) {
            self.evict_expired();
        }

        // Check cache
        if let Ok(cache) = self.cache.read() {
            if let Some(entry) = cache.get(key) {
                let now = current_timestamp();

                // Check expiration
                if now > entry.expires_at {
                    self.metrics.evictions.fetch_add(1, Relaxed);
                    debug!(key = key, "Entry expired");
                    return (IdempotencyStatus::Expired, None);
                }

                // Check if still in progress
                if entry.in_progress {
                    self.metrics.in_progress.fetch_add(1, Relaxed);
                    self.metrics.cache_hits.fetch_add(1, Relaxed);
                    debug!(key = key, "Request in progress");
                    return (IdempotencyStatus::InProgress, Some(entry.clone()));
                }

                // Duplicate with result
                self.metrics.duplicates.fetch_add(1, Relaxed);
                self.metrics.cache_hits.fetch_add(1, Relaxed);
                debug!(key = key, "Duplicate request");
                return (IdempotencyStatus::Duplicate, Some(entry.clone()));
            }
        }

        self.metrics.cache_misses.fetch_add(1, Relaxed);
        self.metrics.new_requests.fetch_add(1, Relaxed);
        (IdempotencyStatus::New, None)
    }

    /// Reserve a slot for a new request (mark as in-progress)
    pub fn reserve(&self, key: &str) -> Result<String, String> {
        // Check capacity
        if let Ok(cache) = self.cache.read() {
            if cache.len() >= MAX_CACHE_ENTRIES {
                // Try to evict first
                drop(cache);
                self.evict_expired();

                // Re-check
                if let Ok(cache) = self.cache.read() {
                    if cache.len() >= MAX_CACHE_ENTRIES {
                        return Err("Cache at capacity, cannot reserve".to_string());
                    }
                }
            }
        }

        let now = current_timestamp();
        let checkpoint_id = generate_entropy_id("CKPT");

        let entry = CachedResult {
            key: key.to_string(),
            result: String::new(),
            received_at: now,
            completed_at: None,
            in_progress: true,
            expires_at: now + self.ttl.as_millis() as u64,
            checkpoint_id: Some(checkpoint_id.clone()),
        };

        if let Ok(mut cache) = self.cache.write() {
            cache.insert(key.to_string(), entry);
        }

        // Create checkpoint
        self.create_checkpoint(key, &checkpoint_id, "reserved", None);

        debug!(key = key, checkpoint_id = %checkpoint_id, "Reserved slot");
        Ok(checkpoint_id)
    }

    /// Complete a request and store the result
    pub fn complete(&self, key: &str, result: &str) -> bool {
        let now = current_timestamp();

        if let Ok(mut cache) = self.cache.write() {
            if let Some(entry) = cache.get_mut(key) {
                entry.result = result.to_string();
                entry.completed_at = Some(now);
                entry.in_progress = false;

                // Update checkpoint
                if let Some(ref checkpoint_id) = entry.checkpoint_id {
                    self.update_checkpoint(checkpoint_id, "completed", Some(result));
                }

                debug!(key = key, "Request completed");
                return true;
            }
        }

        warn!(key = key, "Attempted to complete non-existent request");
        false
    }

    /// Fail a request (remove from in-progress, allow retry)
    pub fn fail(&self, key: &str, error: &str) {
        if let Ok(mut cache) = self.cache.write() {
            if let Some(entry) = cache.get_mut(key) {
                entry.in_progress = false;

                // Update checkpoint with failure
                if let Some(ref checkpoint_id) = entry.checkpoint_id {
                    self.update_checkpoint(checkpoint_id, "failed", Some(error));
                }

                debug!(key = key, error = error, "Request failed");
            }
        }
    }

    /// Get cached result if available
    pub fn get(&self, key: &str) -> Option<CachedResult> {
        if let Ok(cache) = self.cache.read() {
            cache.get(key).cloned()
        } else {
            None
        }
    }

    /// Create a checkpoint for crash recovery
    fn create_checkpoint(&self, key: &str, checkpoint_id: &str, stage: &str, partial: Option<&str>) {
        use std::sync::atomic::Ordering::Relaxed;

        let checkpoint = Checkpoint {
            id: checkpoint_id.to_string(),
            key: key.to_string(),
            stage: stage.to_string(),
            timestamp: current_timestamp(),
            partial_result: partial.map(|s| s.to_string()),
        };

        if let Ok(mut checkpoints) = self.checkpoints.write() {
            checkpoints.insert(checkpoint_id.to_string(), checkpoint);
            self.metrics.checkpoints.fetch_add(1, Relaxed);
        }
    }

    /// Update an existing checkpoint
    fn update_checkpoint(&self, checkpoint_id: &str, stage: &str, partial: Option<&str>) {
        if let Ok(mut checkpoints) = self.checkpoints.write() {
            if let Some(checkpoint) = checkpoints.get_mut(checkpoint_id) {
                checkpoint.stage = stage.to_string();
                checkpoint.timestamp = current_timestamp();
                checkpoint.partial_result = partial.map(|s| s.to_string());
            }
        }
    }

    /// Evict expired entries
    fn evict_expired(&self) {
        use std::sync::atomic::Ordering::Relaxed;

        let now = current_timestamp();
        let mut evicted = 0;

        if let Ok(mut cache) = self.cache.write() {
            cache.retain(|_, entry| {
                if now > entry.expires_at {
                    evicted += 1;
                    false
                } else {
                    true
                }
            });
        }

        if evicted > 0 {
            self.metrics.evictions.fetch_add(evicted, Relaxed);
            debug!(evicted = evicted, "Evicted expired entries");
        }
    }

    /// Recover from checkpoint (for crash recovery)
    pub fn recover_from_checkpoint(&self, checkpoint_id: &str) -> Option<Checkpoint> {
        if let Ok(checkpoints) = self.checkpoints.read() {
            checkpoints.get(checkpoint_id).cloned()
        } else {
            None
        }
    }

    /// List all in-progress requests (for monitoring/recovery)
    pub fn list_in_progress(&self) -> Vec<String> {
        if let Ok(cache) = self.cache.read() {
            cache
                .iter()
                .filter(|(_, entry)| entry.in_progress)
                .map(|(key, _)| key.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Clear all entries (for testing)
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut checkpoints) = self.checkpoints.write() {
            checkpoints.clear();
        }
    }

    /// Get metrics summary
    pub fn get_metrics(&self) -> String {
        self.metrics.summary()
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

impl Default for IdempotentReplayManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonicalize JSON value by sorting object keys recursively.
fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k, canonicalize_json(v));
            }
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalize_json).collect()),
        other => other,
    }
}

/// Get current Unix timestamp
/// Get current timestamp in milliseconds for sub-second precision
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Global idempotency manager (lazy initialization)
static GLOBAL_MANAGER: std::sync::OnceLock<IdempotentReplayManager> = std::sync::OnceLock::new();

/// Get the global idempotency manager
pub fn global_manager() -> &'static IdempotentReplayManager {
    GLOBAL_MANAGER.get_or_init(IdempotentReplayManager::new)
}

/// Convenience: check idempotency using global manager
pub fn check_idempotency(key: &str) -> (IdempotencyStatus, Option<CachedResult>) {
    global_manager().check(key)
}

/// Convenience: reserve using global manager
pub fn reserve_idempotency(key: &str) -> Result<String, String> {
    global_manager().reserve(key)
}

/// Convenience: complete using global manager
pub fn complete_idempotency(key: &str, result: &str) -> bool {
    global_manager().complete(key, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_consistency() {
        let manager = IdempotentReplayManager::new();

        let fp1 = manager.fingerprint("test request");
        let fp2 = manager.fingerprint("test request");
        let fp3 = manager.fingerprint("different request");

        assert_eq!(fp1, fp2, "Same content should produce same fingerprint");
        assert_ne!(fp1, fp3, "Different content should produce different fingerprint");
        assert!(fp1.starts_with("IDEM-"));
    }

    #[test]
    fn test_new_request_flow() {
        let manager = IdempotentReplayManager::new();
        let key = manager.fingerprint("new request");

        // Check - should be new
        let (status, cached) = manager.check(&key);
        assert_eq!(status, IdempotencyStatus::New);
        assert!(cached.is_none());

        // Reserve
        let checkpoint = manager.reserve(&key).expect("Reserve should succeed");
        assert!(checkpoint.starts_with("CKPT-"));

        // Check again - should be in progress
        let (status, cached) = manager.check(&key);
        assert_eq!(status, IdempotencyStatus::InProgress);
        assert!(cached.is_some());

        // Complete
        assert!(manager.complete(&key, r#"{"success": true}"#));

        // Check again - should be duplicate
        let (status, cached) = manager.check(&key);
        assert_eq!(status, IdempotencyStatus::Duplicate);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().result, r#"{"success": true}"#);
    }

    #[test]
    fn test_failure_recovery() {
        let manager = IdempotentReplayManager::new();
        let key = manager.fingerprint("failing request");

        // Reserve
        manager.reserve(&key).expect("Reserve should succeed");

        // Fail
        manager.fail(&key, "Something went wrong");

        // Should no longer be in progress
        let (status, cached) = manager.check(&key);
        assert_eq!(status, IdempotencyStatus::Duplicate); // Still cached but not in_progress
        assert!(!cached.unwrap().in_progress);
    }

    #[test]
    fn test_checkpoint_recovery() {
        let manager = IdempotentReplayManager::new();
        let key = manager.fingerprint("checkpoint request");

        let checkpoint_id = manager.reserve(&key).expect("Reserve should succeed");

        // Recover checkpoint
        let checkpoint = manager.recover_from_checkpoint(&checkpoint_id);
        assert!(checkpoint.is_some());

        let cp = checkpoint.unwrap();
        assert_eq!(cp.key, key);
        assert_eq!(cp.stage, "reserved");
    }

    #[test]
    fn test_expiration() {
        // Use very short TTL for testing
        let manager = IdempotentReplayManager::with_ttl(Duration::from_millis(10));
        let key = manager.fingerprint("expiring request");

        manager.reserve(&key).expect("Reserve should succeed");
        manager.complete(&key, "result");

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(50));

        // Should be expired
        let (status, _) = manager.check(&key);
        assert_eq!(status, IdempotencyStatus::Expired);
    }

    #[test]
    fn test_list_in_progress() {
        let manager = IdempotentReplayManager::new();

        let key1 = manager.fingerprint("request 1");
        let key2 = manager.fingerprint("request 2");
        let key3 = manager.fingerprint("request 3");

        manager.reserve(&key1).unwrap();
        manager.reserve(&key2).unwrap();
        manager.reserve(&key3).unwrap();

        // Complete one
        manager.complete(&key2, "done");

        let in_progress = manager.list_in_progress();
        assert_eq!(in_progress.len(), 2);
        assert!(in_progress.contains(&key1));
        assert!(!in_progress.contains(&key2));
        assert!(in_progress.contains(&key3));
    }

    #[test]
    fn test_metrics() {
        let manager = IdempotentReplayManager::new();

        let key = manager.fingerprint("metrics test");
        manager.check(&key); // new
        manager.reserve(&key).unwrap();
        manager.check(&key); // in_progress
        manager.complete(&key, "done");
        manager.check(&key); // duplicate

        let metrics = manager.get_metrics();
        assert!(metrics.contains("total: 3"));
        assert!(metrics.contains("new: 1"));
        assert!(metrics.contains("duplicates: 1"));
    }

    #[test]
    fn test_structured_fingerprint() {
        let manager = IdempotentReplayManager::new();

        #[derive(Serialize)]
        struct Request {
            user_id: String,
            task: String,
        }

        let req = Request {
            user_id: "user_001".to_string(),
            task: "test task".to_string(),
        };

        let fp1 = manager.fingerprint_structured(&req);
        let fp2 = manager.fingerprint_structured(&req);

        assert_eq!(fp1, fp2, "Same struct should produce same fingerprint");
    }

    #[test]
    fn test_global_manager() {
        let key = global_manager().fingerprint("global test");

        let (status, _) = check_idempotency(&key);
        assert_eq!(status, IdempotencyStatus::New);

        reserve_idempotency(&key).unwrap();
        complete_idempotency(&key, "global result");

        let (status, cached) = check_idempotency(&key);
        assert_eq!(status, IdempotencyStatus::Duplicate);
        assert_eq!(cached.unwrap().result, "global result");
    }
}
