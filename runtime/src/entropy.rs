// src/entropy.rs - EntropyPool Pattern Implementation
// Tiered fallback system for cryptographic operations
//
// Architecture: 4-tier entropy sourcing with latency monitoring
// Tier 1: Pre-filled entropy pool (fastest, ~0.1ms)
// Tier 2: OS-level CSPRNG (reliable, ~1ms)
// Tier 3: Hardware RNG if available (varies, ~2-10ms)
// Tier 4: Emergency fallback (timestamped, >10ms warning)
//
// References:
// - TaskMaster SAPE v1.∞ report: EntropyPool pattern
// - Ihsān principle: Robustness (0.06 weight) - system resilience

use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Pool size in bytes (4096 = 4KB, sufficient for ~128 256-bit keys)
const POOL_SIZE: usize = 4096;

/// Latency thresholds for tier selection (microseconds)
const TIER1_LATENCY_US: u64 = 100; // Pool retrieval
const TIER2_LATENCY_US: u64 = 1000; // OS CSPRNG
const TIER3_LATENCY_US: u64 = 10000; // Hardware RNG
const REFILL_THRESHOLD: usize = 512; // Trigger async refill when pool drops below this

/// Entropy source tier for metrics/debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyTier {
    Pool,      // Tier 1: Pre-filled pool
    OsCsprng,  // Tier 2: OS-level CSPRNG (getrandom/urandom)
    Hardware,  // Tier 3: Hardware RNG (if available)
    Emergency, // Tier 4: Fallback (timestamp + counter)
}

impl std::fmt::Display for EntropyTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntropyTier::Pool => write!(f, "POOL"),
            EntropyTier::OsCsprng => write!(f, "OS_CSPRNG"),
            EntropyTier::Hardware => write!(f, "HARDWARE"),
            EntropyTier::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

/// Result of entropy generation with metadata
#[derive(Debug, Clone)]
pub struct EntropyResult {
    /// Generated entropy bytes
    pub bytes: Vec<u8>,
    /// Source tier used
    pub tier: EntropyTier,
    /// Generation latency
    pub latency: Duration,
    /// Sequence number for audit trail
    pub sequence: u64,
}

/// Metrics for entropy pool operations
#[derive(Debug, Default)]
pub struct EntropyMetrics {
    /// Total entropy requests
    pub total_requests: AtomicU64,
    /// Requests served from pool (Tier 1)
    pub pool_hits: AtomicU64,
    /// Requests requiring OS CSPRNG (Tier 2)
    pub os_requests: AtomicU64,
    /// Requests using hardware RNG (Tier 3)
    pub hardware_requests: AtomicU64,
    /// Emergency fallback invocations (Tier 4)
    pub emergency_fallbacks: AtomicU64,
    /// Pool refill operations
    pub refills: AtomicU64,
    /// Total bytes generated
    pub bytes_generated: AtomicU64,
}

impl EntropyMetrics {
    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "EntropyMetrics {{ requests: {}, pool_hits: {}, os: {}, hw: {}, emergency: {}, refills: {}, bytes: {} }}",
            self.total_requests.load(Ordering::Relaxed),
            self.pool_hits.load(Ordering::Relaxed),
            self.os_requests.load(Ordering::Relaxed),
            self.hardware_requests.load(Ordering::Relaxed),
            self.emergency_fallbacks.load(Ordering::Relaxed),
            self.refills.load(Ordering::Relaxed),
            self.bytes_generated.load(Ordering::Relaxed),
        )
    }
}

/// EntropyPool - tiered cryptographic entropy source
///
/// Provides resilient entropy generation with automatic fallback
/// when primary sources are slow or unavailable.
pub struct EntropyPool {
    /// Pre-filled entropy buffer
    pool: Arc<Mutex<Vec<u8>>>,
    /// Current read position in pool
    position: Arc<Mutex<usize>>,
    /// Sequence counter for audit trail
    sequence: AtomicU64,
    /// Whether hardware RNG is available
    hardware_available: AtomicBool,
    /// Metrics for monitoring
    pub metrics: Arc<EntropyMetrics>,
    /// Whether async refill is in progress
    refilling: AtomicBool,
}

impl EntropyPool {
    /// Create a new entropy pool with pre-filled buffer
    pub fn new() -> Self {
        let mut pool = vec![0u8; POOL_SIZE];

        // Initial fill from OS CSPRNG
        let filled = Self::fill_from_os(&mut pool);
        if filled < POOL_SIZE {
            warn!(
                filled = filled,
                target = POOL_SIZE,
                "Initial entropy pool not fully filled, using emergency supplement"
            );
            Self::emergency_fill(&mut pool[filled..]);
        }

        // Check hardware RNG availability
        let hardware_available = Self::check_hardware_rng();

        info!(
            pool_size = POOL_SIZE,
            hardware_available = hardware_available,
            "🎲 EntropyPool initialized"
        );

        Self {
            pool: Arc::new(Mutex::new(pool)),
            position: Arc::new(Mutex::new(0)),
            sequence: AtomicU64::new(1),
            hardware_available: AtomicBool::new(hardware_available),
            metrics: Arc::new(EntropyMetrics::default()),
            refilling: AtomicBool::new(false),
        }
    }

    /// Get next sequence number
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    /// Generate entropy bytes with tiered fallback
    pub fn generate(&self, len: usize) -> EntropyResult {
        let start = Instant::now();
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

        // Tier 1: Try pool first
        if let Some(bytes) = self.try_pool(len) {
            let latency = start.elapsed();
            if latency.as_micros() <= TIER1_LATENCY_US as u128 {
                self.metrics.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .bytes_generated
                    .fetch_add(len as u64, Ordering::Relaxed);

                // Check if we need to trigger async refill
                self.maybe_trigger_refill();

                return EntropyResult {
                    bytes,
                    tier: EntropyTier::Pool,
                    latency,
                    sequence: self.sequence.fetch_add(1, Ordering::SeqCst),
                };
            }
        }

        // Tier 2: OS CSPRNG
        let start2 = Instant::now();
        if let Some(bytes) = self.try_os_csprng(len) {
            let latency = start2.elapsed();
            if latency.as_micros() <= TIER2_LATENCY_US as u128 {
                self.metrics.os_requests.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .bytes_generated
                    .fetch_add(len as u64, Ordering::Relaxed);
                return EntropyResult {
                    bytes,
                    tier: EntropyTier::OsCsprng,
                    latency: start.elapsed(),
                    sequence: self.sequence.fetch_add(1, Ordering::SeqCst),
                };
            }
        }

        // Tier 3: Hardware RNG (if available and not too slow)
        if self.hardware_available.load(Ordering::Relaxed) {
            let start3 = Instant::now();
            if let Some(bytes) = self.try_hardware_rng(len) {
                let latency = start3.elapsed();
                if latency.as_micros() <= TIER3_LATENCY_US as u128 {
                    self.metrics
                        .hardware_requests
                        .fetch_add(1, Ordering::Relaxed);
                    self.metrics
                        .bytes_generated
                        .fetch_add(len as u64, Ordering::Relaxed);
                    return EntropyResult {
                        bytes,
                        tier: EntropyTier::Hardware,
                        latency: start.elapsed(),
                        sequence: self.sequence.fetch_add(1, Ordering::SeqCst),
                    };
                }
            }
        }

        // Tier 4: Emergency fallback (always succeeds)
        warn!(
            len = len,
            elapsed_us = start.elapsed().as_micros(),
            "⚠️ All primary entropy sources failed/slow, using emergency fallback"
        );
        self.metrics
            .emergency_fallbacks
            .fetch_add(1, Ordering::Relaxed);
        let bytes = self.emergency_generate(len);
        self.metrics
            .bytes_generated
            .fetch_add(len as u64, Ordering::Relaxed);

        EntropyResult {
            bytes,
            tier: EntropyTier::Emergency,
            latency: start.elapsed(),
            sequence: self.sequence.fetch_add(1, Ordering::SeqCst),
        }
    }

    /// Generate 32 bytes (256 bits) - common for cryptographic keys
    pub fn generate_256bit(&self) -> EntropyResult {
        self.generate(32)
    }

    /// Generate a unique ID using entropy
    pub fn generate_id(&self, prefix: &str) -> String {
        let entropy = self.generate(16);
        let hex: String = entropy.bytes.iter().map(|b| format!("{:02x}", b)).collect();
        format!("{}-{}", prefix, &hex[..16])
    }

    /// Try to get bytes from the pre-filled pool
    fn try_pool(&self, len: usize) -> Option<Vec<u8>> {
        let pool = self.pool.lock().ok()?;
        let mut pos = self.position.lock().ok()?;

        // Check if enough bytes available
        let remaining = POOL_SIZE.saturating_sub(*pos);
        if remaining < len {
            debug!(
                remaining = remaining,
                requested = len,
                "Pool exhausted, need refill"
            );
            return None;
        }

        let bytes = pool[*pos..*pos + len].to_vec();
        *pos += len;

        Some(bytes)
    }

    /// Try OS-level CSPRNG (getrandom on Linux, CryptGenRandom on Windows)
    fn try_os_csprng(&self, len: usize) -> Option<Vec<u8>> {
        let mut bytes = vec![0u8; len];

        // Use getrandom crate for cross-platform support
        match getrandom::getrandom(&mut bytes) {
            Ok(()) => Some(bytes),
            Err(e) => {
                debug!(error = %e, "OS CSPRNG failed");
                None
            }
        }
    }

    /// Try hardware RNG (x86 RDRAND)
    fn try_hardware_rng(&self, len: usize) -> Option<Vec<u8>> {
        // Hardware RNG via std::arch intrinsics is complex and platform-specific
        // For now, fallback to OS CSPRNG which may use RDRAND internally
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, getrandom may use RDRAND as entropy source
            self.try_os_csprng(len)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }

    /// Emergency fallback using timestamp + counter + previous entropy
    fn emergency_generate(&self, len: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(len);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let sequence = self.sequence.load(Ordering::Relaxed);

        // Use iterative hashing to expand entropy
        let mut seed = format!("{:032x}{:016x}", timestamp, sequence);

        while result.len() < len {
            let mut hasher = Sha256::new();
            hasher.update(seed.as_bytes());
            hasher.update(&result); // Chain previous output
            let hash = hasher.finalize();

            let take = std::cmp::min(32, len - result.len());
            result.extend_from_slice(&hash[..take]);

            // Update seed for next iteration
            seed = format!("{:064x}", hash);
        }

        result.truncate(len);
        result
    }

    /// Fill buffer from OS CSPRNG, returns number of bytes filled
    fn fill_from_os(buffer: &mut [u8]) -> usize {
        match getrandom::getrandom(buffer) {
            Ok(()) => buffer.len(),
            Err(e) => {
                warn!(error = %e, "Failed to fill from OS CSPRNG");
                0
            }
        }
    }

    /// Emergency fill using timestamp-based entropy
    fn emergency_fill(buffer: &mut [u8]) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let mut hasher = Sha256::new();
        hasher.update(format!("{:032x}", timestamp).as_bytes());
        let seed = hasher.finalize();

        // Expand seed to fill buffer
        let mut pos = 0;
        let mut current_seed = seed.to_vec();

        while pos < buffer.len() {
            let mut h = Sha256::new();
            h.update(&current_seed);
            h.update(pos.to_le_bytes());
            let hash = h.finalize();

            let take = std::cmp::min(32, buffer.len() - pos);
            buffer[pos..pos + take].copy_from_slice(&hash[..take]);
            pos += take;

            current_seed = hash.to_vec();
        }
    }

    /// Check if hardware RNG is available
    fn check_hardware_rng() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, assume hardware RNG is available
            // (most modern CPUs have RDRAND)
            true
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Maybe trigger async pool refill if running low
    fn maybe_trigger_refill(&self) {
        let pos = self.position.lock().map(|p| *p).unwrap_or(POOL_SIZE);
        let remaining = POOL_SIZE.saturating_sub(pos);

        if remaining < REFILL_THRESHOLD && !self.refilling.swap(true, Ordering::AcqRel) {
            debug!(
                remaining = remaining,
                threshold = REFILL_THRESHOLD,
                "Triggering async pool refill"
            );
            self.refill_pool();
        }
    }

    /// Refill the entropy pool
    pub fn refill_pool(&self) {
        let mut new_pool = vec![0u8; POOL_SIZE];
        let filled = Self::fill_from_os(&mut new_pool);

        if filled == POOL_SIZE {
            if let (Ok(mut pool), Ok(mut pos)) = (self.pool.lock(), self.position.lock()) {
                *pool = new_pool;
                *pos = 0;
                self.metrics.refills.fetch_add(1, Ordering::Relaxed);
                debug!("Pool refilled successfully");
            }
        } else {
            warn!(filled = filled, "Pool refill incomplete");
        }

        self.refilling.store(false, Ordering::Release);
    }

    /// Get current pool level (0.0 = empty, 1.0 = full)
    pub fn pool_level(&self) -> f64 {
        let pos = self.position.lock().map(|p| *p).unwrap_or(POOL_SIZE);
        let remaining = POOL_SIZE.saturating_sub(pos);
        remaining as f64 / POOL_SIZE as f64
    }

    /// Get metrics snapshot
    pub fn get_metrics(&self) -> String {
        self.metrics.summary()
    }
}

impl Default for EntropyPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Global entropy pool instance (lazy initialization)
static GLOBAL_POOL: std::sync::OnceLock<EntropyPool> = std::sync::OnceLock::new();

/// Get the global entropy pool
pub fn global_pool() -> &'static EntropyPool {
    GLOBAL_POOL.get_or_init(EntropyPool::new)
}

/// Convenience function: generate bytes from global pool
pub fn generate_entropy(len: usize) -> EntropyResult {
    global_pool().generate(len)
}

/// Convenience function: generate 256-bit (32 bytes) from global pool
pub fn generate_256bit() -> EntropyResult {
    global_pool().generate_256bit()
}

/// Convenience function: generate unique ID from global pool
pub fn generate_entropy_id(prefix: &str) -> String {
    global_pool().generate_id(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_pool_creation() {
        let pool = EntropyPool::new();
        assert!(
            pool.pool_level() > 0.9,
            "Pool should be nearly full on creation"
        );
    }

    #[test]
    fn test_entropy_generation() {
        let pool = EntropyPool::new();

        // Generate various sizes
        let r1 = pool.generate(16);
        assert_eq!(r1.bytes.len(), 16);

        let r2 = pool.generate(32);
        assert_eq!(r2.bytes.len(), 32);

        let r3 = pool.generate(64);
        assert_eq!(r3.bytes.len(), 64);

        // Verify uniqueness
        let r4 = pool.generate(32);
        assert_ne!(r2.bytes, r4.bytes, "Sequential generations should differ");
    }

    #[test]
    fn test_256bit_generation() {
        let pool = EntropyPool::new();
        let result = pool.generate_256bit();

        assert_eq!(result.bytes.len(), 32, "256 bits = 32 bytes");
        assert!(result.sequence > 0, "Should have sequence number");
    }

    #[test]
    fn test_id_generation() {
        let pool = EntropyPool::new();

        let id1 = pool.generate_id("TEST");
        let id2 = pool.generate_id("TEST");

        assert!(id1.starts_with("TEST-"));
        assert!(id2.starts_with("TEST-"));
        assert_ne!(id1, id2, "IDs should be unique");
        assert_eq!(id1.len(), 21, "TEST- (5) + 16 hex chars");
    }

    #[test]
    fn test_emergency_fallback() {
        let pool = EntropyPool::new();

        // Emergency generate always succeeds
        let bytes = pool.emergency_generate(64);
        assert_eq!(bytes.len(), 64);

        // Verify different calls produce different output
        let bytes2 = pool.emergency_generate(64);
        assert_ne!(bytes, bytes2);
    }

    #[test]
    fn test_pool_refill() {
        let pool = EntropyPool::new();

        // Drain most of the pool
        for _ in 0..100 {
            pool.generate(32);
        }

        // Manually refill
        pool.refill_pool();

        assert!(pool.pool_level() > 0.9, "Pool should be full after refill");
    }

    #[test]
    fn test_metrics() {
        let pool = EntropyPool::new();

        // Generate some entropy
        pool.generate(32);
        pool.generate(64);
        pool.generate_256bit();

        let metrics = pool.get_metrics();
        assert!(metrics.contains("requests: 3"));
        assert!(metrics.contains("bytes: 128")); // 32 + 64 + 32
    }

    #[test]
    fn test_global_pool() {
        let id1 = generate_entropy_id("GLOBAL");
        let id2 = generate_entropy_id("GLOBAL");

        assert_ne!(id1, id2, "Global pool should generate unique IDs");
    }

    #[test]
    fn test_tier_selection() {
        let pool = EntropyPool::new();

        // First few requests should come from pool (Tier 1) or OS CSPRNG (Tier 2)
        // Note: Under heavy system load or in CI, pool access may exceed latency threshold
        // causing fallback to OS CSPRNG - both are valid production tiers
        let r1 = pool.generate(32);
        assert!(
            matches!(r1.tier, EntropyTier::Pool | EntropyTier::OsCsprng),
            "Fresh pool should serve from Tier 1 (Pool) or Tier 2 (OsCsprng), got {:?}",
            r1.tier
        );

        // Verify we got valid entropy regardless of tier
        assert_eq!(r1.bytes.len(), 32, "Should generate requested length");
        assert!(
            r1.bytes.iter().any(|&b| b != 0),
            "Entropy should not be all zeros"
        );
    }
}
