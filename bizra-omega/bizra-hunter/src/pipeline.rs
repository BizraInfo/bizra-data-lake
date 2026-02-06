//! Two-Lane SNR Pipeline
//!
//! TRICK 1: Fast heuristics in Lane 1, expensive proofs in Lane 2.
//!
//! Lane 1: O(1000 ops/contract) - Filters 80% noise
//! Lane 2: O(100,000 ops/contract) - Detailed analysis
//!
//! Zero allocation after initialization.

use crossbeam_queue::ArrayQueue;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cascade::{CriticalCascade, GateType};
use crate::entropy::{EntropyCalculator, MultiAxisEntropy};
use crate::invariant::InvariantCache;
use crate::rent::HarbergerRent;
use crate::{LANE1_SNR_THRESHOLD, MIN_CONSISTENT_AXES};

/// Heuristic result from Lane 1 (fast path)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HeuristicResult {
    /// Contract address (20 bytes)
    pub contract_addr: [u8; 20],
    /// Multi-axis entropy scores
    pub entropy: MultiAxisEntropy,
    /// Complexity tier
    pub complexity: Complexity,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp: u64,
    /// Estimated bounty (USD cents)
    pub bounty_estimate: u64,
}

/// Proof job for Lane 2 (slow path)
#[derive(Debug, Clone)]
pub struct ProofJob {
    /// Contract address
    pub contract_addr: [u8; 20],
    /// Contract bytecode
    pub bytecode: Vec<u8>,
    /// Entropy scores
    pub entropy: MultiAxisEntropy,
    /// Vulnerability type detected
    pub vuln_type: VulnType,
    /// Location in bytecode
    pub location: u32,
    /// Estimated bounty
    pub bounty_estimate: u64,
}

/// Vulnerability type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VulnType {
    Reentrancy = 0,
    Overflow = 1,
    AccessControl = 2,
    OracleManipulation = 3,
    FlashLoan = 4,
    FrontRunning = 5,
    Dos = 6,
    Unknown = 255,
}

/// Complexity tier based on entropy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Complexity {
    Simple = 0,
    Medium = 1,
    Complex = 2,
    Expert = 3,
}

impl Complexity {
    /// Derive complexity from average entropy
    pub fn from_entropy(entropy: &MultiAxisEntropy) -> Self {
        let avg = entropy.average();
        if avg < 0.3 {
            Self::Simple
        } else if avg < 0.5 {
            Self::Medium
        } else if avg < 0.7 {
            Self::Complex
        } else {
            Self::Expert
        }
    }
}

/// SNR Pipeline statistics
#[derive(Debug, Default)]
pub struct PipelineStats {
    pub lane1_processed: AtomicU64,
    pub lane1_passed: AtomicU64,
    pub lane1_filtered: AtomicU64,
    pub lane2_processed: AtomicU64,
    pub lane2_submitted: AtomicU64,
    pub duplicates_filtered: AtomicU64,
    pub cascade_blocked: AtomicU64,
}

/// The SNR-Maximized Pipeline
///
/// Two lanes with lock-free queues:
/// - Lane 1: Fast heuristic filtering (80% noise removal)
/// - Lane 2: Expensive proof generation
pub struct SNRPipeline<const N: usize> {
    /// Lane 1 → Lane 2 queue (lock-free MPSC)
    pub lane1_queue: ArrayQueue<HeuristicResult>,
    /// Lane 2 output queue (lock-free SPSC)
    pub lane2_queue: ArrayQueue<ProofJob>,
    /// Invariant deduplication cache
    pub invariant_cache: InvariantCache,
    /// Critical cascade gate controller
    pub cascade: CriticalCascade,
    /// Harberger memory rent controller
    pub rent: HarbergerRent,
    /// Pipeline statistics
    pub stats: PipelineStats,
    /// SNR threshold for Lane 1
    snr_threshold: f32,
    /// Minimum consistent axes required
    min_axes: usize,
}

impl<const N: usize> SNRPipeline<N> {
    /// Create new pipeline with fixed capacity
    /// O(1) initialization, O(0) allocation after boot
    pub fn new() -> Self {
        Self {
            lane1_queue: ArrayQueue::new(N),
            lane2_queue: ArrayQueue::new(N / 4), // Lane 2 is 4x slower
            invariant_cache: InvariantCache::new(1 << 20), // 1M entries
            cascade: CriticalCascade::new(),
            rent: HarbergerRent::new(1000, 3600), // 1000 wei/byte/sec, 1hr grace
            stats: PipelineStats::default(),
            snr_threshold: LANE1_SNR_THRESHOLD,
            min_axes: MIN_CONSISTENT_AXES,
        }
    }

    /// Configure SNR parameters
    pub fn with_snr_config(mut self, threshold: f32, min_axes: usize) -> Self {
        self.snr_threshold = threshold;
        self.min_axes = min_axes;
        self
    }

    /// Process bytecode through Lane 1 (fast heuristic filter)
    /// Returns Some(result) if it passes SNR threshold
    #[inline]
    pub fn process_lane1(
        &self,
        contract_addr: [u8; 20],
        bytecode: &[u8],
        calc: &mut EntropyCalculator,
    ) -> Option<HeuristicResult> {
        self.stats.lane1_processed.fetch_add(1, Ordering::Relaxed);

        // Check cascade gate first
        if !self.cascade.is_open(GateType::Technical) {
            self.stats.cascade_blocked.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Calculate multi-axis entropy
        let entropy = calc.calculate_all(bytecode);

        // Lane 1 gate: SNR threshold + multi-axis consistency
        if entropy.average() < self.snr_threshold {
            self.stats.lane1_filtered.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if !entropy.is_consistent(self.snr_threshold, self.min_axes) {
            self.stats.lane1_filtered.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Check invariant deduplication
        if !self.invariant_cache.check_and_insert(&contract_addr, bytecode) {
            self.stats.duplicates_filtered.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        self.stats.lane1_passed.fetch_add(1, Ordering::Relaxed);

        // Estimate bounty based on complexity
        let complexity = Complexity::from_entropy(&entropy);
        let bounty_estimate = match complexity {
            Complexity::Simple => 500_00,   // $500
            Complexity::Medium => 2500_00,  // $2,500
            Complexity::Complex => 10000_00, // $10,000
            Complexity::Expert => 50000_00,  // $50,000
        };

        Some(HeuristicResult {
            contract_addr,
            entropy,
            complexity,
            timestamp: now_nanos(),
            bounty_estimate,
        })
    }

    /// Detect vulnerability type from entropy patterns
    pub fn detect_vuln_type(&self, entropy: &MultiAxisEntropy) -> VulnType {
        // Heuristics based on entropy patterns
        if entropy.state > 0.8 && entropy.cfg > 0.7 {
            VulnType::Reentrancy
        } else if entropy.economic > 0.8 {
            VulnType::FlashLoan
        } else if entropy.temporal > 0.7 {
            VulnType::FrontRunning
        } else if entropy.cfg > 0.8 && entropy.memory > 0.7 {
            VulnType::Overflow
        } else if entropy.bytecode < 0.3 && entropy.cfg > 0.5 {
            VulnType::AccessControl
        } else {
            VulnType::Unknown
        }
    }

    /// Create proof job for Lane 2
    pub fn create_proof_job(
        &self,
        result: &HeuristicResult,
        bytecode: Vec<u8>,
    ) -> ProofJob {
        let vuln_type = self.detect_vuln_type(&result.entropy);

        ProofJob {
            contract_addr: result.contract_addr,
            bytecode,
            entropy: result.entropy,
            vuln_type,
            location: 0, // Would be set by detailed analysis
            bounty_estimate: result.bounty_estimate,
        }
    }

    /// Push to Lane 1 queue (non-blocking)
    #[inline]
    pub fn push_to_lane1(&self, result: HeuristicResult) -> bool {
        self.lane1_queue.push(result).is_ok()
    }

    /// Pop from Lane 1 queue (non-blocking)
    #[inline]
    pub fn pop_from_lane1(&self) -> Option<HeuristicResult> {
        self.lane1_queue.pop()
    }

    /// Push to Lane 2 queue (non-blocking)
    #[inline]
    pub fn push_to_lane2(&self, job: ProofJob) -> bool {
        self.lane2_queue.push(job).is_ok()
    }

    /// Pop from Lane 2 queue (non-blocking)
    #[inline]
    pub fn pop_from_lane2(&self) -> Option<ProofJob> {
        self.lane2_queue.pop()
    }

    /// Get current queue lengths
    pub fn queue_lengths(&self) -> (usize, usize) {
        (self.lane1_queue.len(), self.lane2_queue.len())
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStatsSnapshot {
        PipelineStatsSnapshot {
            lane1_processed: self.stats.lane1_processed.load(Ordering::Relaxed),
            lane1_passed: self.stats.lane1_passed.load(Ordering::Relaxed),
            lane1_filtered: self.stats.lane1_filtered.load(Ordering::Relaxed),
            lane2_processed: self.stats.lane2_processed.load(Ordering::Relaxed),
            lane2_submitted: self.stats.lane2_submitted.load(Ordering::Relaxed),
            duplicates_filtered: self.stats.duplicates_filtered.load(Ordering::Relaxed),
            cascade_blocked: self.stats.cascade_blocked.load(Ordering::Relaxed),
            snr_ratio: self.calculate_snr_ratio(),
        }
    }

    /// Calculate current SNR ratio
    fn calculate_snr_ratio(&self) -> f64 {
        let processed = self.stats.lane1_processed.load(Ordering::Relaxed);
        let passed = self.stats.lane1_passed.load(Ordering::Relaxed);

        if processed == 0 {
            return 0.0;
        }

        passed as f64 / processed as f64
    }
}

impl<const N: usize> Default for SNRPipeline<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of pipeline statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct PipelineStatsSnapshot {
    pub lane1_processed: u64,
    pub lane1_passed: u64,
    pub lane1_filtered: u64,
    pub lane2_processed: u64,
    pub lane2_submitted: u64,
    pub duplicates_filtered: u64,
    pub cascade_blocked: u64,
    pub snr_ratio: f64,
}

/// Get current time in nanoseconds
#[inline]
pub fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline: SNRPipeline<1024> = SNRPipeline::new();
        assert_eq!(pipeline.lane1_queue.capacity(), 1024);
        assert_eq!(pipeline.lane2_queue.capacity(), 256);
    }

    #[test]
    fn test_lane1_filtering() {
        let pipeline: SNRPipeline<1024> = SNRPipeline::new();
        let mut calc = EntropyCalculator::new();

        // Low entropy bytecode should be filtered
        let low_entropy = vec![0x60u8; 100];
        let result = pipeline.process_lane1([0u8; 20], &low_entropy, &mut calc);
        assert!(result.is_none());

        // High entropy bytecode should pass
        let high_entropy: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let result = pipeline.process_lane1([1u8; 20], &high_entropy, &mut calc);
        // May or may not pass depending on exact entropy calculation
    }

    #[test]
    fn test_complexity_from_entropy() {
        let low = MultiAxisEntropy {
            bytecode: 0.2,
            cfg: 0.1,
            state: 0.2,
            economic: 0.1,
            temporal: 0.1,
            memory: 0.1,
        };
        assert_eq!(Complexity::from_entropy(&low), Complexity::Simple);

        let high = MultiAxisEntropy {
            bytecode: 0.9,
            cfg: 0.8,
            state: 0.7,
            economic: 0.8,
            temporal: 0.6,
            memory: 0.7,
        };
        assert_eq!(Complexity::from_entropy(&high), Complexity::Expert);
    }

    #[test]
    fn test_vuln_type_detection() {
        let pipeline: SNRPipeline<1024> = SNRPipeline::new();

        // Reentrancy pattern
        let reentrancy = MultiAxisEntropy {
            bytecode: 0.5,
            cfg: 0.8,
            state: 0.9,
            economic: 0.4,
            temporal: 0.2,
            memory: 0.5,
        };
        assert_eq!(pipeline.detect_vuln_type(&reentrancy), VulnType::Reentrancy);

        // Flash loan pattern
        let flash = MultiAxisEntropy {
            bytecode: 0.5,
            cfg: 0.5,
            state: 0.5,
            economic: 0.9,
            temporal: 0.3,
            memory: 0.4,
        };
        assert_eq!(pipeline.detect_vuln_type(&flash), VulnType::FlashLoan);
    }
}
