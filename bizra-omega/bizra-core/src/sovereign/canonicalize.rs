//! Canonicalize — State Canonicalization and Chain Integrity
//!
//! Reduces the full autopoietic state to a content-addressed canonical form
//! using domain-separated BLAKE3 hashing. Checkpoints are chained, each
//! referencing its predecessor, forming an immutable audit trail.
//!
//! # Design Principles
//!
//! - **Only hash invariants**: ihsan_ema, total_seed, quality_estimate, cycle_count
//!   (not ephemeral state like learning_rate or prediction_error_history)
//! - **Fixed-point arithmetic**: Floats are converted to fixed-point u64 (P=1_000_000)
//!   for deterministic hashing across platforms
//! - **Chain integrity**: Each checkpoint includes the hash of its predecessor,
//!   forming a hash chain analogous to the ExperienceLedger's episode chain
//!
//! # Standing on Giants
//!
//! - **Merkle** (1979): Hash trees for data integrity
//! - **Nakamoto** (2008): Hash-chained state transitions
//! - **Lamport** (1978): Logical clocks and ordering
//! - **Shannon** (1948): Information-theoretic canonicalization

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use super::autopoiesis::AutopoieticState;

/// Domain separation prefix for canonical hash operations.
const CANONICAL_DOMAIN: &[u8] = b"bizra-canonical-v1:";

/// Fixed-point precision multiplier (P = 1,000,000).
///
/// Converts f64 to u64 for deterministic cross-platform hashing:
///   fixed = (value * P) as u64
const FIXED_POINT_P: f64 = 1_000_000.0;

/// Canonicalize trait — Content-addressed state reduction.
///
/// Types implementing this trait can be reduced to a deterministic
/// 32-byte BLAKE3 hash that uniquely identifies their canonical state.
///
/// # Contract
///
/// - Two values with the same canonical hash are functionally identical
/// - The hash must be deterministic (same input always produces same output)
/// - Only invariant fields are included (not ephemeral state)
pub trait Canonicalize {
    /// Compute the canonical BLAKE3 hash of this value.
    fn to_canonical_hash(&self) -> [u8; 32];
}

/// Canonical Checkpoint — A content-addressed state snapshot.
///
/// Captures the essential invariants of the autopoietic state at a point
/// in time, using fixed-point arithmetic for deterministic hashing.
///
/// # Fields (all are invariants)
///
/// - `state_root`: BLAKE3 hash of the autopoietic state invariants
/// - `era_version`: Constitutional era version at checkpoint time
/// - `cycle_count`: Number of approved cycles at checkpoint time
/// - `ihsan_ema_fixed`: Ihsan EMA as fixed-point u64 (value * 1_000_000)
/// - `total_seed_fixed`: Total SEED as fixed-point u64 (value * 1_000_000)
/// - `timestamp`: Unix epoch seconds at checkpoint creation
/// - `prev_checkpoint`: Hash of the previous checkpoint ([0u8;32] for genesis)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalCheckpoint {
    /// BLAKE3 hash of the autopoietic state invariants.
    pub state_root: [u8; 32],
    /// Constitutional era version at checkpoint time.
    pub era_version: u32,
    /// Number of approved cycles at checkpoint time.
    pub cycle_count: u64,
    /// Ihsan EMA as fixed-point u64 (value * 1_000_000).
    pub ihsan_ema_fixed: u64,
    /// Total SEED as fixed-point u64 (value * 1_000_000).
    pub total_seed_fixed: u64,
    /// Unix epoch seconds at checkpoint creation.
    pub timestamp: u64,
    /// Hash of the previous checkpoint (chain link).
    pub prev_checkpoint: [u8; 32],
}

impl CanonicalCheckpoint {
    /// Create a new checkpoint from an autopoietic state.
    ///
    /// The `prev_checkpoint` links this checkpoint to its predecessor,
    /// forming the canonical chain. Use `[0u8; 32]` for the genesis checkpoint.
    pub fn from_state(
        state: &AutopoieticState,
        era_version: u32,
        prev_checkpoint: [u8; 32],
    ) -> Self {
        let state_root = state.to_canonical_hash();

        Self {
            state_root,
            era_version,
            cycle_count: state.cycle_count,
            ihsan_ema_fixed: to_fixed(state.ihsan_ema),
            total_seed_fixed: to_fixed(state.total_seed),
            timestamp: now_unix_secs(),
            prev_checkpoint,
        }
    }
}

impl Canonicalize for CanonicalCheckpoint {
    /// Compute the canonical hash of this checkpoint.
    ///
    /// Includes all fields plus the prev_checkpoint link, forming a
    /// hash chain where altering any checkpoint invalidates all successors.
    fn to_canonical_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(CANONICAL_DOMAIN);
        hasher.update(&self.state_root);
        hasher.update(&self.era_version.to_le_bytes());
        hasher.update(&self.cycle_count.to_le_bytes());
        hasher.update(&self.ihsan_ema_fixed.to_le_bytes());
        hasher.update(&self.total_seed_fixed.to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&self.prev_checkpoint);
        *hasher.finalize().as_bytes()
    }
}

impl Canonicalize for AutopoieticState {
    /// Compute the canonical hash of the autopoietic state.
    ///
    /// Only hashes invariant fields:
    /// - `total_seed` (fixed-point)
    /// - `ihsan_ema` (fixed-point)
    /// - `quality_estimate` (fixed-point)
    /// - `cycle_count`
    ///
    /// Ephemeral fields (learning_rate, prediction_error_history, improvement_streak)
    /// are excluded because they are implementation details of the learning algorithm,
    /// not canonical properties of the agent's achieved state.
    fn to_canonical_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(CANONICAL_DOMAIN);
        hasher.update(&to_fixed(self.total_seed).to_le_bytes());
        hasher.update(&to_fixed(self.ihsan_ema).to_le_bytes());
        hasher.update(&to_fixed(self.quality_estimate).to_le_bytes());
        hasher.update(&self.cycle_count.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// Canonical Chain — An append-only chain of canonical checkpoints.
///
/// Each checkpoint references its predecessor's hash, forming an
/// immutable audit trail. Analogous to the `ExperienceLedger`'s
/// episode chain but for autopoietic state transitions.
///
/// # Integrity Property
///
/// For any checkpoint C_k in the chain:
///   C_k.prev_checkpoint == canonical_hash(C_{k-1})
///
/// Altering any checkpoint invalidates all subsequent chain links.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalChain {
    /// Ordered list of canonical checkpoints.
    pub checkpoints: Vec<CanonicalCheckpoint>,
    /// Hash of the most recent checkpoint (chain head).
    pub current_root: [u8; 32],
}

impl CanonicalChain {
    /// Create a new empty canonical chain.
    pub fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
            current_root: [0u8; 32],
        }
    }

    /// Append a checkpoint to the chain.
    ///
    /// The checkpoint's `prev_checkpoint` must match `current_root`.
    /// After appending, `current_root` is updated to the new checkpoint's hash.
    ///
    /// Returns `true` if the checkpoint was appended, `false` if the chain
    /// link is invalid.
    pub fn append(&mut self, checkpoint: CanonicalCheckpoint) -> bool {
        if checkpoint.prev_checkpoint != self.current_root {
            return false;
        }

        let new_root = checkpoint.to_canonical_hash();
        self.checkpoints.push(checkpoint);
        self.current_root = new_root;
        true
    }

    /// Create and append a checkpoint from the current autopoietic state.
    ///
    /// Convenience method that creates the checkpoint with the correct
    /// `prev_checkpoint` link and appends it to the chain.
    pub fn checkpoint_state(
        &mut self,
        state: &AutopoieticState,
        era_version: u32,
    ) -> &CanonicalCheckpoint {
        let checkpoint = CanonicalCheckpoint::from_state(state, era_version, self.current_root);
        let new_root = checkpoint.to_canonical_hash();
        self.checkpoints.push(checkpoint);
        self.current_root = new_root;
        self.checkpoints.last().unwrap()
    }

    /// Verify the integrity of the entire chain.
    ///
    /// Checks that each checkpoint's `prev_checkpoint` matches the
    /// canonical hash of its predecessor.
    pub fn verify_integrity(&self) -> bool {
        let mut expected_prev = [0u8; 32];

        for checkpoint in &self.checkpoints {
            if checkpoint.prev_checkpoint != expected_prev {
                return false;
            }
            expected_prev = checkpoint.to_canonical_hash();
        }

        expected_prev == self.current_root
    }

    /// Get the number of checkpoints in the chain.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

impl Default for CanonicalChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert an f64 value to fixed-point u64 with precision P=1_000_000.
#[inline]
fn to_fixed(value: f64) -> u64 {
    (value * FIXED_POINT_P) as u64
}

/// Get current Unix timestamp in seconds.
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_hash_deterministic() {
        let mut state = AutopoieticState::new();
        state.execute_cycle(0.97, 0.90);

        let h1 = state.to_canonical_hash();
        let h2 = state.to_canonical_hash();
        assert_eq!(h1, h2, "Same state must produce same canonical hash");
    }

    #[test]
    fn test_chain_integrity() {
        let mut chain = CanonicalChain::new();
        let mut state = AutopoieticState::new();

        // Build a chain of 10 checkpoints
        for _ in 0..10 {
            state.execute_cycle(0.97, 0.90);
            chain.checkpoint_state(&state, 1);
        }

        assert_eq!(chain.len(), 10);
        assert!(
            chain.verify_integrity(),
            "Chain integrity must hold after sequential appends"
        );
    }

    #[test]
    fn test_different_states_different_hashes() {
        let mut state_a = AutopoieticState::new();
        let mut state_b = AutopoieticState::new();

        state_a.execute_cycle(0.97, 0.90);
        state_b.execute_cycle(0.99, 0.92);

        let hash_a = state_a.to_canonical_hash();
        let hash_b = state_b.to_canonical_hash();

        assert_ne!(
            hash_a, hash_b,
            "Different states must produce different canonical hashes"
        );
    }

    #[test]
    fn test_chain_rejects_invalid_link() {
        let mut chain = CanonicalChain::new();
        let state = AutopoieticState::new();

        // Create a checkpoint with wrong prev_checkpoint
        let bad_checkpoint = CanonicalCheckpoint {
            state_root: state.to_canonical_hash(),
            era_version: 1,
            cycle_count: 0,
            ihsan_ema_fixed: to_fixed(0.95),
            total_seed_fixed: 0,
            timestamp: 0,
            prev_checkpoint: [0xff; 32], // Wrong link
        };

        assert!(
            !chain.append(bad_checkpoint),
            "Chain should reject checkpoint with invalid prev_checkpoint"
        );
    }

    #[test]
    fn test_checkpoint_from_state() {
        let mut state = AutopoieticState::new();
        for _ in 0..5 {
            state.execute_cycle(0.97, 0.90);
        }

        let checkpoint = CanonicalCheckpoint::from_state(&state, 1, [0u8; 32]);
        assert_eq!(checkpoint.cycle_count, state.cycle_count);
        assert_eq!(checkpoint.ihsan_ema_fixed, to_fixed(state.ihsan_ema));
        assert_eq!(checkpoint.state_root, state.to_canonical_hash());
    }

    #[test]
    fn test_empty_chain() {
        let chain = CanonicalChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.verify_integrity());
        assert_eq!(chain.current_root, [0u8; 32]);
    }

    #[test]
    fn test_fixed_point_precision() {
        assert_eq!(to_fixed(0.95), 950_000);
        assert_eq!(to_fixed(1.0), 1_000_000);
        assert_eq!(to_fixed(0.0), 0);
        assert_eq!(to_fixed(0.999999), 999_999);
    }
}
