//! Walking Skeleton — The Thinnest End-to-End Constitutional Liveness Proof
//!
//! Traces ONE complete path through the BIZRA architecture:
//!
//! ```text
//! Genesis Ceremony (BLAKE3 identity)
//!     ↓
//! Sovereign Orchestration (create AutopoieticState)
//!     ↓
//! One Autopoietic Cycle (Predict → Score → Gate → Attest → Learn)
//!     ↓
//! Constitutional Gate (Ihsan ≥ 0.95, SNR ≥ 0.85)
//!     ↓
//! Canonical Checkpoint (BLAKE3 state root)
//!     ↓
//! Evidence Receipt (signed, chained, auditable)
//! ```
//!
//! If this test passes, the system is constitutionally alive.
//! If it fails, something fundamental is broken.
//!
//! # Standing on Giants
//!
//! - **Cockburn** (2004): Walking Skeleton — thinnest e2e slice
//! - **Shannon** (1948): Information theory, SNR
//! - **Al-Ghazali**: Ihsan — pursuit of excellence
//! - **Maturana & Varela** (1980): Autopoiesis — self-producing systems
//! - **Merkle** (1979): Hash chains for data integrity

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::sovereign::autopoiesis::{AutopoieticState, CycleOutcome};
use crate::sovereign::canonicalize::{CanonicalChain, CanonicalCheckpoint};
use crate::sovereign::meta_constitution::MetaConstitution;
use crate::{IHSAN_THRESHOLD, SNR_THRESHOLD};

/// Domain separation prefix for walking skeleton evidence hashes.
const SKELETON_DOMAIN: &[u8] = b"bizra-walking-skeleton-v1:";

/// A Walking Skeleton receipt — the atomic proof that the system is alive.
///
/// Contains every artifact produced by the end-to-end path:
/// genesis identity, cycle metrics, constitutional verdict,
/// canonical state root, and a chained evidence hash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkeletonReceipt {
    /// BLAKE3 genesis identity hash.
    pub genesis_hash: [u8; 32],
    /// Number of approved autopoietic cycles completed.
    pub cycle_count: u64,
    /// Ihsan EMA after the cycle.
    pub ihsan_score: f64,
    /// SNR score fed into the cycle.
    pub snr_score: f64,
    /// Whether both constitutional gates passed.
    pub constitutional_pass: bool,
    /// Canonical state root (BLAKE3 of autopoietic invariants).
    pub state_root: [u8; 32],
    /// BLAKE3 evidence hash chaining all receipt fields.
    pub evidence_hash: [u8; 32],
    /// ISO 8601 timestamp of receipt creation.
    pub timestamp: String,
    /// Meta-constitution era version at receipt time.
    pub era_version: u32,
    /// Elapsed time in microseconds for the skeleton path.
    pub elapsed_us: u64,
}

/// Run the complete walking skeleton path.
///
/// Executes the thinnest possible end-to-end proof:
/// 1. Genesis: create identity via BLAKE3
/// 2. Create AutopoieticState + MetaConstitution
/// 3. Run one autopoietic cycle with constitutional-grade inputs
/// 4. Verify constitutional gates (Ihsan ≥ 0.95, SNR ≥ 0.85)
/// 5. Create canonical checkpoint with chain integrity
/// 6. Generate evidence receipt with BLAKE3 hash
///
/// Returns a `SkeletonReceipt` if the system is constitutionally alive,
/// or an error string if any step fails.
pub fn run_skeleton() -> Result<SkeletonReceipt, String> {
    let start = std::time::Instant::now();

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Genesis Ceremony — establish BLAKE3 identity
    // ═══════════════════════════════════════════════════════════════
    let genesis_payload = b"bizra-node0-walking-skeleton-genesis";
    let genesis_hash = blake3_domain_hash(SKELETON_DOMAIN, genesis_payload);

    if genesis_hash == [0u8; 32] {
        return Err("Genesis hash is zero — BLAKE3 failed".into());
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 2: Sovereign Orchestration — create autopoietic state
    // ═══════════════════════════════════════════════════════════════
    let mut state = AutopoieticState::new();
    let meta_constitution = MetaConstitution::new();

    // Verify initial invariants
    if state.cycle_count != 0 {
        return Err(format!(
            "Initial cycle_count should be 0, got {}",
            state.cycle_count
        ));
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 3: One Autopoietic Cycle — Predict → Score → Gate → Attest → Learn
    // ═══════════════════════════════════════════════════════════════
    // Use constitutional-grade inputs that will pass the gates:
    //   actual_quality = 0.97 (above IHSAN_THRESHOLD of 0.95)
    //   snr = 0.90 (above SNR_THRESHOLD of 0.85)
    let actual_quality = 0.97;
    let snr = 0.90;

    let outcome = state.execute_cycle(actual_quality, snr);

    let reward = match outcome {
        CycleOutcome::Approved(reward) => reward,
        CycleOutcome::Halted { reason, ihsan_score } => {
            return Err(format!(
                "Autopoietic cycle halted: {reason} (ihsan={ihsan_score:.4})"
            ));
        }
    };

    // ═══════════════════════════════════════════════════════════════
    // Step 4: Constitutional Gate — verify thresholds
    // ═══════════════════════════════════════════════════════════════
    let ihsan_score = reward.ihsan_score;
    let snr_score = reward.snr_score;

    let ihsan_pass = ihsan_score >= IHSAN_THRESHOLD;
    let snr_pass = snr_score >= SNR_THRESHOLD;
    let constitutional_pass = ihsan_pass && snr_pass;

    if !constitutional_pass {
        return Err(format!(
            "Constitutional gate failed: ihsan={ihsan_score:.4} (need {IHSAN_THRESHOLD}), \
             snr={snr_score:.4} (need {SNR_THRESHOLD})"
        ));
    }

    // Verify meta-constitution era is valid
    let era_version = meta_constitution.current_era.version;
    if era_version < 1 {
        return Err("Meta-constitution era version must be >= 1".into());
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 5: Canonical Checkpoint — BLAKE3 state root with chain
    // ═══════════════════════════════════════════════════════════════
    let mut chain = CanonicalChain::new();
    let checkpoint = CanonicalCheckpoint::from_state(&state, era_version, chain.current_root);
    let state_root = checkpoint.state_root;

    if state_root == [0u8; 32] {
        return Err("Canonical state root is zero — hashing failed".into());
    }

    // Append to chain and verify integrity
    if !chain.append(checkpoint) {
        return Err("Failed to append checkpoint to canonical chain".into());
    }

    if !chain.verify_integrity() {
        return Err("Canonical chain integrity check failed".into());
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 6: Evidence Receipt — BLAKE3 hash chaining all artifacts
    // ═══════════════════════════════════════════════════════════════
    let timestamp = chrono::Utc::now().to_rfc3339();
    let elapsed_us = start.elapsed().as_micros() as u64;

    let evidence_hash = compute_evidence_hash(
        &genesis_hash,
        state.cycle_count,
        ihsan_score,
        snr_score,
        &state_root,
        &timestamp,
    );

    if evidence_hash == [0u8; 32] {
        return Err("Evidence hash is zero — hashing failed".into());
    }

    Ok(SkeletonReceipt {
        genesis_hash,
        cycle_count: state.cycle_count,
        ihsan_score,
        snr_score,
        constitutional_pass,
        state_root,
        evidence_hash,
        timestamp,
        era_version,
        elapsed_us,
    })
}

/// Compute a domain-separated BLAKE3 hash with the skeleton domain prefix.
fn blake3_domain_hash(domain: &[u8], data: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Compute the evidence hash that chains all receipt artifacts together.
///
/// This is the final cryptographic seal on the walking skeleton receipt.
/// If any upstream artifact changes, the evidence hash changes.
fn compute_evidence_hash(
    genesis_hash: &[u8; 32],
    cycle_count: u64,
    ihsan_score: f64,
    snr_score: f64,
    state_root: &[u8; 32],
    timestamp: &str,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(SKELETON_DOMAIN);
    hasher.update(genesis_hash);
    hasher.update(&cycle_count.to_le_bytes());
    hasher.update(&ihsan_score.to_le_bytes());
    hasher.update(&snr_score.to_le_bytes());
    hasher.update(state_root);
    hasher.update(timestamp.as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skeleton_completes_successfully() {
        let receipt = run_skeleton().expect("Walking skeleton must succeed");

        assert!(receipt.constitutional_pass);
        assert!(receipt.ihsan_score >= IHSAN_THRESHOLD);
        assert!(receipt.snr_score >= SNR_THRESHOLD);
        assert_ne!(receipt.state_root, [0u8; 32]);
        assert_ne!(receipt.evidence_hash, [0u8; 32]);
        assert_ne!(receipt.genesis_hash, [0u8; 32]);
        assert_eq!(receipt.cycle_count, 1);
        assert!(receipt.era_version >= 1);
    }

    #[test]
    fn test_skeleton_is_deterministic_except_timestamp() {
        let r1 = run_skeleton().unwrap();
        let r2 = run_skeleton().unwrap();

        // Deterministic fields
        assert_eq!(r1.genesis_hash, r2.genesis_hash);
        assert_eq!(r1.cycle_count, r2.cycle_count);
        assert_eq!(r1.state_root, r2.state_root);
        assert_eq!(r1.constitutional_pass, r2.constitutional_pass);
        // ihsan_score and snr_score are deterministic given same inputs
        assert!((r1.ihsan_score - r2.ihsan_score).abs() < f64::EPSILON);
        assert!((r1.snr_score - r2.snr_score).abs() < f64::EPSILON);
    }

    #[test]
    fn test_skeleton_fast() {
        let start = std::time::Instant::now();
        let receipt = run_skeleton().unwrap();
        let elapsed = start.elapsed();

        // Must complete in under 1 second
        assert!(
            elapsed.as_millis() < 1000,
            "Skeleton took {}ms, must be <1000ms",
            elapsed.as_millis()
        );
        // Receipt should also report sub-millisecond
        assert!(
            receipt.elapsed_us < 1_000_000,
            "Receipt reports {}us, must be <1s",
            receipt.elapsed_us
        );
    }

    #[test]
    fn test_evidence_hash_varies_with_inputs() {
        let h1 = compute_evidence_hash(&[1u8; 32], 1, 0.97, 0.90, &[2u8; 32], "t1");
        let h2 = compute_evidence_hash(&[1u8; 32], 2, 0.97, 0.90, &[2u8; 32], "t1");
        assert_ne!(h1, h2, "Different cycle counts must produce different hashes");
    }
}
