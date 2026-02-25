//! # Decision Pivots — Chain of Reasoning (Paper 1)
//!
//! Externalises HHMM reasoning into **verifiable checkpoints**.
//! Each level transition in the 8-level HHMM produces a `DecisionPivot`.
//! A pivot is appended to the `ReceiptChain` only when its Iḥsān score
//! meets the gate threshold.  On failure, the caller should try an
//! alternative reasoning branch.
//!
//! Standing on Giants:
//! - Wei et al. (2022): Chain-of-Thought Prompting
//! - Besta et al. (2024): Graph of Thoughts
//! - Al-Ghazali (1095): Iḥsān as incremental excellence
//!
//! ## CPVA impact
//! Early-exit on a failed pivot wastes zero downstream compute.
//! Estimated −15% on cache-miss path.

use serde::{Deserialize, Serialize};

/// Iḥsān threshold below which a pivot is considered failed.
/// Source of truth: `core/integration/constants.py → UNIFIED_IHSAN_THRESHOLD`.
/// Config key: `ihsan_threshold` in `config/proactive_config.yaml`.
pub const PIVOT_IHSAN_DEFAULT: f64 = 0.95;

/// HHMM level that produced this pivot (L0–L7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HhmmLevel {
    L0Runtime,
    L1Reflex,
    L2Cognitive,
    L3Memory,
    L4Reconciliation,
    L5Economic,
    L6Human,
    L7Federation,
}

impl HhmmLevel {
    /// L0/L1 never need reasoning pivots — compiled reflex executes directly.
    #[inline]
    pub fn needs_pivot(self) -> bool {
        !matches!(self, HhmmLevel::L0Runtime | HhmmLevel::L1Reflex)
    }
}

/// A single verifiable checkpoint in a reasoning chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPivot {
    /// Sequential index within this reasoning chain (0-based).
    pub index: usize,
    /// HHMM level that generated this pivot.
    pub level: HhmmLevel,
    /// Human-readable rationale for this transition.
    pub rationale: String,
    /// Iḥsān score for this pivot (0–1).
    pub ihsan: f64,
    /// BLAKE3 hash of (predecessor_hash ∥ rationale) — tamper-evident chain.
    pub hash: [u8; 32],
}

impl DecisionPivot {
    /// Construct a new pivot.  `predecessor_hash` is the hash of the prior
    /// pivot (or all-zeros for the first pivot in a chain).
    pub fn new(
        index: usize,
        level: HhmmLevel,
        rationale: impl Into<String>,
        ihsan: f64,
        predecessor_hash: [u8; 32],
    ) -> Self {
        let rationale = rationale.into();
        let hash = Self::compute_hash(predecessor_hash, &rationale);
        Self {
            index,
            level,
            rationale,
            ihsan,
            hash,
        }
    }

    /// Returns `true` when this pivot passes the Iḥsān gate.
    #[inline]
    pub fn passes(&self, threshold: f64) -> bool {
        self.ihsan >= threshold
    }

    fn compute_hash(predecessor: [u8; 32], rationale: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&predecessor);
        hasher.update(rationale.as_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// An ordered chain of decision pivots produced by one reasoning pass.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ReasoningChain {
    pivots: Vec<DecisionPivot>,
}

impl ReasoningChain {
    pub fn new() -> Self {
        Self { pivots: Vec::new() }
    }

    /// Append a new pivot.  The hash chain is maintained automatically.
    pub fn push(
        &mut self,
        level: HhmmLevel,
        rationale: impl Into<String>,
        ihsan: f64,
    ) -> &DecisionPivot {
        let predecessor = self.tail_hash();
        let index = self.pivots.len();
        let pivot = DecisionPivot::new(index, level, rationale, ihsan, predecessor);
        self.pivots.push(pivot);
        self.pivots.last().unwrap()
    }

    /// Iterate pivots in insertion order.
    pub fn decision_pivots(&self) -> impl Iterator<Item = &DecisionPivot> {
        self.pivots.iter()
    }

    /// Hash of the last pivot, or all-zeros if chain is empty.
    pub fn tail_hash(&self) -> [u8; 32] {
        self.pivots.last().map(|p| p.hash).unwrap_or([0u8; 32])
    }

    /// `true` if every pivot in the chain passes the given Iḥsān threshold.
    pub fn all_pass(&self, threshold: f64) -> bool {
        self.pivots.iter().all(|p| p.passes(threshold))
    }

    pub fn len(&self) -> usize {
        self.pivots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pivots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pivot_chain_hash_links() {
        let mut chain = ReasoningChain::new();
        let p0_hash = {
            let p = chain.push(HhmmLevel::L2Cognitive, "Initial hypothesis", 0.97);
            p.hash
        };
        let p1 = chain.push(HhmmLevel::L3Memory, "Memory retrieval confirms", 0.96);
        // p1.hash must incorporate p0_hash
        let expected = DecisionPivot::compute_hash(p0_hash, "Memory retrieval confirms");
        assert_eq!(p1.hash, expected);
    }

    #[test]
    fn test_l0_l1_no_pivot_needed() {
        assert!(!HhmmLevel::L0Runtime.needs_pivot());
        assert!(!HhmmLevel::L1Reflex.needs_pivot());
        assert!(HhmmLevel::L2Cognitive.needs_pivot());
    }

    #[test]
    fn test_all_pass_below_threshold_fails() {
        let mut chain = ReasoningChain::new();
        chain.push(HhmmLevel::L2Cognitive, "Good step", 0.97);
        chain.push(HhmmLevel::L3Memory, "Weak step", 0.82); // below 0.95
        assert!(!chain.all_pass(PIVOT_IHSAN_DEFAULT));
    }
}
