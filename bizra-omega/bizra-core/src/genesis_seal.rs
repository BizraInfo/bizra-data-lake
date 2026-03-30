//! GenesisSeal v1 — Deterministic Root of Trust
//!
//! The genesis seal is the cryptographic anchor that binds every
//! CanonicalReceipt to a specific constitutional configuration.
//! Two nodes with the same genesis seal share the same constitution.
//! Two nodes with different seals are in different constitutional universes.
//!
//! The seal is computed deterministically from frozen parameters:
//!   - Ihsan threshold
//!   - SNR threshold
//!   - Gini ceiling (Adl invariant)
//!   - Gate chain order
//!   - PAT/SAT topology
//!   - Constitutional version string
//!
//! If ANY parameter changes, the seal changes, and all receipts
//! bound to the old seal become a different chain.
//!
//! Standing on Giants:
//!   - Nakamoto (2008): genesis block as root of trust
//!   - Tezos: self-amending ledger with constitutional governance
//!   - Al-Ghazali: Maqasid al-Shariah as immutable ethical anchors

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::topology_canon::TopologyCanon;

/// Domain prefix for genesis seal hashing.
pub const DOMAIN_GENESIS_SEAL: &str = "bizra-genesis-seal-v1";

/// Frozen constitutional parameters that define a genesis seal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalParams {
    /// Ihsan excellence threshold (e.g., 0.95).
    pub ihsan_threshold: f64,
    /// SNR signal quality threshold (e.g., 0.85).
    pub snr_threshold: f64,
    /// Adl fairness ceiling — Gini coefficient (e.g., 0.35).
    pub gini_ceiling: f64,
    /// Gate chain order (e.g., ["Schema", "Ihsan", "SNR"]).
    pub gate_order: Vec<String>,
    /// PAT agent count.
    pub pat_count: u8,
    /// SAT agent count.
    pub sat_count: u8,
    /// Constitutional version string (e.g., "ihsan_v1").
    pub constitution_id: String,
    /// Verdict precedence order.
    pub verdict_precedence: Vec<String>,
}

impl Default for ConstitutionalParams {
    fn default() -> Self {
        Self {
            ihsan_threshold: 0.95,
            snr_threshold: 0.85,
            gini_ceiling: 0.35,
            gate_order: TopologyCanon::GATE_ORDER.iter().map(|s| s.to_string()).collect(),
            pat_count: TopologyCanon::PAT_COUNT as u8,
            sat_count: TopologyCanon::SAT_COUNT as u8,
            constitution_id: "ihsan_v1".to_string(),
            verdict_precedence: TopologyCanon::VERDICT_PRECEDENCE.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// The GenesisSeal — cryptographic root of trust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisSeal {
    /// The BLAKE3 hash of the frozen constitutional parameters.
    pub seal_hash: [u8; 32],
    /// Human-readable hex of the seal (first 16 chars).
    pub seal_id: String,
    /// The frozen parameters that produced this seal.
    pub params: ConstitutionalParams,
    /// Timestamp when the seal was computed (Unix ms).
    pub sealed_at: u64,
}

impl GenesisSeal {
    /// Compute a genesis seal from constitutional parameters.
    pub fn compute(params: ConstitutionalParams, now_ms: u64) -> Self {
        let seal_hash = Self::hash_params(&params);
        let hex: String = seal_hash.iter().map(|b| format!("{b:02x}")).collect();
        Self {
            seal_hash,
            seal_id: hex[..16].to_string(),
            params,
            sealed_at: now_ms,
        }
    }

    /// Compute the default genesis seal (Node0 canonical configuration).
    pub fn node0_default(now_ms: u64) -> Self {
        Self::compute(ConstitutionalParams::default(), now_ms)
    }

    /// Deterministic BLAKE3 hash of constitutional parameters.
    fn hash_params(p: &ConstitutionalParams) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(DOMAIN_GENESIS_SEAL.as_bytes());
        hasher.update(b":");

        // Fixed-point encoding for reproducibility
        hasher.update(&((p.ihsan_threshold * 1_000_000.0).round() as u64).to_le_bytes());
        hasher.update(&((p.snr_threshold * 1_000_000.0).round() as u64).to_le_bytes());
        hasher.update(&((p.gini_ceiling * 1_000_000.0).round() as u64).to_le_bytes());

        // Gate order
        hasher.update(&(p.gate_order.len() as u32).to_le_bytes());
        for gate in &p.gate_order {
            hasher.update(&(gate.len() as u32).to_le_bytes());
            hasher.update(gate.as_bytes());
        }

        // Topology
        hasher.update(&[p.pat_count, p.sat_count]);

        // Constitution ID
        hasher.update(&(p.constitution_id.len() as u32).to_le_bytes());
        hasher.update(p.constitution_id.as_bytes());

        // Verdict precedence
        hasher.update(&(p.verdict_precedence.len() as u32).to_le_bytes());
        for code in &p.verdict_precedence {
            hasher.update(&(code.len() as u32).to_le_bytes());
            hasher.update(code.as_bytes());
        }

        *hasher.finalize().as_bytes()
    }

    /// Verify that a seal hash matches the parameters.
    pub fn verify(&self) -> bool {
        self.seal_hash == Self::hash_params(&self.params)
    }

    /// Check if a receipt's genesis hash matches this seal.
    pub fn binds_receipt(&self, receipt_genesis_hash: &[u8; 32]) -> bool {
        self.seal_hash == *receipt_genesis_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_seal_deterministic() {
        let s1 = GenesisSeal::node0_default(1000);
        let s2 = GenesisSeal::node0_default(2000);
        // Same params → same hash (timestamp is NOT part of the hash)
        assert_eq!(s1.seal_hash, s2.seal_hash);
        assert_eq!(s1.seal_id, s2.seal_id);
    }

    #[test]
    fn test_genesis_seal_changes_on_param_change() {
        let s1 = GenesisSeal::node0_default(1000);
        let mut params = ConstitutionalParams::default();
        params.ihsan_threshold = 0.90; // Lower threshold = different constitution
        let s2 = GenesisSeal::compute(params, 1000);
        assert_ne!(s1.seal_hash, s2.seal_hash);
    }

    #[test]
    fn test_genesis_seal_verify() {
        let seal = GenesisSeal::node0_default(1000);
        assert!(seal.verify());
    }

    #[test]
    fn test_genesis_seal_tamper_detection() {
        let mut seal = GenesisSeal::node0_default(1000);
        seal.params.gini_ceiling = 0.99; // Tamper with Gini
        assert!(!seal.verify()); // Seal no longer matches params
    }

    #[test]
    fn test_genesis_seal_binds_receipt() {
        let seal = GenesisSeal::node0_default(1000);
        assert!(seal.binds_receipt(&seal.seal_hash));
        assert!(!seal.binds_receipt(&[0xFF; 32]));
    }

    #[test]
    fn test_genesis_seal_topology_sensitivity() {
        let s1 = GenesisSeal::node0_default(1000);
        let mut params = ConstitutionalParams::default();
        params.pat_count = 6; // Wrong PAT count
        let s2 = GenesisSeal::compute(params, 1000);
        assert_ne!(s1.seal_hash, s2.seal_hash);
    }

    #[test]
    fn test_genesis_seal_gate_order_sensitivity() {
        let s1 = GenesisSeal::node0_default(1000);
        let mut params = ConstitutionalParams::default();
        params.gate_order = vec!["Schema".into(), "SNR".into(), "Ihsan".into()]; // Wrong order
        let s2 = GenesisSeal::compute(params, 1000);
        assert_ne!(s1.seal_hash, s2.seal_hash);
    }

    #[test]
    fn test_default_params_match_topology_canon() {
        let p = ConstitutionalParams::default();
        assert_eq!(p.pat_count as usize, TopologyCanon::PAT_COUNT);
        assert_eq!(p.sat_count as usize, TopologyCanon::SAT_COUNT);
        assert_eq!(p.ihsan_threshold, 0.95);
        assert_eq!(p.gini_ceiling, 0.35);
    }
}
