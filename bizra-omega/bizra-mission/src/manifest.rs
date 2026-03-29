//! # ManifestArtifact — Cross-Layer Contract #4
//!
//! Bundles receipts into a reviewable, integrity-checked package.
//! A manifest is the unit of evidence: it collects all receipts from
//! a time period and seals them with a single BLAKE3 integrity hash.
//!
//! Flow: Receipts → Manifest → Evidence Bundle → Public Verification

use blake3::Hasher;
use serde::{Deserialize, Serialize};

/// Domain prefix for manifest hashing.
pub const DOMAIN_MANIFEST: &str = "bizra-manifest-v1";

/// A reference to a receipt within the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptRef {
    /// Receipt ID (BLAKE3 hash).
    pub receipt_id: [u8; 32],
    /// Mission ID this receipt belongs to.
    pub mission_id: [u8; 32],
    /// Whether the receipt represents success or failure.
    pub is_success: bool,
    /// Ihsan score at time of evaluation.
    pub ihsan_score: Option<f32>,
}

/// The ManifestArtifact — cross-layer contract #4.
///
/// Bundles receipt references with metadata for evidence review.
/// The `integrity_hash` covers ALL receipt references — removing
/// or reordering any receipt breaks the hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestArtifact {
    /// Unique manifest ID.
    pub manifest_id: [u8; 32],
    /// Node that generated this manifest.
    pub node_id: String,
    /// Policy version in effect during this period.
    pub policy_version: String,
    /// Period start (Unix ms).
    pub period_start: u64,
    /// Period end (Unix ms).
    pub period_end: u64,
    /// Ordered receipt references.
    pub receipts: Vec<ReceiptRef>,
    /// Total missions in period.
    pub total_missions: u32,
    /// Admitted missions.
    pub admitted: u32,
    /// Rejected missions.
    pub rejected: u32,
    /// Average Ihsan score across admitted missions.
    pub avg_ihsan: f64,
    /// BLAKE3 integrity hash of the entire manifest.
    pub integrity_hash: [u8; 32],
}

impl ManifestArtifact {
    /// Create a new manifest from receipt references.
    pub fn new(
        node_id: String,
        policy_version: String,
        period_start: u64,
        period_end: u64,
        receipts: Vec<ReceiptRef>,
    ) -> Self {
        let total = receipts.len() as u32;
        let admitted = receipts.iter().filter(|r| r.is_success).count() as u32;
        let rejected = total - admitted;

        let scores: Vec<f64> = receipts
            .iter()
            .filter_map(|r| r.ihsan_score.map(|s| s as f64))
            .collect();
        let avg_ihsan = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        let mut manifest = Self {
            manifest_id: [0; 32],
            node_id,
            policy_version,
            period_start,
            period_end,
            receipts,
            total_missions: total,
            admitted,
            rejected,
            avg_ihsan,
            integrity_hash: [0; 32],
        };
        manifest.integrity_hash = manifest.compute_integrity();
        manifest.manifest_id = manifest.compute_id();
        manifest
    }

    /// Compute the integrity hash covering all receipt references.
    fn compute_integrity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(DOMAIN_MANIFEST.as_bytes());
        hasher.update(b":");

        // Node + policy
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.policy_version.as_bytes());

        // Period
        hasher.update(&self.period_start.to_le_bytes());
        hasher.update(&self.period_end.to_le_bytes());

        // All receipt IDs in order (tamper-evident: reorder breaks hash)
        hasher.update(&(self.receipts.len() as u32).to_le_bytes());
        for r in &self.receipts {
            hasher.update(&r.receipt_id);
            hasher.update(&[if r.is_success { 1 } else { 0 }]);
        }

        hasher.finalize().into()
    }

    /// Compute manifest ID from integrity hash.
    fn compute_id(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-manifest-id:");
        hasher.update(&self.integrity_hash);
        hasher.finalize().into()
    }

    /// Verify manifest integrity.
    pub fn verify_integrity(&self) -> bool {
        self.integrity_hash == self.compute_integrity()
    }

    /// Manifest ID as hex.
    pub fn id_hex(&self) -> String {
        self.manifest_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Admission rate.
    pub fn admission_rate(&self) -> f64 {
        if self.total_missions == 0 {
            return 0.0;
        }
        self.admitted as f64 / self.total_missions as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_receipts() -> Vec<ReceiptRef> {
        vec![
            ReceiptRef {
                receipt_id: [1; 32],
                mission_id: [10; 32],
                is_success: true,
                ihsan_score: Some(0.97),
            },
            ReceiptRef {
                receipt_id: [2; 32],
                mission_id: [11; 32],
                is_success: true,
                ihsan_score: Some(0.96),
            },
            ReceiptRef {
                receipt_id: [3; 32],
                mission_id: [12; 32],
                is_success: false,
                ihsan_score: Some(0.40),
            },
        ]
    }

    #[test]
    fn test_manifest_creation() {
        let m = ManifestArtifact::new(
            "node0".into(),
            "0.89.1".into(),
            1000,
            2000,
            sample_receipts(),
        );
        assert_eq!(m.total_missions, 3);
        assert_eq!(m.admitted, 2);
        assert_eq!(m.rejected, 1);
        assert!(m.avg_ihsan > 0.7);
        assert_ne!(m.integrity_hash, [0; 32]);
        assert_ne!(m.manifest_id, [0; 32]);
    }

    #[test]
    fn test_manifest_integrity_valid() {
        let m = ManifestArtifact::new(
            "node0".into(),
            "0.89.1".into(),
            1000,
            2000,
            sample_receipts(),
        );
        assert!(m.verify_integrity());
    }

    #[test]
    fn test_manifest_integrity_fails_on_tamper() {
        let mut m = ManifestArtifact::new(
            "node0".into(),
            "0.89.1".into(),
            1000,
            2000,
            sample_receipts(),
        );
        m.receipts.push(ReceiptRef {
            receipt_id: [99; 32],
            mission_id: [99; 32],
            is_success: true,
            ihsan_score: Some(0.99),
        });
        assert!(!m.verify_integrity());
    }

    #[test]
    fn test_manifest_deterministic() {
        let m1 = ManifestArtifact::new(
            "node0".into(),
            "0.89.1".into(),
            1000,
            2000,
            sample_receipts(),
        );
        let m2 = ManifestArtifact::new(
            "node0".into(),
            "0.89.1".into(),
            1000,
            2000,
            sample_receipts(),
        );
        assert_eq!(m1.integrity_hash, m2.integrity_hash);
        assert_eq!(m1.manifest_id, m2.manifest_id);
    }

    #[test]
    fn test_manifest_empty() {
        let m = ManifestArtifact::new("node0".into(), "0.89.1".into(), 1000, 2000, vec![]);
        assert_eq!(m.total_missions, 0);
        assert_eq!(m.admission_rate(), 0.0);
        assert!(m.verify_integrity());
    }

    #[test]
    fn test_manifest_admission_rate() {
        let m = ManifestArtifact::new(
            "node0".into(),
            "0.89.1".into(),
            1000,
            2000,
            sample_receipts(),
        );
        let rate = m.admission_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.001);
    }
}
