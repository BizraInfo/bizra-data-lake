//! BIZRA ManifestArtifact — §7 Fifth Canonical Contract
//!
//! بسم الله الرحمن الرحيم
//!
//! File: crates/bizra-kernel/src/manifest_artifact.rs
//! Authority: Manifest v0.2 Canon, §7 (Canonical Contracts)
//! Build Step: 7 of 8 (§17)
//! Depends on: Step 2 (ReceiptArtifact)
//!
//! §7 Table 7-1:
//!   Contract: ManifestArtifact
//!   Plane: Proof → Face
//!   Description: Proof-of-life summary over a defined window
//!   Key Fields: manifest_id, window_start, window_end, receipt_refs, integrity_hash
//!   Lifetime: Daily / per-window
//!
//! A ManifestArtifact is NOT a receipt — it is a summary. It aggregates
//! all receipts within a time window into a single integrity-checked
//! proof that the node was alive, lawful, and productive during that window.

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{
    Blake3Hash, ReceiptKind, ReceiptPayload, ReceiptPayloadDecode,
    ByteReader, DecodeError,
};

/// The canonical proof-of-life summary per §7.
///
/// One ManifestArtifact is produced per window (default: daily).
/// It answers: "What did this node prove during this period?"
#[derive(Debug, Clone)]
pub struct ManifestArtifact {
    // ── §7 required fields ──

    /// Unique identifier. blake3(window_start || window_end || integrity_hash).
    pub manifest_id: Blake3Hash,

    /// Window start (nanoseconds, monotonic).
    pub window_start: u64,

    /// Window end (nanoseconds, monotonic).
    pub window_end: u64,

    /// References to all receipts included in this manifest.
    /// These are receipt_ids from ReceiptArtifact, not full payloads.
    pub receipt_refs: Vec<Blake3Hash>,

    /// Integrity hash over all receipt_refs in canonical order.
    /// blake3_domain("bizra-manifest-integrity-v1", sorted(receipt_refs))
    pub integrity_hash: Blake3Hash,

    // ── Operational extensions ──

    /// Total receipt count (convenience; derivable from receipt_refs.len()).
    pub receipt_count: u32,

    /// Chain head at time of manifest generation.
    pub chain_head_at_generation: Blake3Hash,
}

impl ManifestArtifact {
    /// Build a ManifestArtifact from a set of receipt references
    /// within a time window.
    pub fn from_window(
        window_start: u64,
        window_end: u64,
        mut receipt_refs: Vec<Blake3Hash>,
        chain_head: Blake3Hash,
    ) -> Self {
        // Canonical sort + dedup for deterministic integrity hash.
        // Fix C: duplicate receipt_refs are removed. A receipt appearing
        // twice in a manifest would be semantically meaningless and
        // would inflate receipt_count dishonestly.
        receipt_refs.sort();
        receipt_refs.dedup();

        // Compute integrity hash
        let mut integrity_buf = Vec::with_capacity(receipt_refs.len() * 32);
        for r in &receipt_refs {
            integrity_buf.extend_from_slice(r);
        }
        let integrity_hash = blake3_domain(
            "bizra-manifest-integrity-v1",
            &integrity_buf,
        );

        // Compute manifest_id — includes ALL fields that affect identity.
        // Fix A: chain_head_at_generation and receipt_count are now bound
        // into the identity hash, not just operational metadata.
        let mut id_buf = Vec::with_capacity(112);
        id_buf.extend_from_slice(&window_start.to_le_bytes());
        id_buf.extend_from_slice(&window_end.to_le_bytes());
        id_buf.extend_from_slice(&integrity_hash);
        id_buf.extend_from_slice(&chain_head);
        id_buf.extend_from_slice(&(receipt_refs.len() as u32).to_le_bytes());
        let manifest_id = blake3_domain("bizra-manifest-id-v1", &id_buf);

        let receipt_count = receipt_refs.len() as u32;

        ManifestArtifact {
            manifest_id,
            window_start,
            window_end,
            receipt_refs,
            integrity_hash,
            receipt_count,
            chain_head_at_generation: chain_head,
        }
    }

    /// Verify the integrity hash against the receipt_refs.
    /// Any node can call this to check the manifest hasn't been tampered with.
    pub fn verify_integrity(&self) -> bool {
        let mut buf = Vec::with_capacity(self.receipt_refs.len() * 32);
        let mut sorted = self.receipt_refs.clone();
        sorted.sort();
        for r in &sorted {
            buf.extend_from_slice(r);
        }
        let computed = blake3_domain("bizra-manifest-integrity-v1", &buf);
        computed == self.integrity_hash
    }
}

impl ReceiptPayload for ManifestArtifact {
    fn kind(&self) -> ReceiptKind {
        // Cycle-7 G1 — dedicated Manifest kind (was NodeLifecycle pre-Cycle-7).
        // A manifest IS a lifecycle-style summary, but audit clarity favors a
        // distinct receipt kind: manifests bind a mission's full chain footprint
        // into one queryable object, which is semantically narrower than the
        // NodeLifecycle catchall.
        ReceiptKind::Manifest
    }

    /// Fix B: Override timestamp_ns so manifests advance the chain's
    /// latest_timestamp() when appended. window_end is the cleanest
    /// semantics — "this manifest covers up to this point in time."
    fn timestamp_ns(&self) -> u64 {
        self.window_end
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&self.manifest_id);
        buf.extend_from_slice(&self.window_start.to_le_bytes());
        buf.extend_from_slice(&self.window_end.to_le_bytes());
        buf.extend_from_slice(&(self.receipt_refs.len() as u32).to_le_bytes());
        for r in &self.receipt_refs {
            buf.extend_from_slice(r);
        }
        buf.extend_from_slice(&self.integrity_hash);
        buf.extend_from_slice(&self.receipt_count.to_le_bytes());
        buf.extend_from_slice(&self.chain_head_at_generation);
        buf
    }

    fn hash(&self) -> Blake3Hash {
        self.manifest_id
    }
}

impl ReceiptPayloadDecode for ManifestArtifact {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);

        let manifest_id = r.read_hash()?;
        let window_start = r.read_u64()?;
        let window_end = r.read_u64()?;

        let ref_count = r.read_u32()? as usize;
        let mut receipt_refs = Vec::with_capacity(ref_count);
        for _ in 0..ref_count {
            receipt_refs.push(r.read_hash()?);
        }

        let integrity_hash = r.read_hash()?;
        let receipt_count = r.read_u32()?;
        let chain_head_at_generation = r.read_hash()?;

        Ok(ManifestArtifact {
            manifest_id, window_start, window_end,
            receipt_refs, integrity_hash, receipt_count,
            chain_head_at_generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_has_section7_fields() {
        let refs = vec![[1u8; 32], [2u8; 32], [3u8; 32]];
        let m = ManifestArtifact::from_window(
            1000, 2000, refs, [99u8; 32],
        );

        assert_ne!(m.manifest_id, [0u8; 32]);
        assert_eq!(m.window_start, 1000);
        assert_eq!(m.window_end, 2000);
        assert_eq!(m.receipt_refs.len(), 3);
        assert_ne!(m.integrity_hash, [0u8; 32]);
    }

    #[test]
    fn test_manifest_integrity_verifies() {
        let refs = vec![[10u8; 32], [20u8; 32]];
        let m = ManifestArtifact::from_window(
            100, 200, refs, [0u8; 32],
        );
        assert!(m.verify_integrity());
    }

    #[test]
    fn test_manifest_deterministic() {
        let refs = vec![[5u8; 32], [6u8; 32]];
        let m1 = ManifestArtifact::from_window(500, 600, refs.clone(), [0u8; 32]);
        let m2 = ManifestArtifact::from_window(500, 600, refs, [0u8; 32]);
        assert_eq!(m1.manifest_id, m2.manifest_id);
        assert_eq!(m1.integrity_hash, m2.integrity_hash);
    }

    #[test]
    fn test_manifest_roundtrip() {
        let refs = vec![[7u8; 32], [8u8; 32], [9u8; 32]];
        let original = ManifestArtifact::from_window(
            1000, 5000, refs, [50u8; 32],
        );
        let bytes = original.canonical_bytes();
        let decoded = ManifestArtifact::from_canonical_bytes(&bytes).unwrap();

        assert_eq!(original.manifest_id, decoded.manifest_id);
        assert_eq!(original.window_start, decoded.window_start);
        assert_eq!(original.window_end, decoded.window_end);
        assert_eq!(original.receipt_refs, decoded.receipt_refs);
        assert_eq!(original.integrity_hash, decoded.integrity_hash);
    }

    #[test]
    fn test_empty_manifest_valid() {
        let m = ManifestArtifact::from_window(0, 0, vec![], [0u8; 32]);
        assert_eq!(m.receipt_count, 0);
        assert!(m.verify_integrity());
    }
}
