//! BIZRA Receipt v1 Freeze — §7 Contract Alignment + Disk Store + Replay Test
//!
//! بسم الله الرحمن الرحيم
//!
//! File: crates/bizra-kernel/src/receipts/freeze_v1.rs
//! Authority: Manifest v0.2 Canon, §7 (Canonical Contracts), §10 (Proof Law)
//! Build Step: 2 of 8 (§17)
//! Truth Target: PROVEN
//!
//! This file delivers three things:
//!
//!   1. ReceiptArtifact — the frozen §7 contract struct. Once this compiles
//!      and tests pass, its field set is frozen per §14. Implementation
//!      details (serialization, storage) may evolve per §15.
//!
//!   2. SledPayloadStore — disk-backed PayloadStore with fsync. Replaces
//!      InMemoryPayloadStore for production. Chain-is-truth (§10) requires
//!      durable persistence.
//!
//!   3. Replay verification test — proves §16 success condition #4:
//!      "one replay reproduction: deterministic replay matches original."
//!
//! Integration: this file extends, does not replace, receipts.rs. The existing
//! Receipt struct becomes the internal chain record; ReceiptArtifact is the
//! frozen cross-plane contract that wraps it with §7-required fields.
//!
//! Relationship to existing receipts.rs:
//!   Receipt (thin chain record)     = Layer 1 (internal, mutable impl)
//!   ReceiptArtifact (§7 contract)   = Layer 1.5 (frozen interface)
//!   ReceiptEnvelope<T> (payload)    = Layer 2 (internal, mutable impl)

use crate::canonical_hasher::blake3_domain;
#[cfg(test)]
use crate::receipts::InMemoryPayloadStore;
use crate::receipts::{
    Blake3Hash, ByteReader, DecodeError, Receipt, ReceiptChain, ReceiptKind, ReceiptPayload,
    ReceiptPayloadDecode,
};
#[cfg(feature = "sled-store")]
use crate::receipts::{PayloadStore, StoreError};

// ════════════════════════════════════════════════════════════
// ReceiptArtifact — The §7 Frozen Contract
// ════════════════════════════════════════════════════════════

/// The canonical cross-plane receipt contract per Manifest v0.2 §7.
///
/// FROZEN after Step 2 completes. Field set cannot change without
/// constitutional amendment (§14). Serialization format may evolve (§15)
/// but must remain decode-compatible with all previously frozen receipts.
///
/// §7 Table 7-1 specifies:
///   "receipt_id, claim_ref, evidence_hash, lineage, blake3_chain"
///   Plane: Proof
///   Description: Immutable execution/evaluation proof
///   Lifetime: Permanent
///
/// Additional fields (kind, timestamp_ns, prev) are required for chain
/// mechanics and are included as operational extensions that serve the
/// contract without altering its §7 identity.
#[derive(Debug, Clone)]
pub struct ReceiptArtifact {
    // ── §7 required fields ──
    /// Unique identifier for this receipt. Computed as blake3 of canonical_bytes.
    pub receipt_id: Blake3Hash,

    /// Reference to the claim this receipt proves.
    /// For a MissionEnvelope-originated claim: the mission_id.
    /// For a boot receipt: the boot config digest.
    /// For a myelination receipt: the source_s2 edge hash.
    pub claim_ref: Blake3Hash,

    /// Hash of the evidence that supports this receipt.
    /// For execution receipts: hash of the execution trace.
    /// For valuation receipts: hash of the evidence chain.
    /// For boot receipts: hash of the cognition boot payload.
    pub evidence_hash: Blake3Hash,

    /// Lineage: ordered list of prior receipt_ids that this receipt
    /// depends on. Creates a DAG of provenance, not just a chain.
    /// For simple receipts: [prev_receipt_id].
    /// For compound receipts: [all contributing receipt_ids].
    pub lineage: Vec<Blake3Hash>,

    /// The BLAKE3 chain hash linking this receipt to the chain.
    /// Equal to blake3_domain("bizra-receipt-chain-v1", receipt_id || prev_chain_head).
    pub blake3_chain: Blake3Hash,

    // ── Operational extensions (serve §7 contract without altering identity) ──
    /// Receipt kind discriminant.
    pub kind: ReceiptKind,

    /// Nanosecond timestamp (monotonic, not wall-clock).
    pub timestamp_ns: u64,

    /// Previous chain head at time of append.
    pub prev: Blake3Hash,
}

impl ReceiptArtifact {
    /// Build a ReceiptArtifact from its components.
    ///
    /// The receipt_id is computed, not supplied — it is the hash of
    /// the canonical bytes of the content fields. This ensures that
    /// the receipt_id is deterministic and reproducible.
    pub fn new(
        kind: ReceiptKind,
        claim_ref: Blake3Hash,
        evidence_hash: Blake3Hash,
        lineage: Vec<Blake3Hash>,
        prev: Blake3Hash,
        timestamp_ns: u64,
    ) -> Self {
        // Compute receipt_id from content
        let mut content_buf = Vec::new();
        content_buf.push(kind as u8);
        content_buf.extend_from_slice(&claim_ref);
        content_buf.extend_from_slice(&evidence_hash);
        content_buf.extend_from_slice(&(lineage.len() as u32).to_le_bytes());
        for l in &lineage {
            content_buf.extend_from_slice(l);
        }
        content_buf.extend_from_slice(&timestamp_ns.to_le_bytes());
        let receipt_id = blake3_domain("bizra-receipt-id-v1", &content_buf);

        // Compute chain hash
        let mut chain_buf = Vec::new();
        chain_buf.extend_from_slice(&receipt_id);
        chain_buf.extend_from_slice(&prev);
        let blake3_chain = blake3_domain("bizra-receipt-chain-v1", &chain_buf);

        ReceiptArtifact {
            receipt_id,
            claim_ref,
            evidence_hash,
            lineage,
            blake3_chain,
            kind,
            timestamp_ns,
            prev,
        }
    }

    /// Convert to the thin internal Receipt for chain storage.
    /// The ReceiptArtifact is the full contract; the Receipt is the
    /// chain-internal record that ReceiptChain manages.
    pub fn to_chain_receipt(&self) -> Receipt {
        Receipt {
            kind: self.kind,
            hash: self.receipt_id,
            prev: self.prev,
        }
    }
}

impl ReceiptPayload for ReceiptArtifact {
    fn kind(&self) -> ReceiptKind {
        self.kind
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        // receipt_id (32)
        buf.extend_from_slice(&self.receipt_id);
        // claim_ref (32)
        buf.extend_from_slice(&self.claim_ref);
        // evidence_hash (32)
        buf.extend_from_slice(&self.evidence_hash);
        // lineage (4 + N*32)
        buf.extend_from_slice(&(self.lineage.len() as u32).to_le_bytes());
        for l in &self.lineage {
            buf.extend_from_slice(l);
        }
        // blake3_chain (32)
        buf.extend_from_slice(&self.blake3_chain);
        // kind (1)
        buf.push(self.kind as u8);
        // timestamp_ns (8)
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        // prev (32)
        buf.extend_from_slice(&self.prev);
        buf
    }

    fn hash(&self) -> Blake3Hash {
        // The hash of a ReceiptArtifact IS its receipt_id.
        // This is computed in new() and is deterministic.
        self.receipt_id
    }
}

impl ReceiptPayloadDecode for ReceiptArtifact {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);

        let receipt_id = r.read_hash()?;
        let claim_ref = r.read_hash()?;
        let evidence_hash = r.read_hash()?;

        let lineage_len = r.read_u32()? as usize;
        let mut lineage = Vec::with_capacity(lineage_len);
        for _ in 0..lineage_len {
            lineage.push(r.read_hash()?);
        }

        let blake3_chain = r.read_hash()?;

        let kind_byte = r.read_u8()?;
        let kind = ReceiptKind::from_byte(kind_byte).ok_or(DecodeError::UnknownDiscriminant {
            field: "ReceiptArtifact.kind",
            byte: kind_byte,
        })?;

        let timestamp_ns = r.read_u64()?;
        let prev = r.read_hash()?;

        Ok(ReceiptArtifact {
            receipt_id,
            claim_ref,
            evidence_hash,
            lineage,
            blake3_chain,
            kind,
            timestamp_ns,
            prev,
        })
    }
}

// ════════════════════════════════════════════════════════════
// SledPayloadStore — Disk-Backed Durable Storage
// ════════════════════════════════════════════════════════════

/// Production-grade PayloadStore backed by sled embedded database.
///
/// Provides:
///   - fsync on every put (configurable flush mode)
///   - crash-safe: sled uses a log-structured merge tree
///   - content-addressed: key = Blake3Hash, value = canonical bytes
///
/// §10 (Proof Law): "The chain is source truth." An in-memory store
/// cannot be source truth across restarts. This store can.
///
/// Dependency: `sled = "0.34"` in Cargo.toml
///
/// NOTE: This is a reference implementation. The PayloadStore trait
/// allows drop-in replacement with RocksDB, SQLite, or a custom
/// append-only log if sled proves insufficient.
#[cfg(feature = "sled-store")]
pub struct SledPayloadStore {
    db: sled::Db,
}

#[cfg(feature = "sled-store")]
impl SledPayloadStore {
    /// Open or create a sled database at the given path.
    ///
    /// Recommended path: /data/bizra/receipts/payload_store
    pub fn open(path: &str) -> Result<Self, StoreError> {
        let db = sled::open(path).map_err(|e| StoreError::IoError(format!("sled open: {}", e)))?;
        Ok(SledPayloadStore { db })
    }

    /// Flush all pending writes to disk. Called after critical receipts.
    pub fn flush(&self) -> Result<(), StoreError> {
        self.db
            .flush()
            .map_err(|e| StoreError::IoError(format!("sled flush: {}", e)))?;
        Ok(())
    }
}

#[cfg(feature = "sled-store")]
impl PayloadStore for SledPayloadStore {
    fn put(&self, hash: Blake3Hash, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.db
            .insert(hash, bytes.as_slice())
            .map_err(|e| StoreError::IoError(format!("sled put: {}", e)))?;
        // fsync after every put for durability
        self.db
            .flush()
            .map_err(|e| StoreError::IoError(format!("sled flush: {}", e)))?;
        Ok(())
    }

    fn get(&self, hash: &Blake3Hash) -> Result<Option<Vec<u8>>, StoreError> {
        match self.db.get(hash) {
            Ok(Some(bytes)) => Ok(Some(bytes.to_vec())),
            Ok(None) => Ok(None),
            Err(e) => Err(StoreError::IoError(format!("sled get: {}", e))),
        }
    }

    fn contains(&self, hash: &Blake3Hash) -> Result<bool, StoreError> {
        self.db
            .contains_key(hash)
            .map_err(|e| StoreError::IoError(format!("sled contains: {}", e)))
    }
}

// ════════════════════════════════════════════════════════════
// Append helper — ReceiptArtifact through ReceiptChain
// ════════════════════════════════════════════════════════════

/// Extension trait for appending ReceiptArtifacts through the chain.
///
/// This bridges the §7 contract (ReceiptArtifact) with the internal
/// chain mechanics (ReceiptChain::append_with_payload). The chain
/// stores the full ReceiptArtifact as the payload, and the thin
/// Receipt as the chain record.
pub trait ReceiptChainExt {
    /// Append a ReceiptArtifact to the chain.
    /// Persists payload FIRST, then advances chain head.
    /// Returns the receipt_id on success.
    fn append_artifact(
        &mut self,
        artifact: ReceiptArtifact,
    ) -> Result<Blake3Hash, crate::receipts::ChainError>;
}

impl ReceiptChainExt for ReceiptChain {
    fn append_artifact(
        &mut self,
        artifact: ReceiptArtifact,
    ) -> Result<Blake3Hash, crate::receipts::ChainError> {
        // This delegates to the existing append_with_payload,
        // which enforces the persist-before-chain invariant.
        self.append_with_payload(artifact)
    }
}

// ════════════════════════════════════════════════════════════
// Tests — proving Step 2 PROVEN status
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn genesis_hash() -> Blake3Hash {
        // The actual Genesis Block hash prefix, zero-padded to 32 bytes
        let mut h = [0u8; 32];
        let prefix = b"350d642099bde68b";
        h[..prefix.len()].copy_from_slice(prefix);
        h
    }

    fn make_test_chain() -> ReceiptChain {
        let store = Box::new(InMemoryPayloadStore::new());
        ReceiptChain::new(genesis_hash(), store)
    }

    fn make_artifact(
        kind: ReceiptKind,
        claim_ref: Blake3Hash,
        evidence: Blake3Hash,
        prev: Blake3Hash,
        ts: u64,
    ) -> ReceiptArtifact {
        ReceiptArtifact::new(
            kind,
            claim_ref,
            evidence,
            vec![prev], // lineage = [prev]
            prev,
            ts,
        )
    }

    // ── Test 1: ReceiptArtifact has all §7 fields ──

    #[test]
    fn test_receipt_artifact_has_section7_fields() {
        let claim = [1u8; 32];
        let evidence = [2u8; 32];
        let prev = genesis_hash();

        let art = ReceiptArtifact::new(
            ReceiptKind::ReasoningSession,
            claim,
            evidence,
            vec![prev],
            prev,
            1000,
        );

        // §7 required: receipt_id, claim_ref, evidence_hash, lineage, blake3_chain
        assert_ne!(art.receipt_id, [0u8; 32], "receipt_id must be computed");
        assert_eq!(art.claim_ref, claim);
        assert_eq!(art.evidence_hash, evidence);
        assert_eq!(art.lineage.len(), 1);
        assert_eq!(art.lineage[0], prev);
        assert_ne!(art.blake3_chain, [0u8; 32], "blake3_chain must be computed");
    }

    // ── Test 2: Deterministic receipt_id ──

    #[test]
    fn test_receipt_id_is_deterministic() {
        let claim = [3u8; 32];
        let evidence = [4u8; 32];
        let prev = genesis_hash();

        let a1 = ReceiptArtifact::new(
            ReceiptKind::CognitionBoot,
            claim,
            evidence,
            vec![prev],
            prev,
            500,
        );
        let a2 = ReceiptArtifact::new(
            ReceiptKind::CognitionBoot,
            claim,
            evidence,
            vec![prev],
            prev,
            500,
        );

        assert_eq!(
            a1.receipt_id, a2.receipt_id,
            "Same inputs must produce same receipt_id"
        );
        assert_eq!(
            a1.blake3_chain, a2.blake3_chain,
            "Same inputs must produce same blake3_chain"
        );
    }

    // ── Test 3: Different inputs produce different receipt_ids ──

    #[test]
    fn test_different_inputs_different_ids() {
        let prev = genesis_hash();

        let a1 = ReceiptArtifact::new(
            ReceiptKind::Myelination,
            [5u8; 32],
            [6u8; 32],
            vec![prev],
            prev,
            100,
        );
        let a2 = ReceiptArtifact::new(
            ReceiptKind::Myelination,
            [7u8; 32],
            [6u8; 32],
            vec![prev],
            prev,
            100,
        );

        assert_ne!(
            a1.receipt_id, a2.receipt_id,
            "Different claim_ref must produce different receipt_id"
        );
    }

    // ── Test 4: Canonical bytes round-trip ──

    #[test]
    fn test_canonical_bytes_roundtrip() {
        let prev = genesis_hash();
        let original = ReceiptArtifact::new(
            ReceiptKind::ReasoningSession,
            [10u8; 32],
            [11u8; 32],
            vec![prev, [12u8; 32]], // 2-element lineage
            prev,
            999_999,
        );

        let bytes = original.canonical_bytes();
        let decoded = ReceiptArtifact::from_canonical_bytes(&bytes).unwrap();

        assert_eq!(original.receipt_id, decoded.receipt_id);
        assert_eq!(original.claim_ref, decoded.claim_ref);
        assert_eq!(original.evidence_hash, decoded.evidence_hash);
        assert_eq!(original.lineage, decoded.lineage);
        assert_eq!(original.blake3_chain, decoded.blake3_chain);
        assert_eq!(original.kind, decoded.kind);
        assert_eq!(original.timestamp_ns, decoded.timestamp_ns);
        assert_eq!(original.prev, decoded.prev);
    }

    // ── Test 5: Chain append maintains continuity ──

    #[test]
    fn test_chain_append_continuity() {
        let mut chain = make_test_chain();
        let genesis = genesis_hash();

        // Append 10 artifacts
        let mut prev = genesis;
        for i in 0..10u64 {
            let art = make_artifact(
                ReceiptKind::ReasoningSession,
                [(i + 1) as u8; 32],
                [(i + 10) as u8; 32],
                prev,
                i * 1000,
            );
            let id = chain.append_artifact(art).unwrap();
            prev = id;
        }

        assert_eq!(chain.len(), 10);
        assert!(
            chain.verify_continuity(genesis).is_ok(),
            "Chain continuity must hold after 10 appends"
        );
    }

    // ── Test 6: REPLAY VERIFICATION (§16 success condition #4) ──
    //
    // This is the Step 2 proof: build a chain of 50 heterogeneous
    // receipts, serialize all payloads, decode them back, verify
    // every hash matches. This proves deterministic replay.

    #[test]
    fn test_replay_verification_50_receipts() {
        let mut chain = make_test_chain();
        let genesis = genesis_hash();

        let kinds = [
            ReceiptKind::CognitionBoot,
            ReceiptKind::Myelination,
            ReceiptKind::Demyelination,
            ReceiptKind::ReasoningSession,
            ReceiptKind::GovernanceDecision,
        ];

        let mut prev = genesis;
        let mut receipt_ids: Vec<Blake3Hash> = Vec::new();

        // Build 50 heterogeneous receipts
        for i in 0..50u64 {
            let kind = kinds[(i % kinds.len() as u64) as usize];
            let mut claim = [0u8; 32];
            claim[0..8].copy_from_slice(&i.to_le_bytes());
            let mut evidence = [0u8; 32];
            evidence[0..8].copy_from_slice(&(i * 7).to_le_bytes());

            let art = ReceiptArtifact::new(
                kind,
                claim,
                evidence,
                vec![prev],
                prev,
                i * 1_000_000, // monotonic timestamps
            );

            let id = chain.append_artifact(art).unwrap();
            receipt_ids.push(id);
            prev = id;
        }

        assert_eq!(chain.len(), 50);
        assert!(
            chain.verify_continuity(genesis).is_ok(),
            "Chain continuity must hold for 50 receipts"
        );

        // REPLAY: fetch each payload, decode, verify hash match
        for (idx, id) in receipt_ids.iter().enumerate() {
            let bytes = chain
                .fetch_payload_bytes(id)
                .unwrap()
                .unwrap_or_else(|| panic!("Payload missing for receipt {}", idx));

            let decoded = ReceiptArtifact::from_canonical_bytes(&bytes).unwrap();

            // The decoded artifact's receipt_id must match
            assert_eq!(
                &decoded.receipt_id, id,
                "Replay mismatch at receipt {}: decoded id != stored id",
                idx
            );

            // Re-encode and verify bytes are identical (canonical stability)
            let re_encoded = decoded.canonical_bytes();
            assert_eq!(
                bytes, re_encoded,
                "Canonical bytes not stable at receipt {}",
                idx
            );
        }
    }

    // ── Test 7: Empty lineage is valid ──

    #[test]
    fn test_empty_lineage_valid() {
        let art = ReceiptArtifact::new(
            ReceiptKind::Genesis,
            [0u8; 32],
            [0u8; 32],
            vec![], // genesis has no lineage
            [0u8; 32],
            0,
        );

        assert_eq!(art.lineage.len(), 0);
        let bytes = art.canonical_bytes();
        let decoded = ReceiptArtifact::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.lineage.len(), 0);
    }

    // ── Test 8: hash() returns receipt_id ──

    #[test]
    fn test_hash_equals_receipt_id() {
        let art = ReceiptArtifact::new(
            ReceiptKind::NodeLifecycle,
            [20u8; 32],
            [21u8; 32],
            vec![[22u8; 32]],
            [23u8; 32],
            42,
        );

        // ReceiptPayload::hash() must return receipt_id
        assert_eq!(art.hash(), art.receipt_id);
    }
}
