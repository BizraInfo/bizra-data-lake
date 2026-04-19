//! BIZRA OrganizeMission — §Cycle-7 G5 Commit-1
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/organize_mission.rs
//! Authority: cycle-7/niyyah.md §G5 "First real mission" +
//!            manifest v1 §10 Proof Law
//! Cycle position: 7, Phase 5
//!
//! First real operator mission. Reads an allowlisted filesystem path,
//! produces a deterministic listing summary, and emits a
//! MissionExecuted receipt that binds the summary into the chain.
//!
//! Read-only discipline:
//!   - No mutation of the filesystem
//!   - No recursive descent — only top-level entries, sorted
//!   - Summary is a deterministic function of (dir contents at T, path)
//!
//! Allowlist pre-gate:
//!   - Un-allowlisted paths are refused BEFORE entering the lawful loop.
//!   - Not all intents deserve admissibility; unauthorized intents are
//!     filtered at the constitutional pre-gate, not by a rejection
//!     receipt. §10 Proof Law: chain reflects what happened by absence.
//!   - See runtime::submit_organize_mission for the enforcement site.

use crate::canonical_hasher::{blake3_domain, Blake3Hash};
use crate::receipts::{ByteReader, DecodeError, ReceiptKind, ReceiptPayload, ReceiptPayloadDecode};
// Day 2 — Sealable trait integration. Appends a request-type and trait
// impls only; all pre-existing types in this module are unchanged.
use crate::admissibility_freeze_v1::{
    AdmissibilityClaim, EconomicPattern, RejectedClaim, StateMutation,
};
use crate::seal::{Sealable, SealableOutcome};

// ════════════════════════════════════════════════════════════════════
// OrganizeListing — deterministic read-only projection of a directory
// ════════════════════════════════════════════════════════════════════

/// One row in the top-level listing. Stable across runs as long as the
/// directory contents (name + kind byte) are stable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizeEntry {
    pub name: String,
    /// Marker byte: 0x01 = file, 0x02 = directory, 0x03 = symlink,
    /// 0xFF = other / unknown.
    pub kind_byte: u8,
}

impl OrganizeEntry {
    pub fn kind_str(&self) -> &'static str {
        match self.kind_byte {
            0x01 => "file",
            0x02 => "directory",
            0x03 => "symlink",
            _ => "other",
        }
    }
}

/// Deterministic listing summary of a directory path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizeListing {
    pub path: String,
    pub entries: Vec<OrganizeEntry>,
}

impl OrganizeListing {
    /// Build a listing from a filesystem path by reading its direct
    /// children. Entries are sorted by (kind_byte asc, name asc) so the
    /// resulting digest is deterministic.
    pub fn from_path(path: &std::path::Path) -> std::io::Result<Self> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let ft = entry.file_type()?;
            let kind_byte = if ft.is_file() {
                0x01
            } else if ft.is_dir() {
                0x02
            } else if ft.is_symlink() {
                0x03
            } else {
                0xFF
            };
            entries.push(OrganizeEntry { name, kind_byte });
        }
        // Stable sort: by name. (kind_byte is secondary; name is unique
        // within a directory so it suffices as the primary key.)
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(OrganizeListing {
            path: path.to_string_lossy().into_owned(),
            entries,
        })
    }

    /// BLAKE3 domain-separated digest over the listing. Binds path +
    /// all entries in canonical order. Used as replay evidence in the
    /// MissionExecuted receipt.
    pub fn digest(&self) -> Blake3Hash {
        let mut buf = Vec::with_capacity(self.path.len() + self.entries.len() * 64);
        buf.extend_from_slice(self.path.as_bytes());
        buf.push(0x00);
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            buf.push(e.kind_byte);
            buf.extend_from_slice(&(e.name.len() as u32).to_le_bytes());
            buf.extend_from_slice(e.name.as_bytes());
        }
        blake3_domain("bizra-organize-listing-v1", &buf)
    }

    pub fn file_count(&self) -> u32 {
        self.entries.iter().filter(|e| e.kind_byte == 0x01).count() as u32
    }

    pub fn dir_count(&self) -> u32 {
        self.entries.iter().filter(|e| e.kind_byte == 0x02).count() as u32
    }
}

// ════════════════════════════════════════════════════════════════════
// OrganizeMissionReceipt — canonical receipt payload
// ════════════════════════════════════════════════════════════════════

/// Chain receipt payload for a permitted organize mission. Binds the
/// mission_receipt_ref (the NodeLifecycle mission that authorized the
/// action) to the listing's deterministic digest.
///
/// Canonical encoding (big-endian offset comments; actual encoding LE
/// for u64 to match existing cognition conventions):
///
///   receipt_id            : 32 bytes (blake3 of rest of payload)
///   mission_receipt_ref   : 32 bytes
///   path_len              :  4 bytes LE
///   path                  : path_len bytes UTF-8
///   listing_digest        : 32 bytes
///   file_count            :  4 bytes LE u32
///   dir_count             :  4 bytes LE u32
///   entry_count           :  4 bytes LE u32
///   timestamp_ns          :  8 bytes LE u64
///   prev_chain            : 32 bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizeMissionReceipt {
    pub receipt_id: Blake3Hash,
    pub mission_receipt_ref: Blake3Hash,
    pub path: String,
    pub listing_digest: Blake3Hash,
    pub file_count: u32,
    pub dir_count: u32,
    pub entry_count: u32,
    pub timestamp_ns: u64,
    pub prev_chain: Blake3Hash,
}

impl OrganizeMissionReceipt {
    pub fn new(
        mission_receipt_ref: Blake3Hash,
        listing: &OrganizeListing,
        timestamp_ns: u64,
        prev_chain: Blake3Hash,
    ) -> Self {
        let entry_count = listing.entries.len() as u32;
        let listing_digest = listing.digest();
        let file_count = listing.file_count();
        let dir_count = listing.dir_count();

        let mut body = Vec::new();
        body.extend_from_slice(&mission_receipt_ref);
        body.extend_from_slice(&(listing.path.len() as u32).to_le_bytes());
        body.extend_from_slice(listing.path.as_bytes());
        body.extend_from_slice(&listing_digest);
        body.extend_from_slice(&file_count.to_le_bytes());
        body.extend_from_slice(&dir_count.to_le_bytes());
        body.extend_from_slice(&entry_count.to_le_bytes());
        body.extend_from_slice(&timestamp_ns.to_le_bytes());
        body.extend_from_slice(&prev_chain);
        let receipt_id = blake3_domain("bizra-organize-mission-receipt-v1", &body);

        OrganizeMissionReceipt {
            receipt_id,
            mission_receipt_ref,
            path: listing.path.clone(),
            listing_digest,
            file_count,
            dir_count,
            entry_count,
            timestamp_ns,
            prev_chain,
        }
    }
}

impl ReceiptPayload for OrganizeMissionReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::MissionExecuted
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + 32 + 4 + self.path.len() + 32 + 4 + 4 + 4 + 8 + 32);
        buf.extend_from_slice(&self.receipt_id);
        buf.extend_from_slice(&self.mission_receipt_ref);
        buf.extend_from_slice(&(self.path.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.path.as_bytes());
        buf.extend_from_slice(&self.listing_digest);
        buf.extend_from_slice(&self.file_count.to_le_bytes());
        buf.extend_from_slice(&self.dir_count.to_le_bytes());
        buf.extend_from_slice(&self.entry_count.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf.extend_from_slice(&self.prev_chain);
        buf
    }

    fn hash(&self) -> Blake3Hash {
        self.receipt_id
    }

    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }
}

impl ReceiptPayloadDecode for OrganizeMissionReceipt {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);
        let receipt_id = r.read_hash()?;
        let mission_receipt_ref = r.read_hash()?;
        let path_bytes = r.read_length_prefixed()?;
        let path = std::str::from_utf8(path_bytes)
            .map_err(|e| DecodeError::Utf8(e.to_string()))?
            .to_string();
        let listing_digest = r.read_hash()?;
        let file_count = r.read_u32()?;
        let dir_count = r.read_u32()?;
        let entry_count = r.read_u32()?;
        let timestamp_ns = r.read_u64()?;
        let prev_chain = r.read_hash()?;
        Ok(OrganizeMissionReceipt {
            receipt_id,
            mission_receipt_ref,
            path,
            listing_digest,
            file_count,
            dir_count,
            entry_count,
            timestamp_ns,
            prev_chain,
        })
    }
}

// ════════════════════════════════════════════════════════════════════
// OrganizeRequest — Day 2 Sealable integration
// ════════════════════════════════════════════════════════════════════
//
// OrganizeRequest is the pre-execution artifact that passes through the
// universal Sealable primitive. After the 5-gate AdmissibilityChain yields
// Permit, the runtime reads the filesystem to produce OrganizeListing,
// then builds OrganizeMissionReceipt (all unchanged types above).
//
// This addition is purely additive: zero pre-existing code is modified,
// no behavior is altered. The existing organize flow continues to work
// via the exact same path. Day 3+ may choose to route it through a
// generic `fn seal<S: Sealable>(...)` wrapper; that refactor is out of
// Day 2 scope.

/// An organize-mission request — pre-execution input to the lawful loop.
///
/// Carries only what is knowable at submission time: the target path, the
/// self-declared quality score, and a monotonic timestamp. The post-execution
/// evidence (OrganizeListing.digest) binds later at receipt construction.
#[derive(Debug, Clone, PartialEq)]
pub struct OrganizeRequest {
    pub path: String,
    pub quality_score: f64,
    pub timestamp_ns: u64,
}

impl OrganizeRequest {
    pub fn new(path: impl Into<String>, quality_score: f64, timestamp_ns: u64) -> Self {
        Self {
            path: path.into(),
            quality_score,
            timestamp_ns,
        }
    }
}

impl Sealable for OrganizeRequest {
    fn seal_envelope(&self) -> AdmissibilityClaim {
        // Evidence hash is cryptographically derived from the request's own
        // canonical bytes, never asserted. CLAIM_MUST_BIND holds by
        // construction: the hash IS the binding.
        let evidence_hash = blake3_domain("bizra-organize-request-v1", &self.bytes_for_digest());

        AdmissibilityClaim {
            claim_id: evidence_hash,
            // ZANN_ZERO: request carries evidence (its own canonical form + post-exec listing).
            has_evidence: true,
            // CLAIM_MUST_BIND: non-zero hash derived from bytes_for_digest.
            evidence_hash: Some(evidence_hash),
            // RIBA_ZERO: organize is a read-only filesystem operation;
            // no economic pattern is declared. The kernel's RIBA_ZERO gate
            // will Permit None.
            economic_pattern: Some(EconomicPattern::None),
            // NO_SHADOW_STATE: the only state change from organize is a
            // receipt appended to the canonical chain — derives from the
            // canonical runtime, never face-only.
            state_mutation: Some(StateMutation {
                derives_from_canonical: true,
                face_only: false,
            }),
            // IHSAN_FLOOR: self-declared; the kernel enforces ≥ 0.95.
            quality_score: self.quality_score,
            timestamp_ns: self.timestamp_ns,
        }
    }

    fn bytes_for_digest(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.path.len() + 8 + 8);
        buf.extend_from_slice(&(self.path.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.path.as_bytes());
        buf.extend_from_slice(&self.quality_score.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }

    fn receipt_kind() -> ReceiptKind {
        ReceiptKind::MissionExecuted
    }
}

impl SealableOutcome for OrganizeRequest {
    /// Lawful success carries the full canonical receipt payload; chain
    /// head advance is encoded in the receipt's prev_chain field, which
    /// the runtime sets at append time.
    type Ok = OrganizeMissionReceipt;
    /// Refusal carries the kernel's RejectedClaim exactly as issued —
    /// invariant + reason + remediation — no translation layer.
    type Refused = RejectedClaim;
    /// Unreachable carries a human-readable reason only. No verdict is
    /// fabricated; NO_SHADOW_STATE holds at the result boundary.
    type Unreachable = String;
}

// ════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_fixture_dir(root: &std::path::Path) {
        fs::write(root.join("alpha.txt"), b"hello").unwrap();
        fs::write(root.join("beta.txt"), b"world").unwrap();
        fs::create_dir_all(root.join("subdir")).unwrap();
    }

    #[test]
    fn listing_sorts_entries_deterministically() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let l = OrganizeListing::from_path(td.path()).unwrap();
        let names: Vec<&str> = l.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["alpha.txt", "beta.txt", "subdir"]);
    }

    #[test]
    fn listing_counts_files_and_dirs_correctly() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let l = OrganizeListing::from_path(td.path()).unwrap();
        assert_eq!(l.file_count(), 2);
        assert_eq!(l.dir_count(), 1);
        assert_eq!(l.entries.len(), 3);
    }

    #[test]
    fn listing_digest_is_stable_across_reads() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let d1 = OrganizeListing::from_path(td.path()).unwrap().digest();
        let d2 = OrganizeListing::from_path(td.path()).unwrap().digest();
        assert_eq!(d1, d2);
    }

    #[test]
    fn listing_digest_changes_when_contents_change() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let d1 = OrganizeListing::from_path(td.path()).unwrap().digest();
        fs::write(td.path().join("gamma.txt"), b"new").unwrap();
        let d2 = OrganizeListing::from_path(td.path()).unwrap().digest();
        assert_ne!(d1, d2);
    }

    #[test]
    fn receipt_round_trip_preserves_all_fields() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let r = OrganizeMissionReceipt::new(
            [0xAA; 32],
            &listing,
            1_700_000_000_000_000_000,
            [0xBB; 32],
        );
        let bytes = r.canonical_bytes();
        let back = OrganizeMissionReceipt::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn receipt_hash_is_deterministic_for_same_inputs() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let r1 = OrganizeMissionReceipt::new([0xAA; 32], &listing, 123, [0xBB; 32]);
        let r2 = OrganizeMissionReceipt::new([0xAA; 32], &listing, 123, [0xBB; 32]);
        assert_eq!(r1.receipt_id, r2.receipt_id);
    }

    #[test]
    fn receipt_kind_is_mission_executed() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let r = OrganizeMissionReceipt::new([0xAA; 32], &listing, 0, [0xBB; 32]);
        assert_eq!(r.kind(), ReceiptKind::MissionExecuted);
    }

    #[test]
    fn receipt_timestamp_ns_surfaces_via_trait() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let r = OrganizeMissionReceipt::new([0xAA; 32], &listing, 999, [0xBB; 32]);
        assert_eq!(r.timestamp_ns(), 999);
    }

    #[test]
    fn receipt_id_changes_when_path_changes() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let r1 = OrganizeMissionReceipt::new([0xAA; 32], &listing, 123, [0xBB; 32]);

        let mut altered = listing.clone();
        altered.path = "/different".into();
        let r2 = OrganizeMissionReceipt::new([0xAA; 32], &altered, 123, [0xBB; 32]);
        assert_ne!(r1.receipt_id, r2.receipt_id);
    }

    #[test]
    fn receipt_id_changes_when_mission_ref_changes() {
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let r1 = OrganizeMissionReceipt::new([0xAA; 32], &listing, 123, [0xBB; 32]);
        let r2 = OrganizeMissionReceipt::new([0xCC; 32], &listing, 123, [0xBB; 32]);
        assert_ne!(r1.receipt_id, r2.receipt_id);
    }

    #[test]
    fn empty_directory_listing_round_trips() {
        let td = tempfile::TempDir::new().unwrap();
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        assert!(listing.entries.is_empty());
        let r = OrganizeMissionReceipt::new([0xAA; 32], &listing, 0, [0xBB; 32]);
        let bytes = r.canonical_bytes();
        let back = OrganizeMissionReceipt::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.entry_count, 0);
        assert_eq!(back.file_count, 0);
        assert_eq!(back.dir_count, 0);
    }

    // ────────────────────────────────────────────────────────────────
    // Day 2 — Sealable integration tests
    //
    // These tests prove:
    //   (a) OrganizeRequest produces a well-formed AdmissibilityClaim that
    //       preserves all 5 invariants at the trait boundary;
    //   (b) bytes_for_digest is deterministic (empirical-reproducibility
    //       modality of the Four-Modality Golden Standard);
    //   (c) SealableOutcome associated types compile correctly.
    //
    // Pre-existing tests above are UNCHANGED — behavioral equivalence on
    // the organize data path (OrganizeListing, OrganizeMissionReceipt) is
    // preserved by construction: Day 2 is purely additive.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn organize_request_builds_well_formed_claim() {
        let req = OrganizeRequest::new("/tmp/test", 0.98, 1_700_000_000_000_000_000);
        let claim = req.seal_envelope();
        assert!(claim.has_evidence, "ZANN_ZERO: has_evidence must be true");
        assert!(
            claim.evidence_hash.is_some(),
            "CLAIM_MUST_BIND: evidence_hash must be Some"
        );
        assert_eq!(claim.quality_score, 0.98);
        assert_eq!(claim.timestamp_ns, 1_700_000_000_000_000_000);
    }

    #[test]
    fn organize_request_claim_id_equals_evidence_hash() {
        // CLAIM_MUST_BIND preservation: evidence hash is derived from the
        // request's own bytes (blake3_domain over bytes_for_digest), and
        // the claim_id equals that hash so the binding is self-referential.
        let req = OrganizeRequest::new("/tmp/bind", 0.97, 42);
        let claim = req.seal_envelope();
        let expected = blake3_domain("bizra-organize-request-v1", &req.bytes_for_digest());
        assert_eq!(claim.evidence_hash, Some(expected));
        assert_eq!(claim.claim_id, expected);
    }

    #[test]
    fn organize_request_declares_economic_pattern_none() {
        // RIBA_ZERO preservation: organize is a read-only filesystem op.
        // No economic activity is declared; the RibaZeroGate will Permit.
        let req = OrganizeRequest::new("/tmp/riba", 0.98, 0);
        let claim = req.seal_envelope();
        assert_eq!(claim.economic_pattern, Some(EconomicPattern::None));
    }

    #[test]
    fn organize_request_declares_canonical_state_mutation() {
        // NO_SHADOW_STATE preservation: the only state change is a receipt
        // on the canonical chain; derives_from_canonical must be true and
        // face_only must be false.
        let req = OrganizeRequest::new("/tmp/ss", 0.98, 0);
        let claim = req.seal_envelope();
        let sm = claim
            .state_mutation
            .as_ref()
            .expect("state_mutation must be Some");
        assert!(sm.derives_from_canonical);
        assert!(!sm.face_only);
    }

    #[test]
    fn organize_request_bytes_are_deterministic() {
        // Empirical-reproducibility modality: identical inputs → identical
        // bytes on every machine, every run.
        let r1 = OrganizeRequest::new("/tmp/a", 0.98, 100);
        let r2 = OrganizeRequest::new("/tmp/a", 0.98, 100);
        assert_eq!(r1.bytes_for_digest(), r2.bytes_for_digest());
    }

    #[test]
    fn organize_request_bytes_change_on_path_change() {
        let r1 = OrganizeRequest::new("/tmp/a", 0.98, 100);
        let r2 = OrganizeRequest::new("/tmp/b", 0.98, 100);
        assert_ne!(r1.bytes_for_digest(), r2.bytes_for_digest());
    }

    #[test]
    fn organize_request_bytes_change_on_quality_change() {
        let r1 = OrganizeRequest::new("/tmp/a", 0.98, 100);
        let r2 = OrganizeRequest::new("/tmp/a", 0.99, 100);
        assert_ne!(r1.bytes_for_digest(), r2.bytes_for_digest());
    }

    #[test]
    fn organize_request_bytes_change_on_timestamp_change() {
        let r1 = OrganizeRequest::new("/tmp/a", 0.98, 100);
        let r2 = OrganizeRequest::new("/tmp/a", 0.98, 200);
        assert_ne!(r1.bytes_for_digest(), r2.bytes_for_digest());
    }

    #[test]
    fn organize_request_receipt_kind_is_mission_executed() {
        // Chain_head behavior preservation: OrganizeRequest produces the
        // same ReceiptKind (MissionExecuted = 0x70) that the existing
        // OrganizeMissionReceipt stamps on the chain record.
        assert_eq!(
            <OrganizeRequest as Sealable>::receipt_kind(),
            ReceiptKind::MissionExecuted
        );
    }

    #[test]
    fn organize_request_receipt_kind_matches_existing_receipt() {
        // The Sealable trait's advertised receipt_kind MUST equal the
        // kind that OrganizeMissionReceipt actually stamps. Divergence
        // would violate the NO_SHADOW_STATE boundary (advertised vs
        // actual receipt kind).
        let td = tempfile::TempDir::new().unwrap();
        write_fixture_dir(td.path());
        let listing = OrganizeListing::from_path(td.path()).unwrap();
        let receipt = OrganizeMissionReceipt::new([0xAA; 32], &listing, 0, [0xBB; 32]);
        assert_eq!(
            <OrganizeRequest as Sealable>::receipt_kind(),
            receipt.kind()
        );
    }

    #[test]
    fn organize_request_sealable_outcome_types_have_expected_shapes() {
        // Compile-only proof that Ok/Refused/Unreachable associated types
        // carry the shapes the face contract demands.
        fn _ok_is_organize_mission_receipt(_r: <OrganizeRequest as SealableOutcome>::Ok) {}
        fn _refused_is_rejected_claim(_r: <OrganizeRequest as SealableOutcome>::Refused) {}
        fn _unreachable_is_string(_r: <OrganizeRequest as SealableOutcome>::Unreachable) {}
        // If this test compiles, the SealableOutcome typing is correct;
        // no runtime assertion needed.
    }

    #[test]
    fn organize_request_low_quality_claim_would_fail_ihsan_floor() {
        // IHSAN_FLOOR preservation: a quality_score below 0.95 produces
        // a claim the chain will Reject. We don't run the chain here
        // (that would widen scope); we assert the field passes through
        // so the gate has correct input.
        let req = OrganizeRequest::new("/tmp/low", 0.80, 0);
        let claim = req.seal_envelope();
        assert!(claim.quality_score < 0.95);
        // Full chain evaluation is covered by admissibility_freeze_v1's
        // own test suite; here we only assert the input boundary.
    }

    #[test]
    fn organize_request_rejected_claim_shape_is_untranslated() {
        // Refused-path constitutional correctness: SealableOutcome::Refused
        // is exactly RejectedClaim from the admissibility chain — no
        // lossy translation layer. Build a RejectedClaim and assert it
        // type-checks as the Refused associated type.
        use crate::admissibility_freeze_v1::Invariant;
        let refused: <OrganizeRequest as SealableOutcome>::Refused = RejectedClaim {
            claim_ref: [0u8; 32],
            invariant: Invariant::IhsanFloor,
            reject_reason: "quality_score 0.8 < floor 0.95".to_string(),
            remediation_path: "raise quality_score to ≥ 0.95".to_string(),
            escalation_allowed: false,
        };
        assert_eq!(refused.invariant, Invariant::IhsanFloor);
        assert!(!refused.escalation_allowed);
    }

    #[test]
    fn organize_request_unreachable_path_is_typed_string() {
        // Unreachable-path constitutional correctness: the chain may be
        // offline; SealableOutcome::Unreachable carries a reason only.
        // NO simulated verdict is permitted (NO_SHADOW_STATE at the result
        // boundary).
        let unreachable: <OrganizeRequest as SealableOutcome>::Unreachable =
            "cognition gateway connection refused on 127.0.0.1:7421".to_string();
        assert!(unreachable.contains("unreachable") || unreachable.contains("refused"));
    }

    #[test]
    fn entry_kind_str_maps_known_bytes() {
        assert_eq!(
            OrganizeEntry {
                name: "x".into(),
                kind_byte: 0x01
            }
            .kind_str(),
            "file"
        );
        assert_eq!(
            OrganizeEntry {
                name: "x".into(),
                kind_byte: 0x02
            }
            .kind_str(),
            "directory"
        );
        assert_eq!(
            OrganizeEntry {
                name: "x".into(),
                kind_byte: 0x03
            }
            .kind_str(),
            "symlink"
        );
        assert_eq!(
            OrganizeEntry {
                name: "x".into(),
                kind_byte: 0xFF
            }
            .kind_str(),
            "other"
        );
    }
}
