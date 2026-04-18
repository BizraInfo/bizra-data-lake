//! BIZRA Receipt Chain — Two-Layer Model with Decode
//! ===================================================
//! File: crates/bizra-kernel/src/receipts.rs
//! Domain tag: bizra-receipts-v1
//!
//! Layer 1: chain record {kind, hash, prev} — thin, immutable, append-only.
//! Layer 2: payload store — hash-addressed canonical payloads.
//!
//! Encode via ReceiptPayload::canonical_bytes (forward, always required).
//! Decode via ReceiptPayloadDecode::from_canonical_bytes (reverse, only for
//! payload kinds that the rehydrate loop needs to read back).
//!
//! Durability invariant:
//!   1. canonicalize payload
//!   2. hash payload
//!   3. persist payload by hash (fsync or equivalent in production)
//!   4. append chain record using same hash
//!   5. only then promote ctx.receipt_chain
//!
//! If step 3 fails, step 4 must not happen.

use std::collections::HashMap;
use std::sync::Mutex;

pub type Blake3Hash = [u8; 32];

// ============================================================================
// Chain record — Layer 1
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReceiptKind {
    Genesis            = 0x00,
    CognitionBoot      = 0x10,
    Myelination        = 0x20,
    Demyelination      = 0x21,
    ReasoningSession   = 0x30,
    GovernanceDecision = 0x40,
    NodeLifecycle      = 0x50,
    // Cycle-7 G1 — dedicated kind for ManifestArtifact (was previously
    // reusing NodeLifecycle). Non-breaking: new byte; old variants unchanged.
    Manifest           = 0x60,
    // Cycle-7 G2 — dedicated kind for PrincipalActivationReceipt.
    // Binds a NodeLifecycle mission receipt to a principal profile hash,
    // non-transferable and proof-bearing per niyyah §6 (local-only PoI).
    PrincipalActivation = 0x61,
    DegradedPath       = 0xF0,
}

impl ReceiptKind {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Genesis),
            0x10 => Some(Self::CognitionBoot),
            0x20 => Some(Self::Myelination),
            0x21 => Some(Self::Demyelination),
            0x30 => Some(Self::ReasoningSession),
            0x40 => Some(Self::GovernanceDecision),
            0x50 => Some(Self::NodeLifecycle),
            0x60 => Some(Self::Manifest),
            0x61 => Some(Self::PrincipalActivation),
            0xF0 => Some(Self::DegradedPath),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Receipt {
    pub kind: ReceiptKind,
    pub hash: Blake3Hash,
    pub prev: Blake3Hash,
}

// ============================================================================
// Payload traits — encode always, decode only where needed
// ============================================================================

pub trait ReceiptPayload: Send + Sync + 'static {
    fn kind(&self) -> ReceiptKind;
    fn canonical_bytes(&self) -> Vec<u8>;
    fn hash(&self) -> Blake3Hash;
    /// Nanoseconds since UNIX epoch, as embedded in the payload.
    /// Default is 0 ("not present"); payloads that carry a timestamp override.
    /// Consumed by ReceiptChain to expose chain-level latest-timestamp metadata.
    fn timestamp_ns(&self) -> u64 { 0 }
}

#[derive(Debug, Clone)]
pub enum DecodeError {
    ShortInput { need: usize, got: usize },
    UnknownDiscriminant { field: &'static str, byte: u8 },
    Utf8(String),
    HashMismatch { expected: Blake3Hash, computed: Blake3Hash },
}

/// Only implemented for payloads that the rehydrate loop must decode.
/// DegradedPath, ReasoningSession, and GovernanceDecision receipts do NOT
/// need decode — they are chain-of-custody records, not state transitions
/// that affect the graph's reconstructable state.
pub trait ReceiptPayloadDecode: ReceiptPayload + Sized {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError>;
}

#[derive(Debug, Clone)]
pub struct ReceiptEnvelope<T: ReceiptPayload> {
    pub hash: Blake3Hash,
    pub payload: T,
}

// ============================================================================
// Byte-reading helpers — small, explicit, tested
// ============================================================================

/// Reader with bounds checking. Every read returns Result so callers never panic.
pub struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }

    pub fn remaining(&self) -> usize { self.bytes.len() - self.pos }

    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::ShortInput { need: n, got: self.remaining() });
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn read_hash(&mut self) -> Result<Blake3Hash, DecodeError> {
        let mut h = [0u8; 32];
        h.copy_from_slice(self.read_bytes(32)?);
        Ok(h)
    }

    pub fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.read_bytes(1)?[0])
    }

    pub fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub fn read_f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    pub fn read_length_prefixed(&mut self) -> Result<&'a [u8], DecodeError> {
        let len = self.read_u32()? as usize;
        self.read_bytes(len)
    }
}

// ============================================================================
// Payload store — Layer 2
// ============================================================================

pub trait PayloadStore: Send + Sync {
    fn put(&self, hash: Blake3Hash, bytes: Vec<u8>) -> Result<(), StoreError>;
    fn get(&self, hash: &Blake3Hash) -> Result<Option<Vec<u8>>, StoreError>;
    fn contains(&self, hash: &Blake3Hash) -> Result<bool, StoreError>;
}

#[derive(Debug, Clone)]
pub enum StoreError {
    IoError(String),
    Corruption { hash: Blake3Hash, reason: String },
    OutOfSpace,
}

pub struct InMemoryPayloadStore {
    inner: Mutex<HashMap<Blake3Hash, Vec<u8>>>,
}

impl InMemoryPayloadStore {
    pub fn new() -> Self { Self { inner: Mutex::new(HashMap::new()) } }
}

impl Default for InMemoryPayloadStore {
    fn default() -> Self { Self::new() }
}

impl PayloadStore for InMemoryPayloadStore {
    fn put(&self, hash: Blake3Hash, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.inner.lock()
            .map_err(|e| StoreError::IoError(e.to_string()))?
            .insert(hash, bytes);
        Ok(())
    }
    fn get(&self, hash: &Blake3Hash) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.inner.lock()
            .map_err(|e| StoreError::IoError(e.to_string()))?
            .get(hash).cloned())
    }
    fn contains(&self, hash: &Blake3Hash) -> Result<bool, StoreError> {
        Ok(self.inner.lock()
            .map_err(|e| StoreError::IoError(e.to_string()))?
            .contains_key(hash))
    }
}

// ============================================================================
// Receipt chain
// ============================================================================

#[derive(Debug, Clone)]
pub enum ChainError {
    Discontinuity { expected_prev: Blake3Hash, got: Blake3Hash },
    PayloadPersistence(StoreError),
    PayloadMissing(Blake3Hash),
    PayloadDecode(DecodeError),
}

impl From<StoreError> for ChainError {
    fn from(e: StoreError) -> Self { ChainError::PayloadPersistence(e) }
}
impl From<DecodeError> for ChainError {
    fn from(e: DecodeError) -> Self { ChainError::PayloadDecode(e) }
}

pub struct ReceiptChain {
    records: Vec<Receipt>,
    head: Blake3Hash,
    store: Box<dyn PayloadStore>,
    last_timestamp_ns: Option<u64>,
}

impl ReceiptChain {
    pub fn new(genesis: Blake3Hash, store: Box<dyn PayloadStore>) -> Self {
        Self { records: Vec::new(), head: genesis, store, last_timestamp_ns: None }
    }

    pub fn head(&self) -> Blake3Hash { self.head }
    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }

    /// Most recent payload timestamp observed during append, if any.
    /// Returns None when the chain is empty, or when no appended payload
    /// reported a non-zero timestamp_ns via the ReceiptPayload trait.
    pub fn latest_timestamp(&self) -> Option<u64> { self.last_timestamp_ns }

    pub fn append_with_payload<T: ReceiptPayload>(
        &mut self,
        payload: T,
    ) -> Result<Blake3Hash, ChainError> {
        let kind = payload.kind();
        let bytes = payload.canonical_bytes();
        let computed_hash = payload.hash();
        let ts_ns = payload.timestamp_ns();

        // Step 3: persist payload FIRST
        self.store.put(computed_hash, bytes)?;

        // Step 5: only after persistence succeeds, advance chain
        let prev = self.head;
        let record = Receipt { kind, hash: computed_hash, prev };
        self.records.push(record);
        self.head = computed_hash;
        if ts_ns > 0 {
            self.last_timestamp_ns = Some(ts_ns);
        }

        Ok(computed_hash)
    }

    pub fn fetch_payload_bytes(&self, hash: &Blake3Hash) -> Result<Option<Vec<u8>>, ChainError> {
        Ok(self.store.get(hash)?)
    }

    /// Fetch and decode a payload as a specific type. Verifies hash round-trip.
    pub fn fetch_and_decode<T: ReceiptPayloadDecode>(
        &self,
        hash: &Blake3Hash,
    ) -> Result<T, ChainError> {
        let bytes = self.store.get(hash)?
            .ok_or(ChainError::PayloadMissing(*hash))?;
        let payload = T::from_canonical_bytes(&bytes)?;
        let computed = payload.hash();
        if &computed != hash {
            return Err(ChainError::PayloadDecode(DecodeError::HashMismatch {
                expected: *hash,
                computed,
            }));
        }
        Ok(payload)
    }

    pub fn records(&self) -> impl Iterator<Item = &Receipt> {
        self.records.iter()
    }

    pub fn verify_continuity(&self, genesis: Blake3Hash) -> Result<(), ChainError> {
        let mut expected_prev = genesis;
        for record in &self.records {
            if record.prev != expected_prev {
                return Err(ChainError::Discontinuity {
                    expected_prev,
                    got: record.prev,
                });
            }
            expected_prev = record.hash;
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPayload { kind: ReceiptKind, data: Vec<u8> }
    impl ReceiptPayload for DummyPayload {
        fn kind(&self) -> ReceiptKind { self.kind }
        fn canonical_bytes(&self) -> Vec<u8> { self.data.clone() }
        fn hash(&self) -> Blake3Hash {
            let mut h = [0u8; 32];
            for (i, b) in self.data.iter().take(32).enumerate() { h[i] = *b; }
            h
        }
    }

    #[test]
    fn append_advances_head_only_after_persist() {
        let store = Box::new(InMemoryPayloadStore::new());
        let genesis = [0u8; 32];
        let mut chain = ReceiptChain::new(genesis, store);

        let payload = DummyPayload {
            kind: ReceiptKind::CognitionBoot,
            data: vec![1, 2, 3, 4, 5],
        };
        let hash = chain.append_with_payload(payload).unwrap();

        assert_eq!(chain.head(), hash);
        assert_eq!(chain.len(), 1);
        assert!(chain.fetch_payload_bytes(&hash).unwrap().is_some());
    }

    #[test]
    fn continuity_verification_detects_gap() {
        let store = Box::new(InMemoryPayloadStore::new());
        let genesis = [0u8; 32];
        let mut chain = ReceiptChain::new(genesis, store);

        for i in 1..=3u8 {
            chain.append_with_payload(DummyPayload {
                kind: ReceiptKind::ReasoningSession,
                data: vec![i; 5],
            }).unwrap();
        }

        assert!(chain.verify_continuity(genesis).is_ok());
    }

    #[test]
    fn latest_timestamp_tracks_non_zero_and_ignores_zero() {
        struct StampedPayload { ts_ns: u64, data: Vec<u8> }
        impl ReceiptPayload for StampedPayload {
            fn kind(&self) -> ReceiptKind { ReceiptKind::ReasoningSession }
            fn canonical_bytes(&self) -> Vec<u8> { self.data.clone() }
            fn hash(&self) -> Blake3Hash {
                let mut h = [0u8; 32];
                for (i, b) in self.data.iter().take(32).enumerate() { h[i] = *b; }
                h
            }
            fn timestamp_ns(&self) -> u64 { self.ts_ns }
        }

        let store = Box::new(InMemoryPayloadStore::new());
        let mut chain = ReceiptChain::new([0u8; 32], store);
        assert_eq!(chain.latest_timestamp(), None, "empty chain has no timestamp");

        // Default ReceiptPayload (timestamp_ns() = 0) must not shift the accessor.
        chain.append_with_payload(DummyPayload {
            kind: ReceiptKind::CognitionBoot,
            data: vec![10, 11, 12, 13],
        }).unwrap();
        assert_eq!(chain.latest_timestamp(), None, "zero-timestamp payloads do not set latest");

        // Real timestamp propagates.
        chain.append_with_payload(StampedPayload {
            ts_ns: 1_700_000_000_000_000_000,
            data: vec![20, 21, 22, 23],
        }).unwrap();
        assert_eq!(chain.latest_timestamp(), Some(1_700_000_000_000_000_000));

        // Subsequent zero-timestamp append does NOT erase the last known.
        chain.append_with_payload(DummyPayload {
            kind: ReceiptKind::NodeLifecycle,
            data: vec![30, 31, 32, 33],
        }).unwrap();
        assert_eq!(chain.latest_timestamp(), Some(1_700_000_000_000_000_000),
            "zero-timestamp payloads must not clear the last real timestamp");

        // Later real timestamp advances.
        chain.append_with_payload(StampedPayload {
            ts_ns: 1_800_000_000_000_000_000,
            data: vec![40, 41, 42, 43],
        }).unwrap();
        assert_eq!(chain.latest_timestamp(), Some(1_800_000_000_000_000_000));
    }

    #[test]
    fn byte_reader_bounds_check() {
        let bytes = [1u8, 2, 3];
        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.read_u8().unwrap(), 1);
        assert_eq!(r.read_u8().unwrap(), 2);
        assert_eq!(r.read_u8().unwrap(), 3);
        assert!(matches!(r.read_u8(), Err(DecodeError::ShortInput { .. })));
    }
}
