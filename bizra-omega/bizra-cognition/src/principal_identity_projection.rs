//! Typed, authority-neutral projection of a chain-sealed principal identity.
//!
//! This module does not create identity, grant authority, or expose an HTTP
//! contract. It only verifies that an exact chain record is a
//! `PrincipalActivationReceipt`, decodes its canonical payload through the
//! receipt store, verifies the payload hash round-trip, and returns the
//! identity-bearing fields already sealed by the producer.

use crate::principal_activation::PrincipalActivationReceipt;
use crate::receipts::{Blake3Hash, ChainError, ReceiptChain, ReceiptKind};

/// Identity-bearing fields recovered from one verified
/// `PrincipalActivationReceipt` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrincipalIdentityProjection {
    pub receipt_id: Blake3Hash,
    pub activation_receipt_ref: Blake3Hash,
    pub principal_profile_hash: Blake3Hash,
    pub node_pubkey: Blake3Hash,
    pub principal_id: Blake3Hash,
    pub timestamp_ns: u64,
    pub prev_chain: Blake3Hash,
}

/// Fail-closed reasons for principal-identity projection.
#[derive(Debug)]
pub enum PrincipalIdentityProjectionError {
    ReceiptNotFound(Blake3Hash),
    WrongReceiptKind {
        expected: ReceiptKind,
        actual: ReceiptKind,
    },
    Chain(ChainError),
}

impl std::fmt::Display for PrincipalIdentityProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReceiptNotFound(hash) => {
                write!(f, "receipt not found in active chain: {:02x?}", hash)
            }
            Self::WrongReceiptKind { expected, actual } => write!(
                f,
                "wrong receipt kind: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::Chain(err) => write!(f, "receipt payload verification failed: {:?}", err),
        }
    }
}

impl std::error::Error for PrincipalIdentityProjectionError {}

impl From<ChainError> for PrincipalIdentityProjectionError {
    fn from(value: ChainError) -> Self {
        Self::Chain(value)
    }
}

/// Recover the identity material sealed by an exact principal-activation
/// receipt.
///
/// The chain record kind is checked before payload decoding. Payload recovery
/// then uses `ReceiptChain::fetch_and_decode`, which verifies that the decoded
/// payload hashes back to the requested chain hash.
///
/// This function has no side effects and returns no authority or consent.
pub fn project_principal_activation_identity(
    chain: &ReceiptChain,
    receipt_hash: Blake3Hash,
) -> Result<PrincipalIdentityProjection, PrincipalIdentityProjectionError> {
    let record = chain
        .records()
        .find(|record| record.hash == receipt_hash)
        .ok_or(PrincipalIdentityProjectionError::ReceiptNotFound(
            receipt_hash,
        ))?;

    if record.kind != ReceiptKind::PrincipalActivation {
        return Err(PrincipalIdentityProjectionError::WrongReceiptKind {
            expected: ReceiptKind::PrincipalActivation,
            actual: record.kind,
        });
    }

    let receipt = chain.fetch_and_decode::<PrincipalActivationReceipt>(&receipt_hash)?;

    Ok(PrincipalIdentityProjection {
        receipt_id: receipt.receipt_id,
        activation_receipt_ref: receipt.activation_receipt_ref,
        principal_profile_hash: receipt.principal_profile_hash,
        node_pubkey: receipt.node_pubkey,
        principal_id: receipt.principal_id,
        timestamp_ns: receipt.timestamp_ns,
        prev_chain: receipt.prev_chain,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::canonical_hasher::blake3_domain;
    use crate::receipt_history_cache::ReceiptHistorySnapshot;
    use crate::receipts::{
        DecodeError, InMemoryPayloadStore, PayloadStore, ReceiptPayload, StoreError,
    };

    fn activation_receipt() -> PrincipalActivationReceipt {
        PrincipalActivationReceipt::new(
            [0x11; 32],
            [0x22; 32],
            [0x33; 32],
            [0x44; 32],
            1_759_999_999,
            [0x55; 32],
        )
    }

    #[test]
    fn principal_activation_receipt_projects_hash_bound_identity() {
        let mut chain = ReceiptChain::new([0u8; 32], Box::new(InMemoryPayloadStore::new()));
        let receipt = activation_receipt();
        let receipt_hash = chain.append_with_payload(receipt.clone()).unwrap();

        let projected = project_principal_activation_identity(&chain, receipt_hash).unwrap();

        assert_eq!(projected.receipt_id, receipt_hash);
        assert_eq!(projected.receipt_id, receipt.receipt_id);
        assert_eq!(
            projected.activation_receipt_ref,
            receipt.activation_receipt_ref
        );
        assert_eq!(
            projected.principal_profile_hash,
            receipt.principal_profile_hash
        );
        assert_eq!(projected.node_pubkey, receipt.node_pubkey);
        assert_eq!(projected.principal_id, receipt.principal_id);
        assert_eq!(projected.timestamp_ns, receipt.timestamp_ns);
        assert_eq!(projected.prev_chain, receipt.prev_chain);
    }

    #[derive(Clone, Default)]
    struct SharedStore {
        inner: Arc<Mutex<HashMap<Blake3Hash, Vec<u8>>>>,
    }

    impl PayloadStore for SharedStore {
        fn put(&self, hash: Blake3Hash, bytes: Vec<u8>) -> Result<(), StoreError> {
            self.inner
                .lock()
                .map_err(|err| StoreError::IoError(err.to_string()))?
                .insert(hash, bytes);
            Ok(())
        }

        fn get(&self, hash: &Blake3Hash) -> Result<Option<Vec<u8>>, StoreError> {
            Ok(self
                .inner
                .lock()
                .map_err(|err| StoreError::IoError(err.to_string()))?
                .get(hash)
                .cloned())
        }

        fn contains(&self, hash: &Blake3Hash) -> Result<bool, StoreError> {
            Ok(self
                .inner
                .lock()
                .map_err(|err| StoreError::IoError(err.to_string()))?
                .contains_key(hash))
        }
    }

    #[test]
    fn principal_activation_projection_survives_chain_reconstruction() {
        let genesis = [0u8; 32];
        let store = SharedStore::default();
        let mut chain = ReceiptChain::new(genesis, Box::new(store.clone()));
        let receipt = activation_receipt();
        let receipt_hash = chain.append_with_payload(receipt.clone()).unwrap();
        let snapshot = ReceiptHistorySnapshot {
            head: chain.head(),
            last_timestamp_ns: chain.latest_timestamp(),
            records: chain.records().copied().collect(),
        };

        drop(chain);

        let restored =
            ReceiptChain::restore_from_snapshot(genesis, snapshot, Box::new(store)).unwrap();
        let projected = project_principal_activation_identity(&restored, receipt_hash).unwrap();

        assert_eq!(restored.head(), receipt_hash);
        assert_eq!(projected.receipt_id, receipt_hash);
        assert_eq!(projected.node_pubkey, receipt.node_pubkey);
        assert_eq!(projected.principal_id, receipt.principal_id);
        assert_eq!(
            projected.principal_profile_hash,
            receipt.principal_profile_hash
        );
    }

    struct OtherReceipt {
        hash: Blake3Hash,
    }

    impl OtherReceipt {
        fn new() -> Self {
            Self {
                hash: blake3_domain("bizra-projection-other-receipt-v1", b"other"),
            }
        }
    }

    impl ReceiptPayload for OtherReceipt {
        fn kind(&self) -> ReceiptKind {
            ReceiptKind::GovernanceDecision
        }

        fn canonical_bytes(&self) -> Vec<u8> {
            b"other".to_vec()
        }

        fn hash(&self) -> Blake3Hash {
            self.hash
        }
    }

    #[test]
    fn non_principal_receipt_is_refused_before_payload_decode() {
        let mut chain = ReceiptChain::new([0u8; 32], Box::new(InMemoryPayloadStore::new()));
        let hash = chain.append_with_payload(OtherReceipt::new()).unwrap();

        let err = project_principal_activation_identity(&chain, hash).unwrap_err();
        assert!(matches!(
            err,
            PrincipalIdentityProjectionError::WrongReceiptKind {
                expected: ReceiptKind::PrincipalActivation,
                actual: ReceiptKind::GovernanceDecision,
            }
        ));
    }

    #[test]
    fn unknown_receipt_hash_is_refused() {
        let chain = ReceiptChain::new([0u8; 32], Box::new(InMemoryPayloadStore::new()));
        let missing = [0x99; 32];

        let err = project_principal_activation_identity(&chain, missing).unwrap_err();
        assert!(matches!(
            err,
            PrincipalIdentityProjectionError::ReceiptNotFound(hash) if hash == missing
        ));
    }

    #[derive(Default)]
    struct CorruptingStore {
        inner: Mutex<HashMap<Blake3Hash, Vec<u8>>>,
    }

    impl PayloadStore for CorruptingStore {
        fn put(&self, hash: Blake3Hash, mut bytes: Vec<u8>) -> Result<(), StoreError> {
            if let Some(first) = bytes.first_mut() {
                *first ^= 0xff;
            }
            self.inner
                .lock()
                .map_err(|err| StoreError::IoError(err.to_string()))?
                .insert(hash, bytes);
            Ok(())
        }

        fn get(&self, hash: &Blake3Hash) -> Result<Option<Vec<u8>>, StoreError> {
            Ok(self
                .inner
                .lock()
                .map_err(|err| StoreError::IoError(err.to_string()))?
                .get(hash)
                .cloned())
        }

        fn contains(&self, hash: &Blake3Hash) -> Result<bool, StoreError> {
            Ok(self
                .inner
                .lock()
                .map_err(|err| StoreError::IoError(err.to_string()))?
                .contains_key(hash))
        }
    }

    #[test]
    fn corrupted_payload_fails_hash_round_trip() {
        let mut chain = ReceiptChain::new([0u8; 32], Box::new(CorruptingStore::default()));
        let hash = chain.append_with_payload(activation_receipt()).unwrap();

        let err = project_principal_activation_identity(&chain, hash).unwrap_err();
        assert!(matches!(
            err,
            PrincipalIdentityProjectionError::Chain(ChainError::PayloadDecode(
                DecodeError::HashMismatch { .. }
            ))
        ));
    }
}
