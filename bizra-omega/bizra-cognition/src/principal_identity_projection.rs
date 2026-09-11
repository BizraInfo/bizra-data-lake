//! Hash-bound, authority-neutral projection of a principal activation receipt.
//!
//! This module only reads the active receipt chain. It does not create an
//! identity, grant authority, or persist a new truth source.

use crate::principal_activation::PrincipalActivationReceipt;
use crate::receipts::{Blake3Hash, ChainError, ReceiptChain, ReceiptKind};

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
            Self::Chain(error) => write!(f, "receipt payload verification failed: {:?}", error),
        }
    }
}

impl std::error::Error for PrincipalIdentityProjectionError {}

impl From<ChainError> for PrincipalIdentityProjectionError {
    fn from(value: ChainError) -> Self {
        Self::Chain(value)
    }
}

/// Decode one chain-sealed principal activation payload after checking its
/// receipt kind and hash round-trip. This function is read-only by contract.
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
