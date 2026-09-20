//! Typed, chain-sealed witness for the first bounded diagnostic.
//!
//! This module defines identity only. It does not execute a diagnostic or
//! grant authority. A producer must append the receipt through the ordinary
//! governed mission path; observers verify the payload and its references.

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{
    Blake3Hash, ByteReader, DecodeError, ReceiptKind, ReceiptPayload, ReceiptPayloadDecode,
};

pub const ARTIFACT_011_ID: &str = "ARTIFACT-011";
pub const BOUNDED_DIAGNOSTIC_ACTION: &str = "bounded_diagnostic_activation";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedDiagnosticReceipt {
    pub receipt_id: Blake3Hash,
    pub artifact_id: String,
    pub action: String,
    pub mission_receipt_ref: Blake3Hash,
    pub manifest_ref: Blake3Hash,
    pub input_digest: Blake3Hash,
    pub result_digest: Blake3Hash,
    pub pre_state: Blake3Hash,
    pub post_state: Blake3Hash,
    pub timestamp_ns: u64,
    pub prev_chain: Blake3Hash,
}

impl BoundedDiagnosticReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mission_receipt_ref: Blake3Hash,
        manifest_ref: Blake3Hash,
        input_digest: Blake3Hash,
        result_digest: Blake3Hash,
        pre_state: Blake3Hash,
        post_state: Blake3Hash,
        timestamp_ns: u64,
        prev_chain: Blake3Hash,
    ) -> Self {
        let mut receipt = Self {
            receipt_id: [0u8; 32],
            artifact_id: ARTIFACT_011_ID.to_string(),
            action: BOUNDED_DIAGNOSTIC_ACTION.to_string(),
            mission_receipt_ref,
            manifest_ref,
            input_digest,
            result_digest,
            pre_state,
            post_state,
            timestamp_ns,
            prev_chain,
        };
        receipt.receipt_id = receipt.computed_id();
        receipt
    }

    fn body_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.artifact_id.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.artifact_id.as_bytes());
        bytes.extend_from_slice(&(self.action.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.action.as_bytes());
        bytes.extend_from_slice(&self.mission_receipt_ref);
        bytes.extend_from_slice(&self.manifest_ref);
        bytes.extend_from_slice(&self.input_digest);
        bytes.extend_from_slice(&self.result_digest);
        bytes.extend_from_slice(&self.pre_state);
        bytes.extend_from_slice(&self.post_state);
        bytes.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        bytes.extend_from_slice(&self.prev_chain);
        bytes
    }

    fn computed_id(&self) -> Blake3Hash {
        blake3_domain("bizra-bounded-diagnostic-receipt-v1", &self.body_bytes())
    }

    pub fn has_exact_identity(&self) -> bool {
        self.artifact_id == ARTIFACT_011_ID
            && self.action == BOUNDED_DIAGNOSTIC_ACTION
            && self.receipt_id == self.computed_id()
    }
}

impl ReceiptPayload for BoundedDiagnosticReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::BoundedDiagnostic
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let body = self.body_bytes();
        let mut bytes = Vec::with_capacity(32 + body.len());
        bytes.extend_from_slice(&self.receipt_id);
        bytes.extend_from_slice(&body);
        bytes
    }

    fn hash(&self) -> Blake3Hash {
        self.receipt_id
    }

    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }
}

impl ReceiptPayloadDecode for BoundedDiagnosticReceipt {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut reader = ByteReader::new(bytes);
        let receipt_id = reader.read_hash()?;
        let artifact_id = std::str::from_utf8(reader.read_length_prefixed()?)
            .map_err(|e| DecodeError::Utf8(e.to_string()))?
            .to_string();
        let action = std::str::from_utf8(reader.read_length_prefixed()?)
            .map_err(|e| DecodeError::Utf8(e.to_string()))?
            .to_string();
        let mission_receipt_ref = reader.read_hash()?;
        let manifest_ref = reader.read_hash()?;
        let input_digest = reader.read_hash()?;
        let result_digest = reader.read_hash()?;
        let pre_state = reader.read_hash()?;
        let post_state = reader.read_hash()?;
        let timestamp_ns = reader.read_u64()?;
        let prev_chain = reader.read_hash()?;
        if reader.remaining() != 0 {
            return Err(DecodeError::TrailingBytes {
                count: reader.remaining(),
            });
        }

        let receipt = Self {
            receipt_id,
            artifact_id,
            action,
            mission_receipt_ref,
            manifest_ref,
            input_digest,
            result_digest,
            pre_state,
            post_state,
            timestamp_ns,
            prev_chain,
        };
        let computed = receipt.computed_id();
        if computed != receipt.receipt_id {
            return Err(DecodeError::HashMismatch {
                expected: receipt.receipt_id,
                computed,
            });
        }
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_payload_round_trips_and_binds_identity() {
        let receipt = BoundedDiagnosticReceipt::new(
            [1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32], 7, [8; 32],
        );
        let decoded = BoundedDiagnosticReceipt::from_canonical_bytes(&receipt.canonical_bytes())
            .expect("canonical payload must decode");
        assert_eq!(decoded, receipt);
        assert!(decoded.has_exact_identity());
    }

    #[test]
    fn tampered_identity_fails_hash_binding() {
        let receipt = BoundedDiagnosticReceipt::new(
            [1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32], 7, [8; 32],
        );
        let mut bytes = receipt.canonical_bytes();
        bytes[36] ^= 1;
        assert!(matches!(
            BoundedDiagnosticReceipt::from_canonical_bytes(&bytes),
            Err(DecodeError::HashMismatch { .. })
        ));
    }
}
