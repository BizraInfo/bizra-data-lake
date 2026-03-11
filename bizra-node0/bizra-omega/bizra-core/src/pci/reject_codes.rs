//! PCI Reject Codes

use serde::{Deserialize, Serialize};
use std::fmt;

/// Protocol-level result code for PCI envelope processing.
///
/// Each code maps to a `u8` discriminant for wire-format compactness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RejectCode {
    /// Operation completed successfully.
    Success = 0,
    /// Payload failed JSON schema validation.
    RejectSchema = 1,
    /// A required field is absent from the envelope.
    RejectMissingField = 2,
    /// Ed25519 signature verification failed.
    RejectSignature = 10,
    /// Envelope TTL has elapsed.
    RejectExpired = 11,
    /// Duplicate envelope ID (replay attack).
    RejectReplay = 12,
    /// Payload hash does not match `content_hash`.
    RejectHashMismatch = 15,
    /// Schema gate rejected the payload.
    RejectGateSchema = 20,
    /// Signature gate rejected the payload.
    RejectGateSig = 21,
    /// Rate-limit gate exceeded.
    RejectGateRate = 22,
    /// SNR score below constitutional threshold.
    RejectGateSNR = 23,
    /// FATE (Fairness, Accountability, Transparency, Ethics) check failed.
    RejectGateFATE = 24,
    /// Ihsan excellence threshold not met.
    RejectGateIhsan = 25,
    /// Node has exhausted its processing quota.
    RejectQuota = 40,
    /// Processing deadline exceeded.
    RejectTimeout = 42,
    /// Unrecoverable internal error.
    RejectInternal = 50,
}

impl RejectCode {
    /// Returns a human-readable description of this reject code.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::RejectSchema => "Invalid JSON schema",
            Self::RejectMissingField => "Missing required field",
            Self::RejectSignature => "Invalid signature",
            Self::RejectExpired => "Envelope expired",
            Self::RejectReplay => "Replay attack detected",
            Self::RejectHashMismatch => "Content hash mismatch",
            Self::RejectGateSchema => "Schema gate failed",
            Self::RejectGateSig => "Signature gate failed",
            Self::RejectGateRate => "Rate limit exceeded",
            Self::RejectGateSNR => "SNR below threshold",
            Self::RejectGateFATE => "FATE check failed",
            Self::RejectGateIhsan => "Ihsan threshold not met",
            Self::RejectQuota => "Insufficient quota",
            Self::RejectTimeout => "Request timeout",
            Self::RejectInternal => "Internal error",
        }
    }

    /// Returns `true` if this code represents a successful operation.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` if the failure is transient and the request may be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RejectGateRate | Self::RejectQuota | Self::RejectTimeout
        )
    }
}

impl fmt::Display for RejectCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PCI-{:02}: {}", *self as u8, self.description())
    }
}

impl std::error::Error for RejectCode {}
