//! PCI Reject Codes

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RejectCode {
    Success = 0,
    RejectSchema = 1,
    RejectMissingField = 2,
    RejectSignature = 10,
    RejectExpired = 11,
    RejectReplay = 12,
    RejectHashMismatch = 15,
    RejectGateSchema = 20,
    RejectGateSig = 21,
    RejectGateRate = 22,
    RejectGateSNR = 23,
    RejectGateFATE = 24,
    RejectGateIhsan = 25,
    RejectQuota = 40,
    RejectTimeout = 42,
    RejectInternal = 50,
}

impl RejectCode {
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

    pub fn is_success(&self) -> bool { matches!(self, Self::Success) }

    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RejectGateRate | Self::RejectQuota | Self::RejectTimeout)
    }
}

impl fmt::Display for RejectCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PCI-{:02}: {}", *self as u8, self.description())
    }
}

impl std::error::Error for RejectCode {}
