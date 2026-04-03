// src/pci/reject_codes.rs - PCI Protocol Rejection Codes
//
// Status: FROZEN — Changes require version bump + test vector update
// WARNING: These codes are part of the wire protocol.
// Never change existing code values. Only append new codes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{AuditTrail, Gate, GateTier};

/// Stable numeric rejection codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RejectCode {
    /// Success (not a rejection)
    Success = 0,

    // CHEAP tier rejections (1-5)
    RejectSchema = 1,
    RejectSignature = 2,
    RejectNonceReplay = 3,
    RejectTimestampStale = 4,
    RejectTimestampFuture = 5,

    // MEDIUM tier rejections (6-10)
    RejectIhsanBelowMin = 6,
    RejectSnrBelowMin = 7,
    RejectBudgetExceeded = 8,
    RejectPolicyMismatch = 9,
    RejectStateMismatch = 10,

    // ROLE/QUORUM rejections (11-12)
    RejectRoleViolation = 11,
    RejectQuorumFailed = 12,

    // EXPENSIVE tier rejections (13-14)
    RejectFateViolation = 13,
    RejectInvariantFailed = 14,

    // Rate limiting (15)
    RejectRateLimited = 15,

    // Internal error (99)
    RejectInternalError = 99,
}

impl RejectCode {
    pub fn is_success(&self) -> bool {
        matches!(self, RejectCode::Success)
    }

    pub fn is_rejection(&self) -> bool {
        !self.is_success()
    }

    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    pub fn description(&self) -> &'static str {
        match self {
            RejectCode::Success => "Operation completed successfully",
            RejectCode::RejectSchema => "Envelope failed JSON schema validation",
            RejectCode::RejectSignature => "Cryptographic signature invalid",
            RejectCode::RejectNonceReplay => "Nonce already seen within TTL window",
            RejectCode::RejectTimestampStale => "Timestamp outside acceptable skew (past)",
            RejectCode::RejectTimestampFuture => "Timestamp too far in future",
            RejectCode::RejectIhsanBelowMin => "Ihsān score below 0.95 threshold",
            RejectCode::RejectSnrBelowMin => "SNR score below tier threshold",
            RejectCode::RejectBudgetExceeded => "Verification latency exceeded tier budget",
            RejectCode::RejectPolicyMismatch => "policy_hash doesn't match current constitution",
            RejectCode::RejectStateMismatch => "state_hash doesn't match expected state",
            RejectCode::RejectRoleViolation => "Agent attempted unauthorized action",
            RejectCode::RejectQuorumFailed => "Insufficient verifier signatures",
            RejectCode::RejectFateViolation => "FATE invariant check failed",
            RejectCode::RejectInvariantFailed => "Formal invariant verification failed",
            RejectCode::RejectRateLimited => "Too many requests from sender",
            RejectCode::RejectInternalError => "Unexpected internal error (fail-closed)",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            RejectCode::Success => "SUCCESS",
            RejectCode::RejectSchema => "REJECT_SCHEMA",
            RejectCode::RejectSignature => "REJECT_SIGNATURE",
            RejectCode::RejectNonceReplay => "REJECT_NONCE_REPLAY",
            RejectCode::RejectTimestampStale => "REJECT_TIMESTAMP_STALE",
            RejectCode::RejectTimestampFuture => "REJECT_TIMESTAMP_FUTURE",
            RejectCode::RejectIhsanBelowMin => "REJECT_IHSAN_BELOW_MIN",
            RejectCode::RejectSnrBelowMin => "REJECT_SNR_BELOW_MIN",
            RejectCode::RejectBudgetExceeded => "REJECT_BUDGET_EXCEEDED",
            RejectCode::RejectPolicyMismatch => "REJECT_POLICY_MISMATCH",
            RejectCode::RejectStateMismatch => "REJECT_STATE_MISMATCH",
            RejectCode::RejectRoleViolation => "REJECT_ROLE_VIOLATION",
            RejectCode::RejectQuorumFailed => "REJECT_QUORUM_FAILED",
            RejectCode::RejectFateViolation => "REJECT_FATE_VIOLATION",
            RejectCode::RejectInvariantFailed => "REJECT_INVARIANT_FAILED",
            RejectCode::RejectRateLimited => "REJECT_RATE_LIMITED",
            RejectCode::RejectInternalError => "REJECT_INTERNAL_ERROR",
        }
    }
}

impl std::fmt::Display for RejectCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.name(), self.as_u8())
    }
}

/// Structured rejection response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionResponse {
    pub rejected: bool,
    pub code: u8,
    pub name: String,
    pub message: String,
    pub envelope_digest: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_trail: Option<AuditTrail>,
}

impl RejectionResponse {
    /// Create a success response
    pub fn success(envelope_digest: String, timestamp: String) -> Self {
        Self {
            rejected: false,
            code: RejectCode::Success.as_u8(),
            name: RejectCode::Success.name().to_string(),
            message: "Operation completed successfully".to_string(),
            envelope_digest,
            timestamp,
            audit_trail: None,
        }
    }

    /// Create a rejection response
    pub fn rejection(
        code: RejectCode,
        message: String,
        envelope_digest: String,
        timestamp: String,
        audit_trail: Option<AuditTrail>,
    ) -> Self {
        Self {
            rejected: true,
            code: code.as_u8(),
            name: code.name().to_string(),
            message,
            envelope_digest,
            timestamp,
            audit_trail,
        }
    }
}

// =============================================================================
// REJECTION HELPERS
// =============================================================================

pub fn reject_schema(envelope_digest: &str, timestamp: &str, details: &str) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectSchema,
        format!("Schema validation failed: {}", details),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Schema,
            tier: GateTier::Cheap,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert(
                    "error".to_string(),
                    serde_json::Value::String(details.to_string()),
                );
                m
            },
        }),
    )
}

pub fn reject_signature(envelope_digest: &str, timestamp: &str) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectSignature,
        "Cryptographic signature verification failed".to_string(),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Signature,
            tier: GateTier::Cheap,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert(
                    "error".to_string(),
                    serde_json::Value::String("Invalid Ed25519 signature".to_string()),
                );
                m
            },
        }),
    )
}

pub fn reject_ihsan(
    envelope_digest: &str,
    timestamp: &str,
    score: f64,
    threshold: f64,
) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectIhsanBelowMin,
        format!("Ihsān score {:.2} < required {:.2}", score, threshold),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Ihsan,
            tier: GateTier::Medium,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert("score".to_string(), serde_json::json!(score));
                m.insert("threshold".to_string(), serde_json::json!(threshold));
                m
            },
        }),
    )
}

pub fn reject_snr(
    envelope_digest: &str,
    timestamp: &str,
    score: f64,
    threshold: f64,
) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectSnrBelowMin,
        format!("SNR score {:.2} < tier threshold {:.2}", score, threshold),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Snr,
            tier: GateTier::Medium,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert("score".to_string(), serde_json::json!(score));
                m.insert("threshold".to_string(), serde_json::json!(threshold));
                m
            },
        }),
    )
}

pub fn reject_replay(envelope_digest: &str, timestamp: &str, nonce: &str) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectNonceReplay,
        "Nonce already seen within TTL window (replay attack detected)".to_string(),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Replay,
            tier: GateTier::Cheap,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                // Truncate nonce for log safety
                let truncated = if nonce.len() > 16 {
                    format!("{}...", &nonce[..16])
                } else {
                    nonce.to_string()
                };
                m.insert("nonce".to_string(), serde_json::Value::String(truncated));
                m
            },
        }),
    )
}

pub fn reject_timestamp_stale(
    envelope_digest: &str,
    timestamp: &str,
    envelope_ts: &str,
    skew_seconds: f64,
) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectTimestampStale,
        format!(
            "Timestamp {} is {:.1}s in the past (max 120s)",
            envelope_ts,
            skew_seconds.abs()
        ),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Timestamp,
            tier: GateTier::Cheap,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert(
                    "envelope_timestamp".to_string(),
                    serde_json::Value::String(envelope_ts.to_string()),
                );
                m.insert("skew_seconds".to_string(), serde_json::json!(skew_seconds));
                m
            },
        }),
    )
}

pub fn reject_role_violation(
    envelope_digest: &str,
    timestamp: &str,
    agent_type: &str,
    action: &str,
) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectRoleViolation,
        format!("{} agent cannot perform action: {}", agent_type, action),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Role,
            tier: GateTier::Cheap,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert(
                    "agent_type".to_string(),
                    serde_json::Value::String(agent_type.to_string()),
                );
                m.insert(
                    "action".to_string(),
                    serde_json::Value::String(action.to_string()),
                );
                m
            },
        }),
    )
}

pub fn reject_internal_error(
    envelope_digest: &str,
    timestamp: &str,
    error: &str,
) -> RejectionResponse {
    RejectionResponse::rejection(
        RejectCode::RejectInternalError,
        format!("Internal error (fail-closed): {}", error),
        envelope_digest.to_string(),
        timestamp.to_string(),
        Some(AuditTrail {
            gate: Gate::Schema,
            tier: GateTier::Cheap,
            latency_ms: 0.0,
            details: {
                let mut m = HashMap::new();
                m.insert(
                    "error".to_string(),
                    serde_json::Value::String(error.to_string()),
                );
                m
            },
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_code_values() {
        assert_eq!(RejectCode::Success.as_u8(), 0);
        assert_eq!(RejectCode::RejectSchema.as_u8(), 1);
        assert_eq!(RejectCode::RejectSignature.as_u8(), 2);
        assert_eq!(RejectCode::RejectIhsanBelowMin.as_u8(), 6);
        assert_eq!(RejectCode::RejectInternalError.as_u8(), 99);
    }

    #[test]
    fn test_reject_code_display() {
        assert_eq!(format!("{}", RejectCode::Success), "SUCCESS(0)");
        assert_eq!(
            format!("{}", RejectCode::RejectIhsanBelowMin),
            "REJECT_IHSAN_BELOW_MIN(6)"
        );
    }

    #[test]
    fn test_success_response() {
        let resp =
            RejectionResponse::success("digest123".to_string(), "2025-01-08T12:00:00Z".to_string());
        assert!(!resp.rejected);
        assert_eq!(resp.code, 0);
    }

    #[test]
    fn test_rejection_response() {
        let resp = reject_ihsan("digest123", "2025-01-08T12:00:00Z", 0.80, 0.95);
        assert!(resp.rejected);
        assert_eq!(resp.code, 6);
        assert!(resp.message.contains("0.80"));
        assert!(resp.audit_trail.is_some());
    }
}
