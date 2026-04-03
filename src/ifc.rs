// src/ifc.rs - Information Flow Control
// Systematic taint tracking for the dual-agentic pipeline

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Secrecy levels for data classification (least → most restrictive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecrecyLevel {
    Public,       // Can be shared externally
    Internal,     // BIZRA-internal only
    Confidential, // Restricted to specific agents
    Secret,       // Highest sensitivity
}

impl fmt::Display for SecrecyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecrecyLevel::Public => write!(f, "Public"),
            SecrecyLevel::Internal => write!(f, "Internal"),
            SecrecyLevel::Confidential => write!(f, "Confidential"),
            SecrecyLevel::Secret => write!(f, "Secret"),
        }
    }
}

/// Integrity levels for data trust (least → most trusted)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntegrityLevel {
    Untrusted, // User input, external sources
    Validated, // Basic validation passed
    Attested,  // SAT consensus reached
    Sovereign, // Kernel-signed, highest trust
}

impl fmt::Display for IntegrityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrityLevel::Untrusted => write!(f, "Untrusted"),
            IntegrityLevel::Validated => write!(f, "Validated"),
            IntegrityLevel::Attested => write!(f, "Attested"),
            IntegrityLevel::Sovereign => write!(f, "Sovereign"),
        }
    }
}

/// Label attached to data flowing through the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintLabel {
    pub secrecy: SecrecyLevel,
    pub integrity: IntegrityLevel,
    pub source: String, // Origin agent/user ID
    pub timestamp: DateTime<Utc>,
}

impl TaintLabel {
    pub fn new(secrecy: SecrecyLevel, integrity: IntegrityLevel, source: String) -> Self {
        Self {
            secrecy,
            integrity,
            source,
            timestamp: Utc::now(),
        }
    }
}

/// Information flow control violations
#[derive(Error, Debug)]
pub enum IFCViolation {
    #[error("SECRECY VIOLATION: {field} flow from {from_level} → {to_level} (no declassification)")]
    SecrecyViolation {
        from_level: SecrecyLevel,
        to_level: SecrecyLevel,
        field: String,
    },

    #[error("INTEGRITY VIOLATION: {field} from {from_level} treated as {to_level}")]
    IntegrityViolation {
        from_level: IntegrityLevel,
        to_level: IntegrityLevel,
        field: String,
    },

    #[error("UNLABELED DATA: {field} crossed boundary without taint label")]
    UnlabeledData { field: String },
}

/// Audit entry for explicit declassifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintAuditEntry {
    pub timestamp: DateTime<Utc>,
    pub field: String,
    pub from_secrecy: SecrecyLevel,
    pub to_secrecy: SecrecyLevel,
    pub reason: String,
    pub actor: String,
}

/// Tracks taint labels for data fields in the pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintContext {
    labels: HashMap<String, TaintLabel>,
    default_label: TaintLabel,
    audit_log: Vec<TaintAuditEntry>,
}

impl TaintContext {
    /// Create context with Untrusted/Internal defaults
    pub fn new(source: &str) -> Self {
        Self {
            labels: HashMap::new(),
            default_label: TaintLabel::new(
                SecrecyLevel::Internal,
                IntegrityLevel::Untrusted,
                source.to_string(),
            ),
            audit_log: Vec::new(),
        }
    }

    /// Assign taint label to a field
    pub fn taint(&mut self, key: &str, label: TaintLabel) {
        self.labels.insert(key.to_string(), label);
    }

    /// Get label for a field (returns default if unlabeled)
    pub fn get_label(&self, key: &str) -> &TaintLabel {
        self.labels.get(key).unwrap_or(&self.default_label)
    }

    /// Check that data flow doesn't decrease secrecy without declassification
    pub fn check_flow(&self, from_key: &str, to_secrecy: SecrecyLevel) -> Result<(), IFCViolation> {
        let from_label = self.get_label(from_key);

        if from_label.secrecy > to_secrecy {
            return Err(IFCViolation::SecrecyViolation {
                from_level: from_label.secrecy,
                to_level: to_secrecy,
                field: from_key.to_string(),
            });
        }

        Ok(())
    }

    /// Validate that output fields don't leak sensitive data (must be Public)
    pub fn validate_output(&self, keys: &[&str]) -> Result<(), IFCViolation> {
        for key in keys {
            let label = self.get_label(key);
            if label.secrecy > SecrecyLevel::Public {
                return Err(IFCViolation::SecrecyViolation {
                    from_level: label.secrecy,
                    to_level: SecrecyLevel::Public,
                    field: key.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Promote integrity level after validation (only upward)
    pub fn promote(&mut self, key: &str, new_integrity: IntegrityLevel) -> Result<(), IFCViolation> {
        let current_label = self.get_label(key).clone();

        if new_integrity < current_label.integrity {
            return Err(IFCViolation::IntegrityViolation {
                from_level: current_label.integrity,
                to_level: new_integrity,
                field: key.to_string(),
            });
        }

        let mut promoted_label = current_label;
        promoted_label.integrity = new_integrity;
        promoted_label.timestamp = Utc::now();
        self.labels.insert(key.to_string(), promoted_label);

        Ok(())
    }

    /// Explicit declassification with audit trail
    pub fn declassify(&mut self, key: &str, new_secrecy: SecrecyLevel, reason: &str) {
        let current_label = self.get_label(key).clone();

        if new_secrecy < current_label.secrecy {
            tracing::warn!(
                field = %key,
                from = %current_label.secrecy,
                to = %new_secrecy,
                reason = %reason,
                "IFC: Explicit declassification"
            );

            self.audit_log.push(TaintAuditEntry {
                timestamp: Utc::now(),
                field: key.to_string(),
                from_secrecy: current_label.secrecy,
                to_secrecy: new_secrecy,
                reason: reason.to_string(),
                actor: current_label.source.clone(),
            });
        }

        let mut declassified_label = current_label;
        declassified_label.secrecy = new_secrecy;
        declassified_label.timestamp = Utc::now();
        self.labels.insert(key.to_string(), declassified_label);
    }

    /// Merge another context, taking MORE restrictive labels
    pub fn merge(&mut self, other: &TaintContext) {
        for (key, other_label) in &other.labels {
            let merged_label = if let Some(existing_label) = self.labels.get(key) {
                TaintLabel {
                    secrecy: existing_label.secrecy.max(other_label.secrecy),
                    integrity: existing_label.integrity.min(other_label.integrity),
                    source: existing_label.source.clone(),
                    timestamp: Utc::now(),
                }
            } else {
                other_label.clone()
            };

            self.labels.insert(key.clone(), merged_label);
        }

        // Merge audit logs
        self.audit_log.extend(other.audit_log.clone());
    }

    /// Get audit log for receipt embedding
    pub fn audit_log(&self) -> &[TaintAuditEntry] {
        &self.audit_log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secrecy_ordering() {
        assert!(SecrecyLevel::Public < SecrecyLevel::Internal);
        assert!(SecrecyLevel::Internal < SecrecyLevel::Confidential);
        assert!(SecrecyLevel::Confidential < SecrecyLevel::Secret);
    }

    #[test]
    fn test_integrity_ordering() {
        assert!(IntegrityLevel::Untrusted < IntegrityLevel::Validated);
        assert!(IntegrityLevel::Validated < IntegrityLevel::Attested);
        assert!(IntegrityLevel::Attested < IntegrityLevel::Sovereign);
    }

    #[test]
    fn test_check_flow_blocks_secret_to_public() {
        let mut ctx = TaintContext::new("test_user");
        ctx.taint(
            "sensitive_field",
            TaintLabel::new(SecrecyLevel::Secret, IntegrityLevel::Validated, "user".into()),
        );

        let result = ctx.check_flow("sensitive_field", SecrecyLevel::Public);
        assert!(result.is_err());
        assert!(matches!(result, Err(IFCViolation::SecrecyViolation { .. })));
    }

    #[test]
    fn test_check_flow_allows_public_to_secret() {
        let mut ctx = TaintContext::new("test_user");
        ctx.taint(
            "public_field",
            TaintLabel::new(SecrecyLevel::Public, IntegrityLevel::Validated, "user".into()),
        );

        let result = ctx.check_flow("public_field", SecrecyLevel::Secret);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_output_blocks_confidential() {
        let mut ctx = TaintContext::new("test_user");
        ctx.taint(
            "api_key",
            TaintLabel::new(SecrecyLevel::Confidential, IntegrityLevel::Validated, "user".into()),
        );

        let result = ctx.validate_output(&["api_key"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_promote_only_upward() {
        let mut ctx = TaintContext::new("test_user");
        ctx.taint(
            "data",
            TaintLabel::new(SecrecyLevel::Public, IntegrityLevel::Attested, "user".into()),
        );

        // Downgrade should fail
        let result = ctx.promote("data", IntegrityLevel::Validated);
        assert!(result.is_err());

        // Upgrade should succeed
        let result = ctx.promote("data", IntegrityLevel::Sovereign);
        assert!(result.is_ok());
        assert_eq!(ctx.get_label("data").integrity, IntegrityLevel::Sovereign);
    }

    #[test]
    fn test_merge_takes_more_restrictive() {
        let mut ctx1 = TaintContext::new("user1");
        ctx1.taint(
            "field",
            TaintLabel::new(SecrecyLevel::Public, IntegrityLevel::Attested, "user1".into()),
        );

        let mut ctx2 = TaintContext::new("user2");
        ctx2.taint(
            "field",
            TaintLabel::new(SecrecyLevel::Secret, IntegrityLevel::Untrusted, "user2".into()),
        );

        ctx1.merge(&ctx2);

        let merged_label = ctx1.get_label("field");
        assert_eq!(merged_label.secrecy, SecrecyLevel::Secret); // More restrictive
        assert_eq!(merged_label.integrity, IntegrityLevel::Untrusted); // Less trusted
    }

    #[test]
    fn test_declassify_creates_audit_entry() {
        let mut ctx = TaintContext::new("test_user");
        ctx.taint(
            "secret_data",
            TaintLabel::new(SecrecyLevel::Secret, IntegrityLevel::Validated, "user".into()),
        );

        ctx.declassify(
            "secret_data",
            SecrecyLevel::Public,
            "SAT approved public disclosure",
        );

        assert_eq!(ctx.get_label("secret_data").secrecy, SecrecyLevel::Public);
        assert_eq!(ctx.audit_log.len(), 1);
        assert_eq!(ctx.audit_log[0].field, "secret_data");
        assert_eq!(ctx.audit_log[0].from_secrecy, SecrecyLevel::Secret);
        assert_eq!(ctx.audit_log[0].to_secrecy, SecrecyLevel::Public);
    }
}
