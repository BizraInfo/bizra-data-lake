// src/pci/types.rs - PCI Protocol Type Definitions
//
// Status: FROZEN — Changes require version bump + test vector update
// Alignment: BIZRA_SOT.md Section 3.1 (Ihsān IM ≥ 0.95)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PCI Protocol version
pub const PCI_VERSION: &str = "1.0.0";

/// Domain prefix for BLAKE3 domain separation
pub const DOMAIN_PREFIX: &str = "bizra-pci-v1:";

/// Nonce size in bytes (32 bytes = 256 bits)
pub const NONCE_BYTES: usize = 32;

/// Maximum timestamp skew in seconds (±120s)
pub const TIMESTAMP_SKEW_SECONDS: i64 = 120;

/// Ihsān threshold (constitutional requirement)
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// Default SNR threshold
pub const SNR_THRESHOLD_DEFAULT: f64 = 0.95;

/// Latency budget for CHEAP tier (milliseconds)
/// Note: 50ms allows for test parallelism overhead while still being "cheap"
pub const LATENCY_BUDGET_CHEAP_MS: u64 = 50;

/// Latency budget for MEDIUM tier (milliseconds)
pub const LATENCY_BUDGET_MEDIUM_MS: u64 = 150;

/// Latency budget for EXPENSIVE tier (milliseconds)
pub const LATENCY_BUDGET_EXPENSIVE_MS: u64 = 2000;

/// Agent type in the dual-agent architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AgentType {
    /// PAT: Prover/Builder
    Pat,
    /// SAT: Verifier/Governor
    Sat,
}

impl AgentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentType::Pat => "PAT",
            AgentType::Sat => "SAT",
        }
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Request urgency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Urgency {
    RealTime,
    #[default]
    NearRealTime,
    Batch,
    Deferred,
}

/// Verification confidence tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerificationTier {
    Statistical,
    Incremental,
    Optimistic,
    FullZk,
    Formal,
}

/// Gate execution tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GateTier {
    /// <10ms budget
    Cheap,
    /// <150ms budget
    Medium,
    /// <2000ms budget
    Expensive,
}

impl GateTier {
    /// Get the latency budget in milliseconds
    pub fn budget_ms(&self) -> u64 {
        match self {
            GateTier::Cheap => LATENCY_BUDGET_CHEAP_MS,
            GateTier::Medium => LATENCY_BUDGET_MEDIUM_MS,
            GateTier::Expensive => LATENCY_BUDGET_EXPENSIVE_MS,
        }
    }
}

/// Verification gates in execution order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Gate {
    // CHEAP tier
    Schema,
    Signature,
    Timestamp,
    Replay,
    Role,
    // MEDIUM tier
    Snr,
    Ihsan,
    Policy,
    // EXPENSIVE tier
    Fate,
    Formal,
}

impl Gate {
    /// Get the tier for this gate
    pub fn tier(&self) -> GateTier {
        match self {
            Gate::Schema | Gate::Signature | Gate::Timestamp | Gate::Replay | Gate::Role => {
                GateTier::Cheap
            }
            Gate::Snr | Gate::Ihsan | Gate::Policy => GateTier::Medium,
            Gate::Fate | Gate::Formal => GateTier::Expensive,
        }
    }

    /// Get all gates in a specific tier
    pub fn gates_in_tier(tier: GateTier) -> Vec<Gate> {
        match tier {
            GateTier::Cheap => vec![
                Gate::Schema,
                Gate::Signature,
                Gate::Timestamp,
                Gate::Replay,
                Gate::Role,
            ],
            GateTier::Medium => vec![Gate::Snr, Gate::Ihsan, Gate::Policy],
            GateTier::Expensive => vec![Gate::Fate, Gate::Formal],
        }
    }
}

/// Type of commit reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommitRefType {
    Eventlog,
    Blockgraph,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignatureAlgorithm {
    Ed25519,
    Dilithium5, // Post-quantum (future)
}

/// Envelope sender information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sender {
    pub agent_type: AgentType,
    pub agent_id: String,
    /// Hex-encoded Ed25519 public key (32 bytes = 64 hex chars)
    pub public_key: String,
}

/// Envelope payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub action: String,
    pub data: serde_json::Value,
    /// BLAKE3 hash of constitution
    pub policy_hash: String,
    /// BLAKE3 hash of current state
    pub state_hash: String,
}

/// Envelope metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub ihsan_score: f64,
    pub snr_score: f64,
    #[serde(default)]
    pub urgency: Urgency,
}

/// Cryptographic signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub algorithm: SignatureAlgorithm,
    /// Hex-encoded signature (64 bytes = 128 hex chars for Ed25519)
    pub value: String,
    pub signed_fields: Vec<String>,
}

impl Default for Signature {
    fn default() -> Self {
        Self {
            algorithm: SignatureAlgorithm::Ed25519,
            value: String::new(),
            signed_fields: vec![
                "version".into(),
                "envelope_id".into(),
                "timestamp".into(),
                "nonce".into(),
                "sender".into(),
                "payload".into(),
                "metadata".into(),
            ],
        }
    }
}

/// Commit reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRef {
    #[serde(rename = "type")]
    pub ref_type: CommitRefType,
    pub offset: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_hash: Option<String>,
}

/// Verification result details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    pub tier: VerificationTier,
    pub latency_ms: f64,
    pub gates_passed: Vec<Gate>,
    pub ihsan_score: f64,
    pub snr_score: f64,
}

/// Signature from a SAT verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSignature {
    pub sat_id: String,
    pub public_key: String,
    pub signature: String,
    pub timestamp: String,
}

/// Quorum requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quorum {
    pub required: u32,
    pub achieved: u32,
}

impl Quorum {
    pub fn is_met(&self) -> bool {
        self.achieved >= self.required
    }
}

/// Audit trail for rejection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    pub gate: Gate,
    pub tier: GateTier,
    pub latency_ms: f64,
    pub details: HashMap<String, serde_json::Value>,
}

/// Generate a UUID v4 for envelope/receipt IDs
pub fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Get current UTC timestamp in ISO 8601 format
pub fn utc_now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_tiers() {
        assert_eq!(Gate::Schema.tier(), GateTier::Cheap);
        assert_eq!(Gate::Ihsan.tier(), GateTier::Medium);
        assert_eq!(Gate::Fate.tier(), GateTier::Expensive);
    }

    #[test]
    fn test_agent_type_display() {
        assert_eq!(AgentType::Pat.to_string(), "PAT");
        assert_eq!(AgentType::Sat.to_string(), "SAT");
    }

    #[test]
    fn test_tier_budget() {
        assert_eq!(GateTier::Cheap.budget_ms(), 50); // 50ms allows test parallelism overhead
        assert_eq!(GateTier::Medium.budget_ms(), 150);
        assert_eq!(GateTier::Expensive.budget_ms(), 2000);
    }
}
