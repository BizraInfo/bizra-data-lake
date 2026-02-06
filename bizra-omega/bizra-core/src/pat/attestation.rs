//! Agent Action Attestation System
//!
//! Tracks provenance and ensures Standing on Giants protocol compliance.

use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::SATRole;

// =============================================================================
// ACTION ATTESTATION
// =============================================================================

/// ActionAttestation — Immutable record of agent activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAttestation {
    /// Attestation ID
    pub id: Uuid,

    /// Agent that performed the action
    pub agent_id: Uuid,

    /// Action type
    pub action_type: ActionType,

    /// Action description
    pub description: String,

    /// Input data hash (if any)
    #[serde(with = "hex_array_32")]
    pub input_hash: [u8; 32],

    /// Output data hash (if any)
    #[serde(with = "hex_array_32")]
    pub output_hash: [u8; 32],

    /// Giants cited for this action
    pub giants_cited: Vec<GiantCitation>,

    /// Provenance records
    pub provenance: Vec<ProvenanceEntry>,

    /// Action timestamp
    pub timestamp: DateTime<Utc>,

    /// Ihsan score at time of action
    pub ihsan_score: f64,

    /// Attestation hash
    #[serde(with = "hex_array_32")]
    pub attestation_hash: [u8; 32],
}

// Serde helper for 32-byte hex arrays
mod hex_array_32 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 64 {
            return Err(serde::de::Error::custom("expected 64 hex characters"));
        }
        let bytes: Result<Vec<u8>, _> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect();
        let bytes = bytes.map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid length"))
    }
}

/// Types of actions agents can perform
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    /// Reasoning/inference
    Reason,
    /// Code execution
    Execute,
    /// Data retrieval
    Retrieve,
    /// Data storage
    Store,
    /// Validation
    Validate,
    /// Communication
    Communicate,
    /// Travel between places
    Travel,
    /// Meeting with another agent
    Meet,
    /// Consensus participation
    Consensus,
    /// Pool service provision
    PoolService,
}

/// A citation of a giant's work in an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiantCitation {
    /// Giant's name
    pub giant_name: String,
    /// Specific contribution used
    pub contribution: String,
    /// How it was applied
    pub application: String,
}

/// A provenance entry for an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    /// Source type
    pub source_type: ProvenanceSource,
    /// Source identifier
    pub source_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Hash of the sourced content
    #[serde(with = "hex_array_32")]
    pub content_hash: [u8; 32],
}

/// Types of provenance sources
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSource {
    /// Knowledge base
    KnowledgeBase,
    /// External API
    ExternalApi,
    /// Another agent
    Agent,
    /// User input
    UserInput,
    /// Internal computation
    InternalComputation,
    /// Pool resource
    PoolResource,
}

impl ActionAttestation {
    /// Create a new action attestation
    pub fn new(
        agent_id: Uuid,
        action_type: ActionType,
        description: &str,
        input_hash: [u8; 32],
        output_hash: [u8; 32],
        giants_cited: Vec<GiantCitation>,
        ihsan_score: f64,
    ) -> Self {
        let id = Uuid::new_v4();
        let timestamp = Utc::now();

        let mut attestation = Self {
            id,
            agent_id,
            action_type,
            description: description.to_string(),
            input_hash,
            output_hash,
            giants_cited,
            provenance: Vec::new(),
            timestamp,
            ihsan_score,
            attestation_hash: [0u8; 32],
        };

        attestation.attestation_hash = attestation.calculate_hash();
        attestation
    }

    /// Calculate the attestation hash
    fn calculate_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-action-attestation-v1:");
        hasher.update(self.id.as_bytes());
        hasher.update(self.agent_id.as_bytes());
        hasher.update(self.description.as_bytes());
        hasher.update(&self.input_hash);
        hasher.update(&self.output_hash);
        for citation in &self.giants_cited {
            hasher.update(citation.giant_name.as_bytes());
            hasher.update(citation.contribution.as_bytes());
        }
        hasher.update(&self.ihsan_score.to_le_bytes());
        hasher.update(self.timestamp.to_rfc3339().as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Add a provenance entry
    pub fn add_provenance(&mut self, entry: ProvenanceEntry) {
        self.provenance.push(entry);
        self.attestation_hash = self.calculate_hash();
    }

    /// Verify attestation integrity
    pub fn verify(&self) -> bool {
        // Must have at least one giant citation (Standing on Giants protocol)
        if self.giants_cited.is_empty() {
            return false;
        }
        // Verify hash
        self.attestation_hash == self.calculate_hash()
    }

    /// Format as audit log entry
    pub fn format_audit_entry(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "[{}] Agent {} performed {:?}: {}\n",
            self.timestamp.to_rfc3339(),
            self.agent_id,
            self.action_type,
            self.description
        ));
        s.push_str(&format!("  Ihsan Score: {:.3}\n", self.ihsan_score));
        s.push_str("  Giants Cited:\n");
        for citation in &self.giants_cited {
            s.push_str(&format!(
                "    - {} ({}): {}\n",
                citation.giant_name, citation.contribution, citation.application
            ));
        }
        if !self.provenance.is_empty() {
            s.push_str("  Provenance:\n");
            for entry in &self.provenance {
                s.push_str(&format!(
                    "    - {:?} from {}\n",
                    entry.source_type, entry.source_id
                ));
            }
        }
        s
    }
}

// =============================================================================
// ATTESTATION REGISTRY
// =============================================================================

/// AttestationRegistry — Stores and queries action attestations
#[derive(Debug, Default)]
pub struct AttestationRegistry {
    attestations: Vec<ActionAttestation>,
}

impl AttestationRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            attestations: Vec::new(),
        }
    }

    /// Record an attestation
    pub fn record(&mut self, attestation: ActionAttestation) -> bool {
        if !attestation.verify() {
            return false;
        }
        self.attestations.push(attestation);
        true
    }

    /// Get attestations for an agent
    pub fn for_agent(&self, agent_id: Uuid) -> Vec<&ActionAttestation> {
        self.attestations
            .iter()
            .filter(|a| a.agent_id == agent_id)
            .collect()
    }

    /// Get attestations by action type
    pub fn by_action_type(&self, action_type: ActionType) -> Vec<&ActionAttestation> {
        self.attestations
            .iter()
            .filter(|a| a.action_type == action_type)
            .collect()
    }

    /// Get attestations citing a giant
    pub fn citing_giant(&self, giant_name: &str) -> Vec<&ActionAttestation> {
        self.attestations
            .iter()
            .filter(|a| {
                a.giants_cited.iter().any(|c| {
                    c.giant_name
                        .to_lowercase()
                        .contains(&giant_name.to_lowercase())
                })
            })
            .collect()
    }

    /// Calculate impact score for an agent
    pub fn calculate_impact_score(&self, agent_id: Uuid) -> u64 {
        let agent_attestations = self.for_agent(agent_id);
        let mut score = 0u64;

        for attestation in agent_attestations {
            // Base score per action type
            let base = match attestation.action_type {
                ActionType::Reason => 10,
                ActionType::Execute => 8,
                ActionType::Validate => 12,
                ActionType::Consensus => 15,
                ActionType::PoolService => 20,
                _ => 5,
            };

            // Bonus for Ihsan score
            let ihsan_bonus = if attestation.ihsan_score >= 0.98 {
                2
            } else if attestation.ihsan_score >= 0.95 {
                1
            } else {
                0
            };

            // Bonus for giant citations
            let citation_bonus = attestation.giants_cited.len() as u64;

            score += base + ihsan_bonus + citation_bonus;
        }

        score
    }

    /// Get total attestation count
    pub fn count(&self) -> usize {
        self.attestations.len()
    }

    /// Export as audit log
    pub fn export_audit_log(&self) -> String {
        let mut log = String::new();
        log.push_str("=== BIZRA Action Attestation Audit Log ===\n\n");
        for attestation in &self.attestations {
            log.push_str(&attestation.format_audit_entry());
            log.push('\n');
        }
        log
    }
}

// =============================================================================
// POOL USAGE TRACKING
// =============================================================================

/// PoolUsageRecord — Tracks SAT agent usage in the resource pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolUsageRecord {
    /// Record ID
    pub id: Uuid,

    /// SAT agent that provided the service
    pub provider_agent_id: Uuid,

    /// Consumer agent/node
    pub consumer_id: String,

    /// Service type
    pub service_type: SATRole,

    /// Usage duration in milliseconds
    pub duration_ms: u64,

    /// Resources consumed
    pub resources_consumed: ResourceUsage,

    /// Value earned (tokens)
    pub value_earned: u64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Related attestation
    pub attestation_id: Uuid,
}

/// Resource usage breakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU milliseconds used
    pub cpu_ms: u64,
    /// Memory bytes peak
    pub memory_bytes_peak: u64,
    /// Network bytes transferred
    pub network_bytes: u64,
    /// Storage bytes used
    pub storage_bytes: u64,
    /// Inference tokens used
    pub inference_tokens: u64,
}

impl PoolUsageRecord {
    /// Calculate the earnings based on resource usage
    pub fn calculate_earnings(&self, base_rate: u64) -> u64 {
        let duration_factor = (self.duration_ms as f64 / 1000.0).ceil() as u64;

        // Weight different resources
        let cpu_value = self.resources_consumed.cpu_ms / 100;
        let memory_value = self.resources_consumed.memory_bytes_peak / (1024 * 1024);
        let network_value = self.resources_consumed.network_bytes / (1024 * 1024);
        let inference_value = self.resources_consumed.inference_tokens / 100;

        let total_resource_value = cpu_value + memory_value + network_value + inference_value;

        base_rate * duration_factor + total_resource_value
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_attestation_creation() {
        let attestation = ActionAttestation::new(
            Uuid::new_v4(),
            ActionType::Reason,
            "Analyzed data pattern",
            [0u8; 32],
            [1u8; 32],
            vec![GiantCitation {
                giant_name: "Shannon".to_string(),
                contribution: "Information Theory".to_string(),
                application: "Entropy calculation".to_string(),
            }],
            0.97,
        );

        assert!(attestation.verify());
        assert!(!attestation.giants_cited.is_empty());
    }

    #[test]
    fn test_attestation_without_giants_fails() {
        let attestation = ActionAttestation::new(
            Uuid::new_v4(),
            ActionType::Execute,
            "Executed task",
            [0u8; 32],
            [1u8; 32],
            vec![], // No giants cited
            0.95,
        );

        assert!(!attestation.verify()); // Should fail
    }

    #[test]
    fn test_attestation_registry() {
        let mut registry = AttestationRegistry::new();
        let agent_id = Uuid::new_v4();

        let attestation = ActionAttestation::new(
            agent_id,
            ActionType::Validate,
            "Validated transaction",
            [0u8; 32],
            [1u8; 32],
            vec![GiantCitation {
                giant_name: "Lamport".to_string(),
                contribution: "BFT".to_string(),
                application: "Consensus validation".to_string(),
            }],
            0.96,
        );

        assert!(registry.record(attestation));
        assert_eq!(registry.count(), 1);
        assert_eq!(registry.for_agent(agent_id).len(), 1);
    }

    #[test]
    fn test_impact_score_calculation() {
        let mut registry = AttestationRegistry::new();
        let agent_id = Uuid::new_v4();

        // Add multiple attestations
        for _ in 0..5 {
            let attestation = ActionAttestation::new(
                agent_id,
                ActionType::Consensus,
                "Participated in consensus",
                [0u8; 32],
                [1u8; 32],
                vec![
                    GiantCitation {
                        giant_name: "Lamport".to_string(),
                        contribution: "BFT".to_string(),
                        application: "Consensus".to_string(),
                    },
                    GiantCitation {
                        giant_name: "Nakamoto".to_string(),
                        contribution: "PoW".to_string(),
                        application: "Validation".to_string(),
                    },
                ],
                0.98,
            );
            registry.record(attestation);
        }

        let score = registry.calculate_impact_score(agent_id);
        assert!(score > 0);
    }

    #[test]
    fn test_citing_giant_search() {
        let mut registry = AttestationRegistry::new();

        let attestation = ActionAttestation::new(
            Uuid::new_v4(),
            ActionType::Reason,
            "Applied information theory",
            [0u8; 32],
            [1u8; 32],
            vec![GiantCitation {
                giant_name: "Claude Shannon".to_string(),
                contribution: "Information Theory".to_string(),
                application: "SNR calculation".to_string(),
            }],
            0.95,
        );
        registry.record(attestation);

        let shannon_citations = registry.citing_giant("shannon");
        assert_eq!(shannon_citations.len(), 1);

        let missing = registry.citing_giant("einstein");
        assert_eq!(missing.len(), 0);
    }
}
