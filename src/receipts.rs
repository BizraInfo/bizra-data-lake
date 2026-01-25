// src/receipts.rs - Rejection and Execution Receipts
// Machine-verifiable evidence of SAT decisions and FATE escalations
//
// PERSISTENCE: Uses Redis (Synapse) for durable receipt storage + filesystem

use crate::fate::{Escalation, EscalationLevel};
use crate::sat::RejectionCode;
use crate::synapse::SynapseClient;
use anyhow::bail;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use tracing::{info, warn};

/// Receipt types for different outcomes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReceiptType {
    /// Request was approved and executed
    Execution,
    /// Request was rejected by SAT
    Rejection,
    /// Request was quarantined for review
    Quarantine,
    /// Ihsān threshold failure
    IhsanFailure,
    // KEP Receipt Types
    /// Synergy detection between knowledge elements
    SynergyDetection,
    /// Compound knowledge synthesis
    CompoundDiscovery,
    /// Explosion mode entry
    ExplosionModeEntry,
    /// Explosion mode exit
    ExplosionModeExit,
    /// Learning rate acceleration
    LearningAcceleration,
    /// Feedback loop cycle
    FeedbackLoopCycle,
    // Autopoietic Receipt Types
    /// Autopoietic generation cycle
    AutopoieticGeneration,
    /// Blueprint evolution
    BlueprintEvolution,
    /// Proof chain anchor
    ProofChainAnchor,
}

/// Rejection receipt - evidence of SAT blocking a request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Optional request ID for traceability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Timestamp of rejection
    pub timestamp: DateTime<Utc>,
    /// Task that was rejected (truncated for privacy)
    pub task_summary: String,
    /// Rejection codes from SAT
    pub rejection_codes: Vec<String>,
    /// Primary rejection reason
    pub primary_reason: String,
    /// Escalation level assigned by FATE
    pub escalation_level: String,
    /// FATE escalation ID (if escalated)
    pub escalation_id: Option<String>,
    /// Validators that rejected
    pub rejecting_validators: Vec<String>,
    /// Validators that approved (for audit)
    pub approving_validators: Vec<String>,
    /// Recommended action
    pub recommended_action: String,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

/// Execution receipt - evidence of successful SAT approval + execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Optional request ID for traceability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Timestamp of execution
    pub timestamp: DateTime<Utc>,
    /// Task that was executed (truncated for privacy)
    pub task_summary: String,
    /// SAT validation time
    pub sat_validation_ms: u128,
    /// PAT execution time
    pub pat_execution_ms: u128,
    /// Total latency
    pub total_latency_ms: u128,
    /// Synergy score achieved
    pub synergy_score: f64,
    /// Ihsān score achieved
    pub ihsan_score: f64,
    /// Ihsān threshold applied
    pub ihsan_threshold: f64,
    /// Number of PAT agents that contributed
    pub pat_agents_count: usize,
    /// Number of SAT validators that approved
    pub sat_approvers_count: usize,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// KEP RECEIPT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Receipt for synergy detection between knowledge elements (KEP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynergyDetectionReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Timestamp of detection
    pub timestamp: DateTime<Utc>,
    /// Source element ID (principle/pattern)
    pub source_id: String,
    /// Target element ID
    pub target_id: String,
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Semantic similarity score
    pub similarity_score: f64,
    /// Graph structural score
    pub structural_score: f64,
    /// Combined composite score
    pub composite_score: f64,
    /// Hypothesized emergent capability
    pub potential_compound: String,
    /// Pre-validated Ihsān alignment
    pub ihsan_alignment: f64,
    /// Whether threshold was passed
    pub passed_threshold: bool,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

/// Receipt for compound knowledge synthesis (KEP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundDiscoveryReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Timestamp of synthesis
    pub timestamp: DateTime<Utc>,
    /// Compound ID
    pub compound_id: String,
    /// Compound name
    pub compound_name: String,
    /// Emergent capability description
    pub emergent_capability: String,
    /// Synthesis type (additive, multiplicative, transcendent)
    pub synthesis_type: String,
    /// Source principle IDs
    pub source_principles: Vec<String>,
    /// Domains bridged by this compound
    pub domains_bridged: Vec<String>,
    /// Ihsān alignment score
    pub ihsan_alignment: f64,
    /// SAT votes in favor
    pub sat_votes_for: usize,
    /// SAT votes against
    pub sat_votes_against: usize,
    /// Whether SAT consensus was reached (3/5)
    pub sat_consensus_reached: bool,
    /// Related synergy receipt ID
    pub synergy_receipt_id: String,
    /// Synthesis latency in milliseconds
    pub synthesis_latency_ms: u64,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

/// Receipt for explosion mode transitions (KEP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplosionModeReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type (ExplosionModeEntry or ExplosionModeExit)
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Timestamp of transition
    pub timestamp: DateTime<Utc>,
    /// Whether this is an entry (true) or exit (false)
    pub is_entry: bool,
    /// Knowledge mass at transition
    pub knowledge_mass: u64,
    /// Discovery velocity (compounds/hour)
    pub discovery_velocity: f64,
    /// Synergy density ratio
    pub synergy_density: f64,
    /// Learning rate multiplier
    pub learning_rate_multiplier: f64,
    /// System-wide Ihsān average
    pub ihsan_average: f64,
    /// Whether knowledge mass condition was met
    pub mass_threshold_met: bool,
    /// Whether velocity condition was met
    pub velocity_threshold_met: bool,
    /// Whether synergy condition was met
    pub synergy_threshold_met: bool,
    /// Whether Ihsān condition was met
    pub ihsan_threshold_met: bool,
    /// SAT votes for explosion mode (requires 4/5)
    pub sat_votes_for: usize,
    /// SAT votes against
    pub sat_votes_against: usize,
    /// Duration in seconds (for exit receipts)
    pub duration_seconds: u64,
    /// Compounds synthesized during explosion (for exit)
    pub compounds_synthesized: u64,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOPOIETIC RECEIPT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Receipt for autopoietic generation cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticGenerationReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Timestamp of generation
    pub timestamp: DateTime<Utc>,
    /// Generation number
    pub generation: u64,
    /// Generation duration in milliseconds
    pub duration_ms: u64,
    /// Aggregate Ihsān score
    pub aggregate_ihsan: f64,
    /// Individual Ihsān dimension scores
    pub ihsan_dimensions: AutopoieticIhsanDimensions,
    /// KEP state at end of generation
    pub kep_state: String,
    /// KEP progress metrics
    pub kep_progress: AutopoieticKEPProgress,
    /// Number of tasks processed
    pub tasks_processed: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of rejections
    pub rejections: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: u64,
    /// P95 latency in milliseconds
    pub p95_latency_ms: u64,
    /// Number of blueprint improvements applied
    pub improvements_count: usize,
    /// List of improvement descriptions
    pub improvements: Vec<String>,
    /// Proof chain hash for this generation
    pub proof_hash: String,
    /// Previous generation's proof hash (for lineage)
    pub previous_proof_hash: String,
    /// Whether Ihsān gate passed
    pub ihsan_gate_passed: bool,
    /// Whether all SAPE probes passed
    pub sape_all_passed: bool,
    /// Number of active blueprints
    pub active_blueprints: usize,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

/// Ihsān dimensions for autopoietic receipts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticIhsanDimensions {
    pub correctness: f64,
    pub safety: f64,
    pub user_benefit: f64,
    pub efficiency: f64,
    pub auditability: f64,
    pub anti_centralization: f64,
    pub robustness: f64,
    pub adl_fairness: f64,
}

/// KEP progress for autopoietic receipts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticKEPProgress {
    pub knowledge_mass: u64,
    pub discovery_velocity: f64,
    pub synergy_density: f64,
    pub learning_rate_multiplier: f64,
    pub synergies_detected: u64,
    pub compounds_synthesized: u64,
}

/// Receipt for blueprint evolution events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintEvolutionReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Timestamp of evolution
    pub timestamp: DateTime<Utc>,
    /// Blueprint ID (new evolved version)
    pub blueprint_id: String,
    /// Parent blueprint ID
    pub parent_blueprint_id: String,
    /// Blueprint name
    pub blueprint_name: String,
    /// Agent team (PAT or SAT)
    pub team: String,
    /// Capability slot
    pub capability_slot: String,
    /// Generation number
    pub generation: u64,
    /// Number of mutations applied
    pub mutations_count: usize,
    /// Description of mutations
    pub mutations: Vec<String>,
    /// Lineage hash
    pub lineage_hash: String,
    /// Previous Ihsān average
    pub previous_ihsan_avg: Option<f64>,
    /// Fitness score
    pub fitness_score: f64,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

/// Receipt for proof chain blockchain anchoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChainAnchorReceipt {
    /// Schema version
    pub schema: String,
    /// Receipt type
    pub receipt_type: ReceiptType,
    /// Unique receipt ID
    pub receipt_id: String,
    /// Timestamp of anchoring
    pub timestamp: DateTime<Utc>,
    /// Chain name (e.g., "bizra-native")
    pub chain: String,
    /// Transaction hash
    pub tx_hash: String,
    /// Block number
    pub block_number: u64,
    /// Generations included in this anchor
    pub generations: Vec<u64>,
    /// First generation in range
    pub first_generation: u64,
    /// Last generation in range
    pub last_generation: u64,
    /// Merkle root of included proofs
    pub merkle_root: String,
    /// Genesis hash of the proof chain
    pub genesis_hash: String,
    /// Head hash before anchoring
    pub head_hash: String,
    /// Total proof chain length
    pub chain_length: usize,
    /// SHA-256 hash of receipt content
    pub integrity_hash: String,
}

/// Receipt emitter - creates and persists receipts
pub struct ReceiptEmitter {
    /// Directory to store receipts
    output_dir: String,
    /// Counter for receipt IDs
    counter: std::sync::atomic::AtomicU64,
    /// Redis client for persistence (optional)
    synapse: Option<SynapseClient>,
}

impl ReceiptEmitter {
    pub fn new(output_dir: &str) -> Self {
        // Ensure output directory exists
        if let Err(e) = fs::create_dir_all(output_dir) {
            warn!(error = %e, dir = output_dir, "Failed to create receipts directory");
        }
        
        info!(output_dir = output_dir, "📋 Receipt emitter initialized");
        
        Self {
            output_dir: output_dir.to_string(),
            counter: std::sync::atomic::AtomicU64::new(1),
            synapse: None,
        }
    }
    
    /// Create with Redis persistence
    pub fn with_synapse(output_dir: &str, synapse: SynapseClient) -> Self {
        if let Err(e) = fs::create_dir_all(output_dir) {
            warn!(error = %e, dir = output_dir, "Failed to create receipts directory");
        }
        
        info!(output_dir = output_dir, "📋 Receipt emitter initialized with Redis persistence");
        
        Self {
            output_dir: output_dir.to_string(),
            counter: std::sync::atomic::AtomicU64::new(1),
            synapse: Some(synapse),
        }
    }
    
    /// Create from environment (hard fail if Redis unavailable)
    pub async fn from_env(output_dir: &str) -> anyhow::Result<Self> {
        let synapse = crate::synapse::SynapseClient::from_env().await?;
        if !synapse.is_available() {
            bail!("Synapse client reported unavailable state");
        }
        info!("📋 ReceiptEmitter connected to Redis for durable persistence");
        Ok(Self::with_synapse(output_dir, synapse))
    }

    /// Emit a rejection receipt
    pub fn emit_rejection(
        &self,
        task: &str,
        rejection_codes: &[RejectionCode],
        escalation: &Escalation,
        rejecting_validators: Vec<String>,
        approving_validators: Vec<String>,
        request_id: Option<String>,
    ) -> RejectionReceipt {
        let receipt_id = format!(
            "REJ-{}-{:06}",
            Utc::now().format("%Y%m%d%H%M%S"),
            self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );

        let task_summary = if task.len() > 100 {
            format!("{}...", &task[..100])
        } else {
            task.to_string()
        };

        let rejection_code_strings: Vec<String> = rejection_codes
            .iter()
            .map(|c| c.to_string())
            .collect();

        let primary_reason = rejection_codes
            .first()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "Unknown rejection".to_string());

        let escalation_level = match escalation.level {
            EscalationLevel::Low => "LOW",
            EscalationLevel::Medium => "MEDIUM",
            EscalationLevel::High => "HIGH",
            EscalationLevel::Critical => "CRITICAL",
        }
        .to_string();

        let receipt_type = if rejection_codes.iter().any(|c| matches!(c, RejectionCode::Quarantine(_))) {
            ReceiptType::Quarantine
        } else {
            ReceiptType::Rejection
        };

        // Create receipt without hash first
        let mut receipt = RejectionReceipt {
            schema: "bizra-rejection-receipt-v1".to_string(),
            receipt_type,
            receipt_id: receipt_id.clone(),
            request_id,
            timestamp: Utc::now(),
            task_summary,
            rejection_codes: rejection_code_strings,
            primary_reason,
            escalation_level,
            escalation_id: Some(escalation.id.clone()),
            rejecting_validators,
            approving_validators,
            recommended_action: escalation.recommended_action.clone(),
            integrity_hash: String::new(),
        };

        // Calculate integrity hash
        receipt.integrity_hash = self.calculate_hash(&receipt);

        // Persist receipt
        self.persist_receipt(&receipt_id, &receipt);

        info!(
            receipt_id = %receipt_id,
            escalation_id = %escalation.id,
            "🧾 Rejection receipt emitted"
        );

        receipt
    }

    /// Emit an execution receipt (successful flow)
    pub fn emit_execution(
        &self,
        task: &str,
        sat_validation_ms: u128,
        pat_execution_ms: u128,
        total_latency_ms: u128,
        synergy_score: f64,
        ihsan_score: f64,
        ihsan_threshold: f64,
        pat_agents_count: usize,
        sat_approvers_count: usize,
        request_id: Option<String>,
    ) -> ExecutionReceipt {
        let receipt_id = format!(
            "EXEC-{}-{:06}",
            Utc::now().format("%Y%m%d%H%M%S"),
            self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );

        let task_summary = if task.len() > 100 {
            format!("{}...", &task[..100])
        } else {
            task.to_string()
        };

        let mut receipt = ExecutionReceipt {
            schema: "bizra-execution-receipt-v1".to_string(),
            receipt_type: ReceiptType::Execution,
            receipt_id: receipt_id.clone(),
            request_id,
            timestamp: Utc::now(),
            task_summary,
            sat_validation_ms,
            pat_execution_ms,
            total_latency_ms,
            synergy_score,
            ihsan_score,
            ihsan_threshold,
            pat_agents_count,
            sat_approvers_count,
            integrity_hash: String::new(),
        };

        // Calculate integrity hash
        receipt.integrity_hash = self.calculate_execution_hash(&receipt);

        // Persist receipt
        self.persist_execution_receipt(&receipt_id, &receipt);

        info!(
            receipt_id = %receipt_id,
            synergy = synergy_score,
            ihsan = ihsan_score,
            "🧾 Execution receipt emitted"
        );

        receipt
    }

    fn calculate_hash(&self, receipt: &RejectionReceipt) -> String {
        let content = format!(
            "{}|{}|{}|{}|{}",
            receipt.receipt_id,
            receipt.timestamp.to_rfc3339(),
            receipt.task_summary,
            receipt.rejection_codes.join(","),
            receipt.escalation_id.as_deref().unwrap_or("none")
        );
        let hash = Sha256::digest(content.as_bytes());
        format!("sha256:{:x}", hash)
    }

    fn calculate_execution_hash(&self, receipt: &ExecutionReceipt) -> String {
        let content = format!(
            "{}|{}|{}|{:.4}|{:.4}",
            receipt.receipt_id,
            receipt.timestamp.to_rfc3339(),
            receipt.task_summary,
            receipt.synergy_score,
            receipt.ihsan_score
        );
        let hash = Sha256::digest(content.as_bytes());
        format!("sha256:{:x}", hash)
    }

    fn persist_receipt(&self, receipt_id: &str, receipt: &RejectionReceipt) {
        let filename = format!("{}.json", receipt_id);
        let path = Path::new(&self.output_dir).join(&filename);
        
        match serde_json::to_string_pretty(receipt) {
            Ok(json) => {
                // Persist to filesystem (Redis persistence via async method)
                if let Err(e) = fs::write(&path, json) {
                    warn!(error = %e, path = ?path, "Failed to persist rejection receipt");
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to serialize rejection receipt");
            }
        }
    }

    fn persist_execution_receipt(&self, receipt_id: &str, receipt: &ExecutionReceipt) {
        let filename = format!("{}.json", receipt_id);
        let path = Path::new(&self.output_dir).join(&filename);
        
        match serde_json::to_string_pretty(receipt) {
            Ok(json) => {
                // Persist to filesystem (Redis persistence via async method)
                if let Err(e) = fs::write(&path, json) {
                    warn!(error = %e, path = ?path, "Failed to persist execution receipt");
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to serialize execution receipt");
            }
        }
    }
    
    /// Persist receipt to Redis asynchronously
    pub async fn persist_to_synapse(&self, receipt_id: &str, json: &str) -> Result<(), anyhow::Error> {
        if let Some(ref synapse) = self.synapse {
            synapse.store_receipt(receipt_id, json).await?;
        }
        Ok(())
    }
    
    /// Retrieve a receipt from Redis by ID (async)
    pub async fn get_receipt_async(&self, receipt_id: &str) -> Option<String> {
        if let Some(ref synapse) = self.synapse {
            if let Ok(Some(json)) = synapse.get_receipt(receipt_id).await {
                return Some(json);
            }
        }
        
        // Fallback to filesystem
        let filename = format!("{}.json", receipt_id);
        let path = Path::new(&self.output_dir).join(&filename);
        fs::read_to_string(&path).ok()
    }
    
    /// Get recent receipts from Redis (async)
    pub async fn recent_receipts_async(&self, limit: usize) -> Vec<String> {
        if let Some(ref synapse) = self.synapse {
            if let Ok(receipts) = synapse.recent_receipts(limit as isize).await {
                return receipts;
            }
        }
        Vec::new()
    }
    
    /// Sync version: Retrieve a receipt from filesystem only
    pub fn get_receipt(&self, receipt_id: &str) -> Option<String> {
        let filename = format!("{}.json", receipt_id);
        let path = Path::new(&self.output_dir).join(&filename);
        fs::read_to_string(&path).ok()
    }
    
    /// Sync version: returns empty (use async for Redis)
    pub fn recent_receipts(&self, _limit: usize) -> Vec<String> {
        Vec::new()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BLOCKCHAIN ANCHORING
    // ═══════════════════════════════════════════════════════════════════════════

    /// Anchor an execution receipt to BIZRA native blockchain
    ///
    /// This creates an immutable record on the BIZRA chain for:
    /// - Proof of execution
    /// - Ihsān score attestation
    /// - SAT consensus evidence
    pub async fn anchor_execution_to_chain(
        &self,
        receipt: &ExecutionReceipt,
    ) -> Result<crate::blockchain::AnchorResult, anyhow::Error> {
        info!(
            receipt_id = %receipt.receipt_id,
            ihsan = receipt.ihsan_score,
            "⛓️ Anchoring execution receipt to BIZRA chain"
        );

        crate::blockchain::anchor_receipt(
            &receipt.receipt_id,
            &format!("{:?}", receipt.receipt_type),
            &receipt.integrity_hash,
            receipt.ihsan_score,
            receipt.sat_approvers_count as u8,
        ).await
    }

    /// Anchor a rejection receipt to BIZRA native blockchain
    ///
    /// This creates an immutable record for:
    /// - Rejection evidence
    /// - FATE escalation record
    /// - SAT voting record
    pub async fn anchor_rejection_to_chain(
        &self,
        receipt: &RejectionReceipt,
    ) -> Result<crate::blockchain::AnchorResult, anyhow::Error> {
        info!(
            receipt_id = %receipt.receipt_id,
            escalation = %receipt.escalation_level,
            "⛓️ Anchoring rejection receipt to BIZRA chain"
        );

        crate::blockchain::anchor_receipt(
            &receipt.receipt_id,
            &format!("{:?}", receipt.receipt_type),
            &receipt.integrity_hash,
            0.0, // Rejections don't have Ihsān score
            receipt.approving_validators.len() as u8,
        ).await
    }

    /// Anchor any receipt to chain by ID (generic method)
    pub async fn anchor_to_chain(
        &self,
        receipt_id: &str,
        receipt_type: ReceiptType,
        integrity_hash: &str,
        ihsan_score: f64,
        sat_approvers: u8,
    ) -> Result<crate::blockchain::AnchorResult, anyhow::Error> {
        info!(
            receipt_id = %receipt_id,
            receipt_type = ?receipt_type,
            "⛓️ Anchoring receipt to BIZRA chain"
        );

        crate::blockchain::anchor_receipt(
            receipt_id,
            &format!("{:?}", receipt_type),
            integrity_hash,
            ihsan_score,
            sat_approvers,
        ).await
    }
}

impl Default for ReceiptEmitter {
    fn default() -> Self {
        Self::new("docs/evidence/receipts")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fate::FATECoordinator;
    use std::collections::HashMap;

    #[test]
    fn test_rejection_receipt_creation() {
        let emitter = ReceiptEmitter::new("target/test_receipts");
        let mut fate = FATECoordinator::new();
        
        let codes = vec![RejectionCode::SecurityThreat("SQL injection".to_string())];
        let escalation = fate.escalate_rejection(&codes, "DROP TABLE users", &HashMap::new());
        
        let receipt = emitter.emit_rejection(
            "DROP TABLE users",
            &codes,
            &escalation,
            vec!["security_guardian".to_string()],
            vec![],
            Some("REQ-TEST-001".to_string()),
        );

        assert_eq!(receipt.receipt_type, ReceiptType::Rejection);
        assert!(receipt.receipt_id.starts_with("REJ-"));
        assert!(receipt.integrity_hash.starts_with("sha256:"));
        assert!(receipt.rejection_codes[0].contains("SECURITY_THREAT"));
    }

    #[test]
    fn test_quarantine_receipt_type() {
        let emitter = ReceiptEmitter::new("target/test_receipts");
        let mut fate = FATECoordinator::new();
        
        let codes = vec![RejectionCode::Quarantine("uncertain intent".to_string())];
        let escalation = fate.escalate_rejection(&codes, "ambiguous task", &HashMap::new());
        
        let receipt = emitter.emit_rejection(
            "ambiguous task",
            &codes,
            &escalation,
            vec!["ethics_validator".to_string()],
            vec!["security_guardian".to_string()],
            Some("REQ-TEST-002".to_string()),
        );

        assert_eq!(receipt.receipt_type, ReceiptType::Quarantine);
    }

    #[test]
    fn test_execution_receipt_creation() {
        let emitter = ReceiptEmitter::new("target/test_receipts");
        
        let receipt = emitter.emit_execution(
            "Generate unit tests for user module",
            15,
            250,
            275,
            0.87,
            0.92,
            0.90,
            7,
            5,
            Some("REQ-TEST-003".to_string()),
        );

        assert_eq!(receipt.receipt_type, ReceiptType::Execution);
        assert!(receipt.receipt_id.starts_with("EXEC-"));
        assert!(receipt.integrity_hash.starts_with("sha256:"));
        assert!(receipt.ihsan_score >= receipt.ihsan_threshold);
    }
}
