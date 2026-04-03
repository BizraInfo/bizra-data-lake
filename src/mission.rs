// src/mission.rs - Canonical Mission Contracts (Layer 1 Law)
//
// FROZEN: These are the four canonical contracts that define the authoritative
// interface between constitutional layers. All other implementations (Python,
// TypeScript) MUST mirror these definitions exactly.
//
// Contracts:
//   1. MissionEnvelope  - Operator-facing mission representation
//   2. GateVerdict      - Constitutional gate output (PERMIT|REJECT|REVIEW|SCORE_ONLY)
//   3. ReceiptArtifact  - Signed proof artifact with BLAKE3 chain
//   4. ManifestArtifact - Daily proof-of-life heartbeat
//
// Authority chain: Layer 1 defines -> Layer 2 interprets -> Layer 3 enforces
//                  -> Layer 4 experiments -> Layer 5 reveals
//
// Gate order (frozen):
//   Ingress -> State -> Proposal -> Constitution -> Proof -> Receipt -> Refinement -> Reflex

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

// ── Mission State Machine ────────────────────────────────────────────────────

/// Canonical mission states. Transitions are strictly enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionState {
    Draft,
    Submitted,
    Evaluating,
    Permitted,
    Executing,
    Completed,
    Rejected,
    Failed,
    Revoked,
}

impl MissionState {
    /// Returns whether a transition from `self` to `target` is valid.
    pub fn can_transition_to(&self, target: MissionState) -> bool {
        matches!(
            (self, target),
            (MissionState::Draft, MissionState::Submitted)
                | (MissionState::Submitted, MissionState::Evaluating)
                | (MissionState::Evaluating, MissionState::Permitted)
                | (MissionState::Evaluating, MissionState::Rejected)
                | (MissionState::Permitted, MissionState::Executing)
                | (MissionState::Executing, MissionState::Completed)
                | (MissionState::Executing, MissionState::Failed)
                | (MissionState::Permitted, MissionState::Revoked)
                | (MissionState::Executing, MissionState::Revoked)
        )
    }
}

// ── Verdict Types ────────────────────────────────────────────────────────────

/// Constitutional gate layer (1-3). Layer 1 = hard law, Layer 2 = bounded review,
/// Layer 3 = judiciary/advisory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateLayer(u8);

impl GateLayer {
    pub fn new(layer: u8) -> Option<Self> {
        if (1..=3).contains(&layer) {
            Some(Self(layer))
        } else {
            None
        }
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

/// The four canonical verdict kinds.
/// - PERMIT: Approved by constitutional gate (blocking, Layer 1/3)
/// - REJECT: Denied by constitutional gate (blocking, Layer 1/3)
/// - REVIEW: Requires human/bounded review (Layer 2, timeout-aware)
/// - SCORE_ONLY: Advisory score, non-blocking (Layer 2/4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerdictKind {
    Permit,
    Reject,
    Review,
    ScoreOnly,
}

/// Risk classification for missions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    Low,
    Medium,
    High,
    Critical,
}

/// Mission classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionClass {
    Work,
    Query,
    Maintenance,
    Constitutional,
}

/// Evidence item attached to a gate verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub code: String,
    pub description: String,
    pub severity: EvidenceSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSeverity {
    Info,
    Warning,
    Critical,
}

// ── Contract 1: GateVerdict ──────────────────────────────────────────────────

/// Canonical GateVerdict — produced by the authoritative kernel (Layer 1/2).
/// The operator surface receives this; it does NOT produce it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateVerdict {
    /// Unique verdict identifier from the authority
    pub verdict_id: String,
    /// Which constitutional layer produced this verdict
    pub layer: GateLayer,
    /// PERMIT, REJECT, REVIEW, or SCORE_ONLY
    pub kind: VerdictKind,
    /// Machine-readable reason code (e.g. IHSAN_FLOOR_VIOLATION)
    pub reason_code: String,
    /// Human-readable explanation
    pub reason: String,
    /// Bounded advisory score 0-100 (higher = safer)
    pub advisory_score: u8,
    /// Evidence items that contributed to the verdict
    pub evidence: Vec<EvidenceItem>,
    /// Timestamp from the authority
    pub issued_at: DateTime<Utc>,
    /// Ed25519 signature from the authority (hex-encoded)
    pub authority_signature: Option<String>,
}

impl GateVerdict {
    /// Create a PERMIT verdict from Layer 1.
    pub fn permit(reason_code: &str, reason: &str, score: u8) -> Self {
        Self {
            verdict_id: Uuid::new_v4().to_string(),
            layer: GateLayer::new(1).unwrap(),
            kind: VerdictKind::Permit,
            reason_code: reason_code.to_string(),
            reason: reason.to_string(),
            advisory_score: score,
            evidence: Vec::new(),
            issued_at: Utc::now(),
            authority_signature: None,
        }
    }

    /// Create a REJECT verdict from Layer 1.
    pub fn reject(reason_code: &str, reason: &str, evidence: Vec<EvidenceItem>) -> Self {
        Self {
            verdict_id: Uuid::new_v4().to_string(),
            layer: GateLayer::new(1).unwrap(),
            kind: VerdictKind::Reject,
            reason_code: reason_code.to_string(),
            reason: reason.to_string(),
            advisory_score: 0,
            evidence,
            issued_at: Utc::now(),
            authority_signature: None,
        }
    }

    pub fn is_permit(&self) -> bool {
        self.kind == VerdictKind::Permit
    }

    pub fn is_blocking(&self) -> bool {
        matches!(self.kind, VerdictKind::Permit | VerdictKind::Reject)
    }

    /// Compute SHA-256 integrity hash of the verdict payload.
    pub fn integrity_hash(&self) -> String {
        let payload = format!(
            "{}{}{}{}{}",
            self.verdict_id,
            self.issued_at.to_rfc3339(),
            self.reason_code,
            self.reason,
            self.advisory_score,
        );
        let hash = Sha256::digest(payload.as_bytes());
        hex::encode(hash)
    }
}

// ── Contract 2: MissionEnvelope ──────────────────────────────────────────────

/// Canonical MissionEnvelope — the operator-facing mission representation.
/// Created by Layer 5, evaluated by Layer 1/2, executed by Layer 3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionEnvelope {
    /// Unique mission identifier
    pub id: String,
    /// Current state in the MissionStateMachine
    pub state: MissionState,
    /// Mission description / intent
    pub description: String,
    /// Classification
    pub mission_class: MissionClass,
    /// Declared scope paths
    pub scope: Vec<String>,
    /// Declared risk class
    pub risk_class: RiskClass,
    /// Gate verdict if mission has been evaluated
    pub verdict: Option<GateVerdict>,
    /// Ed25519 public key of the submitting operator (hex-encoded)
    pub operator_key: Option<String>,
    /// Agent assigned to execute (if any)
    pub assigned_agent_id: Option<String>,
    /// Token budget constraints
    pub max_tokens: u64,
    pub max_steps: u32,
    /// Resource tracking
    pub budget_used: u64,
    /// Timestamps from authority
    pub submitted_at: Option<DateTime<Utc>>,
    pub evaluated_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl MissionEnvelope {
    /// Create a new mission in Draft state.
    pub fn new(description: &str, mission_class: MissionClass, risk_class: RiskClass) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            state: MissionState::Draft,
            description: description.to_string(),
            mission_class,
            scope: Vec::new(),
            risk_class,
            verdict: None,
            operator_key: None,
            assigned_agent_id: None,
            max_tokens: 100_000,
            max_steps: 50,
            budget_used: 0,
            submitted_at: None,
            evaluated_at: None,
            completed_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Attempt a state transition. Returns Err if the transition is invalid.
    pub fn transition_to(&mut self, target: MissionState) -> Result<(), MissionTransitionError> {
        if self.state.can_transition_to(target) {
            self.state = target;
            self.updated_at = Utc::now();
            match target {
                MissionState::Submitted => self.submitted_at = Some(Utc::now()),
                MissionState::Permitted | MissionState::Rejected => {
                    self.evaluated_at = Some(Utc::now());
                }
                MissionState::Completed | MissionState::Failed => {
                    self.completed_at = Some(Utc::now());
                }
                _ => {}
            }
            Ok(())
        } else {
            Err(MissionTransitionError {
                from: self.state,
                to: target,
            })
        }
    }

    /// Apply a gate verdict to this mission and transition state accordingly.
    pub fn apply_verdict(&mut self, verdict: GateVerdict) -> Result<(), MissionTransitionError> {
        let target = if verdict.is_permit() {
            MissionState::Permitted
        } else {
            MissionState::Rejected
        };
        self.verdict = Some(verdict);
        self.transition_to(target)
    }
}

#[derive(Debug, Clone)]
pub struct MissionTransitionError {
    pub from: MissionState,
    pub to: MissionState,
}

impl std::fmt::Display for MissionTransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid mission transition: {:?} -> {:?}",
            self.from, self.to
        )
    }
}

impl std::error::Error for MissionTransitionError {}

// ── Contract 3: ReceiptArtifact ──────────────────────────────────────────────

/// Receipt states in the ReceiptStateMachine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptState {
    Issued,
    Chained,
    Verified,
    Expired,
    Revoked,
}

/// Action recorded in a receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptAction {
    pub tool: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

/// Verification result for a receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptVerification {
    pub signature_valid: bool,
    pub chain_intact: bool,
    pub payload_intact: bool,
    pub verified_at: DateTime<Utc>,
    pub verified_by: String,
}

/// Canonical ReceiptArtifact — the signed proof artifact produced after
/// constitutional evaluation. The operator surface receives this; it does
/// NOT create it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptArtifact {
    /// Unique receipt identifier (canonical UUID)
    pub receipt_id: String,
    /// Reference to the mission this receipt covers
    pub mission_id: String,
    /// Chain integrity: hash of the previous receipt in the lineage
    pub prev_receipt_hash: String,
    /// Current receipt state in the ReceiptStateMachine
    pub receipt_state: ReceiptState,
    /// The gate verdict that produced this receipt
    pub verdict: GateVerdict,
    /// Actions taken during execution
    pub actions: Vec<ReceiptAction>,
    /// BLAKE3 hash of all evidence
    pub evidence_hash: String,
    /// BLAKE3 hash of the canonical payload
    pub payload_hash: String,
    /// Ed25519 signature from the authority (hex-encoded)
    pub authority_signature: String,
    /// Timestamp
    pub issued_at: DateTime<Utc>,
    /// Verification status — filled by Layer 3 bridge on verification
    pub verification: Option<ReceiptVerification>,
}

impl ReceiptArtifact {
    /// Create a new receipt from a mission and verdict, chaining to the previous receipt.
    pub fn new(
        mission_id: &str,
        prev_receipt_hash: &str,
        verdict: GateVerdict,
        actions: Vec<ReceiptAction>,
    ) -> Self {
        let receipt_id = Uuid::new_v4().to_string();
        let issued_at = Utc::now();

        // Compute BLAKE3 evidence hash from actions
        let evidence_payload = serde_json::to_string(&actions).unwrap_or_default();
        let evidence_hash = blake3::hash(evidence_payload.as_bytes()).to_hex().to_string();

        // Compute BLAKE3 payload hash (receipt_id + mission_id + prev_hash + verdict_id)
        let payload = format!(
            "{}{}{}{}",
            receipt_id, mission_id, prev_receipt_hash, verdict.verdict_id
        );
        let payload_hash = blake3::hash(payload.as_bytes()).to_hex().to_string();

        Self {
            receipt_id,
            mission_id: mission_id.to_string(),
            prev_receipt_hash: prev_receipt_hash.to_string(),
            receipt_state: ReceiptState::Issued,
            verdict,
            actions,
            evidence_hash,
            payload_hash,
            authority_signature: String::new(), // Set by signing module
            issued_at,
            verification: None,
        }
    }

    /// Verify the chain integrity (prev hash matches expected).
    pub fn verify_chain(&self, expected_prev_hash: &str) -> bool {
        self.prev_receipt_hash == expected_prev_hash
    }

    /// Verify the payload hash integrity.
    pub fn verify_payload(&self) -> bool {
        let payload = format!(
            "{}{}{}{}",
            self.receipt_id, self.mission_id, self.prev_receipt_hash, self.verdict.verdict_id
        );
        let expected = blake3::hash(payload.as_bytes()).to_hex().to_string();
        expected == self.payload_hash
    }
}

// ── Contract 4: ManifestArtifact ─────────────────────────────────────────────

/// System health snapshot for manifest generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthSnapshot {
    pub uptime_seconds: u64,
    pub constitutional_layer: LayerStatus,
    pub kernel_bridge: BridgeStatus,
    pub receipt_chain: ChainStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerStatus {
    Active,
    Degraded,
    Offline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BridgeStatus {
    Connected,
    Disconnected,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainStatus {
    Intact,
    Broken,
    Empty,
}

/// Manifest verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestVerification {
    pub signature_valid: bool,
    pub chain_head_matches: bool,
    pub receipt_count_matches: bool,
    pub verified_at: DateTime<Utc>,
}

/// Heartbeat status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeartbeatStatus {
    Alive,
    Degraded,
    Dead,
}

/// Canonical ManifestArtifact — the daily proof-of-life artifact.
/// Generated by the authority heartbeat, NOT by the operator surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestArtifact {
    /// Unique manifest identifier
    pub id: String,
    /// The 24h period this manifest covers
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Heartbeat status from authority
    pub heartbeat_status: HeartbeatStatus,
    /// Head of the receipt chain at manifest generation time
    pub receipt_chain_head: String,
    /// Receipt statistics
    pub receipt_count: u64,
    pub permit_count: u64,
    pub reject_count: u64,
    pub review_count: u64,
    /// System health snapshot
    pub system_health: SystemHealthSnapshot,
    /// Deployment identifier
    pub deployment_id: String,
    /// BLAKE3 hash of the manifest payload
    pub public_proof_hash: String,
    /// Ed25519 authority signature (hex-encoded)
    pub authority_signature: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Verification status
    pub verification: Option<ManifestVerification>,
}

impl ManifestArtifact {
    /// Generate a new manifest for a 24h period.
    pub fn generate(
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        receipt_chain_head: &str,
        receipt_count: u64,
        permit_count: u64,
        reject_count: u64,
        review_count: u64,
        system_health: SystemHealthSnapshot,
        deployment_id: &str,
    ) -> Self {
        let id = Uuid::new_v4().to_string();
        let generated_at = Utc::now();

        // Compute BLAKE3 proof hash
        let payload = format!(
            "{}{}{}{}{}{}{}{}",
            id,
            period_start.to_rfc3339(),
            period_end.to_rfc3339(),
            receipt_chain_head,
            receipt_count,
            permit_count,
            reject_count,
            review_count,
        );
        let public_proof_hash = blake3::hash(payload.as_bytes()).to_hex().to_string();

        Self {
            id,
            period_start,
            period_end,
            heartbeat_status: HeartbeatStatus::Alive,
            receipt_chain_head: receipt_chain_head.to_string(),
            receipt_count,
            permit_count,
            reject_count,
            review_count,
            system_health,
            deployment_id: deployment_id.to_string(),
            public_proof_hash,
            authority_signature: String::new(), // Set by signing module
            generated_at,
            verification: None,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.heartbeat_status == HeartbeatStatus::Alive
    }

    /// Verify the manifest proof hash.
    pub fn verify_proof(&self) -> bool {
        let payload = format!(
            "{}{}{}{}{}{}{}{}",
            self.id,
            self.period_start.to_rfc3339(),
            self.period_end.to_rfc3339(),
            self.receipt_chain_head,
            self.receipt_count,
            self.permit_count,
            self.reject_count,
            self.review_count,
        );
        let expected = blake3::hash(payload.as_bytes()).to_hex().to_string();
        expected == self.public_proof_hash
    }
}

// ── Genesis Seal ─────────────────────────────────────────────────────────────

/// GenesisSeal — the trust anchor for the entire receipt chain.
/// Set once at deployment, never changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisSeal {
    /// BLAKE3 hash of the empty/genesis state
    pub genesis_hash: String,
    /// Ed25519 public key of the genesis authority (hex-encoded)
    pub authority_public_key: String,
    /// Deployment identifier
    pub deployment_id: String,
    /// Timestamp of genesis
    pub created_at: DateTime<Utc>,
}

impl GenesisSeal {
    /// Create a new genesis seal for deployment.
    pub fn new(authority_public_key: &str, deployment_id: &str) -> Self {
        let created_at = Utc::now();
        let genesis_payload = format!("bizra-genesis:{}:{}", deployment_id, created_at.to_rfc3339());
        let genesis_hash = blake3::hash(genesis_payload.as_bytes()).to_hex().to_string();

        Self {
            genesis_hash,
            authority_public_key: authority_public_key.to_string(),
            deployment_id: deployment_id.to_string(),
            created_at,
        }
    }
}

// ── Frozen Gate Order ────────────────────────────────────────────────────────

/// The canonical gate order. All missions traverse these gates in sequence.
pub const GATE_ORDER: &[&str] = &[
    "Ingress",
    "State",
    "Proposal",
    "Constitution",
    "Proof",
    "Receipt",
    "Refinement",
    "Reflex",
];

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mission_state_transitions() {
        let mut mission = MissionEnvelope::new("Test mission", MissionClass::Work, RiskClass::Low);
        assert_eq!(mission.state, MissionState::Draft);

        assert!(mission.transition_to(MissionState::Submitted).is_ok());
        assert_eq!(mission.state, MissionState::Submitted);
        assert!(mission.submitted_at.is_some());

        assert!(mission.transition_to(MissionState::Evaluating).is_ok());

        // Invalid transition: Evaluating -> Executing (must go through Permitted)
        assert!(mission.transition_to(MissionState::Executing).is_err());

        assert!(mission.transition_to(MissionState::Permitted).is_ok());
        assert!(mission.evaluated_at.is_some());

        assert!(mission.transition_to(MissionState::Executing).is_ok());
        assert!(mission.transition_to(MissionState::Completed).is_ok());
        assert!(mission.completed_at.is_some());
    }

    #[test]
    fn test_gate_verdict_permit() {
        let verdict = GateVerdict::permit("IHSAN_PASS", "Ihsan score 0.97 exceeds threshold", 97);
        assert!(verdict.is_permit());
        assert!(verdict.is_blocking());
        assert!(!verdict.integrity_hash().is_empty());
    }

    #[test]
    fn test_gate_verdict_reject() {
        let evidence = vec![EvidenceItem {
            code: "SAFETY_VIOLATION".to_string(),
            description: "Unsafe tool invocation detected".to_string(),
            severity: EvidenceSeverity::Critical,
        }];
        let verdict = GateVerdict::reject("SAFETY_CHECK_FAILED", "Safety probe failed", evidence);
        assert!(!verdict.is_permit());
        assert!(verdict.is_blocking());
    }

    #[test]
    fn test_mission_apply_verdict() {
        let mut mission = MissionEnvelope::new("Test", MissionClass::Query, RiskClass::Low);
        mission.transition_to(MissionState::Submitted).unwrap();
        mission.transition_to(MissionState::Evaluating).unwrap();

        let verdict = GateVerdict::permit("OK", "Passed all gates", 95);
        assert!(mission.apply_verdict(verdict).is_ok());
        assert_eq!(mission.state, MissionState::Permitted);
        assert!(mission.verdict.is_some());
    }

    #[test]
    fn test_receipt_artifact_creation_and_verification() {
        let verdict = GateVerdict::permit("OK", "Passed", 95);
        let receipt = ReceiptArtifact::new(
            "mission-001",
            "0000000000000000000000000000000000000000000000000000000000000000",
            verdict,
            vec![],
        );

        assert_eq!(receipt.receipt_state, ReceiptState::Issued);
        assert!(receipt.verify_payload());
        assert!(receipt.verify_chain(
            "0000000000000000000000000000000000000000000000000000000000000000"
        ));
        assert!(!receipt.evidence_hash.is_empty());
        assert!(!receipt.payload_hash.is_empty());
    }

    #[test]
    fn test_manifest_artifact_generation_and_verification() {
        let now = Utc::now();
        let health = SystemHealthSnapshot {
            uptime_seconds: 86400,
            constitutional_layer: LayerStatus::Active,
            kernel_bridge: BridgeStatus::Connected,
            receipt_chain: ChainStatus::Intact,
        };

        let manifest = ManifestArtifact::generate(
            now - chrono::Duration::hours(24),
            now,
            "abc123",
            42,
            35,
            5,
            2,
            health,
            "node0-dev",
        );

        assert!(manifest.is_healthy());
        assert!(manifest.verify_proof());
        assert_eq!(manifest.receipt_count, 42);
    }

    #[test]
    fn test_genesis_seal() {
        let seal = GenesisSeal::new("ed25519_pubkey_hex", "node0-dev");
        assert!(!seal.genesis_hash.is_empty());
        assert_eq!(seal.deployment_id, "node0-dev");
    }

    #[test]
    fn test_gate_order_frozen() {
        assert_eq!(GATE_ORDER.len(), 8);
        assert_eq!(GATE_ORDER[0], "Ingress");
        assert_eq!(GATE_ORDER[7], "Reflex");
    }
}
