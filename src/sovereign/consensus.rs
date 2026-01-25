// src/sovereign/consensus.rs - Proof-of-Impact Consensus Engine
//
// # PROOF-OF-IMPACT CONSENSUS
//
// A novel consensus mechanism where block production rights and rewards
// are earned through verified positive impact rather than computational
// work (PoW) or stake alone (PoS).
//
// ## Standing on the Shoulders of Giants
// - Nakamoto (2008): Decentralized consensus via proof systems
// - Kiayias et al. (2024): Dual-token tokenomics with impact metrics
// - Buterin (2014): Generalized smart contract execution
//
// ## Impact Categories
// 1. Education (1.2x) - Knowledge sharing and learning
// 2. Healthcare (1.5x) - Health and wellness improvement
// 3. Environment (1.3x) - Sustainability and conservation
// 4. Economic (1.1x) - Financial empowerment
// 5. Governance (1.0x) - Democratic participation
// 6. Technical (1.4x) - Infrastructure contribution
// 7. Community (1.1x) - Social cohesion
//
// ## Validation Flow
// ```
// Attestation → Validators (quorum) → Consensus → BLOOM Mint
// ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ============================================================================
// IMPACT CATEGORY
// ============================================================================

/// Impact category for Proof-of-Impact attestations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImpactCategory {
    /// Educational content creation/curation (1.2x)
    Education = 0,
    /// Healthcare accessibility improvement (1.5x)
    Healthcare = 1,
    /// Environmental sustainability (1.3x)
    Environment = 2,
    /// Economic empowerment (1.1x)
    Economic = 3,
    /// Governance participation (1.0x)
    Governance = 4,
    /// Technical contribution - code, infrastructure (1.4x)
    Technical = 5,
    /// Community building (1.1x)
    Community = 6,
}

impl ImpactCategory {
    /// Get multiplier for category (basis points: 10000 = 1.0x)
    pub fn multiplier(&self) -> u32 {
        match self {
            Self::Education => 12000,   // 1.2x
            Self::Healthcare => 15000,  // 1.5x
            Self::Environment => 13000, // 1.3x
            Self::Economic => 11000,    // 1.1x
            Self::Governance => 10000,  // 1.0x
            Self::Technical => 14000,   // 1.4x
            Self::Community => 11000,   // 1.1x
        }
    }

    /// Parse from u8
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Education),
            1 => Some(Self::Healthcare),
            2 => Some(Self::Environment),
            3 => Some(Self::Economic),
            4 => Some(Self::Governance),
            5 => Some(Self::Technical),
            6 => Some(Self::Community),
            _ => None,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Education => "Education",
            Self::Healthcare => "Healthcare",
            Self::Environment => "Environment",
            Self::Economic => "Economic",
            Self::Governance => "Governance",
            Self::Technical => "Technical",
            Self::Community => "Community",
        }
    }
}

// ============================================================================
// IMPACT ATTESTATION
// ============================================================================

/// Impact attestation: Verified proof of positive impact
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImpactAttestation {
    /// Unique attestation ID (hash)
    pub id: [u8; 32],
    /// Attester's public key
    pub attester: [u8; 32],
    /// Beneficiary's public key (who receives BLOOM)
    pub beneficiary: [u8; 32],
    /// Impact category
    pub category: ImpactCategory,
    /// Base impact score (before multiplier)
    pub base_score: u64,
    /// Evidence hash (IPFS CID or document hash)
    pub evidence_hash: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
    /// Attester's signature (skipped for serde - large array)
    #[serde(skip, default = "default_signature")]
    pub signature: [u8; 64],
    /// Validator approvals
    pub validations: Vec<ValidatorApproval>,
    /// Status
    pub status: AttestationStatus,
}

/// Attestation status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationStatus {
    /// Pending validation
    Pending,
    /// Approved by quorum
    Approved,
    /// Rejected
    Rejected,
    /// Expired (timeout)
    Expired,
    /// BLOOM minted
    Finalized,
}

/// Validator approval for an attestation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidatorApproval {
    /// Validator public key
    pub validator: [u8; 32],
    /// Approval vote
    pub approved: bool,
    /// Optional rejection reason
    pub rejection_reason: Option<String>,
    /// Timestamp
    pub timestamp: u64,
    /// Signature (skipped for serde - large array)
    #[serde(skip, default = "default_signature")]
    pub signature: [u8; 64],
}

/// Default signature for serde deserialization
fn default_signature() -> [u8; 64] {
    [0u8; 64]
}

impl ImpactAttestation {
    /// Create new attestation
    pub fn new(
        attester: [u8; 32],
        beneficiary: [u8; 32],
        category: ImpactCategory,
        base_score: u64,
        evidence_hash: [u8; 32],
        timestamp: u64,
    ) -> Self {
        let mut attestation = Self {
            id: [0u8; 32],
            attester,
            beneficiary,
            category,
            base_score,
            evidence_hash,
            timestamp,
            signature: [0u8; 64],
            validations: Vec::new(),
            status: AttestationStatus::Pending,
        };
        attestation.id = attestation.compute_id();
        attestation
    }

    /// Compute attestation ID (hash)
    pub fn compute_id(&self) -> [u8; 32] {
        let bytes = self.to_bytes();
        let hash = Sha256::digest(&bytes);
        let mut id = [0u8; 32];
        id.copy_from_slice(&hash);
        id
    }

    /// Serialize for hashing/signing
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.attester);
        bytes.extend_from_slice(&self.beneficiary);
        bytes.push(self.category as u8);
        bytes.extend_from_slice(&self.base_score.to_le_bytes());
        bytes.extend_from_slice(&self.evidence_hash);
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes
    }

    /// Calculate final impact score with category multiplier
    pub fn final_score(&self) -> u64 {
        let multiplied =
            u128::from(self.base_score) * u128::from(self.category.multiplier()) / 10_000;
        multiplied as u64
    }

    /// Count approvals
    pub fn approval_count(&self) -> usize {
        self.validations.iter().filter(|v| v.approved).count()
    }

    /// Count rejections
    pub fn rejection_count(&self) -> usize {
        self.validations.iter().filter(|v| !v.approved).count()
    }

    /// Check if attestation has reached quorum
    pub fn has_quorum(&self, required: usize) -> bool {
        self.approval_count() >= required
    }

    /// Add validator approval
    pub fn add_validation(&mut self, approval: ValidatorApproval) {
        // Check for duplicate
        if self
            .validations
            .iter()
            .any(|v| v.validator == approval.validator)
        {
            return;
        }
        self.validations.push(approval);
    }
}

// ============================================================================
// POI VALIDATOR
// ============================================================================

/// Proof-of-Impact validator node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoIValidator {
    /// Validator's public key
    pub address: [u8; 32],
    /// Staked SEED amount (security deposit)
    pub stake: u128,
    /// Reputation score (0-100)
    pub reputation: u8,
    /// Total attestations validated
    pub attestations_validated: u64,
    /// Correct validations (for accuracy tracking)
    pub correct_validations: u64,
    /// Slashed validations
    pub slashed_validations: u64,
    /// Active status
    pub is_active: bool,
    /// Registration timestamp
    pub registered_at: u64,
    /// Last activity timestamp
    pub last_active: u64,
}

impl PoIValidator {
    /// Minimum stake required to be a validator: 10,000 SEED
    pub const MIN_STAKE: u128 = 10_000 * 1_000_000_000_000_000_000u128;

    /// Minimum reputation to validate: 50
    pub const MIN_REPUTATION: u8 = 50;

    /// Create new validator
    pub fn new(address: [u8; 32], stake: u128, timestamp: u64) -> Self {
        Self {
            address,
            stake,
            reputation: 50, // Start at neutral
            attestations_validated: 0,
            correct_validations: 0,
            slashed_validations: 0,
            is_active: true,
            registered_at: timestamp,
            last_active: timestamp,
        }
    }

    /// Calculate validation accuracy
    pub fn accuracy(&self) -> f64 {
        if self.attestations_validated == 0 {
            return 1.0;
        }
        self.correct_validations as f64 / self.attestations_validated as f64
    }

    /// Check if validator meets requirements
    pub fn is_eligible(&self) -> bool {
        self.is_active && self.stake >= Self::MIN_STAKE && self.reputation >= Self::MIN_REPUTATION
    }

    /// Record correct validation
    pub fn record_correct(&mut self, timestamp: u64) {
        self.attestations_validated += 1;
        self.correct_validations += 1;
        self.last_active = timestamp;

        // Increase reputation (max 100)
        self.reputation = self.reputation.saturating_add(1).min(100);
    }

    /// Record incorrect validation (slash)
    pub fn record_incorrect(&mut self, timestamp: u64) {
        self.attestations_validated += 1;
        self.slashed_validations += 1;
        self.last_active = timestamp;

        // Decrease reputation (min 0)
        self.reputation = self.reputation.saturating_sub(5);

        // Slash 1% of stake
        self.stake = self.stake * 99 / 100;
    }
}

// ============================================================================
// POI CONSENSUS ENGINE
// ============================================================================

/// Proof-of-Impact consensus engine
#[derive(Clone, Debug)]
pub struct PoIConsensus {
    /// Registered validators
    validators: HashMap<[u8; 32], PoIValidator>,
    /// Pending attestations
    pending: Vec<ImpactAttestation>,
    /// Finalized attestations
    finalized: Vec<ImpactAttestation>,
    /// Required quorum (number of validators)
    quorum_size: usize,
    /// Current epoch
    current_epoch: u64,
    /// Epoch duration in seconds
    epoch_duration: u64,
    /// Total impact accumulated
    total_impact: u128,
    /// Attestation timeout (epochs)
    attestation_timeout: u64,
}

impl PoIConsensus {
    /// Default quorum size: 3
    pub const DEFAULT_QUORUM: usize = 3;

    /// Default epoch duration: 86400 seconds (1 day)
    pub const DEFAULT_EPOCH_DURATION: u64 = 86400;

    /// Default attestation timeout: 7 epochs
    pub const DEFAULT_TIMEOUT: u64 = 7;

    /// Create new PoI consensus engine
    pub fn new(quorum_size: usize) -> Self {
        Self {
            validators: HashMap::new(),
            pending: Vec::new(),
            finalized: Vec::new(),
            quorum_size,
            current_epoch: 0,
            epoch_duration: Self::DEFAULT_EPOCH_DURATION,
            total_impact: 0,
            attestation_timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// Create with custom configuration
    pub fn with_config(quorum_size: usize, epoch_duration: u64, timeout: u64) -> Self {
        Self {
            validators: HashMap::new(),
            pending: Vec::new(),
            finalized: Vec::new(),
            quorum_size,
            current_epoch: 0,
            epoch_duration,
            total_impact: 0,
            attestation_timeout: timeout,
        }
    }

    /// Register a validator
    pub fn register_validator(&mut self, validator: PoIValidator) -> Result<(), ConsensusError> {
        if validator.stake < PoIValidator::MIN_STAKE {
            return Err(ConsensusError::InsufficientStake);
        }
        if self.validators.contains_key(&validator.address) {
            return Err(ConsensusError::ValidatorExists);
        }
        self.validators.insert(validator.address, validator);
        Ok(())
    }

    /// Unregister a validator
    pub fn unregister_validator(&mut self, address: &[u8; 32]) -> Result<PoIValidator, ConsensusError> {
        self.validators
            .remove(address)
            .ok_or(ConsensusError::UnknownValidator)
    }

    /// Get validator
    pub fn get_validator(&self, address: &[u8; 32]) -> Option<&PoIValidator> {
        self.validators.get(address)
    }

    /// Get validator count
    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }

    /// Get eligible validator count
    pub fn eligible_validator_count(&self) -> usize {
        self.validators.values().filter(|v| v.is_eligible()).count()
    }

    /// Submit attestation for validation
    pub fn submit_attestation(
        &mut self,
        attestation: ImpactAttestation,
    ) -> Result<[u8; 32], ConsensusError> {
        // Verify minimum score
        if attestation.base_score < 10 {
            return Err(ConsensusError::BelowMinimumImpact);
        }

        // Check for duplicate
        if self.pending.iter().any(|a| a.id == attestation.id) {
            return Err(ConsensusError::DuplicateAttestation);
        }

        let id = attestation.id;
        self.pending.push(attestation);
        Ok(id)
    }

    /// Add validator vote to attestation
    pub fn add_validation(
        &mut self,
        attestation_id: &[u8; 32],
        approval: ValidatorApproval,
    ) -> Result<bool, ConsensusError> {
        // Verify validator exists and is eligible
        let validator = self
            .validators
            .get(&approval.validator)
            .ok_or(ConsensusError::UnknownValidator)?;

        if !validator.is_eligible() {
            return Err(ConsensusError::IneligibleValidator);
        }

        // Find attestation
        let attestation = self
            .pending
            .iter_mut()
            .find(|a| &a.id == attestation_id)
            .ok_or(ConsensusError::AttestationNotFound)?;

        // Check for duplicate vote
        if attestation
            .validations
            .iter()
            .any(|v| v.validator == approval.validator)
        {
            return Err(ConsensusError::DuplicateValidation);
        }

        attestation.validations.push(approval);

        // Check if quorum reached
        Ok(attestation.has_quorum(self.quorum_size))
    }

    /// Process attestations and finalize those with quorum
    pub fn process_attestations(&mut self, current_timestamp: u64) -> Vec<ImpactAttestation> {
        let epoch = current_timestamp / self.epoch_duration;
        self.current_epoch = epoch;

        let quorum = self.quorum_size;
        let timeout = self.attestation_timeout;
        let epoch_duration = self.epoch_duration;

        // Partition attestations
        let mut approved = Vec::new();
        let mut still_pending = Vec::new();

        for mut attestation in self.pending.drain(..) {
            let attestation_epoch = attestation.timestamp / epoch_duration;
            let age = epoch.saturating_sub(attestation_epoch);

            if attestation.has_quorum(quorum) {
                attestation.status = AttestationStatus::Approved;
                self.total_impact += attestation.final_score() as u128;
                approved.push(attestation);
            } else if age >= timeout {
                attestation.status = AttestationStatus::Expired;
                // Expired attestations are dropped
            } else if attestation.rejection_count() > quorum {
                attestation.status = AttestationStatus::Rejected;
                // Rejected attestations are dropped
            } else {
                still_pending.push(attestation);
            }
        }

        self.pending = still_pending;
        self.finalized.extend(approved.clone());

        approved
    }

    /// Get pending attestations
    pub fn pending_attestations(&self) -> &[ImpactAttestation] {
        &self.pending
    }

    /// Get finalized attestations
    pub fn finalized_attestations(&self) -> &[ImpactAttestation] {
        &self.finalized
    }

    /// Get total accumulated impact
    pub fn total_impact(&self) -> u128 {
        self.total_impact
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Calculate BLOOM to mint for attestation
    pub fn calculate_bloom_reward(&self, attestation: &ImpactAttestation) -> u128 {
        // BLOOM rate: 0.001 per impact point (10^15 base units)
        const BLOOM_RATE: u128 = 1_000_000_000_000_000;
        let final_score = attestation.final_score() as u128;
        final_score * BLOOM_RATE
    }

    /// Select validators for attestation (weighted by reputation and stake)
    pub fn select_validators(&self, count: usize) -> Vec<[u8; 32]> {
        let mut eligible: Vec<_> = self
            .validators
            .values()
            .filter(|v| v.is_eligible())
            .collect();

        // Sort by reputation * stake (weighted selection)
        eligible.sort_by(|a, b| {
            let weight_a = (a.reputation as u128) * a.stake;
            let weight_b = (b.reputation as u128) * b.stake;
            weight_b.cmp(&weight_a)
        });

        eligible
            .into_iter()
            .take(count)
            .map(|v| v.address)
            .collect()
    }
}

/// Consensus errors
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConsensusError {
    /// Validator not registered
    UnknownValidator,
    /// Validator doesn't meet requirements
    IneligibleValidator,
    /// Validator already exists
    ValidatorExists,
    /// Insufficient stake
    InsufficientStake,
    /// Attestation not found
    AttestationNotFound,
    /// Duplicate attestation
    DuplicateAttestation,
    /// Duplicate validation attempt
    DuplicateValidation,
    /// Below minimum impact score
    BelowMinimumImpact,
    /// Insufficient quorum
    InsufficientQuorum,
    /// Invalid signature
    InvalidSignature,
    /// Attestation expired
    AttestationExpired,
}

impl std::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownValidator => write!(f, "Unknown validator"),
            Self::IneligibleValidator => write!(f, "Validator not eligible"),
            Self::ValidatorExists => write!(f, "Validator already registered"),
            Self::InsufficientStake => write!(f, "Insufficient stake"),
            Self::AttestationNotFound => write!(f, "Attestation not found"),
            Self::DuplicateAttestation => write!(f, "Duplicate attestation"),
            Self::DuplicateValidation => write!(f, "Duplicate validation"),
            Self::BelowMinimumImpact => write!(f, "Impact below minimum"),
            Self::InsufficientQuorum => write!(f, "Insufficient quorum"),
            Self::InvalidSignature => write!(f, "Invalid signature"),
            Self::AttestationExpired => write!(f, "Attestation expired"),
        }
    }
}

impl std::error::Error for ConsensusError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator(id: u8) -> PoIValidator {
        PoIValidator::new(
            [id; 32],
            PoIValidator::MIN_STAKE * 2,
            0,
        )
    }

    fn create_test_attestation(attester: u8, beneficiary: u8) -> ImpactAttestation {
        ImpactAttestation::new(
            [attester; 32],
            [beneficiary; 32],
            ImpactCategory::Technical,
            100,
            [0u8; 32],
            0,
        )
    }

    #[test]
    fn test_impact_category_multipliers() {
        assert_eq!(ImpactCategory::Healthcare.multiplier(), 15000);
        assert_eq!(ImpactCategory::Technical.multiplier(), 14000);
        assert_eq!(ImpactCategory::Governance.multiplier(), 10000);
    }

    #[test]
    fn test_attestation_final_score() {
        let attestation = ImpactAttestation::new(
            [1u8; 32],
            [2u8; 32],
            ImpactCategory::Healthcare, // 1.5x
            100,
            [0u8; 32],
            0,
        );
        assert_eq!(attestation.final_score(), 150);
    }

    #[test]
    fn test_validator_eligibility() {
        let validator = create_test_validator(1);
        assert!(validator.is_eligible());

        let mut low_stake = validator.clone();
        low_stake.stake = 0;
        assert!(!low_stake.is_eligible());

        let mut low_rep = validator.clone();
        low_rep.reputation = 10;
        assert!(!low_rep.is_eligible());
    }

    #[test]
    fn test_consensus_registration() {
        let mut consensus = PoIConsensus::new(3);
        let validator = create_test_validator(1);

        assert!(consensus.register_validator(validator.clone()).is_ok());
        assert_eq!(consensus.validator_count(), 1);

        // Duplicate should fail
        assert!(consensus.register_validator(validator).is_err());
    }

    #[test]
    fn test_attestation_submission() {
        let mut consensus = PoIConsensus::new(3);
        let attestation = create_test_attestation(1, 2);

        let id = consensus.submit_attestation(attestation.clone()).unwrap();
        assert_eq!(id.len(), 32);

        // Duplicate should fail
        assert!(consensus.submit_attestation(attestation).is_err());
    }

    #[test]
    fn test_quorum_reached() {
        let mut consensus = PoIConsensus::new(3);

        // Register validators
        for i in 1..=5 {
            consensus
                .register_validator(create_test_validator(i))
                .unwrap();
        }

        // Submit attestation
        let attestation = create_test_attestation(1, 2);
        let id = consensus.submit_attestation(attestation).unwrap();

        // Add validations
        for i in 1..=3 {
            let approval = ValidatorApproval {
                validator: [i; 32],
                approved: true,
                rejection_reason: None,
                timestamp: 0,
                signature: [0u8; 64],
            };
            let quorum_reached = consensus.add_validation(&id, approval).unwrap();
            if i == 3 {
                assert!(quorum_reached);
            }
        }
    }

    #[test]
    fn test_process_attestations() {
        let mut consensus = PoIConsensus::new(2);

        // Register validators
        for i in 1..=3 {
            consensus
                .register_validator(create_test_validator(i))
                .unwrap();
        }

        // Submit and validate attestation
        let attestation = create_test_attestation(1, 2);
        let id = consensus.submit_attestation(attestation).unwrap();

        for i in 1..=2 {
            let approval = ValidatorApproval {
                validator: [i; 32],
                approved: true,
                rejection_reason: None,
                timestamp: 0,
                signature: [0u8; 64],
            };
            consensus.add_validation(&id, approval).unwrap();
        }

        // Process
        let approved = consensus.process_attestations(0);
        assert_eq!(approved.len(), 1);
        assert!(consensus.total_impact() > 0);
    }

    #[test]
    fn test_bloom_reward_calculation() {
        let consensus = PoIConsensus::new(3);
        let attestation = ImpactAttestation::new(
            [1u8; 32],
            [2u8; 32],
            ImpactCategory::Technical, // 1.4x
            100,
            [0u8; 32],
            0,
        );

        let reward = consensus.calculate_bloom_reward(&attestation);
        // 100 * 1.4 = 140 impact points * 10^15 = 140 * 10^15
        assert_eq!(reward, 140 * 1_000_000_000_000_000);
    }
}
