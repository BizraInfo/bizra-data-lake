// src/sovereign/mod.rs - BIZRA Sovereign Runtime Architecture
//
// # PEAK MASTERPIECE IMPLEMENTATION - NODE0 GENESIS
//
// Standing on the Shoulders of Giants:
// - Kiayias et al. (2024): Dual-Token Tokenomics
// - Szu-Hartley (1987): Fast Simulated Annealing
// - Langevin: Stochastic Dynamics
// - Lyapunov: Stability Theory
// - Shannon: Information Theory (SNR)
//
// ## Architecture
// ```
// ┌─────────────────────────────────────────────────────────────────┐
// │                    BIZRA SOVEREIGN RUNTIME                      │
// ├─────────────────────────────────────────────────────────────────┤
// │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
// │  │  SEED    │  │  BLOOM   │  │   PoI    │  │ Thermal  │       │
// │  │  Token   │──│  Token   │──│Consensus │──│  Engine  │       │
// │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
// │       │              │             │              │            │
// │       └──────────────┴─────────────┴──────────────┘            │
// │                           │                                     │
// │                    ┌──────┴──────┐                              │
// │                    │   NODE0     │                              │
// │                    │  Sovereign  │                              │
// │                    │   Runtime   │                              │
// │                    └─────────────┘                              │
// │                           │                                     │
// │  ┌──────────┐  ┌──────────┴──────────┐  ┌──────────┐          │
// │  │  Block   │  │    Reconciler       │  │ Resource │          │
// │  │  Graph   │──│   (PAT↔SAT)         │──│   Pool   │          │
// │  └──────────┘  └─────────────────────┘  └──────────┘          │
// └─────────────────────────────────────────────────────────────────┘
// ```
//
// إحسان Quality Standard: 99.0

pub mod consensus;
pub mod network;
pub mod node;
pub mod thermal;

pub use consensus::{ImpactAttestation, ImpactCategory, PoIConsensus, PoIValidator};
pub use network::{NetworkMultiplier, ResourcePool, ReverseScaling};
pub use node::{Node0, Node0Config, Node0Status, NodeRole};
pub use thermal::{Reconciler, ReconcilerMode, ThermalConsciousness};

use crate::blockchain::tokens::SeedToken;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Genesis hash from BIZRA_TOKENOMICS_GENESIS.yaml
pub const GENESIS_HASH: &str = "9dfa0bd5375ee06120e72c04618b407b2cf184f110075a573984a4b185f25974";

/// BIZRA Chain ID
pub const CHAIN_ID: u64 = 727866;

/// إحسان (Excellence) threshold for sovereign operations
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// Gini coefficient maximum (anti-concentration)
pub const GINI_MAX: f64 = 0.35;

/// Minimum SNR for network boost
pub const SNR_MIN_BOOST: f64 = 0.95;

/// Compute integrity hash for sovereign state
pub fn compute_integrity_hash(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    format!("sha256:{:x}", hash)
}

/// Sovereign state snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SovereignState {
    /// Current epoch
    pub epoch: u64,
    /// Total SEED supply
    pub seed_supply: u128,
    /// Total BLOOM supply
    pub bloom_supply: u128,
    /// Total impact accumulated
    pub total_impact: u128,
    /// Network efficiency multiplier
    pub network_multiplier: f64,
    /// Current Gini coefficient
    pub gini_coefficient: f64,
    /// إحسان score
    pub ihsan_score: f64,
    /// Active validators
    pub validator_count: usize,
    /// Resource pool size
    pub resource_nodes: usize,
    /// State hash
    pub state_hash: String,
}

impl SovereignState {
    /// Create genesis state
    pub fn genesis() -> Self {
        Self {
            epoch: 0,
            seed_supply: SeedToken::DEFAULT_INITIAL_SUPPLY as u128
                * crate::blockchain::tokens::TokenAmount::BASE_MULTIPLIER,
            bloom_supply: 0,
            total_impact: 0,
            network_multiplier: 1.0,
            gini_coefficient: 0.0,
            ihsan_score: 1.0,
            validator_count: 0,
            resource_nodes: 0,
            state_hash: GENESIS_HASH.to_string(),
        }
    }

    /// Check if state satisfies invariants
    pub fn check_invariants(&self) -> InvariantResult {
        let mut violations = Vec::new();

        if self.gini_coefficient > GINI_MAX {
            violations.push(format!(
                "Gini coefficient {} exceeds maximum {}",
                self.gini_coefficient, GINI_MAX
            ));
        }

        if self.ihsan_score < IHSAN_THRESHOLD {
            violations.push(format!(
                "Ihsān score {} below threshold {}",
                self.ihsan_score, IHSAN_THRESHOLD
            ));
        }

        InvariantResult {
            satisfied: violations.is_empty(),
            violations,
        }
    }
}

/// Result of invariant check
#[derive(Clone, Debug)]
pub struct InvariantResult {
    /// Whether all invariants are satisfied
    pub satisfied: bool,
    /// List of violations
    pub violations: Vec<String>,
}

/// Sovereign transaction types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SovereignTransaction {
    /// Transfer SEED tokens
    SeedTransfer {
        from: [u8; 32],
        to: [u8; 32],
        amount: u128,
    },
    /// Stake SEED for validation
    Stake { staker: [u8; 32], amount: u128 },
    /// Unstake SEED
    Unstake { staker: [u8; 32], amount: u128 },
    /// Submit impact attestation
    ImpactAttest {
        attester: [u8; 32],
        beneficiary: [u8; 32],
        category: u8,
        base_score: u64,
        evidence_hash: [u8; 32],
    },
    /// Cast governance vote
    Vote {
        voter: [u8; 32],
        proposal_id: [u8; 32],
        support: bool,
        bloom_weight: u128,
    },
    /// Contribute resources
    ResourceContribute {
        node: [u8; 32],
        compute: u64,
        storage: u64,
        bandwidth: u64,
    },
    /// Claim staking rewards
    ClaimRewards { staker: [u8; 32] },
    /// Claim BLOOM from impact
    ClaimBloom {
        beneficiary: [u8; 32],
        attestation_id: [u8; 32],
    },
}

impl SovereignTransaction {
    /// Serialize transaction for hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        match self {
            Self::SeedTransfer { from, to, amount } => {
                bytes.push(0);
                bytes.extend_from_slice(from);
                bytes.extend_from_slice(to);
                bytes.extend_from_slice(&amount.to_le_bytes());
            }
            Self::Stake { staker, amount } => {
                bytes.push(1);
                bytes.extend_from_slice(staker);
                bytes.extend_from_slice(&amount.to_le_bytes());
            }
            Self::Unstake { staker, amount } => {
                bytes.push(2);
                bytes.extend_from_slice(staker);
                bytes.extend_from_slice(&amount.to_le_bytes());
            }
            Self::ImpactAttest {
                attester,
                beneficiary,
                category,
                base_score,
                evidence_hash,
            } => {
                bytes.push(3);
                bytes.extend_from_slice(attester);
                bytes.extend_from_slice(beneficiary);
                bytes.push(*category);
                bytes.extend_from_slice(&base_score.to_le_bytes());
                bytes.extend_from_slice(evidence_hash);
            }
            Self::Vote {
                voter,
                proposal_id,
                support,
                bloom_weight,
            } => {
                bytes.push(4);
                bytes.extend_from_slice(voter);
                bytes.extend_from_slice(proposal_id);
                bytes.push(if *support { 1 } else { 0 });
                bytes.extend_from_slice(&bloom_weight.to_le_bytes());
            }
            Self::ResourceContribute {
                node,
                compute,
                storage,
                bandwidth,
            } => {
                bytes.push(5);
                bytes.extend_from_slice(node);
                bytes.extend_from_slice(&compute.to_le_bytes());
                bytes.extend_from_slice(&storage.to_le_bytes());
                bytes.extend_from_slice(&bandwidth.to_le_bytes());
            }
            Self::ClaimRewards { staker } => {
                bytes.push(6);
                bytes.extend_from_slice(staker);
            }
            Self::ClaimBloom {
                beneficiary,
                attestation_id,
            } => {
                bytes.push(7);
                bytes.extend_from_slice(beneficiary);
                bytes.extend_from_slice(attestation_id);
            }
        }

        bytes
    }

    /// Compute transaction hash
    pub fn hash(&self) -> [u8; 32] {
        let hash = Sha256::digest(self.to_bytes());
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        result
    }
}

/// Sovereign execution result
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    /// Transaction hash
    pub tx_hash: [u8; 32],
    /// Success status
    pub success: bool,
    /// State changes
    pub state_changes: Vec<StateChange>,
    /// Events emitted
    pub events: Vec<SovereignEvent>,
    /// Error message if failed
    pub error: Option<String>,
}

/// State change from transaction execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StateChange {
    /// Balance change
    BalanceChange {
        account: [u8; 32],
        token: String,
        delta: i128,
    },
    /// Stake change
    StakeChange { account: [u8; 32], delta: i128 },
    /// Impact recorded
    ImpactRecorded {
        account: [u8; 32],
        score: u64,
        category: u8,
    },
    /// BLOOM minted
    BloomMinted { account: [u8; 32], amount: u128 },
}

/// Sovereign events
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SovereignEvent {
    /// Transfer event
    Transfer {
        from: [u8; 32],
        to: [u8; 32],
        amount: u128,
        token: String,
    },
    /// Stake event
    Staked { staker: [u8; 32], amount: u128 },
    /// Unstake event
    Unstaked { staker: [u8; 32], amount: u128 },
    /// Impact attested
    ImpactAttested {
        attester: [u8; 32],
        beneficiary: [u8; 32],
        score: u64,
        category: u8,
    },
    /// BLOOM minted
    BloomMinted {
        beneficiary: [u8; 32],
        amount: u128,
        impact_score: u64,
    },
    /// Rewards claimed
    RewardsClaimed {
        staker: [u8; 32],
        amount: u128,
        epochs: u64,
    },
    /// Vote cast
    VoteCast {
        voter: [u8; 32],
        proposal: [u8; 32],
        support: bool,
        weight: u128,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_state() {
        let state = SovereignState::genesis();
        assert_eq!(state.epoch, 0);
        assert!(state.seed_supply > 0);
        assert_eq!(state.bloom_supply, 0);
    }

    #[test]
    fn test_invariants_satisfied() {
        let state = SovereignState::genesis();
        let result = state.check_invariants();
        assert!(result.satisfied);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_invariants_violated() {
        let mut state = SovereignState::genesis();
        state.gini_coefficient = 0.5; // Exceeds GINI_MAX
        state.ihsan_score = 0.8; // Below IHSAN_THRESHOLD

        let result = state.check_invariants();
        assert!(!result.satisfied);
        assert_eq!(result.violations.len(), 2);
    }

    #[test]
    fn test_transaction_hash() {
        let tx = SovereignTransaction::SeedTransfer {
            from: [1u8; 32],
            to: [2u8; 32],
            amount: 1000,
        };
        let hash = tx.hash();
        assert_eq!(hash.len(), 32);
    }
}
