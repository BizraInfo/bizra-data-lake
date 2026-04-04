// src/sovereign/node.rs - BIZRA Node0 Sovereign Runtime
//
// # PEAK MASTERPIECE IMPLEMENTATION - UNIFIED RUNTIME
//
// Standing on the Shoulders of Giants:
// - Nakamoto (2008): Decentralized Consensus
// - Buterin (2014): Programmable State Machines
// - Kiayias et al. (2024): Dual-Token Tokenomics
// - Garay et al.: Common Prefix & Chain Quality
// - Lyapunov: Stability Theory
//
// ## Architecture
// ```
// ┌─────────────────────────────────────────────────────────────────┐
// │                       NODE0 SOVEREIGN RUNTIME                    │
// ├─────────────────────────────────────────────────────────────────┤
// │                                                                  │
// │    ┌──────────────────────────────────────────────────────┐     │
// │    │                   EXECUTION LAYER                     │     │
// │    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │     │
// │    │  │ Process │  │ Validate│  │ Execute │  │  Emit   │  │     │
// │    │  │   Tx    │──│ Ihsān   │──│ State Δ │──│ Receipt │  │     │
// │    │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │     │
// │    └──────────────────────────────────────────────────────┘     │
// │                              │                                   │
// │    ┌──────────────────────────────────────────────────────┐     │
// │    │                  CONSENSUS LAYER                      │     │
// │    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │     │
// │    │  │  PoI    │  │  SAT    │  │ Thermal │  │ Network │  │     │
// │    │  │Consensus│──│ Quorum  │──│ Annealer│──│ Multiplr│  │     │
// │    │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │     │
// │    └──────────────────────────────────────────────────────┘     │
// │                              │                                   │
// │    ┌──────────────────────────────────────────────────────┐     │
// │    │                    STATE LAYER                        │     │
// │    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │     │
// │    │  │ Account │  │ Token   │  │ Impact  │  │ Resource│  │     │
// │    │  │  State  │──│ Ledger  │──│ Registry│──│   Pool  │  │     │
// │    │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │     │
// │    └──────────────────────────────────────────────────────┘     │
// │                                                                  │
// └─────────────────────────────────────────────────────────────────┘
// ```
//
// إحسان Quality Standard: 99.0

use crate::blockchain::tokens::{BloomToken, SeedToken, TokenAccount, TokenAmount};
use crate::sovereign::consensus::{ImpactCategory, PoIConsensus};
use crate::sovereign::network::{NetworkMultiplier, ResourcePool, ReverseScaling};
use crate::sovereign::thermal::{Reconciler, ReconcilerMode, ThermalConsciousness};
use crate::sovereign::{
    ExecutionResult, SovereignEvent, SovereignState, SovereignTransaction, StateChange, GINI_MAX,
    IHSAN_THRESHOLD,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Node role in the network
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Genesis node (NODE0) - special privileges
    Genesis,
    /// Validator node - participates in PoI consensus
    Validator,
    /// Resource node - contributes compute/storage
    Resource,
    /// Light node - query-only, no consensus
    Light,
}

/// Node0 configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node0Config {
    /// Node role
    pub role: NodeRole,
    /// Minimum stake for validation (in tokens)
    pub min_stake: u64,
    /// SAT quorum size (3/5 by default)
    pub sat_quorum: usize,
    /// Maximum transactions per block
    pub max_tx_per_block: usize,
    /// Block time in milliseconds
    pub block_time_ms: u64,
    /// Enable thermal annealing
    pub enable_thermal: bool,
    /// Enable network multiplier
    pub enable_network_boost: bool,
}

impl Default for Node0Config {
    fn default() -> Self {
        Self {
            role: NodeRole::Validator,
            min_stake: 10_000,
            sat_quorum: 3,
            max_tx_per_block: 1000,
            block_time_ms: 5000,
            enable_thermal: true,
            enable_network_boost: true,
        }
    }
}

/// Node0 status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node0Status {
    /// Current epoch
    pub epoch: u64,
    /// Block height
    pub block_height: u64,
    /// Is synced with network
    pub synced: bool,
    /// Number of pending transactions
    pub pending_tx: usize,
    /// Current Ihsān score
    pub ihsan_score: f64,
    /// Current Gini coefficient
    pub gini: f64,
    /// Network multiplier
    pub network_multiplier: f64,
    /// Thermal temperature
    pub temperature: f64,
    /// Active validators
    pub validators: usize,
    /// Resource pool utilization
    pub resource_utilization: f64,
}

/// Node0 Sovereign Runtime
///
/// The unified runtime that orchestrates:
/// - Transaction processing
/// - Ihsān gate enforcement
/// - PoI consensus
/// - Thermal annealing
/// - Network multiplier
/// - State management
pub struct Node0 {
    /// Configuration
    config: Node0Config,
    /// Current state
    state: SovereignState,
    /// SEED token
    seed: SeedToken,
    /// BLOOM token
    bloom: BloomToken,
    /// Token accounts
    accounts: HashMap<[u8; 32], TokenAccount>,
    /// PoI consensus engine
    consensus: PoIConsensus,
    /// Thermal consciousness
    thermal: ThermalConsciousness,
    /// PAT-SAT reconciler
    reconciler: Reconciler,
    /// Network multiplier
    network: NetworkMultiplier,
    /// Reverse scaling for costs
    scaling: ReverseScaling,
    /// Resource pool
    resources: ResourcePool,
    /// Pending transactions
    pending_tx: Vec<SovereignTransaction>,
    /// Executed receipts (tx_hash -> result)
    receipts: HashMap<[u8; 32], ExecutionResult>,
    /// Block height
    block_height: u64,
}

impl Node0 {
    /// Create new Node0 runtime with genesis state
    pub fn genesis(config: Node0Config) -> Self {
        let state = SovereignState::genesis();
        // Use 1D thermal engine for Gini optimization
        let thermal = ThermalConsciousness::default_config(1, 1.0);
        let reconciler = Reconciler::new(ReconcilerMode::Balanced, 3, 5);

        Self {
            config,
            state,
            seed: SeedToken::new(),
            bloom: BloomToken::new(),
            accounts: HashMap::new(),
            consensus: PoIConsensus::new(3),
            thermal,
            reconciler,
            network: NetworkMultiplier::new(0, 0.0, 1.0),
            scaling: ReverseScaling::new(100.0, 10.0),
            resources: ResourcePool::new(),
            pending_tx: Vec::new(),
            receipts: HashMap::new(),
            block_height: 0,
        }
    }

    /// Get current status
    pub fn status(&self) -> Node0Status {
        Node0Status {
            epoch: self.state.epoch,
            block_height: self.block_height,
            synced: true, // Simplified for now
            pending_tx: self.pending_tx.len(),
            ihsan_score: self.state.ihsan_score,
            gini: self.state.gini_coefficient,
            network_multiplier: self.network.get_multiplier(),
            temperature: self.thermal.temperature(),
            validators: self.state.validator_count,
            resource_utilization: self.resources.utilization(),
        }
    }

    /// Submit transaction to pending pool
    pub fn submit_transaction(&mut self, tx: SovereignTransaction) -> [u8; 32] {
        let hash = tx.hash();
        self.pending_tx.push(tx);
        hash
    }

    /// Execute a single transaction with full validation
    pub fn execute_transaction(&mut self, tx: SovereignTransaction) -> ExecutionResult {
        let tx_hash = tx.hash();

        // Ihsān gate check
        if self.state.ihsan_score < IHSAN_THRESHOLD {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some(format!(
                    "Ihsān score {} below threshold {}",
                    self.state.ihsan_score, IHSAN_THRESHOLD
                )),
            };
        }

        // Execute based on transaction type
        let result = match &tx {
            SovereignTransaction::SeedTransfer { from, to, amount } => {
                self.execute_seed_transfer(*from, *to, *amount)
            }
            SovereignTransaction::Stake { staker, amount } => self.execute_stake(*staker, *amount),
            SovereignTransaction::Unstake { staker, amount } => {
                self.execute_unstake(*staker, *amount)
            }
            SovereignTransaction::ImpactAttest {
                attester,
                beneficiary,
                category,
                base_score,
                evidence_hash,
            } => self.execute_impact_attestation(
                *attester,
                *beneficiary,
                *category,
                *base_score,
                *evidence_hash,
            ),
            SovereignTransaction::Vote {
                voter,
                proposal_id,
                support,
                bloom_weight,
            } => self.execute_vote(*voter, *proposal_id, *support, *bloom_weight),
            SovereignTransaction::ResourceContribute {
                node,
                compute,
                storage,
                bandwidth,
            } => self.execute_resource_contribution(*node, *compute, *storage, *bandwidth),
            SovereignTransaction::ClaimRewards { staker } => self.execute_claim_rewards(*staker),
            SovereignTransaction::ClaimBloom {
                beneficiary,
                attestation_id,
            } => self.execute_claim_bloom(*beneficiary, *attestation_id),
        };

        // Store receipt
        self.receipts.insert(tx_hash, result.clone());

        result
    }

    /// Execute SEED transfer
    fn execute_seed_transfer(
        &mut self,
        from: [u8; 32],
        to: [u8; 32],
        amount: u128,
    ) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"seed_transfer", &from, &to, amount);
        let token_amount = TokenAmount::from_raw(amount);

        // Get or create accounts
        let from_account = self.accounts.entry(from).or_default();

        // Check balance
        if from_account.seed_balance < token_amount {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some("Insufficient SEED balance".to_string()),
            };
        }

        // Execute transfer
        from_account.seed_balance = TokenAmount::from_raw(from_account.seed_balance.raw() - amount);

        let to_account = self.accounts.entry(to).or_default();
        to_account.seed_balance = TokenAmount::from_raw(to_account.seed_balance.raw() + amount);

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![
                StateChange::BalanceChange {
                    account: from,
                    token: "SEED".to_string(),
                    delta: -(amount as i128),
                },
                StateChange::BalanceChange {
                    account: to,
                    token: "SEED".to_string(),
                    delta: amount as i128,
                },
            ],
            events: vec![SovereignEvent::Transfer {
                from,
                to,
                amount,
                token: "SEED".to_string(),
            }],
            error: None,
        }
    }

    /// Execute stake operation
    fn execute_stake(&mut self, staker: [u8; 32], amount: u128) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"stake", &staker, &[0u8; 32], amount);
        let token_amount = TokenAmount::from_raw(amount);

        let account = self.accounts.entry(staker).or_default();

        // Check balance
        if account.seed_balance < token_amount {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some("Insufficient SEED balance for staking".to_string()),
            };
        }

        // Move to staked
        account.seed_balance = TokenAmount::from_raw(account.seed_balance.raw() - amount);
        account.staked = TokenAmount::from_raw(account.staked.raw() + amount);

        // Check if meets validator minimum
        let min_stake = TokenAmount::from_tokens(self.config.min_stake);
        if account.staked >= min_stake {
            self.state.validator_count += 1;
        }

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![StateChange::StakeChange {
                account: staker,
                delta: amount as i128,
            }],
            events: vec![SovereignEvent::Staked { staker, amount }],
            error: None,
        }
    }

    /// Execute unstake operation
    fn execute_unstake(&mut self, staker: [u8; 32], amount: u128) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"unstake", &staker, &[0u8; 32], amount);

        let account = self.accounts.entry(staker).or_default();

        // Check staked balance
        if account.staked.raw() < amount {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some("Insufficient staked SEED".to_string()),
            };
        }

        // Check if this would drop below validator minimum
        let min_stake = TokenAmount::from_tokens(self.config.min_stake);
        let remaining = account.staked.raw() - amount;
        if account.staked >= min_stake && TokenAmount::from_raw(remaining) < min_stake {
            self.state.validator_count = self.state.validator_count.saturating_sub(1);
        }

        // Move from staked to available
        account.staked = TokenAmount::from_raw(remaining);
        account.seed_balance = TokenAmount::from_raw(account.seed_balance.raw() + amount);

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![StateChange::StakeChange {
                account: staker,
                delta: -(amount as i128),
            }],
            events: vec![SovereignEvent::Unstaked { staker, amount }],
            error: None,
        }
    }

    /// Execute impact attestation
    fn execute_impact_attestation(
        &mut self,
        attester: [u8; 32],
        beneficiary: [u8; 32],
        category: u8,
        base_score: u64,
        evidence_hash: [u8; 32],
    ) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"impact", &attester, &beneficiary, base_score as u128);

        // Validate attester is a validator
        let attester_account = self.accounts.get(&attester);
        let min_stake = TokenAmount::from_tokens(self.config.min_stake);

        let is_validator = attester_account
            .map(|a| a.staked >= min_stake)
            .unwrap_or(false);

        if !is_validator {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some("Attester is not a validator".to_string()),
            };
        }

        // Create attestation
        let impact_category = match category {
            0 => ImpactCategory::Education,
            1 => ImpactCategory::Healthcare,
            2 => ImpactCategory::Environment,
            3 => ImpactCategory::Economic,
            4 => ImpactCategory::Governance,
            5 => ImpactCategory::Technical,
            _ => ImpactCategory::Community,
        };

        // Apply network multiplier to score
        let multiplied_score = (base_score as f64
            * self.network.get_multiplier()
            * impact_category.multiplier() as f64
            / 10000.0) as u64;

        // Record impact
        self.state.total_impact += multiplied_score as u128;

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![StateChange::ImpactRecorded {
                account: beneficiary,
                score: multiplied_score,
                category,
            }],
            events: vec![SovereignEvent::ImpactAttested {
                attester,
                beneficiary,
                score: multiplied_score,
                category,
            }],
            error: None,
        }
    }

    /// Execute governance vote
    fn execute_vote(
        &mut self,
        voter: [u8; 32],
        proposal_id: [u8; 32],
        support: bool,
        bloom_weight: u128,
    ) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"vote", &voter, &proposal_id, bloom_weight);

        // Check voter has BLOOM
        let account = self.accounts.get(&voter);
        let has_bloom = account
            .map(|a| a.bloom_balance.raw() >= bloom_weight)
            .unwrap_or(false);

        if !has_bloom {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some("Insufficient BLOOM for voting".to_string()),
            };
        }

        // Calculate quadratic voting weight
        let voting_power = self.bloom.voting_power(TokenAmount::from_raw(bloom_weight));

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![],
            events: vec![SovereignEvent::VoteCast {
                voter,
                proposal: proposal_id,
                support,
                weight: voting_power as u128,
            }],
            error: None,
        }
    }

    /// Execute resource contribution
    fn execute_resource_contribution(
        &mut self,
        node: [u8; 32],
        compute: u64,
        storage: u64,
        bandwidth: u64,
    ) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"resource", &node, &[0u8; 32], compute as u128);

        use crate::sovereign::network::NodeContribution;

        let contribution = NodeContribution {
            compute,
            storage,
            bandwidth,
            uptime: 100,
            timestamp: 0, // Would be actual timestamp
        };

        self.resources.add_contribution(node, contribution);
        self.state.resource_nodes += 1;

        // Update network multiplier based on new node count
        self.network.update(
            self.resources.contributor_count(),
            self.state.gini_coefficient,
            self.state.ihsan_score,
        );
        self.scaling.update(self.resources.contributor_count());

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![],
            events: vec![],
            error: None,
        }
    }

    /// Execute reward claim
    fn execute_claim_rewards(&mut self, staker: [u8; 32]) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"claim_rewards", &staker, &[0u8; 32], 0);

        let account = self.accounts.entry(staker).or_default();

        // Calculate rewards (simplified: 5% APY, per epoch)
        let staked = account.staked.raw();
        if staked == 0 {
            return ExecutionResult {
                tx_hash,
                success: false,
                state_changes: vec![],
                events: vec![],
                error: Some("No staked tokens".to_string()),
            };
        }

        // Apply network multiplier to rewards
        let base_reward = staked / 20 / 365; // ~5% APY per day
        let multiplied_reward = (base_reward as f64 * self.network.get_multiplier()) as u128;

        // Mint rewards
        account.seed_balance =
            TokenAmount::from_raw(account.seed_balance.raw() + multiplied_reward);
        self.state.seed_supply += multiplied_reward;

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![StateChange::BalanceChange {
                account: staker,
                token: "SEED".to_string(),
                delta: multiplied_reward as i128,
            }],
            events: vec![SovereignEvent::RewardsClaimed {
                staker,
                amount: multiplied_reward,
                epochs: 1,
            }],
            error: None,
        }
    }

    /// Execute BLOOM claim from impact
    fn execute_claim_bloom(
        &mut self,
        beneficiary: [u8; 32],
        _attestation_id: [u8; 32],
    ) -> ExecutionResult {
        let tx_hash = self.compute_op_hash(b"claim_bloom", &beneficiary, &[0u8; 32], 0);

        // Simplified: mint fixed BLOOM amount
        // In full implementation, would look up attestation and calculate
        let bloom_amount = TokenAmount::from_tokens(100);

        let account = self.accounts.entry(beneficiary).or_default();

        account.bloom_balance =
            TokenAmount::from_raw(account.bloom_balance.raw() + bloom_amount.raw());
        self.state.bloom_supply += bloom_amount.raw();

        ExecutionResult {
            tx_hash,
            success: true,
            state_changes: vec![StateChange::BloomMinted {
                account: beneficiary,
                amount: bloom_amount.raw(),
            }],
            events: vec![SovereignEvent::BloomMinted {
                beneficiary,
                amount: bloom_amount.raw(),
                impact_score: 100,
            }],
            error: None,
        }
    }

    /// Process pending transactions into a block
    pub fn process_block(&mut self) -> Vec<ExecutionResult> {
        let txs: Vec<_> = self
            .pending_tx
            .drain(..self.config.max_tx_per_block.min(self.pending_tx.len()))
            .collect();

        let results: Vec<_> = txs
            .into_iter()
            .map(|tx| self.execute_transaction(tx))
            .collect();

        self.block_height += 1;

        // Thermal step for optimization
        if self.config.enable_thermal {
            let gradient = vec![self.state.gini_coefficient - GINI_MAX];
            let energy = (self.state.gini_coefficient - GINI_MAX).abs();
            self.thermal.step(&gradient, energy);
        }

        // Update reconciler
        self.reconciler.step(self.state.ihsan_score);

        results
    }

    /// Advance to next epoch
    pub fn advance_epoch(&mut self) {
        self.state.epoch += 1;

        // Recalculate Gini coefficient
        self.recalculate_gini();

        // Update network state
        self.network.update(
            self.resources.contributor_count(),
            self.state.gini_coefficient,
            self.state.ihsan_score,
        );
        self.scaling.update(self.resources.contributor_count());

        // Check invariants
        let invariants = self.state.check_invariants();
        if !invariants.satisfied {
            // In production, this would trigger recovery mechanisms
            self.state.ihsan_score *= 0.99; // Slight penalty
        }
    }

    /// Recalculate Gini coefficient from account balances
    fn recalculate_gini(&mut self) {
        let mut balances: Vec<f64> = self
            .accounts
            .values()
            .map(|a| a.seed_balance.raw() as f64 + a.staked.raw() as f64)
            .filter(|b| *b > 0.0)
            .collect();

        if balances.is_empty() {
            self.state.gini_coefficient = 0.0;
            return;
        }

        balances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = balances.len() as f64;
        let sum: f64 = balances.iter().sum();

        if sum == 0.0 {
            self.state.gini_coefficient = 0.0;
            return;
        }

        let weighted_sum: f64 = balances
            .iter()
            .enumerate()
            .map(|(i, b)| (2.0 * (i as f64 + 1.0) - n - 1.0) * b)
            .sum();

        self.state.gini_coefficient = (weighted_sum / (n * sum)).abs();
    }

    /// Compute operation hash for receipts
    fn compute_op_hash(
        &self,
        op: &[u8],
        account1: &[u8; 32],
        account2: &[u8; 32],
        amount: u128,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(op);
        hasher.update(account1);
        hasher.update(account2);
        hasher.update(amount.to_le_bytes());
        hasher.update(self.block_height.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Get account balance
    pub fn get_account(&self, address: &[u8; 32]) -> Option<&TokenAccount> {
        self.accounts.get(address)
    }

    /// Get current state
    pub fn state(&self) -> &SovereignState {
        &self.state
    }

    /// Get network multiplier
    pub fn network_multiplier(&self) -> f64 {
        self.network.get_multiplier()
    }

    /// Get effective cost (after reverse scaling)
    pub fn effective_cost(&self) -> f64 {
        self.scaling.get_cost()
    }

    /// Check if system is stable (Lyapunov)
    pub fn is_stable(&self) -> bool {
        self.thermal.is_stable() && self.reconciler.is_stable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node() -> Node0 {
        Node0::genesis(Node0Config::default())
    }

    #[test]
    fn test_genesis_creation() {
        let node = create_test_node();
        assert_eq!(node.state.epoch, 0);
        assert_eq!(node.block_height, 0);
        assert!(node.state.ihsan_score >= IHSAN_THRESHOLD);
    }

    #[test]
    fn test_seed_transfer() {
        let mut node = create_test_node();

        // Fund account
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let amount = TokenAmount::from_tokens(1000);

        node.accounts
            .insert(sender, TokenAccount::with_seed(amount));

        // Execute transfer
        let tx = SovereignTransaction::SeedTransfer {
            from: sender,
            to: receiver,
            amount: TokenAmount::from_tokens(500).raw(),
        };

        let result = node.execute_transaction(tx);
        assert!(result.success);

        // Verify balances
        let sender_balance = node.accounts.get(&sender).unwrap().seed_balance;
        let receiver_balance = node.accounts.get(&receiver).unwrap().seed_balance;

        assert_eq!(sender_balance.raw(), TokenAmount::from_tokens(500).raw());
        assert_eq!(receiver_balance.raw(), TokenAmount::from_tokens(500).raw());
    }

    #[test]
    fn test_staking() {
        let mut node = create_test_node();

        let staker = [1u8; 32];
        let stake_amount = TokenAmount::from_tokens(20_000);

        // Fund account
        node.accounts
            .insert(staker, TokenAccount::with_seed(stake_amount));

        // Stake
        let tx = SovereignTransaction::Stake {
            staker,
            amount: stake_amount.raw(),
        };

        let result = node.execute_transaction(tx);
        assert!(result.success);

        // Verify staked balance
        let account = node.accounts.get(&staker).unwrap();
        assert_eq!(account.staked.raw(), stake_amount.raw());
        assert_eq!(account.seed_balance.raw(), 0);

        // Should be validator now (meets min stake)
        assert_eq!(node.state.validator_count, 1);
    }

    #[test]
    fn test_impact_attestation() {
        let mut node = create_test_node();

        let attester = [1u8; 32];
        let beneficiary = [2u8; 32];

        // Make attester a validator
        let stake = TokenAmount::from_tokens(20_000);
        let mut account = TokenAccount::new();
        account.staked = stake;
        node.accounts.insert(attester, account);
        node.state.validator_count = 1;

        // Submit attestation
        let tx = SovereignTransaction::ImpactAttest {
            attester,
            beneficiary,
            category: 1, // Healthcare
            base_score: 100,
            evidence_hash: [0u8; 32],
        };

        let result = node.execute_transaction(tx);
        assert!(result.success);

        // Total impact should increase
        assert!(node.state.total_impact > 0);
    }

    #[test]
    fn test_ihsan_gate() {
        let mut node = create_test_node();

        // Lower Ihsān below threshold
        node.state.ihsan_score = 0.80;

        let tx = SovereignTransaction::SeedTransfer {
            from: [1u8; 32],
            to: [2u8; 32],
            amount: 100,
        };

        let result = node.execute_transaction(tx);
        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("Ihsān score"));
    }

    #[test]
    fn test_gini_calculation() {
        let mut node = create_test_node();

        // Add accounts with different balances
        for i in 0..10 {
            let address = [i as u8; 32];
            let balance = TokenAmount::from_tokens((i + 1) as u64 * 100);
            node.accounts
                .insert(address, TokenAccount::with_seed(balance));
        }

        node.recalculate_gini();

        // Should have some inequality
        assert!(node.state.gini_coefficient > 0.0);
        assert!(node.state.gini_coefficient < 1.0);
    }

    #[test]
    fn test_block_processing() {
        let mut node = create_test_node();

        // Fund accounts
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        node.accounts.insert(
            sender,
            TokenAccount::with_seed(TokenAmount::from_tokens(10_000)),
        );

        // Add multiple transactions
        for i in 0..5 {
            let tx = SovereignTransaction::SeedTransfer {
                from: sender,
                to: receiver,
                amount: TokenAmount::from_tokens(100).raw(),
            };
            node.submit_transaction(tx);
        }

        assert_eq!(node.pending_tx.len(), 5);

        // Process block
        let results = node.process_block();

        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.success));
        assert_eq!(node.pending_tx.len(), 0);
        assert_eq!(node.block_height, 1);
    }

    #[test]
    fn test_epoch_advancement() {
        let mut node = create_test_node();

        // Add some activity
        let staker = [1u8; 32];
        node.accounts.insert(
            staker,
            TokenAccount::with_seed(TokenAmount::from_tokens(50_000)),
        );

        node.advance_epoch();
        assert_eq!(node.state.epoch, 1);

        node.advance_epoch();
        assert_eq!(node.state.epoch, 2);
    }

    #[test]
    fn test_network_multiplier_update() {
        let mut node = create_test_node();

        // Add resource contributors
        for i in 0..20 {
            let tx = SovereignTransaction::ResourceContribute {
                node: [i; 32],
                compute: 1000,
                storage: 1_000_000,
                bandwidth: 10_000,
            };
            node.execute_transaction(tx);
        }

        // Should have network boost now
        assert!(node.network_multiplier() >= 1.0);
    }

    #[test]
    fn test_status() {
        let node = create_test_node();
        let status = node.status();

        assert_eq!(status.epoch, 0);
        assert_eq!(status.block_height, 0);
        assert!(status.synced);
        assert!(status.ihsan_score >= IHSAN_THRESHOLD);
    }

    #[test]
    fn test_stability_check() {
        let node = create_test_node();
        // Fresh node should be stable
        assert!(node.is_stable());
    }
}
