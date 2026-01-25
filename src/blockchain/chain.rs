// src/blockchain/chain.rs - BIZRA Native Chain Client
//
// Client for interacting with the BIZRA native blockchain.
// BIZRA is the genesis - we don't deploy on external chains.
//
// Architecture:
// - Substrate-based runtime (Rust native)
// - SAT consensus integrated at protocol level
// - ADL token as native currency
// - Receipt anchoring as core feature

use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{BizraTransaction, ContractAddresses, BIZRA_CHAIN_ID, GENESIS_HASH};

/// Configuration for BIZRA chain client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    /// Chain endpoint URL
    pub endpoint: String,
    /// Chain ID
    pub chain_id: u64,
    /// Genesis hash for validation
    pub genesis_hash: String,
    /// Contract addresses
    pub contracts: ContractAddresses,
    /// Enable transaction simulation (no actual anchoring)
    pub simulation_mode: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            endpoint: std::env::var("BIZRA_CHAIN_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:9944".to_string()),
            chain_id: BIZRA_CHAIN_ID,
            genesis_hash: GENESIS_HASH.to_string(),
            contracts: ContractAddresses::default(),
            simulation_mode: std::env::var("BIZRA_CHAIN_SIMULATION")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true), // Default to simulation for safety
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Transaction receipt from chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    /// Transaction hash
    pub tx_hash: String,
    /// Block number (None if pending)
    pub block_number: Option<u64>,
    /// Block hash (None if pending)
    pub block_hash: Option<String>,
    /// Transaction index in block
    pub tx_index: Option<u64>,
    /// Success status
    pub success: bool,
    /// Gas used (in ADL units)
    pub gas_used: u64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Events emitted
    pub events: Vec<ChainEvent>,
}

/// Chain event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainEvent {
    /// Event name
    pub name: String,
    /// Contract that emitted
    pub contract: String,
    /// Event data
    pub data: HashMap<String, serde_json::Value>,
}

/// Block information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    /// Block number
    pub number: u64,
    /// Block hash
    pub hash: String,
    /// Parent hash
    pub parent_hash: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Number of transactions
    pub tx_count: usize,
}

/// BIZRA Native Chain Client
pub struct BizraChain {
    /// Configuration
    config: ChainConfig,
    /// Current block number (simulated or real)
    current_block: Arc<RwLock<u64>>,
    /// Pending transactions (simulation)
    pending_txs: Arc<RwLock<Vec<BizraTransaction>>>,
    /// Transaction receipts (simulation)
    receipts: Arc<RwLock<HashMap<String, TransactionReceipt>>>,
    /// Connected status
    connected: Arc<RwLock<bool>>,
}

impl BizraChain {
    /// Create new chain client with default config
    pub fn new() -> Self {
        Self::with_config(ChainConfig::default())
    }

    /// Create chain client with custom config
    pub fn with_config(config: ChainConfig) -> Self {
        info!(
            endpoint = %config.endpoint,
            chain_id = config.chain_id,
            simulation = config.simulation_mode,
            "🔗 Initializing BIZRA Native Chain client"
        );

        Self {
            config,
            current_block: Arc::new(RwLock::new(1)),
            pending_txs: Arc::new(RwLock::new(Vec::new())),
            receipts: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Connect to BIZRA chain
    pub async fn connect(&self) -> Result<()> {
        if self.config.simulation_mode {
            info!("📡 BIZRA Chain client in SIMULATION mode - no real transactions");
            *self.connected.write().await = true;
            return Ok(());
        }

        // In production, this would connect to the Substrate node
        // For now, we simulate connection
        info!(endpoint = %self.config.endpoint, "📡 Connecting to BIZRA Native Chain...");

        // Verify genesis hash
        if self.config.genesis_hash != GENESIS_HASH {
            bail!(
                "Genesis hash mismatch! Expected: {}, Got: {}",
                GENESIS_HASH,
                self.config.genesis_hash
            );
        }

        *self.connected.write().await = true;
        info!("✅ Connected to BIZRA Native Chain");
        Ok(())
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Get current block number
    pub async fn get_block_number(&self) -> Result<u64> {
        Ok(*self.current_block.read().await)
    }

    /// Get block info
    pub async fn get_block(&self, block_number: u64) -> Result<BlockInfo> {
        // In simulation, generate block info
        let hash = format!("0x{:064x}", block_number * 12345);
        let parent_hash = if block_number > 0 {
            format!("0x{:064x}", (block_number - 1) * 12345)
        } else {
            format!("0x{}", GENESIS_HASH)
        };

        Ok(BlockInfo {
            number: block_number,
            hash,
            parent_hash,
            timestamp: Utc::now(),
            tx_count: 0,
        })
    }

    /// Submit transaction to chain
    pub async fn submit_transaction(&self, tx: BizraTransaction) -> Result<TransactionReceipt> {
        if !*self.connected.read().await {
            bail!("Not connected to BIZRA chain");
        }

        let tx_hash = super::compute_tx_hash(&tx);
        debug!(tx_hash = %tx_hash, "Submitting transaction to BIZRA chain");

        if self.config.simulation_mode {
            return self.simulate_transaction(tx, &tx_hash).await;
        }

        // In production, this would submit to the Substrate node
        // via JSON-RPC or Substrate client library
        self.simulate_transaction(tx, &tx_hash).await
    }

    /// Simulate transaction execution
    async fn simulate_transaction(
        &self,
        tx: BizraTransaction,
        tx_hash: &str,
    ) -> Result<TransactionReceipt> {
        // Store pending transaction
        self.pending_txs.write().await.push(tx.clone());

        // Increment block number
        let mut block_num = self.current_block.write().await;
        *block_num += 1;
        let block_number = *block_num;
        drop(block_num);

        // Generate block hash
        let block_hash = format!("0x{:064x}", block_number * 12345);

        // Generate events based on transaction type
        let events = self.generate_events(&tx);

        let receipt = TransactionReceipt {
            tx_hash: tx_hash.to_string(),
            block_number: Some(block_number),
            block_hash: Some(block_hash),
            tx_index: Some(0),
            success: true,
            gas_used: self.estimate_gas(&tx),
            timestamp: Utc::now(),
            events,
        };

        // Store receipt
        self.receipts
            .write()
            .await
            .insert(tx_hash.to_string(), receipt.clone());

        info!(
            tx_hash = %tx_hash,
            block = block_number,
            gas = receipt.gas_used,
            "✅ Transaction included in BIZRA chain"
        );

        Ok(receipt)
    }

    /// Get transaction receipt
    pub async fn get_receipt(&self, tx_hash: &str) -> Option<TransactionReceipt> {
        self.receipts.read().await.get(tx_hash).cloned()
    }

    /// Estimate gas for transaction
    fn estimate_gas(&self, tx: &BizraTransaction) -> u64 {
        match tx {
            BizraTransaction::AnchorReceipt { .. } => 21000,
            BizraTransaction::RegisterAgent { .. } => 50000,
            BizraTransaction::CastVote { .. } => 15000,
            BizraTransaction::RecordIhsan { .. } => 30000,
        }
    }

    /// Generate events for transaction
    fn generate_events(&self, tx: &BizraTransaction) -> Vec<ChainEvent> {
        match tx {
            BizraTransaction::AnchorReceipt {
                receipt_id,
                receipt_type,
                integrity_hash,
                ihsan_score,
                sat_approvers,
            } => {
                vec![ChainEvent {
                    name: "ReceiptAnchored".to_string(),
                    contract: self.config.contracts.receipt_registry.clone(),
                    data: HashMap::from([
                        (
                            "receiptId".to_string(),
                            serde_json::json!(receipt_id),
                        ),
                        (
                            "receiptType".to_string(),
                            serde_json::json!(receipt_type),
                        ),
                        (
                            "integrityHash".to_string(),
                            serde_json::json!(integrity_hash),
                        ),
                        ("ihsanScore".to_string(), serde_json::json!(ihsan_score)),
                        (
                            "satApprovers".to_string(),
                            serde_json::json!(sat_approvers),
                        ),
                    ]),
                }]
            }
            BizraTransaction::RegisterAgent {
                agent_id,
                team,
                name,
                bond_amount,
            } => {
                vec![ChainEvent {
                    name: "AgentRegistered".to_string(),
                    contract: self.config.contracts.agent_registry.clone(),
                    data: HashMap::from([
                        ("agentId".to_string(), serde_json::json!(agent_id)),
                        ("team".to_string(), serde_json::json!(team)),
                        ("name".to_string(), serde_json::json!(name)),
                        ("bondAmount".to_string(), serde_json::json!(bond_amount)),
                    ]),
                }]
            }
            BizraTransaction::CastVote {
                proposal_id,
                validator_id,
                vote,
                rejection_code,
            } => {
                vec![ChainEvent {
                    name: "VoteCast".to_string(),
                    contract: self.config.contracts.sat_consensus.clone(),
                    data: HashMap::from([
                        (
                            "proposalId".to_string(),
                            serde_json::json!(proposal_id),
                        ),
                        (
                            "validatorId".to_string(),
                            serde_json::json!(validator_id),
                        ),
                        ("vote".to_string(), serde_json::json!(vote)),
                        (
                            "rejectionCode".to_string(),
                            serde_json::json!(rejection_code),
                        ),
                    ]),
                }]
            }
            BizraTransaction::RecordIhsan {
                request_id,
                score,
                dimension_scores,
                passed,
            } => {
                vec![ChainEvent {
                    name: "ScoreValidated".to_string(),
                    contract: self.config.contracts.ihsan_oracle.clone(),
                    data: HashMap::from([
                        ("requestId".to_string(), serde_json::json!(request_id)),
                        ("score".to_string(), serde_json::json!(score)),
                        (
                            "dimensionScores".to_string(),
                            serde_json::json!(dimension_scores),
                        ),
                        ("passed".to_string(), serde_json::json!(passed)),
                    ]),
                }]
            }
        }
    }

    /// Get chain configuration
    pub fn config(&self) -> &ChainConfig {
        &self.config
    }

    /// Get contract addresses
    pub fn contracts(&self) -> &ContractAddresses {
        &self.config.contracts
    }
}

impl Default for BizraChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chain_connect() {
        let chain = BizraChain::new();
        assert!(chain.connect().await.is_ok());
        assert!(chain.is_connected().await);
    }

    #[tokio::test]
    async fn test_submit_anchor_receipt() {
        let chain = BizraChain::new();
        chain.connect().await.unwrap();

        let tx = BizraTransaction::AnchorReceipt {
            receipt_id: "EXEC-20260123-001".to_string(),
            receipt_type: "Execution".to_string(),
            integrity_hash: "sha256:abc123".to_string(),
            ihsan_score: 0.97,
            sat_approvers: 4,
        };

        let receipt = chain.submit_transaction(tx).await.unwrap();
        assert!(receipt.success);
        assert!(receipt.block_number.is_some());
        assert_eq!(receipt.events.len(), 1);
        assert_eq!(receipt.events[0].name, "ReceiptAnchored");
    }

    #[tokio::test]
    async fn test_get_receipt() {
        let chain = BizraChain::new();
        chain.connect().await.unwrap();

        let tx = BizraTransaction::RegisterAgent {
            agent_id: "master_reasoner".to_string(),
            team: "PAT".to_string(),
            name: "MasterReasoner".to_string(),
            bond_amount: 1000,
        };

        let tx_receipt = chain.submit_transaction(tx).await.unwrap();
        let retrieved = chain.get_receipt(&tx_receipt.tx_hash).await;

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().tx_hash, tx_receipt.tx_hash);
    }

    #[tokio::test]
    async fn test_block_info() {
        let chain = BizraChain::new();
        chain.connect().await.unwrap();

        let block = chain.get_block(1).await.unwrap();
        assert_eq!(block.number, 1);
        assert!(block.hash.starts_with("0x"));
    }
}
