// src/blockchain/receipts.rs - Receipt Anchoring to BIZRA Native Chain
//
// Bridges the existing receipt system (src/receipts.rs) to the BIZRA blockchain.
// All receipts can be anchored for immutable evidence.
//
// Receipt anchoring flow:
// 1. Receipt emitted by ReceiptEmitter (file + Redis)
// 2. anchor_receipt() called to anchor to chain
// 3. Transaction submitted to ReceiptRegistry contract
// 4. Event emitted: ReceiptAnchored
// 5. AnchorResult returned with tx hash + block info

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::OnceCell;

use super::chain::{BizraChain, TransactionReceipt};
use super::BizraTransaction;
use crate::receipts::{ExecutionReceipt, RejectionReceipt};

/// Global chain client (lazy initialized)
static CHAIN_CLIENT: OnceCell<Arc<BizraChain>> = OnceCell::const_new();

/// Initialize global chain client
pub async fn init_chain() -> Result<Arc<BizraChain>> {
    let chain = Arc::new(BizraChain::new());
    chain.connect().await?;
    Ok(chain)
}

/// Get or initialize chain client
pub async fn get_chain() -> Result<Arc<BizraChain>> {
    CHAIN_CLIENT
        .get_or_try_init(|| async { init_chain().await })
        .await
        .cloned()
}

/// Result of anchoring a receipt to the chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorResult {
    /// Receipt ID that was anchored
    pub receipt_id: String,
    /// Transaction hash on BIZRA chain
    pub tx_hash: String,
    /// Block number (None if pending)
    pub block_number: Option<u64>,
    /// Block hash (None if pending)
    pub block_hash: Option<String>,
    /// Timestamp of anchoring
    pub anchored_at: DateTime<Utc>,
    /// Success status
    pub success: bool,
    /// Gas used
    pub gas_used: u64,
    /// Chain receipt hash (for verification)
    pub chain_receipt_hash: String,
}

/// Receipt anchor service
pub struct ReceiptAnchor {
    /// Chain client
    chain: Arc<BizraChain>,
    /// Enable auto-anchoring
    auto_anchor: bool,
}

impl ReceiptAnchor {
    /// Create new receipt anchor with existing chain client
    pub fn new(chain: Arc<BizraChain>) -> Self {
        Self {
            chain,
            auto_anchor: false,
        }
    }

    /// Create with auto-anchoring enabled
    pub fn with_auto_anchor(chain: Arc<BizraChain>) -> Self {
        Self {
            chain,
            auto_anchor: true,
        }
    }

    /// Anchor an execution receipt to the chain
    pub async fn anchor_execution(&self, receipt: &ExecutionReceipt) -> Result<AnchorResult> {
        let tx = BizraTransaction::AnchorReceipt {
            receipt_id: receipt.receipt_id.clone(),
            receipt_type: format!("{:?}", receipt.receipt_type),
            integrity_hash: receipt.integrity_hash.clone(),
            ihsan_score: receipt.ihsan_score,
            sat_approvers: receipt.sat_approvers_count as u8,
        };

        let chain_receipt = self.chain.submit_transaction(tx).await?;

        Ok(self.build_anchor_result(&receipt.receipt_id, chain_receipt))
    }

    /// Anchor a rejection receipt to the chain
    pub async fn anchor_rejection(&self, receipt: &RejectionReceipt) -> Result<AnchorResult> {
        let tx = BizraTransaction::AnchorReceipt {
            receipt_id: receipt.receipt_id.clone(),
            receipt_type: format!("{:?}", receipt.receipt_type),
            integrity_hash: receipt.integrity_hash.clone(),
            ihsan_score: 0.0, // Rejections don't have Ihsan score
            sat_approvers: receipt.approving_validators.len() as u8,
        };

        let chain_receipt = self.chain.submit_transaction(tx).await?;

        Ok(self.build_anchor_result(&receipt.receipt_id, chain_receipt))
    }

    /// Anchor any receipt by ID and type
    pub async fn anchor_by_id(
        &self,
        receipt_id: &str,
        receipt_type: &str,
        integrity_hash: &str,
        ihsan_score: f64,
        sat_approvers: u8,
    ) -> Result<AnchorResult> {
        let tx = BizraTransaction::AnchorReceipt {
            receipt_id: receipt_id.to_string(),
            receipt_type: receipt_type.to_string(),
            integrity_hash: integrity_hash.to_string(),
            ihsan_score,
            sat_approvers,
        };

        let chain_receipt = self.chain.submit_transaction(tx).await?;

        Ok(self.build_anchor_result(receipt_id, chain_receipt))
    }

    /// Build anchor result from chain receipt
    fn build_anchor_result(
        &self,
        receipt_id: &str,
        chain_receipt: TransactionReceipt,
    ) -> AnchorResult {
        // Compute chain receipt hash for verification
        let chain_receipt_hash = {
            let content = format!(
                "{}|{}|{}",
                receipt_id,
                chain_receipt.tx_hash,
                chain_receipt.block_number.unwrap_or(0)
            );
            let hash = Sha256::digest(content.as_bytes());
            format!("sha256:{:x}", hash)
        };

        AnchorResult {
            receipt_id: receipt_id.to_string(),
            tx_hash: chain_receipt.tx_hash,
            block_number: chain_receipt.block_number,
            block_hash: chain_receipt.block_hash,
            anchored_at: chain_receipt.timestamp,
            success: chain_receipt.success,
            gas_used: chain_receipt.gas_used,
            chain_receipt_hash,
        }
    }

    /// Check if a receipt has been anchored
    pub async fn is_anchored(&self, _receipt_id: &str) -> bool {
        // In production, this would query the ReceiptRegistry contract
        // For now, we check local receipts
        // TODO: Implement contract query
        false
    }

    /// Get anchor info for a receipt
    pub async fn get_anchor_info(&self, _receipt_id: &str) -> Option<AnchorResult> {
        // In production, query the ReceiptRegistry contract
        // TODO: Implement contract query
        None
    }

    /// Whether auto-anchoring is enabled
    pub fn is_auto_anchor(&self) -> bool {
        self.auto_anchor
    }
}

/// Anchor a receipt to the BIZRA chain (convenience function)
pub async fn anchor_receipt(
    receipt_id: &str,
    receipt_type: &str,
    integrity_hash: &str,
    ihsan_score: f64,
    sat_approvers: u8,
) -> Result<AnchorResult> {
    let chain = get_chain().await?;
    let anchor = ReceiptAnchor::new(chain);
    anchor
        .anchor_by_id(
            receipt_id,
            receipt_type,
            integrity_hash,
            ihsan_score,
            sat_approvers,
        )
        .await
}

/// Record an Ihsan score to the chain
pub async fn record_ihsan(
    request_id: &str,
    score: f64,
    dimension_scores: [f64; 8],
    passed: bool,
) -> Result<TransactionReceipt> {
    let chain = get_chain().await?;

    let tx = BizraTransaction::RecordIhsan {
        request_id: request_id.to_string(),
        score,
        dimension_scores,
        passed,
    };

    chain.submit_transaction(tx).await
}

/// Cast a SAT consensus vote on the chain
pub async fn cast_sat_vote(
    proposal_id: &str,
    validator_id: &str,
    approve: bool,
    rejection_code: Option<&str>,
) -> Result<TransactionReceipt> {
    let chain = get_chain().await?;

    let tx = BizraTransaction::CastVote {
        proposal_id: proposal_id.to_string(),
        validator_id: validator_id.to_string(),
        vote: if approve {
            "Approve".to_string()
        } else {
            "Reject".to_string()
        },
        rejection_code: rejection_code.map(|s| s.to_string()),
    };

    chain.submit_transaction(tx).await
}

/// Register an agent on the chain
pub async fn register_agent(
    agent_id: &str,
    team: &str,
    name: &str,
    bond_amount: u64,
) -> Result<TransactionReceipt> {
    let chain = get_chain().await?;

    let tx = BizraTransaction::RegisterAgent {
        agent_id: agent_id.to_string(),
        team: team.to_string(),
        name: name.to_string(),
        bond_amount,
    };

    chain.submit_transaction(tx).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::ReceiptType;

    #[tokio::test]
    async fn test_anchor_receipt() {
        let result =
            anchor_receipt("EXEC-20260123-001", "Execution", "sha256:abc123", 0.97, 4).await;

        assert!(result.is_ok());
        let anchor = result.unwrap();
        assert!(anchor.success);
        assert!(anchor.tx_hash.starts_with("0x"));
        assert!(anchor.chain_receipt_hash.starts_with("sha256:"));
    }

    #[tokio::test]
    async fn test_record_ihsan() {
        let dimensions = [0.98, 0.97, 0.95, 0.96, 0.94, 0.92, 0.90, 0.88];
        let result = record_ihsan("REQ-001", 0.96, dimensions, true).await;

        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.success);
        assert_eq!(receipt.events[0].name, "ScoreValidated");
    }

    #[tokio::test]
    async fn test_cast_sat_vote() {
        let result = cast_sat_vote("PROP-001", "poi_verifier", true, None).await;

        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.success);
        assert_eq!(receipt.events[0].name, "VoteCast");
    }

    #[tokio::test]
    async fn test_register_agent() {
        let result = register_agent("master_reasoner", "PAT", "MasterReasoner", 1000).await;

        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.success);
        assert_eq!(receipt.events[0].name, "AgentRegistered");
    }

    #[tokio::test]
    async fn test_receipt_anchor_service() {
        let chain = Arc::new(BizraChain::new());
        chain.connect().await.unwrap();

        let anchor = ReceiptAnchor::new(chain);

        let result = anchor
            .anchor_by_id("REJ-20260123-001", "Rejection", "sha256:def456", 0.0, 1)
            .await;

        assert!(result.is_ok());
        let anchor_result = result.unwrap();
        assert!(anchor_result.success);
        assert_eq!(anchor_result.receipt_id, "REJ-20260123-001");
    }
}
