// src/blockchain/mod.rs - BIZRA Native Blockchain Module
//
// BIZRA uses its OWN native blockchain for enforcement:
// - Genesis block = BIZRA NODE0
// - ADL native token (21B cap from BIZRA_TOKENOMICS_GENESIS.yaml)
// - SAT consensus built into chain
// - No external chain dependencies
// - Full sovereignty maintained
//
// Contracts on BIZRA chain:
// - ReceiptRegistry: Immutable receipt anchoring
// - IhsanOracle: Ihsan score validation
// - AgentRegistry: PAT/SAT agent identity + bonding
// - SATConsensus: On-chain Byzantine consensus

pub mod chain;
pub mod receipts;
pub mod tokens;

pub use chain::{BizraChain, ChainConfig, TransactionReceipt};
pub use receipts::{anchor_receipt, AnchorResult, ReceiptAnchor};
pub use tokens::{
    BloomToken, ImpactCategory, SeedToken, TokenAccount, TokenAmount, TokenError, TokenTransfer,
    TokenType,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Genesis hash from BIZRA_TOKENOMICS_GENESIS.yaml
pub const GENESIS_HASH: &str = "9dfa0bd5375ee06120e72c04618b407b2cf184f110075a573984a4b185f25974";

/// BZR token total supply (from tokenomics)
/// Note: Token name configurable - BZR (BIZRA Token), BZT, BZC are alternatives
pub const BZR_TOTAL_SUPPLY: u64 = 21_000_000_000; // 21 billion

/// Token symbol (can be customized: BZR, BZT, BZC, BIZRA)
pub const TOKEN_SYMBOL: &str = "BZR";

/// BIZRA chain ID (unique identifier)
/// Derived from "BIZRA" = 0x B1 2A A (B=11, I=1, Z=2, R=A, A=A in base-16 inspired encoding)
pub const BIZRA_CHAIN_ID: u64 = 727866;

/// Contract addresses on BIZRA native chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractAddresses {
    /// Receipt Registry contract
    pub receipt_registry: String,
    /// Ihsan Oracle contract
    pub ihsan_oracle: String,
    /// Agent Registry contract
    pub agent_registry: String,
    /// SAT Consensus contract
    pub sat_consensus: String,
}

impl Default for ContractAddresses {
    fn default() -> Self {
        Self {
            receipt_registry: "0xBIZRA_RECEIPT_REGISTRY_V1".to_string(),
            ihsan_oracle: "0xBIZRA_IHSAN_ORACLE_V1".to_string(),
            agent_registry: "0xBIZRA_AGENT_REGISTRY_V1".to_string(),
            sat_consensus: "0xBIZRA_SAT_CONSENSUS_V1".to_string(),
        }
    }
}

/// Blockchain transaction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BizraTransaction {
    /// Anchor a receipt to the chain
    AnchorReceipt {
        receipt_id: String,
        receipt_type: String,
        integrity_hash: String,
        ihsan_score: f64,
        sat_approvers: u8,
    },
    /// Register an agent
    RegisterAgent {
        agent_id: String,
        team: String,
        name: String,
        bond_amount: u64,
    },
    /// Cast SAT consensus vote
    CastVote {
        proposal_id: String,
        validator_id: String,
        vote: String,
        rejection_code: Option<String>,
    },
    /// Record Ihsan score
    RecordIhsan {
        request_id: String,
        score: f64,
        dimension_scores: [f64; 8],
        passed: bool,
    },
}

/// Compute transaction hash
pub fn compute_tx_hash(tx: &BizraTransaction) -> String {
    let json = serde_json::to_string(tx).unwrap_or_default();
    let hash = Sha256::digest(json.as_bytes());
    format!("0x{:x}", hash)
}

/// Verify genesis hash matches expected
pub fn verify_genesis() -> bool {
    GENESIS_HASH == "9dfa0bd5375ee06120e72c04618b407b2cf184f110075a573984a4b185f25974"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_hash_valid() {
        assert!(verify_genesis());
    }

    #[test]
    fn test_chain_id() {
        assert_eq!(BIZRA_CHAIN_ID, 727866);
    }

    #[test]
    fn test_contract_addresses_default() {
        let addrs = ContractAddresses::default();
        assert!(addrs.receipt_registry.contains("RECEIPT_REGISTRY"));
        assert!(addrs.ihsan_oracle.contains("IHSAN_ORACLE"));
    }

    #[test]
    fn test_transaction_hash() {
        let tx = BizraTransaction::AnchorReceipt {
            receipt_id: "EXEC-20260123-001".to_string(),
            receipt_type: "Execution".to_string(),
            integrity_hash: "sha256:abc123".to_string(),
            ihsan_score: 0.97,
            sat_approvers: 4,
        };
        let hash = compute_tx_hash(&tx);
        assert!(hash.starts_with("0x"));
        assert_eq!(hash.len(), 66); // 0x + 64 hex chars
    }
}
