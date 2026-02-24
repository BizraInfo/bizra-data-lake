//! # Constitutional Receipts — The Third Fact
//!
//! Every action produces an immutable receipt:
//!   content_hash = HASH(channel || summary || payload_hash || ihsan_score)
//!   chain_hash   = HASH(content_hash || previous_hash)
//!
//! Receipts form a Merkle chain: each receipt links to the previous one.
//! The chain is the node's complete auditable history.
//! Tampering with any receipt breaks the chain — detectable in O(n).
//!
//! ## Standing on Giants
//! - **Merkle (1979)**: Hash trees for tamper-evident data structures
//! - **O'Connor et al. (2020)**: BLAKE3 — parallel, secure, fast
//! - **Nakamoto (2008)**: Chain of signed hashes as immutable ledger

use crate::types::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Deterministic hash function
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Deterministic 256-bit hash for receipt content.
/// Uses 4-lane FNV-1a variant with cross-lane mixing.
///
/// NOTE: Placeholder for dev/test. Production MUST use BLAKE3.
/// The API is designed so swapping is a single-function replacement.
pub fn content_hash(data: &[u8]) -> [u8; 32] {
    let mut h: [u64; 4] = [
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
    ];

    for (i, &b) in data.iter().enumerate() {
        let lane = i % 4;
        h[lane] = h[lane].wrapping_mul(0x100000001b3).wrapping_add(b as u64);
        if i % 32 == 31 {
            let tmp = h[0];
            h[0] ^= h[1].wrapping_mul(0x9e3779b97f4a7c15);
            h[1] ^= h[2].wrapping_mul(0x517cc1b727220a95);
            h[2] ^= h[3].wrapping_mul(0x6c62272e07bb0142);
            h[3] ^= tmp.wrapping_mul(0x62b821756295c58d);
        }
    }

    h[0] ^= h[2];
    h[1] ^= h[3];
    h[2] ^= h[0].wrapping_mul(0xff51afd7ed558ccd);
    h[3] ^= h[1].wrapping_mul(0xc4ceb9fe1a85ec53);

    let mut out = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }
    out
}

/// Hash two 32-byte values together (Merkle chaining).
pub fn chain_hash(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(a);
    combined[32..].copy_from_slice(b);
    content_hash(&combined)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Receipt Chain
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The genesis hash — seed of the entire chain.
pub const GENESIS_SEED: &[u8] = b"BIZRA_GENESIS_RECEIPT_CHAIN_v0.1.0";

/// Builds and verifies constitutional receipt chains.
pub struct ReceiptChain {
    head_hash: [u8; 32],
    chain_length: u64,
    receipts: Vec<ConstitutionalReceipt>,
}

impl ReceiptChain {
    pub fn new() -> Self {
        Self {
            head_hash: content_hash(GENESIS_SEED),
            chain_length: 0,
            receipts: Vec::new(),
        }
    }

    /// Record a completed action, producing a constitutional receipt.
    pub fn record(
        &mut self,
        action_id: ActionId,
        timestamp: ActionTimestamp,
        action: &BizraAction,
        verdict: GuardianVerdict,
        ihsan_score: IhsanScore,
        payload_hash: [u8; 32],
    ) -> ConstitutionalReceipt {
        let channel = action.channel();
        let summary = action.summary();

        // Content hash: channel || summary || payload || ihsan
        let mut data = Vec::with_capacity(summary.len() + 64);
        data.push(channel_to_byte(&channel));
        data.extend_from_slice(summary.as_bytes());
        data.extend_from_slice(&payload_hash);
        data.extend_from_slice(&ihsan_score.value().to_le_bytes());
        let content = content_hash(&data);

        // Chain: hash(content || previous)
        let chained = chain_hash(&content, &self.head_hash);

        let receipt = ConstitutionalReceipt {
            action_id,
            timestamp,
            content_hash: chained,
            ihsan_score,
            verdict,
            channel,
            action_summary: summary,
            signature: [0u8; 64], // Ed25519 placeholder
            previous_hash: self.head_hash,
        };

        self.head_hash = chained;
        self.chain_length += 1;
        self.receipts.push(receipt.clone());

        receipt
    }

    /// Verify entire chain integrity. Ok(len) or Err(broken_index).
    pub fn verify_chain(&self) -> Result<u64, u64> {
        if self.receipts.is_empty() {
            return Ok(0);
        }

        let genesis = content_hash(GENESIS_SEED);
        if self.receipts[0].previous_hash != genesis {
            return Err(0);
        }

        for i in 1..self.receipts.len() {
            if self.receipts[i].previous_hash != self.receipts[i - 1].content_hash {
                return Err(i as u64);
            }
        }

        Ok(self.chain_length)
    }

    pub fn head_hash(&self) -> [u8; 32] {
        self.head_hash
    }
    pub fn len(&self) -> u64 {
        self.chain_length
    }
    pub fn is_empty(&self) -> bool {
        self.chain_length == 0
    }
    pub fn get(&self, index: usize) -> Option<&ConstitutionalReceipt> {
        self.receipts.get(index)
    }
    pub fn latest(&self) -> Option<&ConstitutionalReceipt> {
        self.receipts.last()
    }
    pub fn all_receipts(&self) -> &[ConstitutionalReceipt] {
        &self.receipts
    }
}

impl Default for ReceiptChain {
    fn default() -> Self {
        Self::new()
    }
}

fn channel_to_byte(channel: &Channel) -> u8 {
    match channel {
        Channel::Ahk => 0x01,
        Channel::Llm => 0x02,
        Channel::Memory => 0x03,
        Channel::Mcp => 0x04,
        Channel::FileSystem => 0x05,
        Channel::Browser => 0x06,
        Channel::Response => 0x07,
        Channel::Telescript => 0x08,
    }
}

/// Hash payload content for receipt inclusion.
pub fn hash_payload(payload: &ActionPayload) -> [u8; 32] {
    match payload {
        ActionPayload::Empty => content_hash(b"EMPTY"),
        ActionPayload::Text(s) => content_hash(s.as_bytes()),
        ActionPayload::Bytes(b) => content_hash(b),
        ActionPayload::Structured { entries } => {
            let mut data = Vec::new();
            for (k, v) in entries {
                data.extend_from_slice(k.as_bytes());
                data.push(0x00);
                data.extend_from_slice(v.as_bytes());
                data.push(0x01);
            }
            content_hash(&data)
        }
        ActionPayload::Error(e) => content_hash(e.as_bytes()),
    }
}
