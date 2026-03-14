//! Configuration for BIZRA Hunter

use crate::{LANE1_SNR_THRESHOLD, MIN_CONSISTENT_AXES};

#[derive(Debug, Clone)]
pub struct HunterConfig {
    /// Pipeline capacity (lane1 queue size)
    pub pipeline_capacity: usize,
    /// Lane1 SNR threshold
    pub snr_threshold: f32,
    /// Minimum consistent axes
    pub min_axes: usize,
    /// Loop sleep millis (health loop)
    pub loop_sleep_ms: u64,
    /// JSON-RPC endpoint URL (e.g. "https://eth-mainnet.g.alchemy.com/v2/...")
    pub rpc_url: Option<String>,
    /// Chain ID (1 = mainnet, 11155111 = sepolia, etc.)
    pub chain_id: u64,
    /// Target contract addresses to scan (hex, with 0x prefix)
    pub target_addresses: Vec<String>,
}

impl Default for HunterConfig {
    fn default() -> Self {
        Self {
            pipeline_capacity: 65_536,
            snr_threshold: LANE1_SNR_THRESHOLD,
            min_axes: MIN_CONSISTENT_AXES,
            loop_sleep_ms: 25,
            rpc_url: None,
            chain_id: 1,
            target_addresses: Vec::new(),
        }
    }
}
