// src/autopoietic/step9_implementation.rs - Token Incentive Types for Step 9
//
// Economic/ethical model update types used by the AutopoieticLoop's Step 9.
// These types track BLOOM minting from Proof-of-Impact, generation rewards,
// and Ihsan weight adjustments based on performance trends.

use crate::blockchain::tokens::{BloomToken, SeedToken, TokenAmount};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Token incentive state for economic model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenIncentiveState {
    pub bloom: BloomToken,
    pub seed: SeedToken,
    pub generation_rewards: Vec<GenerationReward>,
    pub ihsan_weight_adjustments: Vec<(String, f64)>,
}

/// Reward issued for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationReward {
    pub generation: u64,
    pub bloom_minted: TokenAmount,
    pub impact_score: u64,
    pub ihsan_score: f64,
    pub timestamp: DateTime<Utc>,
}

impl TokenIncentiveState {
    pub fn new() -> Self {
        Self {
            bloom: BloomToken::new(),
            seed: SeedToken::new(),
            generation_rewards: Vec::new(),
            ihsan_weight_adjustments: Vec::new(),
        }
    }
}
