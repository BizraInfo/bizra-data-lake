//! # Metabolic Ledger — PoI Yield + SEED Emission Decay
//!
//! The Metabolic Ledger tracks the economic metabolism of the sovereign node:
//! - Mints PoI (Proof of Intelligence) yield for every verified action.
//! - Applies TTRL-driven emission decay: as reflex cache hits increase,
//!   SEED emission per action decreases (model is "trained", work is cheap).
//! - Maintains network scaling bonus: larger federation = more shared reflexes.
//!
//! ## Emission Decay Logic (from TTRL paper analysis)
//! This resolves the **gap identified in the original architecture doc**:
//! "PoI emission decay missing from metabolic ledger."
//!
//! The decay is NOT scarcity-based (Bitcoin-style halving).
//! It is **efficiency-based**: emission falls as the node becomes more capable.
//!
//! ```text
//! cache_hit_rate = 0%  → multiplier ≈ 1.0  (full emission, node is learning)
//! cache_hit_rate = 60% → multiplier ≈ 0.46 (model mostly trained)
//! cache_hit_rate = 90% → multiplier ≈ 0.19 (model expert)
//! ```
//!
//! Network scaling bonus: `log2(network_size) / 20`.
//! More nodes share reflexes faster → bonus emission to reward federation.
//!
//! Standing on Giants:
//! - Nakamoto (2008): Emission schedule as economic signal
//! - TTRL paper (2025): Self-improvement as the true scarcity signal

use serde::{Deserialize, Serialize};

/// A single PoI yield event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoiYield {
    /// Base amount minted (before decay multiplier).
    pub base_amount: f64,
    /// Final minted amount after emission multiplier.
    pub amount: f64,
    /// The emission multiplier applied (0.01–1.0).
    pub emission_multiplier: f64,
    /// Cache hit rate at the time of minting.
    pub cache_hit_rate: f64,
    /// Federation node count at time of minting.
    pub network_size: u64,
    /// UNIX ms timestamp.
    pub minted_at_ms: u64,
}

/// Statistics for the metabolic ledger.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LedgerStats {
    pub total_actions: u64,
    pub total_minted: f64,
    pub avg_multiplier: f64,
    /// Running cache hit rate (EMA with α = 0.05).
    pub cache_hit_rate: f64,
}

/// The Metabolic Ledger.
#[derive(Debug)]
pub struct MetabolicLedger {
    /// Base SEED amount minted per verified action (pre-multiplier).
    base_seed_per_action: f64,
    /// EMA smoothing factor for cache hit rate (0 < α < 1).
    ema_alpha: f64,
    pub stats: LedgerStats,
}

impl MetabolicLedger {
    /// Create a new ledger.
    ///
    /// `base_seed_per_action` — raw SEED minted per action (e.g. 1.0).
    /// Read from `config/proactive_config.yaml`; never hardcode.
    pub fn new(base_seed_per_action: f64) -> Self {
        Self {
            base_seed_per_action,
            ema_alpha: 0.05,
            stats: LedgerStats::default(),
        }
    }

    /// Mint PoI yield for a verified action.
    ///
    /// `is_cache_hit`   — was this a Tier-1 or Tier-2 cache hit?
    /// `network_size`   — current federation node count.
    /// `now_ms`         — UNIX milliseconds.
    pub fn mint_poi_yield(
        &mut self,
        is_cache_hit: bool,
        network_size: u64,
        now_ms: u64,
    ) -> PoiYield {
        // Update EMA cache hit rate.
        let hit_signal = if is_cache_hit { 1.0 } else { 0.0 };
        self.stats.cache_hit_rate =
            self.ema_alpha * hit_signal + (1.0 - self.ema_alpha) * self.stats.cache_hit_rate;

        let multiplier = Self::compute_emission_decay(self.stats.cache_hit_rate, network_size);
        let amount = self.base_seed_per_action * multiplier;

        // Update stats.
        self.stats.total_actions += 1;
        self.stats.total_minted += amount;
        let n = self.stats.total_actions as f64;
        self.stats.avg_multiplier = (self.stats.avg_multiplier * (n - 1.0) + multiplier) / n;

        PoiYield {
            base_amount: self.base_seed_per_action,
            amount,
            emission_multiplier: multiplier,
            cache_hit_rate: self.stats.cache_hit_rate,
            network_size,
            minted_at_ms: now_ms,
        }
    }

    /// Compute the emission multiplier.
    ///
    /// Range: [0.01, 1.0].
    /// - Decay term: falls as cache hit rate rises (model is learning less).
    /// - Network bonus: rises logarithmically with federation size.
    pub fn compute_emission_decay(cache_hit_rate: f64, network_size: u64) -> f64 {
        let hit = cache_hit_rate.clamp(0.0, 1.0);
        let decay = 1.0 - hit * 0.90;

        let network_bonus = if network_size > 1 {
            (network_size as f64).log2() / 20.0
        } else {
            0.0
        };

        (decay + network_bonus).clamp(0.01, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_emission_at_zero_hits() {
        let mult = MetabolicLedger::compute_emission_decay(0.0, 1);
        // decay = 1.0, network_bonus = 0 → mult = 1.0
        assert!((mult - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_minimum_emission_floor() {
        // Even at 100% cache hits and 1 node, floor is 0.01
        let mult = MetabolicLedger::compute_emission_decay(1.0, 1);
        assert!(mult >= 0.01);
    }

    #[test]
    fn test_network_bonus_increases_emission() {
        let solo = MetabolicLedger::compute_emission_decay(0.6, 1);
        let large = MetabolicLedger::compute_emission_decay(0.6, 1000);
        assert!(large > solo, "Larger network should yield higher emission");
    }

    #[test]
    fn test_mint_updates_stats() {
        let mut ledger = MetabolicLedger::new(1.0);
        let y = ledger.mint_poi_yield(false, 1, 1000);
        assert_eq!(ledger.stats.total_actions, 1);
        assert!((ledger.stats.total_minted - y.amount).abs() < 1e-12);
    }

    #[test]
    fn test_cache_hit_ema_rises() {
        let mut ledger = MetabolicLedger::new(1.0);
        for _ in 0..50 {
            ledger.mint_poi_yield(true, 1, 1000);
        }
        // After 50 cache hits, EMA should be well above 0.
        assert!(ledger.stats.cache_hit_rate > 0.80);
    }
}
