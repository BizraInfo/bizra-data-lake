// bizra-node/src/seed_ledger.rs
// ============================================================
// SEED Ledger — Economic Settlement for Governed Missions
// ============================================================
//
// Connects the Mission Control Plane (proof) to SEED economics (value).
// Every completed mission that passes constitutional gates earns SEED
// tokens proportional to its Proof-of-Intelligence yield.
//
// Flow:
//   MissionReceipt (proof) → PoI calculation → SEED mint → Gini check
//
// Standing on Giants:
//   Nakamoto (2008): Proof-of-Work → Proof-of-Intelligence
//   Al-Ghazali (1095): Ihsan gate — excellence threshold
//   TTRL paper (2025): Emission decay from cache efficiency
//
// Wire points:
//   mission_bridge.rs:138 (after complete, before sign)
//   omni_kernel.rs (MetabolicLedger already mints PoI)
// ============================================================

use bizra_core::{ADL_GINI_THRESHOLD, IHSAN_THRESHOLD, ZAKAT_RATE};
use bizra_mission::receipt::MissionReceipt;

/// Result of economic settlement for a completed mission.
#[derive(Debug, Clone)]
pub struct SettlementResult {
    /// SEED tokens minted for this mission.
    pub seed_minted: f64,
    /// PoI yield before zakat deduction.
    pub poi_yield_gross: f64,
    /// Zakat (2.5%) deducted at mint time.
    pub zakat_deducted: f64,
    /// Net SEED credited to the node.
    pub seed_net: f64,
    /// Gini coefficient after this mint (if computed).
    pub post_gini: Option<f64>,
    /// Whether the Gini gate passed (Adl: Gini ≤ 0.35).
    pub gini_passed: bool,
    /// Emission multiplier applied (from cache hit rate).
    pub emission_multiplier: f64,
}

/// Accumulated ledger state for a single node's SEED balance.
#[derive(Debug, Clone)]
pub struct SeedLedger {
    /// Cumulative SEED balance (net of zakat).
    balance: f64,
    /// Total missions settled.
    missions_settled: u64,
    /// Total SEED minted (gross, before zakat).
    total_minted_gross: f64,
    /// Total zakat paid.
    total_zakat_paid: f64,
    /// Running cache hit rate (EMA, alpha=0.05) for emission decay.
    cache_hit_rate: f64,
    /// Base SEED per mission (from config, default 1.0).
    base_seed_per_mission: f64,
}

/// EMA smoothing factor for cache hit rate.
const EMA_ALPHA: f64 = 0.05;

impl SeedLedger {
    /// Create a new ledger with the given base SEED rate.
    pub fn new(base_seed_per_mission: f64) -> Self {
        Self {
            balance: 0.0,
            missions_settled: 0,
            total_minted_gross: 0.0,
            total_zakat_paid: 0.0,
            cache_hit_rate: 0.0,
            base_seed_per_mission,
        }
    }

    /// Settle a completed mission receipt into SEED tokens.
    ///
    /// Returns `None` if the receipt is ineligible (failed mission, below Ihsan floor).
    /// Returns `Some(SettlementResult)` with minted SEED on success.
    ///
    /// The Gini check is advisory at this level — the node has only one balance,
    /// so post-mint Gini is always 0.0 for a single-node ledger. The Gini gate
    /// becomes meaningful when the settlement propagates to the ResourcePool (URP)
    /// via federation. At that point, the URP's `check_adl()` enforces the hard gate.
    pub fn settle(
        &mut self,
        receipt: &MissionReceipt,
        was_cache_hit: bool,
    ) -> Option<SettlementResult> {
        // Gate 1: Only successful missions earn SEED
        if !receipt.is_success() {
            return None;
        }

        // Gate 2: Ihsan floor — constitutional minimum for economic participation
        let ihsan = receipt.ihsan_score.unwrap_or(0.0) as f64;
        if ihsan < IHSAN_THRESHOLD {
            return None;
        }

        // Update cache hit rate EMA (for emission decay)
        let hit_signal = if was_cache_hit { 1.0 } else { 0.0 };
        self.cache_hit_rate = EMA_ALPHA * hit_signal + (1.0 - EMA_ALPHA) * self.cache_hit_rate;

        // Compute emission multiplier (TTRL-style decay)
        let emission_multiplier = compute_emission_decay(self.cache_hit_rate);

        // PoI yield: base × ihsan × emission_multiplier
        let poi_yield_gross = self.base_seed_per_mission * ihsan * emission_multiplier;

        // Zakat deduction at mint time (2.5%)
        let zakat = poi_yield_gross * ZAKAT_RATE;
        let seed_net = poi_yield_gross - zakat;

        // Credit to ledger
        self.balance += seed_net;
        self.missions_settled += 1;
        self.total_minted_gross += poi_yield_gross;
        self.total_zakat_paid += zakat;

        // Single-node Gini is always 0.0 (trivially passes)
        // Real Gini enforcement happens at URP level
        let post_gini = 0.0;

        Some(SettlementResult {
            seed_minted: poi_yield_gross,
            poi_yield_gross,
            zakat_deducted: zakat,
            seed_net,
            post_gini: Some(post_gini),
            gini_passed: post_gini <= ADL_GINI_THRESHOLD,
            emission_multiplier,
        })
    }

    /// Current SEED balance (net of all zakat deductions).
    pub fn balance(&self) -> f64 {
        self.balance
    }

    /// Total missions settled through this ledger.
    pub fn missions_settled(&self) -> u64 {
        self.missions_settled
    }

    /// Total zakat paid across all settlements.
    pub fn total_zakat_paid(&self) -> f64 {
        self.total_zakat_paid
    }

    /// Current emission multiplier (based on cache hit rate).
    pub fn emission_multiplier(&self) -> f64 {
        compute_emission_decay(self.cache_hit_rate)
    }
}

/// Emission decay: as cache hit rate rises, emission falls.
/// Not scarcity-based (Bitcoin halving) — efficiency-based (TTRL).
///
/// cache_hit_rate = 0%  → multiplier ≈ 1.0  (node is learning)
/// cache_hit_rate = 60% → multiplier ≈ 0.46 (mostly trained)
/// cache_hit_rate = 90% → multiplier ≈ 0.19 (expert node)
fn compute_emission_decay(cache_hit_rate: f64) -> f64 {
    let hit = cache_hit_rate.clamp(0.0, 1.0);
    let decay = 1.0 - hit * 0.90;
    decay.clamp(0.01, 1.0)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_mission::{
        mission::Mission,
        state::{DegradationReason, MissionState},
    };

    /// Build a mock MissionReceipt by walking a Mission through the state machine.
    fn mock_receipt(ihsan: f32, success: bool) -> MissionReceipt {
        let content_hash: [u8; 32] = blake3::hash(b"test-mission").into();
        let t = 1_700_000_000_000u64;
        let mut m = Mission::new(content_hash, t);
        m.ihsan_score = Some(ihsan);

        // Walk through the constitutional state machine
        m.transition(MissionState::Queued, t + 1, "test").unwrap();
        m.transition(MissionState::WarmingRetrieval, t + 2, "test")
            .unwrap();
        m.transition(MissionState::WarmingModel, t + 3, "test")
            .unwrap();
        m.transition(MissionState::Retrieving, t + 4, "test")
            .unwrap();
        m.transition(MissionState::Routing, t + 5, "test").unwrap();
        m.transition(MissionState::Running, t + 6, "test").unwrap();
        m.transition(MissionState::Scoring, t + 7, "test").unwrap();
        m.transition(MissionState::Persisting, t + 8, "test")
            .unwrap();

        if success {
            m.complete(t + 9).unwrap();
        } else {
            // Back up to Scoring for legal degrade transition
            // (Persisting→Degraded may not be legal; build a separate path)
            let mut m2 = Mission::new(content_hash, t);
            m2.ihsan_score = Some(ihsan);
            m2.transition(MissionState::Queued, t + 1, "test").unwrap();
            m2.transition(MissionState::WarmingRetrieval, t + 2, "test")
                .unwrap();
            m2.transition(MissionState::WarmingModel, t + 3, "test")
                .unwrap();
            m2.transition(MissionState::Retrieving, t + 4, "test")
                .unwrap();
            m2.transition(MissionState::Routing, t + 5, "test").unwrap();
            m2.transition(MissionState::Running, t + 6, "test").unwrap();
            m2.transition(MissionState::Scoring, t + 7, "test").unwrap();
            m2.degrade(vec![DegradationReason::GuardianVeto], t + 8)
                .unwrap();
            return m2.receipt.unwrap();
        }

        m.receipt.unwrap()
    }

    #[test]
    fn test_successful_settlement_mints_seed() {
        let mut ledger = SeedLedger::new(1.0);
        let receipt = mock_receipt(0.97, true);
        let result = ledger.settle(&receipt, false);

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.seed_minted > 0.0);
        assert!(r.zakat_deducted > 0.0);
        assert!(r.seed_net > 0.0);
        assert!(r.seed_net < r.seed_minted); // zakat deducted
        assert!(r.gini_passed);
        assert_eq!(ledger.missions_settled(), 1);
    }

    #[test]
    fn test_failed_mission_earns_nothing() {
        let mut ledger = SeedLedger::new(1.0);
        let receipt = mock_receipt(0.97, false);
        let result = ledger.settle(&receipt, false);
        assert!(result.is_none());
        assert_eq!(ledger.balance(), 0.0);
    }

    #[test]
    fn test_low_ihsan_earns_nothing() {
        let mut ledger = SeedLedger::new(1.0);
        let receipt = mock_receipt(0.80, true); // below 0.95 floor
        let result = ledger.settle(&receipt, false);
        assert!(result.is_none());
    }

    #[test]
    fn test_zakat_is_exactly_2_5_percent() {
        let mut ledger = SeedLedger::new(1.0);
        let receipt = mock_receipt(1.0, true); // perfect Ihsan
        let result = ledger.settle(&receipt, false).unwrap();

        let expected_zakat = result.poi_yield_gross * 0.025;
        assert!((result.zakat_deducted - expected_zakat).abs() < 1e-12);
    }

    #[test]
    fn test_balance_accumulates() {
        let mut ledger = SeedLedger::new(1.0);
        for _ in 0..10 {
            let receipt = mock_receipt(0.96, true);
            ledger.settle(&receipt, false);
        }
        assert_eq!(ledger.missions_settled(), 10);
        assert!(ledger.balance() > 0.0);
        assert!(ledger.total_zakat_paid() > 0.0);
    }

    #[test]
    fn test_emission_decay_full_at_zero_hits() {
        let mult = compute_emission_decay(0.0);
        assert!((mult - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_emission_decay_reduces_with_cache_hits() {
        let mult_low = compute_emission_decay(0.0);
        let mult_high = compute_emission_decay(0.90);
        assert!(mult_high < mult_low);
    }

    #[test]
    fn test_cache_hit_reduces_subsequent_yield() {
        let mut ledger = SeedLedger::new(1.0);

        // First mission: cache miss
        let r1 = ledger.settle(&mock_receipt(0.96, true), false).unwrap();

        // Feed 50 cache hits to raise the rate
        for _ in 0..50 {
            ledger.settle(&mock_receipt(0.96, true), true);
        }

        // Next mission: emission should be lower
        let r2 = ledger.settle(&mock_receipt(0.96, true), false).unwrap();
        assert!(r2.emission_multiplier < r1.emission_multiplier);
    }

    #[test]
    fn test_gini_always_passes_single_node() {
        let mut ledger = SeedLedger::new(1.0);
        let result = ledger.settle(&mock_receipt(0.96, true), false).unwrap();
        assert!(result.gini_passed);
        assert_eq!(result.post_gini, Some(0.0));
    }
}
