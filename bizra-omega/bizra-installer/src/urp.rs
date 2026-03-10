//! Universal Resource Pool (URP) — Sovereign Resource Dedication
//!
//! URP is the opt-in mechanism by which node operators voluntarily
//! dedicate a portion of their idle compute to the BIZRA network.
//! It is NOT mining. It is NOT mandatory. It is Ihsān-scored.
//!
//! Constitutional: ADL Gini ≤ 0.35 — URP must NOT create inequality.
//! The 2.5% Zakat floor is non-negotiable.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §20
//! Standing on Giants: Al-Ghazali (Ihsān, 1095), Shannon (capacity)

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// ADL Gini Threshold (from core/integration/constants.py)
// ─────────────────────────────────────────────────────────────

/// Constitutional hard gate: economic Gini must stay ≤ 0.35
pub const ADL_GINI_THRESHOLD: f64 = 0.35;

/// Zakat floor: minimum 2.5% of earned SEED goes to network commons
pub const ZAKAT_RATE: f64 = 0.025;

// ─────────────────────────────────────────────────────────────
// Resource Pledge
// ─────────────────────────────────────────────────────────────

/// What the user is willing to dedicate to the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourcePledge {
    /// CPU threads to share (0 = none)
    pub cpu_threads: u32,
    /// RAM to share in GB (0 = none)
    pub ram_gb: f32,
    /// GPU memory to share in GB (0 = none)
    pub vram_gb: f32,
    /// Storage to share in GB (0 = none)
    pub storage_gb: f32,
    /// Bandwidth to share in Mbps (0 = none)
    pub bandwidth_mbps: f32,
    /// Operating hours per day (0 = always on, 1-24)
    pub hours_per_day: u8,
    /// Opt-in consent given
    pub consent: bool,
    /// Timestamp of consent
    pub consented_at: Option<String>,
}

impl Default for ResourcePledge {
    fn default() -> Self {
        Self {
            cpu_threads: 0,
            ram_gb: 0.0,
            vram_gb: 0.0,
            storage_gb: 0.0,
            bandwidth_mbps: 0.0,
            hours_per_day: 0,
            consent: false,
            consented_at: None,
        }
    }
}

impl ResourcePledge {
    /// Whether this pledge contributes anything
    pub fn is_active(&self) -> bool {
        self.consent
            && (self.cpu_threads > 0
                || self.ram_gb > 0.0
                || self.vram_gb > 0.0
                || self.storage_gb > 0.0
                || self.bandwidth_mbps > 0.0)
    }

    /// Compute the resource score (0.0 to 1.0)
    /// Used for SEED minting rate calculation
    pub fn resource_score(&self) -> f64 {
        if !self.consent {
            return 0.0;
        }

        // Weighted scoring (spec §20)
        let cpu_score = (self.cpu_threads as f64 / 16.0).min(1.0) * 0.20;
        let ram_score = (self.ram_gb as f64 / 32.0).min(1.0) * 0.20;
        let gpu_score = (self.vram_gb as f64 / 24.0).min(1.0) * 0.25;
        let storage_score = (self.storage_gb as f64 / 100.0).min(1.0) * 0.15;
        let bandwidth_score = (self.bandwidth_mbps as f64 / 100.0).min(1.0) * 0.10;
        let uptime_score = (self.hours_per_day as f64 / 24.0).min(1.0) * 0.10;

        cpu_score + ram_score + gpu_score + storage_score + bandwidth_score + uptime_score
    }
}

// ─────────────────────────────────────────────────────────────
// URP Manager
// ─────────────────────────────────────────────────────────────

/// Manages URP state and SEED calculations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct URPState {
    pub pledge: ResourcePledge,
    /// Accumulated SEED from resource contribution
    pub seed_balance: f64,
    /// SEED sent to Zakat pool
    pub zakat_contributed: f64,
    /// Total uptime seconds since pledge
    pub uptime_seconds: u64,
    /// Last contribution timestamp
    pub last_contribution: Option<String>,
}

impl Default for URPState {
    fn default() -> Self {
        Self {
            pledge: ResourcePledge::default(),
            seed_balance: 0.0,
            zakat_contributed: 0.0,
            uptime_seconds: 0,
            last_contribution: None,
        }
    }
}

impl URPState {
    pub fn new(pledge: ResourcePledge) -> Self {
        Self {
            pledge,
            ..Default::default()
        }
    }

    /// Credit SEED for an interval of contribution.
    /// Automatically deducts Zakat (2.5% floor).
    ///
    /// Returns (net_seed_earned, zakat_deducted)
    pub fn credit_contribution(&mut self, interval_seconds: u64) -> (f64, f64) {
        if !self.pledge.is_active() {
            return (0.0, 0.0);
        }

        let score = self.pledge.resource_score();
        // Base rate: 1 SEED per hour at score=1.0
        let hours = interval_seconds as f64 / 3600.0;
        let gross_seed = score * hours;

        // Zakat deduction (constitutional floor)
        let zakat = gross_seed * ZAKAT_RATE;
        let net_seed = gross_seed - zakat;

        self.seed_balance += net_seed;
        self.zakat_contributed += zakat;
        self.uptime_seconds += interval_seconds;
        self.last_contribution = Some(chrono::Utc::now().to_rfc3339());

        (net_seed, zakat)
    }

    /// Generate a recommended pledge based on device profile.
    /// Conservative defaults: 25% of available resources.
    pub fn recommend_pledge(
        cpu_threads: u32,
        ram_gb: f32,
        vram_gb: f32,
        disk_gb: f32,
    ) -> ResourcePledge {
        ResourcePledge {
            cpu_threads: (cpu_threads / 4).max(1),
            ram_gb: ram_gb * 0.25,
            vram_gb: vram_gb * 0.25,
            storage_gb: (disk_gb * 0.10).min(50.0),
            bandwidth_mbps: 10.0,
            hours_per_day: 8,
            consent: false, // Must be explicitly set
            consented_at: None,
        }
    }
}

/// Save URP state to disk
pub fn save_urp_state(state: &URPState, path: &std::path::Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Load URP state from disk
pub fn load_urp_state(path: &std::path::Path) -> std::io::Result<URPState> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pledge_is_inactive() {
        let p = ResourcePledge::default();
        assert!(!p.is_active());
        assert_eq!(p.resource_score(), 0.0);
    }

    #[test]
    fn active_pledge_has_score() {
        let p = ResourcePledge {
            cpu_threads: 4,
            ram_gb: 8.0,
            vram_gb: 8.0,
            storage_gb: 50.0,
            bandwidth_mbps: 50.0,
            hours_per_day: 12,
            consent: true,
            consented_at: Some("2026-01-01T00:00:00Z".into()),
        };
        assert!(p.is_active());
        let score = p.resource_score();
        assert!(score > 0.0 && score <= 1.0, "Score was {score}");
    }

    #[test]
    fn no_consent_no_score() {
        let p = ResourcePledge {
            cpu_threads: 16,
            ram_gb: 64.0,
            vram_gb: 24.0,
            storage_gb: 100.0,
            bandwidth_mbps: 100.0,
            hours_per_day: 24,
            consent: false,
            consented_at: None,
        };
        assert!(!p.is_active());
        assert_eq!(p.resource_score(), 0.0);
    }

    #[test]
    fn credit_deducts_zakat() {
        let pledge = ResourcePledge {
            cpu_threads: 4,
            ram_gb: 8.0,
            vram_gb: 0.0,
            storage_gb: 10.0,
            bandwidth_mbps: 10.0,
            hours_per_day: 24,
            consent: true,
            consented_at: Some("2026-01-01T00:00:00Z".into()),
        };
        let mut state = URPState::new(pledge);

        let (net, zakat) = state.credit_contribution(3600); // 1 hour
        assert!(net > 0.0, "Net SEED should be positive");
        assert!(zakat > 0.0, "Zakat should be deducted");
        assert!(
            (zakat / (net + zakat) - ZAKAT_RATE).abs() < 0.001,
            "Zakat rate mismatch"
        );
    }

    #[test]
    fn inactive_pledge_gets_nothing() {
        let mut state = URPState::default();
        let (net, zakat) = state.credit_contribution(3600);
        assert_eq!(net, 0.0);
        assert_eq!(zakat, 0.0);
    }

    #[test]
    fn recommend_pledge_is_conservative() {
        let p = URPState::recommend_pledge(16, 32.0, 24.0, 500.0);
        assert_eq!(p.cpu_threads, 4);
        assert_eq!(p.ram_gb, 8.0);
        assert_eq!(p.vram_gb, 6.0);
        assert_eq!(p.storage_gb, 50.0); // Capped
        assert!(!p.consent); // Must be explicitly set
    }

    #[test]
    fn adl_gini_threshold_matches_constants() {
        assert_eq!(ADL_GINI_THRESHOLD, 0.35);
        assert_eq!(ZAKAT_RATE, 0.025);
    }

    #[test]
    fn resource_score_maxes_at_one() {
        let p = ResourcePledge {
            cpu_threads: 64,
            ram_gb: 128.0,
            vram_gb: 80.0,
            storage_gb: 1000.0,
            bandwidth_mbps: 1000.0,
            hours_per_day: 24,
            consent: true,
            consented_at: Some("now".into()),
        };
        assert_eq!(p.resource_score(), 1.0);
    }
}
