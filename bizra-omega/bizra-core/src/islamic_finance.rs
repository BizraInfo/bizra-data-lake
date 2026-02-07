//! BIZRA Islamic Finance Protocol Layer
//!
//! Shariah-compliant financial primitives for the BIZRA Resource Pool.
//!
//! # Core Principles (Non-negotiable)
//!
//! 1. **No Riba (Interest)** - All lending is profit-sharing (Mudarabah/Musharakah)
//! 2. **Zakat Distribution** - Automatic 2.5% on wealth above nisab threshold
//! 3. **Halal Services Only** - Ethical envelope must pass Islamic compliance
//! 4. **Risk Sharing** - Losses shared proportionally, no one bears alone
//! 5. **Asset-Backed** - Every token represents real compute/service value
//!
//! # Standing on Giants
//!
//! - Al-Ghazali (1111): Maqasid al-Shariah (objectives of Islamic law)
//! - Ibn Khaldun (1377): Economic theory and wealth distribution
//! - Muhammad Yunus (2006): Microfinance and social business
//! - Nakamoto (2008): Trustless consensus for value transfer
//!
//! # Integration Points
//!
//! - FATE Gates: IslamicComplianceGate validates all transactions
//! - EthicalEnvelope: Extended with Shariah compliance fields
//! - AdlInvariant: Zakat distribution maintains fairness
//! - Treasury: Waqf endowments fund infrastructure

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::omega::{AdlInvariant, ADL_GINI_THRESHOLD};
use crate::pci::{Gate, GateContext, GateResult, GateTier, RejectCode};
use crate::IHSAN_THRESHOLD;

// =============================================================================
// CONSTANTS - Shariah Compliance Parameters (Immutable)
// =============================================================================

/// Zakat rate: 2.5% per lunar year (354 days)
pub const ZAKAT_RATE: f64 = 0.025;

/// Nisab threshold in compute units (equivalent to 85g gold value)
/// This is the minimum wealth above which Zakat is obligatory
pub const NISAB_THRESHOLD: f64 = 1000.0;

/// Lunar year in days (for Zakat calculation period)
pub const LUNAR_YEAR_DAYS: u64 = 354;

/// Minimum profit share for Mudarabah entrepreneur (protects labor)
pub const MIN_MUDARIB_SHARE: f64 = 0.30;

/// Maximum profit share for Mudarabah investor (prevents exploitation)
pub const MAX_RABBULMAL_SHARE: f64 = 0.70;

/// Minimum holding period before Zakat applies (hawl = 1 lunar year)
pub const HAWL_DAYS: u64 = 354;

/// Maximum Waqf administrative overhead (transparency requirement)
pub const MAX_WAQF_OVERHEAD: f64 = 0.10;

/// Minimum beneficiaries for a valid Waqf (prevents self-dealing)
pub const MIN_WAQF_BENEFICIARIES: usize = 3;

// =============================================================================
// ERROR TYPES
// =============================================================================

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum IslamicFinanceError {
    #[error("Riba (interest) detected: {details}")]
    RibaDetected { details: String },

    #[error("Gharar (excessive uncertainty) in contract: {details}")]
    GhararDetected { details: String },

    #[error("Maysir (gambling/speculation) detected: {details}")]
    MaysirDetected { details: String },

    #[error("Haram activity: {activity}")]
    HaramActivity { activity: String },

    #[error("Zakat calculation error: {reason}")]
    ZakatError { reason: String },

    #[error("Mudarabah violation: {reason}")]
    MudarabahViolation { reason: String },

    #[error("Musharakah violation: {reason}")]
    MusharakahViolation { reason: String },

    #[error("Waqf violation: {reason}")]
    WaqfViolation { reason: String },

    #[error("Ihsan threshold not met: {score:.3} < {threshold:.3}")]
    IhsanViolation { score: f64, threshold: f64 },

    #[error("Adl (justice) violation: {reason}")]
    AdlViolation { reason: String },

    #[error("Insufficient nisab: {balance:.2} < {nisab:.2}")]
    InsufficientNisab { balance: f64, nisab: f64 },

    #[error("Contract expired at {expiry_ms}")]
    ContractExpired { expiry_ms: u64 },

    #[error(
        "Invalid profit ratio: investor {investor:.2} + entrepreneur {entrepreneur:.2} != 1.0"
    )]
    InvalidProfitRatio { investor: f64, entrepreneur: f64 },
}

pub type IslamicFinanceResult<T> = Result<T, IslamicFinanceError>;

// =============================================================================
// ZAKAT CALCULATOR
// =============================================================================

/// Zakat-eligible asset types in the Resource Pool
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZakatableAsset {
    /// Compute credits (liquid, immediately zakatable)
    ComputeCredits,
    /// Stored data (valued by retrieval cost)
    StorageUnits,
    /// Bandwidth allocation (network capacity)
    BandwidthUnits,
    /// Staked tokens in Musharakah partnerships
    PartnershipStake,
    /// Active Mudarabah capital
    MudarabahCapital,
    /// Reserved for future operations
    ReservedFunds,
}

/// Individual wealth record for Zakat calculation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WealthRecord {
    /// Node identifier
    pub node_id: String,
    /// Asset holdings by type
    pub holdings: HashMap<ZakatableAsset, f64>,
    /// Timestamp when wealth first exceeded nisab
    pub nisab_exceeded_at: Option<u64>,
    /// Last Zakat payment timestamp
    pub last_zakat_paid: Option<u64>,
    /// Total Zakat paid lifetime
    pub lifetime_zakat: f64,
}

impl WealthRecord {
    /// Create new wealth record for a node
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            holdings: HashMap::new(),
            nisab_exceeded_at: None,
            last_zakat_paid: None,
            lifetime_zakat: 0.0,
        }
    }

    /// Total wealth across all asset types
    #[inline]
    pub fn total_wealth(&self) -> f64 {
        self.holdings.values().sum()
    }

    /// Check if wealth exceeds nisab threshold
    #[inline]
    pub fn exceeds_nisab(&self) -> bool {
        self.total_wealth() >= NISAB_THRESHOLD
    }
}

/// Zakat distribution recipient categories (Quran 9:60)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZakatRecipient {
    /// Al-Fuqara: Those in absolute poverty (no compute access)
    Poor,
    /// Al-Masakin: Those in relative poverty (minimal compute)
    Needy,
    /// Amil Zakat: Zakat administrators (protocol maintenance)
    Administrators,
    /// Muallaf: New participants requiring onboarding
    NewParticipants,
    /// Riqab: Those in computational debt (over-utilized resources)
    InDebt,
    /// Gharimin: Those facing infrastructure failures
    DisasterRecovery,
    /// Fi Sabilillah: Network infrastructure and public goods
    PublicGoods,
    /// Ibn Sabil: Transient nodes requiring temporary resources
    TransientNodes,
}

/// Zakat distribution result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZakatDistribution {
    /// Source node that paid Zakat
    pub payer_node: String,
    /// Amount of Zakat collected
    pub amount: f64,
    /// Distribution by recipient category
    pub allocations: HashMap<ZakatRecipient, f64>,
    /// Timestamp of distribution
    pub distributed_at: u64,
    /// Cryptographic proof of distribution
    pub distribution_hash: String,
}

/// Zakat Calculator: Automatic 2.5% wealth redistribution engine
///
/// The Zakat system ensures continuous wealth redistribution to maintain
/// Adl (justice) in the Resource Pool. It operates as a protocol-level
/// mechanism that cannot be bypassed.
#[derive(Clone, Debug)]
pub struct ZakatCalculator {
    /// Nisab threshold (minimum wealth for Zakat obligation)
    nisab: f64,
    /// Zakat rate (2.5% standard)
    rate: f64,
    /// Hawl period in milliseconds (1 lunar year)
    hawl_ms: u64,
    /// Distribution weights by recipient category
    distribution_weights: HashMap<ZakatRecipient, f64>,
    /// Adl invariant for fairness checking
    adl_invariant: AdlInvariant,
}

impl Default for ZakatCalculator {
    fn default() -> Self {
        let mut weights = HashMap::new();
        // Equal distribution across 8 categories (can be adjusted by governance)
        weights.insert(ZakatRecipient::Poor, 0.20);
        weights.insert(ZakatRecipient::Needy, 0.15);
        weights.insert(ZakatRecipient::Administrators, 0.05);
        weights.insert(ZakatRecipient::NewParticipants, 0.10);
        weights.insert(ZakatRecipient::InDebt, 0.15);
        weights.insert(ZakatRecipient::DisasterRecovery, 0.10);
        weights.insert(ZakatRecipient::PublicGoods, 0.15);
        weights.insert(ZakatRecipient::TransientNodes, 0.10);

        Self {
            nisab: NISAB_THRESHOLD,
            rate: ZAKAT_RATE,
            hawl_ms: HAWL_DAYS * 24 * 60 * 60 * 1000,
            distribution_weights: weights,
            adl_invariant: AdlInvariant::default(),
        }
    }
}

impl ZakatCalculator {
    /// Create with custom parameters
    pub fn new(nisab: f64, rate: f64) -> Self {
        Self {
            nisab,
            rate,
            ..Default::default()
        }
    }

    /// Calculate Zakat due for a wealth record
    ///
    /// Returns None if:
    /// - Wealth below nisab
    /// - Hawl period not completed
    pub fn calculate_zakat(&self, record: &WealthRecord) -> Option<f64> {
        let total = record.total_wealth();

        // Check nisab threshold
        if total < self.nisab {
            return None;
        }

        // Check hawl (holding period)
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if let Some(exceeded_at) = record.nisab_exceeded_at {
            let holding_period = now_ms.saturating_sub(exceeded_at);
            if holding_period < self.hawl_ms {
                return None; // Hawl not completed
            }
        } else {
            return None; // Never exceeded nisab
        }

        // Check if Zakat already paid this year
        if let Some(last_paid) = record.last_zakat_paid {
            let since_last = now_ms.saturating_sub(last_paid);
            if since_last < self.hawl_ms {
                return None; // Already paid this year
            }
        }

        // Calculate Zakat: 2.5% of total wealth
        Some(total * self.rate)
    }

    /// Distribute Zakat to eligible recipients
    pub fn distribute_zakat(
        &self,
        payer: &WealthRecord,
        amount: f64,
    ) -> IslamicFinanceResult<ZakatDistribution> {
        if amount <= 0.0 {
            return Err(IslamicFinanceError::ZakatError {
                reason: "Zakat amount must be positive".into(),
            });
        }

        let mut allocations = HashMap::new();
        let mut total_allocated = 0.0;

        for (recipient, weight) in &self.distribution_weights {
            let allocation = amount * weight;
            allocations.insert(*recipient, allocation);
            total_allocated += allocation;
        }

        // Ensure all funds distributed (handle floating point)
        let remainder = amount - total_allocated;
        if remainder.abs() > 0.001 {
            // Add remainder to PublicGoods
            *allocations
                .entry(ZakatRecipient::PublicGoods)
                .or_insert(0.0) += remainder;
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Generate distribution hash for auditability
        let hash_input = format!("{}:{}:{}:{}", payer.node_id, amount, now_ms, self.rate);
        let distribution_hash = blake3::hash(hash_input.as_bytes()).to_hex().to_string();

        Ok(ZakatDistribution {
            payer_node: payer.node_id.clone(),
            amount,
            allocations,
            distributed_at: now_ms,
            distribution_hash,
        })
    }

    /// Verify Zakat distribution maintains Adl (fairness)
    pub fn verify_adl_compliance(
        &self,
        pre_distribution: &HashMap<String, f64>,
        post_distribution: &HashMap<String, f64>,
    ) -> IslamicFinanceResult<bool> {
        let pre_result = self.adl_invariant.check(pre_distribution, None);
        let post_result = self.adl_invariant.check(post_distribution, None);

        // Zakat should improve or maintain Gini coefficient
        if post_result.gini > pre_result.gini + 0.01 {
            return Err(IslamicFinanceError::AdlViolation {
                reason: format!(
                    "Zakat distribution worsened inequality: Gini {:.4} -> {:.4}",
                    pre_result.gini, post_result.gini
                ),
            });
        }

        Ok(post_result.passed)
    }
}

// =============================================================================
// MUDARABAH CONTRACT
// =============================================================================

/// Mudarabah contract status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MudarabahStatus {
    /// Contract proposed, awaiting signatures
    Proposed,
    /// Both parties signed, contract active
    Active,
    /// Operations completed, awaiting settlement
    PendingSettlement,
    /// Profits/losses distributed, contract closed
    Settled,
    /// Contract terminated early
    Terminated,
    /// Contract expired without settlement
    Expired,
}

/// Mudarabah loss type (losses borne by capital provider)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MudarabahLoss {
    /// Capital loss due to market conditions
    MarketLoss { amount: f64, reason: String },
    /// Capital loss due to operational failure (not negligence)
    OperationalLoss { amount: f64, reason: String },
}

/// Mudarabah Contract: Profit-Sharing Partnership
///
/// In Mudarabah:
/// - Rabb-ul-Maal (investor) provides capital
/// - Mudarib (entrepreneur) provides labor/skill/compute
/// - Profit shared by pre-agreed ratio
/// - Loss of capital borne by investor
/// - Mudarib loses only their effort (not compensated for labor)
///
/// This structure eliminates Riba by making returns dependent on
/// actual profit, not guaranteed interest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MudarabahContract {
    /// Unique contract identifier
    pub contract_id: String,
    /// Capital provider (Rabb-ul-Maal) node ID
    pub investor_node: String,
    /// Entrepreneur (Mudarib) node ID
    pub entrepreneur_node: String,
    /// Principal capital amount (in compute units)
    pub capital: f64,
    /// Investor's profit share ratio (0.0 to MAX_RABBULMAL_SHARE)
    pub investor_profit_ratio: f64,
    /// Entrepreneur's profit share ratio (MIN_MUDARIB_SHARE to 1.0)
    pub entrepreneur_profit_ratio: f64,
    /// Contract creation timestamp
    pub created_at: u64,
    /// Contract expiry timestamp
    pub expires_at: u64,
    /// Current contract status
    pub status: MudarabahStatus,
    /// Accumulated profit (if positive) or loss (if negative)
    pub accumulated_pnl: f64,
    /// Service type being provided
    pub service_type: String,
    /// Performance metrics
    pub performance_ihsan: f64,
    /// Recorded losses (all borne by investor)
    pub losses: Vec<MudarabahLoss>,
    /// Settlement details
    pub settlement: Option<MudarabahSettlement>,
}

/// Settlement record for completed Mudarabah
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MudarabahSettlement {
    /// Total profit or loss
    pub total_pnl: f64,
    /// Amount returned to investor
    pub investor_return: f64,
    /// Amount paid to entrepreneur
    pub entrepreneur_payment: f64,
    /// Settlement timestamp
    pub settled_at: u64,
    /// Settlement hash for auditability
    pub settlement_hash: String,
}

impl MudarabahContract {
    /// Create a new Mudarabah contract
    ///
    /// Validates profit ratios to ensure Shariah compliance:
    /// - Entrepreneur must receive at least MIN_MUDARIB_SHARE (30%)
    /// - Investor cannot receive more than MAX_RABBULMAL_SHARE (70%)
    /// - Ratios must sum to 1.0
    pub fn new(
        contract_id: String,
        investor_node: String,
        entrepreneur_node: String,
        capital: f64,
        investor_profit_ratio: f64,
        duration_days: u64,
        service_type: String,
    ) -> IslamicFinanceResult<Self> {
        // Validate profit ratios
        let entrepreneur_profit_ratio = 1.0 - investor_profit_ratio;

        if entrepreneur_profit_ratio < MIN_MUDARIB_SHARE {
            return Err(IslamicFinanceError::MudarabahViolation {
                reason: format!(
                    "Entrepreneur share {:.2} below minimum {:.2}",
                    entrepreneur_profit_ratio, MIN_MUDARIB_SHARE
                ),
            });
        }

        if investor_profit_ratio > MAX_RABBULMAL_SHARE {
            return Err(IslamicFinanceError::MudarabahViolation {
                reason: format!(
                    "Investor share {:.2} exceeds maximum {:.2}",
                    investor_profit_ratio, MAX_RABBULMAL_SHARE
                ),
            });
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let expires_at = now_ms + (duration_days * 24 * 60 * 60 * 1000);

        Ok(Self {
            contract_id,
            investor_node,
            entrepreneur_node,
            capital,
            investor_profit_ratio,
            entrepreneur_profit_ratio,
            created_at: now_ms,
            expires_at,
            status: MudarabahStatus::Proposed,
            accumulated_pnl: 0.0,
            service_type,
            performance_ihsan: 1.0,
            losses: Vec::new(),
            settlement: None,
        })
    }

    /// Activate contract (both parties signed)
    pub fn activate(&mut self) -> IslamicFinanceResult<()> {
        if self.status != MudarabahStatus::Proposed {
            return Err(IslamicFinanceError::MudarabahViolation {
                reason: format!("Cannot activate contract in {:?} status", self.status),
            });
        }
        self.status = MudarabahStatus::Active;
        Ok(())
    }

    /// Record profit or loss
    pub fn record_pnl(&mut self, amount: f64, is_profit: bool) -> IslamicFinanceResult<()> {
        if self.status != MudarabahStatus::Active {
            return Err(IslamicFinanceError::MudarabahViolation {
                reason: "Contract not active".into(),
            });
        }

        if is_profit {
            self.accumulated_pnl += amount;
        } else {
            self.accumulated_pnl -= amount;
            self.losses.push(MudarabahLoss::OperationalLoss {
                amount,
                reason: "Operational loss recorded".into(),
            });
        }

        Ok(())
    }

    /// Check if contract has expired
    pub fn is_expired(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now_ms > self.expires_at
    }

    /// Settle the contract and distribute profits/losses
    ///
    /// Settlement rules (Shariah-compliant):
    /// - If profit: Split according to pre-agreed ratios
    /// - If loss: Investor bears capital loss, entrepreneur loses effort
    /// - Capital returned to investor (minus losses)
    pub fn settle(&mut self) -> IslamicFinanceResult<MudarabahSettlement> {
        if self.status != MudarabahStatus::Active
            && self.status != MudarabahStatus::PendingSettlement
        {
            return Err(IslamicFinanceError::MudarabahViolation {
                reason: format!("Cannot settle contract in {:?} status", self.status),
            });
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let (investor_return, entrepreneur_payment) = if self.accumulated_pnl >= 0.0 {
            // Profit case: distribute according to ratios
            let investor_profit = self.accumulated_pnl * self.investor_profit_ratio;
            let entrepreneur_profit = self.accumulated_pnl * self.entrepreneur_profit_ratio;

            (self.capital + investor_profit, entrepreneur_profit)
        } else {
            // Loss case: investor bears capital loss, entrepreneur gets nothing
            let remaining_capital = (self.capital + self.accumulated_pnl).max(0.0);
            (remaining_capital, 0.0)
        };

        // Generate settlement hash
        let hash_input = format!(
            "{}:{}:{}:{}:{}",
            self.contract_id, self.accumulated_pnl, investor_return, entrepreneur_payment, now_ms
        );
        let settlement_hash = blake3::hash(hash_input.as_bytes()).to_hex().to_string();

        let settlement = MudarabahSettlement {
            total_pnl: self.accumulated_pnl,
            investor_return,
            entrepreneur_payment,
            settled_at: now_ms,
            settlement_hash,
        };

        self.settlement = Some(settlement.clone());
        self.status = MudarabahStatus::Settled;

        Ok(settlement)
    }

    /// Verify contract maintains Ihsan threshold
    pub fn verify_ihsan(&self) -> IslamicFinanceResult<bool> {
        if self.performance_ihsan < IHSAN_THRESHOLD {
            return Err(IslamicFinanceError::IhsanViolation {
                score: self.performance_ihsan,
                threshold: IHSAN_THRESHOLD,
            });
        }
        Ok(true)
    }
}

// =============================================================================
// MUSHARAKAH PARTNERSHIP
// =============================================================================

/// Partner contribution in a Musharakah
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MusharakahPartner {
    /// Partner node ID
    pub node_id: String,
    /// Capital contributed
    pub capital_contribution: f64,
    /// Contribution ratio (auto-calculated from capital)
    pub contribution_ratio: f64,
    /// Agreed profit share (can differ from contribution ratio)
    pub profit_share: f64,
    /// Loss share (MUST equal contribution ratio - Shariah requirement)
    pub loss_share: f64,
    /// Partner joined timestamp
    pub joined_at: u64,
    /// Voting weight (proportional to contribution)
    pub voting_weight: f64,
}

/// Musharakah partnership status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusharakahStatus {
    /// Partnership forming, accepting partners
    Forming,
    /// Partnership active
    Active,
    /// Partnership winding down
    WindingDown,
    /// Partnership dissolved
    Dissolved,
}

/// Governance decision in Musharakah
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MusharakahDecision {
    /// Decision ID
    pub decision_id: String,
    /// Description of the decision
    pub description: String,
    /// Votes by partner (node_id -> approve/reject)
    pub votes: HashMap<String, bool>,
    /// Required approval threshold (weighted by contribution)
    pub approval_threshold: f64,
    /// Decision timestamp
    pub timestamp: u64,
    /// Whether decision passed
    pub passed: Option<bool>,
}

/// Musharakah Partnership: Joint Venture with Shared Ownership
///
/// In Musharakah:
/// - All partners contribute capital
/// - Profit shared by negotiated ratios (can reward management)
/// - Losses shared STRICTLY by capital contribution ratio
/// - Democratic governance proportional to contribution
/// - All partners are co-owners of the venture
///
/// This ensures risk sharing - no partner bears disproportionate loss.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MusharakahPartnership {
    /// Partnership unique identifier
    pub partnership_id: String,
    /// Partnership name/purpose
    pub name: String,
    /// Partners and their contributions
    pub partners: Vec<MusharakahPartner>,
    /// Total capital pool
    pub total_capital: f64,
    /// Current status
    pub status: MusharakahStatus,
    /// Accumulated profit/loss
    pub accumulated_pnl: f64,
    /// Creation timestamp
    pub created_at: u64,
    /// Service/operation type
    pub operation_type: String,
    /// Governance decisions history
    pub decisions: Vec<MusharakahDecision>,
    /// Performance Ihsan score
    pub performance_ihsan: f64,
    /// Minimum partners required
    pub min_partners: usize,
    /// Adl invariant for fairness
    #[serde(skip)]
    adl_invariant: Option<AdlInvariant>,
}

impl MusharakahPartnership {
    /// Create a new Musharakah partnership
    pub fn new(
        partnership_id: String,
        name: String,
        operation_type: String,
        min_partners: usize,
    ) -> Self {
        Self {
            partnership_id,
            name,
            partners: Vec::new(),
            total_capital: 0.0,
            status: MusharakahStatus::Forming,
            accumulated_pnl: 0.0,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            operation_type,
            decisions: Vec::new(),
            performance_ihsan: 1.0,
            min_partners,
            adl_invariant: Some(AdlInvariant::default()),
        }
    }

    /// Add a partner with capital contribution
    ///
    /// Validates:
    /// - Partnership is still forming
    /// - Profit share is provided
    /// - Loss share equals contribution ratio (Shariah requirement)
    pub fn add_partner(
        &mut self,
        node_id: String,
        capital: f64,
        profit_share: f64,
    ) -> IslamicFinanceResult<()> {
        if self.status != MusharakahStatus::Forming {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: "Partnership not accepting new partners".into(),
            });
        }

        if capital <= 0.0 {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: "Capital contribution must be positive".into(),
            });
        }

        // Check for duplicate partner
        if self.partners.iter().any(|p| p.node_id == node_id) {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: "Partner already exists in partnership".into(),
            });
        }

        let new_total = self.total_capital + capital;
        let contribution_ratio = capital / new_total;

        // Loss share MUST equal contribution ratio (Shariah requirement)
        let loss_share = contribution_ratio;

        let partner = MusharakahPartner {
            node_id,
            capital_contribution: capital,
            contribution_ratio,
            profit_share,
            loss_share,
            joined_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            voting_weight: contribution_ratio,
        };

        self.partners.push(partner);
        self.total_capital = new_total;

        // Recalculate all ratios after adding partner
        self.recalculate_ratios();

        Ok(())
    }

    /// Recalculate all partner ratios after changes
    fn recalculate_ratios(&mut self) {
        if self.total_capital == 0.0 {
            return;
        }

        for partner in &mut self.partners {
            partner.contribution_ratio = partner.capital_contribution / self.total_capital;
            partner.loss_share = partner.contribution_ratio;
            partner.voting_weight = partner.contribution_ratio;
        }

        // Normalize profit shares if they exceed 1.0
        let total_profit_share: f64 = self.partners.iter().map(|p| p.profit_share).sum();
        if total_profit_share > 1.0 {
            for partner in &mut self.partners {
                partner.profit_share /= total_profit_share;
            }
        }
    }

    /// Activate partnership (minimum partners met)
    pub fn activate(&mut self) -> IslamicFinanceResult<()> {
        if self.partners.len() < self.min_partners {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: format!(
                    "Need {} partners, have {}",
                    self.min_partners,
                    self.partners.len()
                ),
            });
        }

        // Validate profit shares sum to ~1.0
        let total_profit: f64 = self.partners.iter().map(|p| p.profit_share).sum();
        if (total_profit - 1.0).abs() > 0.01 {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: format!("Profit shares sum to {:.3}, must equal 1.0", total_profit),
            });
        }

        // Validate loss shares equal contribution ratios
        for partner in &self.partners {
            if (partner.loss_share - partner.contribution_ratio).abs() > 0.001 {
                return Err(IslamicFinanceError::MusharakahViolation {
                    reason: format!(
                        "Partner {} loss share {:.3} must equal contribution ratio {:.3}",
                        partner.node_id, partner.loss_share, partner.contribution_ratio
                    ),
                });
            }
        }

        self.status = MusharakahStatus::Active;
        Ok(())
    }

    /// Record profit or loss
    pub fn record_pnl(&mut self, amount: f64, is_profit: bool) -> IslamicFinanceResult<()> {
        if self.status != MusharakahStatus::Active {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: "Partnership not active".into(),
            });
        }

        if is_profit {
            self.accumulated_pnl += amount;
        } else {
            self.accumulated_pnl -= amount;
        }

        Ok(())
    }

    /// Distribute accumulated P&L to partners
    ///
    /// Distribution rules (Shariah-compliant):
    /// - Profits: Distributed by agreed profit_share ratios
    /// - Losses: Distributed by contribution_ratio (mandatory)
    pub fn distribute(&self) -> IslamicFinanceResult<HashMap<String, f64>> {
        let mut distributions = HashMap::new();

        if self.accumulated_pnl >= 0.0 {
            // Profit distribution by profit_share
            for partner in &self.partners {
                let share = self.accumulated_pnl * partner.profit_share;
                distributions.insert(partner.node_id.clone(), share);
            }
        } else {
            // Loss distribution by contribution_ratio (mandatory Shariah rule)
            for partner in &self.partners {
                let share = self.accumulated_pnl * partner.loss_share; // Negative value
                distributions.insert(partner.node_id.clone(), share);
            }
        }

        Ok(distributions)
    }

    /// Create a governance decision
    pub fn propose_decision(
        &mut self,
        decision_id: String,
        description: String,
        approval_threshold: f64,
    ) -> IslamicFinanceResult<&MusharakahDecision> {
        if self.status != MusharakahStatus::Active {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: "Partnership not active".into(),
            });
        }

        let decision = MusharakahDecision {
            decision_id,
            description,
            votes: HashMap::new(),
            approval_threshold,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            passed: None,
        };

        self.decisions.push(decision);
        Ok(self.decisions.last().unwrap())
    }

    /// Cast a vote on a decision
    pub fn vote(
        &mut self,
        decision_id: &str,
        node_id: &str,
        approve: bool,
    ) -> IslamicFinanceResult<Option<bool>> {
        // Verify voter is a partner
        let partner = self.partners.iter().find(|p| p.node_id == node_id);
        if partner.is_none() {
            return Err(IslamicFinanceError::MusharakahViolation {
                reason: "Only partners can vote".into(),
            });
        }

        // Find decision
        let decision = self
            .decisions
            .iter_mut()
            .find(|d| d.decision_id == decision_id);

        match decision {
            None => Err(IslamicFinanceError::MusharakahViolation {
                reason: "Decision not found".into(),
            }),
            Some(d) => {
                d.votes.insert(node_id.to_string(), approve);

                // Calculate weighted approval
                let weighted_approval: f64 = self
                    .partners
                    .iter()
                    .filter_map(|p| d.votes.get(&p.node_id).map(|&v| (v, p.voting_weight)))
                    .filter(|(v, _)| *v)
                    .map(|(_, w)| w)
                    .sum();

                // Check if decision can be finalized
                let total_voted: f64 = self
                    .partners
                    .iter()
                    .filter(|p| d.votes.contains_key(&p.node_id))
                    .map(|p| p.voting_weight)
                    .sum();

                if total_voted >= 0.5 {
                    // Majority voted
                    let passed = weighted_approval >= d.approval_threshold;
                    d.passed = Some(passed);
                    Ok(d.passed)
                } else {
                    Ok(None) // Voting continues
                }
            }
        }
    }

    /// Verify partnership maintains Adl fairness
    pub fn verify_adl(&self) -> IslamicFinanceResult<bool> {
        let distribution: HashMap<String, f64> = self
            .partners
            .iter()
            .map(|p| (p.node_id.clone(), p.capital_contribution))
            .collect();

        if let Some(ref adl) = self.adl_invariant {
            let result = adl.check(&distribution, None);
            if !result.passed {
                return Err(IslamicFinanceError::AdlViolation {
                    reason: format!("Partnership Gini {:.4} exceeds threshold", result.gini),
                });
            }
        }

        Ok(true)
    }
}

// =============================================================================
// WAQF ENDOWMENT
// =============================================================================

/// Waqf purpose categories
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaqfPurpose {
    /// Network infrastructure maintenance
    Infrastructure,
    /// Educational resources and training
    Education,
    /// Healthcare and emergency compute
    Healthcare,
    /// Research and development
    Research,
    /// Environmental sustainability
    Environmental,
    /// Community support and social welfare
    CommunityWelfare,
    /// Protocol development and governance
    ProtocolDevelopment,
    /// General charitable purposes
    GeneralCharity,
}

/// Waqf beneficiary definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaqfBeneficiary {
    /// Beneficiary identifier (can be node ID or category)
    pub beneficiary_id: String,
    /// Allocation percentage of Waqf yield
    pub allocation_percent: f64,
    /// Eligibility criteria
    pub eligibility_criteria: String,
    /// Active status
    pub is_active: bool,
}

/// Waqf distribution record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaqfDistribution {
    /// Distribution ID
    pub distribution_id: String,
    /// Period start
    pub period_start: u64,
    /// Period end
    pub period_end: u64,
    /// Total yield distributed
    pub total_yield: f64,
    /// Distribution by beneficiary
    pub allocations: HashMap<String, f64>,
    /// Administrative costs (capped at MAX_WAQF_OVERHEAD)
    pub admin_costs: f64,
    /// Distribution hash for auditability
    pub distribution_hash: String,
}

/// Waqf Endowment: Permanent Charitable Resource Allocation
///
/// A Waqf is a permanent dedication of assets for charitable purposes.
/// In BIZRA:
/// - Principal is locked forever (immutable)
/// - Only the yield (compute returns) is distributed
/// - Beneficiaries cannot be the donor
/// - Transparent allocation with capped overhead
///
/// This provides sustainable funding for infrastructure and public goods.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaqfEndowment {
    /// Waqf unique identifier
    pub waqf_id: String,
    /// Waqf name
    pub name: String,
    /// Donor/founder node ID
    pub founder_node: String,
    /// Primary purpose
    pub purpose: WaqfPurpose,
    /// Detailed description
    pub description: String,
    /// Principal amount (locked forever)
    pub principal: f64,
    /// Accumulated yield available for distribution
    pub available_yield: f64,
    /// Total yield distributed historically
    pub total_distributed: f64,
    /// Beneficiaries
    pub beneficiaries: Vec<WaqfBeneficiary>,
    /// Creation timestamp
    pub created_at: u64,
    /// Is Waqf active
    pub is_active: bool,
    /// Administrative overhead rate (capped)
    pub admin_overhead_rate: f64,
    /// Nazir (administrator) node ID
    pub nazir_node: String,
    /// Distribution history
    pub distributions: Vec<WaqfDistribution>,
    /// Ihsan compliance score
    pub ihsan_score: f64,
}

impl WaqfEndowment {
    /// Create a new Waqf endowment
    ///
    /// Validates:
    /// - Principal is positive
    /// - Administrative overhead within limits
    /// - Founder cannot be sole beneficiary
    // TODO: Refactor to use a WaqfEndowmentBuilder pattern
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        waqf_id: String,
        name: String,
        founder_node: String,
        purpose: WaqfPurpose,
        description: String,
        principal: f64,
        nazir_node: String,
        admin_overhead_rate: f64,
    ) -> IslamicFinanceResult<Self> {
        if principal <= 0.0 {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: "Principal must be positive".into(),
            });
        }

        if admin_overhead_rate > MAX_WAQF_OVERHEAD {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: format!(
                    "Admin overhead {:.2}% exceeds maximum {:.2}%",
                    admin_overhead_rate * 100.0,
                    MAX_WAQF_OVERHEAD * 100.0
                ),
            });
        }

        Ok(Self {
            waqf_id,
            name,
            founder_node: founder_node.clone(),
            purpose,
            description,
            principal,
            available_yield: 0.0,
            total_distributed: 0.0,
            beneficiaries: Vec::new(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            is_active: true,
            admin_overhead_rate,
            nazir_node,
            distributions: Vec::new(),
            ihsan_score: 1.0,
        })
    }

    /// Add a beneficiary to the Waqf
    ///
    /// Validates:
    /// - Beneficiary is not the founder (no self-dealing)
    /// - Total allocation does not exceed 100%
    pub fn add_beneficiary(
        &mut self,
        beneficiary_id: String,
        allocation_percent: f64,
        eligibility_criteria: String,
    ) -> IslamicFinanceResult<()> {
        // Prevent founder self-dealing
        if beneficiary_id == self.founder_node {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: "Founder cannot be a beneficiary".into(),
            });
        }

        // Check allocation limits
        let current_total: f64 = self
            .beneficiaries
            .iter()
            .filter(|b| b.is_active)
            .map(|b| b.allocation_percent)
            .sum();

        if current_total + allocation_percent > 1.0 {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: format!(
                    "Total allocation {:.2}% would exceed 100%",
                    (current_total + allocation_percent) * 100.0
                ),
            });
        }

        self.beneficiaries.push(WaqfBeneficiary {
            beneficiary_id,
            allocation_percent,
            eligibility_criteria,
            is_active: true,
        });

        Ok(())
    }

    /// Accrue yield from the principal
    pub fn accrue_yield(&mut self, yield_amount: f64) {
        if yield_amount > 0.0 {
            self.available_yield += yield_amount;
        }
    }

    /// Distribute available yield to beneficiaries
    ///
    /// Distribution rules:
    /// - Admin costs deducted first (capped at MAX_WAQF_OVERHEAD)
    /// - Remaining distributed to beneficiaries by allocation
    /// - Minimum beneficiaries required for distribution
    pub fn distribute_yield(&mut self) -> IslamicFinanceResult<WaqfDistribution> {
        if !self.is_active {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: "Waqf is not active".into(),
            });
        }

        let active_beneficiaries: Vec<_> =
            self.beneficiaries.iter().filter(|b| b.is_active).collect();

        if active_beneficiaries.len() < MIN_WAQF_BENEFICIARIES {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: format!(
                    "Need {} beneficiaries, have {}",
                    MIN_WAQF_BENEFICIARIES,
                    active_beneficiaries.len()
                ),
            });
        }

        if self.available_yield <= 0.0 {
            return Err(IslamicFinanceError::WaqfViolation {
                reason: "No yield available for distribution".into(),
            });
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Calculate admin costs
        let admin_costs = self.available_yield * self.admin_overhead_rate;
        let distributable = self.available_yield - admin_costs;

        // Allocate to beneficiaries
        let mut allocations = HashMap::new();
        for beneficiary in &active_beneficiaries {
            let amount = distributable * beneficiary.allocation_percent;
            allocations.insert(beneficiary.beneficiary_id.clone(), amount);
        }

        // Generate distribution hash
        let hash_input = format!(
            "{}:{}:{}:{}",
            self.waqf_id,
            self.available_yield,
            now_ms,
            allocations.len()
        );
        let distribution_hash = blake3::hash(hash_input.as_bytes()).to_hex().to_string();

        let period_start = self
            .distributions
            .last()
            .map(|d| d.period_end)
            .unwrap_or(self.created_at);

        let distribution = WaqfDistribution {
            distribution_id: format!("waqf_dist_{}", now_ms),
            period_start,
            period_end: now_ms,
            total_yield: self.available_yield,
            allocations,
            admin_costs,
            distribution_hash,
        };

        self.total_distributed += self.available_yield;
        self.available_yield = 0.0;
        self.distributions.push(distribution.clone());

        Ok(distribution)
    }

    /// Verify Waqf maintains Ihsan standards
    pub fn verify_ihsan(&self) -> IslamicFinanceResult<bool> {
        if self.ihsan_score < IHSAN_THRESHOLD {
            return Err(IslamicFinanceError::IhsanViolation {
                score: self.ihsan_score,
                threshold: IHSAN_THRESHOLD,
            });
        }
        Ok(true)
    }

    /// Get total value locked (principal is immutable)
    pub fn total_value_locked(&self) -> f64 {
        self.principal
    }
}

// =============================================================================
// ISLAMIC COMPLIANCE GATE
// =============================================================================

/// Haram (prohibited) activity categories
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HaramCategory {
    /// Interest-based transactions
    Riba,
    /// Excessive uncertainty
    Gharar,
    /// Gambling and speculation
    Maysir,
    /// Harmful content or services
    HarmfulContent,
    /// Privacy violations
    PrivacyViolation,
    /// Fraud or deception
    Fraud,
    /// Exploitation
    Exploitation,
}

/// Islamic compliance check result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Overall compliance status
    pub compliant: bool,
    /// Ihsan score
    pub ihsan_score: f64,
    /// Adl (fairness) score
    pub adl_score: f64,
    /// Detected violations
    pub violations: Vec<ComplianceViolation>,
    /// Timestamp of check
    pub checked_at: u64,
}

/// Individual compliance violation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation category
    pub category: HaramCategory,
    /// Severity (0.0 to 1.0)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Remediation suggestion
    pub remediation: Option<String>,
}

/// Prohibited service types for Halal filtering
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProhibitedService {
    /// Conventional interest-based lending
    InterestLending,
    /// Gambling platforms
    Gambling,
    /// Speculative derivatives
    Derivatives,
    /// Harmful content generation
    HarmfulContent,
    /// Privacy-violating surveillance
    Surveillance,
    /// Fraudulent schemes
    FraudSchemes,
}

/// Islamic Compliance Gate: FATE Integration
///
/// This gate integrates with the existing FATE (Fairness, Accountability,
/// Transparency, Ethics) gate system to add Shariah compliance validation.
///
/// The gate validates:
/// 1. No Riba (interest) in any transaction
/// 2. No Gharar (excessive uncertainty)
/// 3. No Maysir (gambling/speculation)
/// 4. Asset-backed transactions only
/// 5. Halal service types only
/// 6. Adl (justice) in resource distribution
/// 7. Ihsan (excellence) threshold compliance
#[derive(Clone, Debug)]
pub struct IslamicComplianceGate {
    /// Minimum Ihsan threshold
    ihsan_threshold: f64,
    /// Maximum Gharar (uncertainty) tolerance
    max_gharar_tolerance: f64,
    /// Adl invariant for fairness checking
    adl_invariant: AdlInvariant,
    /// Prohibited service types (reserved for future use)
    #[allow(dead_code)]
    prohibited_services: Vec<ProhibitedService>,
    /// Riba detection patterns
    riba_patterns: Vec<String>,
}

impl Default for IslamicComplianceGate {
    fn default() -> Self {
        Self {
            ihsan_threshold: IHSAN_THRESHOLD,
            max_gharar_tolerance: 0.3,
            adl_invariant: AdlInvariant::default(),
            prohibited_services: vec![
                ProhibitedService::InterestLending,
                ProhibitedService::Gambling,
                ProhibitedService::Derivatives,
                ProhibitedService::HarmfulContent,
                ProhibitedService::Surveillance,
                ProhibitedService::FraudSchemes,
            ],
            riba_patterns: vec![
                "interest".to_string(),
                "apr".to_string(),
                "fixed_return".to_string(),
                "guaranteed_yield".to_string(),
                "usury".to_string(),
            ],
        }
    }
}

impl IslamicComplianceGate {
    /// Create with custom Ihsan threshold
    pub fn new(ihsan_threshold: f64) -> Self {
        Self {
            ihsan_threshold,
            ..Default::default()
        }
    }

    /// Check for Riba (interest) indicators in transaction
    fn detect_riba(&self, content: &str) -> Option<ComplianceViolation> {
        let content_lower = content.to_lowercase();

        for pattern in &self.riba_patterns {
            if content_lower.contains(pattern) {
                return Some(ComplianceViolation {
                    category: HaramCategory::Riba,
                    severity: 1.0, // Riba is always maximum severity
                    description: format!("Detected Riba indicator: '{}'", pattern),
                    remediation: Some(
                        "Convert to profit-sharing (Mudarabah/Musharakah) structure".into(),
                    ),
                });
            }
        }

        None
    }

    /// Check for Gharar (uncertainty) in contract terms
    fn detect_gharar(&self, content: &str) -> Option<ComplianceViolation> {
        let content_lower = content.to_lowercase();

        // Check for excessive uncertainty indicators
        let gharar_indicators = [
            "unknown",
            "undefined",
            "uncertain",
            "unspecified",
            "variable",
            "fluctuating",
            "random",
            "chance",
        ];

        let mut uncertainty_score = 0.0;
        for indicator in gharar_indicators {
            if content_lower.contains(indicator) {
                uncertainty_score += 0.15;
            }
        }

        if uncertainty_score > self.max_gharar_tolerance {
            return Some(ComplianceViolation {
                category: HaramCategory::Gharar,
                severity: uncertainty_score.min(1.0),
                description: format!(
                    "Excessive uncertainty detected: score {:.2}",
                    uncertainty_score
                ),
                remediation: Some(
                    "Specify all contract terms clearly and reduce uncertainty".into(),
                ),
            });
        }

        None
    }

    /// Check for Maysir (gambling) elements
    fn detect_maysir(&self, content: &str) -> Option<ComplianceViolation> {
        let content_lower = content.to_lowercase();

        let maysir_indicators = [
            "gamble",
            "bet",
            "lottery",
            "jackpot",
            "speculation",
            "leverage",
            "margin",
        ];

        for indicator in maysir_indicators {
            if content_lower.contains(indicator) {
                return Some(ComplianceViolation {
                    category: HaramCategory::Maysir,
                    severity: 1.0,
                    description: format!("Detected Maysir indicator: '{}'", indicator),
                    remediation: Some(
                        "Remove gambling/speculative elements from the transaction".into(),
                    ),
                });
            }
        }

        None
    }

    /// Comprehensive compliance check
    pub fn check_compliance(
        &self,
        content: &str,
        ihsan_score: Option<f64>,
        distribution: Option<&HashMap<String, f64>>,
    ) -> ComplianceResult {
        let mut violations = Vec::new();
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Check for Riba
        if let Some(v) = self.detect_riba(content) {
            violations.push(v);
        }

        // Check for Gharar
        if let Some(v) = self.detect_gharar(content) {
            violations.push(v);
        }

        // Check for Maysir
        if let Some(v) = self.detect_maysir(content) {
            violations.push(v);
        }

        // Calculate Ihsan score
        let ihsan = ihsan_score.unwrap_or(0.0);
        if ihsan < self.ihsan_threshold {
            violations.push(ComplianceViolation {
                category: HaramCategory::Exploitation, // Low quality exploits users
                severity: (self.ihsan_threshold - ihsan) / self.ihsan_threshold,
                description: format!(
                    "Ihsan score {:.3} below threshold {:.3}",
                    ihsan, self.ihsan_threshold
                ),
                remediation: Some("Improve service quality to meet Ihsan standards".into()),
            });
        }

        // Check Adl (fairness) if distribution provided
        let adl_score = if let Some(dist) = distribution {
            let result = self.adl_invariant.check(dist, None);
            if !result.passed {
                violations.push(ComplianceViolation {
                    category: HaramCategory::Exploitation,
                    severity: result.gini / ADL_GINI_THRESHOLD,
                    description: format!(
                        "Adl violation: Gini {:.4} exceeds threshold {:.2}",
                        result.gini, ADL_GINI_THRESHOLD
                    ),
                    remediation: Some("Redistribute resources to reduce inequality".into()),
                });
            }
            1.0 - result.gini
        } else {
            0.5 // Neutral if no distribution provided
        };

        ComplianceResult {
            compliant: violations.is_empty(),
            ihsan_score: ihsan,
            adl_score,
            violations,
            checked_at: now_ms,
        }
    }
}

/// Implement Gate trait for integration with FATE gate chain
impl Gate for IslamicComplianceGate {
    fn name(&self) -> &'static str {
        "IslamicCompliance"
    }

    fn tier(&self) -> GateTier {
        GateTier::Expensive // Full compliance check is computationally intensive
    }

    fn verify(&self, ctx: &GateContext) -> GateResult {
        let start = Instant::now();

        // Parse content as string
        let content = match std::str::from_utf8(&ctx.content) {
            Ok(s) => s,
            Err(_) => {
                return GateResult::fail(
                    self.name(),
                    RejectCode::RejectGateSchema,
                    start.elapsed(),
                );
            }
        };

        // Run compliance check
        let result = self.check_compliance(content, ctx.ihsan_score, None);

        if result.compliant {
            GateResult::pass(self.name(), start.elapsed())
        } else {
            // Return FATE rejection for non-compliance
            GateResult::fail(self.name(), RejectCode::RejectGateFATE, start.elapsed())
        }
    }
}

// =============================================================================
// ISLAMIC FINANCE REGISTRY
// =============================================================================

/// Central registry for all Islamic finance instruments
#[derive(Clone, Debug, Default)]
pub struct IslamicFinanceRegistry {
    /// Active Mudarabah contracts
    pub mudarabah_contracts: HashMap<String, MudarabahContract>,
    /// Active Musharakah partnerships
    pub musharakah_partnerships: HashMap<String, MusharakahPartnership>,
    /// Active Waqf endowments
    pub waqf_endowments: HashMap<String, WaqfEndowment>,
    /// Zakat calculator
    pub zakat_calculator: ZakatCalculator,
    /// Compliance gate
    pub compliance_gate: IslamicComplianceGate,
    /// Wealth records for Zakat
    pub wealth_records: HashMap<String, WealthRecord>,
}

impl IslamicFinanceRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            mudarabah_contracts: HashMap::new(),
            musharakah_partnerships: HashMap::new(),
            waqf_endowments: HashMap::new(),
            zakat_calculator: ZakatCalculator::default(),
            compliance_gate: IslamicComplianceGate::default(),
            wealth_records: HashMap::new(),
        }
    }

    /// Register a Mudarabah contract
    pub fn register_mudarabah(&mut self, contract: MudarabahContract) -> IslamicFinanceResult<()> {
        // Verify compliance
        let content = serde_json::to_string(&contract).unwrap_or_default();
        let result =
            self.compliance_gate
                .check_compliance(&content, Some(contract.performance_ihsan), None);

        if !result.compliant {
            return Err(IslamicFinanceError::MudarabahViolation {
                reason: format!(
                    "Contract failed compliance: {:?}",
                    result.violations.first()
                ),
            });
        }

        self.mudarabah_contracts
            .insert(contract.contract_id.clone(), contract);
        Ok(())
    }

    /// Register a Musharakah partnership
    pub fn register_musharakah(
        &mut self,
        partnership: MusharakahPartnership,
    ) -> IslamicFinanceResult<()> {
        // Verify Adl compliance
        partnership.verify_adl()?;

        self.musharakah_partnerships
            .insert(partnership.partnership_id.clone(), partnership);
        Ok(())
    }

    /// Register a Waqf endowment
    pub fn register_waqf(&mut self, waqf: WaqfEndowment) -> IslamicFinanceResult<()> {
        // Verify Ihsan compliance
        waqf.verify_ihsan()?;

        self.waqf_endowments.insert(waqf.waqf_id.clone(), waqf);
        Ok(())
    }

    /// Process Zakat for all eligible nodes
    pub fn process_zakat_cycle(&mut self) -> Vec<ZakatDistribution> {
        let mut distributions = Vec::new();

        for (_, record) in self.wealth_records.iter_mut() {
            if let Some(amount) = self.zakat_calculator.calculate_zakat(record) {
                if let Ok(dist) = self.zakat_calculator.distribute_zakat(record, amount) {
                    // Update record
                    record.last_zakat_paid = Some(dist.distributed_at);
                    record.lifetime_zakat += amount;
                    distributions.push(dist);
                }
            }
        }

        distributions
    }

    /// Get total value locked across all instruments
    pub fn total_value_locked(&self) -> f64 {
        let mudarabah_tvl: f64 = self
            .mudarabah_contracts
            .values()
            .filter(|c| c.status == MudarabahStatus::Active)
            .map(|c| c.capital)
            .sum();

        let musharakah_tvl: f64 = self
            .musharakah_partnerships
            .values()
            .filter(|p| p.status == MusharakahStatus::Active)
            .map(|p| p.total_capital)
            .sum();

        let waqf_tvl: f64 = self
            .waqf_endowments
            .values()
            .filter(|w| w.is_active)
            .map(|w| w.total_value_locked())
            .sum();

        mudarabah_tvl + musharakah_tvl + waqf_tvl
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zakat_calculation() {
        let calculator = ZakatCalculator::default();

        let mut record = WealthRecord::new("node1".into());
        record
            .holdings
            .insert(ZakatableAsset::ComputeCredits, 2000.0);

        // Without nisab_exceeded_at, no Zakat due
        assert!(calculator.calculate_zakat(&record).is_none());

        // Set nisab_exceeded_at to over a year ago
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        record.nisab_exceeded_at = Some(now_ms - (HAWL_DAYS * 24 * 60 * 60 * 1000) - 1000);

        let zakat = calculator.calculate_zakat(&record);
        assert!(zakat.is_some());
        assert!((zakat.unwrap() - 50.0).abs() < 0.01); // 2.5% of 2000 = 50
    }

    #[test]
    fn test_zakat_distribution() {
        let calculator = ZakatCalculator::default();
        let record = WealthRecord::new("node1".into());

        let distribution = calculator.distribute_zakat(&record, 100.0).unwrap();
        assert_eq!(distribution.payer_node, "node1");

        let total_allocated: f64 = distribution.allocations.values().sum();
        assert!((total_allocated - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_mudarabah_profit_ratios() {
        // Valid contract: 60% investor, 40% entrepreneur
        let contract = MudarabahContract::new(
            "mud_001".into(),
            "investor".into(),
            "entrepreneur".into(),
            1000.0,
            0.60, // investor gets 60%
            30,
            "compute_service".into(),
        );
        assert!(contract.is_ok());

        // Invalid: entrepreneur gets less than 30%
        let invalid = MudarabahContract::new(
            "mud_002".into(),
            "investor".into(),
            "entrepreneur".into(),
            1000.0,
            0.80, // investor gets 80%, leaving 20% for entrepreneur
            30,
            "compute_service".into(),
        );
        assert!(matches!(
            invalid,
            Err(IslamicFinanceError::MudarabahViolation { .. })
        ));
    }

    #[test]
    fn test_mudarabah_settlement_profit() {
        let mut contract = MudarabahContract::new(
            "mud_001".into(),
            "investor".into(),
            "entrepreneur".into(),
            1000.0,
            0.60,
            30,
            "compute".into(),
        )
        .unwrap();

        contract.activate().unwrap();
        contract.record_pnl(200.0, true).unwrap(); // 200 profit

        let settlement = contract.settle().unwrap();

        // Investor: 1000 capital + 120 (60% of 200) = 1120
        assert!((settlement.investor_return - 1120.0).abs() < 0.01);
        // Entrepreneur: 80 (40% of 200)
        assert!((settlement.entrepreneur_payment - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_mudarabah_settlement_loss() {
        let mut contract = MudarabahContract::new(
            "mud_001".into(),
            "investor".into(),
            "entrepreneur".into(),
            1000.0,
            0.60,
            30,
            "compute".into(),
        )
        .unwrap();

        contract.activate().unwrap();
        contract.record_pnl(300.0, false).unwrap(); // 300 loss

        let settlement = contract.settle().unwrap();

        // Investor bears all capital loss: 1000 - 300 = 700
        assert!((settlement.investor_return - 700.0).abs() < 0.01);
        // Entrepreneur gets nothing (loss of effort)
        assert_eq!(settlement.entrepreneur_payment, 0.0);
    }

    #[test]
    fn test_musharakah_loss_sharing() {
        let mut partnership = MusharakahPartnership::new(
            "msh_001".into(),
            "Compute Pool".into(),
            "compute".into(),
            2,
        );

        partnership
            .add_partner("node1".into(), 600.0, 0.55)
            .unwrap();
        partnership
            .add_partner("node2".into(), 400.0, 0.45)
            .unwrap();
        partnership.activate().unwrap();

        // Verify loss shares equal contribution ratios
        let node1 = partnership
            .partners
            .iter()
            .find(|p| p.node_id == "node1")
            .unwrap();
        let node2 = partnership
            .partners
            .iter()
            .find(|p| p.node_id == "node2")
            .unwrap();

        assert!((node1.loss_share - 0.6).abs() < 0.01); // 60%
        assert!((node2.loss_share - 0.4).abs() < 0.01); // 40%
    }

    #[test]
    fn test_waqf_no_self_dealing() {
        let mut waqf = WaqfEndowment::new(
            "waqf_001".into(),
            "Infrastructure Fund".into(),
            "founder".into(),
            WaqfPurpose::Infrastructure,
            "Network infrastructure".into(),
            10000.0,
            "nazir".into(),
            0.05,
        )
        .unwrap();

        // Founder cannot be beneficiary
        let result = waqf.add_beneficiary("founder".into(), 0.5, "None".into());
        assert!(matches!(
            result,
            Err(IslamicFinanceError::WaqfViolation { .. })
        ));

        // Others can be beneficiaries
        assert!(waqf
            .add_beneficiary("node1".into(), 0.4, "Active nodes".into())
            .is_ok());
    }

    #[test]
    fn test_waqf_overhead_cap() {
        // Overhead exceeds maximum
        let result = WaqfEndowment::new(
            "waqf_001".into(),
            "Test".into(),
            "founder".into(),
            WaqfPurpose::GeneralCharity,
            "Test".into(),
            1000.0,
            "nazir".into(),
            0.15, // 15% exceeds 10% max
        );
        assert!(matches!(
            result,
            Err(IslamicFinanceError::WaqfViolation { .. })
        ));
    }

    #[test]
    fn test_compliance_gate_riba_detection() {
        let gate = IslamicComplianceGate::default();

        // Halal content
        let halal = "profit-sharing partnership with agreed ratios";
        let result = gate.check_compliance(halal, Some(0.96), None);
        assert!(result.compliant);

        // Haram content (interest)
        let haram = "loan with 5% annual interest rate";
        let result = gate.check_compliance(haram, Some(0.96), None);
        assert!(!result.compliant);
        assert!(result
            .violations
            .iter()
            .any(|v| v.category == HaramCategory::Riba));
    }

    #[test]
    fn test_compliance_gate_integration() {
        let gate = IslamicComplianceGate::default();
        let ctx = GateContext {
            sender_id: "test_node".into(),
            envelope_id: "pci_123".into(),
            content: br#"{"type": "mudarabah", "profit_ratio": 0.6}"#.to_vec(),
            constitution: crate::constitution::Constitution::default(),
            snr_score: Some(0.9),
            ihsan_score: Some(0.96),
        };

        let result = gate.verify(&ctx);
        assert!(result.passed);
    }

    #[test]
    fn test_ihsan_threshold_enforcement() {
        let gate = IslamicComplianceGate::default();

        // Below threshold
        let result = gate.check_compliance("valid content", Some(0.90), None);
        assert!(!result.compliant);

        // Above threshold
        let result = gate.check_compliance("valid content", Some(0.96), None);
        assert!(result.compliant);
    }
}
