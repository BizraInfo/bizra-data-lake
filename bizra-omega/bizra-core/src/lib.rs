//! BIZRA Core — The Sovereign Kernel
//!
//! Identity, PCI Protocol, and Constitutional Governance.
//!
//! Performance optimizations:
//! - SIMD-accelerated gate validation (2x throughput)
//! - Batch signature verification (4x throughput)
//! - Parallel BLAKE3 hashing (2x throughput)
//!
//! Islamic Finance Protocol Layer:
//! - No Riba (interest) — All lending is profit-sharing
//! - Zakat Distribution — Automatic 2.5% on wealth above nisab
//! - Halal Services Only — Ethical envelope with Shariah compliance
//! - Risk Sharing — Losses shared proportionally
//! - Asset-Backed — Every token represents real value
//!
//! PAT/SAT Agent Minting System:
//! - 7 PAT agents (Personal Agentic Team) — Private mastermind council
//! - 5 SAT agents (Shared Agentic Team) — Public utility in Resource Pool
//! - Standing on Giants protocol — Mandatory attribution chain

pub mod constitution;
pub mod genesis;
pub mod identity;
pub mod islamic_finance;
pub mod omega;
pub mod pat;
pub mod pci;
pub mod simd;
pub mod sovereign;

pub use identity::{NodeIdentity, NodeId, domain_separated_digest};
pub use constitution::{Constitution, IhsanThreshold};
pub use genesis::{
    GenesisReceipt, GenesisReceiptBuilder, GenesisError, GenesisResult,
    ExecutionContext, CryptoManifest, blake3_hash, blake3_domain_hash,
};
pub use pci::{PCIEnvelope, RejectCode, GateChain, Gate, GateResult, GateContext};
pub use simd::{validate_gates_batch, verify_signatures_batch, blake3_parallel};
pub use sovereign::{
    SovereignOrchestrator, OrchestratorConfig, SNREngine, SNRConfig, SignalMetrics,
    ThoughtGraph, ThoughtNode, ReasoningPath, GiantRegistry,
    OmegaEngine, OmegaConfig, OmegaMetrics, CircuitState,
    SovereignError, SovereignResult, ErrorContext,
};
pub use islamic_finance::{
    // Zakat Engine
    ZakatCalculator, ZakatDistribution, ZakatableAsset, ZakatRecipient, WealthRecord,
    // Mudarabah (Profit-Sharing)
    MudarabahContract, MudarabahStatus, MudarabahSettlement, MudarabahLoss,
    // Musharakah (Partnership)
    MusharakahPartnership, MusharakahPartner, MusharakahStatus, MusharakahDecision,
    // Waqf (Endowment)
    WaqfEndowment, WaqfPurpose, WaqfBeneficiary, WaqfDistribution,
    // Compliance
    IslamicComplianceGate, ComplianceResult, ComplianceViolation, HaramCategory, ProhibitedService,
    // Registry
    IslamicFinanceRegistry,
    // Errors
    IslamicFinanceError, IslamicFinanceResult,
    // Constants
    ZAKAT_RATE, NISAB_THRESHOLD, MIN_MUDARIB_SHARE, MAX_RABBULMAL_SHARE, HAWL_DAYS,
    MAX_WAQF_OVERHEAD, MIN_WAQF_BENEFICIARIES,
};
pub use omega::{
    // GAP-C1: Ihsan Projector
    IhsanVector, NTUState, IhsanProjector,
    // GAP-C2: Adl Invariant
    AdlInvariant, AdlInvariantResult, AdlViolation, AdlViolationType,
    // GAP-C3: Byzantine Consensus
    ByzantineParams, ByzantineVoteType, ConsensusState,
    // GAP-C4: Treasury Controller
    TreasuryMode, TreasuryModeConfig, TreasuryController,
    // Unified
    ConstitutionalEngine, ConstitutionalError,
    // Constants
    ADL_GINI_THRESHOLD, ADL_GINI_EMERGENCY, BFT_QUORUM_FRACTION, LANDAUER_LIMIT_JOULES,
};
pub use pat::{
    // Agent Roles
    PATRole, SATRole, AgentCapability, AgentState,
    // Types
    AgentCapabilityCard, AgentResourceLimits, MintedAgent,
    PersonalAgentTeam, SharedAgentTeam, AgentType, AgentMintRequest,
    AgentIdentityBlock, AuthorityLink,
    // Standing on Giants
    StandingOnGiantsAttestation, IntellectualFoundation, ProvenanceRecord,
    // Minting
    AgentMintingEngine, MintingError, MintingResult,
    // Attestation
    ActionAttestation, ActionType, GiantCitation, ProvenanceEntry, ProvenanceSource,
    AttestationRegistry, PoolUsageRecord, ResourceUsage,
    // Constants
    PAT_TEAM_SIZE, SAT_TEAM_SIZE, AGENT_MINT_IHSAN_THRESHOLD, MAX_AGENT_DELEGATION_DEPTH,
};

/// Domain separation prefix for all cryptographic operations
pub const DOMAIN_PREFIX: &[u8] = b"bizra-pci-v1:";

/// Ihsan threshold — hard constraint for excellence
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// SNR threshold — minimum signal quality
pub const SNR_THRESHOLD: f64 = 0.85;

/// Maximum envelope TTL in seconds
pub const MAX_TTL_SECONDS: u64 = 3600;
