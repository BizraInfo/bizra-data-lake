#![warn(missing_docs)]
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

/// Constitutional governance — Ihsan thresholds, SNR rules, enforcement policies.
pub mod constitution;
/// Genesis primitives — BLAKE3 hashing, cryptographic manifests, execution receipts.
pub mod genesis;
/// Node identity — Ed25519 key management, domain-separated signing.
pub mod identity;
/// Islamic finance protocol — Zakat, Mudarabah, Musharakah, Waqf, Shariah compliance.
pub mod islamic_finance;
/// Omega governance engine — Adl invariant, Byzantine consensus, treasury, Ihsan projector.
pub mod omega;
/// PAT/SAT agent minting — Personal and Shared Agentic Teams with attestation.
pub mod pat;
/// Proof-Carrying Inference — Envelopes, gate chains, reject codes.
pub mod pci;
/// SIMD-accelerated operations — Parallel hashing, batch signature verification.
pub mod simd;
/// Sovereign reasoning — Graph-of-Thoughts, SNR engine, Giants protocol, Omega circuit.
pub mod sovereign;

pub use constitution::{Constitution, IhsanThreshold};
pub use genesis::{
    blake3_domain_hash, blake3_hash, CryptoManifest, ExecutionContext, GenesisError,
    GenesisReceipt, GenesisReceiptBuilder, GenesisResult,
};
pub use identity::{domain_separated_digest, NodeId, NodeIdentity};
pub use islamic_finance::{
    ComplianceResult,
    ComplianceViolation,
    HaramCategory,
    // Compliance
    IslamicComplianceGate,
    // Errors
    IslamicFinanceError,
    // Registry
    IslamicFinanceRegistry,
    IslamicFinanceResult,
    // Mudarabah (Profit-Sharing)
    MudarabahContract,
    MudarabahLoss,
    MudarabahSettlement,
    MudarabahStatus,
    MusharakahDecision,
    MusharakahPartner,
    // Musharakah (Partnership)
    MusharakahPartnership,
    MusharakahStatus,
    ProhibitedService,
    WaqfBeneficiary,
    WaqfDistribution,
    // Waqf (Endowment)
    WaqfEndowment,
    WaqfPurpose,
    WealthRecord,
    // Zakat Engine
    ZakatCalculator,
    ZakatDistribution,
    ZakatRecipient,
    ZakatableAsset,
    HAWL_DAYS,
    MAX_RABBULMAL_SHARE,
    MAX_WAQF_OVERHEAD,
    MIN_MUDARIB_SHARE,
    MIN_WAQF_BENEFICIARIES,
    NISAB_THRESHOLD,
    // Constants
    ZAKAT_RATE,
};
pub use omega::{
    // GAP-C2: Adl Invariant
    AdlInvariant,
    AdlInvariantResult,
    AdlViolation,
    AdlViolationType,
    // GAP-C3: Byzantine Consensus
    ByzantineParams,
    ByzantineVoteType,
    ConsensusState,
    // Unified
    ConstitutionalEngine,
    ConstitutionalError,
    IhsanProjector,
    // GAP-C1: Ihsan Projector
    IhsanVector,
    NTUState,
    TreasuryController,
    // GAP-C4: Treasury Controller
    TreasuryMode,
    TreasuryModeConfig,
    ADL_GINI_EMERGENCY,
    // Constants
    ADL_GINI_THRESHOLD,
    BFT_QUORUM_FRACTION,
    LANDAUER_LIMIT_JOULES,
};
pub use pat::{
    // Attestation
    ActionAttestation,
    ActionType,
    AgentCapability,
    // Types
    AgentCapabilityCard,
    AgentIdentityBlock,
    AgentMintRequest,
    // Minting
    AgentMintingEngine,
    AgentResourceLimits,
    AgentState,
    AgentType,
    AttestationRegistry,
    AuthorityLink,
    GiantCitation,
    IntellectualFoundation,
    MintedAgent,
    MintingError,
    MintingResult,
    // Agent Roles
    PATRole,
    PersonalAgentTeam,
    PoolUsageRecord,
    ProvenanceEntry,
    ProvenanceRecord,
    ProvenanceSource,
    ResourceUsage,
    SATRole,
    SharedAgentTeam,
    // Standing on Giants
    StandingOnGiantsAttestation,
    AGENT_MINT_IHSAN_THRESHOLD,
    MAX_AGENT_DELEGATION_DEPTH,
    // Constants
    PAT_TEAM_SIZE,
    SAT_MODE_FULL49,
    SAT_MODE_MINI5,
    SAT_TEAM_SIZE,
    SAT_TEAM_SIZE_FULL49,
};
pub use pci::{Gate, GateChain, GateContext, GateResult, PCIEnvelope, RejectCode};
pub use simd::{blake3_parallel, validate_gates_batch, verify_signatures_batch};
pub use sovereign::{
    // Autopoietic loop
    AutopoieticState, Canonicalize, CanonicalChain, CanonicalCheckpoint, ConstitutionalEra,
    ConvergenceReport, CycleOutcome, MetaConstitution, VerifiedReward,
    // Core sovereign types
    CircuitState, Episode, EpisodeAction, EpisodeImpact, ErrorContext, ExperienceLedger,
    GiantRegistry, OmegaConfig, OmegaEngine, OmegaMetrics, OrchestratorConfig, RIRConfig,
    ReasoningPath, SNRConfig, SNREngine, SNRStats, SignalMetrics, SovereignError,
    SovereignOrchestrator, SovereignResult, ThoughtGraph, ThoughtNode, ThoughtType,
};

/// Domain separation prefix for all cryptographic operations
pub const DOMAIN_PREFIX: &[u8] = b"bizra-pci-v1:";

// =============================================================================
// CONSTITUTIONAL CONSTANTS — LOCKED, require constitutional amendment.
//
// Cross-repo alignment:
//   Python: core/integration/constants.py (authoritative)
//   Rust:   this file (bizra-omega/bizra-core/src/lib.rs) + omega.rs
//   TS:     src/core/sovereign/capability-card.ts
//
// Standing on Giants: Shannon · Lamport · Al-Ghazali · Anthropic
// =============================================================================

/// Ihsan threshold — hard constraint for excellence (production)
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// Strict Ihsan — consensus and constitutional operations
pub const STRICT_IHSAN_THRESHOLD: f64 = 0.99;

/// SNR threshold — minimum signal quality (museum floor)
pub const SNR_THRESHOLD: f64 = 0.85;

/// SNR Tier 1 — high-quality operations
pub const SNR_THRESHOLD_T1_HIGH: f64 = 0.95;

/// SNR Tier 0 — elite operations
pub const SNR_THRESHOLD_T0_ELITE: f64 = 0.98;

/// Maximum envelope TTL in seconds
pub const MAX_TTL_SECONDS: u64 = 3600;
