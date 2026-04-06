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

/// Canonical layer — domain-separated hashing, chain integrity, 5 invariants.
pub mod canonical;
/// CanonicalReceipt v1 — the spearpoint artifact. One receipt per visible effect.
pub mod canonical_receipt;
/// Constitutional governance — Ihsan thresholds, SNR rules, enforcement policies.
pub mod constitution;
/// Constitutional Gate Policy — unified enforcement for all threshold violations.
pub mod gate_policy;
/// Genesis primitives — BLAKE3 hashing, cryptographic manifests, execution receipts.
pub mod genesis;
/// GenesisSeal v1 — deterministic root of trust binding receipts to constitution.
pub mod genesis_seal;
/// Golden Vector — cross-language sealing test (Rust/Python digest parity).
pub mod golden_vector;
/// Node identity — Ed25519 key management, domain-separated signing.
pub mod identity;
/// Islamic finance protocol — Zakat, Mudarabah, Musharakah, Waqf, Shariah compliance.
pub mod islamic_finance;
/// Kernel Action Grammar — Constitutional Computability Doctrine v1.
/// Defines the finite action set decidable as blocking syscalls,
/// the four-verdict enum (PERMIT|REJECT|REVIEW|SCORE_ONLY),
/// and budget contracts for bounded constitutional review.
pub mod kernel_action_grammar;
/// MissionState v1 — sovereign mission lifecycle. Human intent through the constitutional pipeline.
pub mod mission_state;
/// Omega governance engine — Adl invariant, Byzantine consensus, treasury, Ihsan projector.
pub mod omega;
/// PAT/SAT agent minting — Personal and Shared Agentic Teams with attestation.
pub mod pat;
/// Proof-Carrying Inference — Envelopes, gate chains, reject codes.
pub mod pci;
/// ReceiptStateMachine v1 — transition law for CanonicalReceipt lifecycle.
pub mod receipt_state_machine;
/// SIMD-accelerated operations — Parallel hashing, batch signature verification.
pub mod simd;
/// Sovereign reasoning — Graph-of-Thoughts, SNR engine, Giants protocol, Omega circuit.
pub mod sovereign;
/// TopologyCanon v1 — frozen agent/node/network topology. PAT-7, SAT-5, gate chain order.
pub mod topology_canon;
/// Walking Skeleton — thinnest end-to-end constitutional liveness proof.
pub mod walking_skeleton;

pub use canonical::{
    block_hash, chain_hash, constitution_hash, domain_hash, episode_hash, hex, identity_hash,
    receipt_hash, DOMAIN_BLOCK, DOMAIN_CHAIN, DOMAIN_CONSTITUTION, DOMAIN_EPISODE, DOMAIN_IDENTITY,
    DOMAIN_POLICY, DOMAIN_RECEIPT,
};
pub use constitution::{Constitution, IhsanThreshold};
pub use gate_policy::{
    apply_gate, env_gate_policy, GateAction, GateMaturationPolicy, GatePolicy, GateVerdict,
    MaturationThresholds,
};
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
    CONSTITUTIONAL_GINI_THRESHOLD,
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
    AutopoieticState,
    CanonicalChain,
    CanonicalCheckpoint,
    Canonicalize,
    // Core sovereign types
    CircuitState,
    ConstitutionalEra,
    ConvergenceReport,
    CycleOutcome,
    Episode,
    EpisodeAction,
    EpisodeImpact,
    ErrorContext,
    ExperienceLedger,
    GiantRegistry,
    MetaConstitution,
    OmegaConfig,
    OmegaEngine,
    OmegaMetrics,
    OrchestratorConfig,
    RIRConfig,
    ReasoningPath,
    SNRConfig,
    SNREngine,
    SNRStats,
    SignalMetrics,
    SovereignError,
    SovereignOrchestrator,
    SovereignResult,
    ThoughtGraph,
    ThoughtNode,
    ThoughtType,
    VerifiedReward,
};
pub use walking_skeleton::{run_skeleton, SkeletonReceipt};

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

/// Runtime Ihsan — Z3-proven floor for live sovereign operations
pub const RUNTIME_IHSAN_THRESHOLD: f64 = 1.0;

/// CI Ihsan — relaxed threshold for continuous integration environments
pub const IHSAN_THRESHOLD_CI: f64 = 0.90;

/// Dev Ihsan — development environment threshold (local iteration)
pub const IHSAN_THRESHOLD_DEV: f64 = 0.80;

/// Maximum envelope TTL in seconds
pub const MAX_TTL_SECONDS: u64 = 3600;
