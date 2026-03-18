//! # bizra-protocol — The Nervous System
//!
//! The 26th crate. Wires 25 organs into one living sovereign system.
//!
//! ## What This Crate Does
//!
//! Every other BIZRA crate builds an organ:
//! - `bizra-core`: Cryptographic identity (Ed25519 + BLAKE3)
//! - `bizra-telescript`: Trust boundary primitives (Place, Agent, Permit)
//! - `bizra-action`: Execution pipeline (Dispatcher → Guardian → Receipt)
//! - `bizra-resourcepool`: URP economics (SEED, BLOOM, Zakat, Gini)
//! - `bizra-mission`: Lifecycle state machine
//!
//! This crate connects them. It defines the runtime protocol:
//!
//! ```text
//! Human → DEMA → PAT (local) → [proof-carry] → SAT (in URP) → Network
//!                                    ↑
//!                            TRUST BOUNDARY
//!                     (the entire architecture)
//! ```
//!
//! ## The Five Protocol Phases
//!
//! 1. **Mint** — HD key derivation: master identity → 7 PAT keys + 5 SAT keys
//! 2. **Execute** — PAT does work locally, Guardian gates every action, Receipt produced
//! 3. **Cross** — Receipt wrapped in Telescript Ticket, crosses trust boundary to URP
//! 4. **Validate** — SAT validates independently, counter-signs attestation
//! 5. **Propagate** — Attestation → SEED mint → proof chain → network
//!
//! ## Constitutional Invariants (compile-time enforced)
//!
//! - All hashes are BLAKE3 with domain separation (NOT SHA-256)
//! - All signatures are Ed25519 with domain prefix "bizra-protocol-v1"
//! - Ihsān floor: 0.95 (actions below this are type errors)
//! - Gini cap: 0.35 (economic operations above this halt)
//! - RIBA_ZERO: no interest-bearing debt can be encoded
//! - ZANN_ZERO: no unattested claim can propagate
//!
//! ## Standing on Giants
//!
//! - General Magic (1990): Telescript — mobile agent primitives
//! - Nakamoto (2008): Proof-carrying state transitions
//! - Lamport (1982): Byzantine fault tolerance in agent consensus
//! - Al-Ghazali (1095): Maqasid al-Shariah → FATE gate ethics
//! - Maturana & Varela (1970s): Autopoiesis → self-loops as architecture

pub mod mint;
pub mod boundary;
pub mod attestation;
pub mod flow;
pub mod autopoiesis;

/// Protocol version — embedded in every signed artifact
pub const PROTOCOL_VERSION: &str = "bizra-protocol-v1";

/// Domain prefix for all BLAKE3 hashes in this crate
pub const DOMAIN_PREFIX: &[u8] = b"bizra-protocol-v1:";

/// Constitutional constants — Single Source of Truth
pub mod constitution {
    /// Ihsān floor: minimum quality score for any action to pass
    pub const IHSAN_FLOOR: f64 = 0.95;

    /// Adl Gini cap: maximum inequality coefficient
    pub const GINI_CAP: f64 = 0.35;

    /// Number of PAT agents per node (personal, local)
    pub const PAT_COUNT: u32 = 7;

    /// Number of SAT agents per node (system, URP)
    pub const SAT_COUNT: u32 = 5;

    /// Total agents minted per identity
    pub const TOTAL_AGENTS: u32 = PAT_COUNT + SAT_COUNT; // 12

    /// HD derivation path prefix for PAT agents
    pub const PAT_DERIVATION_PREFIX: &str = "bizra-pat-agent-v1";

    /// HD derivation path prefix for SAT agents
    pub const SAT_DERIVATION_PREFIX: &str = "bizra-sat-agent-v1";

    /// PAT agent roles (local, serve the human)
    pub const PAT_ROLES: [&str; 7] = [
        "P1-Analyst",
        "P2-Strategist",
        "P3-Technical",
        "P4-Creative",
        "P5-Ethicist",   // Frozen — revelation-derived constants
        "P6-Operational",
        "P7-DEMA-Nexus",  // The Daughter Test gate
    ];

    /// SAT agent roles (URP, serve the constitution)
    pub const SAT_ROLES: [&str; 5] = [
        "S1-Auditor",
        "S2-Oracle",       // Frozen — external truth anchor
        "S3-Compliance",
        "S4-Risk",
        "S5-Constitutional",
    ];
}
