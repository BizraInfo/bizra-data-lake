//! PAT/SAT Agent Minting System
//!
//! Personal Agentic Team (PAT) and Shared Agentic Team (SAT) implementation.
//!
//! # Architecture
//!
//! Each human node mints:
//! - **7 PAT agents** — Personal mastermind team (private)
//! - **5 SAT agents** — Shared agents contributed to Resource Pool (public utility)
//!
//! # Standing on Giants
//!
//! - **General Magic (1990)**: Telescript primitives for mobile agents
//! - **Shannon (1948)**: SNR-based signal quality in agent communication
//! - **Lamport (1982)**: Byzantine fault tolerance in agent consensus
//! - **Bernstein (2012)**: Ed25519 for agent identity signatures
//! - **Al-Ghazali (1095)**: Maqasid al-Shariah for FATE gate ethics
//! - **Anthropic (2023)**: Constitutional AI for Ihsan threshold

mod attestation;
mod minting;
mod types;

/// Agent attestation — action proofs and attestation registry.
pub use attestation::*;
/// Agent minting engine — PAT/SAT creation with full verification.
pub use minting::*;
/// Core agent types — roles, capabilities, identity blocks.
pub use types::*;

/// PAT team size — 7 personal agents (mastermind council)
pub const PAT_TEAM_SIZE: usize = 7;

/// SAT team size — 5 shared agents (public utility)
pub const SAT_TEAM_SIZE: usize = 5;

/// SAT team size for Atlas parity mode (full federation topology)
pub const SAT_TEAM_SIZE_FULL49: usize = 49;

/// SAT operating profile labels.
pub const SAT_MODE_MINI5: &str = "mini5";
/// SAT operating profile label for full 49-validator topology.
pub const SAT_MODE_FULL49: &str = "full49";

/// Minimum Ihsan threshold for agent minting
pub const AGENT_MINT_IHSAN_THRESHOLD: f64 = 0.95;

/// Maximum delegation depth for agent permits
pub const MAX_AGENT_DELEGATION_DEPTH: u8 = 7;
