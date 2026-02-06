//! BIZRA Federation — Distributed Sovereignty
//!
//! SECURITY: All federation communication is cryptographically signed.
//! - Consensus votes: Ed25519 signatures verified before counting
//! - Gossip messages: Ed25519 signatures prevent spoofing
//!
//! Standing on Giants: Lamport (BFT), Das (SWIM)

pub mod gossip;
pub mod consensus;
pub mod node;
pub mod bootstrap;

pub use gossip::{
    GossipProtocol, GossipMessage, NodeState, Member,
    SignedGossipMessage, FederationError,
};
pub use consensus::{
    ConsensusEngine, Proposal, Vote, ConsensusState,
    SignedVote, ConsensusError, KnownPeer,
};
pub use node::{FederationNode, NodeConfig};
pub use bootstrap::{Bootstrapper, BootstrapConfig, BootstrapResult, PeerInfo};

pub const DEFAULT_GOSSIP_PORT: u16 = 7946;
pub const GOSSIP_INTERVAL_MS: u64 = 1000;
