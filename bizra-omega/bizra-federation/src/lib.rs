//! BIZRA Federation — Distributed Sovereignty
//!
//! SECURITY: All federation communication is cryptographically signed.
//! - Consensus votes: Ed25519 signatures verified before counting
//! - Gossip messages: Ed25519 signatures prevent spoofing
//!
//! Standing on Giants: Lamport (BFT), Das (SWIM)

pub mod bootstrap;
pub mod consensus;
pub mod gossip;
pub mod node;

pub use bootstrap::{BootstrapConfig, BootstrapResult, Bootstrapper, PeerInfo};
pub use consensus::{
    ConsensusEngine, ConsensusError, ConsensusState, KnownPeer, Proposal, SignedVote, Vote,
};
pub use gossip::{
    FederationError, GossipMessage, GossipProtocol, Member, NodeState, SignedGossipMessage,
};
pub use node::{FederationNode, NodeConfig};

pub const DEFAULT_GOSSIP_PORT: u16 = 7946;
pub const GOSSIP_INTERVAL_MS: u64 = 1000;
